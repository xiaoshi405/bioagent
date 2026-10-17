"""Development-only external-method canary and native 20-tool preflight."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import httpx

from scripts.external_baselines.audit import audit_run, source_index
from scripts.external_baselines.common import (Budget, ModelClient, Recorder,
                                               ToolFacade, encode, file_digest,
                                               now, tool_catalog)
from scripts.external_baselines.react import ADAPTER_VERSION as REACT_VERSION, run_react


ROOT = Path(__file__).resolve().parents[1]
CANARY = ROOT / "evaluation/biotooleval/external_baselines_dev_canary_v1.json"
DATASET = ROOT / "evaluation/biotooleval/dev70_complex_v2_candidate.json"
MODELS = ("qwen3.7-plus-2026-05-26", "deepseek-v4-flash-0731")


def load_dev_canary():
    specification = json.loads(CANARY.read_text(encoding="utf-8-sig"))
    if file_digest(DATASET, canonical=True) != specification["dataset_canonical_sha256"]:
        raise ValueError("Dev dataset changed; do not silently reuse the canary")
    data = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    if data["status"] != "candidate_not_for_final_metrics":
        raise ValueError("Stage 1 accepts development candidates only")
    by_id = {item["id"]: item for item in data["items"]}
    selected = []
    for entry in specification["tasks"]:
        item = by_id[entry["id"]]
        if item["split"] != "dev":
            raise ValueError("Non-Dev item in canary")
        selected.append(item)
    return specification, selected


def make_request(item: dict):
    # Deliberately excludes Gold, governance, note, preferred chains and rubrics.
    inp = item["input"]
    visible = {k: inp[k] for k in ("input_type", "sequence", "record_id", "record_ids", "chain_id")
               if k in inp}
    files = []
    if "fixture" in inp:
        path = (DATASET.parent / inp["fixture"]).resolve()
        path.relative_to(DATASET.parent.resolve())
        if not path.is_file():
            raise ValueError("Missing Dev fixture")
        files.append(str(path))
        visible["file_path"] = str(path)
    return {"question": item["question_zh"], "input": visible, "supplied_files": files}, files


def write_new(path: Path, value):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encode(value) + "\n")


def code_snapshot():
    paths = [*sorted((ROOT / "scripts/external_baselines").glob("*.py")),
             ROOT / "scripts/run_external_baselines.py", CANARY,
             ROOT / "scripts/run_external_calibration.py",
             ROOT / "scripts/score_external_baselines.py",
             ROOT / "evaluation/biotooleval/external_tools_public_returns_v1.json",
             ROOT / "app/llms/api_clients/openai_api.py",
             ROOT / "app/core/config.py", ROOT / "app/core/runtime_identity.py",
             *sorted((ROOT / "ToolsAgent/ToolsFuns/Biology").glob("*.py")),
             ROOT / "app/tools/safety/toxicity_checker.py", ROOT / "data/toxin_protein.csv",
             *[ROOT / "third_party/codeact_m3tooleval" / p for p in
               ("UPSTREAM.json", "LICENSE", "main.py", "tasks/base.py", "tasks/repl.py")]]
    paths += sorted(p for p in (ROOT / "third_party/scitoolagent_core").rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts)
    paths += sorted(p for p in (ROOT / "third_party/biomni_a1_core").rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts)
    from scripts.external_baselines.f1 import native_identity
    paths += [ROOT / p for p in native_identity()["files"]]
    return {str(p.relative_to(ROOT)): file_digest(p) for p in paths if p.is_file()}


async def preflight(base_url: str, out: Path, budget: Budget):
    async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
        identity = (await client.get(base_url + "/api/runtime-info")).json()
        schema = (await client.get(base_url + "/openapi.json")).json()
    # Do not use a service rooted in another checkout: file guards would differ.
    recorded_root = identity.get("project_root")
    if not recorded_root:
        recorded_root = (identity.get("runtime_identity") or {}).get("project_root")
    if not recorded_root:
        raise ValueError("Biology service did not report its checkout root")
    if Path(recorded_root).resolve() != ROOT.resolve():
        raise ValueError("Biology service belongs to a different checkout")
    from scripts.external_baselines.common import REGISTRY
    missing = [name for name, spec in REGISTRY.items()
               if spec["api_path"] not in schema.get("paths", {})]
    if missing:
        raise ValueError("Missing scientific endpoints: " + ", ".join(missing))
    fasta = ROOT / "tests/fixtures/fasta/single_protein.fasta"
    multi = ROOT / "tests/fixtures/fasta/multi_protein.fasta"
    pdb = DATASET.parent / "fixtures/pdb/mini_two_chain.pdb"
    files = [str(p.resolve()) for p in (fasta, multi, pdb)]
    record = Recorder(out / "preflight_events.jsonl", "preflight_20_tools")
    facade = ToolFacade(base_url, files, record,
                        Budget(**{**asdict(budget), "max_tool_calls": 20}))
    raw = fasta.read_text(encoding="utf-8-sig").splitlines()
    sequence = "".join(line.strip() for line in raw if not line.startswith(">"))
    for name in REGISTRY:
        parameters = REGISTRY[name]["parameters"]
        if "sequence" in parameters:
            args = {"sequence": sequence}
        elif name == "parse_protein_fasta":
            args = {"file_path": files[0]}
        elif name == "parse_multi_protein_fasta":
            args = {"file_path": files[1]}
        elif name == "select_fasta_record":
            args = {"collection": {"$ref": multi_call + "#/collection"}, "record_id": "protein_b"}
        elif name == "align_fasta_records":
            args = {"collection": {"$ref": multi_call + "#/collection"},
                    "record_ids": ["protein_a", "protein_b"]}
        elif name == "summarize_pairwise_alignment":
            args = {"alignment": {"$ref": alignment_call + "#/alignment"}}
        else:
            args = {"file_path": files[2]}
            if name in {"select_pdb_chain", "calculate_protein_anm"}:
                args["chain_id"] = "B"
            if name == "calculate_protein_anm":
                args["n_modes"] = 3
        result = await facade.call(name, args)
        if name == "parse_multi_protein_fasta":
            multi_call = result["call_id"]
        if name == "align_fasta_records":
            alignment_call = result["call_id"]
    calls = [e for e in record.events if e["kind"] == "tool_call"]
    report = {"scope": "local_interface_preflight_not_paper_experiment",
              "actual_tool_base": base_url,
              "service_identity": identity, "registered_tools": len(REGISTRY),
              "endpoint_count": len(REGISTRY) - len(missing),
              "successful_tools": [e["tool_name"] for e in calls if e["status"] == "success"],
              "failed_tools": [{"name": e["tool_name"], "status": e["status"],
                                "code": e["output"].get("code")} for e in calls if e["status"] != "success"],
              "files": [{"path": p, "sha256": file_digest(Path(p))} for p in files]}
    write_new(out / "preflight.json", report)
    if report["failed_tools"]:
        raise ValueError("20-tool preflight failed; inspect preflight.json")
    return report


async def codeact_preflight(base_url: str, out: Path, budget: Budget):
    """Exercise all Python wrappers through the official loop, with no API LLM."""
    from scripts.external_baselines.codeact import run_codeact
    from scripts.external_baselines.common import REGISTRY
    files = [str((ROOT / "tests/fixtures/fasta/single_protein.fasta").resolve()),
             str((ROOT / "tests/fixtures/fasta/multi_protein.fasta").resolve()),
             str((DATASET.parent / "fixtures/pdb/mini_two_chain.pdb").resolve())]
    lines = [f"sequence = ''.join(row.strip() for row in open({files[0]!r}, encoding='utf-8') if not row.startswith('>'))"]
    for name, spec in REGISTRY.items():
        if "sequence" in spec["parameters"]:
            expression = f"{name}(sequence=sequence)"
        elif name == "parse_protein_fasta":
            expression = f"{name}(file_path={files[0]!r})"
        elif name == "parse_multi_protein_fasta":
            expression = f"{name}(file_path={files[1]!r})"
        elif name == "select_fasta_record":
            expression = "select_fasta_record(collection=multi_result['collection'], record_id='protein_b')"
        elif name == "align_fasta_records":
            expression = "align_fasta_records(collection=multi_result['collection'], record_ids=['protein_a', 'protein_b'])"
        elif name == "summarize_pairwise_alignment":
            expression = "summarize_pairwise_alignment(alignment=alignment_result['alignment'])"
        else:
            extra = ", chain_id='B'" if name in {"select_pdb_chain", "calculate_protein_anm"} else ""
            if name == "calculate_protein_anm":
                extra += ", n_modes=3"
            expression = f"{name}(file_path={files[2]!r}{extra})"
        variable = "multi_result" if name == "parse_multi_protein_fasta" else "alignment_result" if name == "align_fasta_records" else "result"
        lines += [f"{variable} = {expression}", f"print({name!r}, {variable}['status'])"]

    class ScriptedPreflight:
        def __init__(self):
            self.responses = iter(["Action:\n" + "\n".join(lines) + "\nEnd Action",
                                   "Answer: Local Python-wrapper interface preflight completed."])

        async def complete_messages(self, messages, model, limit, timeout):
            return {"content": next(self.responses), "response_origin": "local_scripted_preflight",
                    "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None}}

    record = Recorder(out / "codeact_preflight_events.jsonl", "official_codeact_20_wrappers")
    allowance = Budget(**{**asdict(budget), "max_tool_calls": 20})
    facade = ToolFacade(base_url, files, record, allowance)
    result = await run_codeact({"question": "Local interface preflight, not a scientific experiment",
                               "input": {}, "supplied_files": files},
                              "local-scripted-preflight", ScriptedPreflight(), facade, record, allowance)
    calls = [e for e in record.events if e["kind"] == "tool_call"]
    successful = [e["tool_name"] for e in calls if e["status"] == "success"]
    report = {"scope": "local_official_codeact_interface_preflight_not_paper_experiment",
              "external_model_calls": 0, "native_result": result,
              "successful_tools": successful,
              "failed_tools": [e["tool_name"] for e in calls if e["status"] != "success"]}
    write_new(out / "codeact_preflight.json", report)
    if set(successful) != set(REGISTRY) or report["failed_tools"] or result["final_status"] != "answer_delivered":
        raise ValueError("Official CodeAct Python-wrapper preflight failed")
    return report


async def scitoolagent_preflight(base_url: str, out: Path, budget: Budget):
    """Original graph retrieval, plan parsing, executor and summary; mock LLM only."""
    from scripts.external_baselines.scitoolagent import run_scitoolagent
    from scripts.external_baselines.common import REGISTRY
    archived = [json.loads(line) for line in (out / "preflight_events.jsonl").read_text(encoding="utf-8").splitlines()]
    calls = [e for e in archived if e["kind"] == "tool_call"]
    responses = ["Plan Chain:\n" + repr([e["tool_name"] for e in calls])]
    responses += ["Input:" + encode(e["resolved_arguments"]) for e in calls]
    responses += ["IsCompleted:Completed\nFinalAnswer:Local official interface preflight completed."]

    class ScriptedPreflight:
        def __init__(self):
            self.responses = iter(responses)

        async def complete(self, prompt, system, model, limit, timeout):
            return {"content": next(self.responses), "response_origin": "local_scripted_preflight",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}

    files = sorted({e["resolved_arguments"]["file_path"] for e in calls
                    if "file_path" in e["resolved_arguments"]})
    record = Recorder(out / "scitoolagent_preflight_events.jsonl", "official_scitoolagent_20_wrappers")
    allowance = Budget(**{**asdict(budget), "max_tool_calls": 20, "max_llm_calls": 24})
    facade = ToolFacade(base_url, files, record, allowance)
    result = await run_scitoolagent({"question": "Local protein scientific tool interface preflight",
                                   "input": {}, "supplied_files": files},
                                  "local-scripted-preflight", ScriptedPreflight(), facade, record, allowance)
    executed = [e for e in record.events if e["kind"] == "tool_call"]
    successful = [e["tool_name"] for e in executed if e["status"] == "success"]
    report = {"scope": "local_official_scitoolagent_interface_preflight_not_paper_experiment",
              "external_model_calls": 0, "native_result": result,
              "successful_tools": successful,
              "kg_retrieval_count": sum(e["kind"] == "kg_retrieval" for e in record.events),
              "failed_tools": [e["tool_name"] for e in executed if e["status"] != "success"]}
    write_new(out / "scitoolagent_preflight.json", report)
    if set(successful) != set(REGISTRY) or report["failed_tools"] or result["final_status"] != "answer_delivered":
        raise ValueError("Official SciToolAgent native preflight failed")
    return report


async def biomni_preflight(base_url: str, out: Path, budget: Budget):
    """Official resource retrieval and A1 graph, all wrappers; no API LLM."""
    from scripts.external_baselines.biomni import run_biomni
    from scripts.external_baselines.common import REGISTRY
    archived = [json.loads(line) for line in (out / "preflight_events.jsonl").read_text(encoding="utf-8").splitlines()]
    calls = [e for e in archived if e["kind"] == "tool_call"]
    files = sorted({e["resolved_arguments"]["file_path"] for e in calls if "file_path" in e["resolved_arguments"]})
    lines = ["from biomni.tool.protein_tools import " + ", ".join(REGISTRY)]
    for e in calls:
        name = e["tool_name"]
        arguments = e["resolved_arguments"]
        params = [k + "=" + repr(v) for k, v in arguments.items()]
        if name in {"select_fasta_record", "align_fasta_records"}:
            params = [p for p in params if not p.startswith("collection=")]
            params.insert(0, "collection=multi_result['collection']")
        if name == "summarize_pairwise_alignment":
            params = ["alignment=alignment_result['alignment']"]
        variable = "multi_result" if name == "parse_multi_protein_fasta" else "alignment_result" if name == "align_fasta_records" else "result"
        lines += [f"{variable} = {name}(" + ", ".join(params) + ")", f"print({name!r}, {variable}['status'])"]
    class ScriptedPreflight:
        def __init__(self):
            self.responses = iter(["TOOLS: " + repr(list(range(20))) + "\nDATA_LAKE: []\nLIBRARIES: []",
                "<execute>\n" + "\n".join(lines) + "\n</execute>",
                "<solution>Local official A1 interface preflight completed.</solution>"])

        async def complete_messages(self, messages, model, limit, timeout, **kwargs):
            return {"content": next(self.responses), "response_origin": "local_scripted_preflight",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}

    record = Recorder(out / "biomni_preflight_events.jsonl", "official_biomni_20_wrappers")
    allowance = Budget(**{**asdict(budget), "max_tool_calls": 20})
    facade = ToolFacade(base_url, files, record, allowance)
    result = await run_biomni({"question": "Local tool interface preflight, not a scientific experiment",
                              "input": {}, "supplied_files": files},
                             "local-scripted-preflight", ScriptedPreflight(), facade, record, allowance)
    executed = [e for e in record.events if e["kind"] == "tool_call"]
    successful = [e["tool_name"] for e in executed if e["status"] == "success"]
    report = {"scope": "local_official_biomni_a1_interface_preflight_not_paper_experiment",
              "external_model_calls": 0, "native_result": result, "successful_tools": successful,
              "resource_retrieval_count": sum(e["kind"] == "native_resource_retrieval" for e in record.events),
              "failed_tools": [e["tool_name"] for e in executed if e["status"] != "success"]}
    write_new(out / "biomni_preflight.json", report)
    if set(successful) != set(REGISTRY) or report["failed_tools"] or result["final_status"] != "answer_delivered":
        raise ValueError("Official Biomni A1 native preflight failed")
    return report


async def execute(args):
    specification, items = load_dev_canary()
    if args.limit:
        items = items[:args.limit]
    budget = Budget()
    condition = getattr(args, "condition", "ReAct")
    upstream = None
    run_agent = run_react
    adapter_version = REACT_VERSION
    if condition == "CodeAct":
        from scripts.external_baselines.codeact import ADAPTER_VERSION, run_codeact
        from scripts.external_baselines.codeact_upstream import upstream_identity
        upstream = upstream_identity()
        run_agent = run_codeact
        adapter_version = ADAPTER_VERSION
    elif condition == "SciToolAgent":
        from scripts.external_baselines.scitoolagent import ADAPTER_VERSION, run_scitoolagent
        from scripts.external_baselines.scitoolagent_upstream import upstream_identity
        upstream = upstream_identity()
        run_agent = run_scitoolagent
        adapter_version = ADAPTER_VERSION
    elif condition == "Biomni-A1":
        from scripts.external_baselines.biomni import ADAPTER_VERSION, run_biomni
        from scripts.external_baselines.biomni_upstream import upstream_identity
        upstream = upstream_identity()
        run_agent = run_biomni
        adapter_version = ADAPTER_VERSION
    elif condition == "F1":
        from scripts.external_baselines.f1 import ADAPTER_VERSION, run_f1, native_identity
        upstream = native_identity()
        run_agent = run_f1
        adapter_version = ADAPTER_VERSION
    started_keys = set()
    previously_responding_models = set()
    resume_sources = []
    for source_name in getattr(args, "resume_unstarted_from", []) or []:
        if args.limit:
            raise ValueError("Resume cannot be combined with --limit")
        source = (ROOT / "evaluation/results" / source_name).resolve()
        source.relative_to((ROOT / "evaluation/results").resolve())
        frozen = json.loads((source / "manifest.final.json").read_text(encoding="utf-8"))
        if frozen["status"] not in {"completed", "model_blocked"}:
            raise ValueError("Resume requires a finished source batch")
        if frozen["condition"] != condition or frozen["adapter"] != adapter_version:
            raise ValueError("Resume cannot change the method or adapter version")
        if frozen["dataset_canonical_sha256"] != specification["dataset_canonical_sha256"]:
            raise ValueError("Resume cannot change the input dataset")
        current = code_snapshot()
        changed = [p for p, sha in frozen["code_sha256"].items()
                   if p.replace("\\", "/") != "scripts/run_external_baselines.py" and current.get(p) != sha]
        if changed:
            raise ValueError("Resume changes non-runner sources: " + repr(changed))
        archived_events = [json.loads(line) for line in (source / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        starts = {e["run_key"] for e in archived_events if e["kind"] == "run_start"}
        ends = {e["run_key"] for e in archived_events if e["kind"] == "run_complete"}
        if starts != ends:
            raise ValueError("Source has an unfinished started run; do not silently repeat it")
        if started_keys & starts:
            raise ValueError("Resume sources contain duplicate started runs")
        started_keys.update(starts)
        previously_responding_models.update(e["model"] for e in archived_events if e["kind"] == "run_start"
                                            and any(r["kind"] == "llm_response" and r["run_key"] == e["run_key"]
                                                    for r in archived_events))
        resume_sources.append({"run_id": source_name, "final_manifest_sha256": file_digest(source / "manifest.final.json"),
                               "started_run_count": len(starts)})
    planned_keys = {f"{i['id']}|{model}" for i in items for model in args.models
                    if f"{model}|{i['id']}|0" not in started_keys}
    if not planned_keys:
        raise ValueError("No unstarted Dev units remain; do not repeat existing runs")
    output = (ROOT / "evaluation/results" / args.run_id).resolve()
    output.relative_to((ROOT / "evaluation/results").resolve())
    output.mkdir(parents=True, exist_ok=False)
    snapshot = code_snapshot()
    source_output = output / "source_snapshot"
    source_output.mkdir()
    for relative in snapshot:
        target = source_output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    packages = {}
    for package in ("httpx", "biopython", "numpy", "fastapi", "pydantic", "ipython", "tqdm"):
        packages[package] = importlib.metadata.version(package)
    if condition in {"SciToolAgent", "F1"}:
        for package in ("llama-index-core", "llama-index-embeddings-huggingface", "sentence-transformers", "torch", "transformers", "pandas"):
            packages[package] = importlib.metadata.version(package)
    elif condition == "Biomni-A1":
        for package in ("langgraph", "langgraph-checkpoint", "langchain-core", "langchain-openai", "pandas"):
            packages[package] = importlib.metadata.version(package)
    manifest = {"schema_version": "external-baselines-dev/1", "status": "running",
                "actual_tool_base": args.tool_base.rstrip("/"),
                "scope": "development_only_not_for_paper_results", "started_at": now(),
                "adapter": adapter_version, "condition": condition,
                "upstream": upstream, "models": args.models,
                "repeats": 1, "planned_units": len(planned_keys),
                "resume_unstarted_sources": resume_sources,
                "planned_run_keys": sorted(planned_keys),
                "question_ids": [i["id"] for i in items], "budget": asdict(budget),
                "dataset_canonical_sha256": specification["dataset_canonical_sha256"],
                "dataset_raw_sha256": file_digest(DATASET), "code_sha256": snapshot,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "python_version": platform.python_version(), "packages": packages,
                "provider_seed_requested": False, "method_core_modified": False,
                "resource_order": "question_then_model", "automatic_infrastructure_retry": False}
    if condition == "F1":
        from scripts.external_baselines.f1 import native_controls
        manifest["budget"] = native_controls()
        manifest["automatic_infrastructure_retry"] = manifest["budget"]["plan_network_max_attempts"] > 1
        manifest["retry_boundary"] = "native initial plan capture network failures only; never content resampling"
    write_new(output / "manifest.start.json", manifest)
    if condition == "SciToolAgent":
        from scripts.external_baselines.scitoolagent import embedding_identity
        write_new(output / "embedding_identity.json", embedding_identity())
    write_new(output / "tools.json", tool_catalog())
    try:
        report = await preflight(args.tool_base.rstrip("/"), output, budget)
        if condition == "CodeAct":
            native_preflight = await codeact_preflight(args.tool_base.rstrip("/"), output, budget)
            print("OFFICIAL_CODEACT_PREFLIGHT", len(native_preflight["successful_tools"]), "/20", flush=True)
        elif condition == "SciToolAgent":
            native_preflight = await scitoolagent_preflight(args.tool_base.rstrip("/"), output, budget)
            print("OFFICIAL_SCITOOLAGENT_PREFLIGHT", len(native_preflight["successful_tools"]), "/20", flush=True)
        elif condition == "Biomni-A1":
            native_preflight = await biomni_preflight(args.tool_base.rstrip("/"), output, budget)
            print("OFFICIAL_BIOMNI_PREFLIGHT", len(native_preflight["successful_tools"]), "/20", flush=True)
        elif condition == "F1":
            from scripts.external_baselines.f1 import f1_preflight
            native_preflight = await f1_preflight(args.tool_base.rstrip("/"), output, budget)
            print("NATIVE_F1_PREFLIGHT", len(native_preflight["successful_tools"]), "/20", flush=True)
    except Exception:
        write_new(output / "manifest.final.json", {**manifest, "status": "preflight_failed", "finished_at": now()})
        raise
    print("PREFLIGHT", len(report["successful_tools"]), "/20", flush=True)
    if args.preflight_only:
        write_new(output / "manifest.final.json", {**manifest, "status": "preflight_only", "finished_at": now()})
        return
    barrier = getattr(args, "calibration_barrier", None)
    if barrier is not None:
        await barrier.wait()
    client = ModelClient()
    results = []
    blocked_models = set()
    for item in items:
        for model in args.models:
            if model in blocked_models:
                continue
            request, files = make_request(item)
            run_key = f"{model}|{item['id']}|0"
            if f"{item['id']}|{model}" not in planned_keys:
                continue
            recorder = Recorder(output / "events.jsonl", run_key)
            facade = ToolFacade(args.tool_base, files, recorder, budget)
            input_files = [{"path": p, "sha256": file_digest(Path(p)),
                            "raw_text": Path(p).read_text(encoding="utf-8-sig")} for p in files]
            semaphore = getattr(args, "unit_semaphore", None)
            if semaphore is not None:
                await semaphore.acquire()
            recorder.record("run_start", question_id=item["id"], model=model, repeat=0,
                            request=request, input_files=input_files)
            start = time.perf_counter()
            try:
                result = await asyncio.wait_for(run_agent(request, model, client, facade, recorder, budget),
                                                timeout=budget.task_timeout_seconds)
            except Exception as exc:
                result = {"final_status": "task_timeout" if isinstance(exc, TimeoutError) else "adapter_error",
                          "final_response": None, "attempted_steps": facade.tool_calls,
                          "latency_ms": (time.perf_counter() - start) * 1000,
                          "llm_calls": sum(e["kind"] == "llm_request" for e in recorder.events)}
                recorder.record("run_error", exception_type=type(exc).__name__, message=str(exc))
            finally:
                if semaphore is not None:
                    semaphore.release()
            try:
                sources = source_index(item, files)
                diagnostic = audit_run(item, result, recorder.events, sources)
            except Exception as exc:
                diagnostic = {"scoring_status": "audit_error", "exception_type": type(exc).__name__}
            responses = [e for e in recorder.events if e["kind"] == "llm_response"]
            usage = {}
            for field in ("input_tokens", "output_tokens", "total_tokens"):
                values = [e["usage"].get(field) for e in responses]
                usage[field] = sum(values) if values and all(isinstance(v, int) for v in values) else None
            row = {"run_key": run_key, "question_id": item["id"], "model": model,
                   "repeat": 0, "condition": condition, "scope": "development_only",
                   "result": result, "audit": diagnostic, "usage": usage,
                   "llm_response_count": len(responses),
                   "event_count": len(recorder.events)}
            recorder.record("run_complete", summary=row)
            with (output / "runs.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(encode(row) + "\n")
            results.append(row)
            print("RUN", len(results), "/", manifest["planned_units"], item["id"], model,
                  result["final_status"], "tools=", result["attempted_steps"], flush=True)
            if responses:
                previously_responding_models.add(model)
            if result["final_status"] == "llm_error" and not responses and model not in previously_responding_models:
                # Avoid 12 identical charges/failures for a persistently unavailable model.
                blocked_models.add(model)
    final = {**manifest, "status": "completed" if len(results) == manifest["planned_units"] else "model_blocked",
             "finished_at": now(), "recorded_units": len(results),
             "blocked_models": sorted(blocked_models), "code_unchanged_during_run": code_snapshot() == snapshot,
             "final_status_counts": {s: sum(r["result"]["final_status"] == s for r in results)
                                     for s in sorted({r["result"]["final_status"] for r in results})}}
    write_new(output / "manifest.final.json", final)
    print("RESULT_DIR", str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-run", action="store_true", required=True)
    parser.add_argument("--condition", choices=("ReAct", "CodeAct", "SciToolAgent", "Biomni-A1", "F1"), default="ReAct")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tool-base", default="http://127.0.0.1:18080")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--limit", type=int, choices=range(1, 13))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-unstarted-from", nargs="+", metavar="FINISHED_RUN_ID",
                        help="Run only units never started in these finished batches; preserve all failures")
    args = parser.parse_args()
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
