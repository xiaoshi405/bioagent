"""F1 observation adapter: original plan/recovery/executor, original call budgets."""
from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import time
from pathlib import Path

from .common import file_digest
from .f1_worker import worker_main

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_VERSION = "bioagent-f1-native-passive-audit-v1.1"


def native_controls():
    from app.core.config import Config
    c = Config()
    return {"plan_capture_timeout_seconds": 120, "question_timeout_seconds": 120,
            "llm_timeout_seconds": c.LLM_TIMEOUT_SECONDS, "tool_timeout_seconds": 60,
            "replan_max_calls": 1, "replan_max_output_tokens": 256,
            "plan_network_max_attempts": c.EXPERIMENT_PLAN_NETWORK_MAX_ATTEMPTS,
            "external_loop_caps_applied": False, "outer_watchdog_seconds": 600}


def native_identity():
    names = ["scripts/retrieve_tool_info.py", "scripts/hierarchical_recovery.py",
             "scripts/baseline_strategies.py", "scripts/run_biotooleval.py",
             "scripts/typed_planning_runtime.py", "scripts/typed_executor.py",
             "scripts/artifact_pool.py", "scripts/initial_artifacts.py",
             "scripts/structured_plan.py", "scripts/type_checker.py",
             "scripts/plan_sampling.py", "scripts/custom_kg_retrievers.py", "scripts/utils.py",
             "scripts/fix_my_index.py", "app/llms/planning/tool_plan_executor.py",
             "app/llms/planning/prompt_templates.py", "app/llms/extraction/extract_tool.py",
             "app/tools/tool_runner.py", "app/tools/tool_parameters.py", "app/tools/request_guard.py",
             "app/tools/fasta_input_parser.py", "app/core/embedding_runtime.py"]
    return {"method": "existing BioAgent-F1", "native_sources_modified": False,
            "files": {n: file_digest(ROOT/n) for n in names}}


def build_native_graph(out):
    """Original offline registry-to-KG builder; write only a new run resource dir."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    torch.set_num_threads(2)
    from scripts import fix_my_index
    from scripts.external_baselines.scitoolagent import embedding_identity
    embedding = embedding_identity()
    directory = (out/"f1_native_graph").resolve()
    directory.relative_to((ROOT/"evaluation/results").resolve())
    for p in (directory, directory.with_name(directory.name+".building"), directory.with_name(directory.name+".backup")):
        if p.exists():
            raise ValueError("F1 resource directory already exists")
    old_persist, old_bge = fix_my_index._cfg.PERSIST_DIR, fix_my_index._cfg.BGE_MODEL_PATH
    try:
        fix_my_index._cfg.PERSIST_DIR = str(directory)
        fix_my_index._cfg.BGE_MODEL_PATH = embedding["path"]
        result = fix_my_index.refresh_biology_index()
    finally:
        fix_my_index._cfg.PERSIST_DIR, fix_my_index._cfg.BGE_MODEL_PATH = old_persist, old_bge
    if result.get("status") != "success":
        raise RuntimeError("Native F1 offline graph build failed: " + str(result))
    identity = {"builder": "scripts.fix_my_index.refresh_biology_index", "result": result,
                "embedding": embedding, "files": {p.name: file_digest(p) for p in directory.iterdir() if p.is_file()}}
    (out/"f1_resources.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return identity


async def run_f1(request, model, client, facade, recorder, budget):
    # No external 12-call/4096-token cap is imposed on the existing F1 method.
    from app.core.config import Config
    from app.llms.api_clients.openai_api import make_async_request, get_api_url
    started = time.perf_counter()
    directory = recorder.path.parent
    resources = json.loads((directory/"f1_resources.json").read_text(encoding="utf-8"))
    metadata = {"experiment_id": directory.name,
                "question_id": recorder.run_key.split("|")[1], "repeat": int(recorder.run_key.split("|")[2])}
    configuration = {"graph_directory": resources["result"]["persist_dir"],
                     "embedding_path": resources["embedding"]["path"],
                     "cache_directory": str(directory/"f1_embedding_cache"),
                     "dataset_directory": str(ROOT/"evaluation/biotooleval"), "tool_base": facade.base_url,
                     "scripted_model": callable(getattr(client, "complete_native_request", None)),
                     "expected_model": model, "expected_api_url": get_api_url(model).rstrip("/")+"/chat/completions"}
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=worker_main, args=(child, request, model, metadata, configuration))
    process.start()
    child.close()
    recorder.record("kernel_start", process_id=process.pid, protocol="native_f1")
    native = None
    status = "native_incomplete"
    usage_complete = True
    try:
        while True:
            if time.perf_counter()-started >= budget.task_timeout_seconds:
                status = "task_timeout"
                break
            if not parent.poll():
                if not process.is_alive():
                    status = "kernel_process_error"
                    break
                await asyncio.sleep(0.025)
                continue
            try:
                msg = parent.recv()
            except (EOFError, OSError):
                status = "kernel_process_error"
                break
            if msg["kind"] == "event":
                recorder.record(msg["event_kind"], **msg["data"])
                if msg["event_kind"] == "tool_request":
                    facade.tool_calls += 1
                parent.send({"ok": True})
            elif msg["kind"] == "model_request":
                cfg = Config()
                payload = msg["payload"]
                expected = get_api_url(model).rstrip("/")+"/chat/completions"
                if payload["model"] != model or msg["url"] != expected:
                    parent.send({"error": "Native provider/model differs from requested backbone"})
                    continue
                key = cfg.DASHSCOPE_API_KEY if model.startswith("qwen") else cfg.OPENAI_API_KEY
                try:
                    scripted = getattr(client, "complete_native_request", None)
                    raw = (await scripted(msg["url"], payload, msg["timeout_seconds"])
                           if scripted else await make_async_request(msg["url"], key, payload, msg["timeout_seconds"]))
                    parent.send({"value": raw})
                except Exception as exc:
                    usage_complete = False
                    parent.send({"error": str(exc)})
            elif msg["kind"] == "complete":
                native = msg["result"]
                recorder.record("native_f1_complete", **{k:v for k,v in msg.items() if k!="kind"})
                execution = native.get("execution_result") or {}
                if execution.get("status") == "success":
                    status = "answer_delivered"
                else:
                    status = "native_runtime_error" if native.get("execution_result") is not None else "plan_capture_error"
                break
            elif msg["kind"] == "error":
                recorder.record("kernel_error", **{k:v for k,v in msg.items() if k!="kind"})
                status = "adapter_error"
                break
    finally:
        # BGE/torch teardown on Windows takes longer than the lightweight code workers.
        await asyncio.to_thread(process.join, 15 if native is not None else 1)
        if process.is_alive():
            process.terminate()
        await asyncio.to_thread(process.join, 5)
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 5)
        parent.close()
        recorder.record("kernel_stop", process_id=process.pid, exit_code=process.exitcode)
    agent = ((native or {}).get("execution_result") or {}).get("agent_result") or {}
    responses = [e for e in recorder.events if e["kind"] == "llm_response"]
    result = {"final_status": status, "final_response": agent.get("answer"),
              "final_response_status_unclassified": True, "scientific_task_success": None,
              "final_answer_correct": None, "native_execution_summary": (agent.get("reasoning") or {}).get("execution_summary"),
              "native_recovery": (agent.get("reasoning") or {}).get("recovery"),
              "attempted_steps": facade.tool_calls, "llm_calls": sum(e["kind"]=="llm_request" for e in recorder.events),
              "usage_complete": usage_complete and all(all(v is not None for v in e["usage"].values()) for e in responses),
              "latency_ms": (time.perf_counter()-started)*1000, "native_controls": native_controls()}
    recorder.record("agent_final", **result)
    return result


async def f1_preflight(base_url, out, budget):
    """All canonical tools through native typed execution, no cloud model calls."""
    from app.core.config import Config
    from app.tools import tool_runner
    from scripts.typed_planning_runtime import run_typed_planning
    from .common import Recorder, REGISTRY
    from .f1_audit import observe_f1
    graph = build_native_graph(out)
    cfg = Config()
    cfg.TYPE_PLANNING_ENABLED = True
    cfg.TYPE_VALIDATION_ENABLED = True
    cfg.TYPE_REPAIR_ENABLED = True
    fasta = ROOT/"tests/fixtures/fasta/single_protein.fasta"
    multi = ROOT/"tests/fixtures/fasta/multi_protein.fasta"
    pdb = ROOT/"evaluation/biotooleval/fixtures/pdb/mini_two_chain.pdb"
    sequence = "".join(s.strip() for s in fasta.read_text(encoding="utf-8-sig").splitlines() if not s.startswith(">"))
    groups = [
        ("sequence", "结构化氨基酸序列输入：\n```sequence\n"+sequence+"\n```", [],
         [n for n,s in REGISTRY.items() if "sequence" in s["parameters"]]),
        ("single_fasta", "读取上传的单记录FASTA", [str(fasta.resolve())], ["parse_protein_fasta"]),
        ("record", "record_id=protein_b", [str(multi.resolve())], ["parse_multi_protein_fasta", "select_fasta_record"]),
        ("alignment", "record_ids=protein_a,protein_b", [str(multi.resolve())],
         ["parse_multi_protein_fasta", "align_fasta_records", "summarize_pairwise_alignment"]),
        ("pdb", "chain_id=B，计算3个最低频非零ANM模态", [str(pdb.resolve())],
         ["analyze_pdb_structure", "extract_pdb_sequences", "calculate_protein_anm"]),
        ("chain", "chain_id=B", [str(pdb.resolve())], ["select_pdb_chain"]),
    ]
    old_base = tool_runner.BIO_API_BASE
    tool_runner.BIO_API_BASE = base_url
    calls, group_results = [], []
    try:
        for name, question, files, chain in groups:
            record = Recorder(out/"f1_preflight_events.jsonl", "f1_preflight|"+name+"|0")
            def emit(kind, **data):
                if kind == "llm_request":
                    raise AssertionError("Native typed preflight must not call an external model")
                record.record(kind, **data)
            with observe_f1(emit):
                result = await run_typed_planning(chain, question, files, config=cfg)
            group_results.append({"group": name, "completed": result.get("completed"),
                                  "status": result.get("status"), "issues": result.get("issues")})
            calls.extend(e for e in record.events if e["kind"]=="tool_call")
    finally:
        tool_runner.BIO_API_BASE = old_base
    class ScriptedTransport:
        def __init__(self):
            self.responses = iter(["Plan Chain: ['compute_gravy']", "Local native F1 transport preflight completed."])

        async def complete_native_request(self, url, payload, timeout):
            return {"id": "local-scripted-preflight", "response_origin": "local_scripted_preflight",
                    "choices": [{"message": {"content": next(self.responses)}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    from .common import ToolFacade
    record = Recorder(out/"f1_transport_preflight_events.jsonl", "qwen3.7-plus-2026-05-26|preflight_transport|0")
    facade = ToolFacade(base_url, [], record, budget)
    transport_result = await run_f1({"question": "计算该序列的GRAVY", "input": {"input_type": "amino_acid_sequence", "sequence": sequence},
                                   "supplied_files": []}, "qwen3.7-plus-2026-05-26", ScriptedTransport(), facade, record, budget)
    success = sorted({e["tool_name"] for e in calls if e["status"]=="success"})
    report = {"scope": "native_f1_typed_interface_preflight_not_paper_experiment", "external_model_calls": 0,
              "successful_tools": success, "failed_tools": sorted(set(REGISTRY)-set(success)),
              "http_calls": len(calls), "groups": group_results,
              "graph_resources_file": "f1_resources.json", "native_controls": native_controls(),
              "isolated_transport_result": transport_result}
    (out/"f1_preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if (report["failed_tools"] or any(g["status"]!="success" for g in group_results)
            or transport_result["final_status"]!="answer_delivered"
            or not (transport_result["native_execution_summary"] or {}).get("completed")
            or any(e["exit_code"]!=0 for e in record.events if e["kind"]=="kernel_stop")):
        raise RuntimeError("F1 native preflight failed: " + str(report))
    return report
