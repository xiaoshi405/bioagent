"""Official SciToolAgent KG retrieval / planner / executor / summarizer adapter."""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from .common import Budget, Recorder, ToolFacade, digest, encode, file_digest, tool_catalog
from .scitoolagent_upstream import load_core, upstream_identity

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_VERSION = "scitoolagent-official-protein-v1.1-resource"
MODEL_PATH = ROOT.parent / "bge_model"
_CORE = None
_EMBED = None
_LOCK = threading.Lock()
# Inference on repeatedly recreated threads retains private commit on this
# Windows CPU runtime. Keep native retrieval pools, but run BGE on one thread.
_INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scitoolagent-bge")


def infer_embedding(embedding, text, query=False):
    infer = embedding.get_query_embedding if query else embedding.get_text_embedding
    return _INFERENCE_EXECUTOR.submit(infer, text).result()


class BudgetStop(BaseException):
    """Stop official retries at the shared budget boundary, without hidden calls."""

    def __init__(self, status):
        self.status = status


def embedding_identity():
    files = [p for p in sorted(MODEL_PATH.rglob("*")) if p.is_file()
             and ".runtime_cache" not in p.parts]
    if not (MODEL_PATH / "model.safetensors").is_file():
        raise ValueError("Local embedding weights unavailable")
    return {"path": str(MODEL_PATH), "config_model_name": json.loads(
        (MODEL_PATH / "config.json").read_text())["_name_or_path"],
        "files": {str(p.relative_to(MODEL_PATH)): file_digest(p) for p in files},
        "device": "cpu", "network_calls": 0, "adaptation": "local BGE replaces upstream OpenAIEmbedding"}


def get_core():
    global _CORE, _EMBED
    if _CORE is None:
        _CORE = load_core()
    if _EMBED is None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        torch.set_num_threads(2)
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        _EMBED = HuggingFaceEmbedding(model_name=str(MODEL_PATH), device="cpu", embed_batch_size=8)
    return _CORE, _EMBED


def graph_rows(catalog=None):
    """Public tool topology only; original create_triplets builds all edges."""
    return [{"Name": t["name"], "Category": "protein analysis",
             "Function": t["description"],
             "Input": ",".join(t.get("input_types") or ["JSON arguments"]),
             "Output": ",".join(t.get("output_types") or [t["return_type"]]),
             "Source": "shared 20-tool scientific backend", "Safety": "Yes"}
            for t in (catalog or tool_catalog())]


def parse_literal_arguments(text, spec):
    """Serialization only: no sequence selection, inferred IDs or conversion."""
    raw = text.strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        # Official single string input remains usable for one required parameter.
        parameters = spec["parameters"]
        required = [p for p in parameters if not spec.get("parameter_schema", {}).get(p, {}).get("optional", False)]
        if len(required) != 1:
            raise ValueError("Expected one JSON object with documented argument names")
        return {required[0]: raw}
    if not isinstance(value, dict):
        raise ValueError("Expected JSON argument object")
    # References are deliberately unavailable here: the official executor
    # reconstructs literal arguments from previous tool output in its own prompt.
    def has_ref(v):
        return isinstance(v, dict) and ("$ref" in v or any(has_ref(x) for x in v.values())) or isinstance(v, list) and any(has_ref(x) for x in v)
    if has_ref(value):
        raise ValueError("SciToolAgent arguments must be literal; references are not its native API")
    return value


async def run_scitoolagent(request, model, client, facade: ToolFacade, recorder: Recorder, budget: Budget):
    started = time.perf_counter()
    _, embedding = get_core()
    # Functions retain their original global dict, so load a fresh function set
    # for each request. This avoids carrying model/output state across requests.
    retriever_class, ns = load_core()
    catalog = {t["name"]: t for t in tool_catalog()}
    generated = 0
    llm_calls = 0
    status = "native_incomplete"
    final_response = None
    usage_complete = True
    phase = "planning"
    from llama_index.core import Settings
    from llama_index.core.llms import MockLLM
    from llama_index.core.embeddings import BaseEmbedding
    from pydantic import PrivateAttr

    class LoggedEmbedding(BaseEmbedding):
        _cache: dict = PrivateAttr(default_factory=dict)

        def _vector(self, text, query=False):
            with _LOCK:
                key = (text, query)
                cached = key in self._cache
                if not cached:
                    self._cache[key] = infer_embedding(embedding, text, query)
                result = self._cache[key]
                recorder.record("embedding", text=text, query=query, cache_hit=cached,
                                dimension=len(result), vector_sha256=digest(result), backend="local_bge")
                return list(result)

        def _get_text_embedding(self, text):
            return self._vector(text)

        def _get_query_embedding(self, text):
            return self._vector(text, True)

        async def _aget_query_embedding(self, text):
            return self._vector(text, True)

    Settings.llm = MockLLM()
    Settings.embed_model = LoggedEmbedding(model_name="local-bge-small-zh")
    graph_directory = recorder.path.parent / "scitoolagent_graphs" / digest(recorder.run_key)[:20]
    graph_directory.mkdir(parents=True, exist_ok=False)
    import pandas as pd
    rows = graph_rows(list(catalog.values()))
    triplets = ns["create_triplets"](pd.DataFrame(rows))
    recorder.record("kg_build", rows=rows, triplets=triplets,
                    public_catalog_sha256=digest(list(catalog.values())), scope="public_metadata_only")
    ns["build_knowledge_graph"](triplets, str(graph_directory))

    class LoggedRetriever(retriever_class):
        async def _retrieve(self, query):
            result = await super()._retrieve(query)
            recorder.record("kg_retrieval", query=query.query_str, triplets=result)
            return result

        def batch_get_text_embeddings(self, text_list, max_workers=5):
            # Preserve original batch implementation; cap only worker resources.
            return super().batch_get_text_embeddings(text_list, max_workers=min(max_workers, 5))

        def get_tool_info(self, subgraph):
            info = super().get_tool_info(subgraph)
            names = [t[0] for t in subgraph if t[1] == "has the functionality that" and t[0] in catalog]
            if not names:
                raise ValueError("Unknown tool in official executor")
            spec = catalog[names[0]]
            # A single JSON object is the argument of the official /run-func
            # interface, preserving its Input: extraction protocol.
            info["input_format"] = "one JSON object containing the named API arguments: " + encode({
                "parameters": spec["parameters"], "parameter_schema": spec.get("parameter_schema", {})})
            info["output_format"] = encode({"scientific_output_types": spec.get("output_types", []),
                                           "return_fields": spec.get("observed_success_output_fields", [])})
            recorder.record("native_tool_info", tool=names[0], info=info)
            return info

    async def chat(prompt, model=None, **kwargs):
        nonlocal llm_calls, generated, usage_complete
        if llm_calls >= budget.max_llm_calls:
            raise BudgetStop("llm_budget_exhausted")
        remaining = budget.max_total_output_tokens - generated
        if remaining <= 0:
            raise BudgetStop("output_budget_exhausted")
        if model != selected_model:
            raise ValueError("Unexpected upstream model selection")
        limit = min(budget.max_output_tokens_per_call, remaining)
        llm_calls += 1
        system = "You are ChatGPT, a large language model trained by OpenAI.\n"
        recorder.record("llm_request", turn=llm_calls, model=model, system_prompt=system,
                        user_prompt=prompt, output_limit=limit, temperature=0.0,
                        thinking_enabled=False, provider_seed_requested=False, phase=phase)
        call_start = time.perf_counter()
        try:
            reply = await client.complete(prompt, system, model, limit, budget.llm_timeout_seconds)
        except Exception as exc:
            recorder.record("llm_error", turn=llm_calls, exception_type=type(exc).__name__, message=str(exc))
            raise BudgetStop("llm_error")
        recorder.record("llm_response", turn=llm_calls, **reply,
                        latency_ms=(time.perf_counter()-call_start)*1000, phase=phase)
        used = reply.get("usage", {}).get("output_tokens")
        if isinstance(used, int):
            generated += used
        else:
            generated += limit
            usage_complete = False
        return reply["content"]

    async def run_tool(name, arguments, files):
        if facade.tool_calls >= budget.max_tool_calls:
            raise BudgetStop("tool_budget_exhausted")
        try:
            spec = catalog[name]
            parsed = parse_literal_arguments(arguments, spec)
        except (KeyError, ValueError, TypeError) as exc:
            recorder.record("native_argument_error", tool_name=name, raw_input=arguments, message=str(exc))
            return "Tool function execution error," + str(exc)
        recorder.record("native_tool_input", tool_name=name, raw_input=arguments, literal_arguments=parsed)
        result = await facade.call(name, parsed)
        if result["status"] != "success":
            return "Tool function execution error," + encode(result["output"])
        return encode(result["output"])

    original_path_parser = ns["extract_tool_path"]
    def path_parser(text):
        nonlocal phase
        path = original_path_parser(text)
        recorder.record("native_plan", raw_response=text, tool_path=path)
        phase = "execution"
        return path

    original_answer_parser = ns["extract_answer_parts"]
    def answer_parser(text):
        result = original_answer_parser(text)
        recorder.record("native_summary", text=text, is_completed=result[0], final_answer=result[1])
        return result

    original_output = ns["generate_output"]
    async def output(*args, **kwargs):
        nonlocal phase
        phase = "summary"
        return await original_output(*args, **kwargs)

    original_plan = ns["generate_tools_plan"]
    async def plan(*args, **kwargs):
        nonlocal phase
        phase = "planning"
        return await original_plan(*args, **kwargs)

    selected_model = model
    ns.update(make_async_chat=chat, run_tool=run_tool, KGTableRetriever=LoggedRetriever,
              Config=lambda: SimpleNamespace(PERSIST_DIR=str(graph_directory), MODEL_NAME=model),
              asyncio=asyncio, extract_tool_path=path_parser, extract_answer_parts=answer_parser,
              generate_output=output, generate_tools_plan=plan)
    # The 20 analysis APIs retain their own common safety checks. No added
    # chemical generation tools are present; all KG rows are read-only analysis.
    # Those hooks are unreachable with Safety=Yes. Fail loudly if ever called,
    # rather than silently changing upstream checks after a catalog change.
    def unexpected_safety(*args):
        raise ValueError("Additional upstream safety domain is not configured for this analysis-only catalog")
    ns["check_toxicity"] = unexpected_safety
    question = request["question"] + "\nSupplied input:\n" + encode(request["input"])
    recorder.record("native_run_start", upstream=upstream_identity(), model=model,
                    question=question, files=request["supplied_files"])
    try:
        final_response = await ns["run_SciToolAgent"](question, request["supplied_files"], False)
        summaries = [e for e in recorder.events if e["kind"] == "native_summary"]
        status = "answer_delivered" if summaries and summaries[-1]["is_completed"] == "Completed" else "native_incomplete"
    except BudgetStop as exc:
        status = exc.status
    result = {"final_status": status, "final_response": final_response,
              "final_response_status_unclassified": True, "llm_calls": llm_calls,
              "output_budget_used": generated, "usage_complete": usage_complete,
              "attempted_steps": facade.tool_calls, "latency_ms": (time.perf_counter()-started)*1000,
              "native_graph_directory": str(graph_directory)}
    recorder.record("agent_final", **result)
    return result
