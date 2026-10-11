"""Run existing F1 unchanged in an isolated process, with private passive logs."""
from __future__ import annotations

import asyncio
import contextlib
import io
import os
from pathlib import Path


def worker_main(connection, request, model, metadata, resources):
    def emit(kind, **data):
        connection.send({"kind": "event", "event_kind": kind, "data": data})
        reply = connection.recv()
        if reply.get("error"):
            raise RuntimeError(reply["error"])

    async def transport(url, payload, timeout):
        connection.send({"kind": "model_request", "url": url,
                         "payload": payload, "timeout_seconds": timeout})
        reply = connection.recv()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["value"]

    async def execute():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        torch.set_num_threads(2)
        from app.tools import tool_runner
        from scripts import retrieve_tool_info as native
        from scripts.run_biotooleval import prepare_agent_request, _execute_under_condition, _capture_frozen_plan
        from scripts.initial_artifacts import build_initial_artifacts
        from .f1_audit import observe_f1

        native._cfg.PERSIST_DIR = resources["graph_directory"]
        native._cfg.BGE_MODEL_PATH = resources["embedding_path"]
        native._cfg.LLAMA_INDEX_CACHE_DIR = resources["cache_directory"]
        tool_runner.BIO_API_BASE = resources["tool_base"]
        inp = dict(request["input"])
        inp.pop("file_path", None)
        if request["supplied_files"]:
            inp["fixture"] = str(Path(request["supplied_files"][0]).resolve().relative_to(Path(resources["dataset_directory"]).resolve()))
        item = {"question_zh": request["question"], "input": inp}
        prepared = prepare_agent_request(item, dataset_dir=resources["dataset_directory"])
        if prepared.get("status") != "success":
            raise ValueError("Native request preparation failed: " + str(prepared))
        emit("kernel_ready", protocol="native_f1", request=prepared,
             question_timeout_seconds=120, plan_capture_timeout_seconds=120,
             replan_max_output_tokens=256, replan_max_calls=1)
        # Live requests stay on the original asynchronous HTTP client so native
        # wait_for deadlines can cancel them. IPC is only for local scripted preflight.
        with observe_f1(emit, model_transport=transport if resources["scripted_model"] else None,
                        expected_model=resources["expected_model"], expected_url=resources["expected_api_url"]):
            initial = build_initial_artifacts(prepared["question"], prepared["file_path_list"])
            emit("initial_artifacts", result=initial)
            captured = await _capture_frozen_plan(native.capture_SciToolAgent_plan, prepared,
                experiment_id=metadata["experiment_id"], question_id=metadata["question_id"],
                repeat_index=metadata["repeat"], model=model, timeout_seconds=120)
            emit("initial_plan_capture", result=captured)
            if captured.get("status") != "success":
                return {"capture_result": captured, "execution_result": None}
            result = await _execute_under_condition(native.run_SciToolAgent, native._cfg, prepared,
                condition="F1", timeout_seconds=120, frozen_plan=captured["frozen_plan"], model=model,
                replan_callable=native.capture_SciToolAgent_replan,
                initial_types=initial.get("available_input_types") or [],
                initial_artifact_issue_codes=[str(x["code"]) for x in initial.get("issues", []) if x.get("code")],
                replan_max_output_tokens=256)
            emit("native_execution_complete", result=result)
            return {"capture_result": captured, "execution_result": result}

    console = io.StringIO()
    try:
        with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
            result = asyncio.run(execute())
        connection.send({"kind": "complete", "result": result, "console_output": console.getvalue()})
    except BaseException as exc:
        connection.send({"kind": "error", "exception_type": type(exc).__name__,
                         "message": str(exc), "console_output": console.getvalue()})
    finally:
        connection.close()
