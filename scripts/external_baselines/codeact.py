"""Transport/process adaptation of the unchanged official CodeAct native loop."""
from __future__ import annotations

import asyncio
import multiprocessing
import tempfile
import time

from .common import Budget, Recorder, ToolFacade, tool_catalog
from .codeact_upstream import upstream_identity
from .codeact_worker import worker_main


ADAPTER_VERSION = "codeact-official-m3tooleval-protein-v1.1"


async def run_codeact(request, model, client, facade: ToolFacade,
                      recorder: Recorder, budget: Budget):
    identity = upstream_identity()
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    started = time.perf_counter()
    llm_calls = 0
    generated = 0
    usage_complete = True
    kernel_executions = 0
    final_status = "adapter_error"
    final_response = None
    cell_deadline = None
    native_result = None
    with tempfile.TemporaryDirectory(prefix="bioagent-codeact-") as directory:
        process = context.Process(target=worker_main,
                                  args=(child, request, tool_catalog(), model,
                                        budget.max_llm_calls, directory))
        process.start()
        child.close()
        recorder.record("kernel_start", process_id=process.pid,
                        upstream_commit=identity["commit"], temporary_directory=directory)
        try:
            while True:
                elapsed = time.perf_counter() - started
                if elapsed >= budget.task_timeout_seconds:
                    final_status = "task_timeout"
                    break
                if cell_deadline and time.perf_counter() >= cell_deadline:
                    final_status = "kernel_timeout"
                    break
                if not parent.poll():
                    if not process.is_alive():
                        final_status = "kernel_process_error"
                        break
                    await asyncio.sleep(0.025)
                    continue
                try:
                    message = parent.recv()
                except EOFError:
                    final_status = "kernel_process_error"
                    break
                kind = message.pop("kind")
                if kind == "kernel_llm_request":
                    remaining = budget.max_total_output_tokens - generated
                    if llm_calls >= budget.max_llm_calls or remaining <= 0:
                        final_status = "output_budget_exhausted" if remaining <= 0 else "llm_budget_exhausted"
                        parent.send({"stop": final_status})
                        break
                    llm_calls += 1
                    limit = min(remaining, budget.max_output_tokens_per_call)
                    timeout = min(budget.llm_timeout_seconds,
                                  budget.task_timeout_seconds - elapsed)
                    recorder.record("llm_request", turn=llm_calls, model=model,
                                    messages=message["messages"], output_limit=limit,
                                    temperature=0.0, thinking_enabled=False,
                                    provider_seed_requested=False)
                    call_start = time.perf_counter()
                    try:
                        reply = await asyncio.wait_for(client.complete_messages(
                            message["messages"], model, limit, timeout), timeout=timeout)
                    except Exception as exc:
                        recorder.record("llm_error", turn=llm_calls,
                                        exception_type=type(exc).__name__, message=str(exc),
                                        latency_ms=(time.perf_counter() - call_start) * 1000)
                        final_status = "llm_error"
                        parent.send({"stop": final_status})
                        break
                    recorder.record("llm_response", turn=llm_calls, **reply,
                                    latency_ms=(time.perf_counter() - call_start) * 1000)
                    used = reply.get("usage", {}).get("output_tokens")
                    if isinstance(used, int) and used >= 0:
                        generated += used
                    else:
                        generated += limit
                        usage_complete = False
                    parent.send({"value": reply["content"]})
                elif kind == "kernel_tool_request":
                    response = await facade.call(message["tool_name"], message["arguments"])
                    # A Python variable holds the actual scientific return value,
                    # including errors. No object reconstruction or typed binding.
                    parent.send({"value": response["output"]})
                elif kind == "kernel_input_read":
                    parent.send({"value": facade.read_input_file(message["arguments"])})
                elif kind == "kernel_execution_start":
                    kernel_executions += 1
                    # Tool calls already have their own timeout; allow one code
                    # action to compose up to the remaining scientific calls.
                    cell_deadline = time.perf_counter() + max(30, budget.tool_timeout_seconds)
                    recorder.record(kind, execution_index=kernel_executions, **message)
                elif kind == "kernel_execution_end":
                    cell_deadline = None
                    recorder.record(kind, execution_index=kernel_executions, **message)
                elif kind == "native_loop_complete":
                    native_result = message["result"]
                    recorder.record(kind, **message)
                    # Answer parsing uses the unchanged native Task parser. The
                    # worker's native_is_correct flag means delivery only.
                    answers = [e["parsed"]["content"] for e in recorder.events
                               if e["kind"] == "native_action_parse" and e["parsed"]["type"] == "answer"]
                    final_response = answers[-1] if answers else None
                    final_status = "answer_delivered" if answers else "llm_budget_exhausted"
                    break
                elif kind == "kernel_error":
                    recorder.record(kind, **message)
                    final_status = "adapter_error"
                    break
                else:
                    recorder.record(kind, **message)
        finally:
            # Let the native cleanup finish before forcibly closing a stuck cell.
            await asyncio.to_thread(process.join, 1)
            if process.is_alive():
                process.terminate()
            await asyncio.to_thread(process.join, 5)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 5)
            parent.close()
            recorder.record("kernel_stop", process_id=process.pid, exit_code=process.exitcode)
    result = {"final_status": final_status, "final_response": final_response,
              "final_response_status_unclassified": final_status == "answer_delivered",
              "attempted_steps": facade.tool_calls, "llm_calls": llm_calls,
              "kernel_executions": kernel_executions, "output_budget_used": generated,
              "usage_complete": usage_complete,
              "latency_ms": (time.perf_counter() - started) * 1000,
              "upstream_commit": identity["commit"],
              "scientific_task_success": None, "final_answer_correct": None}
    recorder.record("agent_final", **result)
    return result
