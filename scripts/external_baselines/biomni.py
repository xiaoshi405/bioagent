"""Provider and process transport for the original Biomni A1 graph."""
from __future__ import annotations

import asyncio
import multiprocessing
import tempfile
import time

from .biomni_upstream import upstream_identity
from .biomni_worker import worker_main
from .common import tool_catalog

ADAPTER_VERSION = "biomni-a1-official-v0.0.8-protein-v1"


async def run_biomni(request, model, client, facade, recorder, budget):
    identity = upstream_identity()
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    started = time.perf_counter()
    llm_calls = generated = kernel_executions = 0
    usage_complete = True
    final_status = "adapter_error"
    final_response = None
    cell_deadline = None
    with tempfile.TemporaryDirectory(prefix="bioagent-biomni-") as directory:
        process = context.Process(target=worker_main,
            args=(child, request, tool_catalog(), model, budget.max_llm_calls, directory))
        process.start()
        child.close()
        recorder.record("kernel_start", process_id=process.pid, upstream_commit=identity["commit"],
                        temporary_directory=directory)
        try:
            while True:
                elapsed = time.perf_counter() - started
                if elapsed >= budget.task_timeout_seconds:
                    final_status = "task_timeout"
                    break
                if cell_deadline and time.perf_counter() >= cell_deadline:
                    final_status = "kernel_timeout"
                    break
                try:
                    has_message = parent.poll()
                except (OSError, EOFError):
                    final_status = "kernel_process_error"
                    break
                if not has_message:
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
                        final_status = "llm_budget_exhausted" if remaining > 0 else "output_budget_exhausted"
                        parent.send({"stop": final_status})
                        break
                    llm_calls += 1
                    limit = min(budget.max_output_tokens_per_call, remaining)
                    timeout = min(budget.llm_timeout_seconds, budget.task_timeout_seconds - elapsed)
                    recorder.record("llm_request", turn=llm_calls, model=model, **message,
                        output_limit=limit, temperature=0.0, thinking_enabled=False, provider_seed_requested=False)
                    call_start = time.perf_counter()
                    try:
                        reply = await asyncio.wait_for(client.complete_messages(message["messages"], model, limit,
                            timeout, stop=message["stop_sequences"]), timeout=timeout)
                    except Exception as exc:
                        usage_complete = False
                        recorder.record("llm_error", turn=llm_calls, exception_type=type(exc).__name__,
                            message=str(exc), latency_ms=(time.perf_counter()-call_start)*1000)
                        final_status = "llm_error"
                        parent.send({"stop": final_status})
                        break
                    recorder.record("llm_response", turn=llm_calls, **reply,
                                    latency_ms=(time.perf_counter()-call_start)*1000)
                    used = reply.get("usage", {}).get("output_tokens")
                    if isinstance(used, int) and used >= 0:
                        generated += used
                    else:
                        generated += limit
                        usage_complete = False
                    parent.send({"value": reply["content"]})
                elif kind == "kernel_tool_request":
                    if facade.tool_calls >= budget.max_tool_calls:
                        final_status = "tool_budget_exhausted"
                        parent.send({"stop": final_status})
                        break
                    response = await facade.call(message["tool_name"], message["arguments"])
                    parent.send({"value": response["output"]})
                elif kind == "kernel_input_read":
                    parent.send({"value": facade.read_input_file(message["arguments"])})
                elif kind == "kernel_execution_start":
                    kernel_executions += 1
                    cell_deadline = time.perf_counter() + max(30, budget.tool_timeout_seconds)
                    recorder.record(kind, execution_index=kernel_executions, **message)
                elif kind == "kernel_execution_end":
                    cell_deadline = None
                    recorder.record(kind, execution_index=kernel_executions, **message)
                elif kind == "native_biomni_complete":
                    recorder.record(kind, **message)
                    final_response = message["final_solution"]
                    final_status = "answer_delivered" if final_response is not None else "native_incomplete"
                    break
                elif kind == "kernel_error":
                    recorder.record(kind, **message)
                    final_status = "adapter_error"
                    break
                else:
                    recorder.record(kind, **message)
        finally:
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
        "final_response_status_unclassified": True, "attempted_steps": facade.tool_calls,
        "llm_calls": llm_calls, "kernel_executions": kernel_executions,
        "output_budget_used": generated, "usage_complete": usage_complete,
        "latency_ms": (time.perf_counter()-started)*1000, "upstream_commit": identity["commit"],
        "scientific_task_success": None, "final_answer_correct": None}
    recorder.record("agent_final", **result)
    return result
