"""ReAct mechanism adaptation: one action, real observation, next decision.

Source: ysymyth/ReAct, commit 6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9.
This is a zero-shot protein-tool adaptation, not the original QA experiment.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from .common import Budget, Recorder, ToolFacade, digest, tool_catalog


ADAPTER_VERSION = "react-protein-zero-shot-v1.1"


SYSTEM_PROMPT = """Solve the supplied protein-analysis request by interleaving Thought, Action,
and Observation. Select one action at a time. After the environment returns its
real observation, decide the next action. Use the provided scientific tools.
Use only observations and supplied inputs as evidence for the final answer.
Do not invent inputs, identifiers, tool results, or successful execution.

Return ONE JSON object per turn, with these fields:
{"thought": "brief reasoning", "action": {"name": "exact tool name", "arguments": {}}}
Arguments follow each tool's documented contract. They can contain literal JSON
or an explicit result reference {"$ref": "call_0001#/collection"}; paths are JSON
pointers into that call's exact output. References do not perform type conversion,
select biological objects, or insert tools. Failed outputs remain available.
The reference root is the value of Observation.output, not the outer Observation
wrapper. For a collection, use call_0001#/collection, never
call_0001#/output/collection. For an alignment, use call_0002#/alignment.

You may read a supplied file as raw text using the environment action
read_input_file with arguments file_path, start (default 0), count (default 8192,
maximum 65536). This is file access, not a scientific analysis tool.

To finish, use action name finish and arguments:
{"status": "answered" or "rejected", "response": "your final response"}.
The response may be plain text or a JSON object; include the requested results.
If required information is absent or the requested analysis is unsupported,
explain why and what information the user should supply. Do not guess identity.
Do not write an Observation yourself. No tools other than those listed exist.
"""


def parse_action(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1])
    obj = json.loads(cleaned)
    if not isinstance(obj, dict) or not isinstance(obj.get("thought"), str):
        raise ValueError("Expected a JSON object with thought and action")
    action = obj.get("action")
    if not isinstance(action, dict) or not isinstance(action.get("name"), str):
        raise ValueError("action.name must be a string")
    if not isinstance(action.get("arguments"), dict):
        raise ValueError("action.arguments must be an object")
    return obj


async def run_react(request: dict[str, Any], model: str, client, facade: ToolFacade,
                    recorder: Recorder, budget: Budget) -> dict[str, Any]:
    started = time.perf_counter()
    prompt = "Scientific tools:\n" + json.dumps(tool_catalog(), ensure_ascii=False)
    prompt += "\n\nUser request and supplied input:\n" + json.dumps(request, ensure_ascii=False)
    transcript = ""
    generated = 0
    usage_complete = True
    llm_calls = 0
    status = "llm_budget_exhausted"
    final_response = None
    for turn in range(1, budget.max_llm_calls + 1):
        remaining = budget.max_total_output_tokens - generated
        remaining_time = budget.task_timeout_seconds - (time.perf_counter() - started)
        if remaining_time <= 0:
            status = "task_timeout"
            break
        if remaining <= 0:
            status = "output_budget_exhausted"
            break
        text = prompt + "\n\nTrajectory:\n" + transcript + f"\nTurn {turn}:"
        limit = min(budget.max_output_tokens_per_call, remaining)
        llm_calls += 1
        recorder.record("llm_request", turn=turn, model=model, system_prompt=SYSTEM_PROMPT,
                        user_prompt=text, output_limit=limit, temperature=0.0,
                        thinking_enabled=False, provider_seed_requested=False)
        call_started = time.perf_counter()
        try:
            reply = await asyncio.wait_for(
                client.complete(text, SYSTEM_PROMPT, model, limit,
                                min(budget.llm_timeout_seconds, remaining_time)),
                timeout=min(budget.llm_timeout_seconds, remaining_time),
            )
        except Exception as exc:
            recorder.record("llm_error", turn=turn, exception_type=type(exc).__name__,
                            message=str(exc), latency_ms=(time.perf_counter() - call_started) * 1000)
            status = "llm_error"
            break
        recorder.record("llm_response", turn=turn, **reply,
                        latency_ms=(time.perf_counter() - call_started) * 1000)
        used = reply.get("usage", {}).get("output_tokens")
        if isinstance(used, int) and used >= 0:
            generated += used
        else:
            # Conservative budget accounting; never report this estimate as usage.
            generated += limit
            usage_complete = False
        try:
            parsed = parse_action(reply["content"])
            action = parsed["action"]
            recorder.record("agent_action", turn=turn, **parsed)
        except (ValueError, TypeError) as exc:
            observation = {"status": "error", "code": "ACTION_FORMAT_ERROR",
                           "message": str(exc)}
            recorder.record("action_format_error", turn=turn, observation=observation)
        else:
            name, arguments = action["name"], action["arguments"]
            if name == "finish":
                if arguments.get("status") not in {"answered", "rejected"} or not arguments.get("response"):
                    observation = {"status": "error", "code": "INVALID_FINISH_ACTION"}
                    recorder.record("action_format_error", turn=turn, observation=observation)
                else:
                    status = arguments["status"]
                    final_response = arguments["response"]
                    break
            elif name == "read_input_file":
                observation = facade.read_input_file(arguments)
            else:
                observation = await facade.call(name, arguments)
        recorder.record("observation", turn=turn, observation=observation)
        transcript += ("\nThought/Action " + str(turn) + ": " + reply["content"]
                       + "\nObservation " + str(turn) + ": "
                       + json.dumps(observation, ensure_ascii=False))
    result = {"final_status": status, "final_response": final_response,
              "llm_calls": llm_calls, "output_budget_used": generated,
              "usage_complete": usage_complete, "attempted_steps": facade.tool_calls,
              "latency_ms": (time.perf_counter() - started) * 1000,
              "prompt_sha256": digest(SYSTEM_PROMPT)}
    recorder.record("agent_final", **result)
    return result
