"""Shared literal tool transport, explicit references and private audit logs.

No typed binding, biological selection, bridge insertion or plan validation.
Call IDs identify log entries, not scientific artifacts. Error results remain
available for explicit references, just as ordinary program variables would.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from ToolsAgent.ToolsFuns.Biology.tool_name_dict import BIOLOGY_TOOLS_DESC


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def file_digest(path: Path, *, canonical: bool = False) -> str:
    data = path.read_bytes()
    if canonical:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


REGISTRY = {
    name: spec for name, spec in BIOLOGY_TOOLS_DESC.items()
    if not spec.get("is_internal") and spec.get("api_path")
}


def tool_catalog() -> list[dict[str, Any]]:
    """Equivalent documentation; exclude BioAgent-specific recovery metadata."""
    fields = ("description", "parameters", "parameter_schema", "return_type",
              "input_types", "output_types")
    public_path = Path(__file__).resolve().parents[2] / "evaluation/biotooleval/external_tools_public_returns_v1.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))["tools"] if public_path.is_file() else {}
    return [{"name": name, **{k: copy.deepcopy(spec[k]) for k in fields if k in spec},
             **copy.deepcopy(public.get(name, {}))} for name, spec in REGISTRY.items()]


@dataclass(frozen=True)
class Budget:
    max_llm_calls: int = 12
    max_tool_calls: int = 12
    max_output_tokens_per_call: int = 4096
    max_total_output_tokens: int = 16384
    llm_timeout_seconds: float = 120
    tool_timeout_seconds: float = 120
    task_timeout_seconds: float = 600

    def __post_init__(self):
        if any(v <= 0 for v in asdict(self).values()):
            raise ValueError("All budget values must be positive")


class Recorder:
    """Append each event immediately; a crash cannot erase earlier tool calls."""
    def __init__(self, path: Path | None, run_key: str):
        self.path = path
        self.run_key = run_key
        self.events: list[dict[str, Any]] = []

    def record(self, kind: str, **data: Any) -> dict[str, Any]:
        row = {"run_key": self.run_key, "event_index": len(self.events),
               "recorded_at": now(), "kind": kind, **copy.deepcopy(data)}
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encode(row) + "\n")
                handle.flush()
        self.events.append(row)
        return row


class ToolFacade:
    def __init__(self, base_url: str, files: list[str], recorder: Recorder,
                 budget: Budget, *, transport=None):
        self.base_url = base_url.rstrip("/")
        self.files = {Path(p).resolve() for p in files}
        self.recorder = recorder
        self.budget = budget
        self.transport = transport
        self.outputs: dict[str, Any] = {}
        self.tool_calls = 0

    def resolve(self, value: Any, bindings: list, location: str = "") -> Any:
        if isinstance(value, dict) and set(value) == {"$ref"}:
            ref = value["$ref"]
            if not isinstance(ref, str) or "#" not in ref:
                raise ValueError("Reference must be call_id#/json/pointer")
            call_id, pointer = ref.split("#", 1)
            if call_id not in self.outputs:
                raise ValueError("Reference points to an unavailable call")
            result = self.outputs[call_id]
            if pointer and not pointer.startswith("/"):
                raise ValueError("JSON pointer must start with /")
            for raw_part in pointer.split("/")[1:] if pointer else []:
                part = raw_part.replace("~1", "/").replace("~0", "~")
                result = result[int(part)] if isinstance(result, list) else result[part]
            bindings.append({"argument_path": location, "call_id": call_id,
                             "output_pointer": pointer})
            return copy.deepcopy(result)
        if isinstance(value, dict):
            return {k: self.resolve(v, bindings, location + "/" + k)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve(v, bindings, location + "/" + str(i))
                    for i, v in enumerate(value)]
        return copy.deepcopy(value)

    async def call(self, name: str, arguments: Any) -> dict[str, Any]:
        call_id = f"call_{len(self.outputs) + 1:04d}"
        started = time.perf_counter()
        bindings: list[dict[str, Any]] = []
        resolved = None
        http_status = None
        http_requested = False
        error_kind = None
        try:
            if name not in REGISTRY:
                raise ValueError("Unknown or internal tool name")
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            resolved = self.resolve(arguments, bindings)
            if "file_path" in resolved:
                if not isinstance(resolved["file_path"], str):
                    raise ValueError("file_path must be a string")
                if Path(resolved["file_path"]).resolve() not in self.files:
                    raise ValueError("File is not among this request's supplied files")
            if self.tool_calls >= self.budget.max_tool_calls:
                error_kind = "tool_budget_exhausted"
                raise ValueError("Scientific tool call budget exhausted")
            self.tool_calls += 1
            http_requested = True
            self.recorder.record("tool_request", call_id=call_id, tool_name=name,
                                 proposed_arguments=arguments, resolved_arguments=resolved,
                                 reference_bindings=bindings, input_sha256=digest(resolved))
            if self.transport is not None:
                http_status, output = await self.transport(name, resolved)
            else:
                async with httpx.AsyncClient(timeout=self.budget.tool_timeout_seconds,
                                             trust_env=False) as client:
                    response = await client.post(self.base_url + REGISTRY[name]["api_path"],
                                                 json=resolved)
                http_status = response.status_code
                try:
                    output = response.json()
                except ValueError:
                    output = {"status": "error", "code": "NON_JSON_TOOL_RESPONSE",
                              "raw_response": response.text}
            success = http_status == 200 and isinstance(output, dict) and output.get("status") == "success"
            status = "success" if success else "tool_error"
            if http_status >= 500:
                error_kind = "service_error"
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            status = "transport_error"
            error_kind = type(exc).__name__
            output = {"status": "error", "code": "TOOL_TRANSPORT_ERROR",
                      "exception_type": type(exc).__name__}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            status = "interface_error"
            error_kind = error_kind or "invalid_action_arguments"
            output = {"status": "error", "code": "ACTION_ARGUMENT_ERROR",
                      "message": str(exc)}
        self.outputs[call_id] = copy.deepcopy(output)
        self.recorder.record(
            "tool_call", call_id=call_id, tool_name=name,
            proposed_arguments=arguments, resolved_arguments=resolved,
            reference_bindings=bindings, http_requested=http_requested,
            http_status=http_status, status=status, error_kind=error_kind,
            output=output, input_sha256=digest(resolved), output_sha256=digest(output),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return {"call_id": call_id, "status": status, "output": output}

    def read_input_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Raw resource access, comparable to CodeAct's ordinary file reading."""
        try:
            path = Path(arguments["file_path"]).resolve()
            if path not in self.files:
                raise ValueError("File is not among the supplied input files")
            start = int(arguments.get("start", 0))
            count = int(arguments.get("count", 8192))
            if start < 0 or count < 1 or count > 65536:
                raise ValueError("start >= 0; 1 <= count <= 65536")
            text = path.read_text(encoding="utf-8-sig")
            output = {"status": "success", "file_path": str(path),
                      "file_sha256": file_digest(path), "start": start,
                      "end": min(start + count, len(text)), "total_characters": len(text),
                      "content": text[start:start + count]}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            output = {"status": "error", "code": "INPUT_READ_ERROR",
                      "message": str(exc)}
        self.recorder.record("input_read", arguments=arguments, output=output)
        return output


class ModelClient:
    """Reuse the existing HTTP client; explicit Dev controls, no .env edits."""
    async def complete(self, text: str, system_prompt: str, model: str,
                       output_limit: int, timeout: float) -> dict[str, Any]:
        return await self.complete_messages(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
            model, output_limit, timeout)

    async def complete_messages(self, messages: list[dict], model: str,
                                output_limit: int, timeout: float, *, stop=None) -> dict[str, Any]:
        from app.core.config import Config
        from app.llms.api_clients.openai_api import make_async_request, supports_explicit_thinking_toggle
        cfg = Config()
        payload = {"model": model, "temperature": 0.0,
                   "max_tokens": output_limit,
                   "messages": copy.deepcopy(messages)}
        if stop is not None:
            payload["stop"] = copy.deepcopy(stop)
        if supports_explicit_thinking_toggle(model):
            payload["enable_thinking"] = False
        response = await make_async_request(cfg.API_BASE.rstrip("/") + "/chat/completions",
                                            cfg.API_KEY, payload, timeout_seconds=timeout)
        try:
            message = response["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Model response is missing its message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Model returned empty content")
        usage = response.get("usage") or {}
        return {"content": content,
                "usage": {"input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
                          "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
                          "total_tokens": usage.get("total_tokens")},
                "request_id": response.get("request_id") or response.get("id"),
                "provider_model": response.get("model"),
                "finish_reason": response["choices"][0].get("finish_reason"),
                "provider_usage": usage, "response_sha256": digest(response)}
