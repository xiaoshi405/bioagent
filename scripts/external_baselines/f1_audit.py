"""Private passive observation of the existing F1 runtime; no decision changes."""
from __future__ import annotations

import copy
import json
import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from functools import wraps
from unittest.mock import patch

from .common import digest


def normalized_usage(response):
    usage = response.get("usage") or {}
    return {"input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
            "total_tokens": usage.get("total_tokens")}


@contextmanager
def observe_f1(emit, *, model_transport=None, expected_model=None, expected_url=None):
    """Temporarily wrap native functions, returning their exact original results.

    Used within one isolated worker. No credential is passed to emit/transport.
    Public safe_trace and ArtifactPool.snapshot remain unchanged.
    """
    from app.llms.api_clients import openai_api
    from app import tools
    from app.tools import tool_runner
    from scripts import retrieve_tool_info as native
    from scripts import typed_planning_runtime as typed
    from scripts.artifact_pool import ArtifactPool

    phase = ContextVar("f1_audit_phase", default="planning")
    current = ContextVar("f1_audit_tool", default=None)
    count = 0
    original_request = openai_api.make_async_request
    original_runner = tools.run_tool
    original_post = tool_runner.aiohttp.ClientSession.post

    async def request(url, api_key, payload, timeout_seconds=60.0):
        if expected_model is not None and payload["model"] != expected_model:
            raise ValueError("Native F1 request uses a different model")
        if expected_url is not None and url != expected_url:
            raise ValueError("Native F1 request uses a different provider endpoint")
        start = time.perf_counter()
        emit("llm_request", phase=phase.get(), endpoint=url, payload=payload,
             messages=payload["messages"], model=payload["model"],
             output_limit=payload.get("max_tokens"),
             temperature=payload.get("temperature"),
             thinking_enabled=payload.get("enable_thinking"),
             provider_seed_requested="seed" in payload,
             native_timeout_seconds=timeout_seconds)
        try:
            result = (await model_transport(url, payload, timeout_seconds)
                      if model_transport is not None else
                      await original_request(url, api_key, payload, timeout_seconds))
        except BaseException as exc:
            emit("llm_error", phase=phase.get(), exception_type=type(exc).__name__,
                 message=str(exc), latency_ms=(time.perf_counter()-start)*1000)
            raise
        choice = (result.get("choices") or [{}])[0]
        emit("llm_response", phase=phase.get(), content=(choice.get("message") or {}).get("content"),
             raw_provider_response=result, usage=normalized_usage(result),
             request_id=result.get("request_id", result.get("id")),
             finish_reason=choice.get("finish_reason"),
             response_sha256=digest(result), latency_ms=(time.perf_counter()-start)*1000)
        return result

    @wraps(original_runner)
    async def runner(*args, **kwargs):
        nonlocal count
        count += 1
        name = kwargs.get("tool_name", args[0] if args else None)
        call = {"call_id": f"call_{count:04d}", "tool_name": name,
                "typed_inputs": copy.deepcopy(kwargs.get("typed_inputs"))}
        token = current.set(call)
        emit("native_runner_start", **call, native_arguments=kwargs)
        try:
            result = await original_runner(*args, **kwargs)
            emit("native_runner_end", **call, raw_result=result)
            return result
        except BaseException as exc:
            emit("native_runner_error", **call, exception_type=type(exc).__name__)
            raise
        finally:
            current.reset(token)

    class ObservedPost:
        def __init__(self, context, url, payload, call):
            self.context, self.url, self.payload, self.call = context, url, payload, call
            self.response = None
            self.start = time.perf_counter()

        async def __aenter__(self):
            emit("tool_request", **self.call, endpoint=self.url,
                 proposed_arguments=self.payload, resolved_arguments=self.payload,
                 reference_bindings=[], input_sha256=digest(self.payload))
            try:
                self.response = await self.context.__aenter__()
                return self.response
            except BaseException as exc:
                emit("tool_call", **self.call, http_requested=True, http_status=None,
                     status="transport_error", error_kind=type(exc).__name__,
                     proposed_arguments=self.payload, resolved_arguments=self.payload,
                     reference_bindings=[], output=None, latency_ms=(time.perf_counter()-self.start)*1000)
                raise

        async def __aexit__(self, exc_type, exc, tb):
            try:
                body = await self.response.text()
                try:
                    output = json.loads(body)
                except ValueError:
                    output = {"raw_response": body}
                ok = self.response.status == 200 and isinstance(output, dict) and output.get("status") == "success"
                emit("tool_call", **self.call, http_requested=True, http_status=self.response.status,
                     status="success" if ok else "tool_error",
                     error_kind="service_error" if self.response.status >= 500 else None,
                     proposed_arguments=self.payload, resolved_arguments=self.payload,
                     reference_bindings=[], output=output, raw_response=body,
                     input_sha256=digest(self.payload), output_sha256=digest(output),
                     latency_ms=(time.perf_counter()-self.start)*1000)
            finally:
                return await self.context.__aexit__(exc_type, exc, tb)

    def post(session, url, *args, **kwargs):
        call = current.get()
        context = original_post(session, url, *args, **kwargs)
        return ObservedPost(context, url, copy.deepcopy(kwargs.get("json")), call) if call else context

    def pool_wrapper(name):
        original = getattr(ArtifactPool, name)
        @wraps(original)
        def observed(pool, *args, **kwargs):
            result = original(pool, *args, **kwargs)
            emit("artifact_pool_" + name, arguments=args, keyword_arguments=kwargs,
                 result=result, artifacts=[asdict(a) for a in pool._artifacts])
            return result
        return observed

    original_typed = native.run_typed_planning
    @wraps(original_typed)
    async def typed_runtime(*args, **kwargs):
        result = await original_typed(*args, **kwargs)
        emit("typed_runtime_result", result=result)
        return result

    def with_phase(function, label):
        @wraps(function)
        async def observed(*args, **kwargs):
            token = phase.set(label)
            try:
                return await function(*args, **kwargs)
            finally:
                phase.reset(token)
        return observed

    with ExitStack() as stack:
        for target, name, replacement in [
            (openai_api, "make_async_request", request),
            (tools, "run_tool", runner),
            (tool_runner.aiohttp.ClientSession, "post", post),
            (native, "run_typed_planning", typed_runtime),
            (native, "generate_output", with_phase(native.generate_output, "summarization")),
            (native, "capture_SciToolAgent_replan", with_phase(native.capture_SciToolAgent_replan, "replanning")),
        ]:
            stack.enter_context(patch.object(target, name, replacement))
        for name in ("register_initial", "resolve_step_inputs", "register_tool_outputs"):
            stack.enter_context(patch.object(ArtifactPool, name, pool_wrapper(name)))
        yield
