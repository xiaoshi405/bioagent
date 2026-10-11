"""Check passive hooks preserve native F1 returns, contracts and rejection."""
import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.external_baselines.common import Recorder, REGISTRY
from scripts.external_baselines.f1_audit import observe_f1


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k,v in value.items() if not k.endswith("_ms")}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


class F1PassiveAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_async_request_remains_cancellable_at_deadline(self):
        from app.llms.api_clients import openai_api
        cancelled=[]
        async def pending_request(*args, **kwargs):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
        record=Recorder(None,"offline-deadline")
        with patch.object(openai_api,"make_async_request",pending_request):
            with observe_f1(record.record,expected_model="qwen-test"):
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(openai_api.make_async_chat_with_metadata("offline","qwen-test"),timeout=0.03)
        self.assertEqual(cancelled,[True])
        self.assertEqual(next(e for e in record.events if e["kind"]=="llm_error")["exception_type"],"CancelledError")

    async def test_native_model_payload_and_return_unchanged_no_secret_logged(self):
        from app.llms.api_clients import openai_api
        raw = {"id":"offline-test", "choices":[{"message":{"content":"Plan Chain: ['compute_gravy']"}}],
               "usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}}
        sent=[]
        async def native_request(url, key, payload, timeout_seconds=60):
            sent.append(copy.deepcopy(payload))
            return copy.deepcopy(raw)
        record=Recorder(None,"offline")
        with patch.object(openai_api,"make_async_request",native_request):
            plain=await openai_api.make_async_chat_with_metadata("offline request","qwen-test")
            with observe_f1(record.record):
                logged=await openai_api.make_async_chat_with_metadata("offline request","qwen-test")
        self.assertEqual(plain,logged)
        self.assertEqual(sent[0],sent[1])
        self.assertNotIn("max_tokens",sent[1])
        self.assertEqual(next(e for e in record.events if e["kind"]=="llm_response")["usage"]["total_tokens"],9)
        self.assertNotIn("api_key",json.dumps(record.events))

    async def test_native_replan_has_exact_256_token_cap(self):
        from scripts import retrieve_tool_info as native
        record=Recorder(None,"offline")
        payloads=[]
        async def transport(url,payload,timeout):
            payloads.append(payload)
            return {"choices":[{"message":{"content":"Plan Chain: ['compute_gravy']"}}],
                    "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}
        with observe_f1(record.record,model_transport=transport):
            result=await native.capture_SciToolAgent_replan(question="offline",original_plan_chain=["compute_gravy"],
                available_input_types=["amino acid sequence"],type_issues=[],model="qwen-test",max_output_tokens=256)
        self.assertEqual(result["status"],"success")
        self.assertEqual(len(payloads),1)
        self.assertEqual(payloads[0]["max_tokens"],256)
        self.assertEqual(next(e for e in record.events if e["kind"]=="llm_request")["phase"],"replanning")

    async def test_typed_success_and_failure_returns_preserved(self):
        from scripts.typed_planning_runtime import run_typed_planning
        cfg=SimpleNamespace(TYPE_PLANNING_ENABLED=True,TYPE_VALIDATION_ENABLED=True,TYPE_REPAIR_ENABLED=True)
        for output in [{"status":"success","gravy":-0.49,"sequence_length":20},
                       {"status":"error","message":"offline mock tool failure","code":"OFFLINE_FAILURE"}]:
            async def runner(**kwargs):
                return copy.deepcopy(output)
            args=(["compute_gravy"],"结构化氨基酸序列输入：\n```sequence\nACDEFGHIKLMNPQRSTVWY\n```",[])
            plain=await run_typed_planning(*args,config=cfg,runner=runner)
            record=Recorder(None,"offline")
            with observe_f1(record.record):
                observed=await run_typed_planning(*args,config=cfg,runner=runner)
            self.assertEqual(without_timing(plain),without_timing(observed))
            pool_events=[e for e in record.events if e["kind"]=="artifact_pool_register_tool_outputs"]
            if output["status"]=="success":
                self.assertEqual(len(pool_events),1)
                self.assertTrue(any(a["producer_tool"]=="compute_gravy" and a["value"]==-0.49 for a in pool_events[0]["artifacts"]))
            else:
                self.assertEqual(pool_events,[])
            resolved=next(e for e in record.events if e["kind"]=="artifact_pool_resolve_step_inputs")
            self.assertEqual(resolved["result"]["values"]["amino acid sequence"],"ACDEFGHIKLMNPQRSTVWY")
            self.assertNotIn("ACDEFGHIKLMNPQRSTVWY",json.dumps(observed["safe_trace"]))

    async def test_native_http_payload_error_body_and_return_preserved(self):
        from app import tools
        from app.tools import tool_runner
        class Response:
            def __init__(self,status):self.status=status
            async def text(self):return '{"status":"error","code":"OFFLINE_HTTP_ERROR","message":"offline"}' if self.status!=200 else '{"status":"success","gravy":-0.49,"sequence_length":20}'
            async def json(self):return json.loads(await self.text())
        class Context:
            def __init__(self,status):self.status=status
            async def __aenter__(self):return Response(self.status)
            async def __aexit__(self,*args):return False
        for status in (200,400):
            sent=[]
            def post(session,url,**kwargs):
                sent.append(copy.deepcopy(kwargs["json"]))
                return Context(status)
            args={"tool_name":"compute_gravy","tool_args":"","file_path_list":[],
                  "typed_inputs":{"amino acid sequence":"ACDEFGHIKLMNPQRSTVWY"}}
            record=Recorder(None,"offline")
            with patch.object(tool_runner.aiohttp.ClientSession,"post",post):
                plain=await tools.run_tool(**args)
                with observe_f1(record.record):observed=await tools.run_tool(**args)
            self.assertEqual(plain,observed)
            self.assertEqual(sent[0],sent[1])
            event=next(e for e in record.events if e["kind"]=="tool_call")
            self.assertEqual(event["resolved_arguments"],{"sequence":"ACDEFGHIKLMNPQRSTVWY"})
            self.assertEqual(event["http_status"],status)
            self.assertEqual(event["output"]["status"],"success" if status==200 else "error")

    async def test_ambiguous_request_still_rejected_before_tools(self):
        from pathlib import Path
        from scripts.typed_planning_runtime import run_typed_planning
        cfg=SimpleNamespace(TYPE_PLANNING_ENABLED=True,TYPE_VALIDATION_ENABLED=True,TYPE_REPAIR_ENABLED=True)
        files=[str(Path('evaluation/biotooleval/fixtures/fasta/challenge_multirecord.fasta').resolve())]
        record=Recorder(None,"offline")
        with observe_f1(record.record):
            result=await run_typed_planning(["compute_gravy"],"计算一个GRAVY值",files,config=cfg)
        self.assertFalse(result["completed"])
        self.assertFalse(any(e["kind"] in {"tool_request","llm_request"} for e in record.events))


if __name__=="__main__":unittest.main()
