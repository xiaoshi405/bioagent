# -*- coding: utf-8 -*-
"""LLM客户端离线单元测试，不发送真实网络请求。

修改记录：
- 2026-09-08：覆盖百炼 DeepSeek-V4 快照的非思考参数保真。
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.llms.api_clients import openai_api


class DialogTests(unittest.TestCase):
    """验证System Prompt不会被客户端覆盖。"""

    def test_custom_system_prompt_is_preserved(self):
        dialogs = openai_api.make_simple_dialogs(
            "规划蛋白质工具", system_prompt="只能使用已注册工具"
        )
        self.assertEqual(dialogs[0]["content"], "只能使用已注册工具")
        self.assertEqual(dialogs[1]["content"], "规划蛋白质工具")

    def test_default_system_prompt_is_used_when_empty(self):
        dialogs = openai_api.make_simple_dialogs("测试问题")
        self.assertEqual(dialogs[0]["content"], openai_api.DEFAULT_SYSTEM_PROMPT)

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            openai_api.make_simple_dialogs("   ")


class AsyncChatTests(unittest.IsolatedAsyncioTestCase):
    """验证模型配置进入请求体且能解析OpenAI兼容响应。"""

    async def test_qwen_payload_contains_reproducibility_settings(self):
        fake_response = {
            "choices": [{"message": {"content": "Plan Chain:\n['ComputePiMw']"}}],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 8,
                "total_tokens": 28,
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "request_id": "offline-test",
        }
        env = {
            "API_BASE": "https://example.invalid/compatible-mode/v1",
            "API_KEY": "test-key",
            "MODEL_NAME": "qwen3.7-plus-2026-05-26",
            "LLM_ENABLE_THINKING": "false",
            "LLM_TEMPERATURE": "0",
            "LLM_TIMEOUT_SECONDS": "60",
        }

        with patch.dict(os.environ, env, clear=False):
            with patch.object(
                openai_api,
                "make_async_request",
                new=AsyncMock(return_value=fake_response),
            ) as request_mock:
                result = await openai_api.make_async_chat(
                    "计算pI", model="qwen3.7-plus-2026-05-26"
                )

        self.assertIn("ComputePiMw", result)
        payload = request_mock.await_args.args[2]
        self.assertEqual(payload["model"], "qwen3.7-plus-2026-05-26")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertFalse(payload["enable_thinking"])
        self.assertEqual(request_mock.await_args.kwargs["timeout_seconds"], 60.0)

    async def test_malformed_response_is_rejected(self):
        env = {
            "API_BASE": "https://example.invalid/compatible-mode/v1",
            "API_KEY": "test-key",
            "LLM_ENABLE_THINKING": "false",
            "LLM_TEMPERATURE": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch.object(
                openai_api,
                "make_async_request",
                new=AsyncMock(return_value={"usage": {}}),
            ):
                with self.assertRaises(RuntimeError):
                    await openai_api.make_async_chat(
                        "测试", model="qwen3.7-plus-2026-05-26"
                    )

    async def test_deepseek_payload_explicitly_disables_thinking(self):
        """验证 DeepSeek-V4 不会悄悄使用供应商默认思考模式。"""
        fake_response = {
            "choices": [{"message": {"content": "Plan Chain:\n['ComputePiMw']"}}],
            "usage": {
                "prompt_tokens": 18,
                "completion_tokens": 7,
                "total_tokens": 25,
            },
            "id": "deepseek-offline-test",
        }
        env = {
            "API_BASE": "https://example.invalid/compatible-mode/v1",
            "API_KEY": "test-key",
            "MODEL_NAME": "deepseek-v4-flash-0731",
            "LLM_ENABLE_THINKING": "false",
            "LLM_TEMPERATURE": "0",
            "LLM_TIMEOUT_SECONDS": "120",
        }

        with patch.dict(os.environ, env, clear=False):
            with patch.object(
                openai_api,
                "make_async_request",
                new=AsyncMock(return_value=fake_response),
            ) as request_mock:
                result = await openai_api.make_async_chat_with_metadata(
                    "计算pI", model="deepseek-v4-flash-0731"
                )

        payload = request_mock.await_args.args[2]
        self.assertEqual(payload["model"], "deepseek-v4-flash-0731")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertFalse(payload["enable_thinking"])
        self.assertEqual(request_mock.await_args.kwargs["timeout_seconds"], 120.0)
        self.assertEqual(result["usage"]["total_tokens"], 25)


if __name__ == "__main__":
    unittest.main(verbosity=2)
