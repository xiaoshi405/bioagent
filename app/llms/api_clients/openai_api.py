# -*- coding: utf-8 -*-
"""Async client for OpenAI-compatible model APIs."""

import httpx

from app.core import Config
from app.core.logger import get_logger


logger = get_logger("LLMClient", "llm_api.log")
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
THINKING_TOGGLE_MODEL_PREFIXES = ("qwen", "deepseek-v4-")


def make_simple_dialogs(text, system_prompt=""):
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("LLM输入文本不能为空")
        if not isinstance(system_prompt, str):
            raise TypeError("System Prompt必须是字符串")

        effective_system_prompt = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        return [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": text},
        ]
    except Exception as exc:
        logger.error("构造LLM对话失败: %s", exc, exc_info=True)
        raise


async def make_async_request(url, api_key, payload, timeout_seconds=60.0):
    try:
        if not url or not isinstance(url, str):
            raise ValueError("LLM API地址不能为空")
        if not api_key or not isinstance(api_key, str):
            raise ValueError("LLM API Key未配置")
        if not isinstance(payload, dict) or not payload:
            raise ValueError("LLM请求体不能为空")
        if timeout_seconds <= 0:
            timeout_seconds = 60.0

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            request_id = response.headers.get("x-request-id", "unknown")
            logger.error(
                "LLM请求失败 | status=%s | request_id=%s | response=%s",
                response.status_code,
                request_id,
                response.text[:500],
            )
            raise RuntimeError(f"LLM连接失败，HTTP状态码: {response.status_code}")

        try:
            return response.json()
        except Exception as exc:
            logger.error("LLM响应不是有效JSON: %s", exc, exc_info=True)
            raise RuntimeError("LLM响应不是有效JSON") from exc
    except httpx.TimeoutException as exc:
        logger.error("LLM请求超时: %s", exc)
        raise RuntimeError(f"LLM请求超时（{timeout_seconds}秒）") from exc
    except httpx.RequestError as exc:
        logger.error("LLM网络请求失败: %s", exc)
        raise RuntimeError(f"LLM网络请求失败: {exc}") from exc
    except Exception:
        raise


def get_api_url(model):
    try:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("模型名称不能为空")
        config = Config()
        if model.startswith("qwen"):
            return config.DASHSCOPE_API_BASE.rstrip("/")
        return config.OPENAI_API_BASE.rstrip("/")
    except Exception as exc:
        logger.error("选择LLM API地址失败: %s", exc, exc_info=True)
        raise


def supports_explicit_thinking_toggle(model: str) -> bool:
    try:
        normalized = str(model or "").strip().lower()
        return bool(normalized) and normalized.startswith(
            THINKING_TOGGLE_MODEL_PREFIXES
        )
    except Exception as exc:
        logger.warning("判断模型思考开关能力失败: %s", exc)
        return False


def _log_token_usage(model, response):
    try:
        usage = response.get("usage") or {}
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        details = (
            usage.get("output_tokens_details")
            or usage.get("completion_tokens_details")
            or {}
        )
        reasoning_tokens = details.get("reasoning_tokens", 0)
        request_id = response.get("request_id", response.get("id", "unknown"))

        logger.info(
            "LLM Token用量 | model=%s | input=%s | output=%s | reasoning=%s | total=%s | request_id=%s",
            model,
            input_tokens,
            output_tokens,
            reasoning_tokens,
            total_tokens,
            request_id,
        )
    except Exception as exc:
        logger.warning("记录LLM Token用量失败: %s", exc)


async def make_async_chat(text, model, system_prompt="") -> str:
    result = await make_async_chat_with_metadata(
        text,
        model,
        system_prompt=system_prompt,
    )
    return result["content"]


async def make_async_chat_with_metadata(
    text,
    model,
    system_prompt="",
    *,
    max_output_tokens=None,
) -> dict:
    try:
        config = Config()
        selected_model = (model or config.MODEL_NAME).strip()
        if not selected_model:
            raise ValueError("MODEL_NAME未配置")

        url = f"{get_api_url(selected_model)}/chat/completions"
        dialogs = make_simple_dialogs(text, system_prompt=system_prompt)
        api_key = (
            config.DASHSCOPE_API_KEY
            if selected_model.startswith("qwen")
            else config.OPENAI_API_KEY
        )
        payload = {
            "model": selected_model,
            "messages": dialogs,
            "temperature": config.LLM_TEMPERATURE,
        }
        if max_output_tokens is not None:
            try:
                normalized_max_tokens = min(
                    4096,
                    max(1, int(max_output_tokens)),
                )
            except (TypeError, ValueError):
                normalized_max_tokens = 512
            payload["max_tokens"] = normalized_max_tokens
        if supports_explicit_thinking_toggle(selected_model):
            payload["enable_thinking"] = config.LLM_ENABLE_THINKING

        response = await make_async_request(
            url,
            api_key,
            payload,
            timeout_seconds=config.LLM_TIMEOUT_SECONDS,
        )
        _log_token_usage(selected_model, response)

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM响应缺少choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM返回内容为空")
        usage = response.get("usage") or {}
        input_tokens = usage.get(
            "input_tokens",
            usage.get("prompt_tokens"),
        )
        output_tokens = usage.get(
            "output_tokens",
            usage.get("completion_tokens"),
        )
        total_tokens = usage.get("total_tokens")
        return {
            "content": content,
            "usage": {
                "input_tokens": (
                    max(0, int(input_tokens))
                    if isinstance(input_tokens, (int, float))
                    else None
                ),
                "output_tokens": (
                    max(0, int(output_tokens))
                    if isinstance(output_tokens, (int, float))
                    else None
                ),
                "total_tokens": (
                    max(0, int(total_tokens))
                    if isinstance(total_tokens, (int, float))
                    else None
                ),
            },
            "request_id": str(
                response.get("request_id") or response.get("id") or ""
            ),
        }
    except Exception as exc:
        logger.error("LLM聊天调用失败: %s", exc, exc_info=True)
        raise


async def make_async_embedding(text, model) -> list:
    try:
        config = Config()
        url = f"{config.OPENAI_API_BASE.rstrip('/')}/embeddings"
        payload = {"model": model, "input": text}
        response = await make_async_request(
            url,
            config.OPENAI_API_KEY,
            payload,
            timeout_seconds=config.LLM_TIMEOUT_SECONDS,
        )
        return response["data"][0]["embedding"]
    except Exception as exc:
        logger.error("Embedding调用失败: %s", exc, exc_info=True)
        raise
