# -*- coding: utf-8 -*-
"""
显式工具调用请求的规划前安全门禁。

修改记录：
- 2026-08-11：新增显式命名工具注册检查；用户要求调用未注册工具时，
  在KG检索和LLM规划之前确定性拒绝，禁止用其他工具冒充。

设计边界：
1. 只检查“调用/使用某个命名工具”的明确执行意图；
2. “某工具是什么”等知识问答不属于调用请求；
3. 不对自然语言能力需求进行猜测式语义分类；
4. 工具是否注册统一复用执行层注册表，不维护第二份闭集名单；
5. 所有公开函数捕获异常并返回标准结构，主进程不得崩溃。
"""

from __future__ import annotations

import re
from typing import Any

from app.core.logger import get_logger
from app.tools.tool_runner import resolve_api_path


logger = get_logger("ExplicitToolRequestGuard", "agent.log")

GUARD_VERSION = "1.0"
_MAX_TOOL_NAME_LENGTH = 128

# 中文和英文模式都要求出现明确的调用动词。带“工具/API/接口/函数”后缀时，
# 允许普通CamelCase名称；没有后缀时只接受代码风格名称或已注册别名。
_EXPLICIT_INVOCATION_PATTERNS = (
    re.compile(
        r"(?:调用|使用|运行|执行|请用|请使用|请调用)\s*"
        r"(?:一个|这个|该|名为|叫做|不存在的|未注册的|系统中的|本地的|the\s+)*"
        r"[`'\"“”]?([A-Za-z][A-Za-z0-9_.-]{1,127})[`'\"“”]?\s*"
        r"(?:工具|接口|API|函数)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:call|invoke|run|use)\s+"
        r"(?:the\s+)?(?:nonexistent\s+|unregistered\s+|named\s+)*"
        r"[`'\"]?([A-Za-z][A-Za-z0-9_.-]{1,127})[`'\"]?\s+"
        r"(?:tool|api|interface|function)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:调用|使用|运行|执行|请用|请使用|请调用)\s*"
        r"(?:不存在的|未注册的|名为|叫做|系统中的|本地的)*\s*"
        r"[`'\"“”]?([A-Za-z][A-Za-z0-9_.-]{1,127})[`'\"“”]?",
        flags=re.IGNORECASE,
    ),
)


def _looks_like_explicit_identifier(value: Any) -> bool:
    """判断无“工具”后缀的候选是否像代码标识符而非普通英文词。"""
    try:
        if not isinstance(value, str):
            return False
        candidate = value.strip()
        if not candidate or len(candidate) > _MAX_TOOL_NAME_LENGTH:
            return False
        if resolve_api_path(candidate) is not None:
            return True
        return bool(
            "_" in candidate
            or "-" in candidate
            or "." in candidate
            or re.search(r"[a-z][A-Z]", candidate)
            or (len(candidate) >= 3 and candidate.isupper())
        )
    except Exception as exc:
        logger.error("判断显式工具标识符失败: %s", exc, exc_info=True)
        return False


def extract_explicit_tool_invocations(question: Any) -> dict[str, Any]:
    """提取用户明确要求调用的命名工具，不处理一般能力描述。"""
    try:
        if not isinstance(question, str) or not question.strip():
            return {
                "status": "error",
                "message": "问题必须是非空字符串",
                "invocation_intent": False,
                "tool_names": [],
            }

        text = question.strip()
        extracted: list[str] = []
        for pattern_index, pattern in enumerate(_EXPLICIT_INVOCATION_PATTERNS):
            for match in pattern.finditer(text):
                candidate = str(match.group(1) or "").strip()
                if not candidate or len(candidate) > _MAX_TOOL_NAME_LENGTH:
                    continue
                # 第三个模式没有“工具/API”等后缀，必须满足更严格的代码标识符判断。
                if pattern_index == 2 and not _looks_like_explicit_identifier(candidate):
                    continue
                if candidate.casefold() not in {
                    item.casefold() for item in extracted
                }:
                    extracted.append(candidate)

        return {
            "status": "success",
            "message": None,
            "invocation_intent": bool(extracted),
            "tool_names": extracted,
        }
    except Exception as exc:
        logger.error("提取显式工具调用请求失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"提取显式工具调用请求失败: {exc}",
            "invocation_intent": False,
            "tool_names": [],
        }


def evaluate_explicit_tool_request(question: Any) -> dict[str, Any]:
    """检查显式命名工具是否注册，并给出allow/reject确定性决策。"""
    try:
        extraction = extract_explicit_tool_invocations(question)
        if extraction.get("status") != "success":
            return {
                "status": "error",
                "decision": "reject",
                "guard_version": GUARD_VERSION,
                "detail_code": "REQUEST_GUARD_ERROR",
                "message": extraction.get("message", "请求门禁无法解析问题"),
                "invocation_intent": False,
                "requested_tools": [],
                "unknown_tools": [],
                "capability_scope": "explicit_named_tools_only",
            }

        tool_names = list(extraction.get("tool_names") or [])
        requested_tools: list[dict[str, Any]] = []
        unknown_tools: list[str] = []
        for tool_name in tool_names:
            api_path = resolve_api_path(tool_name)
            registered = api_path is not None
            requested_tools.append(
                {
                    "name": tool_name,
                    "registered": registered,
                    "api_path": api_path,
                }
            )
            if not registered:
                unknown_tools.append(tool_name)

        if unknown_tools:
            return {
                "status": "success",
                "decision": "reject",
                "guard_version": GUARD_VERSION,
                "detail_code": "UNREGISTERED_TOOL",
                "message": (
                    "用户明确要求调用未注册工具；系统不会使用其他工具冒充"
                ),
                "invocation_intent": True,
                "requested_tools": requested_tools,
                "unknown_tools": unknown_tools,
                "capability_scope": "explicit_named_tools_only",
            }

        return {
            "status": "success",
            "decision": "allow",
            "guard_version": GUARD_VERSION,
            "detail_code": None,
            "message": None,
            "invocation_intent": bool(tool_names),
            "requested_tools": requested_tools,
            "unknown_tools": [],
            "capability_scope": "explicit_named_tools_only",
        }
    except Exception as exc:
        logger.error("显式工具请求门禁异常: %s", exc, exc_info=True)
        return {
            "status": "error",
            "decision": "reject",
            "guard_version": GUARD_VERSION,
            "detail_code": "REQUEST_GUARD_ERROR",
            "message": f"显式工具请求门禁异常: {exc}",
            "invocation_intent": False,
            "requested_tools": [],
            "unknown_tools": [],
            "capability_scope": "explicit_named_tools_only",
        }
