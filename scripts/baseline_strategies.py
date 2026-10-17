# -*- coding: utf-8 -*-
"""Schema checks and bounded replanning baselines."""

from __future__ import annotations

import copy
import json
import time
from typing import Any

from app.core.logger import get_logger
from app.llms.extraction import extract_tool_path
from app.tools.tool_runner import find_unknown_tool_names
from ToolsAgent.ToolsFuns.Biology.tool_name_dict import BIOLOGY_TOOLS_DESC
from scripts.plan_sampling import (
    build_plan_sample_id,
    safe_plan_trace,
    sha256_text,
    validate_frozen_plan,
)
from scripts.type_checker import get_tool_signatures, validate_tool_plan


logger = get_logger("BaselineStrategies", "baseline_strategies.log")

SCHEMA_RUNTIME_BASELINE = "schema_runtime_validation"
LLM_REPLAN_BASELINE = "matched_budget_llm_replan"


def _error_result(
    message: str,
    code: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        issue = {"code": code, "message": message}
        if isinstance(details, dict):
            issue.update(copy.deepcopy(details))
        return {"status": "error", "message": message, "issues": [issue]}
    except Exception as exc:
        logger.error("构造baseline错误失败: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc), "issues": []}


def validate_schema_runtime_plan(frozen_plan: Any) -> dict[str, Any]:
    try:
        validation = validate_frozen_plan(frozen_plan)
        if validation.get("status") != "success":
            return {
                **_error_result(
                    "原始冻结计划未通过Schema校验",
                    "SCHEMA_INVALID_FROZEN_PLAN",
                ),
                "valid": False,
                "decision": "reject",
            }
        plan = validation["frozen_plan"]
        chain = list(plan.get("planned_chain") or [])
        if plan.get("capture_outcome") != "captured" or not chain:
            return {
                "status": "success",
                "valid": False,
                "decision": "reject",
                "baseline_kind": SCHEMA_RUNTIME_BASELINE,
                "issues": [
                    {
                        "code": "SCHEMA_UNEXECUTABLE_PLAN",
                        "message": "计划未包含可执行的非空Plan Chain",
                    }
                ],
                "planned_chain": chain,
            }
        unknown_tools = find_unknown_tool_names(chain)
        if unknown_tools:
            return {
                "status": "success",
                "valid": False,
                "decision": "reject",
                "baseline_kind": SCHEMA_RUNTIME_BASELINE,
                "issues": [
                    {
                        "code": "SCHEMA_UNKNOWN_TOOL",
                        "message": "计划包含未注册工具",
                        "unknown_tools": list(unknown_tools),
                    }
                ],
                "planned_chain": chain,
            }
        missing_entries = [
            tool_name
            for tool_name in chain
            if not isinstance(BIOLOGY_TOOLS_DESC.get(tool_name), dict)
            or BIOLOGY_TOOLS_DESC[tool_name].get("is_internal")
            or not str(
                BIOLOGY_TOOLS_DESC[tool_name].get("api_path") or ""
            ).strip()
        ]
        if missing_entries:
            return {
                "status": "success",
                "valid": False,
                "decision": "reject",
                "baseline_kind": SCHEMA_RUNTIME_BASELINE,
                "issues": [
                    {
                        "code": "SCHEMA_MISSING_RUNTIME_ENTRY",
                        "message": "工具缺少可验证的运行时入口",
                        "tool_names": missing_entries,
                    }
                ],
                "planned_chain": chain,
            }
        return {
            "status": "success",
            "valid": True,
            "decision": "execute",
            "baseline_kind": SCHEMA_RUNTIME_BASELINE,
            "issues": [],
            "planned_chain": chain,
        }
    except Exception as exc:
        logger.error("Schema/runtime baseline校验异常: %s", exc, exc_info=True)
        return {
            **_error_result(
                "Schema/runtime baseline校验异常",
                "SCHEMA_RUNTIME_VALIDATION_EXCEPTION",
                details={"exception_type": type(exc).__name__},
            ),
            "valid": False,
            "decision": "reject",
        }


def build_schema_rejection_result(
    frozen_plan: Any,
    validation: Any,
) -> dict[str, Any]:
    try:
        issues = (
            copy.deepcopy(validation.get("issues") or [])
            if isinstance(validation, dict)
            else []
        )
        issue = issues[0] if issues else {
            "code": "SCHEMA_RUNTIME_REJECTION",
            "message": "计划未通过Schema/runtime校验",
        }
        chain = (
            list(validation.get("planned_chain") or [])
            if isinstance(validation, dict)
            else []
        )
        return {
            "answer": (
                '{"status":"error","error_type":"schema_runtime_rejection"}'
                "\n\n**Final Answer:**\n"
                f"问题未完成：{issue.get('message')}"
            ),
            "reasoning": {
                "retrieved_tools": [],
                "plan_chain": chain,
                "tool_steps": [],
                "typed_planning": None,
                "request_guard": {
                    "status": "success",
                    "decision": "allow",
                    "detail_code": None,
                    "unknown_tools": [],
                },
                "plan_sampling": safe_plan_trace(frozen_plan),
                "baseline": {
                    "baseline_kind": SCHEMA_RUNTIME_BASELINE,
                    "schema_valid_before": False,
                    "original_plan_preserved": True,
                    "replan_triggered": False,
                    "replan_applied": False,
                    "replan_plan_sample_id": None,
                    "extra_llm_calls": 0,
                    "extra_llm_input_tokens": None,
                    "extra_llm_output_tokens": None,
                    "extra_llm_total_tokens": None,
                    "replan_latency_ms": 0.0,
                    "replan_outcome": "not_triggered",
                    "original_plan_chain": chain,
                    "revised_plan_chain": chain,
                },
                "execution_summary": {
                    "completed": False,
                    "attempted_steps": 0,
                    "successful_steps": 0,
                    "failed_steps": 0,
                    "error_type": "schema_runtime_rejection",
                    "detail_code": issue.get("code"),
                },
            },
        }
    except Exception as exc:
        logger.error("S1拒绝响应构造失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error_code": "SCHEMA_REJECTION_BUILD_EXCEPTION",
            "answer": "错误：无法构造Schema/runtime拒绝响应。",
            "reasoning": {},
        }


def _baseline_metadata(
    *,
    original_chain: list[str],
    original_plan_sample_id: str | None,
    schema_valid_before: bool,
    strategy_kind: str = LLM_REPLAN_BASELINE,
) -> dict[str, Any]:
    return {
        "baseline_kind": str(strategy_kind or LLM_REPLAN_BASELINE),
        "schema_valid_before": bool(schema_valid_before),
        "original_plan_preserved": True,
        "original_plan_sample_id": original_plan_sample_id,
        "replan_triggered": False,
        "replan_applied": False,
        "replan_plan_sample_id": None,
        "extra_llm_calls": 0,
        "extra_llm_input_tokens": None,
        "extra_llm_output_tokens": None,
        "extra_llm_total_tokens": None,
        "replan_latency_ms": 0.0,
        "replan_outcome": "not_triggered",
        "original_plan_chain": list(original_chain),
        "revised_plan_chain": list(original_chain),
        "issue_codes_before": [],
        "issue_codes_after": [],
    }


def _failed_replan_ledger_record(
    original: dict[str, Any],
    *,
    model: Any,
    error_code: str,
    outcome: str,
    raw_response: Any = None,
    planned_chain: Any = None,
    latency_ms: Any = 0.0,
    usage: Any = None,
    exception_type: Any = None,
    plan_namespace: str = "l1",
) -> dict[str, Any]:
    raw_text = str(raw_response or "")
    return {
        "schema_version": "1.0",
        "record_status": "capture_failed",
        "experiment_id": (
            f"{original.get('experiment_id', '')}-{plan_namespace}"
        ),
        "question_id": str(original.get("question_id") or ""),
        "repeat_index": int(original.get("repeat_index", 0)),
        "model": str(model or original.get("model") or ""),
        "error_code": str(error_code),
        "attempts": [
            {
                "attempt_index": 1,
                "attempt_kind": (
                    "content" if raw_text else "infrastructure"
                ),
                "outcome": str(outcome),
                "exception_type": (
                    str(exception_type) if exception_type else None
                ),
                "response_sha256": (
                    sha256_text(raw_text) if raw_text else None
                ),
                "raw_response": raw_text if raw_text else None,
                "parsed_chain": list(planned_chain or []),
                "unknown_tools": find_unknown_tool_names(
                    list(planned_chain or [])
                ),
                "latency_ms": max(0.0, float(latency_ms or 0.0)),
                "usage": copy.deepcopy(
                    usage if isinstance(usage, dict) else {}
                ),
            }
        ],
    }


async def prepare_matched_budget_replan(
    *,
    question: Any,
    frozen_plan: Any,
    available_input_types: Any,
    initial_artifact_issue_codes: Any,
    model: Any,
    replan_callable: Any,
    max_output_tokens: Any,
    strategy_kind: str = LLM_REPLAN_BASELINE,
    plan_namespace: str = "l1",
    required_tool_names: Any = None,
) -> dict[str, Any]:
    try:
        normalized_namespace = str(plan_namespace or "").strip().lower()
        if (
            not normalized_namespace
            or len(normalized_namespace) > 16
            or any(
                char not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for char in normalized_namespace
            )
        ):
            return _error_result(
                "单次重规划命名空间无效",
                "INVALID_REPLAN_NAMESPACE",
            )
        normalized_strategy = str(strategy_kind or "").strip()
        if not normalized_strategy:
            return _error_result(
                "单次重规划策略标识不能为空",
                "INVALID_REPLAN_STRATEGY_KIND",
            )
        normalized_required_tools: list[str] | None = None
        if required_tool_names is not None:
            if not isinstance(required_tool_names, (list, tuple)):
                return _error_result(
                    "重规划必须保留工具必须是列表",
                    "INVALID_REQUIRED_TOOL_NAMES",
                )
            normalized_required_tools = [
                str(value).strip()
                for value in required_tool_names
                if isinstance(value, str) and value.strip()
            ]
            if len(normalized_required_tools) != len(required_tool_names):
                return _error_result(
                    "重规划必须保留工具包含无效名称",
                    "INVALID_REQUIRED_TOOL_NAMES",
                )
        frozen_validation = validate_frozen_plan(frozen_plan)
        if frozen_validation.get("status") != "success":
            return _error_result(
                "L1收到无效原始冻结计划",
                "LLM_REPLAN_INVALID_FROZEN_PLAN",
            )
        original = frozen_validation["frozen_plan"]
        original_chain = list(original.get("planned_chain") or [])
        original_plan_id = str(original.get("plan_sample_id") or "") or None
        metadata = _baseline_metadata(
            original_chain=original_chain,
            original_plan_sample_id=original_plan_id,
            schema_valid_before=(
                original.get("capture_outcome") == "captured"
                and bool(original_chain)
                and not find_unknown_tool_names(original_chain)
            ),
            strategy_kind=normalized_strategy,
        )

        artifact_issues = sorted(
            {
                str(value)
                for value in (initial_artifact_issue_codes or [])
                if str(value).strip()
            }
        )
        if (
            original.get("capture_outcome") != "captured"
            or not original_chain
            or find_unknown_tool_names(original_chain)
            or artifact_issues
        ):
            return {
                "status": "success",
                "active_frozen_plan": original,
                "baseline": metadata,
                "replan_frozen_plan": None,
                "replan_ledger_record": None,
            }

        type_validation = validate_tool_plan(
            original_chain,
            available_input_types,
        )
        if type_validation.get("status") != "success":
            return _error_result(
                "L1触发前类型检查器失败",
                "LLM_REPLAN_TYPE_CHECK_FAILURE",
            )
        issues_before = list(type_validation.get("issues") or [])
        metadata["issue_codes_before"] = sorted(
            {
                str(issue.get("code"))
                for issue in issues_before
                if isinstance(issue, dict) and issue.get("code")
            }
        )
        if type_validation.get("valid") is True:
            return {
                "status": "success",
                "active_frozen_plan": original,
                "baseline": metadata,
                "replan_frozen_plan": None,
                "replan_ledger_record": None,
            }
        if not callable(replan_callable):
            return _error_result(
                "L1需要可调用的单次重规划器",
                "MISSING_LLM_REPLAN_CALLABLE",
            )

        metadata["replan_triggered"] = True
        metadata["extra_llm_calls"] = 1
        started = time.perf_counter()
        try:
            replan_arguments = {
                "question": str(question or ""),
                "original_plan_chain": original_chain,
                "available_input_types": list(available_input_types or []),
                "type_issues": issues_before,
                "model": str(model or ""),
                "max_output_tokens": max_output_tokens,
            }
            if normalized_required_tools is not None:
                replan_arguments["required_tool_names"] = list(
                    normalized_required_tools
                )
            replan_result = await replan_callable(**replan_arguments)
        except Exception as exc:
            logger.error("L1单次重规划调用失败: %s", exc, exc_info=True)
            metadata["replan_latency_ms"] = max(
                0.0, (time.perf_counter() - started) * 1000.0
            )
            metadata["replan_outcome"] = "infrastructure_failure"
            return {
                "status": "success",
                "active_frozen_plan": original,
                "baseline": metadata,
                "replan_frozen_plan": None,
                "replan_ledger_record": _failed_replan_ledger_record(
                    original,
                    model=model,
                    error_code="LLM_REPLAN_REQUEST_EXCEPTION",
                    outcome="planner_exception",
                    latency_ms=metadata["replan_latency_ms"],
                    exception_type=type(exc).__name__,
                    plan_namespace=normalized_namespace,
                ),
            }
        metadata["replan_latency_ms"] = max(
            0.0,
            float(
                replan_result.get("latency_ms")
                if isinstance(replan_result, dict)
                and isinstance(replan_result.get("latency_ms"), (int, float))
                else (time.perf_counter() - started) * 1000.0
            ),
        )
        if not isinstance(replan_result, dict) or (
            replan_result.get("status") != "success"
        ):
            metadata["replan_outcome"] = "infrastructure_failure"
            return {
                "status": "success",
                "active_frozen_plan": original,
                "baseline": metadata,
                "replan_frozen_plan": None,
                "replan_ledger_record": _failed_replan_ledger_record(
                    original,
                    model=model,
                    error_code="LLM_REPLAN_REQUEST_FAILURE",
                    outcome="planner_exception",
                    latency_ms=metadata["replan_latency_ms"],
                    plan_namespace=normalized_namespace,
                ),
            }

        usage = replan_result.get("usage") or {}
        for source_name, target_name in (
            ("input_tokens", "extra_llm_input_tokens"),
            ("output_tokens", "extra_llm_output_tokens"),
            ("total_tokens", "extra_llm_total_tokens"),
        ):
            value = usage.get(source_name) if isinstance(usage, dict) else None
            metadata[target_name] = (
                max(0, int(value)) if isinstance(value, (int, float)) else None
            )
        raw_response = str(
            replan_result.get("raw_response")
            or replan_result.get("content")
            or ""
        )
        revised_chain = replan_result.get("planned_chain")
        if not isinstance(revised_chain, list):
            revised_chain = extract_tool_path(raw_response)
        revised_chain = [
            str(value).strip()
            for value in revised_chain
            if isinstance(value, str) and value.strip()
        ]
        metadata["revised_plan_chain"] = list(revised_chain)
        if not revised_chain:
            metadata["replan_outcome"] = "unparseable_plan"
            return {
                "status": "success",
                "active_frozen_plan": original,
                "baseline": metadata,
                "replan_frozen_plan": None,
                "replan_ledger_record": _failed_replan_ledger_record(
                    original,
                    model=model,
                    error_code="LLM_REPLAN_UNPARSEABLE",
                    outcome="unparseable_plan",
                    raw_response=raw_response,
                    latency_ms=metadata["replan_latency_ms"],
                    usage=usage,
                    plan_namespace=normalized_namespace,
                ),
            }
        unknown_tools = find_unknown_tool_names(revised_chain)
        if unknown_tools:
            metadata["replan_outcome"] = "unknown_tool_plan"
            return {
                "status": "success",
                "active_frozen_plan": original,
                "baseline": metadata,
                "replan_frozen_plan": None,
                "replan_ledger_record": _failed_replan_ledger_record(
                    original,
                    model=model,
                    error_code="LLM_REPLAN_UNKNOWN_TOOL",
                    outcome="unknown_tool_plan",
                    raw_response=raw_response,
                    planned_chain=revised_chain,
                    latency_ms=metadata["replan_latency_ms"],
                    usage=usage,
                    plan_namespace=normalized_namespace,
                ),
            }
        revised_validation = validate_tool_plan(
            revised_chain,
            available_input_types,
        )
        metadata["issue_codes_after"] = sorted(
            {
                str(issue.get("code"))
                for issue in revised_validation.get("issues") or []
                if isinstance(issue, dict) and issue.get("code")
            }
        )
        if revised_validation.get("status") != "success" or (
            revised_validation.get("valid") is not True
        ):
            metadata["replan_outcome"] = "type_invalid_replan"
            return {
                "status": "success",
                "active_frozen_plan": original,
                "baseline": metadata,
                "replan_frozen_plan": None,
                "replan_ledger_record": _failed_replan_ledger_record(
                    original,
                    model=model,
                    error_code="LLM_REPLAN_TYPE_INVALID",
                    outcome="type_invalid_replan",
                    raw_response=raw_response,
                    planned_chain=revised_chain,
                    latency_ms=metadata["replan_latency_ms"],
                    usage=usage,
                    plan_namespace=normalized_namespace,
                ),
            }

        if normalized_required_tools is not None:
            remaining = list(revised_chain)
            missing_required_tools: list[str] = []
            for tool_name in normalized_required_tools:
                if tool_name in remaining:
                    remaining.remove(tool_name)
                else:
                    missing_required_tools.append(tool_name)
            if missing_required_tools:
                metadata["replan_outcome"] = "goal_incomplete_replan"
                return {
                    "status": "success",
                    "active_frozen_plan": original,
                    "baseline": metadata,
                    "replan_frozen_plan": None,
                    "replan_ledger_record": _failed_replan_ledger_record(
                        original,
                        model=model,
                        error_code="LLM_REPLAN_GOAL_INCOMPLETE",
                        outcome="goal_incomplete_replan",
                        raw_response=raw_response,
                        planned_chain=revised_chain,
                        latency_ms=metadata["replan_latency_ms"],
                        usage={
                            **(usage if isinstance(usage, dict) else {}),
                            "missing_required_tools": missing_required_tools,
                        },
                        plan_namespace=normalized_namespace,
                    ),
                }

        response_sha256 = sha256_text(raw_response)
        attempt = {
            "attempt_index": 1,
            "attempt_kind": "content",
            "outcome": "captured",
            "exception_type": None,
            "response_sha256": response_sha256,
            "raw_response": raw_response,
            "parsed_chain": revised_chain,
            "unknown_tools": [],
            "latency_ms": metadata["replan_latency_ms"],
            "usage": copy.deepcopy(usage),
        }
        replanned = {
            "schema_version": "1.0",
            "record_status": "captured",
            "experiment_id": (
                f"{original.get('experiment_id', '')}-{normalized_namespace}"
            ),
            "question_id": str(original.get("question_id") or ""),
            "repeat_index": int(original.get("repeat_index", 0)),
            "model": str(model or original.get("model") or ""),
            "plan_sample_id": build_plan_sample_id(
                experiment_id=(
                    f"{original.get('experiment_id', '')}-{normalized_namespace}"
                ),
                question_id=original.get("question_id"),
                repeat_index=original.get("repeat_index", 0),
                model=model or original.get("model"),
                response_sha256=response_sha256,
            ),
            "capture_outcome": "captured",
            "selected_attempt_index": 1,
            "response_sha256": response_sha256,
            "planned_chain": revised_chain,
            "unknown_tools": [],
            "retrieved_triples": copy.deepcopy(
                original.get("retrieved_triples") or []
            ),
            "retrieved_tool_names": copy.deepcopy(
                original.get("retrieved_tool_names") or []
            ),
            "attempts": [attempt],
        }
        replanned_validation = validate_frozen_plan(replanned)
        if replanned_validation.get("status") != "success":
            return _error_result(
                "L1重规划结果无法冻结",
                "LLM_REPLAN_FREEZE_FAILURE",
            )
        active_plan = replanned_validation["frozen_plan"]
        metadata["replan_applied"] = True
        metadata["replan_outcome"] = "captured"
        metadata["replan_plan_sample_id"] = active_plan["plan_sample_id"]
        return {
            "status": "success",
            "active_frozen_plan": active_plan,
            "baseline": metadata,
            "replan_frozen_plan": active_plan,
            "replan_ledger_record": active_plan,
        }
    except Exception as exc:
        logger.error("L1 baseline准备异常: %s", exc, exc_info=True)
        return _error_result(
            "L1 baseline准备异常",
            "LLM_REPLAN_PREPARATION_EXCEPTION",
            details={"exception_type": type(exc).__name__},
        )


def build_replan_prompt(
    *,
    question: Any,
    original_plan_chain: Any,
    available_input_types: Any,
    type_issues: Any,
    required_tool_names: Any = None,
) -> dict[str, Any]:
    try:
        signatures_result = get_tool_signatures()
        if signatures_result.get("status") != "success":
            return _error_result(
                "无法构造LLM重规划工具签名",
                "REPLAN_PROMPT_REGISTRY_FAILURE",
            )
        compact_signatures = {
            name: {
                "input_types": info.get("input_types") or [],
                "output_types": info.get("output_types") or [],
            }
            for name, info in signatures_result["signatures"].items()
        }
        safe_issues = [
            {
                "code": str(issue.get("code") or ""),
                "step_index": issue.get("step_index"),
                "tool_name": issue.get("tool_name"),
                "missing_types": list(issue.get("missing_types") or []),
            }
            for issue in (type_issues or [])
            if isinstance(issue, dict)
        ]
        required_tools = [
            str(value).strip()
            for value in (required_tool_names or [])
            if isinstance(value, str) and value.strip()
        ]
        preservation_instruction = (
            "下列目标工具不得删除，每个至少保留原计划中的出现次数："
            f"{json.dumps(required_tools, ensure_ascii=False)}\n"
            if required_tool_names is not None
            else ""
        )
        prompt = (
            "你是工具计划修复baseline。你只有一次修改机会，不要解释。\n"
            "只能使用下列注册工具，必须使每步输入类型来自初始输入或前驱输出。\n"
            f"{preservation_instruction}"
            "输出严格限于: Plan Chain: ['tool_a', 'tool_b']\n"
            f"任务: {str(question or '')}\n"
            f"初始可用类型: {json.dumps(list(available_input_types or []), ensure_ascii=False)}\n"
            f"原计划: {json.dumps(list(original_plan_chain or []), ensure_ascii=False)}\n"
            f"类型问题: {json.dumps(safe_issues, ensure_ascii=False)}\n"
            f"工具签名: {json.dumps(compact_signatures, ensure_ascii=False, sort_keys=True)}"
        )
        return {"status": "success", "prompt": prompt}
    except Exception as exc:
        logger.error("构造L1 Prompt失败: %s", exc, exc_info=True)
        return _error_result(
            "构造L1 Prompt失败",
            "REPLAN_PROMPT_BUILD_EXCEPTION",
        )
