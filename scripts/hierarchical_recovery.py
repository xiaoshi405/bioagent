# -*- coding: utf-8 -*-
"""F1 分层恢复：确定性快速路径、一次类型约束重规划、失败关闭。

修改记录：
- 2026-09-05：新增基于工具注册表的计划意图保真门禁；允许替换
  Artifact 适配器，但不允许一次重规划删掉原计划中的目标工具。
- 2026-09-04：修正初始 Artifact 已非法时的判定顺序；在空类型集合计划检查前
  保留精确输入错误并失败关闭，避免误报 F1 检查器故障。
- 2026-09-04：首次建立 F1。复用既有确定性修复器和 L1 单次重规划器，
  不增加针对题目的路线规则；所有候选计划执行前必须通过同一类型验证器。

设计边界：
1. 初始 Artifact 非法、空计划和未注册工具不交给 LLM 猜测；
2. 确定性修复成功后立即返回，不产生额外 LLM 调用；
3. 确定性修复失败后最多调用一次既有重规划器；
4. 重规划仍不合法时保留原计划，由类型门禁零调用拒绝。
"""

from __future__ import annotations

import copy
import json
from typing import Any

from app.core.logger import get_logger
from app.tools.tool_runner import find_unknown_tool_names
from scripts.baseline_strategies import prepare_matched_budget_replan
from scripts.plan_sampling import (
    build_plan_sample_id,
    sha256_text,
    validate_frozen_plan,
)
from scripts.structured_plan import prepare_structured_plan
from scripts.type_checker import get_tool_signatures, validate_tool_plan


logger = get_logger("HierarchicalRecovery", "hierarchical_recovery.log")

HIERARCHICAL_RECOVERY_KIND = "deterministic_then_typed_llm_replan"


def derive_required_goal_tools(original_chain: Any) -> dict[str, Any]:
    """从注册表派生重规划必须保留的目标工具。

    只有注册表显式授权 ``allow_drop`` 的 Artifact 适配器可被替换；
    其他工具保留次数，以防止类型合法但任务不完整的计划进入执行。
    """
    try:
        if not isinstance(original_chain, (list, tuple)):
            return _error_result(
                "原始计划链必须是列表",
                "INVALID_ORIGINAL_PLAN_CHAIN",
            )
        signatures_result = get_tool_signatures()
        if signatures_result.get("status") != "success":
            return _error_result(
                "无法从注册表派生目标保真契约",
                "F1_GOAL_POLICY_REGISTRY_FAILURE",
            )
        signatures = signatures_result["signatures"]
        required: list[str] = []
        droppable: list[str] = []
        for raw_name in original_chain:
            tool_name = str(raw_name or "").strip()
            signature = signatures.get(tool_name)
            if not tool_name or not isinstance(signature, dict):
                return _error_result(
                    "原始计划包含无法派生策略的工具",
                    "F1_GOAL_POLICY_UNKNOWN_TOOL",
                    details={"tool_name": tool_name or None},
                )
            policy = signature.get("replan_policy")
            if isinstance(policy, dict) and policy.get("allow_drop") is True:
                droppable.append(tool_name)
            else:
                required.append(tool_name)
        return {
            "status": "success",
            "required_tool_names": required,
            "droppable_adapter_names": droppable,
        }
    except Exception as exc:
        logger.error("派生F1目标工具失败: %s", exc, exc_info=True)
        return _error_result(
            "派生F1目标工具异常",
            "F1_GOAL_POLICY_EXCEPTION",
            details={"exception_type": type(exc).__name__},
        )


def _error_result(
    message: str,
    code: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造统一错误结构，避免异常逃逸到批量实验主进程。"""
    try:
        issue: dict[str, Any] = {"code": code, "message": message}
        if isinstance(details, dict):
            issue.update(copy.deepcopy(details))
        return {"status": "error", "message": message, "issues": [issue]}
    except Exception:
        return {"status": "error", "message": str(message), "issues": []}


def _issue_codes(value: Any) -> list[str]:
    """从类型问题中提取稳定、去重的错误码。"""
    try:
        return sorted(
            {
                str(issue.get("code"))
                for issue in (value or [])
                if isinstance(issue, dict) and issue.get("code")
            }
        )
    except Exception:
        return []


def _base_recovery_metadata(
    original: dict[str, Any],
    *,
    type_valid_before: bool,
    issue_codes_before: list[str],
) -> dict[str, Any]:
    """建立可公开审计、但不包含问题正文或模型原文的 F1 元数据。"""
    chain = list(original.get("planned_chain") or [])
    return {
        "recovery_kind": HIERARCHICAL_RECOVERY_KIND,
        "original_plan_preserved": True,
        "original_plan_sample_id": str(original.get("plan_sample_id") or "") or None,
        "original_plan_chain": chain,
        "revised_plan_chain": chain,
        "type_valid_before": bool(type_valid_before),
        "issue_codes_before": list(issue_codes_before),
        "issue_codes_after": list(issue_codes_before),
        "deterministic_repair_attempted": False,
        "deterministic_repair_applied": False,
        "deterministic_repair_strategy": "none",
        "deterministic_repair_operations": [],
        "deterministic_plan_sample_id": None,
        "replan_triggered": False,
        "replan_applied": False,
        "replan_plan_sample_id": None,
        "extra_llm_calls": 0,
        "extra_llm_input_tokens": None,
        "extra_llm_output_tokens": None,
        "extra_llm_total_tokens": None,
        "replan_latency_ms": 0.0,
        "replan_outcome": "not_triggered",
        "final_plan_source": "original" if type_valid_before else "rejected",
    }


def _freeze_deterministic_plan(
    original: dict[str, Any],
    repaired_chain: list[str],
    repair_operations: list[dict[str, Any]],
) -> dict[str, Any]:
    """把确定性产物构造成只在本次执行内使用的可校验冻结计划。"""
    try:
        namespace = f"{original.get('experiment_id', '')}-f1-deterministic"
        proof = json.dumps(
            {
                "original_plan_sample_id": original.get("plan_sample_id"),
                "repaired_chain": repaired_chain,
                "repair_operations": repair_operations,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        response_hash = sha256_text(proof)
        attempt = {
            "attempt_index": 1,
            "attempt_kind": "content",
            "outcome": "captured",
            "exception_type": None,
            "response_sha256": response_hash,
            "raw_response": None,
            "parsed_chain": list(repaired_chain),
            "unknown_tools": [],
            "latency_ms": 0.0,
            "usage": {},
        }
        derived = {
            "schema_version": "1.0",
            "record_status": "captured",
            "experiment_id": namespace,
            "question_id": str(original.get("question_id") or ""),
            "repeat_index": int(original.get("repeat_index", 0)),
            "model": str(original.get("model") or ""),
            "plan_sample_id": build_plan_sample_id(
                experiment_id=namespace,
                question_id=original.get("question_id"),
                repeat_index=original.get("repeat_index", 0),
                model=original.get("model"),
                response_sha256=response_hash,
            ),
            "capture_outcome": "captured",
            "selected_attempt_index": 1,
            "response_sha256": response_hash,
            "planned_chain": list(repaired_chain),
            "unknown_tools": [],
            "retrieved_triples": copy.deepcopy(original.get("retrieved_triples") or []),
            "retrieved_tool_names": copy.deepcopy(
                original.get("retrieved_tool_names") or []
            ),
            "attempts": [attempt],
        }
        validation = validate_frozen_plan(derived)
        if validation.get("status") != "success":
            return _error_result(
                "确定性修复计划无法形成执行快照",
                "DETERMINISTIC_PLAN_FREEZE_FAILURE",
            )
        return {"status": "success", "frozen_plan": validation["frozen_plan"]}
    except Exception as exc:
        logger.error("冻结确定性修复计划失败: %s", exc, exc_info=True)
        return _error_result(
            "冻结确定性修复计划异常",
            "DETERMINISTIC_PLAN_FREEZE_EXCEPTION",
            details={"exception_type": type(exc).__name__},
        )


async def prepare_hierarchical_recovery(
    *,
    question: Any,
    frozen_plan: Any,
    available_input_types: Any,
    initial_artifact_issue_codes: Any,
    model: Any,
    replan_callable: Any,
    max_output_tokens: Any,
) -> dict[str, Any]:
    """选择原计划、确定性修复计划或一次 LLM 重规划计划。"""
    try:
        frozen_validation = validate_frozen_plan(frozen_plan)
        if frozen_validation.get("status") != "success":
            return _error_result(
                "F1 收到无效原始冻结计划",
                "F1_INVALID_FROZEN_PLAN",
            )
        original = frozen_validation["frozen_plan"]
        original_chain = list(original.get("planned_chain") or [])
        artifact_issues = sorted(
            {
                str(value)
                for value in (initial_artifact_issue_codes or [])
                if str(value).strip()
            }
        )
        closed_set_valid = (
            original.get("capture_outcome") == "captured"
            and bool(original_chain)
            and not find_unknown_tool_names(original_chain)
        )
        if not closed_set_valid:
            metadata = _base_recovery_metadata(
                original,
                type_valid_before=False,
                issue_codes_before=[],
            )
            metadata["replan_outcome"] = "invalid_original_plan"
            return {
                "status": "success",
                "active_frozen_plan": original,
                "recovery": metadata,
                "replan_frozen_plan": None,
                "replan_ledger_record": None,
            }

        if artifact_issues:
            # 初始对象构建失败时可能没有可用类型；这是已知输入错误，
            # 不应再被空类型集合误报为检查器故障，也不交给 LLM 猜测修复。
            metadata = _base_recovery_metadata(
                original,
                type_valid_before=False,
                issue_codes_before=artifact_issues,
            )
            metadata["replan_outcome"] = "blocked_initial_artifact"
            return {
                "status": "success",
                "active_frozen_plan": original,
                "recovery": metadata,
                "replan_frozen_plan": None,
                "replan_ledger_record": None,
            }

        original_validation = validate_tool_plan(
            original_chain,
            list(available_input_types or []),
        )
        if original_validation.get("status") != "success":
            return _error_result(
                "F1 初始类型检查器失败",
                "F1_INITIAL_TYPE_CHECK_FAILURE",
            )
        valid_before = original_validation.get("valid") is True
        issues_before = list(original_validation.get("issues") or [])
        metadata = _base_recovery_metadata(
            original,
            type_valid_before=valid_before,
            issue_codes_before=_issue_codes(issues_before),
        )
        if valid_before:
            metadata["issue_codes_after"] = []
            return {
                "status": "success",
                "active_frozen_plan": original,
                "recovery": metadata,
                "replan_frozen_plan": None,
                "replan_ledger_record": None,
            }

        metadata["deterministic_repair_attempted"] = True
        deterministic = prepare_structured_plan(
            original_chain,
            list(available_input_types or []),
            enable_repair=True,
        )
        if deterministic.get("status") == "error":
            return _error_result(
                "F1 确定性修复器运行失败",
                "F1_DETERMINISTIC_REPAIR_FAILURE",
            )
        deterministic_applied = (
            deterministic.get("valid") is True
            and deterministic.get("repair_applied") is True
        )
        metadata["deterministic_repair_applied"] = bool(deterministic_applied)
        metadata["deterministic_repair_strategy"] = str(
            deterministic.get("repair_strategy") or "none"
        )
        metadata["deterministic_repair_operations"] = copy.deepcopy(
            deterministic.get("repair_operations") or []
        )
        if deterministic_applied:
            repaired_chain = list(deterministic.get("plan_after") or [])
            derived = _freeze_deterministic_plan(
                original,
                repaired_chain,
                metadata["deterministic_repair_operations"],
            )
            if derived.get("status") != "success":
                return derived
            active = derived["frozen_plan"]
            metadata.update(
                {
                    "revised_plan_chain": repaired_chain,
                    "issue_codes_after": [],
                    "deterministic_plan_sample_id": active["plan_sample_id"],
                    "replan_outcome": "deterministic_repair",
                    "final_plan_source": "deterministic_repair",
                }
            )
            return {
                "status": "success",
                "active_frozen_plan": active,
                "recovery": metadata,
                "replan_frozen_plan": None,
                "replan_ledger_record": None,
            }

        goal_policy = derive_required_goal_tools(original_chain)
        if goal_policy.get("status") != "success":
            return goal_policy
        metadata["required_goal_tools"] = list(
            goal_policy["required_tool_names"]
        )
        metadata["droppable_adapter_tools"] = list(
            goal_policy["droppable_adapter_names"]
        )
        replan = await prepare_matched_budget_replan(
            question=question,
            frozen_plan=original,
            available_input_types=list(available_input_types or []),
            initial_artifact_issue_codes=[],
            model=model,
            replan_callable=replan_callable,
            max_output_tokens=max_output_tokens,
            strategy_kind=HIERARCHICAL_RECOVERY_KIND,
            plan_namespace="f1",
            required_tool_names=goal_policy["required_tool_names"],
        )
        if replan.get("status") != "success":
            return replan
        llm_metadata = replan.get("baseline") or {}
        for field in (
            "replan_triggered",
            "replan_applied",
            "replan_plan_sample_id",
            "extra_llm_calls",
            "extra_llm_input_tokens",
            "extra_llm_output_tokens",
            "extra_llm_total_tokens",
            "replan_latency_ms",
            "replan_outcome",
            "revised_plan_chain",
            "issue_codes_after",
        ):
            metadata[field] = copy.deepcopy(llm_metadata.get(field))
        metadata["final_plan_source"] = (
            "llm_replan" if metadata.get("replan_applied") is True else "rejected"
        )
        return {
            "status": "success",
            "active_frozen_plan": replan["active_frozen_plan"],
            "recovery": metadata,
            "replan_frozen_plan": replan.get("replan_frozen_plan"),
            "replan_ledger_record": replan.get("replan_ledger_record"),
        }
    except Exception as exc:
        logger.error("F1 分层恢复准备异常: %s", exc, exc_info=True)
        return _error_result(
            "F1 分层恢复准备异常",
            "F1_RECOVERY_PREPARATION_EXCEPTION",
            details={"exception_type": type(exc).__name__},
        )
