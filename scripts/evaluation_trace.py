# -*- coding: utf-8 -*-
"""
BioAgent v0.3c 逐题实验Trace构建、校验与本地持久化。

修改记录：
- 2026-09-04：新增 F1 分层恢复条件与独立 v1.3 Trace 扩展，区分原计划、
  确定性快速路径、一次受验证重规划和最终安全拒绝。
- 2026-08-30：新增S1 Schema/runtime与L1固定一次LLM重规划条件，
  使用独立v1.2扩展记录触发、Token、延迟和重规划结果。
- 2026-08-27：在保留历史B0/B1/B2语义的同时新增E0/E1/E2/E3正交
  条件，并在Trace中记录类型化运行时与类型门禁状态。
- 2026-08-15：公开Trace增加脱敏修复策略与固定字段类型证明，区分桥接插入和
  冗余步骤剪枝，不保存自由文本理由或原始输入内容。
- 2026-08-15：公开Trace新增冻结计划ID、内容结果和脱敏尝试摘要；原始
  LLM响应只保存在私有计划账本中。
- 2026-08-12：允许证据完整的规划前未注册工具门禁作为B0/B1/B2共同前置
  拒绝；该路径未进入类型干预层，不能误报为B1/B2条件轨迹缺失。
- 2026-08-09：把可操作错误的detail_code写入新Trace执行摘要，旧R2因字段
  可选仍保持Schema兼容。
- 2026-08-09：离线类型复核优先使用实验运行时实际识别到的初始类型，
  并保存脱敏类型与内容校验问题码，避免按数据集声明重新推断造成口径漂移。
- 2026-07-28：新增B0/B1/B2固定配置、BioToolEval题目到脱敏Trace的
  确定性转换、JSON Schema校验和JSONL保存/加载。

设计原则：
1. Trace不保存问题正文、原始序列、完整文件路径或最终答案全文；
2. B0/B1/B2只改变类型检查和修复开关，其他实验上下文必须显式记录；
3. 类型有效性由同一注册表离线复核，不信任自由文本自报；
4. 人工或评分器尚未给出的标签必须保留null，禁止默认猜成正确。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from app.core.config import Config
from app.core.logger import get_logger
from scripts.type_checker import (
    AMINO_ACID_SEQUENCE,
    PDB_CHAIN_IDENTIFIER,
    PDB_STRUCTURE_FILE,
    SINGLE_PROTEIN_FASTA_FILE,
    validate_tool_plan,
)


logger = get_logger("EvaluationTrace", "evaluation_trace.log")

EXPERIMENT_CONDITIONS = {
    "B0": {
        "TYPE_PLANNING_ENABLED": False,
        "TYPE_VALIDATION_ENABLED": False,
        "TYPE_REPAIR_ENABLED": False,
    },
    "B1": {
        "TYPE_PLANNING_ENABLED": True,
        "TYPE_VALIDATION_ENABLED": True,
        "TYPE_REPAIR_ENABLED": False,
    },
    "B2": {
        "TYPE_PLANNING_ENABLED": True,
        "TYPE_VALIDATION_ENABLED": True,
        "TYPE_REPAIR_ENABLED": True,
    },
    "E0": {
        "TYPE_PLANNING_ENABLED": False,
        "TYPE_VALIDATION_ENABLED": False,
        "TYPE_REPAIR_ENABLED": False,
    },
    "E1": {
        "TYPE_PLANNING_ENABLED": True,
        "TYPE_VALIDATION_ENABLED": False,
        "TYPE_REPAIR_ENABLED": False,
    },
    "E2": {
        "TYPE_PLANNING_ENABLED": True,
        "TYPE_VALIDATION_ENABLED": True,
        "TYPE_REPAIR_ENABLED": False,
    },
    "E3": {
        "TYPE_PLANNING_ENABLED": True,
        "TYPE_VALIDATION_ENABLED": True,
        "TYPE_REPAIR_ENABLED": True,
    },
    "S1": {
        "TYPE_PLANNING_ENABLED": False,
        "TYPE_VALIDATION_ENABLED": False,
        "TYPE_REPAIR_ENABLED": False,
    },
    "L1": {
        "TYPE_PLANNING_ENABLED": True,
        "TYPE_VALIDATION_ENABLED": True,
        "TYPE_REPAIR_ENABLED": False,
    },
    "F1": {
        "TYPE_PLANNING_ENABLED": True,
        "TYPE_VALIDATION_ENABLED": True,
        "TYPE_REPAIR_ENABLED": True,
    },
}

LEGACY_EXPERIMENT_CONDITIONS = ("B0", "B1", "B2")
FACTORIAL_EXPERIMENT_CONDITIONS = ("E0", "E1", "E2", "E3")
BASELINE_EXPERIMENT_CONDITIONS = ("S1", "L1")
HYBRID_EXPERIMENT_CONDITIONS = ("F1",)

_EVALUATION_FIELDS = (
    "final_answer_correct",
    "repair_correct",
    "result_hallucination",
    "original_plan_executable",
    "correct_rejection",
)


def _error_result(
    message: str,
    code: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造Trace模块统一错误结构。"""
    try:
        issue = {"code": code, "message": message}
        if isinstance(details, dict):
            issue.update(copy.deepcopy(details))
        return {
            "status": "error",
            "message": message,
            "issues": [issue],
        }
    except Exception as exc:
        logger.error("构造实验Trace错误失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": str(exc),
            "issues": [],
        }


def get_experiment_condition(condition: Any) -> dict[str, Any]:
    """返回历史或正交实验条件的固定开关副本。"""
    try:
        normalized = condition.strip() if isinstance(condition, str) else ""
        if normalized not in EXPERIMENT_CONDITIONS:
            return _error_result(
                "实验条件必须属于B0/B1/B2、E0-E3、S1/L1或F1",
                "UNKNOWN_EXPERIMENT_CONDITION",
            )
        return {
            "status": "success",
            "condition": normalized,
            "env": copy.deepcopy(EXPERIMENT_CONDITIONS[normalized]),
        }
    except Exception as exc:
        logger.error("读取实验条件失败: %s", exc, exc_info=True)
        return _error_result(
            f"读取实验条件失败: {exc}",
            "CONDITION_EXCEPTION",
        )


def _load_trace_validator(
    *,
    schema_path: Any = None,
    schema_version: str = "1.0",
    config: Any = None,
) -> dict[str, Any]:
    """加载v1.0 Schema；E条件和外部baseline分别使用独立扩展。"""
    try:
        cfg = config if config is not None else Config()
        raw_path = (
            schema_path
            if schema_path is not None
            else getattr(cfg, "EXPERIMENT_TRACE_SCHEMA_PATH", "")
        )
        path = Path(str(raw_path or "")).expanduser()
        if not path.is_file():
            return _error_result(
                "实验Trace Schema文件不存在",
                "TRACE_SCHEMA_NOT_FOUND",
                details={"file_name": path.name},
            )
        schema = json.loads(path.read_text(encoding="utf-8"))
        extension_attributes = {
            "1.1": "EXPERIMENT_TRACE_EXTENSION_PATH",
            "1.2": "EXPERIMENT_TRACE_BASELINE_EXTENSION_PATH",
            "1.3": "EXPERIMENT_TRACE_HYBRID_EXTENSION_PATH",
        }
        if schema_version in extension_attributes:
            extension_attribute = extension_attributes[schema_version]
            extension_path = Path(
                str(
                    getattr(
                        cfg,
                        extension_attribute,
                        "",
                    )
                    or ""
                )
            ).expanduser()
            if not extension_path.is_file():
                return _error_result(
                    "Trace Schema扩展文件不存在",
                    "TRACE_SCHEMA_EXTENSION_NOT_FOUND",
                    details={"file_name": extension_path.name},
                )
            extension = json.loads(
                extension_path.read_text(encoding="utf-8")
            )
            if (
                extension.get("x-version") != schema_version
                or extension.get("base_schema") != path.name
                or not isinstance(extension.get("condition_enum"), list)
                or not isinstance(extension.get("plan_properties"), dict)
            ):
                return _error_result(
                    "Trace Schema扩展格式无效",
                    "TRACE_SCHEMA_EXTENSION_INVALID",
                )
            schema = copy.deepcopy(schema)
            schema["$id"] = extension.get("$id")
            schema["title"] = extension.get("title")
            schema["x-version"] = schema_version
            schema["properties"]["schema_version"] = {
                "const": schema_version
            }
            schema["properties"]["condition"] = {
                "enum": list(extension["condition_enum"])
            }
            plan_schema = schema["properties"]["observation"][
                "properties"
            ]["plan"]
            plan_schema["properties"].update(
                copy.deepcopy(extension["plan_properties"])
            )
            plan_schema["required"] = list(plan_schema["required"]) + list(
                extension["plan_properties"]
            )
        elif schema_version != "1.0":
            return _error_result(
                "不支持的实验Trace Schema版本",
                "UNSUPPORTED_TRACE_SCHEMA_VERSION",
                details={"schema_version": schema_version},
            )
        Draft202012Validator.check_schema(schema)
        return {
            "status": "success",
            "validator": Draft202012Validator(schema),
            "schema_version": schema.get("x-version"),
        }
    except Exception as exc:
        logger.error("加载实验Trace Schema失败: %s", exc, exc_info=True)
        return _error_result(
            f"加载实验Trace Schema失败: {exc}",
            "TRACE_SCHEMA_EXCEPTION",
        )


def validate_experiment_trace(
    trace: Any,
    *,
    schema_path: Any = None,
    config: Any = None,
) -> dict[str, Any]:
    """依据固定Schema校验一条脱敏实验Trace。"""
    try:
        if not isinstance(trace, dict):
            return _error_result(
                "实验Trace必须是字典",
                "INVALID_TRACE_OBJECT",
            )
        validator_result = _load_trace_validator(
            schema_path=schema_path,
            schema_version=str(trace.get("schema_version") or ""),
            config=config,
        )
        if validator_result.get("status") != "success":
            return validator_result
        errors = sorted(
            validator_result["validator"].iter_errors(trace),
            key=lambda item: list(item.absolute_path),
        )
        if errors:
            first = errors[0]
            path_text = ".".join(str(part) for part in first.absolute_path)
            return _error_result(
                f"实验Trace不符合Schema: {first.message}",
                "TRACE_SCHEMA_VALIDATION_FAILED",
                details={"path": path_text},
            )
        return {
            "status": "success",
            "valid": True,
            "schema_version": validator_result.get("schema_version"),
        }
    except Exception as exc:
        logger.error("校验实验Trace失败: %s", exc, exc_info=True)
        return _error_result(
            f"校验实验Trace失败: {exc}",
            "TRACE_VALIDATION_EXCEPTION",
        )


def _initial_types_from_question(question_item: dict[str, Any]) -> list[str]:
    """把BioToolEval输入标注映射为类型检查器的初始类型。"""
    try:
        input_data = question_item.get("input") or {}
        input_type = input_data.get("input_type")
        if input_type == "amino_acid_sequence":
            return [AMINO_ACID_SEQUENCE]
        if input_type == "protein_fasta_file":
            return [SINGLE_PROTEIN_FASTA_FILE]
        if input_type == "pdb_file":
            types = [PDB_STRUCTURE_FILE]
            chain_id = input_data.get("chain_id")
            if isinstance(chain_id, str) and chain_id.strip():
                types.append(PDB_CHAIN_IDENTIFIER)
            return types
        return []
    except Exception as exc:
        logger.error("映射评测题初始类型失败: %s", exc, exc_info=True)
        return []


def _posthoc_plan_validation(
    tool_chain: Any,
    initial_types: list[str],
) -> dict[str, Any]:
    """使用冻结类型注册表离线复核计划，不执行任何工具。"""
    try:
        if not isinstance(tool_chain, list) or not tool_chain:
            return {
                "valid": False,
                "issue_codes": ["EMPTY_PLAN"],
            }
        if not initial_types:
            return {
                "valid": False,
                "issue_codes": ["MISSING_INPUT_TYPE"],
            }
        result = validate_tool_plan(tool_chain, initial_types)
        if result.get("status") != "success":
            return {
                "valid": False,
                "issue_codes": ["TYPE_CHECKER_FAILURE"],
            }
        return {
            "valid": result.get("valid") is True,
            "issue_codes": sorted(
                {
                    str(item.get("code"))
                    for item in result.get("issues") or []
                    if isinstance(item, dict) and item.get("code")
                }
            ),
        }
    except Exception as exc:
        logger.error("离线复核工具计划失败: %s", exc, exc_info=True)
        return {
            "valid": False,
            "issue_codes": ["POSTHOC_VALIDATION_EXCEPTION"],
        }


def _normalize_execution_summary(reasoning: dict[str, Any]) -> dict[str, Any]:
    """规范主链路execution_summary，缺失字段安全回退。"""
    try:
        summary = reasoning.get("execution_summary") or {}
        attempted = int(summary.get("attempted_steps") or 0)
        successful = int(summary.get("successful_steps") or 0)
        failed = int(summary.get("failed_steps") or 0)
        return {
            "completed": bool(summary.get("completed", False)),
            "attempted_steps": max(0, attempted),
            "successful_steps": max(0, successful),
            "failed_steps": max(0, failed),
            "error_type": (
                str(summary.get("error_type"))
                if summary.get("error_type")
                else None
            ),
            "detail_code": (
                str(summary.get("detail_code"))
                if summary.get("detail_code")
                else None
            ),
        }
    except Exception as exc:
        logger.error("规范实验执行摘要失败: %s", exc, exc_info=True)
        return {
            "completed": False,
            "attempted_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "error_type": "invalid_execution_summary",
            "detail_code": "EXECUTION_SUMMARY_NORMALIZATION_FAILURE",
        }


def _normalize_repair_operations(typed_trace: dict[str, Any]) -> list[dict[str, Any]]:
    """仅保留固定类型证明字段，避免自由文本或输入值进入公开实验Trace。"""
    try:
        normalized: list[dict[str, Any]] = []
        for raw_item in typed_trace.get("repair_operations") or []:
            if not isinstance(raw_item, dict):
                continue
            operation = raw_item.get("operation")
            tool_name = raw_item.get("tool_name")
            if operation not in {
                "deterministic_bridge_insert",
                "redundant_step_prune",
            } or not isinstance(tool_name, str) or not tool_name.strip():
                continue

            def _optional_nonnegative_int(value: Any) -> int | None:
                return value if isinstance(value, int) and value >= 0 else None

            def _type_names(value: Any) -> list[str]:
                if not isinstance(value, list):
                    return []
                return list(
                    dict.fromkeys(
                        item.strip()
                        for item in value
                        if isinstance(item, str) and item.strip()
                    )
                )

            normalized.append(
                {
                    "operation": operation,
                    "tool_name": tool_name.strip(),
                    "original_step_index": _optional_nonnegative_int(
                        raw_item.get("original_step_index")
                    ),
                    "before_step_index": _optional_nonnegative_int(
                        raw_item.get("before_step_index")
                    ),
                    "issue_code": (
                        str(raw_item.get("issue_code"))
                        if raw_item.get("issue_code")
                        else None
                    ),
                    "input_types": _type_names(raw_item.get("input_types")),
                    "output_types": _type_names(raw_item.get("output_types")),
                    "downstream_required_types": _type_names(
                        raw_item.get("downstream_required_types")
                    ),
                    "registry_opt_in": bool(
                        raw_item.get("registry_opt_in", False)
                    ),
                    "duplicate_group_size": _optional_nonnegative_int(
                        raw_item.get("duplicate_group_size")
                    ),
                    "output_consumed_downstream": (
                        raw_item.get("output_consumed_downstream")
                        if isinstance(
                            raw_item.get("output_consumed_downstream"), bool
                        )
                        else None
                    ),
                    "candidate_plan_type_valid": bool(
                        raw_item.get("candidate_plan_type_valid", False)
                    ),
                    "bridge_path_length": _optional_nonnegative_int(
                        raw_item.get("bridge_path_length")
                    ),
                }
            )
        return normalized
    except Exception as exc:
        logger.error("规范修复操作证明失败: %s", exc, exc_info=True)
        return []


def _validate_condition_observation(
    condition: str,
    typed_trace: dict[str, Any],
    request_guard: dict[str, Any],
    baseline_trace: dict[str, Any],
    recovery_trace: dict[str, Any],
) -> dict[str, Any]:
    """防止实验运行状态与历史/正交条件标签不一致。"""
    try:
        if condition in {"B0", "E0", "S1"}:
            if typed_trace:
                return _error_result(
                    "B0响应不应包含已启用的类型规划轨迹",
                    "CONDITION_TRACE_MISMATCH",
                )
            if condition == "S1":
                if (
                    baseline_trace.get("baseline_kind")
                    != "schema_runtime_validation"
                    or baseline_trace.get("original_plan_preserved") is not True
                    or baseline_trace.get("extra_llm_calls") != 0
                ):
                    return _error_result(
                        "S1响应缺少一致的Schema/runtime baseline证据",
                        "CONDITION_TRACE_MISMATCH",
                    )
            return {"status": "success"}

        # 显式未注册工具门禁是三个条件共有的规划前前置条件。只有门禁状态、
        # 决策、错误码和未知工具列表四项证据同时成立时，才允许B1/B2没有
        # typed_planning轨迹；其他缺轨迹情形仍按实验配置错误处理。
        unknown_tools = request_guard.get("unknown_tools") or []
        is_verified_preplan_rejection = (
            not typed_trace
            and request_guard.get("status") == "success"
            and request_guard.get("decision") == "reject"
            and request_guard.get("detail_code") == "UNREGISTERED_TOOL"
            and isinstance(unknown_tools, list)
            and any(
                isinstance(tool_name, str) and tool_name.strip()
                for tool_name in unknown_tools
            )
        )
        if is_verified_preplan_rejection:
            if condition != "L1" or (
                baseline_trace.get("baseline_kind")
                == "matched_budget_llm_replan"
            ):
                return {"status": "success"}

        if not typed_trace or typed_trace.get("enabled") is not True:
            return _error_result(
                f"{condition}响应缺少已启用的类型规划轨迹",
                "CONDITION_TRACE_MISMATCH",
            )
        expected_repair = (
            EXPERIMENT_CONDITIONS[condition]["TYPE_REPAIR_ENABLED"]
        )
        if bool(typed_trace.get("repair_enabled", False)) != expected_repair:
            return _error_result(
                f"{condition}响应的修复开关与冻结配置不一致",
                "CONDITION_TRACE_MISMATCH",
            )
        expected_validation = EXPERIMENT_CONDITIONS[condition][
            "TYPE_VALIDATION_ENABLED"
        ]
        observed_validation = typed_trace.get("validation_enabled")
        if observed_validation is None and condition in {"B1", "B2"}:
            observed_validation = True
        if not isinstance(observed_validation, bool) or (
            observed_validation != expected_validation
        ):
            return _error_result(
                f"{condition}响应的类型门禁开关与冻结配置不一致",
                "CONDITION_TRACE_MISMATCH",
            )
        if condition == "L1":
            triggered = baseline_trace.get("replan_triggered")
            applied = baseline_trace.get("replan_applied")
            extra_calls = baseline_trace.get("extra_llm_calls")
            if (
                baseline_trace.get("baseline_kind")
                != "matched_budget_llm_replan"
                or baseline_trace.get("original_plan_preserved") is not True
                or not isinstance(triggered, bool)
                or not isinstance(applied, bool)
                or extra_calls not in {0, 1}
                or extra_calls != int(triggered)
                or (applied and not triggered)
            ):
                return _error_result(
                    "L1响应缺少一次重规划的一致记账证据",
                    "CONDITION_TRACE_MISMATCH",
                )
        if condition == "F1":
            triggered = recovery_trace.get("replan_triggered")
            applied = recovery_trace.get("replan_applied")
            deterministic_applied = recovery_trace.get(
                "deterministic_repair_applied"
            )
            extra_calls = recovery_trace.get("extra_llm_calls")
            final_source = recovery_trace.get("final_plan_source")
            if (
                recovery_trace.get("recovery_kind")
                != "deterministic_then_typed_llm_replan"
                or recovery_trace.get("original_plan_preserved") is not True
                or not isinstance(triggered, bool)
                or not isinstance(applied, bool)
                or not isinstance(deterministic_applied, bool)
                or extra_calls not in {0, 1}
                or extra_calls != int(triggered)
                or (applied and not triggered)
                or (deterministic_applied and triggered)
                or final_source
                not in {
                    "original",
                    "deterministic_repair",
                    "llm_replan",
                    "rejected",
                }
            ):
                return _error_result(
                    "F1响应缺少分层恢复的一致记账证据",
                    "CONDITION_TRACE_MISMATCH",
                )
        return {"status": "success"}
    except Exception as exc:
        logger.error("校验实验条件与响应一致性失败: %s", exc, exc_info=True)
        return _error_result(
            f"校验实验条件与响应一致性失败: {exc}",
            "CONDITION_TRACE_CHECK_EXCEPTION",
        )


def build_experiment_trace(
    question_item: Any,
    agent_result: Any,
    *,
    condition: Any,
    run_context: Any,
    evaluation_labels: Any = None,
    config: Any = None,
) -> dict[str, Any]:
    """把一题Agent响应转换为不含敏感原值的v1.0实验Trace。"""
    try:
        if not isinstance(question_item, dict):
            return _error_result(
                "question_item必须是BioToolEval题目字典",
                "INVALID_QUESTION_ITEM",
            )
        if not isinstance(agent_result, dict):
            return _error_result(
                "agent_result必须是Agent响应字典",
                "INVALID_AGENT_RESULT",
            )
        if not isinstance(run_context, dict):
            return _error_result(
                "run_context必须是实验上下文字典",
                "INVALID_RUN_CONTEXT",
            )

        condition_result = get_experiment_condition(condition)
        if condition_result.get("status") != "success":
            return condition_result

        gold = question_item.get("gold")
        if not isinstance(gold, dict):
            return _error_result(
                "题目缺少已验收的gold字段；candidate submitted_gold不能用于实验",
                "UNVERIFIED_GOLD",
            )

        reasoning = agent_result.get("reasoning") or {}
        if not isinstance(reasoning, dict):
            reasoning = {}
        typed_trace = reasoning.get("typed_planning") or {}
        if not isinstance(typed_trace, dict):
            typed_trace = {}
        request_guard = reasoning.get("request_guard") or {}
        if not isinstance(request_guard, dict):
            request_guard = {}
        plan_sampling = reasoning.get("plan_sampling") or {}
        if not isinstance(plan_sampling, dict):
            plan_sampling = {}
        baseline_trace = reasoning.get("baseline") or {}
        if not isinstance(baseline_trace, dict):
            baseline_trace = {}
        recovery_trace = reasoning.get("recovery") or {}
        if not isinstance(recovery_trace, dict):
            recovery_trace = {}
        condition_trace_result = _validate_condition_observation(
            condition_result["condition"],
            typed_trace,
            request_guard,
            baseline_trace,
            recovery_trace,
        )
        if condition_trace_result.get("status") != "success":
            return condition_trace_result

        planned_chain = (
            recovery_trace.get("original_plan_chain")
            if condition_result["condition"] in HYBRID_EXPERIMENT_CONDITIONS
            else
            baseline_trace.get("original_plan_chain")
            if condition_result["condition"] in BASELINE_EXPERIMENT_CONDITIONS
            else reasoning.get("plan_chain")
        ) or []
        executed_chain = (
            reasoning.get("plan_chain_executed")
            or typed_trace.get("plan_after")
            or planned_chain
        )
        planned_chain = (
            [str(item) for item in planned_chain]
            if isinstance(planned_chain, list)
            else []
        )
        executed_chain = (
            [str(item) for item in executed_chain]
            if isinstance(executed_chain, list)
            else []
        )

        runtime_initial_types = run_context.get("initial_types")
        initial_types = (
            [
                str(value)
                for value in runtime_initial_types
                if isinstance(value, str) and value.strip()
            ]
            if isinstance(runtime_initial_types, list)
            else _initial_types_from_question(question_item)
        )
        initial_issue_codes = sorted(
            {
                str(value)
                for value in (
                    run_context.get("initial_artifact_issue_codes") or []
                )
                if isinstance(value, str) and value.strip()
            }
        )
        before_validation = _posthoc_plan_validation(
            planned_chain,
            initial_types,
        )
        after_validation = _posthoc_plan_validation(
            executed_chain,
            initial_types,
        )
        typed_issue_codes = {
            str(item.get("code"))
            for item in typed_trace.get("planning_issues") or []
            if isinstance(item, dict) and item.get("code")
        }
        issue_codes_before = sorted(
            set(before_validation["issue_codes"])
        )
        issue_codes_after = sorted(
            set(after_validation["issue_codes"])
            | typed_issue_codes
        )
        repair_strategy = typed_trace.get("repair_strategy")
        if condition_result["condition"] in HYBRID_EXPERIMENT_CONDITIONS:
            repair_strategy = recovery_trace.get(
                "deterministic_repair_strategy"
            )
        if repair_strategy not in {
            "none",
            "deterministic_bridge_insertion",
            "duplicate_invalid_step_pruning",
        }:
            repair_strategy = "none"

        labels_source = (
            evaluation_labels
            if isinstance(evaluation_labels, dict)
            else {}
        )
        evaluation = {
            field: (
                labels_source.get(field)
                if isinstance(labels_source.get(field), bool)
                else None
            )
            for field in _EVALUATION_FIELDS
        }

        context = {
            "model": str(run_context.get("model") or "").strip(),
            "temperature": float(run_context.get("temperature", 0.0)),
            "seed": (
                int(run_context["seed"])
                if isinstance(run_context.get("seed"), int)
                else None
            ),
            "git_commit": str(
                run_context.get("git_commit") or ""
            ).strip(),
            "dataset_version": str(
                run_context.get("dataset_version") or ""
            ).strip(),
            "kg_snapshot": (
                str(run_context["kg_snapshot"])
                if run_context.get("kg_snapshot")
                else None
            ),
            "python_version": (
                str(run_context["python_version"])
                if run_context.get("python_version")
                else None
            ),
        }
        total_latency_ms = run_context.get("total_latency_ms")
        if not isinstance(total_latency_ms, (int, float)):
            response_latency = agent_result.get("latency")
            total_latency_ms = (
                float(response_latency) * 1000.0
                if isinstance(response_latency, (int, float))
                else 0.0
            )
        typed_runtime_ms = typed_trace.get("total_duration_ms")
        if not isinstance(typed_runtime_ms, (int, float)):
            typed_runtime_ms = None

        trace_schema_version = (
            "1.1"
            if condition_result["condition"] in FACTORIAL_EXPERIMENT_CONDITIONS
            else "1.2"
            if condition_result["condition"] in BASELINE_EXPERIMENT_CONDITIONS
            else "1.3"
            if condition_result["condition"] in HYBRID_EXPERIMENT_CONDITIONS
            else "1.0"
        )
        trace = {
            "schema_version": trace_schema_version,
            "run_id": str(run_context.get("run_id") or "").strip(),
            "question_id": str(question_item.get("id") or "").strip(),
            "split": question_item.get("split"),
            "category": question_item.get("category"),
            "difficulty": question_item.get("difficulty"),
            "input_type": (question_item.get("input") or {}).get(
                "input_type"
            ),
            "condition": condition_result["condition"],
            "repeat_index": int(run_context.get("repeat_index", 0)),
            "context": context,
            "gold": {
                "preferred_tool_chain": list(
                    gold.get("preferred_tool_chain") or []
                ),
                "acceptable_tool_chains": copy.deepcopy(
                    gold.get("acceptable_tool_chains") or []
                ),
                "order_sensitive": bool(
                    gold.get("order_sensitive", False)
                ),
                "expected_status": gold.get("expected_status"),
                "expected_behavior": gold.get("expected_behavior"),
            },
            "observation": {
                "plan": {
                    **(
                        {
                            "typed_runtime_enabled": bool(
                                EXPERIMENT_CONDITIONS[
                                    condition_result["condition"]
                                ]["TYPE_PLANNING_ENABLED"]
                            ),
                            "validation_enabled": bool(
                                EXPERIMENT_CONDITIONS[
                                    condition_result["condition"]
                                ]["TYPE_VALIDATION_ENABLED"]
                            ),
                        }
                        if trace_schema_version == "1.1"
                        else {}
                    ),
                    **(
                        {
                            "baseline_kind": str(
                                baseline_trace.get("baseline_kind") or ""
                            ),
                            "schema_valid_before": bool(
                                baseline_trace.get(
                                    "schema_valid_before",
                                    False,
                                )
                            ),
                            "original_plan_preserved": bool(
                                baseline_trace.get(
                                    "original_plan_preserved",
                                    False,
                                )
                            ),
                            "original_plan_sample_id": (
                                str(
                                    baseline_trace.get(
                                        "original_plan_sample_id"
                                    )
                                    or plan_sampling.get("plan_sample_id")
                                )
                                if (
                                    baseline_trace.get(
                                        "original_plan_sample_id"
                                    )
                                    or plan_sampling.get("plan_sample_id")
                                )
                                else None
                            ),
                            "replan_triggered": bool(
                                baseline_trace.get("replan_triggered", False)
                            ),
                            "replan_applied": bool(
                                baseline_trace.get("replan_applied", False)
                            ),
                            "replan_plan_sample_id": (
                                str(
                                    baseline_trace.get(
                                        "replan_plan_sample_id"
                                    )
                                )
                                if baseline_trace.get("replan_plan_sample_id")
                                else None
                            ),
                            "extra_llm_calls": max(
                                0,
                                int(
                                    baseline_trace.get(
                                        "extra_llm_calls",
                                        0,
                                    )
                                ),
                            ),
                            "extra_llm_input_tokens": (
                                max(
                                    0,
                                    int(
                                        baseline_trace[
                                            "extra_llm_input_tokens"
                                        ]
                                    ),
                                )
                                if isinstance(
                                    baseline_trace.get(
                                        "extra_llm_input_tokens"
                                    ),
                                    (int, float),
                                )
                                else None
                            ),
                            "extra_llm_output_tokens": (
                                max(
                                    0,
                                    int(
                                        baseline_trace[
                                            "extra_llm_output_tokens"
                                        ]
                                    ),
                                )
                                if isinstance(
                                    baseline_trace.get(
                                        "extra_llm_output_tokens"
                                    ),
                                    (int, float),
                                )
                                else None
                            ),
                            "extra_llm_total_tokens": (
                                max(
                                    0,
                                    int(
                                        baseline_trace[
                                            "extra_llm_total_tokens"
                                        ]
                                    ),
                                )
                                if isinstance(
                                    baseline_trace.get(
                                        "extra_llm_total_tokens"
                                    ),
                                    (int, float),
                                )
                                else None
                            ),
                            "replan_latency_ms": max(
                                0.0,
                                float(
                                    baseline_trace.get(
                                        "replan_latency_ms",
                                        0.0,
                                    )
                                ),
                            ),
                            "replan_outcome": str(
                                baseline_trace.get("replan_outcome")
                                or "not_triggered"
                            ),
                        }
                        if trace_schema_version == "1.2"
                        else {}
                    ),
                    **(
                        {
                            key: copy.deepcopy(recovery_trace.get(key))
                            for key in (
                                "recovery_kind",
                                "original_plan_preserved",
                                "original_plan_sample_id",
                                "deterministic_repair_attempted",
                                "deterministic_repair_applied",
                                "deterministic_repair_strategy",
                                "deterministic_plan_sample_id",
                                "replan_triggered",
                                "replan_applied",
                                "replan_plan_sample_id",
                                "extra_llm_calls",
                                "extra_llm_input_tokens",
                                "extra_llm_output_tokens",
                                "extra_llm_total_tokens",
                                "replan_latency_ms",
                                "replan_outcome",
                                "final_plan_source",
                            )
                        }
                        if trace_schema_version == "1.3"
                        else {}
                    ),
                    "planned_chain": planned_chain,
                    "executed_chain": executed_chain,
                    "initial_types": initial_types,
                    "initial_artifact_issue_codes": initial_issue_codes,
                    "type_valid_before": before_validation["valid"],
                    "type_valid_after": after_validation["valid"],
                    "repair_attempted": bool(
                        recovery_trace.get("deterministic_repair_attempted")
                        if trace_schema_version == "1.3"
                        else typed_trace.get("repair_attempted", False)
                    ),
                    "repair_applied": bool(
                        recovery_trace.get("deterministic_repair_applied")
                        if trace_schema_version == "1.3"
                        else typed_trace.get("repair_applied", False)
                    ),
                    "repair_strategy": repair_strategy,
                    "repair_operations": _normalize_repair_operations(
                        {
                            "repair_operations": recovery_trace.get(
                                "deterministic_repair_operations"
                            )
                        }
                        if trace_schema_version == "1.3"
                        else typed_trace
                    ),
                    "issue_codes_before": issue_codes_before,
                    "issue_codes_after": issue_codes_after,
                    "plan_sample_id": (
                        str(plan_sampling.get("plan_sample_id"))
                        if plan_sampling.get("plan_sample_id")
                        else None
                    ),
                    "shared_plan_replay": bool(
                        plan_sampling.get("shared_plan_replay", False)
                    ),
                    "capture_outcome": (
                        str(plan_sampling.get("capture_outcome"))
                        if plan_sampling.get("capture_outcome")
                        not in {None, "", "invalid"}
                        else None
                    ),
                    "selected_attempt_index": (
                        int(plan_sampling["selected_attempt_index"])
                        if isinstance(
                            plan_sampling.get("selected_attempt_index"),
                            int,
                        )
                        else None
                    ),
                    "plan_attempt_count": max(
                        0,
                        int(plan_sampling.get("attempt_count", 0)),
                    ),
                    "plan_response_sha256": (
                        str(plan_sampling.get("response_sha256"))
                        if plan_sampling.get("response_sha256")
                        else None
                    ),
                    "plan_attempts": [
                        {
                            "attempt_index": max(
                                0,
                                int(item.get("attempt_index", 0)),
                            ),
                            "attempt_kind": str(
                                item.get("attempt_kind") or ""
                            ),
                            "outcome": str(item.get("outcome") or ""),
                            "exception_type": (
                                str(item.get("exception_type"))
                                if item.get("exception_type")
                                else None
                            ),
                            "response_sha256": (
                                str(item.get("response_sha256"))
                                if item.get("response_sha256")
                                else None
                            ),
                            "latency_ms": max(
                                0.0,
                                float(item.get("latency_ms", 0.0)),
                            ),
                        }
                        for item in plan_sampling.get("attempts") or []
                        if isinstance(item, dict)
                    ],
                },
                "execution": _normalize_execution_summary(reasoning),
                "latency": {
                    "total_ms": max(0.0, float(total_latency_ms)),
                    "typed_runtime_ms": (
                        max(0.0, float(typed_runtime_ms))
                        if typed_runtime_ms is not None
                        else None
                    ),
                },
            },
            "evaluation": evaluation,
        }

        validation = validate_experiment_trace(trace, config=config)
        if validation.get("status") != "success":
            return validation
        logger.info(
            "实验Trace构建完成 | run=%s | question=%s | condition=%s",
            trace["run_id"],
            trace["question_id"],
            trace["condition"],
        )
        return {
            "status": "success",
            "trace": trace,
        }
    except Exception as exc:
        logger.error("构建实验Trace失败: %s", exc, exc_info=True)
        return _error_result(
            f"构建实验Trace失败: {exc}",
            "TRACE_BUILD_EXCEPTION",
        )


def append_experiment_trace(
    trace: Any,
    *,
    output_path: Any = None,
    config: Any = None,
) -> dict[str, Any]:
    """校验后把一条Trace追加到本地JSONL；不允许写出项目外默认目录。"""
    try:
        cfg = config if config is not None else Config()
        validation = validate_experiment_trace(trace, config=cfg)
        if validation.get("status") != "success":
            return validation

        if output_path is None:
            results_dir = Path(
                getattr(cfg, "EXPERIMENT_RESULTS_DIR", "")
            ).expanduser()
            path = results_dir / f"{trace['run_id']}.jsonl"
        else:
            path = Path(str(output_path)).expanduser()
        if path.suffix.lower() != ".jsonl":
            return _error_result(
                "实验Trace输出文件必须使用.jsonl扩展名",
                "INVALID_TRACE_OUTPUT_PATH",
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
        return {
            "status": "success",
            "output_path": str(path.resolve()),
        }
    except Exception as exc:
        logger.error("追加实验Trace失败: %s", exc, exc_info=True)
        return _error_result(
            f"追加实验Trace失败: {exc}",
            "TRACE_APPEND_EXCEPTION",
        )


def load_experiment_traces(
    input_path: Any,
    *,
    config: Any = None,
) -> dict[str, Any]:
    """加载并逐条校验JSONL实验Trace。"""
    try:
        path = Path(str(input_path or "")).expanduser()
        if not path.is_file() or path.suffix.lower() != ".jsonl":
            return _error_result(
                "实验Trace JSONL文件不存在或扩展名错误",
                "TRACE_INPUT_NOT_FOUND",
            )
        records: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except Exception as exc:
                return _error_result(
                    f"第{line_number}行不是合法JSON: {exc}",
                    "INVALID_TRACE_JSONL",
                    details={"line_number": line_number},
                )
            validation = validate_experiment_trace(record, config=config)
            if validation.get("status") != "success":
                return _error_result(
                    f"第{line_number}行Trace校验失败",
                    "INVALID_TRACE_RECORD",
                    details={
                        "line_number": line_number,
                        "validation": validation,
                    },
                )
            records.append(record)
        if not records:
            return _error_result(
                "实验Trace文件不包含有效记录",
                "EMPTY_TRACE_FILE",
            )
        return {
            "status": "success",
            "record_count": len(records),
            "records": records,
        }
    except Exception as exc:
        logger.error("加载实验Trace失败: %s", exc, exc_info=True)
        return _error_result(
            f"加载实验Trace失败: {exc}",
            "TRACE_LOAD_EXCEPTION",
        )
