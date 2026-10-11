# -*- coding: utf-8 -*-
"""
BioAgent 结构化工具计划适配层。

修改记录：
- 2026-09-01：空或不可解析计划也保留类型规划、校验与修复开关证据，
  避免E1/E3自然规划失败被误记为实验条件Trace不一致。
- 2026-08-27：新增不启用类型门禁的确定性Artifact绑定计划，支持把
  类型化运行时、类型校验和修复拆成可归因的E1/E2/E3实验因素。
- 2026-08-15：透传安全剪枝/桥接修复策略与脱敏操作证明，供运行时和实验Trace审计。
- 2026-07-25：新增v0.2结构化Plan Step、旧Plan Chain确定性适配、
  保守修复编排、配置开关和统一计划追踪日志。
- 2026-07-26：由v0.3b协调层接入主链路，模块本身仍保持可独立测试。

设计边界：
1. 本模块只把已提取的工具名列表转换为显式数据来源，不调用LLM猜测来源；
2. 输入来源唯一时才生成结构化步骤，多来源歧义必须拒绝；
3. 可选修复只复用type_checker.py中已验收的确定性桥接或安全剪枝；
4. v0.3b通过协调层调用本模块；配置关闭时run_SciToolAgent保持baseline。
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

from app.core.config import Config
from app.core.logger import get_logger
from app.llms.extraction import extract_tool_path
from scripts.type_checker import (
    get_tool_signatures,
    repair_tool_plan,
    validate_tool_plan,
)


logger = get_logger("StructuredPlan", "structured_plan.log")

INITIAL_INPUT_SOURCE = "initial_input"
PLAN_STEP_PREFIX = "step_"


@dataclass(frozen=True)
class StructuredPlanStep:
    """可序列化的结构化计划步骤。

    `input_type`始终使用列表，以兼容select_pdb_chain等多输入工具；
    `input_from`按输入类型记录唯一来源，值为initial_input或step_N。
    """

    step_id: str
    step_index: int
    tool_name: str
    input_type: list[str]
    input_from: dict[str, str]
    output_type: list[str]
    inserted_by_repair: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入JSON和日志的字典；失败时返回标准错误结构。"""
        try:
            return asdict(self)
        except Exception as exc:
            logger.error("结构化Plan Step序列化失败: %s", exc, exc_info=True)
            return {
                "status": "error",
                "message": f"结构化Plan Step序列化失败: {exc}",
            }


def _normalize_tool_path(tool_path: Any) -> list[str]:
    """规范化旧工具名列表；任一元素非法时返回空列表。"""
    try:
        if not isinstance(tool_path, (list, tuple)) or not tool_path:
            return []
        normalized: list[str] = []
        for value in tool_path:
            if not isinstance(value, str) or not value.strip():
                return []
            normalized.append(value.strip())
        return normalized
    except Exception as exc:
        logger.error("规范化旧Plan Chain失败: %s", exc, exc_info=True)
        return []


def _normalize_source_reference(source: Any) -> str:
    """把v0.1来源标记转换为稳定的initial_input或step_N引用。"""
    try:
        if source == INITIAL_INPUT_SOURCE:
            return INITIAL_INPUT_SOURCE
        if not isinstance(source, str):
            return ""
        parts = source.split(":", 2)
        if len(parts) != 3 or parts[0] != "step":
            return ""
        source_index = int(parts[1])
        if source_index < 0:
            return ""
        return f"{PLAN_STEP_PREFIX}{source_index + 1}"
    except Exception as exc:
        logger.error("规范化计划输入来源失败: %s", exc, exc_info=True)
        return ""


def _structured_error(
    message: str,
    code: str,
    *,
    original_plan: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成结构化计划层统一错误结构。"""
    try:
        issue = {"code": code, "message": message}
        if isinstance(details, dict):
            issue.update(details)
        return {
            "status": "error",
            "message": message,
            "valid": False,
            "original_plan": list(original_plan or []),
            "structured_plan": [],
            "issues": [issue],
        }
    except Exception as exc:
        logger.error("生成结构化计划错误结构失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": str(exc),
            "valid": False,
            "structured_plan": [],
        }


def build_structured_plan(
    tool_path: Any,
    available_input_types: Any,
) -> dict[str, Any]:
    """把合法旧Plan Chain确定性转换为结构化Plan Step列表。"""
    try:
        normalized_plan = _normalize_tool_path(tool_path)
        if not normalized_plan:
            return _structured_error(
                "工具计划必须是仅含非空工具名的列表",
                "INVALID_PLAN",
            )

        validation = validate_tool_plan(normalized_plan, available_input_types)
        if validation.get("status") != "success":
            return _structured_error(
                validation.get("message", "类型检查器运行失败"),
                "VALIDATION_ERROR",
                original_plan=normalized_plan,
                details={"validation": validation},
            )
        if validation.get("valid") is not True:
            return {
                "status": "success",
                "valid": False,
                "message": "旧Plan Chain未通过类型检查，不能生成结构化计划",
                "original_plan": normalized_plan,
                "structured_plan": [],
                "issues": list(validation.get("issues") or []),
                "validation": validation,
            }

        structured_steps: list[dict[str, Any]] = []
        for step in validation.get("steps") or []:
            step_index = step.get("step_index")
            tool_name = step.get("tool_name")
            required_types = list(step.get("required_types") or [])
            produced_types = list(step.get("produced_types") or [])
            source_map = step.get("input_sources") or {}
            input_from: dict[str, str] = {}

            for input_type in required_types:
                candidates = source_map.get(input_type) or []
                if len(candidates) != 1:
                    reason = (
                        "同一输入类型存在多个候选数据对象，旧Plan Chain没有足够信息确定来源"
                        if len(candidates) > 1
                        else "类型检查通过但缺少可追踪的数据来源"
                    )
                    return {
                        "status": "success",
                        "valid": False,
                        "message": reason,
                        "original_plan": normalized_plan,
                        "structured_plan": [],
                        "issues": [
                            {
                                "code": (
                                    "AMBIGUOUS_INPUT_SOURCE"
                                    if len(candidates) > 1
                                    else "MISSING_INPUT_SOURCE"
                                ),
                                "step_index": step_index,
                                "tool_name": tool_name,
                                "input_type": input_type,
                                "source_candidates": list(candidates),
                                "message": reason,
                            }
                        ],
                        "validation": validation,
                    }

                source_ref = _normalize_source_reference(candidates[0])
                if not source_ref:
                    return _structured_error(
                        "输入来源引用格式无效",
                        "INVALID_INPUT_SOURCE",
                        original_plan=normalized_plan,
                        details={
                            "step_index": step_index,
                            "tool_name": tool_name,
                            "input_type": input_type,
                            "source": candidates[0],
                        },
                    )
                input_from[input_type] = source_ref

            structured_step = StructuredPlanStep(
                step_id=f"{PLAN_STEP_PREFIX}{step_index + 1}",
                step_index=step_index,
                tool_name=tool_name,
                input_type=required_types,
                input_from=input_from,
                output_type=produced_types,
            ).to_dict()
            if structured_step.get("status") == "error":
                return _structured_error(
                    structured_step.get("message", "Plan Step序列化失败"),
                    "STEP_SERIALIZATION_ERROR",
                    original_plan=normalized_plan,
                )
            structured_steps.append(structured_step)

        logger.info(
            "结构化计划生成完成 | valid=True | plan=%s | steps=%s",
            normalized_plan,
            len(structured_steps),
        )
        return {
            "status": "success",
            "valid": True,
            "original_plan": normalized_plan,
            "structured_plan": structured_steps,
            "issues": [],
            "validation": validation,
        }
    except Exception as exc:
        logger.error("生成结构化计划失败: %s", exc, exc_info=True)
        return _structured_error(
            f"生成结构化计划失败: {exc}",
            "STRUCTURED_PLAN_EXCEPTION",
        )


def build_runtime_binding_plan(
    tool_path: Any,
    available_input_types: Any,
) -> dict[str, Any]:
    """只构造Artifact来源绑定，不执行类型合法性门禁。

    该函数仅用于E1消融：来源选择是确定性的（优先最近上游输出，其次初始
    输入）；若所需类型并不存在，仍绑定到initial_input，让Artifact Pool在
    真实执行时自然失败，从而不把类型门禁的收益混入运行时基线。
    """
    try:
        normalized_plan = _normalize_tool_path(tool_path)
        if not normalized_plan:
            return _structured_error("工具计划格式无效", "INVALID_PLAN")
        if not isinstance(available_input_types, (list, tuple, set)):
            return _structured_error(
                "可用输入类型必须是字符串集合",
                "INVALID_AVAILABLE_TYPES",
                original_plan=normalized_plan,
            )
        initial_types = {
            value.strip()
            for value in available_input_types
            if isinstance(value, str) and value.strip()
        }
        signatures_result = get_tool_signatures()
        if signatures_result.get("status") != "success":
            return _structured_error(
                signatures_result.get("message", "无法读取工具签名"),
                "SIGNATURE_REGISTRY_ERROR",
                original_plan=normalized_plan,
            )
        signatures = signatures_result.get("signatures") or {}
        sources_by_type: dict[str, list[str]] = {
            type_name: [INITIAL_INPUT_SOURCE] for type_name in initial_types
        }
        structured_steps: list[dict[str, Any]] = []
        binding_issues: list[dict[str, Any]] = []
        for step_index, tool_name in enumerate(normalized_plan):
            signature = signatures.get(tool_name)
            if not isinstance(signature, dict):
                return _structured_error(
                    "工具计划包含未注册工具",
                    "UNKNOWN_TOOL",
                    original_plan=normalized_plan,
                    details={"step_index": step_index, "tool_name": tool_name},
                )
            required_types = list(signature.get("input_types") or [])
            produced_types = list(signature.get("output_types") or [])
            input_from: dict[str, str] = {}
            for type_name in required_types:
                candidates = sources_by_type.get(type_name) or []
                input_from[type_name] = (
                    candidates[-1] if candidates else INITIAL_INPUT_SOURCE
                )
                if not candidates:
                    binding_issues.append(
                        {
                            "code": "UNRESOLVED_RUNTIME_BINDING",
                            "step_index": step_index,
                            "tool_name": tool_name,
                            "input_type": type_name,
                            "message": "运行时绑定阶段未找到该输入类型，交由Artifact Pool判定。",
                        }
                    )
            step_id = f"{PLAN_STEP_PREFIX}{step_index + 1}"
            structured_steps.append(
                StructuredPlanStep(
                    step_id=step_id,
                    step_index=step_index,
                    tool_name=tool_name,
                    input_type=required_types,
                    input_from=input_from,
                    output_type=produced_types,
                ).to_dict()
            )
            for type_name in produced_types:
                sources_by_type.setdefault(type_name, []).append(step_id)
        return {
            "status": "success",
            "valid": True,
            "validation_enabled": False,
            "validation_bypassed": True,
            "repair_enabled": False,
            "repair_attempted": False,
            "repair_applied": False,
            "repair_strategy": "none",
            "repair_operations": [],
            "plan_before": normalized_plan,
            "plan_after": list(normalized_plan),
            "structured_plan": structured_steps,
            "issues": binding_issues,
        }
    except Exception as exc:
        logger.error("构造运行时绑定计划失败: %s", exc, exc_info=True)
        return _structured_error(
            "构造运行时绑定计划失败",
            "RUNTIME_BINDING_PLAN_EXCEPTION",
            details={"exception_type": type(exc).__name__},
        )


def _get_inserted_positions(inserted_steps: Any) -> list[int]:
    """把修复器的原计划位置转换为修复后计划中的真实下标。"""
    try:
        if not isinstance(inserted_steps, list):
            return []
        positions: list[int] = []
        offset = 0
        for item in inserted_steps:
            if not isinstance(item, dict):
                return []
            before_index = item.get("before_step_index")
            if not isinstance(before_index, int) or before_index < 0:
                return []
            positions.append(before_index + offset)
            offset += 1
        return positions
    except Exception as exc:
        logger.error("计算修复插入位置失败: %s", exc, exc_info=True)
        return []


def prepare_structured_plan(
    tool_path: Any,
    available_input_types: Any,
    *,
    enable_repair: bool = False,
) -> dict[str, Any]:
    """记录检查前、修复后和结构化后的完整计划追踪。"""
    try:
        normalized_plan = _normalize_tool_path(tool_path)
        if not normalized_plan:
            return _structured_error(
                "工具计划必须是仅含非空工具名的列表",
                "INVALID_PLAN",
            )

        before_result = build_structured_plan(
            normalized_plan,
            available_input_types,
        )
        repair_attempted = False
        repair_applied = False
        repair_result: dict[str, Any] | None = None
        plan_after = list(normalized_plan)
        final_result = before_result

        if before_result.get("valid") is not True and enable_repair:
            repair_attempted = True
            repair_result = repair_tool_plan(
                normalized_plan,
                available_input_types,
            )
            if (
                repair_result.get("status") == "success"
                and repair_result.get("repairable") is True
                and isinstance(repair_result.get("repaired_plan"), list)
            ):
                plan_after = list(repair_result["repaired_plan"])
                repair_applied = bool(repair_result.get("changed"))
                final_result = build_structured_plan(
                    plan_after,
                    available_input_types,
                )
                inserted_positions = _get_inserted_positions(
                    repair_result.get("inserted_steps")
                )
                for step in final_result.get("structured_plan") or []:
                    step["inserted_by_repair"] = (
                        step.get("step_index") in inserted_positions
                    )

        valid = final_result.get("valid") is True
        logger.info(
            "结构化计划准备完成 | valid=%s | repair_enabled=%s | "
            "repair_attempted=%s | repair_applied=%s | before=%s | after=%s",
            valid,
            bool(enable_repair),
            repair_attempted,
            repair_applied,
            normalized_plan,
            plan_after,
        )
        return {
            "status": (
                "error" if final_result.get("status") == "error" else "success"
            ),
            "valid": valid,
            "repair_enabled": bool(enable_repair),
            "repair_attempted": repair_attempted,
            "repair_applied": repair_applied,
            "repair_strategy": (
                repair_result.get("repair_strategy")
                if isinstance(repair_result, dict)
                else "none"
            ),
            "repair_operations": copy.deepcopy(
                repair_result.get("repair_operations") or []
            )
            if isinstance(repair_result, dict)
            else [],
            "plan_before": normalized_plan,
            "plan_after": plan_after,
            "structured_plan": list(final_result.get("structured_plan") or []),
            "issues": list(final_result.get("issues") or []),
            "validation_before": before_result.get("validation"),
            "repair_result": repair_result,
            "final_result": final_result,
        }
    except Exception as exc:
        logger.error("准备结构化计划失败: %s", exc, exc_info=True)
        return _structured_error(
            f"准备结构化计划失败: {exc}",
            "PREPARE_PLAN_EXCEPTION",
        )


def parse_llm_plan_response(
    response_text: Any,
    available_input_types: Any,
    *,
    enable_repair: bool = False,
) -> dict[str, Any]:
    """从旧LLM文本中的Plan Chain生成结构化计划，不再调用第二次LLM。"""
    try:
        if not isinstance(response_text, str) or not response_text.strip():
            return _structured_error(
                "LLM计划响应必须是非空字符串",
                "INVALID_LLM_RESPONSE",
            )
        tool_path = extract_tool_path(response_text)
        if not tool_path:
            return _structured_error(
                "无法从LLM响应中提取合法Plan Chain",
                "PLAN_EXTRACTION_FAILED",
            )
        result = prepare_structured_plan(
            tool_path,
            available_input_types,
            enable_repair=enable_repair,
        )
        result["raw_plan_response"] = response_text
        return result
    except Exception as exc:
        logger.error("解析LLM结构化计划失败: %s", exc, exc_info=True)
        return _structured_error(
            f"解析LLM结构化计划失败: {exc}",
            "PLAN_PARSE_EXCEPTION",
        )


def prepare_structured_plan_from_config(
    tool_path: Any,
    available_input_types: Any,
    *,
    config: Any = None,
) -> dict[str, Any]:
    """根据统一配置开关执行或旁路v0.2结构化计划层。"""
    try:
        cfg = config if config is not None else Config()
        planning_enabled = bool(
            getattr(cfg, "TYPE_PLANNING_ENABLED", False)
        )
        repair_enabled = bool(
            getattr(cfg, "TYPE_REPAIR_ENABLED", False)
        )
        configured_validation = getattr(
            cfg, "TYPE_VALIDATION_ENABLED", None
        )
        validation_enabled = (
            planning_enabled
            if configured_validation is None
            else bool(configured_validation)
        )

        normalized_plan = _normalize_tool_path(tool_path)
        if not normalized_plan:
            # 空计划是可评分的自然规划失败；即使不能进入类型校验，仍必须
            # 记录当前实验条件的开关，避免把方法失败误判为评测管线失败。
            result = _structured_error(
                "工具计划必须是仅含非空工具名的列表",
                "INVALID_PLAN",
            )
            result.update(
                {
                    "enabled": planning_enabled,
                    "bypassed": not planning_enabled,
                    "repair_enabled": repair_enabled,
                    "validation_enabled": validation_enabled,
                    "validation_bypassed": not validation_enabled,
                    "repair_attempted": False,
                    "repair_applied": False,
                    "repair_strategy": "none",
                    "repair_operations": [],
                    "plan_before": [],
                    "plan_after": [],
                }
            )
            return result

        if not planning_enabled:
            logger.info(
                "结构化计划层已旁路 | enabled=False | plan=%s",
                normalized_plan,
            )
            return {
                "status": "success",
                "enabled": False,
                "bypassed": True,
                "valid": None,
                "repair_enabled": repair_enabled,
                "validation_enabled": validation_enabled,
                "validation_bypassed": True,
                "repair_attempted": False,
                "repair_applied": False,
                "repair_strategy": "none",
                "repair_operations": [],
                "plan_before": normalized_plan,
                "plan_after": normalized_plan,
                "structured_plan": [],
                "issues": [],
            }

        if repair_enabled and not validation_enabled:
            return _structured_error(
                "类型修复依赖类型校验，不能在关闭类型门禁时单独开启",
                "INVALID_TYPE_EXPERIMENT_CONFIGURATION",
                original_plan=normalized_plan,
            )
        if not validation_enabled:
            result = build_runtime_binding_plan(
                normalized_plan,
                available_input_types,
            )
            result["enabled"] = True
            result["bypassed"] = False
            return result

        result = prepare_structured_plan(
            normalized_plan,
            available_input_types,
            enable_repair=repair_enabled,
        )
        result["enabled"] = True
        result["bypassed"] = False
        result["validation_enabled"] = True
        result["validation_bypassed"] = False
        return result
    except Exception as exc:
        logger.error("按配置准备结构化计划失败: %s", exc, exc_info=True)
        return _structured_error(
            f"按配置准备结构化计划失败: {exc}",
            "CONFIGURED_PLAN_EXCEPTION",
        )
