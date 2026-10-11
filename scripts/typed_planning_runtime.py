# -*- coding: utf-8 -*-
"""
BioAgent v0.3b 类型约束规划主链路协调层。

修改记录：
- 2026-09-05：空或不可解析计划的拒绝响应改为复用类型化运行时的
  结构化错误；对无初始对象请求补充输入缺失与能力缺口提示。
- 2026-08-27：安全轨迹新增类型门禁状态，支持E1运行时基线与E2校验
  条件的正交比较。
- 2026-09-04：从原始用户问题构造请求级工具参数上下文，并在类型化执行
  全程保留显式ANM参数；不把数值参数注册成数据Artifact。
- 2026-08-15：安全轨迹新增修复策略和固定字段操作证明，支持区分桥接插入与
  注册表授权的重复无效步骤剪枝。
- 2026-08-12：把非法氨基酸字符、标准错误码和修改建议转换为可操作反馈，
  避免安全拒绝退化成泛化的“没有初始Artifact”。
- 2026-08-14：为多记录FASTA的单record_id与record_id pair缺失问题增加
  精确、可执行反馈，替代旧版“请拆成单记录文件”的限制。
- 2026-08-09：保留类型检查器的完整结构化问题，并把缺失PDB链标识、
  多记录FASTA等机器诊断转换为可操作的用户反馈。
- 2026-08-08：初始Artifact构建失败时仍记录当前repair开关，避免B2拒绝
  响应因安全轨迹条件不一致而被实验运行器丢弃。
- 2026-07-26：串联初始Artifact构建、结构化计划检查/保守修复和v0.3a
  执行器，为run_SciToolAgent提供单一、可测试的类型化执行入口。

设计边界：
1. 本模块不负责KG检索和LLM规划，只消费已经提取的旧Plan Chain；
2. TYPE_PLANNING_ENABLED关闭时返回旁路状态，不执行任何工具；
3. 计划或工具失败时返回结构化失败，不调用LLM粉饰错误；
4. safe_trace只包含类型、对象ID和耗时，不包含真实序列或完整文件路径。
"""

from __future__ import annotations

import copy
from typing import Any

from app.core.config import Config
from app.core.logger import get_logger
from app.tools.tool_parameters import build_tool_parameter_context
from scripts.initial_artifacts import build_initial_artifacts
from scripts.structured_plan import prepare_structured_plan_from_config
from scripts.typed_executor import execute_structured_plan
from scripts.type_checker import (
    FASTA_RECORD_IDENTIFIER,
    FASTA_RECORD_IDENTIFIER_PAIR,
    PDB_CHAIN_IDENTIFIER,
)


logger = get_logger("TypedPlanningRuntime", "typed_planning_runtime.log")


def _requested_unavailable_capabilities(question: Any) -> list[str]:
    """从问题中提取少量明示的非当前能力提示，仅用于拒绝解释。

    该函数不参与工具选择或执行决策，避免把关键词规则变成
    第二套规划器。
    """
    try:
        if not isinstance(question, str):
            return []
        normalized = question.casefold()
        hints: list[str] = []
        if any(
            marker in normalized
            for marker in (
                "核酸",
                "dna",
                "rna",
                "反向互补",
                "reverse complement",
                "orf",
            )
        ):
            hints.append("核酸反向互补或ORF识别")
        if any(
            marker in normalized
            for marker in (
                "blast",
                "联网",
                "在线数据库",
                "数据库检索",
                "database search",
            )
        ):
            hints.append("在线BLAST或联网数据库检索")
        return hints
    except Exception as exc:
        logger.error("提取请求能力提示失败: %s", exc, exc_info=True)
        return []


def _explain_runtime_issue(
    issue: Any,
    fallback_message: Any,
    *,
    question: Any = None,
) -> dict[str, Any]:
    """把结构化机器问题转换为确定性、可操作且不泄露原始数据的反馈。"""
    try:
        normalized_issue = copy.deepcopy(issue) if isinstance(issue, dict) else {}
        code = str(normalized_issue.get("code") or "UNKNOWN_RUNTIME_ISSUE")
        fallback = str(fallback_message or "类型约束规划或执行失败")

        if code == "NO_INITIAL_ARTIFACT":
            capability_hints = _requested_unavailable_capabilities(question)
            capability_text = (
                "当前注册表未提供与请求相匹配的"
                f"{'、'.join(capability_hints)}工具。"
                if capability_hints
                else "当前注册表中没有可在缺少输入时完成该请求的工具。"
            )
            return {
                "message": (
                    "未识别到受支持的初始生物数据对象。"
                    f"{capability_text}"
                    "请提供与已注册工具输入类型匹配的数据，"
                    "或先注册所需工具；系统不会编造输入、工具调用或结果。"
                ),
                "detail_code": code,
                "required_user_inputs": [
                    {
                        "name": "biological_input",
                        "type": "registered tool-compatible biological input",
                        "examples": [],
                    }
                ],
            }

        if code == "MISSING_INPUT_TYPE":
            missing_types = [
                str(value)
                for value in normalized_issue.get("missing_types") or []
                if str(value).strip()
            ]
            if PDB_CHAIN_IDENTIFIER in missing_types:
                return {
                    "message": (
                        "该工具计划需要从PDB中选择目标链，但问题中未提供"
                        "chain_id。请指定链编号，例如A或B；系统不会自动猜测。"
                    ),
                    "detail_code": "MISSING_PDB_CHAIN_ID",
                    "required_user_inputs": [
                        {
                            "name": "chain_id",
                            "type": PDB_CHAIN_IDENTIFIER,
                            "examples": ["A", "B"],
                        }
                    ],
                }
            if FASTA_RECORD_IDENTIFIER in missing_types:
                return {
                    "message": (
                        "该计划需要从多记录FASTA中选择一条记录，但问题中未提供"
                        "精确record_id。请使用record_id=<ID>明确指定；系统不会猜测。"
                    ),
                    "detail_code": "MISSING_FASTA_RECORD_IDENTIFIER",
                    "required_user_inputs": [
                        {
                            "name": "record_id",
                            "type": FASTA_RECORD_IDENTIFIER,
                            "examples": ["record_id=protein_a"],
                        }
                    ],
                }
            if FASTA_RECORD_IDENTIFIER_PAIR in missing_types:
                return {
                    "message": (
                        "该计划需要比对多记录FASTA中的两条记录，但问题中未提供"
                        "两个精确record_id。请明确给出record_ids=A,B。"
                    ),
                    "detail_code": "MISSING_FASTA_RECORD_IDENTIFIER_PAIR",
                    "required_user_inputs": [
                        {
                            "name": "record_ids",
                            "type": FASTA_RECORD_IDENTIFIER_PAIR,
                            "examples": ["record_ids=protein_a,protein_b"],
                        }
                    ],
                }
            tool_name = str(normalized_issue.get("tool_name") or "目标工具")
            missing_label = "、".join(missing_types) or "必要输入"
            return {
                "message": f"{tool_name}缺少必要输入类型：{missing_label}。",
                "detail_code": "MISSING_REQUIRED_INPUT",
                "required_user_inputs": [],
            }

        if code == "FASTA_RECORD_COUNT_MISMATCH":
            record_count = normalized_issue.get("record_count")
            count_text = (
                f"检测到{record_count}条记录"
                if isinstance(record_count, int) and record_count >= 0
                else "检测到多条或空记录"
            )
            return {
                "message": (
                    f"该Protein FASTA{count_text}，当前分析要求单条蛋白质记录。"
                    "请明确目标record_id，并只保留该记录后重新上传；系统不会自动选择。"
                ),
                "detail_code": "AMBIGUOUS_FASTA_RECORD",
                "required_user_inputs": [
                    {
                        "name": "record_id",
                        "type": "protein FASTA record identifier",
                        "examples": [],
                    }
                ],
            }

        if code == "INVALID_AMINO_ACID_SEQUENCE_CONTENT":
            invalid_characters = [
                str(value)
                for value in normalized_issue.get("invalid_characters") or []
                if str(value)
            ]
            invalid_text = "、".join(invalid_characters) or "未知字符"
            suggestion = str(
                normalized_issue.get("suggestion")
                or (
                    "请仅使用20种标准氨基酸单字母代码"
                    "ACDEFGHIKLMNPQRSTVWY，并删除或替换非法字符。"
                )
            )
            return {
                "message": (
                    f"氨基酸序列包含非法或非标准字符：{invalid_text}。"
                    f"{suggestion}"
                ),
                "detail_code": code,
                "required_user_inputs": [
                    {
                        "name": "sequence",
                        "type": "amino acid sequence",
                        "allowed_characters": "ACDEFGHIKLMNPQRSTVWY",
                    }
                ],
            }

        return {
            "message": str(normalized_issue.get("message") or fallback),
            "detail_code": code,
            "required_user_inputs": [],
        }
    except Exception as exc:
        logger.error("解释类型化运行问题失败: %s", exc, exc_info=True)
        return {
            "message": str(fallback_message or "类型约束规划或执行失败"),
            "detail_code": "ISSUE_EXPLANATION_FAILURE",
            "required_user_inputs": [],
        }


def build_actionable_empty_plan_rejection(typed_result: Any) -> dict[str, Any]:
    """将空计划与类型化诊断合并为统一、可执行的拒绝结构。"""
    try:
        generic_message = "模型未生成可解析且非空的注册工具计划，已拒绝执行"
        diagnostic = typed_result if isinstance(typed_result, dict) else {}
        has_typed_error = diagnostic.get("status") == "error"
        message = str(
            diagnostic.get("message") if has_typed_error else generic_message
        ).strip() or generic_message
        detail_code = (
            str(diagnostic.get("detail_code") or "").strip() or None
            if has_typed_error
            else None
        )
        issues = (
            copy.deepcopy(diagnostic.get("issues") or [])
            if has_typed_error
            else []
        )
        required_user_inputs = (
            copy.deepcopy(diagnostic.get("required_user_inputs") or [])
            if has_typed_error
            else []
        )
        payload = {
            "status": "error",
            "message": message,
            "error_type": "empty_or_unparseable_plan",
            "detail_code": detail_code,
            "underlying_error_type": (
                diagnostic.get("error_type") if has_typed_error else None
            ),
            "issues": issues,
            "required_user_inputs": required_user_inputs,
        }
        return {
            "status": "success",
            "payload": payload,
            "final_message": f"问题未完成：{message}系统未调用任何工具。",
        }
    except Exception as exc:
        logger.error("构造空计划可行动拒绝失败: %s", exc, exc_info=True)
        fallback = "模型未生成可解析且非空的注册工具计划，已拒绝执行"
        return {
            "status": "error",
            "payload": {
                "status": "error",
                "message": fallback,
                "error_type": "empty_or_unparseable_plan",
                "detail_code": "EMPTY_PLAN_REJECTION_FORMAT_FAILURE",
                "issues": [],
                "required_user_inputs": [],
            },
            "final_message": f"问题未完成：{fallback}系统未调用任何工具。",
        }


def _runtime_error(
    message: str,
    code: str,
    *,
    error_type: str,
    initial_artifacts: dict[str, Any] | None = None,
    planning: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
    issues: list[dict[str, Any]] | None = None,
    detail_code: str | None = None,
    required_user_inputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构造协调层统一失败结构。"""
    try:
        return {
            "status": "error",
            "message": message,
            "completed": False,
            "error_type": error_type,
            "detail_code": detail_code or code,
            "issues": copy.deepcopy(
                issues if isinstance(issues, list) and issues else [
                    {"code": code, "message": message}
                ]
            ),
            "required_user_inputs": copy.deepcopy(required_user_inputs or []),
            "initial_artifacts": copy.deepcopy(initial_artifacts or {}),
            "planning": copy.deepcopy(planning or {}),
            "execution": copy.deepcopy(execution or {}),
            "safe_trace": {},
        }
    except Exception as exc:
        logger.error("构造类型规划协调错误失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": str(exc),
            "completed": False,
            "error_type": "runtime_exception",
            "issues": [],
        }


def build_safe_typed_trace(result: Any) -> dict[str, Any]:
    """从协调结果提取可供前端和论文统计使用的脱敏轨迹。"""
    try:
        if not isinstance(result, dict):
            return {}
        initial = result.get("initial_artifacts") or {}
        planning = result.get("planning") or {}
        execution = result.get("execution") or {}
        return {
            "enabled": bool(result.get("enabled", True)),
            "status": result.get("status"),
            "completed": bool(result.get("completed", False)),
            "error_type": result.get("error_type"),
            "detail_code": result.get("detail_code"),
            "required_user_inputs": copy.deepcopy(
                result.get("required_user_inputs") or []
            ),
            "initial_artifacts": copy.deepcopy(
                initial.get("safe_summary") or []
            ),
            "initial_warnings": copy.deepcopy(
                initial.get("warnings") or []
            ),
            "plan_before": copy.deepcopy(planning.get("plan_before") or []),
            "plan_after": copy.deepcopy(planning.get("plan_after") or []),
            "structured_plan": copy.deepcopy(
                planning.get("structured_plan") or []
            ),
            "repair_enabled": bool(planning.get("repair_enabled", False)),
            "validation_enabled": bool(
                planning.get("validation_enabled", True)
            ),
            "validation_bypassed": bool(
                planning.get("validation_bypassed", False)
            ),
            "repair_attempted": bool(
                planning.get("repair_attempted", False)
            ),
            "repair_applied": bool(planning.get("repair_applied", False)),
            "repair_strategy": planning.get("repair_strategy") or "none",
            "repair_operations": copy.deepcopy(
                planning.get("repair_operations") or []
            ),
            "planning_issues": copy.deepcopy(planning.get("issues") or []),
            "executed_steps": copy.deepcopy(
                execution.get("executed_steps") or []
            ),
            "artifact_pool": copy.deepcopy(
                execution.get("artifact_pool") or {}
            ),
            "total_duration_ms": execution.get("total_duration_ms"),
        }
    except Exception as exc:
        logger.error("构建类型规划安全轨迹失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"构建类型规划安全轨迹失败: {exc}",
        }


async def run_typed_planning(
    tool_path: Any,
    question: Any,
    file_path_list: Any,
    *,
    config: Any = None,
    runner: Any = None,
    tool_parameters: Any = None,
) -> dict[str, Any]:
    """执行v0.3b“初始对象—结构化计划—对象池执行”闭环。"""
    try:
        cfg = config if config is not None else Config()
        if not bool(getattr(cfg, "TYPE_PLANNING_ENABLED", False)):
            result = {
                "status": "success",
                "enabled": False,
                "bypassed": True,
                "completed": False,
                "error_type": None,
                "issues": [],
                "initial_artifacts": {},
                "planning": {},
                "execution": {},
            }
            result["safe_trace"] = build_safe_typed_trace(result)
            return result

        initial_result = build_initial_artifacts(question, file_path_list)
        if initial_result.get("status") != "success":
            issue = (
                initial_result.get("issues")
                or [{"code": "INITIAL_ARTIFACT_FAILURE"}]
            )[0]
            configured_validation = getattr(
                cfg, "TYPE_VALIDATION_ENABLED", None
            )
            validation_enabled = (
                True
                if configured_validation is None
                else bool(configured_validation)
            )
            preplanning_state = {
                "plan_before": copy.deepcopy(
                    tool_path if isinstance(tool_path, list) else []
                ),
                "plan_after": copy.deepcopy(
                    tool_path if isinstance(tool_path, list) else []
                ),
                "repair_enabled": bool(
                    getattr(cfg, "TYPE_REPAIR_ENABLED", False)
                ),
                "validation_enabled": validation_enabled,
                "validation_bypassed": not validation_enabled,
                "repair_attempted": False,
                "repair_applied": False,
                "issues": [],
            }
            explanation = _explain_runtime_issue(
                issue,
                initial_result.get("message", "初始Artifact构建失败"),
                question=question,
            )
            result = _runtime_error(
                explanation["message"],
                issue.get("code", "INITIAL_ARTIFACT_FAILURE"),
                error_type="initial_artifact_failure",
                initial_artifacts=initial_result,
                planning=preplanning_state,
                issues=list(initial_result.get("issues") or []),
                detail_code=explanation.get("detail_code"),
                required_user_inputs=explanation.get("required_user_inputs"),
            )
            result["enabled"] = True
            result["bypassed"] = False
            result["safe_trace"] = build_safe_typed_trace(result)
            return result

        planning_result = prepare_structured_plan_from_config(
            tool_path,
            initial_result.get("available_input_types") or [],
            config=cfg,
        )
        if (
            planning_result.get("status") == "error"
            or planning_result.get("valid") is not True
        ):
            issue = (
                planning_result.get("issues")
                or [{"code": "TYPE_INVALID_PLAN"}]
            )[0]
            explanation = _explain_runtime_issue(
                issue,
                planning_result.get(
                    "message",
                    "工具计划未通过类型约束检查",
                ),
            )
            result = _runtime_error(
                explanation["message"],
                issue.get("code", "TYPE_INVALID_PLAN"),
                error_type="plan_failure",
                initial_artifacts=initial_result,
                planning=planning_result,
                issues=list(planning_result.get("issues") or []),
                detail_code=explanation.get("detail_code"),
                required_user_inputs=explanation.get("required_user_inputs"),
            )
            result["enabled"] = True
            result["bypassed"] = False
            result["safe_trace"] = build_safe_typed_trace(result)
            return result

        execution_result = await execute_structured_plan(
            planning_result.get("structured_plan") or [],
            initial_result.get("artifacts") or {},
            runner=runner,
            tool_parameters=(
                copy.deepcopy(tool_parameters)
                if tool_parameters is not None
                else build_tool_parameter_context(question)
            ),
        )
        if execution_result.get("status") != "success":
            issue = (
                execution_result.get("issues")
                or [{"code": "TYPED_EXECUTION_FAILURE"}]
            )[0]
            explanation = _explain_runtime_issue(
                issue,
                execution_result.get("message", "类型化工具执行失败"),
            )
            result = _runtime_error(
                explanation["message"],
                issue.get("code", "TYPED_EXECUTION_FAILURE"),
                error_type=execution_result.get(
                    "error_type",
                    "execution_failure",
                ),
                initial_artifacts=initial_result,
                planning=planning_result,
                execution=execution_result,
                issues=list(execution_result.get("issues") or []),
                detail_code=explanation.get("detail_code"),
                required_user_inputs=explanation.get("required_user_inputs"),
            )
            result["enabled"] = True
            result["bypassed"] = False
            result["safe_trace"] = build_safe_typed_trace(result)
            return result

        result = {
            "status": "success",
            "enabled": True,
            "bypassed": False,
            "completed": True,
            "error_type": None,
            "issues": [],
            "initial_artifacts": initial_result,
            "planning": planning_result,
            "execution": execution_result,
        }
        result["safe_trace"] = build_safe_typed_trace(result)
        logger.info(
            "v0.3b类型规划闭环完成 | before=%s | after=%s | steps=%s",
            planning_result.get("plan_before"),
            planning_result.get("plan_after"),
            len(execution_result.get("executed_steps") or []),
        )
        return result
    except Exception as exc:
        logger.error("v0.3b类型规划闭环异常: %s", exc, exc_info=True)
        result = _runtime_error(
            f"v0.3b类型规划闭环异常: {exc}",
            "TYPED_PLANNING_RUNTIME_EXCEPTION",
            error_type="runtime_exception",
        )
        result["enabled"] = True
        result["bypassed"] = False
        result["safe_trace"] = build_safe_typed_trace(result)
        return result
