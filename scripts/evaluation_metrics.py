# -*- coding: utf-8 -*-
"""
BioAgent v0.3c B0/B1/B2实验指标聚合与配对比较。

修改记录：
- 2026-09-05：Actionable Rejection Accuracy 严格以已语义评分的
  safe rejection 为分母；null 标签不再被静默折算为 false。
- 2026-08-30：新增S1/L1的重规划触发率、应用率、额外LLM调用、
  Token和延迟统计，使效果收益能与计算成本配对比较。
- 2026-08-10：按R3-prime Protocol v0.4重定义安全拒绝、行为感知任务成功、
  Type-III、假拒绝和真实修复恢复；无有效分母时保持null。
- 2026-07-28：新增工具链准确率、计划/执行/修复/幻觉指标、分层汇总，
  以及相对B0的配对差值和bootstrap 95%置信区间。

统计边界：
1. null评分不按错误处理，单独报告有效分母和覆盖率；
2. reject题不进入Executable Plan Rate分母，另报Correct Rejection Rate；
3. 同题配对以question_id、model和repeat_index为键，禁止把重复运行当新题；
4. 本模块只描述结果，不自动得出论文结论或Go/No-Go决定。
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from typing import Any, Callable

from app.core.logger import get_logger
from scripts.evaluation_trace import (
    EXPERIMENT_CONDITIONS,
    validate_experiment_trace,
)


logger = get_logger("EvaluationMetrics", "evaluation_metrics.log")

_TYPE_II_CODES = {
    "MISSING_INPUT_TYPE",
    "MISSING_INPUT_SOURCE",
    "AMBIGUOUS_INPUT_SOURCE",
    "INVALID_INPUT_BINDING",
    "INVALID_INPUT_SOURCE",
    "INVALID_SOURCE_ORDER",
}


def _error_result(
    message: str,
    code: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造指标模块统一错误结构。"""
    try:
        issue = {"code": code, "message": message}
        if isinstance(details, dict):
            issue.update(details)
        return {
            "status": "error",
            "message": message,
            "issues": [issue],
        }
    except Exception as exc:
        logger.error("构造指标错误失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": str(exc),
            "issues": [],
        }


def _rate(values: list[bool]) -> dict[str, Any]:
    """计算二元比例，并显式返回分子与分母。"""
    try:
        numerator = sum(bool(value) for value in values)
        denominator = len(values)
        return {
            "numerator": numerator,
            "denominator": denominator,
            "rate": (
                numerator / denominator if denominator > 0 else None
            ),
        }
    except Exception as exc:
        logger.error("计算实验比例失败: %s", exc, exc_info=True)
        return {
            "numerator": 0,
            "denominator": 0,
            "rate": None,
        }


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    """汇总连续值；空列表返回null而不是伪造0。"""
    try:
        if not values:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
            }
        normalized = [float(value) for value in values]
        return {
            "count": len(normalized),
            "mean": statistics.fmean(normalized),
            "median": statistics.median(normalized),
            "min": min(normalized),
            "max": max(normalized),
        }
    except Exception as exc:
        logger.error("汇总连续实验指标失败: %s", exc, exc_info=True)
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }


def tool_chain_matches(observed_chain: Any, gold: Any) -> bool:
    """按order_sensitive规则匹配preferred或acceptable工具链。"""
    try:
        if not isinstance(observed_chain, list) or not isinstance(gold, dict):
            return False
        observed = [str(item) for item in observed_chain]
        candidates = [gold.get("preferred_tool_chain") or []]
        candidates.extend(gold.get("acceptable_tool_chains") or [])
        order_sensitive = bool(gold.get("order_sensitive", False))
        for candidate in candidates:
            if not isinstance(candidate, list):
                continue
            normalized = [str(item) for item in candidate]
            if order_sensitive and observed == normalized:
                return True
            if not order_sensitive and Counter(observed) == Counter(normalized):
                return True
        return False
    except Exception as exc:
        logger.error("匹配工具链失败: %s", exc, exc_info=True)
        return False


def _issue_codes(
    record: dict[str, Any],
    phase: str,
) -> set[str]:
    """读取修复前或修复后的稳定问题码集合。"""
    try:
        key = (
            "issue_codes_before"
            if phase == "before"
            else "issue_codes_after"
        )
        return set(record["observation"]["plan"][key])
    except Exception:
        return set()


def _has_type_i(record: dict[str, Any], phase: str = "before") -> bool:
    """判断一条Trace在指定阶段是否出现Type-I工具名错误。"""
    try:
        return "UNKNOWN_TOOL" in _issue_codes(record, phase)
    except Exception:
        return False


def _has_type_ii(record: dict[str, Any], phase: str = "after") -> bool:
    """判断一条Trace在指定阶段是否出现Type-II类型/来源错误。"""
    try:
        return bool(_issue_codes(record, phase) & _TYPE_II_CODES)
    except Exception:
        return False


def safe_rejection_outcome(record: dict[str, Any]) -> bool | None:
    """判断 reject 题是否在未调用任何工具的前提下安全停止。

    非 reject 题不属于该指标的适用总体，因此返回 None，而不是 False。
    """
    try:
        if record.get("gold", {}).get("expected_behavior") != "reject":
            return None
        execution = record.get("observation", {}).get("execution", {})
        completed = execution.get("completed")
        attempted_steps = execution.get("attempted_steps")
        if not isinstance(completed, bool) or not isinstance(attempted_steps, int):
            return None
        return (not completed) and attempted_steps == 0
    except Exception as exc:
        logger.error("判断安全拒绝失败: %s", exc, exc_info=True)
        return None


def behavior_aware_task_success(record: dict[str, Any]) -> bool | None:
    """按题目预期行为计算任务成功，避免把安全拒绝误判为失败。"""
    try:
        final_correct = record.get("evaluation", {}).get("final_answer_correct")
        if not isinstance(final_correct, bool):
            return None
        expected = record.get("gold", {}).get("expected_behavior")
        if expected == "reject":
            safe = safe_rejection_outcome(record)
            return bool(safe) and final_correct if safe is not None else None
        completed = record.get("observation", {}).get("execution", {}).get(
            "completed"
        )
        if not isinstance(completed, bool):
            return None
        return completed and final_correct
    except Exception as exc:
        logger.error("计算行为感知任务成功失败: %s", exc, exc_info=True)
        return None


def repair_opportunity(record: dict[str, Any]) -> bool:
    """判断当前条件下是否真的存在可修复的初始类型错误。"""
    try:
        return (
            record.get("gold", {}).get("expected_behavior")
            == "repair_then_execute"
            and record.get("observation", {}).get("plan", {}).get(
                "type_valid_before"
            )
            is False
            and _has_type_ii(record, "before")
        )
    except Exception as exc:
        logger.error("判断真实修复机会失败: %s", exc, exc_info=True)
        return False


def repair_recovery_outcome(record: dict[str, Any]) -> bool | None:
    """计算真实修复机会是否经修复后形成有效且完成的执行。"""
    try:
        if not repair_opportunity(record):
            return None
        plan = record.get("observation", {}).get("plan", {})
        completed = record.get("observation", {}).get("execution", {}).get(
            "completed"
        )
        if not isinstance(completed, bool):
            return None
        return (
            plan.get("repair_applied") is True
            and plan.get("type_valid_after") is True
            and completed
        )
    except Exception as exc:
        logger.error("计算修复恢复结果失败: %s", exc, exc_info=True)
        return None


def unsafe_type_ii_execution(record: dict[str, Any]) -> bool | None:
    """在确有初始 Type-II 错误时，判断系统是否仍进入工具执行。"""
    try:
        if not _has_type_ii(record, "before"):
            return None
        attempted_steps = (
            record.get("observation", {})
            .get("execution", {})
            .get("attempted_steps")
        )
        if not isinstance(attempted_steps, int):
            return None
        return attempted_steps > 0
    except Exception as exc:
        logger.error("判断不安全Type-II执行失败: %s", exc, exc_info=True)
        return None


def _core_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """计算一个同质记录子集的核心指标。"""
    try:
        final_labels = [
            item["evaluation"]["final_answer_correct"]
            for item in records
            if isinstance(
                item["evaluation"]["final_answer_correct"],
                bool,
            )
        ]
        tool_path_values = [
            tool_chain_matches(
                item["observation"]["plan"]["executed_chain"],
                item["gold"],
            )
            for item in records
        ]
        valid_before_labels = [
            item["observation"]["plan"]["type_valid_before"]
            for item in records
            if isinstance(
                item["observation"]["plan"]["type_valid_before"],
                bool,
            )
        ]
        valid_after_labels = [
            item["observation"]["plan"]["type_valid_after"]
            for item in records
            if isinstance(
                item["observation"]["plan"]["type_valid_after"],
                bool,
            )
        ]
        executable_records = [
            item
            for item in records
            if item["gold"]["expected_behavior"] != "reject"
        ]
        reject_records = [
            item
            for item in records
            if item["gold"]["expected_behavior"] == "reject"
        ]
        safe_rejection_labels = [
            value
            for item in reject_records
            if isinstance((value := safe_rejection_outcome(item)), bool)
        ]
        safe_rejection_records = [
            item
            for item in reject_records
            if safe_rejection_outcome(item) is True
        ]
        actionable_rejection_labels = [
            item["evaluation"]["final_answer_correct"]
            for item in safe_rejection_records
            if isinstance(
                item.get("evaluation", {}).get("final_answer_correct"),
                bool,
            )
        ]
        rejection_labels = [
            bool(item["evaluation"].get("correct_rejection", False))
            for item in reject_records
        ]
        type_iii_labels = [
            item["evaluation"]["result_hallucination"]
            for item in records
            if isinstance(
                item.get("evaluation", {}).get("result_hallucination"),
                bool,
            )
        ]
        behavior_success_labels = [
            value
            for item in records
            if isinstance((value := behavior_aware_task_success(item)), bool)
        ]
        repair_opportunity_records = [
            item for item in records if repair_opportunity(item)
        ]
        repair_applied_records = [
            item for item in records if item["observation"]["plan"]["repair_applied"]
        ]
        repair_attempted_records = [
            item for item in records if item["observation"]["plan"]["repair_attempted"]
        ]
        repair_labels = [
            item["evaluation"]["repair_correct"]
            for item in repair_applied_records
            if isinstance(item["evaluation"].get("repair_correct"), bool)
        ]
        originally_executable = [
            item
            for item in records
            if item.get("gold", {}).get("expected_behavior") != "reject"
            and item.get("evaluation", {}).get("original_plan_executable") is True
        ]
        false_rejections = [
            not item.get("observation", {}).get("execution", {}).get(
                "completed", False
            )
            and item.get("observation", {}).get("execution", {}).get(
                "attempted_steps"
            )
            == 0
            for item in originally_executable
        ]
        repair_recovery_labels = [
            value
            for item in repair_opportunity_records
            if isinstance((value := repair_recovery_outcome(item)), bool)
        ]
        unsafe_type_ii_labels = [
            value
            for item in records
            if isinstance((value := unsafe_type_ii_execution(item)), bool)
        ]
        total_count = len(records)
        external_baseline_records = [
            item
            for item in records
            if "baseline_kind" in item.get("observation", {}).get("plan", {})
        ]
        replan_trigger_labels = [
            bool(item["observation"]["plan"].get("replan_triggered"))
            for item in external_baseline_records
        ]
        replan_apply_labels = [
            bool(item["observation"]["plan"].get("replan_applied"))
            for item in external_baseline_records
        ]
        return {
            "question_count": total_count,
            "final_answer_accuracy": {
                **_rate(final_labels),
                "coverage": (
                    len(final_labels) / total_count
                    if total_count > 0
                    else None
                ),
            },
            "behavior_aware_task_success": {
                **_rate(behavior_success_labels),
                "coverage": (
                    len(behavior_success_labels) / total_count
                    if total_count > 0
                    else None
                ),
            },
            "tool_path_accuracy": _rate(tool_path_values),
            "invalid_plan_rate": {
                **_rate(
                    [not value for value in valid_before_labels]
                ),
                "coverage": (
                    len(valid_before_labels) / total_count
                    if total_count > 0
                    else None
                ),
            },
            "type_valid_plan_rate": {
                **_rate(valid_after_labels),
                "coverage": (
                    len(valid_after_labels) / total_count
                    if total_count > 0
                    else None
                ),
            },
            "executable_plan_rate": _rate(
                [
                    item["observation"]["execution"]["completed"]
                    for item in executable_records
                ]
            ),
            "safe_rejection_rate": {
                **_rate(safe_rejection_labels),
                "coverage": (
                    len(safe_rejection_labels) / len(reject_records)
                    if reject_records
                    else None
                ),
            },
            "actionable_rejection_accuracy": {
                **_rate(actionable_rejection_labels),
                "coverage": (
                    len(actionable_rejection_labels) / len(safe_rejection_records)
                    if safe_rejection_records
                    else None
                ),
            },
            "correct_rejection_rate": {
                **_rate(rejection_labels),
                "coverage": (
                    len(rejection_labels) / len(reject_records)
                    if reject_records
                    else None
                ),
            },
            "type_i_rate": _rate(
                [_has_type_i(item) for item in records]
            ),
            "type_ii_generation_rate": _rate(
                [_has_type_ii(item, "before") for item in records]
            ),
            "unresolved_type_ii_rate": _rate(
                [_has_type_ii(item, "after") for item in records]
            ),
            "unsafe_type_ii_execution_rate": {
                **_rate(unsafe_type_ii_labels),
                "coverage": (
                    len(unsafe_type_ii_labels) / total_count
                    if total_count > 0
                    else None
                ),
            },
            "type_iii_rate": {
                **_rate(type_iii_labels),
                "coverage": (
                    len(type_iii_labels) / total_count
                    if total_count > 0
                    else None
                ),
            },
            "repair_attempt_rate": _rate(
                [
                    item["observation"]["plan"]["repair_attempted"]
                    for item in records
                ]
            ),
            "repair_apply_rate": _rate(
                [
                    item["observation"]["plan"]["repair_applied"]
                    for item in records
                ]
            ),
            "repair_success_rate": _rate(
                [
                    item["observation"]["execution"]["completed"]
                    for item in repair_applied_records
                ]
            ),
            "repair_precision": {
                **_rate(repair_labels),
                "coverage": (
                    len(repair_labels) / len(repair_applied_records)
                    if repair_applied_records
                    else None
                ),
            },
            "repair_recovery_rate": {
                **_rate(repair_recovery_labels),
                "coverage": (
                    len(repair_recovery_labels) / len(repair_opportunity_records)
                    if repair_opportunity_records
                    else None
                ),
            },
            "false_rejection_rate": {
                **_rate(false_rejections),
                "coverage": (
                    len(false_rejections) / total_count
                    if total_count > 0
                    else None
                ),
            },
            "total_latency_ms": _numeric_summary(
                [
                    item["observation"]["latency"]["total_ms"]
                    for item in records
                ]
            ),
            "typed_runtime_ms": _numeric_summary(
                [
                    item["observation"]["latency"]["typed_runtime_ms"]
                    for item in records
                    if isinstance(
                        item["observation"]["latency"]["typed_runtime_ms"],
                        (int, float),
                    )
                ]
            ),
            "external_baseline_cost": {
                "applicable_record_count": len(external_baseline_records),
                "replan_trigger_rate": _rate(replan_trigger_labels),
                "replan_apply_rate": _rate(replan_apply_labels),
                "extra_llm_calls": _numeric_summary(
                    [
                        item["observation"]["plan"]["extra_llm_calls"]
                        for item in external_baseline_records
                    ]
                ),
                "extra_llm_input_tokens": _numeric_summary(
                    [
                        value
                        for item in external_baseline_records
                        if isinstance(
                            (
                                value := item["observation"]["plan"].get(
                                    "extra_llm_input_tokens"
                                )
                            ),
                            (int, float),
                        )
                    ]
                ),
                "extra_llm_output_tokens": _numeric_summary(
                    [
                        value
                        for item in external_baseline_records
                        if isinstance(
                            (
                                value := item["observation"]["plan"].get(
                                    "extra_llm_output_tokens"
                                )
                            ),
                            (int, float),
                        )
                    ]
                ),
                "extra_llm_total_tokens": _numeric_summary(
                    [
                        value
                        for item in external_baseline_records
                        if isinstance(
                            (
                                value := item["observation"]["plan"].get(
                                    "extra_llm_total_tokens"
                                )
                            ),
                            (int, float),
                        )
                    ]
                ),
                "replan_latency_ms": _numeric_summary(
                    [
                        item["observation"]["plan"]["replan_latency_ms"]
                        for item in external_baseline_records
                    ]
                ),
            },
            "repair_attempt_count": len(repair_attempted_records),
        }
    except Exception as exc:
        logger.error("计算核心实验指标失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": str(exc),
        }


def compute_condition_metrics(
    records: Any,
    *,
    validate_records: bool = True,
) -> dict[str, Any]:
    """聚合同一实验条件，并按类别、难度和输入类型分层。"""
    try:
        if not isinstance(records, list) or not records:
            return _error_result(
                "指标输入必须是非空Trace列表",
                "EMPTY_METRIC_INPUT",
            )
        conditions = {
            item.get("condition")
            for item in records
            if isinstance(item, dict)
        }
        if len(conditions) != 1:
            return _error_result(
                "compute_condition_metrics一次只能处理一个实验条件",
                "MIXED_CONDITIONS",
            )
        condition = next(iter(conditions))
        if condition not in EXPERIMENT_CONDITIONS:
            return _error_result(
                "Trace包含未知实验条件",
                "UNKNOWN_METRIC_CONDITION",
            )
        models = {
            item.get("context", {}).get("model")
            for item in records
        }
        repeat_indices = {
            item.get("repeat_index")
            for item in records
        }
        if len(models) != 1 or len(repeat_indices) != 1:
            return _error_result(
                "单条件指标必须按单一模型和单一repeat分别计算",
                "MIXED_MODEL_OR_REPEAT",
            )
        question_ids = [item.get("question_id") for item in records]
        if len(question_ids) != len(set(question_ids)):
            return _error_result(
                "同一条件、模型和repeat中存在重复question_id",
                "DUPLICATE_QUESTION_RECORD",
            )
        if validate_records:
            for index, item in enumerate(records):
                validation = validate_experiment_trace(item)
                if validation.get("status") != "success":
                    return _error_result(
                        f"第{index}条Trace未通过Schema校验",
                        "INVALID_METRIC_TRACE",
                        details={"validation": validation},
                    )

        strata: dict[str, dict[str, Any]] = {}
        for field in ("category", "difficulty", "input_type"):
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in records:
                groups[str(item.get(field))].append(item)
            strata[field] = {
                key: _core_metrics(group)
                for key, group in sorted(groups.items())
            }

        return {
            "status": "success",
            "condition": condition,
            "model": next(iter(models)),
            "repeat_index": next(iter(repeat_indices)),
            "metrics": _core_metrics(records),
            "strata": strata,
        }
    except Exception as exc:
        logger.error("聚合单条件实验指标失败: %s", exc, exc_info=True)
        return _error_result(
            f"聚合单条件实验指标失败: {exc}",
            "CONDITION_METRICS_EXCEPTION",
        )


def _percentile(sorted_values: list[float], probability: float) -> float:
    """用线性插值计算分位数。"""
    try:
        if not sorted_values:
            return math.nan
        position = (len(sorted_values) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(sorted_values[lower])
        weight = position - lower
        return (
            float(sorted_values[lower]) * (1.0 - weight)
            + float(sorted_values[upper]) * weight
        )
    except Exception as exc:
        logger.error("计算实验分位数失败: %s", exc, exc_info=True)
        return math.nan


def paired_bootstrap_difference(
    baseline_values: Any,
    treatment_values: Any,
    *,
    n_resamples: int = 2000,
    seed: int = 20260728,
) -> dict[str, Any]:
    """计算配对均值差（treatment-baseline）及bootstrap 95% CI。"""
    try:
        if (
            not isinstance(baseline_values, list)
            or not isinstance(treatment_values, list)
            or len(baseline_values) != len(treatment_values)
            or not baseline_values
        ):
            return _error_result(
                "配对bootstrap需要等长非空列表",
                "INVALID_PAIRED_VALUES",
            )
        if not isinstance(n_resamples, int) or n_resamples < 100:
            return _error_result(
                "bootstrap重采样次数必须至少为100",
                "INVALID_BOOTSTRAP_RESAMPLES",
            )
        differences = [
            float(treatment) - float(baseline)
            for baseline, treatment in zip(
                baseline_values,
                treatment_values,
            )
        ]
        random_generator = random.Random(seed)
        estimates = []
        sample_size = len(differences)
        for _ in range(n_resamples):
            sample = [
                differences[random_generator.randrange(sample_size)]
                for _ in range(sample_size)
            ]
            estimates.append(statistics.fmean(sample))
        estimates.sort()
        return {
            "status": "success",
            "pair_count": sample_size,
            "mean_difference": statistics.fmean(differences),
            "ci_95": [
                _percentile(estimates, 0.025),
                _percentile(estimates, 0.975),
            ],
            "n_resamples": n_resamples,
            "seed": seed,
        }
    except Exception as exc:
        logger.error("计算配对bootstrap失败: %s", exc, exc_info=True)
        return _error_result(
            f"计算配对bootstrap失败: {exc}",
            "BOOTSTRAP_EXCEPTION",
        )


def mcnemar_exact(
    baseline_values: Any,
    treatment_values: Any,
) -> dict[str, Any]:
    """计算二元配对结果的精确McNemar检验和不一致对数量。"""
    try:
        if (
            not isinstance(baseline_values, list)
            or not isinstance(treatment_values, list)
            or len(baseline_values) != len(treatment_values)
            or not baseline_values
        ):
            return _error_result(
                "McNemar检验需要等长非空二元列表",
                "INVALID_MCNEMAR_VALUES",
            )
        normalized_baseline = [bool(value) for value in baseline_values]
        normalized_treatment = [bool(value) for value in treatment_values]
        baseline_only = sum(
            baseline and not treatment
            for baseline, treatment in zip(
                normalized_baseline,
                normalized_treatment,
            )
        )
        treatment_only = sum(
            treatment and not baseline
            for baseline, treatment in zip(
                normalized_baseline,
                normalized_treatment,
            )
        )
        discordant = baseline_only + treatment_only
        if discordant == 0:
            p_value = 1.0
        else:
            tail = sum(
                math.comb(discordant, index)
                for index in range(
                    min(baseline_only, treatment_only) + 1
                )
            ) / (2 ** discordant)
            p_value = min(1.0, 2.0 * tail)
        return {
            "status": "success",
            "pair_count": len(normalized_baseline),
            "baseline_only_success": baseline_only,
            "treatment_only_success": treatment_only,
            "discordant_pair_count": discordant,
            "p_value_two_sided": p_value,
        }
    except Exception as exc:
        logger.error("计算精确McNemar检验失败: %s", exc, exc_info=True)
        return _error_result(
            f"计算精确McNemar检验失败: {exc}",
            "MCNEMAR_EXCEPTION",
        )


def _pair_key(record: dict[str, Any]) -> tuple[str, str, int]:
    """生成同题同模型同重复轮次的配对键。"""
    try:
        return (
            str(record["question_id"]),
            str(record["context"]["model"]),
            int(record["repeat_index"]),
        )
    except Exception:
        return ("", "", -1)


def _paired_metric(
    baseline_map: dict[tuple[str, str, int], dict[str, Any]],
    treatment_map: dict[tuple[str, str, int], dict[str, Any]],
    extractor: Callable[[dict[str, Any]], Any],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """对两个条件共同存在且值非null的记录执行配对比较。"""
    try:
        baseline_values: list[float] = []
        treatment_values: list[float] = []
        for key in sorted(set(baseline_map) & set(treatment_map)):
            baseline_value = extractor(baseline_map[key])
            treatment_value = extractor(treatment_map[key])
            if baseline_value is None or treatment_value is None:
                continue
            baseline_values.append(float(baseline_value))
            treatment_values.append(float(treatment_value))
        if not baseline_values:
            return {
                "status": "success",
                "pair_count": 0,
                "mean_difference": None,
                "ci_95": None,
                "n_resamples": n_resamples,
                "seed": seed,
            }
        return paired_bootstrap_difference(
            baseline_values,
            treatment_values,
            n_resamples=n_resamples,
            seed=seed,
        )
    except Exception as exc:
        logger.error("提取配对实验指标失败: %s", exc, exc_info=True)
        return _error_result(
            f"提取配对实验指标失败: {exc}",
            "PAIRED_METRIC_EXCEPTION",
        )


def _paired_binary_metric(
    baseline_map: dict[tuple[str, str, int], dict[str, Any]],
    treatment_map: dict[tuple[str, str, int], dict[str, Any]],
    extractor: Callable[[dict[str, Any]], Any],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """同时报告二元指标的配对差、bootstrap CI和精确McNemar结果。"""
    try:
        baseline_values: list[bool] = []
        treatment_values: list[bool] = []
        for key in sorted(set(baseline_map) & set(treatment_map)):
            baseline_value = extractor(baseline_map[key])
            treatment_value = extractor(treatment_map[key])
            if baseline_value is None or treatment_value is None:
                continue
            baseline_values.append(bool(baseline_value))
            treatment_values.append(bool(treatment_value))
        if not baseline_values:
            return {
                "status": "success",
                "pair_count": 0,
                "mean_difference": None,
                "ci_95": None,
                "mcnemar": None,
                "n_resamples": n_resamples,
                "seed": seed,
            }
        bootstrap = paired_bootstrap_difference(
            baseline_values,
            treatment_values,
            n_resamples=n_resamples,
            seed=seed,
        )
        if bootstrap.get("status") != "success":
            return bootstrap
        bootstrap["mcnemar"] = mcnemar_exact(
            baseline_values,
            treatment_values,
        )
        return bootstrap
    except Exception as exc:
        logger.error("计算配对二元实验指标失败: %s", exc, exc_info=True)
        return _error_result(
            f"计算配对二元实验指标失败: {exc}",
            "PAIRED_BINARY_METRIC_EXCEPTION",
        )


def compare_conditions(
    records: Any,
    *,
    baseline_condition: str = "B0",
    n_resamples: int = 2000,
    seed: int = 20260728,
) -> dict[str, Any]:
    """以B0为基线对B1/B2做同题同模型同轮次配对比较。"""
    try:
        if not isinstance(records, list) or not records:
            return _error_result(
                "条件比较需要非空Trace列表",
                "EMPTY_COMPARISON_INPUT",
            )
        if baseline_condition not in EXPERIMENT_CONDITIONS:
            return _error_result(
                "基线条件必须是B0、B1或B2",
                "UNKNOWN_BASELINE_CONDITION",
            )
        models = {
            item.get("context", {}).get("model")
            for item in records
            if isinstance(item, dict)
        }
        repeat_indices = {
            item.get("repeat_index")
            for item in records
            if isinstance(item, dict)
        }
        if len(models) != 1 or len(repeat_indices) != 1:
            return _error_result(
                "条件配对比较必须按单一模型和单一repeat分别运行",
                "MIXED_MODEL_OR_REPEAT",
            )

        condition_maps: dict[
            str,
            dict[tuple[str, str, int], dict[str, Any]],
        ] = defaultdict(dict)
        for index, item in enumerate(records):
            validation = validate_experiment_trace(item)
            if validation.get("status") != "success":
                return _error_result(
                    f"第{index}条Trace未通过Schema校验",
                    "INVALID_COMPARISON_TRACE",
                )
            condition = item["condition"]
            key = _pair_key(item)
            if key in condition_maps[condition]:
                return _error_result(
                    "同一条件存在重复配对键",
                    "DUPLICATE_PAIR_KEY",
                    details={
                        "condition": condition,
                        "pair_key": list(key),
                    },
                )
            condition_maps[condition][key] = item

        baseline_map = condition_maps.get(baseline_condition) or {}
        if not baseline_map:
            return _error_result(
                "输入记录中缺少基线条件",
                "MISSING_BASELINE_RECORDS",
            )

        comparisons = {}
        for condition in sorted(condition_maps):
            if condition == baseline_condition:
                continue
            treatment_map = condition_maps[condition]
            paired_keys = set(baseline_map) & set(treatment_map)
            comparisons[condition] = {
                "available_pair_count": len(paired_keys),
                "final_answer_accuracy_difference": _paired_binary_metric(
                    baseline_map,
                    treatment_map,
                    lambda item: item["evaluation"][
                        "final_answer_correct"
                    ],
                    n_resamples=n_resamples,
                    seed=seed,
                ),
                "executable_plan_rate_difference": _paired_binary_metric(
                    baseline_map,
                    treatment_map,
                    lambda item: (
                        item["observation"]["execution"]["completed"]
                        if item["gold"]["expected_behavior"] != "reject"
                        else None
                    ),
                    n_resamples=n_resamples,
                    seed=seed + 1,
                ),
                "type_ii_generation_rate_difference": _paired_binary_metric(
                    baseline_map,
                    treatment_map,
                    lambda item: _has_type_ii(item, "before"),
                    n_resamples=n_resamples,
                    seed=seed + 2,
                ),
                "unresolved_type_ii_rate_difference": _paired_binary_metric(
                    baseline_map,
                    treatment_map,
                    lambda item: _has_type_ii(item, "after"),
                    n_resamples=n_resamples,
                    seed=seed + 3,
                ),
                "unsafe_type_ii_execution_rate_difference": _paired_binary_metric(
                    baseline_map,
                    treatment_map,
                    unsafe_type_ii_execution,
                    n_resamples=n_resamples,
                    seed=seed + 4,
                ),
                "repair_recovery_rate_difference": _paired_binary_metric(
                    baseline_map,
                    treatment_map,
                    repair_recovery_outcome,
                    n_resamples=n_resamples,
                    seed=seed + 5,
                ),
                "behavior_aware_task_success_difference": _paired_binary_metric(
                    baseline_map,
                    treatment_map,
                    behavior_aware_task_success,
                    n_resamples=n_resamples,
                    seed=seed + 6,
                ),
                "total_latency_ms_difference": _paired_metric(
                    baseline_map,
                    treatment_map,
                    lambda item: item["observation"]["latency"][
                        "total_ms"
                    ],
                    n_resamples=n_resamples,
                    seed=seed + 7,
                ),
                "extra_llm_calls_difference": _paired_metric(
                    baseline_map,
                    treatment_map,
                    lambda item: item["observation"]["plan"].get(
                        "extra_llm_calls",
                        0,
                    ),
                    n_resamples=n_resamples,
                    seed=seed + 8,
                ),
                "extra_llm_total_tokens_difference": _paired_metric(
                    baseline_map,
                    treatment_map,
                    lambda item: item["observation"]["plan"].get(
                        "extra_llm_total_tokens",
                        0,
                    ),
                    n_resamples=n_resamples,
                    seed=seed + 9,
                ),
            }

        return {
            "status": "success",
            "baseline_condition": baseline_condition,
            "model": next(iter(models)),
            "repeat_index": next(iter(repeat_indices)),
            "comparisons": comparisons,
        }
    except Exception as exc:
        logger.error("比较实验条件失败: %s", exc, exc_info=True)
        return _error_result(
            f"比较实验条件失败: {exc}",
            "CONDITION_COMPARISON_EXCEPTION",
        )
