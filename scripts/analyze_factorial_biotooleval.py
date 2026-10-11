# -*- coding: utf-8 -*-
"""BioToolEval E0–E3 与 S1/L1 配对实验分析入口。

修改记录：
- 2026-09-03：新增干预后 baseline 诊断与重规划账本核对，修正成本
  显示层级及缺失值处理；保留原 H1–H3 定义，新增报告版本不覆盖旧结果。
- 2026-08-31：新增六条件分析、共享冻结计划审计、失败单元记账、
  H1–H3 配对统计和 baseline 成本比较；不改变历史 B0–B2 分析口径。

设计边界：
1. 只接受完整声明的 E0/E1/E2/E3/S1/L1 条件集合；
2. 同题同模型同轮次必须共享一个原始计划 ID；
3. H1 比较 E2 vs E1，H2 比较 E3 vs E2，H3 比较 E3 vs E0；
4. 缺少有效分母时返回 not_estimable，禁止用 0 伪造结果；
5. 失败账本进入完整性分母，不把失败 Trace 静默删除。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from app.core.logger import get_logger
from scripts.analyze_biotooleval import (
    _atomic_write_text,
    _control_value_set,
    _error_result,
    _format_ci,
    _format_number,
    _is_type_sensitive,
    _load_failure_summary,
    _load_manifest,
    _metric_rate,
    _normalize_json_value,
    _question_signature,
    _sha256_file,
    holm_adjust,
)
from scripts.evaluation_metrics import (
    behavior_aware_task_success,
    compare_conditions,
    compute_condition_metrics,
    paired_bootstrap_difference,
    repair_opportunity,
)
from scripts.evaluation_trace import load_experiment_traces, validate_experiment_trace
from scripts.baseline_audit import audit_baselines, extra_replan_cost_value


logger = get_logger("BioToolEvalFactorialAnalysis", "factorial_analysis.log")

ANALYSIS_SCHEMA_VERSION = "factorial-1.1"
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_ANALYSIS_SEED = 20260831
REQUIRED_CONDITIONS = ("E0", "E1", "E2", "E3", "S1", "L1")
PRIMARY_HYPOTHESES = (
    {
        "id": "H1",
        "comparison": "E2_vs_E1",
        "metric": "unsafe_type_ii_execution_rate",
        "description": "类型敏感题上验证门禁相对仅类型化运行时的不安全执行差异",
    },
    {
        "id": "H2",
        "comparison": "E3_vs_E2",
        "metric": "repair_recovery_rate",
        "description": "真实修复机会中确定性修复相对仅验证的恢复差异",
    },
    {
        "id": "H3",
        "comparison": "E3_vs_E0",
        "metric": "behavior_aware_task_success",
        "description": "类型敏感题上完整方法相对无类型系统的行为感知成功差异",
    },
)
QUESTION_COMPARISONS = (
    ("E1", "E2"),
    ("E2", "E3"),
    ("E0", "E3"),
    ("E0", "S1"),
    ("E0", "L1"),
    ("E3", "L1"),
)
QUESTION_METRICS = (
    "final_answer_accuracy",
    "behavior_aware_task_success",
    "executable_plan_rate",
    "total_latency_ms",
    "extra_llm_calls",
    "extra_llm_total_tokens",
)


def _extract_metric(record: dict[str, Any], metric: str) -> float | None:
    """提取题目级聚合数值；不适用或未人工标注时返回 None。"""
    try:
        if metric == "final_answer_accuracy":
            value = record.get("evaluation", {}).get("final_answer_correct")
            return float(value) if isinstance(value, bool) else None
        if metric == "behavior_aware_task_success":
            value = behavior_aware_task_success(record)
            return float(value) if isinstance(value, bool) else None
        if metric == "executable_plan_rate":
            if record.get("gold", {}).get("expected_behavior") == "reject":
                return None
            value = record.get("observation", {}).get("execution", {}).get("completed")
            return float(value) if isinstance(value, bool) else None
        if metric == "total_latency_ms":
            value = record.get("observation", {}).get("latency", {}).get("total_ms")
            return float(value) if type(value) in (int, float) and value >= 0 and math.isfinite(value) else None
        if metric in {"extra_llm_calls", "extra_llm_total_tokens"}:
            value = extra_replan_cost_value(record, metric)
            return float(value) if value is not None else None
        return None
    except Exception as exc:
        logger.error("提取六条件题目级指标失败: %s", exc, exc_info=True)
        return None


def _audit_records(
    records: Any,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """审计六条件 Trace、控制变量、Manifest 和共享原始计划。"""
    try:
        if not isinstance(records, list) or not records:
            return _error_result("六条件分析输入不能为空", "EMPTY_FACTORIAL_INPUT")
        if not isinstance(manifest, dict):
            return _error_result("六条件分析必须提供 Manifest", "MISSING_FACTORIAL_MANIFEST")

        required = set(REQUIRED_CONDITIONS)
        observed = {str(item.get("condition")) for item in records}
        declared = set((manifest.get("design") or {}).get("conditions") or [])
        if observed != required or declared != required:
            return _error_result(
                "六条件分析必须同时包含并声明 E0/E1/E2/E3/S1/L1",
                "INCOMPLETE_FACTORIAL_CONDITIONS",
                details={
                    "required": sorted(required),
                    "observed": sorted(observed),
                    "declared": sorted(declared),
                },
            )

        seen: set[tuple[str, str, str, int]] = set()
        signatures: dict[str, str] = {}
        paired_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        for index, record in enumerate(records):
            validation = validate_experiment_trace(record)
            if validation.get("status") != "success":
                return _error_result(
                    f"第 {index} 条六条件 Trace 未通过 Schema 校验",
                    "INVALID_FACTORIAL_TRACE",
                    details={"validation": validation},
                )
            question_id = str(record.get("question_id"))
            model = str(record.get("context", {}).get("model"))
            repeat_index = int(record.get("repeat_index"))
            condition = str(record.get("condition"))
            unit_key = (question_id, model, condition, repeat_index)
            if unit_key in seen:
                return _error_result(
                    "六条件 Trace 存在重复实验单元",
                    "DUPLICATE_FACTORIAL_UNIT",
                    details={"unit_key": list(unit_key)},
                )
            seen.add(unit_key)
            signature = _question_signature(record)
            if question_id in signatures and signatures[question_id] != signature:
                return _error_result(
                    "同一题跨条件的脱敏元数据发生漂移",
                    "FACTORIAL_QUESTION_METADATA_DRIFT",
                    details={"question_id": question_id},
                )
            signatures[question_id] = signature
            paired_groups.setdefault((question_id, model, repeat_index), []).append(record)

        controls: dict[str, Any] = {}
        for field in (
            "model",
            "temperature",
            "seed",
            "git_commit",
            "dataset_version",
            "kg_snapshot",
            "python_version",
        ):
            values = _control_value_set(records, field)
            if len(values) != 1:
                return _error_result(
                    f"六条件 Trace 混入多个 {field}",
                    "MIXED_FACTORIAL_CONTROLS",
                    details={"field": field, "values": sorted(values)},
                )
            controls[field] = records[0].get("context", {}).get(field)

        design = manifest.get("design") or {}
        dataset = manifest.get("dataset") or {}
        environment = manifest.get("environment") or {}
        manifest_controls = {
            "model": design.get("model"),
            "temperature": design.get("temperature"),
            "seed": design.get("seed"),
            "git_commit": environment.get("git_commit"),
            "dataset_version": dataset.get("version"),
            "kg_snapshot": environment.get("kg_snapshot"),
            "python_version": environment.get("python_version"),
        }
        for field, expected in manifest_controls.items():
            if _normalize_json_value(expected) != _normalize_json_value(controls[field]):
                return _error_result(
                    f"Manifest 与 Trace 的 {field} 不一致",
                    "FACTORIAL_MANIFEST_CONTROL_MISMATCH",
                    details={
                        "field": field,
                        "manifest_value": expected,
                        "trace_value": controls[field],
                    },
                )

        plan_pairing_issues: list[dict[str, Any]] = []
        complete_pair_count = 0
        for pair_key, group in sorted(paired_groups.items()):
            group_conditions = {str(item.get("condition")) for item in group}
            plan_ids = {
                str(item.get("observation", {}).get("plan", {}).get("plan_sample_id") or "")
                for item in group
            }
            replay_flags = {
                item.get("observation", {}).get("plan", {}).get("shared_plan_replay")
                for item in group
            }
            if group_conditions == required:
                complete_pair_count += 1
            if len(plan_ids) != 1 or "" in plan_ids or replay_flags != {True}:
                plan_pairing_issues.append(
                    {
                        "pair_key": list(pair_key),
                        "conditions": sorted(group_conditions),
                        "plan_id_count": len(plan_ids - {""}),
                        "shared_plan_replay_values": sorted(str(value) for value in replay_flags),
                    }
                )
        if plan_pairing_issues:
            return _error_result(
                "六条件没有严格共享同一冻结原始计划",
                "FACTORIAL_PLAN_PAIRING_FAILED",
                details={"pairing_issues": plan_pairing_issues[:10]},
            )

        question_ids = design.get("question_ids")
        question_count = (
            len({str(value) for value in question_ids})
            if isinstance(question_ids, list) and question_ids
            else len(signatures)
        )
        repeats = design.get("repeats")
        expected_count = (
            question_count * int(repeats) * len(REQUIRED_CONDITIONS)
            if isinstance(repeats, int) and repeats > 0
            else None
        )
        observed_by_condition = {
            condition: sum(record.get("condition") == condition for record in records)
            for condition in REQUIRED_CONDITIONS
        }
        return {
            "status": "success",
            "controls": controls,
            "repeat_indices": sorted({int(item.get("repeat_index")) for item in records}),
            "question_count": len(signatures),
            "type_sensitive_question_count": len(
                {
                    str(item.get("question_id"))
                    for item in records
                    if _is_type_sensitive(item)
                }
            ),
            "completeness": {
                "expected_trace_count": expected_count,
                "observed_trace_count": len(records),
                "missing_trace_count": (
                    max(0, expected_count - len(records))
                    if isinstance(expected_count, int)
                    else None
                ),
                "observed_by_condition": observed_by_condition,
                "paired_plan_group_count": len(paired_groups),
                "complete_paired_plan_group_count": complete_pair_count,
            },
        }
    except Exception as exc:
        logger.error("审计六条件 Trace 失败: %s", exc, exc_info=True)
        return _error_result(
            f"审计六条件 Trace 失败: {exc}",
            "FACTORIAL_AUDIT_EXCEPTION",
        )


def _not_estimable(specification: dict[str, Any], reason: str) -> dict[str, Any]:
    """构造零分母主要假设结果，保留 null 而非伪造 0。"""
    return {
        **specification,
        "pair_count": 0,
        "mean_difference": None,
        "ci_95": None,
        "p_value_raw": None,
        "discordant_pair_count": None,
        "estimability_status": "not_estimable",
        "estimability_reason": reason,
    }


def _extract_hypothesis(
    specification: dict[str, Any],
    comparison: dict[str, Any],
    treatment_condition: str,
) -> dict[str, Any]:
    """从配对比较中提取一个预注册 H1–H3 结果。"""
    try:
        if comparison.get("status") != "success":
            return _not_estimable(specification, "比较未形成有效结果")
        metric = (
            comparison.get("comparisons", {})
            .get(treatment_condition, {})
            .get(f"{specification['metric']}_difference")
        )
        if not isinstance(metric, dict) or int(metric.get("pair_count") or 0) <= 0:
            return _not_estimable(specification, "没有满足指标定义的成对观测")
        mcnemar = metric.get("mcnemar")
        if not isinstance(mcnemar, dict):
            return _not_estimable(specification, "二元配对检验不可计算")
        return {
            **specification,
            "pair_count": int(metric["pair_count"]),
            "mean_difference": metric.get("mean_difference"),
            "ci_95": metric.get("ci_95"),
            "p_value_raw": mcnemar.get("p_value_two_sided"),
            "discordant_pair_count": mcnemar.get("discordant_pair_count"),
            "estimability_status": "estimable",
            "estimability_reason": None,
        }
    except Exception as exc:
        logger.error("提取六条件主要假设失败: %s", exc, exc_info=True)
        return _not_estimable(specification, "提取比较结果异常")


def _condition_pair(
    records: list[dict[str, Any]],
    baseline: str,
    treatment: str,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """只保留指定两条件，避免无关条件污染配对结果结构。"""
    subset = [item for item in records if item.get("condition") in {baseline, treatment}]
    if not subset:
        return _error_result("指定条件没有记录", "EMPTY_FACTORIAL_PAIR")
    return compare_conditions(
        subset,
        baseline_condition=baseline,
        n_resamples=n_resamples,
        seed=seed,
    )


def _compute_repeat_sections(
    records: list[dict[str, Any]],
    repeat_indices: list[int],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """逐轮计算六条件指标、H1–H3 和 baseline 成本比较。"""
    try:
        sections: dict[str, Any] = {}
        for repeat_index in repeat_indices:
            repeat_records = [
                item for item in records if int(item.get("repeat_index")) == repeat_index
            ]
            condition_metrics = {
                condition: compute_condition_metrics(
                    [item for item in repeat_records if item.get("condition") == condition]
                )
                for condition in REQUIRED_CONDITIONS
            }
            type_sensitive = [item for item in repeat_records if _is_type_sensitive(item)]
            h1_comparison = _condition_pair(
                type_sensitive,
                "E1",
                "E2",
                n_resamples=n_resamples,
                seed=seed + repeat_index * 100,
            )
            repair_records = [
                item
                for item in repeat_records
                if item.get("condition") in {"E2", "E3"} and repair_opportunity(item)
            ]
            h2_comparison = (
                _condition_pair(
                    repair_records,
                    "E2",
                    "E3",
                    n_resamples=n_resamples,
                    seed=seed + repeat_index * 100 + 10,
                )
                if repair_records
                else _error_result("该轮没有真实修复机会", "NO_REPAIR_OPPORTUNITY_RECORDS")
            )
            h3_comparison = _condition_pair(
                type_sensitive,
                "E0",
                "E3",
                n_resamples=n_resamples,
                seed=seed + repeat_index * 100 + 20,
            )
            primary = [
                _extract_hypothesis(PRIMARY_HYPOTHESES[0], h1_comparison, "E2"),
                _extract_hypothesis(PRIMARY_HYPOTHESES[1], h2_comparison, "E3"),
                _extract_hypothesis(PRIMARY_HYPOTHESES[2], h3_comparison, "E3"),
            ]
            baseline_comparisons = {
                f"{treatment}_vs_{baseline}": _condition_pair(
                    repeat_records,
                    baseline,
                    treatment,
                    n_resamples=n_resamples,
                    seed=seed + repeat_index * 100 + 30 + index,
                )
                for index, (baseline, treatment) in enumerate(QUESTION_COMPARISONS)
            }
            sections[str(repeat_index)] = {
                "repeat_index": repeat_index,
                "record_count": len(repeat_records),
                "type_sensitive_record_count": len(type_sensitive),
                "repair_opportunity_record_count": len(repair_records),
                "condition_metrics": condition_metrics,
                "primary_comparisons": {
                    "H1": h1_comparison,
                    "H2": h2_comparison,
                    "H3": h3_comparison,
                },
                "baseline_comparisons": baseline_comparisons,
                "primary_hypothesis_family": holm_adjust(
                    primary,
                    planned_family_size=len(PRIMARY_HYPOTHESES),
                ),
            }
        return {"status": "success", "repeats": sections}
    except Exception as exc:
        logger.error("计算六条件逐轮指标失败: %s", exc, exc_info=True)
        return _error_result(
            f"计算六条件逐轮指标失败: {exc}",
            "FACTORIAL_REPEAT_ANALYSIS_EXCEPTION",
        )


def _question_pair_metric(
    records: list[dict[str, Any]],
    baseline: str,
    treatment: str,
    metric: str,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """题内先聚合重复，再以题目为统计单位做配对 bootstrap。"""
    try:
        grouped: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
        for item in records:
            condition = str(item.get("condition"))
            if condition not in {baseline, treatment}:
                continue
            question_id = str(item.get("question_id"))
            repeat_index = int(item.get("repeat_index"))
            grouped.setdefault(question_id, {}).setdefault(condition, {})[
                repeat_index
            ] = item
        baseline_values: list[float] = []
        treatment_values: list[float] = []
        paired_repeat_count = 0
        unmatched_repeat_count = 0
        null_repeat_pair_count = 0
        without_usable_pair = 0
        for question_id in sorted(grouped):
            baseline_map = grouped[question_id].get(baseline, {})
            treatment_map = grouped[question_id].get(treatment, {})
            common = sorted(set(baseline_map) & set(treatment_map))
            unmatched_repeat_count += len(set(baseline_map) ^ set(treatment_map))
            local_baseline: list[float] = []
            local_treatment: list[float] = []
            for repeat_index in common:
                baseline_value = _extract_metric(baseline_map[repeat_index], metric)
                treatment_value = _extract_metric(treatment_map[repeat_index], metric)
                if baseline_value is None or treatment_value is None:
                    null_repeat_pair_count += 1
                    continue
                local_baseline.append(baseline_value)
                local_treatment.append(treatment_value)
                paired_repeat_count += 1
            if not local_baseline:
                without_usable_pair += 1
                continue
            baseline_values.append(sum(local_baseline) / len(local_baseline))
            treatment_values.append(sum(local_treatment) / len(local_treatment))
        total_questions = len(grouped)
        common_result = {
            "metric": metric,
            "comparison": f"{treatment}_vs_{baseline}",
            "question_count": len(baseline_values),
            "total_question_count": total_questions,
            "question_coverage": (
                len(baseline_values) / total_questions if total_questions else None
            ),
            "paired_repeat_count": paired_repeat_count,
            "unmatched_repeat_count": unmatched_repeat_count,
            "null_repeat_pair_count": null_repeat_pair_count,
            "questions_without_usable_pair": without_usable_pair,
            "statistical_unit": "question",
            "repeat_handling": "mean_within_question_before_bootstrap",
        }
        if not baseline_values:
            return {
                "status": "success",
                **common_result,
                "mean_difference": None,
                "ci_95": None,
                "n_resamples": n_resamples,
                "seed": seed,
            }
        bootstrap = paired_bootstrap_difference(
            baseline_values,
            treatment_values,
            n_resamples=n_resamples,
            seed=seed,
        )
        return {**bootstrap, **common_result}
    except Exception as exc:
        logger.error("计算六条件题目级配对失败: %s", exc, exc_info=True)
        return _error_result(
            f"计算六条件题目级配对失败: {exc}",
            "FACTORIAL_QUESTION_PAIR_EXCEPTION",
        )


def _compute_question_level(
    records: list[dict[str, Any]],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """计算总体与类型敏感题的跨重复题目级效应。"""
    try:
        populations = {
            "all": records,
            "type_sensitive": [item for item in records if _is_type_sensitive(item)],
        }
        output: dict[str, Any] = {}
        counter = 0
        for population_name, population_records in populations.items():
            comparisons: dict[str, Any] = {}
            for baseline, treatment in QUESTION_COMPARISONS:
                comparison_name = f"{treatment}_vs_{baseline}"
                comparisons[comparison_name] = {}
                for metric in QUESTION_METRICS:
                    comparisons[comparison_name][metric] = _question_pair_metric(
                        population_records,
                        baseline,
                        treatment,
                        metric,
                        n_resamples=n_resamples,
                        seed=seed + 1000 + counter,
                    )
                    counter += 1
            output[population_name] = comparisons
        return {
            "status": "success",
            "statistical_unit": "question",
            "populations": output,
        }
    except Exception as exc:
        logger.error("计算六条件题目级分析失败: %s", exc, exc_info=True)
        return _error_result(
            f"计算六条件题目级分析失败: {exc}",
            "FACTORIAL_QUESTION_ANALYSIS_EXCEPTION",
        )


def analyze_factorial_records(
    records: Any,
    *,
    manifest: dict[str, Any],
    failure_summary: dict[str, Any] | None = None,
    trace_sha256: str | None = None,
    replan_ledger: Any = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_ANALYSIS_SEED,
) -> dict[str, Any]:
    """分析完整六条件实验并返回可序列化、可独立复算的报告。"""
    try:
        if not isinstance(n_resamples, int) or n_resamples < 100:
            return _error_result("bootstrap 次数至少为 100", "INVALID_FACTORIAL_RESAMPLES")
        if not isinstance(seed, int):
            return _error_result("分析种子必须为整数", "INVALID_FACTORIAL_SEED")
        audit = _audit_records(records, manifest)
        if audit.get("status") != "success":
            return audit
        baseline_audit = audit_baselines(
            records, experiment_id=manifest.get("experiment_id"), replan_ledger=replan_ledger,
        )
        if baseline_audit.get("status") != "success":
            return baseline_audit
        repeat_analysis = _compute_repeat_sections(
            records,
            audit["repeat_indices"],
            n_resamples=n_resamples,
            seed=seed,
        )
        if repeat_analysis.get("status") != "success":
            return repeat_analysis
        question_level = _compute_question_level(
            records,
            n_resamples=n_resamples,
            seed=seed,
        )
        if question_level.get("status") != "success":
            return question_level
        failures = (
            copy.deepcopy(failure_summary)
            if isinstance(failure_summary, dict)
            else {
                "status": "success",
                "provided": False,
                "unresolved_failure_count": 0,
                "by_condition": {},
                "by_error_code": {},
                "by_category": {},
            }
        )
        if failures.get("status") != "success":
            return failures
        completeness = copy.deepcopy(audit["completeness"])
        failure_count = int(failures.get("unresolved_failure_count") or 0)
        expected_count = completeness.get("expected_trace_count")
        accounted_count = int(completeness["observed_trace_count"]) + failure_count
        completeness["failure_record_count"] = failure_count
        completeness["accounted_unit_count"] = accounted_count
        completeness["unaccounted_unit_count"] = (
            max(0, int(expected_count) - accounted_count)
            if isinstance(expected_count, int)
            else None
        )
        return {
            "status": "success",
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "material_passport": {
                "artifact_id": "BIOTOOL-EVAL-E0-E3-S1-L1-ANALYSIS",
                "artifact_type": "matched_factorial_and_baseline_analysis",
                "verification_status": "ANALYZED",
                "evidence_scope": "DEVELOPMENT_ONLY"
                if manifest.get("development_run") is True
                else "FORMAL_RESULTS_REQUIRE_HUMAN_INTERPRETATION",
            },
            "input": {
                "trace_sha256": str(trace_sha256 or "") or None,
                "manifest_experiment_id": manifest.get("experiment_id"),
                "dataset": manifest.get("dataset"),
            },
            "analysis_config": {
                "conditions": list(REQUIRED_CONDITIONS),
                "bootstrap_resamples": n_resamples,
                "seed": seed,
                "holm_alpha": 0.05,
                "repeat_policy": "mean_within_question_before_bootstrap",
            },
            "controls": audit["controls"],
            "completeness": completeness,
            "failure_outcomes": failures,
            "question_count": audit["question_count"],
            "type_sensitive_question_count": audit["type_sensitive_question_count"],
            "per_repeat": repeat_analysis["repeats"],
            "question_level": question_level,
            "baseline_audit": baseline_audit,
            "interpretation_boundary": [
                "自然修复机会为零时 H2 必须报告 not_estimable",
                "人工标签为 null 时准确率与行为成功不得按错误或正确计入",
                "受控修复微基准不能替代自然 Dev 分布中的修复机会率",
                "本报告不自动宣布显著、因果或投稿 Go/No-Go 结论",
                "旧 unsafe_type_ii_execution_rate 统计初始 Type-II 后任意调用，不能把 L1 的纠正后执行解释为不安全",
                "跨修复/重规划条件另看 baseline_audit 的干预后诊断；原 H1–H3 与历史结果保持可比",
                "额外重规划成本不含共享首次规划、回答生成或其他模型调用，不是整场费用",
            ],
        }
    except Exception as exc:
        logger.error("六条件分析崩溃: %s", exc, exc_info=True)
        return _error_result(
            f"六条件分析崩溃: {exc}",
            "FACTORIAL_ANALYSIS_EXCEPTION",
        )


def render_factorial_markdown(analysis: Any) -> dict[str, Any]:
    """把六条件分析渲染为简洁且可审计的 Markdown。"""
    try:
        if not isinstance(analysis, dict) or analysis.get("status") != "success":
            return _error_result("只有成功分析可以渲染", "INVALID_FACTORIAL_MARKDOWN")
        passport = analysis.get("material_passport", {})
        completeness = analysis.get("completeness", {})
        lines = [
            "# BioToolEval E0–E3 / S1–L1 配对实验分析",
            "",
            f"- Artifact ID：`{passport.get('artifact_id')}`",
            f"- Evidence Scope：`{passport.get('evidence_scope')}`",
            f"- 题目数：{analysis.get('question_count')}",
            f"- 预期/实际/失败 Trace：{completeness.get('expected_trace_count')} / "
            f"{completeness.get('observed_trace_count')} / "
            f"{completeness.get('failure_record_count')}",
            f"- 未解释实验单元：{completeness.get('unaccounted_unit_count')}",
            f"- 完整共享计划组：{completeness.get('complete_paired_plan_group_count')} / "
            f"{completeness.get('paired_plan_group_count')}",
            "",
            "## 逐轮条件指标",
            "",
            "| repeat | condition | Final Answer Accuracy | Executable Plan Rate | Repair Precision | Extra LLM Calls |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for repeat_key, repeat_section in analysis.get("per_repeat", {}).items():
            for condition in REQUIRED_CONDITIONS:
                result = repeat_section.get("condition_metrics", {}).get(condition, {})
                # baseline 成本在嵌套对象中；非 baseline 无此旧字段，不能伪装成缺失用量。
                cost = result.get("metrics", {}).get("external_baseline_cost", {})
                calls = cost.get("extra_llm_calls", {}).get("mean")
                if condition in {"E0", "E1", "E2", "E3"}:
                    calls = 0
                lines.append(
                    f"| {repeat_key} | {condition} | "
                    f"{_metric_rate(result, 'final_answer_accuracy')} | "
                    f"{_metric_rate(result, 'executable_plan_rate')} | "
                    f"{_metric_rate(result, 'repair_precision')} | "
                    f"{_format_number(calls)} |"
                )
        lines.extend(
            [
                "",
                "## 主要假设",
                "",
                "| repeat | hypothesis | comparison | status | effect | 95% CI | raw p | Holm p | pairs |",
                "|---:|---|---|---|---:|---|---:|---:|---:|",
            ]
        )
        for repeat_key, repeat_section in analysis.get("per_repeat", {}).items():
            family = repeat_section.get("primary_hypothesis_family", {})
            for hypothesis in family.get("hypotheses", []):
                lines.append(
                    f"| {repeat_key} | {hypothesis.get('id')} | "
                    f"{hypothesis.get('comparison')} | "
                    f"{hypothesis.get('estimability_status')} | "
                    f"{_format_number(hypothesis.get('mean_difference'))} | "
                    f"{_format_ci(hypothesis.get('ci_95'))} | "
                    f"{_format_number(hypothesis.get('p_value_raw'))} | "
                    f"{_format_number(hypothesis.get('p_value_holm'))} | "
                    f"{hypothesis.get('pair_count', 0)} |"
                )
        baseline = analysis.get("baseline_audit", {})
        lines.extend([
            "", "## 干预后 baseline 诊断（补充描述，不替换 H1–H3）", "",
            "| 条件 | 初始 Type-II | 零调用阻断 | 纠正并通过类型检查后执行 | 未解决 Type-II 仍执行 | 后续状态未知 | 额外调用总数 | 额外 token 总数 | token 覆盖 | 重规划耗时总 ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for condition, section in baseline.get("by_condition", {}).items():
            classes = section["execution_classes"]
            cost = section["extra_replan_cost"]
            tokens = cost["extra_llm_total_tokens"]
            lines.append(
                f"| {condition} | {section['initial_type_ii_count']} | "
                f"{classes.get('blocked_without_tool_attempt', 0)} | "
                f"{classes.get('executed_after_type_valid_correction', 0)} | "
                f"{classes.get('executed_with_unresolved_type_ii', 0)} | "
                f"{classes.get('unknown_after_initial_type_ii', 0)} | "
                f"{_format_number(cost['extra_llm_calls']['total'])} | "
                f"{_format_number(tokens['total'])} | {tokens['known_count']}/{tokens['unit_count']} | "
                f"{_format_number(cost['replan_latency_ms']['total'])} |"
            )
        ledger = baseline.get("ledger", {})
        lines.extend([
            "", f"重规划账本已提供：{ledger.get('provided')}；匹配单元："
            f"{ledger.get('matched_unit_count')}/{ledger.get('expected_triggered_unit_count')}。", "",
        ])
        lines.extend(f"- {item}" for item in baseline.get("interpretation_boundary", []))
        lines.extend(["", "## 解释边界", ""])
        lines.extend(
            f"- {item}" for item in analysis.get("interpretation_boundary", [])
        )
        lines.append("")
        return {"status": "success", "markdown": "\n".join(lines)}
    except Exception as exc:
        logger.error("渲染六条件报告失败: %s", exc, exc_info=True)
        return _error_result(
            f"渲染六条件报告失败: {exc}",
            "FACTORIAL_MARKDOWN_EXCEPTION",
        )


def write_factorial_artifacts(
    analysis: Any,
    *,
    output_prefix: Any,
    overwrite: bool = False,
) -> dict[str, Any]:
    """原子写出六条件 JSON 与 Markdown，默认拒绝覆盖。"""
    try:
        if not isinstance(analysis, dict) or analysis.get("status") != "success":
            return _error_result("只有成功分析可以写出", "INVALID_FACTORIAL_OUTPUT")
        prefix = Path(str(output_prefix or "")).expanduser()
        json_path = prefix.with_name(f"{prefix.name}.json")
        markdown_path = prefix.with_name(f"{prefix.name}.md")
        if not overwrite and (json_path.exists() or markdown_path.exists()):
            return _error_result("分析输出已存在，请使用新前缀", "FACTORIAL_OUTPUT_EXISTS")
        rendered = render_factorial_markdown(analysis)
        if rendered.get("status") != "success":
            return rendered
        json_write = _atomic_write_text(
            json_path,
            json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
            overwrite=overwrite,
        )
        if json_write.get("status") != "success":
            return json_write
        markdown_write = _atomic_write_text(
            markdown_path,
            rendered["markdown"],
            overwrite=overwrite,
        )
        if markdown_write.get("status") != "success":
            return markdown_write
        return {
            "status": "success",
            "json_path": json_write["path"],
            "markdown_path": markdown_write["path"],
        }
    except Exception as exc:
        logger.error("写出六条件分析失败: %s", exc, exc_info=True)
        return _error_result(
            f"写出六条件分析失败: {exc}",
            "FACTORIAL_OUTPUT_EXCEPTION",
        )


def analyze_factorial_trace_file(
    trace_path: Any,
    *,
    manifest_path: Any = None,
    failure_path: Any = None,
    replan_ledger_path: Any = None,
    output_prefix: Any = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_ANALYSIS_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    """加载六条件 Trace、Manifest 和失败账本并导出分析。"""
    try:
        trace = Path(str(trace_path or "")).expanduser()
        loaded = load_experiment_traces(trace)
        if loaded.get("status") != "success":
            return loaded
        resolved_manifest = (
            Path(str(manifest_path)).expanduser()
            if manifest_path is not None
            else trace.with_suffix(".manifest.json")
        )
        manifest_result = _load_manifest(resolved_manifest)
        if manifest_result.get("status") != "success":
            return manifest_result
        manifest = manifest_result["manifest"]
        resolved_failure = (
            Path(str(failure_path)).expanduser()
            if failure_path is not None
            else trace.with_name(f"{trace.stem}.failures.jsonl")
        )
        failure_summary = _load_failure_summary(resolved_failure, manifest)
        if failure_summary.get("status") != "success":
            return failure_summary
        # 只读取调用方显式指定的账本，不猜测版本，也不读取任何数据集题面。
        ledger = None
        ledger_sha256 = None
        if replan_ledger_path is not None:
            ledger_path = Path(str(replan_ledger_path)).expanduser()
            try:
                ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
                ledger_sha256 = _sha256_file(ledger_path)
            except (OSError, UnicodeError, ValueError):
                return _error_result("无法读取有效重规划账本", "INVALID_BASELINE_LEDGER_FILE")
        analysis = analyze_factorial_records(
            loaded["records"],
            manifest=manifest,
            failure_summary=failure_summary,
            trace_sha256=_sha256_file(trace),
            replan_ledger=ledger,
            n_resamples=n_resamples,
            seed=seed,
        )
        if analysis.get("status") != "success":
            return analysis
        analysis["input"]["manifest_sha256"] = _sha256_file(resolved_manifest)
        analysis["input"]["replan_ledger_sha256"] = ledger_sha256
        analysis["input"]["analysis_code_sha256"] = {
            name: _sha256_file(Path(__file__).with_name(name))
            for name in ("analyze_factorial_biotooleval.py", "baseline_audit.py", "evaluation_metrics.py")
        }
        prefix = (
            Path(str(output_prefix)).expanduser()
            if output_prefix is not None
            else trace.parent / f"{trace.stem}.factorial-analysis"
        )
        written = write_factorial_artifacts(
            analysis,
            output_prefix=prefix,
            overwrite=overwrite,
        )
        if written.get("status") != "success":
            return written
        return {
            "status": "success",
            "record_count": loaded["record_count"],
            "question_count": analysis["question_count"],
            "type_sensitive_question_count": analysis["type_sensitive_question_count"],
            "json_path": written["json_path"],
            "markdown_path": written["markdown_path"],
            "completeness": analysis["completeness"],
        }
    except Exception as exc:
        logger.error("分析六条件 Trace 文件失败: %s", exc, exc_info=True)
        return _error_result(
            f"分析六条件 Trace 文件失败: {exc}",
            "FACTORIAL_TRACE_FILE_EXCEPTION",
        )


def _build_cli_parser() -> argparse.ArgumentParser:
    """构建六条件分析命令行。"""
    parser = argparse.ArgumentParser(
        description="分析 BioToolEval E0–E3 与 S1/L1 配对实验 Trace"
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--failures")
    parser.add_argument("--replan-ledger", help="显式指定同实验的 L1 重规划 JSONL 账本")
    parser.add_argument("--output-prefix")
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_ANALYSIS_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口，只打印脱敏摘要。"""
    try:
        args = _build_cli_parser().parse_args(argv)
        result = analyze_factorial_trace_file(
            args.trace,
            manifest_path=args.manifest,
            failure_path=args.failures,
            replan_ledger_path=args.replan_ledger,
            output_prefix=args.output_prefix,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        safe_output = {
            key: result.get(key)
            for key in (
                "status",
                "record_count",
                "question_count",
                "type_sensitive_question_count",
                "json_path",
                "markdown_path",
                "completeness",
                "message",
                "issues",
            )
        }
        print(json.dumps(safe_output, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "success" else 1
    except Exception as exc:
        logger.error("六条件分析 CLI 失败: %s", exc, exc_info=True)
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
