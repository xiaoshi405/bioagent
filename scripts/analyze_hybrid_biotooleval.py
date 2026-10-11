# -*- coding: utf-8 -*-
"""分析 E3/L1/F1 分层恢复对照，不改写既有 H1–H3。

修改记录：
- 2026-09-05：新增三条件完整性、共享原始计划、逐轮配对、题内重复聚合、
  F1 干预后 Type-II 分类和额外重规划成本审计。

设计边界：
1. 仅比较确定性修复 E3、一次 LLM 重规划 L1 与分层恢复 F1；
2. 同题同模型同轮次必须共享原始计划，缺项或漂移时失败关闭；
3. 多轮结果先在题内求均值，再以题目为独立单位做配对 bootstrap；
4. 这是 H1–H3 之外的冻结次要分析，不在看到 Test 后新增显著性假设；
5. 不读取题面、序列或 Gold 正文，只处理脱敏 Trace 与 Manifest。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from scripts.baseline_audit import classify_type_ii_execution, extra_replan_cost_value
from scripts.evaluation_metrics import (
    behavior_aware_task_success,
    compare_conditions,
    compute_condition_metrics,
    paired_bootstrap_difference,
)
from scripts.evaluation_trace import validate_experiment_trace


ANALYSIS_SCHEMA_VERSION = "hybrid-recovery-1.0"
HYBRID_CONDITIONS = ("E3", "L1", "F1")
QUESTION_METRICS = (
    "final_answer_accuracy",
    "behavior_aware_task_success",
    "executable_plan_rate",
    "safe_rejection_rate",
    "total_latency_ms",
    "extra_llm_calls",
    "extra_llm_total_tokens",
)


def _error(message: str, code: str, **details: Any) -> dict[str, Any]:
    """返回统一错误结构，避免分析异常导致主进程崩溃。"""
    issue = {"code": code, "message": message}
    issue.update(details)
    return {"status": "error", "message": message, "issues": [issue]}


def _load_json(path: Path) -> Any:
    """读取 UTF-8 JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 并拒绝非对象或空输入。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"第 {line_number} 行不是 JSON 对象")
            rows.append(row)
    if not rows:
        raise ValueError("混合恢复 Trace 不能为空")
    return rows


def _sha256(path: Path) -> str:
    """计算原始证据文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan_identity(record: dict[str, Any]) -> str | None:
    """统一提取三条件共享的原始计划样本 ID。"""
    plan = record.get("observation", {}).get("plan", {})
    value = plan.get("original_plan_sample_id") or plan.get("plan_sample_id")
    return value if isinstance(value, str) and value else None


def _question_signature(record: dict[str, Any]) -> str:
    """用脱敏元数据和 Gold 结构检测跨条件题目漂移。"""
    payload = {
        "split": record.get("split"),
        "category": record.get("category"),
        "difficulty": record.get("difficulty"),
        "input_type": record.get("input_type"),
        "gold": record.get("gold"),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _numeric(value: Any) -> float | None:
    """只接受非负有限数，布尔值不冒充成本。"""
    if type(value) not in (int, float) or value < 0 or not math.isfinite(value):
        return None
    return float(value)


def _extract_metric(record: dict[str, Any], metric: str) -> float | None:
    """提取题目/轮次级指标；不适用和未知标签保持 None。"""
    try:
        expected = record.get("gold", {}).get("expected_behavior")
        if metric == "final_answer_accuracy":
            value = record.get("evaluation", {}).get("final_answer_correct")
            return float(value) if isinstance(value, bool) else None
        if metric == "behavior_aware_task_success":
            value = behavior_aware_task_success(record)
            return float(value) if isinstance(value, bool) else None
        if metric == "executable_plan_rate":
            if expected == "reject":
                return None
            value = record.get("observation", {}).get("execution", {}).get("completed")
            return float(value) if isinstance(value, bool) else None
        if metric == "safe_rejection_rate":
            if expected != "reject":
                return None
            value = record.get("evaluation", {}).get("correct_rejection")
            return float(value) if isinstance(value, bool) else None
        if metric == "total_latency_ms":
            return _numeric(
                record.get("observation", {}).get("latency", {}).get("total_ms")
            )
        if metric in {"extra_llm_calls", "extra_llm_total_tokens"}:
            return _numeric(extra_replan_cost_value(record, metric))
        return None
    except Exception:
        return None


def _audit(records: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    """审计完整三条件、控制量、共享计划和题目元数据。"""
    try:
        if not isinstance(manifest, dict):
            return _error("必须提供实验 Manifest", "MISSING_HYBRID_MANIFEST")
        observed = {str(row.get("condition")) for row in records}
        declared = set((manifest.get("design") or {}).get("hybrid_conditions") or [])
        required = set(HYBRID_CONDITIONS)
        if observed != required or declared != required:
            return _error(
                "混合恢复分析必须完整包含 E3/L1/F1",
                "INCOMPLETE_HYBRID_CONDITIONS",
                observed=sorted(observed),
                declared=sorted(declared),
            )

        seen: set[tuple[str, str, int, str]] = set()
        groups: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
        signatures: dict[str, str] = {}
        for index, row in enumerate(records):
            validation = validate_experiment_trace(row)
            if validation.get("status") != "success":
                return _error(
                    f"第 {index} 条 Trace 未通过 Schema 校验",
                    "INVALID_HYBRID_TRACE",
                    validation=validation,
                )
            question_id = str(row.get("question_id") or "")
            model = str(row.get("context", {}).get("model") or "")
            repeat_index = row.get("repeat_index")
            condition = str(row.get("condition") or "")
            if not question_id or not model or type(repeat_index) is not int:
                return _error("Trace 配对键不完整", "INVALID_HYBRID_PAIR_KEY")
            unit = (question_id, model, repeat_index, condition)
            if unit in seen:
                return _error("存在重复实验单元", "DUPLICATE_HYBRID_UNIT")
            seen.add(unit)
            group_key = (question_id, model, repeat_index)
            groups[group_key][condition] = row
            signature = _question_signature(row)
            if question_id in signatures and signatures[question_id] != signature:
                return _error("同题脱敏元数据发生漂移", "HYBRID_METADATA_DRIFT")
            signatures[question_id] = signature

        incomplete = [
            list(key) for key, group in groups.items()
            if set(group) != required
        ]
        if incomplete:
            return _error(
                "存在不完整 E3/L1/F1 配对组",
                "INCOMPLETE_HYBRID_PAIR_GROUP",
                examples=incomplete[:5],
                count=len(incomplete),
            )
        drifted = []
        for key, group in groups.items():
            identities = {_plan_identity(row) for row in group.values()}
            if None in identities or len(identities) != 1:
                drifted.append(list(key))
        if drifted:
            return _error(
                "E3/L1/F1 未共享同一原始计划",
                "HYBRID_PLAN_PAIRING_FAILED",
                examples=drifted[:5],
                count=len(drifted),
            )

        controls = {}
        for field, getter in {
            "model": lambda row: row.get("context", {}).get("model"),
            "temperature": lambda row: row.get("context", {}).get("temperature"),
            "seed": lambda row: row.get("context", {}).get("seed"),
            "git_commit": lambda row: row.get("context", {}).get("git_commit"),
            "dataset_version": lambda row: row.get("context", {}).get("dataset_version"),
            "python_version": lambda row: row.get("context", {}).get("python_version"),
        }.items():
            values = sorted({str(getter(row)) for row in records})
            controls[field] = values
            development_cross_version = (
                manifest.get("development_run") is True
                and field in {"git_commit", "python_version"}
            )
            if len(values) != 1 and not development_cross_version:
                return _error(
                    f"控制量 {field} 跨条件不一致",
                    "HYBRID_CONTROL_DRIFT",
                    field=field,
                    values=values,
                )

        repeat_indices = sorted({row["repeat_index"] for row in records})
        declared_repeats = (manifest.get("design") or {}).get("repeats")
        if type(declared_repeats) is not int or declared_repeats < 1:
            return _error("Manifest repeats 非法", "INVALID_HYBRID_REPEATS")
        if repeat_indices != list(range(declared_repeats)):
            return _error(
                "实际轮次与 Manifest 不一致",
                "HYBRID_REPEAT_MISMATCH",
                observed=repeat_indices,
                declared=declared_repeats,
            )
        return {
            "status": "success",
            "question_count": len(signatures),
            "repeat_indices": repeat_indices,
            "paired_group_count": len(groups),
            "controls": controls,
        }
    except Exception as exc:
        return _error(
            f"混合恢复审计异常：{exc}",
            "HYBRID_AUDIT_EXCEPTION",
            exception_type=type(exc).__name__,
        )


def _question_level_pair(
    records: list[dict[str, Any]],
    baseline: str,
    treatment: str,
    metric: str,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """先在题内聚合重复，再计算 treatment-baseline 配对效应。"""
    grouped: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in records:
        condition = str(row.get("condition"))
        if condition in {baseline, treatment}:
            grouped[str(row["question_id"])][condition][int(row["repeat_index"])] = row

    left_values: list[float] = []
    right_values: list[float] = []
    paired_repeat_count = 0
    null_repeat_pair_count = 0
    for question_id in sorted(grouped):
        left = grouped[question_id].get(baseline, {})
        right = grouped[question_id].get(treatment, {})
        local_left: list[float] = []
        local_right: list[float] = []
        for repeat_index in sorted(set(left) & set(right)):
            left_value = _extract_metric(left[repeat_index], metric)
            right_value = _extract_metric(right[repeat_index], metric)
            if left_value is None or right_value is None:
                null_repeat_pair_count += 1
                continue
            local_left.append(left_value)
            local_right.append(right_value)
            paired_repeat_count += 1
        if local_left:
            left_values.append(sum(local_left) / len(local_left))
            right_values.append(sum(local_right) / len(local_right))

    common = {
        "metric": metric,
        "comparison": f"{treatment}_vs_{baseline}",
        "statistical_unit": "question",
        "repeat_handling": "mean_within_question_before_bootstrap",
        "question_count": len(left_values),
        "total_question_count": len(grouped),
        "question_coverage": len(left_values) / len(grouped) if grouped else None,
        "paired_repeat_count": paired_repeat_count,
        "null_repeat_pair_count": null_repeat_pair_count,
    }
    if not left_values:
        return {**common, "status": "not_estimable", "mean_difference": None, "ci_95": None}
    return {
        **paired_bootstrap_difference(
            left_values,
            right_values,
            n_resamples=n_resamples,
            seed=seed,
        ),
        **common,
    }


def _f1_recovery_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 F1 干预后类型状态与触发单元成本。"""
    f1_rows = [row for row in records if row.get("condition") == "F1"]
    classes = Counter(classify_type_ii_execution(row) for row in f1_rows)
    triggered = [
        row for row in f1_rows
        if row.get("observation", {}).get("plan", {}).get("replan_triggered") is True
    ]
    costs = {}
    for field in (
        "extra_llm_calls",
        "extra_llm_input_tokens",
        "extra_llm_output_tokens",
        "extra_llm_total_tokens",
        "replan_latency_ms",
    ):
        values = [extra_replan_cost_value(row, field) for row in triggered]
        costs[field] = {
            "triggered_count": len(triggered),
            "known_count": sum(value is not None for value in values),
            "total": sum(values) if values and all(value is not None for value in values) else None,
        }
    return {
        "execution_classes": dict(sorted(classes.items())),
        "replan_triggered_count": len(triggered),
        "replan_applied_count": sum(
            row["observation"]["plan"].get("replan_applied") is True
            for row in triggered
        ),
        "post_intervention_unresolved_execution_count": classes[
            "executed_with_unresolved_type_ii"
        ],
        "triggered_unit_cost": costs,
    }


def analyze_hybrid_records(
    records: Any,
    *,
    manifest: dict[str, Any],
    n_resamples: int = 10000,
    seed: int = 20260905,
) -> dict[str, Any]:
    """生成 E3/L1/F1 的逐轮与题目级冻结分析。"""
    try:
        if not isinstance(records, list) or not records:
            return _error("Trace 必须是非空列表", "EMPTY_HYBRID_INPUT")
        if type(n_resamples) is not int or n_resamples < 100:
            return _error("bootstrap 次数至少为 100", "INVALID_HYBRID_RESAMPLES")
        audit = _audit(records, manifest)
        if audit.get("status") != "success":
            return audit

        per_repeat: dict[str, Any] = {}
        for repeat_index in audit["repeat_indices"]:
            repeat_rows = [row for row in records if row["repeat_index"] == repeat_index]
            conditions = {
                condition: compute_condition_metrics(
                    [row for row in repeat_rows if row["condition"] == condition]
                )
                for condition in HYBRID_CONDITIONS
            }
            e3_comparison = compare_conditions(
                repeat_rows,
                baseline_condition="E3",
                n_resamples=n_resamples,
                seed=seed + repeat_index * 100,
            )
            l1_comparison = compare_conditions(
                [row for row in repeat_rows if row["condition"] in {"L1", "F1"}],
                baseline_condition="L1",
                n_resamples=n_resamples,
                seed=seed + repeat_index * 100 + 50,
            )
            if any(result.get("status") != "success" for result in conditions.values()):
                return _error("逐轮条件指标失败", "HYBRID_REPEAT_METRIC_FAILED")
            if e3_comparison.get("status") != "success" or l1_comparison.get("status") != "success":
                return _error("逐轮配对比较失败", "HYBRID_REPEAT_COMPARISON_FAILED")
            per_repeat[str(repeat_index)] = {
                "condition_metrics": conditions,
                "E3_baseline_comparisons": e3_comparison,
                "L1_baseline_comparison": l1_comparison,
            }

        question_level: dict[str, Any] = {}
        counter = 0
        for baseline, treatment in (("E3", "F1"), ("L1", "F1")):
            name = f"{treatment}_vs_{baseline}"
            question_level[name] = {}
            for metric in QUESTION_METRICS:
                question_level[name][metric] = _question_level_pair(
                    records,
                    baseline,
                    treatment,
                    metric,
                    n_resamples=n_resamples,
                    seed=seed + 1000 + counter,
                )
                counter += 1

        return {
            "status": "success",
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "evidence_scope": (
                "DEVELOPMENT_ONLY"
                if manifest.get("development_run") is True
                else "FORMAL_RESULTS_REQUIRE_HUMAN_INTERPRETATION"
            ),
            "analysis_config": {
                "conditions": list(HYBRID_CONDITIONS),
                "comparisons": ["F1_vs_E3", "F1_vs_L1"],
                "bootstrap_resamples": n_resamples,
                "seed": seed,
                "statistical_unit": "question",
                "repeat_policy": "mean_within_question_before_bootstrap",
                "hypothesis_role": "secondary_frozen_comparison_outside_H1_H2_H3",
            },
            "audit": audit,
            "per_repeat": per_repeat,
            "question_level": question_level,
            "f1_recovery_audit": _f1_recovery_audit(records),
            "interpretation_boundaries": [
                "本分析不修改 factorial H1-H3 的比较、分母或 Holm family。",
                "未知人工标签按 null 排除并报告覆盖率，不补成正确或错误。",
                "类型有效纠正后执行不等于答案必然正确。",
                "F1 与 L1 的优势若置信区间跨 0，只能报告未检出稳定差异。",
                "额外重规划成本不含共享首轮规划、回答生成或工具执行成本。",
            ],
        }
    except Exception as exc:
        return _error(
            f"混合恢复分析异常：{exc}",
            "HYBRID_ANALYSIS_EXCEPTION",
            exception_type=type(exc).__name__,
        )


def main() -> int:
    """执行只读分析并独占创建 JSON，禁止覆盖既有结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    records = [
        row for row in _load_jsonl(args.trace)
        if row.get("condition") in set(HYBRID_CONDITIONS)
    ]
    manifest = _load_json(args.manifest)
    result = analyze_hybrid_records(
        records,
        manifest=manifest,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    if result.get("status") != "success":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    result["input"] = {
        "trace_path": str(args.trace),
        "trace_sha256": _sha256(args.trace),
        "manifest_path": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "success", "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
