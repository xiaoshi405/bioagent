# -*- coding: utf-8 -*-
"""六条件 baseline 的干预后诊断与额外重规划成本审计。

修改记录：
- 2026-09-05：只要 Trace 同时证明未触发重规划且额外调用数为零，缺省 token
  即为结构性零；不再把合法的前置拒绝误记为成本缺失。
- 2026-09-03：区分未纠正错误与纠正后执行，逐单元核对 L1 账本；
  缺失用量保留未知，不改写 Trace、旧 H1–H3 或人工标签。

本模块只衡量额外重规划，不代表整场实验账单；类型有效不等于答案正确。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from scripts.evaluation_metrics import _TYPE_II_CODES, _error_result


AUDIT_SCHEMA_VERSION = "baseline-audit-1.0"
# 协议常量，不绑定模型、实验路径、价格或具体题目。
NON_REPLAN_CONDITIONS = frozenset({"E0", "E1", "E2", "E3"})
COST_FIELDS = (
    "extra_llm_calls", "extra_llm_input_tokens", "extra_llm_output_tokens",
    "extra_llm_total_tokens", "replan_latency_ms",
)
TOKEN_FIELDS = {
    "extra_llm_input_tokens": "input_tokens",
    "extra_llm_output_tokens": "output_tokens",
    "extra_llm_total_tokens": "total_tokens",
}
LATENCY_REL_TOLERANCE = 1e-9
LATENCY_ABS_TOLERANCE_MS = 1e-6


def _number(value: Any, *, integer: bool = False) -> bool:
    """排除布尔、负数及非有限值，调用数和 token 必须为整数。"""
    return (
        type(value) in ((int,) if integer else (int, float))
        and value >= 0 and math.isfinite(value)
    )


def extra_replan_cost_value(record: Any, field: str) -> int | float | None:
    """仅凭显式用量或协议确定的未调用状态取值，缺失不冒充零。"""
    try:
        if field not in COST_FIELDS:
            return None
        plan = record["observation"]["plan"]
        value = plan.get(field)
        if value is not None:
            return value if _number(value, integer=field != "replan_latency_ms") else None
        if record.get("condition") in NON_REPLAN_CONDITIONS:
            return 0
        if (
            plan.get("replan_triggered") is False
            and type(plan.get("extra_llm_calls")) is int
            and plan["extra_llm_calls"] == 0
        ):
            return 0
        return None
    except (KeyError, TypeError, AttributeError, ValueError, OverflowError):
        return None


def classify_type_ii_execution(record: Any) -> str:
    """按初始错误、干预后类型和实际调用分类；证据矛盾时保留 unknown。"""
    try:
        plan = record["observation"]["plan"]
        before = plan.get("issue_codes_before")
        if not isinstance(before, list) or not all(isinstance(x, str) for x in before):
            return "unknown_initial_state"
        if not set(before) & _TYPE_II_CODES:
            return "no_initial_type_ii"
        attempted = record["observation"]["execution"].get("attempted_steps")
        if not _number(attempted, integer=True):
            return "unknown_after_initial_type_ii"
        if attempted == 0:
            if record["observation"]["execution"].get("completed") is not False:
                return "unknown_after_initial_type_ii"
            return "blocked_without_tool_attempt"
        after = plan.get("issue_codes_after")
        if not isinstance(after, list) or not all(isinstance(x, str) for x in after):
            return "unknown_after_initial_type_ii"
        if set(after) & _TYPE_II_CODES and plan.get("type_valid_after") is False:
            return "executed_with_unresolved_type_ii"
        if not after and plan.get("type_valid_after") is True and (
            (plan.get("replan_applied") is True and plan.get("replan_triggered") is True)
            or (plan.get("repair_applied") is True and plan.get("repair_attempted") is True)
        ):
            return "executed_after_type_valid_correction"
        return "unknown_after_initial_type_ii"
    except (KeyError, TypeError, AttributeError, ValueError, OverflowError):
        return "unknown_initial_state"


def _summary(values: list[int | float | None]) -> dict[str, Any]:
    """同时报告已知小计、完整总数和覆盖率，禁止缺失值缩小分母后冒充总数。"""
    known = [value for value in values if value is not None]
    complete = len(known) == len(values)
    return {
        "unit_count": len(values), "known_count": len(known),
        "missing_count": len(values) - len(known),
        "coverage": len(known) / len(values) if values else None,
        "known_sum": sum(known), "total": sum(known) if complete else None,
        "mean_per_unit": sum(known) / len(values) if complete and values else None,
    }


def _key(record: dict[str, Any], *, ledger: bool = False) -> tuple[str, str, int]:
    """用题目、模型、轮次定位单元；禁止空键、布尔轮次或静默跨轮次拼接。"""
    model = record.get("model") if ledger else record.get("context", {}).get("model")
    question, repeat = record.get("question_id"), record.get("repeat_index")
    if not isinstance(question, str) or not question or not isinstance(model, str) or not model:
        raise ValueError("单元缺少 question_id/model")
    if not _number(repeat, integer=True):
        raise ValueError("repeat_index 必须为非负整数")
    return question, model, repeat


def _ledger_cost(row: dict[str, Any]) -> dict[str, int | float | None]:
    """汇总所有尝试（包括失败）；任一次 usage 缺失则该维度总量未知。"""
    attempts = row.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("重规划账本必须保留尝试记录")
    costs: dict[str, int | float | None] = {"extra_llm_calls": len(attempts)}
    for index, attempt in enumerate(attempts, 1):
        if not isinstance(attempt, dict) or type(attempt.get("attempt_index")) is not int or attempt["attempt_index"] != index:
            raise ValueError("尝试编号必须连续且唯一")
        usage = attempt.get("usage")
        if usage is not None and not isinstance(usage, dict):
            raise ValueError("usage 必须为对象或 null")
        usage = usage or {}
        for name in TOKEN_FIELDS.values():
            value = usage.get(name)
            if value is not None and not _number(value, integer=True):
                raise ValueError("账本 token 必须为非负整数或 null")
        token_values = [usage.get(name) for name in TOKEN_FIELDS.values()]
        if all(value is not None for value in token_values) and token_values[0] + token_values[1] != token_values[2]:
            raise ValueError("账本输入与输出 token 之和不等于总量")
        latency = attempt.get("latency_ms")
        if latency is not None and not _number(latency):
            raise ValueError("账本耗时必须非负且有限")
    for target, source in TOKEN_FIELDS.items():
        values = [(attempt.get("usage") or {}).get(source) for attempt in attempts]
        costs[target] = sum(values) if all(value is not None for value in values) else None
    latencies = [attempt.get("latency_ms") for attempt in attempts]
    costs["replan_latency_ms"] = sum(latencies) if all(value is not None for value in latencies) else None
    return costs


def audit_baselines(
    records: Any, *, experiment_id: str, replan_ledger: Any = None,
) -> dict[str, Any]:
    """只读审计干预后诊断、成本完整性与可选 L1 账本；身份冲突失败关闭。"""
    try:
        if not isinstance(records, list) or not records or not isinstance(experiment_id, str) or not experiment_id:
            return _error_result("审计需要非空 Trace 和实验 ID", "INVALID_BASELINE_AUDIT_INPUT")
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        triggered: dict[tuple[str, str, int], dict[str, Any]] = {}
        seen = set()
        for record in records:
            condition = record.get("condition")
            if condition not in NON_REPLAN_CONDITIONS | {"S1", "L1"}:
                raise ValueError("审计仅适用六条件协议")
            key = _key(record)
            if (condition, key) in seen:
                raise ValueError("Trace 单元重复")
            seen.add((condition, key))
            plan = record["observation"]["plan"]
            for field in COST_FIELDS:
                value = plan.get(field)
                if value is not None and not _number(value, integer=field != "replan_latency_ms"):
                    raise ValueError(f"非法成本字段：{field}")
            tokens = [plan.get(field) for field in TOKEN_FIELDS]
            if all(value is not None for value in tokens) and tokens[0] + tokens[1] != tokens[2]:
                raise ValueError("Trace 输入与输出 token 之和不等于总量")
            if condition in NON_REPLAN_CONDITIONS | {"S1"} and plan.get("replan_triggered") is True:
                raise ValueError("非重规划条件出现额外重规划")
            if condition in NON_REPLAN_CONDITIONS or plan.get("replan_triggered") is False:
                if any(plan.get(field) not in (None, 0) for field in COST_FIELDS):
                    raise ValueError("未重规划状态与非零成本冲突")
            if condition == "L1" and plan.get("replan_triggered") is True:
                if plan.get("extra_llm_calls") == 0:
                    raise ValueError("已触发重规划却声明零次调用")
                triggered[key] = record
            groups[condition].append(record)

        units = []
        checked_ledger_costs = []
        if replan_ledger is not None:
            if not isinstance(replan_ledger, list):
                raise ValueError("重规划账本必须为记录列表")
            ledger_seen = set()
            for row in replan_ledger:
                key = _key(row, ledger=True)
                if row.get("experiment_id") != f"{experiment_id}-l1":
                    return _error_result("重规划账本来自不同实验", "BASELINE_LEDGER_IDENTITY_MISMATCH")
                if key in ledger_seen or key not in triggered:
                    return _error_result("账本有重复或无对应触发单元", "BASELINE_LEDGER_UNIT_MISMATCH")
                ledger_seen.add(key)
                record = triggered[key]
                plan = record["observation"]["plan"]
                captured = row.get("record_status") == "captured"
                if row.get("record_status") not in {"captured", "capture_failed"}:
                    raise ValueError("账本捕获状态未知")
                if (captured and (not row.get("plan_sample_id") or plan.get("replan_applied") is not True)) or (
                    not captured and plan.get("replan_applied") is not False
                ) or row.get("plan_sample_id") != plan.get("replan_plan_sample_id"):
                    return _error_result("账本计划 ID/应用状态与 Trace 不符", "BASELINE_LEDGER_PLAN_MISMATCH")
                costs = _ledger_cost(row)
                for field, value in costs.items():
                    trace_value = extra_replan_cost_value(record, field)
                    if value is None or trace_value is None:
                        continue
                    equal = (
                        math.isclose(value, trace_value, rel_tol=LATENCY_REL_TOLERANCE,
                                     abs_tol=LATENCY_ABS_TOLERANCE_MS)
                        if field == "replan_latency_ms" else value == trace_value
                    )
                    if not equal:
                        return _error_result("Trace 与账本成本冲突", "BASELINE_LEDGER_COST_MISMATCH", details={"field": field, "question_id": key[0]})
                checked_ledger_costs.append(costs)
                units.append({
                    "question_id": key[0], "model": key[1], "repeat_index": key[2],
                    "replan_plan_sample_id": row.get("plan_sample_id"),
                    "ledger_status": row["record_status"], "ledger_cost": costs,
                    "trace_cost": {field: extra_replan_cost_value(record, field) for field in COST_FIELDS},
                    "execution_class": classify_type_ii_execution(record),
                })
            if set(triggered) != ledger_seen:
                return _error_result("账本缺少已触发的重规划单元", "BASELINE_LEDGER_MISSING_UNIT")

        conditions = {}
        for condition, rows in sorted(groups.items()):
            classes = Counter(classify_type_ii_execution(row) for row in rows)
            initial = len(rows) - classes["no_initial_type_ii"] - classes["unknown_initial_state"]
            known = initial - classes["unknown_after_initial_type_ii"]
            conditions[condition] = {
                "record_count": len(rows), "execution_classes": dict(classes),
                "initial_type_ii_count": initial,
                "post_intervention_unresolved_execution_rate": {
                    "numerator": classes["executed_with_unresolved_type_ii"],
                    "denominator": known,
                    "rate": classes["executed_with_unresolved_type_ii"] / known if known else None,
                    "coverage": known / initial if initial else None,
                },
                "extra_replan_cost": {
                    field: _summary([extra_replan_cost_value(row, field) for row in rows])
                    for field in COST_FIELDS
                },
            }
        return {
            "status": "success", "schema_version": AUDIT_SCHEMA_VERSION,
            "by_condition": conditions,
            "ledger": {
                "provided": replan_ledger is not None,
                "expected_triggered_unit_count": len(triggered), "matched_unit_count": len(units),
                "cost_on_triggered_units": {
                    field: _summary([cost[field] for cost in checked_ledger_costs])
                    for field in COST_FIELDS
                } if replan_ledger is not None else None,
                "units": units,
            },
            "interpretation_boundary": [
                "干预后诊断为补充描述，不替换原 H1/H2/H3 或最终答案人工评分。",
                "类型有效的纠正后执行不等于语义正确、任务完成或所有安全条件满足。",
                "成本只含额外重规划；未触发且调用数为零时 token 才可解释为结构性零。",
                "账本和 Trace 分别汇总，缺失成本不补写 Trace；未知总量返回 null。",
                "当前运行器 total_ms 已包含 L1 重规划等待，不可重复相加；不含此前冻结的共享首轮规划。",
            ],
        }
    except Exception as exc:
        return _error_result(f"Baseline 审计失败：{exc}", "INVALID_BASELINE_AUDIT_INPUT")
