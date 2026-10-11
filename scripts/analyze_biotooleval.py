# -*- coding: utf-8 -*-
"""
BioToolEval B0/B1/B2实验分析入口。

修改记录：
- 2026-08-27：历史B0-B2分析显式绑定旧条件集合，防止新增E条件改变
  已冻结分析口径。
- 2026-08-10：主要假设升级为不安全Type-II执行、真实修复恢复和行为感知
  任务成功；删除H2回退指标，并显式报告not_estimable。
- 2026-08-09：主要假设改为H1的B1对B0未解决Type-II比较、H2的
  repair-eligible子集B2对B1恢复比较和H3的B2对B0最终答案比较；
  H2分析尚未实现时显式保持null，使主要假设族标记为不完整。
- 2026-08-09：把失败账本作为正式分析输入，报告失败类别、条件失败率和
  已解释/未解释缺失单元，避免只对成功形成Trace的样本计算而产生幸存者偏差。
- 2026-08-08：完整性统计优先使用Manifest中预注册的question_ids子集，
  避免开发子集实验被错误地按整个数据集题数计算缺失Trace。
- 2026-08-03：新增控制变量审计、逐轮指标、题目级重复聚合、主要假设
  Holm校正，以及JSON/Markdown可复现报告导出。

统计边界：
1. 单轮二元结果使用精确McNemar；
2. 多次重复先在同一题内求均值，再以题目为单位bootstrap，禁止扩充样本量；
3. null保持未评分并报告丢失配对；
4. 本模块只生成描述性结果，不自动宣布假设成立或期刊Go/No-Go。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.core.logger import get_logger
from scripts.evaluation_metrics import (
    behavior_aware_task_success,
    compare_conditions,
    compute_condition_metrics,
    paired_bootstrap_difference,
    repair_opportunity,
)
from scripts.evaluation_trace import (
    LEGACY_EXPERIMENT_CONDITIONS,
    load_experiment_traces,
    validate_experiment_trace,
)


logger = get_logger("BioToolEvalAnalysis", "biotooleval_analysis.log")

ANALYSIS_SCHEMA_VERSION = "1.2"
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_ANALYSIS_SEED = 20260803
PRIMARY_HYPOTHESES = (
    {
        "id": "H1",
        "comparison": "B1_vs_B0",
        "metric": "unsafe_type_ii_execution_rate",
        "description": "类型敏感题上B1相对B0的不安全Type-II执行率差异",
    },
    {
        "id": "H2",
        "comparison": "B2_vs_B1",
        "metric": "repair_recovery_rate",
        "description": "真实修复机会子集上B2相对B1的计划恢复率差异",
    },
    {
        "id": "H3",
        "comparison": "B2_vs_B0",
        "metric": "behavior_aware_task_success",
        "description": "类型敏感题上B2相对B0的行为感知任务成功率差异",
    },
)


def _error_result(
    message: str,
    code: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造分析模块统一错误结构。"""
    try:
        issue = {"code": str(code), "message": str(message)}
        if isinstance(details, dict):
            issue.update(details)
        return {
            "status": "error",
            "message": str(message),
            "issues": [issue],
        }
    except Exception as exc:
        logger.error("构造分析错误结构失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": str(exc),
            "issues": [],
        }


def _sha256_file(path: Path) -> str:
    """计算文件SHA-256；失败返回空字符串并记录日志。"""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()
    except Exception as exc:
        logger.error("计算分析输入哈希失败: %s", exc, exc_info=True)
        return ""


def _load_manifest(path: Any) -> dict[str, Any]:
    """加载实验清单JSON并执行最小结构检查。"""
    try:
        manifest_path = Path(str(path or "")).expanduser()
        if not manifest_path.is_file():
            return _error_result(
                "实验Manifest不存在",
                "MANIFEST_NOT_FOUND",
                details={"path": str(manifest_path)},
            )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return _error_result(
                f"实验Manifest不是合法JSON: {exc}",
                "INVALID_MANIFEST_JSON",
            )
        if not isinstance(payload, dict):
            return _error_result(
                "实验Manifest根节点必须是对象",
                "INVALID_MANIFEST_ROOT",
            )
        required_sections = {"dataset", "design", "environment"}
        missing = sorted(required_sections - set(payload))
        if missing:
            return _error_result(
                "实验Manifest缺少必要区段",
                "INCOMPLETE_MANIFEST",
                details={"missing": missing},
            )
        return {
            "status": "success",
            "path": str(manifest_path.resolve()),
            "manifest": payload,
        }
    except Exception as exc:
        logger.error("加载实验Manifest失败: %s", exc, exc_info=True)
        return _error_result(
            f"加载实验Manifest失败: {exc}",
            "MANIFEST_LOAD_EXCEPTION",
        )


def _load_failure_summary(
    path: Any,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """读取失败账本最新状态，并按条件、错误码和稳定类别显式汇总。"""
    try:
        failure_path = Path(str(path or "")).expanduser()
        if not failure_path.is_file():
            return {
                "status": "success",
                "provided": False,
                "path": None,
                "sha256": None,
                "unresolved_failure_count": 0,
                "by_condition": {},
                "by_error_code": {},
                "by_detail_code": {},
                "by_category": {},
            }
        state: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        for line_number, raw_line in enumerate(
            failure_path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except Exception as exc:
                return _error_result(
                    f"失败账本第{line_number}行不是合法JSON: {exc}",
                    "INVALID_FAILURE_LEDGER_JSON",
                )
            if not isinstance(record, dict):
                return _error_result(
                    f"失败账本第{line_number}行必须是JSON对象",
                    "INVALID_FAILURE_LEDGER_RECORD",
                )
            key = (
                str(record.get("question_id") or ""),
                str(record.get("model") or ""),
                str(record.get("condition") or ""),
                int(record.get("repeat_index", 0)),
            )
            state[key] = record
        unresolved = [
            record
            for record in state.values()
            if str(record.get("status") or "failed") == "failed"
        ]
        condition_counts = Counter(
            str(record.get("condition") or "unknown")
            for record in unresolved
        )
        design = (
            manifest.get("design")
            if isinstance(manifest, dict)
            and isinstance(manifest.get("design"), dict)
            else {}
        )
        dataset = (
            manifest.get("dataset")
            if isinstance(manifest, dict)
            and isinstance(manifest.get("dataset"), dict)
            else {}
        )
        selected_ids = design.get("question_ids")
        question_count = (
            len(selected_ids)
            if isinstance(selected_ids, list) and selected_ids
            else dataset.get("item_count")
        )
        repeats = design.get("repeats")
        planned_per_condition = (
            int(question_count) * int(repeats)
            if isinstance(question_count, int)
            and isinstance(repeats, int)
            and question_count > 0
            and repeats > 0
            else None
        )
        conditions = sorted(
            set(design.get("conditions") or []) | set(condition_counts)
        )
        by_condition = {}
        for condition in conditions:
            count = int(condition_counts.get(condition, 0))
            by_condition[condition] = {
                "failure_count": count,
                "planned_count": planned_per_condition,
                "failure_rate": (
                    count / planned_per_condition
                    if isinstance(planned_per_condition, int)
                    and planned_per_condition > 0
                    else None
                ),
            }
        return {
            "status": "success",
            "provided": True,
            "path": str(failure_path.resolve()),
            "sha256": _sha256_file(failure_path),
            "unresolved_failure_count": len(unresolved),
            "by_condition": by_condition,
            "by_error_code": dict(
                sorted(
                    Counter(
                        str(record.get("error_code") or "UNKNOWN")
                        for record in unresolved
                    ).items()
                )
            ),
            "by_detail_code": dict(
                sorted(
                    Counter(
                        str(record.get("detail_code") or "legacy_missing")
                        for record in unresolved
                    ).items()
                )
            ),
            "by_category": dict(
                sorted(
                    Counter(
                        str(
                            record.get("failure_category")
                            or "legacy_unclassified"
                        )
                        for record in unresolved
                    ).items()
                )
            ),
        }
    except Exception as exc:
        logger.error("汇总实验失败账本失败: %s", exc, exc_info=True)
        return _error_result(
            f"汇总实验失败账本失败: {exc}",
            "FAILURE_SUMMARY_EXCEPTION",
        )


def _normalize_json_value(value: Any) -> str:
    """把嵌套值转换为可比较的稳定JSON文本。"""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except Exception as exc:
        logger.error("规范化控制变量失败: %s", exc, exc_info=True)
        return repr(value)


def _question_signature(record: dict[str, Any]) -> str:
    """生成同一题跨条件应保持一致的脱敏元数据签名。"""
    try:
        return _normalize_json_value(
            {
                "split": record.get("split"),
                "category": record.get("category"),
                "difficulty": record.get("difficulty"),
                "input_type": record.get("input_type"),
                "gold": record.get("gold"),
            }
        )
    except Exception as exc:
        logger.error("生成题目元数据签名失败: %s", exc, exc_info=True)
        return ""


def _is_type_sensitive(record: dict[str, Any]) -> bool:
    """按数据集验收器的冻结规则判断类型敏感题。"""
    try:
        gold = record.get("gold") if isinstance(record.get("gold"), dict) else {}
        preferred_chain = gold.get("preferred_tool_chain")
        plan_length = len(preferred_chain) if isinstance(preferred_chain, list) else 0
        return bool(
            plan_length >= 2
            or gold.get("expected_behavior") in {"repair_then_execute", "reject"}
            or record.get("input_type") in {"protein_fasta_file", "pdb_file"}
        )
    except Exception as exc:
        logger.error("判断类型敏感题失败: %s", exc, exc_info=True)
        return False


def _control_value_set(
    records: list[dict[str, Any]],
    field: str,
) -> set[str]:
    """收集Trace context中的稳定控制变量集合。"""
    try:
        return {
            _normalize_json_value(item.get("context", {}).get(field))
            for item in records
        }
    except Exception as exc:
        logger.error("收集控制变量失败: field=%s error=%s", field, exc, exc_info=True)
        return set()


def _manifest_consistency_issues(
    records: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """检查Manifest与Trace控制变量是否一致。"""
    try:
        if manifest is None:
            return [
                {
                    "code": "MANIFEST_NOT_PROVIDED",
                    "message": "未提供Manifest，控制变量仅依据Trace审计",
                    "severity": "warning",
                }
            ]
        issues: list[dict[str, Any]] = []
        first = records[0]
        context = first.get("context", {})
        design = manifest.get("design") if isinstance(manifest.get("design"), dict) else {}
        dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
        environment = (
            manifest.get("environment")
            if isinstance(manifest.get("environment"), dict)
            else {}
        )
        checks = (
            ("model", design.get("model"), context.get("model")),
            ("temperature", design.get("temperature"), context.get("temperature")),
            ("seed", design.get("seed"), context.get("seed")),
            ("dataset_version", dataset.get("version"), context.get("dataset_version")),
            ("git_commit", environment.get("git_commit"), context.get("git_commit")),
            ("kg_snapshot", environment.get("kg_snapshot"), context.get("kg_snapshot")),
            ("python_version", environment.get("python_version"), context.get("python_version")),
        )
        for field, expected, observed in checks:
            if _normalize_json_value(expected) != _normalize_json_value(observed):
                issues.append(
                    {
                        "code": "MANIFEST_CONTROL_MISMATCH",
                        "message": f"Manifest与Trace的{field}不一致",
                        "severity": "error",
                        "field": field,
                        "manifest_value": expected,
                        "trace_value": observed,
                    }
                )
        manifest_conditions = set(design.get("conditions") or [])
        observed_conditions = {item.get("condition") for item in records}
        if manifest_conditions != observed_conditions:
            issues.append(
                {
                    "code": "MANIFEST_CONDITION_MISMATCH",
                    "message": "Manifest与Trace的实验条件集合不一致",
                    "severity": "error",
                    "manifest_conditions": sorted(manifest_conditions),
                    "trace_conditions": sorted(observed_conditions),
                }
            )
        manifest_repeats = design.get("repeats")
        observed_repeat_indices = sorted({item.get("repeat_index") for item in records})
        expected_repeat_indices = (
            list(range(int(manifest_repeats)))
            if isinstance(manifest_repeats, int) and manifest_repeats > 0
            else []
        )
        if observed_repeat_indices != expected_repeat_indices:
            issues.append(
                {
                    "code": "MANIFEST_REPEAT_MISMATCH",
                    "message": "Manifest与Trace的重复轮次不一致",
                    "severity": "error",
                    "expected": expected_repeat_indices,
                    "observed": observed_repeat_indices,
                }
            )
        if manifest.get("status") != "completed":
            issues.append(
                {
                    "code": "MANIFEST_NOT_COMPLETED",
                    "message": "实验Manifest尚未标记为completed",
                    "severity": "warning",
                    "manifest_status": manifest.get("status"),
                }
            )
        return issues
    except Exception as exc:
        logger.error("检查Manifest一致性失败: %s", exc, exc_info=True)
        return [
            {
                "code": "MANIFEST_AUDIT_EXCEPTION",
                "message": str(exc),
                "severity": "error",
            }
        ]


def _audit_records(
    records: Any,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """验证Trace、重复键、控制变量和题目元数据一致性。"""
    try:
        if not isinstance(records, list) or not records:
            return _error_result(
                "分析输入必须是非空Trace列表",
                "EMPTY_ANALYSIS_INPUT",
            )
        duplicate_keys: list[list[Any]] = []
        seen_keys: set[tuple[str, str, str, int]] = set()
        question_signatures: dict[str, str] = {}
        for index, item in enumerate(records):
            validation = validate_experiment_trace(item)
            if validation.get("status") != "success":
                return _error_result(
                    f"第{index}条Trace未通过Schema校验",
                    "INVALID_ANALYSIS_TRACE",
                    details={"validation": validation},
                )
            key = (
                str(item.get("question_id")),
                str(item.get("context", {}).get("model")),
                str(item.get("condition")),
                int(item.get("repeat_index")),
            )
            if key in seen_keys:
                duplicate_keys.append(list(key))
            seen_keys.add(key)
            question_id = str(item.get("question_id"))
            signature = _question_signature(item)
            if question_id in question_signatures and question_signatures[question_id] != signature:
                return _error_result(
                    "同一question_id跨条件或重复轮次的题目元数据不一致",
                    "QUESTION_METADATA_DRIFT",
                    details={"question_id": question_id},
                )
            question_signatures[question_id] = signature
        if duplicate_keys:
            return _error_result(
                "Trace中存在重复实验单元",
                "DUPLICATE_ANALYSIS_UNIT",
                details={"duplicate_keys": duplicate_keys},
            )

        conditions = {str(item.get("condition")) for item in records}
        required_conditions = set(LEGACY_EXPERIMENT_CONDITIONS)
        if conditions != required_conditions:
            return _error_result(
                "完整B0–B2分析要求同时包含B0、B1和B2",
                "INCOMPLETE_ANALYSIS_CONDITIONS",
                details={
                    "required": sorted(required_conditions),
                    "observed": sorted(conditions),
                },
            )
        control_fields = (
            "model",
            "temperature",
            "seed",
            "git_commit",
            "dataset_version",
            "kg_snapshot",
            "python_version",
        )
        controls: dict[str, Any] = {}
        for field in control_fields:
            values = _control_value_set(records, field)
            if len(values) != 1:
                return _error_result(
                    f"Trace混入多个{field}控制变量",
                    "MIXED_ANALYSIS_CONTROLS",
                    details={"field": field, "values": sorted(values)},
                )
            controls[field] = records[0].get("context", {}).get(field)
        manifest_issues = _manifest_consistency_issues(records, manifest)
        fatal_manifest_issues = [
            issue for issue in manifest_issues if issue.get("severity") == "error"
        ]
        if fatal_manifest_issues:
            return _error_result(
                "Manifest与Trace控制变量不一致",
                "MANIFEST_AUDIT_FAILED",
                details={"manifest_issues": manifest_issues},
            )

        repeat_indices = sorted({int(item.get("repeat_index")) for item in records})
        question_ids = sorted(question_signatures)
        observed_by_condition = {
            condition: sum(item.get("condition") == condition for item in records)
            for condition in sorted(required_conditions)
        }
        expected_trace_count = None
        if manifest is not None:
            dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
            design = manifest.get("design") if isinstance(manifest.get("design"), dict) else {}
            item_count = dataset.get("item_count")
            selected_question_ids = design.get("question_ids")
            if isinstance(selected_question_ids, list):
                normalized_question_ids = {
                    str(item).strip()
                    for item in selected_question_ids
                    if str(item).strip()
                }
                if normalized_question_ids:
                    item_count = len(normalized_question_ids)
            repeats = design.get("repeats")
            manifest_conditions = design.get("conditions")
            if (
                isinstance(item_count, int)
                and isinstance(repeats, int)
                and isinstance(manifest_conditions, list)
            ):
                expected_trace_count = item_count * repeats * len(manifest_conditions)
        missing_trace_count = (
            max(0, expected_trace_count - len(records))
            if isinstance(expected_trace_count, int)
            else None
        )
        return {
            "status": "success",
            "controls": controls,
            "repeat_indices": repeat_indices,
            "question_count": len(question_ids),
            "type_sensitive_question_count": len(
                {
                    str(item.get("question_id"))
                    for item in records
                    if _is_type_sensitive(item)
                }
            ),
            "completeness": {
                "expected_trace_count": expected_trace_count,
                "observed_trace_count": len(records),
                "missing_trace_count": missing_trace_count,
                "observed_by_condition": observed_by_condition,
            },
            "warnings": manifest_issues,
        }
    except Exception as exc:
        logger.error("审计BioToolEval Trace失败: %s", exc, exc_info=True)
        return _error_result(
            f"审计BioToolEval Trace失败: {exc}",
            "ANALYSIS_AUDIT_EXCEPTION",
        )


def holm_adjust(
    hypotheses: Any,
    *,
    alpha: float = 0.05,
    planned_family_size: int | None = None,
) -> dict[str, Any]:
    """对预注册假设族执行Holm校正；缺失检验保留null并计入计划族大小。"""
    try:
        if not isinstance(hypotheses, list) or not hypotheses:
            return _error_result(
                "Holm校正需要非空假设列表",
                "EMPTY_HOLM_FAMILY",
            )
        if not isinstance(alpha, (int, float)) or not 0 < float(alpha) < 1:
            return _error_result(
                "alpha必须位于0和1之间",
                "INVALID_HOLM_ALPHA",
            )
        family_size = (
            int(planned_family_size)
            if isinstance(planned_family_size, int)
            else len(hypotheses)
        )
        if family_size < len(hypotheses) or family_size < 1:
            return _error_result(
                "计划假设族大小不能小于输入假设数",
                "INVALID_HOLM_FAMILY_SIZE",
            )
        normalized: list[dict[str, Any]] = []
        available: list[tuple[int, float]] = []
        for index, item in enumerate(hypotheses):
            if not isinstance(item, dict):
                return _error_result(
                    "每个Holm假设必须是对象",
                    "INVALID_HOLM_HYPOTHESIS",
                )
            copied = dict(item)
            p_value = copied.get("p_value_raw")
            copied["p_value_holm"] = None
            copied["reject_at_alpha"] = None
            normalized.append(copied)
            if p_value is None:
                continue
            numeric_p = float(p_value)
            if numeric_p < 0 or numeric_p > 1:
                return _error_result(
                    "p值必须位于0和1之间",
                    "INVALID_HOLM_P_VALUE",
                    details={"hypothesis": copied.get("id")},
                )
            available.append((index, numeric_p))
        available.sort(key=lambda pair: pair[1])
        running_max = 0.0
        for rank, (index, p_value) in enumerate(available, 1):
            adjusted = min(1.0, (family_size - rank + 1) * p_value)
            running_max = max(running_max, adjusted)
            normalized[index]["p_value_holm"] = running_max
            normalized[index]["reject_at_alpha"] = running_max <= float(alpha)
        return {
            "status": "success",
            "method": "Holm",
            "alpha": float(alpha),
            "planned_family_size": family_size,
            "available_test_count": len(available),
            "family_complete": len(available) == family_size,
            "hypotheses": normalized,
        }
    except Exception as exc:
        logger.error("执行Holm校正失败: %s", exc, exc_info=True)
        return _error_result(
            f"执行Holm校正失败: {exc}",
            "HOLM_ADJUSTMENT_EXCEPTION",
        )


def _empty_primary_hypothesis(
    specification: dict[str, Any],
    reason: str = "没有满足预注册条件的有效配对",
) -> dict[str, Any]:
    """构造无可用配对时的主要假设记录。"""
    try:
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
    except Exception as exc:
        logger.error("构造空主要假设失败: %s", exc, exc_info=True)
        return dict(specification)


def _extract_primary_hypotheses(
    type_sensitive_comparison: dict[str, Any],
    repair_eligible_comparison: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """从类型敏感题和repair-eligible题的单轮比较中提取H1–H3。"""
    try:
        extracted: list[dict[str, Any]] = []
        ts_comparisons = (
            type_sensitive_comparison.get("comparisons") or {}
            if isinstance(type_sensitive_comparison, dict)
            and type_sensitive_comparison.get("status") == "success"
            else {}
        )
        re_comparisons = (
            repair_eligible_comparison.get("comparisons") or {}
            if isinstance(repair_eligible_comparison, dict)
            and repair_eligible_comparison.get("status") == "success"
            else {}
        )

        for specification in PRIMARY_HYPOTHESES:
            hyp_id = specification["id"]
            if hyp_id == "H1":
                # H1: B1 vs B0 on unsafe Type-II execution.
                metric_key = f"{specification['metric']}_difference"
                metric = ts_comparisons.get("B1", {}).get(metric_key)
            elif hyp_id == "H2":
                # H2: B2 vs B1 on true repair opportunities only.
                metric_key = f"{specification['metric']}_difference"
                metric = re_comparisons.get("B2", {}).get(metric_key)
            elif hyp_id == "H3":
                # H3: B2 vs B0 on behavior-aware task success.
                metric_key = f"{specification['metric']}_difference"
                metric = ts_comparisons.get("B2", {}).get(metric_key)
            else:
                metric = None

            if not isinstance(metric, dict):
                extracted.append(
                    _empty_primary_hypothesis(
                        dict(specification),
                        "比较结果未生成该预注册指标",
                    )
                )
                continue
            mcnemar = metric.get("mcnemar") if isinstance(metric.get("mcnemar"), dict) else {}
            pair_count = metric.get("pair_count", 0)
            if not isinstance(pair_count, int) or pair_count <= 0 or not mcnemar:
                extracted.append(
                    _empty_primary_hypothesis(
                        dict(specification),
                        "没有满足指标定义的成对观测，禁止用0替代不可估计",
                    )
                )
                continue
            extracted.append(
                {
                    **specification,
                    "pair_count": pair_count,
                    "mean_difference": metric.get("mean_difference"),
                    "ci_95": metric.get("ci_95"),
                    "p_value_raw": mcnemar.get("p_value_two_sided"),
                    "discordant_pair_count": mcnemar.get("discordant_pair_count"),
                    "estimability_status": "estimable",
                    "estimability_reason": None,
                }
            )
        return extracted
    except Exception as exc:
        logger.error("提取主要假设失败: %s", exc, exc_info=True)
        return [
            _empty_primary_hypothesis(dict(specification))
            for specification in PRIMARY_HYPOTHESES
        ]


def _compute_repeat_sections(
    records: list[dict[str, Any]],
    repeat_indices: list[int],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """按repeat分别计算条件指标、配对比较和主要假设族。"""
    try:
        sections: dict[str, Any] = {}
        for repeat_index in repeat_indices:
            repeat_records = [
                item for item in records if item.get("repeat_index") == repeat_index
            ]
            condition_metrics: dict[str, Any] = {}
            for condition in sorted(LEGACY_EXPERIMENT_CONDITIONS):
                condition_records = [
                    item for item in repeat_records if item.get("condition") == condition
                ]
                condition_metrics[condition] = (
                    compute_condition_metrics(condition_records)
                    if condition_records
                    else _error_result(
                        "该轮次缺少条件记录",
                        "MISSING_REPEAT_CONDITION",
                        details={
                            "repeat_index": repeat_index,
                            "condition": condition,
                        },
                    )
                )
            all_comparison = compare_conditions(
                repeat_records,
                n_resamples=n_resamples,
                seed=seed + repeat_index * 100,
            )
            type_sensitive_records = [
                item for item in repeat_records if _is_type_sensitive(item)
            ]
            if type_sensitive_records:
                type_sensitive_comparison = compare_conditions(
                    type_sensitive_records,
                    n_resamples=n_resamples,
                    seed=seed + repeat_index * 100 + 10,
                )
            else:
                type_sensitive_comparison = _error_result(
                    "该轮次没有类型敏感题",
                    "NO_TYPE_SENSITIVE_RECORDS",
                )
            repair_opportunity_records = [
                item for item in repeat_records if repair_opportunity(item)
            ]
            if repair_opportunity_records:
                repair_eligible_comparison = compare_conditions(
                    repair_opportunity_records,
                    baseline_condition="B1",
                    n_resamples=n_resamples,
                    seed=seed + repeat_index * 100 + 20,
                )
            else:
                repair_eligible_comparison = _error_result(
                    "该轮次没有真实修复机会",
                    "NO_REPAIR_OPPORTUNITY_RECORDS",
                )
            primary = _extract_primary_hypotheses(
                type_sensitive_comparison,
                repair_eligible_comparison,
            )
            holm = holm_adjust(
                primary,
                planned_family_size=len(PRIMARY_HYPOTHESES),
            )
            sections[str(repeat_index)] = {
                "repeat_index": repeat_index,
                "record_count": len(repeat_records),
                "type_sensitive_record_count": len(type_sensitive_records),
                "repair_opportunity_record_count": len(repair_opportunity_records),
                "condition_metrics": condition_metrics,
                "all_question_comparison": all_comparison,
                "type_sensitive_comparison": type_sensitive_comparison,
                "repair_eligible_comparison": repair_eligible_comparison,
                "primary_hypothesis_family": holm,
            }
        return {"status": "success", "repeats": sections}
    except Exception as exc:
        logger.error("计算逐轮分析失败: %s", exc, exc_info=True)
        return _error_result(
            f"计算逐轮分析失败: {exc}",
            "REPEAT_ANALYSIS_EXCEPTION",
        )


def _extract_record_metric(record: dict[str, Any], metric: str) -> float | None:
    """从单条Trace提取题目级聚合所需数值。"""
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
            return float(value) if isinstance(value, (int, float)) else None
        return None
    except Exception as exc:
        logger.error("提取题目级指标失败: metric=%s error=%s", metric, exc, exc_info=True)
        return None


def _question_level_metric_comparison(
    records: list[dict[str, Any]],
    treatment_condition: str,
    metric: str,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """同题内先对配对repeat求均值，再按题目bootstrap。"""
    try:
        if treatment_condition not in {"B1", "B2"}:
            return _error_result(
                "题目级比较只支持B1或B2相对B0",
                "INVALID_QUESTION_LEVEL_TREATMENT",
            )
        grouped: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
        for item in records:
            condition = str(item.get("condition"))
            if condition not in {"B0", treatment_condition}:
                continue
            question_id = str(item.get("question_id"))
            repeat_index = int(item.get("repeat_index"))
            grouped.setdefault(question_id, {}).setdefault(condition, {})[
                repeat_index
            ] = item
        baseline_question_values: list[float] = []
        treatment_question_values: list[float] = []
        paired_repeat_count = 0
        unmatched_repeat_count = 0
        null_repeat_pair_count = 0
        questions_without_usable_pair = 0
        for question_id in sorted(grouped):
            baseline_map = grouped[question_id].get("B0", {})
            treatment_map = grouped[question_id].get(treatment_condition, {})
            common_repeats = sorted(set(baseline_map) & set(treatment_map))
            unmatched_repeat_count += len(set(baseline_map) ^ set(treatment_map))
            baseline_values: list[float] = []
            treatment_values: list[float] = []
            for repeat_index in common_repeats:
                baseline_value = _extract_record_metric(
                    baseline_map[repeat_index],
                    metric,
                )
                treatment_value = _extract_record_metric(
                    treatment_map[repeat_index],
                    metric,
                )
                if baseline_value is None or treatment_value is None:
                    null_repeat_pair_count += 1
                    continue
                baseline_values.append(baseline_value)
                treatment_values.append(treatment_value)
                paired_repeat_count += 1
            if not baseline_values:
                questions_without_usable_pair += 1
                continue
            baseline_question_values.append(
                sum(baseline_values) / len(baseline_values)
            )
            treatment_question_values.append(
                sum(treatment_values) / len(treatment_values)
            )
        total_question_count = len(grouped)
        if not baseline_question_values:
            return {
                "status": "success",
                "metric": metric,
                "comparison": f"{treatment_condition}_vs_B0",
                "question_count": 0,
                "total_question_count": total_question_count,
                "question_coverage": 0.0 if total_question_count > 0 else None,
                "paired_repeat_count": 0,
                "unmatched_repeat_count": unmatched_repeat_count,
                "null_repeat_pair_count": null_repeat_pair_count,
                "questions_without_usable_pair": questions_without_usable_pair,
                "mean_difference": None,
                "ci_95": None,
                "n_resamples": n_resamples,
                "seed": seed,
            }
        bootstrap = paired_bootstrap_difference(
            baseline_question_values,
            treatment_question_values,
            n_resamples=n_resamples,
            seed=seed,
        )
        if bootstrap.get("status") != "success":
            return bootstrap
        return {
            **bootstrap,
            "metric": metric,
            "comparison": f"{treatment_condition}_vs_B0",
            "question_count": len(baseline_question_values),
            "total_question_count": total_question_count,
            "question_coverage": (
                len(baseline_question_values) / total_question_count
                if total_question_count > 0
                else None
            ),
            "paired_repeat_count": paired_repeat_count,
            "unmatched_repeat_count": unmatched_repeat_count,
            "null_repeat_pair_count": null_repeat_pair_count,
            "questions_without_usable_pair": questions_without_usable_pair,
            "statistical_unit": "question",
            "repeat_handling": "mean_within_question_before_bootstrap",
        }
    except Exception as exc:
        logger.error("计算题目级指标失败: %s", exc, exc_info=True)
        return _error_result(
            f"计算题目级指标失败: {exc}",
            "QUESTION_LEVEL_COMPARISON_EXCEPTION",
        )


def _compute_question_level_sections(
    records: list[dict[str, Any]],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """计算总体和类型敏感题的跨重复题目级效应。"""
    try:
        sections: dict[str, Any] = {}
        populations = {
            "all": records,
            "type_sensitive": [item for item in records if _is_type_sensitive(item)],
        }
        metrics = (
            "final_answer_accuracy",
            "behavior_aware_task_success",
            "executable_plan_rate",
            "total_latency_ms",
        )
        counter = 0
        for population_name, population_records in populations.items():
            population_result: dict[str, Any] = {}
            for treatment in ("B1", "B2"):
                treatment_result: dict[str, Any] = {}
                for metric in metrics:
                    treatment_result[metric] = _question_level_metric_comparison(
                        population_records,
                        treatment,
                        metric,
                        n_resamples=n_resamples,
                        seed=seed + 1000 + counter,
                    )
                    counter += 1
                population_result[treatment] = treatment_result
            sections[population_name] = population_result
        return {
            "status": "success",
            "statistical_unit": "question",
            "mcnemar_policy": (
                "精确McNemar仅在逐repeat二元结果中报告；题目级重复均值可能为分数，"
                "因此不错误套用二元检验"
            ),
            "populations": sections,
        }
    except Exception as exc:
        logger.error("计算跨重复题目级分析失败: %s", exc, exc_info=True)
        return _error_result(
            f"计算跨重复题目级分析失败: {exc}",
            "QUESTION_LEVEL_ANALYSIS_EXCEPTION",
        )


def analyze_biotooleval_records(
    records: Any,
    *,
    manifest: dict[str, Any] | None = None,
    failure_summary: dict[str, Any] | None = None,
    trace_sha256: str | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_ANALYSIS_SEED,
) -> dict[str, Any]:
    """分析完整B0–B2 Trace并返回可序列化报告。"""
    try:
        if not isinstance(n_resamples, int) or n_resamples < 100:
            return _error_result(
                "bootstrap重采样次数必须至少为100",
                "INVALID_ANALYSIS_RESAMPLES",
            )
        if not isinstance(seed, int):
            return _error_result(
                "分析随机种子必须是整数",
                "INVALID_ANALYSIS_SEED",
            )
        audit = _audit_records(records, manifest)
        if audit.get("status") != "success":
            return audit
        repeat_analysis = _compute_repeat_sections(
            records,
            audit["repeat_indices"],
            n_resamples=n_resamples,
            seed=seed,
        )
        if repeat_analysis.get("status") != "success":
            return repeat_analysis
        question_level = _compute_question_level_sections(
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
                "by_detail_code": {},
                "by_category": {},
            }
        )
        if failures.get("status") != "success":
            return failures
        completeness = copy.deepcopy(audit["completeness"])
        failure_count = int(failures.get("unresolved_failure_count") or 0)
        expected_count = completeness.get("expected_trace_count")
        accounted_count = completeness.get("observed_trace_count", 0) + failure_count
        completeness["failure_record_count"] = failure_count
        completeness["accounted_unit_count"] = accounted_count
        completeness["unaccounted_unit_count"] = (
            max(0, int(expected_count) - accounted_count)
            if isinstance(expected_count, int)
            else None
        )
        dry_run = bool(manifest.get("dry_run")) if isinstance(manifest, dict) else None
        evidence_scope = (
            "INFRASTRUCTURE_ONLY"
            if dry_run is True
            else "FORMAL_OR_PILOT_RESULTS_REQUIRE_HUMAN_INTERPRETATION"
        )
        result = {
            "status": "success",
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "material_passport": {
                "artifact_id": "BIOTOOL-EVAL-B0-B2-ANALYSIS",
                "artifact_type": "paired_experiment_analysis",
                "verification_status": "ANALYZED",
                "evidence_scope": evidence_scope,
            },
            "input": {
                "trace_sha256": str(trace_sha256 or "") or None,
                "manifest_experiment_id": (
                    manifest.get("experiment_id")
                    if isinstance(manifest, dict)
                    else None
                ),
                "dataset": (
                    manifest.get("dataset")
                    if isinstance(manifest, dict)
                    else None
                ),
            },
            "analysis_config": {
                "bootstrap_resamples": n_resamples,
                "seed": seed,
                "holm_alpha": 0.05,
                "repeat_policy": "mean_within_question_before_bootstrap",
            },
            "controls": audit["controls"],
            "completeness": completeness,
            "failure_outcomes": failures,
            "question_count": audit["question_count"],
            "type_sensitive_question_count": audit[
                "type_sensitive_question_count"
            ],
            "warnings": audit["warnings"],
            "per_repeat": repeat_analysis["repeats"],
            "question_level": question_level,
            "interpretation_boundary": [
                "结果必须同时报告效应大小、95%置信区间、原始分母和coverage",
                "Holm校正只覆盖每个repeat内预注册的H1–H3",
                "题目级重复均值不使用McNemar，因为它不再是二元变量",
                "本报告不自动宣布显著、因果或KBS Go/No-Go结论",
                "失败单元单独报告，不把失败自动等同于任一类幻觉；"
                "只有证据充分时才能进一步分类",
            ],
        }
        logger.info(
            "BioToolEval分析完成 | records=%s | questions=%s | repeats=%s",
            len(records),
            audit["question_count"],
            len(audit["repeat_indices"]),
        )
        return result
    except Exception as exc:
        logger.error("BioToolEval分析崩溃: %s", exc, exc_info=True)
        return _error_result(
            f"BioToolEval分析崩溃: {exc}",
            "ANALYSIS_EXCEPTION",
        )


def _format_number(value: Any, digits: int = 4) -> str:
    """把数值安全格式化为Markdown单元格文本。"""
    try:
        if value is None:
            return "NA"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|")
    except Exception as exc:
        logger.error("格式化报告数值失败: %s", exc, exc_info=True)
        return "NA"


def _format_ci(value: Any) -> str:
    """格式化95%置信区间。"""
    try:
        if not isinstance(value, list) or len(value) != 2:
            return "NA"
        return f"[{_format_number(value[0])}, {_format_number(value[1])}]"
    except Exception as exc:
        logger.error("格式化置信区间失败: %s", exc, exc_info=True)
        return "NA"


def _metric_rate(condition_result: dict[str, Any], metric: str) -> str:
    """读取单条件指标的rate用于Markdown表格。"""
    try:
        value = condition_result.get("metrics", {}).get(metric, {}).get("rate")
        return _format_number(value)
    except Exception as exc:
        logger.error("读取条件指标失败: %s", exc, exc_info=True)
        return "NA"


def render_analysis_markdown(analysis: Any) -> dict[str, Any]:
    """把结构化分析结果渲染为可审计Markdown摘要。"""
    try:
        if not isinstance(analysis, dict) or analysis.get("status") != "success":
            return _error_result(
                "只有成功的分析结果才能渲染Markdown",
                "INVALID_MARKDOWN_ANALYSIS",
            )
        passport = analysis.get("material_passport", {})
        controls = analysis.get("controls", {})
        completeness = analysis.get("completeness", {})
        lines = [
            "# BioToolEval B0–B2实验分析报告",
            "",
            "## Material Passport",
            "",
            f"- Artifact ID：`{passport.get('artifact_id')}`",
            f"- Verification Status：`{passport.get('verification_status')}`",
            f"- Evidence Scope：`{passport.get('evidence_scope')}`",
            f"- Analysis Schema：`{analysis.get('analysis_schema_version')}`",
            "",
            "## 控制变量与完整性",
            "",
            f"- 模型：`{controls.get('model')}`",
            f"- 温度：`{controls.get('temperature')}`",
            f"- Git commit：`{controls.get('git_commit')}`",
            f"- 数据版本：`{controls.get('dataset_version')}`",
            f"- KG快照：`{controls.get('kg_snapshot')}`",
            f"- 题目数：{analysis.get('question_count')}",
            f"- 类型敏感题数：{analysis.get('type_sensitive_question_count')}",
            f"- 预期/实际Trace：{completeness.get('expected_trace_count')} / "
            f"{completeness.get('observed_trace_count')}",
            f"- 缺失Trace：{completeness.get('missing_trace_count')}",
            f"- 失败账本记录：{completeness.get('failure_record_count', 0)}",
            f"- 已解释/未解释实验单元："
            f"{completeness.get('accounted_unit_count')} / "
            f"{completeness.get('unaccounted_unit_count')}",
            "",
            "## 失败单元（不从分母中静默删除）",
            "",
            "| condition | failures | planned | failure rate |",
            "|---|---:|---:|---:|",
        ]
        failure_outcomes = analysis.get("failure_outcomes") or {}
        for condition, failure_result in (
            failure_outcomes.get("by_condition") or {}
        ).items():
            lines.append(
                "| "
                f"{condition} | {failure_result.get('failure_count', 0)} | "
                f"{failure_result.get('planned_count')} | "
                f"{_format_number(failure_result.get('failure_rate'))} |"
            )
        if not (failure_outcomes.get("by_condition") or {}):
            lines.append("| NA | 0 | NA | NA |")
        lines.extend(
            [
                "",
                "- 错误码分布：`"
                + json.dumps(
                    failure_outcomes.get("by_error_code") or {},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "`",
                "- 失败类别分布：`"
                + json.dumps(
                    failure_outcomes.get("by_category") or {},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "`",
            "",
            "## 逐轮条件指标",
            "",
            "| repeat | condition | Final Answer Accuracy | Executable Plan Rate | Type-II Generation Rate | Repair Precision |",
            "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for repeat_key, repeat_section in analysis.get("per_repeat", {}).items():
            for condition in sorted(LEGACY_EXPERIMENT_CONDITIONS):
                condition_result = repeat_section.get("condition_metrics", {}).get(condition, {})
                lines.append(
                    "| "
                    f"{repeat_key} | {condition} | "
                    f"{_metric_rate(condition_result, 'final_answer_accuracy')} | "
                    f"{_metric_rate(condition_result, 'executable_plan_rate')} | "
                    f"{_metric_rate(condition_result, 'type_ii_generation_rate')} | "
                    f"{_metric_rate(condition_result, 'repair_precision')} |"
                )
        lines.extend(
            [
                "",
                "## 主要假设（类型敏感题，逐repeat）",
                "",
                "| repeat | hypothesis | comparison | status | effect | 95% CI | raw p | Holm p | reject@0.05 | pairs |",
                "|---:|---|---|---|---:|---|---:|---:|---|---:|",
            ]
        )
        for repeat_key, repeat_section in analysis.get("per_repeat", {}).items():
            family = repeat_section.get("primary_hypothesis_family", {})
            for hypothesis in family.get("hypotheses", []):
                lines.append(
                    "| "
                    f"{repeat_key} | {hypothesis.get('id')} | "
                    f"{hypothesis.get('comparison')} | "
                    f"{hypothesis.get('estimability_status', 'NA')} | "
                    f"{_format_number(hypothesis.get('mean_difference'))} | "
                    f"{_format_ci(hypothesis.get('ci_95'))} | "
                    f"{_format_number(hypothesis.get('p_value_raw'))} | "
                    f"{_format_number(hypothesis.get('p_value_holm'))} | "
                    f"{_format_number(hypothesis.get('reject_at_alpha'))} | "
                    f"{hypothesis.get('pair_count', 0)} |"
                )
        lines.extend(
            [
                "",
                "## 跨重复题目级效应",
                "",
                "| population | comparison | metric | questions | repeat pairs | effect | 95% CI | coverage |",
                "|---|---|---|---:|---:|---:|---|---:|",
            ]
        )
        populations = analysis.get("question_level", {}).get("populations", {})
        for population_name, population in populations.items():
            for treatment, treatment_result in population.items():
                for metric, metric_result in treatment_result.items():
                    lines.append(
                        "| "
                        f"{population_name} | {treatment}_vs_B0 | {metric} | "
                        f"{metric_result.get('question_count', 0)} | "
                        f"{metric_result.get('paired_repeat_count', 0)} | "
                        f"{_format_number(metric_result.get('mean_difference'))} | "
                        f"{_format_ci(metric_result.get('ci_95'))} | "
                        f"{_format_number(metric_result.get('question_coverage'))} |"
                    )
        lines.extend(["", "## 警告与解释边界", ""])
        warnings = analysis.get("warnings") or []
        if warnings:
            for warning in warnings:
                lines.append(
                    f"- `{warning.get('code')}`：{warning.get('message')}"
                )
        else:
            lines.append("- 控制变量和Manifest一致性检查未发现警告。")
        for boundary in analysis.get("interpretation_boundary", []):
            lines.append(f"- {boundary}")
        lines.append("")
        return {"status": "success", "markdown": "\n".join(lines)}
    except Exception as exc:
        logger.error("渲染分析Markdown失败: %s", exc, exc_info=True)
        return _error_result(
            f"渲染分析Markdown失败: {exc}",
            "MARKDOWN_RENDER_EXCEPTION",
        )


def _atomic_write_text(path: Path, content: str, *, overwrite: bool) -> dict[str, Any]:
    """以UTF-8原子写入分析产物，默认拒绝覆盖。"""
    temporary_path: Path | None = None
    try:
        if path.exists() and not overwrite:
            return _error_result(
                "分析输出已存在；如确认覆盖请使用--overwrite",
                "ANALYSIS_OUTPUT_EXISTS",
                details={"path": str(path)},
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")
        temporary_path.write_text(content, encoding="utf-8", newline="\n")
        temporary_path.replace(path)
        return {"status": "success", "path": str(path.resolve())}
    except Exception as exc:
        logger.error("写入分析产物失败: %s", exc, exc_info=True)
        return _error_result(
            f"写入分析产物失败: {exc}",
            "ANALYSIS_WRITE_EXCEPTION",
            details={"path": str(path)},
        )
    finally:
        try:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        except Exception:
            logger.warning("清理分析临时文件失败: %s", temporary_path)


def write_analysis_artifacts(
    analysis: Any,
    *,
    json_path: Any,
    markdown_path: Any,
    overwrite: bool = False,
) -> dict[str, Any]:
    """写出结构化JSON和人类可读Markdown报告。"""
    try:
        if not isinstance(analysis, dict) or analysis.get("status") != "success":
            return _error_result(
                "只有成功分析才能写出产物",
                "INVALID_ANALYSIS_ARTIFACT",
            )
        target_json = Path(str(json_path or "")).expanduser()
        target_markdown = Path(str(markdown_path or "")).expanduser()
        if target_json.suffix.lower() != ".json":
            return _error_result(
                "结构化分析输出必须使用.json扩展名",
                "INVALID_ANALYSIS_JSON_PATH",
            )
        if target_markdown.suffix.lower() != ".md":
            return _error_result(
                "分析报告必须使用.md扩展名",
                "INVALID_ANALYSIS_MARKDOWN_PATH",
            )
        if not overwrite:
            existing_paths = [
                str(path)
                for path in (target_json, target_markdown)
                if path.exists()
            ]
            if existing_paths:
                return _error_result(
                    "分析输出已存在；如确认覆盖请使用--overwrite",
                    "ANALYSIS_OUTPUT_EXISTS",
                    details={"paths": existing_paths},
                )
        rendered = render_analysis_markdown(analysis)
        if rendered.get("status") != "success":
            return rendered
        json_write = _atomic_write_text(
            target_json,
            json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
            overwrite=overwrite,
        )
        if json_write.get("status") != "success":
            return json_write
        markdown_write = _atomic_write_text(
            target_markdown,
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
        logger.error("导出分析产物失败: %s", exc, exc_info=True)
        return _error_result(
            f"导出分析产物失败: {exc}",
            "ANALYSIS_EXPORT_EXCEPTION",
        )


def analyze_trace_file(
    trace_path: Any,
    *,
    manifest_path: Any = None,
    failure_path: Any = None,
    output_prefix: Any = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_ANALYSIS_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    """加载Trace与Manifest、执行分析并导出两个报告文件。"""
    try:
        trace = Path(str(trace_path or "")).expanduser()
        loaded = load_experiment_traces(trace)
        if loaded.get("status") != "success":
            return loaded
        resolved_manifest_path: Path | None = None
        if manifest_path is not None:
            resolved_manifest_path = Path(str(manifest_path)).expanduser()
        else:
            candidate = trace.with_suffix(".manifest.json")
            if candidate.is_file():
                resolved_manifest_path = candidate
        manifest: dict[str, Any] | None = None
        if resolved_manifest_path is not None:
            manifest_result = _load_manifest(resolved_manifest_path)
            if manifest_result.get("status") != "success":
                return manifest_result
            manifest = manifest_result["manifest"]
        resolved_failure_path: Path | None = None
        if failure_path is not None:
            resolved_failure_path = Path(str(failure_path)).expanduser()
        elif isinstance(manifest, dict):
            artifact_name = (
                (manifest.get("artifacts") or {}).get("failure_file_name")
            )
            if isinstance(artifact_name, str) and artifact_name.strip():
                candidate = trace.parent / artifact_name
                if candidate.is_file():
                    resolved_failure_path = candidate
        if resolved_failure_path is None:
            candidate = trace.with_name(f"{trace.stem}.failures.jsonl")
            if candidate.is_file():
                resolved_failure_path = candidate
        failure_summary = _load_failure_summary(
            resolved_failure_path,
            manifest,
        )
        if failure_summary.get("status") != "success":
            return failure_summary
        analysis = analyze_biotooleval_records(
            loaded["records"],
            manifest=manifest,
            failure_summary=failure_summary,
            trace_sha256=_sha256_file(trace),
            n_resamples=n_resamples,
            seed=seed,
        )
        if analysis.get("status") != "success":
            return analysis
        prefix = (
            Path(str(output_prefix)).expanduser()
            if output_prefix is not None
            else trace.parent / f"{trace.stem}.analysis"
        )
        json_output = prefix.with_name(f"{prefix.name}.json")
        markdown_output = prefix.with_name(f"{prefix.name}.md")
        exported = write_analysis_artifacts(
            analysis,
            json_path=json_output,
            markdown_path=markdown_output,
            overwrite=overwrite,
        )
        if exported.get("status") != "success":
            return exported
        return {
            "status": "success",
            "record_count": loaded["record_count"],
            "question_count": analysis["question_count"],
            "type_sensitive_question_count": analysis[
                "type_sensitive_question_count"
            ],
            "json_path": exported["json_path"],
            "markdown_path": exported["markdown_path"],
            "warnings": analysis["warnings"],
        }
    except Exception as exc:
        logger.error("分析Trace文件失败: %s", exc, exc_info=True)
        return _error_result(
            f"分析Trace文件失败: {exc}",
            "TRACE_FILE_ANALYSIS_EXCEPTION",
        )


def _build_cli_parser() -> argparse.ArgumentParser:
    """构建BioToolEval分析命令行参数。"""
    try:
        parser = argparse.ArgumentParser(
            description="分析BioToolEval B0/B1/B2实验Trace"
        )
        parser.add_argument("--trace", required=True, help="实验Trace JSONL路径")
        parser.add_argument("--manifest", help="实验Manifest路径；默认自动寻找同名文件")
        parser.add_argument(
            "--failures",
            help="失败账本JSONL路径；默认从Manifest或Trace同名前缀查找",
        )
        parser.add_argument("--output-prefix", help="输出前缀；自动生成.json和.md")
        parser.add_argument(
            "--bootstrap-resamples",
            type=int,
            default=DEFAULT_BOOTSTRAP_RESAMPLES,
        )
        parser.add_argument("--seed", type=int, default=DEFAULT_ANALYSIS_SEED)
        parser.add_argument("--overwrite", action="store_true")
        return parser
    except Exception as exc:
        logger.error("构建分析CLI失败: %s", exc, exc_info=True)
        raise RuntimeError("无法构建BioToolEval分析命令行") from exc


def main(argv: list[str] | None = None) -> int:
    """命令行入口；只打印脱敏摘要。"""
    try:
        parser = _build_cli_parser()
        args = parser.parse_args(argv)
        result = analyze_trace_file(
            args.trace,
            manifest_path=args.manifest,
            failure_path=args.failures,
            output_prefix=args.output_prefix,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        safe_output = {
            "status": result.get("status"),
            "record_count": result.get("record_count"),
            "question_count": result.get("question_count"),
            "type_sensitive_question_count": result.get(
                "type_sensitive_question_count"
            ),
            "json_path": result.get("json_path"),
            "markdown_path": result.get("markdown_path"),
            "warnings": result.get("warnings"),
            "message": result.get("message"),
        }
        print(json.dumps(safe_output, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "success" else 1
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("BioToolEval分析CLI失败: %s", exc, exc_info=True)
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": f"BioToolEval分析CLI失败: {exc}",
                },
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
