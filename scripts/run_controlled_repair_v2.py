# -*- coding: utf-8 -*-
"""Run the version 2 controlled repair suite."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Config
from app.core.logger import get_logger
from scripts.controlled_repair_v2_runtime import (
    CONDITIONS_V2,
    ControlledV2Runner,
    load_controlled_repair_v2_dataset,
    local_controlled_v2_runner,
    run_controlled_v2_unit,
)
from scripts.run_controlled_repair import (
    _error_result,
    _git_commit,
    _load_json_validator,
    _rate,
    _write_json_atomic,
    _write_jsonl_atomic,
)


logger = get_logger("ControlledRepairV2", "controlled_repair_v2.log")
_EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _paired_metrics(
    traces: list[dict[str, Any]],
    *,
    planned_pairs: int,
) -> dict[str, Any]:
    pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for trace in traces:
        key = (str(trace["case_id"]), int(trace["repeat_index"]))
        pairs.setdefault(key, {})[str(trace["condition"])] = trace
    complete_pairs = [
        value for value in pairs.values() if set(value) == set(CONDITIONS_V2)
    ]
    opportunities = [
        pair
        for pair in complete_pairs
        if pair["E2"]["evaluation"]["repair_opportunity"]
        and pair["E3"]["evaluation"]["repair_opportunity"]
    ]
    recovered = sum(
        pair["E3"]["evaluation"]["recovery_success"] for pair in opportunities
    )
    applied = [
        trace
        for trace in traces
        if trace["condition"] == "E3" and trace["plan"]["repair_applied"]
    ]
    correct = sum(
        trace["evaluation"]["repair_correct"] is True for trace in applied
    )
    return {
        "pair_coverage": _rate(len(complete_pairs), planned_pairs),
        "repair_recovery_rate": _rate(recovered, len(opportunities)),
        "repair_precision": _rate(correct, len(applied)),
    }


def build_v2_metrics(
    traces: list[dict[str, Any]],
    *,
    planned_family_pairs: dict[str, int],
) -> dict[str, Any]:
    planned_pairs = sum(planned_family_pairs.values())
    overall = _paired_metrics(traces, planned_pairs=planned_pairs)
    by_family: dict[str, dict[str, Any]] = {}
    family_rates: list[float] = []
    fully_recovered_families = 0
    for family, family_planned in sorted(planned_family_pairs.items()):
        family_traces = [trace for trace in traces if trace["family"] == family]
        metrics = _paired_metrics(family_traces, planned_pairs=family_planned)
        by_family[family] = metrics
        recovery = metrics["repair_recovery_rate"]
        if recovery["estimability_status"] == "estimable":
            rate = float(recovery["rate"])
            family_rates.append(rate)
            if rate == 1.0:
                fully_recovered_families += 1
    family_denominator = len(planned_family_pairs)
    return {
        "overall": overall,
        "by_family": by_family,
        "family_macro_recovery_rate": {
            "estimability_status": (
                "estimable"
                if len(family_rates) == family_denominator and family_denominator > 0
                else "not_estimable"
            ),
            "family_numerator_fully_recovered": fully_recovered_families,
            "family_denominator": family_denominator,
            "macro_rate": (
                round(sum(family_rates) / family_denominator, 6)
                if len(family_rates) == family_denominator and family_denominator > 0
                else None
            ),
        },
    }


async def run_controlled_repair_v2_experiment(
    *,
    experiment_id: Any,
    dataset_path: Any = None,
    output_path: Any = None,
    repeats: Any = 1,
    config: Any = None,
    git_commit: Any = None,
    runner: ControlledV2Runner = local_controlled_v2_runner,
) -> dict[str, Any]:
    try:
        cfg = config if config is not None else Config()
        normalized_id = str(experiment_id or "").strip()
        if not _EXPERIMENT_ID_PATTERN.fullmatch(normalized_id):
            return _error_result("experiment_id格式无效", "INVALID_EXPERIMENT_ID")
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
            return _error_result("repeats必须是正整数", "INVALID_REPEATS")
        if not callable(runner):
            return _error_result("runner必须是异步可调用对象", "INVALID_RUNNER")

        dataset_result = load_controlled_repair_v2_dataset(
            dataset_path,
            config=cfg,
        )
        if dataset_result.get("status") != "success":
            return dataset_result
        trace_validator_result = _load_json_validator(
            Path(
                str(getattr(cfg, "CONTROLLED_REPAIR_V2_TRACE_SCHEMA_PATH", ""))
            ).expanduser().resolve(),
            "Controlled Repair v2 Trace",
        )
        if trace_validator_result.get("status") != "success":
            return trace_validator_result

        if output_path is None:
            trace_path = (
                Path(
                    str(getattr(cfg, "CONTROLLED_REPAIR_V2_RESULTS_DIR", ""))
                ).expanduser().resolve()
                / f"{normalized_id}.jsonl"
            )
        else:
            trace_path = Path(str(output_path)).expanduser().resolve()
        if trace_path.suffix.lower() != ".jsonl":
            return _error_result("Trace输出必须使用.jsonl", "INVALID_OUTPUT_PATH")
        manifest_path = trace_path.with_suffix(".manifest.json")
        failure_path = trace_path.with_suffix(".failures.jsonl")
        existing = [
            path.name
            for path in (trace_path, manifest_path, failure_path)
            if path.exists()
        ]
        if existing:
            return _error_result(
                "v2实验输出已存在；no-overwrite规则禁止覆盖",
                "OUTPUT_EXISTS",
                details={"existing_files": existing},
            )
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        commit = str(git_commit or "").strip().lower() or _git_commit(PROJECT_ROOT)
        if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
            return _error_result("git_commit格式无效", "INVALID_GIT_COMMIT")
        context = {
            "dataset_version": dataset_result["version"],
            "dataset_sha256": dataset_result["dataset_sha256"],
            "git_commit": commit,
            "python_version": platform.python_version(),
        }
        try:
            timeout_seconds = float(
                getattr(cfg, "CONTROLLED_REPAIR_TIMEOUT_SECONDS", 30.0)
            )
        except (TypeError, ValueError):
            timeout_seconds = 30.0
        timeout_seconds = min(300.0, max(1.0, timeout_seconds))

        traces: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        started_at = datetime.now(timezone.utc).isoformat()
        for repeat_index in range(repeats):
            for case in dataset_result["items"]:
                for condition in CONDITIONS_V2:
                    try:
                        unit = await asyncio.wait_for(
                            run_controlled_v2_unit(
                                case,
                                condition=condition,
                                repeat_index=repeat_index,
                                experiment_id=normalized_id,
                                context=context,
                                trace_validator=trace_validator_result["validator"],
                                runner=runner,
                            ),
                            timeout=timeout_seconds,
                        )
                        if unit.get("status") == "success":
                            traces.append(unit["trace"])
                        else:
                            failures.append(
                                {
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "experiment_id": normalized_id,
                                    "case_id": str(case.get("id") or ""),
                                    "family": str(case.get("family") or ""),
                                    "condition": condition,
                                    "repeat_index": repeat_index,
                                    "error_code": str(
                                        unit.get("detail_code") or "UNIT_FAILURE"
                                    ),
                                    "exception_type": None,
                                }
                            )
                    except asyncio.TimeoutError:
                        failures.append(
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "experiment_id": normalized_id,
                                "case_id": str(case.get("id") or ""),
                                "family": str(case.get("family") or ""),
                                "condition": condition,
                                "repeat_index": repeat_index,
                                "error_code": "UNIT_TIMEOUT",
                                "exception_type": "TimeoutError",
                            }
                        )
                    except Exception as exc:
                        logger.error(
                            "v2实验单元异常 | case=%s | condition=%s | error=%s",
                            case.get("id"),
                            condition,
                            exc,
                            exc_info=True,
                        )
                        failures.append(
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "experiment_id": normalized_id,
                                "case_id": str(case.get("id") or ""),
                                "family": str(case.get("family") or ""),
                                "condition": condition,
                                "repeat_index": repeat_index,
                                "error_code": "UNIT_EXCEPTION",
                                "exception_type": type(exc).__name__,
                            }
                        )

        planned_family_pairs = {
            family: count * repeats
            for family, count in dataset_result["family_counts"].items()
        }
        planned_pairs = sum(planned_family_pairs.values())
        planned_units = planned_pairs * len(CONDITIONS_V2)
        metrics = build_v2_metrics(
            traces,
            planned_family_pairs=planned_family_pairs,
        )
        manifest = {
            "schema_version": "2.0",
            "experiment_id": normalized_id,
            "status": "completed" if not failures else "completed_with_failures",
            "design": "paired_frozen_invalid_plan_e2_vs_e3_family_stratified",
            "evidence_scope": dataset_result["evidence_scope"],
            "dataset_version": dataset_result["version"],
            "dataset_sha256": dataset_result["dataset_sha256"],
            "git_commit": commit,
            "python_version": platform.python_version(),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "repeats": repeats,
            "family_count": dataset_result["family_count"],
            "family_case_counts": dataset_result["family_counts"],
            "planned_units": planned_units,
            "completed_units": len(traces),
            "failed_units": len(failures),
            "automatic_retries": 0,
            "metrics": metrics,
        }
        _write_jsonl_atomic(trace_path, traces)
        _write_jsonl_atomic(failure_path, failures)
        _write_json_atomic(manifest_path, manifest)
        return {
            "status": "success",
            "completed": True,
            "experiment_id": normalized_id,
            "trace_path": str(trace_path),
            "manifest_path": str(manifest_path),
            "failure_path": str(failure_path),
            "manifest": manifest,
        }
    except Exception as exc:
        logger.error("运行Controlled Repair v2失败: %s", exc, exc_info=True)
        return _error_result(
            f"运行Controlled Repair v2失败: {type(exc).__name__}",
            "CONTROLLED_V2_EXPERIMENT_EXCEPTION",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行Controlled Repair Track v2")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    try:
        args = _parse_args()
        result = asyncio.run(
            run_controlled_repair_v2_experiment(
                experiment_id=args.experiment_id,
                dataset_path=args.dataset,
                output_path=args.output,
                repeats=args.repeats,
            )
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "success" else 1
    except Exception as exc:
        print(
            json.dumps(
                _error_result(
                    f"Controlled Repair v2命令行异常: {type(exc).__name__}",
                    "CONTROLLED_V2_CLI_EXCEPTION",
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
