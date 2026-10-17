#!/usr/bin/env python3
"""Focused Gate3 H1-H3 analysis using the frozen factorial definitions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from scripts.analyze_biotooleval import _is_type_sensitive, holm_adjust
from scripts.evaluation_metrics import (
    behavior_aware_task_success,
    mcnemar_exact,
    paired_bootstrap_difference,
    repair_opportunity,
    repair_recovery_outcome,
    unsafe_type_ii_execution,
)


SPECS = (
    {
        "id": "H1",
        "comparison": "E2_vs_E1",
        "metric": "unsafe_type_ii_execution_rate",
        "baseline": "E1",
        "treatment": "E2",
    },
    {
        "id": "H2",
        "comparison": "E3_vs_E2",
        "metric": "repair_recovery_rate",
        "baseline": "E2",
        "treatment": "E3",
    },
    {
        "id": "H3",
        "comparison": "E3_vs_E0",
        "metric": "behavior_aware_task_success",
        "baseline": "E0",
        "treatment": "E3",
    },
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["question_id"]),
        str(row["context"]["model"]),
        int(row["repeat_index"]),
    )


def _paired_binary(
    rows: list[dict[str, Any]],
    baseline: str,
    treatment: str,
    extractor: Callable[[dict[str, Any]], bool | None],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_map = {_key(row): row for row in rows if row["condition"] == baseline}
    treatment_map = {_key(row): row for row in rows if row["condition"] == treatment}
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
    result = paired_bootstrap_difference(
        baseline_values,
        treatment_values,
        n_resamples=n_resamples,
        seed=seed,
    )
    result["mcnemar"] = mcnemar_exact(baseline_values, treatment_values)
    return result


def analyze(rows: list[dict[str, Any]], n_resamples: int, seed: int) -> dict[str, Any]:
    repeats: dict[str, Any] = {}
    for repeat_index in sorted({int(row["repeat_index"]) for row in rows}):
        repeat_rows = [row for row in rows if int(row["repeat_index"]) == repeat_index]
        type_sensitive = [row for row in repeat_rows if _is_type_sensitive(row)]
        repair_rows = [
            row
            for row in repeat_rows
            if row["condition"] in {"E2", "E3"} and repair_opportunity(row)
        ]
        metric_results = (
            _paired_binary(
                type_sensitive,
                "E1",
                "E2",
                unsafe_type_ii_execution,
                n_resamples=n_resamples,
                seed=seed + repeat_index * 100 + 4,
            ),
            _paired_binary(
                repair_rows,
                "E2",
                "E3",
                repair_recovery_outcome,
                n_resamples=n_resamples,
                seed=seed + repeat_index * 100 + 15,
            ),
            _paired_binary(
                type_sensitive,
                "E0",
                "E3",
                behavior_aware_task_success,
                n_resamples=n_resamples,
                seed=seed + repeat_index * 100 + 26,
            ),
        )
        hypotheses: list[dict[str, Any]] = []
        for spec, result in zip(SPECS, metric_results):
            mcnemar = result.get("mcnemar")
            estimable = int(result.get("pair_count") or 0) > 0 and isinstance(mcnemar, dict)
            hypotheses.append(
                {
                    "id": spec["id"],
                    "comparison": spec["comparison"],
                    "metric": spec["metric"],
                    "pair_count": int(result.get("pair_count") or 0),
                    "mean_difference": result.get("mean_difference"),
                    "ci_95": result.get("ci_95"),
                    "p_value_raw": mcnemar.get("p_value_two_sided") if estimable else None,
                    "discordant_pair_count": mcnemar.get("discordant_pair_count") if estimable else None,
                    "estimability_status": "estimable" if estimable else "not_estimable",
                    "estimability_reason": None if estimable else "no paired observations",
                }
            )
        repeats[str(repeat_index)] = holm_adjust(hypotheses, planned_family_size=3)
    return {
        "status": "success",
        "analysis_schema_version": "gate3-primary-focused-1.0",
        "analysis_config": {
            "bootstrap_resamples": n_resamples,
            "seed": seed,
            "pairing": "question_id+model+repeat_index",
            "implementation": "frozen extractors and Python random bootstrap",
            "holm_alpha": 0.05,
        },
        "record_count": len(rows),
        "per_repeat": repeats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {args.output}")
    result = analyze(_read_jsonl(args.trace), args.bootstrap_resamples, args.seed)
    result["input"] = {
        "trace": str(args.trace.resolve()),
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "success", "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
