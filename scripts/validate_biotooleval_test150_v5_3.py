# -*- coding: utf-8 -*-
"""Validate the frozen Test-150 dataset."""

from __future__ import annotations

import argparse
import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from scripts import validate_biotooleval_dataset as base_validator


_V53_EXACT_INPUT_TYPES = {
    "amino_acid_sequence": 20,
    "protein_fasta_file": 80,
    "pdb_file": 41,
    "none": 9,
}
_V53_EXACT_SOURCE_TYPES = {"real": 106, "synthetic": 35, "none": 9}


@contextmanager
def _v53_profile() -> Iterator[None]:
    original = copy.deepcopy(base_validator._PROFILE_RULES["final150"])
    amended = copy.deepcopy(original)
    amended["exact_input_types"] = copy.deepcopy(_V53_EXACT_INPUT_TYPES)
    amended["exact_source_types"] = copy.deepcopy(_V53_EXACT_SOURCE_TYPES)
    base_validator._PROFILE_RULES["final150"] = amended
    try:
        yield
    finally:
        base_validator._PROFILE_RULES["final150"] = original


def validate_test150_v5_3(
    dataset_path: str | Path,
    *,
    near_duplicate_threshold: float = 0.92,
) -> dict[str, Any]:
    with _v53_profile():
        return base_validator.validate_biotooleval_dataset(
            dataset_path,
            profile="final150",
            near_duplicate_threshold=near_duplicate_threshold,
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(base_validator._markdown_report(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    args = parser.parse_args(argv)
    report = validate_test150_v5_3(
        args.dataset, near_duplicate_threshold=args.near_duplicate_threshold
    )
    if args.json_report:
        _write_json(args.json_report, report)
    if args.markdown_report:
        _write_markdown(args.markdown_report, report)
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "verdict": report.get("verdict"),
                "benchmark_ready": report.get("benchmark_ready"),
                "issue_counts": report.get("issue_counts"),
                "profile": report.get("dataset", {}).get("profile"),
                "profile_amendment": "test150-v5.3-input-source-quota-only",
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.get("verdict") == "pass" and report.get("benchmark_ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
