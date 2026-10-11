# -*- coding: utf-8 -*-
"""BioToolEval Test-150 v5.3 后继冻结校验入口。

修改记录：
- 2026-09-07：保留历史 ``final150`` 校验器及其冻结哈希不变，仅将双人复核后
  的 v5.3 配额修订（41 PDB、9 none、106 real）限定在本入口内。

该模块复用原校验器的全部结构、Gold、类型、近重复、工具覆盖与正式治理门禁；
只在一次函数调用的受控上下文中替换 v5.3 的精确输入/来源配额，随后立即恢复。
"""

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
    """临时安装 v5.3 配额并在异常路径也恢复历史 Profile。"""
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
    """执行完整 final150 门禁，但使用已登记的 v5.3 精确配额。"""
    with _v53_profile():
        return base_validator.validate_biotooleval_dataset(
            dataset_path,
            profile="final150",
            near_duplicate_threshold=near_duplicate_threshold,
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """写出 UTF-8 JSON 报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    """复用历史校验器的脱敏 Markdown 报告格式。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(base_validator._markdown_report(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """命令行入口；失败关闭且不修改数据集。"""
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
