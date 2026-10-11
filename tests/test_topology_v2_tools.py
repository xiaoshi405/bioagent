# -*- coding: utf-8 -*-
"""
Topology v2-lite多记录FASTA与蛋白质比对工具回归测试。

修改记录：
- 2026-08-14：覆盖集合解析、精确选择、成对比对、统计摘要及防御性拒绝。
"""

import tempfile
import unittest
from pathlib import Path

from ToolsAgent.ToolsFuns.Biology.fasta_common import PROJECT_ROOT
from ToolsAgent.ToolsFuns.Biology.fasta_collection import (
    parse_multi_protein_fasta,
    select_fasta_record,
)
from ToolsAgent.ToolsFuns.Biology.protein_alignment import (
    align_fasta_records,
    summarize_pairwise_alignment,
)


FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "fasta" / "multi_protein.fasta"


class TopologyV2ToolTests(unittest.TestCase):
    """验证四个工具形成可执行的数据依赖链，而非共享输入扇出。"""

    def setUp(self):
        self.parsed = parse_multi_protein_fasta(str(FIXTURE))
        self.assertEqual(self.parsed["status"], "success")
        self.collection = self.parsed["collection"]

    def test_multi_fasta_parses_to_named_collection(self):
        self.assertEqual(self.parsed["record_count"], 2)
        self.assertEqual(self.parsed["record_ids"], ["protein_a", "protein_b"])
        self.assertEqual(
            self.collection["schema_version"],
            "protein-sequence-collection/v1",
        )

    def test_exact_record_selection_returns_sequence(self):
        result = select_fasta_record(self.collection, "protein_b")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["record_id"], "protein_b")
        self.assertEqual(result["sequence"], "ACDEYGHIKLMNPQRSAVWY")
        self.assertEqual(result["selection_method"], "exact_record_id")

    def test_unknown_record_id_returns_actionable_error(self):
        result = select_fasta_record(self.collection, "missing")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "FASTA_RECORD_IDENTIFIER_NOT_FOUND")
        self.assertEqual(
            result["issues"][0]["available_record_ids"],
            ["protein_a", "protein_b"],
        )
        self.assertIn("suggestion", result["issues"][0])

    def test_alignment_and_summary_form_two_stage_chain(self):
        aligned = align_fasta_records(
            self.collection,
            ["protein_a", "protein_b"],
        )
        self.assertEqual(aligned["status"], "success")
        self.assertEqual(
            aligned["alignment"]["schema_version"],
            "pairwise-protein-alignment/v1",
        )

        summary = summarize_pairwise_alignment(aligned["alignment"])

        self.assertEqual(summary["status"], "success")
        statistics = summary["statistics"]
        self.assertEqual(statistics["alignment_length"], 20)
        self.assertEqual(statistics["matches"], 18)
        self.assertEqual(statistics["identity_percent"], 90.0)
        self.assertEqual(
            statistics["identity_denominator"],
            "aligned non-gap residue pairs",
        )

    def test_record_pair_must_be_explicit_and_distinct(self):
        result = align_fasta_records(self.collection, ["protein_a", "protein_a"])

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "INVALID_FASTA_RECORD_IDENTIFIER_PAIR")

    def test_single_record_file_is_rejected_by_multi_parser(self):
        single = PROJECT_ROOT / "tests" / "fixtures" / "fasta" / "single_protein.fasta"
        result = parse_multi_protein_fasta(str(single))

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "MULTI_FASTA_RECORD_COUNT_MISMATCH")

    def test_duplicate_ids_and_invalid_residues_are_rejected(self):
        upload_dir = PROJECT_ROOT / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(upload_dir)) as temp_dir:
            duplicate = Path(temp_dir) / "duplicate.fasta"
            duplicate.write_text(
                ">same\nACDEFG\n>same\nACDEFG\n",
                encoding="utf-8",
            )
            invalid = Path(temp_dir) / "invalid.fasta"
            invalid.write_text(
                ">valid\nACDEFG\n>bad\nACDEXZ\n",
                encoding="utf-8",
            )

            duplicate_result = parse_multi_protein_fasta(str(duplicate))
            invalid_result = parse_multi_protein_fasta(str(invalid))

        self.assertEqual(duplicate_result["code"], "DUPLICATE_FASTA_RECORD_ID")
        self.assertEqual(invalid_result["code"], "INVALID_FASTA_SEQUENCE_CONTENT")
        self.assertEqual(
            invalid_result["issues"][0]["invalid_characters"],
            ["X", "Z"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
