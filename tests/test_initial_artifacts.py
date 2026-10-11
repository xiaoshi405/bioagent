# -*- coding: utf-8 -*-
"""
v0.3b 初始Artifact构建测试。

修改记录：
- 2026-09-02：覆盖结构化代码块、UniProt accession消歧、内部空白规范化
  与Swiss-Prot文件语境，防止合法Dev输入在规划前损坏。
- 2026-08-12：增加显式短非法序列、短合法肽、普通英文、PDB ID、chain ID
  与长蛋白序列回归矩阵，验证候选定位和内容校验职责分离。
- 2026-08-09：覆盖中文前置链ID、非法直接序列、多记录与非法FASTA的
  请求级拒绝，验证内容校验与名义类型检查分层工作。
- 2026-07-26：覆盖直接序列、非法序列候选、FASTA/PDB文件、显式链ID、
  多文件歧义拒绝和脱敏摘要。
- 2026-08-14：多记录FASTA改为独立初始类型，并覆盖显式单record_id、
  record_id pair、缺失ID告警和不存在ID的精确拒绝。
"""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.initial_artifacts import build_initial_artifacts
from scripts.type_checker import (
    AMINO_ACID_SEQUENCE,
    FASTA_RECORD_IDENTIFIER,
    FASTA_RECORD_IDENTIFIER_PAIR,
    MULTI_PROTEIN_FASTA_FILE,
    PDB_CHAIN_IDENTIFIER,
    PDB_STRUCTURE_FILE,
    SINGLE_PROTEIN_FASTA_FILE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FASTA_FIXTURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "fasta" / "single_protein.fasta"
)
MULTI_FASTA_FIXTURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "fasta" / "multi_protein.fasta"
)
PDB_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "pdb" / "1HHB.pdb"
TEST_SEQUENCE = "MKWVTFISLLLLFSSAYSRGVFRR"


class InitialArtifactTests(unittest.TestCase):
    """验证请求开始时只构造确定、可审计的真实数据对象。"""

    def test_direct_sequence_becomes_initial_artifact(self):
        result = build_initial_artifacts(
            f"请计算序列 {TEST_SEQUENCE} 的GRAVY值",
            [],
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["artifacts"][AMINO_ACID_SEQUENCE],
            TEST_SEQUENCE,
        )
        self.assertEqual(
            result["available_input_types"],
            [AMINO_ACID_SEQUENCE],
        )

    def test_invalid_sequence_candidate_is_rejected_before_planning(self):
        candidate = "MKWVTFISLL123X"
        result = build_initial_artifacts(
            f"请计算下面序列的pI：{candidate}",
            [],
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "INVALID_AMINO_ACID_SEQUENCE_CONTENT",
        )
        self.assertEqual(result["issues"][0]["sequence_length"], 14)
        self.assertEqual(result["issues"][0]["invalid_characters"], [
            "1",
            "2",
            "3",
            "X",
        ])
        self.assertIn("标准氨基酸", result["issues"][0]["suggestion"])

    def test_explicit_short_invalid_sequences_are_precisely_rejected(self):
        cases = (
            ("请分析序列 X", "X"),
            ("please analyze sequence: Z", "Z"),
            ("请分析短肽 `X`", "X"),
            ('请分析短肽“Z”', "Z"),
            ("请分析下面代码块：\n```\nX\n```", "X"),
        )
        for question, invalid_character in cases:
            with self.subTest(question=question):
                result = build_initial_artifacts(question, [])
                self.assertEqual(result["status"], "error")
                self.assertEqual(
                    result["issues"][0]["code"],
                    "INVALID_AMINO_ACID_SEQUENCE_CONTENT",
                )
                self.assertEqual(
                    result["issues"][0]["invalid_characters"],
                    [invalid_character],
                )

    def test_explicit_short_valid_peptides_are_registered(self):
        for question, expected in (
            ("请分析序列 ACD", "ACD"),
            ("analyze sequence: WY", "WY"),
            ("请分析短肽 `MKW`", "MKW"),
        ):
            with self.subTest(question=question):
                result = build_initial_artifacts(question, [])
                self.assertEqual(result["status"], "success")
                self.assertEqual(
                    result["artifacts"][AMINO_ACID_SEQUENCE],
                    expected,
                )

    def test_structured_sequence_block_has_priority_over_uniprot_accession(self):
        result = build_initial_artifacts(
            "计算真实蛋白质序列 P69905。\n\n"
            "结构化氨基酸序列输入：\n"
            f"```sequence\n{TEST_SEQUENCE}\n```",
            [],
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["artifacts"][AMINO_ACID_SEQUENCE],
            TEST_SEQUENCE,
        )

    def test_structured_short_sequence_normalizes_case_and_whitespace(self):
        result = build_initial_artifacts(
            "规范化输入后计算。\n\n"
            "结构化氨基酸序列输入：\n"
            "```sequence\n acdefg hik \n```",
            [],
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["artifacts"][AMINO_ACID_SEQUENCE],
            "ACDEFGHIK",
        )

    def test_sequence_code_block_identifier_like_content_is_still_validated(self):
        result = build_initial_artifacts(
            "校验明确标注的序列。\n\n"
            "```sequence\nP69905\n```",
            [],
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "INVALID_AMINO_ACID_SEQUENCE_CONTENT",
        )
        self.assertEqual(
            result["issues"][0]["invalid_characters"],
            ["0", "5", "6", "9"],
        )

    def test_uniprot_accession_without_sequence_is_not_invalid_sequence(self):
        result = build_initial_artifacts(
            "请计算真实蛋白质序列 P69905 的GRAVY。",
            [],
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["issues"][0]["code"], "NO_INITIAL_ARTIFACT")

    def test_swiss_prot_name_does_not_shadow_uploaded_fasta(self):
        result = build_initial_artifacts(
            "解析来自 UniProtKB/Swiss-Prot 的单记录 FASTA 文件。",
            [str(FASTA_FIXTURE)],
        )

        self.assertEqual(result["status"], "success")
        self.assertIn(SINGLE_PROTEIN_FASTA_FILE, result["artifacts"])

    def test_common_english_words_are_not_sequence_candidates(self):
        for question in (
            "Please calculate protein stability.",
            "Could you analyze this ordinary English sentence?",
            "Please perform sequence analysis and report stability.",
            "Report identity, similarity, and gap percentages.",
            "TRANSMISSION",
        ):
            with self.subTest(question=question):
                result = build_initial_artifacts(question, [])
                self.assertEqual(result["status"], "error")
                self.assertEqual(
                    result["issues"][0]["code"],
                    "NO_INITIAL_ARTIFACT",
                )

    def test_pdb_ids_and_chain_ids_are_not_sequence_candidates(self):
        for question in (
            "请分析PDB ID 1ABC",
            "Please analyze PDB `1ABC`.",
            "请分析chain ID A",
            "Please analyze chain `A`.",
            "请分析A链",
        ):
            with self.subTest(question=question):
                result = build_initial_artifacts(question, [])
                self.assertEqual(result["status"], "error")
                self.assertEqual(
                    result["issues"][0]["code"],
                    "NO_INITIAL_ARTIFACT",
                )

    def test_normal_long_protein_sequence_remains_supported(self):
        for sequence in (TEST_SEQUENCE, TEST_SEQUENCE.lower()):
            with self.subTest(sequence=sequence):
                result = build_initial_artifacts(sequence, [])
                self.assertEqual(result["status"], "success")
                self.assertEqual(
                    result["artifacts"][AMINO_ACID_SEQUENCE],
                    TEST_SEQUENCE,
                )

    def test_plain_language_is_not_guessed_as_sequence(self):
        result = build_initial_artifacts(
            "请帮我分析这个蛋白质，但是我还没有提供任何数据。",
            [],
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["issues"][0]["code"], "NO_INITIAL_ARTIFACT")

    def test_single_fasta_file_is_registered(self):
        result = build_initial_artifacts(
            "请计算上传蛋白的GRAVY值",
            [str(FASTA_FIXTURE)],
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["artifacts"][SINGLE_PROTEIN_FASTA_FILE],
            str(FASTA_FIXTURE.resolve()),
        )

    def test_pdb_file_and_explicit_chain_are_registered_together(self):
        result = build_initial_artifacts(
            "请选择chain_id=b并计算该链的性质",
            [str(PDB_FIXTURE)],
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["artifacts"][PDB_STRUCTURE_FILE],
            str(PDB_FIXTURE.resolve()),
        )
        self.assertEqual(
            result["artifacts"][PDB_CHAIN_IDENTIFIER],
            "b",
        )

    def test_chinese_prefix_chain_id_is_registered(self):
        result = build_initial_artifacts(
            "请选择A链并计算该链的性质",
            [str(PDB_FIXTURE)],
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["artifacts"][PDB_CHAIN_IDENTIFIER],
            "A",
        )

    def test_multirecord_fasta_is_registered_without_guessing_record(self):
        result = build_initial_artifacts(
            "请解析上传的多记录蛋白FASTA",
            [str(MULTI_FASTA_FIXTURE)],
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["artifacts"][MULTI_PROTEIN_FASTA_FILE],
            str(MULTI_FASTA_FIXTURE.resolve()),
        )
        self.assertNotIn(FASTA_RECORD_IDENTIFIER, result["artifacts"])
        self.assertEqual(
            result["warnings"][0]["code"],
            "MULTI_FASTA_WITHOUT_RECORD_IDENTIFIER",
        )

    def test_multirecord_fasta_explicit_single_and_pair_ids_are_distinct_types(self):
        single = build_initial_artifacts(
            "请分析record_id=protein_b",
            [str(MULTI_FASTA_FIXTURE)],
        )
        pair = build_initial_artifacts(
            "请比较record_ids=protein_a,protein_b",
            [str(MULTI_FASTA_FIXTURE)],
        )

        self.assertEqual(single["artifacts"][FASTA_RECORD_IDENTIFIER], "protein_b")
        self.assertNotIn(FASTA_RECORD_IDENTIFIER_PAIR, single["artifacts"])
        self.assertEqual(
            pair["artifacts"][FASTA_RECORD_IDENTIFIER_PAIR],
            ["protein_a", "protein_b"],
        )
        self.assertNotIn(FASTA_RECORD_IDENTIFIER, pair["artifacts"])

    def test_unknown_explicit_fasta_record_id_is_rejected(self):
        result = build_initial_artifacts(
            "请分析record_id=missing",
            [str(MULTI_FASTA_FIXTURE)],
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "FASTA_RECORD_IDENTIFIER_NOT_FOUND",
        )
        self.assertEqual(
            result["issues"][0]["available_record_ids"],
            ["protein_a", "protein_b"],
        )

    def test_invalid_fasta_sequence_is_rejected_before_planning(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                fasta = Path(temp_dir) / "invalid.fasta"
                fasta.write_text(
                    ">protein\nMKWVTFISLL123X\n",
                    encoding="utf-8",
                )
                result = build_initial_artifacts(
                    "请解析并计算上传蛋白的GRAVY值",
                    [str(fasta)],
                )
        except Exception as exc:
            self.fail(f"构造非法FASTA测试失败: {exc}")

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "INVALID_FASTA_SEQUENCE_CONTENT",
        )

    def test_chain_id_without_pdb_is_not_registered(self):
        result = build_initial_artifacts(
            f"对序列{TEST_SEQUENCE}分析chain B",
            [],
        )

        self.assertEqual(result["status"], "success")
        self.assertNotIn(PDB_CHAIN_IDENTIFIER, result["artifacts"])
        self.assertEqual(
            result["warnings"][0]["code"],
            "CHAIN_ID_WITHOUT_PDB",
        )

    def test_multiple_same_type_files_are_rejected(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                second_fasta = Path(temp_dir) / "second.fasta"
                second_fasta.write_text(
                    ">protein\nACDEFGHIKLMNPQRSTVWY\n",
                    encoding="utf-8",
                )
                result = build_initial_artifacts(
                    "请分析上传文件",
                    [str(FASTA_FIXTURE), str(second_fasta)],
                )
        except Exception as exc:
            self.fail(f"构造多FASTA测试数据失败: {exc}")

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "AMBIGUOUS_INITIAL_FILE",
        )
        self.assertEqual(result["issues"][0]["candidate_count"], 2)

    def test_missing_supported_file_returns_standard_error(self):
        result = build_initial_artifacts(
            "请分析上传文件",
            ["missing_input.pdb"],
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "INITIAL_FILE_NOT_FOUND",
        )

    def test_safe_summary_contains_neither_sequence_nor_full_path(self):
        result = build_initial_artifacts(
            f"分析序列{TEST_SEQUENCE}和chain B",
            [str(PDB_FIXTURE)],
        )
        safe_text = json.dumps(result["safe_summary"], ensure_ascii=False)

        self.assertEqual(result["status"], "success")
        self.assertNotIn(TEST_SEQUENCE, safe_text)
        self.assertNotIn(str(PDB_FIXTURE.resolve()), safe_text)
        self.assertIn(PDB_FIXTURE.name, safe_text)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    except Exception as exc:
        print(f"初始Artifact测试运行失败: {exc}")
        raise
