# -*- coding: utf-8 -*-
"""
单记录 Protein FASTA 本地解析测试。

修改记录：
- 2026-07-24：覆盖合法解析、多记录拒绝、非法字符、路径边界和Runner文件选择。
- 2026-08-14：断言统一FASTA错误码、非法字符证据和多记录迁移建议。
"""

import tempfile
import unittest
from pathlib import Path

from ToolsAgent.ToolsFuns.Biology.fasta_analysis import (
    PROJECT_ROOT,
    parse_protein_fasta,
)
from app.tools.tool_runner import select_fasta_file


class ProteinFASTAAnalysisTests(unittest.TestCase):
    """验证Protein FASTA解析严格而确定，不静默猜测记录或清洗字符。"""

    def setUp(self):
        try:
            upload_dir = PROJECT_ROOT / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir = tempfile.TemporaryDirectory(dir=str(upload_dir))
            self.temp_path = Path(self.temp_dir.name)
        except Exception:
            if hasattr(self, "temp_dir"):
                self.temp_dir.cleanup()
            raise

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def _write(self, name: str, content: str) -> Path:
        """在项目受控上传目录中创建临时FASTA测试文件。"""
        try:
            path = self.temp_path / name
            path.write_text(content, encoding="utf-8")
            return path
        except Exception:
            raise

    def test_valid_multiline_single_record_is_parsed(self):
        path = self._write(
            "protein.fasta",
            ">sp|PTEST|example protein\nMKWVTFISLL\nFLFSSAYSR\n",
        )

        result = parse_protein_fasta(str(path))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["record_id"], "sp|PTEST|example")
        self.assertEqual(result["sequence"], "MKWVTFISLLFLFSSAYSR")
        self.assertEqual(result["sequence_length"], 19)

    def test_multiple_records_are_rejected(self):
        path = self._write(
            "multiple.fa",
            ">protein_a\nACDEFGHIKL\n>protein_b\nMNPQRSTVWY\n",
        )

        result = parse_protein_fasta(str(path))

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "FASTA_RECORD_COUNT_MISMATCH")
        self.assertIn("parse_multi_protein_fasta", result["issues"][0]["suggestion"])

    def test_nonstandard_amino_acids_are_rejected_without_cleaning(self):
        path = self._write("invalid.faa", ">protein_x\nMKWVTFISLLX123\n")

        result = parse_protein_fasta(str(path))

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "INVALID_FASTA_SEQUENCE_CONTENT")
        self.assertEqual(
            result["issues"][0]["invalid_characters"],
            ["1", "2", "3", "X"],
        )

    def test_missing_header_is_rejected(self):
        path = self._write("missing_header.fasta", "MKWVTFISLLFLFSSAYSR\n")

        result = parse_protein_fasta(str(path))

        self.assertEqual(result["status"], "error")
        self.assertIn("必须以 > 开头", result["message"])

    def test_wrong_extension_is_rejected(self):
        path = self._write("protein.txt", ">protein\nACDEFGHIKL\n")

        result = parse_protein_fasta(str(path))

        self.assertEqual(result["status"], "error")
        self.assertIn("仅支持", result["message"])

    def test_project_directory_boundary_is_enforced(self):
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_path = Path(outside_dir) / "outside.fasta"
            outside_path.write_text(">outside\nACDEFGHIKL\n", encoding="utf-8")

            result = parse_protein_fasta(str(outside_path))

        self.assertEqual(result["status"], "error")
        self.assertIn("项目目录内", result["message"])

    def test_runner_selects_only_existing_fasta_suffix(self):
        text_path = self._write("ignore.txt", ">protein\nACDEFGHIKL\n")
        fasta_path = self._write("selected.fa", ">protein\nACDEFGHIKL\n")

        selected = select_fasta_file([str(text_path), str(fasta_path)])

        self.assertEqual(selected, str(fasta_path.resolve()))


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    except Exception as exc:
        print(f"Protein FASTA测试运行失败: {exc}")
        raise
