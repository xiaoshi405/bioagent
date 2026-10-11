# -*- coding: utf-8 -*-
"""
v0.3a类型化Artifact Pool测试。

修改记录：
- 2026-07-26：覆盖真实值保存、精确来源解析、输出字段绑定、原子写入、
  重复对象拒绝和脱敏快照。
"""

import json
import unittest

from scripts.artifact_pool import ArtifactPool, INITIAL_INPUT_SOURCE
from ToolsAgent.ToolsFuns.Biology.tool_name_dict import BIOLOGY_TOOLS_DESC


SEQUENCE_TYPE = "amino acid sequence"
FASTA_TYPE = "single-record protein FASTA file"


class ArtifactPoolTests(unittest.TestCase):
    """验证对象池真实保存数据，但审计快照不泄漏真实值。"""

    def test_initial_value_can_be_resolved_exactly(self):
        pool = ArtifactPool()
        sequence = "MKWVTFISLLLLFSSAYSRGVFRR"

        registered = pool.register_initial({SEQUENCE_TYPE: sequence})
        resolved = pool.resolve(SEQUENCE_TYPE, INITIAL_INPUT_SOURCE)

        self.assertEqual(registered["status"], "success")
        self.assertEqual(resolved["status"], "success")
        self.assertEqual(resolved["value"], sequence)
        self.assertEqual(resolved["artifact"]["value_size"], len(sequence))

    def test_snapshot_does_not_contain_sequence_or_file_path(self):
        pool = ArtifactPool()
        sequence = "MKWVTFISLLLLFSSAYSRGVFRR"
        file_path = "D:/private/user/sample.fasta"
        pool.register_initial(
            {
                SEQUENCE_TYPE: sequence,
                FASTA_TYPE: file_path,
            }
        )

        serialized = json.dumps(pool.snapshot(), ensure_ascii=False)

        self.assertNotIn(sequence, serialized)
        self.assertNotIn(file_path, serialized)
        self.assertIn(SEQUENCE_TYPE, serialized)
        self.assertIn(FASTA_TYPE, serialized)

    def test_duplicate_type_and_source_is_rejected(self):
        pool = ArtifactPool()
        first = pool.register(SEQUENCE_TYPE, "ACDEFGHIKL", INITIAL_INPUT_SOURCE)
        second = pool.register(SEQUENCE_TYPE, "MNPQRSTVWY", INITIAL_INPUT_SOURCE)

        self.assertEqual(first["status"], "success")
        self.assertEqual(second["status"], "error")
        self.assertEqual(second["issues"][0]["code"], "DUPLICATE_ARTIFACT")

    def test_fasta_tool_output_registers_real_sequence(self):
        pool = ArtifactPool()
        result = pool.register_tool_outputs(
            "step_1",
            "parse_protein_fasta",
            [SEQUENCE_TYPE],
            {
                "status": "success",
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
                "sequence_length": 20,
            },
        )
        resolved = pool.resolve(SEQUENCE_TYPE, "step_1")

        self.assertEqual(result["status"], "success")
        self.assertEqual(resolved["value"], "ACDEFGHIKLMNPQRSTVWY")

    def test_failed_tool_does_not_create_output_artifact(self):
        pool = ArtifactPool()
        result = pool.register_tool_outputs(
            "step_1",
            "parse_protein_fasta",
            [SEQUENCE_TYPE],
            {"status": "error", "message": "多记录FASTA不允许自动选择"},
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["issues"][0]["code"], "TOOL_EXECUTION_FAILED")
        self.assertEqual(pool.snapshot()["artifact_count"], 0)

    def test_missing_one_declared_output_is_atomic(self):
        pool = ArtifactPool()
        result = pool.register_tool_outputs(
            "step_1",
            "compute_pI_mw",
            ["pI", "molecular weight"],
            {"status": "success", "pI": 5.4},
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["issues"][0]["code"], "DECLARED_OUTPUT_MISSING")
        self.assertEqual(pool.snapshot()["artifact_count"], 0)

    def test_nested_protein_analysis_bindings_are_extracted(self):
        pool = ArtifactPool()
        output_types = BIOLOGY_TOOLS_DESC["protein_analysis"]["output_types"]
        result = pool.register_tool_outputs(
            "step_1",
            "protein_analysis",
            output_types,
            {
                "status": "success",
                "basic_info": {
                    "status": "success",
                    "pI": 6.2,
                    "molecular_weight": 1234.5,
                },
                "signal_peptide": {"status": "success", "has_signal_peptide": False},
                "transmembrane": {
                    "status": "success",
                    "transmembrane_regions": [],
                },
                "solubility": {
                    "status": "success",
                    "solubility_score": 73,
                },
            },
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(pool.resolve("pI", "step_1")["value"], 6.2)
        self.assertEqual(
            pool.resolve("molecular weight", "step_1")["value"],
            1234.5,
        )
        self.assertEqual(
            pool.resolve("solubility score", "step_1")["value"],
            73,
        )

    def test_nested_failed_output_prevents_all_artifact_registration(self):
        pool = ArtifactPool()
        output_types = BIOLOGY_TOOLS_DESC["protein_analysis"]["output_types"]
        result = pool.register_tool_outputs(
            "step_1",
            "protein_analysis",
            output_types,
            {
                "status": "success",
                "basic_info": {
                    "status": "success",
                    "pI": 6.2,
                    "molecular_weight": 1234.5,
                },
                "signal_peptide": {
                    "status": "error",
                    "message": "信号肽子工具失败",
                },
                "transmembrane": {
                    "status": "success",
                    "transmembrane_regions": [],
                },
                "solubility": {
                    "status": "success",
                    "solubility_score": 73,
                },
            },
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["issues"][0]["code"], "DECLARED_OUTPUT_FAILED")
        self.assertEqual(pool.snapshot()["artifact_count"], 0)

    def test_plan_output_contract_mismatch_is_rejected(self):
        pool = ArtifactPool()
        result = pool.register_tool_outputs(
            "step_1",
            "compute_gravy",
            ["wrong output type"],
            {"status": "success", "gravy": -0.3},
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["issues"][0]["code"], "OUTPUT_CONTRACT_MISMATCH")

    def test_every_external_tool_has_complete_output_bindings(self):
        for tool_name, info in BIOLOGY_TOOLS_DESC.items():
            if info.get("is_internal"):
                continue
            with self.subTest(tool_name=tool_name):
                self.assertEqual(
                    set(info.get("output_types") or []),
                    set((info.get("output_bindings") or {}).keys()),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
