# -*- coding: utf-8 -*-
"""
v0.2结构化工具计划适配层测试。

修改记录：
- 2026-09-01：覆盖空计划在E1/E2/E3配置下保留正交开关证据。
- 2026-07-25：覆盖fan-out、FASTA/PDB桥接、多输入来源、来源歧义、
  保守修复、配置旁路和旧LLM Plan Chain确定性解析。
"""

import unittest

from app.llms.extraction import extract_tool_path
from scripts.structured_plan import (
    INITIAL_INPUT_SOURCE,
    build_structured_plan,
    parse_llm_plan_response,
    prepare_structured_plan,
    prepare_structured_plan_from_config,
)
from scripts.type_checker import (
    AMINO_ACID_SEQUENCE,
    FASTA_TO_SEQUENCE_BRIDGE,
    PDB_CHAIN_IDENTIFIER,
    PDB_STRUCTURE_FILE,
    PDB_TO_SEQUENCE_BRIDGE,
    SINGLE_PROTEIN_FASTA_FILE,
)


class StructuredPlanBuildTests(unittest.TestCase):
    """验证合法旧计划能被唯一、确定地绑定到输入来源。"""

    def test_single_tool_uses_initial_sequence(self):
        result = build_structured_plan(
            ["compute_gravy"],
            [AMINO_ACID_SEQUENCE],
        )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["valid"])
        step = result["structured_plan"][0]
        self.assertEqual(step["step_id"], "step_1")
        self.assertEqual(step["input_type"], [AMINO_ACID_SEQUENCE])
        self.assertEqual(
            step["input_from"],
            {AMINO_ACID_SEQUENCE: INITIAL_INPUT_SOURCE},
        )

    def test_fan_out_steps_reuse_same_initial_sequence(self):
        result = build_structured_plan(
            ["compute_pI_mw", "compute_gravy"],
            [AMINO_ACID_SEQUENCE],
        )

        self.assertTrue(result["valid"])
        self.assertEqual(
            result["structured_plan"][0]["input_from"][AMINO_ACID_SEQUENCE],
            INITIAL_INPUT_SOURCE,
        )
        self.assertEqual(
            result["structured_plan"][1]["input_from"][AMINO_ACID_SEQUENCE],
            INITIAL_INPUT_SOURCE,
        )

    def test_fasta_bridge_output_becomes_step_source(self):
        result = build_structured_plan(
            [FASTA_TO_SEQUENCE_BRIDGE, "compute_gravy"],
            [SINGLE_PROTEIN_FASTA_FILE],
        )

        self.assertTrue(result["valid"])
        first, second = result["structured_plan"]
        self.assertEqual(
            first["input_from"],
            {SINGLE_PROTEIN_FASTA_FILE: INITIAL_INPUT_SOURCE},
        )
        self.assertEqual(
            second["input_from"],
            {AMINO_ACID_SEQUENCE: "step_1"},
        )

    def test_pdb_bridge_records_both_initial_inputs(self):
        result = build_structured_plan(
            [PDB_TO_SEQUENCE_BRIDGE, "compute_gravy"],
            [PDB_STRUCTURE_FILE, PDB_CHAIN_IDENTIFIER],
        )

        self.assertTrue(result["valid"])
        first, second = result["structured_plan"]
        self.assertEqual(
            first["input_type"],
            [PDB_STRUCTURE_FILE, PDB_CHAIN_IDENTIFIER],
        )
        self.assertEqual(
            first["input_from"],
            {
                PDB_STRUCTURE_FILE: INITIAL_INPUT_SOURCE,
                PDB_CHAIN_IDENTIFIER: INITIAL_INPUT_SOURCE,
            },
        )
        self.assertEqual(
            second["input_from"][AMINO_ACID_SEQUENCE],
            "step_1",
        )

    def test_multiple_objects_of_same_type_are_not_guessed(self):
        result = build_structured_plan(
            [FASTA_TO_SEQUENCE_BRIDGE, "compute_gravy"],
            [SINGLE_PROTEIN_FASTA_FILE, AMINO_ACID_SEQUENCE],
        )

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["valid"])
        self.assertEqual(
            result["issues"][0]["code"],
            "AMBIGUOUS_INPUT_SOURCE",
        )
        self.assertEqual(
            result["issues"][0]["source_candidates"],
            [INITIAL_INPUT_SOURCE, f"step:0:{FASTA_TO_SEQUENCE_BRIDGE}"],
        )

    def test_type_invalid_plan_is_not_partially_structured(self):
        result = build_structured_plan(
            ["compute_gravy"],
            [PDB_STRUCTURE_FILE],
        )

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["valid"])
        self.assertEqual(result["structured_plan"], [])
        self.assertEqual(result["issues"][0]["code"], "MISSING_INPUT_TYPE")

    def test_invalid_request_returns_standard_error(self):
        result = build_structured_plan(None, None)

        self.assertEqual(result["status"], "error")
        self.assertFalse(result["valid"])
        self.assertEqual(result["issues"][0]["code"], "INVALID_PLAN")


class StructuredPlanRepairTests(unittest.TestCase):
    """验证修复开关只插入v0.1已验收的确定性桥接工具。"""

    def test_repair_disabled_preserves_invalid_plan_trace(self):
        result = prepare_structured_plan(
            ["compute_gravy"],
            [SINGLE_PROTEIN_FASTA_FILE],
            enable_repair=False,
        )

        self.assertFalse(result["valid"])
        self.assertFalse(result["repair_attempted"])
        self.assertEqual(result["plan_before"], ["compute_gravy"])
        self.assertEqual(result["plan_after"], ["compute_gravy"])

    def test_repair_enabled_marks_inserted_fasta_bridge(self):
        result = prepare_structured_plan(
            ["compute_gravy"],
            [SINGLE_PROTEIN_FASTA_FILE],
            enable_repair=True,
        )

        self.assertTrue(result["valid"])
        self.assertTrue(result["repair_attempted"])
        self.assertTrue(result["repair_applied"])
        self.assertEqual(
            result["plan_after"],
            [FASTA_TO_SEQUENCE_BRIDGE, "compute_gravy"],
        )
        self.assertTrue(result["structured_plan"][0]["inserted_by_repair"])
        self.assertFalse(result["structured_plan"][1]["inserted_by_repair"])

    def test_repair_enabled_marks_inserted_pdb_bridge(self):
        result = prepare_structured_plan(
            ["compute_gravy"],
            [PDB_STRUCTURE_FILE, PDB_CHAIN_IDENTIFIER],
            enable_repair=True,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(
            result["plan_after"],
            [PDB_TO_SEQUENCE_BRIDGE, "compute_gravy"],
        )
        self.assertTrue(result["structured_plan"][0]["inserted_by_repair"])


class StructuredPlanResponseAndConfigTests(unittest.TestCase):
    """验证旧LLM文本解析和统一配置开关。"""

    def test_llm_plan_response_is_parsed_without_second_llm(self):
        response = (
            "Analysis: use two local tools.\n"
            "Plan Chain: ['compute_pI_mw', 'compute_gravy']"
        )
        result = parse_llm_plan_response(
            response,
            [AMINO_ACID_SEQUENCE],
        )

        self.assertTrue(result["valid"])
        self.assertEqual(
            result["plan_before"],
            ["compute_pI_mw", "compute_gravy"],
        )
        self.assertEqual(len(result["structured_plan"]), 2)

    def test_malformed_llm_plan_returns_standard_error(self):
        result = parse_llm_plan_response(
            "Analysis only, no plan chain.",
            [AMINO_ACID_SEQUENCE],
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "PLAN_EXTRACTION_FAILED",
        )

    def test_extract_tool_path_rejects_non_string_tool(self):
        result = extract_tool_path(
            "Plan Chain: ['compute_gravy', 7]"
        )
        self.assertEqual(result, [])

    def test_disabled_config_bypasses_without_claiming_validity(self):
        class DisabledConfig:
            TYPE_PLANNING_ENABLED = False
            TYPE_REPAIR_ENABLED = False

        result = prepare_structured_plan_from_config(
            ["compute_gravy"],
            [AMINO_ACID_SEQUENCE],
            config=DisabledConfig(),
        )

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["enabled"])
        self.assertTrue(result["bypassed"])
        self.assertIsNone(result["valid"])
        self.assertEqual(result["plan_after"], ["compute_gravy"])

    def test_enabled_config_can_apply_conservative_repair(self):
        class EnabledConfig:
            TYPE_PLANNING_ENABLED = True
            TYPE_REPAIR_ENABLED = True

        result = prepare_structured_plan_from_config(
            ["compute_gravy"],
            [SINGLE_PROTEIN_FASTA_FILE],
            config=EnabledConfig(),
        )

        self.assertTrue(result["enabled"])
        self.assertFalse(result["bypassed"])
        self.assertTrue(result["valid"])
        self.assertTrue(result["repair_applied"])

    def test_empty_plan_preserves_factorial_condition_switches(self):
        """空计划必须保留E1/E2/E3开关，供条件Trace一致性校验。"""
        cases = (
            (False, False),
            (True, False),
            (True, True),
        )
        for validation_enabled, repair_enabled in cases:
            with self.subTest(
                validation_enabled=validation_enabled,
                repair_enabled=repair_enabled,
            ):
                config = type(
                    "FactorialConfig",
                    (),
                    {
                        "TYPE_PLANNING_ENABLED": True,
                        "TYPE_VALIDATION_ENABLED": validation_enabled,
                        "TYPE_REPAIR_ENABLED": repair_enabled,
                    },
                )()
                result = prepare_structured_plan_from_config(
                    [],
                    [AMINO_ACID_SEQUENCE],
                    config=config,
                )

                self.assertEqual(result["status"], "error")
                self.assertEqual(result["issues"][0]["code"], "INVALID_PLAN")
                self.assertTrue(result["enabled"])
                self.assertEqual(
                    result["validation_enabled"],
                    validation_enabled,
                )
                self.assertEqual(result["repair_enabled"], repair_enabled)
                self.assertFalse(result["repair_attempted"])
                self.assertFalse(result["repair_applied"])
                self.assertEqual(result["plan_before"], [])
                self.assertEqual(result["plan_after"], [])


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    except Exception as exc:
        print(f"结构化计划测试运行失败: {exc}")
        raise
