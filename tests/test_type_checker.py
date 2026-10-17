# -*- coding: utf-8 -*-
"""Tests for type checker."""

import unittest

from scripts.type_checker import (
    AMINO_ACID_SEQUENCE,
    ALIGNMENT_STATISTICS,
    FASTA_COLLECTION_TO_ALIGNMENT_BRIDGE,
    FASTA_COLLECTION_TO_SEQUENCE_BRIDGE,
    FASTA_RECORD_IDENTIFIER,
    FASTA_RECORD_IDENTIFIER_PAIR,
    FASTA_TO_SEQUENCE_BRIDGE,
    PDB_CHAIN_IDENTIFIER,
    PDB_CHAIN_SEQUENCE_MAP,
    PDB_STRUCTURE_FILE,
    PDB_TO_SEQUENCE_BRIDGE,
    MULTI_FASTA_TO_COLLECTION_BRIDGE,
    MULTI_PROTEIN_FASTA_FILE,
    PAIRWISE_PROTEIN_ALIGNMENT,
    PROTEIN_SEQUENCE_COLLECTION,
    SINGLE_PROTEIN_FASTA_FILE,
    get_tool_signatures,
    repair_tool_plan,
    validate_tool_plan,
)
from ToolsAgent.ToolsFuns.Biology.tool_name_dict import BIOLOGY_TOOLS_DESC


class ToolSignatureTests(unittest.TestCase):

    def test_registry_contains_twenty_external_tools(self):
        result = get_tool_signatures()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool_count"], 20)
        self.assertNotIn("send_expasy_request", result["signatures"])

    def test_sequence_and_pdb_signatures_are_distinct(self):
        signatures = get_tool_signatures()["signatures"]
        self.assertEqual(
            signatures["compute_gravy"]["input_types"],
            [AMINO_ACID_SEQUENCE],
        )
        self.assertEqual(
            signatures["analyze_pdb_structure"]["input_types"],
            [PDB_STRUCTURE_FILE],
        )
        self.assertEqual(
            signatures["parse_protein_fasta"]["input_types"],
            [SINGLE_PROTEIN_FASTA_FILE],
        )
        self.assertEqual(
            signatures["extract_pdb_sequences"]["output_types"],
            [PDB_CHAIN_SEQUENCE_MAP],
        )
        self.assertEqual(
            signatures[PDB_TO_SEQUENCE_BRIDGE]["input_types"],
            [PDB_STRUCTURE_FILE, PDB_CHAIN_IDENTIFIER],
        )
        self.assertEqual(
            signatures[MULTI_FASTA_TO_COLLECTION_BRIDGE]["output_types"],
            [PROTEIN_SEQUENCE_COLLECTION],
        )
        self.assertEqual(
            signatures[FASTA_COLLECTION_TO_SEQUENCE_BRIDGE]["input_types"],
            [PROTEIN_SEQUENCE_COLLECTION, FASTA_RECORD_IDENTIFIER],
        )
        self.assertTrue(
            signatures[FASTA_COLLECTION_TO_SEQUENCE_BRIDGE]["repair_policy"][
                "allow_duplicate_invalid_prune"
            ]
        )
        self.assertEqual(
            signatures[FASTA_COLLECTION_TO_ALIGNMENT_BRIDGE]["output_types"],
            [PAIRWISE_PROTEIN_ALIGNMENT],
        )
        self.assertEqual(
            signatures["summarize_pairwise_alignment"]["output_types"],
            [ALIGNMENT_STATISTICS],
        )

    def test_all_external_tools_explicitly_declare_input_types(self):
        external_tools = {
            tool_name: info
            for tool_name, info in BIOLOGY_TOOLS_DESC.items()
            if isinstance(info, dict) and not info.get("is_internal")
        }
        missing = [
            tool_name
            for tool_name, info in external_tools.items()
            if not info.get("input_types")
        ]
        self.assertEqual(missing, [])

    def test_all_signatures_have_api_path_and_output_types(self):
        signatures = get_tool_signatures()["signatures"]
        incomplete = [
            tool_name
            for tool_name, signature in signatures.items()
            if not signature.get("api_path") or not signature.get("output_types")
        ]
        self.assertEqual(incomplete, [])

    def test_v2_type_topology_has_five_registered_bridges(self):
        signatures = get_tool_signatures()["signatures"]
        input_type_universe = {
            input_type
            for signature in signatures.values()
            for input_type in signature["input_types"]
        }
        bridge_tools = [
            tool_name
            for tool_name, signature in signatures.items()
            if input_type_universe.intersection(signature["output_types"])
        ]

        self.assertEqual(
            input_type_universe,
            {
                AMINO_ACID_SEQUENCE,
                PDB_STRUCTURE_FILE,
                PDB_CHAIN_IDENTIFIER,
                SINGLE_PROTEIN_FASTA_FILE,
                MULTI_PROTEIN_FASTA_FILE,
                PROTEIN_SEQUENCE_COLLECTION,
                FASTA_RECORD_IDENTIFIER,
                FASTA_RECORD_IDENTIFIER_PAIR,
                PAIRWISE_PROTEIN_ALIGNMENT,
            },
        )
        self.assertEqual(
            bridge_tools,
            [
                "parse_protein_fasta",
                MULTI_FASTA_TO_COLLECTION_BRIDGE,
                FASTA_COLLECTION_TO_SEQUENCE_BRIDGE,
                FASTA_COLLECTION_TO_ALIGNMENT_BRIDGE,
                PDB_TO_SEQUENCE_BRIDGE,
            ],
        )
        deterministic_bridges = [
            tool_name
            for tool_name, signature in signatures.items()
            if isinstance(signature.get("repair_policy"), dict)
        ]
        self.assertEqual(deterministic_bridges, bridge_tools)


class ToolPlanValidationTests(unittest.TestCase):

    def test_multiple_sequence_tools_reuse_initial_sequence(self):
        result = validate_tool_plan(
            ["compute_pI_mw", "compute_gravy", "compute_instability_index"],
            [AMINO_ACID_SEQUENCE],
        )
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["valid"])
        self.assertTrue(all(step["valid"] for step in result["steps"]))
        self.assertEqual(
            result["steps"][1]["input_sources"][AMINO_ACID_SEQUENCE],
            ["initial_input"],
        )

    def test_pdb_tools_reuse_uploaded_pdb(self):
        result = validate_tool_plan(
            ["analyze_pdb_structure", "calculate_protein_anm"],
            [PDB_STRUCTURE_FILE],
        )
        self.assertTrue(result["valid"])

    def test_multichain_extraction_does_not_produce_single_sequence(self):
        result = validate_tool_plan(
            ["extract_pdb_sequences", "compute_gravy"],
            [PDB_STRUCTURE_FILE],
        )
        self.assertFalse(result["valid"])
        self.assertIn(PDB_CHAIN_SEQUENCE_MAP, result["available_types"])
        self.assertNotIn(AMINO_ACID_SEQUENCE, result["available_types"])

    def test_explicit_pdb_chain_selection_bridge_is_valid(self):
        result = validate_tool_plan(
            [PDB_TO_SEQUENCE_BRIDGE, "compute_gravy"],
            [PDB_STRUCTURE_FILE, PDB_CHAIN_IDENTIFIER],
        )
        self.assertTrue(result["valid"])
        self.assertIn(AMINO_ACID_SEQUENCE, result["available_types"])
        self.assertIn(
            f"step:0:{PDB_TO_SEQUENCE_BRIDGE}",
            result["steps"][1]["input_sources"][AMINO_ACID_SEQUENCE],
        )

    def test_pdb_chain_selection_requires_both_file_and_chain_id(self):
        result = validate_tool_plan(
            [PDB_TO_SEQUENCE_BRIDGE],
            [PDB_STRUCTURE_FILE],
        )
        self.assertFalse(result["valid"])
        self.assertEqual(
            result["issues"][0]["missing_types"],
            [PDB_CHAIN_IDENTIFIER],
        )

    def test_explicit_fasta_to_sequence_bridge_is_valid(self):
        result = validate_tool_plan(
            [FASTA_TO_SEQUENCE_BRIDGE, "compute_gravy"],
            [SINGLE_PROTEIN_FASTA_FILE],
        )
        self.assertTrue(result["valid"])
        self.assertIn(AMINO_ACID_SEQUENCE, result["available_types"])
        self.assertIn(
            f"step:0:{FASTA_TO_SEQUENCE_BRIDGE}",
            result["steps"][1]["input_sources"][AMINO_ACID_SEQUENCE],
        )

    def test_sequence_tool_without_sequence_or_bridge_is_rejected(self):
        result = validate_tool_plan(["compute_gravy"], [PDB_STRUCTURE_FILE])
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["valid"])
        self.assertEqual(result["issues"][0]["code"], "MISSING_INPUT_TYPE")

    def test_pdb_tool_cannot_consume_sequence(self):
        result = validate_tool_plan(
            ["analyze_pdb_structure"],
            [AMINO_ACID_SEQUENCE],
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["issues"][0]["code"], "MISSING_INPUT_TYPE")

    def test_unknown_tool_is_rejected_as_type_i_signal(self):
        result = validate_tool_plan(
            ["ImaginaryProteinTool"],
            [AMINO_ACID_SEQUENCE],
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["issues"][0]["code"], "UNKNOWN_TOOL")

    def test_internal_tool_is_not_available_to_planner(self):
        result = validate_tool_plan(
            ["send_expasy_request"],
            [AMINO_ACID_SEQUENCE],
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["issues"][0]["code"], "UNKNOWN_TOOL")

    def test_bad_request_returns_standard_error_without_exception(self):
        result = validate_tool_plan(None, None)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["valid"])
        self.assertEqual(result["issues"][0]["code"], "INVALID_PLAN")


class ToolPlanRepairTests(unittest.TestCase):

    def test_prunes_duplicate_invalid_record_selections_with_proof(self):
        plan = [
            MULTI_FASTA_TO_COLLECTION_BRIDGE,
            FASTA_COLLECTION_TO_SEQUENCE_BRIDGE,
            FASTA_COLLECTION_TO_SEQUENCE_BRIDGE,
            FASTA_COLLECTION_TO_ALIGNMENT_BRIDGE,
            "summarize_pairwise_alignment",
        ]
        result = repair_tool_plan(
            plan,
            [MULTI_PROTEIN_FASTA_FILE, FASTA_RECORD_IDENTIFIER_PAIR],
        )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["repairable"])
        self.assertTrue(result["changed"])
        self.assertEqual(
            result["repair_strategy"],
            "duplicate_invalid_step_pruning",
        )
        self.assertEqual(
            result["repaired_plan"],
            [
                MULTI_FASTA_TO_COLLECTION_BRIDGE,
                FASTA_COLLECTION_TO_ALIGNMENT_BRIDGE,
                "summarize_pairwise_alignment",
            ],
        )
        self.assertEqual(result["inserted_steps"], [])
        self.assertEqual(
            [item["original_step_index"] for item in result["removed_steps"]],
            [1, 2],
        )
        self.assertTrue(result["validation"]["valid"])
        for operation in result["repair_operations"]:
            self.assertEqual(operation["operation"], "redundant_step_prune")
            self.assertEqual(operation["issue_code"], "MISSING_INPUT_TYPE")
            self.assertTrue(operation["registry_opt_in"])
            self.assertEqual(operation["duplicate_group_size"], 2)
            self.assertFalse(operation["output_consumed_downstream"])
            self.assertTrue(operation["candidate_plan_type_valid"])
            self.assertNotIn(
                AMINO_ACID_SEQUENCE,
                operation["downstream_required_types"],
            )

    def test_does_not_prune_when_duplicate_output_is_consumed_downstream(self):
        result = repair_tool_plan(
            [
                MULTI_FASTA_TO_COLLECTION_BRIDGE,
                FASTA_COLLECTION_TO_SEQUENCE_BRIDGE,
                FASTA_COLLECTION_TO_SEQUENCE_BRIDGE,
                "compute_gravy",
            ],
            [MULTI_PROTEIN_FASTA_FILE, FASTA_RECORD_IDENTIFIER_PAIR],
        )

        self.assertFalse(result["repairable"])
        self.assertIsNone(result["repaired_plan"])
        self.assertEqual(result["removed_steps"], [])
        self.assertEqual(result["repair_operations"], [])

    def test_does_not_prune_single_invalid_terminal_selection(self):
        result = repair_tool_plan(
            [
                MULTI_FASTA_TO_COLLECTION_BRIDGE,
                FASTA_COLLECTION_TO_SEQUENCE_BRIDGE,
            ],
            [MULTI_PROTEIN_FASTA_FILE, FASTA_RECORD_IDENTIFIER_PAIR],
        )

        self.assertFalse(result["repairable"])
        self.assertIsNone(result["repaired_plan"])
        self.assertEqual(result["removed_steps"], [])

    def test_repair_inserts_pdb_sequence_bridge(self):
        result = repair_tool_plan(
            ["compute_gravy", "compute_pI_mw"],
            [PDB_STRUCTURE_FILE, PDB_CHAIN_IDENTIFIER],
        )
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["repairable"])
        self.assertTrue(result["changed"])
        self.assertEqual(
            result["repaired_plan"],
            [PDB_TO_SEQUENCE_BRIDGE, "compute_gravy", "compute_pI_mw"],
        )
        self.assertEqual(len(result["inserted_steps"]), 1)
        self.assertTrue(result["validation"]["valid"])

    def test_valid_plan_is_not_changed(self):
        plan = ["compute_gravy", "compute_pI_mw"]
        result = repair_tool_plan(plan, [AMINO_ACID_SEQUENCE])
        self.assertTrue(result["repairable"])
        self.assertFalse(result["changed"])
        self.assertEqual(result["repaired_plan"], plan)

    def test_repair_is_idempotent_after_bridge_insertion(self):
        first_result = repair_tool_plan(
            ["compute_gravy", "compute_pI_mw"],
            [PDB_STRUCTURE_FILE, PDB_CHAIN_IDENTIFIER],
        )
        second_result = repair_tool_plan(
            first_result["repaired_plan"],
            [PDB_STRUCTURE_FILE, PDB_CHAIN_IDENTIFIER],
        )

        self.assertTrue(first_result["changed"])
        self.assertFalse(second_result["changed"])
        self.assertEqual(
            second_result["repaired_plan"],
            first_result["repaired_plan"],
        )
        self.assertEqual(second_result["inserted_steps"], [])

    def test_repair_inserts_fasta_sequence_bridge(self):
        result = repair_tool_plan(
            ["compute_gravy", "compute_pI_mw"],
            [SINGLE_PROTEIN_FASTA_FILE],
        )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["repairable"])
        self.assertEqual(
            result["repaired_plan"],
            [FASTA_TO_SEQUENCE_BRIDGE, "compute_gravy", "compute_pI_mw"],
        )
        self.assertEqual(len(result["inserted_steps"]), 1)
        self.assertTrue(result["validation"]["valid"])

    def test_repair_inserts_two_hop_multi_fasta_record_path(self):
        result = repair_tool_plan(
            ["compute_gravy"],
            [MULTI_PROTEIN_FASTA_FILE, FASTA_RECORD_IDENTIFIER],
        )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["repairable"])
        self.assertEqual(
            result["repaired_plan"],
            [
                MULTI_FASTA_TO_COLLECTION_BRIDGE,
                FASTA_COLLECTION_TO_SEQUENCE_BRIDGE,
                "compute_gravy",
            ],
        )
        self.assertEqual(len(result["inserted_steps"]), 2)
        self.assertEqual(len(result["repair_operations"]), 2)
        self.assertTrue(
            all(
                item["output_consumed_downstream"]
                for item in result["repair_operations"]
            )
        )
        self.assertTrue(result["validation"]["valid"])

    def test_repair_inserts_two_hop_alignment_summary_path(self):
        result = repair_tool_plan(
            ["summarize_pairwise_alignment"],
            [MULTI_PROTEIN_FASTA_FILE, FASTA_RECORD_IDENTIFIER_PAIR],
        )

        self.assertTrue(result["repairable"])
        self.assertEqual(
            result["repaired_plan"],
            [
                MULTI_FASTA_TO_COLLECTION_BRIDGE,
                FASTA_COLLECTION_TO_ALIGNMENT_BRIDGE,
                "summarize_pairwise_alignment",
            ],
        )

    def test_multi_fasta_without_explicit_record_id_is_not_repaired(self):
        result = repair_tool_plan(
            ["compute_gravy"],
            [MULTI_PROTEIN_FASTA_FILE],
        )

        self.assertFalse(result["repairable"])
        self.assertIsNone(result["repaired_plan"])

    def test_multiple_file_bridges_are_not_guessed(self):
        result = repair_tool_plan(
            ["compute_gravy"],
            [
                PDB_STRUCTURE_FILE,
                PDB_CHAIN_IDENTIFIER,
                SINGLE_PROTEIN_FASTA_FILE,
            ],
        )

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["repairable"])
        self.assertIsNone(result["repaired_plan"])
        self.assertEqual(
            len(result["blocking_issue"]["bridge_candidates"]),
            2,
        )
        self.assertIn("多个", result["reason"])

    def test_pdb_without_chain_id_is_not_auto_repaired(self):
        result = repair_tool_plan(
            ["compute_gravy"],
            [PDB_STRUCTURE_FILE],
        )

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["repairable"])
        self.assertIsNone(result["repaired_plan"])
        self.assertEqual(
            result["blocking_issue"]["bridge_candidates"],
            [],
        )

    def test_sequence_cannot_be_repaired_into_pdb(self):
        result = repair_tool_plan(
            ["analyze_pdb_structure"],
            [AMINO_ACID_SEQUENCE],
        )
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["repairable"])
        self.assertIsNone(result["repaired_plan"])

    def test_unknown_tool_is_never_guessed(self):
        result = repair_tool_plan(
            ["ComputeImaginaryProperty"],
            [AMINO_ACID_SEQUENCE],
        )
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["repairable"])
        self.assertEqual(result["blocking_issue"]["code"], "UNKNOWN_TOOL")


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    except Exception as exc:
        print(f"类型检查器测试运行失败: {exc}")
        raise
