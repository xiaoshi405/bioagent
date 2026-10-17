# -*- coding: utf-8 -*-
"""Tests for evaluation trace."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation_trace import (
    append_experiment_trace,
    build_experiment_trace,
    get_experiment_condition,
    load_experiment_traces,
    validate_experiment_trace,
)


TEST_SEQUENCE = "MKWVTFISLLLLFSSAYSRGVFRR"


def _question_item():
    try:
        return {
            "id": "bte_001",
            "split": "dev",
            "category": "single_tool",
            "difficulty": "easy",
            "question_zh": "计算给定蛋白质序列的GRAVY值。",
            "question_en": "Calculate the GRAVY score.",
            "input": {
                "input_type": "amino_acid_sequence",
                "sequence": TEST_SEQUENCE,
                "source_type": "synthetic",
            },
            "gold": {
                "preferred_tool_chain": ["compute_gravy"],
                "acceptable_tool_chains": [],
                "order_sensitive": False,
                "expected_status": "success",
                "expected_behavior": "execute",
            },
            "provenance": {
                "gold_generator": "local_tools",
                "verified_by": "A,D",
                "verified_at": "2026-07-28",
            },
        }
    except Exception as exc:
        raise AssertionError(f"构造评测题失败: {exc}") from exc


def _agent_result():
    try:
        return {
            "response": (
                f"原始答案不应进入Trace，序列也不应进入：{TEST_SEQUENCE}"
            ),
            "latency": 1.25,
            "reasoning": {
                "plan_chain": ["compute_gravy"],
                "tool_steps": [
                    {
                        "tool_name": "compute_gravy",
                        "status": "success",
                    }
                ],
                "execution_summary": {
                    "completed": True,
                    "attempted_steps": 1,
                    "successful_steps": 1,
                    "failed_steps": 0,
                    "error_type": None,
                    "detail_code": None,
                },
                "typed_planning": None,
            },
        }
    except Exception as exc:
        raise AssertionError(f"构造Agent响应失败: {exc}") from exc


def _context():
    try:
        return {
            "run_id": "dev-qwen-b0-r0",
            "repeat_index": 0,
            "model": "qwen-test",
            "temperature": 0.0,
            "seed": None,
            "git_commit": "162941f",
            "dataset_version": "biotooleval-dev-0.1",
            "kg_snapshot": "kg-test-v1",
            "python_version": "3.14.4",
        }
    except Exception as exc:
        raise AssertionError(f"构造实验上下文失败: {exc}") from exc


def _preplan_rejection_result():
    try:
        result = _agent_result()
        result["reasoning"]["plan_chain"] = []
        result["reasoning"]["request_guard"] = {
            "status": "success",
            "decision": "reject",
            "detail_code": "UNREGISTERED_TOOL",
            "unknown_tools": ["ComputeMagicScore"],
        }
        result["reasoning"]["execution_summary"] = {
            "completed": False,
            "attempted_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "error_type": "capability_failure",
            "detail_code": "UNREGISTERED_TOOL",
        }
        return result
    except Exception as exc:
        raise AssertionError(f"构造规划前拒绝响应失败: {exc}") from exc


class ExperimentConditionTests(unittest.TestCase):

    def test_b0_b1_b2_switches_are_fixed(self):
        b0 = get_experiment_condition("B0")
        b1 = get_experiment_condition("B1")
        b2 = get_experiment_condition("B2")

        self.assertFalse(b0["env"]["TYPE_PLANNING_ENABLED"])
        self.assertTrue(b1["env"]["TYPE_PLANNING_ENABLED"])
        self.assertFalse(b1["env"]["TYPE_REPAIR_ENABLED"])
        self.assertTrue(b2["env"]["TYPE_REPAIR_ENABLED"])

    def test_e0_e3_switches_are_orthogonal(self):
        expected = {
            "E0": (False, False, False),
            "E1": (True, False, False),
            "E2": (True, True, False),
            "E3": (True, True, True),
        }
        for condition, switches in expected.items():
            with self.subTest(condition=condition):
                env = get_experiment_condition(condition)["env"]
                self.assertEqual(
                    (
                        env["TYPE_PLANNING_ENABLED"],
                        env["TYPE_VALIDATION_ENABLED"],
                        env["TYPE_REPAIR_ENABLED"],
                    ),
                    switches,
                )

    def test_e1_trace_requires_explicit_validation_bypass(self):
        agent_result = _agent_result()
        agent_result["reasoning"]["typed_planning"] = {
            "enabled": True,
            "validation_enabled": False,
            "validation_bypassed": True,
            "repair_enabled": False,
            "repair_attempted": False,
            "repair_applied": False,
            "plan_before": ["compute_gravy"],
            "plan_after": ["compute_gravy"],
            "planning_issues": [],
        }
        result = build_experiment_trace(
            _question_item(),
            agent_result,
            condition="E1",
            run_context=_context(),
        )
        self.assertEqual(result["status"], "success")
        plan = result["trace"]["observation"]["plan"]
        self.assertTrue(plan["typed_runtime_enabled"])
        self.assertFalse(plan["validation_enabled"])

    def test_unknown_condition_is_rejected(self):
        result = get_experiment_condition("B3")

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "UNKNOWN_EXPERIMENT_CONDITION",
        )


class ExperimentTraceTests(unittest.TestCase):

    def test_build_trace_posthoc_validates_baseline_plan(self):
        result = build_experiment_trace(
            _question_item(),
            _agent_result(),
            condition="B0",
            run_context=_context(),
            evaluation_labels={"final_answer_correct": True},
        )

        self.assertEqual(result["status"], "success")
        trace = result["trace"]
        self.assertTrue(
            trace["observation"]["plan"]["type_valid_before"]
        )
        self.assertTrue(
            trace["observation"]["plan"]["type_valid_after"]
        )
        self.assertTrue(
            trace["evaluation"]["final_answer_correct"]
        )
        self.assertIsNone(trace["evaluation"]["repair_correct"])

    def test_runtime_initial_types_override_dataset_declared_type(self):
        context = _context()
        context["initial_types"] = []
        context["initial_artifact_issue_codes"] = [
            "INVALID_AMINO_ACID_SEQUENCE_CONTENT"
        ]

        result = build_experiment_trace(
            _question_item(),
            _agent_result(),
            condition="B0",
            run_context=context,
        )

        self.assertEqual(result["status"], "success")
        plan = result["trace"]["observation"]["plan"]
        self.assertEqual(plan["initial_types"], [])
        self.assertEqual(
            plan["initial_artifact_issue_codes"],
            ["INVALID_AMINO_ACID_SEQUENCE_CONTENT"],
        )
        self.assertFalse(plan["type_valid_before"])
        self.assertIn("MISSING_INPUT_TYPE", plan["issue_codes_before"])

    def test_candidate_submitted_gold_is_not_accepted(self):
        question = _question_item()
        question["submitted_gold"] = question.pop("gold")

        result = build_experiment_trace(
            question,
            _agent_result(),
            condition="B0",
            run_context=_context(),
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["issues"][0]["code"], "UNVERIFIED_GOLD")

    def test_condition_label_must_match_runtime_trace(self):
        result = build_experiment_trace(
            _question_item(),
            _agent_result(),
            condition="B1",
            run_context=_context(),
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "CONDITION_TRACE_MISMATCH",
        )

    def test_verified_preplan_rejection_is_valid_for_b1_and_b2(self):
        for condition in ("B1", "B2"):
            with self.subTest(condition=condition):
                result = build_experiment_trace(
                    _question_item(),
                    _preplan_rejection_result(),
                    condition=condition,
                    run_context=_context(),
                )

                self.assertEqual(result["status"], "success")
                execution = result["trace"]["observation"]["execution"]
                self.assertEqual(execution["attempted_steps"], 0)
                self.assertEqual(
                    execution["detail_code"],
                    "UNREGISTERED_TOOL",
                )

    def test_unverified_preplan_rejection_still_fails_closed(self):
        agent_result = _preplan_rejection_result()
        agent_result["reasoning"]["request_guard"]["unknown_tools"] = []

        result = build_experiment_trace(
            _question_item(),
            agent_result,
            condition="B1",
            run_context=_context(),
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "CONDITION_TRACE_MISMATCH",
        )

    def test_trace_contains_neither_sequence_question_nor_answer(self):
        result = build_experiment_trace(
            _question_item(),
            _agent_result(),
            condition="B0",
            run_context=_context(),
        )
        serialized = json.dumps(result["trace"], ensure_ascii=False)

        self.assertNotIn(TEST_SEQUENCE, serialized)
        self.assertNotIn("计算给定蛋白质序列", serialized)
        self.assertNotIn("原始答案不应进入Trace", serialized)

    def test_pruning_proof_is_schema_valid_and_drops_free_text_reason(self):
        agent_result = _agent_result()
        agent_result["reasoning"]["plan_chain"] = [
            "parse_multi_protein_fasta",
            "select_fasta_record",
            "select_fasta_record",
            "align_fasta_records",
            "summarize_pairwise_alignment",
        ]
        agent_result["reasoning"]["plan_chain_executed"] = [
            "parse_multi_protein_fasta",
            "align_fasta_records",
            "summarize_pairwise_alignment",
        ]
        operation = {
            "operation": "redundant_step_prune",
            "tool_name": "select_fasta_record",
            "original_step_index": 1,
            "before_step_index": None,
            "reason": f"不应公开的理由：{TEST_SEQUENCE}",
            "issue_code": "MISSING_INPUT_TYPE",
            "input_types": [
                "protein sequence collection",
                "FASTA record identifier",
            ],
            "output_types": ["amino acid sequence"],
            "downstream_required_types": [
                "protein sequence collection",
                "FASTA record identifier pair",
                "pairwise protein alignment",
            ],
            "registry_opt_in": True,
            "duplicate_group_size": 2,
            "output_consumed_downstream": False,
            "candidate_plan_type_valid": True,
            "bridge_path_length": None,
        }
        agent_result["reasoning"]["typed_planning"] = {
            "enabled": True,
            "repair_enabled": True,
            "repair_attempted": True,
            "repair_applied": True,
            "repair_strategy": "duplicate_invalid_step_pruning",
            "repair_operations": [operation, {**operation, "original_step_index": 2}],
            "plan_after": agent_result["reasoning"]["plan_chain_executed"],
            "planning_issues": [],
        }
        context = _context()
        context["initial_types"] = [
            "multi-record protein FASTA file",
            "FASTA record identifier pair",
        ]

        result = build_experiment_trace(
            _question_item(),
            agent_result,
            condition="B2",
            run_context=context,
        )

        self.assertEqual(result["status"], "success")
        plan = result["trace"]["observation"]["plan"]
        self.assertEqual(
            plan["repair_strategy"],
            "duplicate_invalid_step_pruning",
        )
        self.assertEqual(len(plan["repair_operations"]), 2)
        serialized = json.dumps(result["trace"], ensure_ascii=False)
        self.assertNotIn("不应公开的理由", serialized)
        self.assertNotIn(TEST_SEQUENCE, serialized)

    def test_schema_rejects_unknown_top_level_field(self):
        result = build_experiment_trace(
            _question_item(),
            _agent_result(),
            condition="B0",
            run_context=_context(),
        )
        trace = result["trace"]
        trace["raw_answer"] = "不允许保存"

        validation = validate_experiment_trace(trace)

        self.assertEqual(validation["status"], "error")
        self.assertEqual(
            validation["issues"][0]["code"],
            "TRACE_SCHEMA_VALIDATION_FAILED",
        )

    def test_jsonl_append_and_load_round_trip(self):
        result = build_experiment_trace(
            _question_item(),
            _agent_result(),
            condition="B0",
            run_context=_context(),
        )
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "trace.jsonl"
                append_result = append_experiment_trace(
                    result["trace"],
                    output_path=output_path,
                )
                load_result = load_experiment_traces(output_path)
        except Exception as exc:
            self.fail(f"Trace JSONL往返测试失败: {exc}")

        self.assertEqual(append_result["status"], "success")
        self.assertEqual(load_result["status"], "success")
        self.assertEqual(load_result["record_count"], 1)
        self.assertEqual(
            load_result["records"][0]["question_id"],
            "bte_001",
        )


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    except Exception as exc:
        print(f"实验Trace测试运行失败: {exc}")
        raise
