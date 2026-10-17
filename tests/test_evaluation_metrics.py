# -*- coding: utf-8 -*-
"""Tests for evaluation metrics."""

import copy
import unittest

from scripts.evaluation_metrics import (
    compare_conditions,
    compute_condition_metrics,
    mcnemar_exact,
    paired_bootstrap_difference,
    tool_chain_matches,
)
from scripts.evaluation_trace import validate_experiment_trace
from tests.test_evaluation_trace import (
    _agent_result,
    _context,
    _question_item,
)
from scripts.evaluation_trace import build_experiment_trace


def _trace(
    question_id,
    condition,
    *,
    completed,
    final_correct,
    type_valid_before=True,
    type_valid_after=True,
    type_ii=False,
    repair_attempted=False,
    repair_applied=False,
    repair_correct=None,
    expected_behavior="execute",
    correct_rejection=None,
    attempted_steps=1,
    original_plan_executable=True,
    result_hallucination=False,
    latency_ms=1000.0,
):
    try:
        question = _question_item()
        question["id"] = question_id
        question["gold"]["expected_behavior"] = expected_behavior
        question["gold"]["expected_status"] = (
            "error" if expected_behavior == "reject" else "success"
        )
        context = _context()
        context["run_id"] = f"synthetic-{condition}"
        context["total_latency_ms"] = latency_ms
        agent_result = _agent_result()
        if condition in {"B1", "B2", "E1", "E2", "E3"}:
            agent_result["reasoning"]["plan_chain_executed"] = [
                "compute_gravy"
            ]
            agent_result["reasoning"]["typed_planning"] = {
                "enabled": True,
                "status": "success",
                "completed": completed,
                "error_type": None if completed else "plan_failure",
                "plan_before": ["compute_gravy"],
                "plan_after": ["compute_gravy"],
                "structured_plan": [],
                "validation_enabled": condition not in {"E1"},
                "validation_bypassed": condition == "E1",
                "repair_enabled": condition in {"B2", "E3"},
                "repair_attempted": repair_attempted,
                "repair_applied": repair_applied,
                "planning_issues": [],
                "executed_steps": [],
                "artifact_pool": {},
                "total_duration_ms": 5.0,
            }
        result = build_experiment_trace(
            question,
            agent_result,
            condition=condition,
            run_context=context,
            evaluation_labels={
                "final_answer_correct": final_correct,
                "repair_correct": repair_correct,
                "result_hallucination": result_hallucination,
                "original_plan_executable": original_plan_executable,
                "correct_rejection": correct_rejection,
            },
        )
        if result.get("status") != "success":
            raise AssertionError(result)
        trace = result["trace"]
        plan = trace["observation"]["plan"]
        plan["type_valid_before"] = type_valid_before
        plan["type_valid_after"] = type_valid_after
        plan["repair_attempted"] = repair_attempted
        plan["repair_applied"] = repair_applied
        if type_ii:
            plan["issue_codes_before"] = ["MISSING_INPUT_TYPE"]
            plan["issue_codes_after"] = ["MISSING_INPUT_TYPE"]
        execution = trace["observation"]["execution"]
        execution["completed"] = completed
        execution["attempted_steps"] = attempted_steps
        execution["successful_steps"] = attempted_steps if completed else 0
        execution["failed_steps"] = 0 if completed else attempted_steps
        execution["error_type"] = None if completed else "plan_failure"
        validation = validate_experiment_trace(trace)
        if validation.get("status") != "success":
            raise AssertionError(validation)
        return trace
    except Exception as exc:
        raise AssertionError(f"构造合成Trace失败: {exc}") from exc


class ToolChainMetricTests(unittest.TestCase):

    def test_order_insensitive_chain_accepts_reordered_tools(self):
        gold = {
            "preferred_tool_chain": ["compute_gravy", "compute_pI_mw"],
            "acceptable_tool_chains": [],
            "order_sensitive": False,
        }

        self.assertTrue(
            tool_chain_matches(
                ["compute_pI_mw", "compute_gravy"],
                gold,
            )
        )

    def test_order_sensitive_chain_rejects_reordering(self):
        gold = {
            "preferred_tool_chain": ["parse_protein_fasta", "compute_gravy"],
            "acceptable_tool_chains": [],
            "order_sensitive": True,
        }

        self.assertFalse(
            tool_chain_matches(
                ["compute_gravy", "parse_protein_fasta"],
                gold,
            )
        )


class ConditionMetricTests(unittest.TestCase):

    def test_null_answer_label_is_excluded_and_coverage_reported(self):
        records = [
            _trace(
                "bte_001",
                "B1",
                completed=True,
                final_correct=True,
            ),
            _trace(
                "bte_002",
                "B1",
                completed=True,
                final_correct=None,
            ),
        ]
        records[1]["observation"]["plan"]["type_valid_before"] = None
        records[1]["observation"]["plan"]["type_valid_after"] = None

        result = compute_condition_metrics(records)
        metric = result["metrics"]["final_answer_accuracy"]

        self.assertEqual(result["status"], "success")
        self.assertEqual(metric["denominator"], 1)
        self.assertEqual(metric["rate"], 1.0)
        self.assertEqual(metric["coverage"], 0.5)
        self.assertEqual(
            result["metrics"]["invalid_plan_rate"]["denominator"],
            1,
        )
        self.assertEqual(
            result["metrics"]["type_valid_plan_rate"]["coverage"],
            0.5,
        )

    def test_reject_question_is_excluded_from_executable_denominator(self):
        records = [
            _trace(
                "bte_001",
                "B1",
                completed=True,
                final_correct=True,
            ),
            _trace(
                "bte_002",
                "B1",
                completed=False,
                final_correct=True,
                expected_behavior="reject",
                correct_rejection=True,
                attempted_steps=0,
            ),
        ]

        result = compute_condition_metrics(records)

        self.assertEqual(
            result["metrics"]["executable_plan_rate"]["denominator"],
            1,
        )
        self.assertEqual(
            result["metrics"]["correct_rejection_rate"]["rate"],
            1.0,
        )

    def test_repair_precision_uses_only_applied_labeled_repairs(self):
        records = [
            _trace(
                "bte_001",
                "B2",
                completed=True,
                final_correct=True,
                repair_attempted=True,
                repair_applied=True,
                repair_correct=True,
            ),
            _trace(
                "bte_002",
                "B2",
                completed=False,
                final_correct=False,
                repair_attempted=True,
                repair_applied=True,
                repair_correct=False,
            ),
            _trace(
                "bte_003",
                "B2",
                completed=True,
                final_correct=True,
            ),
        ]

        result = compute_condition_metrics(records)
        precision = result["metrics"]["repair_precision"]

        self.assertEqual(precision["numerator"], 1)
        self.assertEqual(precision["denominator"], 2)
        self.assertEqual(precision["rate"], 0.5)

    def test_generated_and_unresolved_type_ii_are_separate(self):
        record = _trace(
            "bte_001",
            "B2",
            completed=True,
            final_correct=True,
            type_valid_before=False,
            type_valid_after=True,
            repair_attempted=True,
            repair_applied=True,
            repair_correct=True,
        )
        record["observation"]["plan"]["issue_codes_before"] = [
            "MISSING_INPUT_TYPE"
        ]
        record["observation"]["plan"]["issue_codes_after"] = []

        result = compute_condition_metrics([record])

        self.assertEqual(
            result["metrics"]["type_ii_generation_rate"]["rate"],
            1.0,
        )
        self.assertEqual(
            result["metrics"]["unresolved_type_ii_rate"]["rate"],
            0.0,
        )

    def test_strata_are_reported_by_category_difficulty_and_input(self):
        result = compute_condition_metrics(
            [
                _trace(
                    "bte_001",
                    "B1",
                    completed=True,
                    final_correct=True,
                )
            ]
        )

        self.assertIn("single_tool", result["strata"]["category"])
        self.assertIn("easy", result["strata"]["difficulty"])
        self.assertIn(
            "amino_acid_sequence",
            result["strata"]["input_type"],
        )

    def test_mixed_repeat_indices_are_rejected(self):
        first = _trace(
            "bte_001",
            "B1",
            completed=True,
            final_correct=True,
        )
        second = _trace(
            "bte_002",
            "B1",
            completed=True,
            final_correct=True,
        )
        second["repeat_index"] = 1

        result = compute_condition_metrics([first, second])

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "MIXED_MODEL_OR_REPEAT",
        )

    def test_failed_tool_attempt_is_not_counted_as_safe_rejection(self):
        record = _trace(
            "bte_901",
            "B1",
            completed=False,
            final_correct=True,
            expected_behavior="reject",
            correct_rejection=True,
            attempted_steps=1,
        )

        metrics = compute_condition_metrics([record])["metrics"]

        self.assertEqual(metrics["safe_rejection_rate"]["rate"], 0.0)
        actionable = metrics["actionable_rejection_accuracy"]
        self.assertEqual(actionable["denominator"], 0)
        self.assertIsNone(actionable["rate"])
        self.assertIsNone(actionable["coverage"])

    def test_actionable_rejection_excludes_unlabeled_and_unsafe_records(self):
        safe_labeled = _trace(
            "bte_910",
            "B1",
            completed=False,
            final_correct=True,
            expected_behavior="reject",
            correct_rejection=True,
            attempted_steps=0,
        )
        safe_unlabeled = _trace(
            "bte_911",
            "B1",
            completed=False,
            final_correct=None,
            expected_behavior="reject",
            correct_rejection=None,
            attempted_steps=0,
        )
        unsafe_labeled = _trace(
            "bte_912",
            "B1",
            completed=False,
            final_correct=True,
            expected_behavior="reject",
            correct_rejection=False,
            attempted_steps=1,
        )

        actionable = compute_condition_metrics(
            [safe_labeled, safe_unlabeled, unsafe_labeled]
        )["metrics"]["actionable_rejection_accuracy"]

        self.assertEqual(actionable["numerator"], 1)
        self.assertEqual(actionable["denominator"], 1)
        self.assertEqual(actionable["rate"], 1.0)
        self.assertEqual(actionable["coverage"], 0.5)

    def test_behavior_aware_success_requires_safe_reject_and_answer_label(self):
        safe = _trace(
            "bte_902",
            "B1",
            completed=False,
            final_correct=True,
            expected_behavior="reject",
            correct_rejection=True,
            attempted_steps=0,
        )
        unlabeled = _trace(
            "bte_903",
            "B1",
            completed=False,
            final_correct=None,
            expected_behavior="reject",
            correct_rejection=True,
            attempted_steps=0,
        )

        metric = compute_condition_metrics([safe, unlabeled])["metrics"][
            "behavior_aware_task_success"
        ]

        self.assertEqual(metric["numerator"], 1)
        self.assertEqual(metric["denominator"], 1)
        self.assertEqual(metric["coverage"], 0.5)

    def test_false_rejection_uses_executable_nonreject_population(self):
        executable_but_blocked = _trace(
            "bte_904",
            "B1",
            completed=False,
            final_correct=False,
            attempted_steps=0,
            original_plan_executable=True,
        )
        legitimate_reject = _trace(
            "bte_905",
            "B1",
            completed=False,
            final_correct=True,
            expected_behavior="reject",
            attempted_steps=0,
            original_plan_executable=False,
            correct_rejection=True,
        )

        metric = compute_condition_metrics(
            [executable_but_blocked, legitimate_reject]
        )["metrics"]["false_rejection_rate"]

        self.assertEqual(metric["numerator"], 1)
        self.assertEqual(metric["denominator"], 1)
        self.assertEqual(metric["coverage"], 0.5)

    def test_type_iii_uses_result_hallucination_label_and_null_coverage(self):
        hallucinated = _trace(
            "bte_906",
            "B1",
            completed=False,
            final_correct=True,
            original_plan_executable=False,
            result_hallucination=True,
        )
        unlabeled = _trace(
            "bte_907",
            "B1",
            completed=False,
            final_correct=False,
            result_hallucination=None,
        )

        metric = compute_condition_metrics([hallucinated, unlabeled])[
            "metrics"
        ]["type_iii_rate"]

        self.assertEqual(metric["numerator"], 1)
        self.assertEqual(metric["denominator"], 1)
        self.assertEqual(metric["coverage"], 0.5)

    def test_gold_repair_task_without_initial_error_is_not_an_opportunity(self):
        record = _trace(
            "bte_908",
            "B2",
            completed=True,
            final_correct=True,
            expected_behavior="repair_then_execute",
            type_valid_before=True,
            repair_attempted=False,
            repair_applied=False,
        )

        metric = compute_condition_metrics([record])["metrics"][
            "repair_recovery_rate"
        ]

        self.assertEqual(metric["denominator"], 0)
        self.assertIsNone(metric["rate"])
        self.assertIsNone(metric["coverage"])

    def test_actual_repair_opportunity_requires_applied_valid_completed_path(self):
        recovered = _trace(
            "bte_909",
            "B2",
            completed=True,
            final_correct=True,
            expected_behavior="repair_then_execute",
            type_valid_before=False,
            type_valid_after=True,
            type_ii=True,
            repair_attempted=True,
            repair_applied=True,
            repair_correct=True,
        )
        recovered["observation"]["plan"]["issue_codes_after"] = []

        metric = compute_condition_metrics([recovered])["metrics"][
            "repair_recovery_rate"
        ]

        self.assertEqual(metric["numerator"], 1)
        self.assertEqual(metric["denominator"], 1)
        self.assertEqual(metric["rate"], 1.0)


class PairedComparisonTests(unittest.TestCase):

    def test_paired_bootstrap_reports_expected_difference(self):
        result = paired_bootstrap_difference(
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            n_resamples=500,
            seed=7,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["pair_count"], 4)
        self.assertAlmostEqual(result["mean_difference"], 0.5)
        self.assertEqual(len(result["ci_95"]), 2)

    def test_condition_comparison_pairs_question_model_and_repeat(self):
        records = [
            _trace(
                "bte_001",
                "B0",
                completed=False,
                final_correct=False,
                type_ii=True,
                latency_ms=1000,
            ),
            _trace(
                "bte_001",
                "B1",
                completed=True,
                final_correct=True,
                latency_ms=1100,
            ),
            _trace(
                "bte_002",
                "B0",
                completed=True,
                final_correct=True,
                latency_ms=900,
            ),
            _trace(
                "bte_002",
                "B1",
                completed=True,
                final_correct=True,
                latency_ms=1000,
            ),
        ]

        result = compare_conditions(records, n_resamples=500, seed=9)
        comparison = result["comparisons"]["B1"]

        self.assertEqual(result["status"], "success")
        self.assertEqual(comparison["available_pair_count"], 2)
        self.assertAlmostEqual(
            comparison["final_answer_accuracy_difference"][
                "mean_difference"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            comparison["total_latency_ms_difference"][
                "mean_difference"
            ],
            100.0,
        )
        self.assertEqual(
            comparison["final_answer_accuracy_difference"]["mcnemar"][
                "treatment_only_success"
            ],
            1,
        )

    def test_exact_mcnemar_reports_discordant_pairs(self):
        result = mcnemar_exact(
            [False, False, True, True],
            [True, True, True, False],
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["baseline_only_success"], 1)
        self.assertEqual(result["treatment_only_success"], 2)
        self.assertEqual(result["discordant_pair_count"], 3)
        self.assertGreaterEqual(result["p_value_two_sided"], 0.0)
        self.assertLessEqual(result["p_value_two_sided"], 1.0)

    def test_factorial_validation_comparison_uses_e1_as_baseline(self):
        records = [
            _trace(
                "bte_001",
                "E1",
                completed=False,
                final_correct=False,
                type_valid_before=False,
                type_valid_after=False,
                type_ii=True,
                attempted_steps=1,
            ),
            _trace(
                "bte_001",
                "E2",
                completed=False,
                final_correct=True,
                type_valid_before=False,
                type_valid_after=False,
                type_ii=True,
                attempted_steps=0,
            ),
        ]

        result = compare_conditions(
            records,
            baseline_condition="E1",
            n_resamples=100,
            seed=27,
        )

        self.assertEqual(result["status"], "success")
        comparison = result["comparisons"]["E2"]
        self.assertEqual(comparison["available_pair_count"], 1)
        self.assertEqual(
            comparison["unsafe_type_ii_execution_rate_difference"][
                "mean_difference"
            ],
            -1.0,
        )

    def test_duplicate_pair_key_is_rejected(self):
        first = _trace(
            "bte_001",
            "B0",
            completed=True,
            final_correct=True,
        )
        duplicate = copy.deepcopy(first)

        result = compare_conditions([first, duplicate])

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "DUPLICATE_PAIR_KEY",
        )

    def test_comparison_does_not_pool_repeat_indices(self):
        baseline = _trace(
            "bte_001",
            "B0",
            completed=True,
            final_correct=True,
        )
        treatment = _trace(
            "bte_001",
            "B1",
            completed=True,
            final_correct=True,
        )
        treatment["repeat_index"] = 1

        result = compare_conditions([baseline, treatment])

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "MIXED_MODEL_OR_REPEAT",
        )


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    except Exception as exc:
        print(f"实验指标测试运行失败: {exc}")
        raise
