"""Scoring regression cases are synthetic unit fixtures, never experiment runs."""
import copy
import tempfile
import unittest
from pathlib import Path

from scripts.external_baselines.common import now
from scripts.external_baselines.scoring import (call_identity, evidence_digest, matches,
                                                score_run, tri_and)


def item(sequence="ACDE"):
    return {"id": "unit_fixture", "question_zh": "unit fixture", "input": {
        "input_type": "amino_acid_sequence", "sequence": sequence}, "gold": {
        "expected_behavior": "execute", "preferred_tool_chain": ["irrelevant_preferred_tool"],
        "expected_output": {"gravy": -1.0}, "tolerance": {"gravy": 0.001}}}


def event(output=None, sequence="ACDE", status="success"):
    return {"kind": "tool_call", "event_index": 1, "call_id": "unit_call", "tool_name": "unit_tool",
            "resolved_arguments": {"sequence": sequence}, "status": status,
            "output": {"status": "success", "gravy": -1.0} if output is None else output}


def row(answer="unit answer", steps=1):
    return {"result": {"attempted_steps": steps, "final_response": answer,
                       "final_status": "answer_delivered", "native_execution_summary": {"completed": True}}}


class ScoringTests(unittest.TestCase):
    def test_tri_state_does_not_impute_unknown_as_failure(self):
        self.assertIsNone(tri_and([True, None]))
        self.assertFalse(tri_and([False, None]))
        self.assertTrue(tri_and([True, True]))

    def test_finite_numeric_bool_and_tolerance(self):
        self.assertFalse(matches(1, True))
        self.assertFalse(matches(float("nan"), -1))
        self.assertFalse(matches(float("inf"), -1))
        self.assertTrue(matches(-1.0005, -1.0, 0.001))

    def test_native_completed_does_not_imply_scientific_success(self):
        i = item(); i["gold"]["expected_output"]["pI"] = 6.0
        s = score_run(i, row(), [event()], [])
        self.assertTrue(s["native_completed"])
        self.assertFalse(s["scientific_task_success"])

    def test_wrong_object_correct_numbers_do_not_pass(self):
        s = score_run(item(), row(), [event(sequence="AAAA")], [])
        self.assertFalse(s["scientific_output_correct"])
        self.assertFalse(s["scientific_task_success"])

    def test_failed_output_cannot_supply_gold_values(self):
        s = score_run(item(), row(), [event(status="tool_error")], [])
        self.assertFalse(s["scientific_output_correct"])

    def test_compound_equivalent_tool_not_rejected_by_preferred_path(self):
        s = score_run(item(), row(), [event({"nested": {"gravy": -1.0}})], [])
        self.assertTrue(s["identity_correct"])
        self.assertTrue(s["scientific_output_correct"])
        self.assertIsNone(s["final_answer_correct"])
        self.assertIsNone(s["scientific_task_success"])

    def test_no_tools_crash_is_not_pre_execution_rejection(self):
        i = item(); i["gold"]["expected_behavior"] = "reject"
        s = score_run(i, row(None, 0), [], [])
        self.assertFalse(s["strict_pre_execution_rejection"])

    def test_tool_error_then_rejection_is_not_pre_rejection(self):
        i = item(); i["gold"]["expected_behavior"] = "reject"
        s = score_run(i, row("please specify record", 1), [event(status="tool_error")], [])
        self.assertFalse(s["strict_pre_execution_rejection"])

    def test_model_rejected_flag_alone_not_semantic_evidence(self):
        i = item(); i["gold"]["expected_behavior"] = "reject"
        r = row("unit statement", 0); r["result"]["final_status"] = "rejected"
        self.assertIsNone(score_run(i, r, [], [])["strict_pre_execution_rejection"])

    def test_interrupted_request_counts_as_attempt(self):
        i = item(); i["gold"]["expected_behavior"] = "reject"
        s = score_run(i, row("unit rejection", 0), [{"kind": "tool_request", "event_index": 0}], [])
        self.assertFalse(s["strict_pre_execution_rejection"])
        self.assertEqual(s["attempted_scientific_calls"], 1)

    def test_python_equivalent_computation_gets_review_not_failure(self):
        s = score_run(item(), row(), [{"kind": "kernel_execution_start", "event_index": 0,
                                      "code": "# synthetic unit fixture"}], [])
        self.assertIsNone(s["scientific_output_correct"])
        self.assertIsNone(s["scientific_task_success"])

    def test_duplicate_sequences_require_lineage_not_value_match(self):
        i = item(); i["input"]["record_id"] = "B"
        objects = [{"source": "unit.fasta", "record_id": k, "sequence_sha256": "same"} for k in ["A", "B"]]
        from scripts.external_baselines.audit import sequence_hash
        for o in objects: o["sequence_sha256"] = sequence_hash("ACDE")
        s = call_identity(i, [event()], objects)
        self.assertIsNone(s[0]["identity_correct"])

    def test_final_tool_prefix_not_used_as_answer(self):
        from scripts.external_baselines.scoring import final_response_text
        self.assertEqual(final_response_text(row('{"gravy": -1}\n**Final Answer:**\nwrong answer')["result"]), "wrong answer")

    def test_review_is_bound_to_exact_run_and_separately_typed(self):
        i, r, ev = item(), row(), [event()]
        review = {"evidence_sha256": evidence_digest(i, r, ev), "reviewer_id": "unit",
                  "reviewer_kind": "assistant", "reviewed_at": now(), "reason": "synthetic unit review",
                  "evidence_locations": ["result.final_response"], "labels": {"final_answer_correct": True}}
        s = score_run(i, r, ev, [], review)
        self.assertTrue(s["scientific_task_success"])
        self.assertEqual(s["review_source"], "assistant:unit")
        stale = copy.deepcopy(review); stale["evidence_sha256"] = "wrong"
        with self.assertRaises(ValueError): score_run(i, r, ev, [], stale)

    def test_review_cannot_override_automatic_failure(self):
        i, r, ev = item(), row(None), [event()]
        review = {"evidence_sha256": evidence_digest(i, r, ev), "reviewer_id": "unit", "reviewer_kind": "human",
                  "reviewed_at": now(), "reason": "unit", "evidence_locations": ["unit"],
                  "labels": {"final_answer_correct": True}}
        with self.assertRaises(ValueError): score_run(i, r, ev, [], review)


class CalibrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_slots_and_preflight_barrier(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts import run_external_baselines as runner
        active = 0; maximum = 0; preflights = 0

        async def fake_preflight(*args):
            nonlocal preflights
            preflights += 1
            return {"successful_tools": list(range(20))}

        async def fake_agent(request, model, client, facade, recorder, budget):
            nonlocal active, maximum
            self.assertEqual(preflights, 2)
            active += 1; maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            active -= 1
            return {"final_status": "answered", "final_response": "unit", "attempted_steps": 0}

        with tempfile.TemporaryDirectory(prefix="scoring-unit-") as directory:
            root = Path(directory); semaphore = asyncio.Semaphore(1); barrier = asyncio.Barrier(2)
            with patch.object(runner, "ROOT", root), patch.object(runner, "preflight", fake_preflight), \
                 patch.object(runner, "run_react", fake_agent), patch.object(runner, "code_snapshot", return_value={}), \
                 patch.object(runner.subprocess, "check_output", return_value="unit-head"):
                # Mocked unit transport only, not an experiment or provider call.
                jobs = [runner.execute(SimpleNamespace(condition="ReAct", models=["unit-model"], run_id=f"unit-{n}",
                        tool_base="http://unused", limit=1, preflight_only=False, resume_unstarted_from=[],
                        unit_semaphore=semaphore, calibration_barrier=barrier)) for n in range(2)]
                await asyncio.gather(*jobs)
        self.assertEqual(maximum, 1)


if __name__ == "__main__":
    unittest.main()
