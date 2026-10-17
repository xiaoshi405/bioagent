# -*- coding: utf-8 -*-
"""Tests for plan sampling."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plan_sampling import (
    append_plan_sample,
    build_plan_sample_id,
    classify_planning_exception,
    collect_first_content_response,
    load_plan_samples,
    safe_plan_trace,
    sha256_text,
    validate_frozen_plan,
)


def _frozen_plan(question_id: str = "q1") -> dict:
    raw_response = '["compute_gravy"]'
    response_sha256 = sha256_text(raw_response)
    return {
        "schema_version": "1.0",
        "record_status": "captured",
        "experiment_id": "unit-plan",
        "question_id": question_id,
        "repeat_index": 0,
        "model": "offline-mock",
        "plan_sample_id": build_plan_sample_id(
            experiment_id="unit-plan",
            question_id=question_id,
            repeat_index=0,
            model="offline-mock",
            response_sha256=response_sha256,
        ),
        "capture_outcome": "captured",
        "selected_attempt_index": 2,
        "response_sha256": response_sha256,
        "planned_chain": ["compute_gravy"],
        "unknown_tools": [],
        "retrieved_triples": [],
        "retrieved_tool_names": ["compute_gravy"],
        "attempts": [
            {
                "attempt_index": 1,
                "attempt_kind": "infrastructure",
                "outcome": "network_timeout",
                "exception_type": "TimeoutError",
                "response_sha256": None,
                "raw_response": None,
                "parsed_chain": [],
                "unknown_tools": [],
                "latency_ms": 1.0,
            },
            {
                "attempt_index": 2,
                "attempt_kind": "content",
                "outcome": "captured",
                "exception_type": None,
                "response_sha256": response_sha256,
                "raw_response": raw_response,
                "parsed_chain": ["compute_gravy"],
                "unknown_tools": [],
                "latency_ms": 2.0,
            },
        ],
    }


class PlanSamplingTests(unittest.IsolatedAsyncioTestCase):

    def test_exception_classification_separates_infrastructure(self):
        self.assertEqual(
            classify_planning_exception(TimeoutError("timeout")),
            "network_timeout",
        )
        self.assertEqual(
            classify_planning_exception(ConnectionError("connection")),
            "network_error",
        )
        self.assertEqual(
            classify_planning_exception(ValueError("bad payload")),
            "planner_exception",
        )

    def test_safe_trace_keeps_attempts_but_removes_raw_response(self):
        plan = _frozen_plan()
        self.assertEqual(validate_frozen_plan(plan)["status"], "success")
        public_trace = safe_plan_trace(plan)
        self.assertTrue(public_trace["shared_plan_replay"])
        self.assertEqual(public_trace["attempt_count"], 2)
        self.assertNotIn("raw_response", json.dumps(public_trace))
        self.assertEqual(
            public_trace["attempts"][0]["outcome"],
            "network_timeout",
        )

    def test_ledger_preserves_private_raw_and_indexes_only_captured_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger_path = Path(temp_dir) / "plans.jsonl"
            plan = _frozen_plan()
            captured = append_plan_sample(ledger_path, plan)
            failed = append_plan_sample(
                ledger_path,
                {
                    "schema_version": "1.0",
                    "record_status": "capture_failed",
                    "experiment_id": "unit-plan",
                    "question_id": "q2",
                    "repeat_index": 0,
                    "model": "offline-mock",
                    "attempts": [],
                },
            )
            loaded = load_plan_samples(ledger_path)

            self.assertEqual(captured["status"], "success")
            self.assertEqual(failed["status"], "success")
            self.assertIn("compute_gravy", ledger_path.read_text("utf-8"))
            self.assertEqual(loaded["record_count"], 2)
            self.assertEqual(loaded["captured_record_count"], 1)
            self.assertEqual(loaded["failed_record_count"], 1)
            self.assertEqual(set(loaded["records"]), {("q1", 0)})

    def test_duplicate_captured_plan_key_is_rejected_on_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger_path = Path(temp_dir) / "plans.jsonl"
            append_plan_sample(ledger_path, _frozen_plan())
            append_plan_sample(ledger_path, _frozen_plan())
            loaded = load_plan_samples(ledger_path)
        self.assertEqual(loaded["status"], "error")
        self.assertEqual(loaded["code"], "DUPLICATE_PLAN_SAMPLE_KEY")

    def test_unknown_tool_content_is_kept_as_a_frozen_model_result(self):
        plan = _frozen_plan()
        plan["capture_outcome"] = "unknown_tool_plan"
        plan["planned_chain"] = ["InventedProteinTool"]
        plan["unknown_tools"] = ["InventedProteinTool"]
        plan["attempts"][1]["outcome"] = "unknown_tool_plan"
        plan["attempts"][1]["parsed_chain"] = ["InventedProteinTool"]
        plan["attempts"][1]["unknown_tools"] = ["InventedProteinTool"]
        result = validate_frozen_plan(plan)
        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["frozen_plan"]["capture_outcome"],
            "unknown_tool_plan",
        )

    async def test_network_timeout_retries_but_first_content_does_not(self):
        responses = [TimeoutError("timeout"), "not a parseable plan", "unused"]
        call_count = 0
        sleeps = []

        async def planner():
            nonlocal call_count
            value = responses[call_count]
            call_count += 1
            if isinstance(value, BaseException):
                raise value
            return value

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        result = await collect_first_content_response(
            planner_callable=planner,
            parser_callable=lambda response: [],
            unknown_tool_callable=lambda chain: [],
            max_network_attempts=3,
            backoff_seconds=0.25,
            sleep_callable=fake_sleep,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["capture_outcome"], "unparseable_plan")
        self.assertEqual(call_count, 2)
        self.assertEqual(sleeps, [0.25])
        self.assertEqual(
            [item["outcome"] for item in result["attempts"]],
            ["network_timeout", "unparseable_plan"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
