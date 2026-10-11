"""Offline integration tests; scripted model/transport are explicitly mocks."""
import json
import unittest
from pathlib import Path

from scripts.external_baselines.common import Budget, Recorder, ToolFacade, tool_catalog
from scripts.external_baselines.react import run_react
from scripts.external_baselines.audit import identity_checks, source_index
from scripts.run_external_baselines import ROOT, load_dev_canary, make_request


class MockModel:
    def __init__(self, actions, usage=True):
        self.actions = iter(actions)
        self.prompts = []
        self.usage = usage

    async def complete(self, text, system, model, limit, timeout):
        self.prompts.append(text)
        return {"content": json.dumps(next(self.actions)),
                "usage": {"input_tokens": 1, "output_tokens": 1 if self.usage else None,
                          "total_tokens": 2 if self.usage else None}}


def action(name, arguments):
    return {"thought": "scripted unit test", "action": {"name": name, "arguments": arguments}}


class FacadeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.record = Recorder(None, "unit-test")
        self.received = []

        async def transport(name, arguments):
            self.received.append((name, arguments))
            return 200, {"status": "success", "sequence": "ACDE"}

        self.facade = ToolFacade("http://unused", [], self.record, Budget(), transport=transport)

    async def test_exact_arguments_explicit_reference_and_request_isolation(self):
        await self.facade.call("compute_gravy", {"sequence": "ACDE"})
        await self.facade.call("compute_gravy", {"sequence": {"$ref": "call_0001#/sequence"}})
        self.assertEqual(self.received[1][1], {"sequence": "ACDE"})
        calls = [e for e in self.record.events if e["kind"] == "tool_call"]
        self.assertEqual(calls[1]["reference_bindings"][0]["call_id"], "call_0001")
        another = ToolFacade("http://unused", [], Recorder(None, "other"), Budget())
        rejected = await another.call("compute_gravy", {"sequence": {"$ref": "call_0001#/sequence"}})
        self.assertEqual(rejected["status"], "interface_error")
        self.assertEqual(another.tool_calls, 0)

    async def test_failed_output_is_not_hidden_or_repaired(self):
        async def failure(name, arguments):
            return 400, {"status": "error", "message": "unit-test error"}
        self.facade.transport = failure
        await self.facade.call("compute_gravy", {"sequence": "bad"})
        await self.facade.call("compute_gravy", {"sequence": {"$ref": "call_0001#/message"}})
        self.assertEqual(self.record.events[-1]["resolved_arguments"]["sequence"], "unit-test error")
        self.assertEqual(self.facade.tool_calls, 2)

    async def test_observation_wrapper_is_not_silently_added_to_reference_root(self):
        await self.facade.call("compute_gravy", {"sequence": "ACDE"})
        invalid = await self.facade.call("compute_gravy", {
            "sequence": {"$ref": "call_0001#/output/sequence"}})
        self.assertEqual(invalid["status"], "interface_error")
        self.assertEqual(self.facade.tool_calls, 1)
        valid = await self.facade.call("compute_gravy", {
            "sequence": {"$ref": "call_0001#/sequence"}})
        self.assertEqual(valid["status"], "success")
        self.assertEqual(self.received[-1][1], {"sequence": "ACDE"})

    async def test_file_guard_and_internal_tool(self):
        for name, args in (("parse_protein_fasta", {"file_path": "C:/not-supplied.fasta"}),
                           ("send_expasy_request", {})):
            response = await self.facade.call(name, args)
            self.assertEqual(response["status"], "interface_error")
        self.assertEqual(self.facade.tool_calls, 0)

    async def test_request_record_survives_unexpected_transport_failure(self):
        async def crash(name, arguments):
            raise RuntimeError("mock infrastructure crash")
        self.facade.transport = crash
        with self.assertRaises(RuntimeError):
            await self.facade.call("compute_gravy", {"sequence": "ACDE"})
        self.assertEqual(self.record.events[-1]["kind"], "tool_request")
        self.assertEqual(self.facade.tool_calls, 1)

    async def test_react_observes_real_return_before_next_action(self):
        model = MockModel([action("compute_gravy", {"sequence": "ACDE"}),
                           action("finish", {"status": "answered", "response": "unit test"})])
        result = await run_react({"question": "unit test"}, "mock", model,
                                 self.facade, self.record, Budget())
        self.assertEqual(result["attempted_steps"], 1)
        self.assertIn('"sequence": "ACDE"', model.prompts[1])
        self.assertIn("Observation 1", model.prompts[1])
        self.assertEqual(result["llm_calls"], 2)

    async def test_rejection_without_scientific_execution(self):
        model = MockModel([action("finish", {"status": "rejected", "response": "missing identifier"})])
        result = await run_react({}, "mock", model, self.facade, self.record, Budget())
        self.assertEqual(result["final_status"], "rejected")
        self.assertEqual(result["attempted_steps"], 0)

    async def test_unknown_usage_is_not_treated_as_zero(self):
        model = MockModel([action("compute_gravy", {"sequence": "ACDE"})], usage=False)
        budget = Budget(max_total_output_tokens=4, max_output_tokens_per_call=4)
        result = await run_react({}, "mock", model, self.facade, self.record, budget)
        self.assertEqual(result["final_status"], "output_budget_exhausted")
        self.assertFalse(result["usage_complete"])
        self.assertEqual(result["output_budget_used"], 4)


class ProtocolTests(unittest.TestCase):
    def test_catalog_excludes_runtime_recovery_metadata(self):
        catalog = tool_catalog()
        self.assertEqual(len(catalog), 20)
        for spec in catalog:
            self.assertNotIn("repair_policy", spec)
            self.assertNotIn("output_bindings", spec)
        self.assertNotIn("send_expasy_request", [s["name"] for s in catalog])
        alignment = next(s for s in catalog if s["name"] == "align_fasta_records")
        self.assertIn("alignment", alignment["observed_success_output_fields"])

    def test_frozen_dev_selection_and_gold_is_not_in_actor_request(self):
        _, items = load_dev_canary()
        self.assertEqual(len(items), 12)
        for item in items:
            self.assertEqual(item["split"], "dev")
            request, files = make_request(item)
            self.assertEqual(set(request), {"question", "input", "supplied_files"})
            self.assertNotIn("gold", request["input"])
            self.assertNotIn("note", request["input"])

    def test_successful_execution_can_still_use_wrong_record(self):
        fixture = ROOT / "tests/fixtures/fasta/multi_protein.fasta"
        item = {"input": {"input_type": "fasta", "record_id": "protein_b"}}
        sources = source_index(item, [str(fixture)])
        events = [{"kind": "tool_call", "call_id": "call_0001", "status": "success",
                   "tool_name": "compute_gravy", "resolved_arguments": {"sequence": "ACDEFGHIKLMNPQRSTVWY"}}]
        checks = identity_checks(item, events, sources)
        self.assertFalse(checks[0]["target_value_matches"])


if __name__ == "__main__":
    unittest.main()
