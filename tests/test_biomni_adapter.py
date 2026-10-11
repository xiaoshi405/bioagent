"""Native A1 graph, retrieval, persistent objects and budget tests; no API."""
import unittest
from pathlib import Path

from scripts.external_baselines.biomni import run_biomni
from scripts.external_baselines.biomni_upstream import SOURCE, load_core, upstream_identity
from scripts.external_baselines.common import Budget, Recorder, ToolFacade


class Model:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.requests = []

    async def complete_messages(self, messages, model, limit, timeout, **kwargs):
        self.requests.append((messages, kwargs))
        return {"content": next(self.answers), "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}


class SourceTests(unittest.TestCase):
    def test_original_source_and_release(self):
        identity = upstream_identity()
        self.assertEqual(identity["commit"], "d15b3c9277b8945f3bfa6af8faf20bae92c09628")
        self.assertEqual(identity["tag"], "v0.0.8")
        agent, ns, support, _, retriever = load_core()
        self.assertEqual(Path(agent.configure.__code__.co_filename), SOURCE / "biomni/agent/a1.py")
        self.assertEqual(Path(agent.go.__code__.co_filename), SOURCE / "biomni/agent/a1.py")
        self.assertEqual(Path(retriever.prompt_based_retrieval.__code__.co_filename), SOURCE / "biomni/model/retriever.py")
        self.assertEqual(Path(support.run_python_repl.__code__.co_filename), SOURCE / "biomni/tool/support_tools.py")


class GraphTests(unittest.IsolatedAsyncioTestCase):
    async def run_native(self, answers, budget=None):
        budget = budget or Budget()
        record = Recorder(None, "offline-biomni-test")
        async def transport(name, arguments):
            return 200, {"status": "success", "sequence": "ACDE", "gravy": 0.25}
        facade = ToolFacade("http://unused", [], record, budget, transport=transport)
        model = Model(answers)
        result = await run_biomni({"question": "software fixture", "input": {}, "supplied_files": []},
                                  "mock-model", model, facade, record, budget)
        return result, record.events, model

    async def test_retrieval_native_xml_persistence_and_stop(self):
        result, events, model = await self.run_native([
            "TOOLS: [1]\nDATA_LAKE: []\nLIBRARIES: []",
            "<execute>from biomni.tool.protein_tools import compute_gravy\nresult = compute_gravy(sequence='ACDE')\nprint(result)</execute>",
            "<execute>print(result['gravy'])</execute>",
            "<solution>0.25</solution>"])
        self.assertEqual(result["final_status"], "answer_delivered")
        self.assertEqual(result["attempted_steps"], 1)
        self.assertEqual(result["kernel_executions"], 2)
        self.assertEqual(result["final_response"], "0.25")
        self.assertEqual(len([e for e in events if e["kind"] == "native_resource_retrieval"]), 1)
        self.assertTrue(all(r[1]["stop"] == ["</execute>", "</solution>"] for r in model.requests))
        self.assertEqual([e["exit_code"] for e in events if e["kind"] == "kernel_stop"], [0])

    async def test_native_error_observation_and_correction(self):
        result, events, model = await self.run_native([
            "TOOLS: [1]\nDATA_LAKE: []\nLIBRARIES: []",
            "<execute>print(undefined_variable)</execute>",
            "<execute>print(compute_gravy(sequence='ACDE'))</execute>",
            "<solution>completed</solution>"])
        self.assertEqual(result["final_status"], "answer_delivered")
        self.assertTrue(any("undefined_variable" in e.get("observation", "") for e in events if e["kind"] == "kernel_execution_end"))

    async def test_retrieval_counts_toward_budget(self):
        result, events, _ = await self.run_native([
            "TOOLS: [1]\nDATA_LAKE: []\nLIBRARIES: []",
            "<execute>print(1)</execute>"], Budget(max_llm_calls=2))
        self.assertEqual(result["final_status"], "llm_budget_exhausted")
        self.assertEqual(result["llm_calls"], 2)
        self.assertEqual(sum(e["kind"] == "llm_request" for e in events), 2)

    async def test_original_missing_closing_tag_completion(self):
        result, _, _ = await self.run_native([
            "TOOLS: []\nDATA_LAKE: []\nLIBRARIES: []", "<solution>native completion"])
        self.assertEqual(result["final_response"], "native completion")

    async def test_native_observation_truncation_preserves_full_log(self):
        result, events, _ = await self.run_native([
            "TOOLS: []\nDATA_LAKE: []\nLIBRARIES: []",
            "<execute>print('x' * 12000)</execute>", "<solution>done</solution>"])
        full = next(e["observation"] for e in events if e["kind"] == "kernel_execution_end")
        observed = next(e["content"] for e in events if e["kind"] == "native_graph_message" and "<observation>" in e["content"])
        self.assertEqual(len(full), 12001)
        self.assertIn("first 10K characters", observed)
        self.assertLess(len(observed), len(full))

    async def test_request_isolation_and_unavailable_import(self):
        await self.run_native(["TOOLS: []\nDATA_LAKE: []\nLIBRARIES: []",
            "<execute>sentinel = 123\nprint(sentinel)</execute>", "<solution>done</solution>"])
        result, events, _ = await self.run_native(["TOOLS: []\nDATA_LAKE: []\nLIBRARIES: []",
            "<execute>print(sentinel)</execute>", "<execute>import os</execute>", "<solution>done</solution>"])
        observations = [e["observation"] for e in events if e["kind"] == "kernel_execution_end"]
        self.assertTrue(any("sentinel" in text and "Error:" in text for text in observations))
        self.assertTrue(any("Execution environment error" in text for text in observations))


if __name__ == "__main__":
    unittest.main()
