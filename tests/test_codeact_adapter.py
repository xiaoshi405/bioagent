"""Tests of reused official CodeAct bodies, with scripted model/tool fixtures."""
import ast
import json
import unittest
from pathlib import Path

from scripts.external_baselines.common import Budget, Recorder, ToolFacade
from scripts.external_baselines.codeact import run_codeact
from scripts.external_baselines.codeact_upstream import SOURCE, load_core, upstream_identity
from scripts.external_baselines.codeact_worker import check_environment_code


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    async def complete_messages(self, messages, model, limit, timeout):
        self.messages.append(messages)
        return {"content": next(self.responses),
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}


class OfficialSourceTests(unittest.TestCase):
    def test_pinned_official_files_and_original_loop(self):
        identity = upstream_identity()
        self.assertEqual(identity["commit"], "d607f56c9cfe9e8632ebaf65dcaf2b4b7fe1c6f8")
        base, scope = load_core()
        self.assertEqual(Path(scope["run_task"].__code__.co_filename), SOURCE / "main.py")
        self.assertEqual(Path(base.Task.execute_action.__code__.co_filename), SOURCE / "tasks/base.py")
        self.assertEqual(Path(base.PythonREPL.__call__.__code__.co_filename), SOURCE / "tasks/repl.py")

    def test_environment_preserves_general_python_but_blocks_unrelated_resources(self):
        check_environment_code("import json\nrows = [x for x in range(3)]\nprint(json.dumps(rows))")
        for code in ("import os", "import Bio", "!pip install extra", "open.__globals__", "eval('1')",
                     "get_ipython().system('echo example')", "alias = eval"):
            with self.assertRaises(ValueError):
                check_environment_code(code)


class NativeLoopTests(unittest.IsolatedAsyncioTestCase):
    async def run_agent(self, responses, transport=None):
        recorder = Recorder(None, "offline-codeact-test")
        if transport is None:
            async def transport(name, args):
                return 200, {"status": "success", "sequence": "ACDE", "collection": {"records": []}}
        facade = ToolFacade("http://unused", [], recorder, Budget(), transport=transport)
        client = ScriptedModel(responses)
        result = await run_codeact({"question": "software fixture only", "input": {},
                                   "supplied_files": []}, "mock-model", client, facade, recorder, Budget())
        return result, client, recorder

    async def test_persistent_variables_real_feedback_and_native_answer_stop(self):
        result, client, recorder = await self.run_agent([
            "Action:\nx = 7\nprint(x)\nEnd Action",
            "Action:\nprint(x + 2)\nEnd Action",
            "Answer: 9"])
        self.assertEqual(result["final_status"], "answer_delivered")
        self.assertEqual(result["final_response"], "9")
        self.assertEqual(result["kernel_executions"], 2)
        self.assertIn("7", client.messages[1][-1]["content"])
        self.assertIn("9", client.messages[2][-1]["content"])
        self.assertFalse(any("Your answer is incorrect" in m["content"]
                             for messages in client.messages for m in messages))

    async def test_multiple_scientific_calls_in_one_native_code_action(self):
        received = []
        async def transport(name, args):
            received.append((name, args))
            return 200, {"status": "success", "sequence": args["sequence"]}
        result, _, recorder = await self.run_agent([
            "Action:\na = compute_gravy(sequence='ACDE')\nb = compute_pI_mw(sequence=a['sequence'])\nprint(b)\nEnd Action",
            "Answer: done"], transport)
        self.assertEqual(result["attempted_steps"], 2)
        self.assertEqual(result["kernel_executions"], 1)
        self.assertEqual(received[1], ("compute_pI_mw", {"sequence": "ACDE"}))

    async def test_python_exception_is_observed_and_model_can_correct(self):
        result, client, _ = await self.run_agent([
            "Action:\nprint(missing_variable)\nEnd Action",
            "Action:\nmissing_variable = 3\nprint(missing_variable)\nEnd Action",
            "Answer: 3"])
        self.assertIn("NameError", client.messages[1][-1]["content"])
        self.assertEqual(result["final_response"], "3")

    async def test_invalid_python_reaches_native_syntax_error_feedback(self):
        result, client, _ = await self.run_agent([
            "Action:\n```python\nprint(3)\n```\nEnd Action",
            "Action:\nprint(3)\nEnd Action", "Answer: 3"])
        self.assertIn("SyntaxError", client.messages[1][-1]["content"])
        self.assertEqual(result["final_response"], "3")

    async def test_final_answer_without_execution_is_not_assumed_scientifically_correct(self):
        result, _, _ = await self.run_agent(["Answer: Please provide a chain identifier."])
        self.assertEqual(result["attempted_steps"], 0)
        self.assertTrue(result["final_response_status_unclassified"])
        self.assertIsNone(result["scientific_task_success"])

    async def test_new_request_has_no_prior_python_state(self):
        await self.run_agent(["Action:\nsecret_value = 10\nEnd Action", "Answer: done"])
        _, client, _ = await self.run_agent(["Action:\nprint(secret_value)\nEnd Action", "Answer: done"])
        self.assertIn("NameError", client.messages[1][-1]["content"])

    async def test_unprovided_file_is_denied_in_native_repl(self):
        result, client, _ = await self.run_agent([
            "Action:\nprint(open('C:/unprovided.txt').read())\nEnd Action", "Answer: done"])
        self.assertIn("PermissionError", client.messages[1][-1]["content"])
        self.assertEqual(result["attempted_steps"], 0)


if __name__ == "__main__":
    unittest.main()
