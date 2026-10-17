# -*- coding: utf-8 -*-
"""Tests for typed executor."""

import asyncio
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from app.tools import tool_runner
from app.tools.tool_runner import build_typed_tool_payload
from scripts.structured_plan import build_structured_plan
from scripts.typed_executor import (
    execute_structured_plan,
    validate_runtime_plan,
)
from scripts.type_checker import (
    AMINO_ACID_SEQUENCE,
    FASTA_TO_SEQUENCE_BRIDGE,
    PDB_CHAIN_IDENTIFIER,
    PDB_STRUCTURE_FILE,
    PDB_TO_SEQUENCE_BRIDGE,
    SINGLE_PROTEIN_FASTA_FILE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FASTA_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "fasta" / "single_protein.fasta"
PDB_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "pdb" / "1HHB.pdb"
TEST_SEQUENCE = "MKWVTFISLLLLFSSAYSRGVFRR"


def _structured(tool_path, input_types):
    result = build_structured_plan(tool_path, input_types)
    if result.get("valid") is not True:
        raise AssertionError(f"测试结构化计划构造失败: {result}")
    return result["structured_plan"]


class TypedPayloadTests(unittest.TestCase):

    def test_sequence_payload_preserves_invalid_candidate_exactly(self):
        candidate = "MKWVTFISLL123X"
        result = build_typed_tool_payload(
            "/api/compute_pi_mw",
            {AMINO_ACID_SEQUENCE: candidate},
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["json_data"]["sequence"], candidate)

    def test_unregistered_api_path_is_not_treated_as_sequence_tool(self):
        result = build_typed_tool_payload(
            "/api/imaginary_tool",
            {AMINO_ACID_SEQUENCE: TEST_SEQUENCE},
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "UNREGISTERED_TYPED_API")

    def test_pdb_chain_payload_uses_exact_file_and_case_sensitive_chain(self):
        result = build_typed_tool_payload(
            "/api/select_pdb_chain",
            {
                PDB_STRUCTURE_FILE: str(PDB_FIXTURE),
                PDB_CHAIN_IDENTIFIER: "b",
            },
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["json_data"]["file_path"], str(PDB_FIXTURE.resolve()))
        self.assertEqual(result["json_data"]["chain_id"], "b")

    def test_anm_typed_payload_preserves_explicit_parameters(self):
        result = build_typed_tool_payload(
            "/api/calculate_protein_anm",
            {PDB_STRUCTURE_FILE: str(PDB_FIXTURE)},
            {"n_modes": 3, "cutoff": 10.0, "chain_id": "B"},
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["json_data"]["n_modes"], 3)
        self.assertEqual(result["json_data"]["cutoff"], 10.0)
        self.assertEqual(result["json_data"]["chain_id"], "B")

    def test_fasta_payload_requires_existing_supported_file(self):
        valid = build_typed_tool_payload(
            "/api/parse_protein_fasta",
            {SINGLE_PROTEIN_FASTA_FILE: str(FASTA_FIXTURE)},
        )
        invalid = build_typed_tool_payload(
            "/api/parse_protein_fasta",
            {SINGLE_PROTEIN_FASTA_FILE: "missing.fasta"},
        )

        self.assertEqual(valid["status"], "success")
        self.assertEqual(invalid["status"], "error")

    def test_run_tool_forwards_typed_sequence_without_text_extraction(self):
        captured = {}
        candidate = "MKWVTFISLL123X"

        class FakeResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def json(self):
                return {"status": "success", "sequence": candidate}

        class FakeSession:
            def __init__(self, *args, **kwargs):
                captured["timeout"] = kwargs.get("timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            def post(self, url, json):
                captured["url"] = url
                captured["payload"] = json
                return FakeResponse()

        with patch.object(tool_runner.aiohttp, "ClientSession", FakeSession):
            result = asyncio.run(
                tool_runner.run_tool(
                    "compute_pI_mw",
                    "这里不应参与参数提取",
                    [],
                    typed_inputs={AMINO_ACID_SEQUENCE: candidate},
                )
            )

        self.assertEqual(json.loads(result)["status"], "success")
        self.assertEqual(captured["payload"], {"sequence": candidate})
        self.assertEqual(captured["timeout"], tool_runner.REQUEST_TIMEOUT)


class TypedExecutorTests(unittest.IsolatedAsyncioTestCase):

    async def test_fan_out_reuses_same_initial_sequence(self):
        plan = _structured(
            ["compute_pI_mw", "compute_gravy"],
            [AMINO_ACID_SEQUENCE],
        )
        seen_sequences = []

        async def fake_runner(**kwargs):
            seen_sequences.append(kwargs["typed_inputs"][AMINO_ACID_SEQUENCE])
            if kwargs["tool_name"] == "compute_pI_mw":
                return {
                    "status": "success",
                    "pI": 5.2,
                    "molecular_weight": 2800.0,
                }
            return {"status": "success", "gravy": -0.42}

        result = await execute_structured_plan(
            plan,
            {AMINO_ACID_SEQUENCE: TEST_SEQUENCE},
            runner=fake_runner,
        )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["completed"])
        self.assertEqual(seen_sequences, [TEST_SEQUENCE, TEST_SEQUENCE])
        self.assertEqual(len(result["executed_steps"]), 2)
        self.assertEqual(result["artifact_pool"]["artifact_count"], 4)

    async def test_executor_forwards_request_parameter_context(self):
        plan = _structured(["calculate_protein_anm"], [PDB_STRUCTURE_FILE])
        captured = {}
        parameter_context = {
            "status": "success",
            "by_api": {
                "/api/calculate_protein_anm": {
                    "n_modes": 3,
                    "cutoff": 10.0,
                }
            },
            "issues": [],
        }

        async def fake_runner(**kwargs):
            captured.update(copy.deepcopy(kwargs))
            return {"status": "success", "eigenvalues": [0.1, 0.2, 0.3]}

        result = await execute_structured_plan(
            plan,
            {PDB_STRUCTURE_FILE: str(PDB_FIXTURE)},
            runner=fake_runner,
            tool_parameters=parameter_context,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(captured["tool_parameters"], parameter_context)

    async def test_fasta_bridge_passes_real_output_sequence_to_next_step(self):
        plan = _structured(
            [FASTA_TO_SEQUENCE_BRIDGE, "compute_gravy"],
            [SINGLE_PROTEIN_FASTA_FILE],
        )
        calls = []
        parsed_sequence = "ACDEFGHIKLMNPQRSTVWY"

        async def fake_runner(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            if kwargs["tool_name"] == FASTA_TO_SEQUENCE_BRIDGE:
                return {
                    "status": "success",
                    "sequence": parsed_sequence,
                    "sequence_length": len(parsed_sequence),
                }
            return {"status": "success", "gravy": 0.1}

        result = await execute_structured_plan(
            plan,
            {SINGLE_PROTEIN_FASTA_FILE: str(FASTA_FIXTURE)},
            runner=fake_runner,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            calls[1]["typed_inputs"][AMINO_ACID_SEQUENCE],
            parsed_sequence,
        )
        second_inputs = result["executed_steps"][1]["input_artifact_ids"]
        self.assertEqual(set(second_inputs), {AMINO_ACID_SEQUENCE})

    async def test_pdb_bridge_passes_selected_chain_sequence(self):
        plan = _structured(
            [PDB_TO_SEQUENCE_BRIDGE, "compute_gravy"],
            [PDB_STRUCTURE_FILE, PDB_CHAIN_IDENTIFIER],
        )
        selected_sequence = "VHLTPEEKSAVTALWGKVNVDEVGGEALGRLL"
        calls = []

        async def fake_runner(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            if kwargs["tool_name"] == PDB_TO_SEQUENCE_BRIDGE:
                self.assertEqual(
                    kwargs["typed_inputs"][PDB_CHAIN_IDENTIFIER],
                    "B",
                )
                return {
                    "status": "success",
                    "sequence": selected_sequence,
                    "sequence_length": len(selected_sequence),
                }
            return {"status": "success", "gravy": -0.2}

        result = await execute_structured_plan(
            plan,
            {
                PDB_STRUCTURE_FILE: str(PDB_FIXTURE),
                PDB_CHAIN_IDENTIFIER: "B",
            },
            runner=fake_runner,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            calls[1]["typed_inputs"][AMINO_ACID_SEQUENCE],
            selected_sequence,
        )

    async def test_tool_error_stops_downstream_and_creates_no_fake_output(self):
        plan = _structured(
            [FASTA_TO_SEQUENCE_BRIDGE, "compute_gravy"],
            [SINGLE_PROTEIN_FASTA_FILE],
        )
        calls = []

        async def fake_runner(**kwargs):
            calls.append(kwargs["tool_name"])
            return {
                "status": "error",
                "message": "多记录FASTA不允许自动选择",
            }

        result = await execute_structured_plan(
            plan,
            {SINGLE_PROTEIN_FASTA_FILE: str(FASTA_FIXTURE)},
            runner=fake_runner,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "tool_failure")
        self.assertEqual(result["issues"][0]["code"], "TOOL_EXECUTION_FAILED")
        self.assertEqual(calls, [FASTA_TO_SEQUENCE_BRIDGE])
        self.assertEqual(result["executed_steps"], [])
        self.assertEqual(result["artifact_pool"]["artifact_count"], 1)

    async def test_missing_declared_output_stops_without_partial_artifacts(self):
        plan = _structured(["compute_pI_mw"], [AMINO_ACID_SEQUENCE])

        async def fake_runner(**kwargs):
            return {"status": "success", "pI": 5.2}

        result = await execute_structured_plan(
            plan,
            {AMINO_ACID_SEQUENCE: TEST_SEQUENCE},
            runner=fake_runner,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "artifact_failure")
        self.assertEqual(result["issues"][0]["code"], "DECLARED_OUTPUT_MISSING")
        self.assertEqual(result["artifact_pool"]["artifact_count"], 1)

    async def test_runner_exception_is_caught(self):
        plan = _structured(["compute_gravy"], [AMINO_ACID_SEQUENCE])

        async def fake_runner(**kwargs):
            raise RuntimeError("offline runner crashed")

        result = await execute_structured_plan(
            plan,
            {AMINO_ACID_SEQUENCE: TEST_SEQUENCE},
            runner=fake_runner,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["issues"][0]["code"], "TOOL_RUNNER_EXCEPTION")

    def test_future_step_source_is_rejected_before_execution(self):
        plan = _structured(["compute_gravy"], [AMINO_ACID_SEQUENCE])
        invalid_plan = copy.deepcopy(plan)
        invalid_plan[0]["input_from"][AMINO_ACID_SEQUENCE] = "step_2"

        result = validate_runtime_plan(invalid_plan)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["issues"][0]["code"], "INVALID_SOURCE_ORDER")

    async def test_trace_snapshot_does_not_contain_raw_sequence(self):
        plan = _structured(["compute_gravy"], [AMINO_ACID_SEQUENCE])

        async def fake_runner(**kwargs):
            return {"status": "success", "gravy": -0.5}

        result = await execute_structured_plan(
            plan,
            {AMINO_ACID_SEQUENCE: TEST_SEQUENCE},
            runner=fake_runner,
        )
        safe_trace = json.dumps(
            {
                "executed_steps": result["executed_steps"],
                "artifact_pool": result["artifact_pool"],
            },
            ensure_ascii=False,
        )

        self.assertNotIn(TEST_SEQUENCE, safe_trace)


if __name__ == "__main__":
    unittest.main(verbosity=2)
