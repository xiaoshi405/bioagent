# -*- coding: utf-8 -*-
"""Tests for controlled repair."""

import asyncio
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.initial_artifacts import build_initial_artifacts
from scripts.run_controlled_repair import (
    _load_json_validator,
    load_controlled_repair_dataset,
    run_controlled_repair_experiment,
    run_controlled_unit,
)
from scripts.type_checker import (
    AMINO_ACID_SEQUENCE,
    MULTI_PROTEIN_FASTA_FILE,
    PDB_CHAIN_IDENTIFIER,
    PDB_STRUCTURE_FILE,
    SINGLE_PROTEIN_FASTA_FILE,
    repair_tool_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "evaluation" / "biotooleval"
DATASET_PATH = DATASET_DIR / "controlled_repair_v1.json"
DATASET_SCHEMA = DATASET_DIR / "controlled_repair_dataset_schema_v1.json"
TRACE_SCHEMA = DATASET_DIR / "controlled_repair_trace_schema_v1.json"
MULTI_FASTA = DATASET_DIR / "fixtures" / "fasta" / "challenge_multirecord.fasta"


def _test_config(results_dir: Path, dataset_path: Path = DATASET_PATH) -> SimpleNamespace:
    return SimpleNamespace(
        CONTROLLED_REPAIR_DATASET_PATH=str(dataset_path),
        CONTROLLED_REPAIR_DATASET_SCHEMA_PATH=str(DATASET_SCHEMA),
        CONTROLLED_REPAIR_TRACE_SCHEMA_PATH=str(TRACE_SCHEMA),
        CONTROLLED_REPAIR_RESULTS_DIR=str(results_dir),
        CONTROLLED_REPAIR_TIMEOUT_SECONDS=10.0,
    )


class ControlledRepairDatasetTests(unittest.TestCase):

    def test_frozen_dataset_has_balanced_bridge_families(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = load_controlled_repair_dataset(
                config=_test_config(Path(temp_dir))
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["conditions"], ["B1", "B2"])
        self.assertEqual(len(result["items"]), 8)
        self.assertEqual(
            sum(item["family"] == "fasta_to_sequence" for item in result["items"]),
            4,
        )
        self.assertEqual(
            sum(item["family"] == "pdb_chain_to_sequence" for item in result["items"]),
            4,
        )
        self.assertEqual(len(result["dataset_sha256"]), 64)

    def test_fixture_hash_mismatch_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            copied_dataset = root / "controlled_repair_v1.json"
            copied_dataset.write_text(
                DATASET_PATH.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            source = DATASET_DIR / "fixtures" / "fasta" / "challenge_balanced.fasta"
            target = root / "fixtures" / "fasta" / source.name
            target.parent.mkdir(parents=True)
            shutil.copy2(source, target)
            target.write_text(">changed\nACDEFGHIK\n", encoding="utf-8")
            result = load_controlled_repair_dataset(
                copied_dataset,
                config=_test_config(root, copied_dataset),
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["detail_code"], "FIXTURE_HASH_MISMATCH")
        self.assertEqual(result["details"]["case_id"], "crt_fasta_001")


class ControlledRepairExperimentTests(unittest.TestCase):

    def test_complete_paired_run_produces_estimable_component_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg = _test_config(root)
            result = asyncio.run(
                run_controlled_repair_experiment(
                    experiment_id="unit-controlled-v1",
                    config=cfg,
                    git_commit="abcdef1",
                )
            )
            trace_path = Path(result["trace_path"])
            manifest_path = Path(result["manifest_path"])
            failure_path = Path(result["failure_path"])
            traces = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            failure_text = failure_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(traces), 16)
        self.assertEqual(failure_text, "")
        self.assertEqual(manifest["failed_units"], 0)
        self.assertEqual(manifest["automatic_retries"], 0)
        self.assertEqual(
            manifest["metrics"]["pair_coverage"],
            {
                "estimability_status": "estimable",
                "numerator": 8,
                "denominator": 8,
                "rate": 1.0,
            },
        )
        self.assertEqual(
            manifest["metrics"]["repair_recovery_rate"]["denominator"],
            8,
        )
        self.assertEqual(
            manifest["metrics"]["repair_precision"]["denominator"],
            8,
        )
        self.assertTrue(
            all(
                not trace["plan"]["repair_attempted"]
                and not trace["execution"]["completed"]
                for trace in traces
                if trace["condition"] == "B1"
            )
        )
        self.assertTrue(
            all(
                trace["plan"]["repair_applied"]
                and trace["plan"]["type_valid_after"]
                and trace["execution"]["completed"]
                and trace["evaluation"]["recovery_success"]
                for trace in traces
                if trace["condition"] == "B2"
            )
        )
        serialized = json.dumps(traces, ensure_ascii=False)
        self.assertNotIn(str(PROJECT_ROOT), serialized)
        self.assertNotIn("MVLSPADKT", serialized)

    def test_cli_script_can_start_from_project_root(self):
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_controlled_repair.py"),
                "--help",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--experiment-id", result.stdout)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg = _test_config(root)
            first = asyncio.run(
                run_controlled_repair_experiment(
                    experiment_id="no-overwrite",
                    config=cfg,
                    git_commit="abcdef1",
                )
            )
            original = Path(first["trace_path"]).read_bytes()
            second = asyncio.run(
                run_controlled_repair_experiment(
                    experiment_id="no-overwrite",
                    config=cfg,
                    git_commit="abcdef1",
                )
            )
            unchanged = Path(first["trace_path"]).read_bytes()

        self.assertEqual(first["status"], "success")
        self.assertEqual(second["status"], "error")
        self.assertEqual(second["detail_code"], "OUTPUT_EXISTS")
        self.assertEqual(original, unchanged)

    def test_nonexistent_chain_is_not_counted_as_recovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = _test_config(Path(temp_dir))
            dataset = load_controlled_repair_dataset(config=cfg)
            case = copy.deepcopy(
                next(item for item in dataset["items"] if item["family"] == "pdb_chain_to_sequence")
            )
            case["input"]["chain_id"] = "Z"
            validator_result = _load_json_validator(TRACE_SCHEMA, "Trace")
            result = asyncio.run(
                run_controlled_unit(
                    case,
                    condition="B2",
                    repeat_index=0,
                    experiment_id="invalid-chain",
                    context={
                        "dataset_version": dataset["version"],
                        "dataset_sha256": dataset["dataset_sha256"],
                        "git_commit": "abcdef1",
                        "python_version": "3.test",
                    },
                    trace_validator=validator_result["validator"],
                )
            )

        self.assertEqual(result["status"], "success")
        trace = result["trace"]
        self.assertTrue(trace["plan"]["repair_applied"])
        self.assertFalse(trace["execution"]["completed"])
        self.assertFalse(trace["evaluation"]["recovery_success"])


class ControlledRepairBoundaryTests(unittest.TestCase):

    def test_missing_chain_id_is_not_repairable(self):
        result = repair_tool_plan(["compute_gravy"], [PDB_STRUCTURE_FILE])
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["repairable"])
        self.assertIsNone(result["repaired_plan"])

    def test_multi_record_fasta_is_registered_without_record_guessing(self):
        result = build_initial_artifacts(
            "受控输入：单记录蛋白质 FASTA。",
            [str(MULTI_FASTA)],
        )
        self.assertEqual(result["status"], "success")
        self.assertIn(MULTI_PROTEIN_FASTA_FILE, result["artifacts"])
        self.assertEqual(
            result["warnings"][0]["code"],
            "MULTI_FASTA_WITHOUT_RECORD_IDENTIFIER",
        )

    def test_already_valid_plan_is_idempotent(self):
        result = repair_tool_plan(["compute_gravy"], [AMINO_ACID_SEQUENCE])
        self.assertTrue(result["repairable"])
        self.assertFalse(result["changed"])
        self.assertEqual(result["repaired_plan"], ["compute_gravy"])

    def test_unknown_tool_is_not_guessed(self):
        result = repair_tool_plan(["unknown_protein_tool"], [AMINO_ACID_SEQUENCE])
        self.assertFalse(result["repairable"])
        self.assertEqual(result["blocking_issue"]["code"], "UNKNOWN_TOOL")

    def test_ambiguous_bridge_sources_are_not_guessed(self):
        result = repair_tool_plan(
            ["compute_gravy"],
            [
                SINGLE_PROTEIN_FASTA_FILE,
                PDB_STRUCTURE_FILE,
                PDB_CHAIN_IDENTIFIER,
            ],
        )
        self.assertFalse(result["repairable"])
        self.assertEqual(len(result["blocking_issue"]["bridge_candidates"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
