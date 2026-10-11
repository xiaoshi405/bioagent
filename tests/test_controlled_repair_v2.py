# -*- coding: utf-8 -*-
"""
Controlled Repair Track v2 回归测试。

修改记录：
- 2026-08-30：覆盖30个配对机会、6个非同构家族、E2/E3真实本地闭环、
  家族级指标、Trace脱敏、no-overwrite及错误执行不计为恢复。
"""

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

from scripts.controlled_repair_v2_runtime import (
    load_controlled_repair_v2_dataset,
    run_controlled_v2_unit,
)
from scripts.run_controlled_repair import _load_json_validator
from scripts.run_controlled_repair_v2 import (
    build_v2_metrics,
    run_controlled_repair_v2_experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "evaluation" / "biotooleval"
DATASET_PATH = DATASET_DIR / "controlled_repair_v2.json"
DATASET_SCHEMA = DATASET_DIR / "controlled_repair_dataset_schema_v2.json"
TRACE_SCHEMA = DATASET_DIR / "controlled_repair_trace_schema_v2.json"


def _test_config(results_dir: Path) -> SimpleNamespace:
    """构造只写临时目录、显式绑定v2资产的测试配置。"""
    return SimpleNamespace(
        CONTROLLED_REPAIR_V2_DATASET_PATH=str(DATASET_PATH),
        CONTROLLED_REPAIR_V2_DATASET_SCHEMA_PATH=str(DATASET_SCHEMA),
        CONTROLLED_REPAIR_V2_TRACE_SCHEMA_PATH=str(TRACE_SCHEMA),
        CONTROLLED_REPAIR_V2_RESULTS_DIR=str(results_dir),
        CONTROLLED_REPAIR_TIMEOUT_SECONDS=15.0,
    )


class ControlledRepairV2DatasetTests(unittest.TestCase):
    """验证v2分母和拓扑家族是预先冻结、均衡且可物化的。"""

    def test_dataset_has_thirty_cases_across_six_balanced_families(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = load_controlled_repair_v2_dataset(
                config=_test_config(Path(temp_dir))
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["conditions"], ["E2", "E3"])
        self.assertEqual(result["family_count"], 6)
        self.assertEqual(len(result["items"]), 30)
        self.assertEqual(set(result["family_counts"].values()), {5})
        self.assertEqual(
            len({item["topology_signature"] for item in result["items"]}),
            6,
        )

    def test_dataset_covers_insert_prune_and_two_hop_repairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = load_controlled_repair_v2_dataset(
                config=_test_config(Path(temp_dir))
            )

        strategies = {
            item["gold"]["expected_repair_strategy"] for item in result["items"]
        }
        inserted_counts = {
            len(item["gold"]["expected_inserted_steps"])
            for item in result["items"]
        }
        removed_counts = {
            len(item["gold"]["expected_removed_steps"])
            for item in result["items"]
        }
        self.assertEqual(
            strategies,
            {"deterministic_bridge_insertion", "duplicate_invalid_step_pruning"},
        )
        self.assertEqual(inserted_counts, {0, 1, 2})
        self.assertEqual(removed_counts, {0, 2})

    def test_duplicate_case_design_cannot_inflate_denominator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "biotooleval"
            shutil.copytree(DATASET_DIR / "fixtures", root / "fixtures")
            dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
            duplicate = copy.deepcopy(dataset["cases"][0])
            duplicate["id"] = "crv2_031"
            dataset["cases"].append(duplicate)
            copied = root / "controlled_repair_v2.json"
            copied.write_text(
                json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            result = load_controlled_repair_v2_dataset(
                copied,
                config=_test_config(Path(temp_dir)),
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["detail_code"], "DUPLICATE_CASE_DESIGN")

    def test_duplicate_operation_shape_cannot_pose_as_new_family(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "biotooleval"
            shutil.copytree(DATASET_DIR / "fixtures", root / "fixtures")
            dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
            contracts = {
                item["family"]: item for item in dataset["family_contracts"]
            }
            parser_contract = contracts["multi_fasta_parser_omission"]
            selector_contract = contracts["multi_fasta_selector_omission"]
            for key in (
                "input_type",
                "repair_strategy",
                "injected_plan_template",
                "repaired_plan_template",
                "expected_inserted_steps",
                "expected_removed_steps",
            ):
                selector_contract[key] = copy.deepcopy(parser_contract[key])
            copied = root / "controlled_repair_v2.json"
            copied.write_text(
                json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            result = load_controlled_repair_v2_dataset(
                copied,
                config=_test_config(Path(temp_dir)),
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["detail_code"], "DUPLICATE_FAMILY_CONTRACT")


class ControlledRepairV2ExperimentTests(unittest.TestCase):
    """验证E2只阻断、E3按Gold修复并执行，且证据按家族报告。"""

    def test_complete_run_produces_sixty_safe_traces_and_family_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = asyncio.run(
                run_controlled_repair_v2_experiment(
                    experiment_id="unit-controlled-v2",
                    config=_test_config(root),
                    git_commit="0aca0dd",
                )
            )
            traces = [
                json.loads(line)
                for line in Path(result["trace_path"])
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            failures = Path(result["failure_path"]).read_text(encoding="utf-8")

        manifest = result["manifest"]
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(traces), 60)
        self.assertEqual(failures, "")
        self.assertEqual(manifest["failed_units"], 0)
        self.assertEqual(manifest["automatic_retries"], 0)
        self.assertEqual(manifest["family_count"], 6)
        self.assertEqual(
            manifest["metrics"]["overall"]["repair_recovery_rate"]["rate"],
            1.0,
        )
        self.assertEqual(
            manifest["metrics"]["family_macro_recovery_rate"]["macro_rate"],
            1.0,
        )
        self.assertEqual(len(manifest["metrics"]["by_family"]), 6)
        self.assertTrue(
            all(
                not trace["plan"]["repair_attempted"]
                and not trace["execution"]["completed"]
                for trace in traces
                if trace["condition"] == "E2"
            )
        )
        self.assertTrue(
            all(
                trace["plan"]["repair_applied"]
                and trace["plan"]["type_valid_after"]
                and trace["execution"]["completed"]
                and trace["evaluation"]["recovery_success"]
                for trace in traces
                if trace["condition"] == "E3"
            )
        )
        serialized = json.dumps(traces, ensure_ascii=False)
        self.assertNotIn(str(PROJECT_ROOT), serialized)
        self.assertNotIn("ACDEFGHIK", serialized)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "occupied.jsonl"
            output.write_text("user-owned\n", encoding="utf-8")
            result = asyncio.run(
                run_controlled_repair_v2_experiment(
                    experiment_id="no-overwrite-v2",
                    output_path=output,
                    config=_test_config(root),
                    git_commit="0aca0dd",
                )
            )
            unchanged = output.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["detail_code"], "OUTPUT_EXISTS")
        self.assertEqual(unchanged, "user-owned\n")

    def test_bad_chain_execution_is_not_counted_as_recovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = _test_config(Path(temp_dir))
            dataset = load_controlled_repair_v2_dataset(config=cfg)
            case = copy.deepcopy(
                next(
                    item
                    for item in dataset["items"]
                    if item["family"] == "pdb_chain_composite_bridge"
                )
            )
            case["chain_id"] = "Z"
            validator = _load_json_validator(TRACE_SCHEMA, "v2 Trace")["validator"]
            result = asyncio.run(
                run_controlled_v2_unit(
                    case,
                    condition="E3",
                    repeat_index=0,
                    experiment_id="invalid-chain-v2",
                    context={
                        "dataset_version": dataset["version"],
                        "dataset_sha256": dataset["dataset_sha256"],
                        "git_commit": "0aca0dd",
                        "python_version": "3.test",
                    },
                    trace_validator=validator,
                )
            )

        self.assertEqual(result["status"], "success")
        trace = result["trace"]
        self.assertTrue(trace["plan"]["repair_applied"])
        self.assertFalse(trace["execution"]["completed"])
        self.assertFalse(trace["evaluation"]["recovery_success"])

    def test_cli_can_start_from_project_root(self):
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_controlled_repair_v2.py"),
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


class ControlledRepairV2MetricTests(unittest.TestCase):
    """验证家族缺失时宏平均不可估计，而不是静默当作零或忽略。"""

    def test_missing_family_is_not_estimable(self):
        metrics = build_v2_metrics(
            [],
            planned_family_pairs={"family_a": 1, "family_b": 1},
        )

        macro = metrics["family_macro_recovery_rate"]
        self.assertEqual(macro["estimability_status"], "not_estimable")
        self.assertIsNone(macro["macro_rate"])
        self.assertEqual(macro["family_denominator"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
