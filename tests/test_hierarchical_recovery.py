# -*- coding: utf-8 -*-
"""F1 分层恢复状态机与 v1.3 Trace 回归测试。

修改记录：
- 2026-09-05：增加目标工具保真契约回归，验证适配器可替换而
  分析目标不得被类型合法的重规划静默删除。
- 2026-09-04：覆盖原计划直行、确定性快速路径、一次 LLM 升级、失败关闭、
  Artifact 前置阻断和公开 Trace 记账。
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.controlled_repair_v2_runtime import load_controlled_repair_v2_dataset
from scripts.evaluation_trace import build_experiment_trace
from scripts.hierarchical_recovery import (
    derive_required_goal_tools,
    prepare_hierarchical_recovery,
)
from scripts.plan_sampling import build_plan_sample_id, safe_plan_trace, sha256_text
from scripts.run_biotooleval import _execute_under_condition
from scripts.type_checker import SINGLE_PROTEIN_FASTA_FILE


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _frozen_plan(chain: list[str], *, question_id: str = "bte_901") -> dict:
    """构造不包含真实题面或生物数据的冻结计划。"""
    raw_response = f"Plan Chain: {chain!r}"
    digest = sha256_text(raw_response)
    return {
        "schema_version": "1.0",
        "record_status": "captured",
        "experiment_id": "unit-f1",
        "question_id": question_id,
        "repeat_index": 0,
        "model": "offline-mock",
        "plan_sample_id": build_plan_sample_id(
            experiment_id="unit-f1",
            question_id=question_id,
            repeat_index=0,
            model="offline-mock",
            response_sha256=digest,
        ),
        "capture_outcome": "captured",
        "selected_attempt_index": 1,
        "response_sha256": digest,
        "planned_chain": list(chain),
        "unknown_tools": [],
        "retrieved_triples": [],
        "retrieved_tool_names": list(dict.fromkeys(chain)),
        "attempts": [
            {
                "attempt_index": 1,
                "attempt_kind": "content",
                "outcome": "captured",
                "exception_type": None,
                "response_sha256": digest,
                "raw_response": raw_response,
                "parsed_chain": list(chain),
                "unknown_tools": [],
                "latency_ms": 1.0,
            }
        ],
    }


def _trace_config() -> SimpleNamespace:
    """构造包含 F1 v1.3 扩展的只读测试配置。"""
    root = PROJECT_ROOT / "evaluation" / "biotooleval"
    return SimpleNamespace(
        EXPERIMENT_TRACE_SCHEMA_PATH=str(root / "experiment_trace_schema_v1.json"),
        EXPERIMENT_TRACE_EXTENSION_PATH=str(root / "experiment_trace_schema_e_v1.json"),
        EXPERIMENT_TRACE_BASELINE_EXTENSION_PATH=str(
            root / "experiment_trace_schema_baseline_v1.json"
        ),
        EXPERIMENT_TRACE_HYBRID_EXTENSION_PATH=str(
            root / "experiment_trace_schema_hybrid_v1.json"
        ),
    )


class HierarchicalRecoveryTests(unittest.IsolatedAsyncioTestCase):
    """验证 F1 的四条互斥路径及单次调用上限。"""

    async def test_valid_original_plan_uses_no_recovery_call(self):
        calls = []

        async def forbidden(**kwargs):
            calls.append(kwargs)
            raise AssertionError("合法原计划不得触发 LLM")

        original = _frozen_plan(["compute_gravy"])
        result = await prepare_hierarchical_recovery(
            question="synthetic",
            frozen_plan=original,
            available_input_types=["amino acid sequence"],
            initial_artifact_issue_codes=[],
            model="offline-mock",
            replan_callable=forbidden,
            max_output_tokens=64,
        )
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(calls, [])
        self.assertEqual(result["recovery"]["final_plan_source"], "original")
        self.assertEqual(result["recovery"]["extra_llm_calls"], 0)

    async def test_deterministic_bridge_is_fast_path_without_llm(self):
        calls = []

        async def forbidden(**kwargs):
            calls.append(kwargs)
            raise AssertionError("确定性可修复计划不得触发 LLM")

        original = _frozen_plan(["compute_gravy"])
        result = await prepare_hierarchical_recovery(
            question="synthetic FASTA task",
            frozen_plan=original,
            available_input_types=[SINGLE_PROTEIN_FASTA_FILE],
            initial_artifact_issue_codes=[],
            model="offline-mock",
            replan_callable=forbidden,
            max_output_tokens=64,
        )
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(calls, [])
        recovery = result["recovery"]
        self.assertTrue(recovery["deterministic_repair_applied"])
        self.assertEqual(recovery["final_plan_source"], "deterministic_repair")
        self.assertEqual(
            result["active_frozen_plan"]["planned_chain"],
            ["parse_protein_fasta", "compute_gravy"],
        )

    async def test_unrepairable_route_gets_exactly_one_validated_replan(self):
        calls = []

        async def replanner(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return {
                "status": "success",
                "raw_response": "Plan Chain: ['parse_protein_fasta', 'compute_gravy']",
                "planned_chain": ["parse_protein_fasta", "compute_gravy"],
                "usage": {"input_tokens": 80, "output_tokens": 9, "total_tokens": 89},
                "latency_ms": 3.0,
            }

        original = _frozen_plan(
            ["parse_multi_protein_fasta", "select_fasta_record", "compute_gravy"]
        )
        result = await prepare_hierarchical_recovery(
            question="synthetic FASTA task",
            frozen_plan=original,
            available_input_types=[SINGLE_PROTEIN_FASTA_FILE],
            initial_artifact_issue_codes=[],
            model="offline-mock",
            replan_callable=replanner,
            max_output_tokens=64,
        )
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(len(calls), 1)
        recovery = result["recovery"]
        self.assertTrue(recovery["deterministic_repair_attempted"])
        self.assertFalse(recovery["deterministic_repair_applied"])
        self.assertTrue(recovery["replan_applied"])
        self.assertEqual(recovery["extra_llm_calls"], 1)
        self.assertEqual(recovery["final_plan_source"], "llm_replan")
        self.assertTrue(result["replan_ledger_record"]["experiment_id"].endswith("-f1"))

    def test_goal_policy_is_registry_derived_and_keeps_analysis_tools(self):
        result = derive_required_goal_tools(
            [
                "parse_multi_protein_fasta",
                "select_fasta_record",
                "predict_signalpeptide",
                "predict_transmembrane",
            ]
        )
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(
            result["required_tool_names"],
            ["predict_signalpeptide", "predict_transmembrane"],
        )
        self.assertEqual(
            result["droppable_adapter_names"],
            ["parse_multi_protein_fasta", "select_fasta_record"],
        )

    async def test_type_valid_replan_that_drops_goal_tool_is_rejected(self):
        calls = []

        async def incomplete(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return {
                "status": "success",
                "raw_response": (
                    "Plan Chain: ['parse_protein_fasta', "
                    "'predict_signalpeptide']"
                ),
                "planned_chain": [
                    "parse_protein_fasta",
                    "predict_signalpeptide",
                ],
                "usage": {
                    "input_tokens": 80,
                    "output_tokens": 8,
                    "total_tokens": 88,
                },
                "latency_ms": 2.0,
            }

        original = _frozen_plan(
            [
                "parse_multi_protein_fasta",
                "select_fasta_record",
                "predict_signalpeptide",
                "predict_transmembrane",
            ]
        )
        result = await prepare_hierarchical_recovery(
            question="synthetic FASTA signal and transmembrane task",
            frozen_plan=original,
            available_input_types=[SINGLE_PROTEIN_FASTA_FILE],
            initial_artifact_issue_codes=[],
            model="offline-mock",
            replan_callable=incomplete,
            max_output_tokens=64,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0]["required_tool_names"],
            ["predict_signalpeptide", "predict_transmembrane"],
        )
        self.assertFalse(result["recovery"]["replan_applied"])
        self.assertEqual(
            result["recovery"]["replan_outcome"],
            "goal_incomplete_replan",
        )
        self.assertEqual(result["recovery"]["final_plan_source"], "rejected")
        self.assertEqual(
            result["replan_ledger_record"]["error_code"],
            "LLM_REPLAN_GOAL_INCOMPLETE",
        )

    async def test_invalid_replan_is_not_retried_or_applied(self):
        calls = []

        async def invalid(**kwargs):
            calls.append(kwargs)
            return {
                "status": "success",
                "raw_response": "Plan Chain: ['compute_gravy']",
                "planned_chain": ["compute_gravy"],
                "usage": {},
                "latency_ms": 2.0,
            }

        original = _frozen_plan(
            ["parse_multi_protein_fasta", "select_fasta_record", "compute_gravy"]
        )
        result = await prepare_hierarchical_recovery(
            question="synthetic FASTA task",
            frozen_plan=original,
            available_input_types=[SINGLE_PROTEIN_FASTA_FILE],
            initial_artifact_issue_codes=[],
            model="offline-mock",
            replan_callable=invalid,
            max_output_tokens=64,
        )
        self.assertEqual(len(calls), 1)
        self.assertFalse(result["recovery"]["replan_applied"])
        self.assertEqual(result["recovery"]["final_plan_source"], "rejected")
        self.assertEqual(
            result["active_frozen_plan"]["plan_sample_id"],
            original["plan_sample_id"],
        )

    async def test_invalid_artifact_blocks_replan(self):
        calls = []

        async def forbidden(**kwargs):
            calls.append(kwargs)
            raise AssertionError("Artifact 非法时不得让模型猜测")

        result = await prepare_hierarchical_recovery(
            question="synthetic",
            frozen_plan=_frozen_plan(["compute_gravy"]),
            available_input_types=[SINGLE_PROTEIN_FASTA_FILE],
            initial_artifact_issue_codes=["INVALID_AMINO_ACID_CHARACTER"],
            model="offline-mock",
            replan_callable=forbidden,
            max_output_tokens=64,
        )
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(calls, [])
        self.assertEqual(result["recovery"]["replan_outcome"], "blocked_initial_artifact")

    async def test_invalid_artifact_precedes_empty_type_check(self):
        """初始输入已有精确错误时，空类型集合不得被误报为 F1 故障。"""
        calls = []

        async def forbidden(**kwargs):
            calls.append(kwargs)
            raise AssertionError("Artifact 非法时不得调用重规划")

        result = await prepare_hierarchical_recovery(
            question="对非法序列 X 计算等电点",
            frozen_plan=_frozen_plan(["compute_pI_mw"]),
            available_input_types=[],
            initial_artifact_issue_codes=[
                "INVALID_AMINO_ACID_SEQUENCE_CONTENT"
            ],
            model="offline-mock",
            replan_callable=forbidden,
            max_output_tokens=64,
        )

        self.assertEqual(result["status"], "success", result)
        self.assertEqual(calls, [])
        self.assertEqual(
            result["recovery"]["replan_outcome"],
            "blocked_initial_artifact",
        )
        self.assertEqual(
            result["recovery"]["issue_codes_before"],
            ["INVALID_AMINO_ACID_SEQUENCE_CONTENT"],
        )
        self.assertEqual(result["recovery"]["extra_llm_calls"], 0)

    async def test_f1_runner_builds_v13_trace_and_preserves_original_id(self):
        runtime = SimpleNamespace(
            TYPE_PLANNING_ENABLED=False,
            TYPE_VALIDATION_ENABLED=False,
            TYPE_REPAIR_ENABLED=False,
            MODEL_NAME="original",
        )
        original = _frozen_plan(["compute_gravy"])

        async def agent(question, files, add_name=False, *, frozen_plan=None):
            del question, files, add_name
            chain = list(frozen_plan["planned_chain"])
            return {
                "answer": "offline",
                "reasoning": {
                    "plan_chain": chain,
                    "plan_chain_executed": chain,
                    "typed_planning": {
                        "enabled": True,
                        "validation_enabled": True,
                        "repair_enabled": True,
                        "repair_attempted": False,
                        "repair_applied": False,
                        "repair_strategy": "none",
                        "plan_before": chain,
                        "plan_after": chain,
                        "planning_issues": [],
                    },
                    "request_guard": {},
                    "plan_sampling": safe_plan_trace(frozen_plan),
                    "execution_summary": {
                        "completed": True,
                        "attempted_steps": len(chain),
                        "successful_steps": len(chain),
                        "failed_steps": 0,
                        "error_type": None,
                    },
                },
            }

        execution = await _execute_under_condition(
            agent,
            runtime,
            {"question": "synthetic", "file_path_list": [], "is_add_filename": False},
            condition="F1",
            timeout_seconds=1.0,
            frozen_plan=original,
            model="offline-mock",
            initial_types=[SINGLE_PROTEIN_FASTA_FILE],
            initial_artifact_issue_codes=[],
            replan_max_output_tokens=64,
        )
        self.assertEqual(execution["status"], "success", execution)
        result = build_experiment_trace(
            {
                "id": "bte_901",
                "split": "dev",
                "category": "multi_tool",
                "difficulty": "medium",
                "input": {"input_type": "protein_fasta_file"},
                "gold": {
                    "preferred_tool_chain": ["parse_protein_fasta", "compute_gravy"],
                    "acceptable_tool_chains": [],
                    "order_sensitive": True,
                    "expected_status": "success",
                    "expected_behavior": "execute",
                },
            },
            execution["agent_result"],
            condition="F1",
            run_context={
                "run_id": "unit-f1-r0",
                "repeat_index": 0,
                "model": "offline-mock",
                "temperature": 0.0,
                "git_commit": "abcdef1",
                "dataset_version": "unit-v1",
                "initial_types": [SINGLE_PROTEIN_FASTA_FILE],
                "initial_artifact_issue_codes": [],
                "total_latency_ms": execution["total_latency_ms"],
            },
            config=_trace_config(),
        )
        self.assertEqual(result["status"], "success", result)
        trace = result["trace"]
        self.assertEqual(trace["schema_version"], "1.3")
        plan = trace["observation"]["plan"]
        self.assertEqual(plan["original_plan_sample_id"], original["plan_sample_id"])
        self.assertEqual(plan["final_plan_source"], "deterministic_repair")
        self.assertEqual(plan["extra_llm_calls"], 0)

    async def test_controlled_30_all_use_deterministic_path_without_llm(self):
        """受控分母必须仍是 30/30，且不能偷用 LLM 完成。"""
        loaded = load_controlled_repair_v2_dataset(
            PROJECT_ROOT / "evaluation" / "biotooleval" / "controlled_repair_v2.json"
        )
        self.assertEqual(loaded["status"], "success", loaded)
        calls = []

        async def forbidden(**kwargs):
            calls.append(kwargs)
            raise AssertionError("Controlled-30 应全部走确定性快速路径")

        recovered = 0
        for case in loaded["items"]:
            result = await prepare_hierarchical_recovery(
                question="controlled component case",
                frozen_plan=_frozen_plan(
                    list(case["injected_plan"]),
                    question_id="bte_901",
                ),
                available_input_types=list(case["_initial_types"]),
                initial_artifact_issue_codes=[],
                model="offline-mock",
                replan_callable=forbidden,
                max_output_tokens=64,
            )
            self.assertEqual(result["status"], "success", result)
            recovery = result["recovery"]
            self.assertTrue(recovery["deterministic_repair_applied"], case["id"])
            self.assertEqual(recovery["extra_llm_calls"], 0, case["id"])
            self.assertEqual(
                result["active_frozen_plan"]["planned_chain"],
                case["gold"]["expected_repaired_plan"],
                case["id"],
            )
            recovered += 1
        self.assertEqual(recovered, 30)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
