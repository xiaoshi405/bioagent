# -*- coding: utf-8 -*-
"""
v0.3b 类型约束规划主链路协调测试。

修改记录：
- 2026-09-05：覆盖空计划拒绝对非法序列诊断和能力缺口的
  结构化透传，禁止回退为不可行动的泛化文案。
- 2026-08-27：覆盖E1仅启用Artifact Pool绑定、但不执行类型门禁或修复。
- 2026-08-15：覆盖重复无效记录选择剪枝的运行时闭环、安全证明透传与脱敏。
- 2026-08-12：覆盖显式短非法序列从初始校验到运行时可操作反馈的透传。
- 2026-08-09：覆盖缺失PDB链标识和多记录FASTA的可操作错误解释。
- 2026-08-08：覆盖初始Artifact失败时B1/B2 repair开关仍进入安全轨迹。
- 2026-07-26：覆盖配置旁路、直接序列fan-out、FASTA保守修复、
  修复关闭时阻断、工具失败阻断及安全轨迹脱敏。
"""

import copy
import json
import unittest
from pathlib import Path

from scripts.typed_planning_runtime import (
    build_actionable_empty_plan_rejection,
    run_typed_planning,
)
from scripts.type_checker import (
    AMINO_ACID_SEQUENCE,
    FASTA_RECORD_IDENTIFIER_PAIR,
    FASTA_TO_SEQUENCE_BRIDGE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FASTA_FIXTURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "fasta" / "single_protein.fasta"
)
MULTI_FASTA_FIXTURE = (
    PROJECT_ROOT
    / "evaluation"
    / "biotooleval"
    / "fixtures"
    / "fasta"
    / "challenge_multirecord.fasta"
)
PDB_FIXTURE = (
    PROJECT_ROOT
    / "evaluation"
    / "biotooleval"
    / "fixtures"
    / "pdb"
    / "mini_two_chain.pdb"
)
TEST_SEQUENCE = "MKWVTFISLLLLFSSAYSRGVFRR"


class DisabledConfig:
    TYPE_PLANNING_ENABLED = False
    TYPE_REPAIR_ENABLED = False


class CheckOnlyConfig:
    TYPE_PLANNING_ENABLED = True
    TYPE_REPAIR_ENABLED = False


class RuntimeOnlyConfig:
    TYPE_PLANNING_ENABLED = True
    TYPE_VALIDATION_ENABLED = False
    TYPE_REPAIR_ENABLED = False


class RepairConfig:
    TYPE_PLANNING_ENABLED = True
    TYPE_REPAIR_ENABLED = True


class TypedPlanningRuntimeTests(unittest.IsolatedAsyncioTestCase):
    """验证v0.3b协调层不依赖KG和真实HTTP即可回归。"""

    async def test_disabled_config_bypasses_without_running_tool(self):
        calls = []

        async def fake_runner(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return {"status": "success", "gravy": 0.1}

        result = await run_typed_planning(
            ["compute_gravy"],
            TEST_SEQUENCE,
            [],
            config=DisabledConfig(),
            runner=fake_runner,
        )

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["enabled"])
        self.assertTrue(result["bypassed"])
        self.assertEqual(calls, [])

    async def test_direct_sequence_fan_out_executes_without_second_llm(self):
        calls = []

        async def fake_runner(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            if kwargs["tool_name"] == "compute_pI_mw":
                return {
                    "status": "success",
                    "pI": 5.2,
                    "molecular_weight": 2800.0,
                }
            return {"status": "success", "gravy": -0.4}

        result = await run_typed_planning(
            ["compute_pI_mw", "compute_gravy"],
            f"请分析 {TEST_SEQUENCE}",
            [],
            config=CheckOnlyConfig(),
            runner=fake_runner,
        )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["completed"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[0]["typed_inputs"][AMINO_ACID_SEQUENCE],
            TEST_SEQUENCE,
        )
        self.assertEqual(
            calls[1]["typed_inputs"][AMINO_ACID_SEQUENCE],
            TEST_SEQUENCE,
        )

    async def test_fasta_missing_bridge_is_repaired_and_executed(self):
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

        result = await run_typed_planning(
            ["compute_gravy"],
            "请计算上传蛋白的GRAVY值",
            [str(FASTA_FIXTURE)],
            config=RepairConfig(),
            runner=fake_runner,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["planning"]["plan_after"],
            [FASTA_TO_SEQUENCE_BRIDGE, "compute_gravy"],
        )
        self.assertTrue(result["planning"]["repair_applied"])
        self.assertEqual(
            calls[1]["typed_inputs"][AMINO_ACID_SEQUENCE],
            parsed_sequence,
        )

    async def test_duplicate_record_selection_pruning_is_traced_safely(self):
        calls = []

        async def fake_runner(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            tool_name = kwargs["tool_name"]
            if tool_name == "parse_multi_protein_fasta":
                return {
                    "status": "success",
                    "collection": {
                        "challenge_record_one": {},
                        "challenge_record_two": {},
                    },
                }
            if tool_name == "align_fasta_records":
                self.assertEqual(
                    kwargs["typed_inputs"][FASTA_RECORD_IDENTIFIER_PAIR],
                    ["challenge_record_one", "challenge_record_two"],
                )
                return {"status": "success", "alignment": {"length": 20}}
            return {"status": "success", "statistics": {"identity": 90.0}}

        result = await run_typed_planning(
            [
                "parse_multi_protein_fasta",
                "select_fasta_record",
                "select_fasta_record",
                "align_fasta_records",
                "summarize_pairwise_alignment",
            ],
            "请比较record_ids=challenge_record_one,challenge_record_two的比对统计",
            [str(MULTI_FASTA_FIXTURE)],
            config=RepairConfig(),
            runner=fake_runner,
        )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["completed"])
        self.assertEqual(
            [item["tool_name"] for item in calls],
            [
                "parse_multi_protein_fasta",
                "align_fasta_records",
                "summarize_pairwise_alignment",
            ],
        )
        safe_trace = result["safe_trace"]
        self.assertEqual(
            safe_trace["repair_strategy"],
            "duplicate_invalid_step_pruning",
        )
        self.assertEqual(len(safe_trace["repair_operations"]), 2)
        self.assertTrue(
            all(
                item["candidate_plan_type_valid"]
                for item in safe_trace["repair_operations"]
            )
        )
        safe_text = json.dumps(safe_trace, ensure_ascii=False)
        self.assertNotIn(str(MULTI_FASTA_FIXTURE.resolve()), safe_text)

    async def test_invalid_plan_is_blocked_when_repair_is_disabled(self):
        calls = []

        async def fake_runner(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return {"status": "success", "gravy": 0.1}

        result = await run_typed_planning(
            ["compute_gravy"],
            "请计算上传蛋白的GRAVY值",
            [str(FASTA_FIXTURE)],
            config=CheckOnlyConfig(),
            runner=fake_runner,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "plan_failure")
        self.assertEqual(calls, [])

    async def test_runtime_only_condition_does_not_preempt_with_type_gate(self):
        """E1应进入Artifact Pool后自然失败，而不是在类型检查阶段拒绝。"""
        calls = []

        async def fake_runner(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return {"status": "success", "gravy": 0.1}

        result = await run_typed_planning(
            ["compute_gravy"],
            "请计算上传蛋白的GRAVY值",
            [str(FASTA_FIXTURE)],
            config=RuntimeOnlyConfig(),
            runner=fake_runner,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "artifact_failure")
        self.assertEqual(result["detail_code"], "ARTIFACT_NOT_FOUND")
        self.assertFalse(result["safe_trace"]["validation_enabled"])
        self.assertTrue(result["safe_trace"]["validation_bypassed"])
        self.assertEqual(calls, [])

    async def test_missing_pdb_chain_id_returns_actionable_diagnostic(self):
        """缺少用户意图时应指出chain_id，而不是只返回泛化类型错误。"""
        calls = []

        async def fake_runner(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return {"status": "success"}

        result = await run_typed_planning(
            ["select_pdb_chain", "compute_gravy"],
            "请报告这个多链PDB中目标链的GRAVY值",
            [str(PDB_FIXTURE)],
            config=CheckOnlyConfig(),
            runner=fake_runner,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "plan_failure")
        self.assertEqual(result["detail_code"], "MISSING_PDB_CHAIN_ID")
        self.assertIn("chain_id", result["message"])
        self.assertIn("不会自动猜测", result["message"])
        self.assertEqual(
            result["required_user_inputs"][0]["name"],
            "chain_id",
        )
        self.assertEqual(
            result["issues"][0]["missing_types"],
            ["PDB chain identifier"],
        )
        self.assertEqual(calls, [])

    async def test_multirecord_fasta_requests_explicit_record(self):
        """多记录FASTA应说明选择边界并给出用户可执行的下一步。"""
        result = await run_typed_planning(
            ["select_fasta_record", "compute_gravy"],
            "请报告目标记录的GRAVY值",
            [str(MULTI_FASTA_FIXTURE)],
            config=CheckOnlyConfig(),
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "plan_failure")
        self.assertEqual(result["detail_code"], "MISSING_FASTA_RECORD_IDENTIFIER")
        self.assertIn("record_id", result["message"])
        self.assertIn("不会猜测", result["message"])
        self.assertEqual(
            result["required_user_inputs"][0]["name"],
            "record_id",
        )
        self.assertIn(
            "FASTA record identifier",
            result["issues"][0]["missing_types"],
        )

    async def test_short_invalid_sequence_returns_actionable_diagnostic(self):
        """显式短序列应报告非法字符和可执行修改动作，且不得调用工具。"""
        calls = []

        async def fake_runner(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return {"status": "success"}

        result = await run_typed_planning(
            ["compute_pI_mw"],
            "请分析序列 X",
            [],
            config=CheckOnlyConfig(),
            runner=fake_runner,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "initial_artifact_failure")
        self.assertEqual(
            result["detail_code"],
            "INVALID_AMINO_ACID_SEQUENCE_CONTENT",
        )
        self.assertEqual(result["issues"][0]["invalid_characters"], ["X"])
        self.assertIn("X", result["message"])
        self.assertIn("删除或替换", result["message"])
        self.assertEqual(
            result["required_user_inputs"][0]["name"],
            "sequence",
        )
        self.assertEqual(calls, [])

    async def test_empty_plan_rejection_preserves_invalid_characters(self):
        """空计划分支必须保留非法字符和修改建议。"""
        typed_result = await run_typed_planning(
            [],
            "请分析序列 MKWVTFISLL123X 的pI和分子量",
            [],
            config=CheckOnlyConfig(),
        )
        rejection = build_actionable_empty_plan_rejection(typed_result)
        payload = rejection["payload"]

        self.assertEqual(rejection["status"], "success")
        self.assertEqual(
            payload["detail_code"],
            "INVALID_AMINO_ACID_SEQUENCE_CONTENT",
        )
        self.assertEqual(
            payload["issues"][0]["invalid_characters"],
            ["1", "2", "3", "X"],
        )
        self.assertIn("删除或替换", rejection["final_message"])
        self.assertIn("系统未调用任何工具", rejection["final_message"])

    async def test_no_artifact_rejection_explains_input_and_capability_gap(self):
        """无输入且能力不匹配时应同时说明两类阻断原因。"""
        typed_result = await run_typed_planning(
            [],
            (
                "未提供任何数据，请完成核酸反向互补、"
                "ORF识别和联网数据库检索。"
            ),
            [],
            config=CheckOnlyConfig(),
        )
        rejection = build_actionable_empty_plan_rejection(typed_result)
        final_message = rejection["final_message"]

        self.assertEqual(
            rejection["payload"]["detail_code"],
            "NO_INITIAL_ARTIFACT",
        )
        self.assertIn("未识别到受支持的初始生物数据对象", final_message)
        self.assertIn("核酸反向互补或ORF识别", final_message)
        self.assertIn("联网数据库检索", final_message)
        self.assertIn("不会编造", final_message)
        self.assertEqual(
            rejection["payload"]["required_user_inputs"][0]["name"],
            "biological_input",
        )

    def test_empty_plan_rejection_fallback_is_defensive(self):
        """诊断对象异常时仍应返回标准错误而不是抛出异常。"""
        rejection = build_actionable_empty_plan_rejection(object())

        self.assertIn(rejection["status"], {"success", "error"})
        self.assertEqual(
            rejection["payload"]["error_type"],
            "empty_or_unparseable_plan",
        )
        self.assertIn("系统未调用任何工具", rejection["final_message"])

    async def test_tool_failure_stops_and_is_not_rewritten_as_success(self):
        calls = []
        valid_candidate = "MKWVTFISLLFLFSSAYSRGVFRR"

        async def fake_runner(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return {
                "status": "error",
                "message": "序列包含非法字符",
            }

        result = await run_typed_planning(
            ["compute_pI_mw", "compute_gravy"],
            valid_candidate,
            [],
            config=CheckOnlyConfig(),
            runner=fake_runner,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "tool_failure")
        self.assertEqual(len(calls), 1)
        self.assertFalse(result["completed"])

    async def test_initial_artifact_failure_preserves_repair_flag(self):
        """即使尚未进入规划，B2安全轨迹也必须保留实验条件。"""
        result = await run_typed_planning(
            ["compute_pI_mw"],
            "这不是蛋白质序列，也没有上传支持的文件。",
            [],
            config=RepairConfig(),
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "initial_artifact_failure")
        self.assertTrue(result["safe_trace"]["enabled"])
        self.assertTrue(result["safe_trace"]["repair_enabled"])
        self.assertFalse(result["safe_trace"]["repair_attempted"])
        self.assertFalse(result["safe_trace"]["repair_applied"])

    async def test_safe_trace_does_not_leak_sequence_or_full_path(self):
        async def fake_runner(**kwargs):
            return {
                "status": "success",
                "sequence": TEST_SEQUENCE,
                "sequence_length": len(TEST_SEQUENCE),
            }

        result = await run_typed_planning(
            [FASTA_TO_SEQUENCE_BRIDGE],
            "解析上传的FASTA",
            [str(FASTA_FIXTURE)],
            config=CheckOnlyConfig(),
            runner=fake_runner,
        )
        safe_text = json.dumps(result["safe_trace"], ensure_ascii=False)

        self.assertEqual(result["status"], "success")
        self.assertNotIn(TEST_SEQUENCE, safe_text)
        self.assertNotIn(str(FASTA_FIXTURE.resolve()), safe_text)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    except Exception as exc:
        print(f"v0.3b协调层测试运行失败: {exc}")
        raise
