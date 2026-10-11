# -*- coding: utf-8 -*-
"""请求级工具值参数回归测试。

修改记录：
- 2026-09-04：覆盖ANM显式参数提取、契约校验、接口作用域与默认值保留。
- 2026-09-04：覆盖同一参数的多值冲突，禁止静默选值或回退默认值。
"""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from app.tools import tool_runner
from app.tools.tool_parameters import (
    ANM_API_PATH,
    PDB_CHAIN_API_PATH,
    build_tool_parameter_context,
    extract_pdb_options,
    resolve_tool_parameters,
    validate_tool_parameters,
)
from scripts.type_checker import PDB_STRUCTURE_FILE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDB_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "pdb" / "1CRN.pdb"


class ToolParameterContextTests(unittest.TestCase):
    """验证参数来自用户文本且只绑定到注册接口。"""

    def test_extracts_chinese_anm_parameters_and_preserves_chain_case(self):
        context = build_tool_parameter_context(
            "请对b链按cutoff=10.5计算前三个最低频非零模态"
        )
        resolved = resolve_tool_parameters(context, ANM_API_PATH)

        self.assertEqual(resolved["status"], "success")
        self.assertEqual(
            resolved["parameters"],
            {"n_modes": 3, "cutoff": 10.5, "chain_id": "b"},
        )

    def test_extracts_named_mode_count(self):
        context = build_tool_parameter_context("n_modes=12，计算ANM")
        resolved = resolve_tool_parameters(context, ANM_API_PATH)

        self.assertEqual(resolved["status"], "success")
        self.assertEqual(resolved["parameters"], {"n_modes": 12})

    def test_unspecified_parameters_remain_absent_for_api_defaults(self):
        context = build_tool_parameter_context("分析这个PDB的整体结构")
        resolved = resolve_tool_parameters(context, ANM_API_PATH)

        self.assertEqual(resolved["status"], "success")
        self.assertEqual(resolved["parameters"], {})

    def test_invalid_anm_value_is_scoped_and_does_not_break_chain_selection(self):
        context = build_tool_parameter_context("选择链A并计算前21个ANM模态")
        anm_result = resolve_tool_parameters(context, ANM_API_PATH)
        chain_result = resolve_tool_parameters(context, PDB_CHAIN_API_PATH)

        self.assertEqual(anm_result["status"], "error")
        self.assertEqual(
            anm_result["issues"][0]["code"],
            "TOOL_PARAMETER_OUT_OF_RANGE",
        )
        self.assertEqual(chain_result["status"], "success")
        self.assertEqual(chain_result["parameters"], {"chain_id": "A"})

    def test_decimal_mode_count_is_rejected_instead_of_using_default(self):
        context = build_tool_parameter_context("计算3.5个ANM模态")
        result = resolve_tool_parameters(context, ANM_API_PATH)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["issues"][0]["code"], "INVALID_TOOL_PARAMETER")

    def test_conflicting_mode_counts_are_rejected(self):
        context = build_tool_parameter_context(
            "n_modes=3，但请计算前五个ANM模态"
        )
        result = resolve_tool_parameters(context, ANM_API_PATH)

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "CONFLICTING_TOOL_PARAMETER",
        )

    def test_repeated_equivalent_values_do_not_create_conflict(self):
        context = build_tool_parameter_context(
            "n_modes=3，即计算3个ANM模态"
        )
        result = resolve_tool_parameters(context, ANM_API_PATH)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["parameters"], {"n_modes": 3})

    def test_conflicting_chain_ids_are_scoped_to_pdb_tools(self):
        context = build_tool_parameter_context("选择链A，但chain_id=B")
        anm_result = resolve_tool_parameters(context, ANM_API_PATH)
        chain_result = resolve_tool_parameters(context, PDB_CHAIN_API_PATH)

        self.assertEqual(anm_result["status"], "error")
        self.assertEqual(chain_result["status"], "error")
        self.assertEqual(
            chain_result["issues"][0]["code"],
            "CONFLICTING_TOOL_PARAMETER",
        )

    def test_unregistered_parameter_is_rejected(self):
        result = validate_tool_parameters(ANM_API_PATH, {"answer": 3})

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "UNSUPPORTED_TOOL_PARAMETER",
        )

    def test_legacy_extractor_uses_same_contract(self):
        self.assertEqual(
            extract_pdb_options("计算链A前3个ANM模态，截断距离为10埃"),
            {"n_modes": 3, "cutoff": 10.0, "chain_id": "A"},
        )

    def test_legacy_extractor_keeps_valid_chain_when_anm_value_is_invalid(self):
        self.assertEqual(
            extract_pdb_options("选择链A并计算前21个ANM模态"),
            {"chain_id": "A"},
        )


class ToolParameterRunnerTests(unittest.IsolatedAsyncioTestCase):
    """验证普通与类型化执行形成完全相同的显式ANM请求参数。"""

    async def test_baseline_and_typed_paths_share_anm_parameters(self):
        payloads = []
        context = build_tool_parameter_context(
            "计算链B前三个ANM模态，cutoff=10"
        )

        class FakeResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def json(self):
                return {"status": "success", "eigenvalues": [0.1, 0.2, 0.3]}

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            def post(self, url, json):
                payloads.append(dict(json))
                return FakeResponse()

        with patch.object(tool_runner.aiohttp, "ClientSession", FakeSession):
            baseline = await tool_runner.run_tool(
                "calculate_protein_anm",
                "这段后续文本不包含参数",
                [str(PDB_FIXTURE)],
                tool_parameters=context,
            )
            typed = await tool_runner.run_tool(
                "calculate_protein_anm",
                "",
                [],
                typed_inputs={PDB_STRUCTURE_FILE: str(PDB_FIXTURE)},
                tool_parameters=context,
            )

        self.assertEqual(json.loads(baseline)["status"], "success")
        self.assertEqual(json.loads(typed)["status"], "success")
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[0]["n_modes"], 3)
        self.assertEqual(payloads[0]["cutoff"], 10.0)
        self.assertEqual(payloads[0]["chain_id"], "B")

    async def test_invalid_anm_parameter_returns_structured_error_without_http(self):
        context = build_tool_parameter_context("计算前21个ANM模态")
        with patch.object(
            tool_runner.aiohttp,
            "ClientSession",
            side_effect=AssertionError("非法参数不得进入HTTP调用"),
        ):
            result = await tool_runner.run_tool(
                "calculate_protein_anm",
                "",
                [str(PDB_FIXTURE)],
                tool_parameters=context,
            )

        parsed = json.loads(result)
        self.assertEqual(parsed["status"], "error")
        self.assertEqual(parsed["code"], "TOOL_PARAMETER_OUT_OF_RANGE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
