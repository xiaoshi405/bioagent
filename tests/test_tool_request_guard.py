# -*- coding: utf-8 -*-
"""Tests for tool request guard."""

import unittest

from app.tools.request_guard import (
    evaluate_explicit_tool_request,
    extract_explicit_tool_invocations,
)
from app.tools.tool_runner import resolve_api_path


class TestExplicitToolRequestGuard(unittest.TestCase):

    def test_01_registered_tool_is_allowed(self):
        try:
            result = evaluate_explicit_tool_request("请调用 compute_gravy 工具计算GRAVY。")
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["decision"], "allow")
            self.assertTrue(result["invocation_intent"])
            self.assertEqual(result["unknown_tools"], [])
        except Exception as exc:
            raise AssertionError(f"已注册工具放行测试失败: {exc}") from exc

    def test_02_unregistered_tool_is_rejected(self):
        try:
            result = evaluate_explicit_tool_request(
                "请调用不存在的 ProteinFunctionOracle 工具预测精确功能。"
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["decision"], "reject")
            self.assertEqual(result["detail_code"], "UNREGISTERED_TOOL")
            self.assertEqual(result["unknown_tools"], ["ProteinFunctionOracle"])
        except Exception as exc:
            raise AssertionError(f"未注册工具拦截测试失败: {exc}") from exc

    def test_03_knowledge_question_is_not_invocation(self):
        try:
            result = evaluate_explicit_tool_request("ProteinFunctionOracle是什么工具？")
            self.assertEqual(result["decision"], "allow")
            self.assertFalse(result["invocation_intent"])
            self.assertEqual(result["requested_tools"], [])
        except Exception as exc:
            raise AssertionError(f"知识问答边界测试失败: {exc}") from exc

    def test_04_registered_and_unknown_calls_are_fail_closed(self):
        try:
            result = evaluate_explicit_tool_request(
                "请调用compute_gravy工具，并调用ProteinFunctionOracle工具。"
            )
            self.assertEqual(result["decision"], "reject")
            self.assertEqual(len(result["requested_tools"]), 2)
            self.assertEqual(result["unknown_tools"], ["ProteinFunctionOracle"])
        except Exception as exc:
            raise AssertionError(f"混合工具调用测试失败: {exc}") from exc

    def test_05_case_and_separator_aliases_are_registered(self):
        try:
            self.assertEqual(resolve_api_path("ComputeGRAVY"), "/api/compute_gravy")
            self.assertEqual(resolve_api_path("compute-gravy"), "/api/compute_gravy")
            self.assertEqual(resolve_api_path("COMPUTE_GRAVY"), "/api/compute_gravy")
            result = evaluate_explicit_tool_request("Call the compute-gravy tool.")
            self.assertEqual(result["decision"], "allow")
        except Exception as exc:
            raise AssertionError(f"工具别名规范化测试失败: {exc}") from exc

    def test_06_general_capability_request_is_not_forced_rejected(self):
        try:
            result = evaluate_explicit_tool_request("请预测这个蛋白质的精确生物学功能。")
            self.assertEqual(result["decision"], "allow")
            self.assertFalse(result["invocation_intent"])
            self.assertEqual(result["capability_scope"], "explicit_named_tools_only")
        except Exception as exc:
            raise AssertionError(f"一般能力请求边界测试失败: {exc}") from exc

    def test_07_invalid_sequence_remains_an_artifact_validation_task(self):
        try:
            result = evaluate_explicit_tool_request(
                "请计算MKWVTFISLL123X的等电点和分子量。"
            )
            self.assertEqual(result["decision"], "allow")
            self.assertFalse(result["invocation_intent"])
        except Exception as exc:
            raise AssertionError(f"非法序列边界测试失败: {exc}") from exc

    def test_08_plain_library_name_is_not_treated_as_tool_identifier(self):
        try:
            extracted = extract_explicit_tool_invocations("请使用Biopython计算该序列。")
            self.assertEqual(extracted["status"], "success")
            self.assertEqual(extracted["tool_names"], [])
        except Exception as exc:
            raise AssertionError(f"普通库名误报测试失败: {exc}") from exc

    def test_09_english_unregistered_tool_is_rejected(self):
        try:
            result = evaluate_explicit_tool_request(
                "Call the nonexistent ProteinFunctionOracle tool to predict function."
            )
            self.assertEqual(result["decision"], "reject")
            self.assertEqual(result["unknown_tools"], ["ProteinFunctionOracle"])
        except Exception as exc:
            raise AssertionError(f"英文未注册工具测试失败: {exc}") from exc

    def test_10_invalid_question_fails_closed_without_exception(self):
        try:
            result = evaluate_explicit_tool_request(None)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["decision"], "reject")
            self.assertEqual(result["detail_code"], "REQUEST_GUARD_ERROR")
        except Exception as exc:
            raise AssertionError(f"非法问题结构测试失败: {exc}") from exc


if __name__ == "__main__":
    unittest.main()
