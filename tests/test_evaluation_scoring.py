# -*- coding: utf-8 -*-
"""
BioAgent v0.3e确定性评分、盲评队列和人工标签回填测试。

修改记录：
- 2026-08-09：验证盲审队列在保持条件与题号盲化的同时，提供原始计划、
  初始对象类型和非敏感输入内容证据。
- 2026-07-30：覆盖数值容差、错误值、缺失证据、Type-III人工边界、
  repair/reject标签、条件盲化和非覆盖式人工标签回填。
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluation_scoring import (
    apply_manual_review_labels,
    apply_scoring_to_trace,
    persist_scoring_bundle,
    score_experiment_trace,
)
from scripts.enrich_review_queue import enrich_review_queue
from scripts.evaluation_trace import (
    append_experiment_trace,
    build_experiment_trace,
    load_experiment_traces,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_SCHEMA = (
    PROJECT_ROOT
    / "evaluation"
    / "biotooleval"
    / "experiment_trace_schema_v1.json"
)


def _config(results_dir: Path) -> SimpleNamespace:
    """构造评分测试配置。"""
    try:
        return SimpleNamespace(
            EXPERIMENT_TRACE_SCHEMA_PATH=str(TRACE_SCHEMA),
            EXPERIMENT_RESULTS_DIR=str(results_dir),
        )
    except Exception as exc:
        raise AssertionError(f"构造评分测试配置失败: {exc}") from exc


def _question(
    *,
    expected_behavior: str = "execute",
    expected_status: str = "success",
) -> dict:
    """构造包含结构化Gold的最小题目。"""
    try:
        return {
            "id": "bte_801",
            "split": "pilot",
            "category": "single_tool",
            "difficulty": "easy",
            "question_zh": "计算给定蛋白质序列的等电点和分子量。",
            "question_en": "Calculate pI and molecular weight.",
            "input": {
                "input_type": "amino_acid_sequence",
                "sequence": "MKWVTFISLLLLFSSAYSR",
                "source_type": "synthetic",
            },
            "gold": {
                "preferred_tool_chain": ["compute_pI_mw"],
                "acceptable_tool_chains": [],
                "order_sensitive": False,
                "expected_status": expected_status,
                "expected_behavior": expected_behavior,
                "expected_output": {
                    "pI": 6.5,
                    "molecular_weight": 1234.5,
                },
                "tolerance": {
                    "pI": 0.1,
                    "molecular_weight": 1.0,
                },
            },
            "provenance": {
                "gold_generator": "offline_test",
                "verified_by": "unit_test",
                "verified_at": "2026-07-30",
            },
        }
    except Exception as exc:
        raise AssertionError(f"构造评分题目失败: {exc}") from exc


def _agent_result(
    answer: str,
    *,
    completed: bool = True,
    failed_steps: int = 0,
    attempted_steps: int = 1,
    plan_chain: list[str] | None = None,
) -> dict:
    """构造不依赖真实模型和工具服务的Agent结果。"""
    try:
        chain = (
            list(plan_chain)
            if isinstance(plan_chain, list)
            else ["compute_pI_mw"]
        )
        return {
            "answer": answer,
            "reasoning": {
                "plan_chain": chain,
                "plan_chain_executed": chain,
                "tool_steps": [
                    {
                        "tool_name": tool,
                        "status": (
                            "success" if failed_steps == 0 else "error"
                        ),
                    }
                    for tool in chain
                ],
                "execution_summary": {
                    "completed": completed,
                    "attempted_steps": attempted_steps,
                    "successful_steps": max(
                        0,
                        attempted_steps - failed_steps,
                    ),
                    "failed_steps": failed_steps,
                    "error_type": (
                        None if completed else "tool_failure"
                    ),
                },
                "typed_planning": None,
            },
        }
    except Exception as exc:
        raise AssertionError(f"构造Agent评分结果失败: {exc}") from exc


def _trace(question: dict, agent_result: dict, config: object) -> dict:
    """构造未评分B0 Trace。"""
    try:
        result = build_experiment_trace(
            question,
            agent_result,
            condition="B0",
            run_context={
                "run_id": "score-b0-r0",
                "repeat_index": 0,
                "model": "offline-score",
                "temperature": 0.0,
                "seed": None,
                "git_commit": "abcdef1",
                "dataset_version": "score-test-1.0",
                "kg_snapshot": "kg-test",
                "python_version": "3.14.4",
                "total_latency_ms": 1.0,
            },
            config=config,
        )
        if result.get("status") != "success":
            raise AssertionError(result)
        return result["trace"]
    except Exception as exc:
        raise AssertionError(f"构造评分Trace失败: {exc}") from exc


class DeterministicScoringTests(unittest.TestCase):
    """验证自动标签只在证据充分时给出布尔值。"""

    def test_structured_numeric_gold_matches_within_tolerance(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                cfg = _config(Path(temp_dir))
                question = _question()
                agent = _agent_result(
                    '{"status":"success","pI":6.55,'
                    '"molecular_weight":1235.0}'
                )
                trace = _trace(question, agent, cfg)
                result = score_experiment_trace(
                    question,
                    agent,
                    trace,
                )
        except Exception as exc:
            self.fail(f"数值Gold匹配测试失败: {exc}")

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["labels"]["final_answer_correct"])
        self.assertFalse(result["labels"]["result_hallucination"])
        self.assertTrue(result["labels"]["original_plan_executable"])
        self.assertIsNone(result["labels"]["repair_correct"])

    def test_structured_numeric_mismatch_is_false(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                cfg = _config(Path(temp_dir))
                question = _question()
                agent = _agent_result(
                    '{"status":"success","pI":8.5,'
                    '"molecular_weight":9999.0}'
                )
                result = score_experiment_trace(
                    question,
                    agent,
                    _trace(question, agent, cfg),
                )
        except Exception as exc:
            self.fail(f"数值Gold错误测试失败: {exc}")

        self.assertFalse(result["labels"]["final_answer_correct"])
        self.assertEqual(
            result["decisions"]["final_answer_correct"]["reason_code"],
            "STRUCTURED_GOLD_MISMATCH",
        )

    def test_missing_expected_field_remains_null_for_manual_review(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                cfg = _config(Path(temp_dir))
                question = _question()
                agent = _agent_result(
                    '{"status":"success","pI":6.5}'
                )
                result = score_experiment_trace(
                    question,
                    agent,
                    _trace(question, agent, cfg),
                )
        except Exception as exc:
            self.fail(f"缺失字段人工复核测试失败: {exc}")

        self.assertIsNone(result["labels"]["final_answer_correct"])
        self.assertIn(
            "final_answer_correct",
            result["review_required_fields"],
        )

    def test_conflicting_raw_and_summary_values_require_review(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                cfg = _config(Path(temp_dir))
                question = _question()
                agent = _agent_result(
                    '{"pI":6.5,"molecular_weight":1234.5}'
                    '\n最终pI结果：{"pI":9.9,"molecular_weight":1234.5}'
                )
                result = score_experiment_trace(
                    question,
                    agent,
                    _trace(question, agent, cfg),
                )
        except Exception as exc:
            self.fail(f"冲突字段人工复核测试失败: {exc}")

        self.assertIsNone(result["labels"]["final_answer_correct"])
        self.assertEqual(
            result["decisions"]["final_answer_correct"][
                "field_decisions"
            ]["pI"]["reason_code"],
            "CONFLICTING_OBSERVED_VALUES",
        )

    def test_failed_execution_keeps_type_iii_for_manual_review(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                cfg = _config(Path(temp_dir))
                question = _question()
                agent = _agent_result(
                    "工具失败，但回答中出现了一个数值6.5。",
                    completed=False,
                    failed_steps=1,
                )
                result = score_experiment_trace(
                    question,
                    agent,
                    _trace(question, agent, cfg),
                )
        except Exception as exc:
            self.fail(f"Type-III人工边界测试失败: {exc}")

        self.assertIsNone(result["labels"]["result_hallucination"])
        self.assertIn(
            "result_hallucination",
            result["review_required_fields"],
        )

    def test_repair_and_rejection_labels_are_trace_determined(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                cfg = _config(Path(temp_dir))
                question = _question()
                agent = _agent_result(
                    '{"pI":6.5,"molecular_weight":1234.5}'
                )
                repaired_trace = _trace(question, agent, cfg)
                repaired_trace["observation"]["plan"][
                    "repair_applied"
                ] = True
                repaired_trace["observation"]["plan"][
                    "repair_attempted"
                ] = True
                repair_result = score_experiment_trace(
                    question,
                    agent,
                    repaired_trace,
                )

                reject_question = _question(
                    expected_behavior="reject",
                    expected_status="error",
                )
                reject_question["gold"]["preferred_tool_chain"] = []
                reject_question["gold"]["expected_output"] = {}
                reject_agent = _agent_result(
                    "输入无法安全执行，已拒绝。",
                    completed=False,
                    failed_steps=0,
                    attempted_steps=0,
                    plan_chain=[],
                )
                reject_result = score_experiment_trace(
                    reject_question,
                    reject_agent,
                    _trace(reject_question, reject_agent, cfg),
                )
        except Exception as exc:
            self.fail(f"Repair/Reject标签测试失败: {exc}")

        self.assertTrue(repair_result["labels"]["repair_correct"])
        self.assertTrue(reject_result["labels"]["correct_rejection"])
        self.assertFalse(
            reject_result["labels"]["original_plan_executable"]
        )

    def test_semantic_claims_are_not_scored_by_substring_matching(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                cfg = _config(Path(temp_dir))
                question = _question()
                question["gold"]["expected_output"] = {}
                question["gold"]["required_claims"] = ["存在信号肽"]
                question["gold"]["forbidden_claims"] = ["不存在信号肽"]
                agent = _agent_result("结论：并不存在信号肽。")
                result = score_experiment_trace(
                    question,
                    agent,
                    _trace(question, agent, cfg),
                    config=cfg,
                )
        except Exception as exc:
            self.fail(f"语义主张保守评分测试失败: {exc}")

        self.assertIsNone(result["labels"]["final_answer_correct"])
        self.assertIn(
            "final_answer_correct",
            result["review_required_fields"],
        )

    def test_unexecuted_nonempty_plan_requires_executability_review(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                cfg = _config(Path(temp_dir))
                question = _question()
                agent = _agent_result(
                    "计划未执行。",
                    completed=False,
                    attempted_steps=0,
                )
                result = score_experiment_trace(
                    question,
                    agent,
                    _trace(question, agent, cfg),
                    config=cfg,
                )
        except Exception as exc:
            self.fail(f"原计划可执行性人工复核测试失败: {exc}")

        self.assertIsNone(
            result["labels"]["original_plan_executable"]
        )
        self.assertIn(
            "original_plan_executable",
            result["review_required_fields"],
        )


class ReviewWorkflowTests(unittest.TestCase):
    """验证私有盲评资料与公开Trace严格分离。"""

    def test_review_queue_hides_condition_and_audit_hides_raw_answer(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                cfg = _config(temp_path)
                question = _question()
                raw_answer = "工具失败，需要人工判断是否存在结果幻觉。"
                agent = _agent_result(
                    raw_answer,
                    completed=False,
                    failed_steps=1,
                )
                trace = _trace(question, agent, cfg)
                scoring = score_experiment_trace(
                    question,
                    agent,
                    trace,
                )
                applied = apply_scoring_to_trace(
                    trace,
                    scoring,
                    config=cfg,
                )
                result = persist_scoring_bundle(
                    question,
                    agent,
                    applied["trace"],
                    scoring,
                    experiment_id="blind-test",
                    blinding_key="unit-secret-key",
                    audit_path=temp_path / "audit.jsonl",
                    review_queue_path=temp_path / "queue.jsonl",
                    review_map_path=temp_path / "map.jsonl",
                    review_template_path=temp_path / "template.jsonl",
                )
                queue_text = (temp_path / "queue.jsonl").read_text(
                    encoding="utf-8"
                )
                queue_record = json.loads(queue_text)
                audit_text = (temp_path / "audit.jsonl").read_text(
                    encoding="utf-8"
                )
        except Exception as exc:
            self.fail(f"盲评资料隔离测试失败: {exc}")

        self.assertEqual(result["status"], "success")
        self.assertIn(raw_answer, queue_text)
        self.assertNotIn('"condition"', queue_text)
        self.assertNotIn("bte_801", queue_text)
        self.assertEqual(
            queue_record["plan_evidence"]["planned_chain"],
            ["compute_pI_mw"],
        )
        self.assertEqual(
            queue_record["plan_evidence"]["initial_artifact_types"],
            ["amino acid sequence"],
        )
        self.assertEqual(
            queue_record["plan_evidence"]["content_evidence"][
                "sequence_length"
            ],
            19,
        )
        self.assertNotIn(
            question["input"]["sequence"],
            json.dumps(
                queue_record["plan_evidence"],
                ensure_ascii=False,
            ),
        )
        self.assertNotIn(raw_answer, audit_text)
        self.assertIn('"condition": "B0"', audit_text)

    def test_manual_labels_create_new_trace_without_overwriting_source(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                cfg = _config(temp_path)
                question = _question()
                agent = _agent_result(
                    "失败后仍声称pI为6.5。",
                    completed=False,
                    failed_steps=1,
                )
                original_trace = _trace(question, agent, cfg)
                scoring = score_experiment_trace(
                    question,
                    agent,
                    original_trace,
                )
                applied = apply_scoring_to_trace(
                    original_trace,
                    scoring,
                    config=cfg,
                )
                source_path = temp_path / "source.jsonl"
                append_experiment_trace(
                    applied["trace"],
                    output_path=source_path,
                    config=cfg,
                )
                bundle = persist_scoring_bundle(
                    question,
                    agent,
                    applied["trace"],
                    scoring,
                    experiment_id="manual-test",
                    blinding_key="unit-secret-key",
                    audit_path=temp_path / "audit.jsonl",
                    review_queue_path=temp_path / "queue.jsonl",
                    review_map_path=temp_path / "map.jsonl",
                    review_template_path=temp_path / "template.jsonl",
                )
                labels_path = temp_path / "labels.jsonl"
                labels_path.write_text(
                    json.dumps(
                        {
                            "review_id": bundle["review_id"],
                            "reviewer_id": "adjudicator-D",
                            "reviewed_at": "2026-07-30",
                            "labels": {
                                "final_answer_correct": False,
                                "result_hallucination": True,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                output_path = temp_path / "reviewed.jsonl"
                result = apply_manual_review_labels(
                    source_path,
                    temp_path / "map.jsonl",
                    labels_path,
                    output_path,
                    config=cfg,
                )
                original_loaded = load_experiment_traces(
                    source_path,
                    config=cfg,
                )
                reviewed_loaded = load_experiment_traces(
                    output_path,
                    config=cfg,
                )
        except Exception as exc:
            self.fail(f"人工标签回填测试失败: {exc}")

        self.assertEqual(result["status"], "success")
        self.assertIsNone(
            original_loaded["records"][0]["evaluation"][
                "result_hallucination"
            ]
        )
        self.assertTrue(
            reviewed_loaded["records"][0]["evaluation"][
                "result_hallucination"
            ]
        )
        self.assertFalse(
            reviewed_loaded["records"][0]["evaluation"][
                "final_answer_correct"
            ]
        )

    def test_manual_review_cannot_overwrite_original_trace(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                result = apply_manual_review_labels(
                    temp_path / "trace.jsonl",
                    temp_path / "map.jsonl",
                    temp_path / "labels.jsonl",
                    temp_path / "trace.jsonl",
                    config=_config(temp_path),
                )
        except Exception as exc:
            self.fail(f"人工标签覆盖保护测试失败: {exc}")

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "REVIEW_OUTPUT_OVERWRITE_FORBIDDEN",
        )

    def test_frozen_queue_can_be_enriched_without_revealing_condition(self):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                cfg = _config(root)
                question = _question()
                agent = _agent_result(
                    "工具失败，需要人工判断。",
                    completed=False,
                    failed_steps=1,
                )
                trace = _trace(question, agent, cfg)
                scoring = score_experiment_trace(question, agent, trace)
                applied = apply_scoring_to_trace(trace, scoring, config=cfg)
                trace_path = root / "trace.jsonl"
                append_experiment_trace(
                    applied["trace"],
                    output_path=trace_path,
                    config=cfg,
                )
                persist_scoring_bundle(
                    question,
                    agent,
                    applied["trace"],
                    scoring,
                    experiment_id="enrich-test",
                    blinding_key="unit-secret-key",
                    audit_path=root / "audit.jsonl",
                    review_queue_path=root / "queue.jsonl",
                    review_map_path=root / "map.jsonl",
                    review_template_path=root / "template.jsonl",
                )
                old_record = json.loads(
                    (root / "queue.jsonl").read_text(encoding="utf-8")
                )
                old_record.pop("plan_evidence", None)
                (root / "legacy-queue.jsonl").write_text(
                    json.dumps(old_record, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                result = enrich_review_queue(
                    root / "legacy-queue.jsonl",
                    root / "map.jsonl",
                    trace_path,
                    root / "enriched.jsonl",
                )
                enriched_text = (root / "enriched.jsonl").read_text(
                    encoding="utf-8"
                )
                enriched = json.loads(enriched_text)
        except Exception as exc:
            self.fail(f"冻结盲审队列证据补充测试失败: {exc}")

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            enriched["plan_evidence"]["planned_chain"],
            ["compute_pI_mw"],
        )
        self.assertNotIn('"condition"', enriched_text)
        self.assertNotIn("bte_801", enriched_text)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    except Exception as exc:
        print(f"评分与盲评测试启动失败: {exc}")
        raise
