# -*- coding: utf-8 -*-
"""
BioAgent v0.3e确定性自动评分、评分审计和条件盲化人工复核。

修改记录：
- 2026-08-09：人工盲审队列增加原始计划、初始对象类型和非敏感内容证据，
  使original_plan_executable可在不知道B0/B1/B2条件时独立裁决。
- 2026-07-30：新增数值/布尔Gold容差评分、类型与执行标签推导、manual-required
  三态边界、HMAC盲化复核队列、评分审计和人工标签回填。

研究边界：
1. 自动评分只覆盖可以由结构化Gold和运行Trace确定的事实；
2. 缺少证据、字段冲突或需要语义理解时返回null，不用启发式强行猜测；
3. Type-III在工具成功时可确定为false，工具失败时进入人工复核；
4. 公开Trace不保存问题和完整回答，原文只进入.gitignore下的私有盲评队列；
5. 人工标签回填只接受布尔值，并保留原始Trace不覆盖。
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from app.core.config import Config
from app.core.logger import get_logger
from app.tools.tool_runner import (
    extract_pdb_options,
    extract_sequence_candidate,
)
from scripts.evaluation_metrics import tool_chain_matches
from scripts.evaluation_trace import (
    load_experiment_traces,
    validate_experiment_trace,
)
from scripts.type_checker import (
    AMINO_ACID_SEQUENCE,
    PDB_CHAIN_IDENTIFIER,
    PDB_STRUCTURE_FILE,
    SINGLE_PROTEIN_FASTA_FILE,
)


logger = get_logger("EvaluationScoring", "evaluation_scoring.log")

SCORING_POLICY_VERSION = "deterministic-v1.1"
LABEL_FIELDS = (
    "final_answer_correct",
    "repair_correct",
    "result_hallucination",
    "original_plan_executable",
    "correct_rejection",
)
_STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def build_blind_plan_evidence(
    trace: Any,
    *,
    question_item: Any = None,
    question_text: Any = None,
    agent_result: Any = None,
) -> dict[str, Any]:
    """构造不暴露题号、实验条件和原始序列的初始计划复核证据。"""
    try:
        if not isinstance(trace, dict):
            return {
                "planned_chain": [],
                "initial_artifact_types": [],
                "content_evidence": {},
            }
        plan = (trace.get("observation") or {}).get("plan") or {}
        planned_chain = [
            str(value)
            for value in plan.get("planned_chain") or []
            if isinstance(value, str) and value.strip()
        ]
        input_type = str(trace.get("input_type") or "")
        item = question_item if isinstance(question_item, dict) else {}
        input_data = (
            item.get("input")
            if isinstance(item.get("input"), dict)
            else {}
        )
        question = str(
            question_text
            if isinstance(question_text, str)
            else item.get("question_zh") or ""
        )

        initial_types: list[str] = []
        runtime_types = plan.get("initial_types")
        if isinstance(runtime_types, list):
            initial_types = [
                str(value)
                for value in runtime_types
                if isinstance(value, str) and value.strip()
            ]
        if not initial_types:
            type_mapping = {
                "amino_acid_sequence": AMINO_ACID_SEQUENCE,
                "protein_fasta_file": SINGLE_PROTEIN_FASTA_FILE,
                "pdb_file": PDB_STRUCTURE_FILE,
            }
            mapped = type_mapping.get(input_type)
            if mapped:
                initial_types.append(mapped)
            chain_id = input_data.get("chain_id")
            if not isinstance(chain_id, str) or not chain_id.strip():
                chain_id = extract_pdb_options(question).get("chain_id")
            if (
                input_type == "pdb_file"
                and isinstance(chain_id, str)
                and chain_id.strip()
            ):
                initial_types.append(PDB_CHAIN_IDENTIFIER)

        content_evidence: dict[str, Any] = {"input_type": input_type}
        sequence = input_data.get("sequence")
        if (
            (not isinstance(sequence, str) or not sequence.strip())
            and input_type == "amino_acid_sequence"
        ):
            sequence = extract_sequence_candidate(question)
        if isinstance(sequence, str) and sequence.strip():
            normalized = sequence.strip().upper()
            invalid = sorted(set(normalized) - _STANDARD_AMINO_ACIDS)
            content_evidence.update(
                {
                    "sequence_length": len(normalized),
                    "standard_amino_acid_alphabet": not invalid,
                    "invalid_characters": invalid,
                }
            )
        fixture = input_data.get("fixture")
        if isinstance(fixture, str) and fixture.strip():
            content_evidence["file_name"] = Path(fixture).name
        chain_id = input_data.get("chain_id")
        if isinstance(chain_id, str) and chain_id.strip():
            content_evidence["chain_id_provided"] = True
            content_evidence["chain_id"] = chain_id.strip()
        elif input_type == "pdb_file":
            extracted_chain = extract_pdb_options(question).get("chain_id")
            content_evidence["chain_id_provided"] = bool(extracted_chain)
            if extracted_chain:
                content_evidence["chain_id"] = extracted_chain

        typed = (
            ((agent_result or {}).get("reasoning") or {}).get(
                "typed_planning"
            )
            if isinstance(agent_result, dict)
            else None
        )
        runtime_summary = (
            typed.get("initial_artifacts")
            if isinstance(typed, dict)
            and isinstance(typed.get("initial_artifacts"), list)
            else []
        )
        return {
            "planned_chain": planned_chain,
            "initial_artifact_types": list(dict.fromkeys(initial_types)),
            "content_evidence": content_evidence,
            "runtime_initial_artifact_summary": copy.deepcopy(
                runtime_summary
            ),
        }
    except Exception as exc:
        logger.error("构造盲审计划证据失败: %s", exc, exc_info=True)
        return {
            "planned_chain": [],
            "initial_artifact_types": [],
            "content_evidence": {},
            "evidence_error": type(exc).__name__,
        }


def _error_result(
    message: str,
    code: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造评分模块统一错误结构。"""
    try:
        issue = {"code": code, "message": message}
        if isinstance(details, dict):
            issue.update(copy.deepcopy(details))
        return {
            "status": "error",
            "message": message,
            "issues": [issue],
        }
    except Exception as exc:
        logger.error("构造评分错误结构失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": str(exc),
            "issues": [],
        }


def _normalize_key(value: Any) -> str:
    """把字段名规范为可比较的ASCII字母数字键。"""
    try:
        return re.sub(r"[^a-z0-9]", "", str(value or "").lower())
    except Exception as exc:
        logger.error("规范化评分字段名失败: %s", exc, exc_info=True)
        return ""


def _answer_text(agent_result: Any) -> str:
    """读取Agent最终回答，兼容answer和response字段。"""
    try:
        if not isinstance(agent_result, dict):
            return ""
        value = agent_result.get("answer")
        if not isinstance(value, str):
            value = agent_result.get("response")
        return value if isinstance(value, str) else ""
    except Exception as exc:
        logger.error("读取Agent回答失败: %s", exc, exc_info=True)
        return ""


def _extract_json_objects(text: Any) -> list[dict[str, Any]]:
    """从拼接的工具输出和自然语言中提取所有完整JSON对象。"""
    try:
        content = text if isinstance(text, str) else ""
        decoder = json.JSONDecoder()
        objects: list[dict[str, Any]] = []
        cursor = 0
        while cursor < len(content):
            start = content.find("{", cursor)
            if start < 0:
                break
            try:
                value, consumed = decoder.raw_decode(content[start:])
            except json.JSONDecodeError:
                cursor = start + 1
                continue
            if isinstance(value, dict):
                objects.append(value)
            cursor = start + max(consumed, 1)
        return objects
    except Exception as exc:
        logger.error("提取回答JSON失败: %s", exc, exc_info=True)
        return []


def _collect_observed_fields(value: Any) -> dict[str, list[Any]]:
    """递归收集JSON叶节点；同名字段保留全部候选以检测冲突。"""
    observed: dict[str, list[Any]] = {}
    try:
        def visit(current: Any, key: Any = "") -> None:
            """递归遍历JSON值。"""
            try:
                if isinstance(current, dict):
                    for child_key, child_value in current.items():
                        visit(child_value, child_key)
                    return
                if isinstance(current, list):
                    normalized = _normalize_key(key)
                    if normalized:
                        observed.setdefault(normalized, []).append(
                            copy.deepcopy(current)
                        )
                    return
                normalized = _normalize_key(key)
                if normalized:
                    observed.setdefault(normalized, []).append(current)
            except Exception as exc:
                logger.error("遍历观察字段失败: %s", exc, exc_info=True)

        visit(value)
        return observed
    except Exception as exc:
        logger.error("收集观察字段失败: %s", exc, exc_info=True)
        return {}


def _merge_observed_objects(
    objects: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """合并多个工具JSON对象的叶节点。"""
    try:
        merged: dict[str, list[Any]] = {}
        for item in objects:
            fields = _collect_observed_fields(item)
            for key, values in fields.items():
                merged.setdefault(key, []).extend(values)
        return merged
    except Exception as exc:
        logger.error("合并观察字段失败: %s", exc, exc_info=True)
        return {}


def _tolerance_for_field(tolerance: Any, field_name: str) -> float:
    """读取单字段绝对容差；未提供时数值默认要求近似精确。"""
    try:
        if isinstance(tolerance, (int, float)) and not isinstance(
            tolerance,
            bool,
        ):
            return max(0.0, float(tolerance))
        if isinstance(tolerance, dict):
            direct = tolerance.get(field_name)
            if isinstance(direct, (int, float)) and not isinstance(
                direct,
                bool,
            ):
                return max(0.0, float(direct))
            normalized_target = _normalize_key(field_name)
            for key, value in tolerance.items():
                if (
                    _normalize_key(key) == normalized_target
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    return max(0.0, float(value))
        return 1e-9
    except Exception as exc:
        logger.error("读取数值容差失败: %s", exc, exc_info=True)
        return 1e-9


def _compare_scalar_candidates(
    expected: Any,
    candidates: list[Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """比较一个Gold叶节点；冲突候选返回manual_required。"""
    try:
        if not candidates:
            return {
                "status": "manual_required",
                "reason_code": "EXPECTED_FIELD_NOT_FOUND",
            }
        if isinstance(expected, bool):
            comparable = [
                value for value in candidates if isinstance(value, bool)
            ]
            if not comparable:
                return {
                    "status": "manual_required",
                    "reason_code": "BOOLEAN_FIELD_NOT_PARSEABLE",
                }
            matches = [value is expected for value in comparable]
        elif isinstance(expected, (int, float)) and not isinstance(
            expected,
            bool,
        ):
            comparable = [
                float(value)
                for value in candidates
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ]
            if not comparable:
                return {
                    "status": "manual_required",
                    "reason_code": "NUMERIC_FIELD_NOT_PARSEABLE",
                }
            matches = [
                math.isclose(
                    value,
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=tolerance,
                )
                for value in comparable
            ]
        else:
            return {
                "status": "manual_required",
                "reason_code": "NON_SCALAR_GOLD_REQUIRES_REVIEW",
            }

        if all(matches):
            return {
                "status": "automatic",
                "value": True,
                "reason_code": "ALL_VALUES_WITHIN_GOLD",
            }
        if not any(matches):
            return {
                "status": "automatic",
                "value": False,
                "reason_code": "ALL_VALUES_OUTSIDE_GOLD",
            }
        return {
            "status": "manual_required",
            "reason_code": "CONFLICTING_OBSERVED_VALUES",
        }
    except Exception as exc:
        logger.error("比较Gold字段失败: %s", exc, exc_info=True)
        return {
            "status": "manual_required",
            "reason_code": "FIELD_COMPARISON_EXCEPTION",
        }


def _score_final_answer(
    question_item: dict[str, Any],
    agent_result: dict[str, Any],
) -> dict[str, Any]:
    """保守评分最终回答；无法完整验证时保持null。"""
    try:
        gold = question_item.get("gold") or {}
        answer = _answer_text(agent_result)
        if not answer:
            return {
                "value": False,
                "source": "automatic",
                "reason_code": "EMPTY_FINAL_ANSWER",
            }

        expected_output = gold.get("expected_output")
        field_decisions: dict[str, dict[str, Any]] = {}
        if isinstance(expected_output, dict) and expected_output:
            observed = _merge_observed_objects(
                _extract_json_objects(answer)
            )
            tolerance = gold.get("tolerance") or {}
            for field_name, expected in expected_output.items():
                normalized = _normalize_key(field_name)
                field_decisions[str(field_name)] = (
                    _compare_scalar_candidates(
                        expected,
                        observed.get(normalized, []),
                        tolerance=_tolerance_for_field(
                            tolerance,
                            str(field_name),
                        ),
                    )
                )
            automatic_values = [
                decision.get("value")
                for decision in field_decisions.values()
                if decision.get("status") == "automatic"
            ]
            manual_fields = [
                name
                for name, decision in field_decisions.items()
                if decision.get("status") != "automatic"
            ]
            if any(value is False for value in automatic_values):
                return {
                    "value": False,
                    "source": "automatic",
                    "reason_code": "STRUCTURED_GOLD_MISMATCH",
                    "field_decisions": field_decisions,
                }
            if not manual_fields and len(automatic_values) == len(
                expected_output
            ):
                return {
                    "value": True,
                    "source": "automatic",
                    "reason_code": "STRUCTURED_GOLD_MATCH",
                    "field_decisions": field_decisions,
                }

        return {
            "value": None,
            "source": "manual_required",
            "reason_code": "SEMANTIC_ANSWER_REVIEW_REQUIRED",
            "field_decisions": field_decisions,
        }
    except Exception as exc:
        logger.error("评分最终回答失败: %s", exc, exc_info=True)
        return {
            "value": None,
            "source": "manual_required",
            "reason_code": "FINAL_ANSWER_SCORING_EXCEPTION",
        }


def score_experiment_trace(
    question_item: Any,
    agent_result: Any,
    trace: Any,
    *,
    config: Any = None,
) -> dict[str, Any]:
    """根据Gold、Agent原响应和脱敏Trace生成五类三态标签。"""
    try:
        if not all(
            isinstance(value, dict)
            for value in (question_item, agent_result, trace)
        ):
            return _error_result(
                "评分输入必须是question_item、agent_result和trace字典",
                "INVALID_SCORING_INPUT",
            )
        validation = validate_experiment_trace(trace, config=config)
        if validation.get("status") != "success":
            return validation

        gold = question_item.get("gold") or {}
        plan = (trace.get("observation") or {}).get("plan") or {}
        execution = (trace.get("observation") or {}).get(
            "execution"
        ) or {}
        planned_chain = plan.get("planned_chain") or []
        executed_chain = plan.get("executed_chain") or []

        decisions: dict[str, dict[str, Any]] = {}
        decisions["final_answer_correct"] = _score_final_answer(
            question_item,
            agent_result,
        )

        if plan.get("repair_applied") is True:
            repair_value = (
                bool(execution.get("completed"))
                and tool_chain_matches(executed_chain, trace.get("gold") or {})
                and plan.get("type_valid_after") is True
            )
            decisions["repair_correct"] = {
                "value": repair_value,
                "source": "automatic",
                "reason_code": (
                    "REPAIR_MATCHED_GOLD_AND_EXECUTED"
                    if repair_value
                    else "REPAIR_DID_NOT_MATCH_GOLD_OR_EXECUTE"
                ),
            }
        else:
            decisions["repair_correct"] = {
                "value": None,
                "source": "not_applicable",
                "reason_code": "NO_REPAIR_APPLIED",
            }

        if (
            bool(execution.get("completed"))
            and int(execution.get("failed_steps") or 0) == 0
        ):
            decisions["result_hallucination"] = {
                "value": False,
                "source": "automatic",
                "reason_code": "NO_FAILED_TOOL_TO_MISREPORT",
            }
        else:
            decisions["result_hallucination"] = {
                "value": None,
                "source": "manual_required",
                "reason_code": "FAILED_EXECUTION_NEEDS_SEMANTIC_REVIEW",
            }

        original_chain_executed = (
            bool(planned_chain)
            and bool(execution.get("completed"))
            and plan.get("repair_applied") is not True
            and planned_chain == executed_chain
        )
        if original_chain_executed:
            decisions["original_plan_executable"] = {
                "value": True,
                "source": "automatic",
                "reason_code": "ORIGINAL_PLAN_EXECUTED_SUCCESSFULLY",
            }
        elif not planned_chain:
            decisions["original_plan_executable"] = {
                "value": False,
                "source": "automatic",
                "reason_code": "ORIGINAL_PLAN_EMPTY",
            }
        else:
            decisions["original_plan_executable"] = {
                "value": None,
                "source": "manual_required",
                "reason_code": (
                    "UNEXECUTED_OR_REPAIRED_PLAN_NEEDS_REVIEW"
                ),
            }

        if gold.get("expected_behavior") == "reject":
            rejection_value = (
                not bool(execution.get("completed"))
                and int(execution.get("attempted_steps") or 0) == 0
            )
            decisions["correct_rejection"] = {
                "value": rejection_value,
                "source": "automatic",
                "reason_code": (
                    "EXPECTED_REJECTION_WITHOUT_EXECUTION"
                    if rejection_value
                    else "EXPECTED_REJECTION_WAS_EXECUTED_OR_COMPLETED"
                ),
            }
        else:
            decisions["correct_rejection"] = {
                "value": None,
                "source": "not_applicable",
                "reason_code": "QUESTION_DOES_NOT_EXPECT_REJECTION",
            }

        labels = {
            field: (
                decisions[field].get("value")
                if isinstance(decisions[field].get("value"), bool)
                else None
            )
            for field in LABEL_FIELDS
        }
        review_required_fields = [
            field
            for field in LABEL_FIELDS
            if decisions[field].get("source") == "manual_required"
        ]
        automatic_count = sum(
            decisions[field].get("source") == "automatic"
            for field in LABEL_FIELDS
        )
        return {
            "status": "success",
            "policy_version": SCORING_POLICY_VERSION,
            "labels": labels,
            "decisions": decisions,
            "review_required_fields": review_required_fields,
            "automatic_label_count": automatic_count,
            "manual_required_count": len(review_required_fields),
        }
    except Exception as exc:
        logger.error("评分实验Trace失败: %s", exc, exc_info=True)
        return _error_result(
            f"评分实验Trace失败: {exc}",
            "TRACE_SCORING_EXCEPTION",
        )


def apply_scoring_to_trace(
    trace: Any,
    scoring_result: Any,
    *,
    config: Any = None,
) -> dict[str, Any]:
    """把评分标签写入Trace副本并重新执行Schema校验。"""
    try:
        if not isinstance(trace, dict) or not isinstance(
            scoring_result,
            dict,
        ):
            return _error_result(
                "Trace和scoring_result必须是字典",
                "INVALID_SCORING_APPLICATION",
            )
        if scoring_result.get("status") != "success":
            return scoring_result
        updated = copy.deepcopy(trace)
        evaluation = updated.get("evaluation")
        if not isinstance(evaluation, dict):
            return _error_result(
                "Trace缺少evaluation对象",
                "MISSING_TRACE_EVALUATION",
            )
        for field in LABEL_FIELDS:
            value = (scoring_result.get("labels") or {}).get(field)
            evaluation[field] = value if isinstance(value, bool) else None
        validation = validate_experiment_trace(updated, config=config)
        if validation.get("status") != "success":
            return validation
        return {"status": "success", "trace": updated}
    except Exception as exc:
        logger.error("应用评分标签失败: %s", exc, exc_info=True)
        return _error_result(
            f"应用评分标签失败: {exc}",
            "SCORING_APPLICATION_EXCEPTION",
        )


def _execution_key_text(trace: dict[str, Any]) -> str:
    """生成评分审计内部唯一键。"""
    try:
        context = trace.get("context") or {}
        return "|".join(
            (
                str(trace.get("question_id") or ""),
                str(context.get("model") or ""),
                str(trace.get("condition") or ""),
                str(trace.get("repeat_index") or 0),
            )
        )
    except Exception as exc:
        logger.error("生成评分执行键失败: %s", exc, exc_info=True)
        return ""


def _review_id(execution_key: str, blinding_key: str) -> str:
    """使用HMAC生成不暴露实验条件的复核编号。"""
    try:
        digest = hmac.new(
            blinding_key.encode("utf-8"),
            execution_key.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"review_{digest[:20]}"
    except Exception as exc:
        logger.error("生成盲评编号失败: %s", exc, exc_info=True)
        return ""


def _read_jsonl_records(path: Path, key_field: str) -> dict[str, Any]:
    """读取可原子重写的JSONL索引；文件不存在视为空。"""
    try:
        if not path.exists():
            return {"status": "success", "records": {}}
        records: dict[str, dict[str, Any]] = {}
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except Exception as exc:
                return _error_result(
                    f"{path.name}第{line_number}行不是合法JSON: {exc}",
                    "INVALID_SCORING_JSONL",
                )
            key = str(record.get(key_field) or "").strip()
            if not key:
                return _error_result(
                    f"{path.name}第{line_number}行缺少{key_field}",
                    "MISSING_SCORING_RECORD_KEY",
                )
            if key in records:
                return _error_result(
                    f"{path.name}包含重复{key_field}: {key}",
                    "DUPLICATE_SCORING_RECORD_KEY",
                )
            records[key] = record
        return {"status": "success", "records": records}
    except Exception as exc:
        logger.error("读取评分JSONL失败: %s", exc, exc_info=True)
        return _error_result(
            f"读取评分JSONL失败: {exc}",
            "SCORING_JSONL_READ_EXCEPTION",
        )


def _atomic_write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """原子重写JSONL，防止中断留下半条盲评记录。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        content = "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        )
        temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, path)
        return {"status": "success", "record_count": len(records)}
    except Exception as exc:
        logger.error("原子写入评分JSONL失败: %s", exc, exc_info=True)
        return _error_result(
            f"原子写入评分JSONL失败: {exc}",
            "SCORING_JSONL_WRITE_EXCEPTION",
        )


def persist_scoring_bundle(
    question_item: Any,
    agent_result: Any,
    trace: Any,
    scoring_result: Any,
    *,
    experiment_id: str,
    blinding_key: str,
    audit_path: Any,
    review_queue_path: Any,
    review_map_path: Any,
    review_template_path: Any,
) -> dict[str, Any]:
    """幂等保存脱敏审计、私有盲评队列、映射和人工标签模板。"""
    try:
        if not all(
            isinstance(value, dict)
            for value in (
                question_item,
                agent_result,
                trace,
                scoring_result,
            )
        ):
            return _error_result(
                "评分持久化输入必须是字典",
                "INVALID_SCORING_BUNDLE",
            )
        if scoring_result.get("status") != "success":
            return scoring_result
        if not isinstance(blinding_key, str) or len(blinding_key) < 8:
            return _error_result(
                "人工复核盲化密钥至少需要8个字符",
                "INVALID_REVIEW_BLINDING_KEY",
            )

        paths = {
            "audit": Path(str(audit_path)),
            "queue": Path(str(review_queue_path)),
            "mapping": Path(str(review_map_path)),
            "template": Path(str(review_template_path)),
        }
        loaded: dict[str, dict[str, dict[str, Any]]] = {}
        for name, path in paths.items():
            key_field = (
                "execution_key" if name == "audit" else "review_id"
            )
            result = _read_jsonl_records(path, key_field)
            if result.get("status") != "success":
                return result
            loaded[name] = result["records"]

        execution_key = _execution_key_text(trace)
        review_id = _review_id(execution_key, blinding_key)
        if not execution_key or not review_id:
            return _error_result(
                "无法生成评分或盲评唯一键",
                "SCORING_KEY_FAILURE",
            )
        answer = _answer_text(agent_result)
        review_fields = scoring_result.get(
            "review_required_fields"
        ) or []

        loaded["audit"][execution_key] = {
            "schema_version": "1.0",
            "policy_version": SCORING_POLICY_VERSION,
            "experiment_id": experiment_id,
            "execution_key": execution_key,
            "review_id": review_id,
            "question_id": trace.get("question_id"),
            "model": (trace.get("context") or {}).get("model"),
            "condition": trace.get("condition"),
            "repeat_index": trace.get("repeat_index"),
            "answer_sha256": hashlib.sha256(
                answer.encode("utf-8")
            ).hexdigest(),
            "labels": copy.deepcopy(scoring_result.get("labels") or {}),
            "decisions": copy.deepcopy(
                scoring_result.get("decisions") or {}
            ),
            "review_required_fields": list(review_fields),
        }

        if review_fields:
            gold = question_item.get("gold") or {}
            execution = (trace.get("observation") or {}).get(
                "execution"
            ) or {}
            loaded["queue"][review_id] = {
                "schema_version": "1.0",
                "review_id": review_id,
                "pending_fields": list(review_fields),
                "question": str(
                    question_item.get("question_zh") or ""
                ),
                "answer": answer,
                "rubric": {
                    "expected_status": gold.get("expected_status"),
                    "expected_behavior": gold.get("expected_behavior"),
                    "expected_output": copy.deepcopy(
                        gold.get("expected_output") or {}
                    ),
                    "tolerance": copy.deepcopy(
                        gold.get("tolerance") or {}
                    ),
                    "required_claims": list(
                        gold.get("required_claims") or []
                    ),
                    "forbidden_claims": list(
                        gold.get("forbidden_claims") or []
                    ),
                },
                "plan_evidence": build_blind_plan_evidence(
                    trace,
                    question_item=question_item,
                    agent_result=agent_result,
                ),
                "execution_evidence": {
                    "completed": bool(execution.get("completed")),
                    "attempted_steps": int(
                        execution.get("attempted_steps") or 0
                    ),
                    "failed_steps": int(
                        execution.get("failed_steps") or 0
                    ),
                    "error_type": execution.get("error_type"),
                },
            }
            loaded["mapping"][review_id] = {
                "schema_version": "1.0",
                "review_id": review_id,
                "execution_key": execution_key,
                "question_id": trace.get("question_id"),
                "model": (trace.get("context") or {}).get("model"),
                "condition": trace.get("condition"),
                "repeat_index": trace.get("repeat_index"),
                "pending_fields": list(review_fields),
            }
            loaded["template"][review_id] = {
                "review_id": review_id,
                "reviewer_id": "",
                "reviewed_at": "",
                "labels": {
                    field: None for field in review_fields
                },
            }
        else:
            for name in ("queue", "mapping", "template"):
                loaded[name].pop(review_id, None)

        write_order = (
            ("audit", "execution_key"),
            ("queue", "review_id"),
            ("mapping", "review_id"),
            ("template", "review_id"),
        )
        for name, sort_field in write_order:
            records = sorted(
                loaded[name].values(),
                key=lambda record: str(record.get(sort_field) or ""),
            )
            result = _atomic_write_jsonl(paths[name], records)
            if result.get("status") != "success":
                return result
        return {
            "status": "success",
            "review_id": review_id,
            "review_required": bool(review_fields),
            "review_required_fields": list(review_fields),
        }
    except Exception as exc:
        logger.error("保存评分Bundle失败: %s", exc, exc_info=True)
        return _error_result(
            f"保存评分Bundle失败: {exc}",
            "SCORING_BUNDLE_EXCEPTION",
        )


def apply_manual_review_labels(
    trace_path: Any,
    review_map_path: Any,
    labels_path: Any,
    output_path: Any,
    *,
    config: Any = None,
) -> dict[str, Any]:
    """把已裁决的盲评布尔标签回填到新的Trace文件，绝不覆盖原始Trace。"""
    try:
        cfg = config if config is not None else Config()
        source_path = Path(str(trace_path or "")).resolve()
        destination = Path(str(output_path or "")).resolve()
        if source_path == destination:
            return _error_result(
                "人工标签输出不能覆盖原始Trace",
                "REVIEW_OUTPUT_OVERWRITE_FORBIDDEN",
            )
        if destination.suffix.lower() != ".jsonl":
            return _error_result(
                "人工标签输出必须是.jsonl",
                "INVALID_REVIEW_OUTPUT_PATH",
            )

        traces_result = load_experiment_traces(
            source_path,
            config=cfg,
        )
        if traces_result.get("status") != "success":
            return traces_result
        mapping_result = _read_jsonl_records(
            Path(str(review_map_path)),
            "review_id",
        )
        if mapping_result.get("status") != "success":
            return mapping_result
        labels_result = _read_jsonl_records(
            Path(str(labels_path)),
            "review_id",
        )
        if labels_result.get("status") != "success":
            return labels_result

        trace_index = {
            _execution_key_text(trace): copy.deepcopy(trace)
            for trace in traces_result["records"]
        }
        applied_count = 0
        reviewed_ids: set[str] = set()
        for review_id, label_record in labels_result["records"].items():
            mapping = mapping_result["records"].get(review_id)
            if not isinstance(mapping, dict):
                return _error_result(
                    f"人工标签{review_id}没有盲评映射",
                    "UNKNOWN_REVIEW_ID",
                )
            reviewer_id = str(
                label_record.get("reviewer_id") or ""
            ).strip()
            if not reviewer_id:
                return _error_result(
                    f"人工标签{review_id}缺少reviewer_id",
                    "MISSING_REVIEWER_ID",
                )
            labels = label_record.get("labels")
            if not isinstance(labels, dict):
                return _error_result(
                    f"人工标签{review_id}缺少labels对象",
                    "INVALID_MANUAL_LABELS",
                )
            trace = trace_index.get(mapping.get("execution_key"))
            if not isinstance(trace, dict):
                return _error_result(
                    f"人工标签{review_id}找不到对应Trace",
                    "REVIEW_TRACE_NOT_FOUND",
                )
            allowed_fields = set(mapping.get("pending_fields") or [])
            for field, value in labels.items():
                if field not in allowed_fields:
                    return _error_result(
                        f"{review_id}不允许回填字段{field}",
                        "UNEXPECTED_MANUAL_LABEL_FIELD",
                    )
                if value is None:
                    continue
                if not isinstance(value, bool):
                    return _error_result(
                        f"{review_id}.{field}必须是布尔值或null",
                        "INVALID_MANUAL_LABEL_VALUE",
                    )
                trace["evaluation"][field] = value
                applied_count += 1
            validation = validate_experiment_trace(trace, config=cfg)
            if validation.get("status") != "success":
                return validation
            reviewed_ids.add(review_id)

        ordered_traces = [
            trace_index[_execution_key_text(original)]
            for original in traces_result["records"]
        ]
        write_result = _atomic_write_jsonl(
            destination,
            ordered_traces,
        )
        if write_result.get("status") != "success":
            return write_result
        remaining_null = sum(
            value is None
            for trace in ordered_traces
            for value in (trace.get("evaluation") or {}).values()
        )
        return {
            "status": "success",
            "output_path": str(destination),
            "trace_count": len(ordered_traces),
            "review_record_count": len(reviewed_ids),
            "applied_label_count": applied_count,
            "remaining_null_label_count": remaining_null,
        }
    except Exception as exc:
        logger.error("回填人工复核标签失败: %s", exc, exc_info=True)
        return _error_result(
            f"回填人工复核标签失败: {exc}",
            "MANUAL_REVIEW_APPLICATION_EXCEPTION",
        )
