# -*- coding: utf-8 -*-
"""Validate benchmark tasks and source records offline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from app.core.config import Config
from app.core.logger import get_logger
from scripts.type_checker import (
    AMINO_ACID_SEQUENCE,
    FASTA_RECORD_IDENTIFIER,
    FASTA_RECORD_IDENTIFIER_PAIR,
    MULTI_PROTEIN_FASTA_FILE,
    PDB_CHAIN_IDENTIFIER,
    PDB_STRUCTURE_FILE,
    SINGLE_PROTEIN_FASTA_FILE,
    get_tool_signatures,
    validate_tool_plan,
)


logger = get_logger("BioToolEvalValidator", "biotooleval_validator.log")

_FORMAL_DATASET_STATUSES = {"benchmark_ready", "frozen", "released"}
_VALID_PROFILES = {
    "auto",
    "none",
    "topologyv2pilot12",
    "pilot30",
    "challenge30",
    "dev40",
    "dev70",
    "dev35",
    "test35",
    "final100",
    "final150",
}
_STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
_PLACEHOLDER_PATTERNS = (
    re.compile(r"\.{3,}|…"),
    re.compile(r"\b10[,\s]?000\b", re.IGNORECASE),
    re.compile(r"10000\s*个", re.IGNORECASE),
)
_PROFILE_RULES: dict[str, dict[str, Any]] = {
    "topologyv2pilot12": {
        "item_count": 12,
        "categories": {
            "single_tool": 1,
            "multi_tool": 9,
            "boundary": 2,
            "trap": 0,
        },
        "difficulties": {
            "easy": 1,
            "medium": 4,
            "hard": 7,
        },
        "expected_behaviors": {
            "execute": 1,
            "repair_then_execute": 9,
            "reject": 2,
        },
        "allowed_splits": {"dev"},
        "minimum_input_types": {"protein_fasta_file": 11},
        "minimum_type_sensitive_questions": 11,
        "minimum_tool_coverage": {
            "parse_multi_protein_fasta": 8,
            "select_fasta_record": 5,
            "align_fasta_records": 3,
            "summarize_pairwise_alignment": 3,
        },
    },
    "pilot30": {
        "item_count": 30,
        "categories": {
            "single_tool": 10,
            "multi_tool": 10,
            "boundary": 5,
            "trap": 5,
        },
        "allowed_splits": {"pilot"},
    },
    "challenge30": {
        "item_count": 30,
        "categories": {
            "single_tool": 6,
            "multi_tool": 16,
            "boundary": 4,
            "trap": 4,
        },
        "allowed_splits": {"dev"},
        "minimum_input_types": {
            "protein_fasta_file": 8,
            "pdb_file": 8,
        },
        "minimum_expected_behaviors": {
            "repair_then_execute": 4,
            "reject": 4,
        },
        "minimum_type_sensitive_questions": 20,
    },
    "dev40": {
        "item_count": 40,
        "categories": {
            "single_tool": 10,
            "multi_tool": 18,
            "boundary": 6,
            "trap": 6,
        },
        "difficulties": {
            "easy": 8,
            "medium": 20,
            "hard": 12,
        },
        "expected_behaviors": {
            "execute": 24,
            "repair_then_execute": 8,
            "reject": 8,
        },
        "allowed_splits": {"dev"},
        "minimum_real_inputs": 30,
        "minimum_input_types": {
            "protein_fasta_file": 12,
            "pdb_file": 10,
        },
        "minimum_toxicity_questions": 3,
        "minimum_type_sensitive_questions": 28,
        "require_all_tools": True,
    },
    "dev70": {
        "item_count": 70,
        "categories": {
            "single_tool": 16,
            "multi_tool": 34,
            "boundary": 10,
            "trap": 10,
        },
        "difficulties": {
            "easy": 15,
            "medium": 32,
            "hard": 23,
        },
        "expected_behaviors": {
            "execute": 46,
            "repair_then_execute": 12,
            "reject": 12,
        },
        "allowed_splits": {"dev"},
        "minimum_real_inputs": 30,
        "minimum_input_types": {
            "protein_fasta_file": 21,
            "pdb_file": 19,
        },
        "minimum_toxicity_questions": 3,
        "minimum_type_sensitive_questions": 50,
        "require_all_tools": True,
    },
    "dev35": {
        "item_count": 35,
        "categories": {
            "single_tool": 10,
            "multi_tool": 15,
            "boundary": 5,
            "trap": 5,
        },
        "allowed_splits": {"dev"},
        "minimum_real_inputs": 20,
        "minimum_pdb_inputs": 6,
        "minimum_toxicity_questions": 3,
    },
    "test35": {
        "item_count": 35,
        "categories": {
            "single_tool": 10,
            "multi_tool": 15,
            "boundary": 5,
            "trap": 5,
        },
        "allowed_splits": {"test"},
        "minimum_real_inputs": 20,
        "minimum_pdb_inputs": 6,
        "minimum_toxicity_questions": 3,
        "minimum_type_sensitive_questions": 8,
    },
    "final100": {
        "item_count": 100,
        "categories": {
            "single_tool": 30,
            "multi_tool": 40,
            "boundary": 15,
            "trap": 15,
        },
        "allowed_splits": {"pilot", "dev", "test"},
        "minimum_real_inputs": 40,
        "require_all_tools": True,
    },
    "final150": {
        "item_count": 150,
        "categories": {
            "single_tool": 25,
            "multi_tool": 75,
            "boundary": 25,
            "trap": 25,
        },
        "difficulties": {
            "easy": 25,
            "medium": 75,
            "hard": 50,
        },
        "expected_behaviors": {
            "execute": 95,
            "repair_then_execute": 30,
            "reject": 25,
        },
        "allowed_splits": {"test"},
        "exact_input_types": {
            "amino_acid_sequence": 20,
            "protein_fasta_file": 80,
            "pdb_file": 40,
            "none": 10,
        },
        "exact_source_types": {
            "real": 105,
            "synthetic": 35,
            "none": 10,
        },
        "minimum_tool_coverage": {
            "compute_pI_mw": 8,
            "compute_gravy": 8,
            "compute_secondary_structure_fraction": 8,
            "compute_amino_acid_composition": 8,
            "compute_instability_index": 8,
            "compute_extinction_coefficient": 8,
            "predict_signalpeptide": 8,
            "predict_transmembrane": 8,
            "predict_solubility": 8,
            "check_protein_toxicity": 8,
            "protein_analysis": 8,
            "parse_protein_fasta": 16,
            "parse_multi_protein_fasta": 50,
            "select_fasta_record": 25,
            "align_fasta_records": 18,
            "summarize_pairwise_alignment": 18,
            "select_pdb_chain": 18,
            "analyze_pdb_structure": 8,
            "extract_pdb_sequences": 8,
            "calculate_protein_anm": 8,
        },
        "maximum_tool_coverage": 60,
        "require_all_tools": True,
        "require_contiguous_ids": True,
    },
}


def _runtime_error(message: str, code: str = "VALIDATOR_EXCEPTION") -> dict[str, Any]:
    try:
        return {
            "status": "error",
            "message": str(message or "BioToolEval验收器运行失败"),
            "verdict": "unknown",
            "benchmark_ready": False,
            "issues": [
                {
                    "severity": "error",
                    "code": str(code or "VALIDATOR_EXCEPTION"),
                    "message": str(message or "BioToolEval验收器运行失败"),
                }
            ],
        }
    except Exception as exc:
        logger.error("构造验收器错误结构失败: %s", exc, exc_info=True)
        return {"status": "error", "verdict": "unknown", "benchmark_ready": False}


def _add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    *,
    question_id: Any = None,
    path: Any = None,
    details: dict[str, Any] | None = None,
) -> None:
    try:
        issue: dict[str, Any] = {
            "severity": severity if severity in {"error", "warning", "info"} else "error",
            "code": str(code or "UNKNOWN_ISSUE"),
            "message": str(message or "未说明的问题"),
        }
        normalized_id = str(question_id or "").strip()
        if normalized_id:
            issue["question_id"] = normalized_id
        normalized_path = str(path or "").strip()
        if normalized_path:
            issue["path"] = normalized_path
        if isinstance(details, dict):
            for key, value in details.items():
                if key not in {"question_text", "sequence", "absolute_path"}:
                    issue[key] = copy.deepcopy(value)
        issues.append(issue)
    except Exception as exc:
        logger.error("追加数据集验收问题失败: %s", exc, exc_info=True)


def _load_schema(schema_path: Any) -> dict[str, Any]:
    try:
        path = Path(str(schema_path or "")).expanduser()
        if not path.is_file():
            return _runtime_error("BioToolEval题目Schema不存在", "QUESTION_SCHEMA_NOT_FOUND")
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        return {
            "status": "success",
            "validator": Draft202012Validator(schema),
            "schema_version": schema.get("x-version"),
        }
    except Exception as exc:
        logger.error("加载BioToolEval题目Schema失败: %s", exc, exc_info=True)
        return _runtime_error(f"加载BioToolEval题目Schema失败: {exc}", "QUESTION_SCHEMA_ERROR")


def _normalize_text(value: Any) -> str:
    try:
        text = str(value or "").casefold()
        return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    except Exception as exc:
        logger.error("规范化题面失败: %s", exc, exc_info=True)
        return ""


def _contains_placeholder(value: Any) -> bool:
    try:
        text = str(value or "")
        return any(pattern.search(text) is not None for pattern in _PLACEHOLDER_PATTERNS)
    except Exception as exc:
        logger.error("检查占位文本失败: %s", exc, exc_info=True)
        return False


def _schema_error_path(error: Any) -> str:
    try:
        parts = [str(part) for part in list(error.absolute_path)]
        return ".".join(parts) if parts else "$"
    except Exception as exc:
        logger.error("生成Schema错误路径失败: %s", exc, exc_info=True)
        return "$"


def _schema_error_message(error: Any) -> str:
    try:
        validator_name = str(getattr(error, "validator", "") or "")
        messages = {
            "required": "缺少Schema要求字段",
            "type": "字段类型不符合Schema",
            "enum": "字段值不在Schema允许集合中",
            "pattern": "字段格式不符合Schema",
            "minLength": "字段长度低于Schema下限",
            "maxLength": "字段长度超过Schema上限",
            "minimum": "数值低于Schema下限",
            "maximum": "数值超过Schema上限",
            "additionalProperties": "包含Schema未允许的额外字段",
        }
        return messages.get(validator_name, "字段未通过Schema约束")
    except Exception as exc:
        logger.error("生成脱敏Schema错误消息失败: %s", exc, exc_info=True)
        return "字段未通过Schema约束"


def _initial_types(input_data: Any) -> list[str]:
    try:
        if not isinstance(input_data, dict):
            return []
        input_type = str(input_data.get("input_type") or "").strip()
        if input_type == "amino_acid_sequence":
            return [AMINO_ACID_SEQUENCE]
        if input_type == "protein_fasta_file":
            cardinality = str(input_data.get("fasta_cardinality") or "").strip()
            if cardinality == "multi":
                types = [MULTI_PROTEIN_FASTA_FILE]
                record_ids = input_data.get("record_ids")
                if (
                    isinstance(record_ids, list)
                    and len(record_ids) == 2
                    and all(str(value or "").strip() for value in record_ids)
                ):
                    types.append(FASTA_RECORD_IDENTIFIER_PAIR)
                elif str(input_data.get("record_id") or "").strip():
                    types.append(FASTA_RECORD_IDENTIFIER)
                return types
            return [SINGLE_PROTEIN_FASTA_FILE]
        if input_type == "pdb_file":
            types = [PDB_STRUCTURE_FILE]
            if str(input_data.get("chain_id") or "").strip():
                types.append(PDB_CHAIN_IDENTIFIER)
            return types
        return []
    except Exception as exc:
        logger.error("映射评测输入类型失败: %s", exc, exc_info=True)
        return []


def _iter_numeric_leaf_paths(value: Any, prefix: str = "") -> list[str]:
    try:
        paths: list[str] = []
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{prefix}.{key}" if prefix else str(key)
                paths.extend(_iter_numeric_leaf_paths(child, child_path))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                child_path = f"{prefix}.{index}" if prefix else str(index)
                paths.extend(_iter_numeric_leaf_paths(child, child_path))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            paths.append(prefix)
        return paths
    except Exception as exc:
        logger.error("收集数值Gold字段失败: %s", exc, exc_info=True)
        return []


def _has_semantic_leaf(value: Any) -> bool:
    try:
        if isinstance(value, dict):
            return any(_has_semantic_leaf(child) for child in value.values())
        if isinstance(value, list):
            return True
        return not (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
    except Exception as exc:
        logger.error("识别语义Gold字段失败: %s", exc, exc_info=True)
        return True


def _tolerance_for_path(tolerance: Any, numeric_path: str) -> Any:
    try:
        if not isinstance(tolerance, dict):
            return None
        candidates = [
            numeric_path,
            numeric_path.rsplit(".", 1)[-1],
            numeric_path.split(".", 1)[0],
        ]
        for key in candidates:
            if key in tolerance:
                return tolerance.get(key)
        return None
    except Exception as exc:
        logger.error("读取数值容差失败: %s", exc, exc_info=True)
        return None


def _safe_fixture_path(dataset_path: Path, fixture: Any) -> dict[str, Any]:
    try:
        raw = str(fixture or "").strip()
        if not raw:
            return {"status": "error", "code": "MISSING_FIXTURE", "message": "文件输入缺少fixture"}
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (dataset_path.parent / candidate).resolve()
        root = dataset_path.parent.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            return {
                "status": "error",
                "code": "FIXTURE_OUTSIDE_DATASET_DIR",
                "message": "fixture必须位于数据集目录或其子目录内",
            }
        if not resolved.is_file():
            return {
                "status": "error",
                "code": "FIXTURE_NOT_FOUND",
                "message": "fixture文件不存在",
                "file_name": resolved.name,
            }
        return {
            "status": "success",
            "file_name": resolved.name,
            "suffix": resolved.suffix.lower(),
        }
    except Exception as exc:
        logger.error("检查评测夹具路径失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "code": "FIXTURE_PATH_ERROR",
            "message": f"fixture路径解析失败: {exc}",
        }


def _check_input(
    item: dict[str, Any],
    dataset_path: Path,
    issues: list[dict[str, Any]],
) -> None:
    try:
        question_id = item.get("id")
        input_data = item.get("input")
        gold = item.get("gold")
        if not isinstance(input_data, dict):
            return
        gold = gold if isinstance(gold, dict) else {}
        expected_rejection = (
            gold.get("expected_behavior") == "reject"
            or gold.get("expected_status") == "error"
        )
        input_type = input_data.get("input_type")
        source_type = input_data.get("source_type")

        if input_type == "amino_acid_sequence":
            sequence = str(input_data.get("sequence") or "")
            if _contains_placeholder(sequence):
                _add_issue(
                    issues,
                    "error",
                    "PLACEHOLDER_INPUT",
                    "正式题目不得使用省略号或长度描述代替完整序列",
                    question_id=question_id,
                    path="input.sequence",
                )
            normalized = re.sub(r"\s+", "", sequence).upper()
            invalid = sorted(set(normalized) - _STANDARD_AA)
            if invalid and not expected_rejection:
                _add_issue(
                    issues,
                    "error",
                    "INVALID_SUCCESS_SEQUENCE",
                    "预期成功的题目包含非标准氨基酸字符",
                    question_id=question_id,
                    path="input.sequence",
                    details={"invalid_character_count": len(invalid)},
                )

        if input_type in {"protein_fasta_file", "pdb_file"}:
            fixture_result = _safe_fixture_path(dataset_path, input_data.get("fixture"))
            if fixture_result.get("status") != "success":
                _add_issue(
                    issues,
                    "error",
                    fixture_result.get("code", "FIXTURE_ERROR"),
                    fixture_result.get("message", "文件夹具无效"),
                    question_id=question_id,
                    path="input.fixture",
                    details={"file_name": fixture_result.get("file_name")},
                )
            else:
                suffix = fixture_result.get("suffix")
                expected_suffixes = (
                    {".pdb"} if input_type == "pdb_file" else {".fasta", ".fa", ".faa"}
                )
                if suffix not in expected_suffixes:
                    _add_issue(
                        issues,
                        "error",
                        "FIXTURE_SUFFIX_MISMATCH",
                        "fixture扩展名与input_type不一致",
                        question_id=question_id,
                        path="input.fixture",
                        details={"file_name": fixture_result.get("file_name")},
                    )

        if source_type == "real":
            missing = [
                key
                for key in ("database", "accession", "reference_url")
                if not str(input_data.get(key) or "").strip()
            ]
            if missing:
                _add_issue(
                    issues,
                    "error",
                    "INCOMPLETE_REAL_PROVENANCE",
                    "真实样本必须同时记录数据库、accession/PDB ID和参考链接",
                    question_id=question_id,
                    path="input",
                    details={"missing_fields": missing},
                )
        elif source_type == "synthetic" and not str(input_data.get("note") or "").strip():
            _add_issue(
                issues,
                "warning",
                "MISSING_SYNTHETIC_PURPOSE",
                "合成样本建议在note中说明设计目的",
                question_id=question_id,
                path="input.note",
            )
    except Exception as exc:
        logger.error("检查BioToolEval输入失败: %s", exc, exc_info=True)
        _add_issue(
            issues,
            "error",
            "INPUT_CHECK_EXCEPTION",
            "输入检查器运行异常",
            question_id=item.get("id") if isinstance(item, dict) else None,
        )


def _check_gold(
    item: dict[str, Any],
    signatures: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    try:
        question_id = item.get("id")
        if "submitted_gold" in item:
            _add_issue(
                issues,
                "error",
                "UNVERIFIED_GOLD",
                "submitted_gold尚未经过本地复算与双人复核，不能进入正式指标",
                question_id=question_id,
                path="submitted_gold",
            )
        gold = item.get("gold")
        if not isinstance(gold, dict):
            return

        preferred = gold.get("preferred_tool_chain")
        acceptable = gold.get("acceptable_tool_chains")
        preferred = preferred if isinstance(preferred, list) else []
        acceptable = acceptable if isinstance(acceptable, list) else []
        chains = [preferred] + [chain for chain in acceptable if isinstance(chain, list)]
        behavior = gold.get("expected_behavior")
        expected_status = gold.get("expected_status")

        if behavior == "reject":
            if expected_status != "error" or preferred:
                _add_issue(
                    issues,
                    "error",
                    "REJECT_GOLD_CONFLICT",
                    "reject题必须expected_status=error且preferred_tool_chain为空",
                    question_id=question_id,
                    path="gold",
                )
        elif behavior in {"execute", "repair_then_execute"}:
            if expected_status != "success" or not preferred:
                _add_issue(
                    issues,
                    "error",
                    "EXECUTE_GOLD_CONFLICT",
                    "执行题必须expected_status=success且给出非空首选工具链",
                    question_id=question_id,
                    path="gold",
                )

        for chain_index, chain in enumerate(chains):
            if not chain:
                continue
            unknown = [
                str(tool)
                for tool in chain
                if not isinstance(tool, str) or tool not in signatures
            ]
            if unknown:
                _add_issue(
                    issues,
                    "error",
                    "UNKNOWN_GOLD_TOOL",
                    "Gold工具链包含未注册工具",
                    question_id=question_id,
                    path=f"gold.tool_chains.{chain_index}",
                    details={"unknown_tools": unknown},
                )
                continue
            initial_types = _initial_types(item.get("input"))
            if not initial_types:
                _add_issue(
                    issues,
                    "error",
                    "MISSING_INITIAL_TYPE_FOR_CHAIN",
                    "非空Gold工具链没有可映射的初始输入类型",
                    question_id=question_id,
                    path="input.input_type",
                )
                continue
            validation = validate_tool_plan(chain, initial_types)
            if validation.get("status") != "success" or not validation.get("valid"):
                codes = sorted(
                    {
                        str(issue.get("code"))
                        for issue in validation.get("issues", [])
                        if isinstance(issue, dict) and issue.get("code")
                    }
                )
                _add_issue(
                    issues,
                    "error",
                    "TYPE_INCOMPATIBLE_GOLD_CHAIN",
                    "Gold工具链与题目初始输入类型不兼容",
                    question_id=question_id,
                    path=f"gold.tool_chains.{chain_index}",
                    details={"type_issue_codes": codes},
                )

        if gold.get("order_sensitive") is True and len(preferred) < 2:
            _add_issue(
                issues,
                "error",
                "MEANINGLESS_ORDER_SENSITIVE",
                "order_sensitive=true只适用于至少两个工具的链",
                question_id=question_id,
                path="gold.order_sensitive",
            )

        expected_output = gold.get("expected_output")
        numeric_paths = _iter_numeric_leaf_paths(expected_output)
        tolerance = gold.get("tolerance")
        for numeric_path in numeric_paths:
            value = _tolerance_for_path(tolerance, numeric_path)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < 0
            ):
                _add_issue(
                    issues,
                    "error",
                    "MISSING_NUMERIC_TOLERANCE",
                    "每个数值Gold字段必须配置非负数值容差",
                    question_id=question_id,
                    path=f"gold.expected_output.{numeric_path}",
                )

        requires_semantic_rules = (
            not isinstance(expected_output, dict)
            or not expected_output
            or _has_semantic_leaf(expected_output)
        )
        if requires_semantic_rules:
            required_claims = gold.get("required_claims")
            forbidden_claims = gold.get("forbidden_claims")
            if not (
                isinstance(required_claims, list)
                and any(str(value).strip() for value in required_claims)
                and isinstance(forbidden_claims, list)
                and any(str(value).strip() for value in forbidden_claims)
            ):
                _add_issue(
                    issues,
                    "error",
                    "INCOMPLETE_SEMANTIC_GOLD",
                    "定性或复合输出题必须同时给出非空required_claims和forbidden_claims",
                    question_id=question_id,
                    path="gold",
                )
            else:
                required_set = {str(value).strip().casefold() for value in required_claims}
                forbidden_set = {str(value).strip().casefold() for value in forbidden_claims}
                if required_set & forbidden_set:
                    _add_issue(
                        issues,
                        "error",
                        "CONTRADICTORY_CLAIMS",
                        "required_claims与forbidden_claims存在相同条目",
                        question_id=question_id,
                        path="gold",
                    )

        provenance = item.get("provenance")
        if numeric_paths and isinstance(provenance, dict):
            missing_reproducibility = [
                key
                for key in ("python_version", "biopython_version", "git_commit")
                if not str(provenance.get(key) or "").strip()
            ]
            if missing_reproducibility:
                _add_issue(
                    issues,
                    "warning",
                    "INCOMPLETE_NUMERIC_REPRODUCIBILITY",
                    "数值Gold建议记录Python、Biopython和代码commit",
                    question_id=question_id,
                    path="provenance",
                    details={"missing_fields": missing_reproducibility},
                )
    except Exception as exc:
        logger.error("检查BioToolEval Gold失败: %s", exc, exc_info=True)
        _add_issue(
            issues,
            "error",
            "GOLD_CHECK_EXCEPTION",
            "Gold检查器运行异常",
            question_id=item.get("id") if isinstance(item, dict) else None,
        )


def _check_governance(
    item: dict[str, Any],
    dataset_status: str,
    issues: list[dict[str, Any]],
) -> None:
    try:
        question_id = item.get("id")
        split = str(item.get("split") or "").strip()
        governance = item.get("governance")
        if not isinstance(governance, dict):
            return

        source = str(governance.get("question_source") or "").strip()
        source_reference = str(governance.get("source_reference") or "").strip()
        if source in {"adapted_from_source", "external_submission"} and not source_reference:
            _add_issue(
                issues,
                "error",
                "MISSING_QUESTION_SOURCE_REFERENCE",
                "改编题或外部提交题必须记录可追溯来源",
                question_id=question_id,
                path="governance.source_reference",
            )

        exposure = str(governance.get("exposure_status") or "").strip()
        freeze_commit = str(governance.get("method_freeze_commit") or "").strip()
        if split == "test" and exposure == "seen_by_method_developer":
            _add_issue(
                issues,
                "error",
                "TEST_EXPOSURE_LEAKAGE",
                "test题已被方法开发者查看，不能继续作为独立盲测",
                question_id=question_id,
                path="governance.exposure_status",
            )
        if exposure == "released_after_freeze" and not freeze_commit:
            _add_issue(
                issues,
                "error",
                "MISSING_METHOD_FREEZE_COMMIT",
                "解封后的题目必须记录方法冻结commit",
                question_id=question_id,
                path="governance.method_freeze_commit",
            )
        if split in {"pilot", "dev"} and exposure == "blind_to_method_developer":
            _add_issue(
                issues,
                "info",
                "BLIND_NON_TEST_QUESTION",
                "pilot/dev题仍处于盲态；使用前应确认是否需要保留为test",
                question_id=question_id,
                path="governance.exposure_status",
            )

        gold_status = str(governance.get("gold_status") or "").strip()
        reviewers = governance.get("reviewed_by")
        reviewers = reviewers if isinstance(reviewers, list) else []
        normalized_reviewers = {
            str(reviewer).strip().casefold()
            for reviewer in reviewers
            if str(reviewer).strip()
        }
        author = str(governance.get("question_author") or "").strip().casefold()
        is_formal = dataset_status in _FORMAL_DATASET_STATUSES
        if is_formal and gold_status not in {"human_verified", "adjudicated"}:
            _add_issue(
                issues,
                "error",
                "INSUFFICIENT_GOLD_REVIEW",
                "正式数据集的Gold必须经过人工复核或裁决",
                question_id=question_id,
                path="governance.gold_status",
            )
        if is_formal and len(normalized_reviewers) < 2:
            _add_issue(
                issues,
                "error",
                "INSUFFICIENT_INDEPENDENT_REVIEWERS",
                "正式题目至少需要两名不同复核者",
                question_id=question_id,
                path="governance.reviewed_by",
                details={"reviewer_count": len(normalized_reviewers)},
            )
        if (
            is_formal
            and author
            and normalized_reviewers
            and normalized_reviewers == {author}
        ):
            _add_issue(
                issues,
                "error",
                "AUTHOR_ONLY_REVIEW",
                "正式题目不能只由出题者本人复核",
                question_id=question_id,
                path="governance.reviewed_by",
            )
        reviewed_at = str(governance.get("reviewed_at") or "").strip()
        if gold_status in {"human_verified", "adjudicated"} and not reviewed_at:
            _add_issue(
                issues,
                "error",
                "MISSING_GOLD_REVIEW_DATE",
                "人工复核或裁决的Gold必须记录复核日期",
                question_id=question_id,
                path="governance.reviewed_at",
            )
    except Exception as exc:
        logger.error("检查BioToolEval题目治理元数据失败: %s", exc, exc_info=True)
        _add_issue(
            issues,
            "error",
            "GOVERNANCE_CHECK_EXCEPTION",
            "题目治理检查器运行异常",
            question_id=item.get("id") if isinstance(item, dict) else None,
        )


def _input_fingerprint(item: dict[str, Any]) -> str:
    try:
        input_data = item.get("input")
        if not isinstance(input_data, dict):
            return ""
        input_type = str(input_data.get("input_type") or "").strip()
        identity = ""
        if input_type == "amino_acid_sequence":
            identity = re.sub(r"\s+", "", str(input_data.get("sequence") or "")).upper()
        elif input_type in {"protein_fasta_file", "pdb_file"}:
            accession = str(input_data.get("accession") or "").strip().casefold()
            fixture_name = Path(str(input_data.get("fixture") or "")).name.casefold()
            identity = accession or fixture_name
        if not input_type or not identity:
            return ""
        payload = f"{input_type}|{identity}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    except Exception as exc:
        logger.error("生成BioToolEval输入指纹失败: %s", exc, exc_info=True)
        return ""


def _check_duplicates(
    items: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    threshold: float,
) -> None:
    try:
        exact_seen: dict[str, tuple[str, str]] = {}
        normalized_records: list[tuple[str, str, str]] = []
        template_groups: dict[str, list[tuple[str, str]]] = {}
        input_fingerprints: dict[str, list[tuple[str, str]]] = {}
        for item in items:
            question_id = str(item.get("id") or "")
            split = str(item.get("split") or "")
            zh = _normalize_text(item.get("question_zh"))
            en = _normalize_text(item.get("question_en"))
            combined = f"{zh}|{en}"
            if combined and combined in exact_seen:
                previous_id, previous_split = exact_seen[combined]
                _add_issue(
                    issues,
                    "error",
                    "EXACT_DUPLICATE_QUESTION",
                    "发现题面完全重复",
                    question_id=question_id,
                    details={
                        "paired_question_id": previous_id,
                        "cross_split": previous_split != split,
                    },
                )
            elif combined:
                exact_seen[combined] = (question_id, split)
            normalized_records.append((question_id, split, combined))

            governance = item.get("governance")
            governance = governance if isinstance(governance, dict) else {}
            template_group = str(governance.get("template_group") or "").strip()
            if template_group:
                template_groups.setdefault(template_group, []).append((question_id, split))
            input_fingerprint = _input_fingerprint(item)
            if input_fingerprint:
                input_fingerprints.setdefault(input_fingerprint, []).append(
                    (question_id, split)
                )

        for left_index in range(len(normalized_records)):
            left_id, left_split, left_text = normalized_records[left_index]
            if not left_text:
                continue
            for right_index in range(left_index + 1, len(normalized_records)):
                right_id, right_split, right_text = normalized_records[right_index]
                if not right_text or left_text == right_text:
                    continue
                ratio = SequenceMatcher(None, left_text, right_text).ratio()
                if ratio < threshold:
                    continue
                cross_split = left_split != right_split
                _add_issue(
                    issues,
                    "error" if cross_split else "warning",
                    "CROSS_SPLIT_NEAR_DUPLICATE" if cross_split else "WITHIN_SPLIT_NEAR_DUPLICATE",
                    "发现推理模板可能近重复的题目对",
                    question_id=right_id,
                    details={
                        "paired_question_id": left_id,
                        "similarity": round(ratio, 4),
                    },
                )

        for template_group, records in template_groups.items():
            if len(records) < 2:
                continue
            splits = {split for _, split in records}
            question_ids = [question_id for question_id, _ in records]
            _add_issue(
                issues,
                "error" if len(splits) > 1 else "warning",
                (
                    "CROSS_SPLIT_TEMPLATE_LEAKAGE"
                    if len(splits) > 1
                    else "WITHIN_SPLIT_TEMPLATE_REUSE"
                ),
                (
                    "同一模板组跨split出现，存在模板泄漏"
                    if len(splits) > 1
                    else "同一split内复用模板组，统计时应防止伪重复"
                ),
                question_id=question_ids[-1],
                path="governance.template_group",
                details={
                    "paired_question_ids": question_ids[:-1],
                    "template_group": template_group,
                },
            )

        for records in input_fingerprints.values():
            if len(records) < 2:
                continue
            splits = {split for _, split in records}
            if len(splits) <= 1:
                continue
            question_ids = [question_id for question_id, _ in records]
            _add_issue(
                issues,
                "warning",
                "CROSS_SPLIT_INPUT_REUSE",
                "同一生物输入跨split复用，需要人工确认是否造成属性或模板泄漏",
                question_id=question_ids[-1],
                path="input",
                details={"paired_question_ids": question_ids[:-1]},
            )
    except Exception as exc:
        logger.error("检查BioToolEval重复题失败: %s", exc, exc_info=True)
        _add_issue(issues, "error", "DUPLICATE_CHECK_EXCEPTION", "重复题检查器运行异常")


def _compute_summary(
    items: list[dict[str, Any]],
    signatures: dict[str, Any],
) -> dict[str, Any]:
    try:
        categories = Counter()
        difficulties = Counter()
        splits = Counter()
        input_types = Counter()
        source_types = Counter()
        tool_coverage = Counter()
        expected_behaviors = Counter()
        preferred_chain_lengths = Counter()
        question_sources = Counter()
        exposure_statuses = Counter()
        gold_statuses = Counter()
        template_groups: set[str] = set()
        reviewed_question_count = 0
        blind_test_question_count = 0
        type_sensitive_count = 0
        for item in items:
            categories[str(item.get("category") or "missing")] += 1
            difficulties[str(item.get("difficulty") or "missing")] += 1
            splits[str(item.get("split") or "missing")] += 1
            input_data = item.get("input") if isinstance(item.get("input"), dict) else {}
            input_types[str(input_data.get("input_type") or "missing")] += 1
            source_types[str(input_data.get("source_type") or "missing")] += 1
            gold = item.get("gold") if isinstance(item.get("gold"), dict) else {}
            chain = gold.get("preferred_tool_chain")
            expected_behaviors[str(gold.get("expected_behavior") or "missing")] += 1
            if isinstance(chain, list):
                preferred_chain_lengths[str(len(chain))] += 1
                for tool in set(str(value) for value in chain):
                    if tool in signatures:
                        tool_coverage[tool] += 1
                if (
                    len(chain) >= 2
                    or gold.get("expected_behavior") in {"repair_then_execute", "reject"}
                    or input_data.get("input_type") in {"protein_fasta_file", "pdb_file"}
                ):
                    type_sensitive_count += 1
            else:
                preferred_chain_lengths["missing"] += 1

            governance = item.get("governance")
            governance = governance if isinstance(governance, dict) else {}
            question_sources[str(governance.get("question_source") or "missing")] += 1
            exposure = str(governance.get("exposure_status") or "missing")
            exposure_statuses[exposure] += 1
            gold_status = str(governance.get("gold_status") or "missing")
            gold_statuses[gold_status] += 1
            template_group = str(governance.get("template_group") or "").strip()
            if template_group:
                template_groups.add(template_group)
            reviewers = governance.get("reviewed_by")
            if isinstance(reviewers, list) and any(str(value).strip() for value in reviewers):
                reviewed_question_count += 1
            if item.get("split") == "test" and exposure == "blind_to_method_developer":
                blind_test_question_count += 1
        uncovered_tools = sorted(set(signatures) - set(tool_coverage))
        return {
            "item_count": len(items),
            "categories": dict(sorted(categories.items())),
            "difficulties": dict(sorted(difficulties.items())),
            "splits": dict(sorted(splits.items())),
            "input_types": dict(sorted(input_types.items())),
            "source_types": dict(sorted(source_types.items())),
            "expected_behaviors": dict(sorted(expected_behaviors.items())),
            "preferred_chain_lengths": dict(sorted(preferred_chain_lengths.items())),
            "tool_coverage": dict(sorted(tool_coverage.items())),
            "covered_tool_count": len(tool_coverage),
            "registered_tool_count": len(signatures),
            "uncovered_tools": uncovered_tools,
            "type_sensitive_question_count": type_sensitive_count,
            "question_sources": dict(sorted(question_sources.items())),
            "exposure_statuses": dict(sorted(exposure_statuses.items())),
            "gold_statuses": dict(sorted(gold_statuses.items())),
            "template_group_count": len(template_groups),
            "reviewed_question_count": reviewed_question_count,
            "blind_test_question_count": blind_test_question_count,
        }
    except Exception as exc:
        logger.error("计算BioToolEval统计摘要失败: %s", exc, exc_info=True)
        return {"item_count": len(items)}


def _coverage_signatures_for_profile(
    profile: str,
    signatures: dict[str, Any],
) -> dict[str, Any]:
    try:
        if profile in {"challenge30", "dev40", "dev70"}:
            return {
                tool_name: signature
                for tool_name, signature in signatures.items()
                if isinstance(signature, dict)
                and signature.get("benchmark_generation") == "legacy-v1"
            }
        return dict(signatures)
    except Exception as exc:
        logger.error("选择Profile工具代际失败: %s", exc, exc_info=True)
        return dict(signatures)


def _infer_profile(items: list[dict[str, Any]]) -> str:
    try:
        splits = {str(item.get("split") or "") for item in items}
        if len(items) == 30 and splits == {"dev"}:
            return "challenge30"
        if len(items) == 12 and splits == {"dev"}:
            return "topologyv2pilot12"
        if len(items) == 30:
            return "pilot30"
        if len(items) == 40 and splits == {"dev"}:
            return "dev40"
        if len(items) == 70 and splits == {"dev"}:
            return "dev70"
        if len(items) == 35 and splits == {"dev"}:
            return "dev35"
        if len(items) == 35 and splits == {"test"}:
            return "test35"
        if len(items) == 100:
            return "final100"
        if len(items) == 150:
            return "final150"
        return "none"
    except Exception as exc:
        logger.error("自动推断数据集验收profile失败: %s", exc, exc_info=True)
        return "none"


def _check_profile(
    profile: str,
    summary: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    try:
        if profile == "none":
            return
        rule = _PROFILE_RULES.get(profile)
        if not isinstance(rule, dict):
            _add_issue(issues, "error", "UNKNOWN_PROFILE", "未知的数据集验收profile")
            return
        if summary.get("item_count") != rule.get("item_count"):
            _add_issue(
                issues,
                "error",
                "PROFILE_ITEM_COUNT_MISMATCH",
                "题目总数不符合验收profile",
                details={
                    "expected": rule.get("item_count"),
                    "actual": summary.get("item_count"),
                },
            )
        categories = summary.get("categories") if isinstance(summary.get("categories"), dict) else {}
        for category, expected in rule.get("categories", {}).items():
            actual = int(categories.get(category, 0))
            if actual != expected:
                _add_issue(
                    issues,
                    "error",
                    "PROFILE_CATEGORY_MISMATCH",
                    "题型数量不符合验收profile",
                    path=f"summary.categories.{category}",
                    details={"expected": expected, "actual": actual},
                )
        difficulties = (
            summary.get("difficulties")
            if isinstance(summary.get("difficulties"), dict)
            else {}
        )
        for difficulty, expected in rule.get("difficulties", {}).items():
            actual = int(difficulties.get(difficulty, 0))
            if actual != int(expected):
                _add_issue(
                    issues,
                    "error",
                    "PROFILE_DIFFICULTY_MISMATCH",
                    "难度数量不符合验收profile",
                    path=f"summary.difficulties.{difficulty}",
                    details={"expected": int(expected), "actual": actual},
                )
        actual_splits = {
            split
            for split, count in (summary.get("splits") or {}).items()
            if count
        }
        if not actual_splits.issubset(set(rule.get("allowed_splits", set()))):
            _add_issue(
                issues,
                "error",
                "PROFILE_SPLIT_MISMATCH",
                "数据集split不符合验收profile",
                details={"actual_splits": sorted(actual_splits)},
            )
        source_types = summary.get("source_types") or {}
        real_count = int(source_types.get("real", 0))
        if real_count < int(rule.get("minimum_real_inputs", 0)):
            _add_issue(
                issues,
                "error",
                "INSUFFICIENT_REAL_INPUTS",
                "真实来源样本数量低于预注册下限",
                details={
                    "minimum": rule.get("minimum_real_inputs"),
                    "actual": real_count,
                },
            )
        input_types = summary.get("input_types") or {}
        pdb_count = int(input_types.get("pdb_file", 0))
        if pdb_count < int(rule.get("minimum_pdb_inputs", 0)):
            _add_issue(
                issues,
                "error",
                "INSUFFICIENT_PDB_INPUTS",
                "PDB输入题数量低于预注册下限",
                details={"minimum": rule.get("minimum_pdb_inputs"), "actual": pdb_count},
            )
        for input_type, minimum in rule.get("minimum_input_types", {}).items():
            actual = int(input_types.get(input_type, 0))
            if actual < int(minimum):
                _add_issue(
                    issues,
                    "error",
                    "INSUFFICIENT_PROFILE_INPUT_TYPE",
                    "指定输入类型的题目数量低于验收profile下限",
                    path=f"summary.input_types.{input_type}",
                    details={
                        "input_type": input_type,
                        "minimum": int(minimum),
                        "actual": actual,
                    },
                )
        for input_type, expected in rule.get("exact_input_types", {}).items():
            actual = int(input_types.get(input_type, 0))
            if actual != int(expected):
                _add_issue(
                    issues,
                    "error",
                    "PROFILE_INPUT_TYPE_MISMATCH",
                    "输入类型数量不符合验收profile",
                    path=f"summary.input_types.{input_type}",
                    details={"expected": int(expected), "actual": actual},
                )
        for source_type, expected in rule.get("exact_source_types", {}).items():
            actual = int(source_types.get(source_type, 0))
            if actual != int(expected):
                _add_issue(
                    issues,
                    "error",
                    "PROFILE_SOURCE_TYPE_MISMATCH",
                    "来源类型数量不符合验收profile",
                    path=f"summary.source_types.{source_type}",
                    details={"expected": int(expected), "actual": actual},
                )
        expected_behaviors = summary.get("expected_behaviors") or {}
        for behavior, expected in rule.get("expected_behaviors", {}).items():
            actual = int(expected_behaviors.get(behavior, 0))
            if actual != int(expected):
                _add_issue(
                    issues,
                    "error",
                    "PROFILE_EXPECTED_BEHAVIOR_MISMATCH",
                    "预期行为数量不符合验收profile",
                    path=f"summary.expected_behaviors.{behavior}",
                    details={"expected": int(expected), "actual": actual},
                )
        for behavior, minimum in rule.get("minimum_expected_behaviors", {}).items():
            actual = int(expected_behaviors.get(behavior, 0))
            if actual < int(minimum):
                _add_issue(
                    issues,
                    "error",
                    "INSUFFICIENT_EXPECTED_BEHAVIOR",
                    "指定预期行为的题目数量低于验收profile下限",
                    path=f"summary.expected_behaviors.{behavior}",
                    details={
                        "expected_behavior": behavior,
                        "minimum": int(minimum),
                        "actual": actual,
                    },
                )
        tool_coverage = summary.get("tool_coverage") or {}
        for tool_name, minimum in rule.get("minimum_tool_coverage", {}).items():
            actual = int(tool_coverage.get(tool_name, 0))
            if actual < int(minimum):
                _add_issue(
                    issues,
                    "error",
                    "INSUFFICIENT_PROFILE_TOOL_COVERAGE",
                    "指定工具的题目覆盖数低于验收profile下限",
                    path=f"summary.tool_coverage.{tool_name}",
                    details={
                        "tool_name": tool_name,
                        "minimum": int(minimum),
                        "actual": actual,
                    },
                )
        maximum_tool_coverage = rule.get("maximum_tool_coverage")
        if maximum_tool_coverage is not None:
            overused_tools = {
                str(tool_name): int(count)
                for tool_name, count in tool_coverage.items()
                if int(count) > int(maximum_tool_coverage)
            }
            if overused_tools:
                _add_issue(
                    issues,
                    "error",
                    "EXCESSIVE_PROFILE_TOOL_COVERAGE",
                    "单一工具题目覆盖数超过验收profile上限",
                    details={
                        "maximum": int(maximum_tool_coverage),
                        "overused_tools": dict(sorted(overused_tools.items())),
                    },
                )
        toxicity_count = int(tool_coverage.get("check_protein_toxicity", 0))
        if toxicity_count < int(rule.get("minimum_toxicity_questions", 0)):
            _add_issue(
                issues,
                "error",
                "INSUFFICIENT_TOXICITY_QUESTIONS",
                "毒性相似性筛查题数量低于预注册下限",
                details={
                    "minimum": rule.get("minimum_toxicity_questions"),
                    "actual": toxicity_count,
                },
            )
        type_sensitive_count = int(summary.get("type_sensitive_question_count", 0))
        if type_sensitive_count < int(rule.get("minimum_type_sensitive_questions", 0)):
            _add_issue(
                issues,
                "error",
                "INSUFFICIENT_TYPE_SENSITIVE_QUESTIONS",
                "类型敏感题数量低于预注册下限",
                details={
                    "minimum": rule.get("minimum_type_sensitive_questions"),
                    "actual": type_sensitive_count,
                },
            )
        if rule.get("require_all_tools") and summary.get("uncovered_tools"):
            _add_issue(
                issues,
                "error",
                "INCOMPLETE_TOOL_COVERAGE",
                "数据集尚未覆盖全部外部生物工具",
                details={"uncovered_tools": summary.get("uncovered_tools")},
            )
    except Exception as exc:
        logger.error("检查BioToolEval profile失败: %s", exc, exc_info=True)
        _add_issue(issues, "error", "PROFILE_CHECK_EXCEPTION", "分布验收器运行异常")


def _check_final150_governance(
    profile: str,
    dataset_status: str,
    items: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    try:
        if profile != "final150":
            return

        expected_ids = {f"bte_{index:03d}" for index in range(1, 151)}
        actual_ids = {str(item.get("id") or "").strip() for item in items}
        if actual_ids != expected_ids:
            _add_issue(
                issues,
                "error",
                "PROFILE_ID_SET_MISMATCH",
                "Test-150题号必须恰好为bte_001至bte_150",
                details={
                    "missing_count": len(expected_ids - actual_ids),
                    "unexpected_count": len(actual_ids - expected_ids),
                },
            )

        freeze_commits: list[str] = []
        missing_freeze_ids: list[str] = []
        invalid_freeze_ids: list[str] = []
        for item in items:
            governance = item.get("governance")
            governance = governance if isinstance(governance, dict) else {}
            commit = str(governance.get("method_freeze_commit") or "").strip()
            if commit:
                freeze_commits.append(commit.lower())
                if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
                    invalid_freeze_ids.append(str(item.get("id") or ""))
            else:
                missing_freeze_ids.append(str(item.get("id") or ""))

        is_formal = dataset_status in _FORMAL_DATASET_STATUSES
        if is_formal and missing_freeze_ids:
            _add_issue(
                issues,
                "error",
                "FINAL150_METHOD_FREEZE_INCOMPLETE",
                "正式Test-150的每道题都必须绑定同一方法冻结commit",
                details={"missing_count": len(missing_freeze_ids)},
            )
        if invalid_freeze_ids:
            _add_issue(
                issues,
                "error",
                "INVALID_METHOD_FREEZE_COMMIT",
                "方法冻结commit必须是40位十六进制Git提交",
                details={"invalid_count": len(invalid_freeze_ids)},
            )
        unique_commits = sorted(set(freeze_commits))
        if len(unique_commits) > 1:
            _add_issue(
                issues,
                "error",
                "INCONSISTENT_METHOD_FREEZE_COMMIT",
                "Test-150题目绑定了不一致的方法冻结commit",
                details={"unique_commit_count": len(unique_commits)},
            )
        if not is_formal and freeze_commits:
            _add_issue(
                issues,
                "error",
                "PREMATURE_METHOD_FREEZE_BINDING",
                "候选Test-150尚未完成正式冻结，不应提前写入方法commit",
                details={"bound_item_count": len(freeze_commits)},
            )
    except Exception as exc:
        logger.error("检查Test-150治理门禁失败: %s", exc, exc_info=True)
        _add_issue(
            issues,
            "error",
            "FINAL150_GOVERNANCE_CHECK_EXCEPTION",
            "Test-150治理门禁运行异常",
        )


def _build_dataset_readiness(
    profile: str,
    dataset_status: str,
    verdict: str,
) -> dict[str, Any]:
    try:
        machine_passed = verdict == "pass"
        formal = dataset_status in _FORMAL_DATASET_STATUSES
        formal_ready = profile == "final150" and machine_passed and formal
        if not machine_passed:
            stage = "blocked"
        elif profile == "final150" and not formal:
            stage = "candidate_machine_validated"
        elif formal_ready:
            stage = "formal_dataset_gate_ready"
        else:
            stage = "validated"
        return {
            "stage": stage,
            "machine_validation_passed": machine_passed,
            "formal_dataset_status": formal,
            "formal_dataset_gate_ready": formal_ready,
            "external_gates_still_required": (
                [
                    "round1_review_artifact",
                    "round2_author_review_artifact",
                    "disagreement_adjudication",
                    "gold_recomputation_audit",
                    "dev_test_fingerprint_audit",
                    "method_prompt_model_statistics_freeze",
                ]
                if profile == "final150"
                else []
            ),
        }
    except Exception as exc:
        logger.error("构造数据集就绪状态失败: %s", exc, exc_info=True)
        return {
            "stage": "unknown",
            "machine_validation_passed": False,
            "formal_dataset_gate_ready": False,
        }


def validate_biotooleval_dataset(
    dataset_path: Any,
    *,
    profile: str = "auto",
    near_duplicate_threshold: float = 0.88,
    config: Any = None,
) -> dict[str, Any]:
    try:
        cfg = config if config is not None else Config()
        path = Path(str(dataset_path or "")).expanduser()
        if not path.is_file() or path.suffix.lower() != ".json":
            return _runtime_error("BioToolEval数据集不存在或不是JSON文件", "DATASET_NOT_FOUND")
        normalized_profile = str(profile or "auto").strip().lower()
        if normalized_profile not in _VALID_PROFILES:
            return _runtime_error("不支持的数据集验收profile", "INVALID_PROFILE")
        try:
            threshold = float(near_duplicate_threshold)
        except (TypeError, ValueError):
            return _runtime_error("近重复阈值必须是数值", "INVALID_DUPLICATE_THRESHOLD")
        if threshold < 0.5 or threshold > 1.0:
            return _runtime_error("近重复阈值必须位于0.5到1.0之间", "INVALID_DUPLICATE_THRESHOLD")

        schema_result = _load_schema(cfg.EXPERIMENT_QUESTION_SCHEMA_PATH)
        if schema_result.get("status") != "success":
            return schema_result
        signature_result = get_tool_signatures()
        if signature_result.get("status") != "success":
            return _runtime_error(
                signature_result.get("message", "工具签名注册表不可用"),
                "SIGNATURE_REGISTRY_ERROR",
            )
        signatures = signature_result.get("signatures", {})
        dataset = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(dataset, dict):
            return _runtime_error("BioToolEval数据集顶层必须是JSON对象", "INVALID_DATASET_ROOT")
        raw_items = dataset.get("items")
        if not isinstance(raw_items, list):
            return _runtime_error("BioToolEval数据集缺少items数组", "INVALID_ITEMS")

        issues: list[dict[str, Any]] = []
        items = [item for item in raw_items if isinstance(item, dict)]
        if len(items) != len(raw_items):
            _add_issue(
                issues,
                "error",
                "NON_OBJECT_ITEM",
                "items中存在非对象元素",
                details={"invalid_item_count": len(raw_items) - len(items)},
            )
        if not str(dataset.get("dataset_name") or "").strip():
            _add_issue(issues, "error", "MISSING_DATASET_NAME", "缺少dataset_name")
        if not str(dataset.get("version") or "").strip():
            _add_issue(issues, "error", "MISSING_DATASET_VERSION", "缺少数据集version")
        status_value = str(dataset.get("status") or "").strip()
        if (
            status_value not in _FORMAL_DATASET_STATUSES
            and normalized_profile not in {
                "none",
                "challenge30",
                "dev40",
                "dev70",
                "topologyv2pilot12",
                "final150",
            }
        ):
            _add_issue(
                issues,
                "error",
                "NON_FORMAL_DATASET_STATUS",
                "正式入库只接受benchmark_ready、frozen或released状态",
                path="status",
                details={"actual_status": status_value or "missing"},
            )
        elif status_value not in _FORMAL_DATASET_STATUSES:
            _add_issue(
                issues,
                "warning",
                "NON_FORMAL_DATASET_STATUS",
                "结构检查已执行，但当前状态不能用于正式论文指标",
                path="status",
                details={"actual_status": status_value or "missing"},
            )

        validator = schema_result["validator"]
        seen_ids: set[str] = set()
        for index, item in enumerate(items):
            question_id = str(item.get("id") or "").strip() or f"index:{index}"
            if question_id in seen_ids:
                _add_issue(
                    issues,
                    "error",
                    "DUPLICATE_QUESTION_ID",
                    "题目ID重复",
                    question_id=question_id,
                    path=f"items.{index}.id",
                )
            seen_ids.add(question_id)
            for error in sorted(validator.iter_errors(item), key=lambda value: list(value.path)):
                _add_issue(
                    issues,
                    "error",
                    "QUESTION_SCHEMA_VIOLATION",
                    _schema_error_message(error),
                    question_id=question_id,
                    path=_schema_error_path(error),
                    details={"schema_constraint": str(error.validator or "unknown")},
                )
            _check_input(item, path, issues)
            _check_gold(item, signatures, issues)
            _check_governance(item, status_value, issues)

        selected_profile = _infer_profile(items) if normalized_profile == "auto" else normalized_profile
        _check_final150_governance(selected_profile, status_value, items, issues)
        _check_duplicates(items, issues, threshold)
        coverage_signatures = _coverage_signatures_for_profile(
            selected_profile,
            signatures,
        )
        summary = _compute_summary(items, coverage_signatures)
        summary["tool_registry_scope"] = (
            "legacy-v1"
            if selected_profile in {"challenge30", "dev40", "dev70"}
            else "current"
        )
        _check_profile(selected_profile, summary, issues)

        declared_summary = dataset.get("summary")
        if isinstance(declared_summary, dict):
            declared_count = declared_summary.get("item_count")
            if declared_count is not None and declared_count != len(raw_items):
                _add_issue(
                    issues,
                    "error",
                    "DECLARED_ITEM_COUNT_MISMATCH",
                    "summary.item_count与items实际数量不一致",
                    path="summary.item_count",
                    details={"declared": declared_count, "actual": len(raw_items)},
                )
            ready_count = declared_summary.get("benchmark_ready_count")
            if status_value in _FORMAL_DATASET_STATUSES:
                if ready_count is not None and ready_count != len(raw_items):
                    _add_issue(
                        issues,
                        "error",
                        "BENCHMARK_READY_COUNT_MISMATCH",
                        "正式数据集的benchmark_ready_count必须等于题目总数",
                        path="summary.benchmark_ready_count",
                        details={"declared": ready_count, "actual": len(raw_items)},
                    )
                elif ready_count is None:
                    _add_issue(
                        issues,
                        "error",
                        "MISSING_BENCHMARK_READY_COUNT",
                        "正式数据集summary必须声明benchmark_ready_count",
                        path="summary.benchmark_ready_count",
                    )
            elif ready_count not in {None, 0}:
                _add_issue(
                    issues,
                    "warning",
                    "NON_FORMAL_READY_COUNT",
                    "非正式数据集不应把候选题计入benchmark_ready_count",
                    path="summary.benchmark_ready_count",
                    details={"declared": ready_count},
                )
            for summary_key in (
                "categories",
                "difficulties",
                "splits",
                "input_types",
                "source_types",
                "expected_behaviors",
                "preferred_chain_lengths",
                "tool_coverage",
                "question_sources",
                "exposure_statuses",
                "gold_statuses",
            ):
                declared_distribution = declared_summary.get(summary_key)
                if (
                    declared_distribution is not None
                    and declared_distribution != summary.get(summary_key)
                ):
                    _add_issue(
                        issues,
                        "error",
                        "DECLARED_DISTRIBUTION_MISMATCH",
                        "summary声明的分布与逐题计算结果不一致",
                        path=f"summary.{summary_key}",
                        details={
                            "distribution_name": summary_key,
                            "computed": summary.get(summary_key),
                        },
                    )
        else:
            _add_issue(issues, "error", "MISSING_DATASET_SUMMARY", "缺少数据集summary对象")

        severity_counts = Counter(issue.get("severity", "error") for issue in issues)
        verdict = "fail" if severity_counts.get("error", 0) else "pass"
        readiness = _build_dataset_readiness(
            selected_profile,
            status_value,
            verdict,
        )
        result = {
            "status": "success",
            "verdict": verdict,
            "benchmark_ready": (
                verdict == "pass" and status_value in _FORMAL_DATASET_STATUSES
            ),
            "dataset": {
                "file_name": path.name,
                "dataset_name": str(dataset.get("dataset_name") or ""),
                "version": str(dataset.get("version") or ""),
                "declared_status": status_value,
                "schema_version": schema_result.get("schema_version"),
                "profile": selected_profile,
            },
            "summary": summary,
            "readiness": readiness,
            "issue_counts": {
                "error": int(severity_counts.get("error", 0)),
                "warning": int(severity_counts.get("warning", 0)),
                "info": int(severity_counts.get("info", 0)),
            },
            "issues": issues,
        }
        logger.info(
            "BioToolEval验收完成: file=%s profile=%s verdict=%s errors=%s warnings=%s",
            path.name,
            selected_profile,
            verdict,
            result["issue_counts"]["error"],
            result["issue_counts"]["warning"],
        )
        return result
    except Exception as exc:
        logger.error("BioToolEval数据集验收失败: %s", exc, exc_info=True)
        return _runtime_error(f"BioToolEval数据集验收失败: {exc}")


def _markdown_report(result: dict[str, Any]) -> str:
    try:
        dataset = result.get("dataset") if isinstance(result.get("dataset"), dict) else {}
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        counts = result.get("issue_counts") if isinstance(result.get("issue_counts"), dict) else {}
        readiness = (
            result.get("readiness")
            if isinstance(result.get("readiness"), dict)
            else {}
        )
        lines = [
            "# BioToolEval 数据集验收报告",
            "",
            f"- 文件：`{dataset.get('file_name', 'unknown')}`",
            f"- 版本：`{dataset.get('version', 'unknown')}`",
            f"- Profile：`{dataset.get('profile', 'unknown')}`",
            f"- Verdict：**{str(result.get('verdict', 'unknown')).upper()}**",
            f"- 题目数：{summary.get('item_count', 0)}",
            f"- 就绪阶段：`{readiness.get('stage', 'unknown')}`",
            f"- 正式数据集门禁：`{readiness.get('formal_dataset_gate_ready', False)}`",
            (
                f"- 问题计数：error={counts.get('error', 0)}, "
                f"warning={counts.get('warning', 0)}, info={counts.get('info', 0)}"
            ),
            "",
            "## 分布与覆盖",
            "",
            f"- 题型：`{json.dumps(summary.get('categories', {}), ensure_ascii=False)}`",
            f"- 难度：`{json.dumps(summary.get('difficulties', {}), ensure_ascii=False)}`",
            f"- Split：`{json.dumps(summary.get('splits', {}), ensure_ascii=False)}`",
            f"- 输入类型：`{json.dumps(summary.get('input_types', {}), ensure_ascii=False)}`",
            f"- 来源类型：`{json.dumps(summary.get('source_types', {}), ensure_ascii=False)}`",
            f"- 预期行为：`{json.dumps(summary.get('expected_behaviors', {}), ensure_ascii=False)}`",
            f"- 首选链长度：`{json.dumps(summary.get('preferred_chain_lengths', {}), ensure_ascii=False)}`",
            f"- 题目来源：`{json.dumps(summary.get('question_sources', {}), ensure_ascii=False)}`",
            f"- 暴露状态：`{json.dumps(summary.get('exposure_statuses', {}), ensure_ascii=False)}`",
            f"- Gold状态：`{json.dumps(summary.get('gold_statuses', {}), ensure_ascii=False)}`",
            (
                f"- 工具覆盖：{summary.get('covered_tool_count', 0)}/"
                f"{summary.get('registered_tool_count', 0)}"
            ),
            f"- 未覆盖工具：`{json.dumps(summary.get('uncovered_tools', []), ensure_ascii=False)}`",
            f"- 类型敏感题：{summary.get('type_sensitive_question_count', 0)}",
            f"- 模板组数：{summary.get('template_group_count', 0)}",
            f"- 已复核题：{summary.get('reviewed_question_count', 0)}",
            f"- 仍保持盲态的test题：{summary.get('blind_test_question_count', 0)}",
            "",
            "## 问题清单",
            "",
            "| Severity | Code | Question ID | Path | Message |",
            "|---|---|---|---|---|",
        ]
        for issue in result.get("issues", []):
            if not isinstance(issue, dict):
                continue
            cells = [
                issue.get("severity", ""),
                issue.get("code", ""),
                issue.get("question_id", ""),
                issue.get("path", ""),
                issue.get("message", ""),
            ]
            safe_cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in cells]
            lines.append("| " + " | ".join(safe_cells) + " |")
        if not result.get("issues"):
            lines.append("| info | NO_ISSUES |  |  | 未发现阻断项 |")
        lines.extend(
            [
                "",
                "> 报告只包含ID、统计和错误类型，不包含题面、完整序列或本地绝对路径。",
                "",
            ]
        )
        return "\n".join(lines)
    except Exception as exc:
        logger.error("生成BioToolEval Markdown报告失败: %s", exc, exc_info=True)
        return "# BioToolEval 数据集验收报告\n\n报告生成失败。\n"


def _write_text_atomic(path: Path, content: str) -> dict[str, Any]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
        return {"status": "success", "file_name": path.name}
    except Exception as exc:
        logger.error("原子写入验收报告失败: %s", exc, exc_info=True)
        return _runtime_error(f"写入验收报告失败: {exc}", "REPORT_WRITE_ERROR")


def write_validation_reports(
    result: dict[str, Any],
    *,
    json_report_path: Any = None,
    markdown_report_path: Any = None,
    protected_dataset_path: Any = None,
) -> dict[str, Any]:
    try:
        if not isinstance(result, dict) or result.get("status") not in {"success", "error"}:
            return _runtime_error("验收结果结构无效", "INVALID_REPORT_RESULT")
        protected = (
            Path(str(protected_dataset_path)).expanduser().resolve()
            if protected_dataset_path
            else None
        )
        outputs: list[dict[str, Any]] = []
        for raw_path, report_type in (
            (json_report_path, "json"),
            (markdown_report_path, "markdown"),
        ):
            if not raw_path:
                continue
            output_path = Path(str(raw_path)).expanduser().resolve()
            if protected is not None and output_path == protected:
                return _runtime_error(
                    "验收报告不得覆盖原始数据集",
                    "REPORT_OVERWRITE_DATASET",
                )
            content = (
                json.dumps(result, ensure_ascii=False, indent=2)
                if report_type == "json"
                else _markdown_report(result)
            )
            write_result = _write_text_atomic(output_path, content)
            if write_result.get("status") != "success":
                return write_result
            outputs.append(
                {
                    "report_type": report_type,
                    "file_name": output_path.name,
                }
            )
        return {"status": "success", "outputs": outputs}
    except Exception as exc:
        logger.error("写入BioToolEval验收报告失败: %s", exc, exc_info=True)
        return _runtime_error(f"写入BioToolEval验收报告失败: {exc}", "REPORT_WRITE_ERROR")


def _build_parser() -> argparse.ArgumentParser:
    try:
        parser = argparse.ArgumentParser(description="离线验收BioToolEval数据集")
        parser.add_argument("--dataset", required=True, help="待验收JSON数据集")
        parser.add_argument(
            "--profile",
            default="auto",
            choices=sorted(_VALID_PROFILES),
            help="分布验收profile，默认按题目数和split自动推断",
        )
        parser.add_argument(
            "--near-duplicate-threshold",
            type=float,
            default=0.88,
            help="题面近重复阈值，范围0.5到1.0",
        )
        parser.add_argument("--json-report", default="", help="可选的脱敏JSON报告路径")
        parser.add_argument("--markdown-report", default="", help="可选的Markdown报告路径")
        return parser
    except Exception as exc:
        logger.error("构造验收器CLI失败: %s", exc, exc_info=True)
        raise RuntimeError(f"构造验收器CLI失败: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
        result = validate_biotooleval_dataset(
            args.dataset,
            profile=args.profile,
            near_duplicate_threshold=args.near_duplicate_threshold,
        )
        if args.json_report or args.markdown_report:
            write_result = write_validation_reports(
                result,
                json_report_path=args.json_report,
                markdown_report_path=args.markdown_report,
                protected_dataset_path=args.dataset,
            )
            if write_result.get("status") != "success":
                print(
                    json.dumps(
                        {
                            "status": "error",
                            "message": write_result.get("message"),
                        },
                        ensure_ascii=False,
                    )
                )
                return 1
        print(
            json.dumps(
                {
                    "status": result.get("status"),
                    "verdict": result.get("verdict"),
                    "benchmark_ready": result.get("benchmark_ready"),
                    "issue_counts": result.get("issue_counts", {}),
                    "profile": (result.get("dataset") or {}).get("profile"),
                },
                ensure_ascii=False,
            )
        )
        if result.get("status") != "success":
            return 1
        return 0 if result.get("verdict") == "pass" else 2
    except Exception as exc:
        logger.error("运行BioToolEval验收器CLI失败: %s", exc, exc_info=True)
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
