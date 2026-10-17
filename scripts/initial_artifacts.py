# -*- coding: utf-8 -*-
"""Build initial artifacts from a request and its files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Bio import SeqIO

from app.core.config import Config
from app.core.logger import get_logger
from app.tools.fasta_input_parser import extract_fasta_record_identifiers
from app.tools.tool_runner import (
    extract_pdb_options,
    extract_sequence_candidate,
)
from scripts.type_checker import (
    AMINO_ACID_SEQUENCE,
    FASTA_RECORD_IDENTIFIER,
    FASTA_RECORD_IDENTIFIER_PAIR,
    MULTI_PROTEIN_FASTA_FILE,
    PDB_CHAIN_IDENTIFIER,
    PDB_STRUCTURE_FILE,
    SINGLE_PROTEIN_FASTA_FILE,
)


logger = get_logger("InitialArtifacts", "initial_artifacts.log")

_PDB_SUFFIXES = {".pdb", ".ent"}
_FASTA_SUFFIXES = {".fasta", ".fa", ".faa"}
_VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _error_result(
    message: str,
    code: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        issue = {"code": code, "message": message}
        if isinstance(details, dict):
            issue.update(details)
        return {
            "status": "error",
            "message": message,
            "artifacts": {},
            "available_input_types": [],
            "safe_summary": [],
            "issues": [issue],
            "warnings": [],
        }
    except Exception as exc:
        logger.error("构造初始Artifact错误结构失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": str(exc),
            "artifacts": {},
            "available_input_types": [],
            "safe_summary": [],
            "issues": [],
            "warnings": [],
        }


def _collect_typed_files(file_path_list: Any) -> dict[str, Any]:
    try:
        if file_path_list is None:
            candidates: list[Any] = []
        elif isinstance(file_path_list, (list, tuple)):
            candidates = list(file_path_list)
        else:
            return _error_result(
                "file_path_list必须是文件路径列表",
                "INVALID_FILE_LIST",
            )

        typed_files: dict[str, list[str]] = {
            PDB_STRUCTURE_FILE: [],
            SINGLE_PROTEIN_FASTA_FILE: [],
            MULTI_PROTEIN_FASTA_FILE: [],
        }
        for raw_path in candidates:
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            path = Path(raw_path.strip()).expanduser()
            suffix = path.suffix.lower()
            if suffix not in _PDB_SUFFIXES | _FASTA_SUFFIXES:
                continue
            if not path.is_file():
                return _error_result(
                    "上传列表包含不存在或不可读取的受支持生物文件",
                    "INITIAL_FILE_NOT_FOUND",
                    details={"file_name": path.name, "suffix": suffix},
                )
            type_name = (
                PDB_STRUCTURE_FILE
                if suffix in _PDB_SUFFIXES
                else SINGLE_PROTEIN_FASTA_FILE
            )
            resolved = str(path.resolve())
            if resolved not in typed_files[type_name]:
                typed_files[type_name].append(resolved)

        for type_name, paths in typed_files.items():
            if len(paths) > 1:
                return _error_result(
                    "同一请求中存在多个同类型输入文件，系统拒绝自动猜测数据来源",
                    "AMBIGUOUS_INITIAL_FILE",
                    details={
                        "type_name": type_name,
                        "candidate_count": len(paths),
                        "file_names": [Path(path).name for path in paths],
                    },
                )
        fasta_paths = typed_files[SINGLE_PROTEIN_FASTA_FILE]
        fasta_inspection: dict[str, Any] | None = None
        if len(fasta_paths) == 1:
            fasta_inspection = _inspect_protein_fasta(fasta_paths[0])
            if fasta_inspection.get("status") != "success":
                return fasta_inspection
            if fasta_inspection["record_count"] >= 2:
                typed_files[MULTI_PROTEIN_FASTA_FILE] = list(fasta_paths)
                typed_files[SINGLE_PROTEIN_FASTA_FILE] = []

        return {
            "status": "success",
            "typed_files": typed_files,
            "fasta_inspection": fasta_inspection,
        }
    except Exception as exc:
        logger.error("收集初始类型化文件失败: %s", exc, exc_info=True)
        return _error_result(
            f"收集初始类型化文件失败: {exc}",
            "INITIAL_FILE_COLLECTION_EXCEPTION",
        )


def _safe_artifact_metadata(type_name: str, value: Any) -> dict[str, Any]:
    try:
        metadata = {
            "type_name": type_name,
            "value_kind": type(value).__name__,
        }
        if type_name == AMINO_ACID_SEQUENCE and isinstance(value, str):
            metadata["value_size"] = len(value)
        elif type_name in {
            PDB_STRUCTURE_FILE,
            SINGLE_PROTEIN_FASTA_FILE,
            MULTI_PROTEIN_FASTA_FILE,
        } and isinstance(value, str):
            path = Path(value)
            metadata["file_name"] = path.name
            metadata["suffix"] = path.suffix.lower()
        elif type_name == PDB_CHAIN_IDENTIFIER and isinstance(value, str):
            metadata["value_size"] = len(value)
        elif type_name == FASTA_RECORD_IDENTIFIER and isinstance(value, str):
            metadata["value_size"] = len(value)
        elif (
            type_name == FASTA_RECORD_IDENTIFIER_PAIR
            and isinstance(value, (list, tuple))
        ):
            metadata["value_size"] = len(value)
        return metadata
    except Exception as exc:
        logger.error("生成初始Artifact安全摘要失败: %s", exc, exc_info=True)
        return {
            "type_name": type_name,
            "status": "error",
            "message": str(exc),
        }


def _validate_amino_acid_sequence(sequence: Any) -> dict[str, Any]:
    """Validate sequence content without returning the raw sequence."""
    try:
        if not isinstance(sequence, str) or not sequence.strip():
            return _error_result(
                "氨基酸序列不能为空",
                "EMPTY_AMINO_ACID_SEQUENCE",
            )
        normalized = "".join(sequence.split()).upper()
        if not normalized:
            return _error_result(
                "氨基酸序列不能为空",
                "EMPTY_AMINO_ACID_SEQUENCE",
            )
        invalid = sorted(set(normalized) - _VALID_AMINO_ACIDS)
        if invalid:
            suggestion = (
                "请仅使用20种标准氨基酸单字母代码"
                "ACDEFGHIKLMNPQRSTVWY，并删除或替换列出的非法字符。"
            )
            return _error_result(
                "氨基酸序列包含非法或非标准字符",
                "INVALID_AMINO_ACID_SEQUENCE_CONTENT",
                details={
                    "invalid_characters": invalid,
                    "sequence_length": len(normalized),
                    "suggestion": suggestion,
                },
            )
        return {
            "status": "success",
            "normalized_sequence": normalized,
            "sequence_length": len(normalized),
        }
    except Exception as exc:
        logger.error("校验氨基酸序列内容失败: %s", exc, exc_info=True)
        return _error_result(
            f"校验氨基酸序列内容失败: {exc}",
            "AMINO_ACID_SEQUENCE_VALIDATION_EXCEPTION",
        )


def _validate_single_protein_fasta(file_path: Any) -> dict[str, Any]:
    try:
        inspection = _inspect_protein_fasta(file_path)
        if inspection.get("status") != "success":
            return inspection
        if inspection["record_count"] != 1:
            return _error_result(
                "Protein FASTA必须且只能包含一条蛋白质记录",
                "FASTA_RECORD_COUNT_MISMATCH",
                details={
                    "file_name": inspection["file_name"],
                    "record_count": inspection["record_count"],
                },
            )
        return {
            "status": "success",
            "file_name": inspection["file_name"],
            "record_count": 1,
            "sequence_length": inspection["sequence_lengths"][0],
        }
    except Exception as exc:
        logger.error("校验单记录Protein FASTA失败: %s", exc, exc_info=True)
        return _error_result(
            f"校验单记录Protein FASTA失败: {exc}",
            "FASTA_CONTENT_VALIDATION_EXCEPTION",
        )


def _inspect_protein_fasta(file_path: Any) -> dict[str, Any]:
    """Check records, unique IDs and the configured record limit."""
    try:
        path = Path(str(file_path or "")).expanduser()
        if not path.is_file():
            return _error_result(
                "Protein FASTA文件不存在或不可读取",
                "INITIAL_FILE_NOT_FOUND",
                details={"file_name": path.name},
            )
        records = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                records.append(record)
                if len(records) > Config().MAX_MULTI_FASTA_RECORDS:
                    return _error_result(
                        "Protein FASTA记录数超过配置上限",
                        "FASTA_RECORD_LIMIT_EXCEEDED",
                        details={
                            "file_name": path.name,
                            "max_record_count": Config().MAX_MULTI_FASTA_RECORDS,
                        },
                    )
        if not records:
            return _error_result(
                "Protein FASTA至少需要一条蛋白质记录",
                "FASTA_RECORD_COUNT_MISMATCH",
                details={"file_name": path.name, "record_count": 0},
            )

        record_ids: list[str] = []
        sequence_lengths: list[int] = []
        for record_index, record in enumerate(records):
            record_id = str(record.id or "").strip()
            if not record_id or record_id == "<unknown id>":
                return _error_result(
                    "Protein FASTA记录ID不能为空",
                    "EMPTY_FASTA_RECORD_ID",
                    details={"file_name": path.name, "record_index": record_index},
                )
            if record_id in record_ids:
                return _error_result(
                    "Protein FASTA记录ID必须唯一",
                    "DUPLICATE_FASTA_RECORD_ID",
                    details={"file_name": path.name, "record_id": record_id},
                )
            sequence_result = _validate_amino_acid_sequence(str(record.seq))
            if sequence_result.get("status") != "success":
                issue = (sequence_result.get("issues") or [{}])[0]
                return _error_result(
                    "Protein FASTA包含非法蛋白质序列内容",
                    "INVALID_FASTA_SEQUENCE_CONTENT",
                    details={
                        "file_name": path.name,
                        "record_id": record_id,
                        "sequence_issue_code": issue.get("code"),
                        "invalid_characters": issue.get("invalid_characters", []),
                        "suggestion": issue.get("suggestion"),
                    },
                )
            record_ids.append(record_id)
            sequence_lengths.append(sequence_result["sequence_length"])

        return {
            "status": "success",
            "file_name": path.name,
            "record_count": len(records),
            "record_ids": record_ids,
            "sequence_lengths": sequence_lengths,
        }
    except UnicodeDecodeError:
        return _error_result(
            "Protein FASTA必须使用UTF-8文本编码",
            "INVALID_FASTA_ENCODING",
        )
    except Exception as exc:
        logger.error("检查Protein FASTA内容失败: %s", exc, exc_info=True)
        return _error_result(
            f"检查Protein FASTA内容失败: {exc}",
            "FASTA_CONTENT_VALIDATION_EXCEPTION",
        )


def build_initial_artifacts(
    question: Any,
    file_path_list: Any,
) -> dict[str, Any]:
    """Build request-local artifacts from the question and uploaded files."""
    try:
        if not isinstance(question, str):
            return _error_result(
                "用户问题必须是字符串",
                "INVALID_QUESTION",
            )

        file_result = _collect_typed_files(file_path_list)
        if file_result.get("status") != "success":
            return file_result

        artifacts: dict[str, Any] = {}
        warnings: list[dict[str, Any]] = []
        typed_files = file_result.get("typed_files") or {}
        fasta_inspection = file_result.get("fasta_inspection")

        record_id_result = extract_fasta_record_identifiers(question)
        if record_id_result.get("status") != "success":
            return _error_result(
                record_id_result.get("message", "解析FASTA record_id失败"),
                record_id_result.get(
                    "code", "FASTA_RECORD_IDENTIFIER_PARSE_EXCEPTION"
                ),
            )
        explicit_record_ids = list(record_id_result.get("record_ids") or [])

        sequence_candidate = extract_sequence_candidate(question)
        if (
            sequence_candidate is not None
            and sequence_candidate.casefold()
            not in {value.casefold() for value in explicit_record_ids}
        ):
            sequence_result = _validate_amino_acid_sequence(
                sequence_candidate
            )
            if sequence_result.get("status") != "success":
                return sequence_result
            artifacts[AMINO_ACID_SEQUENCE] = sequence_result[
                "normalized_sequence"
            ]

        for type_name in (
            PDB_STRUCTURE_FILE,
            SINGLE_PROTEIN_FASTA_FILE,
            MULTI_PROTEIN_FASTA_FILE,
        ):
            paths = typed_files.get(type_name) or []
            if len(paths) == 1:
                if type_name == SINGLE_PROTEIN_FASTA_FILE:
                    fasta_result = _validate_single_protein_fasta(paths[0])
                    if fasta_result.get("status") != "success":
                        return fasta_result
                artifacts[type_name] = paths[0]

        if MULTI_PROTEIN_FASTA_FILE in artifacts:
            available_record_ids = (
                list(fasta_inspection.get("record_ids") or [])
                if isinstance(fasta_inspection, dict)
                else []
            )
            missing_record_ids = [
                record_id
                for record_id in explicit_record_ids
                if record_id not in available_record_ids
            ]
            if missing_record_ids:
                return _error_result(
                    "问题中指定的FASTA record_id不在上传文件中",
                    "FASTA_RECORD_IDENTIFIER_NOT_FOUND",
                    details={
                        "missing_record_ids": missing_record_ids,
                        "available_record_ids": available_record_ids,
                        "suggestion": "请从available_record_ids中使用精确ID。",
                    },
                )
            if record_id_result.get("mode") == "single":
                artifacts[FASTA_RECORD_IDENTIFIER] = explicit_record_ids[0]
            elif record_id_result.get("mode") == "pair":
                artifacts[FASTA_RECORD_IDENTIFIER_PAIR] = explicit_record_ids
            else:
                warnings.append(
                    {
                        "code": "MULTI_FASTA_WITHOUT_RECORD_IDENTIFIER",
                        "message": (
                            "已注册多记录FASTA文件；需要选择或比对记录时，"
                            "请显式提供record_id或record_id pair"
                        ),
                    }
                )
        elif explicit_record_ids:
            warnings.append(
                {
                    "code": "FASTA_RECORD_IDENTIFIER_WITHOUT_MULTI_FASTA",
                    "message": "问题中包含FASTA record_id，但没有多记录FASTA输入",
                }
            )

        pdb_options = extract_pdb_options(question)
        chain_id = pdb_options.get("chain_id")
        if isinstance(chain_id, str) and chain_id.strip():
            if PDB_STRUCTURE_FILE in artifacts:
                artifacts[PDB_CHAIN_IDENTIFIER] = chain_id.strip()
            else:
                warnings.append(
                    {
                        "code": "CHAIN_ID_WITHOUT_PDB",
                        "message": "问题中包含链ID，但没有PDB输入文件，因此未注册链ID对象",
                    }
                )

        if not artifacts:
            return _error_result(
                "未从问题或上传文件中识别到受支持的初始生物数据对象",
                "NO_INITIAL_ARTIFACT",
            )

        safe_summary = [
            _safe_artifact_metadata(type_name, value)
            for type_name, value in artifacts.items()
        ]
        logger.info(
            "初始Artifact构建完成 | types=%s | warnings=%s",
            sorted(artifacts),
            [item.get("code") for item in warnings],
        )
        return {
            "status": "success",
            "artifacts": artifacts,
            "available_input_types": list(artifacts),
            "safe_summary": safe_summary,
            "issues": [],
            "warnings": warnings,
        }
    except Exception as exc:
        logger.error("构建初始Artifact失败: %s", exc, exc_info=True)
        return _error_result(
            f"构建初始Artifact失败: {exc}",
            "INITIAL_ARTIFACT_EXCEPTION",
        )
