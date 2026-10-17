# -*- coding: utf-8 -*-
"""Shared FASTA file and sequence validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Bio import SeqIO

from app.core.config import Config
from app.core.logger import get_logger


logger = get_logger("FASTACommon", "fasta_common.log")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FASTA_SUFFIXES = frozenset({".fasta", ".fa", ".faa"})
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def error_result(
    message: str,
    code: str,
    *,
    details: dict[str, Any] | None = None,
    suggestion: str | None = None,
) -> dict[str, Any]:
    try:
        issue: dict[str, Any] = {"code": code, "message": str(message)}
        if isinstance(details, dict):
            issue.update(details)
        if isinstance(suggestion, str) and suggestion.strip():
            issue["suggestion"] = suggestion.strip()
        return {
            "status": "error",
            "code": code,
            "message": str(message),
            "issues": [issue],
        }
    except Exception as exc:
        logger.error("构造FASTA错误结果失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "code": "FASTA_ERROR_RESPONSE_EXCEPTION",
            "message": f"构造FASTA错误结果失败: {exc}",
            "issues": [],
        }


def validate_amino_acid_sequence(sequence: Any) -> dict[str, Any]:
    """Accept only standard amino acids; do not replace invalid characters."""
    try:
        if not isinstance(sequence, str) or not sequence.strip():
            return error_result(
                "蛋白质序列不能为空",
                "EMPTY_AMINO_ACID_SEQUENCE",
                suggestion="请为每条FASTA记录提供非空蛋白质序列。",
            )
        normalized = "".join(sequence.split()).upper()
        invalid = sorted(set(normalized) - VALID_AMINO_ACIDS)
        if invalid:
            return error_result(
                "蛋白质序列包含非法或非标准氨基酸字符",
                "INVALID_AMINO_ACID_SEQUENCE_CONTENT",
                details={
                    "invalid_characters": invalid,
                    "sequence_length": len(normalized),
                },
                suggestion=(
                    "请仅使用20种标准氨基酸单字母代码"
                    "ACDEFGHIKLMNPQRSTVWY，并删除或替换列出的非法字符。"
                ),
            )
        return {
            "status": "success",
            "sequence": normalized,
            "sequence_length": len(normalized),
        }
    except Exception as exc:
        logger.error("校验蛋白质序列失败: %s", exc, exc_info=True)
        return error_result(
            f"校验蛋白质序列失败: {exc}",
            "AMINO_ACID_SEQUENCE_VALIDATION_EXCEPTION",
        )


def resolve_fasta_path(file_path: Any) -> dict[str, Any]:
    """Check the FASTA path, project boundary, extension and size limit."""
    try:
        if not isinstance(file_path, str) or not file_path.strip():
            return error_result("FASTA文件路径不能为空", "EMPTY_FASTA_PATH")

        path = Path(file_path.strip()).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved = path.resolve()
        try:
            resolved.relative_to(PROJECT_ROOT)
        except ValueError:
            return error_result(
                "FASTA文件必须位于项目目录内",
                "FASTA_PATH_OUTSIDE_PROJECT",
            )

        if resolved.suffix.lower() not in FASTA_SUFFIXES:
            return error_result(
                "仅支持 .fasta、.fa 或 .faa 蛋白质FASTA文件",
                "UNSUPPORTED_FASTA_SUFFIX",
                details={"suffix": resolved.suffix.lower()},
            )
        if not resolved.is_file():
            return error_result(
                f"FASTA文件不存在: {resolved.name}",
                "FASTA_FILE_NOT_FOUND",
                details={"file_name": resolved.name},
            )

        file_size = resolved.stat().st_size
        if file_size <= 0:
            return error_result("FASTA文件为空", "EMPTY_FASTA_FILE")
        max_bytes = Config().MAX_FASTA_UPLOAD_MB * 1024 * 1024
        if file_size > max_bytes:
            return error_result(
                "FASTA文件超过配置的大小上限",
                "FASTA_FILE_TOO_LARGE",
                details={
                    "file_size_bytes": file_size,
                    "max_file_size_bytes": max_bytes,
                },
            )
        return {
            "status": "success",
            "path": resolved,
            "file_name": resolved.name,
            "file_size_bytes": file_size,
        }
    except Exception as exc:
        logger.error("解析FASTA文件路径失败: %s", exc, exc_info=True)
        return error_result(
            f"解析FASTA文件路径失败: {exc}",
            "FASTA_PATH_VALIDATION_EXCEPTION",
        )


def read_protein_fasta_records(file_path: Any) -> dict[str, Any]:
    """Read and validate all protein FASTA records."""
    try:
        path_result = resolve_fasta_path(file_path)
        if path_result.get("status") != "success":
            return path_result
        path = path_result["path"]

        first_nonempty = ""
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        first_nonempty = stripped
                        break
        except UnicodeDecodeError:
            return error_result(
                "FASTA文件必须使用UTF-8文本编码",
                "INVALID_FASTA_ENCODING",
            )
        if not first_nonempty:
            return error_result("FASTA文件为空", "EMPTY_FASTA_FILE")
        if not first_nonempty.startswith(">"):
            return error_result(
                "Protein FASTA第一条非空行必须以 > 开头",
                "MISSING_FASTA_HEADER",
            )
        if not first_nonempty[1:].strip():
            return error_result(
                "Protein FASTA记录头不能为空",
                "EMPTY_FASTA_HEADER",
            )

        max_records = Config().MAX_MULTI_FASTA_RECORDS
        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                for raw_record in SeqIO.parse(handle, "fasta"):
                    if len(records) >= max_records:
                        return error_result(
                            "Protein FASTA记录数超过配置上限",
                            "FASTA_RECORD_LIMIT_EXCEEDED",
                            details={"max_record_count": max_records},
                        )
                    record_id = str(raw_record.id or "").strip()
                    if not record_id or record_id == "<unknown id>":
                        return error_result(
                            "Protein FASTA记录ID不能为空",
                            "EMPTY_FASTA_RECORD_ID",
                            details={"record_index": len(records)},
                        )
                    if record_id in seen_ids:
                        return error_result(
                            "Protein FASTA记录ID必须唯一",
                            "DUPLICATE_FASTA_RECORD_ID",
                            details={"record_id": record_id},
                            suggestion="请为重复记录使用不同且稳定的record_id。",
                        )
                    sequence_result = validate_amino_acid_sequence(str(raw_record.seq))
                    if sequence_result.get("status") != "success":
                        issue = (sequence_result.get("issues") or [{}])[0]
                        return error_result(
                            "Protein FASTA包含非法蛋白质序列内容",
                            "INVALID_FASTA_SEQUENCE_CONTENT",
                            details={
                                "record_id": record_id,
                                "sequence_issue_code": issue.get("code"),
                                "invalid_characters": issue.get(
                                    "invalid_characters", []
                                ),
                            },
                            suggestion=issue.get("suggestion"),
                        )
                    seen_ids.add(record_id)
                    records.append(
                        {
                            "record_id": record_id,
                            "description": str(
                                raw_record.description or record_id
                            ),
                            "sequence": sequence_result["sequence"],
                            "sequence_length": sequence_result["sequence_length"],
                        }
                    )
        except UnicodeDecodeError:
            return error_result(
                "FASTA文件必须使用UTF-8文本编码",
                "INVALID_FASTA_ENCODING",
            )

        if not records:
            return error_result(
                "FASTA文件中没有可解析的蛋白质记录",
                "NO_FASTA_RECORDS",
            )
        return {
            "status": "success",
            "path": path,
            "file_name": path.name,
            "record_count": len(records),
            "records": records,
        }
    except Exception as exc:
        logger.error("读取Protein FASTA记录失败: %s", exc, exc_info=True)
        return error_result(
            f"读取Protein FASTA记录失败: {exc}",
            "FASTA_READ_EXCEPTION",
        )
