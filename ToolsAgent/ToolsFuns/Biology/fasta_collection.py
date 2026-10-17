# -*- coding: utf-8 -*-
"""Parse FASTA collections and select explicit records."""

from __future__ import annotations

from typing import Any

from app.core.logger import get_logger
try:
    from .fasta_common import error_result, read_protein_fasta_records
except ImportError:
    from fasta_common import error_result, read_protein_fasta_records


logger = get_logger("FASTACollection", "fasta_collection.log")


def parse_multi_protein_fasta(file_path: Any) -> dict[str, Any]:
    """Read at least two FASTA records into a named collection."""
    try:
        parsed = read_protein_fasta_records(file_path)
        if parsed.get("status") != "success":
            return parsed
        if parsed["record_count"] < 2:
            return error_result(
                "多记录Protein FASTA工具要求文件至少包含两条记录",
                "MULTI_FASTA_RECORD_COUNT_MISMATCH",
                details={"record_count": parsed["record_count"]},
                suggestion="单记录文件请使用parse_protein_fasta。",
            )

        collection = {
            "schema_version": "protein-sequence-collection/v1",
            "source_file_name": parsed["file_name"],
            "record_count": parsed["record_count"],
            "record_ids": [item["record_id"] for item in parsed["records"]],
            "records": {
                item["record_id"]: {
                    "description": item["description"],
                    "sequence": item["sequence"],
                    "sequence_length": item["sequence_length"],
                }
                for item in parsed["records"]
            },
        }
        logger.info(
            "多记录Protein FASTA解析完成 | file=%s | records=%s",
            parsed["file_name"],
            parsed["record_count"],
        )
        return {
            "status": "success",
            "file_name": parsed["file_name"],
            "record_count": parsed["record_count"],
            "record_ids": list(collection["record_ids"]),
            "collection": collection,
            "method": "Biopython SeqIO.parse(format='fasta')",
        }
    except Exception as exc:
        logger.error("解析多记录Protein FASTA失败: %s", exc, exc_info=True)
        return error_result(
            f"解析多记录Protein FASTA失败: {exc}",
            "MULTI_FASTA_PARSE_EXCEPTION",
        )


def _validate_collection(collection: Any) -> dict[str, Any]:
    """Check collection structure without implicit type conversion."""
    try:
        if not isinstance(collection, dict):
            return error_result(
                "ProteinSequenceCollection必须是字典对象",
                "INVALID_SEQUENCE_COLLECTION",
            )
        if collection.get("schema_version") != "protein-sequence-collection/v1":
            return error_result(
                "ProteinSequenceCollection版本不受支持",
                "UNSUPPORTED_SEQUENCE_COLLECTION_SCHEMA",
            )
        records = collection.get("records")
        record_ids = collection.get("record_ids")
        if (
            not isinstance(records, dict)
            or not records
            or not isinstance(record_ids, list)
            or record_ids != list(records)
        ):
            return error_result(
                "ProteinSequenceCollection记录索引不完整或不一致",
                "INVALID_SEQUENCE_COLLECTION",
            )
        return {"status": "success", "records": records, "record_ids": record_ids}
    except Exception as exc:
        logger.error("校验ProteinSequenceCollection失败: %s", exc, exc_info=True)
        return error_result(
            f"校验ProteinSequenceCollection失败: {exc}",
            "SEQUENCE_COLLECTION_VALIDATION_EXCEPTION",
        )


def select_fasta_record(collection: Any, record_id: Any) -> dict[str, Any]:
    """Select the exact requested record ID; do not use approximate matching."""
    try:
        collection_result = _validate_collection(collection)
        if collection_result.get("status") != "success":
            return collection_result
        if not isinstance(record_id, str) or not record_id.strip():
            return error_result(
                "选择FASTA记录必须提供非空record_id",
                "MISSING_FASTA_RECORD_IDENTIFIER",
                suggestion="请使用record_id=<精确ID>明确指定目标记录。",
            )
        normalized_id = record_id.strip()
        records = collection_result["records"]
        if normalized_id not in records:
            return error_result(
                "指定的FASTA record_id不存在",
                "FASTA_RECORD_IDENTIFIER_NOT_FOUND",
                details={
                    "record_id": normalized_id,
                    "available_record_ids": list(collection_result["record_ids"]),
                },
                suggestion="请从available_record_ids中选择一个精确ID。",
            )
        selected = records[normalized_id]
        sequence = selected.get("sequence") if isinstance(selected, dict) else None
        if not isinstance(sequence, str) or not sequence:
            return error_result(
                "目标FASTA记录缺少可用蛋白质序列",
                "INVALID_SEQUENCE_COLLECTION_RECORD",
                details={"record_id": normalized_id},
            )
        logger.info(
            "FASTA记录选择完成 | record=%s | length=%s",
            normalized_id,
            len(sequence),
        )
        return {
            "status": "success",
            "record_id": normalized_id,
            "description": selected.get("description", normalized_id),
            "sequence": sequence,
            "sequence_length": len(sequence),
            "available_record_ids": list(collection_result["record_ids"]),
            "selection_method": "exact_record_id",
        }
    except Exception as exc:
        logger.error("选择FASTA记录失败: %s", exc, exc_info=True)
        return error_result(
            f"选择FASTA记录失败: {exc}",
            "FASTA_RECORD_SELECTION_EXCEPTION",
        )


def validate_sequence_collection(collection: Any) -> dict[str, Any]:
    """Validate a sequence collection without changing it."""
    return _validate_collection(collection)
