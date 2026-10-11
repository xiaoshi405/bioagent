# -*- coding: utf-8 -*-
"""
Protein序列集合的显式两记录全局比对与统计摘要工具。

修改记录：
- 2026-08-14：新增集合到比对、比对到统计的两级离线工具；比对矩阵、
  gap罚分与长度上限均由Config读取，非法输入返回统一错误结构。
"""

from __future__ import annotations

from typing import Any

from Bio.Align import PairwiseAligner, substitution_matrices

from app.core.config import Config
from app.core.logger import get_logger
try:
    from .fasta_collection import validate_sequence_collection
    from .fasta_common import error_result
except ImportError:
    from fasta_collection import validate_sequence_collection
    from fasta_common import error_result


logger = get_logger("ProteinAlignment", "protein_alignment.log")


def _validate_record_pair(record_ids: Any) -> dict[str, Any]:
    """校验record_id pair恰含两个不同的显式字符串ID。"""
    try:
        if not isinstance(record_ids, (list, tuple)) or len(record_ids) != 2:
            return error_result(
                "FASTA记录对必须恰好包含两个record_id",
                "INVALID_FASTA_RECORD_IDENTIFIER_PAIR",
                suggestion="请明确提供两个不同ID，例如record_ids=[A, B]。",
            )
        normalized = [
            value.strip() if isinstance(value, str) else "" for value in record_ids
        ]
        if not all(normalized) or normalized[0] == normalized[1]:
            return error_result(
                "FASTA记录对必须包含两个不同的非空record_id",
                "INVALID_FASTA_RECORD_IDENTIFIER_PAIR",
            )
        return {"status": "success", "record_ids": normalized}
    except Exception as exc:
        logger.error("校验FASTA记录对失败: %s", exc, exc_info=True)
        return error_result(
            f"校验FASTA记录对失败: {exc}",
            "FASTA_RECORD_PAIR_VALIDATION_EXCEPTION",
        )


def align_fasta_records(collection: Any, record_ids: Any) -> dict[str, Any]:
    """对集合中两个显式记录执行配置化BLOSUM全局比对。"""
    try:
        collection_result = validate_sequence_collection(collection)
        if collection_result.get("status") != "success":
            return collection_result
        pair_result = _validate_record_pair(record_ids)
        if pair_result.get("status") != "success":
            return pair_result

        normalized_ids = pair_result["record_ids"]
        records = collection_result["records"]
        missing = [record_id for record_id in normalized_ids if record_id not in records]
        if missing:
            return error_result(
                "比对请求包含不存在的FASTA record_id",
                "FASTA_RECORD_IDENTIFIER_NOT_FOUND",
                details={
                    "missing_record_ids": missing,
                    "available_record_ids": list(collection_result["record_ids"]),
                },
                suggestion="请从available_record_ids中精确选择两个不同ID。",
            )

        first_sequence = records[normalized_ids[0]].get("sequence")
        second_sequence = records[normalized_ids[1]].get("sequence")
        if not isinstance(first_sequence, str) or not isinstance(second_sequence, str):
            return error_result(
                "序列集合中的目标记录结构无效",
                "INVALID_SEQUENCE_COLLECTION_RECORD",
            )

        config = Config()
        max_length = config.MAX_PROTEIN_ALIGNMENT_SEQUENCE_LENGTH
        over_limit = {
            record_id: len(sequence)
            for record_id, sequence in zip(
                normalized_ids, (first_sequence, second_sequence)
            )
            if len(sequence) > max_length
        }
        if over_limit:
            return error_result(
                "目标蛋白质序列超过配置的比对长度上限",
                "PROTEIN_ALIGNMENT_LENGTH_LIMIT_EXCEEDED",
                details={
                    "sequence_lengths": over_limit,
                    "max_sequence_length": max_length,
                },
            )

        aligner = PairwiseAligner(mode="global")
        matrix_name = config.PROTEIN_ALIGNMENT_MATRIX
        try:
            matrix = substitution_matrices.load(matrix_name)
        except Exception as exc:
            return error_result(
                f"无法加载配置的蛋白质替换矩阵: {exc}",
                "PROTEIN_ALIGNMENT_MATRIX_UNAVAILABLE",
                details={"matrix": matrix_name},
            )
        aligner.substitution_matrix = matrix
        aligner.open_gap_score = config.PROTEIN_ALIGNMENT_OPEN_GAP_SCORE
        aligner.extend_gap_score = config.PROTEIN_ALIGNMENT_EXTEND_GAP_SCORE

        alignments = aligner.align(first_sequence, second_sequence)
        try:
            best = alignments[0]
        except IndexError:
            return error_result(
                "全局比对没有产生可用结果",
                "EMPTY_PROTEIN_ALIGNMENT",
            )
        aligned_first = str(best[0])
        aligned_second = str(best[1])
        alignment = {
            "schema_version": "pairwise-protein-alignment/v1",
            "record_ids": normalized_ids,
            "aligned_sequences": [aligned_first, aligned_second],
            "original_lengths": [len(first_sequence), len(second_sequence)],
            "alignment_length": len(aligned_first),
            "score": round(float(best.score), 6),
            "method": "Biopython PairwiseAligner global alignment",
            "matrix": matrix_name,
            "open_gap_score": config.PROTEIN_ALIGNMENT_OPEN_GAP_SCORE,
            "extend_gap_score": config.PROTEIN_ALIGNMENT_EXTEND_GAP_SCORE,
        }
        logger.info(
            "Protein全局比对完成 | records=%s | alignment_length=%s",
            normalized_ids,
            len(aligned_first),
        )
        return {
            "status": "success",
            "record_ids": normalized_ids,
            "alignment_length": len(aligned_first),
            "score": alignment["score"],
            "alignment": alignment,
        }
    except Exception as exc:
        logger.error("Protein记录全局比对失败: %s", exc, exc_info=True)
        return error_result(
            f"Protein记录全局比对失败: {exc}",
            "PROTEIN_ALIGNMENT_EXCEPTION",
        )


def summarize_pairwise_alignment(alignment: Any) -> dict[str, Any]:
    """计算比对的identity、similarity、gap与覆盖率统计。"""
    try:
        if not isinstance(alignment, dict):
            return error_result(
                "PairwiseProteinAlignment必须是字典对象",
                "INVALID_PAIRWISE_ALIGNMENT",
            )
        if alignment.get("schema_version") != "pairwise-protein-alignment/v1":
            return error_result(
                "PairwiseProteinAlignment版本不受支持",
                "UNSUPPORTED_PAIRWISE_ALIGNMENT_SCHEMA",
            )
        aligned_sequences = alignment.get("aligned_sequences")
        if (
            not isinstance(aligned_sequences, list)
            or len(aligned_sequences) != 2
            or not all(isinstance(item, str) for item in aligned_sequences)
            or len(aligned_sequences[0]) != len(aligned_sequences[1])
            or not aligned_sequences[0]
        ):
            return error_result(
                "PairwiseProteinAlignment缺少两个等长比对序列",
                "INVALID_PAIRWISE_ALIGNMENT",
            )

        matrix_name = alignment.get("matrix")
        if not isinstance(matrix_name, str) or not matrix_name:
            return error_result(
                "PairwiseProteinAlignment缺少替换矩阵名称",
                "INVALID_PAIRWISE_ALIGNMENT",
            )
        try:
            matrix = substitution_matrices.load(matrix_name)
        except Exception as exc:
            return error_result(
                f"无法加载比对声明的蛋白质替换矩阵: {exc}",
                "PROTEIN_ALIGNMENT_MATRIX_UNAVAILABLE",
                details={"matrix": matrix_name},
            )

        first, second = aligned_sequences
        aligned_residue_pairs = 0
        matches = 0
        positive_substitutions = 0
        substitutions = 0
        gap_columns = 0
        for left, right in zip(first, second):
            if left == "-" or right == "-":
                gap_columns += 1
                continue
            aligned_residue_pairs += 1
            if left == right:
                matches += 1
            else:
                substitutions += 1
                try:
                    if float(matrix[left, right]) > 0:
                        positive_substitutions += 1
                except Exception:
                    return error_result(
                        "比对包含替换矩阵不支持的残基字符",
                        "INVALID_PAIRWISE_ALIGNMENT_RESIDUE",
                        details={"residues": sorted({left, right})},
                    )

        alignment_length = len(first)
        similarity_count = matches + positive_substitutions
        identity_percent = (
            100.0 * matches / aligned_residue_pairs if aligned_residue_pairs else 0.0
        )
        similarity_percent = (
            100.0 * similarity_count / aligned_residue_pairs
            if aligned_residue_pairs
            else 0.0
        )
        statistics = {
            "schema_version": "alignment-statistics/v1",
            "record_ids": list(alignment.get("record_ids") or []),
            "alignment_length": alignment_length,
            "aligned_residue_pairs": aligned_residue_pairs,
            "matches": matches,
            "positive_substitutions": positive_substitutions,
            "substitutions": substitutions,
            "gap_columns": gap_columns,
            "identity_percent": round(identity_percent, 4),
            "similarity_percent": round(similarity_percent, 4),
            "gap_percent": round(100.0 * gap_columns / alignment_length, 4),
            "coverage_adjusted_identity_percent": round(
                100.0 * matches / alignment_length, 4
            ),
            "identity_denominator": "aligned non-gap residue pairs",
            "similarity_rule": f"identical or {matrix_name} substitution score > 0",
        }
        logger.info(
            "Protein比对统计完成 | records=%s | identity=%s",
            statistics["record_ids"],
            statistics["identity_percent"],
        )
        return {
            "status": "success",
            "record_ids": statistics["record_ids"],
            "statistics": statistics,
        }
    except Exception as exc:
        logger.error("汇总Protein比对失败: %s", exc, exc_info=True)
        return error_result(
            f"汇总Protein比对失败: {exc}",
            "PROTEIN_ALIGNMENT_SUMMARY_EXCEPTION",
        )
