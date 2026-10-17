# -*- coding: utf-8 -*-
"""Screen sequences against the local toxicity reference database."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
from Bio.Align import PairwiseAligner

from app.core.config import Config
from app.core.logger import get_logger


logger = get_logger("ToxicityChecker", "toxicity.log")

PROTEIN_SIMILARITY_THRESHOLD = 95.0
KMER_SIZE = 4
MAX_ALIGNMENT_CANDIDATES = 50
MAX_SEQUENCE_LENGTH = 10000
MAX_ALIGNMENT_CELLS = 25_000_000
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
PROJECT_ROOT = Path(__file__).resolve().parents[3]

ToxicityResult = Dict[str, Union[bool, float, int, str, None]]
ToxinRecord = Tuple[str, str, frozenset[str]]


def _error(message: str) -> ToxicityResult:
    try:
        return {"status": "error", "message": str(message)}
    except Exception as exc:
        return {"status": "error", "message": f"构造毒性错误响应失败: {exc}"}


def _normalize_sequence(sequence: str) -> Tuple[Optional[str], Optional[ToxicityResult]]:
    """Normalize whitespace and case; accept the 20 standard amino acids."""
    try:
        if not isinstance(sequence, str):
            return None, _error("序列必须是字符串")
        normalized = "".join(sequence.split()).upper()
        if not normalized:
            return None, _error("序列不能为空")
        if len(normalized) > MAX_SEQUENCE_LENGTH:
            return None, _error(
                f"序列长度超过上限 {MAX_SEQUENCE_LENGTH}，拒绝执行高开销局部比对"
            )
        invalid = sorted(set(normalized) - VALID_AMINO_ACIDS)
        if invalid:
            return None, _error(
                f"序列包含非法或非标准氨基酸字符: {', '.join(invalid)}"
            )
        return normalized, None
    except Exception as exc:
        logger.error("毒性检测序列校验失败: %s", exc, exc_info=True)
        return None, _error(f"序列校验失败: {exc}")


def _resolve_database_path(db_path: Optional[str]) -> Optional[Path]:
    try:
        configured = str(db_path or Config().TOXIN_PROTEIN or "").strip()
        if not configured:
            return None
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
    except Exception as exc:
        logger.error("解析毒性数据库路径失败: %s", exc, exc_info=True)
        return None


def _kmers(sequence: str, size: int = KMER_SIZE) -> frozenset[str]:
    try:
        if not sequence:
            return frozenset()
        if len(sequence) < size:
            return frozenset({sequence})
        return frozenset(
            sequence[index:index + size]
            for index in range(len(sequence) - size + 1)
        )
    except Exception as exc:
        logger.error("生成 k-mer 失败: %s", exc, exc_info=True)
        raise RuntimeError(f"生成 k-mer 失败: {exc}") from exc


@lru_cache(maxsize=4)
def _load_toxin_database(
    resolved_path: str, modified_time_ns: int
) -> Tuple[ToxinRecord, ...]:
    try:
        # Include the modification time in the database cache key.
        _ = modified_time_ns
        path = Path(resolved_path)
        if not path.is_file():
            raise FileNotFoundError(f"毒性蛋白数据库不存在: {path}")

        frame = pd.read_csv(path, usecols=["Protein names", "Sequence"])
        records = []
        for name, sequence in frame.itertuples(index=False, name=None):
            if pd.isna(sequence):
                continue
            normalized = "".join(str(sequence).split()).upper()
            if normalized and not (set(normalized) - VALID_AMINO_ACIDS):
                records.append((str(name), normalized, _kmers(normalized)))

        if not records:
            raise ValueError("毒性蛋白数据库中没有有效序列")
        logger.info("毒性蛋白数据库加载完成 | 有效序列数: %s", len(records))
        return tuple(records)
    except Exception as exc:
        logger.error("加载毒性蛋白数据库失败: %s", exc, exc_info=True)
        raise RuntimeError(f"加载毒性蛋白数据库失败: {exc}") from exc


def _candidate_score(
    query: str,
    target: str,
    query_kmers: frozenset[str],
    target_kmers: frozenset[str],
) -> float:
    try:
        if not query_kmers or not target_kmers:
            return 0.0
        shared = len(query_kmers & target_kmers)
        denominator = max(1, min(len(query_kmers), len(target_kmers)))
        overlap = shared / denominator
        length_similarity = min(len(query), len(target)) / max(len(query), len(target))
        return 0.8 * overlap + 0.2 * length_similarity
    except Exception as exc:
        logger.error("计算候选 k-mer 分数失败: %s", exc, exc_info=True)
        raise RuntimeError(f"计算候选 k-mer 分数失败: {exc}") from exc


def _smith_waterman_similarity(
    query: str, target: str
) -> Tuple[float, float, float, float]:
    """Return similarity, identity and coverage for both aligned sequences."""
    try:
        aligner = PairwiseAligner()
        aligner.mode = "local"
        aligner.match_score = 1.0
        aligner.mismatch_score = 0.0
        aligner.open_gap_score = -1.0
        aligner.extend_gap_score = -1.0

        alignments = aligner.align(query, target)
        # Read only the first optimal alignment; counting ties can overflow.
        alignment = next(iter(alignments), None)
        if alignment is None:
            return 0.0, 0.0, 0.0, 0.0

        matches = 0
        aligned_length = 0
        query_blocks, target_blocks = alignment.aligned
        for query_block, target_block in zip(query_blocks, target_blocks):
            query_start, query_end = map(int, query_block)
            target_start, target_end = map(int, target_block)
            block_length = min(query_end - query_start, target_end - target_start)
            if block_length <= 0:
                continue
            query_part = query[query_start:query_start + block_length]
            target_part = target[target_start:target_start + block_length]
            matches += sum(a == b for a, b in zip(query_part, target_part))
            aligned_length += block_length

        if aligned_length <= 0:
            return 0.0, 0.0, 0.0, 0.0
        identity = matches / aligned_length * 100.0
        query_coverage = min(100.0, aligned_length / max(1, len(query)) * 100.0)
        target_coverage = min(100.0, aligned_length / max(1, len(target)) * 100.0)
        # Require coverage of both sequences to avoid short incidental matches.
        similarity = identity * min(query_coverage, target_coverage) / 100.0
        return (
            round(similarity, 2),
            round(identity, 2),
            round(query_coverage, 2),
            round(target_coverage, 2),
        )
    except Exception as exc:
        logger.error("Smith-Waterman 比对失败: %s", exc, exc_info=True)
        raise RuntimeError(f"Smith-Waterman 比对失败: {exc}") from exc


def check_protein_toxicity_bio(
    input_sequence: str,
    db_path: Optional[str] = None,
    threshold: float = PROTEIN_SIMILARITY_THRESHOLD,
) -> ToxicityResult:
    """Check for high similarity to proteins in the local toxicity database."""
    try:
        sequence, error = _normalize_sequence(input_sequence)
        if error or sequence is None:
            return error or _error("序列校验失败")
        if not 0.0 <= float(threshold) <= 100.0:
            return _error("相似度阈值必须在 0 到 100 之间")

        database_path = _resolve_database_path(db_path)
        if database_path is None:
            return _error("未配置 TOXIN_PROTEIN 毒性蛋白数据库路径")
        if not database_path.is_file():
            return _error(f"毒性蛋白数据库不存在: {database_path}")
        records = _load_toxin_database(
            str(database_path), database_path.stat().st_mtime_ns
        )

        query_kmers = _kmers(sequence)
        ranked_records = sorted(
            records,
            key=lambda record: _candidate_score(
                sequence, record[1], query_kmers, record[2]
            ),
            reverse=True,
        )
        ranked = []
        estimated_alignment_cells = 0
        for record in ranked_records:
            candidate_cells = len(sequence) * len(record[1])
            if candidate_cells > MAX_ALIGNMENT_CELLS:
                continue
            if estimated_alignment_cells + candidate_cells > MAX_ALIGNMENT_CELLS:
                continue
            ranked.append(record)
            estimated_alignment_cells += candidate_cells
            if len(ranked) >= MAX_ALIGNMENT_CANDIDATES:
                break
        if not ranked:
            return _error("序列与数据库候选组合超过局部比对计算预算")

        best_name = None
        best_similarity = 0.0
        best_identity = 0.0
        best_query_coverage = 0.0
        best_target_coverage = 0.0
        for name, toxin_sequence, _ in ranked:
            similarity, identity, query_coverage, target_coverage = _smith_waterman_similarity(
                sequence, toxin_sequence
            )
            if similarity > best_similarity:
                best_name = name
                best_similarity = similarity
                best_identity = identity
                best_query_coverage = query_coverage
                best_target_coverage = target_coverage
        high_similarity = best_similarity >= float(threshold)
        logger.info(
            "毒性相似性筛查完成 | high_similarity=%s | similarity=%.2f | database_size=%s",
            high_similarity,
            best_similarity,
            len(records),
        )
        return {
            "status": "success",
            "high_similarity_to_known_toxin": high_similarity,
            "similarity_percent": round(best_similarity, 2),
            "identity_percent": round(best_identity, 2),
            "coverage_percent": round(
                min(best_query_coverage, best_target_coverage), 2
            ),
            "query_coverage_percent": round(best_query_coverage, 2),
            "target_coverage_percent": round(best_target_coverage, 2),
            "matched_protein": best_name if high_similarity else None,
            "threshold_percent": float(threshold),
            "database_size": len(records),
            "alignment_candidates": len(ranked),
            "estimated_alignment_cells": estimated_alignment_cells,
            "method": "length-aware k-mer prefilter + Smith-Waterman local alignment",
        }
    except Exception as exc:
        logger.error("蛋白质毒性检测失败: %s", exc, exc_info=True)
        return _error(f"蛋白质毒性检测失败: {exc}")


def check_toxicity_by_type(sequences) -> bool:
    """Return true if any supplied sequence matches the toxicity reference."""
    try:
        if not sequences:
            return False
        for sequence, type_detected in sequences:
            if type_detected != "sequence":
                continue
            result = check_protein_toxicity_bio(sequence)
            if result.get("status") != "success":
                logger.error("毒性安全检查返回错误，采用 fail-closed: %s", result)
                return True
            if result.get("high_similarity_to_known_toxin"):
                return True
        return False
    except Exception as exc:
        logger.error("批量毒性检查失败: %s", exc, exc_info=True)
        # Fail closed if toxicity screening raises an error.
        return True
