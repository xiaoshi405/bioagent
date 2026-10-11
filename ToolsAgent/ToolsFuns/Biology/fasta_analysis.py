# -*- coding: utf-8 -*-
"""
单记录Protein FASTA本地解析工具。

修改记录：
- 2026-07-24：新增单记录Protein FASTA文件解析；严格拒绝多记录、
  非标准氨基酸、项目目录外路径和超限文件。
- 2026-08-14：复用fasta_common统一安全边界和错误结构，保留原成功输出
  契约，避免单记录与多记录工具产生两套校验语义。
"""

from __future__ import annotations

from typing import Any

from app.core.config import Config
from app.core.logger import get_logger

try:
    from .fasta_common import (
        FASTA_SUFFIXES,
        PROJECT_ROOT,
        VALID_AMINO_ACIDS,
        error_result,
        read_protein_fasta_records,
    )
except ImportError:
    from fasta_common import (
        FASTA_SUFFIXES,
        PROJECT_ROOT,
        VALID_AMINO_ACIDS,
        error_result,
        read_protein_fasta_records,
    )


logger = get_logger("FASTAAnalysis", "fasta_analysis.log")
MAX_FASTA_FILE_BYTES = Config().MAX_FASTA_UPLOAD_MB * 1024 * 1024


def parse_protein_fasta(file_path: Any) -> dict[str, Any]:
    """解析恰好一条Protein FASTA记录，返回标准氨基酸序列。"""
    try:
        parsed = read_protein_fasta_records(file_path)
        if parsed.get("status") != "success":
            return parsed
        if parsed["record_count"] != 1:
            return error_result(
                "当前仅支持单记录Protein FASTA；多记录文件请使用"
                "parse_multi_protein_fasta并显式选择record_id",
                "FASTA_RECORD_COUNT_MISMATCH",
                details={"record_count": parsed["record_count"]},
                suggestion=(
                    "请调用parse_multi_protein_fasta；需要单条序列时再使用"
                    "select_fasta_record和精确record_id。"
                ),
            )

        record = parsed["records"][0]
        logger.info(
            "Protein FASTA解析完成 | file=%s | record=%s | length=%s",
            parsed["file_name"],
            record["record_id"],
            record["sequence_length"],
        )
        return {
            "status": "success",
            "file_name": parsed["file_name"],
            "record_id": record["record_id"],
            "description": record["description"],
            "sequence": record["sequence"],
            "sequence_length": record["sequence_length"],
            "method": "Biopython SeqIO.parse(format='fasta')",
        }
    except Exception as exc:
        logger.error("解析Protein FASTA失败: %s", exc, exc_info=True)
        return error_result(
            f"解析Protein FASTA失败: {exc}",
            "SINGLE_FASTA_PARSE_EXCEPTION",
        )
