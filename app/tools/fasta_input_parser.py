# -*- coding: utf-8 -*-
"""
FASTA record_id自然语言输入的保守定位器。

修改记录：
- 2026-08-14：新增单记录ID与记录ID pair的显式语义解析；只识别带
  record/记录标签的形式，不把普通英文、PDB ID或chain ID猜成记录ID。
- 2026-08-14：收紧中文“记录”模式，排除“多记录FASTA”等数据格式描述，
  只有记录ID标签或“记录为/是”结构才能生成record ID Artifact。
"""

from __future__ import annotations

import re
from typing import Any

from app.core.logger import get_logger


logger = get_logger("FASTAInputParser", "fasta_input_parser.log")

_RECORD_ID_TOKEN = r"[A-Za-z0-9][A-Za-z0-9_.|:-]{0,255}"


def extract_fasta_record_identifiers(text: Any) -> dict[str, Any]:
    """定位显式FASTA记录ID；找不到时返回mode=none而不猜测。"""
    try:
        if not isinstance(text, str) or not text.strip():
            return {"status": "success", "mode": "none", "record_ids": []}
        normalized = text.strip()
        pair_patterns = (
            rf"(?:record[_\s-]*ids?|记录(?:\s*ID)?列表)\s*"
            rf"(?:为|是|=|:|：)\s*[\[(]?\s*[`'\"]?({_RECORD_ID_TOKEN})"
            rf"[`'\"]?\s*(?:,|，|和|与|and\b|vs\.?\b|versus\b|/)\s*"
            rf"[`'\"]?({_RECORD_ID_TOKEN})",
            rf"(?:\brecords?\b|(?<!多)记录)\s+[`'\"]?({_RECORD_ID_TOKEN})[`'\"]?\s*"
            rf"(?:,|，|和|与|and\b|vs\.?\b|versus\b|/)\s*"
            rf"(?:records?|记录)?\s*[`'\"]?({_RECORD_ID_TOKEN})",
        )
        for pattern in pair_patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                record_ids = [match.group(1), match.group(2)]
                if record_ids[0] != record_ids[1]:
                    return {
                        "status": "success",
                        "mode": "pair",
                        "record_ids": record_ids,
                    }

        single_patterns = (
            rf"(?:record[_\s-]*id|记录\s*ID)\s*(?:为|是|=|:|：)\s*"
            rf"[`'\"]?({_RECORD_ID_TOKEN})",
            rf"\brecord\b\s+[`'\"]?({_RECORD_ID_TOKEN})",
            rf"(?<!多)记录\s+(?:为|是)\s*[`'\"]?({_RECORD_ID_TOKEN})",
        )
        for pattern in single_patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                return {
                    "status": "success",
                    "mode": "single",
                    "record_ids": [match.group(1)],
                }
        return {"status": "success", "mode": "none", "record_ids": []}
    except Exception as exc:
        logger.error("定位FASTA record_id失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "code": "FASTA_RECORD_IDENTIFIER_PARSE_EXCEPTION",
            "message": f"定位FASTA record_id失败: {exc}",
            "mode": "none",
            "record_ids": [],
        }
