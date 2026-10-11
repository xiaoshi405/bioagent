# -*- coding: utf-8 -*-
"""为已冻结实验的盲审队列补充原始计划与初始对象证据。

修改记录：
- 2026-08-09：新增只读回填工具；不覆盖原队列，不输出题号、实验条件、
  原始序列或完整文件路径，保持既有review_id与人工标签模板兼容。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from app.core.logger import get_logger
from scripts.evaluation_scoring import build_blind_plan_evidence
from scripts.evaluation_trace import load_experiment_traces


logger = get_logger("EnrichReviewQueue", "enrich_review_queue.log")


def _error_result(message: str, code: str) -> dict[str, Any]:
    """构造统一错误结构。"""
    try:
        return {
            "status": "error",
            "message": str(message),
            "issues": [{"code": str(code), "message": str(message)}],
        }
    except Exception as exc:
        logger.error("构造复核队列补充错误失败: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc), "issues": []}


def _read_jsonl(path: Path, key_field: str) -> dict[str, Any]:
    """读取JSONL并按唯一键建立索引。"""
    try:
        if not path.is_file():
            return _error_result(f"输入文件不存在: {path.name}", "INPUT_NOT_FOUND")
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
                    "INVALID_JSONL",
                )
            if not isinstance(record, dict):
                return _error_result(
                    f"{path.name}第{line_number}行必须是JSON对象",
                    "INVALID_JSONL_RECORD",
                )
            key = str(record.get(key_field) or "").strip()
            if not key or key in records:
                return _error_result(
                    f"{path.name}包含空键或重复键: {key_field}",
                    "INVALID_JSONL_KEY",
                )
            records[key] = record
        return {"status": "success", "records": records}
    except Exception as exc:
        logger.error("读取复核JSONL失败: %s", exc, exc_info=True)
        return _error_result(f"读取JSONL失败: {exc}", "JSONL_READ_EXCEPTION")


def _trace_execution_key(trace: Any) -> str:
    """生成与私有review map一致的实验单元键。"""
    try:
        if not isinstance(trace, dict):
            return ""
        return "|".join(
            [
                str(trace.get("question_id") or ""),
                str((trace.get("context") or {}).get("model") or ""),
                str(trace.get("condition") or ""),
                str(trace.get("repeat_index", 0)),
            ]
        )
    except Exception as exc:
        logger.error("生成复核Trace键失败: %s", exc, exc_info=True)
        return ""


def enrich_review_queue(
    queue_path: Any,
    review_map_path: Any,
    trace_path: Any,
    output_path: Any,
) -> dict[str, Any]:
    """把冻结Trace中的计划证据写入新的条件盲化队列。"""
    try:
        queue = Path(str(queue_path or "")).expanduser().resolve()
        review_map = Path(str(review_map_path or "")).expanduser().resolve()
        trace = Path(str(trace_path or "")).expanduser().resolve()
        output = Path(str(output_path or "")).expanduser().resolve()
        if output in {queue, review_map, trace}:
            return _error_result(
                "补充队列不得覆盖原队列、映射或Trace",
                "REVIEW_ENRICH_OVERWRITE_FORBIDDEN",
            )
        queue_result = _read_jsonl(queue, "review_id")
        if queue_result.get("status") != "success":
            return queue_result
        map_result = _read_jsonl(review_map, "review_id")
        if map_result.get("status") != "success":
            return map_result
        trace_result = load_experiment_traces(trace)
        if trace_result.get("status") != "success":
            return trace_result
        traces = {
            _trace_execution_key(record): record
            for record in trace_result.get("records") or []
        }

        enriched: list[dict[str, Any]] = []
        for review_id, queue_record in sorted(queue_result["records"].items()):
            mapping = map_result["records"].get(review_id)
            if not isinstance(mapping, dict):
                return _error_result(
                    f"复核记录缺少私有映射: {review_id}",
                    "REVIEW_MAPPING_NOT_FOUND",
                )
            matched_trace = traces.get(str(mapping.get("execution_key") or ""))
            if not isinstance(matched_trace, dict):
                return _error_result(
                    f"复核记录找不到冻结Trace: {review_id}",
                    "REVIEW_TRACE_NOT_FOUND",
                )
            copied = copy.deepcopy(queue_record)
            copied["plan_evidence"] = build_blind_plan_evidence(
                matched_trace,
                question_text=copied.get("question"),
            )
            # 防御性检查：输出队列不得包含会揭示实验条件的字段。
            serialized = json.dumps(copied, ensure_ascii=False)
            if any(
                forbidden in serialized
                for forbidden in (
                    '"condition"',
                    '"question_id"',
                    '"execution_key"',
                )
            ):
                return _error_result(
                    "补充队列包含禁止的去盲字段",
                    "REVIEW_BLINDING_VIOLATION",
                )
            enriched.append(copied)

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in enriched
            ),
            encoding="utf-8",
        )
        temporary.replace(output)
        logger.info(
            "盲审队列证据补充完成 | records=%s | output=%s",
            len(enriched),
            output.name,
        )
        return {
            "status": "success",
            "record_count": len(enriched),
            "output_path": str(output),
        }
    except Exception as exc:
        logger.error("补充盲审队列失败: %s", exc, exc_info=True)
        return _error_result(
            f"补充盲审队列失败: {exc}",
            "REVIEW_ENRICH_EXCEPTION",
        )


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    try:
        parser = argparse.ArgumentParser(
            description="为冻结BioToolEval盲审队列补充原始计划证据"
        )
        parser.add_argument("--queue", required=True)
        parser.add_argument("--review-map", required=True)
        parser.add_argument("--trace", required=True)
        parser.add_argument("--output", required=True)
        return parser
    except Exception as exc:
        logger.error("构造复核队列补充CLI失败: %s", exc, exc_info=True)
        raise RuntimeError("无法构造复核队列补充CLI") from exc


def main(argv: list[str] | None = None) -> int:
    """执行命令并只输出不含题面和答案的摘要。"""
    try:
        args = _build_parser().parse_args(argv)
        result = enrich_review_queue(
            args.queue,
            args.review_map,
            args.trace,
            args.output,
        )
        print(
            json.dumps(
                {
                    "status": result.get("status"),
                    "record_count": result.get("record_count"),
                    "output_path": result.get("output_path"),
                    "message": result.get("message"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result.get("status") == "success" else 1
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("复核队列补充CLI失败: %s", exc, exc_info=True)
        print(json.dumps(_error_result(str(exc), "CLI_EXCEPTION"), ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
