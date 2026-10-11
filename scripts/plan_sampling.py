# -*- coding: utf-8 -*-
"""
BioToolEval 原始计划采样、冻结与私有账本工具。

修改记录：
- 2026-08-15：新增计划尝试分类、确定性 sample ID、冻结计划校验与 JSONL
  账本；公开 Trace 只保存哈希和安全摘要，原始模型响应仅进入本地私有账本。

设计边界：
1. 网络/基础设施失败可以按配置重试，但内容计划一旦返回便立即冻结；
2. 空计划、无法解析和未知工具都属于模型计划结果，不得通过重新规划静默覆盖；
3. 本模块不调用 LLM、不执行工具，也不修改任何冻结实验 Trace。
"""

from __future__ import annotations

import copy
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any


_CONTENT_OUTCOMES = {
    "captured",
    "unparseable_plan",
    "unknown_tool_plan",
    "request_guard_reject",
}
_INFRASTRUCTURE_OUTCOMES = {
    "network_error",
    "network_timeout",
    "planner_exception",
    "retrieval_failure",
}


def sha256_text(value: Any) -> str:
    """返回 UTF-8 文本的 SHA-256；异常输入按空字符串处理。"""
    try:
        return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(b"").hexdigest()


def classify_planning_exception(exc: BaseException) -> str:
    """把外部规划异常分成网络超时、网络错误和其他规划异常。"""
    try:
        text = f"{type(exc).__name__} {exc}".lower()
        if "timeout" in text or "超时" in text:
            return "network_timeout"
        if any(
            token in text
            for token in (
                "connecterror",
                "connection",
                "network",
                "网络",
                "连接",
            )
        ):
            return "network_error"
        return "planner_exception"
    except Exception:
        return "planner_exception"


async def collect_first_content_response(
    *,
    planner_callable: Any,
    parser_callable: Any,
    unknown_tool_callable: Any,
    max_network_attempts: int,
    backoff_seconds: float,
    sleep_callable: Any = asyncio.sleep,
) -> dict[str, Any]:
    """执行轻量规划重试状态机：基础设施可重试，首个内容响应立即返回。"""
    attempts: list[dict[str, Any]] = []
    try:
        max_attempts = min(5, max(1, int(max_network_attempts)))
        backoff = min(30.0, max(0.0, float(backoff_seconds)))
    except (TypeError, ValueError):
        max_attempts = 1
        backoff = 0.0

    for attempt_index in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            response = await planner_callable()
            raw_response = str(response or "")
            response_sha256 = sha256_text(raw_response)
            parsed = parser_callable(response)
            planned_chain = (
                [str(item) for item in parsed]
                if isinstance(parsed, list)
                else []
            )
            unknown_tools = list(
                unknown_tool_callable(planned_chain) or []
            )
            outcome = (
                "unparseable_plan"
                if not planned_chain
                else "unknown_tool_plan"
                if unknown_tools
                else "captured"
            )
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "attempt_kind": "content",
                    "outcome": outcome,
                    "exception_type": None,
                    "response_sha256": response_sha256,
                    "raw_response": raw_response,
                    "parsed_chain": planned_chain,
                    "unknown_tools": unknown_tools,
                    "latency_ms": max(
                        0.0,
                        (time.perf_counter() - started) * 1000.0,
                    ),
                }
            )
            return {
                "status": "success",
                "capture_outcome": outcome,
                "selected_attempt_index": attempt_index,
                "response_sha256": response_sha256,
                "planned_chain": planned_chain,
                "unknown_tools": unknown_tools,
                "attempts": attempts,
            }
        except Exception as exc:
            outcome = classify_planning_exception(exc)
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "attempt_kind": "infrastructure",
                    "outcome": outcome,
                    "exception_type": type(exc).__name__,
                    "response_sha256": None,
                    "raw_response": None,
                    "parsed_chain": [],
                    "unknown_tools": [],
                    "latency_ms": max(
                        0.0,
                        (time.perf_counter() - started) * 1000.0,
                    ),
                }
            )
            if outcome not in {"network_error", "network_timeout"}:
                break
            if attempt_index < max_attempts and backoff > 0:
                await sleep_callable(backoff)
    return {
        "status": "error",
        "code": "PLAN_CAPTURE_INFRASTRUCTURE_FAILURE",
        "message": "规划基础设施未能返回可冻结的内容响应",
        "attempts": attempts,
    }
def build_plan_sample_id(
    *,
    experiment_id: Any,
    question_id: Any,
    repeat_index: Any,
    model: Any,
    response_sha256: Any,
) -> str:
    """由实验键和首个内容响应哈希生成不可逆、可配对的计划样本 ID。"""
    try:
        payload = json.dumps(
            {
                "experiment_id": str(experiment_id or "").strip(),
                "question_id": str(question_id or "").strip(),
                "repeat_index": int(repeat_index),
                "model": str(model or "").strip(),
                "response_sha256": str(response_sha256 or "").strip(),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"plan_{sha256_text(payload)[:24]}"
    except Exception:
        return ""


def validate_frozen_plan(value: Any) -> dict[str, Any]:
    """防御性校验冻结计划；拒绝缺失 ID、非法状态和非字符串工具链。"""
    try:
        if not isinstance(value, dict):
            return {
                "status": "error",
                "code": "INVALID_FROZEN_PLAN",
                "message": "冻结计划必须是字典",
            }
        sample_id = str(value.get("plan_sample_id") or "").strip()
        capture_outcome = str(value.get("capture_outcome") or "").strip()
        planned_chain = value.get("planned_chain")
        sample_suffix = sample_id.removeprefix("plan_")
        if (
            len(sample_suffix) != 24
            or any(char not in "0123456789abcdef" for char in sample_suffix)
        ):
            return {
                "status": "error",
                "code": "MISSING_PLAN_SAMPLE_ID",
                "message": "冻结计划缺少合法 plan_sample_id",
            }
        if capture_outcome not in _CONTENT_OUTCOMES:
            return {
                "status": "error",
                "code": "INVALID_PLAN_CAPTURE_OUTCOME",
                "message": "冻结计划不是首个内容响应",
            }
        if not isinstance(planned_chain, list) or any(
            not isinstance(item, str) or not item.strip()
            for item in planned_chain
        ):
            return {
                "status": "error",
                "code": "INVALID_FROZEN_PLAN_CHAIN",
                "message": "冻结工具链必须是字符串列表",
            }
        attempts = value.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            return {
                "status": "error",
                "code": "MISSING_PLAN_ATTEMPTS",
                "message": "冻结计划缺少尝试轨迹",
            }
        for expected_index, attempt in enumerate(attempts, 1):
            if not isinstance(attempt, dict) or attempt.get(
                "attempt_index"
            ) != expected_index:
                return {
                    "status": "error",
                    "code": "INVALID_PLAN_ATTEMPT_ORDER",
                    "message": "计划尝试必须从1开始连续编号",
                }
            attempt_kind = attempt.get("attempt_kind")
            attempt_outcome = attempt.get("outcome")
            if (
                attempt_kind == "content"
                and attempt_outcome not in _CONTENT_OUTCOMES
            ) or (
                attempt_kind == "infrastructure"
                and attempt_outcome not in _INFRASTRUCTURE_OUTCOMES
            ) or attempt_kind not in {"content", "infrastructure"}:
                return {
                    "status": "error",
                    "code": "INVALID_PLAN_ATTEMPT_CLASSIFICATION",
                    "message": "计划尝试类型与结果分类不一致",
                }
        selected_index = value.get("selected_attempt_index")
        if not isinstance(selected_index, int) or selected_index < 1:
            return {
                "status": "error",
                "code": "INVALID_SELECTED_PLAN_ATTEMPT",
                "message": "冻结计划缺少首个内容尝试序号",
            }
        selected = [
            item
            for item in attempts
            if isinstance(item, dict)
            and item.get("attempt_index") == selected_index
        ]
        if len(selected) != 1 or selected[0].get("outcome") != capture_outcome:
            return {
                "status": "error",
                "code": "PLAN_ATTEMPT_SELECTION_MISMATCH",
                "message": "冻结计划与被选尝试不一致",
            }
        content_attempts = [
            item for item in attempts if item.get("attempt_kind") == "content"
        ]
        if len(content_attempts) != 1 or selected[0] is not attempts[-1]:
            return {
                "status": "error",
                "code": "CONTENT_PLAN_WAS_RESAMPLED",
                "message": "首个内容响应之后不得存在其他规划尝试",
            }
        if list(selected[0].get("parsed_chain") or []) != planned_chain:
            return {
                "status": "error",
                "code": "PLAN_CHAIN_SELECTION_MISMATCH",
                "message": "冻结工具链与被选内容尝试不一致",
            }
        if str(selected[0].get("response_sha256") or "") != str(
            value.get("response_sha256") or ""
        ):
            return {
                "status": "error",
                "code": "PLAN_RESPONSE_HASH_MISMATCH",
                "message": "冻结计划响应哈希与被选尝试不一致",
            }
        return {"status": "success", "frozen_plan": copy.deepcopy(value)}
    except Exception as exc:
        return {
            "status": "error",
            "code": "FROZEN_PLAN_VALIDATION_EXCEPTION",
            "message": f"校验冻结计划失败: {type(exc).__name__}",
        }


def safe_plan_trace(value: Any) -> dict[str, Any]:
    """构造可进入公开实验 Trace 的无原文计划摘要。"""
    validation = validate_frozen_plan(value)
    if validation.get("status") != "success":
        return {
            "plan_sample_id": None,
            "shared_plan_replay": False,
            "capture_outcome": "invalid",
            "selected_attempt_index": None,
            "attempt_count": 0,
            "response_sha256": None,
            "attempts": [],
        }
    plan = validation["frozen_plan"]
    safe_attempts = []
    for attempt in plan.get("attempts") or []:
        if not isinstance(attempt, dict):
            continue
        safe_attempts.append(
            {
                "attempt_index": int(attempt.get("attempt_index", 0)),
                "attempt_kind": str(attempt.get("attempt_kind") or ""),
                "outcome": str(attempt.get("outcome") or ""),
                "exception_type": (
                    str(attempt.get("exception_type"))
                    if attempt.get("exception_type")
                    else None
                ),
                "response_sha256": (
                    str(attempt.get("response_sha256"))
                    if attempt.get("response_sha256")
                    else None
                ),
                "latency_ms": max(0.0, float(attempt.get("latency_ms", 0.0))),
            }
        )
    return {
        "plan_sample_id": plan["plan_sample_id"],
        "shared_plan_replay": True,
        "capture_outcome": plan["capture_outcome"],
        "selected_attempt_index": plan["selected_attempt_index"],
        "attempt_count": len(safe_attempts),
        "response_sha256": str(plan.get("response_sha256") or ""),
        "attempts": safe_attempts,
    }


def append_plan_sample(path: Any, record: Any) -> dict[str, Any]:
    """把一次完整计划采样写入私有 JSONL 账本，不覆盖既有记录。"""
    try:
        record_status = (
            str(record.get("record_status") or "captured").strip()
            if isinstance(record, dict)
            else ""
        )
        if record_status == "captured":
            validation = validate_frozen_plan(record)
            if validation.get("status") != "success":
                return validation
        elif record_status == "capture_failed":
            if (
                not str(record.get("question_id") or "").strip()
                or not isinstance(record.get("repeat_index"), int)
                or not isinstance(record.get("attempts"), list)
            ):
                return {
                    "status": "error",
                    "code": "INVALID_FAILED_PLAN_SAMPLE",
                    "message": "失败计划样本缺少题号、轮次或尝试轨迹",
                }
        else:
            return {
                "status": "error",
                "code": "INVALID_PLAN_SAMPLE_STATUS",
                "message": "计划样本状态必须是captured或capture_failed",
            }
        output_path = Path(str(path)).expanduser()
        if output_path.suffix.lower() != ".jsonl":
            return {
                "status": "error",
                "code": "INVALID_PLAN_LEDGER_PATH",
                "message": "计划账本必须使用 .jsonl 扩展名",
            }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"status": "success", "path": str(output_path.resolve())}
    except Exception as exc:
        return {
            "status": "error",
            "code": "PLAN_LEDGER_WRITE_EXCEPTION",
            "message": f"写入计划账本失败: {type(exc).__name__}",
        }


def load_plan_samples(path: Any) -> dict[str, Any]:
    """读取私有计划账本并按 question_id/repeat_index 建立唯一索引。"""
    try:
        input_path = Path(str(path)).expanduser()
        if not input_path.exists():
            return {"status": "success", "records": {}, "record_count": 0}
        records: dict[tuple[str, int], dict[str, Any]] = {}
        failed_record_count = 0
        for line_number, raw_line in enumerate(
            input_path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if value.get("record_status") == "capture_failed":
                if (
                    not str(value.get("question_id") or "").strip()
                    or not isinstance(value.get("repeat_index"), int)
                    or not isinstance(value.get("attempts"), list)
                ):
                    return {
                        "status": "error",
                        "code": "INVALID_PLAN_LEDGER_RECORD",
                        "message": f"计划账本第 {line_number} 行无效",
                    }
                failed_record_count += 1
                continue
            validation = validate_frozen_plan(value)
            if validation.get("status") != "success":
                return {
                    "status": "error",
                    "code": "INVALID_PLAN_LEDGER_RECORD",
                    "message": f"计划账本第 {line_number} 行无效",
                }
            key = (
                str(value.get("question_id") or "").strip(),
                int(value.get("repeat_index", 0)),
            )
            if not key[0] or key in records:
                return {
                    "status": "error",
                    "code": "DUPLICATE_PLAN_SAMPLE_KEY",
                    "message": "计划账本存在空题号或重复题目/重复轮次键",
                }
            records[key] = value
        return {
            "status": "success",
            "records": records,
            "record_count": len(records) + failed_record_count,
            "captured_record_count": len(records),
            "failed_record_count": failed_record_count,
        }
    except Exception as exc:
        return {
            "status": "error",
            "code": "PLAN_LEDGER_READ_EXCEPTION",
            "message": f"读取计划账本失败: {type(exc).__name__}",
        }


def is_infrastructure_outcome(value: Any) -> bool:
    """判断尝试结果是否属于可单独统计的基础设施失败。"""
    try:
        return str(value or "").strip() in _INFRASTRUCTURE_OUTCOMES
    except Exception:
        return False
