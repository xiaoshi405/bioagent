# -*- coding: utf-8 -*-
"""Execute structured plans using the artifact pool."""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Awaitable, Callable

from app.core.logger import get_logger
from scripts.artifact_pool import ArtifactPool, INITIAL_INPUT_SOURCE
from scripts.type_checker import get_tool_signatures


logger = get_logger("TypedExecutor", "typed_executor.log")

ToolRunner = Callable[..., Awaitable[Any]]


def _runtime_error(
    message: str,
    code: str,
    *,
    error_type: str,
    failed_step: dict[str, Any] | None = None,
    executed_steps: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    artifact_pool: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        return {
            "status": "error",
            "message": message,
            "completed": False,
            "error_type": error_type,
            "issues": [{"code": code, "message": message}],
            "failed_step": copy.deepcopy(failed_step),
            "executed_steps": copy.deepcopy(executed_steps or []),
            "tool_results": copy.deepcopy(tool_results or []),
            "artifact_pool": copy.deepcopy(artifact_pool or {}),
        }
    except Exception as exc:
        logger.error("构造类型化执行错误结构失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": str(exc),
            "completed": False,
            "error_type": "executor_exception",
            "issues": [],
        }


def _normalize_runner_result(result: Any) -> dict[str, Any]:
    try:
        if isinstance(result, dict):
            return copy.deepcopy(result)
        if isinstance(result, str) and result.strip():
            parsed = json.loads(result)
            return parsed if isinstance(parsed, dict) else {}
        return {}
    except Exception as exc:
        logger.error("解析Runner结果失败: %s", exc, exc_info=True)
        return {}


def validate_runtime_plan(structured_plan: Any) -> dict[str, Any]:
    """Check step order and registry contracts before execution."""
    try:
        if not isinstance(structured_plan, list) or not structured_plan:
            return _runtime_error(
                "结构化计划必须是非空步骤列表",
                "INVALID_STRUCTURED_PLAN",
                error_type="plan_failure",
            )
        signatures = get_tool_signatures()
        if signatures.get("status") != "success":
            return _runtime_error(
                signatures.get("message", "无法读取工具签名"),
                "SIGNATURE_REGISTRY_ERROR",
                error_type="plan_failure",
            )
        tool_signatures = signatures.get("signatures") or {}
        known_step_ids: set[str] = set()

        for index, step in enumerate(structured_plan):
            if not isinstance(step, dict):
                return _runtime_error(
                    "结构化计划包含非字典步骤",
                    "INVALID_PLAN_STEP",
                    error_type="plan_failure",
                )
            expected_step_id = f"step_{index + 1}"
            if (
                step.get("step_id") != expected_step_id
                or step.get("step_index") != index
            ):
                return _runtime_error(
                    "Plan Step编号必须连续且与列表顺序一致",
                    "INVALID_STEP_ORDER",
                    error_type="plan_failure",
                    failed_step=step,
                )
            tool_name = step.get("tool_name")
            signature = tool_signatures.get(tool_name)
            if not isinstance(signature, dict):
                return _runtime_error(
                    "Plan Step包含未知或内部工具",
                    "UNKNOWN_TOOL",
                    error_type="plan_failure",
                    failed_step=step,
                )

            input_types = step.get("input_type")
            output_types = step.get("output_type")
            input_from = step.get("input_from")
            if input_types != signature.get("input_types"):
                return _runtime_error(
                    "Plan Step输入类型与唯一注册表不一致",
                    "INPUT_CONTRACT_MISMATCH",
                    error_type="plan_failure",
                    failed_step=step,
                )
            if output_types != signature.get("output_types"):
                return _runtime_error(
                    "Plan Step输出类型与唯一注册表不一致",
                    "OUTPUT_CONTRACT_MISMATCH",
                    error_type="plan_failure",
                    failed_step=step,
                )
            if not isinstance(input_from, dict) or set(input_from) != set(input_types):
                return _runtime_error(
                    "Plan Step没有为全部输入类型声明唯一来源",
                    "INVALID_INPUT_BINDING",
                    error_type="plan_failure",
                    failed_step=step,
                )
            for type_name in input_types:
                source_id = input_from.get(type_name)
                if source_id != INITIAL_INPUT_SOURCE and source_id not in known_step_ids:
                    return _runtime_error(
                        "Plan Step引用了不存在或尚未执行的数据来源",
                        "INVALID_SOURCE_ORDER",
                        error_type="plan_failure",
                        failed_step=step,
                    )
            known_step_ids.add(expected_step_id)

        return {
            "status": "success",
            "valid": True,
            "step_count": len(structured_plan),
        }
    except Exception as exc:
        logger.error("校验运行时结构化计划失败: %s", exc, exc_info=True)
        return _runtime_error(
            f"校验运行时结构化计划失败: {exc}",
            "RUNTIME_PLAN_VALIDATION_EXCEPTION",
            error_type="executor_exception",
        )


async def execute_structured_plan(
    structured_plan: Any,
    initial_artifacts: Any,
    *,
    runner: ToolRunner | None = None,
    tool_parameters: Any = None,
) -> dict[str, Any]:
    """Execute bound steps and record the artifact trace."""
    executed_steps: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    pool = ArtifactPool()
    started_at = time.perf_counter()

    try:
        plan_validation = validate_runtime_plan(structured_plan)
        if plan_validation.get("status") != "success":
            return plan_validation

        initial_result = pool.register_initial(initial_artifacts)
        if initial_result.get("status") != "success":
            return _runtime_error(
                initial_result.get("message", "初始Artifact注册失败"),
                (
                    initial_result.get("issues")
                    or [{"code": "INITIAL_ARTIFACT_FAILURE"}]
                )[0].get("code", "INITIAL_ARTIFACT_FAILURE"),
                error_type="artifact_failure",
                artifact_pool=pool.snapshot(),
            )

        if runner is None:
            from app.tools import run_tool

            active_runner: ToolRunner = run_tool
        elif callable(runner):
            active_runner = runner
        else:
            return _runtime_error(
                "工具Runner必须是可调用异步函数",
                "INVALID_RUNNER",
                error_type="executor_failure",
                artifact_pool=pool.snapshot(),
            )

        for step in structured_plan:
            step_started_at = time.perf_counter()
            input_result = pool.resolve_step_inputs(step)
            if input_result.get("status") != "success":
                issue = (
                    input_result.get("issues")
                    or [{"code": "STEP_INPUT_FAILURE"}]
                )[0]
                return _runtime_error(
                    input_result.get("message", "步骤输入对象解析失败"),
                    issue.get("code", "STEP_INPUT_FAILURE"),
                    error_type="artifact_failure",
                    failed_step=step,
                    executed_steps=executed_steps,
                    tool_results=tool_results,
                    artifact_pool=pool.snapshot(),
                )

            logger.info(
                "类型化步骤开始 | step=%s | tool=%s | input_artifacts=%s",
                step.get("step_id"),
                step.get("tool_name"),
                sorted(input_result["artifact_ids"].values()),
            )
            try:
                runner_arguments = {
                    "tool_name": step["tool_name"],
                    "tool_args": "",
                    "file_path_list": [],
                    "typed_inputs": input_result["values"],
                }
                has_parameter_context = bool(
                    tool_parameters is not None
                    and (
                        not isinstance(tool_parameters, dict)
                        or tool_parameters.get("status") == "error"
                        or tool_parameters.get("by_api")
                        or tool_parameters.get("errors_by_api")
                    )
                )
                if has_parameter_context:
                    runner_arguments["tool_parameters"] = copy.deepcopy(
                        tool_parameters
                    )
                raw_result = await active_runner(**runner_arguments)
            except Exception as exc:
                logger.error(
                    "类型化工具调用异常 | step=%s | tool=%s | error=%s",
                    step.get("step_id"),
                    step.get("tool_name"),
                    exc,
                    exc_info=True,
                )
                return _runtime_error(
                    f"工具Runner调用异常: {exc}",
                    "TOOL_RUNNER_EXCEPTION",
                    error_type="tool_failure",
                    failed_step=step,
                    executed_steps=executed_steps,
                    tool_results=tool_results,
                    artifact_pool=pool.snapshot(),
                )

            normalized_result = _normalize_runner_result(raw_result)
            if not normalized_result:
                return _runtime_error(
                    "工具Runner返回的结果不是有效JSON对象",
                    "INVALID_TOOL_RESULT",
                    error_type="tool_failure",
                    failed_step=step,
                    executed_steps=executed_steps,
                    tool_results=tool_results,
                    artifact_pool=pool.snapshot(),
                )
            tool_results.append(
                {
                    "step_id": step["step_id"],
                    "tool_name": step["tool_name"],
                    "result": normalized_result,
                }
            )
            if normalized_result.get("status") != "success":
                logger.warning(
                    "类型化步骤失败并阻断下游 | step=%s | tool=%s",
                    step.get("step_id"),
                    step.get("tool_name"),
                )
                return _runtime_error(
                    normalized_result.get("message", "工具返回错误状态"),
                    "TOOL_EXECUTION_FAILED",
                    error_type="tool_failure",
                    failed_step=step,
                    executed_steps=executed_steps,
                    tool_results=tool_results,
                    artifact_pool=pool.snapshot(),
                )

            registration = pool.register_tool_outputs(
                step["step_id"],
                step["tool_name"],
                step["output_type"],
                normalized_result,
            )
            if registration.get("status") != "success":
                issue = (
                    registration.get("issues")
                    or [{"code": "OUTPUT_REGISTRATION_FAILURE"}]
                )[0]
                return _runtime_error(
                    registration.get("message", "工具输出Artifact注册失败"),
                    issue.get("code", "OUTPUT_REGISTRATION_FAILURE"),
                    error_type="artifact_failure",
                    failed_step=step,
                    executed_steps=executed_steps,
                    tool_results=tool_results,
                    artifact_pool=pool.snapshot(),
                )

            duration_ms = round(
                (time.perf_counter() - step_started_at) * 1000.0,
                3,
            )
            trace = {
                "step_id": step["step_id"],
                "tool_name": step["tool_name"],
                "status": "success",
                "input_artifact_ids": copy.deepcopy(
                    input_result["artifact_ids"]
                ),
                "output_artifact_ids": [
                    item["artifact_id"] for item in registration["artifacts"]
                ],
                "duration_ms": duration_ms,
                "inserted_by_repair": bool(
                    step.get("inserted_by_repair", False)
                ),
            }
            executed_steps.append(trace)
            logger.info(
                "类型化步骤完成 | step=%s | tool=%s | outputs=%s | duration_ms=%s",
                step.get("step_id"),
                step.get("tool_name"),
                trace["output_artifact_ids"],
                duration_ms,
            )

        total_duration_ms = round(
            (time.perf_counter() - started_at) * 1000.0,
            3,
        )
        return {
            "status": "success",
            "completed": True,
            "error_type": None,
            "issues": [],
            "executed_steps": executed_steps,
            "tool_results": tool_results,
            "artifact_pool": pool.snapshot(),
            "total_duration_ms": total_duration_ms,
        }
    except Exception as exc:
        logger.error("类型化结构计划执行失败: %s", exc, exc_info=True)
        return _runtime_error(
            f"类型化结构计划执行失败: {exc}",
            "TYPED_EXECUTOR_EXCEPTION",
            error_type="executor_exception",
            executed_steps=executed_steps,
            tool_results=tool_results,
            artifact_pool=pool.snapshot(),
        )
