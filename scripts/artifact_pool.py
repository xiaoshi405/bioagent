# -*- coding: utf-8 -*-
"""
BioAgent 类型化运行时数据对象池（Artifact Pool）。

修改记录：
- 2026-07-26：新增v0.3a真实数据对象保存、按类型与来源精确取值、
  工具输出原子注册和脱敏元数据快照。

设计边界：
1. 对象池保存真实运行值，但普通日志和snapshot绝不写出序列或完整文件路径；
2. 每个(source_id, type_name)只允许一个对象，禁止执行器猜测同类型多对象；
3. 工具只有返回status=success且所有声明输出均可提取时才原子写入；
4. 输出字段绑定只读取BIOLOGY_TOOLS_DESC，不维护第二份工具清单。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from app.core.logger import get_logger
from ToolsAgent.ToolsFuns.Biology.tool_name_dict import BIOLOGY_TOOLS_DESC


logger = get_logger("ArtifactPool", "artifact_pool.log")

INITIAL_INPUT_SOURCE = "initial_input"
_MISSING = object()


def _error_result(
    message: str,
    code: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """返回对象池统一错误结构。"""
    try:
        issue = {"code": code, "message": message}
        if isinstance(details, dict):
            issue.update(details)
        return {
            "status": "error",
            "message": message,
            "issues": [issue],
        }
    except Exception as exc:
        logger.error("构造Artifact Pool错误结构失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": str(exc),
            "issues": [],
        }


def _normalize_tool_result(tool_result: Any) -> dict[str, Any]:
    """把Runner返回的JSON字符串或字典规范化为字典。"""
    try:
        if isinstance(tool_result, dict):
            return copy.deepcopy(tool_result)
        if isinstance(tool_result, str) and tool_result.strip():
            parsed = json.loads(tool_result)
            return parsed if isinstance(parsed, dict) else {}
        return {}
    except Exception as exc:
        logger.error("解析工具结果失败: %s", exc, exc_info=True)
        return {}


def _extract_bound_value(result: dict[str, Any], path: Any) -> Any:
    """按照注册表中的点分路径提取输出；`$`表示完整结果。"""
    try:
        if path == "$":
            return copy.deepcopy(result)
        if not isinstance(path, str) or not path.strip():
            return _MISSING
        current: Any = result
        for key in path.split("."):
            if not isinstance(current, dict) or key not in current:
                return _MISSING
            current = current[key]
        return copy.deepcopy(current)
    except Exception as exc:
        logger.error("提取类型化工具输出失败: %s", exc, exc_info=True)
        return _MISSING


def _value_size(value: Any) -> int | None:
    """只返回值规模，不返回真实内容。"""
    try:
        if isinstance(value, (str, bytes, list, tuple, set, dict)):
            return len(value)
        return None
    except Exception:
        return None


@dataclass(frozen=True)
class Artifact:
    """对象池中的一个真实数据对象。"""

    artifact_id: str
    type_name: str
    source_id: str
    value: Any
    producer_tool: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        """生成不包含真实值的审计元数据。"""
        try:
            return {
                "artifact_id": self.artifact_id,
                "type_name": self.type_name,
                "source_id": self.source_id,
                "producer_tool": self.producer_tool,
                "value_kind": type(self.value).__name__,
                "value_size": _value_size(self.value),
            }
        except Exception as exc:
            logger.error("生成Artifact元数据失败: %s", exc, exc_info=True)
            return {
                "status": "error",
                "message": f"生成Artifact元数据失败: {exc}",
            }


class ArtifactPool:
    """按数据类型和来源保存真实运行值的请求级对象池。"""

    def __init__(self) -> None:
        try:
            self._artifacts: list[Artifact] = []
            self._next_id = 1
        except Exception as exc:
            logger.error("初始化Artifact Pool失败: %s", exc, exc_info=True)
            self._artifacts = []
            self._next_id = 1

    def _find(self, type_name: str, source_id: str) -> list[Artifact]:
        """查找同一类型和来源的对象。"""
        try:
            return [
                artifact
                for artifact in self._artifacts
                if artifact.type_name == type_name
                and artifact.source_id == source_id
            ]
        except Exception as exc:
            logger.error("查找Artifact失败: %s", exc, exc_info=True)
            return []

    def register(
        self,
        type_name: Any,
        value: Any,
        source_id: Any,
        *,
        producer_tool: Any = None,
    ) -> dict[str, Any]:
        """注册单个对象；同类型同来源重复时失败。"""
        try:
            normalized_type = type_name.strip() if isinstance(type_name, str) else ""
            normalized_source = source_id.strip() if isinstance(source_id, str) else ""
            normalized_tool = (
                producer_tool.strip()
                if isinstance(producer_tool, str) and producer_tool.strip()
                else None
            )
            if not normalized_type or not normalized_source:
                return _error_result(
                    "Artifact类型和来源必须是非空字符串",
                    "INVALID_ARTIFACT_METADATA",
                )
            if value is None or (isinstance(value, str) and not value.strip()):
                return _error_result(
                    "Artifact真实值不能为空",
                    "EMPTY_ARTIFACT_VALUE",
                    details={
                        "type_name": normalized_type,
                        "source_id": normalized_source,
                    },
                )
            if self._find(normalized_type, normalized_source):
                return _error_result(
                    "同一来源下已存在相同类型的数据对象",
                    "DUPLICATE_ARTIFACT",
                    details={
                        "type_name": normalized_type,
                        "source_id": normalized_source,
                    },
                )

            artifact = Artifact(
                artifact_id=f"artifact_{self._next_id:04d}",
                type_name=normalized_type,
                source_id=normalized_source,
                value=copy.deepcopy(value),
                producer_tool=normalized_tool,
            )
            self._artifacts.append(artifact)
            self._next_id += 1
            logger.info(
                "Artifact注册完成 | id=%s | type=%s | source=%s | producer=%s",
                artifact.artifact_id,
                artifact.type_name,
                artifact.source_id,
                artifact.producer_tool or "user",
            )
            return {
                "status": "success",
                "artifact": artifact.to_metadata(),
            }
        except Exception as exc:
            logger.error("注册Artifact失败: %s", exc, exc_info=True)
            return _error_result(
                f"注册Artifact失败: {exc}",
                "ARTIFACT_REGISTRATION_EXCEPTION",
            )

    def register_initial(self, initial_artifacts: Any) -> dict[str, Any]:
        """原子注册用户输入提供的初始类型和值。"""
        try:
            if not isinstance(initial_artifacts, dict) or not initial_artifacts:
                return _error_result(
                    "初始Artifact必须是非空类型到值映射",
                    "INVALID_INITIAL_ARTIFACTS",
                )

            normalized: list[tuple[str, Any]] = []
            for type_name, value in initial_artifacts.items():
                if not isinstance(type_name, str) or not type_name.strip():
                    return _error_result(
                        "初始Artifact包含非法类型名",
                        "INVALID_INITIAL_ARTIFACTS",
                    )
                if value is None or (isinstance(value, str) and not value.strip()):
                    return _error_result(
                        "初始Artifact包含空值",
                        "EMPTY_ARTIFACT_VALUE",
                        details={"type_name": type_name},
                    )
                normalized_type = type_name.strip()
                if self._find(normalized_type, INITIAL_INPUT_SOURCE):
                    return _error_result(
                        "初始来源已存在相同类型的数据对象",
                        "DUPLICATE_ARTIFACT",
                        details={"type_name": normalized_type},
                    )
                normalized.append((normalized_type, value))

            prepared = [
                Artifact(
                    artifact_id=f"artifact_{self._next_id + index:04d}",
                    type_name=type_name,
                    source_id=INITIAL_INPUT_SOURCE,
                    value=copy.deepcopy(value),
                    producer_tool=None,
                )
                for index, (type_name, value) in enumerate(normalized)
            ]
            self._artifacts.extend(prepared)
            self._next_id += len(prepared)
            registered = [artifact.to_metadata() for artifact in prepared]
            for artifact in prepared:
                logger.info(
                    "初始Artifact注册完成 | id=%s | type=%s | source=%s",
                    artifact.artifact_id,
                    artifact.type_name,
                    artifact.source_id,
                )
            return {
                "status": "success",
                "artifacts": registered,
            }
        except Exception as exc:
            logger.error("注册初始Artifact失败: %s", exc, exc_info=True)
            return _error_result(
                f"注册初始Artifact失败: {exc}",
                "INITIAL_ARTIFACT_EXCEPTION",
            )

    def resolve(self, type_name: Any, source_id: Any) -> dict[str, Any]:
        """按结构化计划中的类型和来源精确取得一个真实对象。"""
        try:
            normalized_type = type_name.strip() if isinstance(type_name, str) else ""
            normalized_source = source_id.strip() if isinstance(source_id, str) else ""
            if not normalized_type or not normalized_source:
                return _error_result(
                    "Artifact解析需要非空类型和来源",
                    "INVALID_ARTIFACT_REFERENCE",
                )
            matches = self._find(normalized_type, normalized_source)
            if not matches:
                return _error_result(
                    "结构化计划引用的数据对象不存在",
                    "ARTIFACT_NOT_FOUND",
                    details={
                        "type_name": normalized_type,
                        "source_id": normalized_source,
                    },
                )
            if len(matches) != 1:
                return _error_result(
                    "结构化计划引用了多个同类型同来源对象",
                    "AMBIGUOUS_ARTIFACT",
                    details={
                        "type_name": normalized_type,
                        "source_id": normalized_source,
                    },
                )
            artifact = matches[0]
            return {
                "status": "success",
                "artifact": artifact.to_metadata(),
                "value": copy.deepcopy(artifact.value),
            }
        except Exception as exc:
            logger.error("解析Artifact失败: %s", exc, exc_info=True)
            return _error_result(
                f"解析Artifact失败: {exc}",
                "ARTIFACT_RESOLUTION_EXCEPTION",
            )

    def resolve_step_inputs(self, step: Any) -> dict[str, Any]:
        """根据Plan Step的input_from映射取得全部真实输入。"""
        try:
            if not isinstance(step, dict):
                return _error_result(
                    "Plan Step必须是字典",
                    "INVALID_PLAN_STEP",
                )
            input_types = step.get("input_type")
            input_from = step.get("input_from")
            if not isinstance(input_types, list) or not isinstance(input_from, dict):
                return _error_result(
                    "Plan Step缺少input_type或input_from",
                    "INVALID_PLAN_STEP",
                )

            values: dict[str, Any] = {}
            artifact_ids: dict[str, str] = {}
            for type_name in input_types:
                source_id = input_from.get(type_name)
                resolved = self.resolve(type_name, source_id)
                if resolved.get("status") != "success":
                    return resolved
                values[type_name] = resolved["value"]
                artifact_ids[type_name] = resolved["artifact"]["artifact_id"]
            return {
                "status": "success",
                "values": values,
                "artifact_ids": artifact_ids,
            }
        except Exception as exc:
            logger.error("解析Plan Step输入失败: %s", exc, exc_info=True)
            return _error_result(
                f"解析Plan Step输入失败: {exc}",
                "STEP_INPUT_RESOLUTION_EXCEPTION",
            )

    def register_tool_outputs(
        self,
        step_id: Any,
        tool_name: Any,
        expected_output_types: Any,
        tool_result: Any,
    ) -> dict[str, Any]:
        """根据唯一注册表绑定原子写入一个成功工具的全部输出。"""
        try:
            normalized_step = step_id.strip() if isinstance(step_id, str) else ""
            normalized_tool = tool_name.strip() if isinstance(tool_name, str) else ""
            if not normalized_step or not normalized_tool:
                return _error_result(
                    "工具输出注册需要非空step_id和tool_name",
                    "INVALID_TOOL_OUTPUT_METADATA",
                )

            tool_info = BIOLOGY_TOOLS_DESC.get(normalized_tool)
            if not isinstance(tool_info, dict) or tool_info.get("is_internal"):
                return _error_result(
                    "工具不在外部Biology唯一注册表中",
                    "UNKNOWN_TOOL",
                    details={"tool_name": normalized_tool},
                )
            registered_output_types = list(tool_info.get("output_types") or [])
            if (
                not isinstance(expected_output_types, list)
                or expected_output_types != registered_output_types
            ):
                return _error_result(
                    "Plan Step输出类型与工具注册表不一致",
                    "OUTPUT_CONTRACT_MISMATCH",
                    details={
                        "tool_name": normalized_tool,
                        "expected_output_types": list(expected_output_types or []),
                        "registered_output_types": registered_output_types,
                    },
                )

            normalized_result = _normalize_tool_result(tool_result)
            if normalized_result.get("status") != "success":
                return _error_result(
                    normalized_result.get("message", "工具未返回成功状态"),
                    "TOOL_EXECUTION_FAILED",
                    details={"tool_name": normalized_tool},
                )

            output_bindings = tool_info.get("output_bindings")
            if not isinstance(output_bindings, dict):
                return _error_result(
                    "工具缺少Artifact输出字段绑定",
                    "OUTPUT_BINDING_MISSING",
                    details={"tool_name": normalized_tool},
                )

            pending_values: list[tuple[str, Any]] = []
            for output_type in registered_output_types:
                if output_type not in output_bindings:
                    return _error_result(
                        "工具输出类型没有对应字段绑定",
                        "OUTPUT_BINDING_MISSING",
                        details={
                            "tool_name": normalized_tool,
                            "output_type": output_type,
                        },
                    )
                value = _extract_bound_value(
                    normalized_result,
                    output_bindings[output_type],
                )
                if value is _MISSING or value is None:
                    return _error_result(
                        "成功工具结果缺少声明的输出字段",
                        "DECLARED_OUTPUT_MISSING",
                        details={
                            "tool_name": normalized_tool,
                            "output_type": output_type,
                            "binding": output_bindings[output_type],
                        },
                    )
                if isinstance(value, dict) and value.get("status") == "error":
                    return _error_result(
                        value.get("message", "声明输出子结果返回错误状态"),
                        "DECLARED_OUTPUT_FAILED",
                        details={
                            "tool_name": normalized_tool,
                            "output_type": output_type,
                            "binding": output_bindings[output_type],
                        },
                    )
                if self._find(output_type, normalized_step):
                    return _error_result(
                        "步骤来源下已存在相同输出类型",
                        "DUPLICATE_ARTIFACT",
                        details={
                            "source_id": normalized_step,
                            "type_name": output_type,
                        },
                    )
                pending_values.append((output_type, value))

            prepared = [
                Artifact(
                    artifact_id=f"artifact_{self._next_id + index:04d}",
                    type_name=output_type,
                    source_id=normalized_step,
                    value=copy.deepcopy(value),
                    producer_tool=normalized_tool,
                )
                for index, (output_type, value) in enumerate(pending_values)
            ]
            self._artifacts.extend(prepared)
            self._next_id += len(prepared)
            registered = [artifact.to_metadata() for artifact in prepared]
            for artifact in prepared:
                logger.info(
                    "工具输出Artifact注册完成 | id=%s | type=%s | "
                    "source=%s | producer=%s",
                    artifact.artifact_id,
                    artifact.type_name,
                    artifact.source_id,
                    artifact.producer_tool,
                )
            return {
                "status": "success",
                "artifacts": registered,
            }
        except Exception as exc:
            logger.error("注册工具输出Artifact失败: %s", exc, exc_info=True)
            return _error_result(
                f"注册工具输出Artifact失败: {exc}",
                "TOOL_OUTPUT_REGISTRATION_EXCEPTION",
            )

    def snapshot(self) -> dict[str, Any]:
        """返回只含元数据的对象池快照，供日志和论文统计使用。"""
        try:
            metadata = [artifact.to_metadata() for artifact in self._artifacts]
            if any(item.get("status") == "error" for item in metadata):
                return _error_result(
                    "生成Artifact Pool快照失败",
                    "ARTIFACT_SNAPSHOT_ERROR",
                )
            return {
                "status": "success",
                "artifact_count": len(metadata),
                "artifacts": metadata,
            }
        except Exception as exc:
            logger.error("生成Artifact Pool快照失败: %s", exc, exc_info=True)
            return _error_result(
                f"生成Artifact Pool快照失败: {exc}",
                "ARTIFACT_SNAPSHOT_EXCEPTION",
            )
