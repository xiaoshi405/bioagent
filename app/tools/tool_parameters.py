# -*- coding: utf-8 -*-
"""工具值参数的提取、校验与请求级绑定。

修改记录：
- 2026-09-04：新增PDB/ANM显式参数上下文，保证普通执行与类型化执行
  使用同一来源和同一契约，避免Artifact正确但数值参数静默丢失。

本模块只处理用户明确给出的值参数；文件、序列等数据对象仍由Artifact
Pool管理。未出现的参数不写入上下文，由工具接口使用其注册默认值。
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any

from app.core.logger import get_logger
from ToolsAgent.ToolsFuns.Biology.tool_name_dict import BIOLOGY_TOOLS_DESC


logger = get_logger("ToolParameters", "tool_parameters.log")

ANM_API_PATH = "/api/calculate_protein_anm"
PDB_CHAIN_API_PATH = "/api/select_pdb_chain"

_CHINESE_DIGITS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_MODE_TOKEN = r"(?:[-+]?\d+(?:\.\d+)?|[一二两三四五六七八九十]{1,3})"


def _error(message: str, code: str, *, api_path: str, parameter: str) -> dict:
    """构造工具参数统一错误结构。"""
    return {
        "status": "error",
        "message": message,
        "issues": [
            {
                "code": code,
                "message": message,
                "api_path": api_path,
                "parameter": parameter,
            }
        ],
    }


def _conflict_error(*, api_path: str, parameter: str, values: list[Any]) -> dict:
    """构造同一显式参数出现多个不同值时的失败关闭错误。"""
    displayed = "、".join(str(value) for value in values)
    return _error(
        f"参数{parameter}存在互相冲突的显式值：{displayed}",
        "CONFLICTING_TOOL_PARAMETER",
        api_path=api_path,
        parameter=parameter,
    )


def _parse_small_integer(token: str) -> int | None:
    """解析1至99的阿拉伯数字或常见中文整数写法。"""
    try:
        normalized = str(token or "").strip()
        if not normalized:
            return None
        if re.fullmatch(r"[-+]?\d+", normalized):
            return int(normalized)
        if normalized in _CHINESE_DIGITS:
            return _CHINESE_DIGITS[normalized]
        if "十" in normalized:
            left, right = normalized.split("十", 1)
            tens = _CHINESE_DIGITS.get(left, 1) if left else 1
            ones = _CHINESE_DIGITS.get(right, 0) if right else 0
            return tens * 10 + ones
        return None
    except Exception:
        return None


def _parameter_schema() -> dict[str, Any]:
    """从唯一工具注册表读取ANM机器可校验参数契约。"""
    try:
        tool = BIOLOGY_TOOLS_DESC.get("calculate_protein_anm") or {}
        schema = tool.get("parameter_schema") or {}
        return copy.deepcopy(schema) if isinstance(schema, dict) else {}
    except Exception as exc:
        logger.error("读取ANM参数契约失败: %s", exc, exc_info=True)
        return {}


def validate_tool_parameters(api_path: Any, parameters: Any) -> dict:
    """按注册表白名单校验单个接口的显式参数，不补入默认值。"""
    try:
        normalized_api = str(api_path or "").strip()
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            return _error(
                "工具参数必须是字典",
                "INVALID_TOOL_PARAMETERS",
                api_path=normalized_api,
                parameter="*",
            )

        if normalized_api == PDB_CHAIN_API_PATH:
            allowed = {"chain_id"}
        elif normalized_api == ANM_API_PATH:
            allowed = set(_parameter_schema())
        else:
            allowed = set()

        unsupported = sorted(str(key) for key in parameters if key not in allowed)
        if unsupported:
            return _error(
                f"接口包含未注册参数：{'、'.join(unsupported)}",
                "UNSUPPORTED_TOOL_PARAMETER",
                api_path=normalized_api,
                parameter=unsupported[0],
            )

        validated: dict[str, Any] = {}
        schema = _parameter_schema() if normalized_api == ANM_API_PATH else {}
        for name, value in parameters.items():
            if name == "chain_id":
                if not isinstance(value, str) or not value.strip():
                    return _error(
                        "chain_id必须是非空字符串",
                        "INVALID_TOOL_PARAMETER",
                        api_path=normalized_api,
                        parameter=name,
                    )
                validated[name] = value.strip()
                continue

            specification = schema.get(name) or {}
            expected_type = specification.get("type")
            if expected_type == "integer":
                if isinstance(value, bool) or not isinstance(value, int):
                    return _error(
                        f"{name}必须是整数",
                        "INVALID_TOOL_PARAMETER",
                        api_path=normalized_api,
                        parameter=name,
                    )
                normalized_value: int | float = value
            elif expected_type == "number":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return _error(
                        f"{name}必须是数值",
                        "INVALID_TOOL_PARAMETER",
                        api_path=normalized_api,
                        parameter=name,
                    )
                normalized_value = float(value)
                if not math.isfinite(normalized_value):
                    return _error(
                        f"{name}必须是有限数值",
                        "INVALID_TOOL_PARAMETER",
                        api_path=normalized_api,
                        parameter=name,
                    )
            else:
                return _error(
                    f"参数{name}缺少可执行契约",
                    "TOOL_PARAMETER_SCHEMA_ERROR",
                    api_path=normalized_api,
                    parameter=name,
                )

            minimum = specification.get("minimum")
            maximum = specification.get("maximum")
            if minimum is not None and normalized_value < minimum:
                return _error(
                    f"{name}不能小于{minimum}",
                    "TOOL_PARAMETER_OUT_OF_RANGE",
                    api_path=normalized_api,
                    parameter=name,
                )
            if maximum is not None and normalized_value > maximum:
                return _error(
                    f"{name}不能大于{maximum}",
                    "TOOL_PARAMETER_OUT_OF_RANGE",
                    api_path=normalized_api,
                    parameter=name,
                )
            validated[name] = normalized_value

        return {"status": "success", "parameters": validated, "issues": []}
    except Exception as exc:
        logger.error("校验工具参数失败: %s", exc, exc_info=True)
        return _error(
            f"校验工具参数失败: {exc}",
            "TOOL_PARAMETER_VALIDATION_EXCEPTION",
            api_path=str(api_path or ""),
            parameter="*",
        )


def _extract_pdb_parameter_candidates(text: Any) -> dict:
    """提取PDB显式参数候选，并保留非法候选供统一诊断。"""
    try:
        normalized = text if isinstance(text, str) else ""
        candidates: dict[str, Any] = {}
        conflicts: dict[str, list[Any]] = {}

        def register_candidates(name: str, matches: list[tuple[int, Any]]) -> None:
            """按文本顺序去重；不同显式值不静默选择其中一个。"""
            ordered_values: list[Any] = []
            for _, value in sorted(matches, key=lambda item: item[0]):
                if value not in ordered_values:
                    ordered_values.append(value)
            if len(ordered_values) == 1:
                candidates[name] = ordered_values[0]
            elif len(ordered_values) > 1:
                conflicts[name] = ordered_values

        mode_patterns = (
            rf"(?:n[_\s-]*modes?|模态数)\s*(?:为|is|=|:|：)?\s*({_MODE_TOKEN})",
            rf"(?:前|first\s*)?\s*({_MODE_TOKEN})\s*(?:个|阶|种)?\s*"
            rf"(?:ANM\s*)?(?:最低频\s*|低频\s*|非零\s*)*"
            rf"(?:ANM\s*)?(?:模态|modes?)",
        )
        mode_matches: list[tuple[int, Any]] = []
        for pattern in mode_patterns:
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                token = match.group(1)
                parsed = _parse_small_integer(token)
                mode_matches.append(
                    (match.start(1), parsed if parsed is not None else token)
                )
        register_candidates("n_modes", mode_matches)

        cutoff_matches = [
            (match.start(1), float(match.group(1)))
            for match in re.finditer(
                r"(?:cutoff|截断距离|截止距离)\s*(?:为|is|=|:|：)?\s*"
                r"([-+]?\d+(?:\.\d+)?)",
                normalized,
                flags=re.IGNORECASE,
            )
        ]
        register_candidates("cutoff", cutoff_matches)

        chain_patterns = (
            r"(?:链(?:\s*ID)?\s*)(?:为|=|:|：)?\s*"
            r"([A-Za-z0-9])(?![A-Za-z0-9])",
            r"(?<![A-Za-z0-9])([A-Za-z0-9])\s*链"
            r"(?!\s*[A-Za-z0-9])",
            r"(?:chain(?:[\s_-]*id)?\s*)"
            r"(?:is|为|=|:|：)?\s*([A-Za-z0-9])(?![A-Za-z0-9])",
        )
        chain_matches: list[tuple[int, Any]] = []
        for pattern in chain_patterns:
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                chain_matches.append((match.start(1), match.group(1)))
        register_candidates("chain_id", chain_matches)
        return {
            "status": "success",
            "parameters": candidates,
            "conflicts": conflicts,
            "issues": [],
        }
    except Exception as exc:
        logger.error("提取PDB工具参数失败: %s", exc, exc_info=True)
        return _error(
            f"提取PDB工具参数失败: {exc}",
            "TOOL_PARAMETER_EXTRACTION_EXCEPTION",
            api_path=ANM_API_PATH,
            parameter="*",
        )


def extract_pdb_options(text: Any) -> dict:
    """兼容旧调用方：返回通过契约校验的PDB参数，非法参数不伪造成默认值。"""
    extracted = _extract_pdb_parameter_candidates(text)
    if extracted.get("status") != "success":
        return {}
    validated_parameters: dict[str, Any] = {}
    for name, value in (extracted.get("parameters") or {}).items():
        api_path = PDB_CHAIN_API_PATH if name == "chain_id" else ANM_API_PATH
        validated = validate_tool_parameters(api_path, {name: value})
        if validated.get("status") == "success":
            validated_parameters.update(validated.get("parameters") or {})
    return validated_parameters


def build_tool_parameter_context(text: Any) -> dict:
    """从原始用户问题构建按接口绑定的、可审计请求级参数上下文。"""
    try:
        extracted = _extract_pdb_parameter_candidates(text)
        if extracted.get("status") != "success":
            return extracted
        candidates = extracted.get("parameters") or {}
        conflicts = extracted.get("conflicts") or {}
        by_api: dict[str, dict[str, Any]] = {}
        errors_by_api: dict[str, dict[str, Any]] = {}
        anm_conflict = next(
            (
                (name, values)
                for name, values in conflicts.items()
                if name in {"n_modes", "cutoff", "chain_id"}
            ),
            None,
        )
        if anm_conflict:
            name, values = anm_conflict
            errors_by_api[ANM_API_PATH] = _conflict_error(
                api_path=ANM_API_PATH,
                parameter=name,
                values=values,
            )
        else:
            anm_result = validate_tool_parameters(ANM_API_PATH, candidates)
            if anm_result.get("status") == "success":
                anm_parameters = anm_result.get("parameters") or {}
                if anm_parameters:
                    by_api[ANM_API_PATH] = anm_parameters
            else:
                errors_by_api[ANM_API_PATH] = anm_result

        chain_conflict = conflicts.get("chain_id")
        chain_id = candidates.get("chain_id")
        if chain_conflict:
            errors_by_api[PDB_CHAIN_API_PATH] = _conflict_error(
                api_path=PDB_CHAIN_API_PATH,
                parameter="chain_id",
                values=chain_conflict,
            )
        elif chain_id:
            chain_result = validate_tool_parameters(
                PDB_CHAIN_API_PATH,
                {"chain_id": chain_id},
            )
            if chain_result.get("status") == "success":
                by_api[PDB_CHAIN_API_PATH] = chain_result.get("parameters") or {}
            else:
                errors_by_api[PDB_CHAIN_API_PATH] = chain_result
        return {
            "status": "success",
            "by_api": by_api,
            "errors_by_api": errors_by_api,
            "issues": [],
        }
    except Exception as exc:
        logger.error("构建工具参数上下文失败: %s", exc, exc_info=True)
        return _error(
            f"构建工具参数上下文失败: {exc}",
            "TOOL_PARAMETER_CONTEXT_EXCEPTION",
            api_path=ANM_API_PATH,
            parameter="*",
        )


def resolve_tool_parameters(context: Any, api_path: Any) -> dict:
    """按接口从请求级上下文取值，并再次执行防御性契约校验。"""
    try:
        normalized_api = str(api_path or "").strip()
        if context is None:
            return {"status": "success", "parameters": {}, "issues": []}
        if not isinstance(context, dict):
            return _error(
                "工具参数上下文必须是字典",
                "INVALID_TOOL_PARAMETER_CONTEXT",
                api_path=normalized_api,
                parameter="*",
            )
        if context.get("status") == "error":
            return copy.deepcopy(context)
        by_api = context.get("by_api") or {}
        if not isinstance(by_api, dict):
            return _error(
                "工具参数上下文缺少by_api映射",
                "INVALID_TOOL_PARAMETER_CONTEXT",
                api_path=normalized_api,
                parameter="*",
            )
        errors_by_api = context.get("errors_by_api") or {}
        if isinstance(errors_by_api, dict) and normalized_api in errors_by_api:
            scoped_error = errors_by_api.get(normalized_api)
            if isinstance(scoped_error, dict):
                return copy.deepcopy(scoped_error)
        return validate_tool_parameters(normalized_api, by_api.get(normalized_api) or {})
    except Exception as exc:
        logger.error("解析接口参数上下文失败: %s", exc, exc_info=True)
        return _error(
            f"解析接口参数上下文失败: {exc}",
            "TOOL_PARAMETER_CONTEXT_EXCEPTION",
            api_path=str(api_path or ""),
            parameter="*",
        )
