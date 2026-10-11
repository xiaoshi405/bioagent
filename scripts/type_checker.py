# -*- coding: utf-8 -*-
"""
BioAgent 类型约束规划基础模块。

修改记录：
- 2026-09-05：将注册表的重规划角色策略透传到工具签名，供 F1
  目标保真门禁使用。
- 2026-08-15：新增注册表显式授权的重复无效步骤剪枝；仅当候选步骤相邻重复、
  均缺失输入、输出未被下游消费且删除后全计划唯一类型合法时才允许改写，
  并输出可审计的修复操作证明。
- 2026-07-22：新增基于类型化数据对象池的计划检查，并支持
  PDB 文件到氨基酸序列的保守桥接修复。当前模块保持独立，尚未接入主链路。
- 2026-07-23：对缺失显式输入类型的兼容回退增加告警，便于审计元数据质量。
- 2026-07-24：增加单记录Protein FASTA到氨基酸序列的确定性桥接；
  多个可用桥接同时存在时拒绝猜测输入来源。
- 2026-07-24：修正PDB桥语义：多链提取输出链序列映射，只有同时存在
  PDB文件和显式链ID时，才允许插入select_pdb_chain得到单条序列。
- 2026-08-14：桥接能力改由唯一工具注册表声明；新增有界唯一最短路径
  搜索，支持多记录FASTA经集合和显式记录ID进行最多配置跳数的修复。

设计原则：
1. 工具签名只从 BIOLOGY_TOOLS_DESC 唯一注册表生成；
2. 多个工具可以复用仍然存在的用户输入，不能只检查相邻步骤；
3. 自动修复只能插入确定性的注册工具，或执行注册表明确许可且有全图证明的剪枝；
4. 所有公开函数返回标准 status 字段，异常不能传播到主进程。
"""

from __future__ import annotations

from typing import Any, Iterable

from app.core.config import Config
from app.core.logger import get_logger
from ToolsAgent.ToolsFuns.Biology.tool_name_dict import BIOLOGY_TOOLS_DESC


logger = get_logger("TypeChecker", "type_checker.log")

AMINO_ACID_SEQUENCE = "amino acid sequence"
PDB_STRUCTURE_FILE = "PDB structure file"
PDB_CHAIN_IDENTIFIER = "PDB chain identifier"
PDB_CHAIN_SEQUENCE_MAP = "PDB chain sequence map"
SINGLE_PROTEIN_FASTA_FILE = "single-record protein FASTA file"
MULTI_PROTEIN_FASTA_FILE = "multi-record protein FASTA file"
PROTEIN_SEQUENCE_COLLECTION = "protein sequence collection"
FASTA_RECORD_IDENTIFIER = "FASTA record identifier"
FASTA_RECORD_IDENTIFIER_PAIR = "FASTA record identifier pair"
PAIRWISE_PROTEIN_ALIGNMENT = "pairwise protein alignment"
ALIGNMENT_STATISTICS = "alignment statistics"
PDB_TO_SEQUENCE_BRIDGE = "select_pdb_chain"
FASTA_TO_SEQUENCE_BRIDGE = "parse_protein_fasta"
MULTI_FASTA_TO_COLLECTION_BRIDGE = "parse_multi_protein_fasta"
FASTA_COLLECTION_TO_SEQUENCE_BRIDGE = "select_fasta_record"
FASTA_COLLECTION_TO_ALIGNMENT_BRIDGE = "align_fasta_records"


def _normalize_type_name(value: Any) -> str:
    """把类型名称转换为稳定字符串；非法值返回空字符串。"""
    try:
        if not isinstance(value, str):
            return ""
        return " ".join(value.strip().split())
    except Exception as exc:
        logger.error("规范化类型名称失败: %s", exc, exc_info=True)
        return ""


def _normalize_type_collection(values: Any) -> list[str]:
    """规范化类型集合并保持首次出现顺序。"""
    try:
        if isinstance(values, str):
            candidates: Iterable[Any] = [values]
        elif isinstance(values, (list, tuple, set, frozenset)):
            candidates = values
        else:
            return []

        normalized: list[str] = []
        for value in candidates:
            type_name = _normalize_type_name(value)
            if type_name and type_name not in normalized:
                normalized.append(type_name)
        return normalized
    except Exception as exc:
        logger.error("规范化类型集合失败: %s", exc, exc_info=True)
        return []


def get_tool_signatures() -> dict[str, Any]:
    """从唯一工具注册表构造外部工具的输入/输出类型签名。"""
    try:
        signatures: dict[str, dict[str, Any]] = {}
        for tool_name, info in BIOLOGY_TOOLS_DESC.items():
            if not isinstance(info, dict) or info.get("is_internal"):
                continue

            raw_input_types = info.get("input_types")
            if raw_input_types is None:
                logger.warning(
                    "工具 %s 未显式声明 input_types，兼容回退为 %s",
                    tool_name,
                    AMINO_ACID_SEQUENCE,
                )
                raw_input_types = [AMINO_ACID_SEQUENCE]

            input_types = _normalize_type_collection(raw_input_types)
            output_types = _normalize_type_collection(info.get("output_types") or [])
            if not input_types:
                logger.warning("工具 %s 缺少有效输入类型，跳过注册", tool_name)
                continue

            signatures[tool_name] = {
                "input_types": input_types,
                "output_types": output_types,
                "api_path": info.get("api_path"),
                "repair_policy": (
                    dict(info.get("repair_policy"))
                    if isinstance(info.get("repair_policy"), dict)
                    else None
                ),
                "replan_policy": (
                    dict(info.get("replan_policy"))
                    if isinstance(info.get("replan_policy"), dict)
                    else None
                ),
                "benchmark_generation": str(
                    info.get("benchmark_generation") or "legacy-v1"
                ),
            }

        return {
            "status": "success",
            "tool_count": len(signatures),
            "signatures": signatures,
        }
    except Exception as exc:
        logger.error("构造工具签名失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"构造工具签名失败: {exc}",
            "tool_count": 0,
            "signatures": {},
        }


def _invalid_request_result(message: str, code: str = "INVALID_REQUEST") -> dict[str, Any]:
    """生成不会抛异常的统一检查失败结构。"""
    try:
        return {
            "status": "error",
            "message": message,
            "valid": False,
            "issues": [{"code": code, "message": message}],
            "steps": [],
        }
    except Exception as exc:
        logger.error("生成类型检查错误结构失败: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc), "valid": False}


def validate_tool_plan(
    tool_path: Any,
    available_input_types: Any,
) -> dict[str, Any]:
    """
    基于类型化数据对象池验证工具计划。

    `status=success` 表示检查器正常完成；计划本身是否合法由 `valid` 表示。
    这样可以区分“检测到非法计划”和“检查器运行失败”。
    """
    try:
        if not isinstance(tool_path, (list, tuple)) or not tool_path:
            return _invalid_request_result("工具计划必须是非空列表", "INVALID_PLAN")

        initial_types = _normalize_type_collection(available_input_types)
        if not initial_types:
            return _invalid_request_result(
                "至少需要一个有效的初始输入类型", "MISSING_INITIAL_TYPE"
            )

        signature_result = get_tool_signatures()
        if signature_result.get("status") != "success":
            return _invalid_request_result(
                signature_result.get("message", "工具签名不可用"),
                "SIGNATURE_REGISTRY_ERROR",
            )
        signatures = signature_result["signatures"]

        available_types = list(initial_types)
        artifact_sources: dict[str, list[str]] = {
            type_name: ["initial_input"] for type_name in initial_types
        }
        steps: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []

        for index, raw_tool_name in enumerate(tool_path):
            tool_name = raw_tool_name.strip() if isinstance(raw_tool_name, str) else ""
            if not tool_name or tool_name not in signatures:
                issue = {
                    "code": "UNKNOWN_TOOL",
                    "step_index": index,
                    "tool_name": raw_tool_name,
                    "message": f"第 {index + 1} 步工具未在外部工具注册表中注册",
                }
                issues.append(issue)
                steps.append(
                    {
                        "step_index": index,
                        "tool_name": raw_tool_name,
                        "valid": False,
                        "issue_code": "UNKNOWN_TOOL",
                    }
                )
                continue

            signature = signatures[tool_name]
            required_types = signature["input_types"]
            matched_types = [
                type_name for type_name in required_types if type_name in available_types
            ]
            missing_types = [
                type_name for type_name in required_types if type_name not in available_types
            ]

            if missing_types:
                issue = {
                    "code": "MISSING_INPUT_TYPE",
                    "step_index": index,
                    "tool_name": tool_name,
                    "required_types": required_types,
                    "missing_types": missing_types,
                    "available_types": list(available_types),
                    "message": (
                        f"第 {index + 1} 步 {tool_name} 缺少可达输入类型: "
                        f"{', '.join(missing_types)}"
                    ),
                }
                issues.append(issue)
                steps.append(
                    {
                        "step_index": index,
                        "tool_name": tool_name,
                        "valid": False,
                        "required_types": required_types,
                        "missing_types": missing_types,
                        "available_types_before": list(available_types),
                        "issue_code": "MISSING_INPUT_TYPE",
                    }
                )
                continue

            produced_types = signature["output_types"]
            for output_type in produced_types:
                if output_type not in available_types:
                    available_types.append(output_type)
                artifact_sources.setdefault(output_type, []).append(
                    f"step:{index}:{tool_name}"
                )

            steps.append(
                {
                    "step_index": index,
                    "tool_name": tool_name,
                    "valid": True,
                    "required_types": required_types,
                    "matched_input_types": matched_types,
                    "input_sources": {
                        type_name: list(artifact_sources.get(type_name, []))
                        for type_name in matched_types
                    },
                    "produced_types": produced_types,
                    "available_types_after": list(available_types),
                }
            )

        valid = not issues
        log_method = logger.info if valid else logger.warning
        log_method(
            "类型检查完成 | valid=%s | steps=%s | issues=%s",
            valid,
            len(tool_path),
            len(issues),
        )
        return {
            "status": "success",
            "valid": valid,
            "original_plan": list(tool_path),
            "initial_types": initial_types,
            "available_types": available_types,
            "artifact_sources": artifact_sources,
            "steps": steps,
            "issues": issues,
        }
    except Exception as exc:
        logger.error("验证工具计划失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"验证工具计划失败: {exc}",
            "valid": False,
            "issues": [{"code": "CHECKER_EXCEPTION", "message": str(exc)}],
            "steps": [],
        }


def _registered_bridge_signatures(
    signatures: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """从工具签名中选择显式声明的确定性桥接器，并稳定排序。"""
    try:
        bridges: list[tuple[str, dict[str, Any]]] = []
        for tool_name, signature in signatures.items():
            if not isinstance(signature, dict):
                continue
            policy = signature.get("repair_policy")
            if (
                isinstance(policy, dict)
                and policy.get("mode") == "deterministic_bridge"
                and signature.get("input_types")
                and signature.get("output_types")
            ):
                bridges.append((tool_name, signature))
        return sorted(bridges, key=lambda item: item[0])
    except Exception as exc:
        logger.error("读取注册桥接器失败: %s", exc, exc_info=True)
        return []


def _find_unique_shortest_bridge_path(
    current_types: list[str],
    required_types: list[str],
    signatures: dict[str, Any],
    max_hops: int,
) -> dict[str, Any]:
    """在类型超图中搜索有界最短桥接路径；多条最短路径时拒绝猜测。"""
    try:
        target = frozenset(required_types)
        initial_state = frozenset(current_types)
        if target.issubset(initial_state):
            return {
                "status": "success",
                "unique": True,
                "path": [],
                "candidate_paths": [[]],
            }

        bridges = _registered_bridge_signatures(signatures)
        frontier: list[tuple[frozenset[str], list[str]]] = [(initial_state, [])]
        minimum_depth_by_state: dict[frozenset[str], int] = {initial_state: 0}

        for depth in range(1, max_hops + 1):
            next_frontier: list[tuple[frozenset[str], list[str]]] = []
            solutions: list[list[str]] = []
            seen_paths: set[tuple[str, ...]] = set()
            for state, path in frontier:
                for bridge_name, bridge_signature in bridges:
                    if bridge_name in path:
                        continue
                    bridge_inputs = frozenset(bridge_signature["input_types"])
                    bridge_outputs = frozenset(bridge_signature["output_types"])
                    if not bridge_inputs.issubset(state):
                        continue
                    next_state = state | bridge_outputs
                    if next_state == state:
                        continue
                    next_path = path + [bridge_name]
                    path_key = tuple(next_path)
                    if path_key in seen_paths:
                        continue
                    seen_paths.add(path_key)
                    if target.issubset(next_state):
                        solutions.append(next_path)
                        continue
                    known_depth = minimum_depth_by_state.get(next_state)
                    if known_depth is None or depth <= known_depth:
                        minimum_depth_by_state[next_state] = depth
                        next_frontier.append((next_state, next_path))

            if solutions:
                unique_solutions = sorted({tuple(path) for path in solutions})
                return {
                    "status": "success",
                    "unique": len(unique_solutions) == 1,
                    "path": list(unique_solutions[0])
                    if len(unique_solutions) == 1
                    else None,
                    "candidate_paths": [list(path) for path in unique_solutions],
                    "path_length": depth,
                }
            frontier = next_frontier
            if not frontier:
                break

        return {
            "status": "success",
            "unique": False,
            "path": None,
            "candidate_paths": [],
            "path_length": None,
        }
    except Exception as exc:
        logger.error("搜索类型桥接路径失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"搜索类型桥接路径失败: {exc}",
            "unique": False,
            "path": None,
            "candidate_paths": [],
        }


def _find_unique_safe_duplicate_pruning(
    tool_path: list[str],
    available_input_types: Any,
    original_validation: dict[str, Any],
    signatures: dict[str, Any],
    max_pruned_steps: int,
) -> dict[str, Any]:
    """寻找唯一、注册表授权且可由全图类型检查证明安全的重复无效步骤剪枝。"""
    try:
        step_results = original_validation.get("steps") or []
        if len(step_results) != len(tool_path):
            return {
                "status": "error",
                "message": "原计划类型检查步骤数与工具计划不一致",
                "candidate_count": 0,
                "candidate": None,
            }

        candidates: list[dict[str, Any]] = []
        index = 0
        while index < len(tool_path):
            tool_name = tool_path[index]
            group_end = index + 1
            while (
                group_end < len(tool_path)
                and tool_path[group_end] == tool_name
            ):
                group_end += 1

            group_size = group_end - index
            signature = signatures.get(tool_name)
            policy = (
                signature.get("repair_policy")
                if isinstance(signature, dict)
                else None
            )
            registry_opt_in = bool(
                isinstance(policy, dict)
                and policy.get("allow_duplicate_invalid_prune") is True
            )
            invalid_missing_input_group = all(
                isinstance(step_results[position], dict)
                and step_results[position].get("valid") is False
                and step_results[position].get("issue_code")
                == "MISSING_INPUT_TYPE"
                for position in range(index, group_end)
            )

            # 至少保留一个下游步骤，避免删除用户可能明确请求的终端结果。
            has_downstream_step = group_end < len(tool_path)
            if (
                registry_opt_in
                and 2 <= group_size <= max_pruned_steps
                and invalid_missing_input_group
                and has_downstream_step
            ):
                output_types = list(signature.get("output_types") or [])
                downstream_required_types: list[str] = []
                for downstream_tool in tool_path[group_end:]:
                    downstream_signature = signatures.get(downstream_tool)
                    if not isinstance(downstream_signature, dict):
                        continue
                    for required_type in downstream_signature.get("input_types") or []:
                        if required_type not in downstream_required_types:
                            downstream_required_types.append(required_type)

                output_consumed_downstream = any(
                    output_type in downstream_required_types
                    for output_type in output_types
                )
                if not output_consumed_downstream:
                    candidate_plan = tool_path[:index] + tool_path[group_end:]
                    candidate_validation = validate_tool_plan(
                        candidate_plan,
                        available_input_types,
                    )
                    candidate_plan_type_valid = bool(
                        candidate_validation.get("status") == "success"
                        and candidate_validation.get("valid") is True
                    )
                    if candidate_plan_type_valid:
                        candidates.append(
                            {
                                "repaired_plan": candidate_plan,
                                "validation": candidate_validation,
                                "group_start": index,
                                "group_end": group_end,
                                "group_size": group_size,
                                "tool_name": tool_name,
                                "input_types": list(
                                    signature.get("input_types") or []
                                ),
                                "output_types": output_types,
                                "downstream_required_types": (
                                    downstream_required_types
                                ),
                                "registry_opt_in": registry_opt_in,
                                "output_consumed_downstream": False,
                                "candidate_plan_type_valid": True,
                            }
                        )
            index = group_end

        return {
            "status": "success",
            "candidate_count": len(candidates),
            "candidate": candidates[0] if len(candidates) == 1 else None,
        }
    except Exception as exc:
        logger.error("搜索安全重复步骤剪枝失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"搜索安全重复步骤剪枝失败: {exc}",
            "candidate_count": 0,
            "candidate": None,
        }


def _pruning_repair_result(
    original_plan: list[str],
    original_validation: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """把已证明唯一合法的剪枝候选转换为统一修复结果和脱敏证明。"""
    tool_name = str(candidate["tool_name"])
    group_start = int(candidate["group_start"])
    group_end = int(candidate["group_end"])
    group_size = int(candidate["group_size"])
    reason = (
        "删除注册表允许剪枝的相邻重复无效步骤；这些步骤均缺失输入类型，"
        "其输出未被下游消费，且删除后完整计划通过类型检查"
    )
    removed_steps: list[dict[str, Any]] = []
    repair_operations: list[dict[str, Any]] = []
    for original_step_index in range(group_start, group_end):
        removed_steps.append(
            {
                "original_step_index": original_step_index,
                "tool_name": tool_name,
                "reason": reason,
            }
        )
        repair_operations.append(
            {
                "operation": "redundant_step_prune",
                "tool_name": tool_name,
                "original_step_index": original_step_index,
                "before_step_index": None,
                "reason": reason,
                "issue_code": "MISSING_INPUT_TYPE",
                "input_types": list(candidate["input_types"]),
                "output_types": list(candidate["output_types"]),
                "downstream_required_types": list(
                    candidate["downstream_required_types"]
                ),
                "registry_opt_in": bool(candidate["registry_opt_in"]),
                "duplicate_group_size": group_size,
                "output_consumed_downstream": bool(
                    candidate["output_consumed_downstream"]
                ),
                "candidate_plan_type_valid": bool(
                    candidate["candidate_plan_type_valid"]
                ),
                "bridge_path_length": None,
            }
        )
    return {
        "status": "success",
        "repairable": True,
        "changed": True,
        "repair_strategy": "duplicate_invalid_step_pruning",
        "original_plan": original_plan,
        "repaired_plan": list(candidate["repaired_plan"]),
        "inserted_steps": [],
        "removed_steps": removed_steps,
        "repair_operations": repair_operations,
        "original_validation": original_validation,
        "validation": candidate["validation"],
    }


def repair_tool_plan(
    tool_path: Any,
    available_input_types: Any,
) -> dict[str, Any]:
    """通过安全剪枝或注册桥接器的有界唯一最短路径修复非法计划。"""
    try:
        original_validation = validate_tool_plan(tool_path, available_input_types)
        if original_validation.get("status") != "success":
            return {
                "status": "error",
                "message": original_validation.get("message", "原计划无法检查"),
                "repairable": False,
                "changed": False,
                "inserted_steps": [],
                "removed_steps": [],
                "repair_operations": [],
                "original_validation": original_validation,
            }
        if original_validation.get("valid"):
            return {
                "status": "success",
                "repairable": True,
                "changed": False,
                "repair_strategy": "none",
                "original_plan": list(tool_path),
                "repaired_plan": list(tool_path),
                "inserted_steps": [],
                "removed_steps": [],
                "repair_operations": [],
                "original_validation": original_validation,
                "validation": original_validation,
            }

        signature_result = get_tool_signatures()
        if signature_result.get("status") != "success":
            return {
                "status": "error",
                "message": signature_result.get("message", "工具签名不可用"),
                "repairable": False,
                "changed": False,
                "inserted_steps": [],
                "removed_steps": [],
                "repair_operations": [],
            }
        signatures = signature_result["signatures"]

        normalized_plan = [
            value.strip() if isinstance(value, str) else ""
            for value in tool_path
        ]
        config = Config()
        max_pruned_steps = config.TYPE_REPAIR_MAX_PRUNED_STEPS
        pruning_result = _find_unique_safe_duplicate_pruning(
            normalized_plan,
            available_input_types,
            original_validation,
            signatures,
            max_pruned_steps,
        )
        if pruning_result.get("status") != "success":
            return {
                "status": "error",
                "message": pruning_result.get("message", "安全剪枝检查失败"),
                "repairable": False,
                "changed": False,
                "inserted_steps": [],
                "removed_steps": [],
                "repair_operations": [],
                "original_validation": original_validation,
            }
        if int(pruning_result.get("candidate_count") or 0) > 1:
            return {
                "status": "success",
                "repairable": False,
                "changed": False,
                "repair_strategy": None,
                "original_plan": list(tool_path),
                "repaired_plan": None,
                "inserted_steps": [],
                "removed_steps": [],
                "repair_operations": [],
                "reason": "存在多个可行剪枝候选，修复器拒绝猜测用户意图",
                "blocking_issue": {
                    "code": "AMBIGUOUS_PRUNING_CANDIDATES",
                    "candidate_count": pruning_result["candidate_count"],
                    "max_pruned_steps": max_pruned_steps,
                },
                "original_validation": original_validation,
            }
        if isinstance(pruning_result.get("candidate"), dict):
            result = _pruning_repair_result(
                list(tool_path),
                original_validation,
                pruning_result["candidate"],
            )
            logger.info(
                "计划安全剪枝完成 | removed=%s | strategy=%s",
                len(result["removed_steps"]),
                result["repair_strategy"],
            )
            return result

        current_types = _normalize_type_collection(available_input_types)
        repaired_plan: list[str] = []
        inserted_steps: list[dict[str, Any]] = []
        max_bridge_hops = config.TYPE_REPAIR_MAX_BRIDGE_HOPS

        for index, raw_tool_name in enumerate(tool_path):
            tool_name = raw_tool_name.strip() if isinstance(raw_tool_name, str) else ""
            if not tool_name or tool_name not in signatures:
                return {
                    "status": "success",
                    "repairable": False,
                    "changed": False,
                    "original_plan": list(tool_path),
                    "repaired_plan": None,
                    "inserted_steps": inserted_steps,
                    "removed_steps": [],
                    "repair_operations": [],
                    "reason": "未知工具不能通过猜测名称修复",
                    "blocking_issue": {
                        "code": "UNKNOWN_TOOL",
                        "step_index": index,
                        "tool_name": raw_tool_name,
                    },
                }

            signature = signatures[tool_name]
            required_types = signature["input_types"]
            has_required_input = all(
                type_name in current_types for type_name in required_types
            )

            if not has_required_input:
                path_result = _find_unique_shortest_bridge_path(
                    current_types,
                    required_types,
                    signatures,
                    max_bridge_hops,
                )
                candidate_paths = path_result.get("candidate_paths") or []
                if (
                    path_result.get("status") != "success"
                    or path_result.get("unique") is not True
                    or not isinstance(path_result.get("path"), list)
                    or not path_result.get("path")
                ):
                    reason = (
                        "存在多个同长最短类型桥接路径，无法确定输入来源"
                        if len(candidate_paths) > 1
                        else (
                            path_result.get("message")
                            or f"在最多{max_bridge_hops}跳内没有确定性的注册类型桥接"
                        )
                    )
                    return {
                        "status": "success",
                        "repairable": False,
                        "changed": False,
                        "original_plan": list(tool_path),
                        "repaired_plan": None,
                        "inserted_steps": inserted_steps,
                        "removed_steps": [],
                        "repair_operations": [],
                        "reason": reason,
                        "blocking_issue": {
                            "code": "MISSING_INPUT_TYPE",
                            "step_index": index,
                            "tool_name": tool_name,
                            "required_types": required_types,
                            "available_types": list(current_types),
                            "bridge_candidates": [
                                {"tool_path": list(path)}
                                for path in candidate_paths
                            ],
                            "max_bridge_hops": max_bridge_hops,
                        },
                    }

                bridge_path = path_result["path"]
                for bridge_offset, bridge_name in enumerate(bridge_path):
                    bridge_signature = signatures[bridge_name]
                    downstream_tool_name = (
                        bridge_path[bridge_offset + 1]
                        if bridge_offset + 1 < len(bridge_path)
                        else tool_name
                    )
                    downstream_required_types = list(
                        signatures[downstream_tool_name].get("input_types") or []
                    )
                    repaired_plan.append(bridge_name)
                    inserted_steps.append(
                        {
                            "before_step_index": index,
                            "tool_name": bridge_name,
                            "reason": (
                                f"使用注册桥接器把 "
                                f"{' + '.join(bridge_signature['input_types'])} "
                                f"转换为 "
                                f"{' + '.join(bridge_signature['output_types'])}"
                            ),
                            "bridge_path_length": len(bridge_path),
                            "max_bridge_hops": max_bridge_hops,
                            "downstream_required_types": downstream_required_types,
                            "output_consumed_downstream": any(
                                output_type in downstream_required_types
                                for output_type in bridge_signature["output_types"]
                            ),
                        }
                    )
                    for output_type in bridge_signature["output_types"]:
                        if output_type not in current_types:
                            current_types.append(output_type)

            repaired_plan.append(tool_name)
            for output_type in signature["output_types"]:
                if output_type not in current_types:
                    current_types.append(output_type)

        repaired_validation = validate_tool_plan(repaired_plan, available_input_types)
        repairable = (
            repaired_validation.get("status") == "success"
            and repaired_validation.get("valid") is True
        )
        logger.info(
            "计划修复完成 | repairable=%s | inserted=%s",
            repairable,
            len(inserted_steps),
        )
        repair_operations = [
            {
                "operation": "deterministic_bridge_insert",
                "tool_name": str(item["tool_name"]),
                "original_step_index": None,
                "before_step_index": int(item["before_step_index"]),
                "reason": str(item["reason"]),
                "issue_code": "MISSING_INPUT_TYPE",
                "input_types": list(
                    signatures[item["tool_name"]].get("input_types") or []
                ),
                "output_types": list(
                    signatures[item["tool_name"]].get("output_types") or []
                ),
                "downstream_required_types": list(
                    item.get("downstream_required_types") or []
                ),
                "registry_opt_in": True,
                "duplicate_group_size": None,
                "output_consumed_downstream": bool(
                    item.get("output_consumed_downstream", False)
                ),
                "candidate_plan_type_valid": repairable,
                "bridge_path_length": int(item["bridge_path_length"]),
            }
            for item in inserted_steps
        ]
        return {
            "status": "success",
            "repairable": repairable,
            "changed": repaired_plan != list(tool_path),
            "repair_strategy": (
                "deterministic_bridge_insertion" if inserted_steps else "none"
            ),
            "original_plan": list(tool_path),
            "repaired_plan": repaired_plan if repairable else None,
            "inserted_steps": inserted_steps,
            "removed_steps": [],
            "repair_operations": repair_operations,
            "original_validation": original_validation,
            "validation": repaired_validation,
        }
    except Exception as exc:
        logger.error("修复工具计划失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"修复工具计划失败: {exc}",
            "repairable": False,
            "changed": False,
            "inserted_steps": [],
            "removed_steps": [],
            "repair_operations": [],
        }
