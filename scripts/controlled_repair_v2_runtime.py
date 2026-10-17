# -*- coding: utf-8 -*-
"""Load and execute controlled repair cases."""

from __future__ import annotations

import copy
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable

from jsonschema import Draft202012Validator

from app.core.config import Config
from app.core.logger import get_logger
from scripts.initial_artifacts import build_initial_artifacts
from scripts.run_controlled_repair import (
    _error_result,
    _execution_summary,
    _issue_codes,
    _load_json_validator,
    _sha256_file,
    _validation_error,
)
from scripts.structured_plan import prepare_structured_plan
from scripts.typed_executor import execute_structured_plan
from scripts.type_checker import (
    AMINO_ACID_SEQUENCE,
    FASTA_RECORD_IDENTIFIER,
    FASTA_RECORD_IDENTIFIER_PAIR,
    MULTI_PROTEIN_FASTA_FILE,
    PAIRWISE_PROTEIN_ALIGNMENT,
    PDB_CHAIN_IDENTIFIER,
    PDB_STRUCTURE_FILE,
    PROTEIN_SEQUENCE_COLLECTION,
    SINGLE_PROTEIN_FASTA_FILE,
    repair_tool_plan,
)
from ToolsAgent.ToolsFuns.Biology import biologyCommon as biology
from ToolsAgent.ToolsFuns.Biology.fasta_analysis import parse_protein_fasta
from ToolsAgent.ToolsFuns.Biology.fasta_collection import (
    parse_multi_protein_fasta,
    select_fasta_record,
)
from ToolsAgent.ToolsFuns.Biology.pdb_analysis import select_pdb_chain
from ToolsAgent.ToolsFuns.Biology.protein_alignment import (
    align_fasta_records,
    summarize_pairwise_alignment,
)


logger = get_logger("ControlledRepairV2Runtime", "controlled_repair_v2.log")
CONDITIONS_V2 = ("E2", "E3")
_PLAN_PLACEHOLDER = "{terminal_tool}"
_SEQUENCE_TERMINALS: dict[str, Callable[[str], dict[str, Any]]] = {
    "compute_gravy": biology.compute_gravy,
    "compute_instability_index": biology.compute_instability_index,
    "compute_extinction_coefficient": biology.compute_extinction_coefficient,
    "compute_pI_mw": biology.compute_pI_mw,
    "predict_solubility": biology.predict_solubility,
}
_INPUT_TYPE_TO_INITIAL_TYPES = {
    "single_protein_fasta_file": [SINGLE_PROTEIN_FASTA_FILE],
    "pdb_file": [PDB_STRUCTURE_FILE, PDB_CHAIN_IDENTIFIER],
    "multi_protein_fasta_file": [MULTI_PROTEIN_FASTA_FILE],
}

ControlledV2Runner = Callable[..., Awaitable[Any]]


def _render_plan(template: Any, terminal_tool: str) -> list[str]:
    if not isinstance(template, list) or not template:
        return []
    rendered: list[str] = []
    for value in template:
        if not isinstance(value, str) or not value.strip():
            return []
        rendered.append(value.replace(_PLAN_PLACEHOLDER, terminal_tool))
    return rendered


def _safe_step_changes(items: Any, *, index_key: str) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return changes
    for item in items:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool_name") or "").strip()
        step_index = item.get(index_key)
        if tool_name and isinstance(step_index, int) and step_index >= 0:
            changes.append({"tool_name": tool_name, "step_index": step_index})
    return changes


def _validate_case_input_shape(
    case: dict[str, Any],
    contract: dict[str, Any],
) -> str | None:
    input_type = contract.get("input_type")
    if input_type == "pdb_file":
        chain_id = case.get("chain_id")
        if not isinstance(chain_id, str) or len(chain_id.strip()) != 1:
            return "PDB家族必须提供单字符chain_id"
    elif input_type == "multi_protein_fasta_file":
        family = str(case.get("family") or "")
        if family in {
            "multi_fasta_parser_omission",
            "multi_fasta_selector_omission",
        }:
            if not isinstance(case.get("record_id"), str) or not case["record_id"].strip():
                return "多记录选择家族必须提供record_id"
        else:
            record_ids = case.get("record_ids")
            if (
                not isinstance(record_ids, list)
                or len(record_ids) != 2
                or len(set(record_ids)) != 2
                or not all(isinstance(value, str) and value.strip() for value in record_ids)
            ):
                return "多记录比对/剪枝家族必须提供两个不同record_id"
    return None


def _materialize_case(
    raw_case: dict[str, Any],
    *,
    contract: dict[str, Any],
    fixture: Path,
) -> dict[str, Any]:
    terminal_tool = str(raw_case.get("terminal_tool") or "").strip()
    injected_plan = _render_plan(
        contract.get("injected_plan_template"),
        terminal_tool,
    )
    repaired_plan = _render_plan(
        contract.get("repaired_plan_template"),
        terminal_tool,
    )
    item = copy.deepcopy(raw_case)
    item["topology_signature"] = str(contract.get("topology_signature") or "")
    item["input_type"] = str(contract.get("input_type") or "")
    item["injected_plan"] = injected_plan
    item["gold"] = {
        "expected_issue_code": "MISSING_INPUT_TYPE",
        "expected_repair_strategy": str(contract.get("repair_strategy") or ""),
        "expected_repaired_plan": repaired_plan,
        "expected_inserted_steps": list(
            contract.get("expected_inserted_steps") or []
        ),
        "expected_removed_steps": list(
            contract.get("expected_removed_steps") or []
        ),
        "expected_behavior": "repair_then_execute",
    }
    initial_types = list(
        _INPUT_TYPE_TO_INITIAL_TYPES.get(item["input_type"]) or []
    )
    if item["input_type"] == "multi_protein_fasta_file":
        if isinstance(item.get("record_id"), str):
            initial_types.append(FASTA_RECORD_IDENTIFIER)
        if isinstance(item.get("record_ids"), list):
            initial_types.append(FASTA_RECORD_IDENTIFIER_PAIR)
    item["_initial_types"] = initial_types
    item["_resolved_fixture"] = fixture
    return item


def _validate_materialized_gold(item: dict[str, Any]) -> dict[str, Any] | None:
    result = repair_tool_plan(item["injected_plan"], item["_initial_types"])
    gold = item["gold"]
    inserted = [
        str(value.get("tool_name") or "")
        for value in result.get("inserted_steps") or []
        if isinstance(value, dict)
    ]
    removed = [
        str(value.get("tool_name") or "")
        for value in result.get("removed_steps") or []
        if isinstance(value, dict)
    ]
    if (
        result.get("status") != "success"
        or result.get("repairable") is not True
        or result.get("changed") is not True
        or result.get("repair_strategy") != gold["expected_repair_strategy"]
        or result.get("repaired_plan") != gold["expected_repaired_plan"]
        or inserted != gold["expected_inserted_steps"]
        or removed != gold["expected_removed_steps"]
    ):
        return _error_result(
            "v2冻结Gold与当前类型图不一致",
            "DATASET_GOLD_MISMATCH",
            details={"case_id": item.get("id")},
        )
    return None


def load_controlled_repair_v2_dataset(
    dataset_path: Any = None,
    *,
    config: Any = None,
) -> dict[str, Any]:
    try:
        cfg = config if config is not None else Config()
        path = Path(
            str(
                dataset_path
                or getattr(cfg, "CONTROLLED_REPAIR_V2_DATASET_PATH", "")
            )
        ).expanduser().resolve()
        schema_path = Path(
            str(getattr(cfg, "CONTROLLED_REPAIR_V2_DATASET_SCHEMA_PATH", ""))
        ).expanduser().resolve()
        if not path.is_file():
            return _error_result("Controlled Repair v2数据集不存在", "DATASET_NOT_FOUND")
        validator_result = _load_json_validator(schema_path, "v2数据集")
        if validator_result.get("status") != "success":
            return validator_result
        dataset = json.loads(path.read_text(encoding="utf-8"))
        schema_error = _validation_error(
            validator_result["validator"],
            dataset,
            code="DATASET_SCHEMA_VALIDATION_FAILED",
            label="Controlled Repair v2数据集",
        )
        if schema_error:
            return schema_error

        contracts: dict[str, dict[str, Any]] = {}
        signatures: set[str] = set()
        contract_shapes: set[tuple[Any, ...]] = set()
        for contract in dataset.get("family_contracts") or []:
            family = str(contract.get("family") or "")
            signature = str(contract.get("topology_signature") or "")
            contract_shape = (
                str(contract.get("input_type") or ""),
                tuple(contract.get("injected_plan_template") or []),
                tuple(contract.get("repaired_plan_template") or []),
                str(contract.get("repair_strategy") or ""),
                tuple(contract.get("expected_inserted_steps") or []),
                tuple(contract.get("expected_removed_steps") or []),
            )
            if (
                family in contracts
                or signature in signatures
                or contract_shape in contract_shapes
            ):
                return _error_result(
                    "v2家族名称、拓扑签名与操作结构必须分别唯一",
                    "DUPLICATE_FAMILY_CONTRACT",
                )
            contracts[family] = copy.deepcopy(contract)
            signatures.add(signature)
            contract_shapes.add(contract_shape)
        if not 4 <= len(contracts) <= 6:
            return _error_result(
                "v2必须包含4至6个非同构机制家族",
                "INVALID_FAMILY_COUNT",
            )

        fixture_catalog = dataset.get("fixtures") or {}
        dataset_dir = path.parent.resolve()
        resolved_fixtures: dict[str, Path] = {}
        for fixture_ref, fixture_info in fixture_catalog.items():
            fixture = (
                dataset_dir / str((fixture_info or {}).get("path") or "")
            ).resolve()
            try:
                fixture.relative_to(dataset_dir)
            except ValueError:
                return _error_result(
                    "v2 fixture必须位于受控数据目录内",
                    "FIXTURE_PATH_ESCAPE",
                    details={"fixture_ref": fixture_ref},
                )
            if not fixture.is_file():
                return _error_result(
                    "v2 fixture不存在",
                    "FIXTURE_NOT_FOUND",
                    details={"fixture_ref": fixture_ref},
                )
            if _sha256_file(fixture) != fixture_info.get("sha256"):
                return _error_result(
                    "v2 fixture指纹不匹配",
                    "FIXTURE_HASH_MISMATCH",
                    details={"fixture_ref": fixture_ref},
                )
            resolved_fixtures[str(fixture_ref)] = fixture

        case_ids: set[str] = set()
        case_design_keys: set[tuple[Any, ...]] = set()
        family_counts: Counter[str] = Counter()
        materialized: list[dict[str, Any]] = []
        for raw_case in dataset.get("cases") or []:
            case_id = str(raw_case.get("id") or "")
            family = str(raw_case.get("family") or "")
            fixture_ref = str(raw_case.get("fixture_ref") or "")
            contract = contracts.get(family)
            if case_id in case_ids:
                return _error_result(
                    "v2包含重复case_id",
                    "DUPLICATE_CASE_ID",
                    details={"case_id": case_id},
                )
            case_ids.add(case_id)
            case_design_key = (
                family,
                fixture_ref,
                str(raw_case.get("terminal_tool") or ""),
                str(raw_case.get("chain_id") or ""),
                str(raw_case.get("record_id") or ""),
                tuple(raw_case.get("record_ids") or []),
            )
            if case_design_key in case_design_keys:
                return _error_result(
                    "v2同一家族不得用完全相同的输入与计划重复扩充分母",
                    "DUPLICATE_CASE_DESIGN",
                    details={"case_id": case_id},
                )
            case_design_keys.add(case_design_key)
            if not contract or fixture_ref not in resolved_fixtures:
                return _error_result(
                    "v2 case引用未知家族或fixture",
                    "DATASET_REFERENCE_ERROR",
                    details={"case_id": case_id},
                )
            shape_error = _validate_case_input_shape(raw_case, contract)
            if shape_error:
                return _error_result(
                    shape_error,
                    "DATASET_INPUT_CONTRACT_MISMATCH",
                    details={"case_id": case_id},
                )
            terminal_tool = str(raw_case.get("terminal_tool") or "")
            if terminal_tool not in _SEQUENCE_TERMINALS and terminal_tool != "summarize_pairwise_alignment":
                return _error_result(
                    "v2 case使用了未授权终端工具",
                    "UNSUPPORTED_TERMINAL_TOOL",
                    details={"case_id": case_id},
                )
            if contract.get("input_type") == "multi_protein_fasta_file" and (
                isinstance(raw_case.get("record_ids"), list)
                != (terminal_tool == "summarize_pairwise_alignment")
            ):
                return _error_result(
                    "多记录case的标识符类型与终端工具不一致",
                    "DATASET_INPUT_CONTRACT_MISMATCH",
                    details={"case_id": case_id},
                )
            item = _materialize_case(
                raw_case,
                contract=contract,
                fixture=resolved_fixtures[fixture_ref],
            )
            gold_error = _validate_materialized_gold(item)
            if gold_error:
                return gold_error
            materialized.append(item)
            family_counts[family] += 1

        if set(family_counts) != set(contracts) or any(
            count < 5 for count in family_counts.values()
        ):
            return _error_result(
                "每个v2机制家族至少需要5个配对机会",
                "INSUFFICIENT_FAMILY_COVERAGE",
                details={"family_counts": dict(sorted(family_counts.items()))},
            )
        return {
            "status": "success",
            "dataset_path": str(path),
            "dataset_sha256": _sha256_file(path),
            "version": str(dataset.get("version") or ""),
            "evidence_scope": str(dataset.get("evidence_scope") or ""),
            "conditions": list(dataset.get("conditions") or []),
            "family_counts": dict(sorted(family_counts.items())),
            "family_count": len(family_counts),
            "items": materialized,
        }
    except Exception as exc:
        logger.error("加载Controlled Repair v2失败: %s", exc, exc_info=True)
        return _error_result(
            f"加载Controlled Repair v2失败: {type(exc).__name__}",
            "DATASET_LOAD_EXCEPTION",
        )


async def local_controlled_v2_runner(
    *,
    tool_name: str,
    tool_args: str,
    file_path_list: list[str],
    typed_inputs: dict[str, Any],
) -> dict[str, Any]:
    del tool_args, file_path_list
    try:
        if tool_name == "parse_protein_fasta":
            return parse_protein_fasta(typed_inputs[SINGLE_PROTEIN_FASTA_FILE])
        if tool_name == "select_pdb_chain":
            return select_pdb_chain(
                typed_inputs[PDB_STRUCTURE_FILE],
                typed_inputs[PDB_CHAIN_IDENTIFIER],
            )
        if tool_name == "parse_multi_protein_fasta":
            return parse_multi_protein_fasta(typed_inputs[MULTI_PROTEIN_FASTA_FILE])
        if tool_name == "select_fasta_record":
            return select_fasta_record(
                typed_inputs[PROTEIN_SEQUENCE_COLLECTION],
                typed_inputs[FASTA_RECORD_IDENTIFIER],
            )
        if tool_name == "align_fasta_records":
            return align_fasta_records(
                typed_inputs[PROTEIN_SEQUENCE_COLLECTION],
                typed_inputs[FASTA_RECORD_IDENTIFIER_PAIR],
            )
        if tool_name == "summarize_pairwise_alignment":
            return summarize_pairwise_alignment(
                typed_inputs[PAIRWISE_PROTEIN_ALIGNMENT]
            )
        function = _SEQUENCE_TERMINALS.get(tool_name)
        if function is not None:
            return function(typed_inputs[AMINO_ACID_SEQUENCE])
        return {
            "status": "error",
            "message": "v2离线Runner不支持该工具",
            "detail_code": "UNSUPPORTED_CONTROLLED_TOOL",
        }
    except Exception as exc:
        logger.error("v2本地工具异常 | tool=%s | error=%s", tool_name, exc)
        return {
            "status": "error",
            "message": f"v2本地工具执行异常: {type(exc).__name__}",
            "detail_code": "CONTROLLED_TOOL_EXCEPTION",
        }


def _build_v2_initial_result(case: dict[str, Any]) -> dict[str, Any]:
    input_type = case.get("input_type")
    if input_type == "single_protein_fasta_file":
        question = "受控修复v2输入：单记录蛋白质FASTA。"
    elif input_type == "pdb_file":
        question = f"受控修复v2输入：PDB，chain_id={case.get('chain_id', '')}。"
    elif isinstance(case.get("record_id"), str):
        question = f"受控修复v2输入：record_id={case['record_id']}。"
    else:
        record_ids = list(case.get("record_ids") or [])
        question = f"受控修复v2输入：record_ids={','.join(record_ids)}。"
    return build_initial_artifacts(question, [str(case["_resolved_fixture"])])


async def run_controlled_v2_unit(
    case: dict[str, Any],
    *,
    condition: str,
    repeat_index: int,
    experiment_id: str,
    context: dict[str, str],
    trace_validator: Draft202012Validator,
    runner: ControlledV2Runner = local_controlled_v2_runner,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if condition not in CONDITIONS_V2:
        return _error_result("v2实验条件必须是E2或E3", "INVALID_CONDITION")
    initial_result = _build_v2_initial_result(case)
    if initial_result.get("status") != "success":
        issue_codes = _issue_codes(initial_result.get("issues"))
        return _error_result(
            "v2初始Artifact构建失败",
            issue_codes[0] if issue_codes else "INITIAL_ARTIFACT_FAILURE",
            details={"case_id": case.get("id"), "condition": condition},
        )
    injected_plan = list(case.get("injected_plan") or [])
    initial_types = list(initial_result.get("available_input_types") or [])
    planning = prepare_structured_plan(
        injected_plan,
        initial_types,
        enable_repair=condition == "E3",
    )
    validation_before = planning.get("validation_before") or {}
    issue_codes_before = _issue_codes(validation_before.get("issues"))
    final_result = planning.get("final_result") or {}
    issue_codes_after = _issue_codes(final_result.get("issues"))
    repair_result = planning.get("repair_result") or {}
    inserted_steps = _safe_step_changes(
        repair_result.get("inserted_steps"),
        index_key="before_step_index",
    )
    removed_steps = _safe_step_changes(
        repair_result.get("removed_steps"),
        index_key="original_step_index",
    )

    if condition == "E2":
        execution = {
            "status": "blocked",
            "completed": False,
            "error_type": "type_check_block",
            "issues": [{"code": "TYPE_INVALID_PLAN", "message": "E2按契约阻断"}],
            "executed_steps": [],
            "tool_results": [],
        }
    elif planning.get("valid") is True:
        execution = await execute_structured_plan(
            planning.get("structured_plan") or [],
            initial_result.get("artifacts") or {},
            runner=runner,
        )
    else:
        execution = {
            "status": "error",
            "completed": False,
            "error_type": "plan_failure",
            "issues": list(planning.get("issues") or []),
            "executed_steps": [],
            "tool_results": [],
        }

    gold = case.get("gold") or {}
    repair_opportunity = bool(
        validation_before.get("valid") is False
        and gold.get("expected_issue_code") in issue_codes_before
    )
    repair_correct = None
    if condition == "E3":
        repair_correct = bool(
            planning.get("repair_applied") is True
            and planning.get("repair_strategy")
            == gold.get("expected_repair_strategy")
            and planning.get("plan_after") == gold.get("expected_repaired_plan")
            and [item["tool_name"] for item in inserted_steps]
            == gold.get("expected_inserted_steps")
            and [item["tool_name"] for item in removed_steps]
            == gold.get("expected_removed_steps")
        )
    execution_summary = _execution_summary(execution)
    recovery_success = bool(
        condition == "E3"
        and repair_opportunity
        and repair_correct
        and planning.get("valid") is True
        and execution_summary["completed"]
    )
    trace = {
        "schema_version": "2.0",
        "experiment_id": experiment_id,
        "case_id": str(case.get("id") or ""),
        "family": str(case.get("family") or ""),
        "topology_signature": str(case.get("topology_signature") or ""),
        "condition": condition,
        "repeat_index": int(repeat_index),
        "context": copy.deepcopy(context),
        "plan": {
            "injected_plan": injected_plan,
            "initial_types": initial_types,
            "type_valid_before": validation_before.get("valid") is True,
            "issue_codes_before": issue_codes_before,
            "repair_attempted": bool(planning.get("repair_attempted", False)),
            "repair_applied": bool(planning.get("repair_applied", False)),
            "repair_strategy": (
                str(planning.get("repair_strategy"))
                if planning.get("repair_strategy") not in {None, "none"}
                else None
            ),
            "repaired_plan": list(planning.get("plan_after") or injected_plan),
            "type_valid_after": planning.get("valid") is True,
            "issue_codes_after": issue_codes_after,
            "inserted_steps": inserted_steps,
            "removed_steps": removed_steps,
        },
        "execution": execution_summary,
        "evaluation": {
            "repair_opportunity": repair_opportunity,
            "repair_correct": repair_correct,
            "recovery_success": recovery_success,
        },
        "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
    }
    schema_error = _validation_error(
        trace_validator,
        trace,
        code="TRACE_SCHEMA_VALIDATION_FAILED",
        label="Controlled Repair v2 Trace",
    )
    if schema_error:
        return schema_error
    return {"status": "success", "trace": trace}
