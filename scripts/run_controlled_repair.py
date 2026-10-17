# -*- coding: utf-8 -*-
"""Run the version 1 controlled repair suite."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from jsonschema import Draft202012Validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Config
from app.core.logger import get_logger
from scripts.initial_artifacts import build_initial_artifacts
from scripts.structured_plan import prepare_structured_plan
from scripts.typed_executor import execute_structured_plan
from scripts.type_checker import (
    AMINO_ACID_SEQUENCE,
    PDB_CHAIN_IDENTIFIER,
    PDB_STRUCTURE_FILE,
    SINGLE_PROTEIN_FASTA_FILE,
)
from ToolsAgent.ToolsFuns.Biology import biologyCommon as biology
from ToolsAgent.ToolsFuns.Biology.fasta_analysis import parse_protein_fasta
from ToolsAgent.ToolsFuns.Biology.pdb_analysis import select_pdb_chain


logger = get_logger("ControlledRepair", "controlled_repair.log")
_EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CONDITIONS = ("B1", "B2")
_EXPECTED_FAMILY_CONTRACTS = {
    "fasta_to_sequence": {
        "input_type": "protein_fasta_file",
        "initial_types": [SINGLE_PROTEIN_FASTA_FILE],
        "bridge": "parse_protein_fasta",
    },
    "pdb_chain_to_sequence": {
        "input_type": "pdb_file",
        "initial_types": [PDB_STRUCTURE_FILE, PDB_CHAIN_IDENTIFIER],
        "bridge": "select_pdb_chain",
    },
}
_SEQUENCE_TOOL_FUNCTIONS: dict[str, Callable[[str], dict[str, Any]]] = {
    "compute_gravy": biology.compute_gravy,
    "compute_instability_index": biology.compute_instability_index,
    "compute_extinction_coefficient": biology.compute_extinction_coefficient,
    "compute_pI_mw": biology.compute_pI_mw,
    "predict_signalpeptide": biology.predict_signalpeptide,
    "predict_solubility": biology.predict_solubility,
}

ControlledRunner = Callable[..., Awaitable[Any]]


def _error_result(
    message: str,
    code: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        return {
            "status": "error",
            "message": str(message),
            "completed": False,
            "error_type": "controlled_repair_failure",
            "detail_code": str(code),
            "issues": [{"code": str(code), "message": str(message)}],
            "details": copy.deepcopy(details or {}),
        }
    except Exception:
        return {
            "status": "error",
            "message": "构造受控修复错误结构失败",
            "completed": False,
            "error_type": "controlled_repair_failure",
            "detail_code": "ERROR_RESULT_FAILURE",
            "issues": [],
            "details": {},
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_validator(path: Path, label: str) -> dict[str, Any]:
    try:
        if not path.is_file():
            return _error_result(f"{label} Schema 不存在", "SCHEMA_NOT_FOUND")
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        return {
            "status": "success",
            "validator": Draft202012Validator(schema),
            "schema_version": str(schema.get("x-version") or ""),
        }
    except Exception as exc:
        logger.error("加载%s Schema失败: %s", label, exc, exc_info=True)
        return _error_result(
            f"{label} Schema 无效: {exc}",
            "INVALID_SCHEMA",
        )


def _validation_error(
    validator: Draft202012Validator,
    instance: Any,
    *,
    code: str,
    label: str,
) -> dict[str, Any] | None:
    errors = sorted(
        validator.iter_errors(instance),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if not errors:
        return None
    first = errors[0]
    return _error_result(
        f"{label} 不符合冻结 Schema: {first.message}",
        code,
        details={
            "schema_path": ".".join(str(part) for part in first.absolute_path),
        },
    )


def load_controlled_repair_dataset(
    dataset_path: Any = None,
    *,
    config: Any = None,
) -> dict[str, Any]:
    try:
        cfg = config if config is not None else Config()
        path = Path(
            str(dataset_path or getattr(cfg, "CONTROLLED_REPAIR_DATASET_PATH", ""))
        ).expanduser().resolve()
        schema_path = Path(
            str(getattr(cfg, "CONTROLLED_REPAIR_DATASET_SCHEMA_PATH", ""))
        ).expanduser().resolve()
        if not path.is_file():
            return _error_result("受控修复数据集不存在", "DATASET_NOT_FOUND")

        validator_result = _load_json_validator(schema_path, "数据集")
        if validator_result.get("status") != "success":
            return validator_result
        dataset = json.loads(path.read_text(encoding="utf-8"))
        schema_error = _validation_error(
            validator_result["validator"],
            dataset,
            code="DATASET_SCHEMA_VALIDATION_FAILED",
            label="受控修复数据集",
        )
        if schema_error:
            return schema_error

        case_ids: set[str] = set()
        resolved_items: list[dict[str, Any]] = []
        dataset_dir = path.parent.resolve()
        for raw_item in dataset.get("items") or []:
            case_id = str(raw_item.get("id") or "")
            if case_id in case_ids:
                return _error_result(
                    "受控修复数据集包含重复 case_id",
                    "DUPLICATE_CASE_ID",
                    details={"case_id": case_id},
                )
            case_ids.add(case_id)

            family = str(raw_item.get("family") or "")
            contract = _EXPECTED_FAMILY_CONTRACTS.get(family)
            input_data = raw_item.get("input") or {}
            gold = raw_item.get("gold") or {}
            if not contract or input_data.get("input_type") != contract["input_type"]:
                return _error_result(
                    "样例 family 与 input_type 不一致",
                    "DATASET_SEMANTIC_MISMATCH",
                    details={"case_id": case_id},
                )
            if gold.get("expected_bridge") != contract["bridge"]:
                return _error_result(
                    "样例 family 与期望桥接不一致",
                    "DATASET_SEMANTIC_MISMATCH",
                    details={"case_id": case_id},
                )
            expected_plan = [contract["bridge"], *(raw_item.get("injected_plan") or [])]
            if gold.get("expected_repaired_plan") != expected_plan:
                return _error_result(
                    "Gold 修复计划不是桥接加原计划的确定性组合",
                    "DATASET_GOLD_MISMATCH",
                    details={"case_id": case_id},
                )

            fixture = (dataset_dir / str(input_data.get("fixture") or "")).resolve()
            try:
                fixture.relative_to(dataset_dir)
            except ValueError:
                return _error_result(
                    "fixture 必须位于受控数据集目录内",
                    "FIXTURE_PATH_ESCAPE",
                    details={"case_id": case_id},
                )
            if not fixture.is_file():
                return _error_result(
                    "受控 fixture 不存在",
                    "FIXTURE_NOT_FOUND",
                    details={"case_id": case_id},
                )
            actual_sha256 = _sha256_file(fixture)
            if actual_sha256 != input_data.get("fixture_sha256"):
                return _error_result(
                    "受控 fixture 指纹与冻结数据集不一致",
                    "FIXTURE_HASH_MISMATCH",
                    details={"case_id": case_id},
                )

            item = copy.deepcopy(raw_item)
            item["_resolved_fixture"] = fixture
            item["_initial_types"] = list(contract["initial_types"])
            resolved_items.append(item)

        return {
            "status": "success",
            "dataset_path": str(path),
            "dataset_sha256": _sha256_file(path),
            "version": dataset["version"],
            "evidence_scope": dataset["evidence_scope"],
            "conditions": list(dataset["conditions"]),
            "items": resolved_items,
        }
    except Exception as exc:
        logger.error("加载受控修复数据集失败: %s", exc, exc_info=True)
        return _error_result(
            f"加载受控修复数据集失败: {exc}",
            "DATASET_LOAD_EXCEPTION",
        )


async def local_controlled_runner(
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
        function = _SEQUENCE_TOOL_FUNCTIONS.get(tool_name)
        if function is not None:
            return function(typed_inputs[AMINO_ACID_SEQUENCE])
        return {
            "status": "error",
            "message": "受控离线 Runner 不支持该工具",
            "detail_code": "UNSUPPORTED_CONTROLLED_TOOL",
        }
    except Exception as exc:
        logger.error("受控本地工具执行异常 | tool=%s | error=%s", tool_name, exc)
        return {
            "status": "error",
            "message": f"受控本地工具执行异常: {type(exc).__name__}",
            "detail_code": "CONTROLLED_TOOL_EXCEPTION",
        }


def _build_initial_result(case: dict[str, Any]) -> dict[str, Any]:
    if case.get("family") == "fasta_to_sequence":
        question = "受控修复输入：单记录蛋白质 FASTA。"
    else:
        chain_id = str((case.get("input") or {}).get("chain_id") or "")
        question = f"受控修复输入：PDB，chain_id={chain_id}。"
    return build_initial_artifacts(question, [str(case["_resolved_fixture"])])


def _issue_codes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(
        {
            str(item.get("code"))
            for item in value
            if isinstance(item, dict) and item.get("code")
        }
    )


def _execution_summary(execution: dict[str, Any]) -> dict[str, Any]:
    executed_steps = execution.get("executed_steps") or []
    tool_results = execution.get("tool_results") or []
    attempted = len(tool_results)
    successful = len(executed_steps)
    issues = execution.get("issues") or []
    detail_code = (
        str(issues[0].get("code"))
        if issues and isinstance(issues[0], dict) and issues[0].get("code")
        else None
    )
    return {
        "completed": bool(execution.get("completed", False)),
        "attempted_steps": attempted,
        "successful_steps": successful,
        "failed_steps": max(0, attempted - successful),
        "error_type": (
            str(execution.get("error_type"))
            if execution.get("error_type")
            else None
        ),
        "detail_code": detail_code,
    }


async def run_controlled_unit(
    case: dict[str, Any],
    *,
    condition: str,
    repeat_index: int,
    experiment_id: str,
    context: dict[str, str],
    trace_validator: Draft202012Validator,
    runner: ControlledRunner = local_controlled_runner,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if condition not in _CONDITIONS:
        return _error_result("实验条件必须是 B1 或 B2", "INVALID_CONDITION")

    initial_result = _build_initial_result(case)
    if initial_result.get("status") != "success":
        issue_codes = _issue_codes(initial_result.get("issues"))
        return _error_result(
            "受控样例初始 Artifact 构建失败",
            issue_codes[0] if issue_codes else "INITIAL_ARTIFACT_FAILURE",
            details={"case_id": case.get("id"), "condition": condition},
        )

    injected_plan = list(case.get("injected_plan") or [])
    initial_types = list(initial_result.get("available_input_types") or [])
    planning = prepare_structured_plan(
        injected_plan,
        initial_types,
        enable_repair=condition == "B2",
    )
    validation_before = planning.get("validation_before") or {}
    issue_codes_before = _issue_codes(validation_before.get("issues"))
    final_result = planning.get("final_result") or {}
    issue_codes_after = _issue_codes(final_result.get("issues"))
    repair_result = planning.get("repair_result") or {}
    inserted_steps = [
        {
            "tool_name": str(item.get("tool_name") or ""),
            "before_step_index": int(item.get("before_step_index", 0)),
        }
        for item in repair_result.get("inserted_steps") or []
        if isinstance(item, dict)
    ]

    if condition == "B1":
        execution = {
            "status": "blocked",
            "completed": False,
            "error_type": "type_check_block",
            "issues": [
                {
                    "code": "TYPE_INVALID_PLAN",
                    "message": "B1 按契约在非法计划执行前阻断",
                }
            ],
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
    repair_opportunity = (
        validation_before.get("valid") is False
        and gold.get("expected_issue_code") in issue_codes_before
    )
    repair_correct = None
    if condition == "B2":
        repair_correct = bool(
            planning.get("repair_applied") is True
            and planning.get("plan_after") == gold.get("expected_repaired_plan")
            and len(inserted_steps) == 1
            and inserted_steps[0].get("tool_name") == gold.get("expected_bridge")
        )
    execution_summary = _execution_summary(execution)
    recovery_success = bool(
        condition == "B2"
        and repair_opportunity
        and repair_correct
        and planning.get("valid") is True
        and execution_summary["completed"]
    )

    trace = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "case_id": str(case.get("id") or ""),
        "family": str(case.get("family") or ""),
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
            "repaired_plan": list(planning.get("plan_after") or injected_plan),
            "type_valid_after": planning.get("valid") is True,
            "issue_codes_after": issue_codes_after,
            "inserted_steps": inserted_steps,
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
        label="受控修复 Trace",
    )
    if schema_error:
        return schema_error
    return {"status": "success", "trace": trace}


def _git_commit(project_root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().lower()
        if re.fullmatch(r"[0-9a-f]{7,40}", value):
            return value
    except Exception as exc:
        logger.warning("读取Git提交失败: %s", exc)
    return "0000000"


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "estimability_status": "estimable" if denominator > 0 else "not_estimable",
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": round(numerator / denominator, 6) if denominator > 0 else None,
    }


def _build_metrics(
    traces: list[dict[str, Any]],
    *,
    planned_pairs: int,
) -> dict[str, Any]:
    pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for trace in traces:
        key = (trace["case_id"], trace["repeat_index"])
        pairs.setdefault(key, {})[trace["condition"]] = trace
    complete_pairs = [value for value in pairs.values() if set(value) == set(_CONDITIONS)]
    opportunity_pairs = [
        pair
        for pair in complete_pairs
        if pair["B1"]["evaluation"]["repair_opportunity"]
        and pair["B2"]["evaluation"]["repair_opportunity"]
    ]
    recovered = sum(
        bool(pair["B2"]["evaluation"]["recovery_success"])
        for pair in opportunity_pairs
    )
    applied = [
        trace
        for trace in traces
        if trace["condition"] == "B2" and trace["plan"]["repair_applied"]
    ]
    correct = sum(trace["evaluation"]["repair_correct"] is True for trace in applied)
    return {
        "pair_coverage": _rate(len(complete_pairs), planned_pairs),
        "repair_recovery_rate": _rate(recovered, len(opportunity_pairs)),
        "repair_precision": _rate(correct, len(applied)),
    }


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Hard-link creation prevents concurrent trace overwrites.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        # Use the same atomic no-overwrite operation as JSONL.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


async def run_controlled_repair_experiment(
    *,
    experiment_id: Any,
    dataset_path: Any = None,
    output_path: Any = None,
    repeats: Any = 1,
    config: Any = None,
    git_commit: Any = None,
    runner: ControlledRunner = local_controlled_runner,
) -> dict[str, Any]:
    try:
        cfg = config if config is not None else Config()
        normalized_id = str(experiment_id or "").strip()
        if not _EXPERIMENT_ID_PATTERN.fullmatch(normalized_id):
            return _error_result("experiment_id 格式无效", "INVALID_EXPERIMENT_ID")
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
            return _error_result("repeats 必须是正整数", "INVALID_REPEATS")
        if not callable(runner):
            return _error_result("runner 必须是异步可调用对象", "INVALID_RUNNER")

        dataset_result = load_controlled_repair_dataset(dataset_path, config=cfg)
        if dataset_result.get("status") != "success":
            return dataset_result
        trace_validator_result = _load_json_validator(
            Path(str(getattr(cfg, "CONTROLLED_REPAIR_TRACE_SCHEMA_PATH", "")))
            .expanduser()
            .resolve(),
            "Trace",
        )
        if trace_validator_result.get("status") != "success":
            return trace_validator_result

        if output_path is None:
            trace_path = (
                Path(str(getattr(cfg, "CONTROLLED_REPAIR_RESULTS_DIR", "")))
                .expanduser()
                .resolve()
                / f"{normalized_id}.jsonl"
            )
        else:
            trace_path = Path(str(output_path)).expanduser().resolve()
        if trace_path.suffix.lower() != ".jsonl":
            return _error_result("Trace 输出必须使用 .jsonl", "INVALID_OUTPUT_PATH")
        manifest_path = trace_path.with_suffix(".manifest.json")
        failure_path = trace_path.with_suffix(".failures.jsonl")
        existing = [path.name for path in (trace_path, manifest_path, failure_path) if path.exists()]
        if existing:
            return _error_result(
                "实验输出已存在；no-overwrite 规则禁止覆盖",
                "OUTPUT_EXISTS",
                details={"existing_files": existing},
            )
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        commit = str(git_commit or "").strip().lower() or _git_commit(PROJECT_ROOT)
        if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
            return _error_result("git_commit 格式无效", "INVALID_GIT_COMMIT")
        context = {
            "dataset_version": dataset_result["version"],
            "dataset_sha256": dataset_result["dataset_sha256"],
            "git_commit": commit,
            "python_version": platform.python_version(),
        }
        try:
            timeout_seconds = float(
                getattr(cfg, "CONTROLLED_REPAIR_TIMEOUT_SECONDS", 30.0)
            )
        except (TypeError, ValueError):
            timeout_seconds = 30.0
        timeout_seconds = min(300.0, max(1.0, timeout_seconds))

        traces: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        started_at = datetime.now(timezone.utc).isoformat()
        for repeat_index in range(repeats):
            for case in dataset_result["items"]:
                for condition in _CONDITIONS:
                    try:
                        unit = await asyncio.wait_for(
                            run_controlled_unit(
                                case,
                                condition=condition,
                                repeat_index=repeat_index,
                                experiment_id=normalized_id,
                                context=context,
                                trace_validator=trace_validator_result["validator"],
                                runner=runner,
                            ),
                            timeout=timeout_seconds,
                        )
                        if unit.get("status") == "success":
                            traces.append(unit["trace"])
                        else:
                            failures.append(
                                {
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "experiment_id": normalized_id,
                                    "case_id": str(case.get("id") or ""),
                                    "condition": condition,
                                    "repeat_index": repeat_index,
                                    "error_code": str(unit.get("detail_code") or "UNIT_FAILURE"),
                                    "exception_type": None,
                                }
                            )
                    except asyncio.TimeoutError:
                        failures.append(
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "experiment_id": normalized_id,
                                "case_id": str(case.get("id") or ""),
                                "condition": condition,
                                "repeat_index": repeat_index,
                                "error_code": "UNIT_TIMEOUT",
                                "exception_type": "TimeoutError",
                            }
                        )
                    except Exception as exc:
                        logger.error(
                            "受控实验单元异常 | case=%s | condition=%s | error=%s",
                            case.get("id"),
                            condition,
                            exc,
                            exc_info=True,
                        )
                        failures.append(
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "experiment_id": normalized_id,
                                "case_id": str(case.get("id") or ""),
                                "condition": condition,
                                "repeat_index": repeat_index,
                                "error_code": "UNIT_EXCEPTION",
                                "exception_type": type(exc).__name__,
                            }
                        )

        planned_pairs = len(dataset_result["items"]) * repeats
        planned_units = planned_pairs * len(_CONDITIONS)
        metrics = _build_metrics(traces, planned_pairs=planned_pairs)
        manifest = {
            "schema_version": "1.0",
            "experiment_id": normalized_id,
            "status": "completed" if not failures else "completed_with_failures",
            "design": "paired_frozen_invalid_plan_b1_vs_b2",
            "evidence_scope": dataset_result["evidence_scope"],
            "dataset_version": dataset_result["version"],
            "dataset_sha256": dataset_result["dataset_sha256"],
            "git_commit": commit,
            "python_version": platform.python_version(),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "repeats": repeats,
            "planned_units": planned_units,
            "completed_units": len(traces),
            "failed_units": len(failures),
            "automatic_retries": 0,
            "metrics": metrics,
        }

        _write_jsonl_atomic(trace_path, traces)
        _write_jsonl_atomic(failure_path, failures)
        _write_json_atomic(manifest_path, manifest)
        return {
            "status": "success",
            "completed": True,
            "experiment_id": normalized_id,
            "trace_path": str(trace_path),
            "manifest_path": str(manifest_path),
            "failure_path": str(failure_path),
            "manifest": manifest,
        }
    except Exception as exc:
        logger.error("运行受控修复实验失败: %s", exc, exc_info=True)
        return _error_result(
            f"运行受控修复实验失败: {exc}",
            "CONTROLLED_EXPERIMENT_EXCEPTION",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Controlled Repair Track v1")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    try:
        args = _parse_args()
        result = asyncio.run(
            run_controlled_repair_experiment(
                experiment_id=args.experiment_id,
                dataset_path=args.dataset,
                output_path=args.output,
                repeats=args.repeats,
            )
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "success" else 1
    except Exception as exc:
        print(
            json.dumps(
                _error_result(
                    f"受控修复命令行异常: {type(exc).__name__}",
                    "CONTROLLED_CLI_EXCEPTION",
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
