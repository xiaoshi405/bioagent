# -*- coding: utf-8 -*-
"""Run resumable component experiments and score traces."""

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
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from jsonschema import Draft202012Validator

from app.core.config import Config
from app.core.logger import get_logger
from scripts.evaluation_scoring import (
    SCORING_POLICY_VERSION,
    apply_scoring_to_trace,
    persist_scoring_bundle,
    score_experiment_trace,
)
from scripts.evaluation_trace import (
    EXPERIMENT_CONDITIONS,
    append_experiment_trace,
    build_experiment_trace,
    get_experiment_condition,
    load_experiment_traces,
)
from scripts.initial_artifacts import build_initial_artifacts
from scripts.hierarchical_recovery import prepare_hierarchical_recovery
from scripts.baseline_strategies import (
    LLM_REPLAN_BASELINE,
    SCHEMA_RUNTIME_BASELINE,
    build_schema_rejection_result,
    prepare_matched_budget_replan,
    validate_schema_runtime_plan,
)
from scripts.plan_sampling import (
    append_plan_sample,
    build_plan_sample_id,
    load_plan_samples,
    safe_plan_trace,
    sha256_text,
    validate_frozen_plan,
)


logger = get_logger("BioToolEvalRunner", "biotooleval_runner.log")

AgentCallable = Callable[[str, list[str], bool], Awaitable[dict[str, Any]]]
PlanCallable = Callable[..., Awaitable[dict[str, Any]]]
_SAFE_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_FORMAL_DATASET_STATUSES = {"benchmark_ready", "frozen", "released"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _condition_order_for_repeat(
    conditions: list[str] | tuple[str, ...],
    repeat_index: int,
) -> list[str]:
    try:
        ordered = [str(value).strip() for value in conditions]
        if not ordered:
            return []
        offset = int(repeat_index) % len(ordered)
        return ordered[offset:] + ordered[:offset]
    except Exception as exc:
        logger.error("构造条件反平衡顺序失败: %s", exc, exc_info=True)
        return [str(value).strip() for value in conditions]


def _error_result(
    message: str,
    code: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        issue = {"code": code, "message": message}
        if isinstance(details, dict):
            issue.update(copy.deepcopy(details))
        return {
            "status": "error",
            "message": message,
            "issues": [issue],
        }
    except Exception as exc:
        logger.error("构造运行器错误结构失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": str(exc),
            "issues": [],
        }


def _utc_now_text() -> str:
    try:
        return datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        logger.error("生成UTC时间失败: %s", exc, exc_info=True)
        return "unknown"


def _execution_key(
    question_id: Any,
    model: Any,
    condition: Any,
    repeat_index: Any,
) -> tuple[str, str, str, int]:
    try:
        return (
            str(question_id or "").strip(),
            str(model or "").strip(),
            str(condition or "").strip(),
            int(repeat_index),
        )
    except Exception as exc:
        logger.error("构造实验执行键失败: %s", exc, exc_info=True)
        return ("", "", "", -1)


def _load_json_schema(schema_path: Any) -> dict[str, Any]:
    try:
        path = Path(str(schema_path or "")).expanduser()
        if not path.is_file():
            return _error_result(
                "BioToolEval题目Schema不存在",
                "QUESTION_SCHEMA_NOT_FOUND",
                details={"file_name": path.name},
            )
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        return {
            "status": "success",
            "validator": Draft202012Validator(schema),
            "schema_version": schema.get("x-version"),
        }
    except Exception as exc:
        logger.error("加载BioToolEval题目Schema失败: %s", exc, exc_info=True)
        return _error_result(
            f"加载BioToolEval题目Schema失败: {exc}",
            "QUESTION_SCHEMA_EXCEPTION",
        )


def _sha256_file(file_path: Any) -> str:
    try:
        digest = hashlib.sha256()
        path = Path(str(file_path or ""))
        # Universal newline decoding matches Git text normalization.
        with path.open("r", encoding="utf-8", newline=None) as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk.encode("utf-8"))
        return digest.hexdigest()
    except Exception as exc:
        logger.error("计算数据集SHA-256失败: %s", exc, exc_info=True)
        return ""


def _verify_development_freeze_manifest(
    freeze_manifest_path: Any,
    *,
    dataset_result: dict[str, Any],
    experiment_id: str,
    conditions: list[str],
    repeats: int,
    model: str,
    temperature: float,
    seed: int | None,
    trace_path: Path,
    plan_source_path: Path | None = None,
) -> dict[str, Any]:
    try:
        if freeze_manifest_path is None:
            return _error_result(
                "dirty开发实验必须提供冻结清单",
                "MISSING_DEVELOPMENT_FREEZE_MANIFEST",
            )
        project_root = Path(__file__).resolve().parents[1]
        path = Path(str(freeze_manifest_path)).expanduser().resolve()
        try:
            path.relative_to(project_root)
        except ValueError:
            return _error_result(
                "冻结清单必须位于当前项目目录内",
                "FREEZE_MANIFEST_OUTSIDE_PROJECT",
            )
        if not path.is_file() or path.suffix.lower() != ".json":
            return _error_result(
                "冻结清单不存在或不是JSON文件",
                "DEVELOPMENT_FREEZE_MANIFEST_NOT_FOUND",
            )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return _error_result(
                "冻结清单顶层必须是对象",
                "INVALID_DEVELOPMENT_FREEZE_MANIFEST",
            )
        if (
            manifest.get("status") != "FROZEN_DEVELOPMENT_PILOT"
            or manifest.get("scope") != "development_only"
            or manifest.get("not_for_final_metrics") is not True
            or manifest.get("test100_access_authorized") is not False
            or manifest.get("b3_enabled") is not False
        ):
            return _error_result(
                "冻结清单的开发范围或安全边界无效",
                "INVALID_DEVELOPMENT_FREEZE_SCOPE",
            )

        mismatches: list[dict[str, Any]] = []

        def check_file(relative_path: Any, expected_sha256: Any, label: str) -> None:
            try:
                candidate = (project_root / Path(str(relative_path or ""))).resolve()
                candidate.relative_to(project_root)
                expected = str(expected_sha256 or "").strip().lower()
                actual = _sha256_file(candidate) if candidate.is_file() else ""
                if len(expected) != 64 or actual != expected:
                    mismatches.append(
                        {
                            "field": label,
                            "file_name": candidate.name,
                            "expected_sha256": expected,
                            "actual_sha256": actual,
                        }
                    )
            except Exception as exc:
                mismatches.append({"field": label, "error": type(exc).__name__})

        dataset = manifest.get("dataset") or {}
        check_file(dataset.get("path"), dataset.get("sha256"), "dataset")
        if str(dataset_result.get("dataset_sha256") or "") != str(
            dataset.get("sha256") or ""
        ):
            mismatches.append({"field": "active_dataset_sha256"})
        for index, fixture in enumerate(manifest.get("fixtures") or []):
            if isinstance(fixture, dict):
                check_file(
                    fixture.get("path"),
                    fixture.get("sha256"),
                    f"fixtures.{index}",
                )

        snapshot = manifest.get("method_snapshot") or {}
        method_paths = {
            "tool_registry_sha256": "ToolsAgent/ToolsFuns/Biology/tool_name_dict.py",
            "planning_prompt_sha256": "app/llms/planning/prompt_templates.py",
            "llm_client_sha256": "app/llms/api_clients/openai_api.py",
            "agent_runtime_sha256": "scripts/retrieve_tool_info.py",
            "plan_sampling_sha256": "scripts/plan_sampling.py",
            "type_checker_sha256": "scripts/type_checker.py",
            "structured_plan_sha256": "scripts/structured_plan.py",
            "typed_runtime_sha256": "scripts/typed_planning_runtime.py",
            "typed_executor_sha256": "scripts/typed_executor.py",
            "fasta_input_parser_sha256": "app/tools/fasta_input_parser.py",
            "tool_parameters_sha256": "app/tools/tool_parameters.py",
            "tool_runner_sha256": "app/tools/tool_runner.py",
            "initial_artifacts_sha256": "scripts/initial_artifacts.py",
            "evaluation_trace_sha256": "scripts/evaluation_trace.py",
            "trace_schema_sha256": (
                "evaluation/biotooleval/experiment_trace_schema_v1.json"
            ),
            "config_sha256": "app/core/config.py",
            "runner_sha256": "scripts/run_biotooleval.py",
            "validator_sha256": "scripts/validate_biotooleval_dataset.py",
        }
        if "F1" in conditions:
            method_paths.update(
                {
                    "baseline_strategies_sha256": "scripts/baseline_strategies.py",
                    "hierarchical_recovery_sha256": "scripts/hierarchical_recovery.py",
                    "hybrid_trace_schema_sha256": (
                        "evaluation/biotooleval/experiment_trace_schema_hybrid_v1.json"
                    ),
                }
            )
        for hash_field, relative_path in method_paths.items():
            check_file(relative_path, snapshot.get(hash_field), hash_field)

        frozen_run = manifest.get("first_run") or {}
        actual_controls = {
            "experiment_id": experiment_id,
            "model": model,
            "temperature": float(temperature),
            "seed": seed,
            "conditions": list(conditions),
            "repeats": int(repeats),
        }
        for field, actual in actual_controls.items():
            if frozen_run.get(field) != actual:
                mismatches.append(
                    {
                        "field": f"first_run.{field}",
                        "expected": frozen_run.get(field),
                        "actual": actual,
                    }
                )
        frozen_output = (project_root / str(frozen_run.get("output") or "")).resolve()
        if frozen_output != trace_path.resolve():
            mismatches.append(
                {
                    "field": "first_run.output",
                    "expected_file_name": frozen_output.name,
                    "actual_file_name": trace_path.name,
                }
            )
        frozen_plan_source = frozen_run.get("frozen_plan_ledger")
        if plan_source_path is not None:
            expected_source = (
                project_root / str(frozen_plan_source or "")
            ).resolve()
            if expected_source != plan_source_path.resolve():
                mismatches.append({"field": "first_run.frozen_plan_ledger"})
            expected_source_hash = str(
                frozen_run.get("frozen_plan_ledger_sha256") or ""
            )
            if _sha256_file(plan_source_path) != expected_source_hash:
                mismatches.append(
                    {"field": "first_run.frozen_plan_ledger_sha256"}
                )
        elif frozen_plan_source:
            mismatches.append({"field": "first_run.frozen_plan_ledger"})
        if mismatches:
            return _error_result(
                "开发实验冻结清单与当前运行不一致",
                "DEVELOPMENT_FREEZE_MISMATCH",
                details={"mismatches": mismatches},
            )
        return {
            "status": "success",
            "artifact_id": str(manifest.get("artifact_id") or ""),
            "file_name": path.name,
            "sha256": _sha256_file(path),
        }
    except Exception as exc:
        logger.error("校验开发实验冻结清单失败: %s", exc, exc_info=True)
        return _error_result(
            f"校验开发实验冻结清单失败: {exc}",
            "DEVELOPMENT_FREEZE_VALIDATION_EXCEPTION",
        )


def load_biotooleval_dataset(
    dataset_path: Any,
    *,
    dry_run: bool = False,
    development_run: bool = False,
    config: Any = None,
) -> dict[str, Any]:
    try:
        cfg = config if config is not None else Config()
        path = Path(str(dataset_path or "")).expanduser()
        if not path.is_file() or path.suffix.lower() != ".json":
            return _error_result(
                "BioToolEval数据集不存在或不是JSON文件",
                "DATASET_NOT_FOUND",
                details={"file_name": path.name},
            )
        dataset = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(dataset, dict):
            return _error_result(
                "BioToolEval顶层必须是JSON对象",
                "INVALID_DATASET_OBJECT",
            )

        dataset_status = str(dataset.get("status") or "").strip()
        if dry_run and development_run:
            return _error_result(
                "dry-run与development-run不能同时启用",
                "CONFLICTING_RUN_SCOPE",
            )
        if dry_run:
            if dataset_status != "dry_run_fixture":
                return _error_result(
                    "dry-run只接受内置离线夹具，不能伪装运行正式数据",
                    "INVALID_DRY_RUN_DATASET",
                )
        elif development_run:
            if dataset_status in _FORMAL_DATASET_STATUSES:
                return _error_result(
                    "正式数据集不得降级为development-run",
                    "FORMAL_DATASET_IN_DEVELOPMENT_MODE",
                    details={"dataset_status": dataset_status},
                )
            if dataset_status != "candidate_not_for_final_metrics":
                return _error_result(
                    "development-run只接受明确标记的开发候选集",
                    "INVALID_DEVELOPMENT_DATASET",
                    details={"dataset_status": dataset_status or "missing"},
                )
        elif dataset_status not in _FORMAL_DATASET_STATUSES:
            return _error_result(
                "正式实验只接受benchmark_ready、frozen或released数据集",
                "UNVERIFIED_DATASET",
                details={"dataset_status": dataset_status or "missing"},
            )

        items = dataset.get("items")
        if not isinstance(items, list) or not items:
            return _error_result(
                "BioToolEval数据集不包含题目",
                "EMPTY_DATASET",
            )

        schema_result = _load_json_schema(
            getattr(cfg, "EXPERIMENT_QUESTION_SCHEMA_PATH", "")
        )
        if schema_result.get("status") != "success":
            return schema_result
        validator = schema_result["validator"]

        validated_items: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item_index, item in enumerate(items):
            if not isinstance(item, dict):
                return _error_result(
                    f"第{item_index + 1}题不是JSON对象",
                    "INVALID_QUESTION_OBJECT",
                )
            question_id = str(item.get("id") or "").strip()
            if question_id in seen_ids:
                return _error_result(
                    f"题目ID重复: {question_id}",
                    "DUPLICATE_QUESTION_ID",
                    details={"question_id": question_id},
                )
            if "submitted_gold" in item or not isinstance(item.get("gold"), dict):
                return _error_result(
                    f"题目{question_id or item_index + 1}缺少已验收gold",
                    "UNVERIFIED_GOLD",
                    details={"question_id": question_id},
                )
            errors = sorted(
                validator.iter_errors(item),
                key=lambda error: list(error.absolute_path),
            )
            if errors:
                first = errors[0]
                field_path = ".".join(
                    str(part) for part in first.absolute_path
                )
                return _error_result(
                    f"题目{question_id or item_index + 1}不符合Schema: "
                    f"{first.message}",
                    "QUESTION_SCHEMA_VALIDATION_FAILED",
                    details={
                        "question_id": question_id,
                        "path": field_path,
                    },
                )
            seen_ids.add(question_id)
            validated_items.append(copy.deepcopy(item))

        summary = dataset.get("summary") or {}
        if not dry_run and not development_run and isinstance(summary, dict):
            ready_count = summary.get("benchmark_ready_count")
            if isinstance(ready_count, int) and ready_count != len(items):
                return _error_result(
                    "benchmark_ready_count与题目总数不一致",
                    "INCOMPLETE_BENCHMARK",
                    details={
                        "benchmark_ready_count": ready_count,
                        "item_count": len(items),
                    },
                )

        dataset_sha256 = _sha256_file(path)
        if len(dataset_sha256) != 64:
            return _error_result(
                "无法计算BioToolEval数据集SHA-256",
                "DATASET_HASH_FAILURE",
            )
        return {
            "status": "success",
            "dataset_name": str(
                dataset.get("dataset_name") or "BioToolEval"
            ),
            "dataset_version": str(dataset.get("version") or "").strip(),
            "dataset_status": dataset_status,
            "dataset_file_name": path.name,
            "dataset_sha256": dataset_sha256,
            "dataset_dir": str(path.resolve().parent),
            "items": validated_items,
            "item_count": len(validated_items),
            "question_schema_version": schema_result.get("schema_version"),
            "evaluation_scope": (
                "offline_dry_run"
                if dry_run
                else "development_only"
                if development_run
                else "formal"
            ),
        }
    except Exception as exc:
        logger.error("加载BioToolEval数据集失败: %s", exc, exc_info=True)
        return _error_result(
            f"加载BioToolEval数据集失败: {exc}",
            "DATASET_LOAD_EXCEPTION",
        )


def prepare_agent_request(
    question_item: Any,
    *,
    dataset_dir: Any,
) -> dict[str, Any]:
    try:
        if not isinstance(question_item, dict):
            return _error_result(
                "question_item必须是字典",
                "INVALID_QUESTION_ITEM",
            )
        question = str(question_item.get("question_zh") or "").strip()
        input_data = question_item.get("input") or {}
        if not question or not isinstance(input_data, dict):
            return _error_result(
                "题目缺少问题正文或输入对象",
                "INVALID_QUESTION_INPUT",
            )

        input_type = input_data.get("input_type")
        file_paths: list[str] = []
        is_add_filename = False
        if input_type == "amino_acid_sequence":
            sequence = str(input_data.get("sequence") or "").strip()
            if not sequence:
                return _error_result(
                    "氨基酸序列输入为空",
                    "EMPTY_SEQUENCE_INPUT",
                )
            question = (
                f"{question}\n\n"
                "结构化氨基酸序列输入：\n"
                f"```sequence\n{sequence}\n```"
            )
        elif input_type in {"protein_fasta_file", "pdb_file"}:
            fixture_text = str(input_data.get("fixture") or "").strip()
            if not fixture_text:
                return _error_result(
                    "文件型题目缺少fixture",
                    "MISSING_INPUT_FIXTURE",
                )
            base_dir = Path(str(dataset_dir or "")).resolve()
            fixture_path = (base_dir / fixture_text).resolve()
            try:
                fixture_path.relative_to(base_dir)
            except ValueError:
                return _error_result(
                    "fixture必须位于数据集目录内",
                    "FIXTURE_OUTSIDE_DATASET",
                )
            if not fixture_path.is_file():
                return _error_result(
                    "题目fixture文件不存在",
                    "INPUT_FIXTURE_NOT_FOUND",
                    details={"fixture_name": fixture_path.name},
                )
            file_paths = [str(fixture_path)]
            is_add_filename = True
            chain_id = input_data.get("chain_id")
            if input_type == "pdb_file" and chain_id:
                question = f"{question}\n\n指定链ID：{chain_id}"
        elif input_type != "none":
            return _error_result(
                f"不支持的input_type: {input_type}",
                "UNSUPPORTED_INPUT_TYPE",
            )

        return {
            "status": "success",
            "question": question,
            "file_path_list": file_paths,
            "is_add_filename": is_add_filename,
        }
    except Exception as exc:
        logger.error("准备Agent题目输入失败: %s", exc, exc_info=True)
        return _error_result(
            f"准备Agent题目输入失败: {exc}",
            "REQUEST_PREPARATION_EXCEPTION",
        )


def _resolve_git_state() -> dict[str, Any]:
    try:
        project_root = Path(__file__).resolve().parents[1]
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project_root,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project_root,
        )
        commit = commit_result.stdout.strip()
        return {
            "status": "success",
            "git_commit": commit if len(commit) >= 7 else "unknown",
            "git_dirty": bool(status_result.stdout.strip()),
        }
    except Exception as exc:
        logger.warning("读取Git状态失败: %s", exc)
        return {
            "status": "error",
            "git_commit": "unknown",
            "git_dirty": None,
        }


def _load_default_agent_runtime() -> dict[str, Any]:
    try:
        from scripts import retrieve_tool_info

        return {
            "status": "success",
            "agent_callable": retrieve_tool_info.run_SciToolAgent,
            "plan_callable": retrieve_tool_info.capture_SciToolAgent_plan,
            "replan_callable": retrieve_tool_info.capture_SciToolAgent_replan,
            "runtime_config": retrieve_tool_info._cfg,
        }
    except Exception as exc:
        logger.error("加载真实Agent运行时失败: %s", exc, exc_info=True)
        return _error_result(
            f"加载真实Agent运行时失败: {exc}",
            "AGENT_RUNTIME_LOAD_EXCEPTION",
        )


def create_offline_mock_agent(
    runtime_config: Any,
) -> AgentCallable:
    try:
        async def offline_mock_agent(
            question: str,
            file_path_list: list[str],
            is_add_filename: bool = False,
            *,
            frozen_plan: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            try:
                del file_path_list, is_add_filename
                default_tool = (
                    "compute_pI_mw"
                    if "[DRY_PI_MW]" in question
                    else "compute_gravy"
                )
                frozen_chain = (
                    frozen_plan.get("planned_chain")
                    if isinstance(frozen_plan, dict)
                    else None
                )
                tool_name = (
                    str(frozen_chain[0])
                    if isinstance(frozen_chain, list) and frozen_chain
                    else default_tool
                )
                type_enabled = bool(
                    getattr(runtime_config, "TYPE_PLANNING_ENABLED", False)
                )
                repair_enabled = bool(
                    getattr(runtime_config, "TYPE_REPAIR_ENABLED", False)
                )
                validation_enabled = bool(
                    getattr(
                        runtime_config,
                        "TYPE_VALIDATION_ENABLED",
                        type_enabled,
                    )
                )
                typed_trace = None
                if type_enabled:
                    typed_trace = {
                        "enabled": True,
                        "validation_enabled": validation_enabled,
                        "validation_bypassed": not validation_enabled,
                        "repair_enabled": repair_enabled,
                        "repair_attempted": False,
                        "repair_applied": False,
                        "plan_before": [tool_name],
                        "plan_after": [tool_name],
                        "planning_issues": [],
                        "total_duration_ms": 0.1,
                    }
                reasoning = {
                    "plan_chain": [tool_name],
                    "plan_chain_executed": [tool_name],
                    "tool_steps": [
                        {
                            "tool_name": tool_name,
                            "status": "success",
                        }
                    ],
                    "execution_summary": {
                        "completed": True,
                        "attempted_steps": 1,
                        "successful_steps": 1,
                        "failed_steps": 0,
                        "error_type": None,
                    },
                    "typed_planning": typed_trace,
                    "plan_sampling": safe_plan_trace(frozen_plan),
                }
                return {
                    "answer": (
                        json.dumps(
                            (
                                {
                                    "status": "success",
                                    "pI": 6.5,
                                    "molecular_weight": 1234.5,
                                }
                                if tool_name == "compute_pI_mw"
                                else {
                                    "status": "success",
                                    "gravy": 0.1,
                                }
                            ),
                            ensure_ascii=False,
                        )
                        + "\n\n**Final Answer:**\noffline mock result"
                    ),
                    "reasoning": reasoning,
                    "latency": 0.001,
                }
            except Exception as exc:
                logger.error("离线Mock执行失败: %s", exc, exc_info=True)
                return {
                    "answer": "offline mock error",
                    "reasoning": {
                        "execution_summary": {
                            "completed": False,
                            "attempted_steps": 0,
                            "successful_steps": 0,
                            "failed_steps": 0,
                            "error_type": type(exc).__name__,
                        }
                    },
                    "latency": 0.0,
                }

        return offline_mock_agent
    except Exception as exc:
        logger.error("创建离线Mock失败: %s", exc, exc_info=True)

        async def failed_mock_agent(
            question: str,
            file_path_list: list[str],
            is_add_filename: bool = False,
        ) -> dict[str, Any]:
            try:
                del question, file_path_list, is_add_filename
                raise RuntimeError("offline mock unavailable")
            except Exception as inner_exc:
                raise inner_exc

        return failed_mock_agent


def create_offline_mock_plan_sampler() -> PlanCallable:
    async def offline_plan_sampler(
        question: str,
        file_path_list: list[str],
        is_add_filename: bool = False,
        *,
        experiment_id: str = "",
        question_id: str = "",
        repeat_index: int = 0,
        model: str = "offline-mock",
    ) -> dict[str, Any]:
        try:
            del file_path_list, is_add_filename
            tool_name = (
                "compute_pI_mw"
                if "[DRY_PI_MW]" in question
                else "compute_gravy"
            )
            raw_response = json.dumps([tool_name], ensure_ascii=False)
            response_sha256 = sha256_text(raw_response)
            attempt = {
                "attempt_index": 1,
                "attempt_kind": "content",
                "outcome": "captured",
                "exception_type": None,
                "response_sha256": response_sha256,
                "raw_response": raw_response,
                "parsed_chain": [tool_name],
                "unknown_tools": [],
                "latency_ms": 0.0,
            }
            frozen_plan = {
                "schema_version": "1.0",
                "record_status": "captured",
                "experiment_id": str(experiment_id),
                "question_id": str(question_id),
                "repeat_index": int(repeat_index),
                "model": str(model),
                "plan_sample_id": build_plan_sample_id(
                    experiment_id=experiment_id,
                    question_id=question_id,
                    repeat_index=repeat_index,
                    model=model,
                    response_sha256=response_sha256,
                ),
                "capture_outcome": "captured",
                "selected_attempt_index": 1,
                "response_sha256": response_sha256,
                "planned_chain": [tool_name],
                "unknown_tools": [],
                "retrieved_triples": [],
                "retrieved_tool_names": [tool_name],
                "attempts": [attempt],
            }
            validation = validate_frozen_plan(frozen_plan)
            return (
                {
                    "status": "success",
                    "frozen_plan": validation["frozen_plan"],
                }
                if validation.get("status") == "success"
                else validation
            )
        except Exception as exc:
            return _error_result(
                "离线计划采样失败",
                "OFFLINE_PLAN_CAPTURE_EXCEPTION",
                details={"exception_type": type(exc).__name__},
            )

    return offline_plan_sampler


def create_offline_mock_replanner() -> PlanCallable:
    async def offline_replanner(
        *,
        question,
        original_plan_chain,
        available_input_types,
        type_issues,
        model,
        max_output_tokens,
        required_tool_names=None,
    ) -> dict[str, Any]:
        del (
            question,
            available_input_types,
            type_issues,
            model,
            max_output_tokens,
            required_tool_names,
        )
        chain = [str(value) for value in original_plan_chain or []]
        return {
            "status": "success",
            "raw_response": f"Plan Chain: {chain!r}",
            "planned_chain": chain,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "latency_ms": 0.0,
        }

    return offline_replanner


async def _execute_under_condition(
    agent_callable: AgentCallable,
    runtime_config: Any,
    request: dict[str, Any],
    *,
    condition: str,
    timeout_seconds: float,
    frozen_plan: dict[str, Any] | None = None,
    model: str | None = None,
    replan_callable: PlanCallable | None = None,
    initial_types: list[str] | None = None,
    initial_artifact_issue_codes: list[str] | None = None,
    replan_max_output_tokens: int = 256,
) -> dict[str, Any]:
    original_planning = getattr(
        runtime_config,
        "TYPE_PLANNING_ENABLED",
        False,
    )
    had_original_validation = hasattr(
        runtime_config,
        "TYPE_VALIDATION_ENABLED",
    )
    original_validation = getattr(
        runtime_config,
        "TYPE_VALIDATION_ENABLED",
        original_planning,
    )
    original_repair = getattr(
        runtime_config,
        "TYPE_REPAIR_ENABLED",
        False,
    )
    had_original_model = hasattr(runtime_config, "MODEL_NAME")
    original_model = getattr(runtime_config, "MODEL_NAME", None)
    try:
        requested_model = str(model or "").strip()
        if model is not None and not requested_model:
            return _error_result(
                "实验执行模型不得为空",
                "MISSING_EXECUTION_MODEL",
            )
        if requested_model:
            runtime_config.MODEL_NAME = requested_model
        condition_result = get_experiment_condition(condition)
        if condition_result.get("status") != "success":
            return condition_result
        condition_env = condition_result["env"]
        runtime_config.TYPE_PLANNING_ENABLED = condition_env[
            "TYPE_PLANNING_ENABLED"
        ]
        runtime_config.TYPE_VALIDATION_ENABLED = condition_env[
            "TYPE_VALIDATION_ENABLED"
        ]
        runtime_config.TYPE_REPAIR_ENABLED = condition_env[
            "TYPE_REPAIR_ENABLED"
        ]
        started = time.perf_counter()
        active_frozen_plan = frozen_plan
        baseline_trace: dict[str, Any] | None = None
        recovery_trace: dict[str, Any] | None = None
        replan_frozen_plan: dict[str, Any] | None = None
        replan_ledger_record: dict[str, Any] | None = None
        if condition == "S1":
            schema_result = validate_schema_runtime_plan(frozen_plan)
            if schema_result.get("status") != "success":
                return schema_result
            baseline_trace = {
                "baseline_kind": SCHEMA_RUNTIME_BASELINE,
                "schema_valid_before": bool(schema_result.get("valid")),
                "original_plan_preserved": True,
                "original_plan_sample_id": (
                    str(frozen_plan.get("plan_sample_id"))
                    if isinstance(frozen_plan, dict)
                    and frozen_plan.get("plan_sample_id")
                    else None
                ),
                "replan_triggered": False,
                "replan_applied": False,
                "replan_plan_sample_id": None,
                "extra_llm_calls": 0,
                "extra_llm_input_tokens": None,
                "extra_llm_output_tokens": None,
                "extra_llm_total_tokens": None,
                "replan_latency_ms": 0.0,
                "replan_outcome": "not_triggered",
                "original_plan_chain": list(
                    schema_result.get("planned_chain") or []
                ),
                "revised_plan_chain": list(
                    schema_result.get("planned_chain") or []
                ),
            }
            if schema_result.get("decision") == "reject":
                rejection = build_schema_rejection_result(
                    frozen_plan,
                    schema_result,
                )
                rejection.setdefault("reasoning", {})["baseline"] = (
                    baseline_trace
                )
                return {
                    "status": "success",
                    "agent_result": rejection,
                    "total_latency_ms": max(
                        0.0,
                        (time.perf_counter() - started) * 1000.0,
                    ),
                    "replan_frozen_plan": None,
                    "replan_ledger_record": None,
                }
        elif condition == "L1":
            replan_result = await prepare_matched_budget_replan(
                question=request.get("question"),
                frozen_plan=frozen_plan,
                available_input_types=initial_types or [],
                initial_artifact_issue_codes=(
                    initial_artifact_issue_codes or []
                ),
                model=requested_model,
                replan_callable=replan_callable,
                max_output_tokens=replan_max_output_tokens,
            )
            if replan_result.get("status") != "success":
                return replan_result
            active_frozen_plan = replan_result["active_frozen_plan"]
            baseline_trace = copy.deepcopy(replan_result["baseline"])
            replan_frozen_plan = replan_result.get("replan_frozen_plan")
            replan_ledger_record = replan_result.get(
                "replan_ledger_record"
            )
        elif condition == "F1":
            recovery_result = await prepare_hierarchical_recovery(
                question=request.get("question"),
                frozen_plan=frozen_plan,
                available_input_types=initial_types or [],
                initial_artifact_issue_codes=(
                    initial_artifact_issue_codes or []
                ),
                model=requested_model,
                replan_callable=replan_callable,
                max_output_tokens=replan_max_output_tokens,
            )
            if recovery_result.get("status") != "success":
                return recovery_result
            active_frozen_plan = recovery_result["active_frozen_plan"]
            recovery_trace = copy.deepcopy(recovery_result["recovery"])
            replan_frozen_plan = recovery_result.get("replan_frozen_plan")
            replan_ledger_record = recovery_result.get(
                "replan_ledger_record"
            )
        agent_call = (
            agent_callable(
                request["question"],
                request["file_path_list"],
                request["is_add_filename"],
                frozen_plan=active_frozen_plan,
            )
            if active_frozen_plan is not None
            else agent_callable(
                request["question"],
                request["file_path_list"],
                request["is_add_filename"],
            )
        )
        result = await asyncio.wait_for(
            agent_call,
            timeout=float(timeout_seconds),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not isinstance(result, dict):
            return _error_result(
                "Agent响应必须是字典",
                "INVALID_AGENT_RESPONSE",
            )
        if str(result.get("status") or "").strip().lower() == "error":
            agent_error_code = str(
                result.get("error_code") or "AGENT_REPORTED_ERROR"
            ).strip()
            logger.error(
                "Agent显式报告系统错误 | condition=%s | error_code=%s",
                condition,
                agent_error_code,
            )
            return _error_result(
                "Agent显式报告系统错误，本次执行不得写入有效Trace",
                "AGENT_REPORTED_ERROR",
                details={"agent_error_code": agent_error_code},
            )
        if baseline_trace is not None:
            reasoning = result.setdefault("reasoning", {})
            if not isinstance(reasoning, dict):
                return _error_result(
                    "baseline Agent响应的reasoning必须是字典",
                    "INVALID_BASELINE_REASONING",
                )
            reasoning["baseline"] = copy.deepcopy(baseline_trace)
            # Pairing always uses the original frozen plan ID.
            # Record the revised L1 plan ID separately.
            reasoning["plan_sampling"] = safe_plan_trace(frozen_plan)
        if recovery_trace is not None:
            reasoning = result.setdefault("reasoning", {})
            if not isinstance(reasoning, dict):
                return _error_result(
                    "F1 Agent响应的reasoning必须是字典",
                    "INVALID_F1_REASONING",
                )
            reasoning["recovery"] = copy.deepcopy(recovery_trace)
            # Keep the original paired plan ID when F1 executes a derived plan.
            reasoning["plan_sampling"] = safe_plan_trace(frozen_plan)
        return {
            "status": "success",
            "agent_result": result,
            "total_latency_ms": elapsed_ms,
            "replan_frozen_plan": replan_frozen_plan,
            "replan_ledger_record": replan_ledger_record,
        }
    except asyncio.TimeoutError:
        logger.error(
            "单题执行超时 | condition=%s | timeout=%ss",
            condition,
            timeout_seconds,
        )
        return _error_result(
            "单题执行超过硬超时，已取消当前题目",
            "QUESTION_TIMEOUT",
        )
    except Exception as exc:
        logger.error(
            "单题Agent执行异常 | condition=%s | error=%s",
            condition,
            exc,
            exc_info=True,
        )
        return _error_result(
            "单题Agent执行失败，详细异常仅写入本地日志",
            "AGENT_EXECUTION_EXCEPTION",
            details={"exception_type": type(exc).__name__},
        )
    finally:
        try:
            runtime_config.TYPE_PLANNING_ENABLED = original_planning
            if had_original_validation:
                runtime_config.TYPE_VALIDATION_ENABLED = original_validation
            elif hasattr(runtime_config, "TYPE_VALIDATION_ENABLED"):
                delattr(runtime_config, "TYPE_VALIDATION_ENABLED")
            runtime_config.TYPE_REPAIR_ENABLED = original_repair
            if had_original_model:
                runtime_config.MODEL_NAME = original_model
            elif hasattr(runtime_config, "MODEL_NAME"):
                delattr(runtime_config, "MODEL_NAME")
        except Exception as exc:
            logger.error("恢复实验条件配置失败: %s", exc, exc_info=True)


async def _capture_frozen_plan(
    plan_callable: PlanCallable,
    request: dict[str, Any],
    *,
    experiment_id: str,
    question_id: str,
    repeat_index: int,
    model: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        result = await asyncio.wait_for(
            plan_callable(
                request["question"],
                request["file_path_list"],
                request["is_add_filename"],
                experiment_id=experiment_id,
                question_id=question_id,
                repeat_index=repeat_index,
                model=model,
            ),
            timeout=float(timeout_seconds),
        )
        if not isinstance(result, dict):
            return _error_result(
                "计划采样器响应必须是字典",
                "INVALID_PLAN_CAPTURE_RESPONSE",
            )
        if result.get("status") != "success":
            return result
        validation = validate_frozen_plan(result.get("frozen_plan"))
        if validation.get("status") != "success":
            return _error_result(
                "计划采样器返回了无效冻结计划",
                str(validation.get("code") or "INVALID_FROZEN_PLAN"),
            )
        return {
            "status": "success",
            "frozen_plan": validation["frozen_plan"],
        }
    except asyncio.TimeoutError:
        return _error_result(
            "原始计划采样超过硬超时",
            "PLAN_CAPTURE_TIMEOUT",
        )
    except Exception as exc:
        logger.error("原始计划采样异常: %s", exc, exc_info=True)
        return _error_result(
            "原始计划采样异常，详细信息仅写入本地日志",
            "PLAN_CAPTURE_EXCEPTION",
            details={"exception_type": type(exc).__name__},
        )


def _read_completed_keys(
    output_path: Path,
    *,
    config: Any,
) -> dict[str, Any]:
    try:
        if not output_path.exists():
            return {"status": "success", "keys": set(), "record_count": 0}
        loaded = load_experiment_traces(output_path, config=config)
        if loaded.get("status") != "success":
            return loaded
        keys = {
            _execution_key(
                trace.get("question_id"),
                (trace.get("context") or {}).get("model"),
                trace.get("condition"),
                trace.get("repeat_index"),
            )
            for trace in loaded.get("records") or []
        }
        if len(keys) != loaded.get("record_count"):
            return _error_result(
                "现有Trace包含重复实验键，请先审计结果文件",
                "DUPLICATE_TRACE_KEY",
            )
        return {
            "status": "success",
            "keys": keys,
            "record_count": len(keys),
        }
    except Exception as exc:
        logger.error("读取断点续跑Trace失败: %s", exc, exc_info=True)
        return _error_result(
            f"读取断点续跑Trace失败: {exc}",
            "RESUME_TRACE_EXCEPTION",
        )


def _read_failure_keys(failure_path: Path) -> dict[str, Any]:
    try:
        if not failure_path.exists():
            return {"status": "success", "keys": set(), "record_count": 0}
        state: dict[tuple[str, str, str, int], str] = {}
        lines = failure_path.read_text(encoding="utf-8").splitlines()
        for line_number, raw_line in enumerate(lines, 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except Exception as exc:
                return _error_result(
                    f"失败账本第{line_number}行不是合法JSON: {exc}",
                    "INVALID_FAILURE_LEDGER",
                )
            key = _execution_key(
                record.get("question_id"),
                record.get("model"),
                record.get("condition"),
                record.get("repeat_index"),
            )
            state[key] = str(record.get("status") or "failed")
        failed_keys = {
            key for key, status in state.items() if status == "failed"
        }
        return {
            "status": "success",
            "keys": failed_keys,
            "record_count": len(lines),
        }
    except Exception as exc:
        logger.error("读取失败账本失败: %s", exc, exc_info=True)
        return _error_result(
            f"读取失败账本失败: {exc}",
            "FAILURE_LEDGER_EXCEPTION",
        )


def _count_jsonl_records(path: Path) -> int | None:
    try:
        if not path.exists():
            return 0
        return sum(
            bool(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    except Exception as exc:
        logger.error("统计JSONL记录数失败: %s", exc, exc_info=True)
        return None


def _summarize_scoring_audit(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {
                "status": "success",
                "automatic_labels": 0,
                "manual_required_labels": 0,
            }
        automatic_labels = 0
        manual_required_labels = 0
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
                    f"评分审计第{line_number}行不是合法JSON: {exc}",
                    "INVALID_SCORING_AUDIT",
                )
            decisions = record.get("decisions")
            if not isinstance(decisions, dict):
                return _error_result(
                    f"评分审计第{line_number}行缺少decisions",
                    "INVALID_SCORING_AUDIT",
                )
            automatic_labels += sum(
                isinstance(decision, dict)
                and decision.get("source") == "automatic"
                for decision in decisions.values()
            )
            manual_required_labels += sum(
                isinstance(decision, dict)
                and decision.get("source") == "manual_required"
                for decision in decisions.values()
            )
        return {
            "status": "success",
            "automatic_labels": automatic_labels,
            "manual_required_labels": manual_required_labels,
        }
    except Exception as exc:
        logger.error("汇总评分审计失败: %s", exc, exc_info=True)
        return _error_result(
            f"汇总评分审计失败: {exc}",
            "SCORING_AUDIT_SUMMARY_EXCEPTION",
        )


def _append_failure_record(
    failure_path: Path,
    *,
    experiment_id: str,
    question_id: str,
    model: str,
    condition: str,
    repeat_index: int,
    status: str,
    error_code: str | None = None,
    detail_code: str | None = None,
    failure_category: str | None = None,
    exception_type: str | None = None,
) -> dict[str, Any]:
    try:
        if status not in {"failed", "resolved"}:
            return _error_result(
                "失败账本status只能是failed或resolved",
                "INVALID_FAILURE_STATUS",
            )
        record = {
            "schema_version": "1.0",
            "experiment_id": experiment_id,
            "question_id": question_id,
            "model": model,
            "condition": condition,
            "repeat_index": repeat_index,
            "status": status,
            "error_code": error_code,
            "detail_code": detail_code,
            "failure_category": failure_category,
            "exception_type": exception_type,
            "occurred_at": _utc_now_text(),
        }
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"status": "success"}
    except Exception as exc:
        logger.error("追加失败账本失败: %s", exc, exc_info=True)
        return _error_result(
            f"追加失败账本失败: {exc}",
            "FAILURE_APPEND_EXCEPTION",
        )


def _classify_failure(error_code: Any, detail_code: Any) -> str:
    try:
        outer = str(error_code or "UNKNOWN")
        detail = str(detail_code or "")
        if outer == "QUESTION_TIMEOUT":
            return "timeout"
        if outer in {
            "AGENT_EXECUTION_EXCEPTION",
            "INDEX_LOAD_FAILED",
        }:
            return "agent_or_infrastructure_failure"
        if outer == "AGENT_REPORTED_ERROR":
            if detail == "AGENT_RETRIES_EXHAUSTED":
                return "agent_retries_exhausted"
            return "agent_reported_failure"
        if "TRACE" in outer or "SCORING" in outer or "REVIEW" in outer:
            return "evaluation_pipeline_failure"
        if "INPUT" in outer or "FIXTURE" in outer or "DATASET" in outer:
            return "input_or_dataset_failure"
        return "unclassified_failure"
    except Exception as exc:
        logger.error("分类实验失败单元异常: %s", exc, exc_info=True)
        return "failure_classification_exception"


def _write_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = manifest_path.with_suffix(
            manifest_path.suffix + ".tmp"
        )
        temp_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, manifest_path)
        return {"status": "success", "manifest_path": str(manifest_path)}
    except Exception as exc:
        logger.error("写入实验运行清单失败: %s", exc, exc_info=True)
        return _error_result(
            f"写入实验运行清单失败: {exc}",
            "MANIFEST_WRITE_EXCEPTION",
        )


def _validate_resume_manifest(
    manifest_path: Path,
    expected_manifest: dict[str, Any],
    *,
    trace_path: Path,
    failure_path: Path,
) -> dict[str, Any]:
    try:
        has_previous_artifact = trace_path.exists() or failure_path.exists()
        if not manifest_path.exists():
            if has_previous_artifact:
                return _error_result(
                    "已有实验结果但缺少运行清单，禁止盲目追加",
                    "MISSING_RESUME_MANIFEST",
                )
            return {
                "status": "success",
                "resume": False,
                "resume_count": 0,
                "started_at": expected_manifest.get("started_at"),
            }

        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            return _error_result(
                "现有运行清单不是JSON对象",
                "INVALID_RESUME_MANIFEST",
            )

        comparable_paths = (
            ("experiment_id",),
            ("dry_run",),
            ("development_run",),
            ("evaluation_scope",),
            ("dataset", "name"),
            ("dataset", "version"),
            ("dataset", "status"),
            ("dataset", "file_name"),
            ("dataset", "sha256"),
            ("dataset", "item_count"),
            ("dataset", "question_schema_version"),
            ("design", "conditions"),
            ("design", "repeats"),
            ("design", "model"),
            ("design", "temperature"),
            ("design", "seed"),
            ("design", "question_timeout_seconds"),
            ("design", "condition_order"),
            ("design", "question_ids"),
            ("design", "planning_unit"),
            ("design", "condition_plan_pairing"),
            ("design", "plan_source_file_name"),
            ("design", "plan_source_sha256"),
            ("scoring", "policy_version"),
            ("scoring", "review_blinded"),
            ("scoring", "blinding_key_fingerprint"),
            ("environment", "git_commit"),
            ("environment", "git_dirty"),
            ("environment", "python_version"),
            ("environment", "kg_snapshot"),
            ("artifacts", "trace_file_name"),
            ("artifacts", "failure_file_name"),
            ("artifacts", "manifest_file_name"),
            ("artifacts", "scoring_audit_file_name"),
            ("artifacts", "review_queue_file_name"),
            ("artifacts", "review_map_file_name"),
            ("artifacts", "review_template_file_name"),
            ("artifacts", "plan_ledger_file_name"),
        )

        def value_at(source: dict[str, Any], path: tuple[str, ...]) -> Any:
            try:
                current: Any = source
                for part in path:
                    if not isinstance(current, dict):
                        return None
                    current = current.get(part)
                return current
            except Exception as exc:
                logger.error("读取运行清单字段失败: %s", exc, exc_info=True)
                return None

        mismatches = [
            ".".join(path)
            for path in comparable_paths
            if value_at(existing, path) != value_at(expected_manifest, path)
        ]
        if mismatches:
            return _error_result(
                "恢复运行的控制变量与原清单不一致，禁止混写结果",
                "RESUME_MANIFEST_MISMATCH",
                details={"mismatched_fields": mismatches},
            )
        previous_resume_count = existing.get("resume_count")
        return {
            "status": "success",
            "resume": True,
            "resume_count": (
                int(previous_resume_count) + 1
                if isinstance(previous_resume_count, int)
                else 1
            ),
            "started_at": existing.get("started_at"),
        }
    except Exception as exc:
        logger.error("校验恢复运行清单失败: %s", exc, exc_info=True)
        return _error_result(
            f"校验恢复运行清单失败: {exc}",
            "RESUME_MANIFEST_EXCEPTION",
        )


async def run_biotooleval_experiment(
    dataset_path: Any,
    *,
    experiment_id: str,
    conditions: list[str] | tuple[str, ...] = ("B0", "B1", "B2"),
    repeats: int = 1,
    model: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    kg_snapshot: str | None = None,
    output_path: Any = None,
    timeout_seconds: float | None = None,
    dry_run: bool = False,
    development_run: bool = False,
    freeze_manifest_path: Any = None,
    frozen_plan_ledger_path: Any = None,
    question_ids: list[str] | tuple[str, ...] | None = None,
    retry_failures: bool = False,
    agent_callable: AgentCallable | None = None,
    plan_callable: PlanCallable | None = None,
    replan_callable: PlanCallable | None = None,
    runtime_config: Any = None,
    config: Any = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> dict[str, Any]:
    try:
        cfg = config if config is not None else Config()
        normalized_id = str(experiment_id or "").strip()
        if not _SAFE_EXPERIMENT_ID.fullmatch(normalized_id):
            return _error_result(
                "experiment_id只能包含字母、数字、点、短横线和下划线",
                "INVALID_EXPERIMENT_ID",
            )
        if not isinstance(repeats, int) or repeats < 1 or repeats > 100:
            return _error_result(
                "repeats必须是1到100之间的整数",
                "INVALID_REPEAT_COUNT",
            )
        normalized_conditions = [
            str(condition).strip() for condition in conditions
        ]
        if (
            not normalized_conditions
            or len(set(normalized_conditions)) != len(normalized_conditions)
        ):
            return _error_result(
                "conditions不能为空或重复",
                "INVALID_CONDITION_LIST",
            )
        for condition in normalized_conditions:
            condition_result = get_experiment_condition(condition)
            if condition_result.get("status") != "success":
                return condition_result

        dataset_result = load_biotooleval_dataset(
            dataset_path,
            dry_run=dry_run,
            development_run=development_run,
            config=cfg,
        )
        if dataset_result.get("status") != "success":
            return dataset_result

        normalized_question_ids = [
            str(question_id or "").strip()
            for question_id in (question_ids or [])
        ]
        if any(not question_id for question_id in normalized_question_ids):
            return _error_result(
                "question_ids不能包含空值",
                "INVALID_QUESTION_ID_SELECTION",
            )
        if len(set(normalized_question_ids)) != len(normalized_question_ids):
            return _error_result(
                "question_ids不能重复",
                "DUPLICATE_QUESTION_ID_SELECTION",
            )
        if normalized_question_ids:
            item_by_id = {
                str(item.get("id") or "").strip(): item
                for item in dataset_result["items"]
            }
            missing_ids = [
                question_id
                for question_id in normalized_question_ids
                if question_id not in item_by_id
            ]
            if missing_ids:
                return _error_result(
                    "question_ids包含数据集中不存在的题目",
                    "UNKNOWN_QUESTION_ID_SELECTION",
                    details={"question_ids": missing_ids},
                )
            dataset_result["items"] = [
                item_by_id[question_id]
                for question_id in normalized_question_ids
            ]
        else:
            normalized_question_ids = [
                str(item.get("id") or "").strip()
                for item in dataset_result["items"]
            ]
        selected_item_count = len(dataset_result["items"])

        active_model = str(
            (
                "offline-mock"
                if dry_run and model is None
                else model
            )
            if (dry_run or model is not None)
            else getattr(cfg, "MODEL_NAME", "")
        ).strip()
        if not active_model:
            return _error_result(
                "正式实验必须记录非空模型名",
                "MISSING_MODEL_NAME",
            )
        active_temperature = float(
            temperature
            if temperature is not None
            else getattr(cfg, "LLM_TEMPERATURE", 0.0)
        )
        active_timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else getattr(
                cfg,
                "EXPERIMENT_QUESTION_TIMEOUT_SECONDS",
                120.0,
            )
        )
        if active_timeout <= 0:
            return _error_result(
                "单题超时必须大于0秒",
                "INVALID_QUESTION_TIMEOUT",
            )

        results_dir = Path(
            getattr(cfg, "EXPERIMENT_RESULTS_DIR", "evaluation/results")
        ).expanduser()
        trace_path = (
            Path(str(output_path)).expanduser()
            if output_path is not None
            else results_dir / f"{normalized_id}.jsonl"
        )
        if trace_path.suffix.lower() != ".jsonl":
            return _error_result(
                "实验输出必须使用.jsonl扩展名",
                "INVALID_EXPERIMENT_OUTPUT",
            )
        failure_path = trace_path.with_name(
            f"{trace_path.stem}.failures.jsonl"
        )
        manifest_path = trace_path.with_name(
            f"{trace_path.stem}.manifest.json"
        )
        scoring_audit_path = trace_path.with_name(
            f"{trace_path.stem}.scoring.jsonl"
        )
        review_queue_path = trace_path.with_name(
            f"{trace_path.stem}.review_queue.jsonl"
        )
        review_map_path = trace_path.with_name(
            f"{trace_path.stem}.review_map.jsonl"
        )
        review_template_path = trace_path.with_name(
            f"{trace_path.stem}.review_labels.template.jsonl"
        )
        plan_ledger_path = trace_path.with_name(
            f"{trace_path.stem}.plans.jsonl"
        )
        replan_ledger_path = trace_path.with_name(
            f"{trace_path.stem}.replans.jsonl"
        )

        if agent_callable is None or runtime_config is None:
            if dry_run:
                runtime_config = runtime_config or SimpleNamespace(
                    TYPE_PLANNING_ENABLED=False,
                    TYPE_REPAIR_ENABLED=False,
                )
                agent_callable = agent_callable or create_offline_mock_agent(
                    runtime_config
                )
                plan_callable = (
                    plan_callable or create_offline_mock_plan_sampler()
                )
                replan_callable = (
                    replan_callable or create_offline_mock_replanner()
                )
            else:
                runtime_result = _load_default_agent_runtime()
                if runtime_result.get("status") != "success":
                    return runtime_result
                agent_callable = agent_callable or runtime_result[
                    "agent_callable"
                ]
                plan_callable = plan_callable or runtime_result[
                    "plan_callable"
                ]
                replan_callable = replan_callable or runtime_result[
                    "replan_callable"
                ]
                runtime_config = runtime_config or runtime_result[
                    "runtime_config"
                ]
        if not callable(agent_callable) or runtime_config is None:
            return _error_result(
                "Agent执行器或运行配置不可用",
                "INVALID_AGENT_RUNTIME",
            )
        if any(
            condition in {"L1", "F1"}
            for condition in normalized_conditions
        ) and not callable(replan_callable):
            return _error_result(
                "L1/F1条件必须提供单次LLM重规划器",
                "MISSING_LLM_REPLAN_RUNTIME",
            )
        external_plan_replay = frozen_plan_ledger_path is not None
        plan_read_path = plan_ledger_path
        plan_source_sha256 = None
        if external_plan_replay:
            plan_read_path = Path(str(frozen_plan_ledger_path)).expanduser().resolve()
            try:
                plan_read_path.relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                return _error_result(
                    "外部冻结计划账本必须位于项目目录内",
                    "PLAN_SOURCE_OUTSIDE_PROJECT",
                )
            if not plan_read_path.is_file() or plan_read_path.suffix.lower() != ".jsonl":
                return _error_result(
                    "外部冻结计划账本不存在或扩展名错误",
                    "PLAN_SOURCE_NOT_FOUND",
                )
            plan_source_sha256 = _sha256_file(plan_read_path)
        shared_plan_replay = external_plan_replay or callable(plan_callable)
        if not shared_plan_replay and active_model != "offline-mock":
            return _error_result(
                "真实模型实验必须提供计划采样器，禁止B0/B1/B2独立重规划",
                "MISSING_PLAN_CAPTURE_RUNTIME",
            )

        completed_result = _read_completed_keys(trace_path, config=cfg)
        if completed_result.get("status") != "success":
            return completed_result
        failure_result = _read_failure_keys(failure_path)
        if failure_result.get("status") != "success":
            return failure_result
        completed_keys = completed_result["keys"]
        failed_keys = failure_result["keys"]
        plan_samples_result = load_plan_samples(plan_read_path)
        if plan_samples_result.get("status") != "success":
            return _error_result(
                str(plan_samples_result.get("message") or "读取计划账本失败"),
                str(
                    plan_samples_result.get("code")
                    or "PLAN_LEDGER_READ_FAILURE"
                ),
            )
        frozen_plan_by_key = plan_samples_result["records"]
        if external_plan_replay:
            required_plan_keys = {
                (question_id, repeat_index)
                for question_id in normalized_question_ids
                for repeat_index in range(repeats)
            }
            missing_plan_keys = sorted(required_plan_keys - set(frozen_plan_by_key))
            if missing_plan_keys:
                return _error_result(
                    "外部冻结计划账本未覆盖全部选择单元，禁止重新采样",
                    "PLAN_SOURCE_COVERAGE_MISMATCH",
                    details={"missing_count": len(missing_plan_keys)},
                )
            mismatched_models = sorted(
                {
                    str(frozen_plan_by_key[key].get("model") or "")
                    for key in required_plan_keys
                    if str(frozen_plan_by_key[key].get("model") or "")
                    != active_model
                }
            )
            if mismatched_models:
                return _error_result(
                    "外部冻结计划账本模型与本次执行模型不一致",
                    "PLAN_SOURCE_MODEL_MISMATCH",
                    details={"observed_models": mismatched_models},
                )

        git_state = _resolve_git_state()
        resolved_commit = str(
            git_commit or git_state.get("git_commit") or "unknown"
        ).strip()
        resolved_dirty = (
            bool(git_dirty)
            if isinstance(git_dirty, bool)
            else git_state.get("git_dirty")
        )
        if len(resolved_commit) < 7:
            resolved_commit = "unknown"
        freeze_verification: dict[str, Any] | None = None
        if development_run and freeze_manifest_path is not None:
            freeze_verification = _verify_development_freeze_manifest(
                freeze_manifest_path,
                dataset_result=dataset_result,
                experiment_id=normalized_id,
                conditions=normalized_conditions,
                repeats=repeats,
                model=active_model,
                temperature=active_temperature,
                seed=seed,
                trace_path=trace_path,
                plan_source_path=(
                    plan_read_path if external_plan_replay else None
                ),
            )
            if freeze_verification.get("status") != "success":
                return freeze_verification
        dirty_development_snapshot = bool(
            development_run
            and resolved_dirty is True
            and isinstance(freeze_verification, dict)
            and freeze_verification.get("status") == "success"
        )
        if not dry_run and (
            resolved_commit == "unknown"
            or (resolved_dirty is not False and not dirty_development_snapshot)
        ):
            return _error_result(
                "实验必须绑定干净提交；dirty开发运行需提供匹配的逐文件冻结清单",
                "UNFROZEN_CODE_STATE",
                details={
                    "git_commit": resolved_commit,
                    "git_dirty": resolved_dirty,
                },
            )
        review_blinding_key = str(
            getattr(cfg, "EXPERIMENT_REVIEW_BLINDING_KEY", "") or ""
        ).strip()
        if dry_run and len(review_blinding_key) < 8:
            review_blinding_key = "offline-review-key"
        if len(review_blinding_key) < 8:
            return _error_result(
                "正式实验必须在.env设置至少8字符的"
                "EXPERIMENT_REVIEW_BLINDING_KEY",
                "MISSING_REVIEW_BLINDING_KEY",
            )
        review_key_fingerprint = hashlib.sha256(
            review_blinding_key.encode("utf-8")
        ).hexdigest()[:16]
        started_at = _utc_now_text()
        summary = {
            "planned": (
                selected_item_count
                * repeats
                * len(normalized_conditions)
            ),
            "completed": 0,
            "failed": 0,
            "skipped_completed": 0,
            "skipped_failed": 0,
            "automatic_labels": 0,
            "manual_required_labels": 0,
        }
        manifest = {
            "schema_version": "1.0",
            "experiment_id": normalized_id,
            "status": "running",
            "dry_run": bool(dry_run),
            "development_run": bool(development_run),
            "evaluation_scope": dataset_result["evaluation_scope"],
            "development_freeze": (
                copy.deepcopy(freeze_verification)
                if isinstance(freeze_verification, dict)
                else None
            ),
            "started_at": started_at,
            "finished_at": None,
            "dataset": {
                "name": dataset_result["dataset_name"],
                "version": dataset_result["dataset_version"],
                "status": dataset_result["dataset_status"],
                "file_name": dataset_result["dataset_file_name"],
                "sha256": dataset_result["dataset_sha256"],
                "item_count": dataset_result["item_count"],
                "question_schema_version": dataset_result[
                    "question_schema_version"
                ],
            },
            "design": {
                "conditions": normalized_conditions,
                "repeats": repeats,
                "model": active_model,
                "temperature": active_temperature,
                "seed": seed,
                "question_timeout_seconds": active_timeout,
                "condition_order": "cyclic_counterbalanced_within_question",
                "condition_schedule": [
                    _condition_order_for_repeat(normalized_conditions, repeat_index)
                    for repeat_index in range(repeats)
                ],
                "question_ids": normalized_question_ids,
                "retry_failures": bool(retry_failures),
                "planning_unit": "question_repeat",
                "condition_plan_pairing": (
                    "external_shared_frozen_plan"
                    if external_plan_replay
                    else "shared_frozen_plan"
                    if shared_plan_replay
                    else "legacy_independent_agent_calls"
                ),
                "plan_source_file_name": (
                    plan_read_path.name if external_plan_replay else None
                ),
                "plan_source_sha256": plan_source_sha256,
            },
            "environment": {
                "git_commit": resolved_commit,
                "git_dirty": resolved_dirty,
                "python_version": platform.python_version(),
                "kg_snapshot": kg_snapshot,
            },
            "scoring": {
                "policy_version": SCORING_POLICY_VERSION,
                "review_blinded": True,
                "blinding_key_fingerprint": review_key_fingerprint,
                "uncertain_label_policy": "manual_required_as_null",
            },
            "summary": copy.deepcopy(summary),
            "resume_count": 0,
            "artifacts": {
                "trace_file_name": trace_path.name,
                "failure_file_name": failure_path.name,
                "manifest_file_name": manifest_path.name,
                "scoring_audit_file_name": scoring_audit_path.name,
                "review_queue_file_name": review_queue_path.name,
                "review_map_file_name": review_map_path.name,
                "review_template_file_name": review_template_path.name,
                "plan_ledger_file_name": plan_ledger_path.name,
                "replan_ledger_file_name": replan_ledger_path.name,
            },
        }
        resume_result = _validate_resume_manifest(
            manifest_path,
            manifest,
            trace_path=trace_path,
            failure_path=failure_path,
        )
        if resume_result.get("status") != "success":
            return resume_result
        manifest["resume_count"] = resume_result["resume_count"]
        if resume_result.get("started_at"):
            manifest["started_at"] = resume_result["started_at"]
        manifest_result = _write_manifest(manifest_path, manifest)
        if manifest_result.get("status") != "success":
            return manifest_result

        for repeat_index in range(repeats):
            for question_item in dataset_result["items"]:
                request_result = prepare_agent_request(
                    question_item,
                    dataset_dir=dataset_result["dataset_dir"],
                )
                initial_artifact_result = (
                    build_initial_artifacts(
                        request_result.get("question"),
                        request_result.get("file_path_list"),
                    )
                    if request_result.get("status") == "success"
                    else {
                        "status": "error",
                        "available_input_types": [],
                        "issues": request_result.get("issues") or [],
                    }
                )
                initial_types = list(
                    initial_artifact_result.get("available_input_types")
                    or []
                )
                initial_artifact_issue_codes = sorted(
                    {
                        str(issue.get("code"))
                        for issue in initial_artifact_result.get("issues")
                        or []
                        if isinstance(issue, dict) and issue.get("code")
                    }
                )
                ordered_conditions = _condition_order_for_repeat(
                    normalized_conditions,
                    repeat_index,
                )
                question_id = str(question_item.get("id") or "")
                active_conditions = []
                for candidate_condition in ordered_conditions:
                    candidate_key = _execution_key(
                        question_id,
                        active_model,
                        candidate_condition,
                        repeat_index,
                    )
                    if candidate_key in completed_keys:
                        continue
                    if candidate_key in failed_keys and not retry_failures:
                        continue
                    active_conditions.append(candidate_condition)

                frozen_plan = None
                plan_capture_error = None
                plan_key = (question_id, repeat_index)
                if (
                    active_conditions
                    and request_result.get("status") == "success"
                    and shared_plan_replay
                ):
                    frozen_plan = frozen_plan_by_key.get(plan_key)
                    if frozen_plan is None and not external_plan_replay:
                        capture_result = await _capture_frozen_plan(
                            plan_callable,
                            request_result,
                            experiment_id=normalized_id,
                            question_id=question_id,
                            repeat_index=repeat_index,
                            model=active_model,
                            timeout_seconds=active_timeout,
                        )
                        if capture_result.get("status") == "success":
                            frozen_plan = capture_result["frozen_plan"]
                            ledger_result = append_plan_sample(
                                plan_ledger_path,
                                frozen_plan,
                            )
                            if ledger_result.get("status") != "success":
                                return _error_result(
                                    "冻结计划未能写入私有账本，实验已停止",
                                    str(
                                        ledger_result.get("code")
                                        or "PLAN_LEDGER_WRITE_FAILURE"
                                    ),
                                )
                            frozen_plan_by_key[plan_key] = frozen_plan
                        else:
                            plan_capture_error = capture_result
                            capture_issue = (
                                (capture_result.get("issues") or [{}])[0]
                                if isinstance(capture_result, dict)
                                else {}
                            )
                            failed_sample = {
                                "schema_version": "1.0",
                                "record_status": "capture_failed",
                                "experiment_id": normalized_id,
                                "question_id": question_id,
                                "repeat_index": repeat_index,
                                "model": active_model,
                                "occurred_at": _utc_now_text(),
                                "error_code": str(
                                    capture_issue.get("code")
                                    or "PLAN_CAPTURE_FAILURE"
                                ),
                                "attempts": list(
                                    capture_issue.get("attempts") or []
                                ),
                            }
                            ledger_result = append_plan_sample(
                                plan_ledger_path,
                                failed_sample,
                            )
                            if ledger_result.get("status") != "success":
                                return _error_result(
                                    "失败计划尝试未能写入私有账本，实验已停止",
                                    str(
                                        ledger_result.get("code")
                                        or "PLAN_LEDGER_WRITE_FAILURE"
                                    ),
                                )

                for condition in ordered_conditions:
                    key = _execution_key(
                        question_item.get("id"),
                        active_model,
                        condition,
                        repeat_index,
                    )
                    if key in completed_keys:
                        summary["skipped_completed"] += 1
                        continue
                    if key in failed_keys and not retry_failures:
                        summary["skipped_failed"] += 1
                        continue

                    error_result: dict[str, Any] | None = None
                    if request_result.get("status") != "success":
                        error_result = request_result
                    elif plan_capture_error is not None:
                        error_result = plan_capture_error
                    else:
                        execution_result = await _execute_under_condition(
                            agent_callable,
                            runtime_config,
                            request_result,
                            condition=condition,
                            timeout_seconds=active_timeout,
                            frozen_plan=frozen_plan,
                            model=active_model,
                            replan_callable=replan_callable,
                            initial_types=initial_types,
                            initial_artifact_issue_codes=(
                                initial_artifact_issue_codes
                            ),
                            replan_max_output_tokens=int(
                                getattr(
                                    cfg,
                                    "EXPERIMENT_LLM_REPLAN_MAX_OUTPUT_TOKENS",
                                    256,
                                )
                            ),
                        )
                        if execution_result.get("status") != "success":
                            error_result = execution_result
                        else:
                            if isinstance(
                                execution_result.get("replan_ledger_record"),
                                dict,
                            ):
                                replan_ledger_result = append_plan_sample(
                                    replan_ledger_path,
                                    execution_result["replan_ledger_record"],
                                )
                                if (
                                    replan_ledger_result.get("status")
                                    != "success"
                                ):
                                    error_result = _error_result(
                                        "L1重规划未能写入私有账本",
                                        str(
                                            replan_ledger_result.get("code")
                                            or "REPLAN_LEDGER_WRITE_FAILURE"
                                        ),
                                    )
                            if error_result is not None:
                                pass
                            else:
                                run_context = {
                                    "run_id": (
                                        f"{normalized_id}-{condition.lower()}"
                                        f"-r{repeat_index}"
                                    ),
                                    "repeat_index": repeat_index,
                                    "model": active_model,
                                    "temperature": active_temperature,
                                    "seed": seed,
                                    "git_commit": resolved_commit,
                                    "dataset_version": dataset_result[
                                        "dataset_version"
                                    ],
                                    "kg_snapshot": kg_snapshot,
                                    "python_version": platform.python_version(),
                                    "total_latency_ms": execution_result[
                                        "total_latency_ms"
                                    ],
                                    "initial_types": initial_types,
                                    "initial_artifact_issue_codes": (
                                        initial_artifact_issue_codes
                                    ),
                                }
                                trace_result = build_experiment_trace(
                                    question_item,
                                    execution_result["agent_result"],
                                    condition=condition,
                                    run_context=run_context,
                                    evaluation_labels=None,
                                    config=cfg,
                                )
                                if trace_result.get("status") != "success":
                                    error_result = trace_result
                                else:
                                    scoring_result = score_experiment_trace(
                                        question_item,
                                        execution_result["agent_result"],
                                        trace_result["trace"],
                                        config=cfg,
                                    )
                                    if (
                                        scoring_result.get("status")
                                        != "success"
                                    ):
                                        error_result = scoring_result
                                    else:
                                        applied_result = apply_scoring_to_trace(
                                            trace_result["trace"],
                                            scoring_result,
                                            config=cfg,
                                        )
                                        if (
                                            applied_result.get("status")
                                            != "success"
                                        ):
                                            error_result = applied_result
                                        else:
                                            scoring_bundle = (
                                                persist_scoring_bundle(
                                                    question_item,
                                                    execution_result[
                                                        "agent_result"
                                                    ],
                                                    applied_result["trace"],
                                                    scoring_result,
                                                    experiment_id=normalized_id,
                                                    blinding_key=(
                                                        review_blinding_key
                                                    ),
                                                    audit_path=(
                                                        scoring_audit_path
                                                    ),
                                                    review_queue_path=(
                                                        review_queue_path
                                                    ),
                                                    review_map_path=(
                                                        review_map_path
                                                    ),
                                                    review_template_path=(
                                                        review_template_path
                                                    ),
                                                )
                                            )
                                            if (
                                                scoring_bundle.get("status")
                                                != "success"
                                            ):
                                                error_result = scoring_bundle
                                            else:
                                                append_result = (
                                                    append_experiment_trace(
                                                        applied_result["trace"],
                                                        output_path=trace_path,
                                                        config=cfg,
                                                    )
                                                )
                                                if (
                                                    append_result.get("status")
                                                    != "success"
                                                ):
                                                    error_result = append_result
                                                else:
                                                    summary[
                                                        "automatic_labels"
                                                    ] += scoring_result[
                                                        "automatic_label_count"
                                                    ]
                                                    summary[
                                                        "manual_required_labels"
                                                    ] += scoring_result[
                                                        "manual_required_count"
                                                    ]

                    if error_result is not None:
                        issue = (
                            (error_result.get("issues") or [{}])[0]
                            if isinstance(error_result, dict)
                            else {}
                        )
                        ledger_result = _append_failure_record(
                            failure_path,
                            experiment_id=normalized_id,
                            question_id=str(question_item.get("id") or ""),
                            model=active_model,
                            condition=condition,
                            repeat_index=repeat_index,
                            status="failed",
                            error_code=str(issue.get("code") or "UNKNOWN"),
                            detail_code=(
                                str(issue.get("agent_error_code"))
                                if issue.get("agent_error_code")
                                else None
                            ),
                            failure_category=_classify_failure(
                                issue.get("code"),
                                issue.get("agent_error_code"),
                            ),
                            exception_type=(
                                str(issue.get("exception_type"))
                                if issue.get("exception_type")
                                else None
                            ),
                        )
                        if ledger_result.get("status") != "success":
                            logger.error(
                                "失败题目未能写入账本 | question=%s | "
                                "condition=%s",
                                question_item.get("id"),
                                condition,
                            )
                        failed_keys.add(key)
                        summary["failed"] += 1
                        continue

                    if key in failed_keys:
                        _append_failure_record(
                            failure_path,
                            experiment_id=normalized_id,
                            question_id=str(question_item.get("id") or ""),
                            model=active_model,
                            condition=condition,
                            repeat_index=repeat_index,
                            status="resolved",
                        )
                        failed_keys.discard(key)
                    completed_keys.add(key)
                    summary["completed"] += 1

        manifest["status"] = (
            "completed_with_failures"
            if summary["failed"] > 0
            else "completed"
        )
        manifest["finished_at"] = _utc_now_text()
        summary["total_completed_records"] = len(completed_keys)
        summary["total_unresolved_failures"] = len(failed_keys)
        summary["total_review_queue_records"] = _count_jsonl_records(
            review_queue_path
        )
        summary["total_plan_ledger_records"] = _count_jsonl_records(
            plan_read_path
        )
        summary["total_replan_ledger_records"] = _count_jsonl_records(
            replan_ledger_path
        )
        scoring_totals = _summarize_scoring_audit(
            scoring_audit_path
        )
        if scoring_totals.get("status") != "success":
            return scoring_totals
        summary["automatic_labels"] = scoring_totals[
            "automatic_labels"
        ]
        summary["manual_required_labels"] = scoring_totals[
            "manual_required_labels"
        ]
        manifest["summary"] = copy.deepcopy(summary)
        final_manifest = _write_manifest(manifest_path, manifest)
        if final_manifest.get("status") != "success":
            return final_manifest
        logger.info(
            "BioToolEval运行完成 | experiment=%s | planned=%s | "
            "completed=%s | failed=%s | skipped=%s",
            normalized_id,
            summary["planned"],
            summary["completed"],
            summary["failed"],
            summary["skipped_completed"] + summary["skipped_failed"],
        )
        return {
            "status": "success",
            "experiment_id": normalized_id,
            "run_status": manifest["status"],
            "dry_run": bool(dry_run),
            "development_run": bool(development_run),
            "evaluation_scope": dataset_result["evaluation_scope"],
            "summary": summary,
            "trace_path": str(trace_path.resolve()),
            "failure_path": str(failure_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "scoring_audit_path": str(scoring_audit_path.resolve()),
            "review_queue_path": str(review_queue_path.resolve()),
            "review_map_path": str(review_map_path.resolve()),
            "review_template_path": str(review_template_path.resolve()),
            "plan_ledger_path": str(plan_ledger_path.resolve()),
            "plan_source_path": (
                str(plan_read_path) if external_plan_replay else None
            ),
            "replan_ledger_path": str(replan_ledger_path.resolve()),
        }
    except Exception as exc:
        logger.error("BioToolEval批量实验崩溃: %s", exc, exc_info=True)
        return _error_result(
            f"BioToolEval批量实验崩溃: {exc}",
            "EXPERIMENT_RUN_EXCEPTION",
        )


def _build_cli_parser() -> argparse.ArgumentParser:
    try:
        parser = argparse.ArgumentParser(
            description="运行BioToolEval历史、E0-E3、S1/L1或F1分层恢复实验"
        )
        parser.add_argument("--dataset", help="BioToolEval JSON数据集路径")
        parser.add_argument("--experiment-id", help="实验唯一标识")
        parser.add_argument(
            "--conditions",
            nargs="+",
            default=["B0", "B1", "B2"],
            choices=sorted(EXPERIMENT_CONDITIONS),
        )
        parser.add_argument("--repeats", type=int, default=1)
        parser.add_argument("--model")
        parser.add_argument("--temperature", type=float)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--kg-snapshot")
        parser.add_argument("--output")
        parser.add_argument("--timeout-seconds", type=float)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument(
            "--development-run",
            action="store_true",
            help="允许开发候选集进入预实验；产物强制标记development_only",
        )
        parser.add_argument(
            "--freeze-manifest",
            help="dirty开发运行使用的逐文件SHA-256冻结清单",
        )
        parser.add_argument(
            "--frozen-plan-ledger",
            help="只读复用既有共享原始计划JSONL；缺失单元时禁止重新采样",
        )
        parser.add_argument(
            "--question-ids",
            nargs="+",
            default=None,
            help="只运行指定题目ID，用于预注册烟雾子集",
        )
        parser.add_argument("--retry-failures", action="store_true")
        return parser
    except Exception as exc:
        logger.error("构建CLI解析器失败: %s", exc, exc_info=True)
        raise RuntimeError("无法构建BioToolEval命令行") from exc


def _default_experiment_id(dry_run: bool) -> str:
    try:
        prefix = "dry-run" if dry_run else "biotooleval"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{prefix}-{stamp}"
    except Exception as exc:
        logger.error("生成默认实验ID失败: %s", exc, exc_info=True)
        return "biotooleval-unknown"


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _build_cli_parser()
        args = parser.parse_args(argv)
        cfg = Config()
        dataset_path = args.dataset
        if args.dry_run and not dataset_path:
            dataset_path = cfg.EXPERIMENT_DRY_RUN_DATASET_PATH
        if not dataset_path:
            parser.error("正式运行必须提供--dataset；离线检查可使用--dry-run")
        experiment_id = (
            args.experiment_id or _default_experiment_id(args.dry_run)
        )
        result = asyncio.run(
            run_biotooleval_experiment(
                dataset_path,
                experiment_id=experiment_id,
                conditions=args.conditions,
                repeats=args.repeats,
                model=args.model,
                temperature=args.temperature,
                seed=args.seed,
                kg_snapshot=args.kg_snapshot,
                output_path=args.output,
                timeout_seconds=args.timeout_seconds,
                dry_run=args.dry_run,
                development_run=args.development_run,
                freeze_manifest_path=args.freeze_manifest,
                frozen_plan_ledger_path=args.frozen_plan_ledger,
                question_ids=args.question_ids,
                retry_failures=args.retry_failures,
                config=cfg,
            )
        )
        safe_output = {
            "status": result.get("status"),
            "experiment_id": result.get("experiment_id"),
            "run_status": result.get("run_status"),
            "dry_run": result.get("dry_run"),
            "development_run": result.get("development_run"),
            "evaluation_scope": result.get("evaluation_scope"),
            "summary": result.get("summary"),
            "trace_path": result.get("trace_path"),
            "failure_path": result.get("failure_path"),
            "manifest_path": result.get("manifest_path"),
            "scoring_audit_path": result.get("scoring_audit_path"),
            "review_queue_path": result.get("review_queue_path"),
            "review_map_path": result.get("review_map_path"),
            "review_template_path": result.get("review_template_path"),
            "plan_ledger_path": result.get("plan_ledger_path"),
            "replan_ledger_path": result.get("replan_ledger_path"),
            "message": result.get("message"),
        }
        print(json.dumps(safe_output, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "success" else 1
    except SystemExit:
        raise


    except Exception as exc:
        logger.error("BioToolEval CLI运行失败: %s", exc, exc_info=True)
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": f"BioToolEval CLI运行失败: {exc}",
                },
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
