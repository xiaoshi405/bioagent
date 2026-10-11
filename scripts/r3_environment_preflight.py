# -*- coding: utf-8 -*-
"""
BioAgent v0.3e 正式实验双服务预检工具。

修改记录：
- 2026-09-05：数据集摘要统一为 UTF-8/LF 文本哈希，与运行器及冻结清单
  口径一致，避免 Windows CRLF 产生两个看似冲突的数据集指纹。
- 2026-08-27：正式预检新增本地BGE配置、权重和tokenizer资产校验，
  在启动实验前给出精确阻断原因，避免首题Trace之前导入崩溃。
- 2026-08-10：新增根目录、Git、数据集哈希和fixture静态检查。
- 2026-08-11：改为正式模式默认失败关闭；新增真实Agent/Biology服务身份、
  FASTA/PDB HTTP贯通、脏工作树、Python版本和题目Schema阻断。

设计原则：
1. 正式模式必须查询实际运行的两个服务，不能信任调用者填写的路径；
2. 任一身份、版本、数据或HTTP检查失败均返回error和非零退出码；
3. ``--static-only`` 与 ``--allow-dirty`` 仅用于本地诊断，不代表实验放行；
4. 所有异常转换为标准错误结构，预检本身不得导致主进程崩溃。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from jsonschema import Draft202012Validator, FormatChecker

from app.core.config import Config
from app.core.embedding_runtime import validate_local_embedding_model
from app.core.logger import get_logger
from app.core.runtime_identity import collect_git_identity

logger = get_logger("R3EnvironmentPreflight", "r3_preflight.log")


def _get_git_info(repo_dir: Path) -> dict[str, Any]:
    """保留旧接口名称，统一调用共享Git身份采集器。"""
    try:
        return collect_git_identity(repo_dir)
    except Exception as exc:
        logger.warning("获取Git信息失败 (%s): %s", repo_dir, exc)
        return {
            "status": "error",
            "commit": None,
            "is_dirty": None,
            "message": str(exc),
        }


def _calc_sha256(file_path: Path) -> str | None:
    """按 UTF-8/LF 计算数据集SHA-256；失败返回None。"""
    try:
        if not file_path.is_file():
            return None
        digest = hashlib.sha256()
        with file_path.open("r", encoding="utf-8", newline=None) as file_handle:
            while True:
                chunk = file_handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk.encode("utf-8"))
        return digest.hexdigest()
    except Exception as exc:
        logger.error("计算SHA-256失败 (%s): %s", file_path, exc)
        return None


def _version_matches(actual: str, expected: str) -> bool:
    """支持完整版本或主次版本前缀，例如3.14匹配3.14.4。"""
    try:
        actual_value = str(actual or "").strip()
        expected_value = str(expected or "").strip()
        if not actual_value or not expected_value:
            return False
        return actual_value == expected_value or actual_value.startswith(
            f"{expected_value}."
        )
    except Exception:
        return False


def _append_check(
    checks: list[dict[str, Any]],
    blockers: list[str],
    *,
    name: str,
    passed: bool,
    detail: str,
) -> None:
    """追加单项检查；失败项同步写入阻断原因。"""
    try:
        checks.append(
            {
                "name": str(name),
                "status": "pass" if passed else "fail",
                "detail": str(detail),
            }
        )
        if not passed:
            blockers.append(str(detail))
    except Exception as exc:
        blockers.append(f"记录预检项{name}失败: {exc}")


def _load_and_validate_dataset(
    dataset_path: Path,
    schema_path: Path,
) -> dict[str, Any]:
    """验证数据集顶层结构、题目Schema、题目ID唯一性和声明题量。"""
    try:
        if not dataset_path.is_file():
            return {
                "status": "error",
                "items": [],
                "message": f"数据集文件不存在: {dataset_path}",
            }
        if not schema_path.is_file():
            return {
                "status": "error",
                "items": [],
                "message": f"题目Schema不存在: {schema_path}",
            }

        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)

        if not isinstance(dataset, dict):
            return {
                "status": "error",
                "items": [],
                "message": "数据集顶层必须是JSON对象",
            }
        items = dataset.get("items")
        if not isinstance(items, list) or not items:
            return {
                "status": "error",
                "items": [],
                "message": "数据集必须包含非空items数组",
            }
        if any(not isinstance(item, dict) for item in items):
            return {
                "status": "error",
                "items": [],
                "message": "items中的每道题都必须是JSON对象",
            }

        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        schema_errors: list[str] = []
        for index, item in enumerate(items):
            item_errors = sorted(validator.iter_errors(item), key=lambda err: list(err.path))
            for error in item_errors[:3]:
                location = ".".join(str(part) for part in error.path) or "root"
                schema_errors.append(f"items[{index}].{location}: {error.message}")
            if len(schema_errors) >= 10:
                break
        if schema_errors:
            return {
                "status": "error",
                "items": items,
                "message": f"题目Schema校验失败: {schema_errors[:5]}",
            }

        item_ids = [str(item.get("id") or "") for item in items]
        if len(set(item_ids)) != len(item_ids):
            return {
                "status": "error",
                "items": items,
                "message": "数据集包含重复题目ID",
            }

        summary = dataset.get("summary")
        if isinstance(summary, dict) and "item_count" in summary:
            declared_count = summary.get("item_count")
            if not isinstance(declared_count, int) or declared_count != len(items):
                return {
                    "status": "error",
                    "items": items,
                    "message": (
                        "summary.item_count与真实题量不一致: "
                        f"declared={declared_count}, actual={len(items)}"
                    ),
                }

        return {
            "status": "success",
            "items": items,
            "message": f"{len(items)}道题通过顶层结构与题目Schema校验",
        }
    except json.JSONDecodeError as exc:
        return {
            "status": "error",
            "items": [],
            "message": f"数据集JSON解析损坏: {exc}",
        }
    except Exception as exc:
        logger.error("数据集结构校验失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "items": [],
            "message": f"数据集结构校验异常: {exc}",
        }


def _resolve_fixture(
    raw_fixture: str,
    *,
    dataset_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    """在安全根目录内解析fixture，禁止绝对路径和父目录越界。"""
    try:
        raw_value = str(raw_fixture or "").strip()
        if not raw_value:
            return {"status": "error", "path": None, "message": "fixture路径为空"}
        relative_path = Path(raw_value)
        if relative_path.is_absolute():
            return {
                "status": "error",
                "path": None,
                "message": f"fixture禁止使用绝对路径: {raw_value}",
            }

        candidates = [
            dataset_dir / relative_path,
            project_root / relative_path,
            project_root / "evaluation" / "biotooleval" / relative_path,
            project_root / "tests" / relative_path,
        ]
        resolved_candidates: list[str] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if not resolved.is_relative_to(project_root):
                return {
                    "status": "error",
                    "path": None,
                    "message": f"fixture越界逃逸出项目根目录: {raw_value} -> {resolved}",
                }
            resolved_candidates.append(str(resolved))
            if resolved.is_file():
                return {
                    "status": "success",
                    "path": resolved,
                    "message": None,
                }
        return {
            "status": "error",
            "path": None,
            "message": f"fixture不存在: {raw_value}; candidates={resolved_candidates[:3]}",
        }
    except Exception as exc:
        return {
            "status": "error",
            "path": None,
            "message": f"解析fixture失败 ({raw_fixture}): {exc}",
        }


def _normalize_base_url(value: str) -> str:
    """规范化本地服务地址，避免localhost与127.0.0.1造成假不一致。"""
    try:
        parsed = urlsplit(str(value or "").strip().rstrip("/"))
        hostname = parsed.hostname or ""
        if hostname.lower() == "localhost":
            hostname = "127.0.0.1"
        port = f":{parsed.port}" if parsed.port is not None else ""
        netloc = f"{hostname}{port}"
        return urlunsplit((parsed.scheme.lower(), netloc, parsed.path.rstrip("/"), "", ""))
    except Exception:
        return str(value or "").strip().rstrip("/")


def _http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """执行有超时的JSON HTTP请求，并把网络异常转换为标准结构。"""
    try:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            url=str(url),
            data=body,
            headers=headers,
            method=str(method or "GET").upper(),
        )
        with urlopen(request, timeout=float(timeout_seconds)) as response:
            response_body = response.read().decode("utf-8")
            parsed_body = json.loads(response_body)
            if not isinstance(parsed_body, dict):
                return {
                    "status": "error",
                    "http_status": int(response.status),
                    "message": "服务响应不是JSON对象",
                    "data": None,
                }
            return {
                "status": "success",
                "http_status": int(response.status),
                "message": None,
                "data": parsed_body,
            }
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        return {
            "status": "error",
            "http_status": int(exc.code),
            "message": f"HTTP {exc.code}: {detail}",
            "data": None,
        }
    except (URLError, TimeoutError) as exc:
        return {
            "status": "error",
            "http_status": None,
            "message": f"连接服务失败: {exc}",
            "data": None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "http_status": None,
            "message": f"HTTP预检异常: {exc}",
            "data": None,
        }


def _validate_runtime_identity(
    identity: Any,
    *,
    expected_service: str,
    expected_root: Path,
    expected_commit: str,
    expected_python_version: str,
    expected_port: int | None,
    allow_dirty: bool,
) -> dict[str, Any]:
    """校验服务真实身份是否与冻结的本地实验环境一致。"""
    try:
        if not isinstance(identity, dict):
            return {"status": "error", "message": "运行身份响应不是JSON对象"}
        if identity.get("status") != "success":
            return {
                "status": "error",
                "message": f"服务运行身份采集失败: {identity.get('message')}",
            }
        if identity.get("service") != expected_service:
            return {
                "status": "error",
                "message": (
                    f"服务身份错误: expected={expected_service}, "
                    f"actual={identity.get('service')}"
                ),
            }
        actual_root = Path(str(identity.get("project_root") or "")).expanduser().resolve()
        if actual_root != expected_root:
            return {
                "status": "error",
                "message": (
                    f"{expected_service}实际项目根目录不一致: "
                    f"expected={expected_root}, actual={actual_root}"
                ),
            }
        if identity.get("git_commit") != expected_commit:
            return {
                "status": "error",
                "message": (
                    f"{expected_service}实际Git提交不一致: "
                    f"expected={expected_commit}, actual={identity.get('git_commit')}"
                ),
            }
        if identity.get("git_dirty") is True and not allow_dirty:
            return {
                "status": "error",
                "message": f"{expected_service}服务运行于脏工作树",
            }
        if not _version_matches(
            str(identity.get("python_version") or ""),
            expected_python_version,
        ):
            return {
                "status": "error",
                "message": (
                    f"{expected_service} Python版本不一致: "
                    f"expected={expected_python_version}, "
                    f"actual={identity.get('python_version')}"
                ),
            }
        if not str(identity.get("started_at") or "").strip():
            return {"status": "error", "message": f"{expected_service}缺少启动时间"}
        configured_port = identity.get("configured_port")
        if not isinstance(configured_port, int):
            return {"status": "error", "message": f"{expected_service}缺少配置端口"}
        if expected_port is not None and configured_port != expected_port:
            return {
                "status": "error",
                "message": (
                    f"{expected_service}配置端口不一致: "
                    f"expected={expected_port}, actual={configured_port}"
                ),
            }
        return {"status": "success", "message": "运行身份一致"}
    except Exception as exc:
        return {"status": "error", "message": f"校验服务运行身份失败: {exc}"}


def _run_live_service_checks(
    *,
    agent_api_base: str,
    biology_api_base: str,
    agent_root: Path,
    biology_root: Path,
    agent_git: dict[str, Any],
    biology_git: dict[str, Any],
    expected_python_version: str,
    allow_dirty: bool,
    timeout_seconds: float,
    fasta_fixture: Path | None,
    pdb_fixture: Path | None,
    pdb_chain_id: str | None,
) -> dict[str, Any]:
    """查询两个真实服务，并通过Biology HTTP执行FASTA/PDB贯通检查。"""
    try:
        checks: list[dict[str, Any]] = []
        blockers: list[str] = []
        agent_base = _normalize_base_url(agent_api_base)
        biology_base = _normalize_base_url(biology_api_base)

        agent_response = _http_json(
            "GET",
            f"{agent_base}/api/runtime-info",
            timeout_seconds=timeout_seconds,
        )
        biology_response = _http_json(
            "GET",
            f"{biology_base}/api/runtime-info",
            timeout_seconds=timeout_seconds,
        )

        _append_check(
            checks,
            blockers,
            name="agent_runtime_endpoint",
            passed=agent_response.get("status") == "success",
            detail=(
                "Agent运行身份端点可访问"
                if agent_response.get("status") == "success"
                else f"Agent运行身份端点失败: {agent_response.get('message')}"
            ),
        )
        _append_check(
            checks,
            blockers,
            name="biology_runtime_endpoint",
            passed=biology_response.get("status") == "success",
            detail=(
                "Biology运行身份端点可访问"
                if biology_response.get("status") == "success"
                else f"Biology运行身份端点失败: {biology_response.get('message')}"
            ),
        )

        agent_identity = agent_response.get("data") or {}
        biology_identity = biology_response.get("data") or {}
        agent_validation = _validate_runtime_identity(
            agent_identity,
            expected_service="agent",
            expected_root=agent_root,
            expected_commit=str(agent_git.get("commit") or ""),
            expected_python_version=expected_python_version,
            expected_port=urlsplit(agent_base).port,
            allow_dirty=allow_dirty,
        )
        biology_validation = _validate_runtime_identity(
            biology_identity,
            expected_service="biology",
            expected_root=biology_root,
            expected_commit=str(biology_git.get("commit") or ""),
            expected_python_version=expected_python_version,
            expected_port=urlsplit(biology_base).port,
            allow_dirty=allow_dirty,
        )
        _append_check(
            checks,
            blockers,
            name="agent_live_identity",
            passed=agent_validation.get("status") == "success",
            detail=str(agent_validation.get("message")),
        )
        _append_check(
            checks,
            blockers,
            name="biology_live_identity",
            passed=biology_validation.get("status") == "success",
            detail=str(biology_validation.get("message")),
        )

        dependency_base = _normalize_base_url(
            str((agent_identity.get("dependencies") or {}).get("biology_api_base") or "")
        )
        _append_check(
            checks,
            blockers,
            name="agent_biology_endpoint_binding",
            passed=bool(dependency_base) and dependency_base == biology_base,
            detail=(
                "Agent配置的Biology地址与预检目标一致"
                if dependency_base == biology_base and dependency_base
                else (
                    "Agent配置的Biology地址不一致: "
                    f"agent={dependency_base or 'missing'}, expected={biology_base}"
                )
            ),
        )

        identities_match = (
            agent_identity.get("project_root") == biology_identity.get("project_root")
            and agent_identity.get("git_commit") == biology_identity.get("git_commit")
        )
        _append_check(
            checks,
            blockers,
            name="live_service_checkpoint_consistency",
            passed=identities_match,
            detail=(
                "两个运行服务来自同一项目根目录和Git提交"
                if identities_match
                else "两个运行服务的项目根目录或Git提交不一致"
            ),
        )

        if fasta_fixture is None:
            _append_check(
                checks,
                blockers,
                name="fasta_http_roundtrip",
                passed=False,
                detail="数据集中没有可用于正式预检的非拒绝FASTA fixture",
            )
        else:
            fasta_response = _http_json(
                "POST",
                f"{biology_base}/api/parse_protein_fasta",
                payload={"file_path": str(fasta_fixture)},
                timeout_seconds=timeout_seconds,
            )
            fasta_data = fasta_response.get("data") or {}
            fasta_passed = (
                fasta_response.get("status") == "success"
                and fasta_data.get("status") == "success"
            )
            _append_check(
                checks,
                blockers,
                name="fasta_http_roundtrip",
                passed=fasta_passed,
                detail=(
                    f"FASTA HTTP贯通成功: {fasta_fixture.name}"
                    if fasta_passed
                    else f"FASTA HTTP贯通失败: {fasta_response.get('message') or fasta_data}"
                ),
            )

        if pdb_fixture is None:
            _append_check(
                checks,
                blockers,
                name="pdb_http_roundtrip",
                passed=False,
                detail="数据集中没有可用于正式预检的非拒绝PDB fixture",
            )
        else:
            if pdb_chain_id:
                pdb_endpoint = "/api/select_pdb_chain"
                pdb_payload = {
                    "file_path": str(pdb_fixture),
                    "chain_id": str(pdb_chain_id),
                }
            else:
                pdb_endpoint = "/api/extract_pdb_sequences"
                pdb_payload = {"file_path": str(pdb_fixture)}
            pdb_response = _http_json(
                "POST",
                f"{biology_base}{pdb_endpoint}",
                payload=pdb_payload,
                timeout_seconds=timeout_seconds,
            )
            pdb_data = pdb_response.get("data") or {}
            pdb_passed = (
                pdb_response.get("status") == "success"
                and pdb_data.get("status") == "success"
            )
            _append_check(
                checks,
                blockers,
                name="pdb_http_roundtrip",
                passed=pdb_passed,
                detail=(
                    f"PDB HTTP贯通成功: {pdb_fixture.name}"
                    if pdb_passed
                    else f"PDB HTTP贯通失败: {pdb_response.get('message') or pdb_data}"
                ),
            )

        return {
            "status": "error" if blockers else "pass",
            "checks": checks,
            "blockers": blockers,
            "agent_identity": agent_identity,
            "biology_identity": biology_identity,
        }
    except Exception as exc:
        logger.error("真实双服务预检异常: %s", exc, exc_info=True)
        return {
            "status": "error",
            "checks": [],
            "blockers": [f"真实双服务预检异常: {exc}"],
            "agent_identity": {},
            "biology_identity": {},
        }


def run_environment_preflight(
    agent_root: str | Path,
    biology_root: str | Path,
    dataset_path: str | Path,
    *,
    agent_api_base: str | None = None,
    biology_api_base: str | None = None,
    expected_python_version: str | None = None,
    question_schema_path: str | Path | None = None,
    timeout_seconds: float = 60.0,
    require_live_services: bool = True,
    allow_dirty: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """执行正式实验环境预检；默认要求真实双服务和干净工作树。"""
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        config = Config()
        agent_dir = Path(agent_root).expanduser().resolve()
        biology_dir = Path(biology_root).expanduser().resolve()
        dataset_file = Path(dataset_path).expanduser().resolve()
        schema_file = Path(
            question_schema_path or config.EXPERIMENT_QUESTION_SCHEMA_PATH
        ).expanduser().resolve()
        expected_python = str(
            expected_python_version or config.EXPERIMENT_PYTHON_VERSION or ""
        ).strip()
        actual_python = platform.python_version()

        _append_check(
            checks,
            blockers,
            name="agent_root_existence",
            passed=agent_dir.is_dir(),
            detail=(
                str(agent_dir)
                if agent_dir.is_dir()
                else f"Agent根目录不存在: {agent_dir}"
            ),
        )
        _append_check(
            checks,
            blockers,
            name="biology_root_existence",
            passed=biology_dir.is_dir(),
            detail=(
                str(biology_dir)
                if biology_dir.is_dir()
                else f"Biology服务根目录不存在: {biology_dir}"
            ),
        )
        roots_match = agent_dir == biology_dir
        _append_check(
            checks,
            blockers,
            name="root_path_consistency",
            passed=roots_match,
            detail=(
                str(agent_dir)
                if roots_match
                else f"Agent根目录({agent_dir})与Biology根目录({biology_dir})不一致"
            ),
        )

        agent_git = _get_git_info(agent_dir)
        biology_git = _get_git_info(biology_dir)
        for service_name, git_info in (
            ("agent", agent_git),
            ("biology", biology_git),
        ):
            git_ok = git_info.get("status") == "success"
            _append_check(
                checks,
                blockers,
                name=f"{service_name}_git_identity",
                passed=git_ok,
                detail=(
                    f"commit={git_info.get('commit')}, dirty={git_info.get('is_dirty')}"
                    if git_ok
                    else f"{service_name} Git校验失败: {git_info.get('message')}"
                ),
            )
            dirty_ok = git_ok and (
                git_info.get("is_dirty") is False or allow_dirty
            )
            _append_check(
                checks,
                blockers,
                name=f"{service_name}_clean_worktree",
                passed=dirty_ok,
                detail=(
                    f"{service_name}工作树干净"
                    if dirty_ok and git_info.get("is_dirty") is False
                    else (
                        f"{service_name}工作树为dirty；--allow-dirty仅可用于诊断"
                        if dirty_ok
                        else f"{service_name}工作树不干净或状态未知"
                    )
                ),
            )

        commits_match = (
            agent_git.get("status") == "success"
            and biology_git.get("status") == "success"
            and agent_git.get("commit") == biology_git.get("commit")
        )
        _append_check(
            checks,
            blockers,
            name="git_commit_consistency",
            passed=commits_match,
            detail=(
                f"commit={agent_git.get('commit')}"
                if commits_match
                else (
                    "Agent与Biology提交不一致: "
                    f"agent={agent_git.get('commit')}, biology={biology_git.get('commit')}"
                )
            ),
        )

        python_configured = bool(expected_python)
        _append_check(
            checks,
            blockers,
            name="python_version_frozen",
            passed=python_configured,
            detail=(
                f"expected={expected_python}"
                if python_configured
                else "未配置EXPERIMENT_PYTHON_VERSION，正式实验环境未冻结"
            ),
        )
        python_matches = python_configured and _version_matches(
            actual_python,
            expected_python,
        )
        _append_check(
            checks,
            blockers,
            name="local_python_version",
            passed=python_matches,
            detail=(
                f"Python版本匹配: {actual_python}"
                if python_matches
                else f"Python版本不匹配: expected={expected_python or 'missing'}, actual={actual_python}"
            ),
        )

        embedding_validation = validate_local_embedding_model(
            config.BGE_MODEL_PATH
        )
        _append_check(
            checks,
            blockers,
            name="embedding_model_assets",
            passed=embedding_validation.get("status") == "success",
            detail=(
                "本地嵌入模型资产完整: "
                f"path={embedding_validation.get('model_path')}, "
                f"model_type={embedding_validation.get('model_type')}"
                if embedding_validation.get("status") == "success"
                else str(embedding_validation.get("message"))
            ),
        )

        dataset_inside_root = dataset_file.is_relative_to(agent_dir)
        _append_check(
            checks,
            blockers,
            name="dataset_path_safety",
            passed=dataset_inside_root,
            detail=(
                str(dataset_file)
                if dataset_inside_root
                else f"数据集必须位于Agent项目根目录内: {dataset_file}"
            ),
        )
        dataset_sha256 = _calc_sha256(dataset_file)
        _append_check(
            checks,
            blockers,
            name="dataset_hash",
            passed=dataset_sha256 is not None,
            detail=(
                f"path={dataset_file}, sha256={dataset_sha256}"
                if dataset_sha256
                else f"数据集不存在或无法计算SHA-256: {dataset_file}"
            ),
        )
        dataset_validation = _load_and_validate_dataset(dataset_file, schema_file)
        _append_check(
            checks,
            blockers,
            name="dataset_schema_validity",
            passed=dataset_validation.get("status") == "success",
            detail=str(dataset_validation.get("message")),
        )
        items = dataset_validation.get("items") or []

        missing_fixtures: list[str] = []
        unsafe_fixtures: list[str] = []
        checked_fixtures_count = 0
        fasta_fixture: Path | None = None
        pdb_fixture: Path | None = None
        pdb_chain_id: str | None = None
        for item in items:
            input_data = item.get("input") if isinstance(item, dict) else None
            input_data = input_data if isinstance(input_data, dict) else {}
            raw_fixture = input_data.get("fixture")
            if not isinstance(raw_fixture, str) or not raw_fixture.strip():
                continue
            checked_fixtures_count += 1
            resolved = _resolve_fixture(
                raw_fixture,
                dataset_dir=dataset_file.parent,
                project_root=biology_dir,
            )
            if resolved.get("status") != "success":
                message = str(resolved.get("message") or "fixture解析失败")
                if "越界" in message or "绝对路径" in message:
                    unsafe_fixtures.append(message)
                else:
                    missing_fixtures.append(message)
                continue

            fixture_path = resolved.get("path")
            expected_behavior = (item.get("gold") or {}).get("expected_behavior")
            input_type = input_data.get("input_type")
            if expected_behavior != "reject" and isinstance(fixture_path, Path):
                if input_type == "protein_fasta_file" and fasta_fixture is None:
                    fasta_fixture = fixture_path
                if input_type == "pdb_file" and pdb_fixture is None:
                    pdb_fixture = fixture_path
                    chain_value = input_data.get("chain_id")
                    pdb_chain_id = str(chain_value) if chain_value else None

        _append_check(
            checks,
            blockers,
            name="fixture_existence",
            passed=not missing_fixtures,
            detail=(
                f"已检验{checked_fixtures_count}个fixture，无缺失"
                if not missing_fixtures
                else f"缺少{len(missing_fixtures)}个fixture: {missing_fixtures[:3]}"
            ),
        )
        _append_check(
            checks,
            blockers,
            name="fixture_path_safety",
            passed=not unsafe_fixtures,
            detail=(
                "所有fixture均位于项目安全根目录内"
                if not unsafe_fixtures
                else f"发现{len(unsafe_fixtures)}个越界fixture: {unsafe_fixtures[:3]}"
            ),
        )

        live_result: dict[str, Any] = {
            "status": "skipped",
            "checks": [],
            "blockers": [],
            "agent_identity": {},
            "biology_identity": {},
        }
        if require_live_services:
            if not agent_api_base or not biology_api_base:
                _append_check(
                    checks,
                    blockers,
                    name="live_service_configuration",
                    passed=False,
                    detail="正式预检必须提供Agent与Biology服务地址",
                )
            elif agent_git.get("status") == "success" and biology_git.get("status") == "success":
                live_result = _run_live_service_checks(
                    agent_api_base=agent_api_base,
                    biology_api_base=biology_api_base,
                    agent_root=agent_dir,
                    biology_root=biology_dir,
                    agent_git=agent_git,
                    biology_git=biology_git,
                    expected_python_version=expected_python,
                    allow_dirty=allow_dirty,
                    timeout_seconds=timeout_seconds,
                    fasta_fixture=fasta_fixture,
                    pdb_fixture=pdb_fixture,
                    pdb_chain_id=pdb_chain_id,
                )
                checks.extend(live_result.get("checks") or [])
                blockers.extend(live_result.get("blockers") or [])
        else:
            checks.append(
                {
                    "name": "live_service_checks",
                    "status": "skipped",
                    "detail": "静态诊断模式：未核验实际运行服务，不可用于正式实验放行",
                }
            )

        result = {
            "status": "error" if blockers else "pass",
            "mode": "formal_live" if require_live_services else "diagnostic_static",
            "formal_experiment_ready": not blockers and require_live_services and not allow_dirty,
            "checks": checks,
            "blockers": blockers,
            "environment": {
                "python_version": actual_python,
                "expected_python_version": expected_python or None,
                "agent_root": str(agent_dir),
                "biology_root": str(biology_dir),
                "dataset_path": str(dataset_file),
                "question_schema_path": str(schema_file),
                "dataset_sha256": dataset_sha256,
                "embedding_model": {
                    "status": embedding_validation.get("status"),
                    "model_path": embedding_validation.get("model_path"),
                    "model_type": embedding_validation.get("model_type"),
                    "issue_codes": [
                        str(item.get("code"))
                        for item in embedding_validation.get("issues") or []
                        if isinstance(item, dict) and item.get("code")
                    ],
                },
                "agent_git": agent_git,
                "biology_git": biology_git,
                "agent_runtime": live_result.get("agent_identity") or None,
                "biology_runtime": live_result.get("biology_identity") or None,
            },
        }

        if output_path:
            output_file = Path(output_path).expanduser().resolve()
            output_file.parent.mkdir(parents=True, exist_ok=True)
            temporary_file = output_file.with_suffix(f"{output_file.suffix}.tmp")
            temporary_file.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_file.replace(output_file)
            logger.info("预检结果已保存至: %s", output_file)
        return result
    except Exception as exc:
        logger.error("正式实验环境预检异常: %s", exc, exc_info=True)
        return {
            "status": "error",
            "mode": "formal_live" if require_live_services else "diagnostic_static",
            "formal_experiment_ready": False,
            "checks": checks,
            "blockers": blockers + [f"预检系统异常: {exc}"],
            "environment": {
                "python_version": platform.python_version(),
                "agent_root": str(agent_root),
                "biology_root": str(biology_root),
                "dataset_path": str(dataset_path),
            },
        }


def main(argv: list[str] | None = None) -> int:
    """命令行入口；正式模式默认启用真实双服务检查。"""
    try:
        config = Config()
        parser = argparse.ArgumentParser(description="BioAgent正式实验双服务预检工具")
        parser.add_argument("--agent-root", required=True, help="Agent项目根目录")
        parser.add_argument("--biology-root", required=True, help="Biology项目根目录")
        parser.add_argument("--dataset", required=True, help="BioToolEval数据集路径")
        parser.add_argument(
            "--agent-api-base",
            default=config.AGENT_API_BASE,
            help="Agent服务地址，默认读取AGENT_API_BASE",
        )
        parser.add_argument(
            "--biology-api-base",
            default=config.BIO_API_BASE,
            help="Biology服务地址，默认读取BIO_API_BASE",
        )
        parser.add_argument(
            "--expected-python-version",
            default=config.EXPERIMENT_PYTHON_VERSION,
            help="冻结的Python版本，默认读取EXPERIMENT_PYTHON_VERSION",
        )
        parser.add_argument(
            "--question-schema",
            default=config.EXPERIMENT_QUESTION_SCHEMA_PATH,
            help="BioToolEval题目Schema路径",
        )
        parser.add_argument(
            "--timeout",
            type=float,
            default=config.EXPERIMENT_PREFLIGHT_TIMEOUT_SECONDS,
            help="单个HTTP预检请求超时秒数，范围1-60",
        )
        parser.add_argument(
            "--static-only",
            action="store_true",
            help="仅执行静态诊断；该模式不能放行正式实验",
        )
        parser.add_argument(
            "--allow-dirty",
            action="store_true",
            help="允许脏工作树，仅用于诊断；正式实验仍不会标记ready",
        )
        parser.add_argument("--output", default=None, help="预检报告JSON输出路径")
        args = parser.parse_args(argv)

        timeout_seconds = min(60.0, max(1.0, float(args.timeout)))
        result = run_environment_preflight(
            agent_root=args.agent_root,
            biology_root=args.biology_root,
            dataset_path=args.dataset,
            agent_api_base=args.agent_api_base,
            biology_api_base=args.biology_api_base,
            expected_python_version=args.expected_python_version,
            question_schema_path=args.question_schema,
            timeout_seconds=timeout_seconds,
            require_live_services=not args.static_only,
            allow_dirty=args.allow_dirty,
            output_path=args.output,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("formal_experiment_ready") is True else 1
    except Exception as exc:
        logger.error("运行预检CLI失败: %s", exc, exc_info=True)
        print(
            json.dumps(
                {"status": "error", "message": f"运行预检CLI失败: {exc}"},
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
