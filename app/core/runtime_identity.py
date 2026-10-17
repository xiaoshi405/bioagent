# -*- coding: utf-8 -*-
"""Capture service identity at startup."""

from __future__ import annotations

import copy
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def collect_git_identity(project_root: str | Path) -> dict[str, Any]:
    try:
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir():
            return {
                "status": "error",
                "commit": None,
                "is_dirty": None,
                "message": f"项目根目录不存在: {root}",
            }

        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if commit_result.returncode != 0:
            return {
                "status": "error",
                "commit": None,
                "is_dirty": None,
                "message": f"读取Git提交失败: {commit_result.stderr.strip()}",
            }

        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if status_result.returncode != 0:
            return {
                "status": "error",
                "commit": commit_result.stdout.strip() or None,
                "is_dirty": None,
                "message": f"读取Git工作树状态失败: {status_result.stderr.strip()}",
            }

        return {
            "status": "success",
            "commit": commit_result.stdout.strip(),
            "is_dirty": bool(status_result.stdout.strip()),
            "message": None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "commit": None,
            "is_dirty": None,
            "message": f"采集Git身份失败: {exc}",
        }


def build_runtime_identity(
    *,
    service_name: str,
    project_root: str | Path,
    configured_port: int,
    started_at: str,
    dependencies: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        root = Path(project_root).expanduser().resolve()
        git_identity = collect_git_identity(root)
        status = "success" if git_identity.get("status") == "success" else "error"
        return {
            "status": status,
            "service": str(service_name or "unknown"),
            "project_root": str(root),
            "cwd": str(Path.cwd().resolve()),
            "git_commit": git_identity.get("commit"),
            "git_dirty": git_identity.get("is_dirty"),
            "python_version": platform.python_version(),
            "configured_port": int(configured_port),
            "started_at": str(started_at or ""),
            "process_id": int(os.getpid()),
            "dependencies": dict(dependencies or {}),
            "message": git_identity.get("message"),
        }
    except Exception as exc:
        return {
            "status": "error",
            "service": str(service_name or "unknown"),
            "project_root": str(project_root),
            "cwd": None,
            "git_commit": None,
            "git_dirty": None,
            "python_version": platform.python_version(),
            "configured_port": None,
            "started_at": str(started_at or ""),
            "process_id": None,
            "dependencies": {},
            "message": f"构造服务运行身份失败: {exc}",
        }


def capture_runtime_identity(
    *,
    service_name: str,
    project_root: str | Path,
    configured_port: int,
    started_at: str,
    dependencies: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        return build_runtime_identity(
            service_name=service_name,
            project_root=project_root,
            configured_port=configured_port,
            started_at=started_at,
            dependencies=dependencies,
        )
    except Exception as exc:
        return {
            "status": "error",
            "service": str(service_name or "unknown"),
            "project_root": str(project_root),
            "message": f"捕获服务启动身份失败: {exc}",
        }


def read_runtime_identity_snapshot(snapshot: Any) -> dict[str, Any]:
    try:
        if not isinstance(snapshot, dict):
            return {
                "status": "error",
                "message": "服务启动身份快照不是对象",
            }
        return copy.deepcopy(snapshot)
    except Exception as exc:
        return {
            "status": "error",
            "message": f"读取服务启动身份快照失败: {exc}",
        }
