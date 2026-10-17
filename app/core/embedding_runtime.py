# -*- coding: utf-8 -*-
"""Validate and load local embedding assets."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from app.core.logger import get_logger


logger = get_logger("EmbeddingRuntime", "embedding_runtime.log")
_LOAD_LOCK = threading.Lock()
_LOADED_MODEL_PATH: str | None = None


def _error_result(message: str, code: str, **details: Any) -> dict[str, Any]:
    issue = {"code": code, "message": message}
    issue.update(details)
    return {
        "status": "error",
        "valid": False,
        "message": message,
        "issues": [issue],
    }


def validate_local_embedding_model(model_path: Any) -> dict[str, Any]:
    """Check the required files in a local Hugging Face embedding snapshot."""
    try:
        raw_path = str(model_path or "").strip()
        if not raw_path:
            return _error_result(
                "未配置本地嵌入模型路径",
                "EMBEDDING_MODEL_PATH_MISSING",
                suggestion="请通过BGE_MODEL_PATH配置已下载的本地模型目录。",
            )
        resolved = Path(raw_path).expanduser().resolve()
        if not resolved.is_dir():
            return _error_result(
                f"本地嵌入模型目录不存在: {resolved}",
                "EMBEDDING_MODEL_DIRECTORY_MISSING",
                model_path=str(resolved),
                suggestion="请下载并校验完整模型资产，或修正BGE_MODEL_PATH。",
            )

        config_path = resolved / "config.json"
        if not config_path.is_file():
            return _error_result(
                f"本地嵌入模型缺少config.json: {resolved}",
                "EMBEDDING_MODEL_CONFIG_MISSING",
                model_path=str(resolved),
                suggestion="请重新下载完整模型快照，不能只复制权重文件。",
            )
        try:
            model_config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return _error_result(
                "本地嵌入模型config.json无法解析",
                "EMBEDDING_MODEL_CONFIG_INVALID",
                model_path=str(resolved),
                exception_type=type(exc).__name__,
                suggestion="请重新下载或修复config.json后再运行实验。",
            )
        if not isinstance(model_config, dict) or not str(
            model_config.get("model_type") or ""
        ).strip():
            return _error_result(
                "本地嵌入模型config.json缺少model_type",
                "EMBEDDING_MODEL_TYPE_MISSING",
                model_path=str(resolved),
                suggestion="请使用完整、兼容Transformers的模型快照。",
            )

        weight_candidates = (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
        if not any((resolved / name).is_file() for name in weight_candidates):
            return _error_result(
                "本地嵌入模型缺少可识别的权重文件",
                "EMBEDDING_MODEL_WEIGHTS_MISSING",
                model_path=str(resolved),
                suggestion="请重新下载完整模型权重后再运行实验。",
            )
        if not any(
            (resolved / name).is_file()
            for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt")
        ):
            return _error_result(
                "本地嵌入模型缺少tokenizer资产",
                "EMBEDDING_MODEL_TOKENIZER_MISSING",
                model_path=str(resolved),
                suggestion="请重新下载完整tokenizer资产后再运行实验。",
            )
        return {
            "status": "success",
            "valid": True,
            "model_path": str(resolved),
            "model_type": str(model_config["model_type"]),
            "issues": [],
        }
    except Exception as exc:
        logger.error("校验本地嵌入模型失败: %s", exc, exc_info=True)
        return _error_result(
            "校验本地嵌入模型时发生异常",
            "EMBEDDING_MODEL_VALIDATION_EXCEPTION",
            exception_type=type(exc).__name__,
        )


def configure_llama_index_embedding(
    model_path: Any,
    *,
    cache_dir: Any = None,
) -> dict[str, Any]:
    """Load local BGE assets or return a structured error."""
    global _LOADED_MODEL_PATH
    validation = validate_local_embedding_model(model_path)
    if validation.get("status") != "success":
        return validation
    resolved = str(validation["model_path"])
    try:
        normalized_cache = str(cache_dir or "").strip()
        cache_path = (
            Path(normalized_cache).expanduser().resolve()
            if normalized_cache
            else Path(resolved) / ".runtime_cache"
        )
        cache_path.mkdir(parents=True, exist_ok=True)
        if not cache_path.is_dir():
            return _error_result(
                "嵌入模型缓存路径不是目录",
                "EMBEDDING_CACHE_PATH_INVALID",
                cache_path=str(cache_path),
            )
        with _LOAD_LOCK:
            if _LOADED_MODEL_PATH == resolved:
                return {**validation, "already_loaded": True}
            from llama_index.core import Settings
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            Settings.embed_model = HuggingFaceEmbedding(
                model_name=resolved,
                cache_folder=str(cache_path),
            )
            _LOADED_MODEL_PATH = resolved
        logger.info("本地嵌入模型延迟加载完成 | path=%s", resolved)
        return {**validation, "already_loaded": False}
    except Exception as exc:
        logger.error("加载本地嵌入模型失败: %s", exc, exc_info=True)
        return _error_result(
            "本地嵌入模型加载失败",
            "EMBEDDING_MODEL_LOAD_FAILED",
            model_path=resolved,
            cache_path=(str(cache_path) if "cache_path" in locals() else None),
            exception_type=type(exc).__name__,
            suggestion="请核对模型格式、依赖版本和本机可用内存。",
        )
