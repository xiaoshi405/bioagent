# -*- coding: utf-8 -*-
"""Tests for embedding runtime."""

import json
import tempfile
import unittest
from pathlib import Path

from app.core.embedding_runtime import validate_local_embedding_model


class EmbeddingRuntimeTests(unittest.TestCase):

    def _model_dir(self, root: Path) -> Path:
        model_dir = root / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"model_type": "bert"}),
            encoding="utf-8",
        )
        (model_dir / "model.safetensors").write_bytes(b"test")
        (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        return model_dir

    def test_complete_assets_pass_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = validate_local_embedding_model(
                self._model_dir(Path(temp_dir))
            )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["model_type"], "bert")

    def test_missing_config_has_exact_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = validate_local_embedding_model(temp_dir)
        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["issues"][0]["code"],
            "EMBEDDING_MODEL_CONFIG_MISSING",
        )

    def test_invalid_config_and_missing_weights_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "config.json").write_text("{", encoding="utf-8")
            invalid = validate_local_embedding_model(model_dir)
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "bert"}), encoding="utf-8"
            )
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            no_weights = validate_local_embedding_model(model_dir)
        self.assertEqual(
            invalid["issues"][0]["code"],
            "EMBEDDING_MODEL_CONFIG_INVALID",
        )
        self.assertEqual(
            no_weights["issues"][0]["code"],
            "EMBEDDING_MODEL_WEIGHTS_MISSING",
        )
