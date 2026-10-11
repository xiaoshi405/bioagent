"""Load unchanged CodeAct ICML-2024 implementation, without benchmark CLI/data.

Task/parser/executor and run_task/postprocess_fn execute the upstream bodies.
Only selected definitions of main.py are loaded to avoid its import-time CLI,
legacy SDKs and benchmark evaluation. Source hashes are checked before use.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
import traceback
import types
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "third_party/codeact_m3tooleval"


def upstream_identity():
    identity = json.loads((SOURCE / "UPSTREAM.json").read_text(encoding="utf-8"))
    for entry in identity["files"]:
        actual = hashlib.sha256((SOURCE / entry["local_path"]).read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError("Official CodeAct source changed: " + entry["local_path"])
    return identity


def load_core():
    upstream_identity()
    package = "_bioagent_codeact_official_tasks"
    if package + ".base" not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(SOURCE / "tasks")]
        sys.modules[package] = module
        for name in ("repl", "base"):
            spec = importlib.util.spec_from_file_location(package + "." + name,
                                                         SOURCE / "tasks" / (name + ".py"))
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = loaded
            spec.loader.exec_module(loaded)
    base = sys.modules[package + ".base"]
    tree = ast.parse((SOURCE / "main.py").read_text(encoding="utf-8"), filename=str(SOURCE / "main.py"))
    names = {"postprocess_fn", "run_task"}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != names:
        raise ValueError("Missing official CodeAct loop")
    selected = ast.Module(body=definitions, type_ignores=[])
    globals_ = {"re": re, "traceback": traceback, "Task": base.Task,
                "ActionMode": base.ActionMode, "__name__": "_codeact_official_loop"}
    exec(compile(selected, str(SOURCE / "main.py"), "exec"), globals_)
    return base, globals_
