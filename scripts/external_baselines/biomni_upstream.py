"""Load unchanged official A1 graph, retrieval, prompt and REPL implementations."""
from __future__ import annotations

import ast
import glob
import hashlib
import importlib.util
import json
import re
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

SOURCE = Path(__file__).resolve().parents[2] / "third_party/biomni_a1_core"


def upstream_identity():
    identity = json.loads((SOURCE / "UPSTREAM.json").read_text(encoding="utf-8"))
    for row in identity["files"]:
        if hashlib.sha256((SOURCE / row["local_path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Official Biomni source changed: " + row["local_path"])
    return identity


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_core():
    upstream_identity()
    # Isolated worker only: package shells avoid importing the large upstream
    # environment and model SDK initializer. Original runtime imports still work.
    for name in ("biomni", "biomni.tool", "biomni.model"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    support = load_module("biomni.tool.support_tools", "biomni/tool/support_tools.py")
    registry = load_module("biomni.tool.tool_registry", "biomni/tool/tool_registry.py")
    retriever = load_module("biomni.model.retriever", "biomni/model/retriever.py")
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    module = types.ModuleType("_biomni_official_a1_runtime")
    sys.modules[module.__name__] = module
    ns = module.__dict__
    ns.update(glob=glob, re=re, datetime=datetime, Any=Any, Literal=Literal,
              TypedDict=TypedDict, BaseMessage=BaseMessage, AIMessage=AIMessage,
              HumanMessage=HumanMessage, SystemMessage=SystemMessage,
              StateGraph=StateGraph, MemorySaver=MemorySaver, END=END, START=START)
    utils = ast.parse((SOURCE / "biomni/utils.py").read_text(encoding="utf-8"))
    selected = [n for n in utils.body if isinstance(n, ast.FunctionDef)
                and n.name in {"textify_api_dict", "inject_custom_functions_to_repl"}]
    if len(selected) != 2:
        raise ValueError("Missing native Biomni utility")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE / "biomni/utils.py"), "exec", dont_inherit=True), ns)
    tree = ast.parse((SOURCE / "biomni/agent/a1.py").read_text(encoding="utf-8"))
    state = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AgentState")
    a1 = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "A1")
    names = {"_generate_system_prompt", "configure", "_prepare_resources_for_retrieval",
             "go", "update_system_prompt_with_selected_resources",
             "_inject_custom_functions_to_repl", "_clear_execution_plots"}
    a1.body = [n for n in a1.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in a1.body} != names:
        raise ValueError("Missing native A1 graph method")
    exec(compile(ast.Module(body=[state, a1], type_ignores=[]),
                 str(SOURCE / "biomni/agent/a1.py"), "exec", dont_inherit=True), ns)
    return ns["A1"], ns, support, registry.ToolRegistry, retriever.ToolRetriever
