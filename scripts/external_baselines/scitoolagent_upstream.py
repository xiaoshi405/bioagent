"""Load original SciToolAgent algorithms without global SDK/config side effects."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SOURCE = Path(__file__).resolve().parents[2] / "third_party/scitoolagent_core"


def upstream_identity():
    identity = json.loads((SOURCE / "UPSTREAM.json").read_text(encoding="utf-8"))
    for entry in identity["files"]:
        if hashlib.sha256((SOURCE / entry["local_path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Official SciToolAgent source changed: " + entry["local_path"])
    return identity


def definitions(path, namespace, names=None):
    tree = ast.parse((SOURCE / path).read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and (names is None or n.name in names)]
    if names and {n.name for n in selected} != set(names):
        raise ValueError("Official definitions missing in " + path)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE / path), "exec"), namespace)


def load_core():
    upstream_identity()
    spec = importlib.util.spec_from_file_location("_scitoolagent_official_retriever",
                                                  SOURCE / "scripts/custom_kg_retrievers.py")
    retriever = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retriever)
    namespace = {"List": List, "Dict": Dict, "Tuple": Tuple, "Optional": Optional,
                 "Any": Any, "re": re, "ast": ast, "json": json,
                 "logging": logging, "logger": logging.getLogger("scitoolagent_official")}
    definitions("app/llms/planning/prompt_templates.py", namespace)
    definitions("app/llms/planning/tool_plan_executor.py", namespace)
    definitions("app/llms/extraction/extract_tool.py", namespace,
                {"extract_tool_path", "extract_answer_parts", "extract_regex_group"})
    definitions("scripts/utils.py", namespace,
                {"construct_question", "handle_multiple_entries", "create_triplets"})
    # construct_question references os only when adding filenames.
    import os
    namespace["os"] = os
    from llama_index.core import KnowledgeGraphIndex, QueryBundle, load_index_from_storage
    from llama_index.core.graph_stores import SimpleGraphStore
    from llama_index.core.storage.storage_context import StorageContext
    namespace.update(KnowledgeGraphIndex=KnowledgeGraphIndex, QueryBundle=QueryBundle,
                     SimpleGraphStore=SimpleGraphStore, StorageContext=StorageContext)
    definitions("scripts/generate_kg_index.py", namespace, {"build_knowledge_graph"})

    def load_index(directory):
        context = StorageContext.from_defaults(persist_dir=str(directory))
        return context, load_index_from_storage(context)

    namespace["load_index_from_storage_dir"] = load_index
    definitions("scripts/retrieve_tool_info.py", namespace,
                {"execute_tool", "check_toxicity", "run_SciToolAgent"})
    return retriever.KGTableRetriever, namespace
