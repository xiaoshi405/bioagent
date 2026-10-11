"""Request-isolated environment for the official Biomni A1 LangGraph."""
from __future__ import annotations

import builtins
import contextlib
import inspect
import io
import json
import os
import re
import sys
import types
from pathlib import Path

from . import codeact_worker as resources
from .biomni_upstream import load_core


class WorkerStopped(BaseException):
    pass


def api_schema(spec):
    required, optional = [], []
    for name, description in spec["parameters"].items():
        schema = spec.get("parameter_schema", {}).get(name, {})
        entry = {"name": name, "type": schema.get("type", description.split("（")[0]),
                 "description": description}
        if "default" in schema:
            entry["default"] = schema["default"]
            optional.append(entry)
        else:
            required.append(entry)
    detail = spec["description"] + "\nReturns the original JSON dictionary; output types: " + ", ".join(spec.get("output_types", []))
    if spec.get("observed_success_output_fields"):
        detail += ". Observed top-level return fields: " + ", ".join(spec["observed_success_output_fields"])
    return {"name": spec["name"], "description": detail,
            "required_parameters": required, "optional_parameters": optional,
            "module": "biomni.tool.protein_tools"}


def worker_main(connection, request, catalog, model, max_turns, work_directory):
    for name in list(os.environ):
        if any(word in name.upper() for word in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
            os.environ.pop(name, None)
    os.chdir(work_directory)

    def send(kind, **fields):
        connection.send({"kind": kind, **fields})

    def rpc(kind, **fields):
        send(kind, **fields)
        response = connection.recv()
        if response.get("stop"):
            raise WorkerStopped(response["stop"])
        return response["value"]

    agent_class, ns, support, Registry, Retriever = load_core()
    from langchain_core.messages import AIMessage
    original_open = open
    original_import = builtins.__import__
    allowed_files = {Path(p).resolve() for p in request["supplied_files"]}
    work_root = Path(work_directory).resolve()

    def guarded_open(file, mode="r", *args, **kwargs):
        path = Path(file).resolve()
        if not path.is_relative_to(work_root) and (path not in allowed_files or any(c in mode for c in "wax+")):
            raise PermissionError("Only supplied inputs and this task's temporary files are accessible")
        send("kernel_file_open", file_path=str(path), mode=mode)
        return original_open(path, mode, *args, **kwargs)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        allowed_virtual = {"protein_tools", "biomni.tool.protein_tools"}
        if level or (name not in allowed_virtual and name.split(".")[0] not in resources.ALLOWED_IMPORTS):
            raise ImportError("Import is outside the shared protein-tool environment")
        # Only a named registered tool module is allowed under biomni.
        if name.startswith("biomni") and name != "biomni.tool.protein_tools":
            raise ImportError("Only biomni.tool.protein_tools is available to generated code")
        return original_import(name, globals, locals, fromlist, level)

    functions = {}
    for spec in catalog:
        parameters = []
        for name in spec["parameters"]:
            entry = spec.get("parameter_schema", {}).get(name, {})
            parameters.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              default=entry["default"] if "default" in entry else inspect.Parameter.empty))
        signature = inspect.Signature(parameters)

        def make_function(name, signature):
            def call(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                return rpc("kernel_tool_request", tool_name=name, arguments=dict(bound.arguments))
            call.__name__, call.__signature__ = name, signature
            return call

        functions[spec["name"]] = make_function(spec["name"], signature)
    tools_module = types.ModuleType("biomni.tool.protein_tools")
    tools_module.__dict__.update(functions)
    tools_module.__all__ = list(functions)
    sys.modules[tools_module.__name__] = tools_module
    sys.modules["protein_tools"] = tools_module
    sys.modules["biomni"].tool = sys.modules["biomni.tool"]
    sys.modules["biomni.tool"].protein_tools = tools_module
    safe_builtins = dict(vars(builtins))
    safe_builtins.update(open=guarded_open, __import__=guarded_import)
    support._persistent_namespace.update(__builtins__=safe_builtins, open=guarded_open,
        read_input_file=lambda **kw: rpc("kernel_input_read", arguments=kw))

    class OpenAICompatibleModelHook:
        model_name = model
        phase = "retrieval"

        def invoke(self, messages):
            mapped = [{"role": {"human": "user", "ai": "assistant", "system": "system"}[m.type],
                       "content": m.content} for m in messages]
            text = rpc("kernel_llm_request", messages=mapped, phase=self.phase,
                       stop_sequences=["</execute>", "</solution>"])
            return AIMessage(content=text)

    class LoggedRetriever(Retriever):
        def prompt_based_retrieval(self, query, resources, llm=None):
            result = super().prompt_based_retrieval(query, resources, llm=llm)
            send("native_resource_retrieval", query=query, resources=resources, selected=result)
            if llm is not None:
                llm.phase = "execution"
            return result

    native_repl = support.run_python_repl
    def execute_with_watchdog(func, args=None, kwargs=None, timeout=600):
        code = args[0] if args else ""
        send("kernel_execution_start", code=code, native_executor=getattr(func, "__name__", "unknown"))
        try:
            if func is not native_repl:
                observation = "Execution environment error: R/Bash are unavailable; use the provided Python tools."
            else:
                before = set(resources.ALLOWED_IMPORTS)
                resources.ALLOWED_IMPORTS.update({"biomni", "protein_tools"})
                try:
                    resources.check_environment_code(code)
                except ValueError as exc:
                    observation = "Execution environment error: " + str(exc)
                else:
                    observation = func(*(args or []), **(kwargs or {}))
                finally:
                    resources.ALLOWED_IMPORTS.clear()
                    resources.ALLOWED_IMPORTS.update(before)
            send("kernel_execution_end", observation=observation, status="returned")
            return observation
        except BaseException:
            send("kernel_execution_end", observation=None, status="interrupted")
            raise

    def pretty_print(message):
        send("native_graph_message", role=message.type, content=message.content)
        return message.content

    ns.update(run_python_repl=native_repl, run_with_timeout=execute_with_watchdog,
              run_r_code=lambda *a: None, run_bash_script=lambda *a: None,
              pretty_print=pretty_print)
    agent = agent_class.__new__(agent_class)
    agent.path = str(work_root)
    (work_root / "data_lake").mkdir()
    agent.module2api = {"biomni.tool.protein_tools": [api_schema(spec) for spec in catalog]}
    agent.tool_registry = Registry(agent.module2api)
    agent.retriever = LoggedRetriever()
    agent.use_tool_retriever = True
    agent.llm = OpenAICompatibleModelHook()
    agent.timeout_seconds = 120
    agent.data_lake_dict = {}
    agent.library_content_dict = {name: "Python standard-library module" for name in sorted(resources.ALLOWED_IMPORTS)}
    agent._custom_functions = functions
    agent.configure(self_critic=False, test_time_scale_round=0)
    environment = ("\nShared environment: only the listed scientific tools and standard-library Python are available. "
                   "Import tools from biomni.tool.protein_tools. Persistent variables, indexing and loops are supported. "
                   "open() and read_input_file(file_path=..., start=0, count=8192) can read supplied files. "
                   "R, Bash, package installation, additional scientific libraries and network access are unavailable. "
                   "The data_lake directory is empty; supplied input paths are given separately.")
    prompt = json.dumps(request, ensure_ascii=False) + environment
    transcript = io.StringIO()
    send("kernel_ready", protocol="official_biomni_a1_langgraph", registered_tools=len(functions), tool_retriever=True)
    try:
        with contextlib.redirect_stdout(transcript):
            result = agent.go(prompt)
        last = result[1]
        solution = re.search(r"<solution>(.*?)</solution>", last, re.DOTALL | re.IGNORECASE)
        send("native_biomni_complete", final_message=last, final_solution=solution.group(1).strip() if solution else None,
             next_step=agent._conversation_state.get("next_step"), console_output=transcript.getvalue())
    except WorkerStopped:
        pass
    except BaseException as exc:
        send("kernel_error", exception_type=type(exc).__name__, message=str(exc), console_output=transcript.getvalue())
    finally:
        connection.close()
