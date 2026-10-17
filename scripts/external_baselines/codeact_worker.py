"""Process/IPC adapter around unchanged official CodeAct Task and PythonREPL."""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import os
import re
from pathlib import Path
import types

from .codeact_upstream import load_core


ALLOWED_IMPORTS = {"json", "math", "re", "statistics", "collections", "itertools",
                   "functools", "decimal", "fractions", "string", "typing",
                   "hashlib", "operator", "csv", "datetime"}
BLOCKED_CALLS = {"eval", "exec", "compile", "getattr", "setattr", "delattr",
                 "globals", "locals", "vars", "dir", "input", "help", "breakpoint",
                 "get_ipython"}


def check_environment_code(code):
    """Control available resources; never inspect types or biological identity."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        if re.search(r"(?m)^\s*(?:[%!]|[A-Za-z]\w*\s*=\s*[%!])", code):
            raise ValueError("Shell/package-install and notebook magics are unavailable")
        # Syntax diagnostics belong to the official IPython executor. Resource
        # checks must not replace its SyntaxError with vague adapter feedback.
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if getattr(node, "level", 0) or any(n.split(".")[0] not in ALLOWED_IMPORTS for n in names):
                raise ValueError("Use the provided scientific tools; this import is outside the controlled environment")
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in BLOCKED_CALLS):
            raise ValueError("Private runtime names are unavailable")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private runtime attributes are unavailable")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            raise ValueError("Runtime introspection/dynamic execution is unavailable")
    return tree


class WorkerStopped(Exception):
    pass


def worker_main(connection, request, catalog, model, max_turns, work_directory):
    # Do not expose provider credentials to generated code. No .env is loaded here.
    for name in list(os.environ):
        if any(token in name.upper() for token in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
            os.environ.pop(name, None)
    os.chdir(work_directory)
    os.environ["IPYTHONDIR"] = str(Path(work_directory) / ".ipython")

    def send(kind, **fields):
        connection.send({"kind": kind, **fields})

    def rpc(kind, **fields):
        send(kind, **fields)
        response = connection.recv()
        if response.get("stop"):
            raise WorkerStopped(response["stop"])
        return response["value"]

    base, native_globals = load_core()
    allowed_files = {Path(p).resolve() for p in request.get("supplied_files", [])}
    work_root = Path(work_directory).resolve()
    original_open = open

    def guarded_open(file, mode="r", *args, **kwargs):
        path = Path(file).resolve()
        in_work = path.is_relative_to(work_root)
        if not in_work and (path not in allowed_files or any(flag in mode for flag in "wax+")):
            raise PermissionError("Only supplied inputs and this task's temporary files are accessible")
        send("kernel_file_open", file_path=str(path), mode=mode)
        return original_open(path, mode, *args, **kwargs)

    class WindowsREPL(base.PythonREPL):
        def reset(self):
            super().reset()
            # IPython installs its own open() during initialization. Inject the
            # request environment afterwards, as for predefined kernel tools.
            self.shell.user_ns.update(self.user_ns)

        @contextlib.contextmanager
        def time_limit(self, seconds):
            # Windows has no SIGALRM. Parent enforces the cell deadline by
            # terminating this owned process; upstream execution body is intact.
            yield

    tools = {}
    for spec in catalog:
        parameters = []
        for name in spec["parameters"]:
            entry = (spec.get("parameter_schema") or {}).get(name, {})
            parameters.append(inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=entry["default"] if "default" in entry else inspect.Parameter.empty))
        signature = inspect.Signature(parameters)

        def make_function(tool_name, tool_signature):
            def call(*args, **kwargs):
                bound = tool_signature.bind(*args, **kwargs)
                bound.apply_defaults()
                return rpc("kernel_tool_request", tool_name=tool_name, arguments=dict(bound.arguments))
            call.__name__ = tool_name
            call.__signature__ = tool_signature
            return call

        function = make_function(spec["name"], signature)
        description = spec["description"] + "\nReturns the original tool JSON dictionary."
        if spec.get("observed_success_output_fields"):
            description += " Observed top-level return fields: " + ", ".join(spec["observed_success_output_fields"])
        tools[spec["name"]] = base.ToolType(spec["name"], description, function,
                                            spec["name"] + str(signature))

    class ProteinTask(base.Task):
        def print_task(self):
            pass

        def reset(self):
            super().reset()
            self._ns.update({"open": guarded_open,
                             "read_input_file": lambda **kw: rpc("kernel_input_read", arguments=kw)})

        def get_prompt(self, action_mode):
            prompt = super().get_prompt(action_mode)
            return prompt.replace(
                "Please ONLY output the answer (e.g., single number), without any other text.",
                "Provide all requested results in your final answer, or ask for the missing information. "
                "Use only supplied inputs and actual tool results; do not guess biological identifiers.")

        def check_answer(self, answer):
            # Delivery terminates this native loop. Scientific correctness is
            # scored only after execution, with no Gold-derived model feedback.
            return True

        def parse_generation(self, generation):
            parsed = super().parse_generation(generation)
            send("native_action_parse", generation=generation, parsed=parsed)
            return parsed

        def execute_action(self, action, action_mode):
            send("kernel_execution_start", code=action)
            try:
                check_environment_code(action)
            except ValueError as exc:
                observation = "Execution environment error: " + str(exc)
                send("kernel_execution_end", observation=observation, status="environment_error")
                return observation
            if self.repl is None:
                self.repl = WindowsREPL(self._ns)
            observation = super().execute_action(action, action_mode)
            send("kernel_execution_end", observation=observation, status="returned")
            return observation

    environment = ("\n\nEnvironment: read_input_file(file_path=..., start=0, count=8192) "
                   "reads supplied files as raw text; open() also supports the supplied files. "
                   "Ordinary Python variables, indexing, loops and conditional statements persist "
                   "within this request. Scientific analyses use only the listed tools. "
                   "Additional packages, shell commands, network access and runtime introspection "
                   "are unavailable. Allowed standard imports: " + ", ".join(sorted(ALLOWED_IMPORTS)))
    instruction = json.dumps(request, ensure_ascii=False) + environment
    instruction += "\nFunctional tool contracts:\n" + json.dumps(catalog, ensure_ascii=False)
    task = ProteinTask("protein_request", tools, instruction, expected_output=None,
                       is_single_tool_task=False)

    class NativeInvalidRequest(Exception):
        pass

    class NativeAPIError(Exception):
        pass

    native_globals.update({
        "args": types.SimpleNamespace(model=model),
        "openai": types.SimpleNamespace(error=types.SimpleNamespace(
            InvalidRequestError=NativeInvalidRequest, APIError=NativeAPIError)),
        "chat_completion": lambda messages: rpc("kernel_llm_request", messages=messages),
    })
    transcript = io.StringIO()
    try:
        send("kernel_ready", process_id=os.getpid())
        with contextlib.redirect_stdout(transcript):
            result = native_globals["run_task"]("protein_request", task,
                                                  base.ActionMode.CODE_AS_ACTION, max_turns)
        send("native_loop_complete", result=result, native_stdout=transcript.getvalue())
    except BaseException as exc:
        send("kernel_error", exception_type=type(exc).__name__, message=str(exc),
             native_stdout=transcript.getvalue())
    finally:
        task.free_resource()
        connection.close()
