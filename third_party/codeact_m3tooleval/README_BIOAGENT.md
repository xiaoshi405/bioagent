# Unmodified upstream CodeAct sources

Repository: https://github.com/xingyaoww/code-act

Commit: `d607f56c9cfe9e8632ebaf65dcaf2b4b7fe1c6f8`.

The four files listed in `UPSTREAM.json` are exact Git blob bytes from upstream,
including its MIT license. Do not edit these core sources for local adaptation.

`scripts/external_baselines/codeact_upstream.py` verifies SHA256 and loads the
official `run_task` and `postprocess_fn` definitions without invoking the CLI or
loading upstream benchmark tasks. The official Task and PythonREPL classes are
also loaded directly.

Process, model, protein-tool and Windows compatibility adapters live separately
in `scripts/external_baselines/`. Their scope and differences are documented in
[baseline documentation](../../docs/BASELINES.md).
