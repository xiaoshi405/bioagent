# Official Biomni A1 core, controlled protein-tool adaptation

Upstream: https://github.com/snap-stanford/Biomni

Release: v0.0.8, commit d15b3c9277b8945f3bfa6af8faf20bae92c09628.

UPSTREAM.json lists exact original Git bytes and SHA256. Apache-2.0 LICENSE is
retained. Original files are not modified. Adapter code is under
scripts/external_baselines/biomni*.py.

The original A1 configure/go, system prompt, resource selection/update methods,
ToolRetriever, ToolRegistry and persistent Python REPL run the workflow.
Selected methods are compiled from their original AST bodies, including the
original nested LangGraph nodes/routing. This skips only the general environment
initializer, large data lake/benchmark downloads and provider SDK configuration.

The common 20 registered protein tools replace the original heterogeneous tools;
only standard-library Python and supplied inputs are available. Each request has
an isolated process/namespace and a temporary empty data_lake. Scientific tool
calls return exact raw JSON dictionaries as ordinary Python objects. No BioAgent
binding, validation, bridge insertion or guessed identifiers are provided.

Native checklist prompts, XML parser and closing-tag completion, 10K-character
observation truncation, LLM resource retrieval, graph routing and solution
termination are preserved. Self-critic is disabled (official configure default).
The original custom-provider stop sequences </execute> and </solution> are sent
to the designated Qwen/DeepSeek backbone, including the retrieval call.

The provider and execution timeouts/budgets are enforced externally. Windows
uses the parent process watchdog rather than the upstream timeout thread. Full
tool results and code observations are logged before native context truncation.
Resource controls restrict imports, files and extra tools; they do not resolve
scientific identity/provenance and are not an OS security sandbox.

This adapts the published A1 architecture to shared models/tools. It does not
evaluate Biomni-R0 weights or the complete original biomedical software/data lake.
