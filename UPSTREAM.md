# Source provenance

BioAgent derives from the official [HICAI-ZJU/SciToolAgent](https://github.com/HICAI-ZJU/SciToolAgent) implementation. The original import's exact upstream revision was not preserved. The recorded local import was `727abad63c82d036ca42cd6d8fc2ef5aafa811b0`; reference release V1.0.0 is `0bca493534ed2c87cf940a0bb6a20cbcd7c0ffae`. These are provenance records, not a claim that the original import used the reference release.

Inherited components include knowledge-graph tool retrieval, the Planner–Executor–Summarizer pattern, tool service scaffolding, selected prompts/utilities and the local toxicity-data snapshot. The root MIT copyright is retained.

BioAgent additions provide request-scoped artifact binding, scientific type/identity/provenance checks, complete-plan validation, deterministic authorized recovery and structured execution evaluation. The public source paths remain compatible with the research implementation.

The independently evaluated official CodeAct, SciToolAgent and Biomni core files are recorded separately in `third_party/*/UPSTREAM.json`; their upstream revisions, exact hashes and individual licenses are retained. Adapter boundaries are documented in `docs/BASELINES.md` and each vendored README.

`RELEASE_MANIFEST.json` records the research snapshot revision, per-file source locations, original hashes and any generated documentation or result-path normalization. Exporting into a new Git history does not erase upstream attribution or replace historical method/input freezes.
