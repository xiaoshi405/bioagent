# BioAgent v0.1.0

BioAgent manages scientific-object identity and provenance across protein-analysis tool workflows using typed artifact binding, complete-plan validation and deterministic recovery.

## Included

- The execution runtime and registry of 20 protein-analysis tools.
- BioToolEval Test-150, controlled repair fixtures, schemas and source manifests.
- ReAct, CodeAct, SciToolAgent, Biomni-A1 and BioAgent F1 adapters with pinned official source licenses.
- Archived result labels, recorded model usage, paired-effect tables and source data for the two result plots.
- English and Chinese READMEs, offline examples, environment diagnostics and a three-service local demo launcher.
- Offline regression tests, Windows/Linux CI, verified source packaging and frozen-asset maintenance guards.

## Run

Follow [installation](INSTALLATION.md). The lightweight examples and table/plot reconstruction need no model key or embedding download. Full model-backed planning requires a user-supplied API configuration and a local BGE model.

The result package preserves the confirmed experiments and historical missing records. It reconstructs archived summaries and plots without a new experiment. Full historical model/tool events are not in the source ZIP; no independent raw-event-rescoring archive is claimed. See [evidence levels](REPRODUCIBILITY.md).

The source ZIP is accompanied by a SHA256 sidecar. Upstream MIT and Apache-2.0 notices and biological database attribution remain applicable. This version does not assert journal acceptance or publication.
