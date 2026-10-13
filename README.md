# BioAgent: Type-Safe Execution for Protein Tool-Using Agents

BioAgent controls scientific object identity and provenance at runtime. A request-scoped artifact pool binds tool inputs to their source objects; complete-plan validation checks the workflow before execution; deterministic recovery inserts uniquely authorized bridges or prunes redundant invalid steps. Model replanning is an optional follow-up to deterministic runtime control.

This repository contains the protein-tool execution layer, the frozen BioToolEval benchmark, adapters for ReAct, CodeAct, SciToolAgent and Biomni-A1, and an offline evidence package for the accompanying manuscript.

The review candidate is hosted at [xiaoshi405/bioagent-release](https://github.com/xiaoshi405/bioagent-release). It remains private until the maintainer approves public release.

## Start offline

The lightweight path runs without a model key, embedding download, or paid API call. Python 3.13 was used for the local release checks; the historical experiment environment is recorded in `results/environment/frozen_external_packages.json`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_release_assets.py
python examples/offline_identity.py
python examples/offline_api.py
python scripts/reproduce_paper_results.py --output outputs/reproduced
```

The identity example uses the supplied multi-record FASTA fixture and real selection/GRAVY tools. Result reproduction reads archived per-run labels and recorded token counts. It performs no model calls and generates no new experimental outcomes.

## Runtime and optional agent interface

The registry contains 20 protein-analysis tools covering sequences, FASTA records/collections, PDB chains, and alignment objects. The tool API can run independently:

```bash
python -m uvicorn ToolsAgent.ToolsFuns.Biology.main:app --host 127.0.0.1 --port 8000
```

For model-backed planning, install `requirements-agent.txt`, copy `example.env` to `.env`, supply your own model endpoint/key, and configure local embedding weights. Then build the index and start the optional interfaces:

```bash
python -m scripts.fix_my_index
python scripts/start_agent.py --host 127.0.0.1 --port 50005
streamlit run web_demo.py
```

The public configuration enables binding, validation and deterministic repair. API keys and runtime outputs remain local. See [installation](docs/INSTALLATION.md) for the separate core, agent and baseline environments.

## Benchmark and paper evidence

BioToolEval provides 150 frozen protein-analysis tasks, including 125 execution/repair tasks and 25 rejection tasks, with the input fixtures, gold contracts, source records, schemas and freeze manifests. The evaluated behaviors are explained in [benchmark documentation](docs/BENCHMARK.md).

| Study | Design | Released evidence |
|---|---|---|
| Component study | Shared initial plans; seven conditions; two models; three repeats; 6,300 runs | Archived repeat metrics and source data for the result plot |
| Framework comparison | Independent planning; five methods; two models; three repeats; 4,500 planned units | All 4,500 per-run analysis rows, recorded token counts, summary and paired-effect tables; historical missing units remain marked |
| Controlled repair | 30 cases in six registered families | Frozen suite, runner, trace schemas and offline regression tests |

Natural recovery and the controlled bridge suite are separate evidence sources. The main execution endpoints are expected-behavior outcome, execute-task completion and safe rejection. Object/output correctness and answer correctness have their own labels and coverage.

See [reproduction](docs/REPRODUCIBILITY.md) for the evidence levels, frozen code/input verification, offline table/plot reconstruction and instructions for a separately versioned new run.

## Main modules

```text
app/                          configuration and model/tool interfaces
ToolsAgent/ToolsFuns/Biology/  protein tool API and registry
scripts/                      artifact binding, validation, execution, recovery and evaluation
scripts/external_baselines/   framework adapters and budget/resource controls
evaluation/biotooleval/        frozen tasks, schemas, sources and fixtures
third_party/                  pinned official baseline core files and licenses
tests/                        offline runtime, tool and adapter regression tests
examples/                     runnable fixture-based example
results/                      archived labels, metrics, figure data and environment records
```

## Source, license and citation

BioAgent derives from [SciToolAgent](https://github.com/HICAI-ZJU/SciToolAgent). The inherited retrieval/planning architecture and BioAgent runtime additions are described in [UPSTREAM.md](UPSTREAM.md). Official CodeAct, SciToolAgent and Biomni core files are pinned and verified independently; see [baseline adaptation](docs/BASELINES.md).

The root MIT license retains the original copyright. Each vendored component retains its own license, including Biomni's Apache-2.0 license. Biological source data retain their recorded database terms; see [data sources](docs/DATA_SOURCES.md).

Use `CITATION.cff` to cite the software and its upstream work. Manuscript metadata are not presented as an accepted or published journal article. See [release checks](docs/RELEASE.md) and [changelog](CHANGELOG.md) for the packaged version and maintenance workflow.
