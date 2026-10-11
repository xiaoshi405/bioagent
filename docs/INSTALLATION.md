# Installation

Run commands from the repository root. Start with Python 3.13 and a fresh virtual environment. `requirements.txt` pins the lightweight tool/runtime and offline-result dependencies used in local verification. `requirements-agent.txt` adds retrieval, model interfaces and Streamlit. `requirements-baselines.txt` adds official-core adapter dependencies.

The external experiment's recorded package versions are kept separately in `results/environment/frozen_external_packages.json`. That record describes the historical environment; it is not changed to match a new installation.

## Offline runtime

```bash
python -m pip install -r requirements.txt
python scripts/verify_release_assets.py
python examples/offline_identity.py
```

The example executes the existing multi-record FASTA fixture with explicit record selection. No LLM, local embedding model or background service is required. Python's built-in unittest runner is used for the released tests:

```bash
python scripts/run_release_tests.py --group core
```

## Protein tool service

```bash
python -m uvicorn ToolsAgent.ToolsFuns.Biology.main:app --host 127.0.0.1 --port 8000
```

This serves the registered computational tools. The optional agent and web interface are separate processes. Local output and logging directories are created when needed.

## Model planning and retrieval

Install `requirements-agent.txt`. Copy `example.env` to `.env`, set `API_KEY`, `API_BASE` and `MODEL_NAME`, and set `BGE_MODEL_PATH` to your own embedding installation. Historical embedding identity is in the archived external freeze. No model weights or credentials are included in this repository.

Enable `TYPE_PLANNING_ENABLED`, `TYPE_VALIDATION_ENABLED` and `TYPE_REPAIR_ENABLED` for the deterministic-first BioAgent runtime. These are independent experimental switches; changing them creates a different experimental condition.

`python scripts/fix_my_index.py` builds local knowledge-graph storage from the public tool descriptions. Agent-backed requests can send protein inputs and tool returns to the configured model provider and can incur API charges. The offline example and result reconstruction do neither.

## Baseline adapters

Install `requirements-baselines.txt` to run the official-core adapter tests:

```bash
python scripts/run_release_tests.py --group baselines
```

These tests use local transports and scripted test responses. They exercise adapter behavior without paid model calls. They are software regression tests, not replacements for the archived experimental comparison.

Historical online runners retain their original freeze checks. A new execution on another machine needs a new preparation/freeze with its own paths, installed dependencies and resources. Do not rewrite the archived freeze or present a new run as the historical experiment.
