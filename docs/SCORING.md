# Scoring

## Execution outcomes

The component study and framework comparison report three execution endpoints:

- **Execute-task completion:** the executor reports completion, or the execution trace contains a closed successful workflow.
- **Safe rejection:** an incomplete reject task has zero attempted scientific operations.
- **Expected-behavior outcome:** completion on execution/repair tasks and safe rejection on rejection tasks.

The framework comparison has 450 planned units per model/method: 375 execution/repair units and 75 rejection units. Missing units retain unknown labels. Rates use the planned denominators; unresolved labels give lower and upper identification bounds.

Completion describes the executed workflow. Object identity, output correctness, complete request coverage and final-answer correctness have separate labels. A model's completion claim alone is not execution evidence.

Direct Python scientific computation counts as an operation. File inspection, printing and calls blocked before execution do not. A successful later invocation can resolve an earlier tool failure; argument changes remain available for identity scoring. Unresolved execution evidence enters the review queue.

The zero-operation definition also includes budget or infrastructure stops with no scientific operation. Terminal reasons are retained separately. Safe rejection and correctness of the rejection explanation are distinct labels.

## Scientific correctness

The original frozen scientific-task endpoint combines authorized object identity, requested execution, output agreement and supported final-answer correctness. Its original labels and coverage remain in the released scientific-score tables. It is distinct from the derived execution endpoints above.

Scientific scoring permits equivalent computations supported by actual execution evidence. Gold tool paths and BioAgent artifact IDs are not required. Ambiguous identity or unavailable evidence retains a null label. Automatic checks and selective answer reviews retain their recorded reviewer provenance and label coverage.

## Comparisons and resources

Paired comparisons use each question's mean across three repeats. Complete-pair 95% percentile bootstrap intervals use 10,000 resamples and seed 20261001. Complete-pair coverage and full-set effect bounds are reported separately.

Methods share the benchmark, protein tools and model backbones and retain their native interaction budgets. Recorded token usage and usage coverage are reported separately from provider billing.

## Code and records

- `scripts/analyze_external_operational.py`: execution endpoints and paired analysis from archived raw events.
- `scripts/external_baselines/scoring_v2.py`: scientific correctness checks.
- `scripts/reproduce_paper_results.py`: reconstruction from the released labels and resource records.
- `results/framework/`: archived per-run labels, summaries and paired estimates.

Raw model/tool events are not bundled. See [reproducibility](REPRODUCIBILITY.md) for the available evidence and original frozen-source snapshot.
