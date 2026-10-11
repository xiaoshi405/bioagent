# BioToolEval

The final held-out benchmark is `evaluation/biotooleval/test_150_frozen_v5_3.json`; its freeze manifest is `test_150_freeze_manifest_v5_3.json`. It contains 150 tasks: 95 execute, 30 repair-then-execute and 25 reject. The first two groups provide the 125-task execution denominator.

Inputs include amino-acid sequences, single/multi-record FASTA files and PDB structures. Tasks require preserving explicit record or chain identity and satisfying tool dependencies. The release contains input snapshots, source manifests, task/trace schemas, required development inputs and the two controlled-suite versions used by the retained software tests.

The historical manifests and dataset bytes are preserved. The release asset verifier checks the raw byte hashes recorded by the independent-planning external freeze. JSON canonical hashes in older source manifests are a separate check and should not be confused with raw file hashes.

## Execution endpoints

- **Execute-task completion:** required workflow execution is complete, based on executor summaries or successful closed execution traces.
- **Safe rejection:** a reject task stops before any scientific operation is attempted.
- **Expected-behavior outcome:** completion on execute/repair tasks, and safe rejection on reject tasks.

A completed workflow is not automatically a correct final answer. Identity, scientific output and final-answer correctness are tracked separately with determinate-label coverage. An unclassified label remains unclassified. Historical missing framework units remain missing rather than being counted as observed failures.

## Experimental boundaries

Component ablations share the initial plan across conditions. The framework comparison lets each method plan independently under the same tasks, protein registry and model backbones. These are complementary designs; their F1 rates are not interchangeable.

Natural repair and the 30-case controlled repair suite are separate sets. In the paper, the 38 natural recoveries prune redundant invalid steps; registered bridge families are tested separately in the controlled suite. Do not use a controlled bridge case as an individual natural-recovery trace.

The frozen main metrics and scoring procedures are documented in `docs/EXTERNAL_BASELINES_FORMAL_SCORING_V2_ZH.md`, the source-level evaluation scripts, and the released result tables. The public documentation does not relabel execution outcomes as a new scientific-success metric.
