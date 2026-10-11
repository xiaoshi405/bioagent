# Reproducing the archived results

## Offline result package

```bash
python scripts/verify_release_assets.py
python scripts/reproduce_paper_results.py --output outputs/reproduced
```

The lightweight reproduction program reconstructs the framework summary counts and bounds from all 4,500 archived analysis rows, verifies plotted repeat rates, checks token aggregates against archived resource rows, and renders the two result plots with the existing plotting routines and palettes. It checks the already-confirmed numerical results; it does not invoke a model or rerun tasks.

`results/framework/main_metrics.csv` contains the three execution endpoints. `run_analysis.csv` supplies their per-run labels. `resource_runs.csv` supplies observed model-token usage and coverage. `paired_effects.csv` retains the question-level paired estimates and recorded intervals. Plot CSVs preserve their actual means and three repeat points.

There are 4,500 planned external units and 4,492 archived complete results; the eight historical missing units remain identified. Rates use planned denominators and retain missing-outcome bounds. Tokens are based on recorded usage; paired token savings and pooled plotted means are distinct aggregations.

For the plotted observed-token means, a recorded run with no model response contributes zero observed tokens, following the archived aggregation. Its usage-completeness flag remains false; this is not an estimate of the provider's billed or unobserved consumption.

The component plot uses archived per-repeat metrics from the 6,300-run shared-plan study. Natural recovery counts and controlled bridge validation are kept distinct in the benchmark documentation and existing suite.

## Evidence levels

1. **Source and input verification:** `scripts/verify_release_assets.py` verifies release-file SHA256, frozen external code and Test-150 input byte hashes, and vendored official-source hashes. New-clone line endings are protected by `.gitattributes`.
2. **Archived-label reconstruction:** the offline reproduction reconstructs tables and plot values from released per-run/repeat records. The retained scientific-score files preserve their original labels and coverage.
3. **Raw event rescoring:** historical raw events contain the planning and execution observations used to obtain those labels. They are not bundled into the lightweight code repository. Original raw-result/event hashes are retained for provenance. No external archive URL is invented; one must be provided with the eventual archival release before claiming independent raw-trace rescoring is available.
4. **New online execution:** historical runners and adapters are provided, but a new installation needs a new versioned preparation/freeze and its own resources/API authorization. A new execution is not the archived experiment.

The original external freeze includes machine-specific runtime/preparation metadata. It is preserved as evidence rather than edited for portability. The public asset verifier validates the portable source/input hashes without pretending that the historical machine paths match the new installation.

## Outputs and provenance

`RELEASE_MANIFEST.json` maps each exported/generated file to its origin and checksum. Private manuscript-management materials and temporary directories are not part of the export. Local tests, generated figures and application logs go under ignored runtime directories.

Plot PDFs can differ in font fallback or embedded creation metadata across machines; the verified rates, repeat points, palettes, and rendering logic are reproducible. Figures are regenerated into the requested output directory, leaving archived plot assets unchanged.
