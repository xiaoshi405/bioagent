# Framework adaptation

ReAct uses a local implementation of the thought/action/observation protocol. CodeAct, SciToolAgent and Biomni-A1 use pinned official core sources. Native planning/execution routines are retained and adapted to the same registered protein-tool environment. BioAgent F1 uses the existing BioAgent runtime with deterministic-first recovery and bounded model replanning.

| Method | Official source / implementation | Retained source license |
|---|---|---|
| ReAct | Protocol implementation in `scripts/external_baselines/react.py` | Root project license |
| CodeAct | `xingyaoww/code-act`, commit `d607f56c9cfe9e8632ebaf65dcaf2b4b7fe1c6f8` | MIT |
| SciToolAgent | `HICAI-ZJU/SciToolAgent`, commit `ac1cf19fcef84e69f149db6da424afe4e9b2f11f` | MIT |
| Biomni-A1 | `snap-stanford/Biomni`, v0.0.8, commit `d15b3c9277b8945f3bfa6af8faf20bae92c09628` | Apache-2.0 |

Each `third_party/*/UPSTREAM.json` enumerates exact official file hashes. The corresponding `README_BIOAGENT.md` specifies the adapter boundary, native control flow and environment substitutions. Adapters verify source bytes before loading. Baseline sources must not be silently rewritten to add BioAgent artifact binding or repair.

Protein tools and model backbones are shared. Native framework interaction patterns and F1's native controls remain different; the experiment is not described as a full original-domain leaderboard or an equal-token-budget comparison. Code-execution resource hooks are documented in the source and are not an operating-system sandbox.
