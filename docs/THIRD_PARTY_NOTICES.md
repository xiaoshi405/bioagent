# Source and data notices

## Software distributed in this repository

| Component | Pinned provenance | License file |
|---|---|---|
| BioAgent's inherited SciToolAgent implementation | [Source history and adaptation](../UPSTREAM.md) | [Root MIT license](../LICENSE), retaining the 2024 Jing Yu notice |
| CodeAct official core | `third_party/codeact_m3tooleval/UPSTREAM.json` | [MIT](../third_party/codeact_m3tooleval/LICENSE) |
| SciToolAgent comparison core | `third_party/scitoolagent_core/UPSTREAM.json` | [MIT](../third_party/scitoolagent_core/LICENSE) |
| Biomni-A1 official core | `third_party/biomni_a1_core/UPSTREAM.json` | [Apache-2.0](../third_party/biomni_a1_core/LICENSE) |

Each vendored `UPSTREAM.json` records exact revisions and hashes. The corresponding `README_BIOAGENT.md` documents adaptations. Root licensing does not replace third-party licenses. Installed dependencies are obtained through their distributions and retain their own license terms; their binaries and model weights are not bundled in this source package.

## Biological inputs and generated evidence

UniProt-derived sequences and the inherited toxicity CSV retain UniProt attribution and source terms. PDB coordinate snapshots retain entry identifiers and their source records. Real source records and synthetic software fixtures remain distinguishable through the supplied benchmark manifests. See [data provenance and terms](DATA_SOURCES.md).

Archived labels, resource metrics and plot source data are identified in [reproducibility](REPRODUCIBILITY.md). The software source archive includes processed evidence and input fixtures; full historical model/tool-event payloads are a separate evidence layer and are not bundled.

The default API credential is blank. Downloaded BGE snapshots, provider keys, uploaded user inputs and local indexes are runtime assets supplied or created by the user.
