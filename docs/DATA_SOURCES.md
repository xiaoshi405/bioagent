# Biological input sources

The source manifests and task provenance supplied with BioToolEval retain the actual accessions, PDB IDs, URLs, snapshot hashes and real/synthetic designations. Inputs are copied without changing sequences, chain/record identifiers or byte-level freeze records.

- UniProt sequence data retain UniProt attribution and database terms: [UniProt licensing](https://www.uniprot.org/help/license).
- PDB archive coordinates are available under CC0; preserve the entry identity and source authors in citations: [RCSB PDB usage policies](https://www.rcsb.org/pages/policies).
- Synthetic software fixtures retain their recorded synthetic designation; they are not represented as experimentally observed proteins.
- `data/toxin_protein.csv` is inherited from the SciToolAgent source tree. Its entries use UniProt accessions and the upstream data documentation identifies UniProt as the resource. The original extraction date/query were not retained; this release preserves the inherited snapshot and provenance instead of claiming a new download.

The root code license is not a blanket reassignment of database rights. Source terms and the vendored licenses remain separately applicable.
