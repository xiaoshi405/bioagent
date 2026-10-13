# Maintaining and publishing the release

## Check a release

Run from the repository root in the documented Python 3.13 environment:

```bash
python scripts/verify_release_assets.py
python scripts/run_release_tests.py --group core
python examples/offline_identity.py
python examples/offline_api.py
python scripts/reproduce_paper_results.py --output outputs/reproduced
python scripts/build_release_archive.py --output outputs/BioAgent-source.zip
```

The archive builder verifies every manifested file before packaging, includes exactly the manifested files plus `RELEASE_MANIFEST.json`, and writes a SHA256 sidecar. It excludes Git history, local `.env`, runtime logs, installed environments and generated outputs.

## Versioned assets

`RELEASE_MANIFEST.json` records exact bytes for this release. The historical source/data freeze and official-source manifests independently protect research inputs and upstream code. Changing documentation or portable release scripts requires a corresponding release-manifest update; changing a frozen method or dataset requires a separately versioned research configuration. Do not replace historical hashes to hide a scientific change.

The initial release is assembled from the preserved research checkout with its allowlisted exporter. That private preparation workflow is separate from the public repository. Public contributions are reviewed against the release checks before the maintainer prepares the next version and updates its manifest.

## GitHub checks

The supplied workflow installs the lightweight environment and checks assets, core tests, fixture examples and table/figure reconstruction on Windows and Linux. It needs no model-provider secrets. Full agent/baseline dependencies and online experiments are outside this lightweight job.

Local checks recorded for this candidate are Windows checks. The Linux job is configured but must actually run before its result is reported as verified.

## Evidence scope

The bundled labels and metrics support offline result reconstruction. Full historical model/tool events are not included in this source archive. Their hashes are retained; independent event rescoring needs the original event archive. See [reproducibility](REPRODUCIBILITY.md).

The review candidate is uploaded separately to `xiaoshi405/bioagent-release` as a private repository. The existing `xiaoshi405/bioagent` research repository and its development remotes remain unchanged. Public release requires maintainer approval. Publish only this clean candidate and verify its CI results before documenting a public release tag and download URL.
