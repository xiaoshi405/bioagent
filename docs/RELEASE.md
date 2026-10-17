# Maintaining and publishing a release

## Verify a checkout or source archive

Use Python 3.13 and the dependency group documented in [installation](INSTALLATION.md):

```bash
python scripts/verify_release_assets.py
python scripts/check_environment.py --mode core
python scripts/run_release_tests.py --group core
python examples/offline_identity.py
python examples/offline_api.py
python scripts/reproduce_paper_results.py --output outputs/reproduced
```

Core checks run on Windows and Linux in GitHub Actions. Full agent/baseline tests require `requirements-baselines.txt`; they use local transports and scripted responses, not a model provider. For those checks use `python scripts/run_release_tests.py --group all`.

## Maintain the file manifest

`RELEASE_MANIFEST.json` covers every tracked release file. The historical source/input freeze and the official-source manifests independently protect research assets. To update documentation, release wrappers or tests:

```bash
git add <intended-files>
python scripts/update_release_manifest.py
python scripts/update_release_manifest.py --write
git add RELEASE_MANIFEST.json
python scripts/verify_release_assets.py
```

The default mode previews changes. Only tracked files enter the manifest; local untracked notes and credentials are excluded. The updater rejects changed or removed frozen code, benchmark inputs, archived results, source plots, vendored sources and source licenses. It also rejects environment files, runtime directories and manuscript administration. Historical changes require a separately versioned research preparation, rather than replacing historical hashes. The command requires a Git checkout; an extracted ZIP can be verified without Git.

## Build a source package

```bash
python scripts/build_release_archive.py --output outputs/BioAgent-v0.1.0-source.zip
```

The builder checks the release first, packages exactly the manifest files and `RELEASE_MANIFEST.json`, and writes a `.zip.sha256` sidecar. It excludes local Git history, credentials, installed environments, model weights and generated outputs. An existing output is not overwritten.

## Review and publish

1. Keep the repository private while reviewing a prepared release. Check the CI runs attached to the intended commit.
2. Run the complete local tests, inspect the source ZIP independently and review all secret-scan findings. The scanner configuration records only specifically reviewed historical false positives.
3. Check the version in `CITATION.cff`, `RELEASE_MANIFEST.json`, changelog and release notes. A software version is not a claim of manuscript acceptance or journal publication.
4. Prepare a GitHub draft release targeting the verified commit. Attach the source ZIP and SHA256 sidecar.
5. After the maintainer authorizes public release, change the repository visibility and publish the draft. Verify access from an unauthenticated session before announcing public availability.

No DOI or archival repository identifier is claimed until one has actually been assigned. The software/data-source terms and evidence levels are described in [third-party notices](THIRD_PARTY_NOTICES.md), [data sources](DATA_SOURCES.md) and [reproducibility](REPRODUCIBILITY.md).
