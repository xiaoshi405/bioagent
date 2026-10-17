"""Refresh public-file checksums while rejecting changes to historical evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.source_provenance import ARCHIVE, verify_source

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "RELEASE_MANIFEST.json"
FROZEN = "evaluation/biotooleval/external_baselines_freeze_v3_1_resource.json"
PROTECTED_PREFIXES = ("evaluation/biotooleval/", "third_party/", "results/", "reproduction/")
EXCLUDED_PARTS = {".git", ".venv", "venv", "node_modules", "__pycache__", "outputs", "logs", "uploads", "storage", "bge_model"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          check=True).stdout.decode("utf-8")


def checked_path(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    parts = Path(relative).parts
    if (root / relative).is_symlink() or set(parts) & EXCLUDED_PARTS:
        raise ValueError("Excluded or symbolic-link path: " + relative)
    if any(part.startswith(".env") for part in parts):
        raise ValueError("Local environment file must not be released: " + relative)
    if relative.startswith(("docs/reviews/", "docs/submission/")) or any(
            word in relative for word in ["Cover_Letter", "Title_Page", "ORCID.docx", "Author Agreement"]):
        raise ValueError("Manuscript administration file must not be released: " + relative)
    return path


def plan_update(root=ROOT):
    root = Path(root).resolve()
    if not (root / ".git").exists():
        raise ValueError("Use a Git checkout; source ZIPs can be verified but not maintained with this command.")
    baseline = json.loads(git(root, "show", "HEAD:" + MANIFEST))
    current = json.loads((root / MANIFEST).read_text(encoding="utf-8-sig"))
    tracked = set(git(root, "ls-files", "-z").rstrip("\0").split("\0"))
    if MANIFEST not in tracked:
        raise ValueError("The release manifest must be tracked.")
    previous = baseline["files"]
    # Validate the historical manifest before reading the list of frozen code.
    frozen_path = root / FROZEN
    if FROZEN in previous and sha(frozen_path) != previous[FROZEN]["sha256"]:
        raise ValueError("Historical freeze changed; prepare a new versioned experiment instead.")
    frozen_code = set()
    frozen_hashes = {}
    if FROZEN in previous:
        frozen_hashes = {name.replace("\\", "/"): value for name, value in json.loads(
            frozen_path.read_text(encoding="utf-8-sig"))["code_sha256"].items()}
        frozen_code = set(frozen_hashes)
    protected = {name for name in previous if name.startswith(PROTECTED_PREFIXES)}
    protected |= frozen_code | ({"LICENSE", "data/toxin_protein.csv"} & set(previous))
    for name in sorted(protected):
        path = checked_path(root, name)
        if name in frozen_hashes and ARCHIVE in previous:
            if sha(root / ARCHIVE) != previous[ARCHIVE]["sha256"]:
                raise ValueError("Protected historical/source asset changed: " + ARCHIVE)
            verify_source(root, name, frozen_hashes[name])
            if name not in tracked and Path(name).suffix != '.md':
                raise ValueError("Protected historical/source asset removed: " + name)
            continue
        if name not in tracked or not path.is_file() or sha(path) != previous[name]["sha256"]:
            raise ValueError("Protected historical/source asset changed: " + name)
    files = {}
    added, changed = [], []
    for name in sorted(tracked - {MANIFEST}):
        path = checked_path(root, name)
        if not path.is_file():
            raise ValueError("Tracked file is missing: " + name)
        digest = sha(path)
        if name in previous:
            entry = dict(previous[name])
            if digest != entry["sha256"]:
                changed.append(name)
                entry["mode"] = "public_maintenance"
        else:
            added.append(name)
            entry = {"source": "public repository", "mode": "public_maintenance",
                     "reason": "public documentation, maintenance or regression test"}
        entry["sha256"] = digest
        files[name] = entry
    removed = sorted(set(previous) - tracked)
    current["files"] = files
    report = {"status": "PASS", "added": added, "changed": changed, "removed": removed,
              "protected_files_checked": len(protected), "release_files": len(files) + 1,
              "experiments_run": 0}
    return current, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        manifest, report = plan_update()
        if args.write:
            target = ROOT / MANIFEST
            temporary = ROOT / (MANIFEST + ".tmp")
            temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
            temporary.replace(target)
        report["written"] = args.write
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        report = {"status": "FAIL", "error": str(exc), "written": False}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
