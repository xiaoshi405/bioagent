"""Check local installation and agent assets without loading weights or calling a model."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def requirement_pins(filename):
    pins = {}
    for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            pins.update(requirement_pins(line[3:].strip()))
        elif "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


def check_environment(mode="core", before_index=False):
    errors = []
    if sys.version_info < (3, 11):
        errors.append("Use Python 3.13, the verified installation environment.")
    filename = {"core": "requirements.txt", "agent": "requirements-agent.txt",
                "baselines": "requirements-baselines.txt"}[mode]
    pins = requirement_pins(filename)
    for name, expected in pins.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            errors.append(f"Missing package: {name}; run python -m pip install -r {filename}")
            continue
        # A CPU-local build such as 2.10.0+cpu satisfies the public ==2.10.0 pin.
        if actual.split("+", 1)[0] != expected:
            errors.append(f"Package version differs: {name}={actual}; expected {expected}")
    for relative in ["tests/fixtures/fasta/multi_protein.fasta", "data/toxin_protein.csv"]:
        if not (ROOT / relative).is_file():
            errors.append("Missing bundled input: " + relative)
    if mode == "agent" and not errors:
        from app.core.config import Config
        from app.core.embedding_runtime import validate_local_embedding_model
        cfg = Config()
        if not cfg.API_KEY.strip():
            errors.append("API_KEY is empty; configure your own provider in .env.")
        if not cfg.MODEL_NAME.strip():
            errors.append("MODEL_NAME is empty; configure a model available to your account in .env.")
        if not cfg.API_BASE.strip():
            errors.append("API_BASE is empty; configure an OpenAI-compatible base URL in .env.")
        embedding = validate_local_embedding_model(cfg.BGE_MODEL_PATH)
        if embedding.get("status") != "success":
            codes = [issue["code"] for issue in embedding.get("issues", [])]
            errors.append("Embedding assets are incomplete: " + ", ".join(codes)
                          + "; prepare a complete local BGE snapshot and set BGE_MODEL_PATH.")
        if not cfg.TOXIN_PROTEIN or not Path(cfg.TOXIN_PROTEIN).is_file():
            errors.append("Set TOXIN_PROTEIN=data/toxin_protein.csv in .env.")
        if not before_index:
            for name in ["graph_store.json", "index_store.json", "docstore.json"]:
                path = Path(cfg.PERSIST_DIR) / name
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(value, dict):
                        raise ValueError("Index JSON must be an object")
                except (OSError, ValueError):
                    errors.append(f"Missing or invalid index file: {name}; run python -m scripts.fix_my_index")
    return {"status": "PASS" if not errors else "FAIL", "mode": mode,
            "packages_checked": len(pins), "index_check_skipped": before_index,
            "model_requests": 0, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["core", "agent", "baselines"], default="core")
    parser.add_argument("--before-index", action="store_true",
                        help="Check model/configuration before building the local index.")
    args = parser.parse_args()
    try:
        report = check_environment(args.mode, args.before_index)
    except (OSError, ValueError, ImportError) as exc:
        # Do not expose configuration values or credential-containing exception messages.
        report = {"status": "FAIL", "mode": args.mode,
                  "errors": ["Environment check failed: " + type(exc).__name__], "model_requests": 0}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
