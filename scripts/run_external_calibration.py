"""Current-version, fixed-12 Dev calibration only. Never loads Test or Dev-70."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.run_external_baselines import ROOT, MODELS, code_snapshot, execute, load_dev_canary, write_new
from scripts.external_baselines.common import now, file_digest

CONDITIONS = ("ReAct", "CodeAct", "SciToolAgent", "Biomni-A1", "F1")


async def calibrate(args):
    specification, items = load_dev_canary()
    if len(items) != 12:
        raise ValueError("Calibration is restricted to the authorized fixed twelve Dev tasks")
    output = (ROOT / "evaluation/results" / args.run_id).resolve()
    output.relative_to((ROOT / "evaluation/results").resolve())
    output.mkdir(parents=True, exist_ok=False)
    semaphore = asyncio.Semaphore(args.concurrency)
    barrier = asyncio.Barrier(len(CONDITIONS))
    run_ids = {condition: args.run_id + "-" + condition.lower() for condition in CONDITIONS}
    for name in run_ids.values():
        if (output.parent / name).exists():
            raise ValueError("A calibration child directory already exists: " + name)
    manifest = {"scope": "fixed12_current_version_dev_calibration_not_paper_results",
                "status": "running", "started_at": now(), "conditions": list(CONDITIONS),
                "models": list(MODELS), "question_ids": [i["id"] for i in items],
                "repeats": 1, "planned_units": 120, "agent_concurrency": args.concurrency,
                "ordering": "FIFO shared slots; one active unit per method; all preflights before online calls",
                "queue_wait_excluded_from_adapter_latency": True,
                "dataset_canonical_sha256": specification["dataset_canonical_sha256"],
                "code_sha256": code_snapshot(), "child_run_ids": run_ids,
                "automatic_run_resampling": False}
    write_new(output / "manifest.start.json", manifest)
    try:
        async with asyncio.TaskGroup() as group:
            for condition in CONDITIONS:
                group.create_task(execute(SimpleNamespace(
                    condition=condition, models=list(MODELS), run_id=run_ids[condition],
                    tool_base=args.tool_base, limit=None, preflight_only=False,
                    resume_unstarted_from=[], unit_semaphore=semaphore, calibration_barrier=barrier)))
    except BaseException as exc:
        write_new(output / "manifest.final.json", {**manifest, "status": "interrupted",
                  "finished_at": now(), "exception_type": type(exc).__name__,
                  "code_unchanged_during_run": code_snapshot() == manifest["code_sha256"]})
        raise
    children = {}
    for condition, name in run_ids.items():
        path = output.parent / name / "manifest.final.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        children[condition] = {"run_id": name, "status": value["status"],
                              "adapter": value["adapter"], "recorded_units": value["recorded_units"],
                              "manifest_sha256": file_digest(path),
                              "code_unchanged_during_run": value["code_unchanged_during_run"]}
    complete = all(c["status"] == "completed" and c["recorded_units"] == 24 for c in children.values())
    write_new(output / "manifest.final.json", {**manifest,
              "status": "completed" if complete else "incomplete",
              "finished_at": now(), "children": children,
              "code_unchanged_during_run": code_snapshot() == manifest["code_sha256"]})
    print("CALIBRATION_DIR", str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tool-base", default="http://127.0.0.1:18080")
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3, 4), default=4)
    asyncio.run(calibrate(parser.parse_args()))


if __name__ == "__main__":
    main()
