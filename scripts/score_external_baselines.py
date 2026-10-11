"""Offline fixed-12 Dev scoring with immutable raw archives and blind review."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from scripts.run_external_baselines import ROOT, load_dev_canary, make_request, write_new
from scripts.external_baselines.common import file_digest, now, encode
from scripts.external_baselines.scoring import SCORER_VERSION, score_run, review_packet


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(source_names, run_id, labels_path=None):
    specification, items = load_dev_canary()
    by_id = {i["id"]: i for i in items}
    output = (ROOT / "evaluation/results" / run_id).resolve()
    output.relative_to((ROOT / "evaluation/results").resolve())
    if output.exists():
        raise ValueError("Scoring output must be new; never overwrite prior decisions")
    reviews = {}
    if labels_path:
        for review in read_jsonl(Path(labels_path)):
            key = review["evidence_sha256"]
            if key in reviews:
                raise ValueError("Duplicate review")
            reviews[key] = review
    scores, packets, mapping, sources, planned = [], [], [], [], set()
    seen = set()
    used_reviews = set()
    for name in source_names:
        path = (ROOT / "evaluation/results" / name).resolve()
        path.relative_to((ROOT / "evaluation/results").resolve())
        manifest = json.loads((path / "manifest.final.json").read_text(encoding="utf-8"))
        condition = manifest.get("condition")
        if condition is None and manifest["adapter"] == "react-protein-zero-shot-v1":
            condition = "ReAct"
        if condition is None:
            raise ValueError("Archive method identity is missing")
        if manifest["scope"] != "development_only_not_for_paper_results":
            raise ValueError("Scorer accepts archived Dev runs only")
        if manifest["dataset_canonical_sha256"] != specification["dataset_canonical_sha256"]:
            raise ValueError("Different Dev input version")
        for relative, sha in manifest["code_sha256"].items():
            if file_digest(path / "source_snapshot" / relative) != sha:
                raise ValueError("Archived source snapshot changed: " + relative)
        plan_keys = manifest.get("planned_run_keys")
        if plan_keys is None:
            if manifest.get("repeats") != 1:
                raise ValueError("Legacy Dev archive has an unsupported repeat specification")
            plan_keys = [question+"|"+model for question in manifest["question_ids"] for model in manifest["models"]]
            if len(plan_keys) != manifest["planned_units"]:
                raise ValueError("Legacy Dev planned count cannot be reconstructed from its manifest")
        for key in plan_keys:
            question, model = key.split("|", 1)
            if question not in by_id:
                raise ValueError("Only the fixed twelve Dev questions are authorized")
            planned.add((condition, model, question, 0))
        events = collections.defaultdict(list)
        for event in read_jsonl(path / "events.jsonl"):
            events[event["run_key"]].append(event)
        for row in read_jsonl(path / "runs.jsonl"):
            row_condition = row.get("condition", condition)
            key = (row_condition, row["model"], row["question_id"], row["repeat"])
            if key in seen:
                raise ValueError("Duplicate unit; do not select a better attempt")
            if key not in planned or row["repeat"] != 0:
                raise ValueError("Run not present in fixed Dev manifest")
            seen.add(key)
            item = by_id[row["question_id"]]
            _, files = make_request(item)
            run_events = events[row["run_key"]]
            starts = [e for e in run_events if e["kind"] == "run_start"]
            ends = [e for e in run_events if e["kind"] == "run_complete"]
            if len(starts) != 1 or len(ends) != 1:
                raise ValueError("Incomplete or duplicate run boundary")
            for file in starts[0].get("input_files", []):
                if file_digest(Path(file["path"])) != file["sha256"]:
                    raise ValueError("Dev fixture changed since execution")
            score = score_run(item, row, run_events, files)
            review = reviews.get(score["evidence_sha256"])
            if review:
                score = score_run(item, row, run_events, files, review)
                used_reviews.add(score["evidence_sha256"])
            scores.append({"source_run_id": name, "run_key": row["run_key"],
                           "condition": row_condition, "adapter": manifest["adapter"],
                           "model": row["model"], "question_id": row["question_id"],
                           "repeat": row["repeat"], **score})
            packet = review_packet(item, row, run_events, files, score)
            if score["pending_review_fields"]:
                packets.append(packet)
            mapping.append({"review_id": packet["review_id"], "source_run_id": name,
                            "run_key": row["run_key"], "evidence_sha256": score["evidence_sha256"]})
        sources.append({"run_id": name, "adapter": manifest["adapter"],
                        "files_sha256": {file: file_digest(path / file)
                        for file in ("manifest.final.json", "events.jsonl", "runs.jsonl")}})
    if set(reviews) != used_reviews:
        raise ValueError("Review file includes unknown/stale runs")
    groups = []
    for condition, model in sorted({(k[0], k[1]) for k in planned}):
        group = [s for s in scores if s["condition"] == condition and s["model"] == model]
        n = sum(k[:2] == (condition, model) for k in planned)
        success = sum(s["scientific_task_success"] is True for s in group)
        failure = sum(s["scientific_task_success"] is False for s in group)
        unresolved = n-success-failure
        groups.append({"condition": condition, "model": model, "planned": n, "recorded": len(group),
                       "confirmed_success": success, "confirmed_failure": failure,
                       "unresolved_including_unrecorded": unresolved,
                       "coverage": (success+failure)/n, "success_rate_lower": success/n,
                       "success_rate_upper": (success+unresolved)/n})
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("scores.jsonl", scores), ("review_queue.blind.jsonl", packets),
                       ("review_map.private.jsonl", mapping)):
        with (output / name).open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(encode(row)+"\n")
    templates = [{"review_id": p["review_id"], "evidence_sha256": p["evidence_sha256"],
                  "reviewer_id": "", "reviewer_kind": "", "reviewed_at": "",
                  "reason": "", "evidence_locations": [],
                  "labels": {k: None for k in p["automatic_score"]["pending_review_fields"]}}
                 for p in packets]
    with (output / "review_template.jsonl").open("x", encoding="utf-8") as handle:
        for row in templates:
            handle.write(encode(row)+"\n")
    write_new(output / "summary.json", {"scope": "fixed12_dev_only_not_paper_accuracy",
              "scorer_version": SCORER_VERSION, "created_at": now(), "sources": sources,
              "scorer_source_sha256": file_digest(Path(__file__).parent / "external_baselines/scoring.py"),
              "scoring_entrypoint_sha256": file_digest(Path(__file__)),
              "labels_file_sha256": file_digest(Path(labels_path)) if labels_path else None,
              "planned_units": len(planned), "recorded_units": len(scores),
              "primary_resolved_units": sum(s["scientific_task_success"] is not None for s in scores),
              "review_queue_units": len(packets), "groups": groups,
              "human_reviewed_units": sum((s.get("review_source") or "").startswith("human:") for s in scores),
              "assistant_reviewed_units": sum((s.get("review_source") or "").startswith("assistant:") for s in scores)})
    print("SCORING_DIR", str(output), "planned", len(planned), "recorded", len(scores),
          "review_queue", len(packets), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-only", action="store_true", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--labels-file")
    args = parser.parse_args()
    evaluate(args.sources, args.run_id, args.labels_file)


if __name__ == "__main__":
    main()
