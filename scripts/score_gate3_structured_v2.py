#!/usr/bin/env python3
"""Versioned, conservative structured-evidence scorer for Gate3 Test-150.

This post-processor never mutates the frozen raw traces or deterministic-v1.1
artifacts.  It compares the nested expected result with JSON evidence already
present in each captured answer, writes a derived trace, and creates a small
blinded, stratified human-audit queue.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


POLICY_VERSION = "gate3-structured-evidence-v2.0"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _extract_json_objects(text: str) -> list[Any]:
    """Extract JSON objects/arrays without assuming Markdown formatting."""
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    while index < len(text):
        brace = text.find("{", index)
        bracket = text.find("[", index)
        starts = [candidate for candidate in (brace, bracket) if candidate >= 0]
        if not starts:
            break
        start = min(starts)
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        values.append(value)
        index = start + consumed
    return values


def _iter_nodes(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_nodes(child)


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _tolerance_for(path: tuple[str, ...], tolerance: Any) -> float:
    if isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool):
        return float(tolerance)
    if not isinstance(tolerance, dict):
        return 0.0
    full_path = ".".join(path)
    leaf = path[-1] if path else ""
    for key in (full_path, leaf, "absolute", "abs", "default"):
        candidate = tolerance.get(key)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return float(candidate)
    return 0.0


def deep_contains(
    expected: Any,
    observed: Any,
    *,
    tolerance: Any = None,
    path: tuple[str, ...] = (),
) -> bool:
    """Return whether observed contains expected, recursively and conservatively."""
    if isinstance(expected, bool) or isinstance(observed, bool):
        return type(expected) is type(observed) and expected == observed
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        allowed = _tolerance_for(path, tolerance)
        return math.isclose(float(expected), float(observed), rel_tol=0.0, abs_tol=allowed)
    if expected is None:
        return observed is None
    if isinstance(expected, str):
        return isinstance(observed, str) and _normalized_text(expected) == _normalized_text(observed)
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            return False
        return all(
            deep_contains(item, observed[index], tolerance=tolerance, path=path + (str(index),))
            for index, item in enumerate(expected)
        )
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            return False
        for key, value in expected.items():
            if key not in observed:
                return False
            if not deep_contains(value, observed[key], tolerance=tolerance, path=path + (str(key),)):
                return False
        return True
    return expected == observed


def _target_for_rubric(rubric: dict[str, Any]) -> tuple[str, Any]:
    expected_output = rubric.get("expected_output")
    behavior = rubric.get("expected_behavior")
    if not isinstance(expected_output, dict):
        return "expected_output", expected_output
    if behavior in {"execute", "repair", "repair_then_execute"} and "final_output" in expected_output:
        return "final_output", expected_output["final_output"]
    if behavior == "reject" and "actual_output" in expected_output:
        return "actual_output", expected_output["actual_output"]
    return "expected_output", expected_output


def score_review_record(
    record: dict[str, Any], existing_scoring: dict[str, Any] | None = None
) -> dict[str, Any]:
    rubric = record.get("rubric") if isinstance(record.get("rubric"), dict) else {}
    answer = str(record.get("answer") or "")
    tolerance = rubric.get("tolerance")
    target_name, target = _target_for_rubric(rubric)
    json_values = _extract_json_objects(answer)

    matched = False
    matched_root: int | None = None
    if target is not None:
        for root_index, value in enumerate(json_values):
            if any(deep_contains(target, node, tolerance=tolerance) for node in _iter_nodes(value)):
                matched = True
                matched_root = root_index
                break

    normalized_answer = _normalized_text(answer)
    forbidden_hits = [
        claim
        for claim in rubric.get("forbidden_claims", [])
        if isinstance(claim, str)
        and len(_normalized_text(claim)) >= 8
        and _normalized_text(claim) in normalized_answer
    ]

    behavior = rubric.get("expected_behavior")
    prior_labels = (existing_scoring or {}).get("labels")
    if not isinstance(prior_labels, dict):
        prior_labels = {}
    correct_rejection = prior_labels.get("correct_rejection")
    reject_confirmation_fields: list[str] = []
    if behavior == "reject":
        expected_output = rubric.get("expected_output")
        if isinstance(expected_output, dict):
            for field in (
                "actual_status",
                "actual_error_code",
                "actual_message",
                "primary_failure_reason",
            ):
                expected_value = expected_output.get(field)
                if (
                    isinstance(expected_value, (str, int, float, bool))
                    and len(str(expected_value).strip()) >= 2
                    and _normalized_text(str(expected_value)) in normalized_answer
                ):
                    reject_confirmation_fields.append(field)

    if forbidden_hits:
        value: bool | None = False
        reason = "forbidden_claim_literal_hit"
        confidence = "high"
    elif behavior == "reject" and correct_rejection is False:
        value = False
        reason = "deterministic_rejection_check_failed"
        confidence = "high"
    elif (
        behavior == "reject"
        and correct_rejection is True
        and (matched or reject_confirmation_fields)
    ):
        value = True
        reason = "deterministic_rejection_check_and_answer_evidence_match"
        confidence = "high"
    elif target is None:
        value = None
        reason = "no_structured_gold_target"
        confidence = "none"
    elif matched:
        value = True
        reason = "structured_gold_contained_in_answer_evidence"
        confidence = "high"
    elif not json_values:
        value = None
        reason = "no_json_evidence_in_answer"
        confidence = "none"
    else:
        value = None
        reason = "structured_gold_not_found"
        confidence = "none"

    return {
        "value": value,
        "source": "automatic" if value is not None else "manual",
        "reason": reason,
        "confidence": confidence,
        "evidence": {
            "target": target_name,
            "json_root_count": len(json_values),
            "matched_json_root": matched_root,
            "forbidden_literal_hit_count": len(forbidden_hits),
            "reject_confirmation_fields": reject_confirmation_fields,
        },
    }


def _trace_key(record: dict[str, Any]) -> str:
    explicit = record.get("execution_key")
    if isinstance(explicit, str) and explicit:
        return explicit
    fields = (
        record.get("question_id"),
        record.get("model"),
        record.get("condition"),
        record.get("repeat_index"),
    )
    return "|".join(str(value) for value in fields)


def _stable_sample(
    candidates: list[dict[str, Any]],
    *,
    per_stratum: int,
    seed: int,
) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        key = (
            str(candidate["map"].get("model", "")),
            str(candidate["map"].get("condition", "")),
            str(candidate["behavior"]),
        )
        strata[key].append(candidate)
    selected: list[dict[str, Any]] = []
    for stratum in sorted(strata):
        rows = sorted(strata[stratum], key=lambda row: str(row["queue"].get("review_id", "")))
        stratum_seed = int.from_bytes(
            hashlib.sha256(f"{seed}|{'|'.join(stratum)}".encode("utf-8")).digest()[:8],
            "big",
        )
        rng = random.Random(stratum_seed)
        rng.shuffle(rows)
        selected.extend(rows[:per_stratum])
    return selected


def process_run(args: argparse.Namespace) -> dict[str, Any]:
    trace_rows = read_jsonl(args.trace)
    queue_rows = read_jsonl(args.review_queue)
    map_rows = read_jsonl(args.review_map)
    scoring_rows = read_jsonl(args.scoring)

    queue_by_id = {str(row["review_id"]): row for row in queue_rows}
    map_by_id = {str(row["review_id"]): row for row in map_rows}
    scoring_by_id = {str(row["review_id"]): row for row in scoring_rows}
    if set(queue_by_id) != set(map_by_id) or set(queue_by_id) != set(scoring_by_id):
        raise ValueError("review queue, map, and scoring IDs are not identical")

    decisions: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    for review_id in sorted(queue_by_id):
        queue_record = queue_by_id[review_id]
        mapping = map_by_id[review_id]
        decision = score_review_record(queue_record, scoring_by_id[review_id])
        decisions[review_id] = decision
        behavior = str((queue_record.get("rubric") or {}).get("expected_behavior", "unknown"))
        candidates.append(
            {
                "queue": queue_record,
                "map": mapping,
                "decision": decision,
                "behavior": behavior,
            }
        )

    updated_scoring: list[dict[str, Any]] = []
    decision_by_execution: dict[str, dict[str, Any]] = {}
    decision_by_trace_tuple: dict[tuple[str, str, int], dict[str, Any]] = {}
    for review_id in sorted(scoring_by_id):
        row = copy.deepcopy(scoring_by_id[review_id])
        decision = decisions[review_id]
        row["policy_version"] = POLICY_VERSION
        labels = row.setdefault("labels", {})
        labels["final_answer_correct"] = decision["value"]
        row.setdefault("decisions", {})["final_answer_correct"] = decision
        pending = [field for field in row.get("review_required_fields", []) if field != "final_answer_correct"]
        if decision["value"] is None and "final_answer_correct" not in pending:
            pending.append("final_answer_correct")
        row["review_required_fields"] = sorted(pending)
        updated_scoring.append(row)
        execution_key = str(row.get("execution_key") or map_by_id[review_id].get("execution_key"))
        decision_by_execution[execution_key] = decision
        mapping = map_by_id[review_id]
        trace_tuple = (
            str(mapping.get("question_id")),
            str(mapping.get("condition")),
            int(mapping.get("repeat_index")),
        )
        if trace_tuple in decision_by_trace_tuple:
            raise ValueError(f"duplicate trace tuple in one model run: {trace_tuple}")
        decision_by_trace_tuple[trace_tuple] = decision

    updated_trace: list[dict[str, Any]] = []
    missing_trace_keys: list[str] = []
    for row in trace_rows:
        updated = copy.deepcopy(row)
        key = _trace_key(row)
        decision = decision_by_execution.get(key)
        if decision is None:
            trace_tuple = (
                str(row.get("question_id")),
                str(row.get("condition")),
                int(row.get("repeat_index")),
            )
            decision = decision_by_trace_tuple.get(trace_tuple)
        if decision is None:
            missing_trace_keys.append(key)
        else:
            evaluation = updated.setdefault("evaluation", {})
            evaluation["final_answer_correct"] = decision["value"]
        updated_trace.append(updated)
    if missing_trace_keys:
        raise ValueError(f"{len(missing_trace_keys)} trace rows have no v2 scoring decision")

    uncertain = [row for row in candidates if row["decision"]["value"] is None]
    automatic = [row for row in candidates if row["decision"]["value"] is not None]
    selected_uncertain = _stable_sample(
        uncertain, per_stratum=args.uncertain_per_stratum, seed=args.seed + 1
    )
    selected_audit = _stable_sample(
        automatic, per_stratum=args.audit_per_stratum, seed=args.seed
    )
    selected_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    for reason, rows in (("uncertain", selected_uncertain), ("automatic_audit", selected_audit)):
        for row in rows:
            selected_by_id[str(row["queue"]["review_id"])] = (reason, row)

    blinded_queue: list[dict[str, Any]] = []
    review_template: list[dict[str, Any]] = []
    private_map: list[dict[str, Any]] = []
    for review_id in sorted(selected_by_id):
        selection_reason, candidate = selected_by_id[review_id]
        queue_copy = copy.deepcopy(candidate["queue"])
        queue_copy["v2_selection_reason"] = selection_reason
        queue_copy["v2_proposed_final_answer_correct"] = candidate["decision"]["value"]
        queue_copy["v2_evidence_summary"] = candidate["decision"]["evidence"]
        blinded_queue.append(queue_copy)
        fields = sorted(set(queue_copy.get("pending_fields", [])) | {"final_answer_correct"})
        review_template.append(
            {
                "review_id": review_id,
                "reviewer_id": "",
                "reviewed_at": "",
                "labels": {field: None for field in fields},
                "notes": "",
            }
        )
        private = copy.deepcopy(candidate["map"])
        private["v2_selection_reason"] = selection_reason
        private_map.append(private)

    write_jsonl(args.output_trace, updated_trace)
    write_jsonl(args.output_scoring, updated_scoring)
    write_jsonl(args.output_review_queue, blinded_queue)
    write_jsonl(args.output_review_template, review_template)
    write_jsonl(args.output_review_map, private_map)
    if args.manifest is not None and args.output_manifest is not None:
        source_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest = copy.deepcopy(source_manifest)
        manifest.setdefault("design", {})["hybrid_conditions"] = ["E3", "L1", "F1"]
        manifest["derived_scoring"] = {
            "policy_version": POLICY_VERSION,
            "source_manifest": str(args.manifest.resolve()),
            "source_trace": str(args.trace.resolve()),
            "derived_trace": str(args.output_trace.resolve()),
        }
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.output_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        factorial_conditions = ["E0", "E1", "E2", "E3", "S1", "L1"]
        hybrid_conditions = ["E3", "L1", "F1"]
        if args.output_factorial_trace is not None and args.output_factorial_manifest is not None:
            write_jsonl(
                args.output_factorial_trace,
                (row for row in updated_trace if row.get("condition") in factorial_conditions),
            )
            factorial_manifest = copy.deepcopy(manifest)
            factorial_manifest["design"]["conditions"] = factorial_conditions
            schedule = factorial_manifest["design"].get("condition_schedule")
            if isinstance(schedule, list):
                factorial_manifest["design"]["condition_schedule"] = [
                    [condition for condition in cycle if condition in factorial_conditions]
                    for cycle in schedule
                ]
            factorial_manifest["derived_scoring"]["analysis_slice"] = "factorial"
            factorial_manifest["derived_scoring"]["derived_trace"] = str(
                args.output_factorial_trace.resolve()
            )
            args.output_factorial_manifest.write_text(
                json.dumps(factorial_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if args.replan_ledger is not None and args.output_factorial_replan_ledger is not None:
                replan_rows = read_jsonl(args.replan_ledger)
                write_jsonl(
                    args.output_factorial_replan_ledger,
                    (
                        row
                        for row in replan_rows
                        if str(row.get("experiment_id", "")).endswith("-l1")
                    ),
                )
        if args.output_hybrid_trace is not None and args.output_hybrid_manifest is not None:
            write_jsonl(
                args.output_hybrid_trace,
                (row for row in updated_trace if row.get("condition") in hybrid_conditions),
            )
            hybrid_manifest = copy.deepcopy(manifest)
            hybrid_manifest["design"]["conditions"] = hybrid_conditions
            hybrid_manifest["design"]["hybrid_conditions"] = hybrid_conditions
            hybrid_manifest["derived_scoring"]["analysis_slice"] = "hybrid"
            hybrid_manifest["derived_scoring"]["derived_trace"] = str(
                args.output_hybrid_trace.resolve()
            )
            args.output_hybrid_manifest.write_text(
                json.dumps(hybrid_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    values = Counter(
        "manual" if candidate["decision"]["value"] is None else str(candidate["decision"]["value"]).lower()
        for candidate in candidates
    )
    reasons = Counter(candidate["decision"]["reason"] for candidate in candidates)
    behavior_values: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in candidates:
        value = candidate["decision"]["value"]
        behavior_values[candidate["behavior"]]["manual" if value is None else str(value).lower()] += 1

    summary = {
        "schema_version": "1.0",
        "policy_version": POLICY_VERSION,
        "inputs": {
            "trace": str(args.trace.resolve()),
            "review_queue": str(args.review_queue.resolve()),
            "review_map": str(args.review_map.resolve()),
            "scoring": str(args.scoring.resolve()),
        },
        "counts": {
            "records": len(candidates),
            "automatic_true": values["true"],
            "automatic_false": values["false"],
            "manual_unresolved": values["manual"],
            "human_queue": len(blinded_queue),
            "human_queue_uncertain": sum(
                row["v2_selection_reason"] == "uncertain" for row in blinded_queue
            ),
            "human_queue_automatic_audit": sum(
                row["v2_selection_reason"] == "automatic_audit" for row in blinded_queue
            ),
        },
        "decision_reasons": dict(sorted(reasons.items())),
        "by_expected_behavior": {
            behavior: dict(sorted(counts.items()))
            for behavior, counts in sorted(behavior_values.items())
        },
        "sampling": {
            "seed": args.seed,
            "audit_per_model_condition_behavior": args.audit_per_stratum,
            "uncertain_per_model_condition_behavior": args.uncertain_per_stratum,
        },
        "outputs": {
            "derived_trace": str(args.output_trace.resolve()),
            "derived_scoring": str(args.output_scoring.resolve()),
            "blinded_review_queue": str(args.output_review_queue.resolve()),
            "review_template": str(args.output_review_template.resolve()),
            "private_review_map": str(args.output_review_map.resolve()),
            "derived_manifest": (
                str(args.output_manifest.resolve()) if args.output_manifest is not None else None
            ),
        },
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--review-map", type=Path, required=True)
    parser.add_argument("--scoring", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--replan-ledger", type=Path)
    parser.add_argument("--output-trace", type=Path, required=True)
    parser.add_argument("--output-scoring", type=Path, required=True)
    parser.add_argument("--output-review-queue", type=Path, required=True)
    parser.add_argument("--output-review-template", type=Path, required=True)
    parser.add_argument("--output-review-map", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--output-factorial-trace", type=Path)
    parser.add_argument("--output-factorial-manifest", type=Path)
    parser.add_argument("--output-factorial-replan-ledger", type=Path)
    parser.add_argument("--output-hybrid-trace", type=Path)
    parser.add_argument("--output-hybrid-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--audit-per-stratum", type=int, default=2)
    parser.add_argument("--uncertain-per-stratum", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (args.manifest is None) != (args.output_manifest is None):
        raise SystemExit("--manifest and --output-manifest must be supplied together")
    if (args.output_factorial_trace is None) != (args.output_factorial_manifest is None):
        raise SystemExit("factorial trace and manifest outputs must be supplied together")
    if (args.replan_ledger is None) != (args.output_factorial_replan_ledger is None):
        raise SystemExit("replan ledger input and factorial ledger output must be supplied together")
    if (args.output_hybrid_trace is None) != (args.output_hybrid_manifest is None):
        raise SystemExit("hybrid trace and manifest outputs must be supplied together")
    summary = process_run(args)
    print(json.dumps(summary["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
