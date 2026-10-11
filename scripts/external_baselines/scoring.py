"""Method-independent offline scoring; no Gold or decisions reach an agent.

Structured execution evidence and semantic reply review are separate. None is
unresolved, never a failure imputation. Python actions are preserved for review.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .audit import _named_values, normalized_sequence, sequence_hash, source_index
from .common import digest

SCORER_VERSION = "external-scientific-outcome-v1"
REVIEW_FIELDS = ("identity_correct", "scientific_output_correct", "final_answer_correct",
                 "rejection_response_correct")


def tri_and(values):
    values = list(values)
    if False in values:
        return False
    return True if values and all(v is True for v in values) else None


def matches(actual, expected, tolerance=1e-6):
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isfinite(expected)
                and abs(actual-expected) <= tolerance)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and matches(actual[k], v, tolerance)
                                                for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            matches(a, b, tolerance) for a, b in zip(actual, expected))
    return actual == expected


def reference_objects(item, files):
    """Independent input parsing only; never execute a scientific tool."""
    from Bio import SeqIO
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    objects = []
    inp = item.get("input", {})
    if isinstance(inp.get("sequence"), str):
        objects.append({"source": "request_sequence", "sequence": normalized_sequence(inp["sequence"])})
    for file in files:
        path = Path(file)
        if path.suffix.lower() in {".fasta", ".fa", ".faa"}:
            with path.open(encoding="utf-8-sig") as handle:
                objects += [{"source": str(path), "record_id": r.id,
                             "sequence": normalized_sequence(str(r.seq))} for r in SeqIO.parse(handle, "fasta")]
        elif path.suffix.lower() == ".pdb":
            model = next(PDBParser(QUIET=True).get_structure("input", str(path)).get_models(), None)
            for chain in model or []:
                seq = "".join(seq1(r.resname, custom_map={"MSE": "M"}) for r in chain if r.id[0] == " ")
                if seq:
                    objects.append({"source": str(path), "chain_id": chain.id, "sequence": seq})
    for obj in objects:
        obj["sequence_sha256"] = sequence_hash(obj["sequence"])
    return objects


def output_contract(item, objects):
    expected = copy.deepcopy(item["gold"].get("expected_output", {}))
    inp = item.get("input", {})
    target = [o for o in objects if (not inp.get("record_id") or o.get("record_id") == inp["record_id"])
              and (not inp.get("chain_id") or o.get("chain_id") == inp["chain_id"])]
    # Fixed Dev rubrics omit two explicitly requested outputs. These additions
    # come from source characters, not a preferred tool/path or new experiment.
    if item.get("id") in {"bte_108", "bte_120"} and len(target) == 1:
        expected["counts"] = {aa: target[0]["sequence"].count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    if item.get("id") == "bte_116":
        expected["chains"] = {o["chain_id"]: {"sequence": o["sequence"],
                             "sequence_length": len(o["sequence"])} for o in target}
    return expected


def _file_same(a, b):
    try:
        return Path(a).resolve() == Path(b).resolve()
    except (TypeError, ValueError, OSError):
        return False


def call_identity(item, events, objects):
    inp = item.get("input", {})
    targets = [o for o in objects if (not inp.get("record_id") or o.get("record_id") == inp["record_id"])
               and (not inp.get("chain_id") or o.get("chain_id") == inp["chain_id"])]
    evaluated = []
    for e in events:
        if e.get("kind") != "tool_call":
            continue
        args = e.get("resolved_arguments") or {}
        checks = []
        reasons = []
        source = args.get("file_path")
        if source is not None:
            checks.append(any(_file_same(source, o["source"]) for o in objects))
            reasons.append("supplied_file_boundary")
        for key in ("record_id", "chain_id", "record_ids"):
            if inp.get(key) is not None and key in args:
                checks.append(args[key] == inp[key]); reasons.append("explicit_"+key)
        if isinstance(args.get("sequence"), str):
            actual = sequence_hash(args["sequence"])
            candidates = [o for o in objects if o["sequence_sha256"] == actual]
            wanted = [o for o in targets if o["sequence_sha256"] == actual]
            if not wanted:
                checks.append(False); reasons.append("sequence_outside_requested_object")
            elif len(candidates) == 1:
                if candidates[0]["source"] == "request_sequence":
                    checks.append(True); reasons.append("request_sequence_value")
                else:
                    reads = [p for p in events if p["event_index"] < e["event_index"] and (
                        (p.get("kind") == "tool_call" and p.get("status") == "success"
                         and _file_same((p.get("resolved_arguments") or {}).get("file_path"), candidates[0]["source"]))
                        or (p.get("kind") == "input_read" and (p.get("output") or {}).get("status") == "success"
                            and _file_same((p.get("output") or {}).get("file_path"), candidates[0]["source"]))
                        or (p.get("kind") == "kernel_file_open" and _file_same(p.get("file_path"), candidates[0]["source"])))]
                    checks.append(True if reads else None)
                    reasons.append("unique_source_value_with_read" if reads else "source_read_not_recorded")
            else:
                selected = [p for p in evaluated if p["identity_correct"] is True
                            and (p["event"].get("tool_name") in {"select_fasta_record", "select_pdb_chain"})
                            and any(isinstance(s, str) and sequence_hash(s) == actual
                                    for s in _named_values(p["event"].get("output"), "sequence"))]
                # Literal value copying cannot distinguish equal-valued records.
                # Require an explicit log reference/artifact link, not selection
                # merely appearing somewhere earlier in the execution.
                refs = {r.get("call_id") for r in e.get("reference_bindings", [])}
                linked = any(p["call_id"] in refs for p in selected)
                checks.append(True if linked else None); reasons.append("duplicate_value_explicit_link" if linked else "duplicate_value_lineage_unresolved")
        for arg_key, producer_key in (("collection", "collection"), ("alignment", "alignment")):
            if arg_key in args:
                producers = [p for p in evaluated if p["event"].get("status") == "success"
                             and any(args[arg_key] == v for v in _named_values(p["event"].get("output"), producer_key))]
                checks.append(True if any(p["identity_correct"] is True for p in producers) else None)
                reasons.append(arg_key+"_producer_evidence")
        if e.get("tool_name") == "calculate_protein_anm" and source is not None:
            chains = [o for o in objects if "chain_id" in o and _file_same(o["source"], source)]
            if inp.get("chain_id"):
                target_chain = inp["chain_id"]
            else:
                longest = [o for o in chains if len(o["sequence"]) == max([len(x["sequence"]) for x in chains], default=-1)]
                target_chain = longest[0]["chain_id"] if len(longest) == 1 else None
            actual_chain = (e.get("output") or {}).get("chain_id")
            checks.append(actual_chain == target_chain if target_chain and actual_chain else None)
            reasons.append("requested_or_unique_largest_ANM_chain")
        value = tri_and(checks)
        evaluated.append({"call_id": e["call_id"], "event_index": e["event_index"],
                          "identity_correct": value, "reasons": reasons, "event": e})
    return evaluated


def final_response_text(result):
    response = result.get("final_response")
    if isinstance(response, str) and "**Final Answer:**" in response:
        # F1/SciToolAgent prefixes are tool transcripts, not the final answer.
        return response.rsplit("**Final Answer:**", 1)[1].strip()
    return response


def score_run(item, row, events, files, review=None):
    result = row["result"]
    objects = reference_objects(item, files)
    identities = call_identity(item, events, objects)
    expected = output_contract(item, objects)
    calls = [x for x in identities if x["event"].get("status") == "success"]
    python_events = [e for e in events if e.get("kind") == "kernel_execution_start"]
    fields = []
    for key, value in expected.items():
        tolerance = item["gold"].get("tolerance", {})
        tol = tolerance.get(key, 1e-6) if isinstance(tolerance, dict) else tolerance
        if not isinstance(tol, (int, float)):
            tol = 1e-6
        possible = [x for x in calls if any(matches(a, value, tol) for a in _named_values(x["event"].get("output"), key))]
        good = [x for x in possible if x["identity_correct"] is True]
        if good:
            decision = True
        elif any(x["identity_correct"] is None for x in possible) or python_events:
            decision = None
        else:
            decision = False
        fields.append({"field": key, "correct_on_requested_object": decision,
                       "evidence_event_indices": [x["event_index"] for x in good],
                       "candidate_event_indices": [x["event_index"] for x in possible], "tolerance": tol})
    attempted = sum(e.get("kind") == "tool_request" for e in events)
    # Interrupted HTTP requests are attempts even when no return was archived.
    attempted = max(attempted, int(result.get("attempted_steps", 0)))
    answer = final_response_text(result)
    has_answer = isinstance(answer, (str, dict, list)) and bool(answer)
    reject = item["gold"]["expected_behavior"] == "reject"
    labels = {"identity_correct": True if fields and all(f["correct_on_requested_object"] is True for f in fields) else None,
              "scientific_output_correct": tri_and(f["correct_on_requested_object"] for f in fields),
              "final_answer_correct": None if has_answer else False,
              "rejection_response_correct": None if has_answer else False}
    if reject:
        labels["identity_correct"] = None
        labels["scientific_output_correct"] = None
    review_source = None
    if review:
        if review.get("evidence_sha256") != evidence_digest(item, row, events):
            raise ValueError("Semantic review does not match this exact archived run")
        if not review.get("reviewer_id") or review.get("reviewer_kind") not in {"human", "assistant"}:
            raise ValueError("Review must identify reviewer and whether human or assistant")
        if not review.get("reviewed_at") or not review.get("reason") or not review.get("evidence_locations"):
            raise ValueError("Review requires timestamp, reason and evidence locations")
        for key, value in review.get("labels", {}).items():
            if key not in REVIEW_FIELDS or (value is not None and type(value) is not bool):
                raise ValueError("Invalid semantic review label")
            if labels[key] is not None and value is not None and labels[key] != value:
                raise ValueError("Review conflicts with a determinate automatic decision")
            if value is not None:
                labels[key] = value
        review_source = review["reviewer_kind"] + ":" + review["reviewer_id"]
    strict = tri_and((attempted == 0, labels["rejection_response_correct"])) if reject else None
    primary = strict if reject else tri_and((labels["identity_correct"], labels["scientific_output_correct"], labels["final_answer_correct"]))
    pending = [k for k in (("rejection_response_correct",) if reject else REVIEW_FIELDS[:3]) if labels[k] is None]
    return {"scorer_version": SCORER_VERSION, "scientific_task_success": primary,
            "strict_pre_execution_rejection": strict, **labels, "attempted_scientific_calls": attempted,
            "native_completed": (result.get("native_execution_summary") or {}).get("completed"),
            "native_status": result.get("final_status"), "field_checks": fields,
            "identity_checks": [{k: v for k, v in x.items() if k != "event"} for x in identities],
            "python_execution_event_indices": [e["event_index"] for e in python_events],
            "pending_review_fields": pending, "review_source": review_source,
            "evidence_sha256": evidence_digest(item, row, events)}


def evidence_digest(item, row, events):
    return digest({"item": item, "run": row, "events": events})


def review_packet(item, row, events, files, score):
    return {"review_id": "review_"+score["evidence_sha256"][:20],
            "evidence_sha256": score["evidence_sha256"], "question": item["question_zh"],
            "input": {k: v for k, v in item["input"].items() if k not in {"note", "fixture"}},
            "source_objects": reference_objects(item, files),
            "rubric": {"expected_behavior": item["gold"]["expected_behavior"],
                       "expected_output": output_contract(item, reference_objects(item, files)),
                       "tolerance": item["gold"].get("tolerance", {}),
                       "required_claims": item["gold"].get("required_claims", []),
                       "forbidden_claims": item["gold"].get("forbidden_claims", [])},
            "final_response": final_response_text(row["result"]),
            "evidence": [{k: v for k, v in e.items() if k not in {"run_key", "recorded_at"}}
                         for e in events if e.get("kind") in {"tool_request", "tool_call", "input_read", "kernel_file_open",
                                                              "kernel_execution_start", "kernel_execution_end",
                                                              "artifact_pool_resolve_step_inputs",
                                                              "artifact_pool_register_tool_outputs"}],
            "automatic_score": {k: v for k, v in score.items() if k not in {"native_status", "native_completed"}}}
