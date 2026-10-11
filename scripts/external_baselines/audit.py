"""Passive, offline Dev diagnostics. Never fed back to the agent.

These checks are not a frozen semantic final-answer scorer. They distinguish
tool execution, identity evidence and response behavior without assuming that
an agent's own 'answered' flag proves scientific success.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def normalized_sequence(sequence: str) -> str:
    return "".join(sequence.split()).upper()


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(normalized_sequence(sequence).encode("utf-8")).hexdigest()


def source_index(item: dict, files: list[str]) -> list[dict]:
    from Bio import SeqIO
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    sources = []
    if isinstance(item.get("input", {}).get("sequence"), str):
        seq = item["input"]["sequence"]
        sources.append({"source": "request_sequence", "sequence_sha256": sequence_hash(seq)})
    for filename in files:
        path = Path(filename)
        if path.suffix.lower() in {".fasta", ".fa", ".faa"}:
            with path.open(encoding="utf-8-sig") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    sources.append({"source": filename, "record_id": record.id,
                                    "sequence_sha256": sequence_hash(str(record.seq))})
        elif path.suffix.lower() == ".pdb":
            structure = PDBParser(QUIET=True).get_structure("input", filename)
            model = next(structure.get_models(), None)
            for chain in model or []:
                residues = [r for r in chain if r.id[0] == " "]
                seq = "".join(seq1(r.resname, custom_map={"MSE": "M"}) for r in residues)
                if seq:
                    sources.append({"source": filename, "chain_id": chain.id,
                                    "sequence_sha256": sequence_hash(seq)})
    return sources


def identity_checks(item: dict, events: list[dict], sources: list[dict]) -> list[dict]:
    inp = item.get("input", {})
    record = inp.get("record_id")
    chain = inp.get("chain_id")
    expected = [s for s in sources if (not record or s.get("record_id") == record)
                and (not chain or s.get("chain_id") == chain)]
    checks = []
    for event in events:
        if event.get("kind") != "tool_call":
            continue
        args = event.get("resolved_arguments") or {}
        seq = args.get("sequence")
        if isinstance(seq, str):
            actual = sequence_hash(seq)
            matches = [s for s in sources if s["sequence_sha256"] == actual]
            target_matches = [s for s in expected if s["sequence_sha256"] == actual]
            if record or chain or inp.get("input_type") == "amino_acid_sequence":
                valid = bool(target_matches) if expected else None
                if valid and len(matches) > 1:
                    valid = None  # An identical value alone cannot establish lineage.
            else:
                valid = None
            checks.append({"call_id": event["call_id"], "tool_name": event["tool_name"],
                           "sequence_sha256": actual, "matching_sources": matches,
                           "target_value_matches": valid,
                           "reference_bindings": event.get("reference_bindings", [])})
        for key in ("record_id", "chain_id", "record_ids"):
            target = inp.get(key)
            if target is not None and key in args:
                checks.append({"call_id": event["call_id"], "constraint": key,
                               "requested": target, "actual": args[key],
                               "explicit_identifier_matches": args[key] == target})
    return checks


def _named_values(obj: Any, key: str) -> list[Any]:
    values = []
    if isinstance(obj, dict):
        if key in obj:
            values.append(obj[key])
        for child in obj.values():
            values.extend(_named_values(child, key))
    elif isinstance(obj, list):
        for child in obj:
            values.extend(_named_values(child, key))
    return values


def _matches(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and _matches(actual[k], v, tolerance) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _matches(a, b, tolerance) for a, b in zip(actual, expected))
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and abs(actual - expected) <= tolerance
    return actual == expected


def audit_run(item: dict, result: dict, events: list[dict], sources: list[dict]) -> dict:
    calls = [e for e in events if e.get("kind") == "tool_call"]
    success = [e for e in calls if e.get("status") == "success"]
    gold = item["gold"]
    fields = []
    for key, expected in gold.get("expected_output", {}).items():
        tol = gold.get("tolerance", {})
        tolerance = tol.get(key, 1e-6) if isinstance(tol, dict) else tol
        if not isinstance(tolerance, (int, float)):
            tolerance = 1e-6
        candidates = [(e["call_id"], value) for e in success
                      for value in _named_values(e["output"], key)]
        matched = [call_id for call_id, actual in candidates if _matches(actual, expected, tolerance)]
        fields.append({"field": key, "matched": bool(matched),
                       "evidence_call_ids": matched,
                       "candidate_count": len(candidates)})
    identity = identity_checks(item, events, sources)
    reject = gold["expected_behavior"] == "reject"
    return {
        "expected_behavior": gold["expected_behavior"],
        "attempted_steps": result["attempted_steps"],
        "successful_steps": len(success),
        "failed_http_steps": sum(e["http_requested"] and e["status"] != "success" for e in calls),
        "interface_error_actions": sum(e["status"] == "interface_error" for e in calls),
        "strict_pre_execution_rejection": (
            False if result["attempted_steps"] > 0 else
            None if result.get("final_response_status_unclassified") else
            result["final_status"] == "rejected") if reject else None,
        "gold_output_fields": fields,
        "structured_tool_outputs_match": all(f["matched"] for f in fields) if fields and not reject else None,
        "identity_checks": identity,
        "identity_mismatch_observed": any(c.get("target_value_matches") is False
                                          or c.get("explicit_identifier_matches") is False for c in identity),
        "scientific_task_success": None,
        "final_answer_correct": None,
        "scoring_status": "dev_diagnostics_only_semantic_review_required",
    }
