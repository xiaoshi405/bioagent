"""Offline object-identity probe with the real FASTA and GRAVY tools.

This is a controlled mechanism check, not a Test-150 identity-rate estimate.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

from scripts.initial_artifacts import build_initial_artifacts
from scripts.structured_plan import prepare_structured_plan
from scripts.typed_executor import execute_structured_plan
from scripts.type_checker import AMINO_ACID_SEQUENCE, FASTA_RECORD_IDENTIFIER, MULTI_PROTEIN_FASTA_FILE
from ToolsAgent.ToolsFuns.Biology.biologyCommon import compute_gravy
from ToolsAgent.ToolsFuns.Biology.fasta_collection import parse_multi_protein_fasta, select_fasta_record


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "fasta" / "multi_protein.fasta"
COLLECTION_TYPE = "protein sequence collection"


async def run_case(record_id: str, *, corrupt_source: bool = False) -> dict:
    initial = build_initial_artifacts(
        f"Compute GRAVY for record_id={record_id}", [str(FIXTURE)]
    )
    assert initial["status"] == "success"
    assert initial["artifacts"][FASTA_RECORD_IDENTIFIER] == record_id
    prepared = prepare_structured_plan(
        ["compute_gravy"], initial["available_input_types"], enable_repair=True
    )
    assert prepared["valid"] is True
    assert prepared["plan_after"] == [
        "parse_multi_protein_fasta", "select_fasta_record", "compute_gravy"
    ]
    plan = copy.deepcopy(prepared["structured_plan"])
    if corrupt_source:
        plan[1]["input_from"][COLLECTION_TYPE] = "nonexistent_source"

    observed = []

    async def runner(**kwargs):
        tool = kwargs["tool_name"]
        values = kwargs["typed_inputs"]
        observed.append(tool)
        if tool == "parse_multi_protein_fasta":
            return parse_multi_protein_fasta(values[MULTI_PROTEIN_FASTA_FILE])
        if tool == "select_fasta_record":
            return select_fasta_record(values[COLLECTION_TYPE], values[FASTA_RECORD_IDENTIFIER])
        if tool == "compute_gravy":
            return compute_gravy(values[AMINO_ACID_SEQUENCE])
        raise AssertionError(tool)

    result = await execute_structured_plan(plan, initial["artifacts"], runner=runner)
    if corrupt_source:
        assert result["status"] == "error"
        assert "select_fasta_record" not in observed
        assert "compute_gravy" not in observed
        assert not result["completed"]
        return {
            "requested_record": record_id,
            "source_reference": "nonexistent_source",
            "completed": False,
            "called_tools": observed,
            "downstream_tool_called": False,
        }

    parsed = parse_multi_protein_fasta(str(FIXTURE))
    assert parsed["status"] == "success"
    oracle = select_fasta_record(parsed["collection"], record_id)
    assert oracle["status"] == "success"
    assert result["status"] == "success" and result["completed"]
    selected = result["tool_results"][1]["result"]["sequence"]
    actual = result["tool_results"][2]["result"]["gravy"]
    expected = compute_gravy(oracle["sequence"])["gravy"]
    assert selected == oracle["sequence"] and actual == expected
    assert result["executed_steps"][2]["input_artifact_ids"][AMINO_ACID_SEQUENCE] == \
        result["executed_steps"][1]["output_artifact_ids"][0]
    return {
        "requested_record": record_id,
        "selected_record_matches_gold": True,
        "sequence_length": len(selected),
        "gravy": actual,
        "gravy_matches_gold": True,
        "downstream_artifact_matches_selector_output": True,
        "completed": True,
    }


async def main() -> None:
    results = [await run_case(record) for record in ("protein_a", "protein_b")]
    assert results[0]["gravy"] != results[1]["gravy"]
    results.append(await run_case("protein_b", corrupt_source=True))
    print(json.dumps({"fixture": str(FIXTURE.relative_to(ROOT)), "cases": results}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
