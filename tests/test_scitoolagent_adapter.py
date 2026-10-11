"""Official source provenance and literal-interface boundary tests (no API)."""
import unittest
from pathlib import Path

from scripts.external_baselines.common import tool_catalog
from scripts.external_baselines.scitoolagent import graph_rows, parse_literal_arguments
from scripts.external_baselines.scitoolagent_upstream import SOURCE, load_core, upstream_identity


class OfficialSourceTests(unittest.TestCase):
    def test_original_bodies_and_pinned_commit(self):
        self.assertEqual(upstream_identity()["commit"], "ac1cf19fcef84e69f149db6da424afe4e9b2f11f")
        retriever, ns = load_core()
        self.assertEqual(Path(retriever._retrieve.__code__.co_filename), SOURCE / "scripts/custom_kg_retrievers.py")
        for name in ("run_SciToolAgent", "execute_tool"):
            self.assertEqual(Path(ns[name].__code__.co_filename), SOURCE / "scripts/retrieve_tool_info.py")
        self.assertEqual(ns["extract_tool_path"]("Plan Chain:\n['compute_gravy', 'compute_pI_mw']"),
                         ["compute_gravy", "compute_pI_mw"])

    def test_graph_uses_only_public_topology(self):
        _, ns = load_core()
        import pandas as pd
        triples = ns["create_triplets"](pd.DataFrame(graph_rows()))
        self.assertIn(("select_pdb_chain", "outputs", "amino acid sequence"), triples)
        self.assertIn(("amino acid sequence", "is the input of", "compute_gravy"), triples)
        self.assertEqual(sum(t[1] == "has the functionality that" for t in triples), 20)
        self.assertFalse(any("repair_policy" in str(t) or "gold" in str(t) for t in triples))

    def test_literal_arguments_do_not_select_or_convert(self):
        catalog = {t["name"]: t for t in tool_catalog()}
        self.assertEqual(parse_literal_arguments('ACDE', catalog["compute_gravy"]), {"sequence": "ACDE"})
        self.assertEqual(parse_literal_arguments('{"file_path":"x","chain_id":"B"}', catalog["select_pdb_chain"]),
                         {"file_path": "x", "chain_id": "B"})
        with self.assertRaises(ValueError):
            parse_literal_arguments('x', catalog["select_pdb_chain"])
        with self.assertRaises(ValueError):
            parse_literal_arguments('{"sequence":{"$ref":"call_0001#/sequence"}}', catalog["compute_gravy"])


if __name__ == "__main__":
    unittest.main()
