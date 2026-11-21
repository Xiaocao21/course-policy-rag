"""Tests for the report data layer and the section-path normaliser.

These guard the property the supervisor-facing report depends on: every figure is
derived from the repository and the derived set is internally consistent.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


extract = load_module("extract_clean_rc", "02_extract_clean.py")
import report_content  # noqa: E402


class TestSectionPathCleaning(unittest.TestCase):
    """section_path is display-only, but it leaked raw prose into the reports."""

    def test_long_lead_in_sentence_is_dropped(self):
        parts = ["Academic Conduct Code",
                 "Boston University's Academic Conduct Code (the Code) is designed to assist "
                 "in the development of a supportive environment for learning and scholarship"]
        self.assertNotIn("designed", extract.clean_section_path(parts))

    def test_depth_is_capped(self):
        parts = ["A", "B", "C", "D", "E"]
        self.assertEqual(extract.clean_section_path(parts, max_depth=3), "C > D > E")

    def test_trailing_punctuation_is_stripped(self):
        self.assertEqual(extract.clean_section_path(["H. Title."]), "H. Title")

    def test_duplicate_levels_collapse(self):
        self.assertEqual(extract.clean_section_path(["Appeals", "Appeals"]), "Appeals")

    def test_empty_and_whitespace_parts_are_ignored(self):
        self.assertEqual(extract.clean_section_path(["", "   ", "Real Heading"]), "Real Heading")

    def test_overlong_part_is_dropped_not_truncated(self):
        long_part = "x" * 200
        self.assertEqual(extract.clean_section_path([long_part, "Short"]), "Short")


class TestReportContent(unittest.TestCase):
    """The report must never print a figure the repository contradicts."""

    def test_collect_on_real_repo_is_consistent(self):
        data = report_content.collect(ROOT)
        self.assertTrue(data["consistency"]["ok"],
                        f"inconsistencies: {data['consistency']['problems']}")

    def test_corpus_membership_matches_curation(self):
        data = report_content.collect(ROOT)
        doc_ids = {d["doc_id"] for d in data["documents"]}
        self.assertEqual(len(doc_ids), data["corpus"]["documents"])
        # every corpus document must contribute clause statistics
        for d in data["documents"]:
            self.assertGreaterEqual(d["clauses"], 1, f"{d['doc_id']} has no clauses")

    def test_tier_shares_respect_the_cap(self):
        data = report_content.collect(ROOT)
        for tier, share in data["corpus"]["tier_shares"].items():
            self.assertLessEqual(share, 0.4001, f"tier {tier} over the 40% cap")

    def test_all_acceptance_gates_pass(self):
        data = report_content.collect(ROOT)
        failed = [g[0] for g in data["gates"] if not g[3]]
        self.assertEqual(failed, [], f"failed gates: {failed}")

    def test_counts_come_from_clause_stats_not_raw_hits(self):
        """Regression: raw page-level regex hits were once reported as clause counts."""
        import csv
        data = report_content.collect(ROOT)
        with (ROOT / "reports" / "clause_stats.csv").open(encoding="utf-8") as f:
            stats = list(csv.DictReader(f))
        by_doc = {r["doc_id"]: r for r in stats}
        corpus_ids = {d["doc_id"] for d in data["documents"]}
        expected = sum(int(by_doc[d]["clauses_total"]) for d in corpus_ids if d in by_doc)
        self.assertEqual(data["capability"]["clauses_total"], expected)

    def test_consistency_flags_a_missing_corpus(self):
        with tempfile.TemporaryDirectory() as td:
            problems = report_content.consistency_check(
                set(), {}, {}, {"span_errors": 0}, Path(td) / "nope.jsonl", chunk_count=0)
            self.assertFalse(problems["ok"])
            self.assertTrue(any("corpus file missing" in p for p in problems["problems"]))

    def test_consistency_flags_span_errors_and_tier_overflow(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.jsonl"
            p.write_text("", encoding="utf-8")
            res = report_content.consistency_check(
                {"d1"}, {"d1": {}}, {"faculty": 0.9}, {"span_errors": 3}, p, chunk_count=1)
            joined = " ".join(res["problems"])
            self.assertIn("span errors", joined)
            self.assertIn("40% cap", joined)

    def test_gates_are_evaluated_not_hard_coded(self):
        data = report_content.collect(ROOT)
        names = [g[0] for g in data["gates"]]
        self.assertTrue(any("机构数" in n for n in names))
        for _name, have, need, ok in data["gates"]:
            self.assertIsInstance(ok, bool)
            self.assertTrue(have != "" or need != "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
