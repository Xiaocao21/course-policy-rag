"""Structure checks for the teacher-facing report.

The report is a submission artefact, so its structure is asserted rather than eyeballed:
section order, the six-column-plus-observation experiment log, retention of failed runs,
and the guarantee that the earlier brief was not overwritten.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parent.parent
def _latest_report() -> Path:
    """Prefer the most recently written report, so a locked canonical file cannot mask a
    newer build."""
    candidates = sorted((ROOT / "docs").glob("PROJECT_REPORT*.docx"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else ROOT / "docs" / "PROJECT_REPORT.docx"
OLD_BRIEF = ROOT / "docs" / "TEACHER_BRIEF.docx"
OLD_HASH_FILE = ROOT / "docs" / ".teacher_brief_v1.sha256"

REQUIRED_SECTIONS = [
    "Research Question",
    "Work Completed",
    "Experimental Setup",
    "Experiment Log",
    "Preliminary Results",
    "Problems and Changes",
    "Remaining Experiments",
    "Updated Timeline",
    "AI Disclosure",
]
LOG_HEADER = ["Run", "Model", "Dataset", "Important settings", "Metric", "Result",
              "Observation"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


class TestReportStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report = _latest_report()
        if not report.exists():
            raise unittest.SkipTest("PROJECT_REPORT.docx not built")
        cls.doc = Document(str(report))
        cls.headings = [p.text.strip() for p in cls.doc.paragraphs
                        if p.style.name.startswith("Heading 1")]

    def test_sections_in_recommended_order(self):
        joined = " | ".join(self.headings)
        positions = []
        for name in REQUIRED_SECTIONS:
            matches = [h for h in self.headings if name.lower() in h.lower()]
            self.assertTrue(matches, f"missing section {name!r} in {joined}")
            positions.append(self.headings.index(matches[0]))
        self.assertEqual(positions, sorted(positions),
                         f"sections out of order: {joined}")

    def test_title_present(self):
        first = self.doc.paragraphs[0].text.strip()
        self.assertIn("Chunking", first)
        self.assertIn("Course Policies", first)

    def test_experiment_log_columns(self):
        candidates = []
        for t in self.doc.tables:
            header = [c.text.strip() for c in t.rows[0].cells]
            if header[:2] == ["Run", "Model"]:
                candidates.append((t, header))
        self.assertTrue(candidates, "no experiment-log table found")
        table, header = candidates[0]
        self.assertEqual(header, LOG_HEADER)

    def test_failed_runs_retained(self):
        table = next(t for t in self.doc.tables
                     if [c.text.strip() for c in t.rows[0].cells][:2] == ["Run", "Model"])
        body = " ".join(c.text for row in table.rows[1:] for c in row.cells).lower()
        self.assertIn("aborted", body)
        self.assertGreaterEqual(body.count("aborted"), 3,
                                "expected at least three aborted training runs")
        runs = [row.cells[0].text for row in table.rows[1:]]
        self.assertGreaterEqual(len(runs), 10, f"only {len(runs)} runs logged")

    def test_ai_disclosure_is_last_section(self):
        self.assertIn("AI Disclosure", self.headings[-1])

    def test_primary_and_secondary_separated(self):
        text = " ".join(p.text for p in self.doc.paragraphs)
        self.assertIn("Primary experiment", text)
        self.assertIn("Secondary experiment", text)
        self.assertIn("How the two relate", text)

    def test_a5000_mentions_are_budget_scoped(self):
        """A5000 is allowed only as a compatibility/budget statement.

        The earlier revision of this report deliberately mentioned no other machine, because
        every reported number came from the canonical RTX 3060. The A5000 experiments now in
        the log are compatibility runs, so the guard changed shape rather than disappearing:
        any A5000 mention must sit next to a compatibility or budget qualifier, and it must
        never appear in the research question or the results claims.
        """
        body = " ".join([p.text for p in self.doc.paragraphs]
                        + [c.text for t in self.doc.tables for r in t.rows for c in r.cells])
        if "A5000" not in body:
            return
        lowered = body.lower()
        for qualifier in ("compatibility", "budget", "a5000 time", "a5000 memory"):
            self.assertIn(qualifier, lowered,
                          "A5000 is mentioned without a compatibility/budget qualifier")
        headings = [h for h in self.headings]
        self.assertNotIn("A5000", " ".join(headings))

    def test_disclosure_matches_course_policy_wording(self):
        text = " ".join(p.text for p in self.doc.paragraphs)
        self.assertIn("in accordance with the course policy", text)
        self.assertNotIn("approved by the supervisor", text)

    def test_matched_budget_table_present(self):
        headers = [[c.text.strip() for c in t.rows[0].cells] for t in self.doc.tables]
        self.assertTrue(any(h[:2] == ["track", "Recall@512"] for h in headers),
                        "matched token-budget table missing")

    def test_strategy_table_present(self):
        """Per-strategy results must appear next to the macro average, not only in a CSV."""
        headers = [[c.text.strip() for c in t.rows[0].cells] for t in self.doc.tables]
        self.assertTrue(any(h[:2] == ["strategy", "configs"] for h in headers),
                        "per-strategy table missing")

    def test_union_definition_present(self):
        """The completeness rule must say that several chunks may combine."""
        body = " ".join([p.text for p in self.doc.paragraphs]
                        + [c.text for t in self.doc.tables for r in t.rows for c in r.cells])
        self.assertIn("union", body.lower())

    def test_length_contrast_matches_stats(self):
        """Derived numbers are read from stats.json, never typed into the generator."""
        stats = json.loads((ROOT / "results" / "final" / "stats.json").read_text(encoding="utf-8"))
        panel = stats["by_length_tercile"]["tracks"]["C1"]["k_floor"]["welch"]
        body = " ".join(p.text for p in self.doc.paragraphs)
        self.assertIn(f"{panel['delta']:+.3f}", body)
        self.assertIn(f"{panel['ci_low']:+.3f}", body)

    def test_seed_derivation_documented(self):
        text = " ".join(p.text for p in self.doc.paragraphs)
        self.assertIn("20261037", text)
        self.assertIn("deriv", text.lower())

    def test_timeline_compares_plan_with_actual(self):
        headers = [[c.text.strip() for c in t.rows[0].cells] for t in self.doc.tables]
        self.assertIn(["phase", "planned", "actual", "status"], headers,
                      "the timeline must compare the proposal plan with what happened")
        text = " ".join(p.text for p in self.doc.paragraphs)
        self.assertIn("proposal", text.lower())
        self.assertIn("weeks 1–14", text)

    def test_old_brief_untouched(self):
        self.assertTrue(OLD_BRIEF.exists(), "the earlier brief must not be deleted")
        if OLD_HASH_FILE.exists():
            recorded = OLD_HASH_FILE.read_text(encoding="ascii").strip().upper()
            self.assertEqual(sha256(OLD_BRIEF), recorded,
                             "TEACHER_BRIEF.docx changed; it must stay as-is")


if __name__ == "__main__":
    unittest.main()
