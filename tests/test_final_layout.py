"""Layout and completeness of the formal four-track artefacts."""
from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "results" / "final"
TRACKS = ("main", "C1", "C2", "C3")


class TestFinalLayout(unittest.TestCase):
    def setUp(self):
        if not FINAL.exists():
            self.skipTest("formal run not executed yet")

    def test_all_configs_present_per_track(self):
        summary = json.loads((ROOT / "reports" / "index_summary.json").read_text(encoding="utf-8"))
        expected = len(summary["configs"])
        for track in TRACKS:
            files = list((FINAL / track).glob("*.json"))
            self.assertEqual(len(files), expected, f"{track}: {len(files)} != {expected}")

    def test_summaries_have_budget_columns_filled(self):
        for track in TRACKS:
            path = FINAL / f"summary_{track}.csv"
            self.assertTrue(path.exists(), path)
            with path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(rows)
            for row in rows:
                for col in ("tok512_recall", "tok1024_recall", "tok2048_recall"):
                    self.assertNotEqual(row[col], "", f"{track} {row['config']} {col} empty")

    def test_attempts_and_final_run_recorded(self):
        self.assertTrue((FINAL / "FINAL_RUN.json").exists())
        attempts = FINAL / "ATTEMPTS.json"
        if attempts.exists():
            data = json.loads(attempts.read_text(encoding="utf-8"))
            self.assertIn("attempts", data)

    def test_pilot_archive_is_marked(self):
        readme = ROOT / "results" / "pilot" / "README.md"
        self.assertTrue(readme.exists())
        self.assertIn("NOT the final result", readme.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
