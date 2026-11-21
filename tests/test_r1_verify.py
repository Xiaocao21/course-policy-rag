"""R1 comparison rules: exclusions are declared, not chosen after seeing a difference."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import verify  # noqa: E402


class TestR1Comparison(unittest.TestCase):
    def test_only_created_at_is_excluded(self):
        self.assertEqual(verify.EXCLUDE["results/test2/OUTCOME.json"], {"created_at"})

    def test_diff_respects_declared_exclusion(self):
        a = {"primary": {"delta": 0.8}, "created_at": "2026-11-13T02:01:39+08:00"}
        b = {"primary": {"delta": 0.8}, "created_at": "2026-11-21T18:00:00+08:00"}
        self.assertTrue(verify.diff(a, b))
        self.assertEqual(verify.diff(a, b, skip={"created_at"}), [])

    def test_diff_reports_nested_change(self):
        a = {"tracks": {"C3": {"complete@10": {"mean": 0.405}}}}
        b = {"tracks": {"C3": {"complete@10": {"mean": 0.406}}}}
        diffs = verify.diff(a, b)
        self.assertEqual(len(diffs), 1)
        self.assertIn("/tracks/C3/complete@10/mean", diffs[0])

    def test_tracks_are_the_four_reported(self):
        self.assertEqual(verify.TRACKS, ("main", "C1", "C2", "C3"))


if __name__ == "__main__":
    unittest.main()
