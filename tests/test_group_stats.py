"""Group-level statistics: units, correction, and determinism.

The statistics are only honest if (a) the resampling unit is the group rather than the
question or the question-by-configuration cell, (b) the correction is monotone and
conservative, and (c) the same seed reproduces the same numbers.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def load_stats_module():
    spec = importlib.util.spec_from_file_location("stats16", ROOT / "scripts" / "16_stats.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


STATS = load_stats_module()


class TestGroupUnit(unittest.TestCase):
    def test_holm_is_monotone_and_conservative(self):
        raw = [0.01, 0.04, 0.03]
        adj = STATS.holm(raw)
        self.assertEqual(len(adj), len(raw))
        # adjusted >= raw, and ordering is preserved
        self.assertTrue(all(a >= r - 1e-12 for a, r in zip(adj, raw)))
        order = sorted(range(len(raw)), key=lambda i: raw[i])
        self.assertTrue(all(adj[order[i]] <= adj[order[i + 1]] + 1e-12
                            for i in range(len(order) - 1)))
        self.assertTrue(all(a <= 1.0 for a in adj))

    def test_bootstrap_p_is_bounded(self):
        deltas = np.array([0.1, 0.2, -0.05, 0.3])
        p = STATS.bootstrap_p(deltas)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_spearman_identity_and_reversal(self):
        a = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(STATS.spearman(a, a), 1.0, places=6)
        self.assertAlmostEqual(STATS.spearman(a, a[::-1]), -1.0, places=6)

    def test_wilcoxon_needs_enough_pairs(self):
        self.assertTrue(np.isnan(STATS.wilcoxon(np.array([0.1, -0.1, 0.2]))))
        self.assertFalse(np.isnan(STATS.wilcoxon(np.array([0.1] * 8 + [-0.1] * 2))))


class TestStatsArtifacts(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "results" / "final" / "stats.json"
        if not self.path.exists():
            self.skipTest("stats.json not produced yet")
        self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def test_units_are_groups(self):
        self.assertIn("group_id", self.data["unit"])
        self.assertEqual(self.data["groups"], 59)

    def test_invariance_reported_not_tested(self):
        for key in ("C1-main", "C2-main"):
            entry = self.data["invariance"][key]
            self.assertTrue(entry["passed"])
            self.assertIn("no p-value", entry["note"])

    def test_primary_comparison_present(self):
        self.assertEqual(self.data["primary_comparison"], "C3-C1 on complete@10")
        self.assertIn("p_holm", self.data["contrasts"]["C3-C1"])


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_bytes(self):
        stats = ROOT / "results" / "final" / "stats.json"
        if not stats.exists():
            self.skipTest("stats.json not produced yet")
        before = stats.read_bytes()
        out = subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"),
                              str(ROOT / "scripts" / "16_stats.py"), "--split", "test"],
                             capture_output=True, text=True, timeout=900, cwd=str(ROOT))
        self.assertEqual(out.returncode, 0, out.stderr[-800:])
        self.assertEqual(before, stats.read_bytes(),
                         "stats.json changed on an identical re-run with the same seed")


if __name__ == "__main__":
    unittest.main()
