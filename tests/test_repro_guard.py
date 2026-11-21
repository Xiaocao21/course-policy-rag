"""Guards for the reproduction entry points: official dirs and non-comparable outputs."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import evaluate  # noqa: E402
from repro.compare import num  # noqa: E402


class TestOutDirGuard(unittest.TestCase):
    def test_rejects_official_results_dir(self):
        with self.assertRaises(SystemExit):
            evaluate.resolve_out(ROOT, "results/final_test2")

    def test_rejects_subdirectory_of_official_results(self):
        with self.assertRaises(SystemExit):
            evaluate.resolve_out(ROOT, "results/test2/repro")

    def test_accepts_repro_dir(self):
        path = evaluate.resolve_out(ROOT, "results/repro/_guard_test")
        self.assertTrue(path.name == "_guard_test")
        path.rmdir()


class TestCompareHandlesNullMetrics(unittest.TestCase):
    def test_none_becomes_zero(self):
        self.assertEqual(num(None), 0.0)
        self.assertEqual(num(0.5), 0.5)


class TestOfficialTreeIsDeclared(unittest.TestCase):
    def test_official_paths_cover_both_runs(self):
        self.assertIn("results/final", evaluate.OFFICIAL)
        self.assertIn("results/final_test2", evaluate.OFFICIAL)
        self.assertIn("models/reranker/FROZEN.json", evaluate.GUARDED)
        self.assertIn("experiments/log.jsonl", evaluate.GUARDED)


if __name__ == "__main__":
    unittest.main()
