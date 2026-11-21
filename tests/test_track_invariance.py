"""The two arithmetic identities of the four-track design.

Reranking a fixed ten-chunk set cannot change Recall@10 / Hit@10 / Complete@10, and the
first ten of a dense top-50 are the dense top-10. Both are checked question by question on
the produced artefacts rather than trusted from the implementation.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SET_METRICS = ("recall@10", "hit@10", "complete@10")


def load(track: str, root: Path) -> dict[tuple[str, str], dict]:
    path = root / f"perq_{track}.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {(r["qid"], r["config"]): r for r in rows}


class TestTrackInvariance(unittest.TestCase):
    def _check(self, root: Path):
        main = load("main", root)
        if not main:
            self.skipTest("evaluation artefacts not present")
        for track in ("C1", "C2"):
            other = load(track, root)
            self.assertTrue(other, f"missing perq_{track}.jsonl")
            worst = 0.0
            for key, row in other.items():
                ref = main.get(key)
                if ref is None:
                    continue
                for metric in SET_METRICS:
                    worst = max(worst, abs(float(row[metric]) - float(ref[metric])))
            self.assertLess(worst, 1e-9, f"{track} deviates from main by {worst}")

    def test_dev_track_invariance(self):
        self._check(ROOT / "results" / "final_dev")

    def test_test_track_invariance(self):
        self._check(ROOT / "results" / "final")

    def test_c3_may_differ(self):
        root = ROOT / "results" / "final"
        main, c3 = load("main", root), load("C3", root)
        if not main or not c3:
            self.skipTest("evaluation artefacts not present")
        diffs = [abs(float(c3[k]["complete@10"]) - float(main[k]["complete@10"]))
                 for k in c3 if k in main]
        self.assertGreater(max(diffs), 0.0,
                           "C3 should be allowed to differ on set-level metrics")


if __name__ == "__main__":
    unittest.main()
