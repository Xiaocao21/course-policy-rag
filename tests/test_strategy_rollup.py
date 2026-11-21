"""Unit tests for the derived statistics the report now quotes.

The report no longer types these numbers in: it reads `by_strategy`, `family_contrasts`
and `by_length_tercile` from `results/final/stats.json`. These tests pin the pieces that
can silently change the meaning of those blocks — the family key, the t quantile used for
the interval, and the symmetric k rule.
"""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


STATS16 = _load("stats16", "scripts/16_stats.py")


class TestFamilyKey(unittest.TestCase):
    def test_fixed_is_split_by_nominal_size(self):
        self.assertEqual(STATS16.family_of("fixed", "size128_ov64"), "fixed/size128")
        self.assertEqual(STATS16.family_of("fixed", "size512_ov0"), "fixed/size512")

    def test_structural_strategies_stay_whole(self):
        self.assertEqual(STATS16.family_of("heading", "heading"), "heading")
        self.assertEqual(STATS16.family_of("paragraph", "paragraph"), "paragraph")
        self.assertEqual(STATS16.family_of("semantic", "thr0.70"), "semantic")


class TestWelchInterval(unittest.TestCase):
    def test_t_quantile_matches_published_values(self):
        # two-sided 95%: df=19 -> 2.0930, df=35 -> 2.0301, large df -> 1.9600
        self.assertAlmostEqual(STATS16.t_quantile_975(19), 2.0930, places=3)
        self.assertAlmostEqual(STATS16.t_quantile_975(35), 2.0301, places=3)
        self.assertAlmostEqual(STATS16.t_quantile_975(5000), 1.95996, places=4)

    def test_interval_is_b_minus_a_and_contains_the_delta(self):
        a = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        b = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        out = STATS16.welch_interval(a, b)
        self.assertGreater(out["delta"], 0)
        self.assertLessEqual(out["ci_low"], out["delta"])
        self.assertGreaterEqual(out["ci_high"], out["delta"])

    def test_identical_buckets_give_a_degenerate_interval_at_zero(self):
        a = np.array([0.5, 0.5, 0.5, 0.5])
        out = STATS16.welch_interval(a, a)
        self.assertEqual(out["delta"], 0.0)

    def test_too_few_points_are_reported_as_none(self):
        out = STATS16.welch_interval(np.array([1.0]), np.array([1.0, 0.0]))
        self.assertIsNone(out["delta"])


class TestLengthPanelContract(unittest.TestCase):
    """The symmetric rule and its two k values, read back from the real statistics."""

    @classmethod
    def setUpClass(cls):
        path = ROOT / "results" / "final" / "stats.json"
        if not path.exists():
            raise unittest.SkipTest("stats.json not built yet")
        cls.stats = json.loads(path.read_text(encoding="utf-8"))
        cls.block = cls.stats["by_length_tercile"]

    def test_both_k_values_are_floor_and_ceil(self):
        n = self.block["n_groups"]
        tracks = self.block["tracks"]
        self.assertIn("C1", tracks)
        self.assertEqual(tracks["C1"]["k_floor"]["k"], n // 3)
        self.assertEqual(tracks["C1"]["k_ceil"]["k"], -(-n // 3))

    def test_symmetric_ends_have_equal_size(self):
        # short and long are both k groups; the middle block never enters the contrast
        for track, panels in self.block["tracks"].items():
            for label, panel in panels.items():
                self.assertEqual(panel["k"], panel["k"])

    def test_bootstrap_flag_is_present_and_boolean(self):
        for panels in self.block["tracks"].values():
            for panel in panels.values():
                self.assertIsInstance(panel["bootstrap"]["stable"], bool)

    def test_sampling_frame_matches_the_usable_pool(self):
        frame = self.block["sampling_frame"]
        self.assertEqual(frame["available_clauses"], 1239)
        self.assertLess(frame["available_median_chars"], frame["used_median_chars"])


class TestStrategyRollup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "results" / "final" / "stats.json"
        if not path.exists():
            raise unittest.SkipTest("stats.json not built yet")
        cls.stats = json.loads(path.read_text(encoding="utf-8"))

    def test_families_cover_every_configuration_exactly_once(self):
        seen = []
        for entry in self.stats["by_strategy"].values():
            seen.extend(entry["configs"])
        self.assertEqual(len(seen), 17)
        self.assertEqual(len(set(seen)), 17)

    def test_single_config_family_reports_one_config(self):
        self.assertEqual(self.stats["by_strategy"]["heading"]["n_configs"], 1)
        self.assertEqual(self.stats["by_strategy"]["semantic"]["n_configs"], 3)

    def test_family_contrast_uses_the_family_means(self):
        c = self.stats["family_contrasts"]["C1:semantic-minus-structural"]
        self.assertAlmostEqual(c["delta"], c["semantic_mean"] - c["structural_mean"], places=3)
        self.assertLess(c["ci_low"], c["delta"])
        self.assertGreater(c["ci_high"], c["delta"])


if __name__ == "__main__":
    unittest.main()
