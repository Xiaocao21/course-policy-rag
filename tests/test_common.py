"""Offline tests for the text-layer, span-arithmetic and robots logic.

These run with no network access. They cover the failure modes that would
silently corrupt gold-span annotations if they regressed.

Run:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from common import (  # noqa: E402
    RobotsCache, build_text_layers, contains_span, covered_token_fraction,
    jaccard, minhash_signature, minhash_similarity, shingles,
)

DEFAULT_LAYER_CFG = {
    "retrieval": {
        "nfkc": True, "unify_quotes": True, "unify_dashes": True, "unify_bullets": True,
        "fullwidth_to_halfwidth": True, "expand_ligatures": True,
        "strip_zero_width": True, "collapse_whitespace": True, "require_charmap": True,
    }
}


class TestTextLayers(unittest.TestCase):
    def test_charmap_is_exact_for_identical_text(self):
        raw = "Section 3.2 Late work\n\nA penalty applies."
        layers = build_text_layers(raw, DEFAULT_LAYER_CFG)
        self.assertEqual(len(layers.charmap), len(layers.retrieval))
        for i, ev in enumerate(layers.charmap):
            self.assertLess(ev, len(layers.evidence))

    def test_curly_quotes_and_dashes_do_not_break_mapping(self):
        raw = "The instructor\u2019s policy \u2014 \u201clate\u201d work \u2013 is clear."
        layers = build_text_layers(raw, DEFAULT_LAYER_CFG)
        self.assertIn("'", layers.retrieval)
        self.assertIn('"', layers.retrieval)
        self.assertNotIn("\u2014", layers.retrieval)
        # every retrieval char must map back to a real evidence index
        self.assertEqual(len(layers.charmap), len(layers.retrieval))
        self.assertTrue(all(0 <= i < len(layers.evidence) for i in layers.charmap))

    def test_ligature_expansion_records_both_chars(self):
        raw = "The \ufb01nal grade is decided by the instructor."
        layers = build_text_layers(raw, DEFAULT_LAYER_CFG)
        self.assertIn("final", layers.retrieval)
        # both 'f' and 'i' map to the single evidence char holding the ligature
        idx = layers.retrieval.index("fi")
        self.assertEqual(layers.charmap[idx], layers.charmap[idx + 1])

    def test_evidence_span_roundtrip_for_ascii_clause(self):
        raw = "Rule 1: late work loses 10 percent per day.\nRule 2: except with approval."
        layers = build_text_layers(raw, DEFAULT_LAYER_CFG)
        needle = "except with approval"
        rs = layers.retrieval.index(needle)
        ev_start, ev_end = layers.evidence_span(rs, rs + len(needle))
        self.assertEqual(layers.evidence[ev_start:ev_end], needle)

    def test_evidence_layer_is_character_faithful(self):
        raw = "Line one.\r\nLine two \u2014 with dash."
        layers = build_text_layers(raw, DEFAULT_LAYER_CFG)
        # evidence layer must be unchanged; only retrieval is normalised
        self.assertEqual(layers.evidence, raw)

    def test_zero_width_is_stripped_only_from_retrieval(self):
        raw = "Integrity\u200b policy applies."
        layers = build_text_layers(raw, DEFAULT_LAYER_CFG)
        self.assertIn("\u200b", layers.evidence)
        self.assertNotIn("\u200b", layers.retrieval)

    def test_whitespace_collapse_keeps_charmap_consistent(self):
        raw = "A    B\n\n\nC"
        layers = build_text_layers(raw, DEFAULT_LAYER_CFG)
        self.assertEqual(layers.retrieval, "A B C")
        self.assertEqual(len(layers.charmap), 5)


class TestContainment(unittest.TestCase):
    def test_full_clause_hit_requires_entire_span(self):
        text = "0123456789"
        self.assertTrue(contains_span(text, [(2, 6)], (2, 6)))
        self.assertTrue(contains_span(text, [(0, 6)], (2, 6)))
        self.assertTrue(contains_span(text, [(2, 4), (4, 6)], (2, 6)))

    def test_partial_overlap_is_not_a_hit(self):
        text = "0123456789"
        # covers only 2..5 of a 2..6 span
        self.assertFalse(contains_span(text, [(2, 5)], (2, 6)))
        self.assertFalse(contains_span(text, [(4, 8)], (2, 6)))

    def test_disjoint_ranges_are_not_a_hit(self):
        text = "0123456789"
        self.assertFalse(contains_span(text, [(2, 3), (3, 4)], (2, 6)))

    def test_partial_span_coverage_fraction(self):
        span = (0, 10)
        self.assertAlmostEqual(covered_token_fraction("x" * 10, [(0, 5)], span), 0.5)
        self.assertAlmostEqual(covered_token_fraction("x" * 10, [(0, 10)], span), 1.0)
        self.assertAlmostEqual(covered_token_fraction("x" * 10, [(2, 4)], span), 0.2)
        self.assertAlmostEqual(covered_token_fraction("x" * 10, [], span), 0.0)

    def test_off_by_one_span_is_not_a_full_hit(self):
        """Plan section G self-check: a span shifted by one char must not count."""
        text = "the quick brown fox"
        true_span = (4, 9)          # "quick"
        shifted = (5, 10)           # "uick "
        self.assertFalse(contains_span(text, [shifted], true_span))
        self.assertAlmostEqual(covered_token_fraction(text, [shifted], true_span), 0.8)

    def test_empty_span_is_never_a_hit(self):
        self.assertFalse(contains_span("abc", [(0, 3)], (1, 1)))


class TestDedup(unittest.TestCase):
    def test_minhash_detects_near_duplicates(self):
        a = ("Students must submit work by the stated deadline unless an extension "
             "has been granted by the instructor in advance.")
        b = ("Students must submit work by the stated deadline, unless an extension "
             "has been granted by the instructor in advance.")
        c = "The library opens at eight in the morning on weekdays during term."
        sig_a = minhash_signature(a)
        sig_b = minhash_signature(b)
        sig_c = minhash_signature(c)
        self.assertGreater(minhash_similarity(sig_a, sig_b), 0.7)
        self.assertLess(minhash_similarity(sig_a, sig_c), 0.3)

    def test_jaccard_is_symmetric(self):
        a = shingles("alpha beta gamma delta epsilon")
        b = shingles("alpha beta gamma delta zeta")
        self.assertAlmostEqual(jaccard(a, b), jaccard(b, a))
        self.assertEqual(jaccard(a, a), 1.0)

    def test_empty_text_is_safe(self):
        self.assertEqual(minhash_signature(""), tuple([0] * 128))
        self.assertEqual(jaccard(set(), set()), 0.0)


class TestRobots(unittest.TestCase):
    def test_disallowed_path_is_blocked(self):
        """Real UAlberta rule: Disallow: /catalogue/syllabus must decide BLOCK."""
        cache = RobotsCache(user_agent="CPS5801-research-bot/0.1")
        from urllib import robotparser
        p = robotparser.RobotFileParser()
        p.parse([
            "User-agent: *",
            "Disallow: /catalogue/archive",
            "Disallow: /catalogue/syllabus",
            "Disallow: /ezsend",
        ])
        cache.cache["https://apps.ualberta.ca"] = (p, "test")
        cache.raw["https://apps.ualberta.ca"] = "test"
        cache.fetched_at["https://apps.ualberta.ca"] = "test"
        self.assertFalse(
            cache.decision("https://apps.ualberta.ca/catalogue/syllabus/download/197050289")["allowed"])
        self.assertTrue(cache.decision("https://apps.ualberta.ca/catalogue/courses")["allowed"])

    def test_blanket_disallow_blocks_everything(self):
        cache = RobotsCache(user_agent="CPS5801-research-bot/0.1")
        from urllib import robotparser
        p = robotparser.RobotFileParser()
        p.parse(["User-agent: *", "Disallow: /"])
        cache.cache["https://ro.umich.edu"] = (p, "test")
        cache.raw["https://ro.umich.edu"] = "test"
        cache.fetched_at["https://ro.umich.edu"] = "test"
        self.assertFalse(cache.decision("https://ro.umich.edu/policies")["allowed"])

    def test_allow_all_permits_policy_page(self):
        cache = RobotsCache(user_agent="CPS5801-research-bot/0.1")
        from urllib import robotparser
        p = robotparser.RobotFileParser()
        p.parse(["User-agent: *", "Allow: /"])
        cache.cache["https://ocw.mit.edu"] = (p, "test")
        cache.raw["https://ocw.mit.edu"] = "test"
        cache.fetched_at["https://ocw.mit.edu"] = "test"
        self.assertTrue(
            cache.decision("https://ocw.mit.edu/courses/6-033/pages/syllabus/")["allowed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
