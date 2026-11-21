"""Tests for the test2 tooling: sampling, bundle invariants, grouping.

The second held-out set is only valuable if its design cannot drift while it is being built.
These tests pin the deterministic pieces:

* `26` length bins and readability screen (so the sampling frame is what the prereg says),
* `26` relation checks are *candidate* checks - a valid exception phrase must be required, a
  bare keyword must not pass,
* `27` grouping: a unique group id per question is not independence, so questions sharing a
  concrete template and inference shape must collapse into one group, while questions that
  merely share a broad signature must not.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


S26 = _load("s26", "scripts/26_select_test2_clauses.py")
S27 = _load("s27", "scripts/27_check_test2.py")
GOLD = _load("goldset", "scripts/goldset.py")


def clause(clause_id: str, text: str, *, has_exception: bool = False) -> "GOLD.Clause":
    return GOLD.Clause(doc_id="DOC", clause_id=clause_id, char_start=0,
                       char_end=len(text), text=text, has_exception=has_exception)


class TestLengthBins(unittest.TestCase):
    def test_bin_edges_follow_the_preregistered_quartiles(self):
        self.assertEqual(S26.bin_of(100), 1)
        self.assertEqual(S26.bin_of(727.9), 1)
        self.assertEqual(S26.bin_of(728), 2)
        self.assertEqual(S26.bin_of(1009.5), 3)
        self.assertEqual(S26.bin_of(1121.8), 4)
        self.assertEqual(S26.bin_of(5000), 4)


class TestReadability(unittest.TestCase):
    def test_numbered_prose_is_usable(self):
        c = clause("c1", "1.3 The Student Contract is made up of the following documents, "
                         "and the University's Regulations apply to every enrolled student.")
        self.assertTrue(S26.is_readable(c))

    def test_citation_dump_is_rejected(self):
        c = clause("c2", "Litchfield v. Spielberg, 736 F.2d 1352, 1357 (9th Cir. 1984) and "
                         "other authority concerning prior art and obviousness of claims.")
        self.assertFalse(S26.is_readable(c))

    def test_fragment_is_rejected(self):
        self.assertFalse(S26.is_readable(clause("c3", "prior art. Both issues, however, are")))


class TestRelationCandidates(unittest.TestCase):
    def test_rule_exception_requires_an_exception_phrase(self):
        texts = ["Students may sit the exam.", "Except with the Dean's approval."]
        flags = [{"has_exception": False}, {"has_exception": True}]
        self.assertEqual(S26.check_relation("rule_exception", "Except", texts, flags), [])
        bad = S26.check_relation("rule_exception", "Except", ["Nothing relevant here."],
                                 [{"has_exception": False}])
        self.assertTrue(bad)

    def test_defined_term_must_appear_in_two_clauses(self):
        texts = ["The assessment board decides.", "The assessment board must meet."]
        self.assertEqual(S26.check_relation("defined_term", "assessment board", texts, []), [])
        bad = S26.check_relation("defined_term", "assessment board",
                                 ["assessment board decides.", "nothing else"], [])
        self.assertTrue(bad)

    def test_quote_must_be_verbatim(self):
        bad = S26.check_relation("explicit_reference", "see section 4",
                                 ["no reference here"], [])
        self.assertTrue(any("verbatim" in p for p in bad))


class TestGrouping(unittest.TestCase):
    def _q(self, qid, *, family, concrete, shape, text, clause_ids):
        return {"qid": qid, "question": text, "group_id": qid,
                "provenance": {"template_family": family, "concrete_template": concrete,
                               "inference_shape": shape},
                "evidence_sets": [{"clauses": [{"clause_id": c} for c in clause_ids]}]}

    def test_same_concrete_template_and_shape_must_merge(self):
        qs = [self._q("Q1", family="DEADLINE", concrete="late-submission", shape="rule_then_exception",
                      text="Can I submit an assignment after the deadline?", clause_ids=["a"]),
              self._q("Q2", family="DEADLINE", concrete="late-submission", shape="rule_then_exception",
                      text="Is a late exam submission accepted by the school?", clause_ids=["b"])]
        mapping, report = S27.assign_groups(qs)
        self.assertEqual(mapping["Q1"], mapping["Q2"])
        self.assertEqual(report["groups"], 1)

    def test_same_broad_signature_only_is_not_merged(self):
        qs = [self._q("Q1", family="EXCEPTION_APPLIES", concrete="extension-medical",
                      shape="rule_then_exception", text="I was ill before the final exam.",
                      clause_ids=["a"]),
              self._q("Q2", family="EXCEPTION_APPLIES", concrete="extension-travel",
                      shape="rule_then_exception", text="I had to travel for a family emergency.",
                      clause_ids=["b"])]
        mapping, report = S27.assign_groups(qs)
        self.assertNotEqual(mapping["Q1"], mapping["Q2"])
        self.assertTrue(report["similar_candidates"])

    def test_shared_evidence_set_must_merge(self):
        qs = [self._q("Q1", family="A", concrete="one", shape="s1",
                      text="Question about policy alpha?", clause_ids=["x"]),
              self._q("Q2", family="B", concrete="two", shape="s2",
                      text="Different wording, same evidence?", clause_ids=["x"])]
        mapping, _ = S27.assign_groups(qs)
        self.assertEqual(mapping["Q1"], mapping["Q2"])

    def test_jaccard_based_merge(self):
        text = "Can a student appeal the decision of the assessment board within ten days"
        qs = [self._q("Q1", family="A", concrete="one", shape="s1", text=text, clause_ids=["x"]),
              self._q("Q2", family="B", concrete="two", shape="s2", text=text, clause_ids=["y"])]
        mapping, _ = S27.assign_groups(qs)
        self.assertEqual(mapping["Q1"], mapping["Q2"])


class TestJaccard(unittest.TestCase):
    def test_identical_sets(self):
        self.assertEqual(S27.jaccard({"a", "b"}, {"a", "b"}), 1.0)

    def test_disjoint_sets(self):
        self.assertEqual(S27.jaccard({"a"}, {"b"}), 0.0)


if __name__ == "__main__":
    unittest.main()
