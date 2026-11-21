"""Tests for goldset.py - the shared span/validation implementation.

These run offline against a synthetic corpus built in a temporary directory, so they
pin the protocol rules independently of the real 36-document corpus.

Run:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import goldset  # noqa: E402
from common import sha256_text  # noqa: E402

DOC_A = "DOC-A"
DOC_B = "DOC-B"

# Two documents. Spans below are computed from these exact strings.
BLOCKS_A = [
    "Article 3 Late Work",
    "3.1 Work submitted after the deadline loses ten percent per day.",
    "3.2 An extension may be granted in exceptional circumstances by the instructor.",
]
BLOCKS_B = [
    "Section 5 Examinations",
    "5.1 A student who misses an examination for medical reasons may request a deferred examination.",
]


def text_of(blocks: list[str]) -> str:
    return "\n".join(blocks)


def build_corpus(tmp: Path) -> Path:
    """Create a minimal but structurally real corpus under tmp."""
    (tmp / "extracted").mkdir(parents=True, exist_ok=True)
    (tmp / "cleaned").mkdir(parents=True, exist_ok=True)

    docs = {DOC_A: BLOCKS_A, DOC_B: BLOCKS_B}
    rows_by_doc = {}
    for doc_id, blocks in docs.items():
        (tmp / "extracted" / f"{doc_id}.json").write_text(
            json.dumps({"blocks": [{"text": b} for b in blocks]}), encoding="utf-8")

    text_a = text_of(BLOCKS_A)
    text_b = text_of(BLOCKS_B)
    # clause spans cover blocks 1 and 2 of each document
    start_a1 = len(BLOCKS_A[0]) + 1
    end_a1 = start_a1 + len(BLOCKS_A[1])
    start_a2 = end_a1 + 1
    end_a2 = start_a2 + len(BLOCKS_A[2])
    start_b1 = len(BLOCKS_B[0]) + 1
    end_b1 = start_b1 + len(BLOCKS_B[1])

    def rec(doc_id, cid, text, s, e, exc=False):
        return {"doc_id": doc_id, "clause_id": cid, "char_start": s, "char_end": e,
                "text": text[s:e], "evidence_sha256": sha256_text(text[s:e]),
                "has_exception": exc, "has_crossref": False, "has_alt_evidence": False,
                "boilerplate": False, "section_path": "", "char_len": e - s}

    rows_by_doc[DOC_A] = [
        rec(DOC_A, f"{DOC_A}#3.1", text_a, start_a1, end_a1),
        rec(DOC_A, f"{DOC_A}#3.2", text_a, start_a2, end_a2, exc=True),
    ]
    rows_by_doc[DOC_B] = [rec(DOC_B, f"{DOC_B}#5.1", text_b, start_b1, end_b1)]

    for doc_id, rows in rows_by_doc.items():
        with (tmp / "cleaned" / f"{doc_id}.evidence.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    with (tmp / "sources.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "tier", "institution", "include"])
        w.writeheader()
        w.writerow({"doc_id": DOC_A, "tier": "course", "institution": "Uni A", "include": "TRUE"})
        w.writerow({"doc_id": DOC_B, "tier": "institution", "institution": "Uni B",
                    "include": "TRUE"})
    return tmp


class GoldsetTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = build_corpus(Path(self._tmp.name))
        self.corpus = goldset.load_corpus(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def ref(self, clause_id: str, doc_id: str, quote: str | None = None) -> dict:
        c = self.corpus.clauses[clause_id]
        text = goldset.evidence_text(self.corpus, doc_id)
        return {"doc_id": doc_id, "clause_id": clause_id,
                "char_start": c.char_start, "char_end": c.char_end,
                "quote_evidence": quote if quote is not None else text[c.char_start:c.char_end]}


class TestCorpusLoading(GoldsetTestBase):
    def test_loads_included_documents_only(self):
        self.assertEqual(set(self.corpus.by_doc), {DOC_A, DOC_B})
        self.assertEqual(len(self.corpus.clauses), 3)

    def test_tier_comes_from_sources_csv(self):
        self.assertEqual(self.corpus.doc_tier[DOC_A], "course")
        self.assertEqual(self.corpus.doc_tier[DOC_B], "institution")

    def test_excluded_documents_are_absent(self):
        with (self.root / "sources.csv").open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([DOC_B, "institution", "Uni B", "FALSE"])
        corpus = goldset.load_corpus(self.root, doc_ids=[DOC_A])
        self.assertEqual(set(corpus.by_doc), {DOC_A})

    def test_clause_spans_match_evidence_text(self):
        for cid, clause in self.corpus.clauses.items():
            text = goldset.evidence_text(self.corpus, clause.doc_id)
            self.assertEqual(sha256_text(text[clause.char_start:clause.char_end]),
                             clause.evidence_sha256, f"span mismatch for {cid}")


class TestResolveRef(GoldsetTestBase):
    def test_valid_ref_resolves_cleanly(self):
        ref = self.ref(f"{DOC_A}#3.1", DOC_A)
        clause, problems = goldset.resolve_clause_ref(ref, self.corpus)
        self.assertIsNotNone(clause)
        self.assertEqual(problems, [])

    def test_unknown_clause_id_is_reported(self):
        ref = self.ref(f"{DOC_A}#3.1", DOC_A)
        ref["clause_id"] = "NOPE#1"
        clause, problems = goldset.resolve_clause_ref(ref, self.corpus)
        self.assertIsNone(clause)
        self.assertTrue(any("unknown clause_id" in p for p in problems))

    def test_quote_mismatch_is_reported(self):
        ref = self.ref(f"{DOC_A}#3.1", DOC_A, quote="something else entirely")
        _clause, problems = goldset.resolve_clause_ref(ref, self.corpus)
        self.assertTrue(any("quote_evidence does not match" in p for p in problems))

    def test_span_outside_evidence_is_reported(self):
        ref = self.ref(f"{DOC_A}#3.1", DOC_A)
        ref["char_end"] = 10 ** 6
        _clause, problems = goldset.resolve_clause_ref(ref, self.corpus)
        self.assertTrue(any("outside evidence length" in p for p in problems))

    def test_inverted_span_is_reported(self):
        ref = self.ref(f"{DOC_A}#3.1", DOC_A)
        ref["char_start"], ref["char_end"] = ref["char_end"], ref["char_start"]
        _clause, problems = goldset.resolve_clause_ref(ref, self.corpus)
        self.assertTrue(any("invalid span" in p for p in problems))

    def test_doc_id_mismatch_is_reported(self):
        ref = self.ref(f"{DOC_A}#3.1", DOC_A)
        ref["doc_id"] = DOC_B
        _clause, problems = goldset.resolve_clause_ref(ref, self.corpus)
        self.assertTrue(any("doc_id mismatch" in p for p in problems))


class TestValidateQuestion(GoldsetTestBase):
    def q(self, **kw) -> dict:
        base = {
            "qid": "Q1", "question": "May I submit late?", "reference_answer": "No.",
            "group_id": "G1", "split": "dev", "evidence_type": "single-clause",
            "domain": "late", "source_level": "course",
            "evidence_sets": [{"set_id": "S1", "clauses": [self.ref(f"{DOC_A}#3.1", DOC_A)]}],
        }
        base.update(kw)
        return base

    def test_valid_question_has_no_problems(self):
        problems = goldset.validate_question(
            self.q(), self.corpus,
            valid_evidence_types={"single-clause", "rule-and-exception", "cross-clause"},
            valid_domains={"late", "exam"}, valid_levels={"course", "institution"})
        self.assertEqual(problems, [])

    def test_missing_required_field(self):
        q = self.q()
        del q["domain"]
        problems = goldset.validate_question(q, self.corpus)
        self.assertTrue(any("missing required field 'domain'" in p for p in problems))

    def test_prefilled_question_cannot_enter_test(self):
        q = self.q(split="test", prefilled=True, manual_reviewed=False)
        problems = goldset.validate_question(q, self.corpus)
        self.assertTrue(any("may not enter the test split" in p for p in problems))

    def test_prefilled_question_allowed_in_test_once_reviewed(self):
        q = self.q(split="test", prefilled=True, manual_reviewed=True)
        self.assertEqual(goldset.validate_question(q, self.corpus), [])

    def test_non_prefilled_question_needs_no_review_flag(self):
        q = self.q(split="test", prefilled=False)
        self.assertEqual(goldset.validate_question(q, self.corpus), [])

    def test_empty_evidence_sets_rejected(self):
        self.assertTrue(goldset.validate_question(self.q(evidence_sets=[]), self.corpus))

    def test_sufficient_set_with_no_clauses_rejected(self):
        q = self.q(evidence_sets=[{"set_id": "S1", "clauses": []}])
        self.assertTrue(any("has no clauses" in p
                            for p in goldset.validate_question(q, self.corpus)))

    def test_bad_clause_ref_rejected(self):
        q = self.q()
        q["evidence_sets"][0]["clauses"][0]["clause_id"] = "NOPE#1"
        self.assertTrue(goldset.validate_question(q, self.corpus))

    def test_unknown_domain_and_type_rejected_when_validated(self):
        q = self.q(domain="nonsense", evidence_type="nonsense")
        problems = goldset.validate_question(
            q, self.corpus, valid_evidence_types={"single-clause"},
            valid_domains={"late"}, valid_levels={"course"})
        self.assertTrue(any("unknown evidence_type" in p for p in problems))
        self.assertTrue(any("unknown domain" in p for p in problems))


class TestScoring(GoldsetTestBase):
    def setUp(self):
        super().setUp()
        self.clause = self.corpus.clauses[f"{DOC_A}#3.1"]
        self.ref_a = self.ref(f"{DOC_A}#3.1", DOC_A)
        self.text = goldset.evidence_text(self.corpus, DOC_A)

    def question(self, sets: list[list[dict]]) -> dict:
        return {"qid": "Q", "evidence_sets": [
            {"set_id": f"S{i}", "clauses": clauses} for i, clauses in enumerate(sets)]}

    def span(self):
        c = self.clause
        return (c.doc_id, c.char_start, c.char_end)

    def test_exact_retrieval_is_full_hit(self):
        q = self.question([[self.ref_a]])
        res = goldset.score_question(q, self.corpus, [self.span()])
        self.assertTrue(res.hit)
        self.assertTrue(res.complete)
        self.assertEqual(res.recall, 1.0)
        self.assertAlmostEqual(res.partial, 1.0)

    def test_one_char_short_is_not_a_hit(self):
        _d, s, e = self.span()
        q = self.question([[self.ref_a]])
        res = goldset.score_question(q, self.corpus, [(_d, s, e - 1)])
        self.assertFalse(res.hit)
        self.assertFalse(res.complete)
        self.assertEqual(res.recall, 0.0)
        self.assertLess(res.partial, 1.0)
        self.assertGreater(res.partial, 0.98)

    def test_wider_retrieval_still_covers(self):
        _d, s, e = self.span()
        q = self.question([[self.ref_a]])
        res = goldset.score_question(q, self.corpus, [(_d, max(0, s - 50), e + 50)])
        self.assertTrue(res.hit)

    def test_retrieval_from_wrong_document_does_not_count(self):
        _d, s, e = self.span()
        q = self.question([[self.ref_a]])
        res = goldset.score_question(q, self.corpus, [(DOC_B, s, e)])
        self.assertFalse(res.hit)

    def test_split_retrieval_union_covers_when_adjacent(self):
        _d, s, e = self.span()
        mid = s + (e - s) // 2
        q = self.question([[self.ref_a]])
        res = goldset.score_question(q, self.corpus, [(_d, s, mid), (_d, mid, e)])
        self.assertTrue(res.hit)

    def test_gap_in_retrieval_is_not_a_hit(self):
        _d, s, e = self.span()
        mid = s + (e - s) // 2
        q = self.question([[self.ref_a]])
        res = goldset.score_question(q, self.corpus, [(_d, s, mid - 2), (_d, mid + 2, e)])
        self.assertFalse(res.hit)

    def test_alternative_sets_use_best_set(self):
        ref_b = self.ref(f"{DOC_B}#5.1", DOC_B)
        q = self.question([[self.ref_a], [ref_b]])
        # only the second set is retrieved
        b = self.corpus.clauses[f"{DOC_B}#5.1"]
        res = goldset.score_question(q, self.corpus, [(DOC_B, b.char_start, b.char_end)])
        self.assertTrue(res.hit)
        self.assertTrue(res.complete)
        self.assertEqual(res.recall, 1.0)
        self.assertEqual(res.full_sets, [1])

    def test_multi_clause_set_requires_all_for_completeness(self):
        exc = self.ref(f"{DOC_A}#3.2", DOC_A)
        q = self.question([[self.ref_a, exc]])
        # retrieve only the first clause
        res = goldset.score_question(q, self.corpus, [self.span()])
        self.assertTrue(res.hit)             # any-hit
        self.assertFalse(res.complete)       # but not complete
        self.assertEqual(res.recall, 0.5)

    def test_no_retrieval_scores_zero(self):
        q = self.question([[self.ref_a]])
        res = goldset.score_question(q, self.corpus, [])
        self.assertFalse(res.hit)
        self.assertFalse(res.complete)
        self.assertEqual(res.recall, 0.0)
        self.assertEqual(res.partial, 0.0)


class TestGroupCounts(GoldsetTestBase):
    def test_detects_group_crossing_split(self):
        qs = [{"qid": "a", "group_id": "G", "split": "dev"},
              {"qid": "b", "group_id": "G", "split": "test"},
              {"qid": "c", "group_id": "H", "split": "test"}]
        stats = goldset.test_group_counts(qs)
        self.assertEqual(stats["crossing_groups"], ["G"])
        self.assertEqual(stats["groups"]["dev"], 1)
        self.assertEqual(stats["groups"]["test"], 2)

    def test_counts_review_flags(self):
        qs = [{"qid": "a", "group_id": "G", "split": "dev",
               "prefilled": True, "manual_reviewed": False},
              {"qid": "b", "group_id": "H", "split": "dev",
               "prefilled": True, "manual_reviewed": True}]
        stats = goldset.test_group_counts(qs)
        self.assertEqual(stats["prefilled"], 2)
        self.assertEqual(stats["manual_reviewed"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
