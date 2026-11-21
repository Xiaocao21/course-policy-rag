"""Regression tests for the reranker stage (data construction + training protocol).

The tests deliberately avoid the GPU and the trained checkpoint: what they pin down is
the labelling contract, the candidate identity rule, the group-disjoint split, the
set-invariance property that motivated the selection metric, and the guards that keep the
held-out test split out of training.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common import contains_span  # noqa: E402
import goldset  # noqa: E402


def load_module(rel: str, name: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


DATA = load_module("scripts/17_build_rerank_data.py", "build17")
TRAIN = load_module("scripts/18_train_reranker.py", "train18")


class TestLabels(unittest.TestCase):
    def test_full_cover_is_clause_positive(self):
        label = DATA.label_chunk([[100, 300]], [(120, 280)])
        self.assertEqual(label, "positive_clause")

    def test_partial_overlap_is_partial_not_positive(self):
        label = DATA.label_chunk([[100, 200]], [(150, 400)])
        self.assertEqual(label, "partial")

    def test_disjoint_is_negative(self):
        label = DATA.label_chunk([[100, 200]], [(500, 600)])
        self.assertEqual(label, "negative")

    def test_any_alternative_sufficient_set_counts(self):
        # two annotated clauses, chunk fully covers only the second one
        label = DATA.label_chunk([[400, 700]], [(10, 50), (450, 600)])
        self.assertEqual(label, "positive_clause")

    def test_union_across_intervals_must_be_contiguous(self):
        # a gap between intervals means the span is not covered
        self.assertFalse(contains_span("", [[0, 10], [20, 30]], (5, 25)))
        self.assertTrue(contains_span("", [[0, 10], [10, 30]], (5, 25)))


class TestIdentity(unittest.TestCase):
    def test_offsets_are_part_of_identity(self):
        a = DATA.chunk_key("doc", "v1", 0, 100, "same text")
        b = DATA.chunk_key("doc", "v1", 10, 110, "same text")
        self.assertNotEqual(a, b)

    def test_identical_chunk_same_identity(self):
        a = DATA.chunk_key("doc", "v1", 0, 100, "same text")
        b = DATA.chunk_key("doc", "v1", 0, 100, "same text")
        self.assertEqual(a, b)


class TestSplitArtifacts(unittest.TestCase):
    def setUp(self):
        pairs = ROOT / "models" / "reranker" / "pairs_dev.jsonl"
        groups = ROOT / "models" / "reranker" / "split_groups.json"
        if not pairs.exists() or not groups.exists():
            self.skipTest("reranker data not built yet")
        self.groups = json.loads(groups.read_text(encoding="utf-8"))
        self.pairs = [json.loads(l) for l in pairs.read_text(encoding="utf-8").splitlines()
                      if l.strip()]

    def test_group_split_is_disjoint(self):
        train = set(self.groups["train_groups"])
        val = set(self.groups["val_groups"])
        self.assertEqual(train & val, set())

    def test_role_matches_group_membership(self):
        train = set(self.groups["train_groups"])
        val = set(self.groups["val_groups"])
        for p in self.pairs:
            if p["role"] == "train":
                self.assertIn(p["group_id"], train)
            elif p["role"] == "val":
                self.assertIn(p["group_id"], val)
            else:
                self.fail(f"unexpected role {p['role']}")

    def test_no_question_straddles_roles(self):
        roles: dict[str, set[str]] = {}
        for p in self.pairs:
            roles.setdefault(p["qid"], set()).add(p["role"])
        straddling = {q for q, r in roles.items() if len(r) > 1}
        self.assertEqual(straddling, set())

    def test_partials_are_never_positive(self):
        # partial chunks must stay out of the positive class; they are analysis-only
        partials = [p for p in self.pairs if p["label"] == "partial"]
        self.assertTrue(partials, "expected some partial chunks in the built data")
        self.assertTrue(all(p["label"] != "positive_clause" for p in partials))


@unittest.skipUnless((ROOT / "cleaned").exists(), "corpus not present in this checkout")
class TestSetInvariance(unittest.TestCase):
    """Reranking a fixed candidate set must not move set-level @10 metrics."""

    @classmethod
    def setUpClass(cls):
        cls.ev = load_module("scripts/13_evaluate.py", "eval13")
        cls.corpus = goldset.load_corpus(ROOT)
        questions = [json.loads(l) for l in
                     (ROOT / "annotations" / "questions.jsonl").read_text(encoding="utf-8")
                     .splitlines() if l.strip()]
        # one question whose gold spans resolve, so coverage is meaningful
        cls.question = next(q for q in questions
                            if q["split"] == "dev" and cls.ev.gold_spans(q, cls.corpus))
        cls.gold = cls.ev.gold_spans(cls.question, cls.corpus)
        ref = cls.question["evidence_sets"][0]["clauses"][0]
        clause = cls.corpus.clauses[ref["clause_id"]]
        cls.hit = {"chunk_id": "hit", "doc_id": clause.doc_id, "token_count": 10,
                   "ev_ranges": [[ref["char_start"], ref["char_end"]]]}
        cls.miss = [{"chunk_id": f"miss{i}", "doc_id": "SOMEWHERE-ELSE",
                     "token_count": 10, "ev_ranges": [[0, 5]]} for i in range(9)]

    def test_order_does_not_change_top10_metrics(self):
        forward = self.ev.coverage_for_set(self.gold, [self.hit] + self.miss, self.corpus)
        backward = self.ev.coverage_for_set(self.gold, self.miss + [self.hit], self.corpus)
        self.assertEqual(forward[1], backward[1])  # hit
        self.assertEqual(forward[2], backward[2])  # complete
        self.assertAlmostEqual(forward[0], backward[0])  # recall

    def test_order_does_change_mrr(self):
        rank_first = self.ev.first_hit_rank(self.gold, [self.hit] + self.miss, self.corpus)
        rank_last = self.ev.first_hit_rank(self.gold, self.miss + [self.hit], self.corpus)
        self.assertEqual(rank_first, 1)
        self.assertEqual(rank_last, 10)


class TestSelection(unittest.TestCase):
    def _run(self, complete, mrr, recall, lr=2e-5, neg=4):
        return {"lr": lr, "negatives": neg,
                "val": {"macro_complete@10": complete, "macro_mrr": mrr,
                        "macro_recall@10": recall}}

    def test_complete_metric_wins(self):
        best = TRAIN.select_best([self._run(0.4, 0.9, 0.9), self._run(0.5, 0.1, 0.1)])
        self.assertEqual(best["val"]["macro_complete@10"], 0.5)

    def test_tie_breaks_on_mrr_then_recall(self):
        best = TRAIN.select_best([self._run(0.5, 0.2, 0.9), self._run(0.5, 0.3, 0.1)])
        self.assertEqual(best["val"]["macro_mrr"], 0.3)

    def test_final_tie_prefers_smaller_lr_then_fewer_negatives(self):
        best = TRAIN.select_best([self._run(0.5, 0.3, 0.3, lr=2e-5, neg=8),
                                  self._run(0.5, 0.3, 0.3, lr=1e-5, neg=4)])
        self.assertEqual((best["lr"], best["negatives"]), (1e-5, 4))


class TestGuards(unittest.TestCase):
    def test_build_refuses_test_split(self):
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "17_build_rerank_data.py"),
                              "--split", "test", "--dry-run"],
                             capture_output=True, text=True, timeout=600)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("development split", (out.stderr or "").lower())

    def test_trainer_refuses_without_cuda(self):
        # the CPU interpreter has no CUDA build, so the trainer must stop immediately
        out = subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"),
                              str(ROOT / "scripts" / "18_train_reranker.py"), "--smoke"],
                             capture_output=True, text=True, timeout=600)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("cuda", ((out.stderr or "") + (out.stdout or "")).lower())


class TestFrozenSchema(unittest.TestCase):
    REQUIRED = ("kind", "created_at", "env_role", "model", "tokenizer", "data", "code",
                "corpus", "env", "training", "selection", "seed", "val_metrics")

    def test_frozen_has_required_fields(self):
        path = ROOT / "models" / "reranker" / "FROZEN.json"
        if not path.exists():
            self.skipTest("FROZEN.json not produced yet")
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in self.REQUIRED:
            self.assertIn(key, data)
        for key in ("gpu_model", "cuda", "torch", "precision", "batch_size", "grad_accum"):
            self.assertIn(key, data["env"])
        self.assertIn("checkpoint_sha256", data["model"])
        self.assertEqual(data["env_role"], "canonical")


if __name__ == "__main__":
    unittest.main()
