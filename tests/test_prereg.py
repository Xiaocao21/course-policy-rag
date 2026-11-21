"""Guards for the second held-out set: locking, pre-registration, and the one-shot rule.

The test split's lock hashes `annotations/questions.jsonl`, and `19_rerank_eval.py`
refuses to evaluate it again. A second held-out set must not disturb either property, so
these tests pin the two mechanisms that protect it:

* `05_audit_splits.lock_test(split="test2")` writes its own directory and leaves
  `annotations/LOCKED` untouched;
* `19_rerank_eval.verify_preregistration` reports every artefact whose hash has moved,
  which is what makes "evaluate once, against frozen settings" checkable.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


AUDIT = _load("audit05", "scripts/05_audit_splits.py")
EVAL19 = _load("eval19", "scripts/19_rerank_eval.py")


class TestLockIsolation(unittest.TestCase):
    def _paths(self, tmp: Path) -> dict:
        ann = tmp / "annotations"
        ann.mkdir(parents=True, exist_ok=True)
        reports = tmp / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        return {"annotations": ann, "reports": reports}

    def test_test2_lock_does_not_touch_the_test_lock(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            paths = self._paths(tmp)
            (paths["annotations"] / "LOCKED").mkdir(parents=True, exist_ok=True)
            (paths["annotations"] / "LOCKED" / "LOCKED.json").write_text("{}", encoding="utf-8")
            before = (paths["annotations"] / "LOCKED" / "LOCKED.json").read_bytes()

            qs = [{"qid": "Q9001", "split": "test2", "group_id": "G9001",
                   "manual_reviewed": True}]
            res = AUDIT.lock_test(qs, paths, split="test2")
            self.assertTrue(res["locked"])
            self.assertTrue((paths["annotations"] / "LOCKED_TEST2" / "LOCKED2.json").exists())
            self.assertTrue((paths["annotations"] / "LOCKED_TEST2" / "test2.jsonl").exists())
            self.assertEqual(before,
                             (paths["annotations"] / "LOCKED" / "LOCKED.json").read_bytes())

    def test_test2_meta_records_its_own_questions_file(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            paths = self._paths(tmp)
            qfile = paths["annotations"] / "questions_test2.jsonl"
            qfile.write_text(json.dumps({"qid": "Q1", "split": "test2"}) + "\n",
                             encoding="utf-8")
            res = AUDIT.lock_test([{"qid": "Q1", "split": "test2", "group_id": "G1"}],
                                  paths, split="test2", questions_path=qfile)
            meta = json.loads((paths["annotations"] / "LOCKED_TEST2" / "LOCKED2.json")
                              .read_text(encoding="utf-8"))
            self.assertEqual(meta["split"], "test2")
            self.assertTrue(meta["questions_jsonl_sha256"])
            self.assertEqual(res["questions"], 1)

    def test_lock_refuses_an_empty_split(self):
        with tempfile.TemporaryDirectory() as td:
            paths = self._paths(Path(td))
            res = AUDIT.lock_test([], paths, split="test2")
            self.assertFalse(res["locked"])


class TestPreregistration(unittest.TestCase):
    def test_missing_file_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            problems = EVAL19.verify_preregistration(tmp, tmp / "nope.json", tmp / "q.jsonl")
            self.assertTrue(problems)

    def test_intact_artifacts_pass(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            art = tmp / "data.txt"
            art.write_text("frozen", encoding="utf-8")
            qs = tmp / "questions_test2.jsonl"
            qs.write_text("{}", encoding="utf-8")
            pre = tmp / "PREREG.json"
            pre.write_text(json.dumps({
                "artifacts": {"data.txt": EVAL19.sha256_file(art)},
                "questions_sha256": EVAL19.sha256_file(qs),
            }), encoding="utf-8")
            self.assertEqual(EVAL19.verify_preregistration(tmp, pre, qs), [])

    def test_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            art = tmp / "data.txt"
            art.write_text("frozen", encoding="utf-8")
            qs = tmp / "questions_test2.jsonl"
            qs.write_text("{}", encoding="utf-8")
            pre = tmp / "PREREG.json"
            pre.write_text(json.dumps({
                "artifacts": {"data.txt": EVAL19.sha256_file(art)},
                "questions_sha256": EVAL19.sha256_file(qs),
            }), encoding="utf-8")
            art.write_text("edited after pre-registration", encoding="utf-8")
            problems = EVAL19.verify_preregistration(tmp, pre, qs)
            self.assertEqual(len(problems), 1)
            self.assertEqual(problems[0]["problem"], "hash mismatch")

    def test_question_file_change_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            qs = tmp / "questions_test2.jsonl"
            qs.write_text("{}", encoding="utf-8")
            pre = tmp / "PREREG.json"
            pre.write_text(json.dumps({"artifacts": {},
                                       "questions_sha256": EVAL19.sha256_file(qs)}),
                           encoding="utf-8")
            qs.write_text('{"qid": "extra"}', encoding="utf-8")
            problems = EVAL19.verify_preregistration(tmp, pre, qs)
            self.assertEqual(len(problems), 1)


class TestOneShotRule(unittest.TestCase):
    def test_eval_has_no_force_flag(self):
        import inspect
        src = inspect.getsource(EVAL19.main)
        # the prose explains that there is no --force; the parser must not offer one
        self.assertNotIn('add_argument("--force"', src)
        self.assertIn("FINAL_RUN.json", src)


if __name__ == "__main__":
    unittest.main()
