"""Freeze and gate for the required-source list (reproduction package).

The point of these tests is that the public repository cannot silently redefine the corpus:
the 36 required documents are frozen in data/REQUIRED_SOURCES.csv, and a crawl that returns
fewer documents must fail the gate rather than look complete.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import make_required_sources as req  # noqa: E402


class TestRequiredSources(unittest.TestCase):
    def test_freeze_exists_and_matches_hash(self):
        self.assertTrue(req.OUT.exists())
        digest = hashlib.sha256(req.OUT.read_bytes()).hexdigest()
        self.assertEqual(req.SHA.read_text(encoding="utf-8").strip(), digest)

    def test_counts_match_manifest(self):
        with req.OUT.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 36)
        tiers = {t: sum(1 for r in rows if r["tier"] == t) for t in {r["tier"] for r in rows}}
        self.assertEqual(tiers, {"course": 14, "faculty": 9, "institution": 13})
        self.assertEqual(len({r["institution"] for r in rows}), 12)
        manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["documents_included"], 36)
        self.assertEqual(manifest["tier_counts"], tiers)

    def test_every_row_is_usable(self):
        with req.OUT.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            self.assertTrue(r["final_url"].startswith("http"))
            self.assertEqual(len(r["sha256_raw"]), 64)
            self.assertTrue(r["seed_id"])
            self.assertTrue(r["doc_id"])

    def test_check_passes_on_clean_tree(self):
        self.assertEqual(req.check(ROOT), 0)


if __name__ == "__main__":
    unittest.main()
