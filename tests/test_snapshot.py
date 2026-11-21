"""Corpus snapshot tests: packaging and verification must be provable, not assumed."""
from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import make_corpus_snapshot as snap  # noqa: E402
import fetch_data  # noqa: E402


@unittest.skipUnless((ROOT / "cleaned").exists(), "corpus not present in this checkout")
class TestSnapshotPackaging(unittest.TestCase):
    def test_collect_matches_manifest(self):
        files, entries = snap.collect(ROOT)
        self.assertEqual(len(files), 110)          # 36 docs x 3 layers + 2 corpus files
        self.assertEqual(len(entries), 110)
        self.assertEqual(sum(1 for e in entries if e["layer"] == "corpus_safe_512"), 2)
        for entry in entries:
            self.assertEqual(len(entry["sha256"]), 64)
            if entry["manifest_sha256"]:
                self.assertEqual(entry["sha256"], entry["manifest_sha256"],
                                 f"{entry['path']} differs from MANIFEST")

    def test_required_docs_are_the_36(self):
        docs = {r["doc_id"] for r in snap.required_docs(ROOT)}
        self.assertEqual(len(docs), 36)


class TestArchiveSafety(unittest.TestCase):
    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "evil.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                data = b"x"
                info = tarfile.TarInfo("../escaped.txt")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            ok, report = fetch_data.verify(ROOT, archive, Path(tmp) / "out")
            self.assertFalse(ok)
            self.assertIn("unsafe path", report.get("reason", ""))

    def test_rejects_archive_without_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "empty.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                data = b"x"
                info = tarfile.TarInfo("readme.txt")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            ok, report = fetch_data.verify(ROOT, archive, Path(tmp) / "out")
            self.assertFalse(ok)
            self.assertIn("SNAPSHOT_MANIFEST", report.get("reason", ""))


if __name__ == "__main__":
    unittest.main()
