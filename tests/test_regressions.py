"""Regression tests for bugs found while building the pipeline.

Each test corresponds to a defect that produced *silently wrong* output rather than
an error. They exist so the same class of failure cannot return unnoticed.

Run:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def load_module(name: str, filename: str):
    """Import a module whose name starts with a digit (e.g. 02_extract_clean.py)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod            # dataclasses look up sys.modules
    spec.loader.exec_module(mod)
    return mod


extract = load_module("extract_clean", "02_extract_clean.py")
corpus = load_module("build_safe_corpus", "03_build_safe_corpus.py")

from common import load_taxonomy  # noqa: E402

TAXONOMY = load_taxonomy(ROOT)


def write_html(tmp: Path, body: str) -> Path:
    p = tmp / "page.html"
    p.write_text(
        "<html><head><title>t</title></head><body>"
        "<nav><ul><li>Nav One</li><li>Nav Two</li></ul></nav>"
        f'<div role="main">{body}</div>'
        "</body></html>",
        encoding="utf-8",
    )
    return p


class TestHtmlParagraphsAreExtracted(unittest.TestCase):
    """Bug: blocks_from_html used tree.traverse(), which skips text nodes, so real
    <p> paragraphs were dropped and pages looked empty."""

    def test_paragraph_content_is_captured(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            body = "".join(
                f"<p>Policy paragraph number {i} states that students must comply with the "
                f"stated requirements except where an exception is granted.</p>"
                for i in range(1, 13)
            )
            html = write_html(tmp, body)
            blocks, _info = extract.blocks_from_html(html, TAXONOMY)
            paras = [b for b in blocks if b.block_type == "paragraph"]
            self.assertGreaterEqual(len(paras), 12,
                                    f"expected the 12 paragraphs, got {len(paras)}")
            self.assertIn("Policy paragraph number 1", " ".join(b.text for b in paras))

    def test_body_text_is_not_mislabelled_as_heading(self):
        """Bug: a 'short title-case line' heuristic labelled paragraphs as headings."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            html = write_html(
                tmp,
                "<p>Late Work Policy</p>"          # title case, short, but a paragraph
                "<p>Students who submit work after the deadline receive a penalty unless an "
                "extension was granted in advance.</p>",
            )
            blocks, _ = extract.blocks_from_html(html, TAXONOMY)
            headings = [b.text for b in blocks if b.block_type == "heading"]
            self.assertNotIn("Late Work Policy", headings,
                             "a <p> must not be reclassified as a heading from capitalisation")

    def test_navigation_is_removed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            html = write_html(Path(td), "<p>" + "word " * 300 + "</p>")
            blocks, _ = extract.blocks_from_html(html, TAXONOMY)
            self.assertNotIn("Nav One", " ".join(b.text for b in blocks))


class TestClauseIdsAreUnique(unittest.TestCase):
    """Bug: labels such as 'A.' legitimately repeat across sections, so clause ids
    collided inside one document and downstream lookups silently overwrote."""

    def test_repeated_labels_get_distinct_ids(self):
        blocks = [
            extract.Block(block_type="heading", text="Section One", heading_level=2),
            extract.Block(block_type="clause", text="A. First requirement " + "x" * 100,
                          clause_label="A."),
            extract.Block(block_type="clause", text="B. Second requirement " + "x" * 100,
                          clause_label="B."),
            extract.Block(block_type="heading", text="Section Two", heading_level=2),
            extract.Block(block_type="clause", text="A. Third requirement " + "x" * 100,
                          clause_label="A."),
        ]
        detectors = {}
        for name in ("exception", "crossref", "alt_evidence"):
            import re
            detectors[name] = re.compile(r"(?!x)x")   # never matches
        records = extract.build_clause_records(blocks, detectors, "DOC", "faculty", "Uni")
        ids = [r["clause_id"] for r in records]
        self.assertEqual(len(ids), len(set(ids)), f"duplicate clause ids: {ids}")
        self.assertIn("DOC#A.", ids)
        self.assertIn("DOC#A.~2", ids)


class TestNoSilentTruncation(unittest.TestCase):
    """Bug class: an over-length block must be split, never truncated or dropped."""

    def test_overlength_block_is_split_not_truncated(self):
        tok = corpus.Tokenizer("unused", mode="regex")
        text = " ".join(f"sentence{i} about policy." for i in range(400))
        max_tokens = 100
        pieces = corpus.split_recursive(text, tok, max_tokens)
        self.assertGreater(len(pieces), 1)
        for p in pieces:
            self.assertLessEqual(p["tokens"], max_tokens + 5)
        joined = " ".join(p["text"] for p in pieces)
        self.assertEqual(joined.split(), text.split(),
                         "splitting must preserve every token")

    def test_split_reason_is_recorded(self):
        tok = corpus.Tokenizer("unused", mode="regex")
        text = " ".join(f"w{i}" for i in range(500))
        pieces = corpus.split_recursive(text, tok, 50)
        self.assertTrue(all(p["split_reason"] for p in pieces if p["tokens"] > 0))


class TestParserAvailabilityFailsLoudly(unittest.TestCase):
    """Bug: a missing HTML parser returned '' and real pages were classified
    'pending_review', making the failure invisible."""

    def test_probe_raises_without_parser(self):
        probe = load_module("probe_seeds_mod", "probe_seeds.py")
        original = probe._HTML_PARSER
        try:
            probe._HTML_PARSER = ("none", None) if False else None
            # force the "no parser" path by emptying the cache and hiding imports
            probe._HTML_PARSER = None
            import builtins
            real_import = builtins.__import__

            def fake_import(name, *a, **kw):
                if name.startswith("selectolax") or name.startswith("lxml"):
                    raise ImportError(f"blocked {name}")
                return real_import(name, *a, **kw)

            saved_lh = probe.LH
            probe.LH = None
            builtins.__import__ = fake_import
            try:
                with self.assertRaises(RuntimeError):
                    probe.text_of(b"<html><body><p>hi</p></body></html>")
            finally:
                builtins.__import__ = real_import
                probe.LH = saved_lh
        finally:
            probe._HTML_PARSER = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
