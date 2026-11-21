"""02_extract_clean.py - structure-aware extraction, dual text layers, dedup.

Pipeline per document:
    raw/<tier>/<doc_id>/raw.{html,pdf}
        -> structure blocks (heading / clause / paragraph / list / table) with page+line provenance
        -> header/footer removal with protected-line REVIEW (exception clauses must survive)
        -> evidence layer (character-faithful) + retrieval layer (normalised) + charmap
        -> clause records with stable ids and evidence-space spans
    outputs:
        extracted/<doc_id>.json
        cleaned/<doc_id>.evidence.jsonl   <- ALL gold spans anchor here
        cleaned/<doc_id>.retrieval.jsonl
        cleaned/<doc_id>.charmap.jsonl
        cleaned/removed_lines.log
        reports/duplicates.csv
        reports/quality_report.csv

Key invariants (see plan sections E and G):
  * evidence layer is character-faithful; normalisation happens only in retrieval
  * a clause span must never be split by chunking; clause_id stays stable
  * header/footer lines containing exception/qualifier language are REVIEW, not deleted

Usage:
    .venv\\Scripts\\python.exe scripts\\02_extract_clean.py
    .venv\\Scripts\\python.exe scripts\\02_extract_clean.py --doc-id MIT-OCW-6-033-...
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_ROOT, build_text_layers, compile_detectors, decode_html, load_settings,
    load_taxonomy, minhash_signature, minhash_similarity, project_paths, read_json,
    sha256_text, write_json, write_jsonl,
)

CLAUSE_HEAD_RE = re.compile(
    r"^\s*(?:(?P<num>\d+(?:\.\d+){1,3})\b|(?P<art>Article\s+\d+[A-Za-z]?)\b|"
    r"(?P<sec>Section\s+\d+(?:\.\d+)*)\b|(?P<rule>Rule\s+\d+)\b|"
    r"(?P<alnum>[A-Z]\.)\s|(?P<paren>\(\s*[a-z]\s*\))\s)"
)
HEADING_ONLY_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Za-z ,\-/&']{3,80}$")

# lines that commonly repeat as page furniture
PAGE_NUM_RE = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*(?:of\s*\d{1,4})?\s*$", re.I)


@dataclass
class Block:
    block_type: str                 # heading | clause | paragraph | list_item | table | title
    text: str
    page: int = 0
    line_start: int = 0
    line_end: int = 0
    heading_level: int = 0
    clause_label: str = ""
    flags: dict = field(default_factory=dict)


# ------------------------------------------------------------------ extraction
def blocks_from_pdf(path: Path, taxonomy: dict) -> tuple[list[Block], dict]:
    """Layout-aware PDF extraction using pdfplumber word geometry."""
    import pdfplumber

    detectors = compile_detectors(taxonomy)
    blocks: list[Block] = []
    info: dict = {"pages": 0, "extract_ratio": 0.0, "has_text_layer": True, "error": None}
    try:
        with pdfplumber.open(path) as pdf:
            info["pages"] = len(pdf.pages)
            for pno, page in enumerate(pdf.pages, 1):
                words = page.extract_words(use_text_flow=False, keep_blank_chars=False,
                                           extra_attrs=["size"])
                if not words:
                    continue
                # group words into visual lines by rounded top coordinate
                lines: dict[int, list[dict]] = {}
                for w in words:
                    key = int(round(w["top"] / 3.0))
                    lines.setdefault(key, []).append(w)
                ordered = [lines[k] for k in sorted(lines)]
                body_sizes = Counter(round(w.get("size") or 0) for w in words)
                common_size = body_sizes.most_common(1)[0][0] if body_sizes else 0
                for line_words in ordered:
                    text = " ".join(w["text"] for w in sorted(line_words, key=lambda x: x["x0"]))
                    text = re.sub(r"\s+", " ", text).strip()
                    if not text:
                        continue
                    size = round(sum((w.get("size") or 0) for w in line_words) / len(line_words))
                    blocks.append(_classify(text, page=pno, size=size,
                                            common_size=common_size, detectors=detectors))
            chars = sum(len(b.text) for b in blocks)
            info["extract_ratio"] = round(chars / max(1, info["pages"]), 1)
            info["has_text_layer"] = chars > 200 * max(1, info["pages"]) * 0.15
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
    return blocks, info


def _html_parser_cls():
    """selectolax >= 1.0 exposes LexborHTMLParser; fall back to the legacy name."""
    try:
        from selectolax.lexbor import LexborHTMLParser
        return LexborHTMLParser
    except ImportError:
        from selectolax.parser import HTMLParser
        return HTMLParser


def blocks_from_html(path: Path, taxonomy: dict) -> tuple[list[Block], dict]:
    """HTML extraction preserving heading hierarchy, paragraphs and list items.

    NOTE: an earlier version walked `root.traverse()`. That only yields *element*
    nodes, so real `<p>` paragraphs were never visited, and a "short title-case
    line" heuristic then mislabelled body text as headings. It selected semantic
    tags explicitly now, ordered by document position, with duplicate suppression.
    """
    HTMLParser = _html_parser_cls()

    detectors = compile_detectors(taxonomy)
    html = decode_html(path.read_bytes())
    tree = HTMLParser(html)
    for tag in ("script", "style", "noscript", "nav", "header", "footer", "aside", "form"):
        for node in tree.css(tag):
            node.decompose()

    # Pick the content container. Picking purely by text length can select a wrapper
    # whose bulk is a navigation list, so containers dense in list items are
    # penalised: policy prose is paragraphs and headings, navigation is <li> chips.
    best_root, best_score = None, float("-inf")
    for sel in ("main", "article", "[role=main]", "#main-content", "#content",
                ".content-panel", ".entry-content", ".page-content", ".main", "div.container"):
        nodes = tree.css(sel)
        if not nodes:
            continue
        for node in nodes:
            text_len = len(node.text() or "")
            li_count = len(node.css("li"))
            score = text_len - 25 * li_count
            if score > best_score:
                best_root, best_score = node, score
    root = best_root or tree.body or tree.root

    blocks: list[Block] = []
    if root is None:
        return blocks, {"pages": 1, "extract_ratio": 0.0, "has_text_layer": False, "error": "no_root"}

    # ---- collect candidates, then order them by real document position ---------
    # NOTE: order must come from a DOM-order text walk. Assigning an incrementing
    # counter while iterating `root.css("h1")`, then `root.css("h2")` ... puts every
    # heading before every paragraph, which scrambled section_path (headings from the
    # end of the page appeared to precede body text from the top). selectolax exposes
    # no node index, so position is recovered by matching element text in walk order.
    candidates: list[Block] = []

    def add(node, block: Block) -> None:
        candidates.append(block)

    heading_map = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
    for tag, level in heading_map.items():
        for node in root.css(tag):
            text = re.sub(r"\s+", " ", (node.text() or "")).strip()
            if text:
                add(node, Block(block_type="heading", text=text, heading_level=level, page=1))

    # list items: skip wrappers that nest a paragraph or a nested list, because the
    # paragraph / inner items are extracted separately and would otherwise duplicate
    for node in root.css("li"):
        if node.css("p") or node.css("ul, ol"):
            continue
        text = re.sub(r"\s+", " ", (node.text() or "")).strip()
        if text:
            add(node, _classify(text, page=1, block_type="list_item", detectors=detectors))

    # paragraphs, then leaf divs/sections/blockquotes
    for node in root.css("p"):
        text = re.sub(r"\s+", " ", (node.text() or "")).strip()
        if len(text) > 1:
            add(node, _classify(text, page=1, detectors=detectors,
                                allow_heading_guess=False))

    for sel in ("div", "section", "blockquote"):
        for node in root.css(sel):
            if node.css("p, li, table, ul, ol, div"):
                continue                      # container, not a leaf block
            text = re.sub(r"\s+", " ", (node.text() or "")).strip()
            if len(text) > 1:
                add(node, _classify(text, page=1, detectors=detectors,
                                    allow_heading_guess=False))

    for node in root.css("tr"):
        cells = [re.sub(r"\s+", " ", (c.text() or "")).strip() for c in node.css("td, th")]
        cells = [c for c in cells if c]
        if cells:
            add(node, Block(block_type="table", text=" | ".join(cells), page=1))

    # ---- order by real document position, then drop duplicates ----------------
    order_of: dict[str, int] = {}
    pos = 0
    try:
        for node in root.traverse(include_text=True):
            if node.tag == "-text":
                txt = re.sub(r"\s+", " ", (node.text_content or "")).strip()
                if txt and txt not in order_of:
                    order_of[txt] = pos
                    pos += 1
    except Exception:  # noqa: BLE001
        order_of = {}

    def sort_key(b: Block) -> tuple[int, int]:
        idx = order_of.get(b.text)
        return (0, idx) if idx is not None else (1, len(candidates))

    ordered = sorted(candidates, key=sort_key)

    seen: set[str] = set()
    for b in ordered:
        if b.text in seen:
            continue
        seen.add(b.text)
        blocks.append(b)

    chars = sum(len(b.text) for b in blocks)
    return blocks, {"pages": 1, "extract_ratio": chars, "has_text_layer": chars > 800, "error": None}


def _classify(text: str, page: int, detectors: dict, size: int = 0, common_size: int = 0,
              block_type: str | None = None, allow_heading_guess: bool = True) -> Block:
    """Classify one text block.

    `allow_heading_guess` should be False for HTML: real heading tags are available
    there, and guessing from capitalisation mislabelled body paragraphs as headings.
    For PDFs font size is the only signal, so the guess stays on.
    """
    m = CLAUSE_HEAD_RE.match(text)
    is_heading = False
    if allow_heading_guess:
        # A heading is short AND does not read like prose. Both a word cap and the
        # absence of sentence-final punctuation are required, plus a hard ceiling, so
        # that a large-font (or truncated) body sentence cannot become a heading and
        # pollute every section_path beneath it.
        words = len(text.split())
        prose_tail = text.rstrip().endswith((".", "?", "!", ";", ":"))
        short = words <= 12 and len(text) <= 120 and not prose_tail
        # hard ceiling: nothing this long is a heading, whatever the font says
        plausible_heading = words <= 16 and len(text) <= 140
        if size and common_size and size > common_size + 0.5 and short and plausible_heading:
            is_heading = True
        if HEADING_ONLY_RE.match(text) and short and plausible_heading and not m:
            is_heading = True

    if block_type:
        btype = block_type
    elif is_heading:
        btype = "heading"
    elif m:
        btype = "clause"
    else:
        btype = "paragraph"

    label = ""
    if m:
        label = (m.group("num") or m.group("art") or m.group("sec")
                 or m.group("rule") or m.group("alnum") or m.group("paren") or "").strip()

    flags = {}
    if detectors["exception"].search(text):
        flags["has_exception"] = True
    if detectors["crossref"].search(text):
        flags["has_crossref"] = True
    if detectors["alt_evidence"].search(text):
        flags["has_alt_evidence"] = True
    return Block(block_type=btype, text=text, page=page, clause_label=label, flags=flags)


# ---------------------------------------------------------- header/footer removal
def filter_noise_blocks(blocks: list[Block], min_chars: int, detectors: dict,
                        doc_id: str) -> tuple[list[Block], list[dict]]:
    """Drop navigation crumbs that are too short to be a policy provision.

    Index pages contribute many 3-10 character fragments ("Policies", "Grades").
    They are not answerable evidence and would inflate chunk counts, so they are
    removed unless they are long enough, or carry a clause label, or contain
    exception / cross-reference / alternative-evidence language.
    """
    kept: list[Block] = []
    dropped: list[dict] = []
    for b in blocks:
        text = b.text.strip()
        if len(text) >= min_chars or b.clause_label or b.flags:
            kept.append(b)
            continue
        dropped.append({
            "doc_id": doc_id, "page": b.page, "where": "block",
            "text": text, "confidence": "high", "status": "removed",
            "reason": f"short non-clause block ({len(text)} < {min_chars} chars)",
        })
    return kept, dropped


def remove_headers_footers(blocks: list[Block], cfg: dict, detectors: dict,
                           doc_id: str) -> tuple[list[Block], list[dict]]:
    hf = cfg["header_footer"]
    pages: dict[int, list[Block]] = {}
    for b in blocks:
        pages.setdefault(b.page, []).append(b)

    edge = hf["edge_lines"]
    repeat_ratio = hf["repeat_page_ratio"]
    protect = re.compile(hf["protect_regex"])

    # count how often each edge-position text appears across pages
    counts: Counter = Counter()
    for pno, plist in pages.items():
        for b in plist[:edge]:
            counts[("top", b.text)] += 1
        for b in plist[-edge:]:
            counts[("bottom", b.text)] += 1

    n_pages = max(1, len(pages))
    candidates: set[tuple[str, str]] = set()
    for (where, text), c in counts.items():
        if n_pages >= 3 and c / n_pages >= repeat_ratio:
            candidates.add((where, text))
        elif n_pages < 3 and c >= 2:
            candidates.add((where, text))
        if PAGE_NUM_RE.match(text):
            candidates.add((where, text))

    kept: list[Block] = []
    removed: list[dict] = []
    for pno, plist in pages.items():
        last = len(plist) - 1
        for idx, b in enumerate(plist):
            where = "top" if idx < edge else ("bottom" if idx > last - edge else "middle")
            hit = (where, b.text) in candidates or PAGE_NUM_RE.match(b.text)
            if not hit:
                kept.append(b)
                continue
            protected = bool(protect.search(b.text)) or bool(b.flags)
            removed.append({
                "doc_id": doc_id, "page": pno, "where": where, "text": b.text,
                "confidence": "high" if (where, b.text) in candidates else "medium",
                "status": "REVIEW_PROTECTED" if protected else "removed",
                "reason": ("matches protect_regex or carries exception/crossref/alt flag"
                           if protected else "repeated page furniture or page number"),
            })
            if protected:
                kept.append(b)          # never drop protected lines
    return kept, removed


def clean_section_path(parts: list[str], max_depth: int = 3, max_part_len: int = 100) -> str:
    """Normalise a heading stack into a short, human-readable section path.

    Raw heading text arrives with trailing punctuation, doubled whitespace and
    (from PDF extraction) occasional large-font lead-in sentences. The path is only
    used for display and grouping — it never affects span offsets or containment —
    so the goal is readability: drop noise, cap depth and length, remove repeats.
    """
    out: list[str] = []
    for raw in parts:
        s = re.sub(r"\s+", " ", (raw or "")).strip()
        s = s.strip(" \t.:;,>|-–—")               # trailing separators and句号
        if not s or len(s) > max_part_len:
            continue
        if s.lower() in {p.lower() for p in out}:   # drop repeated levels
            continue
        out.append(s)
    return " > ".join(out[-max_depth:])


# ------------------------------------------------------------------ clause spans
def build_clause_records(blocks: list[Block], detectors: dict, doc_id: str,
                         tier: str, institution: str,
                         max_chars: int = 1200) -> list[dict]:
    """Assign stable clause ids and evidence-space spans.

    A clause record covers a clause head block plus following paragraph/list
    blocks until the next heading or clause head. Spans are half-open
    [char_start, char_end) in the evidence text, which is the concatenation of
    the retained block texts joined by newlines.

    `max_chars` bounds how much text one clause may absorb. This matters because
    gold spans are matched by full containment: a clause spanning 34,000 characters
    can never be contained by any retrieved chunk (the model maximum is 512 tokens),
    so such a clause could never register as a hit and would silently make questions
    unanswerable. Continuation stops at `max_chars`, which keeps clause boundaries
    aligned with real block boundaries.

    A single block longer than `max_chars` cannot be split here without inventing a
    boundary, so it is kept whole and flagged `oversized` for manual review.
    """
    records: list[dict] = []
    section_stack: list[str] = []
    cursor = 0
    offsets: list[tuple[int, int]] = []   # (block_index, evidence_offset)
    parts: list[str] = []
    for i, b in enumerate(blocks):
        offsets.append((i, cursor))
        parts.append(b.text)
        cursor += len(b.text) + 1          # +1 for the joining newline
    evidence_text = "\n".join(parts)

    current: dict | None = None
    seq = 0
    label_seen: Counter = Counter()

    def start_record(block: Block, start: int, end: int) -> dict:
        nonlocal seq
        seq += 1
        label = block.clause_label or f"c{seq}"
        # Labels like "A." or "1." legitimately repeat across sections of the same
        # document, so the id must be made unique while staying human-readable.
        label_seen[label] += 1
        occurrence = label_seen[label]
        cid = f"{doc_id}#{label}" if occurrence == 1 else f"{doc_id}#{label}~{occurrence}"
        return {
            "doc_id": doc_id,
            "tier": tier,
            "institution": institution,
            "clause_id": cid,
            "clause_label": block.clause_label,
            "section_path": clean_section_path(section_stack),
            "block_type": block.block_type,
            "page": block.page,
            "char_start": start,
            "char_end": end,
            "text": block.text,
            "has_exception": bool(block.flags.get("has_exception")),
            "has_crossref": bool(block.flags.get("has_crossref")),
            "has_alt_evidence": bool(block.flags.get("has_alt_evidence")),
            "boilerplate": False,
        }

    for i, b in enumerate(blocks):
        start = offsets[i][1]
        end = start + len(b.text)

        if b.block_type == "heading":
            if current:
                records.append(current)
                current = None
            level = b.heading_level or 1
            section_stack = section_stack[: level - 1]
            section_stack.append(b.text)
            continue

        if b.block_type in ("clause", "list_item") or current is None:
            if current:
                records.append(current)
            current = start_record(b, start, end)
        else:
            # continuation: extend across paragraph/list/table blocks, but only while
            # the clause stays small enough to be retrievable as a single unit
            would_be = end - current["char_start"]
            if would_be > max_chars:
                records.append(current)
                current = start_record(b, start, end)
                continue
            current["char_end"] = end
            current["text"] = evidence_text[current["char_start"]:end]
            if b.flags.get("has_exception"):
                current["has_exception"] = True
            if b.flags.get("has_crossref"):
                current["has_crossref"] = True
            if b.flags.get("has_alt_evidence"):
                current["has_alt_evidence"] = True
            if b.block_type == "table" and current["block_type"] != "table":
                current["block_type"] = "table"

    if current:
        records.append(current)

    for r in records:
        r["char_len"] = r["char_end"] - r["char_start"]
        r["oversized"] = r["char_len"] > max_chars
        r["sha256"] = sha256_text(r["text"])
        r["evidence_sha256"] = sha256_text(evidence_text[r["char_start"]:r["char_end"]])
    return records


# ------------------------------------------------------------------ main
def process_doc(doc_id: str, raw_dir: Path, meta: dict, cfg: dict, taxonomy: dict,
                detectors: dict, paths: dict) -> dict:
    fmt = meta.get("format", "html")
    raw_file = raw_dir / ("raw.pdf" if fmt == "pdf" else "raw.html")
    if not raw_file.exists():
        return {"doc_id": doc_id, "status": "missing_raw"}

    if fmt == "pdf":
        blocks, info = blocks_from_pdf(raw_file, taxonomy)
    else:
        blocks, info = blocks_from_html(raw_file, taxonomy)

    if not blocks:
        return {"doc_id": doc_id, "status": "no_blocks", "info": info}

    kept, removed = remove_headers_footers(blocks, cfg, detectors, doc_id)
    min_chars = int(cfg.get("quality", {}).get("min_clause_chars", 80))
    kept, noise = filter_noise_blocks(kept, min_chars, detectors, doc_id)
    removed += noise
    tier = meta.get("tier", "")
    institution = meta.get("institution", "")
    max_clause_chars = int(cfg.get("quality", {}).get("max_clause_chars", 1200))
    clauses = build_clause_records(kept, detectors, doc_id, tier, institution,
                                   max_chars=max_clause_chars)

    # rebuild the evidence text exactly as build_clause_records saw it
    parts = [b.text for b in kept]
    evidence_text = "\n".join(parts)
    layers = build_text_layers(evidence_text, cfg["text_layers"])

    # verify every clause span still matches its own slice (offset integrity)
    span_errors = []
    for c in clauses:
        slice_ = evidence_text[c["char_start"]:c["char_end"]]
        if sha256_text(slice_) != c["evidence_sha256"]:
            span_errors.append(c["clause_id"])

    # boilerplate detection across clauses (per document; cross-doc pass is in 06)
    bl_cfg = cfg["dedup"]["clause_boilerplate"]
    if bl_cfg.get("enabled", True):
        sigs = [(c, minhash_signature(c["text"], cfg["dedup"]["document_minhash"]["shingle_size"],
                                      cfg["dedup"]["document_minhash"]["num_hashes"]))
                for c in clauses]
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                if minhash_similarity(sigs[i][1], sigs[j][1]) >= bl_cfg["jaccard_threshold"]:
                    sigs[j][0]["boilerplate"] = True

    write_json(paths["extracted"] / f"{doc_id}.json", {
        "doc_id": doc_id, "tier": tier, "institution": institution,
        "format": fmt, "info": info,
        "blocks": [b.__dict__ for b in kept],
        "blocks_removed": len(blocks) - len(kept),
        "clauses": clauses,
        "evidence_chars": len(evidence_text),
        "retrieval_chars": len(layers.retrieval),
        "charmap_len": len(layers.charmap),
        "span_errors": span_errors,
    })

    ev_rows = []
    for c in clauses:
        row = dict(c)
        row["layer"] = "evidence"
        ev_rows.append(row)
    write_jsonl(paths["cleaned"] / f"{doc_id}.evidence.jsonl", ev_rows)

    # retrieval layer: clauses re-anchored through the charmap
    ret_rows = []
    for c in clauses:
        rs, re_ = _evidence_to_retrieval(layers, c["char_start"], c["char_end"])
        row = dict(c)
        row.update({"layer": "retrieval", "ret_char_start": rs, "ret_char_end": re_,
                    "ret_text": layers.retrieval[rs:re_]})
        ret_rows.append(row)
    write_jsonl(paths["cleaned"] / f"{doc_id}.retrieval.jsonl", ret_rows)

    write_jsonl(paths["cleaned"] / f"{doc_id}.charmap.jsonl",
                [{"doc_id": doc_id, "n_evidence": len(layers.evidence),
                  "n_retrieval": len(layers.retrieval), "charmap": layers.charmap}])

    usable_clauses = [c for c in clauses
                      if not c.get("boilerplate") and c.get("char_len", 0) >= 80]
    return {
        "doc_id": doc_id, "status": "ok", "tier": tier, "institution": institution,
        "format": fmt, "pages": info.get("pages", 0),
        "extract_ratio": info.get("extract_ratio", 0),
        "has_text_layer": info.get("has_text_layer", False),
        "blocks_total": len(blocks), "blocks_kept": len(kept),
        "blocks_removed": len(blocks) - len(kept),
        "removed_review": sum(1 for r in removed if r["status"] == "REVIEW_PROTECTED"),
        "clauses": len(clauses),
        "clauses_with_exception": sum(1 for c in clauses if c["has_exception"]),
        "clauses_with_crossref": sum(1 for c in clauses if c["has_crossref"]),
        "clauses_with_alt": sum(1 for c in clauses if c["has_alt_evidence"]),
        "boilerplate_clauses": sum(1 for c in clauses if c["boilerplate"]),
        "evidence_chars": len(evidence_text),
        "retrieval_chars": len(layers.retrieval),
        "span_errors": len(span_errors),
        # clause-level authoritative counts, consumed by reports/clause_stats.csv
        "clauses_total": len(clauses),
        "clauses_usable": len(usable_clauses),
        "clauses_exception": sum(1 for c in clauses if c["has_exception"]),
        "clauses_crossref": sum(1 for c in clauses if c["has_crossref"]),
        "clauses_alt": sum(1 for c in clauses if c["has_alt_evidence"]),
        "_removed": removed,
    }


def _evidence_to_retrieval(layers, ev_start: int, ev_end: int) -> tuple[int, int]:
    """Map an evidence span to the enclosing retrieval span.

    Because normalisation can expand characters, the mapping is: first retrieval
    index whose evidence index >= ev_start, and last retrieval index whose
    evidence index < ev_end (inclusive -> +1).
    """
    cm = layers.charmap
    if not cm:
        return 0, 0
    import bisect
    rs = bisect.bisect_left(cm, ev_start)
    re_ = bisect.bisect_left(cm, ev_end)
    return rs, max(rs, re_)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--doc-id", action="append", default=[])
    ap.add_argument("--tier", action="append", default=[])
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)
    taxonomy = load_taxonomy(root)
    detectors = compile_detectors(taxonomy)

    raw_root = paths["raw"]
    metas: list[tuple[str, Path, dict]] = []
    for tier_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        for doc_dir in sorted(p for p in tier_dir.iterdir() if p.is_dir()):
            meta_file = doc_dir / "meta.json"
            if not meta_file.exists():
                continue
            metas.append((doc_dir.name, doc_dir, read_json(meta_file)))

    if args.doc_id:
        want = set(args.doc_id)
        metas = [m for m in metas if m[0] in want]
    if args.tier:
        want = set(args.tier)
        metas = [m for m in metas if m[2].get("tier") in want]

    if not metas:
        print("no documents found under raw/. Run 01_collect.py first.", file=sys.stderr)
        return 2

    results = []
    all_removed: list[dict] = []
    for i, (doc_id, doc_dir, meta) in enumerate(metas, 1):
        res = process_doc(doc_id, doc_dir, meta, cfg, taxonomy, detectors, paths)
        all_removed.extend(res.pop("_removed", []))
        results.append(res)
        print(f"[{i:2d}/{len(metas)}] {res.get('status','?'):12s} clauses={res.get('clauses',0):>4} "
              f"exc={res.get('clauses_with_exception',0):>3} xr={res.get('clauses_with_crossref',0):>3} "
              f"span_err={res.get('span_errors',0)}  {doc_id}", flush=True)

    if all_removed:
        log_path = paths["cleaned"] / "removed_lines.log"
        exists = log_path.exists()
        with log_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_removed[0].keys()))
            if not exists:
                w.writeheader()
            for r in all_removed:
                w.writerow(r)
        # also write a review-only extract so protected lines are easy to inspect
        review = [r for r in all_removed if r["status"] == "REVIEW_PROTECTED"]
        if review:
            write_json(paths["reports"] / "protected_lines_review.json", review)

    qfields = ["doc_id", "status", "tier", "institution", "format", "pages", "extract_ratio",
               "has_text_layer", "blocks_total", "blocks_kept", "blocks_removed", "removed_review",
               "clauses", "clauses_with_exception", "clauses_with_crossref", "clauses_with_alt",
               "boilerplate_clauses", "evidence_chars", "retrieval_chars", "span_errors"]
    with (paths["reports"] / "quality_report.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=qfields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in qfields})

    # Authoritative clause-level statistics. These are measured on the cleaned
    # evidence layer, so they are the single source of truth for every downstream
    # report. The collector's raw regex hit counts are NOT clause counts (they count
    # numbered-looking lines in the whole page, navigation included) and were
    # previously mistaken for them.
    csfields = ["doc_id", "tier", "institution", "clauses_total", "clauses_usable",
                "clauses_exception", "clauses_crossref", "clauses_alt", "clauses_boilerplate",
                "evidence_chars"]
    with (paths["reports"] / "clause_stats.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csfields)
        w.writeheader()
        for r in results:
            if r.get("status") != "ok":
                continue
            w.writerow({k: r.get(k, "") for k in csfields})

    ok = [r for r in results if r.get("status") == "ok"]
    span_bad = [r for r in ok if r.get("span_errors")]
    no_text = [r for r in ok if not r.get("has_text_layer")]
    print("\n=== extraction summary ===")
    print(f"  documents processed     : {len(results)}")
    print(f"  ok                      : {len(ok)}")
    print(f"  clause span errors      : {len(span_bad)}  (must be 0)")
    print(f"  suspicious text layer   : {len(no_text)}  (scanned PDFs -> pending_review)")
    print(f"  protected lines kept    : {sum(r.get('removed_review', 0) for r in results)}")
    print(f"  total clauses           : {sum(r.get('clauses', 0) for r in ok)}")
    print(f"  clauses w/ exception    : {sum(r.get('clauses_with_exception', 0) for r in ok)}")
    print(f"  clauses w/ crossref     : {sum(r.get('clauses_with_crossref', 0) for r in ok)}")
    print(f"  clauses w/ alt evidence : {sum(r.get('clauses_with_alt', 0) for r in ok)}")
    print(f"\nwrote {paths['reports'] / 'quality_report.csv'}")
    return 1 if span_bad else 0


if __name__ == "__main__":
    sys.exit(main())
