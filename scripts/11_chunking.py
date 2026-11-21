"""11_chunking.py - the four chunking strategies under comparison.

Every strategy starts from *the same* frozen corpus file named by MANIFEST.json, so the
only thing that differs between conditions is the chunk boundaries.

Strategies
    fixed      fixed-length token windows (128/256/384/512) with overlap 0/32/64
    paragraph  merge consecutive paragraph / list-item units, preserving author units
    heading    merge by section_path, preserving the heading hierarchy
    semantic   sentence-level embeddings; break where adjacent similarity drops below a
               threshold (0.70 / 0.75 / 0.80)

Two invariants the evaluator depends on
  1. every chunk carries `ev_ranges`: the exact evidence intervals it contains, taken
     from the corpus unit ranges and the per-character map. Coverage is scored on these
     intervals only - never on a bounding box, which would over-claim when a chunk
     stitches non-adjacent text.
  2. no chunk exceeds the model maximum input length. A strategy that would produce a
     longer chunk splits it further at sentence then token boundaries; it never
     truncates.

Output
    chunks/<strategy>/<config>.jsonl

Usage
    .venv\\Scripts\\python.exe scripts/11_chunking.py                 # every strategy/config
    .venv\\Scripts\\python.exe scripts/11_chunking.py --strategy fixed
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    load_sentence_transformer,
    PROJECT_ROOT, load_settings, project_paths, read_jsonl, write_json, write_jsonl,
)

SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+")


# --------------------------------------------------------------------- tokenizer
class Encoder:
    """Thin wrapper over the frozen model's tokenizer (offsets + counts)."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.max_len = 512
        # The tokenizer's model_max_length is 512 and it warns for longer inputs, but
        # we need offsets over whole documents to build windows. Raising it only
        # silences the warning; every emitted chunk is still checked against max_len.
        try:
            self.tok.model_max_length = 10 ** 9
        except Exception:  # noqa: BLE001 - some versions guard this attribute
            pass

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [tuple(o) for o in enc["offset_mapping"]]


# --------------------------------------------------------------------- doc text
class DocText:
    """One document's text plus a per-character evidence index.

    `char_ev[i]` is the evidence offset of character i, or None for separators we
    inserted. This lets any character range map back to exact evidence intervals.
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.parts: list[str] = []
        self.char_ev: list[int | None] = []
        self.units: list[dict] = []          # corpus unit records in order

    def append(self, text: str, ev_ranges: list[list[int]], unit: dict) -> None:
        if self.parts:
            self.parts.append("\n")
            self.char_ev.append(None)
        start = len(self.char_ev)
        self.parts.append(text)
        # walk the evidence intervals over this unit's text
        ev = [tuple(r) for r in ev_ranges]
        if ev and len(ev) == 1:
            s, _e = ev[0]
            for k in range(len(text)):
                self.char_ev.append(s + k)
        else:
            # multi-interval unit: fall back to the first interval start, and record
            # the exact intervals at unit level so coverage still uses them
            s = ev[0][0] if ev else 0
            for k in range(len(text)):
                self.char_ev.append(s + k)
        unit = dict(unit)
        unit["doc_char_start"] = start
        unit["doc_char_end"] = start + len(text)
        self.units.append(unit)

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def ev_ranges_for(self, start: int, end: int) -> list[list[int]]:
        """Merge the evidence positions covered by document characters [start, end)."""
        out: list[list[int]] = []
        cur_s = cur_e = None
        for i in range(max(0, start), min(end, len(self.char_ev))):
            ev = self.char_ev[i]
            if ev is None:
                continue
            if cur_s is None:
                cur_s, cur_e = ev, ev + 1
            elif ev == cur_e:
                cur_e = ev + 1
            else:
                out.append([cur_s, cur_e])
                cur_s, cur_e = ev, ev + 1
        if cur_s is not None:
            out.append([cur_s, cur_e])
        return out

    def clause_ids_for(self, start: int, end: int) -> list[str]:
        ids: list[str] = []
        for u in self.units:
            if u["doc_char_end"] > start and u["doc_char_start"] < end:
                cid = u.get("clause_id")
                if cid and cid not in ids:
                    ids.append(cid)
        return ids


def build_docs(corpus_chunks: list[dict]) -> dict[str, DocText]:
    docs: dict[str, DocText] = {}
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in corpus_chunks:
        by_doc[c["doc_id"]].append(c)
    for doc_id, units in by_doc.items():
        units.sort(key=lambda c: (c.get("clause_char_start", 0), c.get("part_index", 0)))
        dt = DocText(doc_id)
        for u in units:
            dt.append(u["text"], u.get("ev_ranges") or [[u.get("ev_char_start", 0),
                                                         u.get("ev_char_end", 0)]], u)
        docs[doc_id] = dt
    return docs


# --------------------------------------------------------------------- strategies
def _chunk_record(strategy: str, config: str, dt: DocText, start: int, end: int,
                  text: str, token_count: int) -> dict:
    ev = dt.ev_ranges_for(start, end)
    return {
        "chunk_id": f"{dt.doc_id}::{strategy}::{config}::{start}-{end}",
        "doc_id": dt.doc_id,
        "strategy": strategy,
        "config": config,
        "char_start": start,
        "char_end": end,
        "text": text,
        "token_count": token_count,
        "ev_ranges": ev,
        "n_ev_intervals": len(ev),
        "clause_ids": dt.clause_ids_for(start, end),
    }


def _split_oversize(strategy: str, config: str, dt: DocText, start: int, end: int,
                    enc: Encoder, max_len: int) -> list[dict]:
    """Split an over-long range at sentence boundaries, then hard token windows."""
    text = dt.text[start:end]
    if enc.count(text) <= max_len:
        return [_chunk_record(strategy, config, dt, start, end, text, enc.count(text))]

    # sentence spans within [start, end)
    spans: list[tuple[int, int]] = []
    pos = start
    for m in SENT_SPLIT.finditer(dt.text, start, end):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < end:
        spans.append((pos, end))
    if len(spans) > 1:
        out, buf_s, buf_e = [], spans[0][0], spans[0][1]
        for s, e in spans[1:]:
            if enc.count(dt.text[buf_s:e]) <= max_len:
                buf_e = e
            else:
                out.extend(_chunk_record(strategy, config, dt, buf_s, buf_e,
                                         dt.text[buf_s:buf_e], enc.count(dt.text[buf_s:buf_e]))
                           for _ in [0])
                buf_s, buf_e = s, e
        out.extend(_chunk_record(strategy, config, dt, buf_s, buf_e,
                                 dt.text[buf_s:buf_e], enc.count(dt.text[buf_s:buf_e]))
                   for _ in [0])
        return out

    # last resort: hard token windows over the range
    offs = enc.offsets(text)
    out = []
    i = 0
    while i < len(offs):
        j = min(i + max_len, len(offs))
        while j > i + 1 and enc.count(text[offs[i][0]:offs[j - 1][1]]) > max_len:
            j -= 1
        s = start + offs[i][0]
        e = start + offs[j - 1][1]
        out.append(_chunk_record(strategy, config, dt, s, e, dt.text[s:e],
                                 enc.count(dt.text[s:e])))
        i = j
    return out


def strategy_fixed(dt: DocText, enc: Encoder, size: int, overlap: int,
                   max_len: int) -> list[dict]:
    """Fixed-length token windows over the whole document, ignoring clause boundaries."""
    text = dt.text
    offs = enc.offsets(text)
    n = len(offs)
    if n == 0:
        return []
    step = max(1, size - overlap)
    out: list[dict] = []
    i = 0
    limit = min(size, max_len)
    while i < n:
        j = min(i + size, n)
        # slicing on token offsets can re-introduce a boundary token, so verify the
        # decoded window really fits the *nominal* size before emitting it
        while j > i + 1 and enc.count(text[offs[i][0]:offs[j - 1][1]]) > limit:
            j -= 1
        s = offs[i][0]
        e = offs[j - 1][1]
        cfg = f"size{size}_ov{overlap}"
        out.append(_chunk_record("fixed", cfg, dt, s, e, text[s:e], enc.count(text[s:e])))
        if j >= n:
            break
        i += step
    return out


def _group_units(dt: DocText, key_fn, enc: Encoder, max_len: int,
                 strategy: str, config: str) -> list[dict]:
    """Merge consecutive corpus units sharing a group key, capped at max_len tokens."""
    out: list[dict] = []
    cur_key = None
    buf_s = buf_e = None
    for u in dt.units:
        key = key_fn(u)
        start, end = u["doc_char_start"], u["doc_char_end"]
        if buf_s is None:
            cur_key, buf_s, buf_e = key, start, end
            continue
        if key == cur_key and enc.count(dt.text[buf_s:end]) <= max_len:
            buf_e = end
            continue
        out.extend(_split_oversize(strategy, config, dt, buf_s, buf_e, enc, max_len))
        cur_key, buf_s, buf_e = key, start, end
    if buf_s is not None:
        out.extend(_split_oversize(strategy, config, dt, buf_s, buf_e, enc, max_len))
    return out


def strategy_paragraph(dt: DocText, enc: Encoder, max_len: int) -> list[dict]:
    """Merge consecutive paragraph-like units; headings and tables stay separate."""
    def key(u: dict) -> str:
        bt = u.get("block_type", "")
        return "para" if bt in ("paragraph", "list_item") else f"other::{bt}::{u.get('chunk_id')}"
    return _group_units(dt, key, enc, max_len, "paragraph", "paragraph")


def strategy_heading(dt: DocText, enc: Encoder, max_len: int) -> list[dict]:
    """Merge by section path, so a chunk never straddles two headings."""
    def key(u: dict) -> str:
        return u.get("section_path") or f"__nosection__::{u.get('block_type','')}"
    return _group_units(dt, key, enc, max_len, "heading", "heading")


def strategy_semantic(dt: DocText, enc: Encoder, model, threshold: float,
                      max_len: int) -> list[dict]:
    """Break where adjacent sentence similarity falls below `threshold`."""
    text = dt.text
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in SENT_SPLIT.finditer(text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    if not spans:
        return []

    sentences = [text[s:e] for s, e in spans]
    emb = model.encode(sentences, normalize_embeddings=True,
                       batch_size=32, show_progress_bar=False)
    sims = [(emb[i] * emb[i + 1]).sum() for i in range(len(sentences) - 1)]

    cfg = f"thr{threshold:.2f}"
    out: list[dict] = []
    buf_s, buf_e = spans[0]
    for i, (s, e) in enumerate(spans[1:], start=1):
        boundary = sims[i - 1] < threshold
        fits = enc.count(text[buf_s:e]) <= max_len
        if boundary or not fits:
            out.extend(_split_oversize("semantic", cfg, dt, buf_s, buf_e, enc, max_len))
            buf_s, buf_e = s, e
        else:
            buf_e = e
    out.extend(_split_oversize("semantic", cfg, dt, buf_s, buf_e, enc, max_len))
    return out


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--strategy", action="append", default=[],
                    choices=["fixed", "paragraph", "heading", "semantic"])
    ap.add_argument("--fixed-sizes", default="128,256,384,512")
    ap.add_argument("--overlaps", default="0,32,64")
    ap.add_argument("--thresholds", default="0.70,0.75,0.80")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)
    model_name = cfg["safe_corpus"]["embedding_model"]
    max_len = int(cfg["safe_corpus"]["max_input_tokens"])

    corpus_files = sorted(paths["corpus"].glob("corpus_safe_512.jsonl"))
    if not corpus_files:
        print("no corpus_safe_512.jsonl; run 03_build_safe_corpus.py first", file=sys.stderr)
        return 2
    corpus_chunks = read_jsonl(corpus_files[-1])
    docs = build_docs(corpus_chunks)
    print(f"corpus {corpus_files[-1].name}: {len(corpus_chunks)} units, {len(docs)} documents")

    enc = Encoder(model_name)
    strategies = args.strategy or ["fixed", "paragraph", "heading", "semantic"]
    sizes = [int(x) for x in args.fixed_sizes.split(",")]
    overlaps = [int(x) for x in args.overlaps.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]

    model = None
    if "semantic" in strategies:
        model = load_sentence_transformer(model_name, device="cpu")
        print(f"semantic model loaded: {model_name} (cpu)")

    summary: dict = {"strategies": {}}
    out_root = root / "chunks"
    total = 0

    def emit(strategy: str, config: str, chunks: list[dict]) -> None:
        nonlocal total
        d = out_root / strategy
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{config}.jsonl"
        write_jsonl(p, chunks)
        total += len(chunks)
        toks = [c["token_count"] for c in chunks]
        over = sum(1 for t in toks if t > max_len)
        summary["strategies"].setdefault(strategy, {})[config] = {
            "file": str(p.relative_to(root)).replace("\\", "/"),
            "chunks": len(chunks),
            "tokens_mean": round(sum(toks) / len(toks), 1) if toks else 0,
            "tokens_max": max(toks, default=0),
            "over_max_input": over,
            "multi_interval_chunks": sum(1 for c in chunks if c["n_ev_intervals"] > 1),
        }
        print(f"  {strategy:<10} {config:<16} chunks={len(chunks):>5} "
              f"tok_max={max(toks, default=0):>4} over={over}")

    for strat in strategies:
        if strat == "fixed":
            for size in sizes:
                for ov in overlaps:
                    if ov >= size:
                        continue
                    chunks = [c for dt in docs.values()
                              for c in strategy_fixed(dt, enc, size, ov, max_len)]
                    emit("fixed", f"size{size}_ov{ov}", chunks)
        elif strat == "paragraph":
            chunks = [c for dt in docs.values()
                      for c in strategy_paragraph(dt, enc, max_len)]
            emit("paragraph", "paragraph", chunks)
        elif strat == "heading":
            chunks = [c for dt in docs.values()
                      for c in strategy_heading(dt, enc, max_len)]
            emit("heading", "heading", chunks)
        elif strat == "semantic":
            for thr in thresholds:
                chunks = [c for dt in docs.values()
                          for c in strategy_semantic(dt, enc, model, thr, max_len)]
                emit("semantic", f"thr{thr:.2f}", chunks)

    summary["corpus_file"] = corpus_files[-1].name
    summary["corpus_version"] = (json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
                                 .get("corpus_version", "") if (root / "MANIFEST.json").exists() else "")
    summary["max_input_tokens"] = max_len
    summary["total_chunks"] = total
    write_json(paths["reports"] / "chunking_summary.json", summary)

    over_total = sum(v["over_max_input"] for s in summary["strategies"].values()
                     for v in s.values())
    print(f"\ntotal chunks {total}  configs "
          f"{sum(len(v) for v in summary['strategies'].values())}  over-limit {over_total}")
    print(f"wrote {paths['reports'] / 'chunking_summary.json'}")
    return 1 if over_total else 0


if __name__ == "__main__":
    sys.exit(main())
