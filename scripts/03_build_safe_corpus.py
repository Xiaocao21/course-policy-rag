"""03_build_safe_corpus.py - shared 512-token corpus for all four chunking strategies.

Why this stage exists: the plan requires that no strategy silently receives more
model input than another. Any block longer than the model maximum is split
recursively (sentence boundary -> token boundary) and the reason is recorded.
Nothing is ever silently truncated.

Also emits the chunk-level metadata needed by the complete-chunk token-budget
simulation: every chunk carries its token count so that ranked chunks can be
admitted in order and selection stops when the next one does not fit.

Tokenisation:
  * default: the Hugging Face tokenizer of BAAI/bge-small-en-v1.5
  * fallback: --tokenizer regex (approximate; outputs are suffixed *_approx so an
    approximate run can never be mistaken for the frozen corpus)

Usage:
    .venv\\Scripts\\python.exe scripts\\03_build_safe_corpus.py
    .venv\\Scripts\\python.exe scripts\\03_build_safe_corpus.py --tokenizer regex
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_ROOT, load_settings, project_paths, read_json, read_jsonl,
    write_json, write_jsonl,
)

SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")
_TOKEN_APPROX = re.compile(r"\w+|[^\w\s]")


class Tokenizer:
    """Wrapper so the corpus build can run before the ML dependencies exist."""

    def __init__(self, model_name: str, mode: str = "huggingface"):
        self.model_name = model_name
        self.mode = mode
        self.approx = False
        self._tok = None
        if mode == "huggingface":
            try:
                from transformers import AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: could not load tokenizer '{model_name}' ({type(e).__name__}: {e})",
                      file=sys.stderr)
                print("         falling back to approximate tokenisation; outputs get _approx suffix",
                      file=sys.stderr)
                self.approx = True
        if self._tok is None:
            self.approx = True

    def count(self, text: str) -> int:
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False))
        return len(_TOKEN_APPROX.findall(text))

    def suffix(self) -> str:
        return "_approx" if self.approx else ""


def split_recursive(text: str, tok: Tokenizer, max_tokens: int, reason: str = "overlength",
                    depth: int = 0) -> list[dict]:
    """Split text into <= max_tokens pieces, recording why each split happened."""
    n = tok.count(text)
    if n <= max_tokens:
        return [{"text": text, "tokens": n, "split_reason": "" if depth == 0 else reason,
                 "split_depth": depth}]

    if depth > 6:                      # hard stop; still never truncate
        return [{"text": text, "tokens": n, "split_reason": f"{reason}:unsplittable",
                 "split_depth": depth}]

    # 1) try sentence boundaries
    sentences = [s for s in SENTENCE_END.split(text) if s.strip()]
    if len(sentences) > 1:
        return _pack(sentences, tok, max_tokens, f"{reason}>sentence", depth + 1)

    # 2) fall back to token boundaries (whitespace-safe chunking of the raw text)
    words = text.split(" ")
    if len(words) > 1:
        return _pack(words, tok, max_tokens, f"{reason}>token", depth + 1, joiner=" ")

    # 3) single unbreakable token run: character split as last resort
    return _char_split(text, tok, max_tokens, f"{reason}>char", depth + 1)


def _pack(units: list[str], tok: Tokenizer, max_tokens: int, reason: str,
          depth: int, joiner: str = " ") -> list[dict]:
    out: list[dict] = []
    buf: list[str] = []
    for unit in units:
        trial = joiner.join(buf + [unit])
        if buf and tok.count(trial) > max_tokens:
            piece = joiner.join(buf)
            out.extend(split_recursive(piece, tok, max_tokens, reason, depth))
            buf = [unit]
        else:
            buf.append(unit)
    if buf:
        piece = joiner.join(buf)
        out.extend(split_recursive(piece, tok, max_tokens, reason, depth))
    return out


def _char_split(text: str, tok: Tokenizer, max_tokens: int, reason: str, depth: int) -> list[dict]:
    out: list[dict] = []
    step = max(64, len(text) // max(1, (tok.count(text) // max_tokens) + 1))
    for i in range(0, len(text), step):
        piece = text[i:i + step]
        n = tok.count(piece)
        out.append({"text": piece, "tokens": n, "split_reason": reason, "split_depth": depth})
    return out


def _locate_pieces(source: str, pieces: list[dict]) -> bool:
    """Record each piece's character range within `source`.

    Pieces are produced by splitting in order and (for sentence/word splits) are
    exact substrings, so a sequential scan recovers the offsets. Splitting drops the
    separator between pieces (typically one space or newline), which would leave
    1-character gaps; those separators are then assigned to the preceding piece so the
    parts tile the source exactly. Returns False if a piece cannot be located.
    """
    pos = 0
    for p in pieces:
        t = p.get("text", "")
        if not t:
            p["rel_start"] = pos
            p["rel_end"] = pos
            continue
        if source.startswith(t, pos):
            p["rel_start"] = pos
            p["rel_end"] = pos + len(t)
            pos += len(t)
            continue
        idx = source.find(t, pos)
        if idx < 0:
            return False
        p["rel_start"] = idx
        p["rel_end"] = idx + len(t)
        pos = idx + len(t)

    if not pieces:
        return True
    # close small gaps (dropped separators) by extending the preceding piece
    if pieces[0]["rel_start"] <= 4:
        pieces[0]["rel_start"] = 0
    for a, b in zip(pieces, pieces[1:]):
        gap = b["rel_start"] - a["rel_end"]
        if 0 <= gap <= 4:
            a["rel_end"] = b["rel_start"]
    if len(source) - pieces[-1]["rel_end"] <= 4:
        pieces[-1]["rel_end"] = len(source)

    for a, b in zip(pieces, pieces[1:]):
        if a["rel_end"] != b["rel_start"]:
            return False
    return True


def _map_to_evidence(charmap: list[int], rs: int, re_: int) -> tuple[int, int]:
    """Map a [rs, re_) range in retrieval space onto evidence space via the charmap."""
    if not charmap:
        return rs, re_
    rs = max(0, min(rs, len(charmap) - 1))
    re_ = max(rs, min(re_, len(charmap)))
    if re_ <= rs:
        return charmap[rs], charmap[rs] + 1
    return charmap[rs], charmap[re_ - 1] + 1


def build_doc_corpus(doc_id: str, clauses: list[dict], retrieval: list[dict],
                     tok: Tokenizer, max_tokens: int, cfg: dict,
                     tier: str | None = None,
                     institution: str | None = None,
                     charmap: list[int] | None = None,
                     span_errors: list[dict] | None = None) -> list[dict]:
    """One chunk per clause where possible; overlength clauses are split but keep their span.

    `tier` / `institution` override the values stored in the clause records.

    Why the override exists: clause records are written by 02_extract_clean.py and
    carry the *seed* tier from meta.json. Curation can promote a document (e.g. a
    university's General Regulations reached from a department landing page becomes
    an institution-level source), and it records that decision only in sources.csv.
    Reading the tier from the clause record therefore left 9 documents labelled
    'faculty' inside the corpus while sources.csv said 'institution', which would
    silently corrupt any tier-level analysis.

    Evidence ranges: every chunk records `ev_ranges`, the exact evidence intervals it
    contains. This matters because an over-length clause can be split into several
    chunks; an earlier version gave every part the clause's full span, so retrieving
    only the first part was scored as covering the whole clause and inflated Recall
    and Complete-Answer Rate. `ev_char_start/end` is kept as the bounding range for
    display only.
    """
    ret_by_clause = {r["clause_id"]: r for r in retrieval}
    chunks: list[dict] = []
    for c in clauses:
        r = ret_by_clause.get(c["clause_id"], {})
        text = r.get("ret_text") or c["text"]
        pieces = split_recursive(text, tok, max_tokens)
        # ---- per-piece evidence ranges (the fix described in the docstring) ----
        use_evidence_text = not r.get("ret_text")
        located = _locate_pieces(text, pieces)
        if not located:
            if span_errors is not None:
                span_errors.append({"doc_id": doc_id, "clause_id": c["clause_id"],
                                    "issue": "piece offsets could not be located"})
            # do not guess: fall back to the whole clause span but record the problem
            for piece in pieces:
                piece["ev_range"] = (c["char_start"], c["char_end"])
        else:
            clause_ret_start = r.get("ret_char_start", 0)
            for piece in pieces:
                rel_s, rel_e = piece["rel_start"], piece["rel_end"]
                if use_evidence_text:
                    ev_s = c["char_start"] + rel_s
                    ev_e = c["char_start"] + rel_e
                else:
                    ev_s, ev_e = _map_to_evidence(charmap or [],
                                                  clause_ret_start + rel_s,
                                                  clause_ret_start + rel_e)
                piece["ev_range"] = (ev_s, ev_e)

        # contiguity check: the parts of one clause must tile the clause span exactly
        if span_errors is not None and len(pieces) > 1:
            ranges = sorted(p["ev_range"] for p in pieces)
            covered = sum(e - s for s, e in ranges)
            contiguous = all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))
            if (not contiguous
                    or ranges[0][0] != c["char_start"]
                    or ranges[-1][1] != c["char_end"]
                    or covered != c["char_end"] - c["char_start"]):
                span_errors.append({
                    "doc_id": doc_id, "clause_id": c["clause_id"],
                    "issue": "split parts do not tile the clause span",
                    "clause_span": [c["char_start"], c["char_end"]],
                    "part_ranges": ranges[:6], "covered": covered,
                })

        for j, piece in enumerate(pieces):
            ev_s, ev_e = piece["ev_range"]
            chunks.append({
                "chunk_id": f"{c['clause_id']}::p{j}",
                "doc_id": doc_id,
                "tier": tier if tier is not None else c.get("tier", ""),
                "seed_tier": c.get("tier", ""),
                "institution": institution if institution is not None else c.get("institution", ""),
                # clause identity is preserved across a split: a clause may span
                # several chunks, but the clause span itself is never redefined
                "clause_id": c["clause_id"],
                "clause_label": c.get("clause_label", ""),
                "section_path": c.get("section_path", ""),
                "block_type": c.get("block_type", ""),
                "part_index": j,
                "n_parts": len(pieces),
                "text": piece["text"],
                "token_count": piece["tokens"],
                "split_reason": piece["split_reason"],
                "split_depth": piece["split_depth"],
                # ev_ranges is authoritative for coverage scoring; ev_char_* is display only
                "ev_ranges": [[ev_s, ev_e]],
                "ev_char_start": ev_s,
                "ev_char_end": ev_e,
                "clause_char_start": c["char_start"],
                "clause_char_end": c["char_end"],
                "ret_char_start": r.get("ret_char_start", 0),
                "ret_char_end": r.get("ret_char_end", 0),
                "has_exception": c.get("has_exception", False),
                "has_crossref": c.get("has_crossref", False),
                "has_alt_evidence": c.get("has_alt_evidence", False),
                "boilerplate": c.get("boilerplate", False),
                "strategy": "clause_unit",      # the shared unit all four strategies start from
            })
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--tokenizer", choices=["huggingface", "regex"], default="huggingface")
    ap.add_argument("--doc-id", action="append", default=[])
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)
    sc = cfg["safe_corpus"]
    max_tokens = sc["max_input_tokens"]

    tok = Tokenizer(sc["embedding_model"], args.tokenizer)
    suffix = tok.suffix()
    if tok.approx:
        print(f"NOTE: approximate tokenisation active; writing *_approx outputs", flush=True)

    ev_files = sorted(paths["cleaned"].glob("*.evidence.jsonl"))
    if args.doc_id:
        want = set(args.doc_id)
        ev_files = [p for p in ev_files if p.stem.replace(".evidence", "") in want]
    if not ev_files:
        print("no cleaned/*.evidence.jsonl found. Run 02_extract_clean.py first.", file=sys.stderr)
        return 2

    # Membership is decided by curate_corpus.py, which writes include=TRUE/FALSE into
    # sources.csv and a per-document reason into reports/curation_decisions.csv. The
    # clause floor below is only a fallback for runs that are not curated yet; keeping
    # the curated decision authoritative is what makes the frozen corpus match the
    # curation report instead of silently admitting extra documents.
    min_clauses = int(cfg.get("quality", {}).get("min_clauses_per_doc", 3))
    # Membership authority, in order:
    #   1. reports/curation_decisions.csv - the curated corpus (in_corpus column)
    #   2. sources.csv include=TRUE/FALSE  - written by curate_corpus.py
    #   3. clause floor                    - fallback for an uncurated first run
    #
    # NOTE: an earlier version preferred the previously built corpus file, which
    # stopped re-extraction from silently changing membership but also meant the
    # corpus ignored a new curation - the corpus and sources.csv then disagreed.
    # The curation decision is authoritative; a stale corpus is the worse failure.
    min_clauses = int(cfg.get("quality", {}).get("min_clauses_per_doc", 3))

    curated: dict[str, str] = {}
    cur_path = root / "reports" / "curation_decisions.csv"
    if cur_path.exists():
        with cur_path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("doc_id"):
                    curated[r["doc_id"]] = "TRUE" if str(r.get("in_corpus", "")).lower() in (
                        "true", "1", "yes") else "FALSE"

    sources_csv = root / "sources.csv"
    if not curated and sources_csv.exists():
        with sources_csv.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if str(r.get("include", "")).upper() in ("TRUE", "FALSE"):
                    curated[r.get("doc_id", "")] = r.get("include", "").upper()

    # Curation's effective tier/institution is authoritative for the corpus. The
    # clause records only carry the seed tier (see build_doc_corpus docstring).
    doc_meta: dict[str, dict] = {}
    if sources_csv.exists():
        with sources_csv.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                doc_id = r.get("doc_id", "")
                if doc_id:
                    doc_meta[doc_id] = {"tier": r.get("tier", ""),
                                        "institution": r.get("institution", "")}

    membership: list[dict] = []
    admitted: list[Path] = []
    for p in ev_files:
        doc_id = p.name.replace(".evidence.jsonl", "")
        n = len(read_jsonl(p))
        if curated and doc_id in curated:
            keep = curated[doc_id] == "TRUE"
            reason = "" if keep else "excluded by curate_corpus.py"
        elif curated:
            keep = False
            reason = "not present in the curated corpus"
        else:
            keep = n >= min_clauses
            reason = "" if keep else f"below min_clauses_per_doc ({min_clauses})"
        membership.append({"doc_id": doc_id, "clauses": n, "in_corpus": keep, "reason": reason})
        if keep:
            admitted.append(p)

    with (paths["reports"] / "corpus_membership.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "clauses", "in_corpus", "reason"])
        w.writeheader()
        for r in membership:
            w.writerow(r)

    ev_files = admitted
    if not ev_files:
        print(f"no document reached {min_clauses} clauses; lower min_clauses_per_doc "
              f"or collect denser sources", file=sys.stderr)
        return 2

    all_chunks: list[dict] = []
    per_doc: list[dict] = []
    span_errors: list[dict] = []

    for i, ev_path in enumerate(ev_files, 1):
        doc_id = ev_path.name.replace(".evidence.jsonl", "")
        clauses = read_jsonl(ev_path)
        ret_path = paths["cleaned"] / f"{doc_id}.retrieval.jsonl"
        retrieval = read_jsonl(ret_path) if ret_path.exists() else []
        # charmap lets a piece offset in retrieval space map back to evidence space
        charmap: list[int] = []
        cm_path = paths["cleaned"] / f"{doc_id}.charmap.jsonl"
        if cm_path.exists():
            cm_rows = read_jsonl(cm_path)
            if cm_rows:
                charmap = cm_rows[0].get("charmap", []) or []
        meta = doc_meta.get(doc_id, {})
        chunks = build_doc_corpus(doc_id, clauses, retrieval, tok, max_tokens, cfg,
                                  tier=meta.get("tier") or None,
                                  institution=meta.get("institution") or None,
                                  charmap=charmap, span_errors=span_errors)
        all_chunks.extend(chunks)

        tokens = [ch["token_count"] for ch in chunks]
        overs = [ch for ch in chunks if ch["split_reason"]]
        per_doc.append({
            "doc_id": doc_id,
            "tier": meta.get("tier") or (clauses[0].get("tier", "") if clauses else ""),
            "institution": meta.get("institution") or (clauses[0].get("institution", "") if clauses else ""),
            "clauses": len(clauses),
            "chunks": len(chunks),
            "chunks_from_split": len(overs),
            "max_split_depth": max([ch["split_depth"] for ch in chunks], default=0),
            "tokens_total": sum(tokens),
            "tokens_mean": round(sum(tokens) / len(tokens), 1) if tokens else 0,
            "tokens_max": max(tokens, default=0),
            "tokens_min": min(tokens, default=0),
            "over_max_input": sum(1 for t in tokens if t > max_tokens),
            "over_512": sum(1 for t in tokens if t > 512),
            "over_384": sum(1 for t in tokens if t > 384),
            "over_256": sum(1 for t in tokens if t > 256),
            "over_128": sum(1 for t in tokens if t > 128),
        })
        print(f"[{i:2d}/{len(ev_files)}] {doc_id:52s} clauses={len(clauses):>4} chunks={len(chunks):>4} "
              f"split={len(overs):>3} maxtok={max(tokens, default=0):>4}", flush=True)

    out_corpus = paths["corpus"] / f"corpus_safe_512{suffix}.jsonl"
    n = write_jsonl(out_corpus, all_chunks)

    # chunk-level index used by the token-budget simulation
    write_jsonl(paths["corpus"] / f"chunk_index{suffix}.jsonl",
                [{"chunk_id": ch["chunk_id"], "doc_id": ch["doc_id"],
                  "clause_id": ch["clause_id"], "token_count": ch["token_count"]}
                 for ch in all_chunks])

    with (paths["reports"] / f"corpus_capability{suffix}.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(per_doc[0].keys()) if per_doc else ["doc_id"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_doc:
            w.writerow(r)

    # token distribution figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tokens = [ch["token_count"] for ch in all_chunks]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(tokens, bins=40, color="#3b6ea5", edgecolor="white")
        axes[0].axvline(max_tokens, color="crimson", linestyle="--",
                        label=f"model max = {max_tokens}")
        axes[0].set_title("Chunk token distribution")
        axes[0].set_xlabel("tokens per chunk")
        axes[0].set_ylabel("chunks")
        axes[0].legend()
        axes[1].hist([len(ch["text"]) for ch in all_chunks], bins=40, color="#7aa35b", edgecolor="white")
        axes[1].set_title("Chunk character length")
        axes[1].set_xlabel("characters per chunk")
        fig.suptitle(f"Shared corpus: {len(all_chunks)} chunks from {len(per_doc)} documents "
                     f"({'approximate' if tok.approx else 'bge-small-en-v1.5'} tokenizer)")
        fig.tight_layout()
        fig.savefig(paths["reports"] / f"token_distribution{suffix}.png", dpi=140)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not render token_distribution.png ({type(e).__name__}: {e})",
              file=sys.stderr)

    tokens = [ch["token_count"] for ch in all_chunks]
    over = [ch for ch in all_chunks if ch["token_count"] > max_tokens]

    # ---- contract assertion: the corpus must agree with the curation record ----
    # If these drift, every downstream tier-level analysis is silently wrong and the
    # frozen manifest stops describing the corpus. Fail loudly instead.
    corpus_tier = {ch["doc_id"]: ch["tier"] for ch in all_chunks}
    mismatches = [(d, t, doc_meta.get(d, {}).get("tier", "<missing>"))
                  for d, t in corpus_tier.items()
                  if doc_meta.get(d, {}).get("tier", "") != t]
    # NOTE: curated maps doc_id -> "TRUE"/"FALSE" strings; comparing truthiness
    # would treat "FALSE" as selected, so compare against the literal string.
    curated_set = {d for d, keep in curated.items() if str(keep).upper() == "TRUE"}
    set_diff = sorted(curated_set ^ set(corpus_tier))

    print("\n=== shared corpus summary ===")
    print(f"  tokenizer                : {'APPROX (regex)' if tok.approx else sc['embedding_model']}")
    print(f"  documents admitted       : {len(per_doc)} of {len(membership)} "
          f"({len(membership) - len(per_doc)} not selected by curation)")
    print(f"  chunks                   : {n}")
    print(f"  chunks over {max_tokens} tokens    : {len(over)}  (must be 0; no silent truncation)")
    print(f"  chunks produced by split : {sum(1 for ch in all_chunks if ch['split_reason'])}")
    print(f"  chunks with ev_ranges    : {sum(1 for ch in all_chunks if ch.get('ev_ranges'))}")
    print(f"  split-span errors        : {len(span_errors)}  (must be 0)")
    print(f"  token mean/median/max    : "
          f"{round(sum(tokens)/len(tokens),1) if tokens else 0} / "
          f"{sorted(tokens)[len(tokens)//2] if tokens else 0} / {max(tokens, default=0)}")

    if span_errors:
        write_json(paths["reports"] / f"split_span_errors{suffix}.json", span_errors)
        print(f"\nERROR: {len(span_errors)} clause(s) have split parts that do not tile "
              f"their clause span; coverage scoring would be wrong.", file=sys.stderr)
        for e in span_errors[:5]:
            print(f"   {e['clause_id']}: {e['issue']}", file=sys.stderr)
        print(f"   details: reports/split_span_errors{suffix}.json", file=sys.stderr)
        return 1

    tier_counts: dict[str, int] = {}
    for t in corpus_tier.values():
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print(f"  tier composition         : {tier_counts}")

    if mismatches:
        print(f"\nERROR: {len(mismatches)} document(s) have a tier in the corpus that "
              f"disagrees with sources.csv:", file=sys.stderr)
        for d, got, want in mismatches[:10]:
            print(f"   {d}: corpus={got!r} sources={want!r}", file=sys.stderr)
        print("Refusing to leave an inconsistent corpus in place.", file=sys.stderr)
        return 1
    if set_diff:
        print(f"\nERROR: corpus/sources document sets differ: {set_diff[:10]}", file=sys.stderr)
        return 1
    print("  tier contract            : OK (corpus tiers match sources.csv)")
    print(f"\nwrote {out_corpus}")
    print(f"wrote {paths['reports'] / f'corpus_capability{suffix}.csv'}")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
