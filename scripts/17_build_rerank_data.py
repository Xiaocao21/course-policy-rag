"""17_build_rerank_data.py - build the development training pairs for the reranker.

One reranker is trained for all 17 chunking configurations, so a candidate is identified
by the chunk itself, never by "the chunk as produced by config X". Identity is
`(doc_id, corpus_version, char_start, char_end, sha1(text))`: two chunks that merely share
normalised text but start at different offsets are different candidates, because their
evidence coverage differs.

Labels come in three classes, and only two of them are trained:

    positive_clause  the chunk's evidence intervals fully contain at least one annotated
                     clause span. This is a *clause-level* property of one chunk.
    partial          the chunk overlaps a required span without containing it. Kept in the
                     artifact for analysis, excluded from binary training so that a
                     near-miss is not taught as a full negative.
    negative_hard    dense-ranked but neither of the above.
    negative_easy    same-document chunk that was never retrieved (optional, <=20%).

`complete_evidence` - one whole minimal sufficient set covered by the retrieved *set* - is
deliberately absent here: the reranker scores one (query, chunk) pair at a time and cannot
observe the rest of the ranking, so set-level completeness is an emergent property of the
ordering, not a training target.

Usage
    .venv\\Scripts\\python.exe scripts/17_build_rerank_data.py
    .venv\\Scripts\\python.exe scripts/17_build_rerank_data.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_ROOT, contains_span, load_settings, load_sentence_transformer,
    project_paths, read_jsonl, write_json, write_jsonl,
)
import goldset  # noqa: E402


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def chunk_key(doc_id: str, version: str, start: int, end: int, text: str) -> tuple:
    """Identity of a candidate chunk.

    Offsets are part of the key on purpose: two chunks that share normalised text but
    start at different positions cover different evidence, so they must not be merged.
    """
    return (doc_id, version, int(start), int(end), sha1(text))


def load_chunks(path: Path) -> dict[str, dict]:
    return {c["chunk_id"]: c for c in read_jsonl(path)}


def label_chunk(ev_ranges: list[list[int]], spans: list[tuple[int, int]]) -> str:
    """positive_clause / partial / negative for one chunk against one question."""
    text = ""
    if any(contains_span(text, ev_ranges, span) for span in spans):
        return "positive_clause"
    for span in spans:
        s, e = span
        for a, b in ev_ranges:
            if b > a and min(b, e) > max(a, s):
                return "partial"
    return "negative"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.split != "dev":
        print("REFUSING: reranker data may only be built from the development split.",
              file=sys.stderr)
        return 2

    root = Path(args.root)
    paths = project_paths(root)
    settings = load_settings(root)
    seed = int(settings.get("project", {}).get("random_seed", 20261013))
    rng = random.Random(seed)

    corpus = goldset.load_corpus(root)
    questions = read_jsonl(paths["annotations"] / "questions.jsonl")
    dev = [q for q in questions if q.get("split") == "dev"]
    if not dev:
        print("no development questions", file=sys.stderr)
        return 2
    if any(q.get("split") != "dev" for q in dev):
        print("REFUSING: non-dev question leaked into the build", file=sys.stderr)
        return 2

    summary = json.loads((paths["reports"] / "index_summary.json").read_text(encoding="utf-8"))
    configs = sorted(summary.get("configs", {}))
    if not configs:
        print("no indexes found; run 12_build_index.py first", file=sys.stderr)
        return 2

    model_name = settings.get("embedding", {}).get("model", "BAAI/bge-small-en-v1.5")
    model = load_sentence_transformer(model_name, device="cpu")
    qtexts = [q["question"] for q in dev]
    qemb = model.encode(qtexts, batch_size=32, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)

    # ------------------------------------------------------------ candidate pool
    # key: (qid, doc_id, version, char_start, char_end, text_sha1)
    candidates: dict[tuple, dict] = {}
    version = corpus.corpus_version

    for cfg in configs:
        strategy, config = cfg.split("/", 1)
        index_dir = root / "indexes" / strategy
        import faiss
        index = faiss.read_index(str(index_dir / f"{config}.faiss"))
        meta = json.loads((index_dir / f"{config}.meta.json").read_text(encoding="utf-8"))
        chunks = load_chunks(root / "chunks" / strategy / f"{config}.jsonl")
        chunk_ids = meta["chunk_ids"]

        D, I = index.search(qemb, args.top_k)
        for qi, (dists, idxs) in enumerate(zip(D, I)):
            q = dev[qi]
            for rank, (dist, idx) in enumerate(zip(dists, idxs)):
                if idx < 0:
                    continue
                chunk = chunks.get(chunk_ids[int(idx)])
                if chunk is None:
                    continue
                key = (q["qid"],) + chunk_key(chunk["doc_id"], version,
                                              chunk["char_start"], chunk["char_end"],
                                              chunk["text"])
                entry = candidates.get(key)
                if entry is None:
                    entry = candidates[key] = {
                        "qid": q["qid"], "group_id": q["group_id"],
                        "domain": q["domain"], "evidence_type": q["evidence_type"],
                        "doc_id": chunk["doc_id"], "corpus_version": version,
                        "char_start": int(chunk["char_start"]),
                        "char_end": int(chunk["char_end"]),
                        "text_sha1": sha1(chunk["text"]),
                        "text": chunk["text"],
                        "token_count": int(chunk.get("token_count", 0)),
                        "source_offsets": chunk.get("ev_ranges", []),
                        "clause_ids": chunk.get("clause_ids", []),
                        "config_ids": [], "dense_rank_by_config": {},
                        "merged_from_chunk_ids": [],
                    }
                if config not in entry["dense_rank_by_config"]:
                    entry["config_ids"].append(config)
                    entry["dense_rank_by_config"][config] = rank
                    entry["merged_from_chunk_ids"].append(chunk["chunk_id"])
                entry["dense_rank_by_config"][config] = min(
                    entry["dense_rank_by_config"][config], rank)

    # ------------------------------------------------------------ labels
    spans_by_qid: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for q in dev:
        for _si, _clause, span in goldset.iter_gold_spans(q, corpus):
            if span[1] > span[0]:
                spans_by_qid[q["qid"]].append(span)

    label_counts = Counter()
    for key, entry in candidates.items():
        spans = spans_by_qid.get(entry["qid"], [])
        entry["label"] = label_chunk(entry["source_offsets"], spans)
        label_counts[entry["label"]] += 1

    # ------------------------------------------------------------ cap per question
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for entry in candidates.values():
        by_qid[entry["qid"]].append(entry)

    kept: list[dict] = []
    for qid, rows in by_qid.items():
        positives = [r for r in rows if r["label"] == "positive_clause"]
        near = [r for r in rows if min(r["dense_rank_by_config"].values()) < 10]
        chosen = {id(r) for r in positives} | {id(r) for r in near}
        positive_ids = {id(r) for r in positives}
        selected = list(positives)
        selected += [r for r in near
                     if id(r) not in positive_ids and r["label"] != "positive_clause"]
        rest = [r for r in rows if id(r) not in chosen]
        rest.sort(key=lambda r: (min(r["dense_rank_by_config"].values()),
                                 r["doc_id"], r["char_start"]))
        for r in rest:
            if len(selected) >= args.cap:
                break
            selected.append(r)
        kept.extend(selected)

    # ------------------------------------------------------------ group split
    groups = sorted({q["group_id"] for q in dev})
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * args.val_fraction)))
    val_groups = set(groups[:n_val])
    train_groups = set(groups[n_val:])
    if val_groups & train_groups:
        print("REFUSING: group split overlaps", file=sys.stderr)
        return 2

    for entry in kept:
        entry["role"] = "val" if entry["group_id"] in val_groups else "train"

    # A question whose gold clause never appears in any config's top-k has no positive
    # pair. Keeping its rows in *training* would teach BCE that "nothing is relevant" for
    # that query, so its train rows are dropped (val rows stay, for honest evaluation).
    pos_by_qid = Counter(e["qid"] for e in kept if e["label"] == "positive_clause")
    no_positive = sorted({e["qid"] for e in kept} - set(pos_by_qid))
    dropped = [e for e in kept
               if e["role"] == "train" and e["qid"] in set(no_positive)]
    if dropped:
        kept = [e for e in kept if not (e["role"] == "train"
                                        and e["qid"] in set(no_positive))]

    train_q = {e["qid"] for e in kept if e["role"] == "train"}
    val_q = {e["qid"] for e in kept if e["role"] == "val"}
    if train_q & val_q:
        print("REFUSING: a question appears in both roles", file=sys.stderr)
        return 2

    # ------------------------------------------------------------ truncation estimate
    # bge-reranker-base has 512 position embeddings, and corpus chunks run up to 509
    # tokens, so a (question, chunk) pair cannot always fit: something must be cut.
    # The rule is fixed here rather than left to the trainer: the question keeps its
    # first QUERY_MAX=128 tokens (a question carries its intent early), and the chunk
    # gets whatever remains. Each pair records how many tokens were cut so that a
    # sensitivity analysis on the untruncated subset stays possible.
    QUERY_MAX = 128
    PAIR_MAX = 512
    trunc = {"pairs": 0, "truncated": 0, "fraction": 0.0, "max_pair_tokens": 0,
             "query_max": QUERY_MAX, "pair_max": PAIR_MAX,
             "truncated_tokens_total": 0, "truncated_tokens_max": 0,
             "tokenizer": None}
    try:
        from transformers import AutoTokenizer
        rt = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
        rt.model_max_length = 10 ** 9  # counting only; the model still sees <=512
        trunc["tokenizer"] = "BAAI/bge-reranker-base"
        qlen = {q["qid"]: min(QUERY_MAX,
                              len(rt(q["question"], add_special_tokens=False)["input_ids"]))
                for q in dev}
        clen: dict[str, int] = {}
        for e in kept:
            h = e["text_sha1"]
            if h not in clen:
                clen[h] = len(rt(e["text"], add_special_tokens=False)["input_ids"])
        for e in kept:
            pair = qlen[e["qid"]] + clen[e["text_sha1"]] + 3
            cut = max(0, pair - PAIR_MAX)
            e["pair_tokens"] = pair
            e["truncated_tokens"] = cut
            trunc["pairs"] += 1
            trunc["max_pair_tokens"] = max(trunc["max_pair_tokens"], pair)
            if cut:
                trunc["truncated"] += 1
                trunc["truncated_tokens_total"] += cut
                trunc["truncated_tokens_max"] = max(trunc["truncated_tokens_max"], cut)
    except Exception as exc:  # noqa: BLE001
        trunc["error"] = f"{type(exc).__name__}: {exc}"

    if trunc["pairs"]:
        trunc["fraction"] = round(trunc["truncated"] / trunc["pairs"], 4)
        trunc["truncated_tokens_mean"] = round(
            trunc["truncated_tokens_total"] / max(1, trunc["truncated"]), 1)

    # ------------------------------------------------------------ stats + write
    per_q_positive = Counter()
    for e in kept:
        if e["label"] == "positive_clause":
            per_q_positive[e["qid"]] += 1

    stats = {
        "split": "dev",
        "corpus_version": version,
        "seed": seed,
        "top_k": args.top_k,
        "cap": args.cap,
        "configs": len(configs),
        "questions": len(dev),
        "groups": len(groups),
        "candidates_before_dedup": int(sum(len(c["merged_from_chunk_ids"])
                                           for c in candidates.values())),
        "candidates_after_dedup": len(candidates),
        "merged_across_configs": sum(1 for c in candidates.values()
                                     if len(c["config_ids"]) > 1),
        "kept_pairs": len(kept),
        "label_counts": dict(label_counts),
        "kept_label_counts": dict(Counter(e["label"] for e in kept)),
        "questions_with_positive": len(per_q_positive),
        "questions_without_positive_qids": no_positive,
        "train_rows_dropped_no_positive": len(dropped),
        "kept_pairs_final": len(kept),
        "questions_without_positive": len(dev) - len(per_q_positive),
        "mean_positives_per_question": round(
            sum(per_q_positive.values()) / max(1, len(per_q_positive)), 3),
        "split": {"val_groups": len(val_groups), "train_groups": len(train_groups),
                  "val_questions": len(val_q), "train_questions": len(train_q)},
        "truncation": trunc,
    }

    print("=== reranker data ===")
    print(f"  dev questions/groups : {len(dev)} / {len(groups)}")
    print(f"  candidates (dedup)   : {stats['candidates_after_dedup']} "
          f"(merged across configs: {stats['merged_across_configs']})")
    print(f"  kept pairs           : {len(kept)}")
    print(f"  labels (kept)        : {stats['kept_label_counts']}")
    print(f"  questions with pos   : {stats['questions_with_positive']} / {len(dev)}")
    print(f"  train/val groups     : {len(train_groups)} / {len(val_groups)}")
    print(f"  truncation           : {trunc['truncated']}/{trunc['pairs']} "
          f"({trunc['fraction']:.1%}) max pair {trunc['max_pair_tokens']}")

    if args.dry_run:
        print("\n  dry run: nothing written")
        return 0

    out = root / "models" / "reranker"
    out.mkdir(parents=True, exist_ok=True)
    kept.sort(key=lambda e: (e["qid"], min(e["dense_rank_by_config"].values()),
                             e["doc_id"], e["char_start"]))
    write_jsonl(out / "pairs_dev.jsonl", kept)
    stats["pairs_sha256"] = hashlib.sha256(
        (out / "pairs_dev.jsonl").read_bytes()).hexdigest()
    write_json(out / "data_stats.json", stats)
    write_json(out / "split_groups.json",
               {"val_groups": sorted(val_groups), "train_groups": sorted(train_groups)})
    print(f"\nwrote {out / 'pairs_dev.jsonl'}")
    print(f"wrote {out / 'data_stats.json'}")
    print(f"wrote {out / 'split_groups.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
