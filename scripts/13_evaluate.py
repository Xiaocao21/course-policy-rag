"""13_evaluate.py - retrieval metrics for every chunking configuration.

Metrics follow the revised protocol exactly.

  full-clause hit    a clause counts as covered only when the union of retrieved
                     evidence intervals from the correct document contains its whole
                     annotated span. Partial overlap never counts as a hit; it is
                     reported separately as partial_span_coverage.
  Recall@k           share of required clauses fully covered. With several alternative
                     sufficient sets, the largest covered share over sets.
  Hit@k              1 if any required clause in any set is fully covered.
  Complete@k         1 only when one whole sufficient set is covered.
  MRR                reciprocal rank of the first chunk that fully covers a required
                     clause.
  token budget       chunks are admitted in rank order and only as complete chunks; if
                     the next chunk does not fit, selection stops immediately. It is not
                     truncated and no smaller lower-ranked chunk is skipped ahead to.
                     Tokens actually used and the unused budget are both reported.

Usage
    .venv\\Scripts\\python.exe scripts/13_evaluate.py --split dev
    .venv\\Scripts\\python.exe scripts/13_evaluate.py --split test     # requires lock
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    load_sentence_transformer,
    PROJECT_ROOT, contains_span, covered_token_fraction, load_settings, project_paths,
    read_jsonl, write_json,
)
import goldset  # noqa: E402

KS = (1, 3, 5, 10)
TOKEN_BUDGETS = (512, 1024, 2048)


def gold_spans(q: dict, corpus: goldset.Corpus) -> list[list[tuple[str, tuple[int, int]]]]:
    """Per sufficient set, the list of (doc_id, span) that must be covered."""
    sets: list[list[tuple[str, tuple[int, int]]]] = []
    for eset in q.get("evidence_sets") or []:
        spans = []
        for ref in eset.get("clauses") or []:
            clause, _ = goldset.resolve_clause_ref(ref, corpus)
            if clause is None:
                continue
            spans.append((clause.doc_id, goldset.span_of(ref)))
        if spans:
            sets.append(spans)
    return sets


def covered_intervals(chunks: list[dict], doc_id: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for c in chunks:
        if c["doc_id"] != doc_id:
            continue
        for s, e in c.get("ev_ranges") or []:
            out.append((s, e))
    return out


def coverage_for_set(sets: list[list[tuple[str, tuple[int, int]]]], chunks: list[dict],
                     corpus: goldset.Corpus) -> tuple[float, bool, bool, float]:
    """Return (best_recall, hit, complete, mean_partial) over the sufficient sets."""
    by_doc: dict[str, list[tuple[int, int]]] = defaultdict(list)
    seen: set[str] = set()
    for c in chunks:
        if c["doc_id"] in seen and False:
            continue
        for s, e in c.get("ev_ranges") or []:
            by_doc[c["doc_id"]].append((s, e))

    best_recall = 0.0
    hit = False
    complete = False
    partials: list[float] = []
    for spans in sets:
        full = 0
        for doc_id, span in spans:
            text = goldset.evidence_text(corpus, doc_id)
            covers = by_doc.get(doc_id, [])
            if contains_span(text, covers, span):
                full += 1
                hit = True
            partials.append(covered_token_fraction(text, covers, span))
        share = full / len(spans) if spans else 0.0
        best_recall = max(best_recall, share)
        if full == len(spans):
            complete = True
    partial = sum(partials) / len(partials) if partials else 0.0
    return best_recall, hit, complete, partial


def first_hit_rank(sets: list[list[tuple[str, tuple[int, int]]]], ranked: list[dict],
                   corpus: goldset.Corpus) -> int | None:
    """1-based rank of the first chunk that fully covers any required clause."""
    for rank, c in enumerate(ranked, start=1):
        covers = [(s, e) for s, e in (c.get("ev_ranges") or [])]
        for spans in sets:
            for doc_id, span in spans:
                if doc_id != c["doc_id"]:
                    continue
                text = goldset.evidence_text(corpus, doc_id)
                if contains_span(text, covers, span):
                    return rank
    return None


def admission_token_budget(ranked: list[dict], budget: int) -> tuple[list[dict], int, bool]:
    """Admit complete chunks in rank order; stop at the first that does not fit."""
    used = 0
    taken: list[dict] = []
    stopped = False
    for c in ranked:
        t = int(c.get("token_count") or 0)
        if used + t > budget:
            stopped = True
            break
        taken.append(c)
        used += t
    return taken, used, stopped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--questions", default="")
    ap.add_argument("--strategy", action="append", default=[])
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-unlocked-test", action="store_true",
                    help="evaluate test without the lock (records that it was unlocked)")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)

    # ---------------------------------------------------------------- test gate
    locked_meta = None
    lock_path = paths["annotations"] / "LOCKED" / "LOCKED.json"
    if args.split == "test":
        if lock_path.exists():
            locked_meta = json.loads(lock_path.read_text(encoding="utf-8"))
        elif not args.allow_unlocked_test:
            print("REFUSING to evaluate the test split: it is not locked.", file=sys.stderr)
            print("Review the questions, then run: scripts/05_audit_splits.py --lock-test",
                  file=sys.stderr)
            return 2

    qpath = Path(args.questions) if args.questions else paths["annotations"] / "questions.jsonl"
    if not qpath.is_absolute():
        qpath = root / qpath
    questions = [q for q in read_jsonl(qpath) if q.get("split") == args.split]
    if not questions:
        print(f"no questions for split={args.split}", file=sys.stderr)
        return 2

    unreviewed = [q.get("qid") for q in questions
                  if q.get("prefilled") and not q.get("manual_reviewed")]
    if args.split == "test" and unreviewed and not args.allow_unlocked_test:
        print(f"REFUSING: {len(unreviewed)} test question(s) not marked manual_reviewed.",
              file=sys.stderr)
        return 2

    corpus = goldset.load_corpus(root)
    metas = sorted((root / "indexes").glob("*/*.meta.json"))
    if args.strategy:
        metas = [m for m in metas if m.parent.name in set(args.strategy)]
    if args.config:
        metas = [m for m in metas if m.stem.replace(".meta", "") in set(args.config)]
    if not metas:
        print("no indexes; run 12_build_index.py first", file=sys.stderr)
        return 2

    import faiss
    import numpy as np
    
    model_name = cfg["safe_corpus"]["embedding_model"]
    model = load_sentence_transformer(model_name, device=args.device)

    qtexts = [q.get("question", "") for q in questions]
    qemb = model.encode(qtexts, batch_size=32, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
    gold = [gold_spans(q, corpus) for q in questions]
    missing = [q.get("qid") for q, g in zip(questions, gold) if not g]
    if missing:
        print(f"WARNING: {len(missing)} question(s) have no resolvable gold spans: {missing[:5]}",
              file=sys.stderr)

    results_root = root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    perq_path = results_root / f"perq_{args.split}.jsonl"
    perq_fh = perq_path.open("w", encoding="utf-8")

    for meta_path in metas:
        strategy = meta_path.parent.name
        config = meta_path.name.replace(".meta.json", "")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(meta_path.with_suffix("").with_suffix(".faiss")))
        # meta path is <strategy>/<config>.meta.json ; index is <strategy>/<config>.faiss
        index = faiss.read_index(str(meta_path.parent / f"{config}.faiss"))

        chunks = list(zip(meta["chunk_ids"], meta["doc_ids"], meta["token_counts"],
                          meta["ev_ranges"]))

        t0 = time.time()
        D, I = index.search(np.asarray(qemb, dtype="float32"), args.top_k)
        search_time = time.time() - t0
        latency_ms = round(search_time / max(1, len(questions)) * 1000, 3)

        per_k: dict[int, dict[str, list[float]]] = {
            k: defaultdict(list) for k in KS}
        per_budget: dict[int, dict[str, list[float]]] = {
            b: defaultdict(list) for b in TOKEN_BUDGETS}
        budgets_unused = {b: [] for b in TOKEN_BUDGETS}
        budgets_zero = {b: 0 for b in TOKEN_BUDGETS}
        mrr_vals: list[float] = []
        by_type: dict[str, list[float]] = defaultdict(list)
        by_domain: dict[str, list[float]] = defaultdict(list)
        errors: list[dict] = []

        for qi, q in enumerate(questions):
            ranked = []
            for idx in I[qi]:
                if idx < 0 or idx >= len(chunks):
                    continue
                cid, doc_id, tok, ev = chunks[idx]
                ranked.append({"chunk_id": cid, "doc_id": doc_id,
                               "token_count": tok, "ev_ranges": ev})

            rank = first_hit_rank(gold[qi], ranked, corpus)
            mrr_vals.append(1.0 / rank if rank else 0.0)

            for k in KS:
                rec, hit, comp, part = coverage_for_set(gold[qi], ranked[:k], corpus)
                per_k[k]["recall"].append(rec)
                per_k[k]["hit"].append(1.0 if hit else 0.0)
                per_k[k]["complete"].append(1.0 if comp else 0.0)
                per_k[k]["partial"].append(part)
                if k == 5:
                    by_type[q.get("evidence_type", "?")].append(rec)
                    by_domain[q.get("domain", "?")].append(rec)

            for b in TOKEN_BUDGETS:
                taken, used, stopped = admission_token_budget(ranked, b)
                rec, hit, comp, part = coverage_for_set(gold[qi], taken, corpus)
                per_budget[b]["recall"].append(rec)
                per_budget[b]["hit"].append(1.0 if hit else 0.0)
                per_budget[b]["complete"].append(1.0 if comp else 0.0)
                per_budget[b]["partial"].append(part)
                budgets_unused[b].append(b - used)
                if not taken:
                    budgets_zero[b] += 1

            rec10 = per_k[10]["recall"][-1]
            # per-question dump feeds group-disjoint cross-validation in 14_analyze
            perq_fh.write(json.dumps({
                "config": f"{strategy}/{config}",
                "strategy": strategy,
                "qid": q.get("qid"), "group_id": q.get("group_id"),
                "split": args.split,
                "evidence_type": q.get("evidence_type"), "domain": q.get("domain"),
                "recall@1": per_k[1]["recall"][-1],
                "recall@5": per_k[5]["recall"][-1],
                "recall@10": rec10,
                "complete@10": per_k[10]["complete"][-1],
                "hit@10": per_k[10]["hit"][-1],
                "partial@10": per_k[10]["partial"][-1],
                "tok512_recall": per_budget[512]["recall"][-1],
                "tok1024_recall": per_budget[1024]["recall"][-1],
                "tok2048_recall": per_budget[2048]["recall"][-1],
            }, ensure_ascii=False) + "\n")

            if rec10 < 1.0 and len(errors) < 25:
                errors.append({"qid": q.get("qid"), "recall_at_10": round(rec10, 3),
                               "first_hit_rank": rank,
                               "evidence_type": q.get("evidence_type"),
                               "domain": q.get("domain")})

        def mean(xs: list[float]) -> float:
            return round(statistics.fmean(xs), 4) if xs else 0.0

        block = {
            "strategy": strategy, "config": config,
            "split": args.split,
            "questions": len(questions),
            "gold_unresolved": len(missing),
            "n_chunks": meta["n_chunks"],
            "model": model_name, "device": args.device,
            "latency_ms_per_query": latency_ms,
            "mrr": mean(mrr_vals),
            "by_k": {str(k): {m: mean(v) for m, v in per_k[k].items()} for k in KS},
            "by_token_budget": {
                str(b): {**{m: mean(v) for m, v in per_budget[b].items()},
                         "tokens_unused_mean": round(statistics.fmean(budgets_unused[b]), 1)
                         if budgets_unused[b] else 0,
                         "zero_chunk_queries": budgets_zero[b]}
                for b in TOKEN_BUDGETS},
            "recall_at_5_by_type": {k: mean(v) for k, v in sorted(by_type.items())},
            "recall_at_5_by_domain": {k: mean(v) for k, v in sorted(by_domain.items())},
            "error_examples": errors,
        }
        out = results_root / f"metrics_{strategy}_{config}.json"
        write_json(out, block)
        rows.append(block)

        print(f"  {strategy:<10} {config:<16} R@1={block['by_k']['1']['recall']:.3f} "
              f"R@5={block['by_k']['5']['recall']:.3f} R@10={block['by_k']['10']['recall']:.3f} "
              f"CE@10={block['by_k']['10']['complete']:.3f} MRR={block['mrr']:.3f} "
              f"lat={latency_ms:.2f}ms")

    # ---------------------------------------------------------------- summary csv
    perq_fh.close()
    csv_path = results_root / f"summary_{args.split}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "config", "split", "questions", "chunks",
                    "recall@1", "recall@5", "recall@10", "hit@10", "complete@10",
                    "partial@10", "mrr", "latency_ms",
                    "tok512_recall", "tok1024_recall", "tok2048_recall",
                    "tok512_complete", "tok1024_complete", "tok2048_complete",
                    "tok512_zero_chunk", "tok2048_unused_mean"])
        for b in sorted(rows, key=lambda r: (r["strategy"], -r["by_k"]["10"]["recall"])):
            w.writerow([
                b["strategy"], b["config"], b["split"], b["questions"], b["n_chunks"],
                b["by_k"]["1"]["recall"], b["by_k"]["5"]["recall"], b["by_k"]["10"]["recall"],
                b["by_k"]["10"]["hit"], b["by_k"]["10"]["complete"], b["by_k"]["10"]["partial"],
                b["mrr"], b["latency_ms_per_query"],
                b["by_token_budget"]["512"]["recall"],
                b["by_token_budget"]["1024"]["recall"],
                b["by_token_budget"]["2048"]["recall"],
                b["by_token_budget"]["512"]["complete"],
                b["by_token_budget"]["1024"]["complete"],
                b["by_token_budget"]["2048"]["complete"],
                b["by_token_budget"]["512"]["zero_chunk_queries"],
                b["by_token_budget"]["2048"]["tokens_unused_mean"],
            ])

    write_json(results_root / f"run_{args.split}.json", {
        "split": args.split,
        "questions": len(questions),
        "unreviewed_questions": len(unreviewed),
        "lock": locked_meta,
        "configs": len(rows),
        "corpus_version": corpus.corpus_version,
        "model": model_name,
    })
    print(f"\n{len(rows)} configurations evaluated on {args.split}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
