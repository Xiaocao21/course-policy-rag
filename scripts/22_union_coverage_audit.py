"""22_union_coverage_audit.py - how completeness is actually achieved.

Why this exists
---------------
A reviewer asked whether `Complete@10` can be satisfied by *combining* several retrieved
chunks, because a rule that demanded one whole clause inside one chunk would handicap
sentence-level chunking by construction. It can: `coverage_for_set` accumulates the
`ev_ranges` of every retrieved chunk in a document and then asks whether their **union**
contains the annotated span (`13_evaluate.py:76-102`, `common.py:471-486`,
`tests/test_goldset.py:271-283`). Only MRR is single-chunk.

This script measures the consequence instead of asserting it: for each configuration it
recomputes the retrieved lists, then splits completeness into

    single    one retrieved chunk contains the whole clause on its own
    union     only the union of >= 2 retrieved chunks contains it
    chunks    how many chunks the cheapest covering union needs

and reports the share of complete answers that needed >= 2 chunks. Those shares are
qualitative quantities: they move by a whole chunk, not by a last-digit flip, so they can
be compared across machines while a per-cell `Complete@10` cannot.

Machine discipline
------------------
* **Same machine, strict.** For the development split the reference is
  `results/compat_dev/dev_full-vs-partial/partial`, produced on the A5000 by
  `20_compat_dev_compare.py`, so an exact per-cell comparison is meaningful and asserted.
* **Across machines, loose.** For the test split the only reference is the canonical
  RTX 3060 run. The same checkpoint re-evaluated on the A5000 differs by ~0.003 in macro
  accuracy (val 0.3904 here vs 0.3877 there), so the script asserts only
  |macro difference| <= 0.005 and records the device; per-cell differences are reported
  but never asserted.

Test-split policy
-----------------
`--split test` is restricted to the dense tracks. They are computed on CPU with a frozen
encoder, reproduce the canonical per-cell values exactly (0/1904 differing cells measured
before this script existed), and are therefore a *re-analysis* of the frozen measurement,
not a new evaluation. The reranked decomposition is run on development only, because
reranking a test pool again would be a second use of the split. The output never goes to
`results/final` - it lands in `results/analysis/`.

Usage
    .venv-gpu\\Scripts\\python.exe scripts/22_union_coverage_audit.py --split test
    .venv-gpu\\Scripts\\python.exe scripts/22_union_coverage_audit.py --split dev
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from common import (PROJECT_ROOT, contains_span, load_settings, project_paths,  # noqa: E402
                    read_jsonl, write_json)
import goldset  # noqa: E402

TRACKS = {"main": {"pool": 10, "rerank": False},
          "C1": {"pool": 50, "rerank": False},
          "C3": {"pool": 50, "rerank": True}}
MACRO_TOLERANCE = 0.005


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_module(rel: str, name: str, root: Path):
    spec = importlib.util.spec_from_file_location(name, root / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def family_of(strategy: str, config: str) -> str:
    if strategy == "fixed":
        return f"fixed/{config.split('_')[0]}"
    return strategy


def chunk_ranges(chunks: list[dict], doc_id: str) -> list[list[tuple[int, int]]]:
    """Per chunk, the evidence intervals it contributes for one document."""
    return [[(s, e) for s, e in (c.get("ev_ranges") or []) if c["doc_id"] == doc_id]
            for c in chunks if c["doc_id"] == doc_id]


def covered_by_one(chunks: list[dict], text: str, doc_id: str,
                   span: tuple[int, int]) -> bool:
    for c in chunks:
        if c["doc_id"] != doc_id:
            continue
        if contains_span(text, [(s, e) for s, e in (c.get("ev_ranges") or [])], span):
            return True
    return False


def covered_by_union(chunks: list[dict], text: str, doc_id: str,
                     span: tuple[int, int]) -> bool:
    ranges = [(s, e) for c in chunks if c["doc_id"] == doc_id
              for s, e in (c.get("ev_ranges") or [])]
    return contains_span(text, ranges, span)


def min_chunks(chunks: list[dict], text: str, doc_id: str, span: tuple[int, int]) -> int | None:
    """Fewest chunks whose union covers `span`; None when the pool cannot cover it.

    Interval covering is solved greedily, which is optimal for intervals: from the current
    cursor, take the chunk that reaches furthest right.
    """
    s, e = span
    if e <= s:
        return None
    per_chunk = chunk_ranges(chunks, doc_id)
    cursor = s
    used = 0
    remaining = [(i, rs) for i, rs in enumerate(per_chunk) if rs]
    while cursor < e:
        best_gain, best_i = cursor, None
        for i, rs in remaining:
            reach = max([b for a, b in rs if a <= cursor] or [cursor])
            if reach > best_gain:
                best_gain, best_i = reach, i
        if best_i is None:
            return None
        cursor = best_gain
        used += 1
        remaining = [(i, rs) for i, rs in remaining if i != best_i]
    return used


def build_pools(root: Path, questions: list[dict], configs: list[str], embed_model,
                top_k: int = 50) -> dict[str, list[list[dict]]]:
    import faiss

    qemb = embed_model.encode([q["question"] for q in questions], batch_size=32,
                              normalize_embeddings=True, show_progress_bar=False,
                              convert_to_numpy=True)
    pools: dict[str, list[list[dict]]] = {}
    for cfg in configs:
        strategy, config = cfg.split("/", 1)
        index = faiss.read_index(str(root / "indexes" / strategy / f"{config}.faiss"))
        meta = json.loads((root / "indexes" / strategy / f"{config}.meta.json")
                          .read_text(encoding="utf-8"))
        chunks = {c["chunk_id"]: c for c in
                  read_jsonl(root / "chunks" / strategy / f"{config}.jsonl")}
        _D, I = index.search(qemb, top_k)
        per_q = []
        for row in I:
            cands = []
            for idx in row:
                if idx < 0:
                    continue
                c = chunks.get(meta["chunk_ids"][int(idx)])
                if c is None:
                    continue
                cands.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                              "text": c["text"], "ev_ranges": c.get("ev_ranges", []),
                              "token_count": int(c.get("token_count", 0))})
            per_q.append(cands)
        pools[cfg] = per_q
    return pools


def audit_track(*, track: str, spec: dict, pools: dict, questions: list[dict],
                gold: list, corpus, ev, tr, model, tok, device: str, batch: int) -> dict:
    """Recompute completeness and decompose it into single-chunk vs union coverage."""
    per_cell = []
    for cfg, per_q in pools.items():
        strategy, config = cfg.split("/", 1)
        for qi, q in enumerate(questions):
            pool = per_q[qi]
            if spec["rerank"]:
                ranked = tr.rerank(model, tok, q["question"], pool[: spec["pool"]], device, batch)
            else:
                ranked = pool[: spec["pool"]]
            top10 = ranked[:10]
            rec, hit, complete, _partial = ev.coverage_for_set(gold[qi], top10, corpus)
            # per required clause: single-chunk or union-only coverage, and how many chunks
            union_only = 0
            single = 0
            needed = []
            for spans in gold[qi]:
                for doc_id, span in spans:
                    text = goldset.evidence_text(corpus, doc_id)
                    if covered_by_one(top10, text, doc_id, span):
                        single += 1
                        needed.append(1)
                    elif covered_by_union(top10, text, doc_id, span):
                        union_only += 1
                        need = min_chunks(top10, text, doc_id, span)
                        needed.append(need if need else 99)
                    else:
                        need = min_chunks(pool[:50], text, doc_id, span)
                        needed.append(need if need else None)
            complete50 = ev.coverage_for_set(gold[qi], pool[:50], corpus)[2]
            complete20 = ev.coverage_for_set(gold[qi], ranked[:20], corpus)[2]
            per_cell.append({
                "qid": q["qid"], "group_id": q.get("group_id"), "strategy": strategy,
                "config": config, "family": family_of(strategy, config),
                "complete@10": 1.0 if complete else 0.0,
                "complete@20": 1.0 if complete20 else 0.0,
                "complete@50": 1.0 if complete50 else 0.0,
                "recall@10": round(rec, 6), "hit@10": 1.0 if hit else 0.0,
                "clauses_covered_single": single,
                "clauses_covered_union_only": union_only,
                "needed_chunks": needed,
            })
    return {**summarize(per_cell, track), "cells": per_cell}


def summarize(per_cell: list[dict], track: str) -> dict:
    complete_cells = [c for c in per_cell if c["complete@10"] == 1.0]
    multi = [c for c in complete_cells if c["clauses_covered_union_only"] > 0]
    needed = [n for c in complete_cells for n in c["needed_chunks"] if isinstance(n, int) and n > 0]
    by_family: dict[str, dict] = {}
    fam_cells = defaultdict(list)
    for c in per_cell:
        fam_cells[c["family"]].append(c)
    for fam, cells in sorted(fam_cells.items()):
        comp = [c for c in cells if c["complete@10"] == 1.0]
        by_family[fam] = {
            "n_cells": len(cells),
            "complete@10": round(float(np.mean([c["complete@10"] for c in cells])), 4),
            "complete@20": round(float(np.mean([c["complete@20"] for c in cells])), 4),
            "complete@50": round(float(np.mean([c["complete@50"] for c in cells])), 4),
            "share_of_complete_needing_union": (round(len([c for c in comp
                                                          if c["clauses_covered_union_only"] > 0])
                                                       / len(comp), 4) if comp else None),
            "median_chunks_for_covered_clause":
                round(float(np.median([n for c in comp for n in c["needed_chunks"]
                                       if isinstance(n, int) and n > 0])), 2) if comp else None,
        }
    return {
        "track": track,
        "n_cells": len(per_cell),
        "complete@10": round(float(np.mean([c["complete@10"] for c in per_cell])), 4),
        "complete@20": round(float(np.mean([c["complete@20"] for c in per_cell])), 4),
        "complete@50": round(float(np.mean([c["complete@50"] for c in per_cell])), 4),
        "complete_cells": len(complete_cells),
        "complete_cells_needing_union": len(multi),
        "share_of_complete_needing_union": (round(len(multi) / len(complete_cells), 4)
                                            if complete_cells else None),
        "median_chunks_for_covered_clause": (round(float(np.median(needed)), 2)
                                             if needed else None),
        "chunks_needed_p75": (round(float(np.percentile(needed, 75)), 2)
                              if needed else None),
        "by_family": by_family,
        "note": ("complete@k is union-based; 'needing union' means at least one required "
                 "clause was covered only by >= 2 retrieved chunks"),
    }


def compare_with_reference(audit: dict, perq_rows: list[dict], strict: bool) -> dict:
    """Cell-level and macro agreement between the audit and a stored perq file."""
    ref = {(r["qid"], r["config"]): float(r["complete@10"]) for r in perq_rows}
    mine = {}
    for row in audit["cells"]:
        mine[(row["qid"], row["config"])] = row["complete@10"]
    common = sorted(set(ref) & set(mine))
    diffs = [k for k in common if abs(ref[k] - mine[k]) > 1e-9]
    macro_ref = float(np.mean([ref[k] for k in common])) if common else 0.0
    macro_mine = float(np.mean([mine[k] for k in common])) if common else 0.0
    out = {
        "reference": audit.get("reference"),
        "cells_compared": len(common),
        "cells_differing": len(diffs),
        "max_abs_cell_difference": (round(max(abs(ref[k] - mine[k]) for k in diffs), 6)
                                    if diffs else 0.0),
        "macro_reference": round(macro_ref, 4),
        "macro_audit": round(macro_mine, 4),
        "macro_abs_difference": round(abs(macro_mine - macro_ref), 4),
        "strict": strict,
    }
    if strict:
        out["passed"] = (len(diffs) == 0)
    else:
        out["passed"] = out["macro_abs_difference"] <= MACRO_TOLERANCE
        out["tolerance"] = MACRO_TOLERANCE
    if diffs:
        out["examples"] = [{"qid": k[0], "config": k[1], "reference": ref[k],
                            "audit": mine[k]} for k in diffs[:5]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--tracks", nargs="*", default=None,
                    help="tracks to audit (test: main/C1 only; dev: C1/C3)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-batch", type=int, default=32)
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    settings = load_settings(root)

    tracks = args.tracks or (["main", "C1"] if args.split == "test" else ["C1", "C3"])
    if args.split == "test" and any(t == "C3" for t in tracks):
        print("REFUSING: the test split is not reranked again. The reranked decomposition "
              "runs on development only; the test audit is limited to the dense tracks, "
              "which reproduce the frozen run exactly.", file=sys.stderr)
        return 2

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = args.device if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    questions = [q for q in read_jsonl(paths["annotations"] / "questions.jsonl")
                 if q.get("split") == args.split]
    corpus = goldset.load_corpus(root)
    ev = load_module("scripts/13_evaluate.py", "eval13", root)
    tr = load_module("scripts/18_train_reranker.py", "train18", root)
    gold = [ev.gold_spans(q, corpus) for q in questions]

    summary = json.loads((paths["reports"] / "index_summary.json").read_text(encoding="utf-8"))
    configs = sorted(summary.get("configs", {}))

    print(f"=== union coverage audit ({args.split}) ===")
    print(f"  device     : {gpu}  torch {torch.__version__}")
    print(f"  questions  : {len(questions)}  configs: {len(configs)}  tracks: {tracks}")

    from common import load_sentence_transformer
    embed_model = load_sentence_transformer(
        settings.get("embedding", {}).get("model", "BAAI/bge-small-en-v1.5"), device="cpu")
    pools = build_pools(root, questions, configs, embed_model, top_k=50)

    model = tok = None
    if any(TRACKS[t]["rerank"] for t in tracks):
        frozen = json.loads((root / "models/reranker/FROZEN.json").read_text(encoding="utf-8"))
        want = frozen["model"]["checkpoint_sha256"]
        runs = root / "models" / "reranker" / "runs"
        ckpt = None
        for cand in sorted(runs.iterdir()):
            w = cand / "model.safetensors"
            if w.exists() and hashlib.sha256(w.read_bytes()).hexdigest() == want:
                ckpt = cand
                break
        if ckpt is None:
            print("REFUSING: no checkpoint matches FROZEN.json", file=sys.stderr)
            return 2
        print(f"  checkpoint : {ckpt.name}")
        tok = AutoTokenizer.from_pretrained(str(ckpt))
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt), num_labels=1)
        model.to(device).eval()

    out_root = root / "results" / "analysis"
    out_root.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "kind": "union_coverage_audit", "created_at": now(), "split": args.split,
        "device": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
        "precision": "fp16" if device == "cuda" else "fp32",
        "questions": len(questions), "configs": len(configs),
        "coverage_rule": ("a clause is covered when the union of retrieved intervals from "
                          "the same document version contains its whole span"),
        "tracks": {},
    }

    rows_for_csv = []
    for track in tracks:
        audit = audit_track(track=track, spec=TRACKS[track], pools=pools, questions=questions,
                            gold=gold, corpus=corpus, ev=ev, tr=tr, model=model, tok=tok,
                            device=device, batch=args.eval_batch)
        # keep the raw cells for the agreement checks, then drop them before writing json
        cells = audit.pop("cells", None)
        cell_rows = []
        for row in (cells or []):
            cell_rows.append(row)
            rows_for_csv.append({**{k: v for k, v in row.items() if k != "needed_chunks"},
                                 "needed_chunks_max": max(
                                     [n for n in row["needed_chunks"] if isinstance(n, int)]
                                     or [0])})
        audit["cells"] = cell_rows

        perq_path = None
        if args.split == "test":
            perq_path = root / "results" / "final" / f"perq_{track}.jsonl"
            strict = False
        else:
            perq_path = root / "results" / "compat_dev" / "dev_full-vs-partial" / "partial" / f"perq_{track}.jsonl"
            strict = True
        if perq_path and perq_path.exists():
            audit["reference"] = perq_path.relative_to(root).as_posix()
            check = compare_with_reference(audit, read_jsonl(perq_path), strict=strict)
            audit["agreement"] = check
            print(f"  {track}: {check['cells_compared']} cells, "
                  f"{check['cells_differing']} differing, "
                  f"macro ref {check['macro_reference']:.4f} vs audit {check['macro_audit']:.4f} "
                  f"({'strict' if strict else 'tolerance ' + str(MACRO_TOLERANCE)}) "
                  f"-> {'PASS' if check['passed'] else 'FAIL'}")
            if not check["passed"]:
                print(f"ASSERTION FAILED for {track}", file=sys.stderr)
                return 3
        audit.pop("cells", None)
        result["tracks"][track] = audit
        print(f"  {track}: complete@10 {audit['complete@10']:.3f}  "
              f"share needing union {audit['share_of_complete_needing_union']}  "
              f"median chunks {audit['median_chunks_for_covered_clause']}")
        for fam, e in audit["by_family"].items():
            print(f"      {fam:<14} complete@10 {e['complete@10']:.3f}  "
                  f"@20 {e['complete@20']:.3f}  @50 {e['complete@50']:.3f}  "
                  f"union-share {e['share_of_complete_needing_union']}  "
                  f"median chunks {e['median_chunks_for_covered_clause']}")

    write_json(out_root / f"union_audit_{args.split}.json", result)
    with (out_root / f"union_audit_{args.split}.csv").open("w", encoding="utf-8",
                                                           newline="") as f:
        fields = ["qid", "group_id", "strategy", "config", "family", "complete@10",
                  "complete@20", "complete@50", "recall@10", "hit@10",
                  "clauses_covered_single", "clauses_covered_union_only", "needed_chunks_max"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows_for_csv:
            w.writerow(row)
    print(f"  wrote {out_root / f'union_audit_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
