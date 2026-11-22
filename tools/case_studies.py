"""Diagnostic case studies for the paper (not part of the official evaluation).

Reads the frozen indexes and corpus, retrieves the top-50 candidates for a small set of
pre-selected questions, and records which gold spans each chunk covers. Output is written to
results/analysis/case_studies.json; no official result file is modified.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import faiss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import load_sentence_transformer, load_settings, read_jsonl  # noqa: E402
import goldset  # noqa: E402


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = load_module(ROOT / "scripts" / "13_evaluate.py", "eval13")

CASES = [
    ("Q0001", "semantic", "thr0.70"),
    ("Q0001", "heading", "heading"),
    ("Q0001", "fixed", "size128_ov0"),
    ("Q0088", "fixed", "size128_ov0"),
    ("Q0088", "heading", "heading"),
]


def main() -> int:
    questions = {q["qid"]: q for q in read_jsonl(ROOT / "annotations" / "questions.jsonl")}
    official = {(r["qid"], r["config"]): r for r in
                read_jsonl(ROOT / "results" / "final" / "perq_C1.jsonl")
                if r.get("track") == "C1"}
    corpus = goldset.load_corpus(ROOT)
    settings = load_settings(ROOT)
    encoder = load_sentence_transformer(
        settings.get("embedding", {}).get("model", "BAAI/bge-small-en-v1.5"), device="cpu")

    out = {"note": "Diagnostic retrieval for paper case studies; not an official evaluation.",
           "cases": []}

    for qid, strategy, config in CASES:
        q = questions[qid]
        index = faiss.read_index(str(ROOT / "indexes" / strategy / f"{config}.faiss"))
        meta = json.loads((ROOT / "indexes" / strategy / f"{config}.meta.json")
                          .read_text(encoding="utf-8"))
        chunks = {c["chunk_id"]: c for c in
                  read_jsonl(ROOT / "chunks" / strategy / f"{config}.jsonl")}
        emb = encoder.encode([q["question"]], batch_size=1, normalize_embeddings=True,
                             show_progress_bar=False, convert_to_numpy=True)
        _d, ind = index.search(emb, 50)
        ranked = []
        for idx in ind[0]:
            if idx < 0:
                continue
            c = chunks.get(meta["chunk_ids"][int(idx)])
            if c is None:
                continue
            ranked.append({
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "token_count": int(c.get("token_count", 0)),
                "clause_ids": c.get("clause_ids", []),
                "ev_ranges": c.get("ev_ranges", []),
                "snippet": c["text"][:220].replace("\n", " "),
            })

        sets = ev.gold_spans(q, corpus)
        recall, hit, complete, partial = ev.coverage_for_set(sets, ranked[:10], corpus)

        # Per-clause coverage and first rank for each gold span.
        gold = []
        seen = set()
        for si, spans in enumerate(sets):
            for ci, (doc_id, span) in enumerate(spans):
                key = (doc_id, span)
                if key in seen:
                    continue
                seen.add(key)
                text = goldset.evidence_text(corpus, doc_id)
                first = None
                for rank, c in enumerate(ranked[:10], start=1):
                    if c["doc_id"] != doc_id:
                        continue
                    if ev.contains_span(text, [tuple(x) for x in c["ev_ranges"]], span):
                        first = rank
                        break
                ref = q["evidence_sets"][si]["clauses"][ci]
                gold.append({"clause_id": ref.get("clause_id"), "doc_id": doc_id,
                             "span": list(span), "first_covering_rank": first,
                             "covered": first is not None})

        # Mark, for each ranked chunk, which gold spans it fully covers on its own.
        for rank, c in enumerate(ranked[:10], start=1):
            text = goldset.evidence_text(corpus, c["doc_id"])
            covers = [g["clause_id"] for g in gold
                      if g["doc_id"] == c["doc_id"]
                      and ev.contains_span(text, [tuple(x) for x in c["ev_ranges"]],
                                           tuple(g["span"]))]
            c["rank"] = rank
            c["covers_clause_ids"] = covers

        off = official[(qid, config)]
        checks = {
            "hit@10": (float(hit), float(off["hit@10"])),
            "complete@10": (float(complete), float(off["complete@10"])),
            "recall@10": (float(recall), float(off["recall@10"])),
        }
        for field, (got, want) in checks.items():
            if abs(got - want) > 1e-9:
                raise AssertionError(f"{qid} {config} {field}: {got} != {want}")
        out["cases"].append({
            "qid": qid, "question": q["question"], "evidence_type": q.get("evidence_type"),
            "strategy": strategy, "config": config, "track": "C1",
            "coverage": {
                "recall@10": recall, "hit@10": float(hit),
                "complete@10": float(complete), "partial@10": partial},
            "official_check": {k: {"computed": v[0], "official": v[1]}
                               for k, v in checks.items()},
            "gold": gold, "top10": ranked[:10],
        })
        print(f"{qid} {strategy}/{config}: hit={hit} complete={complete} "
              f"recall={recall:.3f}")

    dest = ROOT / "results" / "analysis" / "case_studies.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
