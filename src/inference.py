"""inference.py - retrieve chunks for one query (inference component).

Loads the frozen encoder, searches one configuration's FAISS index, and prints the top-k chunks
with their evidence ranges. This is the retrieval half of the system; the reported metrics are
produced by `src/evaluate.py`.

Usage
    python src/inference.py --query "How do I appeal a grade?" \
        --strategy heading --config heading --top-k 5
    python src/inference.py --query "..." --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--strategy", default="heading")
    ap.add_argument("--config", default="heading")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    index_path = root / "indexes" / args.strategy / f"{args.config}.faiss"
    chunks_path = root / "chunks" / args.strategy / f"{args.config}.jsonl"
    print(f"index : {index_path}")
    print(f"chunks: {chunks_path}")
    if args.dry_run:
        return 0
    for path in (index_path, chunks_path):
        if not path.exists():
            print(f"REFUSING: {path} is missing. Run tools/fetch_indexes.py (official) or "
                  f"src/preprocess.py (rebuild).", file=sys.stderr)
            return 2

    import faiss
    from sentence_transformers import SentenceTransformer

    texts = {}
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                texts[row["chunk_id"]] = row
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=args.device)
    query = model.encode([args.query], normalize_embeddings=True, convert_to_numpy=True)
    index = faiss.read_index(str(index_path))
    scores, ids = index.search(query.astype("float32"), args.top_k)
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        row = texts.get(f"{args.strategy}::{args.config}::{idx}") or texts.get(
            f"{args.strategy}::{args.config}::{int(idx)}")
        if row is None:
            # chunk ids are doc-scoped; fall back to the insertion order
            row = list(texts.values())[idx] if idx < len(texts) else None
        if row is None:
            print(f"{rank:>2} score={score:.4f} id={idx} (text unavailable)")
            continue
        print(f"{rank:>2} score={score:.4f} doc={row['doc_id']} "
              f"chars=[{row['char_start']},{row['char_end']}]")
        print(f"   {row['text'][:240].replace(chr(10), ' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
