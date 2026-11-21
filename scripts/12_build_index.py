"""12_build_index.py - embed every chunking configuration and build a local FAISS index.

One index per (strategy, config), all using the same embedding model named in the frozen
manifest, so differences between conditions come from chunking alone.

Recorded per configuration (the efficiency outcomes the protocol asks for):
    chunks, index_size_bytes, build_time_s, embedding_time_s, peak_rss_mb, device

Usage
    .venv\\Scripts\\python.exe scripts/12_build_index.py
    .venv\\Scripts\\python.exe scripts/12_build_index.py --strategy fixed --config size256_ov0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    load_sentence_transformer,
    PROJECT_ROOT, load_settings, project_paths, read_jsonl, write_json,
)


def peak_rss_mb() -> float:
    """Peak working set in MB.

    `resource` is POSIX-only, so Windows uses GetProcessMemoryInfo through ctypes.
    A missing measurement is reported as 0 rather than failing the run.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            import ctypes.wintypes as wt

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wt.DWORD),
                    ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb):
                return round(counters.PeakWorkingSetSize / (1024 * 1024), 1)
            return 0.0
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(usage / 1024, 1)
    except Exception:  # noqa: BLE001
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--strategy", action="append", default=[])
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)
    model_name = cfg["safe_corpus"]["embedding_model"]

    chunk_files = sorted((root / "chunks").glob("*/*.jsonl"))
    if args.strategy:
        chunk_files = [p for p in chunk_files if p.parent.name in set(args.strategy)]
    if args.config:
        chunk_files = [p for p in chunk_files if p.stem in set(args.config)]
    if not chunk_files:
        print("no chunk files; run 11_chunking.py first", file=sys.stderr)
        return 2

    import faiss
    import numpy as np
    
    t0 = time.time()
    model = load_sentence_transformer(model_name, device=args.device)
    load_time = round(time.time() - t0, 1)
    dim = model.get_sentence_embedding_dimension()
    print(f"model {model_name} dim={dim} device={args.device} load={load_time}s")

    index_root = root / "indexes"
    summary: dict = {"model": model_name, "device": args.device, "configs": {}}

    for p in chunk_files:
        strategy, config = p.parent.name, p.stem
        chunks = read_jsonl(p)
        if not chunks:
            continue
        texts = [c["text"] for c in chunks]

        t_emb = time.time()
        emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
        emb_time = round(time.time() - t_emb, 1)

        t_idx = time.time()
        index = faiss.IndexFlatIP(dim)
        index.add(np.asarray(emb, dtype="float32"))
        build_time = round(time.time() - t_idx, 2)

        out_dir = index_root / strategy
        out_dir.mkdir(parents=True, exist_ok=True)
        index_path = out_dir / f"{config}.faiss"
        faiss.write_index(index, str(index_path))

        write_json(out_dir / f"{config}.meta.json", {
            "strategy": strategy, "config": config,
            "chunks_file": str(p.relative_to(root)).replace("\\", "/"),
            "n_chunks": len(chunks),
            "model": model_name, "dim": dim, "device": args.device,
            "chunk_ids": [c["chunk_id"] for c in chunks],
            "doc_ids": [c["doc_id"] for c in chunks],
            "token_counts": [c["token_count"] for c in chunks],
            "ev_ranges": [c["ev_ranges"] for c in chunks],
        })

        summary["configs"][f"{strategy}/{config}"] = {
            "n_chunks": len(chunks),
            "dim": dim,
            "index_size_bytes": index_path.stat().st_size,
            "build_time_s": build_time,
            "embedding_time_s": emb_time,
            "peak_rss_mb": peak_rss_mb(),
            "model_load_s": load_time,
            "device": args.device,
        }
        print(f"  {strategy:<10} {config:<16} n={len(chunks):>5} "
              f"emb={emb_time:>6.1f}s idx={build_time:>5.2f}s "
              f"size={index_path.stat().st_size/1e6:>6.2f}MB")

    summary["total_configs"] = len(summary["configs"])
    write_json(paths["reports"] / "index_summary.json", summary)
    print(f"\nindexed {summary['total_configs']} configurations")
    print(f"wrote {paths['reports'] / 'index_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
