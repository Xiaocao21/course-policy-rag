"""20_compat_dev_compare.py - same-machine dev comparison of two reranker checkpoints.

Why this script exists
----------------------
The A5000 experiment is a *compatibility-mode* measurement: `18_train_reranker.py` writes
`COMPAT.json`, never `FROZEN.json`, and `19_rerank_eval.py` refuses to run on anything that
is not the canonical 3060 freeze. That guard is deliberate, so this script does not try to
get around it. It answers a different, narrower question that is still legitimate:

    on one machine, with one fp16 code path, the same dense pools and the same
    development questions, how much does *full* fine-tuning change the result relative
    to the frozen *partial* fine-tune?

Both checkpoints are therefore evaluated here, on this machine, in the same process. The
comparison is internally valid (only the checkpoint varies), but its absolute level is not
comparable with the canonical numbers, and every artefact it writes says so.

Hard limits, enforced in code
-----------------------------
* `--split test` is rejected. The test split has had its one formal use and is not touched
  by this programme.
* Output goes only to `results/compat_dev/<tag>/`. `results/final` and `results/final_dev`
  are never written.
* The four-track definitions, the metric functions and the truncation rule are imported
  from the same modules `19_rerank_eval.py` uses (`13_evaluate`, `18_train_reranker`), so
  this is a re-execution of the same measurement, not a second implementation of it.

Self-check. `main` and `C1` do not touch the reranker at all, so they must come out
identical for every checkpoint; the script asserts that instead of reporting it. A mismatch
means the dense path diverged from `19_rerank_eval.py` and the run is aborted.

Usage
    .venv-gpu\\Scripts\\python.exe scripts/20_compat_dev_compare.py \\
        --model partial=models/reranker/runs/rk-1e-05-n4-s20261027-canonical-partial-3060 \\
        --model full=models/reranker/runs/rk-1e-05-n4-s20261027

then, for group-level bootstrap CIs and Holm-corrected contrasts:
    .venv\\Scripts\\python.exe scripts/16_stats.py --split dev \\
        --results results/compat_dev/<tag>/<model-tag>
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

from common import PROJECT_ROOT, load_settings, project_paths, read_jsonl, write_json, write_jsonl  # noqa: E402
import goldset  # noqa: E402

TRACKS = {
    "main": {"pool": 10, "rerank": False},
    "C1": {"pool": 50, "rerank": False},
    "C2": {"pool": 10, "rerank": True},
    "C3": {"pool": 50, "rerank": True},
}
KS = (1, 3, 5, 10)
TOKEN_BUDGETS = (512, 1024, 2048)
SET_METRIC_KEYS = ("recall@10", "hit@10", "complete@10")
DENSE_ONLY_TRACKS = ("main", "C1")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(rel: str, name: str, root: Path):
    spec = importlib.util.spec_from_file_location(name, root / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def metric_block(per_k: dict, per_budget: dict, budgets_unused: dict, budgets_zero: dict,
                 total: int) -> dict:
    """Identical to 19_rerank_eval.metric_block; kept verbatim so the schema matches."""

    def mean(xs):
        return round(float(np.mean(xs)), 4) if xs else 0.0

    out = {}
    for k in KS:
        out[f"recall@{k}"] = mean(per_k[k]["recall"])
        out[f"hit@{k}"] = mean(per_k[k]["hit"])
        out[f"complete@{k}"] = mean(per_k[k]["complete"])
        out[f"partial@{k}"] = mean(per_k[k]["partial"])
    out["mrr"] = mean(per_k["mrr"])
    out["questions"] = total
    out["by_token_budget"] = {}
    for b in TOKEN_BUDGETS:
        out["by_token_budget"][str(b)] = {
            "recall": mean(per_budget[b]["recall"]),
            "hit": mean(per_budget[b]["hit"]),
            "complete": mean(per_budget[b]["complete"]),
            "tokens_unused_mean": round(float(np.mean(budgets_unused[b])), 1)
            if budgets_unused[b] else 0.0,
            "zero_chunk_queries": budgets_zero[b],
        }
    return out


def parse_models(items: list[str]) -> list[tuple[str, Path]]:
    models: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--model expects tag=path, got {item!r}")
        tag, path = item.split("=", 1)
        models.append((tag.strip(), Path(path.strip())))
    if len(models) < 2:
        raise SystemExit("give at least two --model tag=path, otherwise there is nothing "
                         "to compare")
    return models


def dense_pools(root: Path, questions: list[dict], configs: list[str],
                embed_model, top_k: int = 50) -> dict[str, list[list[dict]]]:
    """Dense top-k per configuration, built once and shared by every checkpoint."""
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
        per_q: list[list[dict]] = []
        for row in I:
            cands = []
            for idx in row:
                if idx < 0:
                    continue
                c = chunks.get(meta["chunk_ids"][int(idx)])
                if c is None:
                    continue
                cands.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                              "token_count": int(c.get("token_count", 0)),
                              "ev_ranges": c.get("ev_ranges", []), "text": c["text"]})
            per_q.append(cands)
        pools[cfg] = per_q
    print(f"    dense top-{top_k} pools built for {len(pools)} configurations", flush=True)
    return pools


def eval_tracks(*, root: Path, corpus, questions: list[dict], gold: list, pools: dict,
                ev, tr, model, tok, device: str, eval_batch: int, split: str,
                tag: str) -> tuple[dict, dict]:
    """Replay the four tracks of 19_rerank_eval.py for one checkpoint."""
    trunc_cache: dict[tuple[str, str], bool] = {}

    def pair_truncated(qid: str, text: str, chunk_text: str) -> bool:
        key = (qid, text[:64] + str(len(chunk_text)))
        if key not in trunc_cache:
            q_ids = tok(text, add_special_tokens=False)["input_ids"]
            c_ids = tok(chunk_text, add_special_tokens=False)["input_ids"]
            q_len = min(len(q_ids), tr.QUERY_MAX)
            trunc_cache[key] = (q_len + len(c_ids) + 3) > tr.PAIR_MAX
        return trunc_cache[key]

    perq: dict[str, list[dict]] = {t: [] for t in TRACKS}
    results: dict[str, dict[str, dict]] = {t: {} for t in TRACKS}

    for cfg, dense_per_q in pools.items():
        strategy, config = cfg.split("/", 1)
        for track, spec in TRACKS.items():
            per_k = {k: defaultdict(list) for k in KS}
            per_budget = {b: defaultdict(list) for b in TOKEN_BUDGETS}
            budgets_unused = {b: [] for b in TOKEN_BUDGETS}
            budgets_zero = {b: 0 for b in TOKEN_BUDGETS}
            per_k["mrr"] = []
            rows: list[dict] = []

            for qi, q in enumerate(questions):
                base = dense_per_q[qi]
                if spec["rerank"]:
                    pool = base[: spec["pool"]]
                    ranked = tr.rerank(model, tok, q["question"], pool, device, eval_batch)
                    cut = sum(1 for c in pool
                              if pair_truncated(q["qid"], q["question"], c["text"]))
                    trunc_frac = round(cut / max(1, len(pool)), 4)
                else:
                    ranked = base[: spec["pool"]]
                    trunc_frac = None

                rank = ev.first_hit_rank(gold[qi], ranked, corpus)
                per_k["mrr"].append(1.0 / rank if rank else 0.0)
                per_q = {"qid": q["qid"], "group_id": q.get("group_id"),
                         "domain": q.get("domain"), "evidence_type": q.get("evidence_type"),
                         "strategy": strategy, "config": config, "track": track,
                         "split": split, "pool": spec["pool"],
                         "rerank": spec["rerank"], "first_hit_rank": rank,
                         "ckpt_tag": tag}
                for k in KS:
                    rec, hit, comp, part = ev.coverage_for_set(gold[qi], ranked[:k], corpus)
                    per_k[k]["recall"].append(rec)
                    per_k[k]["hit"].append(1.0 if hit else 0.0)
                    per_k[k]["complete"].append(1.0 if comp else 0.0)
                    per_k[k]["partial"].append(part)
                    per_q[f"recall@{k}"] = round(rec, 6)
                    per_q[f"hit@{k}"] = 1.0 if hit else 0.0
                    per_q[f"complete@{k}"] = 1.0 if comp else 0.0
                    per_q[f"partial@{k}"] = round(part, 6)
                for b in TOKEN_BUDGETS:
                    taken, used, _stopped = ev.admission_token_budget(ranked, b)
                    rec, hit, comp, _part = ev.coverage_for_set(gold[qi], taken, corpus)
                    per_budget[b]["recall"].append(rec)
                    per_budget[b]["hit"].append(1.0 if hit else 0.0)
                    per_budget[b]["complete"].append(1.0 if comp else 0.0)
                    budgets_unused[b].append(b - used)
                    if not taken:
                        budgets_zero[b] += 1
                    per_q[f"tok{b}_recall"] = round(rec, 6)
                    per_q[f"tok{b}_complete"] = 1.0 if comp else 0.0
                per_q["mrr"] = round(1.0 / rank if rank else 0.0, 6)
                if trunc_frac is not None:
                    per_q["trunc_frac"] = trunc_frac
                rows.append(per_q)

            block = metric_block(per_k, per_budget, budgets_unused, budgets_zero,
                                 len(questions))
            for b in TOKEN_BUDGETS:
                bb = block["by_token_budget"][str(b)]
                block[f"tok{b}_recall"] = bb["recall"]
                block[f"tok{b}_complete"] = bb["complete"]
                block[f"tok{b}_hit"] = bb["hit"]
            block.update({"strategy": strategy, "config": config, "track": track,
                          "pool": spec["pool"], "rerank": spec["rerank"],
                          "split": split})
            results[track][f"{strategy}_{config}"] = block
            perq[track].extend(rows)
    return results, perq


def macro_track(perq: dict, track: str) -> dict:
    """Macro over configurations, the pre-registered collapse, from the perq rows."""
    by_cfg = defaultdict(list)
    for r in perq[track]:
        by_cfg[r["config"]].append(r)
    per_cfg = {}
    for cfg, rs in by_cfg.items():
        per_cfg[cfg] = {
            "complete@10": float(np.mean([r["complete@10"] for r in rs])),
            "recall@10": float(np.mean([r["recall@10"] for r in rs])),
            "mrr": float(np.mean([r["mrr"] for r in rs])),
            "tok2048_recall": float(np.mean([r["tok2048_recall"] for r in rs])),
        }
    macro = {k: float(np.mean([v[k] for v in per_cfg.values()]))
             for k in ("complete@10", "recall@10", "mrr", "tok2048_recall")}
    return {"macro": macro, "per_config": per_cfg, "n_configs": len(per_cfg)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--model", action="append", default=[], metavar="TAG=PATH")
    ap.add_argument("--tag", default="", help="output subdirectory under results/compat_dev")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-batch", type=int, default=32)
    args = ap.parse_args()

    if args.split == "test":
        print("REFUSING: the test split has had its one formal use and is not evaluated "
              "here. This script is a development-split compatibility check only.",
              file=sys.stderr)
        return 2

    root = Path(args.root)
    paths = project_paths(root)
    models = parse_models(args.model)
    for tag, path in models:
        if not (path / "config.json").exists():
            print(f"REFUSING: {tag}: no config.json under {path}", file=sys.stderr)
            return 2

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = args.device if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    corpus = goldset.load_corpus(root)
    questions = [q for q in read_jsonl(paths["annotations"] / "questions.jsonl")
                 if q.get("split") == args.split]
    if not questions:
        print(f"no {args.split} questions", file=sys.stderr)
        return 2
    ev = load_module("scripts/13_evaluate.py", "eval13", root)
    tr = load_module("scripts/18_train_reranker.py", "train18", root)
    gold = [ev.gold_spans(q, corpus) for q in questions]

    summary = json.loads((paths["reports"] / "index_summary.json").read_text(encoding="utf-8"))
    configs = sorted(summary.get("configs", {}))

    out_tag = args.tag or ("dev_" + "-vs-".join(t for t, _ in models))
    out_root = root / "results" / "compat_dev" / out_tag
    if out_root.exists() and any(out_root.iterdir()):
        print(f"REFUSING: {out_root} already has content; pick another --tag",
              file=sys.stderr)
        return 2
    out_root.mkdir(parents=True, exist_ok=True)

    print("=== compatibility dev comparison (NOT a canonical measurement) ===")
    print(f"  gpu        : {gpu_name}")
    print(f"  torch      : {torch.__version__}  cuda {torch.version.cuda}")
    print(f"  split      : {args.split}  questions {len(questions)}  configs {len(configs)}")
    print(f"  output     : {out_root}")

    # dense pools depend only on the frozen bi-encoder, so they are built once
    settings = load_settings(root)
    from common import load_sentence_transformer
    embed_model = load_sentence_transformer(
        settings.get("embedding", {}).get("model", "BAAI/bge-small-en-v1.5"), device="cpu")
    print("  building dense top-50 pools once for all checkpoints ...")
    pools = dense_pools(root, questions, configs, embed_model, top_k=50)

    summary_all: dict[str, dict] = {}
    reference_tag: str | None = None

    for tag, path in models:
        print(f"\n  --- {tag}: {path}")
        tok = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path), num_labels=1)
        model.to(device).eval()
        results, perq = eval_tracks(root=root, corpus=corpus, questions=questions, gold=gold,
                                    pools=pools, ev=ev, tr=tr, model=model, tok=tok,
                                    device=device, eval_batch=args.eval_batch,
                                    split=args.split, tag=tag)

        # ---- assertion: the two dense-only tracks must not depend on the checkpoint
        fp = {t: {cfg: {k: results[t][cfg][k] for k in SET_METRIC_KEYS}
                  for cfg in results[t]} for t in DENSE_ONLY_TRACKS}
        if reference_tag is None:
            reference_tag = tag
            reference_fp = fp
        elif fp != reference_fp:
            print(f"\nASSERTION FAILED: dense-only tracks for {tag} differ from "
                  f"{reference_tag}; the dense path is not checkpoint-independent.",
                  file=sys.stderr)
            return 3

        macro = {t: macro_track(perq, t) for t in TRACKS}
        summary_all[tag] = {
            "checkpoint": str(path),
            "model_sha256": sha256_file(path / "model.safetensors"),
            "dense_only_tracks_identical_to_reference": True,
            "tracks": macro,
        }
        for t in TRACKS:
            write_jsonl(out_root / f"perq_{t}__{tag}.jsonl", perq[t])
            print(f"    {t:<4} macro complete@10 {macro[t]['macro']['complete@10']:.4f}  "
                  f"recall@10 {macro[t]['macro']['recall@10']:.4f}  "
                  f"mrr {macro[t]['macro']['mrr']:.4f}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # 16_stats expects perq_<track>.jsonl without a tag, so each checkpoint gets its own
    # directory that 16_stats can be pointed at with --results.
    for tag, _path in models:
        sub = out_root / tag
        sub.mkdir(parents=True, exist_ok=True)
        for t in TRACKS:
            src = out_root / f"perq_{t}__{tag}.jsonl"
            dst = sub / f"perq_{t}.jsonl"
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            src.unlink()
        fields = ["strategy", "config", "track", "pool", "rerank",
                  "recall@1", "recall@5", "recall@10", "hit@10", "complete@10",
                  "partial@10", "mrr", "tok512_recall", "tok1024_recall",
                  "tok2048_recall", "tok512_complete", "tok1024_complete",
                  "tok2048_complete"]
        for t in TRACKS:
            with (sub / f"summary_{t}.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                for cfg in sorted(summary_all[tag]["tracks"][t]["per_config"]):
                    strategy = cfg.split("/", 1)[0] if "/" in cfg else ""
                    config = cfg.split("/", 1)[1] if "/" in cfg else cfg
                    w.writerow({"strategy": strategy, "config": config, "track": t,
                                "pool": TRACKS[t]["pool"], "rerank": TRACKS[t]["rerank"],
                                **summary_all[tag]["tracks"][t]["per_config"][cfg]})

    meta = {
        "kind": "compat_dev_comparison",
        "created_at": now(),
        "env_role": "compatibility",
        "note": ("A5000 fp16 compatibility measurement. NOT comparable with "
                 "results/final or results/final_dev (canonical RTX 3060). The only "
                 "inference supported is the within-machine difference between the "
                 "checkpoints evaluated together here."),
        "split": args.split,
        "questions": len(questions),
        "configs": len(configs),
        "env": {"gpu_model": gpu_name, "torch": torch.__version__,
                "cuda": torch.version.cuda, "precision": "fp16",
                "eval_batch": args.eval_batch},
        "dense_only_tracks_identical_across_checkpoints": True,
        "models": summary_all,
    }
    write_json(out_root / "META.json", meta)
    print(f"\n  wrote {out_root / 'META.json'}")
    print("  next: 16_stats.py --split dev --results "
          f"{out_root / '<model-tag>'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
