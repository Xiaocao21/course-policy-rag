"""18_train_reranker.py - train ONE cross-encoder reranker for all 17 configurations.

Design notes that matter for the write-up.

* One model, not 17. Each configuration keeps its own frozen bi-encoder index; the
  reranker only rescores (question, chunk) pairs, so nothing is re-indexed and the
  configuration effect stays separate from the model effect.

* Model selection uses **set-level** evidence completeness, not question-level relevance.
  The decisive detail: reranking a fixed top-10 pool cannot change Recall@10 / Hit@10 /
  Complete@10, because the retrieved set is the same set in a different order. Selection
  therefore reranks the **top-50** pool and scores the top-10 of that reordered list, and
  the score is a **macro average over the 17 configurations** so no single configuration
  dominates.

* `FROZEN.json` is written only on the canonical environment. Another GPU may produce
  slightly different numbers under fp16, so its runs write `COMPAT.json` and must never be
  compared with canonical results.

Usage
    .venv-gpu\\Scripts\\python.exe scripts/18_train_reranker.py --smoke
    .venv-gpu\\Scripts\\python.exe scripts/18_train_reranker.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from common import (  # noqa: E402
    PROJECT_ROOT, load_settings, load_sentence_transformer, project_paths,
    read_jsonl, write_json, write_jsonl,
)
import goldset  # noqa: E402

PAIR_MAX = 512
QUERY_MAX = 128
MODEL_ID = "BAAI/bge-reranker-base"
CANONICAL = {"gpu_substring": "3060", "precision": "fp16", "batch": 8, "grad_accum": 2}
TIE_BREAK = ["macro_complete@10", "macro_mrr", "macro_recall@10"]


def select_best(runs: list[dict]) -> dict:
    """Pre-registered selection: macro Complete@10, then MRR, then Recall@10.

    Remaining ties are broken deterministically by the smaller learning rate, then fewer
    negatives, so the outcome never depends on iteration order.
    """
    return sorted(runs, key=lambda r: (-r["val"]["macro_complete@10"],
                                       -r["val"]["macro_mrr"],
                                       -r["val"]["macro_recall@10"],
                                       r["lr"], r["negatives"]))[0]


def load_eval_module(root: Path):
    """Import 13_evaluate by path so metric semantics are literally the same code."""
    path = root / "scripts" / "13_evaluate.py"
    spec = importlib.util.spec_from_file_location("eval13", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------- val pools
def build_val_pools(root: Path, questions: list[dict], configs: list[str],
                    top_k: int, model_name: str, cache: Path) -> dict:
    """Dense top-k per configuration for the validation questions (built once)."""
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("top_k") == top_k and data.get("n_questions") == len(questions):
            return data
    import faiss

    model = load_sentence_transformer(model_name, device="cpu")
    qemb = model.encode([q["question"] for q in questions], batch_size=32,
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
                chunk = chunks.get(meta["chunk_ids"][int(idx)])
                if chunk is None:
                    continue
                cands.append({"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
                              "token_count": int(chunk.get("token_count", 0)),
                              "ev_ranges": chunk.get("ev_ranges", []),
                              "text": chunk["text"]})
            per_q.append(cands)
        pools[cfg] = per_q
    data = {"top_k": top_k, "n_questions": len(questions),
            "qids": [q["qid"] for q in questions], "pools": pools}
    write_json(cache, data)
    return data


# --------------------------------------------------------------------- scoring
def macro_scores(model, tok, pools: dict, val_questions: list[dict], gold_sets: dict,
                 ev, corpus, device: str, batch: int, max_pairs: int | None = None,
                 cache: dict | None = None, use_reranker: bool = True) -> dict:
    """Macro-average over configurations of top-10 metrics.

    `use_reranker=False` scores the dense order of the same top-k pool, which is the
    baseline the reranker has to beat; without it a reranked number means nothing.
    """
    import torch

    per_config: dict[str, dict[str, float]] = {}
    rows: list[dict] = []
    for cfg, per_q in pools.items():
        rec, comp, mrr = [], [], []
        for qi, cands in enumerate(per_q):
            if not cands:
                continue
            if use_reranker:
                ranked = rerank(model, tok, val_questions[qi]["question"], cands, device,
                                batch, max_pairs, cache=cache,
                                qid=val_questions[qi]["qid"])
            else:
                ranked = cands if max_pairs is None else cands[:max_pairs]
            gold = gold_sets[val_questions[qi]["qid"]]
            rank = ev.first_hit_rank(gold, ranked, corpus)
            mrr.append(1.0 / rank if rank else 0.0)
            for k in (10,):
                r, _hit, c, _p = ev.coverage_for_set(gold, ranked[:k], corpus)
                rec.append(r)
                comp.append(1.0 if c else 0.0)
        per_config[cfg] = {
            "complete@10": float(np.mean(comp)) if comp else 0.0,
            "recall@10": float(np.mean(rec)) if rec else 0.0,
            "mrr": float(np.mean(mrr)) if mrr else 0.0,
        }
        rows.append({"config": cfg, **per_config[cfg], "n": len(comp)})
    macro = {k: float(np.mean([v[k] for v in per_config.values()]))
             for k in ("complete@10", "recall@10", "mrr")}
    return {"macro_complete@10": macro["complete@10"],
            "macro_recall@10": macro["recall@10"],
            "macro_mrr": macro["mrr"],
            "per_config": per_config, "n_configs": len(per_config)}


def pad_batch(ids_list: list[list[int]], pad_id: int):
    import torch
    maxlen = max(len(x) for x in ids_list)
    ids = torch.tensor([x + [pad_id] * (maxlen - len(x)) for x in ids_list])
    mask = torch.tensor([[1] * len(x) + [0] * (maxlen - len(x)) for x in ids_list])
    return {"input_ids": ids, "attention_mask": mask}


def pair_ids(tok, query: str, chunk: str) -> list[int]:
    """Single (question, chunk) token ids under the fixed truncation rule."""
    q_ids = tok(query, add_special_tokens=False, truncation=True,
                max_length=QUERY_MAX)["input_ids"]
    room = PAIR_MAX - len(q_ids) - 3
    c_ids = tok(chunk, add_special_tokens=False, truncation=True,
                max_length=max(8, room))["input_ids"]
    return [tok.cls_token_id] + q_ids + [tok.sep_token_id] + c_ids + [tok.sep_token_id]


def rerank(model, tok, query: str, cands: list[dict], device: str, batch: int,
           max_pairs: int | None = None, cache: dict | None = None,
           qid: str | None = None) -> list[dict]:
    """Score (query, chunk) pairs and return candidates sorted by score desc.

    `cache` memoises token ids per (qid, chunk_id): the validation pools are fixed, so
    re-tokenising 18k pairs on every epoch would dominate the wall clock.
    """
    import torch

    order = cands if max_pairs is None else cands[:max_pairs]
    if not order:
        return []
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(order), batch):
            chunk = order[start:start + batch]
            ids_list = []
            for c in chunk:
                key = (qid, c["chunk_id"]) if cache is not None and qid else None
                if key is not None and key in cache:
                    ids_list.append(cache[key])
                else:
                    ids = pair_ids(tok, query, c["text"])
                    if key is not None:
                        cache[key] = ids
                    ids_list.append(ids)
            enc = pad_batch(ids_list, tok.pad_token_id or 0)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**enc).logits.view(-1)
            scores.extend(logits.float().cpu().tolist())
    ranked = [dict(c) for c in order]
    for c, s in zip(ranked, scores):
        c["rerank_score"] = float(s)
    ranked.sort(key=lambda c: -c["rerank_score"])
    return ranked


def encode_pairs(tok, queries: list[str], chunks: list[str]):
    """Fixed truncation rule: question <=128 tokens, chunk gets the remainder."""
    import torch

    input_ids, attn = [], []
    sep_id = tok.sep_token_id
    cls_id = tok.cls_token_id
    for q, c in zip(queries, chunks):
        q_ids = tok(q, add_special_tokens=False, truncation=True, max_length=QUERY_MAX)["input_ids"]
        room = PAIR_MAX - len(q_ids) - 3
        c_ids = tok(c, add_special_tokens=False, truncation=True, max_length=max(8, room))["input_ids"]
        ids = [cls_id] + q_ids + [sep_id] + c_ids + [sep_id]
        input_ids.append(ids)
        attn.append([1] * len(ids))
    maxlen = max(len(x) for x in input_ids)
    pad_id = tok.pad_token_id or 0
    ids_t = torch.tensor([x + [pad_id] * (maxlen - len(x)) for x in input_ids])
    mask_t = torch.tensor([x + [0] * (maxlen - len(x)) for x in attn])
    return {"input_ids": ids_t, "attention_mask": mask_t}


# --------------------------------------------------------------------- training
def epoch_examples(pairs: list[dict], negatives_per_positive: int, rng: random.Random,
                   qtext: dict[str, str]) -> list[dict]:
    """All positives, each with `negatives_per_positive` sampled hard negatives."""
    by_q: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"pos": [], "neg": []})
    for p in pairs:
        if p["role"] != "train":
            continue
        bucket = "pos" if p["label"] == "positive_clause" else (
            "neg" if p["label"] == "negative" else None)
        if bucket:
            by_q[p["qid"]][bucket].append(p)
    out: list[dict] = []
    for qid, b in by_q.items():
        if not b["pos"] or not b["neg"]:
            continue
        for pos in b["pos"]:
            out.append(pos)
            k = min(negatives_per_positive, len(b["neg"]))
            out.extend(rng.sample(b["neg"], k))
    rng.shuffle(out)
    return out


def freeze_lower_layers(model, n: int) -> int:
    """Freeze the first `n` encoder layers and report how many parameters stay trainable."""
    body = None
    for name in ("roberta", "bert", "xlm_roberta"):
        body = getattr(model, name, None)
        if body is not None:
            break
    if body is None or n <= 0:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    layers = getattr(body, "layer", None)
    if layers is not None:
        for layer in list(layers)[:n]:
            for p in layer.parameters():
                p.requires_grad = False
    for p in body.embeddings.parameters():
        p.requires_grad = False
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one(*, model, tok, train_pairs, qtext, cfg, device, args, ev, corpus,
              gold_sets, val_questions, pools, seed, ckpt_path, cache) -> dict:
    import torch
    import warnings
    from transformers import get_linear_schedule_with_warmup

    warnings.filterwarnings("ignore", message="Detected call of")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    if args.optimizer == "adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(model.parameters(), lr=cfg["lr"], scale_parameter=False,
                              relative_step=False, warmup_init=False,
                              weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                      weight_decay=0.01)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    except TypeError:  # older torch signature
        scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    probe = epoch_examples(train_pairs, cfg["negatives"], rng, qtext)
    steps_per_epoch = max(1, len(probe) // (args.batch * args.grad_accum))
    total_steps = max(1, steps_per_epoch * args.epochs_max)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps)

    history: list[dict] = []
    best = {"score": -1.0, "epoch": -1, "ckpt": None, "metrics": None}
    patience_left = args.patience

    for epoch in range(1, args.epochs_max + 1):
        examples = epoch_examples(train_pairs, cfg["negatives"], rng, qtext)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        losses: list[float] = []
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for i in range(0, len(examples), args.batch):
            batch = examples[i:i + args.batch]
            enc = encode_pairs(tok, [qtext[b["qid"]] for b in batch],
                               [b["text"] for b in batch])
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = torch.tensor([1.0 if b["label"] == "positive_clause" else 0.0
                                   for b in batch], device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**enc).logits.view(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            scaler.scale(loss / args.grad_accum).backward()
            if (i // args.batch + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            if args.max_steps and len(losses) >= args.max_steps:
                break
        train_time = round(time.time() - t0, 1)

        model.eval()
        t_eval = time.time()
        metrics = macro_scores(model, tok, pools, val_questions, gold_sets, ev, corpus,
                               device, args.eval_batch, cache=cache)
        eval_time = round(time.time() - t_eval, 1)
        score = metrics["macro_complete@10"]
        peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device == "cuda" else 0.0
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5),
                        "train_time_s": train_time, "eval_time_s": eval_time,
                        "val": metrics, "peak_gpu_gb": round(peak, 2)})
        print(f"    epoch {epoch}: loss {np.mean(losses):.4f} "
              f"macro_complete@10 {score:.4f} "
              f"macro_recall@10 {metrics['macro_recall@10']:.4f} "
              f"({train_time}s train / {eval_time}s eval)", flush=True)

        if score > best["score"]:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                       ckpt_path)
            best = {"score": score, "epoch": epoch, "ckpt": str(ckpt_path),
                    "metrics": metrics}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("    early stop")
                break
        if args.max_steps:
            break

    return {"history": history, "best": best}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--pairs", default="models/reranker/pairs_dev.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs-max", type=int, default=5)
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--batch", type=int, default=CANONICAL["batch"])
    ap.add_argument("--grad-accum", type=int, default=CANONICAL["grad_accum"])
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--val-pool", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seed-mode", choices=["derived", "paired"], default="derived",
                    help="derived (default) keeps the frozen contract: the run seed is a "
                         "function of the hyper-parameters, so a re-run of either "
                         "configuration is exact. That also makes the seed change together "
                         "with the learning rate, which is why an lr *comparison* must use "
                         "paired: there the requested --seed is used verbatim for every "
                         "grid point, so lr and seed vary independently")
    ap.add_argument("--env-role", choices=["canonical", "compatibility"], default="canonical")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap optimizer batches per epoch (smoke tests)")
    ap.add_argument("--freeze-encoder-layers", type=int, default=0,
                    help="freeze this many leading encoder layers (0 = full fine-tuning). "
                         "On a 6 GB card the fp32 weights plus gradients of all 278M "
                         "parameters do not fit, so the top layers are trained instead and "
                         "the count is recorded in FROZEN.json")
    ap.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor",
                    help="Adafactor keeps factorised second moments (~MB) instead of two "
                         "full buffers per parameter; on a 6 GB card AdamW states alone "
                         "exhaust VRAM and the driver pages to system memory")
    ap.add_argument("--grid", default="1e-5:4,2e-5:4")
    ap.add_argument("--run-tag", default="",
                    help="suffix for the run directory/run_id, so a grid experiment cannot "
                         "collide with (or overwrite) the frozen checkpoint's directory")
    ap.add_argument("--no-freeze", action="store_true",
                    help="keep the run directory but write no COMPAT/FROZEN record and no "
                         "experiment-log row; used by 25_lr_grid.py, which is an experiment "
                         "rather than a model selection")
    ap.add_argument("--model-path", default="models/reranker/base",
                    help="local directory with the reranker weights; falls back to the hub id")
    ap.add_argument("--smoke", action="store_true", help="1 tiny run, no FROZEN write")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    settings = load_settings(root)
    seed = int(args.seed if args.seed is not None else
               settings.get("project", {}).get("random_seed", 20261013))

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # CPU environment: no CUDA build, no transformers
        print("REFUSING: no CUDA device (torch/transformers are not installed in this "
              f"interpreter: {exc}). Install the CUDA torch build in .venv-gpu.",
              file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("REFUSING: no CUDA device. Install the CUDA torch build in .venv-gpu.",
              file=sys.stderr)
        return 2
    gpu_name = torch.cuda.get_device_name(0)
    canonical = CANONICAL["gpu_substring"] in gpu_name
    role = args.env_role if not canonical else "canonical"
    if role == "canonical" and not canonical:
        print(f"REFUSING: {gpu_name} is not the canonical training environment. "
              f"Use --env-role compatibility to produce COMPAT.json.", file=sys.stderr)
        return 2

    pairs = read_jsonl(root / args.pairs)
    questions = read_jsonl(paths["annotations"] / "questions.jsonl")
    dev = [q for q in questions if q["split"] == "dev"]
    by_qid = {q["qid"]: q for q in dev}
    qtext = {q["qid"]: q["question"] for q in dev}
    val_qids = [qid for qid in {p["qid"] for p in pairs if p["role"] == "val"}]
    val_qids.sort()
    val_questions = [by_qid[q] for q in val_qids]

    corpus = goldset.load_corpus(root)
    ev = load_eval_module(root)
    gold_sets = {q["qid"]: ev.gold_spans(q, corpus) for q in val_questions}

    summary = json.loads((paths["reports"] / "index_summary.json").read_text(encoding="utf-8"))
    configs = sorted(summary.get("configs", {}))
    model_name = settings.get("embedding", {}).get("model", "BAAI/bge-small-en-v1.5")
    pools = build_val_pools(root, val_questions, configs, args.val_pool, model_name,
                            root / "models" / "reranker" / "val_pools.json")["pools"]

    local = root / args.model_path
    source = str(local) if (local / "config.json").exists() else MODEL_ID
    print(f"  model src  : {source}")
    tok = AutoTokenizer.from_pretrained(source)
    base_model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=1)
    base_model.to(args.device)

    print(f"=== reranker training ({role}) ===")
    print(f"  gpu        : {gpu_name}")
    print(f"  pairs      : {len(pairs)} total, "
          f"{sum(1 for p in pairs if p['role']=='train')} train / "
          f"{sum(1 for p in pairs if p['role']=='val')} val")
    print(f"  val configs: {len(pools)}  (top-{args.val_pool} pool, macro over configs)")

    cache: dict = {}
    dense = macro_scores(None, tok, pools, val_questions, gold_sets, ev, corpus,
                         args.device, args.eval_batch, use_reranker=False)
    print(f"  dense macro_complete@10 {dense['macro_complete@10']:.4f} "
          f"macro_recall@10 {dense['macro_recall@10']:.4f}  (no reranker, same pool)")

    base_model.eval()
    base = macro_scores(base_model, tok, pools, val_questions, gold_sets, ev, corpus,
                        args.device, args.eval_batch, cache=cache)
    print(f"  base  macro_complete@10 {base['macro_complete@10']:.4f} "
          f"macro_recall@10 {base['macro_recall@10']:.4f}  (off-the-shelf reranker)")
    # The baseline model is no longer needed: keeping 278M extra parameters resident
    # pushed the 6 GB card into allocator churn and roughly doubled the step time.
    base_model.to("cpu")
    del base_model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    grid = [(float(lr), int(neg)) for lr, neg in
            (item.split(":") for item in args.grid.split(","))]
    if args.smoke:
        grid = grid[:1]

    runs: list[dict] = []
    for lr, neg in grid:
        run_seed = seed + int(lr * 1e5) * 10 + neg
        if args.seed_mode == "paired":
            # identical seed for every grid point: the only factor that varies is the lr
            run_seed = seed
        print(f"\n  run lr={lr} negatives={neg} seed={run_seed}")
        model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=1)
        model.to(args.device)
        if args.freeze_encoder_layers:
            trainable = freeze_lower_layers(model, args.freeze_encoder_layers)
            print(f"    frozen first {args.freeze_encoder_layers} encoder layers; "
                  f"trainable params {trainable/1e6:.1f}M")
        cfg = {"lr": lr, "negatives": neg}
        ckpt = root / "models" / "reranker" / "tmp" / f"run{len(runs)}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        out = train_one(model=model, tok=tok, train_pairs=pairs, qtext=qtext, cfg=cfg,
                        device=args.device, args=args, ev=ev, corpus=corpus,
                        gold_sets=gold_sets, val_questions=val_questions, pools=pools,
                        seed=run_seed, ckpt_path=ckpt, cache=cache)
        best = out["best"]
        runs.append({"lr": lr, "negatives": neg, "seed": run_seed,
                     "history": out["history"], "best_epoch": best["epoch"],
                     "val": best["metrics"], "ckpt": best["ckpt"]})

    best = select_best(runs)
    print(f"\n  selected lr={best['lr']} negatives={best['negatives']} "
          f"macro_complete@10 {best['val']['macro_complete@10']:.4f}")

    if args.smoke:
        print("  smoke run: nothing written")
        return 0

    run_id = f"rk-{best['lr']:.0e}-n{best['negatives']}-s{best['seed']}".replace("+", "")
    if args.run_tag:
        run_id = f"{run_id}-{args.run_tag}"
    run_dir = root / "models" / "reranker" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    final = AutoModelForSequenceClassification.from_pretrained(source, num_labels=1)
    final.to(args.device)
    state = torch.load(best["ckpt"], map_location="cpu")
    final.load_state_dict(state)
    final.save_pretrained(run_dir)
    tok.save_pretrained(run_dir)

    ckpt_hash = sha256(run_dir / "model.safetensors") if (run_dir / "model.safetensors").exists() else None
    for r in runs:
        Path(r["ckpt"]).unlink(missing_ok=True)
    train_log = {"run_id": run_id, "dense_val": dense, "base_val": base,
                 "runs": [{k: v for k, v in r.items() if k != "ckpt"} for r in runs],
                 "selected": {"lr": best["lr"], "negatives": best["negatives"],
                              "seed": best["seed"], "best_epoch": best["best_epoch"],
                              "tie_break_ladder": TIE_BREAK}}
    write_json(run_dir / "train_log.json", train_log)

    if args.no_freeze:
        print(f"\nwrote {run_dir}")
        print("  --no-freeze: no COMPAT/FROZEN record and no experiment-log row")
        return 0

    stats = json.loads((root / "models" / "reranker" / "data_stats.json").read_text(encoding="utf-8"))
    frozen = {
        "kind": "reranker_freeze" if role == "canonical" else "reranker_compat",
        "created_at": now(), "env_role": role, "run_id": run_id,
        "model": {"id": MODEL_ID, "source": source, "checkpoint_sha256": ckpt_hash},
        "tokenizer": {"id": MODEL_ID, "source": source,
                      "max_query_len": QUERY_MAX, "max_pair_len": PAIR_MAX,
                      "truncation": "question<=128; chunk gets 512-len(q)-3 (tail cut)"},
        "data": {"pairs_sha256": stats.get("pairs_sha256"),
                 "split_groups_sha256": sha256(root / "models/reranker/split_groups.json"),
                 "train_questions": stats["split"]["train_questions"],
                 "val_questions": stats["split"]["val_questions"],
                 "train_groups": stats["split"]["train_groups"],
                 "val_groups": stats["split"]["val_groups"]},
        "code": {"scripts_sha256": None},
        "corpus": {"version": corpus.corpus_version},
        "env": {"gpu_model": gpu_name,
                "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
                "cuda": torch.version.cuda, "torch": torch.__version__,
                "precision": "fp16", "batch_size": args.batch,
                "grad_accum": args.grad_accum,
                "effective_batch": args.batch * args.grad_accum,
                "device": args.device},
        "training": {"objective": "bce_with_logits",
                     "optimizer": args.optimizer,
                     "frozen_encoder_layers": args.freeze_encoder_layers,
                     "lr": best["lr"],
                     "negatives_per_positive": best["negatives"],
                     "epochs_max": args.epochs_max,
                     "early_stop": {"metric": "macro_rerank_complete@10_top50",
                                    "patience": args.patience},
                     "warmup_ratio": 0.1, "max_grad_norm": 1.0},
        "selection": {"metric": "macro_rerank_complete@10", "macro_over": f"{len(configs)} configs",
                      "val_pool": args.val_pool, "tie_break_ladder": TIE_BREAK},
        "seed": best["seed"],
        "dense_val_metrics": {"macro_complete@10": dense["macro_complete@10"],
                              "macro_recall@10": dense["macro_recall@10"],
                              "macro_mrr": dense["macro_mrr"]},
        "val_metrics": {"macro_complete@10": best["val"]["macro_complete@10"],
                        "macro_recall@10": best["val"]["macro_recall@10"],
                        "macro_mrr": best["val"]["macro_mrr"]},
        "base_val_metrics": {"macro_complete@10": base["macro_complete@10"],
                             "macro_recall@10": base["macro_recall@10"],
                             "macro_mrr": base["macro_mrr"]},
        "truncation": stats.get("truncation"),
    }

    sys.path.insert(0, str(root / "scripts"))
    import importlib
    logmod = importlib.import_module("15_experiment_log")
    frozen["code"]["scripts_sha256"] = logmod.code_sha256(root)
    frozen["run_id"] = logmod.run_id(corpus_version=corpus.corpus_version,
                                     strategy="reranker", config=run_id, split="dev",
                                     track="rerank_train",
                                     code_sha=frozen["code"]["scripts_sha256"])

    target = root / "models" / "reranker" / ("FROZEN.json" if role == "canonical" else "COMPAT.json")
    write_json(target, frozen)
    logmod.append_log(root, {"kind": "training", "run_id": frozen["run_id"],
                             "track": "rerank_train", "env_role": role,
                             "model": MODEL_ID, "corpus_version": corpus.corpus_version,
                             "gpu": gpu_name, "seed": best["seed"],
                             "hyper": {"lr": best["lr"], "negatives": best["negatives"]},
                             "val": frozen["val_metrics"], "base_val": frozen["base_val_metrics"]})
    print(f"\nwrote {run_dir}")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
