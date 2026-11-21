"""25_lr_grid.py - compare learning rates with the seed held fixed.

The frozen selection compared lr 1e-5 (val 0.3877) with lr 2e-5 (val 0.3422) and reported the
first as better. It could not support that reading: `18_train_reranker.py` derives the seed
from the hyper-parameters,

    run_seed = base_seed + 10 * (lr * 1e5) + negatives

so the two runs differ in the seed as well as in the learning rate. A reviewer spotted this.
This script re-runs the same two learning rates with the **same seed set**, so the paired
difference per seed isolates the learning rate.

What it does not do: it trains nothing new for the reported system. Every run is a
compatibility-mode re-run with the frozen recipe (10 of 12 encoder layers frozen, Adafactor,
batch 8 x accum 2), started from the same base weights, and no COMPAT/FROZEN record is
written (`--no-freeze`). The result belongs in the report as a compatibility experiment,
next to the disclosure that the frozen selection was seed-confounded.

Usage
    .venv-gpu\\Scripts\\python.exe scripts/25_lr_grid.py
    .venv-gpu\\Scripts\\python.exe scripts/25_lr_grid.py --lrs 1e-5,2e-5 --seeds 20261027,20261047

Output
    results/compat_lr/lr_grid.json     per-run validation metrics and the paired summary
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, write_json  # noqa: E402

DEFAULT_SEEDS = (20261027, 20261047, 20261057)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--lrs", default="1e-5,2e-5")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--negatives", type=int, default=4)
    ap.add_argument("--tag", default="lrgrid")
    ap.add_argument("--epochs-max", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.root)
    lrs = [lr.strip() for lr in args.lrs.split(",") if lr.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        print("REFUSING: a paired comparison needs at least two seeds", file=sys.stderr)
        return 2

    print(f"=== learning-rate grid (matched seeds) ===")
    print(f"  lrs   : {lrs}")
    print(f"  seeds : {seeds} (identical for every lr)")
    print(f"  recipe: 10 layers frozen, Adafactor, batch 8 x accum 2, {args.epochs_max} epochs max")

    runs = []
    for lr in lrs:
        for seed in seeds:
            run_dir = (root / "models" / "reranker" / "runs"
                       / f"rk-{float(lr):.0e}-n{args.negatives}-s{seed}-{args.tag}".replace("+", ""))
            log_path = run_dir / "train_log.json"
            if log_path.exists():
                print(f"  {lr} seed {seed}: already trained, reusing {run_dir.name}")
            else:
                cmd = [sys.executable, str(root / "scripts" / "18_train_reranker.py"),
                       "--root", str(root),
                       "--grid", f"{lr}:{args.negatives}",
                       "--seed", str(seed),
                       "--seed-mode", "paired",
                       "--freeze-encoder-layers", "10",
                       "--optimizer", "adafactor",
                       "--batch", "8", "--grad-accum", "2",
                       "--epochs-max", str(args.epochs_max),
                       "--env-role", "compatibility",
                       "--run-tag", args.tag, "--no-freeze"]
                print(f"  {lr} seed {seed}: training ...", flush=True)
                res = subprocess.run(cmd, cwd=str(root))
                if res.returncode != 0:
                    print(f"REFUSING: training failed for lr={lr} seed={seed}", file=sys.stderr)
                    return 3
                if not log_path.exists():
                    print(f"REFUSING: expected {log_path} after training", file=sys.stderr)
                    return 3
            data = json.loads(log_path.read_text(encoding="utf-8"))
            selected = data["selected"]
            best = next(r for r in data["runs"] if r["lr"] == float(lr) and r["seed"] == seed)
            runs.append({"lr": float(lr), "seed": seed,
                         "run_id": data["run_id"],
                         "best_epoch": best["best_epoch"],
                         "val": best["val"],
                         "dense_val": data["dense_val"]})

    by_lr = {}
    for lr in lrs:
        vals = [r["val"]["macro_complete@10"] for r in runs if r["lr"] == float(lr)]
        by_lr[lr] = {
            "n_seeds": len(vals),
            "mean": round(statistics.mean(vals), 4),
            "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
            "values": [round(v, 4) for v in vals],
        }

    paired = []
    for seed in seeds:
        if len(lrs) != 2:
            break
        a = next((r for r in runs if r["lr"] == float(lrs[0]) and r["seed"] == seed), None)
        b = next((r for r in runs if r["lr"] == float(lrs[1]) and r["seed"] == seed), None)
        if a and b:
            paired.append({
                "seed": seed,
                "lr_low": float(lrs[0]), "lr_high": float(lrs[1]),
                "complete_low": a["val"]["macro_complete@10"],
                "complete_high": b["val"]["macro_complete@10"],
                "delta_high_minus_low": round(b["val"]["macro_complete@10"]
                                              - a["val"]["macro_complete@10"], 4),
            })

    summary = {
        "kind": "lr_grid_matched_seeds",
        "created_at": now(),
        "env_role": "compatibility",
        "recipe": {"frozen_encoder_layers": 10, "optimizer": "adafactor",
                   "batch": 8, "grad_accum": 2, "negatives": args.negatives,
                   "epochs_max": args.epochs_max,
                   "selection_metric": "macro reranked Complete@10 over the top-50 pool"},
        "seeds": seeds, "lrs": [float(lr) for lr in lrs],
        "by_lr": by_lr, "runs": runs, "paired": paired,
        "note": ("Each run is a compatibility-mode re-run of the frozen recipe with the seed "
                 "held fixed across learning rates, so the paired difference isolates lr. "
                 "No COMPAT/FROZEN record is written and nothing here replaces the frozen "
                 "checkpoint."),
    }
    out = root / "results" / "compat_lr" / "lr_grid.json"
    write_json(out, summary)

    print("\n  validation macro Complete@10 by learning rate:")
    for lr, e in by_lr.items():
        sd = f"  sd {e['sd']:.4f}" if e["sd"] is not None else ""
        print(f"    lr {lr}: mean {e['mean']:.4f}{sd}  values {e['values']}")
    if paired:
        print("  paired by seed (high - low):")
        for p in paired:
            print(f"    seed {p['seed']}: {p['complete_low']:.4f} -> {p['complete_high']:.4f}"
                  f"  delta {p['delta_high_minus_low']:+.4f}")
        deltas = [p["delta_high_minus_low"] for p in paired]
        print(f"    mean paired delta {statistics.mean(deltas):+.4f}  "
              f"(range {min(deltas):+.4f} .. {max(deltas):+.4f}, n={len(deltas)} seeds)")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
