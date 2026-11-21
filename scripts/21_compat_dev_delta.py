"""21_compat_dev_delta.py - paired, group-level comparison of two dev result sets.

`16_stats.py` compares tracks *within* one result set (C3 vs C1, C2 vs main, ...). This
script compares two result sets that were produced by the same code on the same machine
with only the reranker checkpoint differing - which is exactly the full-versus-partial
fine-tuning question.

It deliberately reuses the estimators from `16_stats.py` instead of restating them:

  * the resampling unit is `group_id`, because the dev split has 112 questions in 106
    groups and treating questions as independent would understate the variance;
  * configurations are collapsed by the pre-registered macro-average before resampling;
  * one shared bootstrap resample (B = 2000, seed 20261013) is used for both sides, so
    the delta is paired at the group level;
  * the p-value is the same two-sided bootstrap p as `16_stats.bootstrap_p`.

Both result sets must be compatibility-mode outputs (`results/compat_dev/...`); the script
refuses anything under `results/final` or `results/final_dev`, because a canonical result
must never be silently compared with an A5000 one.

Usage
    .venv\\Scripts\\python.exe scripts/21_compat_dev_delta.py \\
        --a results/compat_dev/dev_full-vs-partial/partial \\
        --b results/compat_dev/dev_full-vs-partial/full
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from common import PROJECT_ROOT, load_settings, read_jsonl, write_json  # noqa: E402
from importlib import import_module  # noqa: E402

TRACKS = ("main", "C1", "C2", "C3")
METRICS = ("complete@10", "recall@10", "mrr", "tok2048_recall")


def group_macro(perq_rows: list[dict], metric: str) -> tuple[list[str], np.ndarray]:
    """Per group, the macro average over configurations of one metric."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in perq_rows:
        g = r.get("group_id") or r["qid"]
        if metric in r:
            acc[g][r["config"]].append(float(r[metric]))
    groups = sorted(acc)
    out = np.array([float(np.mean([np.mean(v) for v in acc[g].values()])) for g in groups])
    return groups, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--a", required=True, help="reference result dir (e.g. partial)")
    ap.add_argument("--b", required=True, help="comparison result dir (e.g. full)")
    ap.add_argument("--label-a", default="partial")
    ap.add_argument("--label-b", default="full")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="", help="json output (default: <b>/delta_vs_a.json)")
    args = ap.parse_args()

    root = Path(args.root)
    a_dir = (root / args.a) if not Path(args.a).is_absolute() else Path(args.a)
    b_dir = (root / args.b) if not Path(args.b).is_absolute() else Path(args.b)
    for d in (a_dir, b_dir):
        if not d.exists():
            print(f"REFUSING: {d} does not exist", file=sys.stderr)
            return 2
        rel = d.resolve().as_posix()
        if "/results/final" in rel:
            print(f"REFUSING: {d} is a canonical result set. Compatibility results live "
                  f"under results/compat_dev/ and must not be compared with "
                  f"results/final or results/final_dev.", file=sys.stderr)
            return 2

    settings = load_settings(root)
    seed = int(args.seed if args.seed is not None else
               settings.get("project", {}).get("random_seed", 20261013))

    stats16 = import_module("16_stats")
    holm = stats16.holm
    bootstrap_p = stats16.bootstrap_p

    out: dict = {"kind": "compat_dev_paired_delta", "env_role": "compatibility",
                 "a": {"label": args.label_a, "dir": str(a_dir)},
                 "b": {"label": args.label_b, "dir": str(b_dir)},
                 "boot": args.boot, "seed": seed,
                 "note": ("paired group-level bootstrap; both result sets come from the "
                          "same machine and the same code path, so the only difference "
                          "is the reranker checkpoint. Not comparable with results/final*.")}

    # ---- shared resample over the union of groups (both sets must cover the same ones)
    rows = {}
    for label, d in (("a", a_dir), ("b", b_dir)):
        rows[label] = {}
        for t in TRACKS:
            path = d / f"perq_{t}.jsonl"
            if not path.exists():
                print(f"REFUSING: missing {path}", file=sys.stderr)
                return 2
            rows[label][t] = read_jsonl(path)

    groups, _ = group_macro(rows["a"]["C3"], "complete@10")
    groups_b, _ = group_macro(rows["b"]["C3"], "complete@10")
    if groups != groups_b:
        print("REFUSING: the two result sets cover different groups", file=sys.stderr)
        return 2

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), size=(args.boot, len(groups)))

    contrasts: dict[str, dict] = {}
    pvals: list[float] = []
    names: list[str] = []
    for t in ("C3", "C2", "C1"):
        for metric in METRICS if t != "C1" else ("complete@10",):
            _g, va = group_macro(rows["a"][t], metric)
            _g, vb = group_macro(rows["b"][t], metric)
            delta = vb - va
            boot = np.array([np.nanmean(delta[i]) for i in idx])
            name = f"{t}:{metric}"
            contrasts[name] = {
                "metric": metric, "track": t,
                "a_mean": round(float(np.nanmean(va)), 4),
                "b_mean": round(float(np.nanmean(vb)), 4),
                "delta_b_minus_a": round(float(np.nanmean(delta)), 4),
                "ci_low": round(float(np.nanpercentile(boot, 2.5)), 4),
                "ci_high": round(float(np.nanpercentile(boot, 97.5)), 4),
                "p_boot": round(bootstrap_p(boot), 4),
                "n_groups": len(groups),
                "n_groups_changed": int(np.sum(np.abs(delta) > 1e-12)),
            }
            if name.startswith("C3:") and metric == "complete@10":
                names.insert(0, name)      # primary first
                pvals.insert(0, contrasts[name]["p_boot"])
            else:
                names.append(name)
                pvals.append(contrasts[name]["p_boot"])

    for name, adj in zip(names, holm(pvals)):
        contrasts[name]["p_holm"] = round(adj, 4)

    out["contrasts"] = contrasts

    # ---- per configuration, so a level shift can be separated from a reshuffle
    def cfg_means(rows_list: list[dict], metric: str) -> dict[str, float]:
        acc: dict[str, list[float]] = defaultdict(list)
        for r in rows_list:
            acc[r["config"]].append(float(r[metric]))
        return {c: float(np.mean(v)) for c, v in acc.items()}

    cfg_delta: dict[str, dict] = {}
    for t in ("C3", "C2"):
        ma = cfg_means(rows["a"][t], "complete@10")
        mb = cfg_means(rows["b"][t], "complete@10")
        cfg_delta[t] = {c: round(mb[c] - ma[c], 4) for c in sorted(ma)}
    out["per_config_delta_complete@10"] = cfg_delta

    # ---- rank stability between the two checkpoints, per track
    def spearman_map(ma: dict[str, float], mb: dict[str, float]) -> float:
        cs = sorted(ma)
        return round(stats16.spearman([ma[c] for c in cs], [mb[c] for c in cs]), 4)

    out["spearman_between_checkpoints"] = {
        t: spearman_map(cfg_means(rows["a"][t], "complete@10"),
                        cfg_means(rows["b"][t], "complete@10")) for t in ("C3", "C2")}

    target = Path(args.out) if args.out else b_dir / "delta_vs_a.json"
    write_json(target, out)

    print(f"=== paired dev comparison: {args.label_b} - {args.label_a} "
          f"({len(groups)} groups, boot {args.boot}) ===")
    for name in names:
        c = contrasts[name]
        star = "  <-- primary" if name == "C3:complete@10" else ""
        print(f"  {name:<22} {args.label_a} {c['a_mean']:.3f} -> {args.label_b} "
              f"{c['b_mean']:.3f}  delta {c['delta_b_minus_a']:+.3f} "
              f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  p_holm={c['p_holm']:.3f}"
              f"{star}")
    print(f"  groups changed (C3 complete@10): "
          f"{contrasts['C3:complete@10']['n_groups_changed']}/{len(groups)}")
    print(f"  wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
