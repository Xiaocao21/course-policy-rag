"""23_make_report_figures.py - regenerate every figure the report embeds.

Why this exists: `results/figures/final_tracks.png` and `final_rank.png` were produced once,
by hand, and then read by three different consumers (`make_project_report.py`,
`make_teacher_brief.py`, `make_dashboard.py`). Nothing in the repository could rebuild them,
so a re-run of the statistics could silently disagree with the submitted report. This script
makes the figures a function of `results/final/stats.json`.

Figures
    final_tracks.png    Complete@10 per track with group-level 95% CI
    final_rank.png      per-configuration Complete@10 before vs after reranking
    final_strategy.png  per-family Complete@10 with CI, C1 vs C3 (used next to the
                        aggregate table, so the families are visible without reading a CSV)

Usage
    .venv\\Scripts\\python.exe scripts/23_make_report_figures.py
    .venv\\Scripts\\python.exe scripts/23_make_report_figures.py --results results/final_dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, write_json  # noqa: E402

TRACKS = ("main", "C1", "C2", "C3")
FAMILY_ORDER = ["heading", "paragraph", "fixed/size512", "fixed/size384",
                "fixed/size256", "fixed/size128", "semantic"]


def plt_module():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_tracks(plt, stats: dict, out: Path) -> None:
    tracks = list(TRACKS)
    means = [stats["tracks"][t]["complete@10"]["mean"] for t in tracks]
    lows = [stats["tracks"][t]["complete@10"]["ci_low"] for t in tracks]
    highs = [stats["tracks"][t]["complete@10"]["ci_high"] for t in tracks]
    colors = ["#9aa5b1", "#9aa5b1", "#9aa5b1", "#2f6f9f"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
    bars = ax.bar(tracks, means, color=colors, width=0.6)
    ax.errorbar(tracks, means, yerr=[[m - l for m, l in zip(means, lows)],
                                    [h - m for m, h in zip(means, highs)]],
                fmt="none", ecolor="#1b1f24", elinewidth=1.2, capsize=4)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.015, f"{mean:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Complete@10 (macro over 17 configurations)")
    ax.set_ylim(0, max(highs) + 0.08)
    ax.set_title("Evidence completeness by track (95% group bootstrap CI)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_rank(plt, stats: dict, out: Path) -> None:
    per_c1 = stats["per_config_complete@10"]["C1"]
    per_c3 = stats["per_config_complete@10"]["C3"]
    cfgs = sorted(per_c1, key=lambda c: -per_c1[c])
    fig, ax = plt.subplots(figsize=(5.0, 4.6), dpi=160)
    ax.plot([0, 1], [0, 1], color="#c9ced6", linewidth=0.8, zorder=0)
    ax.scatter([per_c1[c] for c in cfgs], [per_c3[c] for c in cfgs],
               s=28, color="#2f6f9f", zorder=3)
    for c in cfgs:
        label = c.split("/")[-1] if len(c.split("/")) == 2 else c
        ax.annotate(label, (per_c1[c], per_c3[c]), fontsize=5.6,
                    xytext=(3, 3), textcoords="offset points", color="#3a4149")
    rho = stats["rank_stability"]["C1_vs_C3"]
    ax.set_xlabel("Complete@10, dense top-50 (C1)")
    ax.set_ylabel("Complete@10, reranked (C3)")
    ax.set_title(f"Configuration ranking before vs after reranking (Spearman ρ = {rho:.3f})",
                 fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_strategy(plt, stats: dict, out: Path) -> None:
    fams = [f for f in FAMILY_ORDER if f in stats.get("by_strategy", {})]
    if not fams:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
    y = range(len(fams))
    h = 0.36
    for offset, track, color, label in ((-h / 2, "C1", "#9aa5b1", "dense top-50 (C1)"),
                                        (h / 2, "C3", "#2f6f9f", "reranked (C3)")):
        vals = [stats["by_strategy"][f][track]["complete@10"] for f in fams]
        lo = [stats["by_strategy"][f][track]["ci_low"] for f in fams]
        hi = [stats["by_strategy"][f][track]["ci_high"] for f in fams]
        ax.barh([i + offset for i in y], vals, height=h, color=color, label=label)
        ax.errorbar(vals, [i + offset for i in y],
                    xerr=[[v - l for v, l in zip(vals, lo)], [hh - v for v, hh in zip(vals, hi)]],
                    fmt="none", ecolor="#1b1f24", elinewidth=1.0, capsize=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(fams, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Complete@10 (group-level mean over the family's configurations)")
    ax.set_title("Per-strategy completeness, before and after reranking", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--results", default="results/final")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    root = Path(args.root)
    stats_path = root / args.results / "stats.json"
    if not stats_path.exists():
        print(f"missing {stats_path}; run 16_stats.py first", file=sys.stderr)
        return 2
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    plt = plt_module()
    fig_tracks(plt, stats, out / "final_tracks.png")
    fig_rank(plt, stats, out / "final_rank.png")
    fig_strategy(plt, stats, out / "final_strategy.png")

    manifest = {
        "source": args.results + "/stats.json",
        "split": stats.get("split"), "groups": stats.get("groups"),
        "figures": ["final_tracks.png", "final_rank.png", "final_strategy.png"],
        "note": ("regenerated from stats.json so the report cannot drift from the "
                 "statistics it quotes"),
    }
    write_json(out / "figures_manifest.json", manifest)
    print(f"wrote {out / 'final_tracks.png'}")
    print(f"wrote {out / 'final_rank.png'}")
    print(f"wrote {out / 'final_strategy.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
