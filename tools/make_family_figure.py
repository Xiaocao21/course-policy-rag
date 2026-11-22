"""make_family_figure.py - regenerate the per-family figure with readable labels.

The report figure is sized for a single-column Word page; at IEEE column width its labels are
too small. This script draws the same numbers from results/final/stats.json at a size and font
appropriate for the paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "family_bars.png"


def main() -> int:
    stats = json.loads((ROOT / "results/final/stats.json").read_text(encoding="utf-8"))
    fam = stats["by_strategy"]
    order = ["heading", "paragraph", "fixed/size512", "fixed/size384",
             "fixed/size256", "fixed/size128", "semantic"]
    labels = ["heading", "paragraph", "fixed/512", "fixed/384", "fixed/256",
              "fixed/128", "semantic"]
    c1 = [fam[k]["C1"]["complete@10"] for k in order]
    c3 = [fam[k]["C3"]["complete@10"] for k in order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.barh(y - 0.19, c1, height=0.36, label="dense top-50 (C1)", color="#7f8fa6")
    ax.barh(y + 0.19, c3, height=0.36, label="top-50 + reranker (C3)", color="#1f4e79")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Complete@10 (macro over the family's configurations)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
