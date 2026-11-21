# Figures

The report figures live in `results/figures/` (they are generated from the committed result
files, not drawn by hand):

| figure | file | source |
|---|---|---|
| Complete@10 by track, with group bootstrap CI | `results/figures/final_tracks.png` | `results/final/stats.json` |
| chunking ranking before/after reranking | `results/figures/final_rank.png` | `results/final/perq_*.jsonl` |

Regenerate with:

```powershell
python scripts/23_make_report_figures.py
```

`results/dashboard.html` is a self-contained demo page built by `scripts/make_dashboard.py`.
