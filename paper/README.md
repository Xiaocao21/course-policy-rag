# Paper source

Build:

```powershell
$tectonic = "$env:LOCALAPPDATA\tectonic\tectonic.exe"
& $tectonic -X compile main.tex
```

Output: `main.pdf` (copied to `../docs/PAPER.pdf`).

Current state: **7 pages of main content + 1 page of references**, IEEEtran two-column,
17 references (15 scholarly + 2 model cards). Tables: corpus, 17 configurations, training settings, four-track test, per-family,
test2, matched token budget. Figure: per-family C1→C3 (`figures/final_strategy.png`).

The five references added beyond the student-verified literature-review list are marked for
verification in `../docs/PAPER_NOTES.md` before final submission.
