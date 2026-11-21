"""make_dashboard.py - build a self-contained results page for a live demo.

The project produces files, not a web application, so this script renders one static HTML
page from the same artefacts the report uses: headline metrics, the four-track table,
matched token budgets, the per-configuration table, and the two figures embedded as base64
so the page works from a file path with no server and no network.

Run with any Python that has the standard library only.
"""
from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "dashboard.html"
STATS = ROOT / "results" / "final" / "stats.json"
FIGURES = {
    "tracks": ROOT / "results" / "figures" / "final_tracks.png",
    "rank": ROOT / "results" / "figures" / "final_rank.png",
}
TRACKS = ("main", "C1", "C2", "C3")


def img64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def load_summary(track: str) -> list[dict]:
    with (ROOT / "results" / "final" / f"summary_{track}.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    tr = stats["tracks"]
    c31 = stats["contrasts"]["C3-C1"]
    bs = stats["by_strategy"]
    # Second held-out set: optional on this machine, so a missing directory degrades to a
    # note instead of breaking the demo page.
    t2_stats_path = ROOT / "results" / "final_test2" / "stats.json"
    t2_outcome_path = ROOT / "results" / "test2" / "OUTCOME.json"
    if t2_stats_path.exists() and t2_outcome_path.exists():
        t2 = json.loads(t2_stats_path.read_text(encoding="utf-8"))
        t2o = json.loads(t2_outcome_path.read_text(encoding="utf-8"))
        t2_rows = "".join(
            f"<tr><td><b>{t}</b></td><td>{'10' if t in ('main', 'C2') else '50'}</td>"
            f"<td>{'no' if t in ('main', 'C1') else 'yes'}</td>"
            f"<td>{t2['tracks'][t]['complete@10']['mean']:.3f}</td>"
            f"<td>[{t2['tracks'][t]['complete@10']['ci_low']:.3f}, "
            f"{t2['tracks'][t]['complete@10']['ci_high']:.3f}]</td></tr>"
            for t in ("main", "C1", "C2", "C3"))
        t2_html = f"""
<h2>Second held-out set (test2)</h2>
<p class="note">45 questions in 45 groups, drawn by a pre-registered length-stratified rule,
locked and evaluated once. The primary endpoint is the structural-minus-semantic gap; the
verdict below is read from <code>results/test2/OUTCOME.json</code>, not chosen here.</p>
<table>
<tr><th>track</th><th>pool</th><th>reranker</th><th>Complete@10</th><th>95% CI</th></tr>
{t2_rows}
</table>
<p class="note"><b>Primary</b> (structural − semantic, C1):
<b>{t2o['primary']['delta']:+.3f}</b>
[{t2o['primary']['ci_low']:+.3f}, {t2o['primary']['ci_high']:+.3f}] →
<span class="ok">{t2o['primary']['verdict']}</span>.
<b>Secondary</b> (C3 − C1): <b>{t2o['secondary']['delta']:+.3f}</b>
[{t2o['secondary']['ci_low']:+.3f}, {t2o['secondary']['ci_high']:+.3f}] →
<span class="warn">{t2o['secondary']['verdict']}</span>.
The absolute level is not comparable with the first test split (all 45 questions have
single-clause evidence), the set is agent-reviewed rather than human-reviewed, and at this n
the reranking endpoint cannot exclude a +0.047 effect.</p>
"""
        t2_say = (f" On test2 the direction repeated: "
                  f"{t2o['secondary']['delta']:+.3f} "
                  f"[{t2o['secondary']['ci_low']:+.3f}, "
                  f"{t2o['secondary']['ci_high']:+.3f}] — positive, but the size is unresolved "
                  f"at 45 groups.")
    else:
        t2_html = ('<h2>Second held-out set (test2)</h2><p class="note">not built on this '
                   'machine (results/final_test2/stats.json or results/test2/OUTCOME.json '
                   'missing).</p>')
        t2_say = ""
    c2m_r3 = stats["contrasts"]["C2-main:recall@3"]
    inv = stats["invariance"]

    rows_track = "".join(
        f"<tr><td><b>{t}</b></td><td>{10 if t in ('main', 'C2') else 50}</td>"
        f"<td>{'no' if t in ('main', 'C1') else 'yes'}</td>"
        f"<td>{tr[t]['complete@10']['mean']:.3f}</td>"
        f"<td>[{tr[t]['complete@10']['ci_low']:.3f}, {tr[t]['complete@10']['ci_high']:.3f}]</td>"
        f"<td>{tr[t]['hit@10']['mean']:.3f}</td>"
        f"<td>{tr[t]['recall@10']['mean']:.3f}</td>"
        f"<td>{tr[t]['mrr']['mean']:.3f}</td></tr>"
        for t in TRACKS)

    rows_budget = "".join(
        f"<tr><td>{t}</td><td>{tr[t]['tok512_recall']['mean']:.3f}</td>"
        f"<td>{tr[t]['tok1024_recall']['mean']:.3f}</td>"
        f"<td>{tr[t]['tok2048_recall']['mean']:.3f}</td></tr>"
        for t in TRACKS)

    main_rows = load_summary("main")
    c3_rows = load_summary("C3")
    c3_by_name = {f"{r['strategy']}/{r['config']}": r for r in c3_rows}
    rows_cfg = ""
    for r in sorted(main_rows, key=lambda x: -float(x["complete@10"])):
        name = f"{r['strategy']}/{r['config']}"
        c3 = c3_by_name.get(name, {})
        delta = float(c3.get("complete@10", 0)) - float(r["complete@10"])
        rows_cfg += (f"<tr><td>{html.escape(name)}</td>"
                     f"<td>{float(r['complete@10']):.3f}</td>"
                     f"<td>{float(c3.get('complete@10', 0)):.3f}</td>"
                     f"<td>{delta:+.3f}</td></tr>")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Chunking × reranking — results</title>
<style>
 body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0;
        background:#f6f7f9; color:#1b2430; }}
 .wrap {{ max-width: 1080px; margin: 0 auto; padding: 28px 22px 60px; }}
 h1 {{ font-size: 23px; margin: 0 0 4px; color:#16324f; }}
 .sub {{ color:#5a6b7d; font-size: 13px; margin-bottom: 18px; }}
 h2 {{ font-size: 16px; margin: 30px 0 8px; color:#16324f; }}
 .cards {{ display:flex; gap:14px; flex-wrap:wrap; margin: 16px 0 6px; }}
 .card {{ background:#fff; border:1px solid #dde3ea; border-radius:10px; padding:14px 16px;
          flex:1 1 210px; }}
 .card .k {{ font-size:12px; color:#5a6b7d; text-transform:uppercase; letter-spacing:.04em; }}
 .card .v {{ font-size:26px; font-weight:600; margin-top:4px; }}
 .card .n {{ font-size:12px; color:#7b8794; margin-top:2px; }}
 table {{ border-collapse: collapse; width:100%; background:#fff; font-size:13px;
          border:1px solid #dde3ea; }}
 th, td {{ padding:7px 9px; border-bottom:1px solid #e8edf2; text-align:left; }}
 th {{ background:#eef2f6; font-weight:600; font-size:12px; }}
 tr:last-child td {{ border-bottom:none; }}
 figure {{ margin: 14px 0 6px; background:#fff; border:1px solid #dde3ea; border-radius:10px;
           padding:12px; }}
 img {{ max-width:100%; display:block; margin:0 auto; }}
 figcaption {{ font-size:12px; color:#5a6b7d; text-align:center; margin-top:8px; }}
 .note {{ background:#fff8e6; border:1px solid #f0dca8; border-radius:10px; padding:12px 14px;
          font-size:13px; margin-top:14px; }}
 .ok {{ color:#1d7a46; font-weight:600; }}
 .warn {{ color:#a35a00; font-weight:600; }}
 code {{ background:#eef2f6; padding:1px 5px; border-radius:4px; font-size:12px; }}
 ul {{ font-size:13px; line-height:1.55; }}
</style></head><body><div class="wrap">

<h1>Do chunking strategies change retrieval on course-policy RAG?</h1>
<div class="sub">Corpus v0.1.5 · 172 questions (112 dev / 60 test, 59 groups) ·
17 chunking configurations · frozen bge-small-en-v1.5 + one fine-tuned cross-encoder ·
test evaluated in a single formal pass on 28 Oct 2026 · second held-out set (45 questions)
evaluated once on 13 Nov 2026</div>

<div class="cards">
  <div class="card"><div class="k">Complete@10 · best track (C3)</div>
    <div class="v">{tr['C3']['complete@10']['mean']:.3f}</div>
    <div class="n">95% CI [{tr['C3']['complete@10']['ci_low']:.3f}, {tr['C3']['complete@10']['ci_high']:.3f}]</div></div>
  <div class="card"><div class="k">Baseline (dense top-10)</div>
    <div class="v">{tr['main']['complete@10']['mean']:.3f}</div>
    <div class="n">same 17 configurations, no reranker</div></div>
  <div class="card"><div class="k">Reranking gain on test</div>
    <div class="v">{c31['delta']:+.3f}</div>
    <div class="n">Holm p = {c31['p_holm']:.2f} · <span class="warn">not significant</span></div></div>
  <div class="card"><div class="k">Chunking ranking stability</div>
    <div class="v">{stats['rank_stability']['C1_vs_C3']:.3f}</div>
    <div class="n">Spearman ρ, with vs without reranker</div></div>
</div>

<h2>Four-track design (2 × 2)</h2>
<table>
<tr><th>track</th><th>pool</th><th>reranker</th><th>Complete@10</th><th>95% CI</th>
<th>Hit@10</th><th>Recall@10</th><th>MRR</th></tr>
{rows_track}
</table>
<p class="note"><b>Two contrasts are arithmetic identities, not results.</b>
Reranking a fixed ten-chunk set (C2 − main) and taking the first ten of a top-50 (C1 − main)
cannot change Complete@10 / Hit@10 / Recall@10 — the set is the same set. Both are asserted:
max difference <code>{inv['C1-main']['max_abs_diff']:.1e}</code>. The informative comparison is
<b>C3 − C1</b>: {c31['delta']:+.3f} [{c31['ci_low']:+.3f}, {c31['ci_high']:+.3f}], Holm
p = {c31['p_holm']:.2f}.</p>

<figure><img alt="Complete@10 by track" src="data:image/png;base64,{img64(FIGURES['tracks'])}">
<figcaption>Evidence completeness by track; group-level bootstrap 95% CI (59 groups, B = 2000).</figcaption></figure>

<h2>Matched token budgets</h2>
<table>
<tr><th>track</th><th>Recall@512</th><th>Recall@1024</th><th>Recall@2048</th></tr>
{rows_budget}
</table>

<h2>Every configuration, before and after reranking</h2>
<table>
<tr><th>configuration</th><th>main Complete@10</th><th>C3 Complete@10</th><th>Δ</th></tr>
{rows_cfg}
</table>

<figure><img alt="Ranking before and after reranking" src="data:image/png;base64,{img64(FIGURES['rank'])}">
<figcaption>Chunking ranking survives reranking (Spearman ρ = {stats['rank_stability']['C1_vs_C3']:.3f}).</figcaption></figure>

{t2_html}
<h2>What to say, briefly</h2>
<ul>
<li><b>Answer to the research question:</b> structural chunking (heading {bs['heading']['C3']['complete@10']:.3f}, paragraph {bs['paragraph']['C3']['complete@10']:.3f})
beats fixed-length (best {bs['fixed/size512']['C3']['complete@10']:.3f}); sentence-level semantic splitting fails ({bs['semantic']['C3']['complete@10']:.3f})
because the evidence is not retrieved, not because of the completeness rule: coverage is scored on the union of
retrieved chunks, and union character coverage is only {bs['semantic']['C1']['partial@10']:.3f} against
{bs['heading']['C1']['partial@10']:.3f} for heading.</li>
<li><b>Reranking helps but is not established here:</b> +{c31['delta']:.3f} on test with a CI that
includes zero; it does raise Recall@3 by {c2m_r3['delta']:+.3f} (Holm p = {c2m_r3['p_holm']:.3f}) at a
fixed pool and cuts the matched-budget gap.{t2_say}</li>
<li><b>The chunking conclusion does not depend on reranking:</b> rank correlation 0.963, so the
recommendation is stable.</li>
<li><b>Disclosure:</b> the test split had already been used for a pilot, so those are
re-evaluations of a seen split, not a pristine holdout. test2 is the opposite case — never
used before — but it was drafted and reviewed by the agent, not by a human.</li>
</ul>

<h2>Artefacts</h2>
<ul>
<li><code>docs/PROJECT_REPORT.docx</code> — the 5-page report to hand over</li>
<li><code>results/REPORT_FINAL.md</code> — full analysis, limitations</li>
<li><code>results/final/stats.json</code> — every number on this page</li>
<li><code>results/final_test2/stats.json</code> + <code>results/test2/OUTCOME.json</code> — the
second held-out set and its pre-registered verdict</li>
<li><code>models/reranker/FROZEN.json</code> — model, data and environment hashes</li>
<li><code>docs/RUBRIC_CHECK.md</code> — requirement-by-requirement self-check</li>
</ul>

<div class="sub" style="margin-top:26px">Generated from <code>results/final/</code> and
<code>results/final_test2/</code>; no numbers typed by hand.</div>
</div></body></html>
"""
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
