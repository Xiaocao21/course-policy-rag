"""make_teacher_brief.py - build the 3-5 page Word brief from the final results.

The brief is a submission artefact, so the build is scripted rather than hand-edited: the
numbers below are the ones produced by 16_stats.py and 19_rerank_eval.py, and the script
fails loudly if a figure or a source document is missing.

Run with the bundled Python that ships python-docx:

    python scripts/make_teacher_brief.py
"""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "TEACHER_BRIEF.docx"
FIG_TRACKS = ROOT / "results" / "figures" / "final_tracks.png"
FIG_RANK = ROOT / "results" / "figures" / "final_rank.png"
STATS = ROOT / "results" / "final" / "stats.json"


def style_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.9)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(9.5)
    normal.paragraph_format.space_after = Pt(3)
    normal.paragraph_format.line_spacing = 1.05

    for name, size in (("Heading 1", 12.5), ("Heading 2", 10.5)):
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0x1F, 0x3B, 0x57)
        style.paragraph_format.space_before = Pt(8)
        style.paragraph_format.space_after = Pt(3)


def _cap(text: str) -> str:
    """Capitalise the first character so every bold lead is followed consistently."""
    return text[:1].upper() + text[1:] if text else text


def heading(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def para(doc: Document, text: str, *, bold_lead: str | None = None) -> None:
    p = doc.add_paragraph()
    if bold_lead:
        run = p.add_run(bold_lead)
        run.bold = True
    p.add_run(_cap(text))
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def bullet(doc: Document, text: str, *, bold_lead: str | None = None) -> None:
    p = doc.add_paragraph(style="List Bullet")
    if bold_lead:
        run = p.add_run(bold_lead)
        run.bold = True
    p.add_run(_cap(text))
    p.paragraph_format.space_after = Pt(1.5)


def table(doc: Document, header: list[str], rows: list[list[str]],
          widths: list[float] | None = None) -> None:
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(8)
        hdr[i].paragraphs[0].paragraph_format.space_after = Pt(0)
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(value))
            run.font.size = Pt(8)
            cells[i].paragraphs[0].paragraph_format.space_after = Pt(0)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(doc: Document, path: Path, width_cm: float, caption: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing figure: {path}")
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(8)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(6)


def main() -> int:
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    c3 = stats["tracks"]["C3"]["complete@10"]
    main_t = stats["tracks"]["main"]["complete@10"]
    primary = stats["contrasts"]["C3-C1"]

    doc = Document()
    style_document(doc)

    title = doc.add_paragraph()
    run = title.add_run("Course-Policy RAG: Chunking and Reranking")
    run.bold = True
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor(0x1F, 0x3B, 0x57)
    title.paragraph_format.space_after = Pt(1)

    sub = doc.add_paragraph()
    run = sub.add_run("CPS5801 progress report · corpus v0.1.5 · 172 questions · "
                      "17 chunking configurations · two-stage retrieval with a frozen reranker")
    run.font.size = Pt(8.5)
    run.italic = True
    sub.paragraph_format.space_after = Pt(4)

    # ---------------------------------------------------------------- 1
    heading(doc, "1. Project status and completed work")
    para(doc, "265 pages were crawled from 13 institutions under robots.txt and crawl-delay "
              "rules; 36 documents entered the frozen corpus (14 course / 9 faculty / "
              "13 institution, 12 institutions), giving 1,988 clauses and 2,007 retrieval "
              "units. Refused or blocked sources are documented in COMPLIANCE.md.",
         bold_lead="Data collection. ")
    para(doc, "each document is stored twice: an evidence layer that preserves character "
              "offsets exactly, and a normalised retrieval layer with a character map back to "
              "evidence. Clause spans are validated and split pieces must tile their clause "
              "span (split-span errors: 0). Chunk sizes use the real model tokenizer "
              "(maximum 509 tokens, over-limit 0).",
         bold_lead="Preprocessing. ")
    para(doc, "the bi-encoder BAAI/bge-small-en-v1.5 is used frozen — 384 dimensions, CLS "
              "pooling, L2-normalised vectors, exact FAISS IndexFlatIP search. No encoder "
              "fine-tuning was performed, so all 17 configurations share one embedding model "
              "and differ only in how text is cut.",
         bold_lead="Retrieval system. ")
    para(doc, "four families, 17 configurations: fixed-length (128/256/384/512 tokens × "
              "overlap 0/32/64), paragraph, heading, and semantic splitting "
              "(0.70/0.75/0.80 thresholds).",
         bold_lead="Baselines. ")
    para(doc, "one cross-encoder, BAAI/bge-reranker-base, fine-tuned on the development split "
              "only and applied unchanged to every configuration. On a 6 GB laptop GPU the "
              "full 278 M-parameter update does not fit, so the first 10 of 12 encoder layers "
              "are frozen (85.6 M trainable) and Adafactor replaces AdamW.",
         bold_lead="Training. ")
    para(doc, "tool-assisted annotation produced 172 questions (112 development / 60 test, "
              "59 test groups); all were reviewed and accepted by the student. A clause counts "
              "only when the retrieved intervals contain its entire annotated span, and a "
              "question is complete only when one whole minimal sufficient evidence set is "
              "covered.",
         bold_lead="Evaluation. ")

    # ---------------------------------------------------------------- 2
    heading(doc, "2. Experimental log")
    para(doc, "Rows are taken from experiments/log.jsonl and experiments/aborted_runs.jsonl; "
              "failed and weaker runs are retained rather than deleted.")
    table(doc,
          ["date", "run", "model / training configuration", "seed", "result"],
          [["10-21", "index build", "bge-small, FAISS, 17 configs, 2,007 units", "20261013",
            "17 indexes, 0 over limit"],
           ["10-24", "pilot test", "dense top-10, no reranker", "20261013",
            "macro Complete@10 0.357"],
           ["10-25 01:59", "full fine-tune", "AdamW, batch 8×2, fp16", "20261027",
            "aborted: 6 GB paging, 20 min/epoch"],
           ["10-25 03:05", "full fine-tune", "AdamW, batch 2×8", "20261027",
            "aborted: optimizer state is the floor"],
           ["10-25 03:20", "full fine-tune", "Adafactor, batch 8×2", "20261027",
            "aborted: weights + gradients too large"],
           ["10-25 03:27", "fine-tune, lr 1e-5", "Adafactor, 10 layers frozen, 4 negatives",
            "20261027", "selected: val 0.388"],
           ["10-25 03:41", "fine-tune, lr 2e-5", "same, lr 2e-5", "20261037",
            "weaker: 0.342, kept in log"],
           ["10-28", "four-track test", "frozen reranker, 4 tracks × 17 configs", "20261013",
            "C3 0.405 vs main 0.358"]],
          widths=[2.2, 3.2, 6.6, 2.0, 4.6])

    # ---------------------------------------------------------------- 3
    heading(doc, "3. Preliminary results and analysis")
    para(doc, "The evaluation is a 2×2 design: candidate pool (top-10 vs top-50) × reranker "
              "(off vs on). Two contrasts are arithmetic identities — reranking a fixed "
              "ten-chunk set, or taking the first ten of a top-50, cannot change "
              "Recall@10 / Hit@10 / Complete@10 — so they are asserted (maximum difference "
              "0.0) rather than tested.")
    table(doc,
          ["track", "pool", "reranker", "Complete@10", "95% CI", "MRR"],
          [["main", "10", "no", f"{main_t['mean']:.3f}",
            f"[{main_t['ci_low']:.3f}, {main_t['ci_high']:.3f}]",
            f"{stats['tracks']['main']['mrr']['mean']:.3f}"],
           ["C1", "50", "no", f"{stats['tracks']['C1']['complete@10']['mean']:.3f}",
            f"[{stats['tracks']['C1']['complete@10']['ci_low']:.3f}, "
            f"{stats['tracks']['C1']['complete@10']['ci_high']:.3f}]",
            f"{stats['tracks']['C1']['mrr']['mean']:.3f}"],
           ["C2", "10", "yes", f"{stats['tracks']['C2']['complete@10']['mean']:.3f}",
            f"[{stats['tracks']['C2']['complete@10']['ci_low']:.3f}, "
            f"{stats['tracks']['C2']['complete@10']['ci_high']:.3f}]",
            f"{stats['tracks']['C2']['mrr']['mean']:.3f}"],
           ["C3", "50", "yes", f"{c3['mean']:.3f}",
            f"[{c3['ci_low']:.3f}, {c3['ci_high']:.3f}]",
            f"{stats['tracks']['C3']['mrr']['mean']:.3f}"]],
          widths=[2.4, 1.8, 2.4, 3.2, 4.4, 2.2])
    figure(doc, FIG_TRACKS, 12.6,
           "Figure 1. Evidence completeness by track on the test split; bars show the "
           "group-level bootstrap 95% CI (59 groups, B = 2000).")
    para(doc, f"the primary comparison C3 − C1 is {primary['delta']:+.3f} "
              f"[{primary['ci_low']:+.3f}, {primary['ci_high']:+.3f}] with a Holm-adjusted "
              f"p = {primary['p_holm']:.2f}: the direction is consistent but the interval "
              "includes zero. On the development split the same comparison was +0.110 "
              "[+0.079, +0.141], p = 0.001. The effect did not replicate at the same size, so "
              "the honest reading is suggestive rather than established.",
         bold_lead="Interpretation. ")
    para(doc, "reranking does help where it can. At a fixed ten-chunk pool it raises Recall@3 "
              "by +0.058 (Holm p = 0.014) and MRR by +0.036, and it narrows the gap at matched "
              "token budgets (recall@512 tokens: 0.215 → 0.254).")
    figure(doc, FIG_RANK, 8.2,
           "Figure 2. Chunking ranking before and after reranking; the order is essentially "
           "unchanged (Spearman ρ = 0.963).")
    para(doc, "first, the chunking ranking survives reranking: structural splitting "
              "(heading 0.833, paragraph 0.750) stays ahead of fixed-length splitting "
              "(best 0.717), and sentence-level semantic splitting remains unusable "
              "(0.000–0.067) because a sentence chunk cannot contain a multi-sentence clause. "
              "Second, the reranker shifts the level rather than the order — it adds roughly "
              "five points of completeness to every configuration, including weak ones.",
         bold_lead="What this suggests. ")

    # ---------------------------------------------------------------- 4
    heading(doc, "4. Problems, changes, and research decisions")
    bullet(doc, "a single normalised corpus silently breaks span annotation, so a dual-layer "
                "design with a character map was adopted.", bold_lead="Offsets. ")
    bullet(doc, "each split piece originally claimed its whole clause span, inflating "
                "completeness; per-piece evidence ranges plus a tiling assertion fixed it.",
           bold_lead="Over-claimed coverage. ")
    bullet(doc, "the first audit compared only question wording and missed 20 clauses used on "
                "both sides of the split. Clusters defined by shared gold evidence are now "
                "never divided; leakage is 0.", bold_lead="Leakage was real. ")
    bullet(doc, "the test split was used once as an exploratory pilot; the formal run is a "
                "re-evaluation with frozen settings and is reported as such. It is not an "
                "unseen holdout.", bold_lead="The split is no longer unseen. ")
    bullet(doc, "selecting on Complete@10 over a reranked top-10 selects a constant; selection "
                "now uses the top-50 pool, and the two invariant contrasts are asserted, not "
                "tested.", bold_lead="A broken selection metric. ")
    bullet(doc, "60 questions come from 59 groups and each appears under 17 configurations, so "
                "questions or cells are not independent; the bootstrap resamples groups.",
           bold_lead="Statistics. ")
    bullet(doc, "the 6 GB card could not hold a full AdamW update; Adafactor plus freezing 10 "
                "of 12 layers made training feasible at the cost of a weaker model. The A5000 "
                "was unavailable, so every number comes from the canonical 3060.",
           bold_lead="Hardware. ")

    # ---------------------------------------------------------------- 5
    heading(doc, "5. Remaining work and feasibility")
    table(doc,
          ["item", "status", "effort"],
          [["four-track test run", "done, one formal attempt", "—"],
           ["statistics and figures", "done", "—"],
           ["this brief (3–5 pages)", "done", "—"],
           ["fine-tune on an A5000", "optional; full fine-tuning reached 0.396 at epoch 1",
            "1–2 h GPU"],
           ["second held-out set from 1,239 unused clauses", "optional; restores an unseen split",
            "2 days"],
           ["expand sparse domains (resources: 7 test questions)", "optional", "1 day"]],
          widths=[8.4, 6.2, 2.4])
    para(doc, "The required work is complete. The optional items would strengthen the weak "
              "result in section 3 but are not needed for the assignment; the first is the "
              "cheapest and would also remove the partial fine-tuning limitation.")

    # ---------------------------------------------------------------- 6
    heading(doc, "6. AI disclosure")
    para(doc, "AI assistance was used, and the arrangement was approved by the supervisor for "
              "this course assignment.")
    bullet(doc, "questions and reference answers were drafted by an AI assistant (DeepSeek) "
                "from clauses in the frozen corpus, with provenance recorded on every question. "
                "The student reviewed all 172 questions, raised 23 specific defects (wrong "
                "answers, non-minimal evidence, unnatural pairings, leakage), which were "
                "corrected, and then accepted every question; the sign-off is timestamped in "
                "reports/lock_audit.jsonl.", bold_lead="Question drafting. ")
    bullet(doc, "pipeline scripts were written with AI assistance. Data transformations are "
                "asserted and covered by 95 automated tests; the corpus manifest hashes every "
                "artefact.", bold_lead="Code. ")
    bullet(doc, "this brief and the analysis narrative were drafted with AI assistance and "
                "reviewed by the student.", bold_lead="Text. ")
    bullet(doc, "the AI did not set the research question, the evaluation protocol, or the "
                "interpretation. No inter-annotator agreement is claimed, because there was no "
                "second human annotator.", bold_lead="What the AI did not do. ")
    bullet(doc, "BAAI/bge-small-en-v1.5 (frozen encoder), BAAI/bge-reranker-base (fine-tuned "
                "reranker), DeepSeek (annotation and drafting).", bold_lead="Models. ")

    doc.save(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
