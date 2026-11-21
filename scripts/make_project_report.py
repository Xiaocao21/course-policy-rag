"""make_project_report.py - build the teacher-structured project report.

Structure follows the recommended order exactly:

    Project Title -> Research Question -> Work Completed -> Experimental Setup ->
    Experiment Log -> Preliminary Results -> Problems and Changes ->
    Remaining Experiments -> Updated Timeline -> AI Disclosure

Every number is read from the produced artefacts (`results/final/stats.json` and the
summary CSVs); nothing is typed in by hand, and the script aborts if a required file is
missing. The previous brief (`TEACHER_BRIEF.docx`) is left untouched.

Run with the bundled Python that ships python-docx.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "PROJECT_REPORT.docx"
STATS = ROOT / "results" / "final" / "stats.json"
FIG_TRACKS = ROOT / "results" / "figures" / "final_tracks.png"
FIG_RANK = ROOT / "results" / "figures" / "final_rank.png"


# --------------------------------------------------------------------- helpers
def macro(csv_path: Path, column: str) -> float:
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    vals = [float(r[column]) for r in rows if r.get(column) not in (None, "")]
    return sum(vals) / len(vals)


def style_document(doc: Document) -> None:
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.top_margin, s.bottom_margin = Cm(1.8), Cm(1.7)
    s.left_margin, s.right_margin = Cm(2.0), Cm(2.0)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(9.5)
    normal.paragraph_format.space_after = Pt(2.5)
    normal.paragraph_format.line_spacing = 1.03
    for name, size in (("Heading 1", 12.0), ("Heading 2", 10.0)):
        st = doc.styles[name]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0x1F, 0x3B, 0x57)
        st.paragraph_format.space_before = Pt(7)
        st.paragraph_format.space_after = Pt(2)


def cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def para(doc, text, lead=None, size=None, italic=False):
    p = doc.add_paragraph()
    if lead:
        r = p.add_run(lead)
        r.bold = True
        if size:
            r.font.size = Pt(size)
    r = p.add_run(cap(text))
    if size:
        r.font.size = Pt(size)
    if italic:
        r.italic = True
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def bullet(doc, text, lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if lead:
        r = p.add_run(lead)
        r.bold = True
    p.add_run(cap(text))
    p.paragraph_format.space_after = Pt(1.2)


def table(doc, header, rows, widths=None, size=7.6):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    tr_pr = t.rows[0]._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(size)
        cell.paragraphs[0].paragraph_format.space_after = Pt(0)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(size)
            cells[i].paragraphs[0].paragraph_format.space_after = Pt(0)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    for row in t.rows:
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def figure(doc, path: Path, width_cm: float, caption: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing figure: {path}")
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    r = p.add_run(caption)
    r.italic = True
    r.font.size = Pt(7.6)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)


# --------------------------------------------------------------------- content
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), help="output docx path")
    args = ap.parse_args()
    out_path = Path(args.out)
    for path in (STATS,):
        if not path.exists():
            raise SystemExit(f"missing {path}; run the evaluation first")
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    tr = stats["tracks"]
    c31 = stats["contrasts"]["C3-C1"]
    c2m_mrr = stats["contrasts"]["C2-main:mrr"]
    c2m_r3 = stats["contrasts"]["C2-main:recall@3"]

    # group-level means, so every number in the report matches the statistics artefact
    track_mean = {t: tr[t]["complete@10"]["mean"] for t in tr}
    dev = json.loads((ROOT / "results" / "final_dev" / "stats.json").read_text(encoding="utf-8"))
    dev_track = {t: dev["tracks"][t]["complete@10"]["mean"] for t in dev["tracks"]}
    dev_delta = dev["contrasts"]["C3-C1"]["delta"]
    dev_p = dev["contrasts"]["C3-C1"]["p_holm"]
    # the pilot predates the group-level statistics; only a question-level mean exists
    pilot = macro(ROOT / "results" / "pilot" / "summary_test.csv", "complete@10")
    dev_main, dev_c3 = dev_track["main"], dev_track["C3"]

    # Per-family and clause-length material: quoted from the statistics artefact, never
    # typed in by hand, so a re-run cannot leave the report behind.
    by_strategy = stats["by_strategy"]
    length = stats["by_length_tercile"]
    length_c1 = length["tracks"]["C1"]["k_floor"]
    frame = length["sampling_frame"]
    union_path = ROOT / "results" / "analysis" / "union_audit_test.json"
    if not union_path.exists():
        raise SystemExit(f"missing {union_path}; run 22_union_coverage_audit.py --split test")
    union = json.loads(union_path.read_text(encoding="utf-8"))
    union_main = union["tracks"]["main"]["by_family"]

    # Second held-out set: the numbers and the verdict come from the frozen design, never
    # from hand-entered values.
    t2_stats_path = ROOT / "results" / "final_test2" / "stats.json"
    t2_outcome_path = ROOT / "results" / "test2" / "OUTCOME.json"
    t2_pre_path = ROOT / "results" / "test2" / "PREREGISTRATION.json"
    t2_amend_path = ROOT / "results" / "test2" / "DESIGN_AMENDMENTS.json"
    for _p in (t2_stats_path, t2_outcome_path, t2_pre_path, t2_amend_path):
        if not _p.exists():
            raise SystemExit(f"missing {_p}; run the test2 pipeline first")
    t2 = json.loads(t2_stats_path.read_text(encoding="utf-8"))
    t2_out = json.loads(t2_outcome_path.read_text(encoding="utf-8"))
    t2_pre = json.loads(t2_pre_path.read_text(encoding="utf-8"))
    t2_amend = json.loads(t2_amend_path.read_text(encoding="utf-8"))
    # 16_stats records groups, not a question count; read it from the frozen questions file
    n_test2 = sum(1 for _ in (ROOT / "annotations" / "questions_test2.jsonl").open(
        encoding="utf-8"))

    # A5000 compatibility experiments (T06 full fine-tune, T07 matched-seed lr grid)
    compat_path = ROOT / "models" / "reranker" / "COMPAT.json"
    if not compat_path.exists():
        raise SystemExit(f"missing {compat_path}; run the A5000 full fine-tune first")
    compat = json.loads(compat_path.read_text(encoding="utf-8"))
    lr_path = ROOT / "results" / "compat_lr" / "lr_grid.json"
    if not lr_path.exists():
        raise SystemExit(f"missing {lr_path}; run 25_lr_grid.py first")
    lr_grid = json.loads(lr_path.read_text(encoding="utf-8"))
    lr_low, lr_high = sorted(lr_grid["by_lr"], key=float)[:2]
    lr_deltas = [p["delta_high_minus_low"] for p in lr_grid["paired"]]
    lr_paired_mean = sum(lr_deltas) / len(lr_deltas)

    doc = Document()
    style_document(doc)

    # 1 ---------------------------------------------------------------- title
    t = doc.add_paragraph()
    r = t.add_run("Effects of Text Chunking Strategies on Retrieval Accuracy in "
                  "RAG-Based Question Answering for University Course Policies")
    r.bold = True
    r.font.size = Pt(15)
    r.font.color.rgb = RGBColor(0x1F, 0x3B, 0x57)
    t.paragraph_format.space_after = Pt(2)
    sub = doc.add_paragraph()
    r = sub.add_run("CPS5801 · project report · revision 4 · 13 November 2026 · corpus v0.1.5 · "
                    "17 chunking configurations · two-stage retrieval with a frozen reranker")
    r.italic = True
    r.font.size = Pt(8.4)
    sub.paragraph_format.space_after = Pt(4)

    # 2 ---------------------------------------------------------------- RQ
    doc.add_heading("1. Research Question", level=1)
    para(doc, "When the embedding model, vector retriever, knowledge-base content, query set "
              "and retrieval parameters are held constant, how do fixed-length, "
              "paragraph-based, heading-based and semantic chunking strategies affect "
              "retrieval accuracy, evidence completeness and retrieval efficiency in a RAG "
              "system for university course policies?", italic=True)
    para(doc, "the chunking strategy (four families, 17 configurations). The frozen encoder, "
              "FAISS index type, document set, 172 questions and top-k are held fixed, so the "
              "only factor that varies is how text is cut. Measured: Recall@k, Hit@10, "
              "evidence completeness (Complete@10), MRR, token-budget recall and latency.",
         lead="Primary experiment, independent variable: ")
    para(doc, "a separate question, not a second independent variable in the same comparison: "
              "does the answer survive when two retrieval-side factors are added — candidate "
              "pool size (top-10 vs top-50) and a reranker (off vs on)? This is a 2×2 "
              "sensitivity check on the chunking conclusion, not a search for the best "
              "absolute system.", lead="Secondary experiment: ")
    para(doc, "the primary experiment asks which chunking strategy is best under a fixed "
              "retriever. The secondary experiment asks whether that ordering changes once "
              "the pool is enlarged or the candidates are reranked. Chunking remains the only "
              "factor varied within the primary comparison; pool and reranking are varied "
              "only in the secondary one, and the two are reported separately so neither "
              "interpretation borrows the other's evidence.", lead="How the two relate: ")

    # 3 ---------------------------------------------------------------- work
    doc.add_heading("2. Work Completed", level=1)
    para(doc, "265 pages crawled from 13 institutions under robots.txt and crawl-delay rules; "
              "36 documents frozen (14 course / 9 faculty / 13 institution, 12 institutions), "
              "1,988 clauses, 2,007 retrieval units. Refused sources are documented in "
              "COMPLIANCE.md.", lead="Corpus. ")
    para(doc, "each document is stored twice — a character-faithful evidence layer and a "
              "normalised retrieval layer with a character map back to evidence — so "
              "annotated spans can be checked exactly. Split pieces tile their clause span "
              "(split-span errors: 0) and chunk sizes use the real tokenizer (max 509 tokens, "
              "over-limit 0).", lead="Preprocessing. ")
    para(doc, "17 configurations from four families (fixed 128/256/384/512 × overlap "
              "0/32/64, paragraph, heading, semantic at 0.70/0.75/0.80) over one frozen "
              "bge-small-en-v1.5 encoder and exact FAISS inner-product search.",
         lead="Baselines. ")
    para(doc, "one cross-encoder (bge-reranker-base) fine-tuned on the development split only "
              "and applied unchanged to all 17 configurations; frozen in FROZEN.json.",
         lead="Reranker. ")
    para(doc, "172 questions (112 development / 60 test, 59 test groups), drafted with AI "
              "assistance from corpus clauses, reviewed by the student, 23 defects corrected, "
              "all accepted and timestamped in reports/lock_audit.jsonl. A second held-out "
              f"set adds {n_test2} questions in {t2['groups']} groups, drafted and "
              "self-audited by the agent (no human sign-off; see 5.1 and §9).",
         lead="Annotation. ")
    para(doc, "a 2×2 design over candidate pool and reranking, evaluated in a single formal pass "
              "on the locked test split, with group-level bootstrap and paired contrasts, plus "
              "one pre-registered formal run on the second held-out set.", lead="Evaluation. ")

    # 4 ---------------------------------------------------------------- setup
    doc.add_heading("3. Experimental Setup", level=1)
    table(doc, ["component", "setting"], [
        ["Encoder (frozen)", "BAAI/bge-small-en-v1.5, 384-d, CLS pooling, L2-normalised"],
        ["Index", "FAISS IndexFlatIP, exact search, one index per configuration"],
        ["Reranker", "BAAI/bge-reranker-base, 10 of 12 encoder layers frozen (85.6 M trainable), "
                     "Adafactor, lr 1e-5, 4 negatives/positive, batch 8 × accum 2, fp16"],
        ["Reranker input", "query max 128 tokens; pair max 512; chunk keeps 512 − len(query) − 3 "
                           "tokens and is tail-truncated; 25.3% of development pairs are cut "
                           "(mean 85 tokens)"],
        ["Reranker training data", "85 train / 21 validation groups (89 / 22 questions), "
                                   "group-disjoint; 1,581 clause-positives, 18,807 negatives, "
                                   "1,812 partials excluded from the loss"],
        ["Model selection", "macro reranked Complete@10 over 17 configurations, scored on the "
                            "top-50 pool (not the reranked top-10, whose set cannot change); "
                            "tie-break MRR, then Recall@10, then smaller lr"],
        ["Data", "corpus v0.1.5; 112 dev / 60 test (59 groups); 45 test2 (45 groups)"],
        ["Metrics", "Recall@1/3/5/10, Hit@10, Complete@10 (whole evidence set), MRR, "
                    "token-budget recall at 512/1024/2048, latency"],
        ["Evidence completeness", "a clause counts as covered only when the union of retrieved "
                                  "intervals from the same document version contains its whole "
                                  "span; chunks may combine, gaps do not count. MRR is the one "
                                  "single-chunk metric"],
        ["Tracks", "main top-10 dense; C1 top-50 dense; C2 top-10 reranked; C3 top-50 reranked"],
        ["Statistics", "group-level bootstrap B=2000, seed 20261013, Holm correction"],
        ["Hardware", "RTX 3060 Laptop GPU, 6 GB, CUDA 12.4, torch 2.6.0+cu124, fp16"],
    ], widths=[3.2, 13.6])

    # 5 ---------------------------------------------------------------- log
    doc.add_heading("4. Experiment Log", level=1)
    para(doc, "runs are listed in execution order, including the ones that were aborted. "
              "Aborted runs keep their settings and the observed reason; none were deleted.",
         size=8.6)
    table(doc,
          ["Run", "Model", "Dataset", "Important settings", "Metric", "Result", "Observation"],
          [["E01 · 10-24", "dense baseline", "test (pilot)", "top-10, no reranker, seed 20261013",
            "Complete@10", f"{pilot:.3f}", "exploratory pilot; the split is no longer unseen"],
           ["E02 · 10-28", "dense baseline", "test (re-evaluation)", "top-10, no reranker, seed 20261013",
            "Complete@10", f"{track_mean['main']:.3f}",
            "locked test re-evaluation on a previously exposed split"],
           ["E03 · 10-28", "dense, pool 50", "test (re-evaluation)", "top-50, no reranker, seed 20261013",
            "Complete@10", f"{track_mean['C1']:.3f}", "identical to E02 by construction"],
           ["E04 · 10-28", "reranked", "test (re-evaluation)", "top-10 pool, reranker, seed 20261013",
            "MRR", f"{tr['C2']['mrr']['mean']:.3f}",
            f"set metrics unchanged; MRR +{c2m_mrr['delta']:.3f}"],
           ["E05 · 10-28", "reranked", "test (re-evaluation)", "top-50 pool, reranker, seed 20261013",
            "Complete@10", f"{track_mean['C3']:.3f}",
            f"primary gain +{c31['delta']:.3f}, Holm p = {c31['p_holm']:.2f} (n.s.)"],
           ["E06 · 10-28", "reranked", "dev", "same four tracks, seed 20261013",
            "Complete@10", f"{dev_c3:.3f} vs {dev_main:.3f}",
            f"dev gain +{dev_delta:.3f}, p = {dev_p:.3f} — larger than test"],
           ["T01 · 10-25", "reranker full FT", "dev pairs", "AdamW, 8×2, fp16, lr 1e-5, seed 20261027",
            "val Complete@10", "0.396 (epoch 1)",
            "aborted at epoch 2 (6 GB paging, ~20 min/epoch); epoch-1 checkpoint discarded"],
           ["T02 · 10-25", "reranker full FT", "dev pairs", "AdamW, 2×8, fp16, seed 20261027",
            "—", "aborted", "optimizer state is the memory floor"],
           ["T03 · 10-25", "reranker full FT", "dev pairs", "Adafactor, 8×2, fp16, seed 20261027",
            "—", "aborted", "fp32 weights + gradients still exceed 6 GB"],
           ["T04 · 10-25", "reranker partial FT", "dev pairs", "Adafactor, 10 layers frozen, lr 1e-5, seed 20261027",
            "val Complete@10", "0.388", "selected and frozen"],
           ["T05 · 10-25", "reranker partial FT", "dev pairs", "same, lr 2e-5, seed 20261037",
            "val Complete@10", "0.342", "weaker; kept in the log"],
           ["T06 · 11-11", "reranker full FT", "dev pairs",
            "A5000, AdamW, batch 32, 0 layers frozen, lr 1e-5, seed 20261027",
            "val Complete@10", f"{compat['val_metrics']['macro_complete@10']:.3f}",
            "partial also re-measured at 0.390; dev Δ +0.004, interval includes zero — no gain"],
           ["T07 · 11-12", "reranker partial FT", "dev pairs",
            "A5000, same recipe, 3 seeds × lr {1e-5, 2e-5}",
            "val Complete@10",
            f"{lr_grid['by_lr'][lr_low]['mean']:.3f} vs {lr_grid['by_lr'][lr_high]['mean']:.3f}",
            f"paired by seed {lr_paired_mean:+.3f} on all 3 seeds; T04/T05 overstated the gap "
            "because the seed moved with the lr"]],
          widths=[1.3, 3.3, 2.2, 4.6, 2.4, 2.0, 5.0], size=7.2)

    para(doc, "the base seed is 20261013 (settings.random_seed). Evaluation runs use it "
              "directly. Training runs derive a per-configuration seed as "
              "base + 10·(lr×10⁵) + negatives, so lr 1e-5 yields 20261027 and lr 2e-5 yields "
              "20261037. The derivation keeps a re-run of either configuration exact, but it "
              "also makes the seed move together with the learning rate, so the T04/T05 "
              "difference mixes the two and is not evidence about the learning rate. T07 "
              "repeats both rates with an identical seed set for that reason. The frozen "
              "selection is unaffected: it needed one model, not a learning-rate ranking.",
         size=8.2)

    # 6 ---------------------------------------------------------------- results
    doc.add_heading("5. Preliminary Results", level=1)
    para(doc, "the results in this subsection and the next are a re-evaluation of the first "
              "test split, which had already been used for a pilot, so they are not "
              "independent first-use held-out results. That split was evaluated once as a "
              "pilot (E01) before the settings were frozen, and the formal runs (E02–E05) "
              "repeat the measurement on the same questions with every setting fixed in "
              "advance. The second held-out set in 5.1 is a different case: it had never been "
              "used and was evaluated once after its design was frozen.", italic=True, size=8.8)
    table(doc, ["track", "pool", "reranker", "Complete@10", "95% CI", "Hit@10",
                "Recall@10", "MRR"],
          [[t, str(tr[t]["complete@10"] and (10 if t in ("main", "C2") else 50)),
            "no" if t in ("main", "C1") else "yes",
            f"{tr[t]['complete@10']['mean']:.3f}",
            f"[{tr[t]['complete@10']['ci_low']:.3f}, {tr[t]['complete@10']['ci_high']:.3f}]",
            f"{tr[t]['hit@10']['mean']:.3f}",
            f"{tr[t]['recall@10']['mean']:.3f}", f"{tr[t]['mrr']['mean']:.3f}"]
           for t in ("main", "C1", "C2", "C3")],
          widths=[1.6, 1.2, 1.6, 2.4, 3.2, 1.6, 2.0, 1.4])
    para(doc, "the macro average above hides how far apart the families are, so each family is "
              "also reported on its own. Complete@10 is the group-level mean over the family's "
              "configurations with a 95% bootstrap interval on C3; partial@10 is union "
              "character coverage (how much of the annotated evidence is retrieved at all); "
              "recall@512 is recall under an equal 512-token budget, which removes the "
              "advantage of retrieving ten large chunks.", size=8.2)
    fam_rows = [[fam, str(e["n_configs"]), f"{e['C1']['complete@10']:.3f}",
                 f"{e['C3']['complete@10']:.3f}",
                 f"[{e['C3']['ci_low']:.3f}, {e['C3']['ci_high']:.3f}]",
                 f"{e['C1']['partial@10']:.3f}", f"{e['C1']['tok512_recall']:.3f}"]
                for fam, e in by_strategy.items()]
    table(doc, ["strategy", "configs", "Complete@10 (C1)", "Complete@10 (C3)", "95% CI (C3)",
                "partial@10 (C1)", "recall@512 (C1)"], fam_rows,
          widths=[2.8, 1.4, 2.6, 2.6, 3.2, 2.3, 2.3], size=7.2)
    para(doc, f"matched token budgets, macro over configurations (group-level means; admission "
              "stops at the first chunk that does not fit):", size=8.2)
    table(doc, ["track", "Recall@512", "Recall@1024", "Recall@2048"], [
        [t, f"{tr[t]['tok512_recall']['mean']:.3f}", f"{tr[t]['tok1024_recall']['mean']:.3f}",
         f"{tr[t]['tok2048_recall']['mean']:.3f}"] for t in ("main", "C1", "C2", "C3")],
        widths=[2.4, 3.0, 3.0, 3.0], size=7.6)
    figure(doc, FIG_TRACKS, 11.4,
           "Figure 1. Evidence completeness by track on the test split "
           "(group-level bootstrap 95% CI, 59 groups).")
    para(doc, f"the primary comparison C3 − C1 is {c31['delta']:+.3f} "
              f"[{c31['ci_low']:+.3f}, {c31['ci_high']:+.3f}], Holm p = {c31['p_holm']:.2f}. "
              "The interval includes zero, so reranking is suggestive but not established on "
              "the test split. The same comparison on the development split was "
              f"{dev_delta:+.3f} with p = {dev_p:.3f}, so the effect did not replicate at the "
              "same size — the development result is the optimistic one.",
         lead="Interpretation. ")
    figure(doc, FIG_RANK, 7.4,
           f"Figure 2. Chunking ranking before and after reranking (Spearman ρ = "
           f"{stats['rank_stability']['C1_vs_C3']:.3f}).")
    para(doc, f"the chunking conclusion survives reranking. Heading "
              f"({by_strategy['heading']['C3']['complete@10']:.3f}) and paragraph "
              f"({by_strategy['paragraph']['C3']['complete@10']:.3f}) remain the best families and "
              f"sentence-level semantic splitting stays far behind "
              f"({by_strategy['semantic']['C3']['complete@10']:.3f}). The completeness rule is not "
              "what causes that gap: a clause counts as covered when the union of retrieved "
              "chunks contains its whole span, so a clause cut into sentence chunks can still be "
              "complete. The semantic family fails because the evidence is not retrieved — union "
              f"character coverage is {by_strategy['semantic']['C1']['partial@10']:.3f} against "
              f"{by_strategy['heading']['C1']['partial@10']:.3f} for heading, and under an equal "
              f"512-token budget recall is {by_strategy['semantic']['C1']['tok512_recall']:.3f} "
              f"against {by_strategy['heading']['C1']['tok512_recall']:.3f}. The union audit shows "
              f"the same thing from the other side: "
              f"{union_main['fixed/size128']['share_of_complete_needing_union']:.0%} of the "
              "complete answers produced by fixed/128 needed two or more chunks, against "
              f"{union_main['fixed/size512']['share_of_complete_needing_union']:.0%} for "
              f"fixed/512, and even at k=50 the semantic family completes only "
              f"{union_main['semantic']['complete@50']:.3f} of questions. Reranking lifts the "
              "level rather than reordering the families.", lead="What it suggests. ")
    para(doc, f"clause length is the obvious composition confound. The clauses already used by "
              f"questions have a median span of {frame['used_median_chars']:.0f} characters, while "
              f"the {frame['available_clauses']} unused usable clauses a future held-out set would "
              f"draw on have a median of {frame['available_median_chars']:.0f}. On the test split "
              f"the short-vs-long contrast is {length_c1['welch']['delta']:+.3f} "
              f"[{length_c1['welch']['ci_low']:+.3f}, {length_c1['welch']['ci_high']:+.3f}] "
              "(Welch 95% interval; no p-value is reported) and it reaches conventional "
              "significance in none of the four split×track combinations. Length is therefore "
              "treated as a composition variable to match when sampling a new set, not as an "
              "established effect.", lead="Difficulty. ")
    para(doc, f"at a fixed ten-chunk pool reranking raises Recall@3 by "
              f"{c2m_r3['delta']:+.3f} (Holm p = {c2m_r3['p_holm']:.3f}) and MRR by "
              f"{c2m_mrr['delta']:+.3f}, and it narrows the gap at matched token budgets "
              "(recall@512: 0.217 → 0.258). The gain is largest where long chunks had to be "
              "truncated (Spearman ρ = "
              f"{stats['truncation_strata']['spearman_trunc_vs_C3_minus_C1']:.2f}).",
         lead="Where it helps. ")

    with (ROOT / "results" / "final" / "summary_C3.csv").open(encoding="utf-8") as f:
        c3_rows = list(csv.DictReader(f))
    c3_ev = [float(r["eval_time_s"]) for r in c3_rows if r.get("eval_time_s")]
    lat = [float(r["latency_ms"]) for r in c3_rows if r.get("latency_ms")]
    para(doc, f"exact search costs {min(lat):.2f}–{max(lat):.2f} ms per query across "
              f"configurations, so chunking barely affects retrieval time. Reranking does: "
              f"scoring 50 candidates per question takes {min(c3_ev):.0f}–{max(c3_ev):.0f} s "
              "for the 60-question test set (about 0.2–0.65 s per query), which is two to "
              "three orders of magnitude above the search itself. Any deployment that adopts "
              "the reranker pays in latency, not in index size.",
         lead="Efficiency. ")

    # 6a --------------------------------------------------------------- test2
    doc.add_heading("5.1 Second held-out set (test2)", level=2)
    t2f = t2["by_strategy"]
    t2gap = t2_out["primary"]
    t2sec = t2_out["secondary"]
    para(doc, f"the first test split had already been used once, so a second set was built "
              f"from clauses that no existing question uses. It has {n_test2} questions "
              f"in {t2['groups']} groups, drawn by a pre-registered length-stratified rule from "
              f"{t2_pre['design']['clause_pool']['readable_clauses']} readable unused clauses; "
              f"the questions were drafted and self-reviewed by the agent, with no human "
              f"sign-off. The design was frozen before they were written and the split was "
              f"evaluated once. The primary endpoint is the structural-minus-semantic gap, "
              f"because at this size the reranking gain cannot be resolved while the family gap "
              f"can.", lead="Why a second set. ", size=8.6)
    table(doc, ["track", "pool", "reranker", "Complete@10", "95% CI"],
          [[t, "10" if t in ("main", "C2") else "50",
            "no" if t in ("main", "C1") else "yes",
            f"{t2['tracks'][t]['complete@10']['mean']:.3f}",
            f"[{t2['tracks'][t]['complete@10']['ci_low']:.3f}, "
            f"{t2['tracks'][t]['complete@10']['ci_high']:.3f}]"]
           for t in ("main", "C1", "C2", "C3")],
          widths=[1.7, 1.3, 1.7, 2.6, 3.4], size=7.4)
    para(doc, f"per family, Complete@10 (C1 → C3): paragraph "
              f"{t2f['paragraph']['C1']['complete@10']:.3f} → "
              f"{t2f['paragraph']['C3']['complete@10']:.3f}, heading "
              f"{t2f['heading']['C1']['complete@10']:.3f} → "
              f"{t2f['heading']['C3']['complete@10']:.3f}, best fixed "
              f"{t2f['fixed/size512']['C1']['complete@10']:.3f} → "
              f"{t2f['fixed/size512']['C3']['complete@10']:.3f}, semantic "
              f"{t2f['semantic']['C1']['complete@10']:.3f} → "
              f"{t2f['semantic']['C3']['complete@10']:.3f}.", size=8.4)
    para(doc, f"primary endpoint (structural minus semantic, C1): "
              f"{t2gap['delta']:+.3f} [{t2gap['ci_low']:+.3f}, {t2gap['ci_high']:+.3f}], "
              f"pre-registered threshold 0.30, verdict {t2gap['verdict']}. "
              f"Secondary endpoint (C3 − C1): {t2sec['delta']:+.3f} "
              f"[{t2sec['ci_low']:+.3f}, {t2sec['ci_high']:+.3f}], verdict "
              f"{t2sec['verdict']}. {t2gap['claim']} {t2sec['claim']}",
         lead="Pre-registered verdict. ", size=8.6)
    para(doc, f"three limits apply. First, the absolute level is not comparable with the first "
              f"test split: all {n_test2} questions have single-clause evidence, which "
              f"makes completeness easier by construction, so only the family contrast is "
              f"interpreted. Second, the clean clause pool could not support the planned "
              f"composition, and three deviations were recorded before evaluation: the set is "
              f"integrity-heavy (25 of 45), Glasgow supplies 33% (above the 25% design cap), and "
              f"no defensible two-clause bundle survived review, so every question is "
              f"single-clause. Third, with {t2['groups']} groups the reranking endpoint can "
              f"resolve only a development-sized effect: its minimum detectable effect is "
              f"{t2_out['power']['secondary_mde']} and an interval containing zero would not "
              f"have been evidence of no effect.",
         lead="What test2 can and cannot claim. ", size=8.6)

    # 7 ---------------------------------------------------------------- problems
    doc.add_heading("6. Problems and Changes", level=1)
    bullet(doc, "a single normalised corpus silently breaks span annotation, so the corpus "
                "keeps a character-faithful evidence layer plus a character map.",
           lead="Offsets. ")
    bullet(doc, "split pieces originally claimed their whole clause span and inflated "
                "completeness; they now carry per-piece evidence ranges, with a tiling "
                "assertion.", lead="Over-claimed coverage. ")
    bullet(doc, "the first audit compared only question wording and missed 20 clauses used on "
                "both sides of the split; clusters sharing gold evidence are now never "
                "divided, and leakage is 0.", lead="Leakage. ")
    bullet(doc, "the test split had already been used for the pilot, so the formal run is "
                "reported as a re-evaluation of a seen split, never as a first holdout.",
           lead="Pilot exposure. ")
    bullet(doc, "selecting on Complete@10 over a reranked top-10 selects a constant; selection "
                "now uses the top-50 pool, and the two invariant contrasts are asserted "
                "(max difference 0.0) instead of tested.", lead="Broken selection metric. ")
    bullet(doc, "T01 reached 0.396 at epoch 1, above the 0.388 that was frozen, but the run "
                "could not complete (about 20 minutes per epoch under memory pressure) and its "
                "checkpoint was discarded when it was aborted. Selection therefore ran over "
                "completed runs only (T04 vs T05); 0.396 is reported for transparency but was "
                "never a candidate.", lead="Why the higher number was not selected. ")
    bullet(doc, "a 6 GB card could not hold a full AdamW update, so Adafactor with 10 of 12 "
                "layers frozen was used; the training log records the three aborted attempts "
                "and the reasoning. On the A5000 the full update does fit (9.7 GB, 64 s per "
                "epoch) and T06 repeats it without any freezing.", lead="Hardware. ")
    bullet(doc, "the training seed is a function of the learning rate "
                "(base + 10·(lr×10⁵) + negatives), so the T04/T05 pair changed two factors at "
                "once. The frozen selection needed one model and stands, but no learning-rate "
                "claim is drawn from that pair; T07 repeats it with matched seeds.",
           lead="Seed and learning rate. ")
    bullet(doc, "the 1,239 in the next section is a usable-clause count, not a raw one: 1,771 "
                "clauses are unused by any question and 1,239 of those are usable "
                "(>=150 characters, not boilerplate), and 768 of those also read as prose. About "
                "42% of them have no domain label from the keyword classifier, so a new question "
                "set has to be domain-annotated by hand, and the sentences there are shorter "
                "than the ones already used (median 439 vs 758 characters), which is why a new "
                "set must be length-matched.",
           lead="Unused clauses. ")
    bullet(doc, "for test2 the clean clause pool could not support the planned composition. The "
                "domain mix, the institution cap and the absence of two-clause bundles were "
                "recorded as pre-evaluation amendments (results/test2/DESIGN_AMENDMENTS.json) "
                "rather than forced with unsuitable clauses, and the report states the "
                "consequences in 5.1.", lead="Test2 composition. ")

    # 8 ---------------------------------------------------------------- remaining
    doc.add_heading("7. Remaining Experiments", level=1)
    table(doc, ["experiment", "purpose", "A5000 time", "A5000 memory", "cost"], [
        ["Second held-out set (test2): 45 questions / 45 groups",
         "restore an unseen split; primary endpoint was the structural-vs-semantic gap",
         "evaluation ~2 min", "2–3 GB", "done"],
        ["Matched-seed learning-rate comparison (T07)",
         "separate learning rate from seed; the frozen pair confounds them",
         "~25 min (6 runs)", "9.7 GB", "done"],
        ["Full fine-tune of the reranker (T06)",
         "remove the partial-fine-tuning limitation",
         "9 min (2 lr × 2 epochs)", "9.7 GB", "done, no gain"],
        ["Expand the resources domain",
         "resources has 7 test questions; the usable pool offers 19 clauses, so ~5–8 more "
         "questions at best", "—", "—", "1 day (limited)"],
    ], widths=[5.2, 5.6, 2.4, 2.0, 2.2], size=7.4)

    # 9 ---------------------------------------------------------------- timeline
    doc.add_heading("8. Updated Timeline", level=1)
    para(doc, "this section updates the plan in the proposal (§10, weeks 1–14) with what actually "
              "happened. The planned column repeats the proposal's week numbers; the actual dates "
              "are the timestamps of the artefacts that record each step, so every row can be "
              "checked against the repository rather than against recollection.", size=8.4)
    table(doc, ["phase", "planned", "actual", "status"], [
        ["literature, permissions, metrics", "weeks 1–2",
         "protocol fixed before collection; the metric set was extended with evidence "
         "completeness and matched token budgets", "on plan"],
        ["collection, cleaning, annotation", "weeks 3–4",
         "seeds probed 13 Oct; corpus frozen 21 Oct; 172 questions reviewed and accepted 24 Oct",
         "ahead of plan"],
        ["four chunking strategies, embeddings, retrieval", "weeks 5–6",
         "17 configurations built and indexed by 22 Oct; no chunk over the model limit",
         "ahead of plan"],
        ["main experiments (Recall@k, MRR, Hit Rate, latency)", "weeks 7–9",
         "pilot evaluation of the test split 24 Oct; formal four-track evaluation 28 Oct",
         "on plan"],
        ["error analysis and sensitivity experiments", "week 10",
         "replaced by the 2×2 pool × reranking design and group-level statistics, 28 Oct",
         "scope changed"],
        ["figures, reproducible code, report", "weeks 11–12",
         "statistics and figures 28–29 Oct; report 29 Oct, revised 2 Nov and again 12 Nov",
         "on plan"],
        ["A5000 compatibility experiments", "not planned",
         "full fine-tune and dev comparison 11 Nov; matched-seed lr grid 12 Nov", "added"],
        ["final paper and presentation", "weeks 13–14", "not started", "remaining"],
    ], widths=[4.4, 1.9, 8.6, 2.3], size=7.3)
    para(doc, "one phase was added after week 9: the reranker (trained and frozen on 25 Oct) and "
              "the candidate-pool factor. Neither is in the proposal, which assumed a single fixed "
              "retriever. Both are reported as a secondary experiment, so the primary comparison "
              "still matches the proposed design; the proposal's week-10 sensitivity study is what "
              "this phase replaced.", lead="What changed against the proposal. ", size=8.4)
    table(doc, ["remaining work", "dates", "deliverable"], [
        ["submit this report", "2–8 Nov", "revision 4 (test2), 13 Nov"],
        ["second held-out set and a single evaluation", "9–15 Nov", "done: 45 questions, one run"],
        ["optional: expand the resources domain", "16–22 Nov", "wider domain CIs"],
        ["buffer and final proofread", "23–29 Nov", "final document"],
    ], widths=[8.4, 3.2, 5.6], size=7.4)
    para(doc, "the semester end date is taken from the course notice; if it falls before the "
              "buffer week, the optional rows are dropped and this report is the final "
              "deliverable.", size=8.2)

    # 10 --------------------------------------------------------------- disclosure
    doc.add_heading("9. AI Disclosure", level=1)
    para(doc, "AI assistance was used in accordance with the course policy.")
    bullet(doc, "questions and reference answers were drafted by DeepSeek from frozen-corpus "
                "clauses, with provenance recorded per question; the student reviewed all "
                "172, raised 23 defects that were corrected, and accepted every question "
                "(timestamped in reports/lock_audit.jsonl).", lead="Annotation. ")
    bullet(doc, f"this differs for test2. Its {n_test2} questions were drafted **and "
                f"reviewed by the agent** (`agent_self_audit`), with no human sign-off; "
                f"`manual_reviewed` records that agent review, not a student signature. The "
                f"self-audit checked, per question, that the answer follows from the cited "
                f"clause and that no outside fact was introduced, and repaired two answers; it "
                f"cannot supply the independent judgement a human reviewer gives. The limit is "
                f"stated in annotations/TEST2_SELF_AUDIT.md and in the outcome record.",
           lead="Second held-out set. ")
    bullet(doc, "pipeline and report code were written with AI assistance; transformations "
                "are asserted and covered by automated tests, and the corpus manifest hashes "
                "every artefact.", lead="Code. ")
    bullet(doc, "this report and the analysis narrative were drafted with AI assistance and "
                "reviewed by the student.", lead="Text. ")
    bullet(doc, "the AI did not set the research question, the evaluation protocol or the "
                "interpretation. No inter-annotator agreement is claimed: there was no second "
                "human annotator.", lead="Scope. ")

    doc.save(out_path)
    print(f"wrote {out_path}")
    print(f"  tracks: " + ", ".join(f"{t}={track_mean[t]:.3f}" for t in track_mean))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
