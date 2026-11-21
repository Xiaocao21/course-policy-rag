"""07_annotation_feasibility.py - can this corpus actually support the question plan?

Run this BEFORE annotating. It answers one question: for each
(domain x evidence_type) cell, how many independent questions can the frozen corpus
plausibly support, and where are the gaps against the planned quotas?

Why it matters: the corpus is skewed (one institution supplies most clauses, and the
course tier is thin). Discovering a shortfall after 150 questions are annotated would
waste the most expensive work in the project, and the held-out test set needs >= 40
independent groups that the corpus may or may not be able to support.

Capacity model (deliberately conservative)
    single-clause       : one usable clause anchors one question
    rule-and-exception  : each clause carrying exception language can anchor one
                          question (the partner rule clause is taken from the same
                          document); capacity = count of such clauses
    cross-clause        : each clause with a cross-reference anchors one question;
                          if a cell has few cross-references, sections containing >= 2
                          usable same-domain clauses count as half capacity each,
                          because a multi-clause question can be built from them
                          without an explicit cross-reference

Outputs
    reports/annotation_feasibility.json
    reports/annotation_feasibility.png   heatmap: capacity / quota ratio per cell
    stdout                                gap list + a clear go / collect-more verdict

Usage:
    .venv\\Scripts\\python.exe scripts/07_annotation_feasibility.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_ROOT, load_settings, load_taxonomy, project_paths, read_jsonl, write_json,
)

# Planned marginals. Sum each to the 150-question target.
DOMAIN_QUOTA = {
    "grading": 35, "integrity": 30, "late": 25,
    "exam": 25, "attendance": 20, "resources": 15,
}
TYPE_QUOTA = {
    "single-clause": 50, "rule-and-exception": 55, "cross-clause": 45,
}


def classify_domain(text: str, taxonomy: dict, context: str = "") -> tuple[str, int]:
    """Local mirror of goldset.classify_domain (kept import-light for this script)."""
    low = text.lower()
    ctx = (context or "").lower()
    best, best_n = "", 0
    for name, spec in taxonomy.get("domains", {}).items():
        n = sum(low.count(kw.lower()) for kw in spec.get("keywords", []))
        if ctx:
            n += sum(1 for kw in spec.get("keywords", []) if kw.lower() in ctx)
        if n > best_n:
            best, best_n = name, n
    return best, best_n


def joint_quota(domain_quota: dict[str, int], type_quota: dict[str, int],
                total: int) -> dict[tuple[str, str], int]:
    """Allocate the joint quota so both marginals hold.

    A cell's target is proportional to (domain share x type share), then cells are
    rounded with largest-remainder so the totals match the plan exactly.
    """
    raw = {}
    for d, dq in domain_quota.items():
        for t, tq in type_quota.items():
            raw[(d, t)] = total * (dq / total) * (tq / total)

    out = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = total - sum(out.values())
    order = sorted(raw.items(), key=lambda kv: (-(kv[1] - math.floor(kv[1])), kv[0]))
    for i in range(remainder):
        out[order[i % len(order)][0]] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--total", type=int, default=150)
    ap.add_argument("--min-test-groups", type=int, default=40)
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)
    taxonomy = load_taxonomy(root)

    sources = list(csv.DictReader((root / "sources.csv").open(encoding="utf-8"))) \
        if (root / "sources.csv").exists() else []
    included = {s["doc_id"]: s for s in sources
                if str(s.get("include", "")).upper() == "TRUE"}
    if not included:
        print("no include=TRUE documents in sources.csv; run curate_corpus.py first",
              file=sys.stderr)
        return 2

    # ---------------------------------------------------------------- clause pool
    clauses: list[dict] = []
    for doc_id, meta in included.items():
        ev = paths["cleaned"] / f"{doc_id}.evidence.jsonl"
        if not ev.exists():
            continue
        for c in read_jsonl(ev):
            text = c.get("text", "")
            usable = c.get("char_len", 0) >= 80 and not c.get("boilerplate")
            domain, hits = classify_domain(text, taxonomy, context=c.get("section_path", ""))
            clauses.append({
                "doc_id": doc_id,
                "clause_id": c["clause_id"],
                "tier": meta.get("tier", ""),
                "institution": meta.get("institution", ""),
                "section_path": c.get("section_path", ""),
                "usable": usable,
                "domain": domain,
                "domain_hits": hits,
                "has_exception": bool(c.get("has_exception")),
                "has_crossref": bool(c.get("has_crossref")),
                "has_alt": bool(c.get("has_alt_evidence")),
                "char_len": c.get("char_len", 0),
            })

    usable = [c for c in clauses if c["usable"]]
    labelless = [c for c in usable if not c["domain"]]
    print(f"clauses total {len(clauses)}  usable {len(usable)}  "
          f"no domain label {len(labelless)}")

    # ---------------------------------------------------------------- capacity cells
    # (domain, type) capacity
    by_cell_anchors: Counter = Counter()
    for c in usable:
        if not c["domain"]:
            continue
        by_cell_anchors[(c["domain"], "single-clause")] += 1
        if c["has_exception"]:
            by_cell_anchors[(c["domain"], "rule-and-exception")] += 1
        if c["has_crossref"]:
            by_cell_anchors[(c["domain"], "cross-clause")] += 1

    # cross-clause fallback: sections holding >= 2 usable same-domain clauses
    section_groups: Counter = Counter()
    for c in usable:
        if c["domain"] and c["section_path"]:
            section_groups[(c["doc_id"], c["section_path"], c["domain"])] += 1
    for (doc_id, section, domain), n in section_groups.items():
        if n >= 2:
            by_cell_anchors[(domain, "cross-clause")] += n // 2

    quota = joint_quota(DOMAIN_QUOTA, TYPE_QUOTA, args.total)

    domains = list(taxonomy.get("domains", {}).keys())
    types = ["single-clause", "rule-and-exception", "cross-clause"]
    cells = []
    for d in domains:
        for t in types:
            cap = by_cell_anchors.get((d, t), 0)
            need = quota.get((d, t), 0)
            cells.append({
                "domain": d, "evidence_type": t,
                "capacity": cap, "quota": need,
                "ratio": round(cap / need, 2) if need else None,
                "ok": cap >= need,
            })

    # ---------------------------------------------------------------- tier x domain
    tier_domain: dict[tuple[str, str], int] = Counter()
    for c in usable:
        if c["domain"]:
            tier_domain[(c["tier"], c["domain"])] += 1
    tier_table = []
    for tier in ("course", "faculty", "institution"):
        row = {"tier": tier}
        for d in domains:
            row[d] = tier_domain.get((tier, d), 0)
        tier_table.append(row)

    # ---------------------------------------------------------------- institution
    inst_clauses = Counter(c["institution"] for c in usable)
    inst_domains: dict[str, set[str]] = defaultdict(set)
    for c in usable:
        if c["domain"]:
            inst_domains[c["institution"]].add(c["domain"])
    total_usable = len(usable) or 1
    institutions = [{
        "institution": i,
        "usable_clauses": n,
        "share": round(n / total_usable, 3),
        "domains": sorted(inst_domains[i]),
        "max_questions_estimate": n,      # one clause anchors at most one question
    } for i, n in inst_clauses.most_common()]

    # ---------------------------------------------------------------- doc size skew
    doc_chars: Counter = Counter()
    for c in usable:
        doc_chars[c["doc_id"]] += c["char_len"]
    total_chars = sum(doc_chars.values()) or 1
    doc_rows = sorted(
        ({"doc_id": d, "chars": n, "share": round(n / total_chars, 3)} for d, n in doc_chars.items()),
        key=lambda r: -r["chars"])
    top3_share = round(sum(r["share"] for r in doc_rows[:3]), 3)

    # ---------------------------------------------------------------- test-group check
    # 50 test questions in >= 40 independent groups: a group is a template, so at
    # least 40 distinct answerable evidence situations must exist.
    test_quota = round(args.total * cfg["annotation"]["target_test_questions"] /
                       cfg["annotation"]["target_questions_total"])
    capacity_total = sum(c["capacity"] for c in cells)
    feasible_groups = capacity_total >= args.min_test_groups

    # ---------------------------------------------------------------- gaps
    gaps = [c for c in cells if not c["ok"]]
    gaps.sort(key=lambda c: (c["ratio"] if c["ratio"] is not None else 0))

    verdict = "GO" if (not gaps and feasible_groups) else "COLLECT_MORE"
    report = {
        "status": verdict,
        "corpus": {
            "documents": len(included),
            "clauses_total": len(clauses),
            "clauses_usable": len(usable),
            "clauses_without_domain": len(labelless),
            "institutions": len(inst_clauses),
            "coverage_note": ("institution coverage is as frozen in MANIFEST.json "
                              "(12 institutions collected, corpus subset per curation)"),
        },
        "quota_model": {
            "total": args.total,
            "domain_quota": DOMAIN_QUOTA,
            "type_quota": TYPE_QUOTA,
            "joint": {f"{d}|{t}": v for (d, t), v in quota.items()},
        },
        "cells": cells,
        "tier_by_domain": tier_table,
        "institutions": institutions,
        "document_skew": {"top3_share": top3_share, "documents": doc_rows[:10]},
        "test_group_check": {
            "planned_test_questions": test_quota,
            "min_groups": args.min_test_groups,
            "estimated_capacity": capacity_total,
            "feasible": feasible_groups,
        },
        "gaps": gaps,
        "recommendation": [],
    }

    # action-oriented recommendation
    if gaps:
        by_domain = Counter(c["domain"] for c in gaps)
        for d, n in by_domain.most_common():
            worst = min((c for c in gaps if c["domain"] == d),
                        key=lambda c: c["ratio"] if c["ratio"] is not None else 0)
            report["recommendation"].append(
                f"collect more '{worst['domain']}' sources: cell "
                f"{worst['domain']}x{worst['evidence_type']} has capacity "
                f"{worst['capacity']} vs quota {worst['quota']}")
    if not feasible_groups:
        report["recommendation"].append(
            f"corpus capacity ({capacity_total}) is below the {args.min_test_groups} "
            f"independent test groups required by the protocol")
    if top3_share > 0.30:
        report["recommendation"].append(
            f"top-3 documents hold {top3_share:.0%} of corpus characters; report "
            f"per-document macro averages alongside pooled scores")
    course_late = next((c for c in cells
                        if c["domain"] == "late" and c["evidence_type"] == "single-clause"), None)
    if course_late:
        course_late_n = next((r["late"] for r in tier_table if r["tier"] == "course"), 0)
        if course_late_n < 10:
            report["recommendation"].append(
                f"course tier has only {course_late_n} usable 'late' clauses; late-work "
                f"questions are the one domain institutional rules cannot supply, so "
                f"collect more CC-licensed syllabi")

    write_json(paths["reports"] / "annotation_feasibility.json", report)

    # ---------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        grid = np.zeros((len(domains), len(types)))
        for i, d in enumerate(domains):
            for j, t in enumerate(types):
                cell = next(c for c in cells if c["domain"] == d and c["evidence_type"] == t)
                need = max(1, cell["quota"])
                grid[i, j] = cell["capacity"] / need

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=2.5, aspect="auto")
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(types, rotation=18, ha="right")
        ax.set_yticks(range(len(domains)))
        ax.set_yticklabels(domains)
        for i in range(len(domains)):
            for j in range(len(types)):
                cell = next(c for c in cells if c["domain"] == domains[i]
                            and c["evidence_type"] == types[j])
                ax.text(j, i, f"{cell['capacity']}/{cell['quota']}",
                        ha="center", va="center", fontsize=9,
                        color="black" if grid[i, j] >= 0.9 else "darkred")
        ax.set_title("Annotation feasibility: capacity / quota per cell\n"
                     f"corpus {len(included)} docs, {len(usable)} usable clauses, "
                     f"verdict {verdict}")
        fig.colorbar(im, ax=ax, label="capacity / quota (green >= 1.0)")
        fig.tight_layout()
        fig.savefig(paths["reports"] / "annotation_feasibility.png", dpi=140)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not render feasibility heatmap ({type(e).__name__}: {e})",
              file=sys.stderr)

    # ---------------------------------------------------------------- stdout
    print("\n=== corpus ===")
    print(f"  documents {len(included)}   usable clauses {len(usable)}   "
          f"institutions {len(inst_clauses)}")
    print("\n=== capacity / quota per cell ===")
    print(f"  {'domain':<12} {'single':>12} {'rule+exc':>12} {'cross':>12}")
    for d in domains:
        row = []
        for t in types:
            cell = next(c for c in cells if c["domain"] == d and c["evidence_type"] == t)
            mark = " " if cell["ok"] else "!"
            row.append(f"{cell['capacity']}/{cell['quota']}{mark}")
        print(f"  {d:<12} {row[0]:>12} {row[1]:>12} {row[2]:>12}")
    print("   (! = capacity below quota)")

    print("\n=== tier x domain (usable clauses) ===")
    header = f"  {'tier':<12}" + "".join(f"{d[:8]:>9}" for d in domains)
    print(header)
    for r in tier_table:
        print(f"  {r['tier']:<12}" + "".join(f"{r[d]:>9}" for d in domains))

    print("\n=== institutions ===")
    for i in institutions[:8]:
        print(f"  {i['institution'][:40]:<42} {i['usable_clauses']:>5} clauses "
              f"({i['share']:.0%})  domains={len(i['domains'])}")

    print(f"\n=== document skew === top-3 share {top3_share:.1%}")
    for r in doc_rows[:3]:
        print(f"  {r['share']:>6.1%}  {r['doc_id'][:60]}")

    print(f"\n=== test group check ===")
    print(f"  planned test questions {test_quota} across >= {args.min_test_groups} groups")
    print(f"  estimated total capacity {capacity_total}  -> "
          f"{'feasible' if feasible_groups else 'NOT feasible'}")

    if verdict == "GO":
        print("\nVERDICT: GO - the corpus can support the planned question set.")
    else:
        print(f"\nVERDICT: COLLECT_MORE - {len(gaps)} cell(s) below quota")
        for rec in report["recommendation"]:
            print(f"  - {rec}")
    print(f"\nwrote {paths['reports'] / 'annotation_feasibility.json'}")
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
