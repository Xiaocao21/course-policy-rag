"""curate_corpus.py - select the final documented corpus from everything collected.

The crawl is deliberately generous (it is cheap to collect and expensive to go back),
but the study needs a corpus that is **documented and balanced**, not just large. The
plan's constraints are:

    * 20-40 documents in total
    * three tiers, no tier above 40%
    * >= 6 institutions
    * exception clauses in >= 10 documents, cross-references in >= 8,
      alternative-evidence structure in >= 6
    * each document rich enough to annotate clauses in

This stage enforces them explicitly and records every decision in
`reports/curation_decisions.csv`, so the corpus is reproducible from the same inputs
rather than the result of manual cherry-picking.

Selection order:
  1. drop documents below the clause floor
  2. drop documents with no exception, no cross-reference and no alternative evidence
     (they cannot support rule-and-exception or cross-clause questions)
  3. per tier, rank by richness (clauses + weighted evidence signals) and take up to
     the tier cap, preferring new institutions before adding a second document from
     one already represented
  4. verify the acceptance gates and report any that are still unmet

Usage:
    .venv\\Scripts\\python.exe scripts\\curate_corpus.py
    .venv\\Scripts\\python.exe scripts\\curate_corpus.py --total 30 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_ROOT, load_settings, project_paths, read_jsonl, write_json,
)

# Target composition, mirroring the plan's tier table
TIER_TARGET = {"course": 0.40, "faculty": 0.25, "institution": 0.35}

# A crawled page inherits its seed's tier, which mislabels institution-wide
# regulations reached from a department landing page (e.g. a university's General
# Regulations crawled under a schools/faculty seed). These path markers indicate the
# document governs the whole institution, so it is promoted to the institution tier.
INSTITUTIONAL_MARKERS = (
    "general-regulations", "general-student", "senate", "university-regulations",
    "academic-regulations-current", "regulations-current",
    "student-contract", "code-of-conduct", "honor-code", "academic-integrity",
    "academic-conduct", "student-responsibilities", "grade-grievance",
    "grading-practices", "plagiarism", "academic-appeals",
)


def effective_tier(doc_id: str, meta_tier: str) -> tuple[str, str]:
    """Return (tier, reason). Promotes institution-wide documents out of the faculty tier."""
    low = doc_id.lower().replace("_", "-")
    if meta_tier != "faculty":
        return meta_tier, ""
    for marker in INSTITUTIONAL_MARKERS:
        if marker in low:
            return "institution", f"promoted from faculty: path marker '{marker}'"
    return meta_tier, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--total", type=int, default=32, help="target corpus size")
    ap.add_argument("--min-clauses", type=int, default=6)
    ap.add_argument("--min-institutions", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)
    acc = cfg["acceptance"]

    # ---------------------------------------------------------------- gather
    sources: list[dict] = []
    sources_csv = root / "sources.csv"
    if sources_csv.exists():
        with sources_csv.open(encoding="utf-8", newline="") as f:
            sources = list(csv.DictReader(f))
    by_doc = {s["doc_id"]: s for s in sources if s.get("doc_id")}

    stats: list[dict] = []
    for p in sorted(paths["cleaned"].glob("*.evidence.jsonl")):
        doc_id = p.name.replace(".evidence.jsonl", "")
        clauses = read_jsonl(p)
        if not clauses:
            continue
        meta = by_doc.get(doc_id, {})
        exc = sum(1 for c in clauses if c.get("has_exception"))
        xr = sum(1 for c in clauses if c.get("has_crossref"))
        alt = sum(1 for c in clauses if c.get("has_alt_evidence"))
        usable = sum(1 for c in clauses if not c.get("boilerplate") and c.get("char_len", 0) >= 80)
        raw_tier = meta.get("tier") or clauses[0].get("tier", "")
        tier, tier_note = effective_tier(doc_id, raw_tier)
        stats.append({
            "doc_id": doc_id,
            "tier": tier,
            "tier_note": tier_note,
            "seed_tier": raw_tier,
            "institution": meta.get("institution") or clauses[0].get("institution", ""),
            "clauses": len(clauses),
            "usable_clauses": usable,
            "exception_clauses": exc,
            "crossref_clauses": xr,
            "alt_evidence_clauses": alt,
            "richness": usable + 3 * exc + 3 * xr + 2 * alt,
        })

    # ---------------------------------------------------------------- gates 1-2
    decisions: list[dict] = []
    eligible: list[dict] = []

    # The same policy is often reachable at more than one URL (a hub page and the
    # canonical path), which yields byte-identical documents. Deduplicate on the raw
    # content hash before selection, otherwise one policy occupies two corpus slots
    # and is over-weighted in every retrieval metric.
    seen_hash: dict[str, str] = {}
    for s in stats:
        h = (by_doc.get(s["doc_id"], {}) or {}).get("sha256_raw", "")
        if h and h in seen_hash:
            decisions.append({**s, "in_corpus": False,
                              "reason": f"duplicate content of {seen_hash[h]} (same sha256_raw)"})
            continue
        if h:
            seen_hash[h] = s["doc_id"]

        if s["usable_clauses"] < args.min_clauses:
            decisions.append({**s, "in_corpus": False,
                              "reason": f"usable_clauses {s['usable_clauses']} < {args.min_clauses}"})
            continue
        if s["exception_clauses"] == 0 and s["crossref_clauses"] == 0 and s["alt_evidence_clauses"] == 0:
            decisions.append({**s, "in_corpus": False,
                              "reason": "no exception, crossref or alternative-evidence clause"})
            continue
        eligible.append(s)

    # ---------------------------------------------------------------- gate 3: tiers
    # Selection alternates by "largest shortfall against the target share", which
    # keeps the tier composition near the plan's table while filling the corpus.
    # A previous version took one breadth pass then a richness pass and stalled at
    # 19 documents, because a naive share guard rejected every remaining candidate.
    caps = {t: max(1, round(args.total * share)) for t, share in TIER_TARGET.items()}
    chosen: list[dict] = []
    chosen_ids: set[str] = set()
    tier_count: Counter = Counter()
    inst_count: Counter = Counter()

    def take(s: dict) -> None:
        chosen.append(s)
        chosen_ids.add(s["doc_id"])
        tier_count[s["tier"]] += 1
        inst_count[s["institution"]] += 1

    pools: dict[str, list[dict]] = {
        tier: sorted([s for s in eligible if s["tier"] == tier],
                     key=lambda s: (-s["richness"], s["doc_id"]))
        for tier in caps
    }

    # Fill the tier whose realised share falls furthest short of its target share,
    # preferring an institution not yet represented in that tier.
    #
    # `caps` is derived from the target shares and can round ABOVE the hard share
    # limit (round(32 * 0.40) = 13, which is 40.6% of 32). The per-tier ceiling is
    # therefore floor(total * tier_share_max), which by construction cannot breach the
    # limit. An earlier attempt also vetoed any pick that would exceed the limit on the
    # running total; that deadlocked at three documents, because a one-document
    # corpus is always 100% of whatever tier was just added.
    max_per_tier = int(args.total * acc["tier_share_max"])
    ceiling = {t: min(caps[t], max_per_tier) for t in caps}

    while len(chosen) < args.total:
        total = len(chosen)
        deficits = {}
        for tier in pools:
            if tier_count[tier] >= ceiling[tier]:
                continue
            if not any(s["doc_id"] not in chosen_ids for s in pools[tier]):
                continue
            realised = tier_count[tier] / total if total else 0.0
            deficits[tier] = TIER_TARGET[tier] - realised
        if not deficits:
            break
        tier = max(deficits, key=lambda t: deficits[t])
        pool = [s for s in pools[tier] if s["doc_id"] not in chosen_ids]
        fresh = [s for s in pool if inst_count[s["institution"]] == 0]
        take((fresh or pool)[0])

    # Any leftover capacity goes to the richest remaining documents of the tiers
    # that still have room, but never past the maximum share for a single tier.
    for s in sorted(eligible, key=lambda s: (-s["richness"], s["doc_id"])):
        if len(chosen) >= args.total:
            break
        if s["doc_id"] in chosen_ids:
            continue
        if tier_count[s["tier"]] >= ceiling[s["tier"]]:
            continue
        take(s)

    for s in eligible:
        if s["doc_id"] not in chosen_ids:
            decisions.append({**s, "in_corpus": False,
                              "reason": f"not selected: tier '{s['tier']}' at cap "
                                        f"{caps.get(s['tier'])} or below selection rank"})
    for s in chosen:
        decisions.append({**s, "in_corpus": True, "reason": f"selected (richness {s['richness']})"})

    # ---------------------------------------------------------------- report
    fields = ["doc_id", "tier", "seed_tier", "tier_note", "institution", "clauses",
              "usable_clauses", "exception_clauses", "crossref_clauses",
              "alt_evidence_clauses", "richness", "in_corpus", "reason"]
    out = paths["reports"] / "curation_decisions.csv"
    if not args.dry_run:
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for d in sorted(decisions, key=lambda d: (d["tier"], not d["in_corpus"], d["doc_id"])):
                w.writerow({k: d.get(k, "") for k in fields})

        # Reflect the decision in sources.csv so include=TRUE is the corpus definition
        # rather than a hand-maintained flag that can drift from the reports. The
        # effective tier is written back too, so every downstream report agrees with
        # the curation instead of repeating the seed's tier.
        if sources_csv.exists():
            reason_by_doc = {d["doc_id"]: d for d in decisions}
            with sources_csv.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                src_fields = list(reader.fieldnames or [])
                rows = list(reader)

            # The collector's clause columns are raw page-level regex hits, NOT clause
            # counts (they count numbered-looking lines across the whole page, nav
            # included). Rename them so they can never be mistaken for clause counts
            # again, and add the authoritative clause-level columns below.
            renames = {
                "clause_count": "clause_count_raw_hits",
                "exception_count": "exception_count_raw_hits",
                "crossref_count": "crossref_count_raw_hits",
                "alt_evidence_count": "alt_evidence_count_raw_hits",
            }
            for old, new in renames.items():
                if old in src_fields:
                    src_fields[src_fields.index(old)] = new
                    for r in rows:
                        if old in r:
                            r[new] = r.pop(old)
            for col in ("clause_count", "clauses_usable", "clauses_exception",
                        "clauses_crossref", "clauses_alt"):
                if col not in src_fields:
                    src_fields.append(col)

            for r in rows:
                d = reason_by_doc.get(r.get("doc_id", ""))
                if d is None:
                    r["include"] = "FALSE"
                    r["exclude_reason"] = "no usable clauses extracted"
                    continue
                # authoritative counts, measured on the cleaned evidence layer
                r["clause_count"] = d["clauses"]
                r["clauses_usable"] = d["usable_clauses"]
                r["clauses_exception"] = d["exception_clauses"]
                r["clauses_crossref"] = d["crossref_clauses"]
                r["clauses_alt"] = d["alt_evidence_clauses"]
                r["tier"] = d["tier"]
                if d["in_corpus"]:
                    r["include"] = "TRUE"
                    r["exclude_reason"] = d["tier_note"] or ""
                else:
                    r["include"] = "FALSE"
                    r["exclude_reason"] = d["reason"]
            tmp = sources_csv.with_suffix(".csv.tmp")
            with tmp.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=src_fields)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            try:
                tmp.replace(sources_csv)
            except PermissionError:
                # Windows refuses to replace a file another process holds open (Excel
                # is the usual culprit). Keep the new table under a sidecar name so the
                # run still produces its output, and say plainly what to do.
                side = sources_csv.with_name(sources_csv.stem + ".updated.csv")
                try:
                    if side.exists():
                        side.unlink()
                    tmp.replace(side)
                except OSError:
                    pass
                print(f"\nWARNING: could not replace {sources_csv.name} - it is open in "
                      f"another program (usually Excel).", file=sys.stderr)
                print(f"         The updated table was written to {side.name} instead.",
                      file=sys.stderr)
                print(f"         Close the file in Excel, then re-run this script.",
                      file=sys.stderr)

    tier_chosen = Counter(s["tier"] for s in chosen)
    institutions = sorted({s["institution"] for s in chosen})
    total = len(chosen)
    shares = {t: round(n / total, 3) for t, n in tier_chosen.items()} if total else {}

    gates = {
        "documents": (total, f"20-40 target"),
        "institutions": (len(institutions), f">= {args.min_institutions}"),
        "docs_with_exceptions": (sum(1 for s in chosen if s["exception_clauses"]), f">= {acc['min_docs_with_exceptions']}"),
        "docs_with_crossrefs": (sum(1 for s in chosen if s["crossref_clauses"]), f">= {acc['min_docs_with_crossrefs']}"),
        "docs_with_alt_evidence": (sum(1 for s in chosen if s["alt_evidence_clauses"]), f">= {acc['min_docs_with_alt_evidence']}"),
    }

    print("=== corpus curation ===")
    print(f"  documents with clauses : {len(stats)}")
    print(f"  eligible after gates   : {len(eligible)}")
    print(f"  selected               : {total}  (target ~{args.total}, plan asks 20-40)")
    print(f"  tier composition       : {dict(tier_chosen)}  shares {shares}")
    print(f"  tier caps              : {caps}")
    print(f"  institutions           : {len(institutions)}")
    for inst in institutions:
        n = sum(1 for s in chosen if s["institution"] == inst)
        print(f"      {inst[:48]:<50} {n}")
    print("\n  acceptance gates:")
    for name, (have, need) in gates.items():
        flag = "OK " if _meets(name, have, need) else "!! "
        print(f"    {flag}{name:24s} {have:>3}   ({need})")
    print(f"\n  {'wrote ' + str(out) if not args.dry_run else 'dry run: nothing written'}")
    if total and max(shares.values()) > acc["tier_share_max"]:
        print(f"\n  WARNING: tier '{max(shares, key=shares.get)}' is {max(shares.values()):.0%} "
              f"> {acc['tier_share_max']:.0%} max")
    return 0


def _meets(name: str, have: int, need: str) -> bool:
    try:
        threshold = int("".join(ch for ch in need if ch.isdigit()) or 0)
    except ValueError:
        return True
    if name == "documents":
        return 20 <= have <= 40
    return have >= threshold


if __name__ == "__main__":
    sys.exit(main())
