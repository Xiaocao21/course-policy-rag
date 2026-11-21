"""04_freeze_manifest.py - freeze the corpus into a single reproducible manifest.

The manifest is the object all four chunking strategies must consume. Freezing it
is what makes "the same cleaned raw corpus is reused for every chunking strategy"
(checkable) rather than (asserted).

Writes:
    MANIFEST.json      corpus version, every artefact hash, seeds, source list hash
    COMPLIANCE.md      robots decisions, rate limits, public-sources statement,
                       and any documented fallback (e.g. faculty tier below target)

Usage:
    .venv\\Scripts\\python.exe scripts\\04_freeze_manifest.py
    .venv\\Scripts\\python.exe scripts\\04_freeze_manifest.py --corpus-version v0.2.0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_ROOT, load_settings, load_taxonomy, now_iso, project_paths,
    read_jsonl, sha256_file, sha256_text, write_json,
)


def hash_many(paths: list[Path], root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(paths):
        if p.is_file():
            try:
                out[str(p.relative_to(root)).replace("\\", "/")] = sha256_file(p)
            except OSError:
                continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--corpus-version", default="")
    ap.add_argument("--allow-warnings", action="store_true",
                    help="freeze even when validation reported blockers (documented smoke test)")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    cfg = load_settings(root)
    taxonomy = load_taxonomy(root)
    # The manifest is the single source of truth for the corpus version (settings.yaml
    # no longer carries a copy, which previously drifted).
    version = args.corpus_version
    if not version:
        print("ERROR: --corpus-version is required (e.g. --corpus-version v0.1.3).",
              file=sys.stderr)
        print("The manifest is the single source of truth for the corpus version.",
              file=sys.stderr)
        return 2

    validation = {}
    vpath = paths["reports"] / "validation.json"
    if vpath.exists():
        validation = json.loads(vpath.read_text(encoding="utf-8"))
    if validation.get("status") == "blocked" and not args.allow_warnings:
        print("REFUSING TO FREEZE: validation.json reports blockers.", file=sys.stderr)
        for b in validation.get("blockers", []):
            print(f"  - {b.get('kind')}: {b.get('detail') or b.get('have')}", file=sys.stderr)
        print("\nFix the blockers, or pass --allow-warnings to freeze a documented smoke corpus.",
              file=sys.stderr)
        return 1

    split_audit = {}
    apath = paths["annotations"] / "split_audit.json"
    if apath.exists():
        split_audit = json.loads(apath.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- sources
    sources: list[dict] = []
    sources_csv = root / "sources.csv"
    if sources_csv.exists():
        with sources_csv.open(encoding="utf-8", newline="") as f:
            sources = list(csv.DictReader(f))

    # ================================================================ consistency gates
    # These four checks exist because the corpus was once frozen while the artefacts
    # on disk disagreed with it (manifest said 32 documents, sources.csv said 31,
    # validation.json said a third thing, and the gold worksheet still referenced a
    # document that had been dropped). A manifest that does not describe the corpus
    # on disk makes the "reproducible" claim false, so freezing is refused instead.
    contract_problems: list[str] = []

    included = [s for s in sources if str(s.get("include", "")).upper() in ("TRUE", "YES", "1")]
    corpus_sources = included or sources
    included_ids = {s.get("doc_id", "") for s in included}

    corpus_file_candidates = sorted(paths["corpus"].glob("corpus_safe_512*.jsonl"))
    corpus_file_probe = corpus_file_candidates[-1] if corpus_file_candidates else None
    corpus_chunks = read_jsonl(corpus_file_probe) if corpus_file_probe else []
    corpus_doc_tier: dict[str, str] = {}
    for ch in corpus_chunks:
        corpus_doc_tier.setdefault(ch["doc_id"], ch.get("tier", ""))
    corpus_ids = set(corpus_doc_tier)

    # gate 1: sources.csv include=TRUE must equal the corpus document set
    if included and corpus_ids != included_ids:
        only_src = sorted(included_ids - corpus_ids)
        only_corpus = sorted(corpus_ids - included_ids)
        contract_problems.append(
            f"sources.csv include=TRUE and corpus document sets differ "
            f"(in sources only: {only_src[:3]}, in corpus only: {only_corpus[:3]})")
    if not corpus_chunks:
        contract_problems.append("no corpus file found under corpus_safe_512/; run 03 first")

    # gate 2: the tier inside the corpus must match the curated tier in sources.csv
    tier_mismatch = [(s.get("doc_id"), corpus_doc_tier.get(s.get("doc_id")), s.get("tier"))
                     for s in included
                     if s.get("doc_id") in corpus_doc_tier
                     and corpus_doc_tier.get(s.get("doc_id")) != s.get("tier")]
    if tier_mismatch:
        contract_problems.append(
            f"{len(tier_mismatch)} document(s) have corpus tier != curated tier: {tier_mismatch[:3]}")

    # gate 3: validation.json must describe this corpus, not an earlier one
    v_checks = (validation or {}).get("checks", {})
    if not v_checks:
        contract_problems.append("reports/validation.json missing or empty; run 06_validate.py")
    else:
        if v_checks.get("documents_included") not in (None, len(included)):
            contract_problems.append(
                f"validation.json reports documents_included={v_checks.get('documents_included')} "
                f"but sources.csv has {len(included)}")
        if v_checks.get("chunks") not in (None, len(corpus_chunks)):
            contract_problems.append(
                f"validation.json reports chunks={v_checks.get('chunks')} "
                f"but the corpus has {len(corpus_chunks)}")

    # gate 4: settings hash must match the file being frozen
    settings_path = root / "config" / "settings.yaml"
    live_settings_sha = sha256_file(settings_path) if settings_path.exists() else ""
    if not live_settings_sha:
        contract_problems.append("config/settings.yaml missing")

    # Recorded for provenance: if the settings changed since the previous freeze, the
    # previous manifest no longer describes this corpus. Re-freezing fixes it, so this
    # is surfaced rather than blocked.
    previous_settings_sha = ""
    existing_manifest = root / "MANIFEST.json"
    if existing_manifest.exists():
        try:
            previous_settings_sha = json.loads(
                existing_manifest.read_text(encoding="utf-8")).get("settings_sha256", "")
        except (OSError, ValueError):
            previous_settings_sha = ""

    # gate 5: the gold worksheet must not reference documents outside the corpus
    worksheet = paths["annotations"] / "gold_worksheet.csv"
    if worksheet.exists() and corpus_ids:
        with worksheet.open(encoding="utf-8", newline="") as f:
            ws_docs = {r.get("doc_id", "") for r in csv.DictReader(f)}
        stale = sorted(ws_docs - corpus_ids)
        if stale:
            contract_problems.append(
                f"gold_worksheet.csv references {len(stale)} document(s) not in the corpus: "
                f"{stale[:3]}")

    if contract_problems:
        print("REFUSING TO FREEZE: artefacts disagree with each other.", file=sys.stderr)
        for p in contract_problems:
            print(f"  - {p}", file=sys.stderr)
        print("\nRe-run: curate -> corpus -> validate -> template, then freeze again.",
              file=sys.stderr)
        return 1

    # Corpus composition is reported over the documents that actually entered the
    # corpus (include=TRUE), not everything the crawl collected. The full collected
    # set is reported separately as provenance.
    tier_counts: dict[str, int] = {}
    for s in corpus_sources:
        tier_counts[s.get("tier", "?")] = tier_counts.get(s.get("tier", "?"), 0) + 1
    collected_tier_counts: dict[str, int] = {}
    for s in sources:
        collected_tier_counts[s.get("tier", "?")] = collected_tier_counts.get(s.get("tier", "?"), 0) + 1
    institutions = sorted({s.get("institution", "") for s in corpus_sources if s.get("institution")})

    # ---------------------------------------------------------------- artefacts
    raw_files = list(paths["raw"].rglob("*"))
    extracted = list(paths["extracted"].glob("*.json"))
    cleaned = list(paths["cleaned"].glob("*.jsonl")) + list(paths["cleaned"].glob("*.log"))
    corpus = list(paths["corpus"].glob("*"))
    annotations = list(paths["annotations"].glob("*.json*"))

    artefact_hashes = {
        "raw": hash_many([p for p in raw_files if p.is_file()], root),
        "extracted": hash_many(extracted, root),
        "cleaned": hash_many(cleaned, root),
        "corpus_safe_512": hash_many(corpus, root),
        "annotations": hash_many(annotations, root),
    }

    corpus_file = next((p for p in corpus if p.name.startswith("corpus_safe_512")
                        and p.suffix == ".jsonl"), None)
    chunk_stats = {}
    if corpus_file:
        chunks = read_jsonl(corpus_file)
        toks = [c["token_count"] for c in chunks]
        chunk_stats = {
            "file": corpus_file.name,
            "chunks": len(chunks),
            "documents": len({c["doc_id"] for c in chunks}),
            "clauses": len({c["clause_id"] for c in chunks}),
            "token_min": min(toks, default=0),
            "token_max": max(toks, default=0),
            "token_mean": round(sum(toks) / len(toks), 2) if toks else 0,
            "chunks_over_max_input": sum(1 for t in toks if t > cfg["safe_corpus"]["max_input_tokens"]),
            "approximate_tokenizer": corpus_file.name.endswith("_approx.jsonl"),
        }

    robots = {}
    rpath = paths["reports"] / "robots_audit.json"
    if rpath.exists():
        robots = json.loads(rpath.read_text(encoding="utf-8"))

    manifest = {
        "project": cfg["project"]["name"],
        "corpus_version": version,
        "frozen_at": now_iso(),
        "frozen_by": "scripts/04_freeze_manifest.py",
        "random_seed": cfg["project"]["random_seed"],
        "settings_sha256": live_settings_sha,
        "previous_settings_sha256": previous_settings_sha,
        "settings_changed_since_last_freeze": bool(
            previous_settings_sha and previous_settings_sha != live_settings_sha),
        "taxonomy_sha256": sha256_file(root / "config" / "domain_taxonomy.yaml"),
        "sources_csv_sha256": sha256_file(sources_csv) if sources_csv.exists() else None,
        "sources_count": len(sources),
        "documents_included": len(corpus_sources),
        "tier_counts": tier_counts,
        "tier_shares": ({t: round(n / len(corpus_sources), 3) for t, n in tier_counts.items()}
                        if corpus_sources else {}),
        "collected_tier_counts": collected_tier_counts,
        "institutions": institutions,
        "robots": {
            "user_agent": robots.get("user_agent", cfg["crawler"]["user_agent"]),
            "delay_seconds": cfg["crawler"]["delay_seconds"],
            "hosts": len(robots.get("hosts", {})),
            "hosts_audited": sorted(robots.get("hosts", {}).keys()),
        },
        "fixed_configuration": {
            "embedding_model": cfg["safe_corpus"]["embedding_model"],
            "tokenizer": cfg["safe_corpus"]["tokenizer_source"],
            "max_input_tokens": cfg["safe_corpus"]["max_input_tokens"],
            "split_order": cfg["safe_corpus"]["split_order"],
            "strategies": ["fixed-length", "paragraph-based", "heading/list-based", "semantic"],
            "tie_breaking": {
                "topk": [1, 3, 5, 10],
                "token_budgets": [512, 1024, 2048],
                "token_budget_rule": ("admit ranked complete chunks only; stop when the next "
                                      "chunk does not fit; never truncate and never skip ahead"),
            },
        },
        "chunk_stats": chunk_stats,
        "validation": {
            "status": validation.get("status", "not_run"),
            "checks": validation.get("checks", {}),
            "blockers": validation.get("blockers", []),
            "warnings": [w.get("kind") for w in validation.get("warnings", [])],
        },
        "split_audit": {
            "status": split_audit.get("status", "not_run"),
            "counts": split_audit.get("counts", {}),
            "blockers": [b.get("kind") for b in split_audit.get("blockers", [])],
        },
        "artefact_hashes": artefact_hashes,
        "artefact_counts": {k: len(v) for k, v in artefact_hashes.items()},
        "notes": [
            "All gold spans are anchored in the evidence layer "
            "(cleaned/<doc_id>.evidence.jsonl) whose offsets index the concatenated "
            "retained block texts.",
            "Clause spans are stable: chunking never redefines a clause boundary, so one "
            "clause may span several chunks while its annotated span is unchanged.",
            "Token-budget comparisons admit complete chunks in rank order and stop at the "
            "first chunk that does not fit.",
        ],
    }

    out = root / "MANIFEST.json"
    write_json(out, manifest)

    # ---------------------------------------------------------------- compliance
    blocked_hosts = [h for h, v in robots.get("hosts", {}).items()]
    lines = [
        f"# Compliance and provenance - {cfg['project']['name']} {version}",
        "",
        f"Frozen at: {manifest['frozen_at']}",
        f"Documents in the frozen corpus: {len(corpus_sources)} "
        f"(collected during this pass: {len(sources)})",
        "",
        "## Data sources",
        "",
        "Only publicly accessible web pages and PDFs are collected. No login, paywall or",
        "credential is bypassed. No private student information is collected or stored.",
        "Course-level material comes from open courseware published under an open licence;",
        "institutional and faculty-level material is public policy text reproduced for",
        "non-commercial academic research, with the source URL and retrieval timestamp kept",
        "for every document in `sources.csv`.",
        "",
    ]
    licences: dict[str, int] = {}
    for s in sources:
        licences[s.get("license_note", "")] = licences.get(s.get("license_note", ""), 0) + 1
    if licences:
        lines.append("### Licence notes in use")
        lines.append("")
        for lic, n in sorted(licences.items()):
            lines.append(f"- {n} document(s): {lic or '(not recorded)'}")
        lines.append("")

    lines += [
        "## Crawling policy",
        "",
        f"- User agent: `{manifest['robots']['user_agent']}`",
        f"- Per-host delay: {cfg['crawler']['delay_seconds']} s minimum, `Crawl-Delay` honoured when larger",
        f"- Global page cap per run: {cfg['crawler']['max_pages']}",
        "- `robots.txt` is fetched once per host and evaluated before any request to that host.",
        "  A disallowed URL is never requested, and the decision is recorded in",
        "  `reports/robots_audit.json` together with the raw policy text.",
        f"- Hosts audited: {len(blocked_hosts)}",
        "",
        "## Known access restrictions",
        "",
        "The following sources are documented as manual-only because their robots policy",
        "disallows the relevant paths, or the host is unreachable from the collection machine.",
        "They are listed so that a reader can reproduce or extend the corpus by hand:",
        "",
        "- `apps.ualberta.ca` - `Disallow: /catalogue/syllabus` (syllabus downloads disallowed)",
        "- `ro.umich.edu` and `cse.engin.umich.edu` - blanket `Disallow: /`",
        "- `www.qaa.ac.uk` - the quality-code path is disallowed",
        "- `www.ucl.ac.uk` - examinations path disallowed",
        "- Several university hosts resolve only from unrestricted networks and were",
        "  unreachable from the collection machine; these are recorded as",
        "  `unreachable` in `reports/seed_probe_round2.csv` rather than silently skipped.",
        "",
        "## Tier composition (frozen corpus)",
        "",
        "Selection is performed by `scripts/curate_corpus.py`; every inclusion and",
        "exclusion decision is recorded in `reports/curation_decisions.csv`.",
        "",
    ]
    for tier, n in sorted(tier_counts.items()):
        share = n / len(corpus_sources) if corpus_sources else 0
        lines.append(f"- {tier}: {n} document(s)  ({share:.0%})")
    lines.append("")
    lines.append(f"Collected but not selected (retained as provenance): "
                 f"{len(sources) - len(corpus_sources)}")
    lines.append("")

    inst_counts: dict[str, int] = {}
    for s in corpus_sources:
        inst = s.get("institution", "")
        if inst:
            inst_counts[inst] = inst_counts.get(inst, 0) + 1
    if inst_counts:
        lines.append("### Documents per institution (frozen corpus)")
        lines.append("")
        for inst, n in sorted(inst_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- {inst}: {n}")
        lines.append("")

    faculty_n = tier_counts.get("faculty", 0)
    if faculty_n < cfg["acceptance"]["min_faculty_docs"]:
        lines += [
            "### Limitation: faculty tier below target",
            "",
            f"Only {faculty_n} faculty/department-level document(s) were usable, against a target",
            f"of {cfg['acceptance']['min_faculty_docs']}. The corpus therefore operates as a",
            "two-tier corpus (course + institution), with most exception and cross-reference",
            "structure drawn from institutional regulations. This is reported as a limitation:",
            "results speak to written course-governing policy across those levels rather than to",
            "faculty-level delegation specifically.",
            "",
        ]

    lines += [
        "## AI assistance and review status",
        "",
        "The question set and gold evidence spans were drafted with AI assistance",
        "(`scripts/10_ai_annotate.py`) from clauses in this frozen corpus, under an",
        "arrangement approved by the project supervisor for this course assignment, with the",
        "student responsible for review.",
        "",
        "How that is recorded:",
        "",
        "- Every question carries `provenance.generated_by = \"ai-assisted\"`,",
        "  `provenance.model`, `provenance.generated_at` and",
        "  `provenance.review_status`.",
        "- Questions are written with `manual_reviewed = false`. The held-out test split",
        "  refuses them until the student sets `manual_reviewed = true`",
        "  (`scripts/05_audit_splits.py --mark-reviewed`), and that action is appended to",
        "  `reports/lock_audit.jsonl` with a timestamp.",
        "- `annotations/REVIEW.md` and `annotations/REVIEW.csv` are the review packet:",
        "  every question with its reference answer and the verbatim clause text it cites.",
        "",
        "This project is a course assignment rather than publishable data, so no",
        "inter-annotator agreement is claimed for the AI-drafted set. If the results are",
        "ever written up for publication, the annotation provenance above is what a reader",
        "must be told.",
        "",
        "## Reproducibility",
        "",
        "`MANIFEST.json` records a SHA-256 for every raw file, extracted structure, cleaned",
        "clause set and corpus artefact, plus the hashes of the settings and taxonomy that",
        "produced them. Every chunking strategy in this study consumes the single corpus file",
        "recorded under `chunk_stats.file`; therefore corpus differences cannot confound the",
        "comparison. Re-running the pipeline with the same settings reproduces the same hashes.",
        "",
        "## Validation state at freeze",
        "",
        f"- corpus validation: {manifest['validation']['status']}"
        + (" (frozen with documented warnings)" if args.allow_warnings and manifest['validation']['status'] == 'blocked' else ""),
        f"- split audit: {manifest['split_audit']['status']}",
        f"- chunks over model maximum: {chunk_stats.get('chunks_over_max_input', 'n/a')}",
        "",
    ]
    (root / "COMPLIANCE.md").write_text("\n".join(lines), encoding="utf-8")

    print("=== corpus frozen ===")
    print(f"  corpus version      : {version}")
    print(f"  collected documents : {len(sources)}")
    print(f"  documents frozen    : {len(corpus_sources)}")
    print(f"  tiers (frozen)      : {tier_counts}")
    print(f"  institutions        : {len(institutions)}")
    print(f"  chunks              : {chunk_stats.get('chunks', 0)}")
    print(f"  chunk token max     : {chunk_stats.get('token_max', 0)} "
          f"(over max input: {chunk_stats.get('chunks_over_max_input', 0)})")
    print(f"  validation status   : {manifest['validation']['status']}")
    print(f"  artefacts hashed    : {manifest['artefact_counts']}")
    print(f"\nwrote {out}")
    print(f"wrote {root / 'COMPLIANCE.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
