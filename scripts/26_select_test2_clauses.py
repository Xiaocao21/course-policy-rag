"""26_select_test2_clauses.py - draw and freeze the evidence bundles for the second held-out set.

Why bundles, not clauses
------------------------
A rule-and-exception or cross-clause question needs more than one clause to be answerable, so
freezing 45 *clauses* would not freeze the evidence of 45 *questions*. This script therefore
freezes **bundles**: a primary clause plus every eligible companion candidate in the same
document, ranked, with exclusions recorded. The annotator may only choose among the frozen
candidates (or downgrade to single-clause, or amend); the corpus is never re-searched for a
"better" companion after the freeze.

Two phases
----------
    propose   deterministic draw of 60 primaries (15 per length quartile) plus the full
              candidate list for each; writes results/test2/bundle_candidates.json
    --freeze  validate the annotated bundle file against every invariant and write
              results/test2/selected_bundles.json (+ selection_sha256, frozen_at)

Sampling is deterministic: `random.Random(20261013)` over clause ids sorted before shuffling,
with a domain quota and a per-institution cap. Nothing here looks at retrieval results - the
test2 questions must not be tuned against the metric.

Length
------
`bin` is decided by `max_evidence_chars = max(primary_char_len, companion_char_len...)`, which
is the same statistic the test split's difficulty target was computed from (max annotated
clause span per question). Companions are *not* required to be shorter than the primary: that
would be an evidence constraint that the relation does not need. Where a shorter companion is
chosen deliberately to land a bundle in a given length bin, that is a **sampling design**
choice and must be written into the bundle's `rationale`.

Usage
    .venv\\Scripts\\python.exe scripts/26_select_test2_clauses.py
    .venv\\Scripts\\python.exe scripts/26_select_test2_clauses.py --freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, project_paths, read_jsonl, write_json  # noqa: E402
import goldset  # noqa: E402

SEED = 20261013
MIN_CHARS = 150
# Independent cap on a companion's length: the corpus clause cap from settings.yaml
# (quality.max_clause_chars). It is deliberately *not* "shorter than the primary" - the
# relation does not require that, and imposing it would exclude legitimate exception or
# definition clauses.
CLAUSE_CAP = 1200
# Quartiles of the test split's per-question maximum annotated clause span (preregistered).
QUARTILES = (728.0, 1009.5, 1121.8)
BINS = (0.0, *QUARTILES, float("inf"))
PER_BIN = 15
TOTAL = 60
TOTAL_RANGE = (50, 60)     # the clean pool may not fill every cell; deviations are recorded
FIRST_WAVE = 45
BIN_RANGE = (10, 18)
DOMAIN_TARGETS = {"exam": 12, "grading": 12, "integrity": 11, "late": 7,
                  "attendance": 7, "resources": 5}
UNCLASSIFIED_CAP = 8
INSTITUTION_CAP = 10
DOC_CAP = 5

REFERENCE_RE = re.compile(
    r"(?i)\b(see|refer to|set out in|in accordance with|as described in|"
    r"under (?:the )?(?:section|policy|regulation)|subject to)\b")
PROCEDURAL_RE = re.compile(
    r"(?i)\b(before|after|first|then|within \d+ (?:days|weeks|hours)|must|prior to)\b")
# A readability screen, needed because the corpus mixes policy prose with OCR-damaged
# fragments and legal-citation dumps. Numbers alone do not make a clause question-usable:
# 1,239 clauses meet the length/boilerplate definition, but only 918 also read as prose.
_CITATION_RE = re.compile(r"\b\d+\s+F\.\s?\d?d|\b\d+\s+U\.\S*S\.\s?\d+|§|\bv\.\s+[A-Z]")
_POLICY_CUE_RE = re.compile(
    r"(?i)\b(must|shall|may|will|required|permitted|prohibited|expected|responsible|"
    r"deadline|grade|appeal|exam|student|course)\b")


def is_readable(clause) -> bool:
    """Prose screen for question-bearing clauses.

    Three failure modes were found by reading the pool: OCR fragments from a patent-law
    report, course-content notes (a finance syllabus), and layout debris (URL lines, table
    rows, bullet fragments whose lead-in lives in another clause). None of them can carry a
    question, so they are excluded from the sampling frame rather than discovered later.
    """
    text = clause.text.strip()
    if len(text.split()) < 15:
        return False
    if sum(ch.isalpha() for ch in text) / max(1, len(text)) < 0.55:
        return False
    if re.search(r"www\.|https?://", text):
        return False
    if text.count("|") >= 2:                      # table row
        return False
    if re.match(r"^\([a-z0-9]{1,3}[).]", text):   # bullet fragment without its lead-in
        return False
    if re.match(r"^[a-z]", text):                 # starts mid-sentence
        return False
    if _CITATION_RE.search(text):
        return False
    if not _POLICY_CUE_RE.search(text):
        return False
    return bool(re.search(r"[A-Za-z]{4,}", text[:30]))

STOP = set("""a an the of to in on for by with and or is are was were be been being that this
these those it its as at from not no any all such which who whom whose shall may must can could
should will would if then than there their they them he she his her you your we our""".split())


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_json(data) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOP}


def bin_of(length: float) -> int:
    for i in range(len(BINS) - 1):
        if BINS[i] <= length < BINS[i + 1]:
            return i + 1
    return len(BINS) - 1


def load_state(root: Path):
    corpus = goldset.load_corpus(root)
    questions = read_jsonl(project_paths(root)["annotations"] / "questions.jsonl")
    used = {ref.get("clause_id")
            for q in questions
            for eset in (q.get("evidence_sets") or [])
            for ref in (eset.get("clauses") or [])}
    pool = [c for c in corpus.clauses.values()
            if c.clause_id not in used and not c.boilerplate and len(c.text) >= MIN_CHARS
            and is_readable(c)]
    return corpus, used, pool


def companion_candidates(primary, doc_clauses, used: set[str]) -> tuple[list[dict], list[dict]]:
    """Every eligible companion in the same document, ranked; and the exclusions."""
    words = content_words(primary.text)
    scored = []
    excluded = []
    for c in doc_clauses:
        if c.clause_id == primary.clause_id:
            continue
        reason = None
        if c.clause_id in used:
            reason = "used_by_dev_or_test"
        elif c.boilerplate:
            reason = "boilerplate"
        elif len(c.text) < MIN_CHARS:
            reason = "too_short"
        elif c.char_len > CLAUSE_CAP:
            reason = "over_clause_cap"
        if reason:
            excluded.append({"clause_id": c.clause_id, "reason": reason})
            continue
        overlap = len(words & content_words(c.text))
        scored.append({"clause_id": c.clause_id,
                       "char_len": c.char_len,
                       "shared_content_words": overlap,
                       "section_path": c.section_path,
                       "has_exception": c.has_exception,
                       "has_crossref": c.has_crossref})
    scored.sort(key=lambda r: (-r["shared_content_words"], r["clause_id"]))
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
        row["eligible"] = True
        row["reason"] = ""
    return scored, excluded


def propose(root: Path) -> int:
    corpus, used, pool = load_state(root)
    taxonomy = _taxonomy(root)
    # `used` is recomputed below for the companion check; the pool already excludes it.
    by_doc = defaultdict(list)
    for c in pool:
        by_doc[c.doc_id].append(c)

    by_bin = defaultdict(list)
    for c in pool:
        by_bin[bin_of(c.char_len)].append(c)

    rng = random.Random(SEED)
    institution_count: Counter = Counter()
    doc_count: Counter = Counter()
    domain_count: Counter = Counter()
    per_bin_count: Counter = Counter()
    chosen: list = []
    relaxations: list = []

    def domain_of(clause) -> str:
        return goldset.classify_domain(clause.text, taxonomy, clause.section_path)[0] or ""

    def take(b: int, clause, domain: str) -> None:
        chosen.append((b, clause, domain))
        doc_count[clause.doc_id] += 1
        institution_count[clause.institution] += 1
        domain_count[domain] += 1
        per_bin_count[b] += 1

    # Pass 1: scarcest label first, so resources/integrity are not crowded out by the
    # unclassified majority. Within a domain, bins are filled in order.
    for domain in sorted(DOMAIN_TARGETS, key=lambda d: len([c for c in pool if domain_of(c) == d])):
        needed = DOMAIN_TARGETS[domain]
        if needed <= 0:
            continue
        for b in range(1, len(BINS)):
            if needed <= 0:
                break
            if per_bin_count[b] >= PER_BIN:
                continue
            cands = sorted([c for c in by_bin.get(b, []) if domain_of(c) == domain],
                           key=lambda c: c.clause_id)
            rng.shuffle(cands)
            for c in cands:
                if needed <= 0 or per_bin_count[b] >= PER_BIN:
                    break
                if institution_count[c.institution] >= INSTITUTION_CAP:
                    continue
                if doc_count[c.doc_id] >= DOC_CAP:
                    continue
                take(b, c, domain)
                needed -= 1
        if needed > 0:
            relaxations.append({"domain": domain, "unfilled": needed,
                                "reason": "not enough eligible clauses under the caps"})

    # Pass 2: fill the remaining bin slots, preferring unclassified clauses (bounded) so the
    # pool's large empty-label mass does not dominate the sample.
    taken_ids = {c.clause_id for _, c, _ in chosen}
    for b in range(1, len(BINS)):
        if per_bin_count[b] >= PER_BIN:
            continue
        cands = sorted([c for c in by_bin.get(b, []) if c.clause_id not in taken_ids],
                       key=lambda c: c.clause_id)
        rng.shuffle(cands)
        # labelled-but-over-quota clauses first, then unclassified under its cap
        order = [c for c in cands if domain_of(c) and domain_count[domain_of(c)] < 99] + \
                [c for c in cands if not domain_of(c)]
        for c in order:
            if per_bin_count[b] >= PER_BIN:
                break
            if c.clause_id in taken_ids:
                continue
            if institution_count[c.institution] >= INSTITUTION_CAP:
                continue
            if doc_count[c.doc_id] >= DOC_CAP:
                continue
            domain = domain_of(c)
            if not domain and domain_count[""] >= UNCLASSIFIED_CAP:
                continue
            take(b, c, domain)
            taken_ids.add(c.clause_id)
        if per_bin_count[b] < PER_BIN:
            relaxations.append({"bin": b, "reason": f"only {per_bin_count[b]} of {PER_BIN}"})

    chosen.sort(key=lambda x: (x[0], x[1].clause_id))
    chosen = sorted(chosen, key=lambda x: (x[0], x[1].clause_id))

    bundles = []
    for i, (b, c, domain) in enumerate(chosen, start=1):
        cands, excluded = companion_candidates(c, by_doc[c.doc_id], used)
        bundles.append({
            "bundle_id": f"T2B{i:02d}",
            "wave": "first" if i <= FIRST_WAVE else "reserve",
            "primary": {"clause_id": c.clause_id, "doc_id": c.doc_id,
                        "char_start": c.char_start, "char_end": c.char_end,
                        "char_len": c.char_len, "version": corpus.corpus_version,
                        "has_exception": c.has_exception, "has_crossref": c.has_crossref,
                        "has_alt_evidence": c.has_alt_evidence},
            "bin": b,
            "domain_classifier": domain,
            "institution": c.institution,
            "tier": c.tier,
            "companion_candidates": cands[:8],
            "excluded_candidates": excluded[:12],
            "chosen_companions": [],
            "type": None,
            "relation_evidence": None,
            "rationale": None,
        })

    out = {
        "kind": "test2_bundle_candidates",
        "created_at": now(),
        "seed": SEED,
        "quartiles": list(QUARTILES),
        "bins": [[BINS[i], BINS[i + 1]] for i in range(len(BINS) - 1)],
        "domain_targets": DOMAIN_TARGETS,
        "institution_cap": INSTITUTION_CAP,
        "n_pool": len(pool),
        "n_bundles": len(bundles),
        "per_bin": dict(Counter(b["bin"] for b in bundles)),
        "per_domain_classifier": dict(Counter(b["domain_classifier"] or "(none)"
                                              for b in bundles)),
        "per_institution": dict(Counter(b["institution"] for b in bundles)),
        "relaxations": relaxations,
        "bundles": bundles,
        "note": ("Primaries are frozen; the annotator chooses companions only from "
                 "companion_candidates, downgrades to single-clause, or amends."),
    }
    path = root / "results" / "test2" / "bundle_candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, out)
    print("=== test2 bundle candidates ===")
    print(f"  pool            : {len(pool)}")
    print(f"  bundles         : {len(bundles)}  per bin {out['per_bin']}")
    print(f"  domain (classifier): {out['per_domain_classifier']}")
    print(f"  institutions    : {len(out['per_institution'])}  max "
          f"{max(out['per_institution'].values()) if out['per_institution'] else 0}")
    print(f"  relaxations     : {len(relaxations)}")
    print(f"  wrote {path}")
    return 0


def _taxonomy(root: Path):
    from common import load_taxonomy
    return load_taxonomy(root)


def check_relation(basis: str, quote: str, texts: list[str], flags: list[dict]) -> list[str]:
    """Machine *candidate* checks. These never prove the relation - the review does."""
    problems = []
    if not quote:
        problems.append("relation_evidence.quote is empty")
        return problems
    if not any(quote in t for t in texts):
        problems.append("relation_evidence.quote is not verbatim in any clause")
    if basis == "rule_exception":
        if not any(f.get("has_exception") for f in flags):
            problems.append("rule_exception requires a clause flagged has_exception")
        if not re.search(r"(?i)\b(except|unless|provided that|however|notwithstanding|subject to)\b",
                         " ".join(texts)):
            problems.append("rule_exception: no exception language in the bundle")
    elif basis == "explicit_reference":
        if not REFERENCE_RE.search(quote):
            problems.append("explicit_reference: quote has no reference phrase")
    elif basis == "procedural_dependency":
        if not PROCEDURAL_RE.search(quote):
            problems.append("procedural_dependency: quote has no procedural marker")
    elif basis == "defined_term":
        term = quote.strip()
        hits = sum(1 for t in texts if term in t)
        if hits < 2:
            problems.append("defined_term: the term does not appear in at least two clauses")
    else:
        problems.append(f"unknown basis {basis!r}")
    return problems


def freeze(root: Path) -> int:
    corpus, used, pool = load_state(root)
    src = root / "results" / "test2" / "bundles_proposed.json"
    if not src.exists():
        print(f"REFUSING: {src} not found; construct the bundles first", file=sys.stderr)
        return 2
    proposed = json.loads(src.read_text(encoding="utf-8"))
    bundles = proposed.get("bundles") or []
    errors: list[str] = []
    if not TOTAL_RANGE[0] <= len(bundles) <= TOTAL_RANGE[1]:
        errors.append(f"expected {TOTAL_RANGE[0]}-{TOTAL_RANGE[1]} bundles, found {len(bundles)}")
    waves = Counter(b.get("wave") for b in bundles)
    if waves.get("first", 0) not in (40, 45):
        errors.append(f"first wave must be {FIRST_WAVE}, got {waves.get('first', 0)}")

    seen: set[str] = set()
    bins = Counter()
    for b in bundles:
        bid = b.get("bundle_id")
        primary = corpus.clauses.get((b.get("primary") or {}).get("clause_id", ""))
        if primary is None:
            errors.append(f"{bid}: primary clause not in corpus")
            continue
        if primary.clause_id in used:
            errors.append(f"{bid}: primary clause is already used by dev/test")
        if primary.boilerplate or len(primary.text) < MIN_CHARS:
            errors.append(f"{bid}: primary clause is not usable")
        clauses = [primary]
        for ref in b.get("chosen_companions") or []:
            cid = ref.get("clause_id", "")
            companion = corpus.clauses.get(cid)
            if companion is None:
                errors.append(f"{bid}: companion {cid} not in corpus")
                continue
            if not ref.get("relation"):
                errors.append(f"{bid}: companion {cid} missing relation")
            if companion.doc_id != primary.doc_id:
                if not (b.get("relation_evidence") or {}).get("basis") == "explicit_reference":
                    errors.append(f"{bid}: cross-document companion needs explicit_reference")
            clauses.append(companion)
        if len({c.clause_id for c in clauses}) != len(clauses):
            errors.append(f"{bid}: duplicate clause inside the bundle")
        for c in clauses:
            if c.clause_id in seen:
                errors.append(f"{bid}: clause {c.clause_id} appears in another bundle")
            seen.add(c.clause_id)
            if c.clause_id in used:
                errors.append(f"{bid}: clause {c.clause_id} is used by dev/test")

        btype = b.get("type")
        if btype == "single-clause" and len(clauses) != 1:
            errors.append(f"{bid}: single-clause with {len(clauses)} clauses")
        if btype == "cross-clause" and len(clauses) < 2:
            errors.append(f"{bid}: cross-clause needs >= 2 clauses")
        if btype == "rule-and-exception":
            if len(clauses) < 2:
                errors.append(f"{bid}: rule-and-exception needs >= 2 clauses")
            if not any(c.has_exception for c in clauses):
                errors.append(f"{bid}: rule-and-exception without an exception clause")
        if btype not in ("single-clause", "rule-and-exception", "cross-clause"):
            errors.append(f"{bid}: unknown type {btype!r}")

        # chosen companions must come from the frozen candidate list
        eligible = {c["clause_id"] for c in (b.get("companion_candidates") or [])}
        for ref in b.get("chosen_companions") or []:
            if ref.get("clause_id") not in eligible:
                errors.append(f"{bid}: companion {ref.get('clause_id')} is not a frozen "
                              f"candidate (a new candidate needs an amendment)")

        if btype != "single-clause":
            ev = b.get("relation_evidence") or {}
            flags = [{"has_exception": c.has_exception} for c in clauses]
            problems = check_relation(ev.get("basis", ""), ev.get("quote", ""),
                                      [c.text for c in clauses], flags)
            errors.extend(f"{bid}: {p}" for p in problems)
        if not (b.get("rationale") or "").strip():
            errors.append(f"{bid}: rationale is empty")

        lengths = [c.char_len for c in clauses]
        max_len = max(lengths)
        declared = (b.get("lengths") or {})
        if declared.get("max_evidence_chars") != max_len:
            errors.append(f"{bid}: max_evidence_chars should be {max_len}, "
                          f"got {declared.get('max_evidence_chars')}")
        if b.get("bin") != bin_of(max_len):
            errors.append(f"{bid}: bin should be {bin_of(max_len)} for {max_len} chars")
        bins[bin_of(max_len)] += 1

    for b in range(1, len(BINS)):
        if not BIN_RANGE[0] <= bins.get(b, 0) <= BIN_RANGE[1]:
            errors.append(f"bin {b} has {bins.get(b, 0)} bundles (allowed {BIN_RANGE[0]}-{BIN_RANGE[1]})")

    if errors:
        print("REFUSING: bundle file violates the freeze invariants:", file=sys.stderr)
        for e in errors[:40]:
            print(f"  {e}", file=sys.stderr)
        return 3

    frozen = {
        "kind": "test2_selected_bundles",
        "frozen_at": now(),
        "seed": SEED,
        "quartiles": list(QUARTILES),
        "n_bundles": len(bundles),
        "per_bin": {str(k): v for k, v in sorted(bins.items())},
        "amendment_policy": ("frozen after this point; changes only via "
                             "results/test2/bundle_amendments.jsonl"),
        "bundles": bundles,
    }
    frozen["selection_sha256"] = sha256_json(bundles)
    path = root / "results" / "test2" / "selected_bundles.json"
    write_json(path, frozen)
    print("=== test2 bundles frozen ===")
    print(f"  bundles: {len(bundles)}  per bin {frozen['per_bin']}")
    print(f"  selection_sha256: {frozen['selection_sha256']}")
    print(f"  wrote {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--freeze", action="store_true",
                    help="validate bundles_proposed.json and write selected_bundles.json")
    args = ap.parse_args()
    root = Path(args.root)
    return freeze(root) if args.freeze else propose(root)


if __name__ == "__main__":
    sys.exit(main())
