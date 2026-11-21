"""10_ai_annotate.py - AI-assisted annotation with full provenance and hard validation.

Context: the student's supervisor approved AI-generated annotation drafts for this
course assignment, with the student performing final review. This script therefore
produces the drafts. It records that they are AI-assisted and NOT yet reviewed
(`provenance.generated_by`, `review_status = pending_student_review`,
`manual_reviewed = false`). The student's review is what flips `manual_reviewed`.

Why a two-step worklist/ingest design
    The author supplies only the natural language (question + reference answer) and
    which clause(s) answer it. Offsets, quotes, spans and metadata are derived from
    the frozen corpus by `goldset`, so a transcription slip cannot enter the gold set.
    Every batch is validated with the same code that scores retrieval, so a question
    that cannot be resolved is rejected before it is written.

Modes
    --worklist --domain D [--type T] [--n N]   compact list of clauses to write from
    --ingest batch.jsonl [--allow-test]        validate and merge a batch
    --status                                   quota completion
    --restamp-version V                        update clause_ref.version everywhere

Batch record (one JSON object per line)
    {
      "clause_ids": ["DOC#3.1"],                  # or "evidence_sets" below
      "evidence_sets": [["i1","i2"], ["i3"]],     # optional: alternative sufficient sets
      "question": "...", "reference_answer": "...",
      "domain": "grading", "evidence_type": "single-clause",
      "source_level": "course", "condition": "baseline",
      "group_id": "G01", "split": "dev"
    }

Usage
    .venv\\Scripts\\python.exe scripts/10_ai_annotate.py --worklist --domain grading
    .venv\\Scripts\\python.exe scripts/10_ai_annotate.py --ingest annotations/batches/b1.jsonl
    .venv\\Scripts\\python.exe scripts/10_ai_annotate.py --status
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_ROOT, jaccard, load_settings, load_taxonomy, project_paths, read_jsonl,
    shingles, write_json, write_jsonl,
)
import goldset  # noqa: E402

import re  # noqa: E402

GENERATED_BY = "ai-assisted (10_ai_annotate.py)"
MODEL = "deepseek-flash"

DOMAIN_QUOTA = {"grading": 35, "integrity": 30, "late": 25,
                "exam": 25, "attendance": 20, "resources": 15}
TYPE_QUOTA = {"single-clause": 50, "rule-and-exception": 55, "cross-clause": 45}

# Obligation / consequence markers. A real policy provision states what must, may or
# will happen, often with a consequence or a numeric rule. Course-description prose
# ("this class focuses on...") scores near zero and is ranked last, which cuts the
# amount of useless reading when selecting clauses to write questions from.
OBLIGATION_RE = re.compile(
    r"(?i)\b(must|shall|required?|require|not permitted|may not|may be|is entitled|"
    r"will be|are expected|prohibited|liable|penalt\w*|sanction\w*|forfeit\w*|"
    r"deduct\w*|not accepted|no later than|within \w+ (calendar )?days?)\b")
NUMBER_RE = re.compile(r"(\d+\s?%|\b\d+\s?(calendar\s+)?(days?|weeks?|hours?|marks?|points?)\b)")
PROSE_RE = re.compile(r"(?i)\b(this (class|course|seminar) (focuses|examines|covers|explores)|"
                      r"we will (discuss|cover|examine)|students will learn)\b")


def policy_score(text: str) -> int:
    score = 3 * len(OBLIGATION_RE.findall(text))
    score += 2 * len(NUMBER_RE.findall(text))
    if PROSE_RE.search(text):
        score -= 6
    return score


# --------------------------------------------------------------------- helpers
def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_questions(paths: dict) -> list[dict]:
    return read_jsonl(paths["annotations"] / "questions.jsonl")


def save_questions(paths: dict, questions: list[dict]) -> None:
    target = paths["annotations"] / "questions.jsonl"
    tmp = target.with_suffix(".jsonl.tmp")
    write_jsonl(tmp, questions)
    tmp.replace(target)


def next_qids(existing: list[dict], n: int) -> list[str]:
    used = {q.get("qid") for q in existing}
    out, i = [], 1
    while len(out) < n:
        qid = f"Q{i:04d}"
        if qid not in used:
            out.append(qid)
        i += 1
    return out


def build_ref(corpus: goldset.Corpus, clause_id: str) -> dict:
    """Construct a clause_ref from the corpus, so offsets are never transcribed."""
    clause = corpus.clauses[clause_id]
    text = goldset.evidence_text(corpus, clause.doc_id)
    return {
        "doc_id": clause.doc_id,
        "version": corpus.corpus_version,
        "clause_id": clause.clause_id,
        "section_path": clause.section_path,
        "char_start": clause.char_start,
        "char_end": clause.char_end,
        "quote_evidence": text[clause.char_start:clause.char_end],
        "token_start": None,
        "token_end": None,
    }


def usable_clause_pool(corpus: goldset.Corpus, taxonomy: dict,
                       min_score: int = 4, min_hits: int = 2) -> list[dict]:
    """Clauses that plausibly encode a policy provision.

    Filters matter: the corpus contains course-description prose and off-topic
    documents, and a question written from those would not be answerable from a
    policy clause. `min_score` drops prose via obligation/number markers, `min_hits`
    drops weak domain assignments (keyword homonyms such as "grade" in an
    intellectual-property document).
    """
    pool = []
    for doc_id, clauses in corpus.by_doc.items():
        for c in clauses:
            if c.boilerplate or not (150 <= c.char_len <= 1500):
                continue
            domain, hits = goldset.classify_domain(c.text, taxonomy, context=c.section_path)
            if not domain or hits < min_hits:
                continue
            score = policy_score(c.text)
            if score < min_score:
                continue
            pool.append({"clause": c, "domain": domain, "hits": hits,
                         "tier": corpus.doc_tier.get(doc_id, ""),
                         "institution": corpus.doc_institution.get(doc_id, ""),
                         "policy_score": score})
    return pool


# --------------------------------------------------------------------- worklist
def emit_worklist(args, corpus, taxonomy, paths) -> int:
    pool = usable_clause_pool(corpus, taxonomy, args.min_score, args.min_hits)
    wanted = [args.domain] if args.domain else list(DOMAIN_QUOTA)
    out_dir = paths["annotations"] / "worklists"
    out_dir.mkdir(parents=True, exist_ok=True)

    # exception partners, so rule+exception questions can be written directly
    pairs = {}
    pair_csv = paths["root"] / "candidates" / "exception_pairs.csv"
    if pair_csv.exists():
        with pair_csv.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                pairs[(r["doc_id"], r["exception_clause_id"])] = r["rule_clause_id"]

    total = 0
    for domain in wanted:
        items = [p for p in pool if p["domain"] == domain]

        # ---- cross-clause worklist: emit same-document clause PAIRS.
        # A cross-clause question needs two or more clauses that jointly answer it,
        # so emitting single clauses would force the author to guess a partner.
        if args.type == "cross-clause":
            by_doc_items: dict[str, list[dict]] = defaultdict(list)
            for p in items:
                by_doc_items[p["clause"].doc_id].append(p)
            pair_rows = []
            for doc_id, lst in by_doc_items.items():
                lst.sort(key=lambda p: p["clause"].char_start)
                for a, b in zip(lst, lst[1:]):
                    # prefer a pair where at least one side carries a cross-reference
                    if not (a["clause"].has_crossref or b["clause"].has_crossref
                            or a["clause"].has_alt_evidence or b["clause"].has_alt_evidence):
                        continue
                    pair_rows.append({
                        "clause_ids": [a["clause"].clause_id, b["clause"].clause_id],
                        "doc_id": doc_id,
                        "tier": a["tier"],
                        "institution": a["institution"],
                        "domain": domain,
                        "section_path": (a["clause"].section_path or "")[:100],
                        "has_crossref": a["clause"].has_crossref or b["clause"].has_crossref,
                        "text_a": a["clause"].text[:900],
                        "text_b": b["clause"].text[:900],
                        "char_len": a["clause"].char_len + b["clause"].char_len,
                    })
            pair_rows.sort(key=lambda r: -r["char_len"])
            picked_pairs = pair_rows[: (args.n or 40)]
            path = out_dir / f"{domain}.cross-clause.jsonl"
            write_jsonl(path, picked_pairs)
            total += len(picked_pairs)
            print(f"  {domain:<12} {len(picked_pairs):>3} clause PAIRS -> "
                  f"{path.relative_to(paths['root'])}")
            continue

        if args.type == "rule-and-exception":
            items = [p for p in items if p["clause"].has_exception]
        # rank by policy-likeness with a per-document cap, so the question set is
        # not dominated by one large document while still preferring real provisions
        items.sort(key=lambda p: (-p["policy_score"], p["clause"].char_start))
        cap = max(1, (args.n or 30) // 3)
        per_doc_count: Counter = Counter()
        picked = []
        for p in items:
            if per_doc_count[p["clause"].doc_id] >= cap:
                continue
            picked.append(p)
            per_doc_count[p["clause"].doc_id] += 1
            if len(picked) >= (args.n or 30):
                break

        rows = []
        for p in picked:
            c = p["clause"]
            partner = pairs.get((c.doc_id, c.clause_id))
            row = {
                "clause_id": c.clause_id,
                "doc_id": c.doc_id,
                "tier": p["tier"],
                "institution": p["institution"],
                "char_len": c.char_len,
                "policy_score": p["policy_score"],
                "has_exception": c.has_exception,
                "has_crossref": c.has_crossref,
                "has_alt_evidence": c.has_alt_evidence,
                "section_path": c.section_path[:120],
                "text": c.text[:1200],
            }
            if partner and partner in corpus.clauses:
                row["partner_clause_id"] = partner
                row["partner_text"] = corpus.clauses[partner].text[:800]
            rows.append(row)
        path = out_dir / f"{domain}.jsonl" if not args.type else out_dir / f"{domain}.{args.type}.jsonl"
        write_jsonl(path, rows)
        total += len(rows)
        print(f"  {domain:<12} {len(rows):>3} clauses -> {path.relative_to(paths['root'])}")
    print(f"\nworklist records: {total}")
    return 0


# --------------------------------------------------------------------- ingest
def ingest(args, corpus, taxonomy, paths) -> int:
    batch_path = Path(args.ingest)
    if not batch_path.is_absolute():
        batch_path = paths["root"] / batch_path
    if not batch_path.exists():
        print(f"no batch file at {batch_path}", file=sys.stderr)
        return 2

    batch = read_jsonl(batch_path)
    existing = load_questions(paths)
    problems: list[str] = []
    prepared: list[dict] = []

    existing_sig = {tuple(sorted(r["clause_id"] for s in q.get("evidence_sets", [])
                               for r in s.get("clauses", [])))
                    for q in existing}
    existing_text = [(q.get("qid"), shingles(q.get("question", ""))) for q in existing]
    groups = {str(q.get("group_id")): q.get("split") for q in existing}
    qids = next_qids(existing, len(batch))

    for i, item in enumerate(batch):
        where = f"{batch_path.name}[{i}]"
        # ---- resolve evidence sets
        if item.get("evidence_sets"):
            sets_of_ids = item["evidence_sets"]
        elif item.get("clause_ids"):
            sets_of_ids = [item["clause_ids"]]
        else:
            problems.append(f"{where}: needs 'clause_ids' or 'evidence_sets'")
            continue

        missing = [cid for ids in sets_of_ids for cid in ids if cid not in corpus.clauses]
        if missing:
            problems.append(f"{where}: unknown clause_id(s) {missing[:3]}")
            continue

        evidence_sets = [{"set_id": f"S{j+1}", "clauses": [build_ref(corpus, cid) for cid in ids]}
                         for j, ids in enumerate(sets_of_ids)]

        first_clause = corpus.clauses[sets_of_ids[0][0]]
        # qid: honour a supplied one, otherwise take the next free sequential id
        if item.get("qid"):
            qid = str(item["qid"])
        elif i < len(qids):
            qid = qids[i]
        else:
            qid = None
        question = {
            "qid": qid,
            "question": (item.get("question") or "").strip(),
            "reference_answer": (item.get("reference_answer") or "").strip(),
            "group_id": str(item.get("group_id") or "").strip(),
            "split": item.get("split") or "dev",
            "evidence_type": item.get("evidence_type") or args.type or "single-clause",
            "domain": item.get("domain") or args.domain or "",
            "source_level": item.get("source_level") or corpus.doc_tier.get(first_clause.doc_id, ""),
            "condition": item.get("condition") or "baseline",
            "evidence_sets": evidence_sets,
            "prefilled": True,
            "manual_reviewed": False,
            "provenance": {
                "generated_by": GENERATED_BY,
                "model": MODEL,
                "generated_at": _now(),
                "skeleton_id": item.get("skeleton_id"),
                "review_status": "pending_student_review",
                "batch": batch_path.name,
            },
        }

        # ---- protocol validation (span/quote/group/type/domain)
        vp = goldset.validate_question(
            question, corpus,
            valid_evidence_types=STATE_CFG["annotation"]["evidence_types"],
            valid_domains=list(taxonomy.get("domains", {}).keys()),
            valid_levels=STATE_CFG["annotation"]["source_levels"],
            require_manual_review_for_test=False)
        if vp:
            problems.extend(f"{where}: {p}" for p in vp)
            continue

        if not question["question"] or not question["reference_answer"]:
            problems.append(f"{where}: empty question or reference_answer")
            continue
        if not question["group_id"]:
            problems.append(f"{where}: missing group_id")
            continue
        if question["split"] == "test" and not args.allow_test:
            problems.append(f"{where}: split=test requires --allow-test")
            continue

        # ---- duplicates
        sig = tuple(sorted(r["clause_id"] for s in evidence_sets for r in s["clauses"]))
        if sig in existing_sig:
            problems.append(f"{where}: duplicate evidence set already annotated "
                            f"({first_clause.clause_id})")
            continue
        new_sh = shingles(question["question"])
        near = [(qid, round(jaccard(new_sh, sh), 2)) for qid, sh in existing_text
                if jaccard(new_sh, sh) >= 0.85]
        if near:
            problems.append(f"{where}: question too similar to {near[0][0]} ({near[0][1]})")
            continue

        # ---- group/split discipline
        gid = question["group_id"]
        if gid in groups and groups[gid] != question["split"]:
            problems.append(f"{where}: group {gid} already in split {groups[gid]}, "
                            f"cannot also be {question['split']}")
            continue
        groups[gid] = question["split"]

        existing_sig.add(sig)
        existing_text.append((question["qid"], new_sh))
        prepared.append(question)

    if problems:
        print(f"REJECTED BATCH: {len(problems)} problem(s) in {batch_path.name}", file=sys.stderr)
        for p in problems[:25]:
            print(f"  - {p}", file=sys.stderr)
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more", file=sys.stderr)
        return 1

    merged = existing + prepared
    save_questions(paths, merged)
    print(f"ingested {len(prepared)} question(s) from {batch_path.name}")
    print(f"questions.jsonl now {len(merged)}")
    return 0


# --------------------------------------------------------------------- status
def status(paths, taxonomy) -> int:
    questions = load_questions(paths)
    stats = goldset.test_group_counts(questions)
    print("=== annotation status ===")
    print(f"  questions        : {len(questions)}")
    print(f"  dev / test       : {stats['counts']['dev']} / {stats['counts']['test']}")
    print(f"  groups dev/test  : {stats['groups']['dev']} / {stats['groups']['test']} "
          f"(test target >= {STATE_CFG['annotation']['target_test_groups_min']})")
    print(f"  prefilled        : {stats['prefilled']}   manual_reviewed: {stats['manual_reviewed']}")

    def table(title, key, quota):
        have = Counter(q.get(key, "?") for q in questions)
        print(f"\n  {title:<14} {'have':>5} {'quota':>6}  left")
        for k, want in quota.items():
            print(f"    {k:<14} {have.get(k, 0):>5} {want:>6}  {max(0, want - have.get(k, 0)):>5}")
        extra = {k: v for k, v in have.items() if k not in quota}
        if extra:
            print(f"    (unexpected values: {extra})")

    table("domain", "domain", DOMAIN_QUOTA)
    table("type", "evidence_type", TYPE_QUOTA)
    print(f"\n  review_status    : "
          f"{dict(Counter(q.get('provenance', {}).get('review_status', '?') for q in questions))}")
    return 0


# --------------------------------------------------------------------- restamp
def restamp(args, paths) -> int:
    questions = load_questions(paths)
    n = 0
    for q in questions:
        for eset in q.get("evidence_sets") or []:
            for ref in eset.get("clauses") or []:
                if ref.get("version") != args.restamp_version:
                    ref["version"] = args.restamp_version
                    n += 1
    save_questions(paths, questions)
    print(f"restamped {n} clause reference(s) to {args.restamp_version}")
    return 0


# --------------------------------------------------------------------- peek
def peek(args, paths) -> int:
    """Print a worklist compactly so questions can be written without huge files."""
    path = Path(args.peek)
    if not path.is_absolute():
        path = paths["root"] / path
    if not path.exists():
        print(f"no worklist at {path}", file=sys.stderr)
        return 2
    rows = read_jsonl(path)
    limit = args.n or len(rows)
    for i, r in enumerate(rows[:limit]):
        cid = r.get("clause_id") or ",".join(r.get("clause_ids", []))
        flags = "".join(["E" if r.get("has_exception") else "-",
                         "X" if r.get("has_crossref") else "-",
                         "A" if r.get("has_alt_evidence") else "-"])
        print(f"[{i}] {cid} | {r.get('tier','')} | {flags} | {r.get('char_len','')}c")
        if r.get("text"):
            print(f"    {r['text'][:args.chars]}")
        if r.get("partner_text"):
            print(f"    PAIR {r.get('partner_clause_id')}: {r['partner_text'][:args.chars]}")
        if r.get("text_a"):
            print(f"    A {r['clause_ids'][0]}: {r['text_a'][:args.chars]}")
            print(f"    B {r['clause_ids'][1]}: {r['text_b'][:args.chars]}")
        if args.blank:
            print()
    return 0


# --------------------------------------------------------------------- main
STATE_CFG: dict = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--worklist", action="store_true")
    ap.add_argument("--domain", default="")
    ap.add_argument("--type", default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--ingest", default="")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--peek", default="")
    ap.add_argument("--chars", type=int, default=420)
    ap.add_argument("--blank", action="store_true")
    ap.add_argument("--min-score", type=int, default=4)
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--restamp-version", default="")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    STATE_CFG.update(load_settings(root))
    taxonomy = load_taxonomy(root)
    corpus = goldset.load_corpus(root)
    if not corpus.clauses:
        print("no corpus; run 02_extract_clean.py first", file=sys.stderr)
        return 2

    if args.peek:
        return peek(args, paths)
    if args.worklist:
        return emit_worklist(args, corpus, taxonomy, paths)
    if args.ingest:
        return ingest(args, corpus, taxonomy, paths)
    if args.status:
        return status(paths, taxonomy)
    if args.restamp_version:
        return restamp(args, paths)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
