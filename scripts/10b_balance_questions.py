"""10b_balance_questions.py - reduce an over-represented evidence type.

The first annotation pass produced far more `single-clause` questions than the protocol's
evidence-type quota calls for, because single-clause items are the easiest to write. The
study compares strategies per evidence type, so a skewed set would make the
rule-and-exception and cross-clause comparisons statistically thin.

This script removes surplus questions of one type (single-clause by default) until the
type shares approach their targets. Selection is deterministic and documented:
    1. prefer dropping items whose evidence carries no exception / cross-reference /
       alternative-evidence signal (the least informative single-clause items)
    2. then items in domains whose share of that type is furthest above its target
    3. tie-break by qid, so the result is reproducible
It never drops below `--min-per-domain` questions of that type in any domain, and it
keeps the dev/test ratio close to its previous value by alternating splits.

Dropped items are recorded in reports/question_balance.json with a reason, and the
action is appended to reports/lock_audit.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, project_paths, read_jsonl, write_json, write_jsonl  # noqa: E402

DEFAULT_TARGET = {"single-clause": 0.33, "rule-and-exception": 0.37, "cross-clause": 0.30}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def signal_strength(q: dict) -> int:
    """Higher means the item's evidence is structurally richer, so keep it."""
    score = 0
    for eset in q.get("evidence_sets") or []:
        for ref in eset.get("clauses") or []:
            if ref.get("has_exception"):
                score += 2
            if ref.get("has_crossref"):
                score += 1
            if ref.get("has_alt_evidence"):
                score += 1
        if len(eset.get("clauses") or []) > 1:
            score += 1
    if len(q.get("evidence_sets") or []) > 1:
        score += 2
    return score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--drop-type", default="single-clause")
    ap.add_argument("--target-share", type=float, default=0.0,
                    help="target share for --drop-type (default 0.33 for single-clause)")
    ap.add_argument("--max-drop", type=int, default=0, help="cap on how many to drop")
    ap.add_argument("--target-count", type=int, default=0,
                    help="keep exactly this many of --drop-type (overrides --target-share)")
    ap.add_argument("--min-per-domain", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    paths = project_paths(root)
    qfile = paths["annotations"] / "questions.jsonl"
    questions = read_jsonl(qfile)
    if not questions:
        print("no questions", file=sys.stderr)
        return 2

    target_share = args.target_share or (0.33 if args.drop_type == "single-clause" else 0.25)
    by_type = Counter(q.get("evidence_type") for q in questions)
    total = len(questions)
    droppable = [q for q in questions if q.get("evidence_type") == args.drop_type]
    keep_other = total - len(droppable)

    # solve: drop / (total - drop) such that remaining share == target
    # remaining_drop_type = total - drop - keep_other ; solve for drop
    # (len(droppable) - drop) / (total - drop) = target_share
    if args.target_count:
        # absolute target is easier to reason about than a share, and avoids the
        # feedback loop where each drop raises the other types' shares
        ideal_drop = max(0, len(droppable) - args.target_count)
    else:
        numer = len(droppable) - target_share * total
        denom = 1 - target_share
        ideal_drop = int(round(numer / denom)) if denom else 0
    ideal_drop = max(0, min(ideal_drop, len(droppable)))
    if args.max_drop:
        ideal_drop = min(ideal_drop, args.max_drop)

    # domain caps: never go below min_per_domain of this type
    dom_counts = Counter(q.get("domain") for q in droppable)
    dom_budget = {d: max(0, c - args.min_per_domain) for d, c in dom_counts.items()}

    def rank(q: dict) -> tuple:
        d = q.get("domain", "?")
        dom_share = dom_counts[d] / len(droppable) if droppable else 0
        over = -(dom_share)          # domains carrying more of the type drop first
        return (signal_strength(q), over, q.get("split") != "dev", q.get("qid", ""))

    ordered = sorted(droppable, key=rank)

    # Drop proportionally to the existing dev:test split so the ratio does not drift.
    split_totals = Counter(q.get("split") for q in questions)
    total_q = sum(split_totals.values()) or 1
    split_quota = {sp: int(round(ideal_drop * n / total_q)) for sp, n in split_totals.items()}

    drop: list[dict] = []
    split_used: Counter = Counter()
    for q in ordered:
        if len(drop) >= ideal_drop:
            break
        d = q.get("domain", "?")
        if dom_budget.get(d, 0) <= 0:
            continue
        sp = q.get("split")
        if split_quota.get(sp, 0) <= 0:
            continue
        dom_budget[d] -= 1
        split_quota[sp] -= 1
        split_used[sp] += 1
        drop.append(q)

    dropped_ids = {q.get("qid") for q in drop}
    kept = [q for q in questions if q.get("qid") not in dropped_ids]

    before = dict(Counter(q.get("evidence_type") for q in questions))
    after = dict(Counter(q.get("evidence_type") for q in kept))

    report = {
        "at": _now(),
        "drop_type": args.drop_type,
        "target_share": target_share,
        "before": {"total": len(questions), "types": before,
                   "splits": dict(Counter(q.get("split") for q in questions))},
        "after": {"total": len(kept), "types": after,
                  "splits": dict(Counter(q.get("split") for q in kept)),
                  "shares": {k: round(v / len(kept), 3) for k, v in after.items()}},
        "dropped": [{"qid": q.get("qid"), "domain": q.get("domain"),
                     "split": q.get("split"), "signal": signal_strength(q),
                     "reason": "surplus single-clause item; lowest structural signal first"}
                    for q in drop],
        "note": ("Dropping is a deliberate rebalance to the protocol's evidence-type "
                 "quota; the dropped drafts remain reproducible from the batch files."),
    }
    write_json(paths["reports"] / "question_balance.json", report)

    print("=== question balance ===")
    print(f"  before: {len(questions)} questions {before}")
    print(f"  after : {len(kept)} questions {after}")
    print(f"  shares: " + ", ".join(f"{k} {v / len(kept):.0%}" for k, v in after.items()))
    print(f"  split : {report['after']['splits']}")
    print(f"  dropped {len(drop)} x {args.drop_type}"
          + (" (dry run, nothing written)" if args.dry_run else ""))

    if not args.dry_run and drop:
        tmp = qfile.with_suffix(".jsonl.tmp")
        write_jsonl(tmp, kept)
        tmp.replace(qfile)
        log = paths["reports"] / "lock_audit.jsonl"
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": _now(), "event": "balance_questions",
                                "dropped": len(drop),
                                "dropped_qids": sorted(dropped_ids)},
                               ensure_ascii=False) + "\n")
        print(f"\nwrote {qfile}")
    print(f"wrote {paths['reports'] / 'question_balance.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
