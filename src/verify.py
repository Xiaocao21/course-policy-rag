"""verify.py - statistical recomputation (R1) against the committed official results.

This does **not** re-run retrieval: it reads the committed per-question results and recomputes
every statistic with the same script and seed, then compares every field. The two official
splits were checked this way when the package was built: `stats.json` is byte-identical after
JSON parsing for both splits, and for test2 the `OUTCOME.json` matches except `created_at`.

Declared exclusions (fixed here, never widened at run time):
    results/final*/stats.json     none
    results/test2/OUTCOME.json    created_at (a wall-clock field, not a result)

The official files are hashed before and after; a mismatch in either is a failure.

Usage
    python src/verify.py --split test
    python src/verify.py --split test2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKS = ("main", "C1", "C2", "C3")
EXCLUDE = {"results/test2/OUTCOME.json": {"created_at"}}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diff(a, b, path: str = "", skip: set[str] | None = None) -> list[str]:
    skip = skip or set()
    out: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key in skip:
                continue
            if key not in a:
                out.append(f"{path}/{key}: missing on official side")
            elif key not in b:
                out.append(f"{path}/{key}: missing on recomputed side")
            else:
                out.extend(diff(a[key], b[key], f"{path}/{key}", skip))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(diff(x, y, f"{path}[{i}]", skip))
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run([sys.executable, *cmd], cwd=str(ROOT), check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["test", "test2"])
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    if args.split == "test":
        src = root / "results" / "final"
        official_stats = src / "stats.json"
        questions = "annotations/questions.jsonl"
    else:
        src = root / "results" / "final_test2"
        official_stats = src / "stats.json"
        questions = "annotations/questions_test2.jsonl"

    work = root / "results" / "repro" / "r1" / args.split
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for track in TRACKS:
        shutil.copy2(src / f"perq_{track}.jsonl", work / f"perq_{track}.jsonl")

    guarded = [official_stats]
    official_outcome = root / "results" / "test2" / "OUTCOME.json"
    if args.split == "test2":
        guarded.append(official_outcome)
    before = {str(p.relative_to(root)): sha256(p) for p in guarded}

    stats_cmd = ["scripts/16_stats.py", "--split", args.split, "--results", str(work)]
    # The official test statistics were produced without --questions (the sampling frame is
    # computed from the corpus); only test2 requires the questions file. Passing it for test
    # changes by_length_tercile.sampling_frame and is therefore a different computation.
    if args.split == "test2":
        stats_cmd += ["--questions", questions]
    run(stats_cmd)
    if args.split == "test2":
        run(["scripts/28_decide_test2.py", "--stats", str(work / "stats.json"),
             "--prereg", "results/test2/PREREGISTRATION.json",
             "--out", str(work / "OUTCOME.json")])

    report = {"split": args.split, "created_at": datetime.now(timezone.utc).isoformat(),
              "inputs": [f"perq_{t}.jsonl" for t in TRACKS] + [questions], "checks": {}}
    stats_diffs = diff(json.loads(official_stats.read_text(encoding="utf-8")),
                       json.loads((work / "stats.json").read_text(encoding="utf-8")))
    report["checks"]["stats"] = {"files": [f"results/{'final' if args.split == 'test' else 'final_test2'}/stats.json",
                                           str((work / 'stats.json').relative_to(root))],
                                 "equal": not stats_diffs, "diffs": stats_diffs[:200]}
    if args.split == "test2":
        skip = EXCLUDE["results/test2/OUTCOME.json"]
        outcome_diffs = diff(json.loads(official_outcome.read_text(encoding="utf-8")),
                             json.loads((work / "OUTCOME.json").read_text(encoding="utf-8")),
                             skip=skip)
        report["checks"]["outcome"] = {"files": ["results/test2/OUTCOME.json",
                                                 str((work / "OUTCOME.json").relative_to(root))],
                                       "excluded_fields": sorted(skip),
                                       "equal": not outcome_diffs,
                                       "diffs": outcome_diffs[:200]}
    after = {str(p.relative_to(root)): sha256(p) for p in guarded}
    report["official_hashes_unchanged"] = before == after
    report["ok"] = all(c["equal"] for c in report["checks"].values()) and before == after
    (work / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    for name, check in report["checks"].items():
        print(f"{name:<8} {'PASS' if check['equal'] else 'FAIL'}  "
              f"(diffs: {len(check['diffs'])})")
        for line in check["diffs"][:5]:
            print(f"  {line}")
    print(f"official hashes unchanged: {report['official_hashes_unchanged']}")
    print(f"report: {work / 'report.json'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
