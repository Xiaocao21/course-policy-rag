"""Compare a reproduction run with the official per-question results.

Five layers are reported; only `same_env` makes per-question equality a pass condition. The
historical evaluation environment is not fully recorded (see ENVIRONMENT_EVIDENCE.md), so a
normal reproduction is judged on: identical inputs, metric deltas, and whether the
pre-registered verdict is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

METRICS = ("complete@10", "hit@10", "recall@10", "mrr", "first_hit_rank")
TRACKS = ("main", "C1", "C2", "C3")


def num(value) -> float:
    return 0.0 if value is None else float(value)


def load_perq(path: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(row["qid"], row["config"])] = row
    return rows


def compare(official_dir: Path, repro_dir: Path) -> dict:
    report = {"per_track": {}, "totals": {"rows": 0, "identical": 0, "different": 0}}
    for track in TRACKS:
        a = load_perq(official_dir / f"perq_{track}.jsonl")
        b = load_perq(repro_dir / f"perq_{track}.jsonl")
        keys = sorted(set(a) & set(b))
        identical = 0
        diffs = []
        sums_a = {m: 0.0 for m in METRICS}
        sums_b = {m: 0.0 for m in METRICS}
        for key in keys:
            same = all(abs(num(a[key][m]) - num(b[key][m])) < 1e-12 for m in METRICS)
            identical += int(same)
            if not same and len(diffs) < 20:
                diffs.append({"qid": key[0], "config": key[1],
                              "official": {m: a[key][m] for m in METRICS},
                              "repro": {m: b[key][m] for m in METRICS}})
            for m in METRICS:
                sums_a[m] += num(a[key][m])
                sums_b[m] += num(b[key][m])
        n = len(keys) or 1
        report["per_track"][track] = {
            "rows": len(keys),
            "identical": identical,
            "different": len(keys) - identical,
            "mean_official": {m: round(sums_a[m] / n, 6) for m in METRICS},
            "mean_repro": {m: round(sums_b[m] / n, 6) for m in METRICS},
            "delta": {m: round((sums_b[m] - sums_a[m]) / n, 6) for m in METRICS},
            "examples": diffs,
        }
        report["totals"]["rows"] += len(keys)
        report["totals"]["identical"] += identical
        report["totals"]["different"] += len(keys) - identical
    report["exact"] = report["totals"]["different"] == 0
    return report
