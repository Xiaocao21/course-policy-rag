"""export_results.py - write the flat result tables the rubric asks for.

    results/results.csv          one row per split x track (paper Table 1 and 5.1)
    results/experiment_log.csv   experiments/log.jsonl flattened
    results/artifact_map.csv     paper location -> split -> run -> config -> checkpoint -> files

Everything is derived from committed JSON; nothing is typed by hand.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKS = ("main", "C1", "C2", "C3")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def results_table() -> list[dict]:
    rows = []
    for split, stats_path, run_path in (
            ("test", ROOT / "results/final/stats.json", ROOT / "results/final/FINAL_RUN.json"),
            ("test2", ROOT / "results/final_test2/stats.json",
             ROOT / "results/final_test2/FINAL_RUN.json")):
        stats, run = load(stats_path), load(run_path)
        for track in TRACKS:
            block = stats["tracks"][track]["complete@10"]
            rows.append({
                "split": split, "track": track,
                "complete@10": round(block["mean"], 6),
                "ci_low": round(block["ci_low"], 6), "ci_high": round(block["ci_high"], 6),
                "groups": stats["groups"], "run_attempt": run.get("attempt"),
                "source": str(stats_path.relative_to(ROOT)).replace("\\", "/"),
            })
    return rows


def experiment_log() -> list[dict]:
    rows = []
    path = ROOT / "experiments/log.jsonl"
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append({
            "logged_at": row.get("logged_at", ""), "kind": row.get("kind", ""),
            "track": row.get("track", ""), "split": row.get("split", ""),
            "run_id": row.get("run_id", ""), "env_role": row.get("env_role", ""),
            "gpu": row.get("gpu", ""),
            "val_complete@10": (row.get("val") or {}).get("macro_complete@10", ""),
            "questions": row.get("questions", ""), "configs": row.get("configs", ""),
        })
    return rows


def artifact_map() -> list[dict]:
    frozen = load(ROOT / "models/reranker/FROZEN.json")
    test_run = load(ROOT / "results/final/FINAL_RUN.json")
    test2_run = load(ROOT / "results/final_test2/FINAL_RUN.json")
    ckpt = frozen["model"]["checkpoint_sha256"]
    eval_ids = {}
    for line in (ROOT / "experiments/log.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("kind") == "eval" and row.get("split"):
                eval_ids[row["split"]] = row.get("run_id", "")
    common = {"checkpoint_sha256": ckpt,
              "frozen_json_sha256": test_run["frozen_sha256"],
              "train_run_id": frozen["run_id"]}
    rows = []
    for locator, split, run, device, files, note in (
            ("Table 1, four tracks", "test", test_run, "RTX 3060 (FROZEN.env)",
             "results/final/stats.json; results/final/perq_*.jsonl",
             "code_sha256 6f0bcff6; evaluation process env not recorded"),
            ("Figure 1 (track CI)", "test", test_run, "RTX 3060 (FROZEN.env)",
             "results/figures/final_tracks.png", "same run as Table 1"),
            ("Figure 2 (rank stability)", "test", test_run, "RTX 3060 (FROZEN.env)",
             "results/figures/final_rank.png", "same run as Table 1"),
            ("Table 5.1, four tracks", "test2", test2_run, "NVIDIA RTX A5000",
             "results/final_test2/stats.json; results/final_test2/perq_*.jsonl",
             "code_sha256 da6a039b"),
            ("Table 5.1, primary/secondary", "test2", test2_run, "NVIDIA RTX A5000",
             "results/test2/OUTCOME.json", "pre-registered decision matrix"),
            ("T06 full fine-tune", "dev", {"attempt": "", "split": "dev"}, "NVIDIA RTX A5000",
             "models/reranker/COMPAT.json", "compatibility run; writes COMPAT.json only"),
            ("T07 matched-seed lr grid", "dev", {"attempt": "", "split": "dev"},
             "NVIDIA RTX A5000", "results/compat_lr/lr_grid.json", "6 runs, 3 seeds x 2 lr"),
    ):
        rows.append({
            "paper_locator": locator, "split": split,
            "run_attempt": run.get("attempt"),
            "run_id": eval_ids.get(split, run.get("run_id", "")),
            "device": device, "result_files": files, "note": note, **common,
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def main() -> int:
    write_csv(ROOT / "results/results.csv", results_table())
    write_csv(ROOT / "results/experiment_log.csv", experiment_log())
    write_csv(ROOT / "results/artifact_map.csv", artifact_map())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
