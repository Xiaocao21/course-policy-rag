"""train.py - retrain the reranker in an isolated root (R3, method check).

`18_train_reranker.py` writes `FROZEN.json`/`COMPAT.json` under `<root>/models/reranker/`. To
protect the official metadata, this entry builds a temporary root where `models/reranker` is a
real directory (the base model and the training pairs are linked, not copied), runs the pair
builder and the trainer there, and copies the resulting run directory back to
`results/repro/train/<tag>/`. The official `FROZEN.json`/`COMPAT.json` are hashed before and after.

The A5000 recipe is `configs/a5000.yaml`; it is a compatibility run (`env_role: compatibility`),
so it writes `COMPAT.json` and never replaces the canonical frozen model.

Usage
    python src/train.py --config configs/a5000.yaml --dry-run
    python src/train.py --config configs/a5000.yaml --smoke
    python src/train.py --config configs/a5000.yaml --full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config as cfg  # noqa: E402

LINKS = ("scripts", "config", "annotations", "indexes", "chunks", "cleaned",
         "corpus_safe_512", "reports")
GUARDED = ("models/reranker/FROZEN.json", "models/reranker/COMPAT.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def link_dir(src: Path, dst: Path) -> None:
    if os.name == "nt":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"REFUSING: cannot link {dst}: {result.stderr or result.stdout}")
    else:
        os.symlink(src, dst, target_is_directory=True)


def link_file(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(root: Path, work: Path) -> list[Path]:
    work.mkdir(parents=True, exist_ok=True)
    for name in LINKS:
        if (root / name).exists():
            link_dir(root / name, work / name)
    for name in ("sources.csv", "MANIFEST.json"):
        if (root / name).exists():
            link_file(root / name, work / name)
    reranker = work / "models" / "reranker"
    reranker.mkdir(parents=True)
    links = []
    base = root / "models" / "reranker" / "base"
    if base.exists():
        link_dir(base, reranker / "base")
        links.append(reranker / "base")
    # copy the pairs: 17/18 rewrite this file, and a hard link would modify the repository copy
    pairs = root / "models" / "reranker" / "pairs_dev.jsonl"
    if pairs.exists():
        shutil.copy2(pairs, reranker / "pairs_dev.jsonl")
    return links


def cleanup(work: Path, links: list[Path]) -> None:
    for path in links:
        try:
            if path.is_symlink():
                path.unlink()
            elif getattr(path, "is_junction", lambda: False)():
                os.rmdir(path)
        except OSError:
            pass
    shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "a5000.yaml"))
    ap.add_argument("--tag", default="a5000_repro")
    ap.add_argument("--smoke", action="store_true", help="one tiny run, no FROZEN write")
    ap.add_argument("--full", action="store_true", help="full recipe from the config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python", default="")
    args = ap.parse_args()
    conf = cfg.load(Path(args.config))
    python = Path(args.python) if args.python else None
    if python is None:
        gpu = ROOT / ".venv-gpu" / "Scripts" / "python.exe"
        python = gpu if gpu.exists() else Path(sys.executable)

    work = Path(tempfile.mkdtemp(prefix="cpr-train-"))
    links = build(ROOT, work)
    out = ROOT / "results" / "repro" / "train" / args.tag
    neg = conf.get("negatives_per_positive", 4)
    lr = conf.get("learning_rate", 1e-05)
    train_cmd = [str(python), str(ROOT / "scripts" / "18_train_reranker.py"),
                 "--root", str(work), "--device", conf.get("device", "cuda"),
                 "--freeze-encoder-layers", str(conf.get("freeze_encoder_layers", 0)),
                 "--optimizer", conf.get("optimizer", "adamw"),
                 "--batch", str(conf.get("batch_size", 32)),
                 "--grad-accum", str(conf.get("grad_accum", 1)),
                 "--epochs-max", str(conf.get("epochs_max", 3)),
                 "--seed", str(conf.get("seed", 20261027)),
                 "--env-role", conf.get("env_role", "compatibility"),
                 "--grid", f"{lr:g}:{neg}", "--run-tag", "repro"]
    if conf.get("base_model"):
        train_cmd += ["--model-path", f"models/reranker/base"]
    if args.smoke:
        train_cmd += ["--smoke", "--max-steps", "1"]
    pair_cmd = [str(python), str(ROOT / "scripts" / "17_build_rerank_data.py"),
                "--root", str(work), "--split", conf.get("split", "dev")]
    print("$ " + " ".join(pair_cmd[:4]) + " ...")
    print("$ " + " ".join(train_cmd[:6]) + " ...")
    print(f"staging root: {work}\noutput: {out}")
    if args.dry_run:
        cleanup(work, links)
        return 0

    before = {rel: sha256(ROOT / rel) for rel in GUARDED}
    subprocess.run(pair_cmd, cwd=str(ROOT), check=True)
    subprocess.run(train_cmd, cwd=str(ROOT), check=True)

    out.mkdir(parents=True, exist_ok=True)
    for name in ("COMPAT.json", "train.log", "data_stats.json", "split_groups.json"):
        src = work / "models" / "reranker" / name
        if src.exists():
            shutil.copy2(src, out / name)
    runs = work / "models" / "reranker" / "runs"
    if runs.exists():
        for run in runs.iterdir():
            if "repro" in run.name:
                shutil.copytree(run, out / "runs" / run.name, dirs_exist_ok=True)
    after = {rel: sha256(ROOT / rel) for rel in GUARDED}
    unchanged = before == after
    (out / "run_meta.json").write_text(json.dumps({
        "mode": "smoke" if args.smoke else "full", "config": args.config,
        "official_hashes_unchanged": unchanged, "staging_root": str(work),
    }, indent=1), encoding="utf-8")
    cleanup(work, links)
    print(f"wrote {out}; official FROZEN/COMPAT unchanged: {unchanged}")
    return 0 if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
