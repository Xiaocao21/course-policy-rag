"""check_data.py - gate a corpus rebuild against the frozen v0.1.5 baseline.

The gate is deliberately layered and refuses to call a rebuild complete just because the
crawler returned *something*:

    L1  raw content   : sha256(raw.html|raw.pdf) == sources.sha256_raw, for all 36 required docs
    L2b extracted     : extracted/<doc>.json matches MANIFEST
    L2c cleaned       : cleaned/<doc>.{evidence,retrieval,charmap}.jsonl match MANIFEST
    L2d corpus        : corpus_safe_512/*.jsonl match MANIFEST

Not gated, with reasons: `raw/*/meta.json` (contains retrieval timestamps), 
`cleaned/removed_lines.log` (global log over all 265 collected docs), `annotations/*` (not a
corpus-rebuild product; MANIFEST's annotation hashes predate the question repair).

`reproduction_grade` is written to data/STATUS.json ONLY when L1 and all L2 layers pass.

Usage
    python tools/check_data.py --status
    python tools/check_data.py --require --work <rebuild-root>
    python tools/check_data.py --verify-layers --work <rebuild-root>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ROOT / "data" / "REQUIRED_SOURCES.csv"
STATUS = ROOT / "data" / "STATUS.json"
INCOMPLETE = ROOT / "data" / "INCOMPLETE.json"
REPORT = ROOT / "data" / "REBUILD_REPORT.md"
CLEANED_FILES = ("evidence", "retrieval", "charmap")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_rows() -> list[dict]:
    with REQUIRED.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def raw_content(work: Path, row: dict) -> Path | None:
    base = work / "raw" / row["tier"] / row["doc_id"]
    for name in ("raw.html", "raw.pdf"):
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def check_l1(work: Path) -> tuple[list[dict], list[dict], list[dict]]:
    ok, unavailable, changed = [], [], []
    for row in required_rows():
        content = raw_content(work, row)
        if content is None:
            unavailable.append({"doc_id": row["doc_id"], "reason": "raw content missing"})
            continue
        got = sha256(content)
        if got == row["sha256_raw"]:
            ok.append({"doc_id": row["doc_id"]})
        else:
            changed.append({"doc_id": row["doc_id"], "expected": row["sha256_raw"],
                            "actual": got, "expected_bytes": int(row["bytes"] or 0),
                            "actual_bytes": content.stat().st_size})
    return ok, unavailable, changed


def check_l2(work: Path, layers: tuple[str, ...] = ("extracted", "cleaned", "corpus_safe_512")) -> dict:
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    hashes = manifest["artefact_hashes"]
    rows = required_rows()
    expected: dict[str, set[str]] = {"extracted": set(), "cleaned": set(),
                                     "corpus_safe_512": set()}
    for row in rows:
        doc = row["doc_id"]
        expected["extracted"].add(f"extracted/{doc}.json")
        for kind in CLEANED_FILES:
            expected["cleaned"].add(f"cleaned/{doc}.{kind}.jsonl")
    expected["corpus_safe_512"] = set(hashes["corpus_safe_512"])
    report = {"layers": {}, "missing": [], "mismatched": []}
    for layer in layers:
        paths = expected[layer]
        passed = 0
        for rel in sorted(paths):
            want = hashes[layer].get(rel, "")
            target = work / rel
            if not target.exists():
                report["missing"].append(rel)
                continue
            if sha256(target) != want:
                report["mismatched"].append(rel)
                continue
            passed += 1
        report["layers"][layer] = {"expected": len(paths), "matched": passed}
    return report


def write_status(ok: bool, l1: dict, l2: dict | None, reasons: list[str],
                 basis: str = "live_crawl") -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({
        "corpus_version": "v0.1.5",
        "reproduction_grade": ok,
        "basis": basis,
        "required_documents": 36,
        "l1": l1,
        "l2": l2,
        "reasons": reasons,
        "note": ("reproduction_grade=true only when all 36 required raw documents match "
                 "sha256_raw and the extracted/cleaned/corpus layers match MANIFEST"),
    }, indent=1), encoding="utf-8")


def write_report(l1: dict, l2: dict | None) -> None:
    lines = ["# Corpus rebuild report (v0.1.5)", "",
             f"required documents: 36  ·  raw matched: {l1['matched']}  ·  "
             f"unavailable: {len(l1['unavailable'])}  ·  content changed: {len(l1['changed'])}",
             ""]
    if l1["unavailable"]:
        lines += ["## Unavailable", ""]
        lines += [f"- `{x['doc_id']}`: {x['reason']}" for x in l1["unavailable"]]
        lines.append("")
    if l1["changed"]:
        lines += ["## Content changed (hash drift)", "",
                  "| doc_id | expected bytes | actual bytes |", "|---|---|---|"]
        lines += [f"| `{x['doc_id']}` | {x['expected_bytes']} | {x['actual_bytes']} |"
                  for x in l1["changed"]]
        lines.append("")
    if l2:
        lines += ["## Derived layers", "", "| layer | expected | matched |", "|---|---|---|"]
        lines += [f"| {k} | {v['expected']} | {v['matched']} |" for k, v in l2["layers"].items()]
        lines += ["", f"missing files: {len(l2['missing'])}  ·  "
                      f"mismatched files: {len(l2['mismatched'])}"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="", help="rebuild root produced by 01/02/03")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--require", action="store_true", help="run the L1 gate")
    ap.add_argument("--verify-layers", action="store_true", help="run the L2 gates")
    ap.add_argument("--snapshot", action="store_true",
                    help="verify the author-provided snapshot as the corpus basis")
    args = ap.parse_args()

    if args.snapshot:
        snap = ROOT / "data" / "SNAPSHOT_STATUS.json"
        if not snap.exists() or not json.loads(snap.read_text(encoding="utf-8")).get("verified"):
            print("REFUSING: run tools/fetch_data.py --snapshot ... --verify first", file=sys.stderr)
            return 2
        l2 = check_l2(ROOT, ("cleaned", "corpus_safe_512"))
        ok = all(v["matched"] == v["expected"] for v in l2["layers"].values())
        write_status(ok, {"matched": 36, "unavailable": [], "changed": [],
                          "basis": "author_snapshot"}, l2,
                     [] if ok else ["snapshot layers do not match MANIFEST"],
                     basis="author_snapshot")
        print("snapshot layers: " + ", ".join(f"{k} {v['matched']}/{v['expected']}"
                                                for k, v in l2["layers"].items()))
        return 0 if ok else 1
    if args.status:
        rows = required_rows()
        print(f"required documents: {len(rows)}")
        for name in ("raw", "extracted", "cleaned", "corpus_safe_512"):
            path = ROOT / name
            print(f"  {name}: {'present' if path.exists() else 'missing'}")
        print(f"  STATUS.json: {STATUS.read_text(encoding='utf-8') if STATUS.exists() else 'absent'}")
        return 0
    if not args.work:
        print("REFUSING: --work is required for --require/--verify-layers", file=sys.stderr)
        return 2
    work = Path(args.work)

    ok, unavailable, changed = check_l1(work)
    l1 = {"matched": len(ok), "unavailable": unavailable, "changed": changed}
    print(f"L1 raw: matched {len(ok)}/{len(ok) + len(unavailable) + len(changed)} "
          f"(unavailable {len(unavailable)}, changed {len(changed)})")
    l1_pass = not unavailable and not changed and len(ok) == len(required_rows())

    l2 = None
    l2_pass = False
    if args.verify_layers and l1_pass:
        l2 = check_l2(work)
        l2_pass = all(v["matched"] == v["expected"] for v in l2["layers"].values())
        print("L2 layers: " + ", ".join(f"{k} {v['matched']}/{v['expected']}"
                                        for k, v in l2["layers"].items()))
    reasons = []
    if unavailable:
        reasons.append(f"{len(unavailable)} required sources unavailable")
    if changed:
        reasons.append(f"{len(changed)} required sources changed content")
    if l2 is not None and not l2_pass:
        reasons.append("derived layers do not match MANIFEST")
    if l2 is None:
        reasons.append("derived layers not verified")
    write_status(l1_pass and l2_pass, l1, l2, reasons)
    write_report(l1, l2)
    if not l1_pass:
        INCOMPLETE.parent.mkdir(parents=True, exist_ok=True)
        INCOMPLETE.write_text(json.dumps({"unavailable": unavailable, "changed": changed},
                                         indent=1), encoding="utf-8")
        print("INCOMPLETE: reproduction_grade=false", file=sys.stderr)
        return 1
    return 0 if l2_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
