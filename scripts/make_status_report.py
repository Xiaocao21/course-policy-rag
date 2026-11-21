"""make_status_report.py - thin wrapper kept for backward compatibility.

The status summary and the supervisor progress report were separate scripts that both
composed a Word document from the same figures, which risked the two drifting apart and
printing different numbers. All content now lives in `report_content.py` and is rendered
by `make_progress_report.py`.

This entry point forwards to it. It also reports the source of the change so an existing
caller is not silently surprised by a different output path.

Requires python-docx (bundled runtime, not the project venv).

Usage:
    <bundled python> scripts/make_status_report.py
    <bundled python> scripts/make_status_report.py --out PATH
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_progress_report  # noqa: E402
import report_content  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LEGACY_OUT = Path(r"<home>\Desktop\WKUclass\高级人工智能CPS5801\work"
                  r"\数据收集与预处理_进展报告.docx")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default="",
                    help="output path; defaults to the progress report location")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(make_progress_report.DEFAULT_OUT)

    print("make_status_report.py now delegates to make_progress_report.py so that the "
          "two documents cannot report different numbers.")
    data = report_content.collect(Path(args.root))
    if not data["consistency"]["ok"]:
        print("REFUSING TO GENERATE: report figures are not self-consistent.", file=sys.stderr)
        for p in data["consistency"]["problems"]:
            print(f"  - {p}", file=sys.stderr)
        return 1

    make_progress_report.build(data, out)
    print(f"wrote {out}")
    if out != LEGACY_OUT:
        print(f"NOTE: the previous default was {LEGACY_OUT.name}; that file is no longer "
              f"written. Delete it or pass --out to keep it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
