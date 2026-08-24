#!/usr/bin/env python3
"""Validate the structure and optional scientific gates of a teacher audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--require-teacher", action="store_true")
    parser.add_argument("--require-q", action="store_true")
    args = parser.parse_args()
    receipt_path = args.audit / "teacher_audit_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "status",
        "contract",
        "objects",
        "method_summaries",
        "q_only_importance",
        "checks",
        "teacher_ready",
        "q_ready",
        "next_action",
    }
    missing = sorted(required - set(receipt))
    if missing:
        raise SystemExit("teacher audit receipt is missing: " + ", ".join(missing))
    if receipt["objects"] != 8:
        raise SystemExit(f"teacher audit has {receipt['objects']} objects, expected 8")
    if not (args.audit / "DONE").is_file():
        raise SystemExit("teacher audit DONE marker is missing")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    failed = []
    if args.require_teacher and not receipt["teacher_ready"]:
        failed.append("teacher_ready")
    if args.require_q and not receipt["q_ready"]:
        failed.append("q_ready")
    if failed:
        raise SystemExit("failed required gates: " + ", ".join(failed))


if __name__ == "__main__":
    main()
