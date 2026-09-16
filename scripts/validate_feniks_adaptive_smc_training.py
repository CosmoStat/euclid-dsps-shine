#!/usr/bin/env python3
"""Fail closed unless a FENIKS adaptive-SMC training receipt passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--expect-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt_path = args.train / "training_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if bool(receipt.get("smoke")) != bool(args.expect_smoke):
        raise ValueError("training receipt smoke contract mismatch")
    if receipt.get("truth_used_for_training_or_selection") is not False:
        raise ValueError("training receipt does not prove the no-truth contract")
    failed = [name for name, value in receipt.get("checks", {}).items() if not value]
    if receipt.get("status") != "PASS" or failed:
        raise SystemExit(
            "adaptive-SMC training gate failed: " + ", ".join(failed or ["status"])
        )
    for relative in (
        "checkpoints/last.eqx",
        "checkpoints/best.eqx",
        "checkpoints/bootstrap.eqx",
        "checkpoints/training_state_last.eqx",
        "adaptive_training_log.csv",
        "adaptive_validation_log.csv",
        "prior_macro_log.csv",
        "hard_object_queue.csv",
        "training_progress.json",
        "selection_gradient_preflight.json",
    ):
        path = args.train / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
