#!/usr/bin/env python3
"""Validate and print a completed frozen adaptive-SMC E-step pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.pilot / "pilot_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "DIAGNOSTIC_COMPLETE":
        raise ValueError("pilot did not complete its diagnostic contract")
    if receipt.get("truth_used_for_training_or_selection") is not False:
        raise ValueError("pilot no-truth contract is not proven")
    if receipt.get("q_updates_applied") != 0:
        raise ValueError("frozen pilot unexpectedly updated q")
    if receipt.get("prior_updates_applied") != 0:
        raise ValueError("frozen pilot unexpectedly updated the prior")
    for relative in (
        "DONE",
        "e_step_pilot_log.csv",
        "hard_object_queue.csv",
        "selection_gradient_preflight.json",
    ):
        path = args.pilot / relative
        if not path.is_file():
            raise FileNotFoundError(path)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
