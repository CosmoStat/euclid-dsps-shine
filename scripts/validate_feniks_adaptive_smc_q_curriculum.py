#!/usr/bin/env python3
"""Validate and print a completed exact-cohort q curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(
        (args.curriculum / "curriculum_receipt.json").read_text(encoding="utf-8")
    )
    if receipt.get("status") != "DIAGNOSTIC_COMPLETE":
        raise ValueError("curriculum did not complete its diagnostic contract")
    if receipt.get("truth_used_for_training_or_selection") is not False:
        raise ValueError("curriculum no-truth contract is not proven")
    if receipt.get("prior_updates_applied") != 0:
        raise ValueError("curriculum unexpectedly updated the parent prior")
    if int(receipt.get("q_updates_applied", 0)) > 0:
        checkpoint = receipt.get("curriculum_checkpoint")
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
    for relative in (
        "DONE",
        "curriculum_e_step_log.csv",
        "q_distillation_log.csv",
        "hard_object_queue.csv",
        "selection_gradient_preflight.json",
    ):
        if not (args.curriculum / relative).is_file():
            raise FileNotFoundError(args.curriculum / relative)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
