#!/usr/bin/env python3
"""Fail-closed receipt for the four independent SC-DRWS inference shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--full-pass", type=Path, required=True)
    args = parser.parse_args()
    full = json.loads(args.full_pass.read_text())
    shards = []
    for index in range(4):
        root = args.root / f"shard_{index}"
        summary = root / "inference_summary.json"
        if not (root / "DONE").is_file() or not summary.is_file():
            raise SystemExit(f"incomplete inference shard {index}")
        shards.append(json.loads(summary.read_text()))
    payload = {
        "status": "PASS",
        "workflow": "SC-DRWS full selected-catalogue inference",
        "checkpoint": full["selected_checkpoint"],
        "shards": 4,
        "truth_used": False,
        "report_artifacts": {
            "posterior_joint_draws": "shard_*/posterior_samples/*.parquet",
            "photometric_ppc": "shard_*/posterior_predictive_*.parquet",
            "parent_and_selected_prior": "derive beta-weighted selected draws from the single learned parent prior",
            "closure": "run only after this frozen receipt",
        },
    }
    (args.root / "inference_receipt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (args.root / "DONE").touch()
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
