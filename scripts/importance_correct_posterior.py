#!/usr/bin/env python3
"""Importance-correct a dense joint amortized posterior bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from euclid_dsps.amortized.posthoc_calibration import run_importance_correction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--target-checkpoint", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--truth-column", default="redshift_true")
    parser.add_argument("--resample-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior-eval-batch-size", type=int, default=65_536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_importance_correction(
        posterior=args.posterior,
        out_dir=args.out,
        config_path=args.config,
        target_checkpoint=args.target_checkpoint,
        truth_path=args.truth,
        truth_column=args.truth_column,
        resample_count=args.resample_count,
        seed=args.seed,
        prior_eval_batch_size=args.prior_eval_batch_size,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
