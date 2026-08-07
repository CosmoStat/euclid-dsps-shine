#!/usr/bin/env python3
"""Update a learned population prior from a cached joint proposal bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from euclid_dsps.amortized.posthoc_calibration import run_generalized_em


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--mstep-epochs", type=int, default=5)
    parser.add_argument("--object-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--trust-strength", type=float, default=0.05)
    parser.add_argument("--trust-samples", type=int, default=512)
    parser.add_argument("--weight-kind", choices=("raw", "psis"), default="raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--min-median-ess-fraction", type=float, default=0.01)
    parser.add_argument(
        "--max-fraction-pareto-k-gt-0p7", type=float, default=0.5
    )
    parser.add_argument("--allow-low-ess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_generalized_em(
        posterior=args.posterior,
        config_path=args.config,
        checkpoint=args.checkpoint,
        out_dir=args.out,
        iterations=args.iterations,
        mstep_epochs=args.mstep_epochs,
        object_batch_size=args.object_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        trust_strength=args.trust_strength,
        trust_samples=args.trust_samples,
        weight_kind=args.weight_kind,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        min_median_ess_fraction=args.min_median_ess_fraction,
        max_fraction_pareto_k_gt_0p7=args.max_fraction_pareto_k_gt_0p7,
        allow_low_ess=args.allow_low_ess,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
