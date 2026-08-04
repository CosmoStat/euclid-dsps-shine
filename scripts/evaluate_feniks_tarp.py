#!/usr/bin/env python3
"""Evaluate TARP coverage from held-out FENIKS posterior parquet samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD", "0")

from euclid_dsps.amortized.mira import parse_posterior_spec  # noqa: E402
from euclid_dsps.amortized.tarp import evaluate_feniks_tarp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute TARP expected-coverage curves from inference_truth.parquet "
            "and one or more encoder posterior parquet sources."
        )
    )
    parser.add_argument(
        "--truth",
        type=Path,
        required=True,
        help="inference_truth.parquet or its containing inference directory.",
    )
    parser.add_argument(
        "--posterior",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help=(
            "Named posterior source. PATH may be an inference directory, a "
            "posterior_samples directory, or a monolithic parquet. Repeat to "
            "compare models with shared TARP references."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--num-alpha-bins",
        type=int,
        default=0,
        help="ECP histogram bins; 0 uses the upstream default L//10.",
    )
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--samples-per-object",
        type=int,
        default=128,
        help="Use the first N sample IDs per object; pass 0 to use every sample.",
    )
    parser.add_argument("--seed", type=int, default=260730)
    parser.add_argument(
        "--limit",
        type=int,
        help="Use the first N truth objects for a smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    posterior_specs = [parse_posterior_spec(value) for value in args.posterior]
    samples_per_object = (
        None if args.samples_per_object == 0 else args.samples_per_object
    )
    num_alpha_bins = None if args.num_alpha_bins == 0 else args.num_alpha_bins
    summary = evaluate_feniks_tarp(
        truth_path=args.truth,
        posterior_specs=posterior_specs,
        out_dir=args.out,
        num_alpha_bins=num_alpha_bins,
        num_bootstrap=args.num_bootstrap,
        samples_per_object=samples_per_object,
        seed=args.seed,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[tarp] complete: {args.out / 'DONE'}")


if __name__ == "__main__":
    main()
