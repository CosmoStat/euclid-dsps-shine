#!/usr/bin/env python3
"""Evaluate MIRA from held-out FENIKS encoder-posterior parquet samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD", "0")

from euclid_dsps.amortized.mira import (  # noqa: E402
    evaluate_feniks_mira,
    parse_posterior_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute MIRA scores from inference_truth.parquet and one or more "
            "encoder posterior_samples parquet sources."
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
            "compare models with shared random regions."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-regions", type=int, default=100)
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
    summary = evaluate_feniks_mira(
        truth_path=args.truth,
        posterior_specs=posterior_specs,
        out_dir=args.out,
        num_regions=args.num_regions,
        num_bootstrap=args.num_bootstrap,
        samples_per_object=samples_per_object,
        seed=args.seed,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[mira] complete: {args.out / 'DONE'}")


if __name__ == "__main__":
    main()
