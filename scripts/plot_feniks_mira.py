#!/usr/bin/env python3
"""Regenerate a FENIKS MIRA plot from an existing score CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from euclid_dsps.amortized.mira import write_mira_score_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate mira_scores.png from mira_scores.csv without rerunning "
            "the MIRA calculation."
        )
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = pd.read_csv(args.scores)
    required = {
        "model",
        "group",
        "num_objects",
        "num_posterior_samples",
        "num_regions",
        "score",
        "ideal_score",
        "bootstrap_mean",
        "bootstrap_std",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"MIRA score CSV is missing columns: {missing}")
    write_mira_score_plot(scores, args.out)
    print(f"[mira] plot written: {args.out}")


if __name__ == "__main__":
    main()
