#!/usr/bin/env python3
"""Prepare a fixed 4-8-object NUTS cohort after SC-ASMC-EM freeze."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from euclid_dsps.amortized.sc_asmc_postfreeze import (  # noqa: E402
    prepare_postfreeze_nuts_cohort,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=range(4, 9), default=8)
    parser.add_argument(
        "--rows",
        help="Optional comma-separated frozen-bank row indices; must match --count.",
    )
    args = parser.parse_args()
    rows = None if args.rows is None else [int(value) for value in args.rows.split(",")]
    result = prepare_postfreeze_nuts_cohort(
        args.run_root,
        args.out,
        count=args.count,
        row_indices=rows,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
