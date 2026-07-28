#!/usr/bin/env python3
"""Prepare deterministic A24-like COSMOS2020 Farmer parquet subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

from euclid_dsps.cosmos2020 import (
    DEFAULT_SUBSET_SIZES,
    prepare_farmer_catalog,
    read_farmer_table,
    sha256_file,
    write_json,
    write_nested_subsets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--sizes",
        default=",".join(str(value) for value in DEFAULT_SUBSET_SIZES),
    )
    parser.add_argument("--seed", type=int, default=260727)
    parser.add_argument(
        "--expected-selected",
        type=int,
        default=None,
        help="Fail unless the selected sample has this exact cardinality.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(",") if value.strip())
    frame = read_farmer_table(args.input)
    selected, manifest = prepare_farmer_catalog(frame)
    if (
        args.expected_selected is not None
        and len(selected) != args.expected_selected
    ):
        raise SystemExit(
            f"Expected {args.expected_selected} selected rows, got {len(selected)}"
        )
    subsets = write_nested_subsets(selected, args.out, sizes=sizes, seed=args.seed)
    manifest.update(
        {
            "input": str(args.input),
            "input_sha256": sha256_file(args.input),
            "subsets": subsets,
            "expected_selected": args.expected_selected,
        }
    )
    write_json(args.out / "preparation_manifest.json", manifest)
    print(f"[cosmos2020] selected {len(selected)} / {len(frame)} rows")
    print(f"[cosmos2020] manifest -> {args.out / 'preparation_manifest.json'}")


if __name__ == "__main__":
    main()
