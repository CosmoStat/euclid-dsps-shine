#!/usr/bin/env python3
"""Prepare deterministic A24-like COSMOS2020 Farmer parquet subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
        "--public-summary",
        type=Path,
        help="T24 summaries.txt used to select the published same-object cohort.",
    )
    parser.add_argument(
        "--expected-selected",
        type=int,
        default=None,
        help="Fail unless the selected sample has this exact cardinality.",
    )
    return parser.parse_args()


def _public_r25_non_xray_rows(
    frame: pd.DataFrame, summary_path: Path
) -> np.ndarray:
    reference = pd.read_csv(
        summary_path,
        sep=r"\s+",
        usecols=["INDEX_COSMOS", "RA", "DEC", "XRAY", "MAGCUT_r"],
    )
    reference = reference.loc[
        (reference["MAGCUT_r"] == "Y") & (reference["XRAY"] == "N")
    ].copy()
    public_ids = pd.to_numeric(
        reference["INDEX_COSMOS"], errors="raise"
    ).to_numpy(np.int64)
    if len(np.unique(public_ids)) != len(public_ids):
        raise ValueError("T24 r<25 non-X-ray cohort has duplicate Farmer IDs")

    catalog_ids = pd.to_numeric(frame["ID"], errors="raise").to_numpy(np.int64)
    if len(np.unique(catalog_ids)) != len(catalog_ids):
        raise ValueError("Farmer v2.1 catalog contains duplicate IDs")
    row_by_id = pd.Series(np.arange(len(frame), dtype=np.int64), index=catalog_ids)
    resolved = row_by_id.reindex(public_ids)
    if resolved.isna().any():
        missing = public_ids[resolved.isna().to_numpy()]
        raise KeyError(
            f"{len(missing)} T24 INDEX_COSMOS IDs are absent from Farmer v2.1"
        )
    rows = resolved.to_numpy(np.int64)

    catalog_ra = pd.to_numeric(
        frame.iloc[rows]["ALPHA_J2000"], errors="coerce"
    ).to_numpy(float)
    catalog_dec = pd.to_numeric(
        frame.iloc[rows]["DELTA_J2000"], errors="coerce"
    ).to_numpy(float)
    reference_ra = pd.to_numeric(reference["RA"], errors="coerce").to_numpy(float)
    reference_dec = pd.to_numeric(reference["DEC"], errors="coerce").to_numpy(float)
    separation_arcsec = np.hypot(
        (catalog_ra - reference_ra) * np.cos(np.deg2rad(reference_dec)),
        catalog_dec - reference_dec,
    ) * 3600.0
    if not np.all(np.isfinite(separation_arcsec)):
        raise ValueError("Non-finite coordinate match in the public T24 cohort")
    if float(separation_arcsec.max(initial=0.0)) > 0.01:
        raise ValueError(
            "T24 INDEX_COSMOS IDs do not match Farmer v2.1 coordinates: "
            f"max separation={separation_arcsec.max():.6f} arcsec"
        )
    return rows


def main() -> None:
    args = parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(",") if value.strip())
    frame = read_farmer_table(args.input)
    public_rows = (
        _public_r25_non_xray_rows(frame, args.public_summary)
        if args.public_summary is not None
        else None
    )
    selected, manifest = prepare_farmer_catalog(
        frame, public_catalog_rows=public_rows
    )
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
