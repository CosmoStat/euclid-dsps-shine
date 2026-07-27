#!/usr/bin/env python3
"""Join existing closure photometry with exact spline-15D truth columns."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir, write_json
from euclid_dsps.prior_learning.spline15d import SPLINE15D_PARAMETER_NAMES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--spline-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_catalogs(
    source_dir: Path,
    spline_dir: Path,
    out_dir: Path,
    *,
    splits: tuple[str, ...],
    limit: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Build photometry plus exact-spline catalogs without regenerating Diffsky."""
    out_dir = ensure_dir(out_dir)
    records: dict[str, dict] = {}
    truth_columns = list(SPLINE15D_PARAMETER_NAMES)
    for split in splits:
        source_path = source_dir / f"{split}.parquet"
        spline_path = spline_dir / f"{split}_exact.parquet"
        output_path = out_dir / f"{split}.parquet"
        for path in (source_path, spline_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {output_path}")

        source = pd.read_parquet(source_path)
        spline = pd.read_parquet(spline_path, columns=["object_id", *truth_columns])
        if (
            source["object_id"].duplicated().any()
            or spline["object_id"].duplicated().any()
        ):
            raise ValueError(f"Duplicate object_id in {split}")
        if limit is not None:
            source = source.head(max(int(limit), 0))
            wanted = set(source["object_id"].tolist())
            spline = spline[spline["object_id"].isin(wanted)]
        source_ids = set(source["object_id"].tolist())
        spline_ids = set(spline["object_id"].tolist())
        if source_ids != spline_ids:
            raise ValueError(
                f"object_id contract mismatch for {split}: "
                f"source_only={len(source_ids - spline_ids)}, "
                f"spline_only={len(spline_ids - source_ids)}"
            )
        source = source.drop(columns=truth_columns, errors="ignore")
        joined = source.merge(spline, on="object_id", how="left", validate="one_to_one")
        if not np.isfinite(joined[truth_columns].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite join result for {split}")
        joined.to_parquet(output_path, index=False)
        records[split] = {
            "source": str(source_path),
            "spline_truth": str(spline_path),
            "output": str(output_path),
            "rows": int(len(joined)),
        }
        print(f"[spline15d-amortized] {split}: rows={len(joined)}", flush=True)

    contract = {
        "version": 1,
        "regenerates_diffsky": False,
        "join_key": "object_id",
        "truth_kind": "exact_spline15d",
        "parameter_names": list(SPLINE15D_PARAMETER_NAMES),
        "source_dataset_dir": str(source_dir),
        "spline_dataset_dir": str(spline_dir),
        "output_dataset_dir": str(out_dir),
        "splits": records,
    }
    write_json(out_dir / "amortized_catalog_contract.json", contract)
    return contract


def main() -> None:
    args = _parse_args()
    build_catalogs(
        args.source_dir,
        args.spline_dir,
        args.out,
        splits=tuple(args.splits),
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
