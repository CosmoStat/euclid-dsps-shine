#!/usr/bin/env python
"""Merge MCLMC posterior batch outputs from disjoint run directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TABLES = (
    "batch_posterior_summary",
    "batch_posterior_predictive",
    "batch_posterior_predictive_flux_residual_summary",
    "batch_mcmc_diagnostics",
    "batch_posterior_samples",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Merged output directory.")
    parser.add_argument("runs", nargs="+", help="Run directories to merge.")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = [Path(run) for run in args.runs]
    manifest_rows = []
    for stem in TABLES:
        frames = []
        for run_dir in run_dirs:
            frame = _read_table(run_dir, stem)
            if frame is None:
                continue
            frame = frame.copy()
            frame["source_run"] = str(run_dir)
            frames.append(frame)
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(out / f"{stem}.csv", index=False)
        if stem == "batch_posterior_predictive_flux_residual_summary":
            merged.to_parquet(out / f"{stem}.parquet", index=False)
        manifest_rows.append(
            {
                "table": stem,
                "rows": int(len(merged)),
                "sources": int(len(frames)),
            }
        )
        print(f"{stem}: rows={len(merged)} sources={len(frames)}")
    pd.DataFrame(manifest_rows).to_csv(out / "merge_manifest.csv", index=False)


def _read_table(run_dir: Path, stem: str) -> pd.DataFrame | None:
    csv = run_dir / f"{stem}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    parquet = run_dir / f"{stem}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    return None


if __name__ == "__main__":
    main()
