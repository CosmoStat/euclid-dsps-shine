#!/usr/bin/env python3
"""Select a deterministic observed-property cohort for exact COSMOS posteriors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _nearest_unused(values: np.ndarray, target: float, used: set[int]) -> int:
    order = np.argsort(np.abs(values - target))
    for index in order:
        if int(index) not in used and np.isfinite(values[index]):
            used.add(int(index))
            return int(index)
    raise ValueError("Could not select a distinct finite cohort row")


def main() -> None:
    args = parse_args()
    columns = [
        "row_index",
        "object_id",
        "r_abmag_mw_corrected",
        "lp_zbest",
        "flux_hsc_r",
        "fluxerr_hsc_r",
    ]
    frame = pd.read_parquet(args.dataset, columns=columns)
    rmag = frame["r_abmag_mw_corrected"].to_numpy(float)
    z = frame["lp_zbest"].to_numpy(float)
    snr = frame["flux_hsc_r"].to_numpy(float) / frame["fluxerr_hsc_r"].to_numpy(
        float
    )
    used: set[int] = set()
    targets = (
        ("typical", rmag, np.nanmedian(rmag)),
        ("bright", rmag, np.nanquantile(rmag, 0.05)),
        ("faint", rmag, np.nanquantile(rmag, 0.95)),
        ("low_photoz", z, np.nanquantile(z, 0.10)),
        ("high_photoz", z, np.nanquantile(z, 0.90)),
        ("high_snr", snr, np.nanquantile(snr, 0.98)),
    )
    rows = []
    for order, (label, values, target) in enumerate(targets, start=1):
        local = _nearest_unused(values, target, used)
        item = frame.iloc[local]
        rows.append(
            {
                "order": order,
                "example_key": label,
                "row_index": int(item["row_index"]),
                "object_id": int(item["object_id"]),
                "lp_zbest_reference_only": float(item["lp_zbest"]),
                "r_abmag": float(item["r_abmag_mw_corrected"]),
                "hsc_r_snr": float(snr[local]),
            }
        )
    output = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"[cosmos-mclmc] cohort -> {args.out}")


if __name__ == "__main__":
    main()
