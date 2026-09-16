#!/usr/bin/env python3
"""Read completed branch diagnostics; never select a finite-difference reference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(root: Path, point: int = 0):
    final = json.loads((root / "FINAL.json").read_text())
    if final["status"] != "REDSHIFT_DECOMPOSITION_COMPLETE":
        raise ValueError("redshift decomposition has no complete receipt")
    for name, artifact in final["artifacts"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError(f"changed artifact: {name}")
    report = json.loads((root / "REDSHIFT_DECOMPOSITION.json").read_text())
    print(report["status"])
    print(report["precision_scope"])
    for entry in report["points"]:
        print(
            f"point={entry['point_index']} physical_z={entry['physical_z']:.8g} dz/dx={entry['dz_dx']:.8g}"
        )
        print(
            "  max |sum branch AD - full AD| in sigma/z:",
            np.max(np.abs(entry["chain_minus_full_ad_sigma"])),
        )
        print(
            "  max |branch center - native center| in sigma:",
            {
                k: float(np.max(np.abs(v)))
                for k, v in entry["center_delta_sigma"].items()
            },
        )
        print("  precision traces:", entry["floating_trace_dtypes"])
    df = pd.read_csv(root / "redshift_decomposition.csv")
    selected = df[df.point_index == point].copy()
    if selected.empty:
        raise ValueError("requested point absent")
    print(f"\nPoint {point}: finite rows {selected.finite.sum()}/{len(selected)}")
    selected["abs_derivative_delta"] = abs(selected.fd - selected.ad)
    flux = selected[~selected.branch.str.startswith("age_mass")]
    table = (
        flux.groupby(["branch", "equivalent_x_step", "physical_z_step"])
        .agg(
            loglike_AD=("loglike_ad_contribution", "sum"),
            loglike_FD=("loglike_centered_fd_contribution", "sum"),
            max_flux_AD_FD_sigma=("abs_derivative_delta", "max"),
        )
        .reset_index()
    )
    print(table.to_string(index=False))
    age = selected[selected.branch.str.startswith("age_mass")]
    print(
        "\nAge/mass weights, errors normalized by fixed target stellar mass (NOT photometric sigma):"
    )
    print(
        age.groupby(["branch", "equivalent_x_step"])
        .agg(max_AD_FD=("abs_derivative_delta", "max"))
        .to_string()
    )
    print("No posterior validation or training authorized by this diagnostic.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--point", type=int, default=0)
    args = parser.parse_args()
    summarize(args.root, args.point)
