#!/usr/bin/env python3
"""Evaluate only native-DSPS redshift inference on public COSMOS spectroscopy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def redshift_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    truth = pd.to_numeric(frame["redshift_true"], errors="coerce").to_numpy(float)
    median = pd.to_numeric(frame["z_obs_median"], errors="coerce").to_numpy(float)
    q16 = pd.to_numeric(frame["z_obs_q16"], errors="coerce").to_numpy(float)
    q84 = pd.to_numeric(frame["z_obs_q84"], errors="coerce").to_numpy(float)
    valid = np.isfinite(truth) & (truth >= 0.0) & np.isfinite(median)
    if not valid.any():
        return {
            "n_spec": 0,
            "median_bias": None,
            "nmad": None,
            "rmse": None,
            "outlier_fraction_0p15": None,
            "coverage_68": None,
            "median_interval_width_68": None,
        }
    dz = (median[valid] - truth[valid]) / (1.0 + truth[valid])
    center = float(np.median(dz))
    interval = np.isfinite(q16[valid]) & np.isfinite(q84[valid])
    covered = (
        (truth[valid][interval] >= q16[valid][interval])
        & (truth[valid][interval] <= q84[valid][interval])
    )
    widths = q84[valid][interval] - q16[valid][interval]
    return {
        "n_spec": int(valid.sum()),
        "median_bias": center,
        "nmad": float(1.48 * np.median(np.abs(dz - center))),
        "rmse": float(np.sqrt(np.mean(dz**2))),
        "outlier_fraction_0p15": float(np.mean(np.abs(dz) > 0.15)),
        "coverage_68": float(np.mean(covered)) if covered.size else None,
        "median_interval_width_68": (
            float(np.median(widths)) if widths.size else None
        ),
    }


def _write_plots(frame: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    spec = frame.loc[frame["has_public_specz"]].copy()
    if spec.empty:
        return
    truth = spec["redshift_true"].to_numpy(float)
    median = spec["z_obs_median"].to_numpy(float)
    q16 = spec["z_obs_q16"].to_numpy(float)
    q84 = spec["z_obs_q84"].to_numpy(float)
    valid = np.isfinite(truth) & np.isfinite(median)
    if not valid.any():
        return
    truth = truth[valid]
    median = median[valid]
    q16 = q16[valid]
    q84 = q84[valid]
    lower = np.maximum(median - q16, 0.0)
    upper = np.maximum(q84 - median, 0.0)

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    ax.errorbar(
        truth,
        median,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        ms=3,
        alpha=0.45,
        elinewidth=0.6,
    )
    upper_limit = float(max(np.nanmax(truth), np.nanmax(median), 0.1))
    ax.plot([0.0, upper_limit], [0.0, upper_limit], color="black", lw=1.0)
    ax.set_xlabel("public spectroscopic redshift")
    ax.set_ylabel("native 15D posterior redshift")
    ax.set_title("COSMOS2020 redshift inference")
    fig.tight_layout()
    fig.savefig(out / "redshift_truth_vs_posterior.png", dpi=180)
    plt.close(fig)

    dz = (median - truth) / (1.0 + truth)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(dz[np.isfinite(dz)], bins=40)
    ax.axvline(0.0, color="black", lw=1.0)
    ax.axvline(-0.15, color="0.5", lw=1.0, ls="--")
    ax.axvline(0.15, color="0.5", lw=1.0, ls="--")
    ax.set_xlabel("(z posterior median - z spec) / (1 + z spec)")
    ax.set_ylabel("object count")
    fig.tight_layout()
    fig.savefig(out / "redshift_normalized_error.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = args.out or args.inference
    out.mkdir(parents=True, exist_ok=True)

    posterior_path = args.inference / "posterior_summary.parquet"
    posterior = pd.read_parquet(
        posterior_path,
        columns=[
            "object_id",
            "row_index",
            "z_obs_q16",
            "z_obs_median",
            "z_obs_q84",
        ],
    )
    available = set(pq.ParquetFile(args.dataset).schema.names)
    truth_columns = ["object_id", "redshift_true"]
    truth_columns.extend(
        name
        for name in (
            "redshift_spec",
            "specz_confidence_level",
            "specz_survey",
            "t24_specz_flag",
        )
        if name in available
    )
    truth = pd.read_parquet(args.dataset, columns=truth_columns)
    merged = posterior.merge(truth, on="object_id", how="left", validate="one_to_one")
    if len(merged) != len(posterior):
        raise RuntimeError("Redshift evaluation changed the inference row count")
    merged["redshift_true"] = pd.to_numeric(
        merged["redshift_true"], errors="coerce"
    )
    merged["has_public_specz"] = (
        np.isfinite(merged["redshift_true"].to_numpy(float))
        & (merged["redshift_true"].to_numpy(float) >= 0.0)
    )
    merged["normalized_redshift_error"] = (
        merged["z_obs_median"] - merged["redshift_true"]
    ) / (1.0 + merged["redshift_true"])
    merged.to_parquet(out / "redshift_predictions.parquet", index=False)

    metrics = redshift_metrics(merged)
    pd.DataFrame([metrics]).to_csv(out / "photoz_metrics.csv", index=False)
    payload = {
        "status": "complete",
        "science_target": "z_obs_only",
        "latent_model": "native_feniks_spline15d_with_14_nuisance_coordinates",
        "dataset": str(args.dataset),
        "posterior_summary": str(posterior_path),
        "n_inference": int(len(posterior)),
        "metrics": metrics,
        "truth": {
            "column": "redshift_true",
            "semantics": (
                "COSMOS DR1.1 public spectroscopy with Confidence_level >= 50, "
                "joined by Farmer object_id"
            ),
        },
        "excluded_comparisons": [
            "A24 physical latent parameters",
            "A24 marginal posterior medians",
        ],
    }
    (out / "redshift_metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_plots(merged, out)
    print(
        "[cosmos-redshift] "
        f"inference={len(posterior)} specz={metrics['n_spec']} -> {out}"
    )


if __name__ == "__main__":
    main()
