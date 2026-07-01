"""Small MAP-under-learned-prior sweeps for Diffsky diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir, write_json

from .map_adam import run_map_adam_under_prior


def run_map_prior_weight_sweep(
    config: dict[str, Any],
    out_dir: str | Path,
    *,
    checkpoint: str | Path,
    feature_stats_path: str | Path | None = None,
    weights: tuple[float, ...] = (0.0, 0.03, 0.1, 0.3, 1.0),
    row_indices_file: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 128,
    n_starts: int = 8,
    maxiter: int = 160,
    learning_rate: float = 0.015,
    start_mode: str = "mixed",
    start_chunk_size: int = 1,
    selection_mode: str | None = None,
    stratified_strategy: str | None = None,
    selection_seed: int | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the same selected galaxies under several learned-prior strengths."""
    out = ensure_dir(out_dir)
    summary_rows = []
    run_dirs = []
    for weight in weights:
        label = _weight_label(weight)
        run_dir = out / label
        run_dirs.append(run_dir)
        if verbose:
            print(f"[map-sweep] prior_weight={weight:g} -> {run_dir}")
        run_map_adam_under_prior(
            config,
            run_dir,
            checkpoint=Path(checkpoint),
            feature_stats_path=Path(feature_stats_path) if feature_stats_path else None,
            limit=limit,
            batch_size=int(batch_size),
            n_starts=int(n_starts),
            maxiter=int(maxiter),
            learning_rate=float(learning_rate),
            prior_weight=float(weight),
            seed=int(seed),
            start_mode=str(start_mode),
            start_chunk_size=int(start_chunk_size),
            selection_mode=selection_mode,
            stratified_strategy=stratified_strategy,
            selection_seed=selection_seed,
            row_indices_file=row_indices_file,
            dataset_label="Diffsky HLTDS MAP learned-prior sweep",
            verbose=verbose,
        )
        estimates_path = run_dir / "map_estimates.parquet"
        if not estimates_path.exists():
            continue
        estimates = pd.read_parquet(estimates_path)
        truth_path = run_dir / "inference_truth.parquet"
        truth = pd.read_parquet(truth_path) if truth_path.exists() else pd.DataFrame()
        row = _summary_row(estimates, truth, weight=float(weight), run_dir=run_dir)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary.to_csv(out / "map_prior_weight_sweep_summary.csv", index=False)
        _write_sweep_plots(out, summary)
    payload = {
        "checkpoint": str(checkpoint),
        "feature_stats_path": str(feature_stats_path) if feature_stats_path else None,
        "weights": [float(weight) for weight in weights],
        "limit": limit,
        "row_indices_file": str(row_indices_file) if row_indices_file else None,
        "n_runs": int(len(run_dirs)),
        "run_dirs": [str(path) for path in run_dirs],
        "summary": "map_prior_weight_sweep_summary.csv",
    }
    write_json(out / "map_prior_weight_sweep_summary.json", payload)
    return payload


def _weight_label(weight: float) -> str:
    text = f"{float(weight):.4g}".replace("-", "m").replace(".", "p")
    return f"prior_weight_{text}"


def _summary_row(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    weight: float,
    run_dir: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "prior_weight": float(weight),
        "run_dir": str(run_dir),
        "n_objects": int(len(estimates)),
        "median_map_photometric_nll": _median(estimates, "map_photometric_nll"),
        "median_map_prior_logprob": _median(estimates, "map_prior_logprob"),
        "median_map_chi2": _median(estimates, "map_chi2"),
        "median_z_obs": _median(estimates, "z_obs"),
        "frac_z_within_upper_1pct": _frac_z_upper(estimates, 0.01),
        "frac_z_within_upper_5pct": _frac_z_upper(estimates, 0.05),
    }
    if not truth.empty and "redshift_true" in truth and "row_index" in estimates:
        merged = estimates.merge(
            truth[["row_index", "redshift_true"]].drop_duplicates("row_index"),
            on="row_index",
            how="inner",
        )
        if not merged.empty and "z_obs" in merged:
            dz = pd.to_numeric(merged["z_obs"], errors="coerce") - pd.to_numeric(
                merged["redshift_true"],
                errors="coerce",
            )
            row["median_delta_z"] = float(np.nanmedian(dz))
            row["rmse_delta_z"] = float(np.sqrt(np.nanmean(np.asarray(dz) ** 2)))
            row["frac_abs_delta_z_gt_0p05"] = float(np.nanmean(np.abs(dz) > 0.05))
    return row


def _median(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    return float(np.nanmedian(pd.to_numeric(frame[column], errors="coerce")))


def _frac_z_upper(frame: pd.DataFrame, fraction: float) -> float:
    if "z_obs" not in frame:
        return float("nan")
    z = pd.to_numeric(frame["z_obs"], errors="coerce").to_numpy(float)
    finite = np.isfinite(z)
    if not finite.any():
        return float("nan")
    upper = np.nanmax(z[finite])
    lower = np.nanmin(z[finite])
    threshold = upper - float(fraction) * max(upper - lower, 1.0e-12)
    return float(np.mean(z[finite] >= threshold))


def _write_sweep_plots(out: Path, summary: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x = summary["prior_weight"].to_numpy(float)
    for column, ylabel, filename in (
        ("median_delta_z", "median z_MAP - z_true", "z_bias_vs_prior_weight.png"),
        ("median_map_photometric_nll", "median photometric NLL", "nll_vs_prior_weight.png"),
        ("frac_z_within_upper_5pct", "fraction z near upper 5%", "z_upper_fraction_vs_prior_weight.png"),
    ):
        if column not in summary:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(x, summary[column].to_numpy(float), marker="o")
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlabel("prior_weight")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)
