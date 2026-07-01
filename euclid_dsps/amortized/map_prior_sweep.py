"""Small MAP-under-learned-prior sweeps for Diffsky diagnostics."""

from __future__ import annotations

import json
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
    prior_density_space: str = "x",
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
            prior_density_space=str(prior_density_space),
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
        "prior_density_space": str(prior_density_space),
        "limit": limit,
        "row_indices_file": str(row_indices_file) if row_indices_file else None,
        "n_runs": int(len(run_dirs)),
        "run_dirs": [str(path) for path in run_dirs],
        "summary": "map_prior_weight_sweep_summary.csv",
    }
    write_json(out / "map_prior_weight_sweep_summary.json", payload)
    return payload


def finalize_map_prior_weight_sweep(
    out_dir: str | Path,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Combine sharded MAP-prior sweep outputs into per-weight tables."""
    out = ensure_dir(out_dir)
    summary_rows = []
    finalized = []
    for weight_dir in sorted(out.glob("prior_weight_*")):
        if not weight_dir.is_dir():
            continue
        shard_dirs = sorted(path for path in weight_dir.glob("shard_*") if path.is_dir())
        if not shard_dirs:
            continue
        if verbose:
            print(f"[map-sweep] finalizing {weight_dir}: {len(shard_dirs)} shards")
        estimates = _concat_optional_tables(
            shard / "map_estimates.parquet" for shard in shard_dirs
        )
        if estimates.empty:
            continue
        by_start = _concat_optional_tables(
            shard / "map_estimates_by_start.parquet" for shard in shard_dirs
        )
        trace = _concat_optional_tables(
            shard / "map_optimizer_trace.parquet" for shard in shard_dirs
        )
        truth = _concat_optional_tables(
            shard / "inference_truth.parquet" for shard in shard_dirs
        )
        estimates.to_parquet(weight_dir / "map_estimates.parquet", index=False)
        estimates.to_csv(weight_dir / "map_estimates.csv", index=False)
        if not by_start.empty:
            by_start.to_parquet(weight_dir / "map_estimates_by_start.parquet", index=False)
            by_start.to_csv(weight_dir / "map_estimates_by_start.csv", index=False)
        if not trace.empty:
            trace.to_parquet(weight_dir / "map_optimizer_trace.parquet", index=False)
        if not truth.empty:
            truth = truth.drop_duplicates("row_index")
            truth.to_parquet(weight_dir / "inference_truth.parquet", index=False)
        weight = _weight_from_shards_or_label(weight_dir, shard_dirs)
        summary_rows.append(
            _summary_row(estimates, truth, weight=float(weight), run_dir=weight_dir)
        )
        finalized.append(
            {
                "run_dir": str(weight_dir),
                "prior_weight": float(weight),
                "n_shards": int(len(shard_dirs)),
                "n_objects": int(len(estimates)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values("prior_weight").reset_index(drop=True)
        summary.to_csv(out / "map_prior_weight_sweep_summary.csv", index=False)
        _write_sweep_plots(out, summary)
    payload = {
        "out_dir": str(out),
        "n_weight_runs": int(len(finalized)),
        "finalized": finalized,
        "summary": "map_prior_weight_sweep_summary.csv",
    }
    write_json(out / "map_prior_weight_sweep_summary.json", payload)
    return payload


def _weight_label(weight: float) -> str:
    text = f"{float(weight):.4g}".replace("-", "m").replace(".", "p")
    return f"prior_weight_{text}"


def _concat_optional_tables(paths) -> pd.DataFrame:
    frames = []
    for path in paths:
        path = Path(path)
        if path.exists():
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _weight_from_shards_or_label(weight_dir: Path, shard_dirs: list[Path]) -> float:
    for shard in shard_dirs:
        summary_path = shard / "map_summary.json"
        if not summary_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            return float(payload["prior_weight"])
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    text = weight_dir.name.removeprefix("prior_weight_").replace("m", "-").replace(
        "p",
        ".",
    )
    return float(text)


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
