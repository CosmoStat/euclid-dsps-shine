"""Projected-truth closure diagnostics against DSPS optima."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.calibration import (
    apply_global_sed_scale_to_flux,
    apply_per_band_flux_calibration_to_flux,
    global_sed_scale_config,
    per_band_flux_calibration_config,
)
from euclid_dsps.filters import load_filters
from euclid_dsps.io import ensure_dir, write_json
from euclid_dsps.model import dynamic_model_args, load_context

from .catalog_identity import (
    select_catalog_row_indices,
    write_catalog_fingerprint,
    write_truth_snapshot,
)
from .config import amortized_config
from .data import load_photometry_arrays_from_config
from .decoder import model_flux_from_x
from .diagnostics import _truth_parameter_series
from .features import read_feature_stats
from .latent import initial_theta_from_config, latent_spec_from_config, theta_to_x
from .likelihood import photometric_loglike, photometric_normalized_residual
from .map_adam import run_map_adam_under_prior
from .train import load_checkpoint


def run_closure_optimum_diagnostics(
    config: dict[str, Any],
    out_dir: str | Path,
    *,
    checkpoint: str | Path,
    feature_stats_path: str | Path | None = None,
    nn_run: str | Path | None = None,
    row_indices_file: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 128,
    map_n_starts: int = 8,
    map_maxiter: int = 160,
    map_learning_rate: float = 0.015,
    map_start_mode: str = "mixed",
    map_start_chunk_size: int = 1,
    selection_mode: str | None = None,
    stratified_strategy: str | None = None,
    selection_seed: int | None = None,
    redshift_profile_count: int = 24,
    redshift_grid_size: int = 96,
    redshift_profile_source: str = "truth_projected",
    run_map: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compare projected truth, NN median, and flat-prior MAP log-likelihoods."""
    out = ensure_dir(out_dir)
    cfg = amortized_config(config)
    inference_cfg = cfg["inference"]
    seed = int(
        selection_seed
        if selection_seed is not None
        else cfg["training"].get("seed", 42)
    )
    redshift_bins = inference_cfg.get(
        "redshift_bins",
        (config.get("amortized", {}) or {}).get("data", {}).get("redshift_bins"),
    )
    catalog_fingerprint = write_catalog_fingerprint(
        out,
        config,
        redshift_bins=redshift_bins,
    )
    row_indices, selection = select_catalog_row_indices(
        config,
        limit=limit,
        selection_mode=str(selection_mode or inference_cfg.get("selection_mode", "sequential")),
        stratified_strategy=str(
            stratified_strategy or inference_cfg.get("stratified_strategy", "balanced")
        ),
        seed=seed,
        redshift_bins=redshift_bins,
        row_indices_file=row_indices_file,
    )
    if row_indices is not None:
        np.save(out / "closure_indices.npy", row_indices)
        selection["row_indices_path"] = "closure_indices.npy"
    write_json(out / "closure_selection.json", selection)
    truth = write_truth_snapshot(
        out,
        config,
        row_indices=row_indices,
        limit=limit,
        batch_size=int(inference_cfg.get("catalog_batch_size", 10_000)),
        filename="closure_truth.parquet",
    )
    arrays = load_photometry_arrays_from_config(
        config,
        batch_size=int(inference_cfg.get("catalog_batch_size", 10_000)),
        limit=limit if row_indices is None else None,
        row_indices=row_indices,
    )
    latent_spec = latent_spec_from_config(config)
    runtime = _load_runtime(
        config,
        checkpoint=Path(checkpoint),
        feature_stats_path=Path(feature_stats_path) if feature_stats_path else None,
    )
    theta_frames: dict[str, pd.DataFrame] = {
        "truth_projected": _truth_theta_frame(config, truth, latent_spec),
    }
    nn_frame = _nn_median_theta_frame(nn_run, latent_spec) if nn_run else pd.DataFrame()
    if not nn_frame.empty:
        theta_frames["nn_median"] = nn_frame

    map_dir = out / "map_flat"
    if run_map:
        if verbose:
            print(f"[closure] running flat-prior MAP -> {map_dir}")
        run_map_adam_under_prior(
            config,
            map_dir,
            checkpoint=Path(checkpoint),
            feature_stats_path=(
                Path(feature_stats_path) if feature_stats_path is not None else None
            ),
            limit=limit,
            batch_size=int(batch_size),
            n_starts=int(map_n_starts),
            maxiter=int(map_maxiter),
            learning_rate=float(map_learning_rate),
            prior_weight=0.0,
            seed=seed,
            start_mode=str(map_start_mode),
            start_chunk_size=int(map_start_chunk_size),
            selection_mode=selection_mode,
            stratified_strategy=stratified_strategy,
            selection_seed=seed,
            row_indices_file=row_indices_file,
            dataset_label="Diffsky HLTDS closure MAP flat",
            verbose=verbose,
        )
    map_path = map_dir / "map_estimates.parquet"
    if map_path.exists():
        theta_frames["map_flat"] = _map_theta_frame(map_path, latent_spec)

    object_tables = []
    residual_tables = []
    for label, frame in theta_frames.items():
        if frame.empty:
            continue
        metrics, residuals = _evaluate_theta_frame(
            label,
            frame,
            arrays=arrays,
            latent_spec=latent_spec,
            runtime=runtime,
            likelihood_config=cfg["likelihood"],
            batch_size=int(batch_size),
        )
        object_tables.append(metrics)
        residual_tables.append(residuals)
    object_long = (
        pd.concat(object_tables, ignore_index=True) if object_tables else pd.DataFrame()
    )
    residual_long = (
        pd.concat(residual_tables, ignore_index=True)
        if residual_tables
        else pd.DataFrame()
    )
    object_wide = _wide_object_metrics(object_long, truth)
    object_long.to_parquet(out / "closure_optimum_object_metrics_long.parquet", index=False)
    object_long.to_csv(out / "closure_optimum_object_metrics_long.csv", index=False)
    object_wide.to_parquet(out / "closure_optimum_summary.parquet", index=False)
    object_wide.to_csv(out / "closure_optimum_summary.csv", index=False)
    if not residual_long.empty:
        residual_long.to_parquet(out / "closure_residuals_by_band.parquet", index=False)
        _residual_band_summary(residual_long).to_csv(
            out / "closure_residuals_by_band_summary.csv",
            index=False,
        )

    profile_summary: dict[str, Any] = {}
    if int(redshift_profile_count) > 0 and int(redshift_grid_size) > 1:
        source = theta_frames.get(str(redshift_profile_source), pd.DataFrame())
        if not source.empty:
            profiles = _write_redshift_profiles(
                out,
                source,
                object_wide,
                arrays=arrays,
                latent_spec=latent_spec,
                runtime=runtime,
                likelihood_config=cfg["likelihood"],
                n_objects=int(redshift_profile_count),
                grid_size=int(redshift_grid_size),
                source_label=str(redshift_profile_source),
            )
            profile_summary = profiles

    _write_closure_plots(out, object_wide, residual_long)
    summary = {
        "checkpoint": str(checkpoint),
        "nn_run": str(nn_run) if nn_run else None,
        "feature_stats_path": str(runtime["feature_stats_path"]),
        "selection": selection,
        "catalog_fingerprint": catalog_fingerprint,
        "truth_rows": int(len(truth)),
        "n_objects": int(len(object_wide)),
        "methods": sorted(object_long["method"].unique().tolist())
        if not object_long.empty
        else [],
        "map_flat": {
            "enabled": bool(run_map),
            "path": str(map_dir),
            "n_starts": int(map_n_starts),
            "maxiter": int(map_maxiter),
            "learning_rate": float(map_learning_rate),
            "start_mode": str(map_start_mode),
        },
        "redshift_profiles": profile_summary,
        "outputs": {
            "summary": "closure_optimum_summary.parquet",
            "object_metrics_long": "closure_optimum_object_metrics_long.parquet",
            "residuals_by_band": "closure_residuals_by_band.parquet",
        },
    }
    write_json(out / "closure_optimum_summary.json", summary)
    return summary


def _load_runtime(
    config: dict[str, Any],
    *,
    checkpoint: Path,
    feature_stats_path: Path | None,
) -> dict[str, Any]:
    if feature_stats_path is None:
        feature_stats_path = checkpoint.parent.parent / "feature_stats.json"
    feature_stats = read_feature_stats(feature_stats_path)
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    model = load_checkpoint(checkpoint, config)
    scale_cfg = global_sed_scale_config({"calibration": config.get("calibration", {}) or {}})
    band_cfg = per_band_flux_calibration_config(
        {"calibration": config.get("calibration", {}) or {}}
    )
    return {
        "feature_stats": feature_stats,
        "feature_stats_path": feature_stats_path,
        "context": context,
        "model_args": dynamic_model_args(context),
        "model": model,
        "scale_cfg": scale_cfg,
        "band_cfg": band_cfg,
        "log_alpha_sed": model.sed_scale.log_alpha_sed
        if scale_cfg.enabled
        else jnp.asarray(0.0, dtype=jnp.float32),
        "log_alpha_band": model.band_calibration.log_alpha_band
        if band_cfg.enabled and model.band_calibration is not None
        else jnp.zeros((len(config["bands"]),), dtype=jnp.float32),
    }


def _truth_theta_frame(config: dict[str, Any], truth: pd.DataFrame, latent_spec) -> pd.DataFrame:
    if truth.empty:
        return pd.DataFrame()
    rows = pd.DataFrame(
        {
            "row_index": pd.to_numeric(truth["row_index"], errors="coerce").astype("Int64"),
            "object_id": truth.get("object_id", truth["row_index"]),
        }
    )
    specs = dict((config.get("truth", {}) or {}).get("parameter_columns") or {})
    for name in latent_spec.names:
        spec = specs.get(name)
        if isinstance(spec, dict) and str(spec.get("kind", "")).lower() == "missing":
            continue
        series = _truth_parameter_series(truth, str(name), spec)
        if series is not None:
            rows[str(name)] = pd.to_numeric(series, errors="coerce")
    return _complete_theta_frame(rows, config, latent_spec)


def _nn_median_theta_frame(nn_run: str | Path | None, latent_spec) -> pd.DataFrame:
    if nn_run is None:
        return pd.DataFrame()
    path = Path(nn_run) / "posterior_summary.parquet"
    if not path.exists():
        return pd.DataFrame()
    summary = pd.read_parquet(path)
    if "row_index" not in summary:
        return pd.DataFrame()
    frame = summary[["row_index"]].copy()
    if "object_id" in summary:
        frame["object_id"] = summary["object_id"]
    for name in latent_spec.names:
        column = f"{name}_median"
        if column in summary:
            frame[str(name)] = pd.to_numeric(summary[column], errors="coerce")
    return frame


def _map_theta_frame(map_path: str | Path, latent_spec) -> pd.DataFrame:
    estimates = pd.read_parquet(map_path)
    if "row_index" not in estimates:
        return pd.DataFrame()
    columns = ["row_index"]
    if "object_id" in estimates:
        columns.append("object_id")
    for name in latent_spec.names:
        if name in estimates:
            columns.append(str(name))
    return estimates[columns].copy()


def _complete_theta_frame(
    frame: pd.DataFrame,
    config: dict[str, Any],
    latent_spec,
) -> pd.DataFrame:
    out = frame.copy()
    theta0 = initial_theta_from_config(
        config,
        latent_spec.names,
        np.asarray(latent_spec.lower, dtype=float),
        np.asarray(latent_spec.upper, dtype=float),
    )
    for index, name in enumerate(latent_spec.names):
        if name not in out:
            out[str(name)] = float(theta0[index])
        else:
            out[str(name)] = pd.to_numeric(out[str(name)], errors="coerce").fillna(
                float(theta0[index])
            )
    return out


def _evaluate_theta_frame(
    label: str,
    theta_frame: pd.DataFrame,
    *,
    arrays,
    latent_spec,
    runtime: dict[str, Any],
    likelihood_config: dict[str, Any],
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_row = theta_frame.drop_duplicates("row_index").set_index("row_index")
    row_indices = np.asarray(arrays.row_index, dtype=np.int64)
    theta = np.full((len(row_indices), len(latent_spec.names)), np.nan, dtype=np.float32)
    object_id = np.asarray(arrays.object_id)
    present = np.zeros(len(row_indices), dtype=bool)
    for out_index, row_index in enumerate(row_indices):
        if row_index not in by_row.index:
            continue
        row = by_row.loc[row_index]
        theta[out_index] = [float(row[str(name)]) for name in latent_spec.names]
        present[out_index] = np.isfinite(theta[out_index]).all()
    order = np.nonzero(present)[0]
    object_rows = []
    residual_rows = []
    if order.size == 0:
        return pd.DataFrame(), pd.DataFrame()
    for start in range(0, order.size, int(batch_size)):
        idx = order[start : start + int(batch_size)]
        theta_batch = jnp.asarray(theta[idx], dtype=jnp.float32)
        model_flux = _model_flux_for_theta(theta_batch, latent_spec, runtime)
        loglike = photometric_loglike(
            obs_flux=jnp.asarray(arrays.flux[idx], dtype=jnp.float32),
            model_flux=model_flux,
            obs_err=jnp.asarray(arrays.flux_err[idx], dtype=jnp.float32),
            mask=jnp.asarray(arrays.mask[idx], dtype=bool),
            likelihood_type=str(likelihood_config.get("type", "student_t")),
            student_t_dof=float(likelihood_config.get("student_t_dof", 2.0)),
            error_floor_frac=float(likelihood_config.get("error_floor_frac", 0.02)),
            error_jitter=float(likelihood_config.get("error_jitter", 0.0)),
        )[0]
        chi = photometric_normalized_residual(
            obs_flux=jnp.asarray(arrays.flux[idx], dtype=jnp.float32),
            model_flux=model_flux,
            obs_err=jnp.asarray(arrays.flux_err[idx], dtype=jnp.float32),
            mask=jnp.asarray(arrays.mask[idx], dtype=bool),
            error_floor_frac=float(likelihood_config.get("error_floor_frac", 0.02)),
            error_jitter=float(likelihood_config.get("error_jitter", 0.0)),
        )[0]
        model_np = np.asarray(jax.device_get(model_flux[0]), dtype=float)
        chi_np = np.asarray(jax.device_get(chi), dtype=float)
        loglike_np = np.asarray(jax.device_get(loglike), dtype=float)
        for local, array_index in enumerate(idx):
            row = {
                "method": label,
                "row_index": int(row_indices[array_index]),
                "object_id": object_id[array_index],
                "photometric_loglike": float(loglike_np[local]),
                "photometric_nll": float(-loglike_np[local]),
            }
            for param_index, name in enumerate(latent_spec.names):
                row[str(name)] = float(theta[array_index, param_index])
            object_rows.append(row)
            for band_index, band in enumerate(arrays.band_names):
                if not bool(arrays.mask[array_index, band_index]):
                    continue
                obs_flux = float(arrays.flux[array_index, band_index])
                model_value = float(model_np[local, band_index])
                residual = model_value - obs_flux
                residual_rows.append(
                    {
                        "method": label,
                        "row_index": int(row_indices[array_index]),
                        "object_id": object_id[array_index],
                        "band": str(band),
                        "obs_flux_fnu_cgs": obs_flux,
                        "obs_err_fnu_cgs": float(arrays.flux_err[array_index, band_index]),
                        "model_flux_fnu_cgs": model_value,
                        "flux_residual_model_minus_obs": residual,
                        "abs_flux_residual": abs(residual),
                        "chi_likelihood": float(chi_np[local, band_index]),
                        "abs_chi_likelihood": abs(float(chi_np[local, band_index])),
                    }
                )
    return pd.DataFrame(object_rows), pd.DataFrame(residual_rows)


def _model_flux_for_theta(theta: jnp.ndarray, latent_spec, runtime: dict[str, Any]) -> jnp.ndarray:
    x = theta_to_x(theta, latent_spec)
    if x.ndim == 1:
        x = x[None, None, :]
    elif x.ndim == 2:
        x = x[None, :, :]
    elif x.ndim != 3:
        raise ValueError(f"theta must be rank 1, 2, or 3; got {theta.shape}")
    model_flux_raw = model_flux_from_x(
        x,
        latent_spec,
        runtime["context"],
        runtime["model_args"],
        latent_spec.names,
    )
    model_flux = (
        apply_global_sed_scale_to_flux(model_flux_raw, runtime["log_alpha_sed"])
        if runtime["scale_cfg"].enabled
        else model_flux_raw
    )
    return (
        apply_per_band_flux_calibration_to_flux(model_flux, runtime["log_alpha_band"])
        if runtime["band_cfg"].enabled
        else model_flux
    )


def _wide_object_metrics(long: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()
    base_columns = ["row_index", "object_id"]
    methods = sorted(long["method"].unique())
    base = long[base_columns].drop_duplicates("row_index").copy()
    for method in methods:
        group = long.loc[long["method"] == method].copy()
        keep = ["row_index", "photometric_loglike", "photometric_nll", "z_obs"]
        keep = [column for column in keep if column in group]
        renamed = {
            column: f"{method}_{column}" for column in keep if column != "row_index"
        }
        base = base.merge(group[keep].rename(columns=renamed), on="row_index", how="left")
    if "redshift_true" in truth:
        ztruth = truth[["row_index", "redshift_true"]].drop_duplicates("row_index")
        base = base.merge(ztruth, on="row_index", how="left")
    if "map_flat_photometric_loglike" in base and "truth_projected_photometric_loglike" in base:
        base["delta_loglike_map_flat_minus_truth"] = (
            base["map_flat_photometric_loglike"]
            - base["truth_projected_photometric_loglike"]
        )
    for method in methods:
        column = f"{method}_z_obs"
        if column in base and "redshift_true" in base:
            base[f"{method}_delta_z"] = base[column] - base["redshift_true"]
    return base


def _residual_band_summary(residuals: pd.DataFrame) -> pd.DataFrame:
    if residuals.empty:
        return pd.DataFrame()
    return (
        residuals.groupby(["method", "band"], dropna=False)
        .agg(
            n=("chi_likelihood", "size"),
            median_chi=("chi_likelihood", "median"),
            median_abs_chi=("abs_chi_likelihood", "median"),
            p95_abs_chi=("abs_chi_likelihood", lambda x: float(np.nanquantile(x, 0.95))),
            median_abs_flux_residual=("abs_flux_residual", "median"),
        )
        .reset_index()
    )


def _write_redshift_profiles(
    out: Path,
    source: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    arrays,
    latent_spec,
    runtime: dict[str, Any],
    likelihood_config: dict[str, Any],
    n_objects: int,
    grid_size: int,
    source_label: str,
) -> dict[str, Any]:
    z_index = latent_spec.names.index("z_obs")
    lower = float(np.asarray(latent_spec.lower)[z_index])
    upper = float(np.asarray(latent_spec.upper)[z_index])
    selected = _profile_row_indices(metrics, n_objects=n_objects)
    if not selected:
        selected = source["row_index"].head(int(n_objects)).astype(int).tolist()
    by_row = source.drop_duplicates("row_index").set_index("row_index")
    row_to_array = {int(row): idx for idx, row in enumerate(np.asarray(arrays.row_index))}
    z_grid = np.linspace(lower, upper, int(grid_size), dtype=np.float32)
    rows = []
    for row_index in selected:
        if row_index not in by_row.index or row_index not in row_to_array:
            continue
        theta0 = np.asarray(
            [float(by_row.loc[row_index, str(name)]) for name in latent_spec.names],
            dtype=np.float32,
        )
        theta_grid = np.repeat(theta0[None, :], len(z_grid), axis=0)
        theta_grid[:, z_index] = z_grid
        array_index = row_to_array[row_index]
        model_flux = _model_flux_for_theta(
            jnp.asarray(theta_grid[:, None, :], dtype=jnp.float32),
            latent_spec,
            runtime,
        )
        loglike = photometric_loglike(
            obs_flux=jnp.asarray(arrays.flux[[array_index]], dtype=jnp.float32),
            model_flux=model_flux,
            obs_err=jnp.asarray(arrays.flux_err[[array_index]], dtype=jnp.float32),
            mask=jnp.asarray(arrays.mask[[array_index]], dtype=bool),
            likelihood_type=str(likelihood_config.get("type", "student_t")),
            student_t_dof=float(likelihood_config.get("student_t_dof", 2.0)),
            error_floor_frac=float(likelihood_config.get("error_floor_frac", 0.02)),
            error_jitter=float(likelihood_config.get("error_jitter", 0.0)),
        )[:, 0]
        loglike_np = np.asarray(jax.device_get(loglike), dtype=float)
        best = int(np.nanargmax(loglike_np)) if np.isfinite(loglike_np).any() else 0
        for z_value, value in zip(z_grid, loglike_np, strict=True):
            rows.append(
                {
                    "row_index": int(row_index),
                    "object_id": arrays.object_id[array_index],
                    "source": source_label,
                    "z_grid": float(z_value),
                    "photometric_loglike": float(value),
                    "delta_loglike_from_profile_max": float(value - loglike_np[best]),
                    "profile_best_z": float(z_grid[best]),
                }
            )
    profiles = pd.DataFrame(rows)
    if profiles.empty:
        return {"enabled": False, "reason": "no_profile_rows"}
    profiles.to_parquet(out / "redshift_profile_samples.parquet", index=False)
    profiles.to_csv(out / "redshift_profile_samples.csv", index=False)
    profile_summary = (
        profiles.groupby("row_index", dropna=False)
        .agg(
            object_id=("object_id", "first"),
            source=("source", "first"),
            profile_best_z=("profile_best_z", "first"),
            max_loglike=("photometric_loglike", "max"),
        )
        .reset_index()
    )
    profile_summary.to_csv(out / "redshift_profile_summary.csv", index=False)
    _write_profile_plot(out, profiles)
    return {
        "enabled": True,
        "source": source_label,
        "n_objects": int(profile_summary["row_index"].nunique()),
        "grid_size": int(grid_size),
        "samples": "redshift_profile_samples.parquet",
        "summary": "redshift_profile_summary.csv",
    }


def _profile_row_indices(metrics: pd.DataFrame, *, n_objects: int) -> list[int]:
    if metrics.empty or "row_index" not in metrics:
        return []
    score = None
    if "delta_loglike_map_flat_minus_truth" in metrics:
        score = pd.to_numeric(
            metrics["delta_loglike_map_flat_minus_truth"],
            errors="coerce",
        ).abs()
    elif "map_flat_delta_z" in metrics:
        score = pd.to_numeric(metrics["map_flat_delta_z"], errors="coerce").abs()
    if score is None:
        return metrics["row_index"].head(int(n_objects)).astype(int).tolist()
    work = metrics.assign(__score=score).sort_values("__score", ascending=False)
    return work["row_index"].head(int(n_objects)).astype(int).tolist()


def _write_closure_plots(out: Path, metrics: pd.DataFrame, residuals: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not metrics.empty and {
        "truth_projected_photometric_loglike",
        "map_flat_photometric_loglike",
    }.issubset(metrics.columns):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            metrics["truth_projected_photometric_loglike"],
            metrics["map_flat_photometric_loglike"],
            s=8,
            alpha=0.45,
        )
        lo = float(
            np.nanmin(
                metrics[
                    ["truth_projected_photometric_loglike", "map_flat_photometric_loglike"]
                ].to_numpy(float)
            )
        )
        hi = float(
            np.nanmax(
                metrics[
                    ["truth_projected_photometric_loglike", "map_flat_photometric_loglike"]
                ].to_numpy(float)
            )
        )
        ax.plot([lo, hi], [lo, hi], color="#555555", lw=1)
        ax.set_xlabel("projected truth logL")
        ax.set_ylabel("flat MAP logL")
        fig.tight_layout()
        fig.savefig(out / "loglike_truth_vs_map.png", dpi=160)
        plt.close(fig)
    if not metrics.empty and {"redshift_true", "map_flat_z_obs"}.issubset(metrics.columns):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(metrics["redshift_true"], metrics["map_flat_z_obs"], s=8, alpha=0.45)
        ax.plot([0.0, 0.35], [0.0, 0.35], color="#555555", lw=1)
        ax.set_xlabel("truth redshift")
        ax.set_ylabel("flat MAP z_obs")
        fig.tight_layout()
        fig.savefig(out / "z_true_vs_z_map_flat.png", dpi=160)
        plt.close(fig)
    if not residuals.empty:
        summary = _residual_band_summary(residuals)
        if not summary.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            for method, group in summary.groupby("method", sort=True):
                ax.plot(
                    group["band"].astype(str),
                    group["median_abs_chi"].to_numpy(float),
                    marker="o",
                    label=str(method),
                )
            ax.set_ylabel("median abs likelihood residual")
            ax.tick_params(axis="x", rotation=45)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "residuals_truth_nn_map_by_band.png", dpi=160)
            plt.close(fig)


def _write_profile_plot(out: Path, profiles: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for row_index, group in profiles.groupby("row_index", sort=False):
        ax.plot(
            group["z_grid"],
            group["delta_loglike_from_profile_max"],
            lw=1,
            alpha=0.65,
            label=str(row_index),
        )
    ax.set_xlabel("z_obs")
    ax.set_ylabel("logL - max profile logL")
    ax.set_title("Fixed-nuisance redshift likelihood profiles")
    fig.tight_layout()
    fig.savefig(out / "redshift_profiles_fixed_nuisance.png", dpi=160)
    plt.close(fig)
