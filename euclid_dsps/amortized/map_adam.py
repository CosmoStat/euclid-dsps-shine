"""MAP-Adam inference under a learned amortized RealNVP prior."""

from __future__ import annotations

import time
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
from .collapse_gates import write_inference_collapse_gate
from .config import amortized_config, require_equinox
from .data import (
    compute_feature_stats_from_config,
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from .decoder import model_flux_from_x
from .features import read_feature_stats, write_feature_stats
from .latent import (
    network_x_to_raw_x,
    theta_to_x,
    x_to_theta,
)
from .likelihood import photometric_loglike
from .train import (
    JitLatentSpec,
    _latent_spec_for_amortized_config,
    _StaticArg,
    build_amortized_model,
    load_checkpoint,
)
from .truth_diagnostics import write_extended_truth_diagnostics

eqx = require_equinox()


def run_map_adam_under_prior(
    config: dict[str, Any],
    out_dir: Path,
    *,
    checkpoint: Path | None,
    feature_stats_path: Path | None,
    limit: int | None,
    batch_size: int,
    n_starts: int,
    maxiter: int,
    learning_rate: float,
    prior_weight: float,
    prior_density_space: str = "x",
    seed: int,
    start_mode: str = "encoder",
    start_chunk_size: int | None = None,
    shard_outputs: bool = True,
    resume: bool = True,
    selection_mode: str | None = None,
    stratified_strategy: str | None = None,
    selection_seed: int | None = None,
    row_indices_file: str | Path | None = None,
    dataset_label: str = "Diffsky HLTDS",
    verbose: bool = True,
) -> dict[str, Any]:
    """Run per-object MAP optimization, optionally under a learned NF prior."""
    out = ensure_dir(out_dir)
    cfg = amortized_config(config)
    inference_cfg = cfg["inference"]
    selection_mode = str(
        selection_mode or inference_cfg.get("selection_mode", "sequential")
    )
    stratified_strategy = str(
        stratified_strategy or inference_cfg.get("stratified_strategy", "balanced")
    )
    selection_seed = int(selection_seed if selection_seed is not None else seed)
    redshift_bins = inference_cfg.get(
        "redshift_bins",
        (config.get("amortized", {}) or {}).get("data", {}).get("redshift_bins"),
    )
    catalog_fingerprint = write_catalog_fingerprint(
        out,
        config,
        redshift_bins=redshift_bins,
    )
    row_indices, selection_summary = select_catalog_row_indices(
        config,
        limit=limit,
        selection_mode=selection_mode,
        stratified_strategy=stratified_strategy,
        seed=selection_seed,
        redshift_bins=redshift_bins,
        row_indices_file=row_indices_file,
    )
    if row_indices is not None:
        np.save(out / "map_indices.npy", row_indices)
        selection_summary["row_indices_path"] = "map_indices.npy"
    write_json(out / "map_selection.json", selection_summary)
    truth_snapshot = write_truth_snapshot(
        out,
        config,
        row_indices=row_indices,
        limit=limit,
        filename="inference_truth.parquet",
        batch_size=int(inference_cfg.get("catalog_batch_size", 10_000)),
    )
    checkpoint = Path(checkpoint) if checkpoint is not None else None
    _validate_checkpoint_free_map(
        checkpoint=checkpoint,
        prior_weight=float(prior_weight),
        start_mode=str(start_mode),
    )
    arrays = load_photometry_arrays_from_config(
        config,
        batch_size=int(inference_cfg.get("catalog_batch_size", 10_000)),
        limit=limit if row_indices is None else None,
        row_indices=row_indices,
    )
    if feature_stats_path is None and checkpoint is not None:
        feature_stats_path = checkpoint.parent.parent / "feature_stats.json"
    if feature_stats_path is not None:
        feature_stats_path = Path(feature_stats_path)
        feature_stats = read_feature_stats(feature_stats_path)
    else:
        stats_config = dict(config)
        stats_catalog = cfg["features"].get("stats_catalog_path")
        if stats_catalog:
            stats_config["catalog_path"] = str(stats_catalog)
        feature_stats = compute_feature_stats_from_config(
            stats_config,
            batch_size=int(inference_cfg.get("catalog_batch_size", 10_000)),
        )
        feature_stats_path = out / "feature_stats.json"
        write_feature_stats(feature_stats_path, feature_stats)
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    model_args = dynamic_model_args(context)
    latent_spec = _latent_spec_for_amortized_config(config)
    jit_latent_spec = _jit_latent_spec(latent_spec)
    jit_context = _StaticArg(context)
    model_key, key = jax.random.split(jax.random.PRNGKey(int(seed)))
    model = (
        load_checkpoint(checkpoint, config)
        if checkpoint is not None
        else build_amortized_model(config, model_key, latent_spec=latent_spec)
    )
    scale_cfg = global_sed_scale_config(
        {"calibration": config.get("calibration", {}) or {}}
    )
    band_cfg = per_band_flux_calibration_config(
        {"calibration": config.get("calibration", {}) or {}}
    )
    log_alpha_sed = (
        model.sed_scale.log_alpha_sed
        if scale_cfg.enabled
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    log_alpha_band = (
        model.band_calibration.log_alpha_band
        if band_cfg.enabled and model.band_calibration is not None
        else jnp.zeros((len(config["bands"]),), dtype=jnp.float32)
    )
    start_chunk_size = _effective_start_chunk_size(start_chunk_size, int(n_starts))
    estimate_shard_dir = out / "map_estimates_shards"
    by_start_shard_dir = out / "map_estimates_by_start_shards"
    trace_shard_dir = out / "map_optimizer_trace_shards"
    photometry_shard_dir = out / "map_photometry_shards"
    if shard_outputs:
        estimate_shard_dir.mkdir(parents=True, exist_ok=True)
        by_start_shard_dir.mkdir(parents=True, exist_ok=True)
        trace_shard_dir.mkdir(parents=True, exist_ok=True)
        photometry_shard_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(
            "[map-prior] checkpoint: "
            f"{checkpoint if checkpoint is not None else 'none (likelihood-only)'}"
        )
        print(f"[map-prior] output directory: {out}")
        print(
            "[map-prior] run config: "
            f"limit={limit} batch_size={batch_size} n_starts={n_starts} "
            f"maxiter={maxiter} lr={learning_rate} prior_weight={prior_weight} "
            f"prior_density_space={prior_density_space} "
            f"start_mode={start_mode} start_chunk_size={start_chunk_size} "
            f"shard_outputs={shard_outputs} resume={resume}"
        )
    rows = []
    rows_by_start = []
    traces = []
    photometry_rows = []
    progress_rows = []
    started_at = time.perf_counter()
    for batch_index, batch in enumerate(
        iter_photometry_batches_from_arrays(
            arrays,
            batch_size=int(batch_size),
            feature_stats=feature_stats,
        ),
        start=1,
    ):
        estimate_shard = estimate_shard_dir / f"part_{batch_index:06d}.parquet"
        by_start_shard = by_start_shard_dir / f"part_{batch_index:06d}.parquet"
        trace_shard = trace_shard_dir / f"part_{batch_index:06d}.parquet"
        photometry_shard = photometry_shard_dir / f"part_{batch_index:06d}.parquet"
        if (
            shard_outputs
            and resume
            and estimate_shard.exists()
            and photometry_shard.exists()
        ):
            if verbose:
                print(
                    "[map-prior] batch "
                    f"{batch_index}: existing shard -> skip {estimate_shard}"
                )
            progress_rows.append(
                {
                    "batch": batch_index,
                    "n_objects": int(batch.flux.shape[0]),
                    "status": "skipped_existing_shard",
                    "elapsed_s": 0.0,
                }
            )
            continue
        if verbose:
            print(f"[map-prior] batch {batch_index}: n_objects={batch.flux.shape[0]}")
        key, batch_key = jax.random.split(key)
        batch_started = time.perf_counter()
        result = _map_adam_batch(
            model,
            batch,
            jit_latent_spec,
            jit_context,
            model_args,
            latent_spec.names,
            batch_key,
            n_starts=int(n_starts),
            maxiter=int(maxiter),
            learning_rate=float(learning_rate),
            prior_weight=float(prior_weight),
            prior_density_space=str(prior_density_space),
            start_mode=str(start_mode),
            start_chunk_size=int(start_chunk_size),
            likelihood_config=cfg["likelihood"],
            log_alpha_sed=log_alpha_sed,
            log_alpha_band=log_alpha_band,
            use_global_scale=bool(scale_cfg.enabled),
            use_band_calibration=bool(band_cfg.enabled),
        )
        jax.block_until_ready(result["best_objective"])
        batch_elapsed = time.perf_counter() - batch_started
        theta = np.asarray(jax.device_get(x_to_theta(result["best_x"], latent_spec)))
        model_flux = model_flux_from_x(
            result["best_x"],
            jit_latent_spec,
            context,
            model_args,
            latent_spec.names,
        )
        if scale_cfg.enabled:
            model_flux = apply_global_sed_scale_to_flux(model_flux, log_alpha_sed)
        if band_cfg.enabled:
            model_flux = apply_per_band_flux_calibration_to_flux(
                model_flux,
                log_alpha_band,
            )
        model_flux = np.asarray(jax.device_get(model_flux))
        mask = np.asarray(jax.device_get(batch.mask), dtype=bool)
        n_valid = np.sum(mask, axis=1, dtype=np.int64)
        nominal_dof = np.maximum(n_valid - len(latent_spec.names), 1)
        chi2 = np.asarray(jax.device_get(result["best_chi2"]))
        row = {
            "object_id": np.asarray(batch.object_id),
            "row_index": (
                np.asarray(batch.row_index, dtype=np.int64)
                if batch.row_index is not None
                else np.arange(theta.shape[0], dtype=np.int64)
            ),
            "map_objective": np.asarray(jax.device_get(result["best_objective"])),
            "map_photometric_nll": np.asarray(jax.device_get(result["best_nll"])),
            "map_prior_logprob": np.asarray(jax.device_get(result["best_logprior"])),
            "map_chi2": chi2,
            "map_n_valid_bands": n_valid,
            "map_n_parameters": len(latent_spec.names),
            "map_nominal_dof": nominal_dof,
            "map_chi2_per_valid_band": chi2 / np.maximum(n_valid, 1),
            "map_reduced_chi2": chi2 / nominal_dof,
            "map_start_index": np.asarray(jax.device_get(result["best_start"])),
            "map_start_family": np.asarray(result["start_family"], dtype=object)[
                np.asarray(jax.device_get(result["best_start"]))
            ],
            "map_grad_norm": np.asarray(jax.device_get(result["grad_norm"])),
        }
        for index, name in enumerate(latent_spec.names):
            row[name] = theta[:, index]
        estimates_piece = pd.DataFrame(row)
        photometry_piece = _map_photometry_frame(
            batch,
            model_flux,
            arrays.band_names,
        )
        by_start = _map_by_start_frame(
            result,
            batch,
            latent_spec,
        )
        if not by_start.empty:
            by_start["batch"] = batch_index
        trace = pd.DataFrame(
            {
                "batch": batch_index,
                "iteration": np.arange(result["trace_objective"].shape[0]),
                "mean_objective": np.asarray(jax.device_get(result["trace_objective"])),
            }
        )
        if shard_outputs:
            if not by_start.empty:
                _write_parquet_atomic(by_start, by_start_shard)
            _write_parquet_atomic(trace, trace_shard)
            _write_parquet_atomic(photometry_piece, photometry_shard)
            # This shard is the completion marker for resume, so write it last.
            _write_parquet_atomic(estimates_piece, estimate_shard)
        else:
            rows.append(estimates_piece)
            if not by_start.empty:
                rows_by_start.append(by_start)
            traces.append(trace)
            photometry_rows.append(photometry_piece)
        throughput = (
            float(batch.flux.shape[0]) / batch_elapsed
            if batch_elapsed > 0.0
            else float("nan")
        )
        progress_rows.append(
            {
                "batch": batch_index,
                "n_objects": int(batch.flux.shape[0]),
                "status": "completed",
                "elapsed_s": float(batch_elapsed),
                "objects_per_s": float(throughput),
                "wall_elapsed_s": float(time.perf_counter() - started_at),
            }
        )
        pd.DataFrame(progress_rows).to_csv(out / "map_progress.csv", index=False)
        if verbose:
            print(
                "[map-prior] batch "
                f"{batch_index} done in {batch_elapsed:.1f}s "
                f"({throughput:.2f} objects/s)"
            )
    if shard_outputs:
        rows = _read_shard_frames(estimate_shard_dir)
        rows_by_start = _read_shard_frames(by_start_shard_dir)
        traces = _read_shard_frames(trace_shard_dir)
        photometry_rows = _read_shard_frames(photometry_shard_dir)
    estimates = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    by_start_frame = (
        pd.concat(rows_by_start, ignore_index=True) if rows_by_start else pd.DataFrame()
    )
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    photometry_frame = (
        pd.concat(photometry_rows, ignore_index=True)
        if photometry_rows
        else pd.DataFrame()
    )
    estimates.to_parquet(out / "map_estimates.parquet", index=False)
    estimates.to_csv(out / "map_estimates.csv", index=False)
    if not by_start_frame.empty:
        by_start_frame.to_parquet(out / "map_estimates_by_start.parquet", index=False)
        by_start_frame.to_csv(out / "map_estimates_by_start.csv", index=False)
        _write_start_family_summary(by_start_frame, out)
    trace_frame.to_parquet(out / "map_optimizer_trace.parquet", index=False)
    photometry_frame.to_parquet(out / "map_photometry.parquet", index=False)
    _write_map_photometry_summary(photometry_frame, out)
    _write_map_plots(estimates, trace_frame, out, by_start_frame)
    summary = {
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "uses_learned_prior": bool(checkpoint is not None and prior_weight != 0.0),
        "feature_stats_path": str(feature_stats_path),
        "limit": limit,
        "selection": selection_summary,
        "catalog_fingerprint": catalog_fingerprint,
        "truth_snapshot_rows": int(len(truth_snapshot)),
        "n_objects": int(len(estimates)),
        "batch_size": int(batch_size),
        "n_starts": int(n_starts),
        "maxiter": int(maxiter),
        "learning_rate": float(learning_rate),
        "prior_weight": float(prior_weight),
        "prior_density_space": str(prior_density_space),
        "start_mode": str(start_mode),
        "start_chunk_size": int(start_chunk_size),
        "shard_outputs": bool(shard_outputs),
        "resume": bool(resume),
        "shard_paths": {
            "estimates": str(estimate_shard_dir) if shard_outputs else None,
            "by_start": str(by_start_shard_dir) if shard_outputs else None,
            "trace": str(trace_shard_dir) if shard_outputs else None,
            "photometry": str(photometry_shard_dir) if shard_outputs else None,
        },
        "dataset_label": dataset_label,
        "photometric_fit": _map_fit_summary(estimates),
    }
    write_json(out / "map_summary.json", summary)
    try:
        _write_map_closure_metrics(out, estimates)
    except Exception as exc:
        summary["map_closure_warning"] = str(exc)
        write_json(out / "map_summary.json", summary)
    try:
        extended_outputs = write_extended_truth_diagnostics(out)
    except Exception as exc:
        extended_outputs = {"warning": str(exc)}
    if extended_outputs:
        summary["extended_truth_diagnostics"] = extended_outputs
    try:
        gate = write_inference_collapse_gate(out)
        summary["collapse_gate"] = {
            "path": str(out / "collapse_gate.json"),
            "status": gate.get("status"),
        }
    except Exception as exc:
        summary["collapse_gate_warning"] = str(exc)
    if (
        extended_outputs
        or "collapse_gate" in summary
        or "collapse_gate_warning" in summary
    ):
        write_json(out / "map_summary.json", summary)
    return summary


def _validate_checkpoint_free_map(
    *,
    checkpoint: Path | None,
    prior_weight: float,
    start_mode: str,
) -> None:
    if checkpoint is not None:
        return
    if not np.isclose(float(prior_weight), 0.0):
        raise ValueError(
            "MAP without an amortized checkpoint requires prior_weight=0"
        )
    mode = str(start_mode or "").strip().lower()
    if mode not in {"latin_hypercube", "z_grid", "lowz_grid"}:
        raise ValueError(
            "MAP without an amortized checkpoint requires an encoder-independent "
            "start_mode: latin_hypercube, z_grid, or lowz_grid"
        )


def _map_photometry_frame(
    batch,
    model_flux: np.ndarray,
    band_names: tuple[str, ...],
) -> pd.DataFrame:
    observed = np.asarray(jax.device_get(batch.flux), dtype=float)
    error = np.asarray(jax.device_get(batch.flux_err), dtype=float)
    mask = np.asarray(jax.device_get(batch.mask), dtype=bool)
    predicted = np.asarray(model_flux, dtype=float)
    if predicted.shape != observed.shape:
        raise ValueError(
            f"MAP model flux shape {predicted.shape} != observed {observed.shape}"
        )
    n_objects, n_bands = observed.shape
    if len(band_names) != n_bands:
        raise ValueError(f"Expected {n_bands} band names, got {len(band_names)}")
    object_id = np.asarray(batch.object_id)
    row_index = (
        np.asarray(batch.row_index, dtype=np.int64)
        if batch.row_index is not None
        else np.arange(n_objects, dtype=np.int64)
    )
    residual = np.full(observed.shape, np.nan, dtype=float)
    valid = mask & np.isfinite(error) & (error > 0.0)
    residual[valid] = (predicted[valid] - observed[valid]) / error[valid]
    return pd.DataFrame(
        {
            "object_id": np.repeat(object_id, n_bands),
            "row_index": np.repeat(row_index, n_bands),
            "band": np.tile(np.asarray(band_names, dtype=object), n_objects),
            "observed_flux_fnu_cgs": observed.ravel(),
            "observed_error_fnu_cgs": error.ravel(),
            "map_flux_fnu_cgs": predicted.ravel(),
            "normalized_residual": residual.ravel(),
            "is_valid": valid.ravel(),
        }
    )


def _map_fit_summary(estimates: pd.DataFrame) -> dict[str, Any]:
    if estimates.empty:
        return {}
    chi2_per_band = estimates["map_chi2_per_valid_band"].to_numpy(dtype=float)
    reduced_chi2 = estimates["map_reduced_chi2"].to_numpy(dtype=float)
    chi2_per_band = chi2_per_band[np.isfinite(chi2_per_band)]
    reduced_chi2 = reduced_chi2[np.isfinite(reduced_chi2)]
    if not len(reduced_chi2):
        return {}
    return {
        "n_objects": int(len(estimates)),
        "n_parameters": int(estimates["map_n_parameters"].iloc[0]),
        "median_chi2_per_valid_band": float(np.median(chi2_per_band)),
        "median_reduced_chi2": float(np.median(reduced_chi2)),
        "p84_reduced_chi2": float(np.percentile(reduced_chi2, 84)),
        "fraction_reduced_chi2_le_2": float(np.mean(reduced_chi2 <= 2.0)),
        "fraction_reduced_chi2_le_5": float(np.mean(reduced_chi2 <= 5.0)),
        "fraction_reduced_chi2_le_10": float(np.mean(reduced_chi2 <= 10.0)),
    }


def _write_map_photometry_summary(frame: pd.DataFrame, out: Path) -> None:
    if frame.empty:
        return
    valid = frame.loc[
        frame["is_valid"].astype(bool)
        & np.isfinite(frame["normalized_residual"].to_numpy(dtype=float))
    ].copy()
    if valid.empty:
        return
    rows = []
    for band, group in valid.groupby("band", sort=False):
        residual = group["normalized_residual"].to_numpy(dtype=float)
        rows.append(
            {
                "band": str(band),
                "n": int(len(residual)),
                "mean": float(np.mean(residual)),
                "std": float(np.std(residual)),
                "median": float(np.median(residual)),
                "p16": float(np.percentile(residual, 16)),
                "p84": float(np.percentile(residual, 84)),
                "frac_abs_gt_3": float(np.mean(np.abs(residual) > 3.0)),
                "frac_abs_gt_5": float(np.mean(np.abs(residual) > 5.0)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "map_normalized_residuals_by_band.csv", index=False)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    y = np.arange(len(summary))
    median = summary["median"].to_numpy(dtype=float)
    lower = median - summary["p16"].to_numpy(dtype=float)
    upper = summary["p84"].to_numpy(dtype=float) - median
    fig, ax = plt.subplots(figsize=(8.0, max(5.0, 0.28 * len(summary))))
    ax.errorbar(
        median,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        markersize=4,
        capsize=2,
    )
    ax.axvline(0.0, color="black", lw=1.0, alpha=0.6)
    ax.axvspan(-3.0, 3.0, color="0.9", zorder=-1)
    ax.set_yticks(y, summary["band"])
    ax.set_xlabel("(MAP model flux - observed flux) / reported error")
    ax.set_title("Native 15D MAP photometric residuals")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "map_normalized_residuals_by_band.png", dpi=160)
    plt.close(fig)


def _map_adam_batch(
    model,
    batch,
    latent_spec,
    context,
    model_args,
    parameter_names: tuple[str, ...],
    key,
    *,
    n_starts: int,
    maxiter: int,
    learning_rate: float,
    prior_weight: float,
    prior_density_space: str,
    start_mode: str,
    start_chunk_size: int,
    likelihood_config: dict[str, Any],
    log_alpha_sed,
    log_alpha_band,
    use_global_scale: bool,
    use_band_calibration: bool,
) -> dict[str, jnp.ndarray]:
    mode = str(start_mode or "encoder").strip().lower()
    if mode in {"encoder", "mixed"}:
        mean, log_std = model.encoder(batch.features)
    else:
        mean = jnp.zeros(
            (batch.flux.shape[0], len(parameter_names)),
            dtype=batch.flux.dtype,
        )
        log_std = jnp.zeros_like(mean)
    n_objects = mean.shape[0]
    n_starts = max(1, int(n_starts))
    starts, start_family = _make_map_starts(
        model,
        mean,
        log_std,
        latent_spec,
        key,
        n_starts=n_starts,
        start_mode=start_mode,
    )
    chunk_size = _effective_start_chunk_size(start_chunk_size, n_starts)
    chunk_results = []
    weighted_trace = None
    for start in range(0, n_starts, chunk_size):
        end = min(start + chunk_size, n_starts)
        chunk = starts[start:end]
        chunk_result = _optimize_map_start_chunk(
            model,
            batch,
            latent_spec,
            context,
            model_args,
            parameter_names,
            chunk,
            maxiter=int(maxiter),
            learning_rate=float(learning_rate),
            prior_weight=float(prior_weight),
            prior_density_space=str(prior_density_space),
            likelihood_config=likelihood_config,
            log_alpha_sed=log_alpha_sed,
            log_alpha_band=log_alpha_band,
            use_global_scale=use_global_scale,
            use_band_calibration=use_band_calibration,
        )
        chunk_results.append(chunk_result)
        weight = float(end - start)
        trace_piece = chunk_result["trace_objective"] * weight
        weighted_trace = (
            trace_piece if weighted_trace is None else weighted_trace + trace_piece
        )
    best_x = jnp.concatenate([result["best_x"] for result in chunk_results], axis=0)
    best_obj = jnp.concatenate(
        [result["best_objective"] for result in chunk_results], axis=0
    )
    best_nll = jnp.concatenate([result["best_nll"] for result in chunk_results], axis=0)
    best_logprior = jnp.concatenate(
        [result["best_logprior"] for result in chunk_results],
        axis=0,
    )
    best_chi2 = jnp.concatenate(
        [result["best_chi2"] for result in chunk_results], axis=0
    )
    grad_norm_all = jnp.concatenate(
        [result["grad_norm"] for result in chunk_results],
        axis=0,
    )
    trace = weighted_trace / float(n_starts)
    start_index = jnp.argmin(best_obj, axis=0)
    object_index = jnp.arange(n_objects)
    chosen_x = best_x[start_index, object_index]
    chosen_obj = best_obj[start_index, object_index]
    chosen_nll = best_nll[start_index, object_index]
    chosen_logprior = best_logprior[start_index, object_index]
    chosen_chi2 = best_chi2[start_index, object_index]
    grad_norm = grad_norm_all[start_index, object_index]
    return {
        "best_x": chosen_x,
        "best_objective": chosen_obj,
        "best_nll": chosen_nll,
        "best_logprior": chosen_logprior,
        "best_chi2": chosen_chi2,
        "best_start": start_index,
        "grad_norm": grad_norm,
        "trace_objective": jnp.asarray(trace),
        "all_best_x": best_x,
        "all_best_objective": best_obj,
        "all_best_nll": best_nll,
        "all_best_logprior": best_logprior,
        "all_best_chi2": best_chi2,
        "start_x": starts,
        "start_family": tuple(start_family),
    }


def _optimize_map_start_chunk(
    model,
    batch,
    latent_spec,
    context,
    model_args,
    parameter_names: tuple[str, ...],
    starts: jnp.ndarray,
    *,
    maxiter: int,
    learning_rate: float,
    prior_weight: float,
    prior_density_space: str,
    likelihood_config: dict[str, Any],
    log_alpha_sed,
    log_alpha_band,
    use_global_scale: bool,
    use_band_calibration: bool,
) -> dict[str, jnp.ndarray]:
    density_space = _normalize_prior_density_space(prior_density_space)
    if density_space == "theta" and "log10_gas_metallicity" in parameter_names:
        raise ValueError(
            "prior_density_space='theta' is not implemented for the coupled "
            "stellar/gas metallicity latent transform"
        )
    likelihood_type = str(likelihood_config.get("type", "student_t"))
    student_t_dof = float(likelihood_config.get("student_t_dof", 2.0))
    error_floor_frac = float(likelihood_config.get("error_floor_frac", 0.02))
    error_jitter = float(likelihood_config.get("error_jitter", 0.0))
    return _optimize_map_start_chunk_jit(
        model,
        batch.flux,
        batch.flux_err,
        batch.mask,
        latent_spec,
        context,
        model_args,
        parameter_names,
        starts,
        maxiter=int(maxiter),
        learning_rate=float(learning_rate),
        prior_weight=float(prior_weight),
        prior_density_space=density_space,
        likelihood_type=likelihood_type,
        student_t_dof=student_t_dof,
        error_floor_frac=error_floor_frac,
        error_jitter=error_jitter,
        log_alpha_sed=log_alpha_sed,
        log_alpha_band=log_alpha_band,
        use_global_scale=bool(use_global_scale),
        use_band_calibration=bool(use_band_calibration),
    )


@eqx.filter_jit
def _optimize_map_start_chunk_jit(
    model,
    flux: jnp.ndarray,
    flux_err: jnp.ndarray,
    mask: jnp.ndarray,
    latent_spec,
    context,
    model_args,
    parameter_names: tuple[str, ...],
    starts: jnp.ndarray,
    *,
    maxiter: int,
    learning_rate: float,
    prior_weight: float,
    prior_density_space: str,
    likelihood_type: str,
    student_t_dof: float,
    error_floor_frac: float,
    error_jitter: float,
    log_alpha_sed,
    log_alpha_band,
    use_global_scale: bool,
    use_band_calibration: bool,
) -> dict[str, jnp.ndarray]:
    actual_context = context.value if isinstance(context, _StaticArg) else context

    def metrics_for_x(x):
        model_flux_raw = model_flux_from_x(
            x,
            latent_spec,
            actual_context,
            model_args,
            parameter_names,
        )
        model_flux = (
            apply_global_sed_scale_to_flux(model_flux_raw, log_alpha_sed)
            if use_global_scale
            else model_flux_raw
        )
        model_flux = (
            apply_per_band_flux_calibration_to_flux(model_flux, log_alpha_band)
            if use_band_calibration
            else model_flux
        )
        loglike = photometric_loglike(
            obs_flux=flux,
            model_flux=model_flux,
            obs_err=flux_err,
            mask=mask,
            likelihood_type=likelihood_type,
            student_t_dof=float(student_t_dof),
            error_floor_frac=float(error_floor_frac),
            error_jitter=float(error_jitter),
        )
        logprior = _prior_log_prob_for_map(
            model,
            x,
            latent_spec,
            prior_density_space=prior_density_space,
        )
        chi = (model_flux - flux[None, :, :]) / flux_err[None, :, :]
        chi = jnp.where(mask[None, :, :], chi, 0.0)
        chi2 = jnp.sum(chi**2, axis=-1)
        objective = -loglike - float(prior_weight) * logprior
        return objective, -loglike, logprior, chi2

    def scalar_objective_with_aux(x):
        objective, _nll, _logprior, _chi2 = metrics_for_x(x)
        return jnp.mean(objective), metrics_for_x(x)

    value_and_grad = jax.value_and_grad(scalar_objective_with_aux, has_aux=True)
    x = starts
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    best_x = x
    best_obj, best_nll, best_logprior, best_chi2 = metrics_for_x(x)
    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1.0e-8

    def step(carry, iteration_zero):
        x, m, v, best_x, best_obj, best_nll, best_logprior, best_chi2 = carry
        (value, (obj, nll, logprior, chi2)), grad = value_and_grad(x)
        improved = obj < best_obj
        best_x = jnp.where(improved[..., None], x, best_x)
        best_obj = jnp.where(improved, obj, best_obj)
        best_nll = jnp.where(improved, nll, best_nll)
        best_logprior = jnp.where(improved, logprior, best_logprior)
        best_chi2 = jnp.where(improved, chi2, best_chi2)
        grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)
        iteration = iteration_zero + 1
        m_hat = m / (1.0 - beta1**iteration)
        v_hat = v / (1.0 - beta2**iteration)
        x = x - float(learning_rate) * m_hat / (jnp.sqrt(v_hat) + eps_adam)
        return (x, m, v, best_x, best_obj, best_nll, best_logprior, best_chi2), value

    init = (x, m, v, best_x, best_obj, best_nll, best_logprior, best_chi2)
    final, trace = jax.lax.scan(
        step,
        init,
        jnp.arange(int(maxiter), dtype=jnp.float32),
    )
    _x, _m, _v, best_x, best_obj, best_nll, best_logprior, best_chi2 = final
    (_value, _aux), final_grad = value_and_grad(best_x)
    grad_norm = jnp.linalg.norm(final_grad, axis=-1)
    return {
        "best_x": best_x,
        "best_objective": best_obj,
        "best_nll": best_nll,
        "best_logprior": best_logprior,
        "best_chi2": best_chi2,
        "grad_norm": grad_norm,
        "trace_objective": jnp.asarray(trace),
    }


def _normalize_prior_density_space(value: str) -> str:
    text = str(value or "x").strip().lower()
    if text in {"x", "latent", "network"}:
        return "x"
    if text in {"theta", "physical"}:
        return "theta"
    raise ValueError("prior_density_space must be 'x' or 'theta'")


def _prior_log_prob_for_map(
    model,
    x: jnp.ndarray,
    latent_spec,
    *,
    prior_density_space: str,
) -> jnp.ndarray:
    logp_x = model.prior.log_prob(x)
    if prior_density_space == "x":
        return logp_x
    raw = network_x_to_raw_x(x, latent_spec)
    unit = jax.nn.sigmoid(raw)
    scale = jnp.asarray(latent_spec.raw_scale, dtype=jnp.float32)
    span = jnp.asarray(latent_spec.upper - latent_spec.lower, dtype=jnp.float32)
    logdet = jnp.sum(
        jnp.log(jnp.maximum(span, 1.0e-12))
        + jnp.log(jnp.maximum(scale, 1.0e-12))
        + jnp.log(jnp.maximum(unit, 1.0e-12))
        + jnp.log(jnp.maximum(1.0 - unit, 1.0e-12)),
        axis=-1,
    )
    return logp_x - logdet


def _effective_start_chunk_size(value: int | None, n_starts: int) -> int:
    if value is None:
        return int(n_starts)
    chunk = int(value)
    if chunk <= 0:
        raise ValueError("MAP start_chunk_size must be positive")
    return max(1, min(chunk, int(n_starts)))


def _jit_latent_spec(latent_spec) -> JitLatentSpec:
    return JitLatentSpec(
        names=latent_spec.names,
        lower=latent_spec.lower,
        upper=latent_spec.upper,
        raw_center=latent_spec.raw_center,
        raw_scale=latent_spec.raw_scale,
        normalization=latent_spec.normalization,
        transform_family=latent_spec.transform_family,
        transform_location=latent_spec.transform_location,
        transform_lambda=latent_spec.transform_lambda,
    )


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _read_shard_frames(directory: Path) -> list[pd.DataFrame]:
    if not directory.exists():
        return []
    paths = sorted(directory.glob("part_*.parquet"))
    return [pd.read_parquet(path) for path in paths]


def _make_map_starts(
    model,
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    latent_spec,
    key,
    *,
    n_starts: int,
    start_mode: str,
) -> tuple[jnp.ndarray, tuple[str, ...]]:
    """Build MAP starting points and a family label for each start."""
    mode = str(start_mode or "encoder").strip().lower()
    if mode not in {
        "encoder",
        "prior",
        "z_grid",
        "lowz_grid",
        "latin_hypercube",
        "mixed",
    }:
        raise ValueError(
            "Unsupported MAP start_mode. Use encoder, prior, z_grid, "
            "lowz_grid, latin_hypercube, or mixed."
        )
    n_starts = max(1, int(n_starts))
    if mode == "encoder":
        return _encoder_map_starts(mean, log_std, key, n_starts)
    if mode == "prior":
        return (
            model.prior.sample(key, (n_starts, int(mean.shape[0]))),
            tuple("prior" for _ in range(n_starts)),
        )
    if mode == "z_grid":
        return _z_grid_map_starts(latent_spec, int(mean.shape[0]), n_starts)
    if mode == "lowz_grid":
        return _z_grid_map_starts(
            latent_spec,
            int(mean.shape[0]),
            n_starts,
            z_upper_override=0.45,
            family="lowz_grid",
        )
    if mode == "latin_hypercube":
        return _latin_hypercube_map_starts(
            latent_spec, key, int(mean.shape[0]), n_starts
        )
    return _mixed_map_starts(model, mean, log_std, latent_spec, key, n_starts)


def _encoder_map_starts(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    key,
    n_starts: int,
) -> tuple[jnp.ndarray, tuple[str, ...]]:
    n_objects, latent_dim = mean.shape
    eps = jax.random.normal(
        key,
        (n_starts, n_objects, latent_dim),
        dtype=mean.dtype,
    )
    starts = mean[None, :, :] + jnp.exp(log_std)[None, :, :] * eps
    starts = starts.at[0].set(mean)
    labels = ["encoder"] * n_starts
    labels[0] = "encoder_mean"
    return starts, tuple(labels)


def _z_grid_map_starts(
    latent_spec,
    n_objects: int,
    n_starts: int,
    *,
    z_upper_override: float | None = None,
    family: str = "z_grid",
) -> tuple[jnp.ndarray, tuple[str, ...]]:
    try:
        z_index = latent_spec.names.index("z_obs")
    except ValueError as exc:
        raise ValueError("MAP z_grid start_mode requires z_obs in the latent") from exc
    lower = jnp.asarray(latent_spec.lower, dtype=jnp.float32)
    upper = jnp.asarray(latent_spec.upper, dtype=jnp.float32)
    theta = 0.5 * (lower + upper)
    z_low = lower[z_index]
    z_high = upper[z_index]
    if z_upper_override is not None:
        z_high = jnp.minimum(
            z_high, jnp.asarray(float(z_upper_override), dtype=jnp.float32)
        )
        z_high = jnp.maximum(z_high, z_low + jnp.asarray(1.0e-4, dtype=jnp.float32))
    z_values = jnp.linspace(z_low, z_high, int(n_starts), dtype=jnp.float32)
    theta = jnp.broadcast_to(theta, (int(n_starts), int(n_objects), theta.shape[0]))
    theta = theta.at[:, :, z_index].set(z_values[:, None])
    starts = theta_to_x(theta, latent_spec)
    return starts, tuple(family for _ in range(int(n_starts)))


def _latin_hypercube_map_starts(
    latent_spec,
    key,
    n_objects: int,
    n_starts: int,
) -> tuple[jnp.ndarray, tuple[str, ...]]:
    lower = jnp.asarray(latent_spec.lower, dtype=jnp.float32)
    upper = jnp.asarray(latent_spec.upper, dtype=jnp.float32)
    unit = jax.random.uniform(
        key,
        (int(n_starts), int(n_objects), int(lower.shape[0])),
        minval=0.02,
        maxval=0.98,
        dtype=jnp.float32,
    )
    theta = lower + (upper - lower) * unit
    return theta_to_x(theta, latent_spec), tuple(
        "latin_hypercube" for _ in range(int(n_starts))
    )


def _mixed_map_starts(
    model,
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    latent_spec,
    key,
    n_starts: int,
) -> tuple[jnp.ndarray, tuple[str, ...]]:
    keys = jax.random.split(key, 4)
    counts = _split_start_counts(int(n_starts), 4)
    pieces = []
    labels: list[str] = []
    if counts[0]:
        starts, family = _encoder_map_starts(mean, log_std, keys[0], counts[0])
        pieces.append(starts)
        labels.extend(family)
    if counts[1]:
        pieces.append(model.prior.sample(keys[1], (counts[1], int(mean.shape[0]))))
        labels.extend(["prior"] * counts[1])
    if counts[2]:
        starts, family = _z_grid_map_starts(latent_spec, int(mean.shape[0]), counts[2])
        pieces.append(starts)
        labels.extend(family)
    if counts[3]:
        starts, family = _latin_hypercube_map_starts(
            latent_spec,
            keys[3],
            int(mean.shape[0]),
            counts[3],
        )
        pieces.append(starts)
        labels.extend(family)
    return jnp.concatenate(pieces, axis=0), tuple(labels)


def _split_start_counts(total: int, n_groups: int) -> list[int]:
    base = int(total) // int(n_groups)
    remainder = int(total) % int(n_groups)
    return [base + (1 if index < remainder else 0) for index in range(int(n_groups))]


def _map_by_start_frame(result: dict[str, Any], batch, latent_spec) -> pd.DataFrame:
    all_best_x = np.asarray(jax.device_get(result["all_best_x"]))
    if all_best_x.size == 0:
        return pd.DataFrame()
    best_theta = np.asarray(
        jax.device_get(x_to_theta(result["all_best_x"], latent_spec))
    )
    start_theta = np.asarray(jax.device_get(x_to_theta(result["start_x"], latent_spec)))
    n_starts, n_objects, _latent_dim = best_theta.shape
    object_id = np.asarray(batch.object_id)
    row_index = (
        np.asarray(batch.row_index, dtype=np.int64)
        if batch.row_index is not None
        else np.arange(n_objects, dtype=np.int64)
    )
    start_index = np.broadcast_to(
        np.arange(n_starts, dtype=np.int64)[:, None],
        (n_starts, n_objects),
    )
    best_start = np.asarray(jax.device_get(result["best_start"]), dtype=np.int64)
    selected = start_index == best_start[None, :]
    family = np.asarray(result["start_family"], dtype=object)
    frame = pd.DataFrame(
        {
            "object_id": np.broadcast_to(
                object_id[None, :], (n_starts, n_objects)
            ).ravel(),
            "row_index": np.broadcast_to(
                row_index[None, :], (n_starts, n_objects)
            ).ravel(),
            "start_index": start_index.ravel(),
            "start_family": np.repeat(family, n_objects),
            "is_selected": selected.ravel(),
            "map_objective": np.asarray(
                jax.device_get(result["all_best_objective"])
            ).ravel(),
            "map_photometric_nll": np.asarray(
                jax.device_get(result["all_best_nll"])
            ).ravel(),
            "map_prior_logprob": np.asarray(
                jax.device_get(result["all_best_logprior"])
            ).ravel(),
            "map_chi2": np.asarray(jax.device_get(result["all_best_chi2"])).ravel(),
        }
    )
    for index, name in enumerate(latent_spec.names):
        frame[f"map_{name}"] = best_theta[:, :, index].ravel()
        frame[f"start_{name}"] = start_theta[:, :, index].ravel()
    return frame


def _write_start_family_summary(frame: pd.DataFrame, out: Path) -> None:
    if frame.empty or "start_family" not in frame:
        return
    grouped = frame.groupby("start_family", dropna=False)
    summary = grouped.agg(
        n_rows=("start_family", "size"),
        n_objects=("row_index", "nunique"),
        selected_count=("is_selected", "sum"),
        median_objective=("map_objective", "median"),
        median_photometric_nll=("map_photometric_nll", "median"),
        median_chi2=("map_chi2", "median"),
    ).reset_index()
    summary["selected_fraction"] = summary["selected_count"] / np.maximum(
        summary["n_rows"],
        1,
    )
    if "map_z_obs" in frame:
        z_summary = (
            grouped["map_z_obs"]
            .agg(
                median_map_z_obs="median",
                p16_map_z_obs=lambda value: float(np.nanpercentile(value, 16)),
                p84_map_z_obs=lambda value: float(np.nanpercentile(value, 84)),
            )
            .reset_index()
        )
        summary = summary.merge(z_summary, on="start_family", how="left")
    summary.to_csv(out / "map_start_family_summary.csv", index=False)
    selected = frame.loc[frame["is_selected"]].copy()
    if not selected.empty:
        winners = (
            selected.groupby("start_family", dropna=False)
            .size()
            .reset_index(name="n_winning_objects")
        )
        winners.to_csv(out / "map_best_by_start_family.csv", index=False)
        winners.to_csv(out / "map_start_family_winners.csv", index=False)


def _write_map_plots(
    estimates: pd.DataFrame,
    trace: pd.DataFrame,
    out: Path,
    by_start_frame: pd.DataFrame | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not estimates.empty and "z_obs" in estimates:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(estimates["z_obs"].to_numpy(dtype=float), bins=40)
        ax.set_xlabel("MAP z_obs")
        ax.set_ylabel("object count")
        ax.set_title("MAP redshift distribution")
        fig.tight_layout()
        fig.savefig(out / "map_z_distribution.png", dpi=160)
        plt.close(fig)
    if not estimates.empty and {"map_prior_logprob", "map_chi2"}.issubset(estimates):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(
            estimates["map_prior_logprob"].to_numpy(dtype=float),
            np.log10(np.maximum(estimates["map_chi2"].to_numpy(dtype=float), 1.0e-12)),
            s=10,
            alpha=0.6,
        )
        ax.set_xlabel("log p_NF(x_MAP)")
        ax.set_ylabel("log10 chi2")
        ax.set_title("MAP prior density vs photometric chi2")
        fig.tight_layout()
        fig.savefig(out / "map_prior_logprob_vs_chi2.png", dpi=160)
        plt.close(fig)
    if not trace.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        grouped = trace.groupby("iteration", sort=True)["mean_objective"].mean()
        ax.plot(grouped.index.to_numpy(dtype=float), grouped.to_numpy(dtype=float))
        ax.set_xlabel("iteration")
        ax.set_ylabel("mean MAP objective")
        ax.set_title("MAP Adam trace")
        fig.tight_layout()
        fig.savefig(out / "map_optimizer_trace.png", dpi=160)
        plt.close(fig)
    if (
        by_start_frame is not None
        and not by_start_frame.empty
        and {"start_z_obs", "map_chi2", "start_family"}.issubset(by_start_frame)
    ):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for family, group in by_start_frame.groupby("start_family", dropna=False):
            sample = group
            if len(sample) > 3000:
                sample = sample.sample(n=3000, random_state=0)
            ax.scatter(
                sample["start_z_obs"].to_numpy(dtype=float),
                np.log10(np.maximum(sample["map_chi2"].to_numpy(dtype=float), 1.0e-12)),
                s=8,
                alpha=0.35,
                label=str(family),
            )
        ax.set_xlabel("initial z_obs")
        ax.set_ylabel("log10 best chi2 from that start")
        ax.set_title("MAP multistart redshift basin diagnostic")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "map_chi2_vs_start_z.png", dpi=160)
        plt.close(fig)
    if (
        by_start_frame is not None
        and not by_start_frame.empty
        and {"map_z_obs", "start_family", "is_selected"}.issubset(by_start_frame)
    ):
        selected = by_start_frame.loc[by_start_frame["is_selected"]].copy()
        if not selected.empty:
            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            labels = []
            values = []
            for family, group in selected.groupby("start_family", dropna=False):
                vals = group["map_z_obs"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    labels.append(str(family))
                    values.append(vals)
            if values:
                ax.boxplot(values, labels=labels, showfliers=False)
                ax.set_ylabel("selected MAP z_obs")
                ax.set_title("Winning MAP redshift by start family")
                ax.tick_params(axis="x", labelrotation=20)
                fig.tight_layout()
                fig.savefig(out / "map_selected_z_by_start_family.png", dpi=160)
            plt.close(fig)


def _write_map_closure_metrics(out: Path, estimates: pd.DataFrame) -> None:
    truth_path = out / "inference_truth.parquet"
    if estimates.empty or not truth_path.exists() or "z_obs" not in estimates:
        return
    truth = pd.read_parquet(truth_path)
    if "row_index" in estimates and "row_index" in truth:
        merged = estimates.merge(
            truth,
            on="row_index",
            how="inner",
            suffixes=("_map", "_truth"),
        )
    elif "object_id" in estimates and "object_id" in truth:
        merged = estimates.merge(
            truth,
            on="object_id",
            how="inner",
            suffixes=("_map", "_truth"),
        )
    else:
        return
    if merged.empty or "redshift_true" not in merged:
        return
    map_z_column = "z_obs_map" if "z_obs_map" in merged else "z_obs"
    if map_z_column not in merged:
        return
    dz = (
        merged[map_z_column].to_numpy(dtype=float)
        - merged["redshift_true"].to_numpy(dtype=float)
    ) / (1.0 + merged["redshift_true"].to_numpy(dtype=float))
    finite = dz[np.isfinite(dz)]
    if finite.size == 0:
        return
    metrics = {
        "n_objects": int(finite.size),
        "median_bias": float(np.median(finite)),
        "rmse": float(np.sqrt(np.mean(finite**2))),
        "outlier_fraction_0p15": float(np.mean(np.abs(finite) > 0.15)),
    }
    pd.DataFrame([metrics]).to_csv(out / "map_closure_photoz_metrics.csv", index=False)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        merged["redshift_true"].to_numpy(dtype=float),
        merged[map_z_column].to_numpy(dtype=float),
        s=10,
        alpha=0.55,
    )
    lo = float(np.nanmin([merged["redshift_true"].min(), merged[map_z_column].min()]))
    hi = float(np.nanmax([merged["redshift_true"].max(), merged[map_z_column].max()]))
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, alpha=0.6)
    ax.set_xlabel("true redshift")
    ax.set_ylabel("MAP z_obs")
    ax.set_title("Closure MAP redshift")
    fig.tight_layout()
    fig.savefig(out / "closure_photoz_truth_vs_map.png", dpi=160)
    plt.close(fig)
