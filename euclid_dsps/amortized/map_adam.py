"""MAP-Adam inference under a learned amortized RealNVP prior."""

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
from .collapse_gates import write_inference_collapse_gate
from .config import amortized_config
from .data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from .decoder import model_flux_from_x
from .features import read_feature_stats
from .latent import latent_spec_from_config, theta_to_x, x_to_theta
from .likelihood import photometric_loglike
from .train import load_checkpoint
from .truth_diagnostics import write_extended_truth_diagnostics


def run_map_adam_under_prior(
    config: dict[str, Any],
    out_dir: Path,
    *,
    checkpoint: Path,
    feature_stats_path: Path | None,
    limit: int | None,
    batch_size: int,
    n_starts: int,
    maxiter: int,
    learning_rate: float,
    prior_weight: float,
    seed: int,
    start_mode: str = "encoder",
    start_chunk_size: int | None = None,
    selection_mode: str | None = None,
    stratified_strategy: str | None = None,
    selection_seed: int | None = None,
    dataset_label: str = "Diffsky HLTDS",
    verbose: bool = True,
) -> dict[str, Any]:
    """Run per-object MAP optimization with the learned NF prior."""
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
    checkpoint = Path(checkpoint)
    if feature_stats_path is None:
        feature_stats_path = checkpoint.parent.parent / "feature_stats.json"
    feature_stats = read_feature_stats(feature_stats_path)
    arrays = load_photometry_arrays_from_config(
        config,
        batch_size=int(inference_cfg.get("catalog_batch_size", 10_000)),
        limit=limit if row_indices is None else None,
        row_indices=row_indices,
    )
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
    latent_spec = latent_spec_from_config(config)
    model = load_checkpoint(checkpoint, config)
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
    if verbose:
        print(f"[map-prior] checkpoint: {checkpoint}")
        print(f"[map-prior] output directory: {out}")
        print(
            "[map-prior] run config: "
            f"limit={limit} batch_size={batch_size} n_starts={n_starts} "
            f"maxiter={maxiter} lr={learning_rate} prior_weight={prior_weight} "
            f"start_mode={start_mode} start_chunk_size={start_chunk_size}"
        )
    key = jax.random.PRNGKey(int(seed))
    rows = []
    rows_by_start = []
    traces = []
    for batch_index, batch in enumerate(
        iter_photometry_batches_from_arrays(
            arrays,
            batch_size=int(batch_size),
            feature_stats=feature_stats,
        ),
        start=1,
    ):
        if verbose:
            print(f"[map-prior] batch {batch_index}: n_objects={batch.flux.shape[0]}")
        key, batch_key = jax.random.split(key)
        result = _map_adam_batch(
            model,
            batch,
            latent_spec,
            context,
            model_args,
            latent_spec.names,
            batch_key,
            n_starts=int(n_starts),
            maxiter=int(maxiter),
            learning_rate=float(learning_rate),
            prior_weight=float(prior_weight),
            start_mode=str(start_mode),
            start_chunk_size=int(start_chunk_size),
            likelihood_config=cfg["likelihood"],
            log_alpha_sed=log_alpha_sed,
            log_alpha_band=log_alpha_band,
            use_global_scale=bool(scale_cfg.enabled),
            use_band_calibration=bool(band_cfg.enabled),
        )
        theta = np.asarray(jax.device_get(x_to_theta(result["best_x"], latent_spec)))
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
            "map_chi2": np.asarray(jax.device_get(result["best_chi2"])),
            "map_start_index": np.asarray(jax.device_get(result["best_start"])),
            "map_start_family": np.asarray(result["start_family"], dtype=object)[
                np.asarray(jax.device_get(result["best_start"]))
            ],
            "map_grad_norm": np.asarray(jax.device_get(result["grad_norm"])),
        }
        for index, name in enumerate(latent_spec.names):
            row[name] = theta[:, index]
        rows.append(pd.DataFrame(row))
        by_start = _map_by_start_frame(
            result,
            batch,
            latent_spec,
        )
        if not by_start.empty:
            by_start["batch"] = batch_index
            rows_by_start.append(by_start)
        trace = pd.DataFrame(
            {
                "batch": batch_index,
                "iteration": np.arange(result["trace_objective"].shape[0]),
                "mean_objective": np.asarray(jax.device_get(result["trace_objective"])),
            }
        )
        traces.append(trace)
    estimates = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    by_start_frame = (
        pd.concat(rows_by_start, ignore_index=True) if rows_by_start else pd.DataFrame()
    )
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    estimates.to_parquet(out / "map_estimates.parquet", index=False)
    estimates.to_csv(out / "map_estimates.csv", index=False)
    if not by_start_frame.empty:
        by_start_frame.to_parquet(out / "map_estimates_by_start.parquet", index=False)
        by_start_frame.to_csv(out / "map_estimates_by_start.csv", index=False)
        _write_start_family_summary(by_start_frame, out)
    trace_frame.to_parquet(out / "map_optimizer_trace.parquet", index=False)
    _write_map_plots(estimates, trace_frame, out, by_start_frame)
    summary = {
        "checkpoint": str(checkpoint),
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
        "start_mode": str(start_mode),
        "start_chunk_size": int(start_chunk_size),
        "dataset_label": dataset_label,
    }
    write_json(out / "map_summary.json", summary)
    _write_map_closure_metrics(out, estimates)
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
    start_mode: str,
    start_chunk_size: int,
    likelihood_config: dict[str, Any],
    log_alpha_sed,
    log_alpha_band,
    use_global_scale: bool,
    use_band_calibration: bool,
) -> dict[str, jnp.ndarray]:
    mean, log_std = model.encoder(batch.features)
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
    likelihood_config: dict[str, Any],
    log_alpha_sed,
    log_alpha_band,
    use_global_scale: bool,
    use_band_calibration: bool,
) -> dict[str, jnp.ndarray]:
    def metrics_for_x(x):
        model_flux_raw = model_flux_from_x(
            x,
            latent_spec,
            context,
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
            obs_flux=batch.flux,
            model_flux=model_flux,
            obs_err=batch.flux_err,
            mask=batch.mask,
            likelihood_type=str(likelihood_config.get("type", "student_t")),
            student_t_dof=float(likelihood_config.get("student_t_dof", 2.0)),
            error_floor_frac=float(likelihood_config.get("error_floor_frac", 0.02)),
            error_jitter=float(likelihood_config.get("error_jitter", 0.0)),
        )
        logprior = model.prior.log_prob(x)
        chi = (model_flux - batch.flux[None, :, :]) / batch.flux_err[None, :, :]
        chi = jnp.where(batch.mask[None, :, :], chi, 0.0)
        chi2 = jnp.sum(chi**2, axis=-1)
        objective = -loglike - float(prior_weight) * logprior
        return objective, -loglike, logprior, chi2

    def scalar_objective(x):
        objective, _nll, _logprior, _chi2 = metrics_for_x(x)
        return jnp.mean(objective)

    value_and_grad = jax.value_and_grad(scalar_objective)
    x = starts
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    best_x = x
    best_obj, best_nll, best_logprior, best_chi2 = metrics_for_x(x)
    trace = []
    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1.0e-8
    for iteration in range(1, int(maxiter) + 1):
        value, grad = value_and_grad(x)
        trace.append(value)
        obj, nll, logprior, chi2 = metrics_for_x(x)
        improved = obj < best_obj
        best_x = jnp.where(improved[..., None], x, best_x)
        best_obj = jnp.where(improved, obj, best_obj)
        best_nll = jnp.where(improved, nll, best_nll)
        best_logprior = jnp.where(improved, logprior, best_logprior)
        best_chi2 = jnp.where(improved, chi2, best_chi2)
        grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)
        m_hat = m / (1.0 - beta1**iteration)
        v_hat = v / (1.0 - beta2**iteration)
        x = x - float(learning_rate) * m_hat / (jnp.sqrt(v_hat) + eps_adam)
    _value, final_grad = value_and_grad(best_x)
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


def _effective_start_chunk_size(value: int | None, n_starts: int) -> int:
    if value is None:
        return 1
    chunk = int(value)
    if chunk <= 0:
        raise ValueError("MAP start_chunk_size must be positive")
    return max(1, min(chunk, int(n_starts)))


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
        merged = estimates.merge(truth, on="row_index", how="inner")
    elif "object_id" in estimates and "object_id" in truth:
        merged = estimates.merge(truth, on="object_id", how="inner")
    else:
        return
    if merged.empty or "redshift_true" not in merged:
        return
    dz = (
        merged["z_obs"].to_numpy(dtype=float)
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
        merged["z_obs"].to_numpy(dtype=float),
        s=10,
        alpha=0.55,
    )
    lo = float(np.nanmin([merged["redshift_true"].min(), merged["z_obs"].min()]))
    hi = float(np.nanmax([merged["redshift_true"].max(), merged["z_obs"].max()]))
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, alpha=0.6)
    ax.set_xlabel("true redshift")
    ax.set_ylabel("MAP z_obs")
    ax.set_title("Closure MAP redshift")
    fig.tight_layout()
    fig.savefig(out / "closure_photoz_truth_vs_map.png", dpi=160)
    plt.close(fig)
