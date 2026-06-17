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
from .config import amortized_config
from .data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from .decoder import model_flux_from_x
from .features import read_feature_stats
from .latent import latent_spec_from_config, x_to_theta
from .likelihood import photometric_loglike
from .train import load_checkpoint


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
    selection_mode = str(selection_mode or inference_cfg.get("selection_mode", "sequential"))
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
    scale_cfg = global_sed_scale_config({"calibration": config.get("calibration", {}) or {}})
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
    if verbose:
        print(f"[map-prior] checkpoint: {checkpoint}")
        print(f"[map-prior] output directory: {out}")
        print(
            "[map-prior] run config: "
            f"limit={limit} batch_size={batch_size} n_starts={n_starts} "
            f"maxiter={maxiter} lr={learning_rate} prior_weight={prior_weight}"
        )
    key = jax.random.PRNGKey(int(seed))
    rows = []
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
            "map_grad_norm": np.asarray(jax.device_get(result["grad_norm"])),
        }
        for index, name in enumerate(latent_spec.names):
            row[name] = theta[:, index]
        rows.append(pd.DataFrame(row))
        trace = pd.DataFrame(
            {
                "batch": batch_index,
                "iteration": np.arange(result["trace_objective"].shape[0]),
                "mean_objective": np.asarray(
                    jax.device_get(result["trace_objective"])
                ),
            }
        )
        traces.append(trace)
    estimates = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    estimates.to_parquet(out / "map_estimates.parquet", index=False)
    estimates.to_csv(out / "map_estimates.csv", index=False)
    trace_frame.to_parquet(out / "map_optimizer_trace.parquet", index=False)
    _write_map_plots(estimates, trace_frame, out)
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
        "dataset_label": dataset_label,
    }
    write_json(out / "map_summary.json", summary)
    _write_map_closure_metrics(out, estimates)
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
    likelihood_config: dict[str, Any],
    log_alpha_sed,
    log_alpha_band,
    use_global_scale: bool,
    use_band_calibration: bool,
) -> dict[str, jnp.ndarray]:
    mean, log_std = model.encoder(batch.features)
    n_objects, latent_dim = mean.shape
    n_starts = max(1, int(n_starts))
    eps = jax.random.normal(
        key,
        (n_starts, n_objects, latent_dim),
        dtype=mean.dtype,
    )
    starts = mean[None, :, :] + jnp.exp(log_std)[None, :, :] * eps
    starts = starts.at[0].set(mean)

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
    start_index = jnp.argmin(best_obj, axis=0)
    object_index = jnp.arange(n_objects)
    chosen_x = best_x[start_index, object_index]
    chosen_obj = best_obj[start_index, object_index]
    chosen_nll = best_nll[start_index, object_index]
    chosen_logprior = best_logprior[start_index, object_index]
    chosen_chi2 = best_chi2[start_index, object_index]
    _value, final_grad = value_and_grad(best_x)
    grad_norm = jnp.linalg.norm(final_grad[start_index, object_index], axis=-1)
    return {
        "best_x": chosen_x,
        "best_objective": chosen_obj,
        "best_nll": chosen_nll,
        "best_logprior": chosen_logprior,
        "best_chi2": chosen_chi2,
        "best_start": start_index,
        "grad_norm": grad_norm,
        "trace_objective": jnp.asarray(trace),
    }


def _write_map_plots(
    estimates: pd.DataFrame,
    trace: pd.DataFrame,
    out: Path,
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
    dz = (merged["z_obs"].to_numpy(dtype=float) - merged["redshift_true"].to_numpy(dtype=float)) / (
        1.0 + merged["redshift_true"].to_numpy(dtype=float)
    )
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
