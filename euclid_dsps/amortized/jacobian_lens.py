"""Physical Jacobian Lens diagnostics for amortized DSPS inference."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
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

from .catalog_identity import select_catalog_row_indices
from .config import amortized_config
from .data import (
    iter_photometry_batches_from_arrays,
    iter_photometry_batches_from_config,
    load_photometry_arrays_from_config,
)
from .decoder import model_flux_from_x
from .elbo import is_deterministic_reconstruction
from .features import make_encoder_features, read_feature_stats
from .latent import LatentSpec, latent_spec_from_config, x_to_theta
from .likelihood import photometric_sigma_eff
from .posterior import posterior_reference_from_base_mean, sample_posterior
from .train import _effective_jax_batch_size, load_checkpoint

RANK_THRESHOLDS = (1.0e-1, 1.0e-2, 1.0e-3)


@dataclass(frozen=True)
class LensTables:
    """Container for per-object J-lens output tables."""

    object_summary: pd.DataFrame
    singular_values: pd.DataFrame
    latent_direction_loadings: pd.DataFrame
    physical_direction_loadings: pd.DataFrame
    band_loadings: pd.DataFrame
    posterior_variance_directions: pd.DataFrame
    prior_score_directions: pd.DataFrame
    ae_singular_values: pd.DataFrame


def decoder_jacobian_lens(
    decoder_flux_from_x: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    latent_spec: LatentSpec,
    obs_flux: jnp.ndarray,
    obs_err: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    likelihood_type: str = "student_t",
    student_t_dof: float = 2.0,
    error_floor_frac: float = 0.02,
    error_jitter: float = 0.0,
    log_std: jnp.ndarray | None = None,
    posterior_covariance: jnp.ndarray | None = None,
    prior: Any | None = None,
    prior_active: bool = False,
) -> dict[str, jnp.ndarray]:
    """Compute local latent-to-flux J-lens arrays for one object.

    The photometric metric follows the same effective sigma convention as the
    training/inference likelihood. The ELBO KL convention is not changed here:
    training still uses ``E_q[logq - logp_beta]``.
    """
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    obs_flux = jnp.asarray(obs_flux, dtype=jnp.float32)
    obs_err = jnp.asarray(obs_err, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=bool)
    model_flux = jnp.asarray(decoder_flux_from_x(x0), dtype=jnp.float32)
    j_flux_x = jax.jacrev(decoder_flux_from_x)(x0)
    sigma_eff = jnp.squeeze(
        photometric_sigma_eff(
            obs_flux[None, :],
            model_flux[None, :],
            obs_err[None, :],
            mask[None, :],
            error_floor_frac=error_floor_frac,
            error_jitter=error_jitter,
        )
    )
    residual = (model_flux - obs_flux) / sigma_eff
    kind = _likelihood_kind(likelihood_type)
    if kind == "student_t":
        weight = (float(student_t_dof) + 1.0) / (
            float(student_t_dof) + residual**2
        )
    else:
        weight = jnp.ones_like(residual)
    finite_band = (
        mask
        & jnp.isfinite(obs_flux)
        & jnp.isfinite(model_flux)
        & jnp.isfinite(sigma_eff)
        & (sigma_eff > 0.0)
    )
    white_scale = jnp.where(
        finite_band,
        jnp.sqrt(jnp.maximum(weight, 0.0)) / sigma_eff,
        0.0,
    )
    j_white = j_flux_x * white_scale[:, None]
    _u, singular_values, vt_full = jnp.linalg.svd(j_white, full_matrices=True)
    fisher = j_white.T @ j_white
    j_theta = jax.jacrev(lambda xx: x_to_theta(xx, latent_spec))(x0)
    physical_dirs = j_theta @ vt_full.T
    if posterior_covariance is not None:
        covariance = jnp.asarray(posterior_covariance, dtype=x0.dtype)
        posterior_var = jnp.einsum("di,ij,dj->d", vt_full, covariance, vt_full)
    elif log_std is not None:
        posterior_var = jnp.sum(
            (vt_full**2) * jnp.exp(2.0 * jnp.asarray(log_std))[None, :],
            axis=1,
        )
    else:
        posterior_var = jnp.full((vt_full.shape[0],), jnp.nan, dtype=x0.dtype)
    if prior is not None:
        prior_score = jax.grad(lambda xx: prior.log_prob(xx[None, :])[0])(x0)
        prior_projection = vt_full @ prior_score
    else:
        prior_score = jnp.full_like(x0, jnp.nan)
        prior_projection = jnp.full((vt_full.shape[0],), jnp.nan, dtype=x0.dtype)
    return {
        "x0": x0,
        "theta0": x_to_theta(x0, latent_spec),
        "model_flux": model_flux,
        "j_flux_x": j_flux_x,
        "j_white": j_white,
        "singular_values": singular_values,
        "vt_full": vt_full,
        "fisher": fisher,
        "j_theta": j_theta,
        "physical_dirs": physical_dirs,
        "posterior_var": posterior_var,
        "prior_score": prior_score,
        "prior_projection": prior_projection,
        "residual": residual,
        "sigma_eff": sigma_eff,
        "finite_band": finite_band,
        "prior_active": jnp.asarray(bool(prior_active)),
    }


def autoencoder_jacobian_lens(
    autoencoder_flux_from_flux: Callable[[jnp.ndarray], jnp.ndarray],
    obs_flux: jnp.ndarray,
    obs_err: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    error_floor_frac: float = 0.02,
    error_jitter: float = 0.0,
) -> dict[str, jnp.ndarray]:
    """Compute flux-in to flux-out Jacobian diagnostics for one object."""
    obs_flux = jnp.asarray(obs_flux, dtype=jnp.float32)
    obs_err = jnp.asarray(obs_err, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=bool)
    model_flux = jnp.asarray(autoencoder_flux_from_flux(obs_flux), dtype=jnp.float32)
    jac = jax.jacrev(autoencoder_flux_from_flux)(obs_flux)
    sigma_out = jnp.squeeze(
        photometric_sigma_eff(
            obs_flux[None, :],
            model_flux[None, :],
            obs_err[None, :],
            mask[None, :],
            error_floor_frac=error_floor_frac,
            error_jitter=error_jitter,
        )
    )
    sigma_in = jnp.squeeze(
        photometric_sigma_eff(
            obs_flux[None, :],
            obs_flux[None, :],
            obs_err[None, :],
            mask[None, :],
            error_floor_frac=error_floor_frac,
            error_jitter=error_jitter,
        )
    )
    finite = mask & jnp.isfinite(sigma_out) & jnp.isfinite(sigma_in)
    finite &= (sigma_out > 0.0) & (sigma_in > 0.0)
    left = jnp.where(finite, 1.0 / sigma_out, 0.0)
    right = jnp.where(finite, sigma_in, 0.0)
    whitened = left[:, None] * jac * right[None, :]
    singular_values = jnp.linalg.svd(whitened, compute_uv=False)
    return {
        "model_flux": model_flux,
        "jacobian": jac,
        "whitened_jacobian": whitened,
        "singular_values": singular_values,
    }


def lens_tables_for_object(
    *,
    object_id: Any,
    row_index: int | None,
    decoder_flux_from_x: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    latent_spec: LatentSpec,
    obs_flux: jnp.ndarray,
    obs_err: jnp.ndarray,
    mask: jnp.ndarray,
    band_names: tuple[str, ...],
    likelihood_config: dict[str, Any],
    log_std: jnp.ndarray | None = None,
    posterior_covariance: jnp.ndarray | None = None,
    prior: Any | None = None,
    prior_active: bool = False,
    direction_top_k: int = 5,
    autoencoder_flux_from_flux: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> LensTables:
    """Return compact tables for one object's J-lens diagnostics."""
    base = _base_identity(object_id, row_index)
    try:
        arrays = decoder_jacobian_lens(
            decoder_flux_from_x,
            x0,
            latent_spec,
            obs_flux,
            obs_err,
            mask,
            likelihood_type=str(likelihood_config.get("type", "student_t")),
            student_t_dof=float(likelihood_config.get("student_t_dof", 2.0)),
            error_floor_frac=float(likelihood_config.get("error_floor_frac", 0.02)),
            error_jitter=float(likelihood_config.get("error_jitter", 0.0)),
            log_std=log_std,
            posterior_covariance=posterior_covariance,
            prior=prior,
            prior_active=prior_active,
        )
        tables = _tables_from_decoder_arrays(
            base=base,
            arrays={key: np.asarray(jax.device_get(value)) for key, value in arrays.items()},
            latent_spec=latent_spec,
            band_names=band_names,
            direction_top_k=int(direction_top_k),
        )
    except Exception as exc:
        return _failure_tables(base, str(exc))
    if autoencoder_flux_from_flux is not None:
        try:
            ae_arrays = autoencoder_jacobian_lens(
                autoencoder_flux_from_flux,
                obs_flux,
                obs_err,
                mask,
                error_floor_frac=float(likelihood_config.get("error_floor_frac", 0.02)),
                error_jitter=float(likelihood_config.get("error_jitter", 0.0)),
            )
            tables = _append_ae_tables(
                tables,
                base,
                {key: np.asarray(jax.device_get(value)) for key, value in ae_arrays.items()},
            )
        except Exception as exc:
            summary = tables.object_summary.copy()
            summary["ae_finite"] = False
            summary["ae_error"] = str(exc)
            tables = LensTables(
                object_summary=summary,
                singular_values=tables.singular_values,
                latent_direction_loadings=tables.latent_direction_loadings,
                physical_direction_loadings=tables.physical_direction_loadings,
                band_loadings=tables.band_loadings,
                posterior_variance_directions=tables.posterior_variance_directions,
                prior_score_directions=tables.prior_score_directions,
                ae_singular_values=tables.ae_singular_values,
            )
    return tables


def run_jacobian_lens_diffsky(
    config: dict[str, Any],
    out_dir: str | Path,
    *,
    checkpoint: str | Path,
    feature_stats_path: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 16,
    row_indices_file: str | Path | None = None,
    selection_mode: str = "sequential",
    stratified_strategy: str = "balanced",
    selection_seed: int = 260617,
    mode: str = "decoder",
    posterior_point: str = "mean",
    posterior_samples: int = 128,
    posterior_seed: int = 260722,
    max_objects: int | None = None,
    direction_top_k: int = 5,
    include_prior_score: bool = True,
    include_ae_lens: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
    resume: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run a sharded Physical J-lens pass over a trained Diffsky model."""
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(posterior_samples) < 2:
        raise ValueError("posterior_samples must be at least 2")
    posterior_point = str(posterior_point).lower()
    if posterior_point not in {"mean", "median"}:
        raise ValueError("posterior_point must be mean or median")
    if int(num_shards) <= 0:
        raise ValueError("num_shards must be positive")
    if int(shard_index) < 0 or int(shard_index) >= int(num_shards):
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    mode = str(mode).lower()
    if mode not in {"decoder", "autoencoder", "both"}:
        raise ValueError("--mode must be decoder, autoencoder, or both")
    include_ae = bool(include_ae_lens or mode in {"autoencoder", "both"})
    include_decoder = mode in {"decoder", "both"}
    if not include_decoder and not include_ae:
        raise ValueError("At least one J-lens mode must be enabled")
    out = ensure_dir(out_dir)
    shard_dir = ensure_dir(out / "jacobian_lens_shards" / f"part_{int(shard_index):06d}")
    if resume and _lens_shard_complete(shard_dir):
        if verbose:
            print(f"[jlens] shard exists; skipping {shard_dir}")
        return _read_json(shard_dir / "shard_summary.json")

    cfg = amortized_config(config)
    inference_cfg = cfg["inference"]
    if feature_stats_path is None:
        feature_stats_path = Path(checkpoint).parent.parent / "feature_stats.json"
    feature_stats = read_feature_stats(feature_stats_path)
    model = load_checkpoint(Path(checkpoint), config)
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
    band_names = tuple(str(band["name"]) for band in config["bands"])
    calibration_runtime_config = {"calibration": config.get("calibration", {}) or {}}
    scale_cfg = global_sed_scale_config(calibration_runtime_config)
    band_cfg = per_band_flux_calibration_config(calibration_runtime_config)
    log_alpha_sed = model.sed_scale.log_alpha_sed if scale_cfg.enabled else 0.0
    log_alpha_band = (
        model.band_calibration.log_alpha_band
        if band_cfg.enabled and model.band_calibration is not None
        else jnp.zeros((len(band_names),), dtype=jnp.float32)
    )
    deterministic = is_deterministic_reconstruction(cfg.get("objective", {}))
    kl_weight_max = float(cfg["training"].get("kl_weight_max", 1.0))
    prior_active = bool(include_prior_score and not deterministic and kl_weight_max > 0.0)
    redshift_bins = inference_cfg.get(
        "redshift_bins",
        ((config.get("amortized", {}) or {}).get("data", {}) or {}).get(
            "redshift_bins",
            None,
        ),
    )
    selected_limit = limit
    if max_objects is not None:
        selected_limit = (
            int(max_objects)
            if selected_limit is None
            else min(int(selected_limit), int(max_objects))
        )
    row_indices, selection_summary = select_catalog_row_indices(
        config,
        limit=selected_limit,
        selection_mode=selection_mode,
        stratified_strategy=stratified_strategy,
        seed=int(selection_seed),
        redshift_bins=redshift_bins,
        row_indices_file=row_indices_file,
    )
    if row_indices is None and int(num_shards) > 1:
        total = int(selection_summary.get("selected_rows", 0))
        row_indices = np.arange(total, dtype=np.int64)
    if row_indices is not None:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        row_indices = row_indices[int(shard_index) :: int(num_shards)]
        selected_limit = None
    if verbose:
        print(
            "[jlens] start "
            f"checkpoint={checkpoint} out={shard_dir} "
            f"mode={mode} rows={len(row_indices) if row_indices is not None else selected_limit} "
            f"shard={int(shard_index)}/{int(num_shards)}"
        )
    jax_batch_size = _effective_jax_batch_size(inference_cfg, int(batch_size))
    if row_indices is not None:
        arrays = load_photometry_arrays_from_config(
            config,
            batch_size=int(inference_cfg.get("catalog_batch_size", 10_000)),
            row_indices=row_indices,
        )
        batches = iter_photometry_batches_from_arrays(
            arrays,
            batch_size=int(jax_batch_size),
            feature_stats=feature_stats,
        )
    else:
        batches = iter_photometry_batches_from_config(
            config,
            batch_size=int(jax_batch_size),
            limit=selected_limit,
            feature_stats=feature_stats,
        )
    started = time.time()
    all_tables: list[LensTables] = []
    for batch_index, batch in enumerate(batches, start=1):
        posterior = sample_posterior(
            model,
            jax.random.fold_in(
                jax.random.PRNGKey(int(posterior_seed)),
                int(batch_index - 1),
            ),
            batch.features,
            int(posterior_samples),
        )
        posterior_x = posterior.x
        posterior_center = (
            jnp.mean(posterior_x, axis=0)
            if posterior_point == "mean"
            else jnp.median(posterior_x, axis=0)
        )
        centered = posterior_x - jnp.mean(posterior_x, axis=0, keepdims=True)
        posterior_covariance = jnp.einsum(
            "sbi,sbj->bij", centered, centered
        ) / float(int(posterior_samples) - 1)
        for local_index in range(int(batch.flux.shape[0])):
            x0 = posterior_center[local_index]
            covariance0 = (
                None if deterministic else posterior_covariance[local_index]
            )
            flux_err0 = batch.flux_err[local_index]

            def decode_x(xx):
                raw = model_flux_from_x(
                    xx,
                    latent_spec,
                    context,
                    model_args,
                    latent_spec.names,
                )
                scaled = (
                    apply_global_sed_scale_to_flux(raw, log_alpha_sed)
                    if scale_cfg.enabled
                    else raw
                )
                return (
                    apply_per_band_flux_calibration_to_flux(scaled, log_alpha_band)
                    if band_cfg.enabled
                    else scaled
                )

            def ae_flux(flux_in, fixed_flux_err=flux_err0):
                features = make_encoder_features(
                    flux_in[None, :],
                    fixed_flux_err[None, :],
                    feature_stats,
                )
                return decode_x(posterior_reference_from_base_mean(model, features)[0])

            all_tables.append(
                lens_tables_for_object(
                    object_id=np.asarray(batch.object_id)[local_index],
                    row_index=(
                        int(np.asarray(batch.row_index)[local_index])
                        if batch.row_index is not None
                        else None
                    ),
                    decoder_flux_from_x=decode_x,
                    x0=x0,
                    latent_spec=latent_spec,
                    obs_flux=batch.flux[local_index],
                    obs_err=batch.flux_err[local_index],
                    mask=batch.mask[local_index],
                    band_names=band_names,
                    likelihood_config=cfg["likelihood"],
                    posterior_covariance=covariance0,
                    prior=model.prior if include_prior_score else None,
                    prior_active=prior_active,
                    direction_top_k=int(direction_top_k),
                    autoencoder_flux_from_flux=ae_flux if include_ae else None,
                )
            )
        if verbose:
            print(f"[jlens] batch {batch_index} done")
    outputs = _write_lens_tables(shard_dir, all_tables)
    summary = {
        "command": "amortized-jacobian-lens-diffsky",
        "out_dir": str(out),
        "shard_dir": str(shard_dir),
        "checkpoint": str(checkpoint),
        "feature_stats_path": str(feature_stats_path),
        "config_catalog_path": str(config.get("catalog_path")),
        "mode": mode,
        "posterior_point": posterior_point,
        "posterior_samples": int(posterior_samples),
        "posterior_seed": int(posterior_seed),
        "posterior_covariance": "empirical_full_transformed_posterior",
        "autoencoder_reference": "flow_pushforward_of_base_mean",
        "include_prior_score": bool(include_prior_score),
        "prior_active": bool(prior_active),
        "include_ae_lens": bool(include_ae),
        "kl_definition": "E_q[logq - logp_beta]",
        "kl_direction": "q_to_p",
        "kl_weight_max": float(kl_weight_max),
        "prior_source": str(cfg["prior"].get("source", "joint_realnvp")),
        "prior_train_jointly": bool(cfg["prior"].get("train_jointly", True)),
        "selection": selection_summary,
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "n_objects": int(sum(len(t.object_summary) for t in all_tables)),
        "latent_names": list(latent_spec.names),
        "band_names": list(band_names),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "git": _git_payload(),
        "started_at_unix": float(started),
        "elapsed_time_s": float(time.time() - started),
        "outputs": outputs,
    }
    write_json(shard_dir / "shard_summary.json", summary)
    write_json(shard_dir / "jacobian_lens_artifact_manifest.json", _artifact_manifest(shard_dir))
    if int(num_shards) == 1:
        finalize_jacobian_lens(out, verbose=verbose)
    return summary


def finalize_jacobian_lens(out_dir: str | Path, *, verbose: bool = True) -> dict[str, Any]:
    """Combine J-lens shards and write global summaries/plots."""
    out = ensure_dir(out_dir)
    shard_root = out / "jacobian_lens_shards"
    shard_dirs = (
        sorted(path for path in shard_root.glob("part_*") if path.is_dir())
        if shard_root.exists()
        else []
    )
    if not shard_dirs and (out / "object_summary.parquet").exists():
        shard_dirs = [out]
    if not shard_dirs:
        raise FileNotFoundError(f"No J-lens shards found under {out}")
    names = [
        "object_summary",
        "singular_values",
        "latent_direction_loadings",
        "physical_direction_loadings",
        "band_loadings",
        "posterior_variance_directions",
        "prior_score_directions",
        "ae_singular_values",
    ]
    row_counts = {}
    for name in names:
        frames = []
        for shard in shard_dirs:
            path = shard / f"{name}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True)
        frame.to_parquet(out / f"{name}.parquet", index=False)
        if name in {"object_summary", "singular_values"}:
            frame.to_csv(out / f"{name}.csv", index=False)
        row_counts[name] = int(len(frame))
    plots = _write_lens_plots(out)
    summary = _global_summary(out, row_counts=row_counts, plots=plots)
    write_json(out / "jacobian_lens_summary.json", summary)
    write_json(out / "jacobian_lens_artifact_manifest.json", _artifact_manifest(out))
    if verbose:
        print(f"[jlens] finalized {out} objects={summary.get('n_objects', 0)}")
    return summary


def _tables_from_decoder_arrays(
    *,
    base: dict[str, Any],
    arrays: dict[str, np.ndarray],
    latent_spec: LatentSpec,
    band_names: tuple[str, ...],
    direction_top_k: int,
) -> LensTables:
    singular = np.asarray(arrays["singular_values"], dtype=float)
    vt_full = np.asarray(arrays["vt_full"], dtype=float)
    physical_dirs = np.asarray(arrays["physical_dirs"], dtype=float)
    posterior_var = np.asarray(arrays["posterior_var"], dtype=float)
    prior_projection = np.asarray(arrays["prior_projection"], dtype=float)
    prior_score = np.asarray(arrays["prior_score"], dtype=float)
    finite_band = np.asarray(arrays["finite_band"], dtype=bool)
    latent_dim = int(vt_full.shape[1])
    n_singular = int(singular.size)
    exact_nullity = max(latent_dim - n_singular, 0)
    rel = _relative_singular_values(singular)
    summary_row = {
        **base,
        "finite": bool(np.all(np.isfinite(singular))),
        "n_bands": int(len(band_names)),
        "n_valid_bands": int(finite_band.sum()),
        "latent_dim": latent_dim,
        "n_singular": n_singular,
        "exact_nullity": int(exact_nullity),
        "condition_number": _condition_number(singular),
        "singular_value_max": _safe_float(singular[0] if singular.size else np.nan),
        "singular_value_min": _safe_float(singular[-1] if singular.size else np.nan),
        "prior_active": bool(np.asarray(arrays["prior_active"]).item()),
        "prior_score_norm": _safe_float(np.linalg.norm(prior_score)),
    }
    for threshold in RANK_THRESHOLDS:
        summary_row[f"effective_rank_{_threshold_label(threshold)}"] = int(
            np.sum(rel >= float(threshold))
        )
    weak = rel < 1.0e-2
    visible = rel >= 1.0e-1
    prior_visible = _norm(prior_projection[:n_singular][visible[:n_singular]])
    prior_weak = _norm(prior_projection[:n_singular][weak[:n_singular]])
    prior_null = _norm(prior_projection[n_singular:])
    prior_total = _norm(prior_projection)
    summary_row.update(
        {
            "prior_score_visible_norm": prior_visible,
            "prior_score_weak_norm": prior_weak,
            "prior_score_exact_null_norm": prior_null,
            "prior_score_null_fraction": (
                prior_null / prior_total if prior_total > 0 else np.nan
            ),
            "posterior_var_visible_median": _nanmedian(posterior_var[:n_singular][visible[:n_singular]]),
            "posterior_var_weak_median": _nanmedian(posterior_var[:n_singular][weak[:n_singular]]),
            "posterior_var_exact_null_median": _nanmedian(posterior_var[n_singular:]),
        }
    )
    singular_rows = []
    for direction_index in range(latent_dim):
        singular_value = singular[direction_index] if direction_index < n_singular else 0.0
        relative = rel[direction_index] if direction_index < n_singular else 0.0
        singular_rows.append(
            {
                **base,
                "direction_index": int(direction_index),
                "direction_kind": _direction_kind(direction_index, rel, n_singular),
                "singular_value": float(singular_value),
                "relative_singular_value": float(relative),
            }
        )
    latent_rows = _top_loading_rows(
        base,
        matrix=vt_full,
        names=tuple(f"x_{name}" for name in latent_spec.names),
        singular=singular,
        rel=rel,
        n_singular=n_singular,
        value_key="latent_coordinate",
        direction_top_k=direction_top_k,
    )
    physical_rows = _top_loading_rows(
        base,
        matrix=physical_dirs.T,
        names=latent_spec.names,
        singular=singular,
        rel=rel,
        n_singular=n_singular,
        value_key="parameter",
        direction_top_k=direction_top_k,
    )
    band_rows = _band_loading_rows(
        base,
        j_white=np.asarray(arrays["j_white"], dtype=float),
        vt_full=vt_full,
        band_names=band_names,
        singular=singular,
        rel=rel,
        direction_top_k=direction_top_k,
    )
    posterior_rows = [
        {
            **base,
            "direction_index": int(index),
            "direction_kind": _direction_kind(index, rel, n_singular),
            "singular_value": float(singular[index]) if index < n_singular else 0.0,
            "relative_singular_value": float(rel[index]) if index < n_singular else 0.0,
            "posterior_variance": _safe_float(value),
        }
        for index, value in enumerate(posterior_var)
    ]
    prior_rows = [
        {
            **base,
            "direction_index": int(index),
            "direction_kind": _direction_kind(index, rel, n_singular),
            "singular_value": float(singular[index]) if index < n_singular else 0.0,
            "relative_singular_value": float(rel[index]) if index < n_singular else 0.0,
            "prior_score_projection": _safe_float(value),
            "abs_prior_score_projection": _safe_float(abs(value)),
        }
        for index, value in enumerate(prior_projection)
    ]
    return LensTables(
        object_summary=pd.DataFrame([summary_row]),
        singular_values=pd.DataFrame(singular_rows),
        latent_direction_loadings=pd.DataFrame(latent_rows),
        physical_direction_loadings=pd.DataFrame(physical_rows),
        band_loadings=pd.DataFrame(band_rows),
        posterior_variance_directions=pd.DataFrame(posterior_rows),
        prior_score_directions=pd.DataFrame(prior_rows),
        ae_singular_values=pd.DataFrame(),
    )


def _append_ae_tables(
    tables: LensTables,
    base: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> LensTables:
    singular = np.asarray(arrays["singular_values"], dtype=float)
    rel = _relative_singular_values(singular)
    ae_rows = [
        {
            **base,
            "ae_direction_index": int(index),
            "ae_singular_value": float(value),
            "ae_relative_singular_value": float(rel[index]) if index < rel.size else 0.0,
        }
        for index, value in enumerate(singular)
    ]
    summary = tables.object_summary.copy()
    summary["ae_finite"] = bool(np.all(np.isfinite(singular)))
    summary["ae_spectral_norm"] = _safe_float(singular[0] if singular.size else np.nan)
    summary["ae_trace"] = _safe_float(np.trace(np.asarray(arrays["whitened_jacobian"])))
    summary["ae_noise_amplification_count"] = int(np.sum(singular > 1.0))
    summary["ae_noise_amplification_max"] = _safe_float(
        np.max(singular[singular > 1.0]) if np.any(singular > 1.0) else np.nan
    )
    summary["ae_copy_like_count"] = int(np.sum((singular >= 0.8) & (singular <= 1.2)))
    summary["ae_denoise_like_count"] = int(np.sum(singular < 0.2))
    for threshold in RANK_THRESHOLDS:
        summary[f"ae_effective_rank_{_threshold_label(threshold)}"] = int(
            np.sum(rel >= float(threshold))
        )
    return LensTables(
        object_summary=summary,
        singular_values=tables.singular_values,
        latent_direction_loadings=tables.latent_direction_loadings,
        physical_direction_loadings=tables.physical_direction_loadings,
        band_loadings=tables.band_loadings,
        posterior_variance_directions=tables.posterior_variance_directions,
        prior_score_directions=tables.prior_score_directions,
        ae_singular_values=pd.DataFrame(ae_rows),
    )


def _write_lens_tables(out: Path, tables: list[LensTables]) -> dict[str, str]:
    names = LensTables.__dataclass_fields__.keys()
    outputs = {}
    for name in names:
        frames = [getattr(item, name) for item in tables if not getattr(item, name).empty]
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        path = out / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        outputs[name] = str(path)
        if name in {"object_summary", "singular_values"}:
            csv_path = out / f"{name}.csv"
            frame.to_csv(csv_path, index=False)
            outputs[f"{name}_csv"] = str(csv_path)
    return outputs


def _write_lens_plots(out: Path) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    outputs: dict[str, str] = {}
    sv_path = out / "singular_values.parquet"
    obj_path = out / "object_summary.parquet"
    if sv_path.exists():
        sv = pd.read_parquet(sv_path)
        if not sv.empty:
            pivot = sv.pivot_table(
                index="object_id",
                columns="direction_index",
                values="singular_value",
                aggfunc="median",
            )
            med = pivot.median(axis=0, skipna=True)
            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            ax.semilogy(med.index.to_numpy(), med.to_numpy(), marker="o")
            ax.set_xlabel("latent direction index")
            ax.set_ylabel("median singular value")
            ax.set_title("J-lens singular spectrum")
            fig.tight_layout()
            path = out / "singular_spectrum_median.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            outputs["singular_spectrum_median"] = str(path)
    if obj_path.exists():
        obj = pd.read_parquet(obj_path)
        rank_col = "effective_rank_1e_2"
        if rank_col in obj:
            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            ax.hist(pd.to_numeric(obj[rank_col], errors="coerce").dropna(), bins=20)
            ax.set_xlabel("effective rank (1e-2)")
            ax.set_ylabel("object count")
            ax.set_title("Photometric effective rank")
            fig.tight_layout()
            path = out / "effective_rank_hist.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            outputs["effective_rank_hist"] = str(path)
        if "ae_spectral_norm" in obj:
            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            ax.hist(
                pd.to_numeric(obj["ae_spectral_norm"], errors="coerce").dropna(),
                bins=20,
            )
            ax.set_xlabel("AE spectral norm")
            ax.set_ylabel("object count")
            ax.set_title("AE noise amplification")
            fig.tight_layout()
            path = out / "ae_noise_amplification_hist.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            outputs["ae_noise_amplification_hist"] = str(path)
    return outputs


def _global_summary(
    out: Path,
    *,
    row_counts: dict[str, int],
    plots: dict[str, str],
) -> dict[str, Any]:
    summary = {
        "command": "amortized-finalize-jacobian-lens",
        "out_dir": str(out),
        "row_counts": row_counts,
        "plots": plots,
        "git": _git_payload(),
    }
    path = out / "object_summary.parquet"
    if path.exists():
        frame = pd.read_parquet(path)
        summary["n_objects"] = int(len(frame))
        for column in (
            "exact_nullity",
            "condition_number",
            "effective_rank_1e_1",
            "effective_rank_1e_2",
            "effective_rank_1e_3",
            "prior_score_null_fraction",
            "ae_spectral_norm",
        ):
            if column in frame:
                values = pd.to_numeric(frame[column], errors="coerce")
                summary[f"{column}_median"] = _safe_float(values.median(skipna=True))
    return summary


def _top_loading_rows(
    base: dict[str, Any],
    *,
    matrix: np.ndarray,
    names: tuple[str, ...],
    singular: np.ndarray,
    rel: np.ndarray,
    n_singular: int,
    value_key: str,
    direction_top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for direction_index, values in enumerate(matrix):
        order = np.argsort(-np.abs(values))[: max(int(direction_top_k), 0)]
        for item_index in order:
            rows.append(
                {
                    **base,
                    "direction_index": int(direction_index),
                    "direction_kind": _direction_kind(direction_index, rel, n_singular),
                    value_key: names[item_index],
                    "loading": _safe_float(values[item_index]),
                    "abs_loading": _safe_float(abs(values[item_index])),
                    "singular_value": (
                        float(singular[direction_index])
                        if direction_index < n_singular
                        else 0.0
                    ),
                    "relative_singular_value": (
                        float(rel[direction_index]) if direction_index < rel.size else 0.0
                    ),
                }
            )
    return rows


def _band_loading_rows(
    base: dict[str, Any],
    *,
    j_white: np.ndarray,
    vt_full: np.ndarray,
    band_names: tuple[str, ...],
    singular: np.ndarray,
    rel: np.ndarray,
    direction_top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    n_singular = int(singular.size)
    for direction_index in range(n_singular):
        band_values = j_white @ vt_full[direction_index]
        order = np.argsort(-np.abs(band_values))[: max(int(direction_top_k), 0)]
        for band_index in order:
            rows.append(
                {
                    **base,
                    "direction_index": int(direction_index),
                    "direction_kind": _direction_kind(direction_index, rel, n_singular),
                    "band": band_names[band_index],
                    "loading": _safe_float(band_values[band_index]),
                    "abs_loading": _safe_float(abs(band_values[band_index])),
                    "singular_value": float(singular[direction_index]),
                    "relative_singular_value": float(rel[direction_index]),
                }
            )
    return rows


def _failure_tables(base: dict[str, Any], error: str) -> LensTables:
    return LensTables(
        object_summary=pd.DataFrame([{**base, "finite": False, "error": error}]),
        singular_values=pd.DataFrame(),
        latent_direction_loadings=pd.DataFrame(),
        physical_direction_loadings=pd.DataFrame(),
        band_loadings=pd.DataFrame(),
        posterior_variance_directions=pd.DataFrame(),
        prior_score_directions=pd.DataFrame(),
        ae_singular_values=pd.DataFrame(),
    )


def _base_identity(object_id: Any, row_index: int | None) -> dict[str, Any]:
    return {
        "object_id": _jsonable_scalar(object_id),
        **({} if row_index is None else {"row_index": int(row_index)}),
    }


def _jsonable_scalar(value: Any) -> int | float | str:
    item = np.asarray(value).item() if np.asarray(value).shape == () else value
    if isinstance(item, (np.integer, int)):
        return int(item)
    if isinstance(item, (np.floating, float)):
        return float(item)
    return str(item)


def _relative_singular_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values[0]) or values[0] <= 0.0:
        return np.zeros_like(values, dtype=float)
    return values / values[0]


def _direction_kind(index: int, rel: np.ndarray, n_singular: int) -> str:
    if int(index) >= int(n_singular):
        return "exact_null"
    if rel.size and rel[int(index)] >= 1.0e-1:
        return "visible_top"
    return "weak"


def _threshold_label(value: float) -> str:
    exponent = int(round(-np.log10(float(value))))
    return f"1e_{exponent}"


def _condition_number(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return float("nan")
    if finite[-1] <= 0.0:
        return float("inf")
    return float(finite[0] / finite[-1])


def _norm(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.linalg.norm(values)) if values.size else 0.0


def _nanmedian(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _likelihood_kind(value: str) -> str:
    kind = str(value).strip().lower().replace("-", "_")
    if kind in {"student", "studentt"}:
        return "student_t"
    if kind in {"gaussian", "normal"}:
        return "gaussian"
    if kind == "student_t":
        return kind
    raise ValueError(f"Unsupported likelihood_type: {value}")


def _lens_shard_complete(path: Path) -> bool:
    required = ("object_summary.parquet", "singular_values.parquet", "shard_summary.json")
    return all((path / name).exists() for name in required)


def _artifact_manifest(out: Path) -> dict[str, Any]:
    files = []
    for path in sorted(out.glob("*")):
        if path.is_file():
            record = {"path": str(path), "bytes": int(path.stat().st_size)}
            if path.suffix == ".parquet":
                try:
                    import pyarrow.parquet as pq

                    record["rows"] = int(pq.ParquetFile(path).metadata.num_rows)
                except Exception:
                    record["rows"] = None
            files.append(record)
    return {"root": str(out), "files": files}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_payload() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ("git", *args),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short"),
    }
