"""Inference entry points for trained amortized FS2 models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.diffsky_redshift_ablation import write_redshift_metrics_for_run
from euclid_dsps.filters import load_filters
from euclid_dsps.io import ensure_dir, write_json
from euclid_dsps.model import dynamic_model_args, load_context

from .catalog import (
    learned_prior_samples_frame,
    posterior_predictive_flux_frame,
    posterior_samples_frame,
    posterior_summary_frame,
)
from .config import amortized_config
from .data import iter_photometry_batches_from_config
from .decoder import model_flux_from_x
from .diagnostics import (
    feature_diagnostics_frame,
    posterior_predictive_residual_frame,
    summarize_inference_outputs,
)
from .features import read_feature_stats
from .latent import latent_spec_from_config, x_to_theta
from .likelihood import photometric_loglike
from .train import _effective_jax_batch_size, load_checkpoint


def infer_amortized_fs2(
    config: dict[str, Any],
    out_dir: Path,
    *,
    checkpoint: Path,
    limit: int | None,
    batch_size: int,
    posterior_samples: int,
    prior_samples: int = 8192,
    seed: int,
    feature_stats_path: Path | None = None,
    decoder_sample_chunk_size: int = 1,
    verbose: bool = True,
    dataset_label: str = "FS2",
) -> None:
    """Run amortized posterior inference for configured catalog rows."""
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(posterior_samples) <= 0:
        raise ValueError("posterior_samples must be positive")
    if int(prior_samples) <= 0:
        raise ValueError("prior_samples must be positive")
    if int(decoder_sample_chunk_size) <= 0:
        raise ValueError("decoder_sample_chunk_size must be positive")
    out = ensure_dir(out_dir)
    cfg = amortized_config(config)
    jax_batch_size = _effective_jax_batch_size(
        cfg.get("inference", {}),
        int(batch_size),
    )
    checkpoint = Path(checkpoint)
    if feature_stats_path is None:
        feature_stats_path = checkpoint.parent.parent / "feature_stats.json"
    if verbose:
        print(f"[amortized] {dataset_label} amortized inference")
        print(f"[amortized] checkpoint: {checkpoint}")
        print(f"[amortized] output directory: {out}")
        print(
            "[amortized] run config: "
            f"limit={limit} batch_size={int(batch_size)} "
            f"posterior_samples={int(posterior_samples)} "
            f"prior_samples={int(prior_samples)} "
            f"decoder_sample_chunk_size={int(decoder_sample_chunk_size)}"
        )
        if jax_batch_size != int(batch_size):
            print(
                "[amortized] capping JAX/DSPS inference batch size: "
                f"requested_batch_size={int(batch_size)} "
                f"jax_batch_size={jax_batch_size}"
            )
        print(f"[amortized] JAX backend: {jax.default_backend()} devices={jax.devices()}")
        print(f"[amortized] feature stats: {feature_stats_path}")
    feature_stats = read_feature_stats(feature_stats_path)
    model = load_checkpoint(checkpoint, config)
    if verbose:
        print("[amortized] loading configured filters...")
    filters = load_filters(config["bands"])
    if verbose:
        print(f"[amortized] loading DSPS context from {config['ssp_path']}...")
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    model_args = dynamic_model_args(context)
    if verbose:
        print(f"[amortized] DSPS context ready: {len(filters)} filters")
    latent_spec = latent_spec_from_config(config)
    likelihood_cfg = cfg["likelihood"]
    key = jax.random.PRNGKey(int(seed))

    sample_frames = []
    summary_frames = []
    predictive_frames = []
    residual_frames = []
    feature_frames = []
    n_objects_total = 0
    band_names = tuple(str(band["name"]) for band in config["bands"])
    for batch_index, batch in enumerate(
        iter_photometry_batches_from_config(
            config,
            batch_size=int(jax_batch_size),
            limit=limit,
            feature_stats=feature_stats,
        ),
        start=1,
    ):
        if verbose:
            print(
                "[amortized] inference batch "
                f"{batch_index}: n_objects={int(batch.flux.shape[0])}"
            )
        key, sample_key = jax.random.split(key)
        x_samples, logq = model.encoder.sample_and_log_prob(
            sample_key,
            batch.features,
            int(posterior_samples),
        )
        theta = x_to_theta(x_samples, latent_spec)
        model_flux = _model_flux_from_x_sample_chunks(
            x_samples,
            latent_spec,
            context,
            model_args,
            latent_spec.names,
            sample_chunk_size=int(decoder_sample_chunk_size),
        )
        logprior = model.prior.log_prob(x_samples)
        loglike = photometric_loglike(
            obs_flux=batch.flux,
            model_flux=model_flux,
            obs_err=batch.flux_err,
            mask=batch.mask,
            likelihood_type=str(likelihood_cfg.get("type", "student_t")),
            student_t_dof=float(likelihood_cfg.get("student_t_dof", 2.0)),
            error_floor_frac=float(likelihood_cfg.get("error_floor_frac", 0.02)),
            error_jitter=float(likelihood_cfg.get("error_jitter", 0.0)),
        )
        chi2 = _posterior_predictive_chi2(batch, model_flux)
        object_id = np.asarray(batch.object_id)
        theta_np = jax.device_get(theta)
        logq_np = jax.device_get(logq)
        logprior_np = jax.device_get(logprior)
        loglike_np = jax.device_get(loglike)
        model_flux_np = jax.device_get(model_flux)
        mask_np = jax.device_get(batch.mask)
        sample_frames.append(
            posterior_samples_frame(
                object_id,
                theta_np,
                latent_spec.names,
                logq_np,
                logprior_np,
                loglike_np,
            )
        )
        summary_frames.append(
            posterior_summary_frame(
                object_id,
                theta_np,
                latent_spec.names,
                loglike_np,
                jax.device_get(chi2),
                mask_np,
            )
        )
        predictive_frames.append(
            posterior_predictive_flux_frame(
                object_id,
                model_flux_np,
                band_names,
            )
        )
        residual_frames.append(
            posterior_predictive_residual_frame(
                object_id,
                jax.device_get(batch.flux),
                jax.device_get(batch.flux_err),
                mask_np,
                model_flux_np,
                band_names,
            )
        )
        feature_frames.append(
            feature_diagnostics_frame(
                object_id,
                jax.device_get(batch.features),
                n_flux_bands=len(band_names),
            )
        )
        n_objects_total += int(batch.flux.shape[0])

    samples = (
        pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    )
    summary = (
        pd.concat(summary_frames, ignore_index=True)
        if summary_frames
        else pd.DataFrame()
    )
    predictive = (
        pd.concat(predictive_frames, ignore_index=True)
        if predictive_frames
        else pd.DataFrame()
    )
    residuals = (
        pd.concat(residual_frames, ignore_index=True)
        if residual_frames
        else pd.DataFrame()
    )
    feature_diagnostics = (
        pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    )
    key, prior_key = jax.random.split(key)
    prior_x = model.prior.sample(prior_key, int(prior_samples))
    prior_theta = x_to_theta(prior_x, latent_spec)
    prior_logprob = model.prior.log_prob(prior_x)
    learned_prior = learned_prior_samples_frame(
        jax.device_get(prior_x),
        jax.device_get(prior_theta),
        latent_spec.names,
        jax.device_get(prior_logprob),
    )
    samples.to_parquet(out / "posterior_samples.parquet", index=False)
    summary.to_parquet(out / "posterior_summary.parquet", index=False)
    predictive.to_parquet(out / "posterior_predictive_flux.parquet", index=False)
    residuals.to_parquet(out / "posterior_predictive_residuals.parquet", index=False)
    feature_diagnostics.to_parquet(out / "feature_diagnostics.parquet", index=False)
    learned_prior.to_parquet(out / "learned_prior_samples.parquet", index=False)
    learned_prior.to_parquet(
        out / "learned_or_loaded_prior_samples.parquet",
        index=False,
    )
    try:
        metric_outputs = {
            key: str(value)
            for key, value in write_redshift_metrics_for_run(
                dataset_path=config["catalog_path"],
                run_dir=out,
                out_dir=out,
                label=dataset_label,
            ).items()
        }
    except Exception as exc:
        metric_outputs = {"warning": str(exc)}
    write_json(
        out / "inference_summary.json",
        {
            "checkpoint": str(checkpoint),
            "feature_stats_path": str(feature_stats_path),
            "limit": limit,
            "batch_size": int(batch_size),
            "jax_batch_size": int(jax_batch_size),
            "posterior_samples": int(posterior_samples),
            "prior_samples": int(prior_samples),
            "decoder_sample_chunk_size": int(decoder_sample_chunk_size),
            "n_objects": int(n_objects_total),
            "samples_rows": int(len(samples)),
            "summary_rows": int(len(summary)),
            "predictive_rows": int(len(predictive)),
            "residual_rows": int(len(residuals)),
            "feature_diagnostics_rows": int(len(feature_diagnostics)),
            "learned_prior_rows": int(len(learned_prior)),
            "prior_source": str(cfg["prior"].get("source", "joint_realnvp")),
            "prior_train_jointly": bool(cfg["prior"].get("train_jointly", True)),
            "metric_outputs": metric_outputs,
        },
    )
    write_json(out / "normalized_config.json", config)
    if verbose:
        print("[amortized] writing posterior predictive diagnostics...")
    summarize_inference_outputs(
        out / "posterior_summary.parquet",
        out,
        config=config,
        limit=limit,
    )
    if verbose:
        print("[amortized] inference complete")
        print(f"[amortized] summary: {out / 'inference_summary.json'}")


def _model_flux_from_x_sample_chunks(
    x_samples: jnp.ndarray,
    latent_spec,
    context,
    model_args,
    parameter_names: tuple[str, ...],
    *,
    sample_chunk_size: int,
) -> jnp.ndarray:
    """Decode posterior samples in small chunks to cap DSPS peak memory."""
    x_samples = jnp.asarray(x_samples)
    if int(sample_chunk_size) <= 0:
        raise ValueError("sample_chunk_size must be positive")
    if x_samples.ndim != 3:
        return model_flux_from_x(
            x_samples,
            latent_spec,
            context,
            model_args,
            parameter_names,
        )
    chunks = []
    for start in range(0, int(x_samples.shape[0]), int(sample_chunk_size)):
        x_chunk = x_samples[start : start + int(sample_chunk_size)]
        flux_chunk = model_flux_from_x(
            x_chunk,
            latent_spec,
            context,
            model_args,
            parameter_names,
        )
        chunks.append(jax.block_until_ready(flux_chunk))
    return jnp.concatenate(chunks, axis=0)


def _posterior_predictive_chi2(batch, model_flux):
    obs = batch.flux[None, :, :]
    err = batch.flux_err[None, :, :]
    mask = batch.mask[None, :, :]
    chi = np.asarray((model_flux - obs) / err)
    valid = np.asarray(mask)
    return np.sum(np.where(valid, chi**2, 0.0), axis=-1)
