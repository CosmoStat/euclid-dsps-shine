"""Negative ELBO objective for joint amortized inference training."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from euclid_dsps.calibration import (
    GlobalSedScaleState,
    PerBandFluxCalibrationState,
    alpha_from_log_alpha,
    apply_global_sed_scale_to_flux,
    apply_per_band_flux_calibration_to_flux,
    global_sed_scale_config,
    global_sed_scale_prior_penalty,
    per_band_flux_calibration_config,
    per_band_flux_calibration_prior_penalty,
)

from .config import require_equinox
from .decoder import mock_model_flux_from_x, model_flux_from_x
from .latent import LatentSpec
from .likelihood import photometric_loglike

eqx = require_equinox()


@dataclass
class AmortizedModel(eqx.Module):
    encoder: object
    prior: object
    sed_scale: GlobalSedScaleState
    band_calibration: PerBandFluxCalibrationState | None = None


def negative_elbo(
    model: AmortizedModel,
    batch,
    latent_spec: LatentSpec,
    context,
    model_args,
    parameter_names: tuple[str, ...],
    key,
    n_samples: int,
    kl_weight: float,
    likelihood_config: dict,
    calibration_config: dict | None = None,
    use_mock_decoder: bool = False,
    mock_decoder_params=None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return the Monte Carlo negative ELBO and diagnostic metrics.

    ``logq`` and ``logp`` are exact pointwise. The KL is estimated by Monte
    Carlo as ``E_q[logq - logp]`` because the RealNVP prior has nonlinear
    coupling networks, so the expectation has no Gaussian closed form.
    """
    x_samples, logq = model.encoder.sample_and_log_prob(
        key,
        batch.features,
        int(n_samples),
    )
    if use_mock_decoder:
        if mock_decoder_params is None:
            raise ValueError(
                "mock_decoder_params is required with use_mock_decoder=True"
            )
        model_flux_raw = mock_model_flux_from_x(
            x_samples,
            mock_decoder_params["weights"],
            mock_decoder_params["bias"],
        )
    else:
        model_flux_raw = model_flux_from_x(
            x_samples,
            latent_spec,
            context,
            model_args,
            parameter_names,
        )
    scale_cfg = global_sed_scale_config(calibration_config)
    band_cfg = per_band_flux_calibration_config(calibration_config)
    log_alpha_sed = model.sed_scale.log_alpha_sed
    model_flux = (
        apply_global_sed_scale_to_flux(model_flux_raw, log_alpha_sed)
        if scale_cfg.enabled
        else model_flux_raw
    )
    log_alpha_band = (
        model.band_calibration.log_alpha_band
        if band_cfg.enabled and model.band_calibration is not None
        else jnp.zeros((model_flux_raw.shape[-1],), dtype=model_flux_raw.dtype)
    )
    model_flux = (
        apply_per_band_flux_calibration_to_flux(model_flux, log_alpha_band)
        if band_cfg.enabled
        else model_flux
    )
    alpha_prior_penalty = (
        global_sed_scale_prior_penalty(
            log_alpha_sed,
            scale_cfg.prior_sigma_log_alpha,
        )
        if scale_cfg.enabled
        else jnp.asarray(0.0, dtype=model_flux.dtype)
    )
    band_prior_penalty = (
        per_band_flux_calibration_prior_penalty(
            log_alpha_band,
            band_cfg.prior_sigma_log_alpha,
        )
        if band_cfg.enabled
        else jnp.asarray(0.0, dtype=model_flux.dtype)
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
    logp = model.prior.log_prob(x_samples)
    kl_mc = logq - logp
    loss_terms = -loglike + float(kl_weight) * kl_mc
    loss = jnp.mean(loss_terms) + alpha_prior_penalty + band_prior_penalty
    obs = batch.flux[None, :, :]
    mask = batch.mask[None, :, :]
    residual = jnp.where(mask, model_flux - obs, 0.0)
    n_valid = jnp.maximum(jnp.sum(mask), 1)
    metrics = {
        "loss": loss,
        "negative_loglike": jnp.mean(-loglike),
        "loglike_mean": jnp.mean(loglike),
        "logprior_mean": jnp.mean(logp),
        "logq_mean": jnp.mean(logq),
        "kl_mc_mean": jnp.mean(kl_mc),
        "model_flux_mean": jnp.mean(model_flux),
        "mean_model_flux_raw": jnp.mean(model_flux_raw),
        "mean_model_flux_scaled": jnp.mean(model_flux),
        "log_alpha_sed": log_alpha_sed,
        "alpha_sed": alpha_from_log_alpha(log_alpha_sed),
        "alpha_prior_penalty": alpha_prior_penalty,
        "band_alpha_prior_penalty": band_prior_penalty,
        "max_abs_band_delta_mag": jnp.max(
            jnp.abs(-2.5 * log_alpha_band / jnp.log(jnp.asarray(10.0)))
        ),
        "residual_rms": jnp.sqrt(jnp.sum(residual**2) / n_valid),
        "finite_fraction": jnp.mean(jnp.isfinite(loss_terms).astype(jnp.float32)),
    }
    return loss, metrics
