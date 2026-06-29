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
from .likelihood import photometric_loglike, photometric_normalized_residual

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
    objective_config: dict | None = None,
    use_mock_decoder: bool = False,
    mock_decoder_params=None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return the Monte Carlo negative ELBO and diagnostic metrics.

    ``logq`` and ``logp`` are exact pointwise. The KL is estimated by Monte
    Carlo as ``E_q[logq - logp]`` because the RealNVP prior has nonlinear
    coupling networks, so the expectation has no Gaussian closed form.
    """
    objective_config = objective_config or {}
    deterministic = is_deterministic_reconstruction(objective_config)
    mean, log_std = model.encoder(batch.features)
    if deterministic:
        x_samples = mean[None, ...]
        logq = jnp.zeros(mean.shape[:-1], dtype=mean.dtype)[None, ...]
    else:
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
    logp = (
        jnp.zeros_like(logq)
        if deterministic
        else model.prior.log_prob(x_samples)
    )
    kl_mc = logq - logp
    likelihood_temperature = jnp.asarray(
        max(float(objective_config.get("likelihood_temperature", 1.0)), 1.0e-6),
        dtype=loglike.dtype,
    )
    entropy_penalty = (
        jnp.asarray(0.0, dtype=loglike.dtype)
        if deterministic
        else _posterior_entropy_floor_penalty(log_std, objective_config)
    )
    loss_terms = -loglike / likelihood_temperature + float(kl_weight) * kl_mc
    loss = (
        jnp.mean(loss_terms)
        + alpha_prior_penalty
        + band_prior_penalty
        + entropy_penalty
    )
    obs = batch.flux[None, :, :]
    mask = batch.mask[None, :, :]
    residual = jnp.where(mask, model_flux - obs, 0.0)
    n_valid = jnp.maximum(jnp.sum(mask), 1)
    chi = photometric_normalized_residual(
        obs_flux=batch.flux,
        model_flux=model_flux,
        obs_err=batch.flux_err,
        mask=batch.mask,
        error_floor_frac=float(likelihood_config.get("error_floor_frac", 0.02)),
        error_jitter=float(likelihood_config.get("error_jitter", 0.0)),
    )
    n_valid_residual = jnp.maximum(jnp.sum(mask) * x_samples.shape[0], 1)
    metrics = {
        "loss": loss,
        "negative_loglike": jnp.mean(-loglike),
        "loglike_mean": jnp.mean(loglike),
        "logprior_mean": jnp.mean(logp),
        "logq_mean": jnp.mean(logq),
        "kl_mc_mean": jnp.mean(kl_mc),
        "likelihood_temperature": likelihood_temperature,
        "entropy_floor_penalty": entropy_penalty,
        "posterior_entropy_mean": (
            jnp.asarray(0.0, dtype=loglike.dtype)
            if deterministic
            else _diag_gaussian_entropy(log_std).mean()
        ),
        "posterior_min_log_std": (
            jnp.asarray(0.0, dtype=loglike.dtype)
            if deterministic
            else jnp.min(log_std)
        ),
        "posterior_median_log_std": (
            jnp.asarray(0.0, dtype=loglike.dtype)
            if deterministic
            else jnp.median(log_std)
        ),
        "posterior_max_log_std": (
            jnp.asarray(0.0, dtype=loglike.dtype)
            if deterministic
            else jnp.max(log_std)
        ),
        "deterministic_reconstruction": jnp.asarray(
            1.0 if deterministic else 0.0,
            dtype=loglike.dtype,
        ),
        "effective_n_samples": jnp.asarray(x_samples.shape[0], dtype=loglike.dtype),
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
        "residual_rms": jnp.sqrt(jnp.sum(chi**2) / n_valid_residual),
        "flux_residual_rms": jnp.sqrt(jnp.sum(residual**2) / n_valid),
        "finite_fraction": jnp.mean(jnp.isfinite(loss_terms).astype(jnp.float32)),
    }
    return loss, metrics


def objective_mode(objective_config: dict | None) -> str:
    """Return normalized amortized objective mode."""
    raw = str((objective_config or {}).get("mode", "stochastic_elbo"))
    mode = raw.strip().lower().replace("-", "_")
    aliases = {
        "stochastic": "stochastic_elbo",
        "elbo": "stochastic_elbo",
        "stochastic_elbo": "stochastic_elbo",
        "deterministic": "deterministic_reconstruction",
        "autoencoder": "deterministic_reconstruction",
        "deterministic_autoencoder": "deterministic_reconstruction",
        "deterministic_reconstruction": "deterministic_reconstruction",
    }
    if mode not in aliases:
        raise ValueError(
            "amortized.objective.mode must be 'stochastic_elbo' or "
            "'deterministic_reconstruction'"
        )
    return aliases[mode]


def is_deterministic_reconstruction(objective_config: dict | None) -> bool:
    return objective_mode(objective_config) == "deterministic_reconstruction"


def _diag_gaussian_entropy(log_std: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * jnp.sum(
        1.0 + jnp.log(2.0 * jnp.pi) + 2.0 * log_std,
        axis=-1,
    )


def _posterior_entropy_floor_penalty(
    log_std: jnp.ndarray,
    objective_config: dict,
) -> jnp.ndarray:
    regularization = dict(objective_config.get("posterior_regularization", {}) or {})
    enabled = bool(regularization.get("entropy_floor_enabled", False))
    weight = float(regularization.get("weight", 0.0))
    if not enabled or weight <= 0.0:
        return jnp.asarray(0.0, dtype=log_std.dtype)
    min_log_std = regularization.get("min_log_std")
    if min_log_std is None:
        min_scale = float(regularization.get("min_scale", 0.0))
        if min_scale <= 0.0:
            return jnp.asarray(0.0, dtype=log_std.dtype)
        min_log_std = float(jnp.log(jnp.asarray(min_scale)))
    deficit = jax_nn_relu(jnp.asarray(float(min_log_std), dtype=log_std.dtype) - log_std)
    return jnp.asarray(weight, dtype=log_std.dtype) * jnp.mean(deficit**2)


def jax_nn_relu(value: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(value, jnp.asarray(0.0, dtype=value.dtype))
