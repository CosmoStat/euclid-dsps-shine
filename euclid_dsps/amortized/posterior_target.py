"""Canonical learned-prior posterior target for amortized and exact inference."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.calibration import (
    apply_global_sed_scale_to_flux,
    apply_per_band_flux_calibration_to_flux,
    global_sed_scale_config,
    per_band_flux_calibration_config,
)

from .decoder import model_flux_from_x
from .latent import theta_to_x, x_to_theta
from .likelihood import photometric_loglike


class PosteriorObservation(NamedTuple):
    """Observed photometry held fixed by every posterior target evaluator."""

    flux: jnp.ndarray
    flux_err: jnp.ndarray
    mask: jnp.ndarray


class PosteriorTargetValues(NamedTuple):
    """Decomposed canonical target values in latent-x density space."""

    logtarget: jnp.ndarray
    loglike: jnp.ndarray
    logprior: jnp.ndarray
    physical_valid: jnp.ndarray
    model_flux: jnp.ndarray
    model_flux_raw: jnp.ndarray


def safe_decoder_inputs(
    x: jnp.ndarray,
    latent_spec: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return finite decoder inputs and the shared physical-support mask.

    Invalid values are replaced only to keep the fixed DSPS decoder numerically
    evaluable. Their target density is still exactly ``-inf``.
    """
    x = jnp.asarray(x)
    theta = x_to_theta(x, latent_spec)
    lower = jnp.asarray(latent_spec.lower, dtype=theta.dtype)
    upper = jnp.asarray(latent_spec.upper, dtype=theta.dtype)
    finite = jnp.all(jnp.isfinite(x), axis=-1)
    finite &= jnp.all(jnp.isfinite(theta), axis=-1)
    in_bounds = jnp.all((theta >= lower) & (theta <= upper), axis=-1)
    valid = finite & in_bounds
    midpoint = 0.5 * (lower + upper)
    sanitized = jnp.where(jnp.isfinite(theta), theta, midpoint)
    margin = jnp.maximum((upper - lower) * 1.0e-6, 1.0e-7)
    safe_theta = jnp.clip(sanitized, lower + margin, upper - margin)
    safe_x = theta_to_x(safe_theta, latent_spec)
    return safe_x, valid


def posterior_log_target(
    model: Any,
    x: jnp.ndarray,
    observation: PosteriorObservation | Any,
    latent_spec: Any,
    context: Any,
    model_args: Any,
    parameter_names: tuple[str, ...],
    likelihood_config: dict[str, Any],
    calibration_config: dict[str, Any] | None = None,
    *,
    model_flux_fn: Callable[..., jnp.ndarray] = model_flux_from_x,
) -> PosteriorTargetValues:
    """Evaluate the one canonical target ``log p(y|x) + log p_eta(x)``.

    Selection probabilities are intentionally absent. For an object already
    selected using the same observed photometry, the catalog normalizer is
    constant across its particles and therefore cancels from normalized RWS/IS
    weights.
    """
    safe_x, physical_valid = safe_decoder_inputs(x, latent_spec)
    model_flux_raw = model_flux_fn(
        safe_x,
        latent_spec,
        context,
        model_args,
        parameter_names,
    )
    model_flux = _apply_model_calibration(
        model,
        model_flux_raw,
        calibration_config or {},
    )
    return posterior_log_target_from_model_flux(
        model,
        x,
        observation,
        latent_spec,
        model_flux,
        likelihood_config,
        physical_valid=physical_valid,
        safe_x=safe_x,
        model_flux_raw=model_flux_raw,
    )


def posterior_log_target_from_model_flux(
    model: Any,
    x: jnp.ndarray,
    observation: PosteriorObservation | Any,
    latent_spec: Any,
    model_flux: jnp.ndarray,
    likelihood_config: dict[str, Any],
    *,
    physical_valid: jnp.ndarray | None = None,
    safe_x: jnp.ndarray | None = None,
    model_flux_raw: jnp.ndarray | None = None,
) -> PosteriorTargetValues:
    """Evaluate the canonical density from an already calibrated model flux.

    This entry point lets memory-bounded inference decode samples in chunks
    while retaining exactly the same support, likelihood and prior contract.
    """
    x = jnp.asarray(x)
    model_flux = jnp.asarray(model_flux)
    if safe_x is None or physical_valid is None:
        safe_x, physical_valid = safe_decoder_inputs(x, latent_spec)
    physical_valid = jnp.asarray(physical_valid, dtype=bool)
    physical_valid &= jnp.all(jnp.isfinite(model_flux), axis=-1)
    flux, flux_err, mask = _observation_arrays(observation)
    loglike = photometric_loglike(
        obs_flux=flux,
        model_flux=model_flux,
        obs_err=flux_err,
        mask=mask,
        likelihood_type=str(likelihood_config.get("type", "student_t")),
        student_t_dof=float(likelihood_config.get("student_t_dof", 2.0)),
        error_floor_frac=float(likelihood_config.get("error_floor_frac", 0.0)),
        error_jitter=float(likelihood_config.get("error_jitter", 0.0)),
    )
    loglike = jnp.where(physical_valid, loglike, -jnp.inf)
    prior_x = jnp.where(physical_valid[..., None], x, safe_x)
    logprior = model.prior.log_prob(prior_x)
    logprior = jnp.where(physical_valid, logprior, -jnp.inf)
    logtarget = jnp.where(physical_valid, loglike + logprior, -jnp.inf)
    raw = model_flux if model_flux_raw is None else jnp.asarray(model_flux_raw)
    return PosteriorTargetValues(
        logtarget=logtarget,
        loglike=loglike,
        logprior=logprior,
        physical_valid=physical_valid,
        model_flux=model_flux,
        model_flux_raw=raw,
    )


def physical_bounds_diagnostics(
    x: Any,
    latent_spec: Any,
) -> dict[str, Any]:
    """Summarize chain samples outside the configured physical fit bounds."""
    values = jnp.asarray(x)
    theta = np.asarray(jax.device_get(x_to_theta(values, latent_spec)), dtype=float)
    lower = np.asarray(latent_spec.lower, dtype=float)
    upper = np.asarray(latent_spec.upper, dtype=float)
    outside = ~np.isfinite(theta) | (theta < lower) | (theta > upper)
    per_parameter = {
        str(name): float(np.mean(outside[..., index]))
        for index, name in enumerate(latent_spec.names)
    }
    global_outside = np.any(outside, axis=-1)
    return {
        "n_samples": int(global_outside.size),
        "fraction_of_samples_outside_fit_bounds": float(np.mean(global_outside)),
        "fraction_outside_fit_bounds_by_parameter": per_parameter,
    }


def _apply_model_calibration(
    model: Any,
    model_flux_raw: jnp.ndarray,
    calibration_config: dict[str, Any],
) -> jnp.ndarray:
    scale_cfg = global_sed_scale_config(calibration_config)
    band_cfg = per_band_flux_calibration_config(calibration_config)
    model_flux = (
        apply_global_sed_scale_to_flux(
            model_flux_raw,
            model.sed_scale.log_alpha_sed,
        )
        if scale_cfg.enabled
        else model_flux_raw
    )
    if band_cfg.enabled and getattr(model, "band_calibration", None) is not None:
        model_flux = apply_per_band_flux_calibration_to_flux(
            model_flux,
            model.band_calibration.log_alpha_band,
        )
    return model_flux


def _observation_arrays(
    observation: PosteriorObservation | Any,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    try:
        return observation.flux, observation.flux_err, observation.mask
    except AttributeError as exc:
        raise TypeError(
            "observation must expose flux, flux_err and mask arrays"
        ) from exc
