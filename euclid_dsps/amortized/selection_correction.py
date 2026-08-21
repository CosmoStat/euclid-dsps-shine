"""Differentiable normalization for observed-flux selected samples."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, logsumexp

from euclid_dsps.photometry import abmag_to_fnu_cgs_jax


def observed_magnitude_flux_limit_jax(max_mag_ab: Any) -> jnp.ndarray:
    """Convert an observed AB-magnitude upper limit to an fnu-cgs threshold."""
    return jnp.asarray(abmag_to_fnu_cgs_jax(jnp.asarray(max_mag_ab)))


def observed_flux_selection_log_beta(
    model_flux: Any,
    flux_error: Any,
    flux_limit: Any,
) -> jnp.ndarray:
    """Return ``log P(noisy_flux > flux_limit | model_flux)`` for Gaussian noise."""
    model_flux_array = jnp.asarray(model_flux)
    if not jnp.issubdtype(model_flux_array.dtype, jnp.inexact):
        model_flux_array = model_flux_array.astype(jnp.float32)
    error_array = jnp.asarray(flux_error, dtype=model_flux_array.dtype)
    limit_array = jnp.asarray(flux_limit, dtype=model_flux_array.dtype)
    valid = jnp.isfinite(model_flux_array)
    valid &= jnp.isfinite(error_array) & (error_array > 0.0)
    valid &= jnp.isfinite(limit_array)
    unit = jnp.maximum(jnp.abs(model_flux_array), jnp.abs(limit_array))
    unit = jnp.maximum(unit, jnp.abs(error_array))
    unit = jax.lax.stop_gradient(
        jnp.maximum(unit, jnp.asarray(1.0e-30, dtype=model_flux_array.dtype))
    )
    model_scaled = model_flux_array / unit
    limit_scaled = limit_array / unit
    error_scaled = error_array / unit
    safe_error = jnp.where(valid, error_scaled, jnp.ones_like(error_scaled))
    z = (model_scaled - limit_scaled) / safe_error
    return jnp.where(valid, log_ndtr(z), -jnp.inf)


def observed_flux_selection_beta(
    model_flux: Any,
    flux_error: Any,
    flux_limit: Any,
) -> jnp.ndarray:
    """Return the Gaussian observed-flux selection probability ``beta(x)``."""
    return jnp.exp(observed_flux_selection_log_beta(model_flux, flux_error, flux_limit))


def selection_log_alpha_from_log_beta(
    log_beta: Any,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return a stable log-mean selection probability and MC diagnostics."""
    values = jnp.ravel(jnp.asarray(log_beta))
    if values.size == 0:
        raise ValueError("Selection alpha estimation requires at least one sample")
    n_samples = int(values.size)
    log_alpha = logsumexp(values) - jnp.log(jnp.asarray(n_samples, dtype=values.dtype))
    beta = jnp.exp(values)
    beta_mean = jnp.mean(beta)
    centered = beta - beta_mean
    variance = (
        jnp.sum(centered**2) / float(n_samples - 1)
        if n_samples > 1
        else jnp.asarray(0.0, dtype=values.dtype)
    )
    mc_error = jnp.sqrt(variance / float(n_samples))
    metrics = {
        "selection/enabled": jnp.asarray(1.0, dtype=values.dtype),
        "selection/evaluated": jnp.asarray(1.0, dtype=values.dtype),
        "selection/alpha": jnp.exp(log_alpha),
        "selection/log_alpha": log_alpha,
        "selection/beta_mean": beta_mean,
        "selection/beta_min": jnp.min(beta),
        "selection/beta_max": jnp.max(beta),
        "selection/beta_q05": jnp.quantile(beta, 0.05),
        "selection/beta_q50": jnp.quantile(beta, 0.50),
        "selection/beta_q95": jnp.quantile(beta, 0.95),
        "selection/beta_transition_fraction": jnp.mean(
            ((beta > 0.1) & (beta < 0.9)).astype(values.dtype)
        ),
        "selection/n_prior_samples": jnp.asarray(n_samples, dtype=values.dtype),
        "selection/alpha_mc_error": mc_error,
        "selection/alpha_mc_relative_error": mc_error
        / jnp.maximum(beta_mean, jnp.asarray(1.0e-12, dtype=values.dtype)),
    }
    return log_alpha, metrics


def estimate_log_alpha_reparameterized(
    prior: Any,
    key: jax.Array,
    *,
    n_prior_samples: int,
    log_beta_fn: Callable[[jnp.ndarray], jnp.ndarray],
    prior_sample_batch_size: int | None = None,
    dtype: Any = jnp.float32,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Estimate ``log alpha`` from independent reparameterized prior draws.

    Base-normal draws are transformed with ``prior.forward`` rather than
    ``prior.sample`` so gradients explicitly reach the prior-flow parameters.
    """
    count = int(n_prior_samples)
    if count <= 0:
        raise ValueError("n_prior_samples must be positive")
    batch_size = (
        count
        if prior_sample_batch_size is None
        else min(int(prior_sample_batch_size), count)
    )
    if batch_size <= 0:
        raise ValueError("prior_sample_batch_size must be positive")
    n_batches = (count + batch_size - 1) // batch_size
    padded_count = n_batches * batch_size
    base = jax.random.normal(
        key,
        (padded_count, int(prior.latent_dim)),
        dtype=dtype,
    )

    def transform_and_score(base_batch):
        prior_samples, _logdet = prior.forward(base_batch)
        return log_beta_fn(prior_samples)

    batched_base = base.reshape(
        n_batches,
        batch_size,
        int(prior.latent_dim),
    )
    log_beta = jax.lax.map(jax.checkpoint(transform_and_score), batched_base)
    log_alpha, metrics = selection_log_alpha_from_log_beta(jnp.ravel(log_beta)[:count])
    metrics = dict(metrics)
    metrics["selection/prior_sample_batch_size"] = jnp.asarray(
        batch_size, dtype=log_alpha.dtype
    )
    return log_alpha, metrics


def disabled_selection_metrics(dtype: Any = jnp.float32) -> dict[str, jnp.ndarray]:
    """Return neutral selection metrics when the correction is disabled."""
    zero = jnp.asarray(0.0, dtype=dtype)
    return {
        "selection/enabled": zero,
        "selection/evaluated": zero,
        "selection/alpha": jnp.asarray(1.0, dtype=dtype),
        "selection/log_alpha": zero,
        "selection/beta_mean": jnp.asarray(1.0, dtype=dtype),
        "selection/beta_min": jnp.asarray(1.0, dtype=dtype),
        "selection/beta_max": jnp.asarray(1.0, dtype=dtype),
        "selection/beta_q05": jnp.asarray(1.0, dtype=dtype),
        "selection/beta_q50": jnp.asarray(1.0, dtype=dtype),
        "selection/beta_q95": jnp.asarray(1.0, dtype=dtype),
        "selection/beta_transition_fraction": zero,
        "selection/n_prior_samples": zero,
        "selection/prior_sample_batch_size": zero,
        "selection/alpha_mc_error": zero,
        "selection/alpha_mc_relative_error": zero,
    }
