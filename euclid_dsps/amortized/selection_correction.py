"""Differentiable normalization for observed-flux selected samples."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, logsumexp

from euclid_dsps.photometry import abmag_to_fnu_cgs_jax

_MAX_SCORE_FUNCTION_BATCH_SIZE = 64


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
    finite_model_abs = jnp.where(
        jnp.isfinite(model_flux_array),
        jnp.abs(model_flux_array),
        jnp.zeros_like(model_flux_array),
    )
    finite_limit_abs = jnp.where(
        jnp.isfinite(limit_array),
        jnp.abs(limit_array),
        jnp.zeros_like(limit_array),
    )
    finite_error_abs = jnp.where(
        jnp.isfinite(error_array),
        jnp.abs(error_array),
        jnp.zeros_like(error_array),
    )
    unit = jnp.maximum(finite_model_abs, finite_limit_abs)
    unit = jnp.maximum(unit, finite_error_abs)
    unit = jax.lax.stop_gradient(
        jnp.maximum(unit, jnp.asarray(1.0e-30, dtype=model_flux_array.dtype))
    )
    model_scaled = model_flux_array / unit
    limit_scaled = limit_array / unit
    error_scaled = error_array / unit
    safe_model = jnp.where(valid, model_scaled, jnp.zeros_like(model_scaled))
    safe_limit = jnp.where(valid, limit_scaled, jnp.zeros_like(limit_scaled))
    safe_error = jnp.where(valid, error_scaled, jnp.ones_like(error_scaled))
    z = (safe_model - safe_limit) / safe_error
    return jnp.where(valid, log_ndtr(z), -jnp.inf)


def observed_flux_selection_beta(
    model_flux: Any,
    flux_error: Any,
    flux_limit: Any,
) -> jnp.ndarray:
    """Return the Gaussian observed-flux selection probability ``beta(x)``."""
    return jnp.exp(observed_flux_selection_log_beta(model_flux, flux_error, flux_limit))


def observed_flux_selection_log_beta_gaussian_m5(
    model_flux: Any,
    flux_limit: Any,
    m5: Any,
    gamma: Any,
    *,
    sigma_sys_mag: float = 0.0,
    min_sigma_fnu_cgs: float = 1.0e-40,
) -> jnp.ndarray:
    """Return Gaussian-m5 completeness without cgs-scale autodiff adjoints.

    This is algebraically the same PhotoErr model used by
    :func:`m5_depth_flux_error_jax`, but flux, threshold, depth and noise are
    kept in one detached common unit through ``log_ndtr``. The fused form
    avoids multiplying reverse-mode sensitivities near ``1e30`` by DSPS flux
    derivatives near ``1e-30``.
    """
    flux = jnp.asarray(model_flux)
    if not jnp.issubdtype(flux.dtype, jnp.inexact):
        flux = flux.astype(jnp.float32)
    dtype = flux.dtype
    limit = jnp.asarray(flux_limit, dtype=dtype)
    m5_array = jnp.asarray(m5, dtype=dtype)
    gamma_array = jnp.asarray(gamma, dtype=dtype)
    f5 = jnp.asarray(abmag_to_fnu_cgs_jax(m5_array), dtype=dtype)
    finite_flux_abs = jnp.where(
        jnp.isfinite(flux),
        jnp.abs(flux),
        jnp.zeros_like(flux),
    )
    unit = jax.lax.stop_gradient(
        jnp.maximum(
            jnp.maximum(
                finite_flux_abs,
                jnp.maximum(jnp.abs(limit), jnp.abs(f5)),
            ),
            jnp.asarray(1.0e-30, dtype=dtype),
        )
    )
    flux_scaled = flux / unit
    limit_scaled = limit / unit
    f5_scaled = f5 / unit
    sigma2_scaled = (jnp.asarray(0.04, dtype=dtype) - gamma_array) * jnp.abs(
        flux_scaled
    ) * f5_scaled + gamma_array * jnp.square(f5_scaled)
    if float(sigma_sys_mag) > 0.0:
        sys_fraction = jnp.expm1(
            jnp.log(jnp.asarray(10.0, dtype=dtype)) * float(sigma_sys_mag) / 2.5
        )
        sigma2_scaled = sigma2_scaled + jnp.square(sys_fraction * flux_scaled)
    floor_scaled = jnp.asarray(min_sigma_fnu_cgs, dtype=dtype) / unit
    sigma_scaled = jnp.sqrt(jnp.maximum(sigma2_scaled, jnp.square(floor_scaled)))
    return observed_flux_selection_log_beta(
        flux_scaled,
        sigma_scaled,
        limit_scaled,
    )


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


def estimate_log_alpha_score_function_diagnostic(
    prior: Any,
    key: jax.Array,
    *,
    n_prior_samples: int,
    log_beta_fn: Callable[[jnp.ndarray], jnp.ndarray],
    prior_sample_batch_size: int | None = None,
    dtype: Any = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return ``log alpha`` and a score-gradient surrogate for diagnostics.

    The scalar surrogate has the score-function gradient

    ``sum_k (normalized_beta_k - 1/M) * grad log p_eta(x_k)``.

    Its value is not ``log alpha`` and it must not replace the pathwise
    production objective without an explicit scientific decision. Samples,
    completeness values, and normalized selection weights are stopped so the
    only gradient is through ``prior.log_prob``.
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
    batch_size = min(batch_size, _MAX_SCORE_FUNCTION_BATCH_SIZE)
    n_batches = (count + batch_size - 1) // batch_size
    padded_count = n_batches * batch_size
    base = jax.random.normal(
        key,
        (padded_count, int(prior.latent_dim)),
        dtype=dtype,
    )
    batched_base = base.reshape(n_batches, batch_size, int(prior.latent_dim))

    def transform_and_score(base_batch):
        samples, _logdet = prior.forward(base_batch)
        stopped = jax.lax.stop_gradient(samples)
        return jax.lax.stop_gradient(log_beta_fn(stopped))

    batched_log_beta = jax.lax.map(
        jax.checkpoint(transform_and_score),
        batched_base,
    )
    log_beta = jnp.ravel(batched_log_beta)[:count]
    log_alpha, metrics = selection_log_alpha_from_log_beta(log_beta)
    finite = jnp.isfinite(log_beta)
    any_finite = jnp.any(finite)
    safe_log_beta = jnp.where(finite, log_beta, -jnp.inf)
    safe_log_beta = jnp.where(any_finite, safe_log_beta, jnp.zeros_like(log_beta))
    selection_weights = jax.lax.stop_gradient(jax.nn.softmax(safe_log_beta))
    uniform_weight = jnp.asarray(1.0 / count, dtype=selection_weights.dtype)
    centered_weights = jax.lax.stop_gradient(selection_weights - uniform_weight)
    padded_centered_weights = jnp.pad(
        centered_weights,
        (0, padded_count - count),
        constant_values=0.0,
    ).reshape(n_batches, batch_size)

    def accumulate_score(score, inputs):
        base_batch, weight_batch = inputs
        samples, _logdet = prior.forward(base_batch)
        stopped = jax.lax.stop_gradient(samples)
        contribution = jnp.sum(
            jax.lax.stop_gradient(weight_batch) * prior.log_prob(stopped)
        )
        return score + contribution, None

    score_surrogate, _ = jax.lax.scan(
        jax.checkpoint(accumulate_score),
        jnp.asarray(0.0, dtype=selection_weights.dtype),
        (batched_base, padded_centered_weights),
    )
    score_surrogate = jnp.where(
        any_finite,
        score_surrogate,
        jnp.asarray(jnp.nan, dtype=score_surrogate.dtype),
    )
    metrics = dict(metrics)
    metrics["selection/score_function_diagnostic"] = jnp.asarray(
        1.0, dtype=log_alpha.dtype
    )
    metrics["selection/prior_sample_batch_size"] = jnp.asarray(
        batch_size, dtype=log_alpha.dtype
    )
    metrics["selection/score_weight_ess"] = 1.0 / jnp.sum(jnp.square(selection_weights))
    metrics["selection/score_weight_ess_fraction"] = metrics[
        "selection/score_weight_ess"
    ] / float(count)
    metrics["selection/maximum_score_weight"] = jnp.max(selection_weights)
    metrics["selection/score_weights_finite"] = jnp.all(
        jnp.isfinite(selection_weights)
    ).astype(log_alpha.dtype)
    metrics["selection/score_control_variate_centered"] = jnp.asarray(
        1.0, dtype=log_alpha.dtype
    )
    return log_alpha, score_surrogate, metrics


def estimate_log_alpha_score_function(
    prior: Any,
    key: jax.Array,
    *,
    n_prior_samples: int,
    log_beta_fn: Callable[[jnp.ndarray], jnp.ndarray],
    prior_sample_batch_size: int | None = None,
    dtype: Any = jnp.float32,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Estimate ``log alpha`` with its exact score-function gradient.

    The returned scalar has the Monte-Carlo value ``log(mean(beta))`` and the
    gradient

    ``sum_k stop(beta_k / sum_j beta_j - 1/M) * grad log p_eta(x_k)``.

    This avoids differentiating through DSPS and Gaussian-m5 while preserving
    the population selection-normalization objective. It must only be used for
    the prior loss; selection still never enters object-level posterior weights.
    """
    log_alpha, score_surrogate, metrics = estimate_log_alpha_score_function_diagnostic(
        prior,
        key,
        n_prior_samples=n_prior_samples,
        log_beta_fn=log_beta_fn,
        prior_sample_batch_size=prior_sample_batch_size,
        dtype=dtype,
    )
    objective = jax.lax.stop_gradient(log_alpha - score_surrogate) + score_surrogate
    metrics = dict(metrics)
    metrics["selection/gradient_estimator_score_function"] = jnp.asarray(
        1.0, dtype=log_alpha.dtype
    )
    return objective, metrics


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
        "selection/score_weight_ess": zero,
        "selection/score_weight_ess_fraction": zero,
        "selection/maximum_score_weight": zero,
        "selection/score_weights_finite": jnp.asarray(1.0, dtype=dtype),
        "selection/score_control_variate_centered": zero,
    }
