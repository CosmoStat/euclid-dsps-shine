"""JAX photometric likelihoods for amortized inference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def student_t_logpdf(y, loc, scale, nu=2.0) -> jnp.ndarray:
    """Student-t log-density with location ``loc`` and positive ``scale``."""
    y = jnp.asarray(y)
    loc = jnp.asarray(loc)
    scale = jnp.maximum(jnp.asarray(scale), 1.0e-30)
    nu = jnp.asarray(nu, dtype=y.dtype)
    z = (y - loc) / scale
    log_norm = (
        jsp.special.gammaln((nu + 1.0) / 2.0)
        - jsp.special.gammaln(nu / 2.0)
        - 0.5 * (jnp.log(nu) + jnp.log(jnp.pi))
        - jnp.log(scale)
    )
    return log_norm - 0.5 * (nu + 1.0) * jnp.log1p((z**2) / nu)


def gaussian_logpdf(y, loc, scale) -> jnp.ndarray:
    """Gaussian log-density with location ``loc`` and positive ``scale``."""
    y = jnp.asarray(y)
    scale = jnp.maximum(jnp.asarray(scale), 1.0e-30)
    z = (y - loc) / scale
    return -0.5 * (z**2 + jnp.log(2.0 * jnp.pi)) - jnp.log(scale)


def photometric_loglike(
    obs_flux: jnp.ndarray,
    model_flux: jnp.ndarray,
    obs_err: jnp.ndarray,
    mask: jnp.ndarray,
    likelihood_type: str = "student_t",
    student_t_dof: float = 2.0,
    error_floor_frac: float = 0.02,
    error_jitter: float = 0.0,
) -> jnp.ndarray:
    """Return per-sample, per-object photometric log-likelihoods.

    ``model_flux`` is expected to be shaped ``[K,N,B]``. Observed arrays may be
    shaped ``[N,B]`` or ``[1,N,B]`` and broadcast over Monte Carlo samples.
    """
    obs_flux = _with_sample_axis(jnp.asarray(obs_flux, dtype=jnp.float32))
    obs_err = _with_sample_axis(jnp.asarray(obs_err, dtype=jnp.float32))
    mask = _with_sample_axis(jnp.asarray(mask, dtype=bool))
    model_flux = jnp.asarray(model_flux, dtype=jnp.float32)
    unit = _likelihood_unit(obs_flux, obs_err)
    obs_flux_scaled = obs_flux / unit
    model_flux_scaled = model_flux / unit
    obs_err_scaled = obs_err / unit
    error_jitter_scaled = float(error_jitter) / unit
    sigma_eff = jnp.sqrt(
        obs_err_scaled**2
        + (float(error_floor_frac) * jnp.abs(model_flux_scaled)) ** 2
        + error_jitter_scaled**2
        + 1.0e-12
    )
    finite = mask & jnp.isfinite(obs_flux_scaled) & jnp.isfinite(model_flux_scaled)
    finite &= jnp.isfinite(sigma_eff) & (sigma_eff > 0.0)
    kind = likelihood_type.strip().lower().replace("-", "_")
    if kind in {"student", "studentt"}:
        kind = "student_t"
    if kind == "student_t":
        logpdf = student_t_logpdf(
            obs_flux_scaled,
            model_flux_scaled,
            sigma_eff,
            nu=student_t_dof,
        )
    elif kind in {"gaussian", "normal"}:
        logpdf = gaussian_logpdf(obs_flux_scaled, model_flux_scaled, sigma_eff)
    else:
        raise ValueError(f"Unsupported likelihood_type: {likelihood_type}")
    logpdf = logpdf - jnp.log(unit)
    return jnp.sum(jnp.where(finite, logpdf, 0.0), axis=-1)


def photometric_normalized_residual(
    obs_flux: jnp.ndarray,
    model_flux: jnp.ndarray,
    obs_err: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    error_floor_frac: float = 0.02,
    error_jitter: float = 0.0,
) -> jnp.ndarray:
    """Return stable likelihood-space residuals ``(model - obs) / sigma_eff``."""
    obs_flux = _with_sample_axis(jnp.asarray(obs_flux, dtype=jnp.float32))
    obs_err = _with_sample_axis(jnp.asarray(obs_err, dtype=jnp.float32))
    mask = _with_sample_axis(jnp.asarray(mask, dtype=bool))
    model_flux = jnp.asarray(model_flux, dtype=jnp.float32)
    unit = _likelihood_unit(obs_flux, obs_err)
    obs_flux_scaled = obs_flux / unit
    model_flux_scaled = model_flux / unit
    obs_err_scaled = obs_err / unit
    error_jitter_scaled = float(error_jitter) / unit
    sigma_eff = jnp.sqrt(
        obs_err_scaled**2
        + (float(error_floor_frac) * jnp.abs(model_flux_scaled)) ** 2
        + error_jitter_scaled**2
        + 1.0e-12
    )
    finite = mask & jnp.isfinite(obs_flux_scaled) & jnp.isfinite(model_flux_scaled)
    finite &= jnp.isfinite(sigma_eff) & (sigma_eff > 0.0)
    chi = (model_flux_scaled - obs_flux_scaled) / sigma_eff
    return jnp.where(finite, chi, 0.0)


def _with_sample_axis(value: jnp.ndarray) -> jnp.ndarray:
    if value.ndim == 2:
        return value[None, :, :]
    if value.ndim == 3:
        return value
    raise ValueError(f"Expected rank 2 or 3 photometry array, got {value.shape}")


def _likelihood_unit(obs_flux: jnp.ndarray, obs_err: jnp.ndarray) -> jnp.ndarray:
    """Return a stop-gradient flux unit that avoids float32 variance underflow."""
    unit = jnp.maximum(jnp.abs(obs_flux), obs_err)
    unit = jnp.maximum(unit, 1.0e-30)
    return jax.lax.stop_gradient(unit)
