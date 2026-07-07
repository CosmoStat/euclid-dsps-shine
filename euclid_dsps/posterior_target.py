"""Pure-JAX posterior targets for non-NumPyro samplers."""

# ruff: noqa: I001, E402

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .jax_runtime import configure_jax_runtime

configure_jax_runtime()

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as jstats
import numpy as np

from .fit import (
    _initial_value,
    _likelihood_space,
    _photometric_likelihood,
    _photometric_objective_from_chi,
    _student_t_dof,
)
from .model import (
    DspsContext,
    gas_metallicity_constraint_penalty_jax,
    model_mags_jax_dynamic,
)
from .photometry import abmag_to_fnu_cgs_jax


@dataclass(frozen=True)
class BoundedParameterTransform:
    """Logistic transform between unconstrained sampler space and fit bounds."""

    names: tuple[str, ...]
    lower: jnp.ndarray
    upper: jnp.ndarray
    gas_metallicity_constraint: tuple[int, int] | None = None

    def to_bounded(self, y: jnp.ndarray) -> jnp.ndarray:
        theta = self.lower + (self.upper - self.lower) * jax.nn.sigmoid(y)
        if self.gas_metallicity_constraint is None:
            return theta
        stellar_index, gas_index = self.gas_metallicity_constraint
        stellar_low = self.lower[stellar_index]
        stellar_high = jnp.minimum(self.upper[stellar_index], self.upper[gas_index])
        stellar = stellar_low + (stellar_high - stellar_low) * jax.nn.sigmoid(
            y[stellar_index]
        )
        gas_low = jnp.maximum(self.lower[gas_index], stellar)
        gas_high = self.upper[gas_index]
        gas = gas_low + (gas_high - gas_low) * jax.nn.sigmoid(y[gas_index])
        return theta.at[stellar_index].set(stellar).at[gas_index].set(gas)

    def to_unconstrained(self, theta: jnp.ndarray) -> jnp.ndarray:
        span = self.upper - self.lower
        scaled = jnp.clip((theta - self.lower) / span, 1.0e-6, 1.0 - 1.0e-6)
        y = jnp.log(scaled) - jnp.log1p(-scaled)
        if self.gas_metallicity_constraint is None:
            return y
        stellar_index, gas_index = self.gas_metallicity_constraint
        stellar_low = self.lower[stellar_index]
        stellar_high = jnp.minimum(self.upper[stellar_index], self.upper[gas_index])
        stellar_span = jnp.maximum(stellar_high - stellar_low, 1.0e-12)
        stellar_scaled = jnp.clip(
            (theta[stellar_index] - stellar_low) / stellar_span,
            1.0e-6,
            1.0 - 1.0e-6,
        )
        gas_low = jnp.maximum(self.lower[gas_index], theta[stellar_index])
        gas_high = self.upper[gas_index]
        gas_span = jnp.maximum(gas_high - gas_low, 1.0e-12)
        gas_scaled = jnp.clip(
            (theta[gas_index] - gas_low) / gas_span,
            1.0e-6,
            1.0 - 1.0e-6,
        )
        return (
            y.at[stellar_index]
            .set(jnp.log(stellar_scaled) - jnp.log1p(-stellar_scaled))
            .at[gas_index]
            .set(jnp.log(gas_scaled) - jnp.log1p(-gas_scaled))
        )

    def log_abs_det_jacobian(self, y: jnp.ndarray) -> jnp.ndarray:
        span = self.upper - self.lower
        terms = jnp.log(span) + jax.nn.log_sigmoid(y) + jax.nn.log_sigmoid(-y)
        if self.gas_metallicity_constraint is None:
            return jnp.sum(terms)
        stellar_index, gas_index = self.gas_metallicity_constraint
        stellar_low = self.lower[stellar_index]
        stellar_high = jnp.minimum(self.upper[stellar_index], self.upper[gas_index])
        stellar_span = jnp.maximum(stellar_high - stellar_low, 1.0e-12)
        stellar = stellar_low + stellar_span * jax.nn.sigmoid(y[stellar_index])
        gas_low = jnp.maximum(self.lower[gas_index], stellar)
        gas_span = jnp.maximum(self.upper[gas_index] - gas_low, 1.0e-12)
        terms = terms.at[stellar_index].set(
            jnp.log(stellar_span)
            + jax.nn.log_sigmoid(y[stellar_index])
            + jax.nn.log_sigmoid(-y[stellar_index])
        )
        terms = terms.at[gas_index].set(
            jnp.log(gas_span)
            + jax.nn.log_sigmoid(y[gas_index])
            + jax.nn.log_sigmoid(-y[gas_index])
        )
        return jnp.sum(terms)


@dataclass(frozen=True)
class PosteriorTarget:
    """A BlackJAX-friendly posterior over an unconstrained parameter vector."""

    context: DspsContext
    model_args: tuple[Any, ...]
    base_params: dict[str, float]
    transform: BoundedParameterTransform
    prior_specs: tuple[dict[str, float | str], ...]
    observed: jnp.ndarray
    sigma: jnp.ndarray
    finite_mask: jnp.ndarray
    fit_config: dict[str, Any]
    likelihood_space: str
    photometric_likelihood: str
    student_t_dof: float
    band_offsets: jnp.ndarray

    @property
    def free_names(self) -> list[str]:
        return list(self.transform.names)

    def physical_from_unconstrained(self, y: jnp.ndarray) -> dict[str, Any]:
        theta = self.transform.to_bounded(y)
        params: dict[str, Any] = {
            key: jnp.asarray(value) for key, value in self.base_params.items()
        }
        params.update(
            {
                name: theta[index]
                for index, name in enumerate(self.transform.names)
            }
        )
        return params

    def theta_from_unconstrained(self, y: jnp.ndarray) -> jnp.ndarray:
        return self.transform.to_bounded(y)

    def unconstrained_from_theta(self, theta: jnp.ndarray) -> jnp.ndarray:
        return self.transform.to_unconstrained(theta)

    def logdensity(self, y: jnp.ndarray) -> jnp.ndarray:
        theta = self.transform.to_bounded(y)
        params = self.physical_from_unconstrained(y)
        model_mag = model_mags_jax_dynamic(self.context, self.model_args, params)
        if self.band_offsets.size:
            model_mag = model_mag + self.band_offsets
        if self.likelihood_space == "flux":
            model_obs = abmag_to_fnu_cgs_jax(model_mag)
        else:
            model_obs = model_mag
        loglike = _masked_observation_logprob(
            observed=self.observed,
            model_obs=model_obs,
            sigma=self.sigma,
            finite_mask=self.finite_mask,
            photometric_likelihood=self.photometric_likelihood,
            student_t_dof=self.student_t_dof,
        )
        logprior = _bounded_log_prior(theta, self.prior_specs)
        logjac = self.transform.log_abs_det_jacobian(y)
        if self.transform.gas_metallicity_constraint is None:
            gas_penalty = gas_metallicity_constraint_penalty_jax(
                params, self.context.model_config, penalty=jnp.inf
            )
        else:
            gas_penalty = jnp.asarray(0.0, dtype=theta.dtype)
        return loglike + logprior + logjac - gas_penalty


def build_posterior_target(
    context: DspsContext,
    model_args: tuple[Any, ...],
    base_params: dict[str, float],
    fit_config: dict[str, Any],
    sample_config: dict[str, Any],
    observed_mag: np.ndarray,
    sigma_mag: np.ndarray,
    observed_flux: np.ndarray,
    flux_error: np.ndarray,
) -> PosteriorTarget:
    """Build the unconstrained-space posterior target used by BlackJAX."""
    free = fit_config["free_parameters"]
    free_names = tuple(free)
    bounds = np.asarray(
        [tuple(float(value) for value in free[name]["bounds"]) for name in free_names],
        dtype=float,
    )
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("fit.free_parameters bounds must be [low, high] pairs")
    lower = jnp.asarray(bounds[:, 0], dtype=jnp.float32)
    upper = jnp.asarray(bounds[:, 1], dtype=jnp.float32)
    transform = BoundedParameterTransform(
        names=free_names,
        lower=lower,
        upper=upper,
        gas_metallicity_constraint=_gas_metallicity_constraint_indices(free_names),
    )
    likelihood_space = _likelihood_space(fit_config)
    if likelihood_space == "flux":
        floor_frac = float(fit_config.get("flux_error_floor_frac", 0.0))
        jitter = float(fit_config.get("flux_error_jitter", 0.0))
        observed = np.asarray(observed_flux, dtype=float)
        sigma = np.sqrt(
            np.asarray(flux_error, dtype=float) ** 2
            + (floor_frac * np.asarray(observed_flux, dtype=float)) ** 2
            + jitter**2
        )
    elif likelihood_space == "mag":
        observed = np.asarray(observed_mag, dtype=float)
        sigma = np.asarray(sigma_mag, dtype=float)
    else:
        raise ValueError(f"Unsupported fit.likelihood_space: {likelihood_space}")
    finite = np.isfinite(observed) & np.isfinite(sigma) & (sigma > 0.0)
    prior_specs = tuple(
        _resolved_prior_spec(
            name,
            free[name],
            (sample_config.get("priors", {}) or {}).get(name, {}),
            base_params,
        )
        for name in free_names
    )
    return PosteriorTarget(
        context=context,
        model_args=model_args,
        base_params=base_params,
        transform=transform,
        prior_specs=prior_specs,
        observed=jnp.asarray(observed, dtype=jnp.float32),
        sigma=jnp.asarray(sigma, dtype=jnp.float32),
        finite_mask=jnp.asarray(finite),
        fit_config=fit_config,
        likelihood_space=likelihood_space,
        photometric_likelihood=_photometric_likelihood(fit_config),
        student_t_dof=_student_t_dof(fit_config),
        band_offsets=jnp.asarray(
            fit_config.get("band_calibration_offsets_mag", []), dtype=jnp.float32
        ),
    )


def initial_unconstrained_position(
    target: PosteriorTarget,
    initial_params: dict[str, float] | None,
    fit_config: dict[str, Any],
) -> jnp.ndarray:
    """Return a finite unconstrained initial vector from MAP or config initial values."""
    free = fit_config["free_parameters"]
    values = []
    for name in target.free_names:
        if initial_params and name in initial_params and np.isfinite(initial_params[name]):
            value = float(initial_params[name])
        else:
            value = _initial_value(free[name], name, target.base_params)
        values.append(value)
    theta = jnp.asarray(values, dtype=jnp.float32)
    eps = jnp.maximum(
        (target.transform.upper - target.transform.lower) * 1.0e-6, 1.0e-7
    )
    theta = jnp.clip(theta, target.transform.lower + eps, target.transform.upper - eps)
    return target.unconstrained_from_theta(theta)


def _gas_metallicity_constraint_indices(
    free_names: tuple[str, ...],
) -> tuple[int, int] | None:
    try:
        stellar_index = free_names.index("log10_stellar_metallicity")
        gas_index = free_names.index("log10_gas_metallicity")
    except ValueError:
        return None
    return stellar_index, gas_index


def _masked_observation_logprob(
    *,
    observed: jnp.ndarray,
    model_obs: jnp.ndarray,
    sigma: jnp.ndarray,
    finite_mask: jnp.ndarray,
    photometric_likelihood: str,
    student_t_dof: float,
) -> jnp.ndarray:
    chi = jnp.where(finite_mask, (observed - model_obs) / sigma, 0.0)
    return -0.5 * _photometric_objective_from_chi(
        chi, photometric_likelihood, student_t_dof
    )


def _resolved_prior_spec(
    name: str,
    fit_spec: dict[str, Any],
    prior_spec: dict[str, Any],
    base_params: dict[str, float],
) -> dict[str, float | str]:
    low, high = [float(value) for value in fit_spec["bounds"]]
    prior_type = str(prior_spec.get("type", "truncated_normal"))
    loc = _prior_location(name, fit_spec, prior_spec, base_params)
    scale = _prior_scale(name, prior_spec, base_params, max((high - low) / 4.0, 1.0e-3))
    return {
        "type": prior_type,
        "low": low,
        "high": high,
        "loc": loc,
        "scale": scale,
        "alpha": max(float(prior_spec.get("alpha", 1.0)), 1.0e-6),
        "beta": max(float(prior_spec.get("beta", 1.0)), 1.0e-6),
    }


def _bounded_log_prior(
    theta: jnp.ndarray, prior_specs: tuple[dict[str, float | str], ...]
) -> jnp.ndarray:
    terms = []
    for index, spec in enumerate(prior_specs):
        value = theta[index]
        low = jnp.asarray(float(spec["low"]), dtype=theta.dtype)
        high = jnp.asarray(float(spec["high"]), dtype=theta.dtype)
        span = high - low
        prior_type = str(spec["type"])
        if prior_type == "uniform":
            logprob = -jnp.log(span)
        elif prior_type == "normal":
            loc = jnp.asarray(float(spec["loc"]), dtype=theta.dtype)
            scale = jnp.maximum(
                jnp.asarray(float(spec["scale"]), dtype=theta.dtype), 1.0e-6
            )
            logprob = jstats.norm.logpdf(value, loc=loc, scale=scale)
        elif prior_type == "truncated_normal":
            loc = jnp.asarray(float(spec["loc"]), dtype=theta.dtype)
            scale = jnp.maximum(
                jnp.asarray(float(spec["scale"]), dtype=theta.dtype), 1.0e-6
            )
            norm = jnp.maximum(
                jsp.special.ndtr((high - loc) / scale)
                - jsp.special.ndtr((low - loc) / scale),
                1.0e-12,
            )
            logprob = jstats.norm.logpdf(value, loc=loc, scale=scale) - jnp.log(norm)
        elif prior_type == "scaled_beta":
            alpha = jnp.asarray(float(spec["alpha"]), dtype=theta.dtype)
            beta = jnp.asarray(float(spec["beta"]), dtype=theta.dtype)
            unit = jnp.clip((value - low) / span, 1.0e-6, 1.0 - 1.0e-6)
            logprob = (
                (alpha - 1.0) * jnp.log(unit)
                + (beta - 1.0) * jnp.log1p(-unit)
                - jsp.special.betaln(alpha, beta)
                - jnp.log(span)
            )
        else:
            raise ValueError(f"Unsupported sample prior type: {prior_type}")
        terms.append(logprob)
    return jnp.sum(jnp.asarray(terms, dtype=theta.dtype))


def _prior_location(
    name: str,
    fit_spec: dict[str, Any],
    prior_spec: dict[str, Any],
    base_params: dict[str, float],
) -> float:
    value = prior_spec.get("loc", _initial_value(fit_spec, name, base_params))
    if value == "from_base":
        return float(base_params[name])
    return float(value)


def _prior_scale(
    name: str,
    prior_spec: dict[str, Any],
    base_params: dict[str, float],
    fallback: float,
) -> float:
    value = prior_spec.get("scale", fallback)
    if value == "from_base":
        scale_name = str(prior_spec.get("scale_parameter", f"{name}_prior_sigma"))
        return max(float(base_params.get(scale_name, fallback)), 1.0e-6)
    return max(float(value), 1.0e-6)
