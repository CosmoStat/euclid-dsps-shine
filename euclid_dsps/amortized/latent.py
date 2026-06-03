"""Latent-space transforms for FS2 amortized inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.parameter_vectors import free_parameter_bounds_from_config
from euclid_dsps.parameters import POPCOSMOS_PARAMETER_NAMES


@dataclass(frozen=True)
class LatentSpec:
    names: tuple[str, ...]
    lower: jnp.ndarray
    upper: jnp.ndarray


def latent_spec_from_config(config: dict[str, Any]) -> LatentSpec:
    """Build the PopCosmos 16D latent transform spec from fit bounds."""
    amortized = config.get("amortized", {}) or {}
    latent = amortized.get("latent", {}) or {}
    schema = str(latent.get("schema", "popcosmos_16"))
    if schema != "popcosmos_16":
        raise ValueError("Amortized FS2 currently supports only schema='popcosmos_16'")
    if not bool(latent.get("include_redshift", True)):
        raise ValueError("Amortized FS2 requires redshift in the 16D latent")
    configured_names = tuple(config.get("fit", {}).get("free_parameters", {}))
    if configured_names == POPCOSMOS_PARAMETER_NAMES:
        names = configured_names
    elif set(configured_names) == set(POPCOSMOS_PARAMETER_NAMES):
        names = POPCOSMOS_PARAMETER_NAMES
    else:
        raise ValueError(
            "Amortized FS2 theta order must match POPCOSMOS_PARAMETER_NAMES; "
            f"got {configured_names}"
        )
    lower, upper = free_parameter_bounds_from_config(config, names)
    return LatentSpec(
        names=names,
        lower=jnp.asarray(lower, dtype=jnp.float32),
        upper=jnp.asarray(upper, dtype=jnp.float32),
    )


def x_to_theta(x: jnp.ndarray, spec: LatentSpec) -> jnp.ndarray:
    """Map unconstrained latent ``x`` to bounded physical ``theta``."""
    x = _validate_last_dim(jnp.asarray(x, dtype=jnp.float32), spec)
    theta = spec.lower + (spec.upper - spec.lower) * jax.nn.sigmoid(x)
    constraint = _gas_metallicity_indices(spec.names)
    if constraint is None:
        return theta
    stellar_index, gas_index = constraint
    stellar_low = spec.lower[stellar_index]
    stellar_high = jnp.minimum(spec.upper[stellar_index], spec.upper[gas_index])
    stellar_span = jnp.maximum(stellar_high - stellar_low, 1.0e-12)
    stellar = stellar_low + stellar_span * jax.nn.sigmoid(x[..., stellar_index])
    gas_low = jnp.maximum(spec.lower[gas_index], stellar)
    gas_span = jnp.maximum(spec.upper[gas_index] - gas_low, 1.0e-12)
    gas = gas_low + gas_span * jax.nn.sigmoid(x[..., gas_index])
    theta = theta.at[..., stellar_index].set(stellar)
    return theta.at[..., gas_index].set(gas)


def theta_to_x(theta: jnp.ndarray, spec: LatentSpec) -> jnp.ndarray:
    """Map bounded physical ``theta`` to unconstrained latent ``x``."""
    theta = _validate_last_dim(jnp.asarray(theta, dtype=jnp.float32), spec)
    eps = jnp.asarray(1.0e-6, dtype=theta.dtype)
    scaled = _safe_unit_interval((theta - spec.lower) / (spec.upper - spec.lower), eps)
    x = _logit(scaled)
    constraint = _gas_metallicity_indices(spec.names)
    if constraint is None:
        return x
    stellar_index, gas_index = constraint
    stellar_low = spec.lower[stellar_index]
    stellar_high = jnp.minimum(spec.upper[stellar_index], spec.upper[gas_index])
    stellar_span = jnp.maximum(stellar_high - stellar_low, 1.0e-12)
    stellar_scaled = _safe_unit_interval(
        (theta[..., stellar_index] - stellar_low) / stellar_span,
        eps,
    )
    gas_low = jnp.maximum(spec.lower[gas_index], theta[..., stellar_index])
    gas_span = jnp.maximum(spec.upper[gas_index] - gas_low, 1.0e-12)
    gas_scaled = _safe_unit_interval(
        (theta[..., gas_index] - gas_low) / gas_span,
        eps,
    )
    x = x.at[..., stellar_index].set(_logit(stellar_scaled))
    return x.at[..., gas_index].set(_logit(gas_scaled))


def theta_matrix_to_param_dict(
    theta: jnp.ndarray,
    names: tuple[str, ...],
) -> dict[str, jnp.ndarray]:
    """Convert a theta array to a JAX pytree keyed by parameter name."""
    theta = jnp.asarray(theta)
    if theta.shape[-1] != len(names):
        raise ValueError(
            f"theta last dimension must be {len(names)}, got {theta.shape[-1]}"
        )
    return {name: theta[..., index] for index, name in enumerate(names)}


def _validate_last_dim(value: jnp.ndarray, spec: LatentSpec) -> jnp.ndarray:
    if value.shape[-1] != len(spec.names):
        raise ValueError(
            f"Expected last dimension {len(spec.names)}, got {value.shape[-1]}"
        )
    return value


def _safe_unit_interval(value: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(value, eps, 1.0 - eps)


def _logit(value: jnp.ndarray) -> jnp.ndarray:
    return jnp.log(value) - jnp.log1p(-value)


def _gas_metallicity_indices(names: tuple[str, ...]) -> tuple[int, int] | None:
    try:
        return (
            names.index("log10_stellar_metallicity"),
            names.index("log10_gas_metallicity"),
        )
    except ValueError:
        return None


def latent_spec_to_jsonable(spec: LatentSpec) -> dict[str, Any]:
    """Return a JSON-serializable latent spec payload."""
    return {
        "names": list(spec.names),
        "lower": np.asarray(spec.lower).astype(float).tolist(),
        "upper": np.asarray(spec.upper).astype(float).tolist(),
    }
