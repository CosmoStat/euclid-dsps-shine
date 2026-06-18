"""Latent-space transforms for amortized inference."""

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
    raw_center: jnp.ndarray | None = None
    raw_scale: jnp.ndarray | None = None
    normalization: str = "identity"


def latent_spec_from_config(config: dict[str, Any]) -> LatentSpec:
    """Build the latent transform spec from configured fit bounds."""
    amortized = config.get("amortized", {}) or {}
    latent = amortized.get("latent", {}) or {}
    schema = str(latent.get("schema", "popcosmos_16"))
    if not bool(latent.get("include_redshift", True)):
        raise ValueError("Amortized inference requires redshift in the latent")
    configured_names = tuple(config.get("fit", {}).get("free_parameters", {}))
    if schema == "popcosmos_16" and configured_names == POPCOSMOS_PARAMETER_NAMES:
        names = configured_names
    elif schema == "popcosmos_16" and set(configured_names) == set(
        POPCOSMOS_PARAMETER_NAMES
    ):
        names = POPCOSMOS_PARAMETER_NAMES
    elif schema in {
        "config_free_parameters",
        "diffsky_hltds_prior_v1",
        "diffsky_truth_basic",
    }:
        names = configured_names
        if not names:
            raise ValueError("config.fit.free_parameters must be non-empty")
        if "z_obs" not in names:
            raise ValueError(
                f"Amortized schema {schema!r} requires z_obs in free_parameters"
            )
    else:
        raise ValueError(
            "Unsupported amortized latent schema. Use 'popcosmos_16', "
            "'config_free_parameters', 'diffsky_hltds_prior_v1', or "
            "'diffsky_truth_basic'."
        )
    lower, upper = free_parameter_bounds_from_config(config, names)
    raw_center, raw_scale, normalization = _latent_normalization_from_config(
        config,
        names,
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    return LatentSpec(
        names=names,
        lower=jnp.asarray(lower, dtype=jnp.float32),
        upper=jnp.asarray(upper, dtype=jnp.float32),
        raw_center=jnp.asarray(raw_center, dtype=jnp.float32),
        raw_scale=jnp.asarray(raw_scale, dtype=jnp.float32),
        normalization=normalization,
    )


def x_to_theta(x: jnp.ndarray, spec: LatentSpec) -> jnp.ndarray:
    """Map network latent ``x`` to bounded physical ``theta``."""
    x = _validate_last_dim(jnp.asarray(x, dtype=jnp.float32), spec)
    raw_x = network_x_to_raw_x(x, spec)
    theta = spec.lower + (spec.upper - spec.lower) * jax.nn.sigmoid(raw_x)
    constraint = _gas_metallicity_indices(spec.names)
    if constraint is None:
        return theta
    stellar_index, gas_index = constraint
    stellar_low = spec.lower[stellar_index]
    stellar_high = jnp.minimum(spec.upper[stellar_index], spec.upper[gas_index])
    stellar_span = jnp.maximum(stellar_high - stellar_low, 1.0e-12)
    stellar = stellar_low + stellar_span * jax.nn.sigmoid(raw_x[..., stellar_index])
    gas_low = jnp.maximum(spec.lower[gas_index], stellar)
    gas_span = jnp.maximum(spec.upper[gas_index] - gas_low, 1.0e-12)
    gas = gas_low + gas_span * jax.nn.sigmoid(raw_x[..., gas_index])
    theta = theta.at[..., stellar_index].set(stellar)
    return theta.at[..., gas_index].set(gas)


def theta_to_x(theta: jnp.ndarray, spec: LatentSpec) -> jnp.ndarray:
    """Map bounded physical ``theta`` to network latent ``x``."""
    theta = _validate_last_dim(jnp.asarray(theta, dtype=jnp.float32), spec)
    eps = jnp.asarray(1.0e-6, dtype=theta.dtype)
    scaled = _safe_unit_interval((theta - spec.lower) / (spec.upper - spec.lower), eps)
    raw_x = _logit(scaled)
    constraint = _gas_metallicity_indices(spec.names)
    if constraint is not None:
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
        raw_x = raw_x.at[..., stellar_index].set(_logit(stellar_scaled))
        raw_x = raw_x.at[..., gas_index].set(_logit(gas_scaled))
    return raw_x_to_network_x(raw_x, spec)


def network_x_to_raw_x(x: jnp.ndarray, spec: LatentSpec) -> jnp.ndarray:
    """Map standardized network latent coordinates to raw bounded logits."""
    return _latent_center(spec) + _latent_scale(spec) * x


def raw_x_to_network_x(raw_x: jnp.ndarray, spec: LatentSpec) -> jnp.ndarray:
    """Map raw bounded logits to standardized network latent coordinates."""
    return (raw_x - _latent_center(spec)) / _latent_scale(spec)


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
        "raw_center": np.asarray(_latent_center(spec)).astype(float).tolist(),
        "raw_scale": np.asarray(_latent_scale(spec)).astype(float).tolist(),
        "normalization": str(spec.normalization),
    }


def _latent_center(spec: LatentSpec) -> jnp.ndarray:
    if spec.raw_center is None:
        return jnp.zeros_like(spec.lower)
    return jnp.asarray(spec.raw_center, dtype=jnp.float32)


def _latent_scale(spec: LatentSpec) -> jnp.ndarray:
    if spec.raw_scale is None:
        return jnp.ones_like(spec.lower)
    return jnp.maximum(jnp.asarray(spec.raw_scale, dtype=jnp.float32), 1.0e-6)


def _latent_normalization_from_config(
    config: dict[str, Any],
    names: tuple[str, ...],
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    latent = (config.get("amortized", {}) or {}).get("latent", {}) or {}
    mode = str(latent.get("normalization", latent.get("transform", "identity"))).lower()
    if mode in {"identity", "none", "raw_logit"}:
        return np.zeros(len(names), dtype=float), np.ones(len(names), dtype=float), "identity"
    if mode not in {"standardized_logit", "standardized", "centered_logit"}:
        raise ValueError(
            "amortized.latent.normalization must be 'identity' or 'standardized_logit'"
        )
    theta0 = _initial_theta_for_normalization(config, names, lower, upper)
    span = np.maximum(upper - lower, 1.0e-12)
    eps = 1.0e-6
    scaled = np.clip((theta0 - lower) / span, eps, 1.0 - eps)
    raw_center = np.log(scaled) - np.log1p(-scaled)
    derivative = np.maximum(span * scaled * (1.0 - scaled), 1.0e-6)
    physical_scales = dict(latent.get("physical_scales", {}) or {})
    default_fraction = float(latent.get("default_physical_scale_fraction", 0.15))
    default_physical = np.maximum(default_fraction * span, 1.0e-6)
    raw_scale = np.empty(len(names), dtype=float)
    for index, name in enumerate(names):
        physical = float(physical_scales.get(name, default_physical[index]))
        physical = max(abs(physical), 1.0e-6)
        raw_scale[index] = physical / derivative[index]
    raw_scale = np.clip(
        raw_scale,
        float(latent.get("min_raw_scale", 0.05)),
        float(latent.get("max_raw_scale", 5.0)),
    )
    return raw_center, raw_scale, "standardized_logit"


def _initial_theta_for_normalization(
    config: dict[str, Any],
    names: tuple[str, ...],
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    defaults = {
        "z_obs": 0.8,
        "log10_stellar_mass": 10.0,
        "log10_stellar_metallicity": -0.7,
        "tau2": 0.4,
        "dust_index_n": -0.7,
        "tau1_over_tau2": 1.0,
        "log10_gas_metallicity": -0.3,
        "log10_gas_ionization": -2.5,
        "ln_fagn": -8.0,
        "ln_tauagn": float(np.log(20.0)),
    }
    defaults.update({f"dlog10_sfr_{index}": 0.0 for index in range(1, 7)})
    free = (config.get("fit", {}) or {}).get("free_parameters", {}) or {}
    values = []
    for index, name in enumerate(names):
        configured = (free.get(name, {}) or {}).get("initial")
        if isinstance(configured, int | float) and np.isfinite(float(configured)):
            value = float(configured)
        else:
            value = float(defaults.get(name, 0.5 * (lower[index] + upper[index])))
        values.append(np.clip(value, lower[index] + 1.0e-5, upper[index] - 1.0e-5))
    return np.asarray(values, dtype=float)
