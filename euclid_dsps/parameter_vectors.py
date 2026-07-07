"""Public helpers for JAX parameter-vector model calls."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .model import (
    _jax_result_derived_array,
    model_mags_jax_dynamic,
    run_dsps_model_jax_dynamic,
)


def parameter_names_from_config(config: dict[str, Any]) -> tuple[str, ...]:
    """Return the configured free-parameter order as an immutable tuple."""
    free = config.get("fit", {}).get("free_parameters", {})
    if not isinstance(free, dict) or not free:
        raise ValueError("config.fit.free_parameters must be a non-empty mapping")
    return tuple(str(name) for name in free)


def free_parameter_bounds_from_config(
    config: dict[str, Any],
    parameter_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper bounds for ``parameter_names`` from a config."""
    free = config.get("fit", {}).get("free_parameters", {})
    lower = []
    upper = []
    for name in parameter_names:
        if name not in free:
            raise ValueError(f"Missing fit.free_parameters entry for {name!r}")
        bounds = free[name].get("bounds")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"fit.free_parameters.{name}.bounds must be [low, high]")
        low, high = float(bounds[0]), float(bounds[1])
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError(f"Invalid bounds for {name!r}: {bounds!r}")
        lower.append(low)
        upper.append(high)
    return np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)


def theta_vector_to_param_dict(
    theta: jnp.ndarray,
    parameter_names: tuple[str, ...],
) -> dict[str, jnp.ndarray]:
    """Convert one physical parameter vector to a JAX parameter dictionary."""
    theta = jnp.asarray(theta)
    if theta.ndim != 1:
        raise ValueError(f"theta must be rank 1, got shape {theta.shape}")
    if theta.shape[0] != len(parameter_names):
        raise ValueError(
            "theta length mismatch: "
            f"expected {len(parameter_names)}, got {theta.shape[0]}"
        )
    return {name: theta[index] for index, name in enumerate(parameter_names)}


def theta_vector_to_model_param_dict(
    theta: jnp.ndarray,
    parameter_names: tuple[str, ...],
    model_config: dict[str, Any] | None,
) -> dict[str, jnp.ndarray]:
    """Convert free parameters to a full model-parameter dictionary.

    Amortized configs often expose only a compact latent subset while the DSPS
    forward model still expects additional nuisance/SFH parameters. Those values
    come from ``model.fixed_parameters`` and are overwritten by the free latent
    parameters when names overlap.
    """
    fixed = {}
    if isinstance(model_config, dict):
        raw_fixed = model_config.get("fixed_parameters", {})
        if isinstance(raw_fixed, dict):
            fixed = {
                str(name): jnp.asarray(value, dtype=jnp.float32)
                for name, value in raw_fixed.items()
                if np.isscalar(value)
            }
    fixed.update(theta_vector_to_param_dict(theta, parameter_names))
    return _adapt_truth_basic_to_popcosmos_params(
        fixed,
        parameter_names=parameter_names,
        model_config=model_config,
    )


def _adapt_truth_basic_to_popcosmos_params(
    params: dict[str, jnp.ndarray],
    *,
    parameter_names: tuple[str, ...],
    model_config: dict[str, Any] | None,
) -> dict[str, jnp.ndarray]:
    """Map Diffsky truth-basic nuisance names onto the PopCosmos DSPS decoder.

    ``diffsky_truth_basic`` is the supervised-prior schema.  The current
    compact Diffsky amortized decoder is still ``popcosmos_bins``, so the
    basic truth quantities must be projected into the parameters actually used
    by that decoder.  This keeps the prior and posterior in the same latent
    space while making the decoder dependency explicit.
    """
    if not isinstance(model_config, dict):
        return params
    if str(model_config.get("sfh_model", "lognormal")) != "popcosmos_bins":
        return params
    free = set(parameter_names)
    adapted = dict(params)
    if _has_truth_basic_sfr(adapted) and not any(
        f"dlog10_sfr_{index}" in free for index in range(1, 7)
    ):
        target_logssfr = _truth_basic_logssfr(adapted)
        reference = jnp.asarray(
            float(model_config.get("truth_basic_ssfr_reference", -10.0)),
            dtype=jnp.float32,
        )
        slope = jnp.clip((target_logssfr - reference) / 6.0, -3.0, 3.0)
        for index in range(1, 7):
            adapted[f"dlog10_sfr_{index}"] = slope
    if "dust_av" in adapted and "tau2" not in free:
        adapted["tau2"] = jnp.clip(
            jnp.asarray(adapted["dust_av"], dtype=jnp.float32) / 1.086,
            0.0,
            jnp.inf,
        )
    if "dust_delta" in adapted and "dust_index_n" not in free:
        adapted["dust_index_n"] = jnp.asarray(adapted["dust_delta"], dtype=jnp.float32)
    return adapted


def _has_truth_basic_sfr(params: dict[str, jnp.ndarray]) -> bool:
    return "log10_ssfr_at_obs" in params or "log10_sfr_at_obs" in params


def _truth_basic_logssfr(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
    if "log10_ssfr_at_obs" in params:
        return jnp.asarray(params["log10_ssfr_at_obs"], dtype=jnp.float32)
    log_sfr = jnp.asarray(params["log10_sfr_at_obs"], dtype=jnp.float32)
    log_mass = jnp.asarray(params["log10_stellar_mass"], dtype=jnp.float32)
    return log_sfr - log_mass


def theta_matrix_to_param_dicts_or_pytree(
    theta: jnp.ndarray,
    parameter_names: tuple[str, ...],
):
    """Return a JAX pytree mapping parameter names to leading-shaped arrays."""
    theta = jnp.asarray(theta)
    if theta.shape[-1] != len(parameter_names):
        raise ValueError(
            "theta last dimension mismatch: "
            f"expected {len(parameter_names)}, got {theta.shape[-1]}"
        )
    return {name: theta[..., index] for index, name in enumerate(parameter_names)}


def model_mags_from_theta_matrix_jax(
    context,
    model_args,
    theta: jnp.ndarray,
    parameter_names: tuple[str, ...],
) -> jnp.ndarray:
    """Evaluate model magnitudes for ``theta`` vectors while preserving gradients.

    ``theta`` may be shaped ``[D]``, ``[N,D]``, or ``[K,N,D]``. The returned
    magnitudes preserve the leading dimensions and append the configured band
    dimension.
    """
    theta = jnp.asarray(theta, dtype=jnp.float32)
    if theta.shape[-1] != len(parameter_names):
        raise ValueError(
            "theta last dimension mismatch: "
            f"expected {len(parameter_names)}, got {theta.shape[-1]}"
        )

    def single(theta_row):
        params = theta_vector_to_model_param_dict(
            theta_row,
            parameter_names,
            getattr(context, "model_config", None),
        )
        return model_mags_jax_dynamic(context, model_args, params)

    if theta.ndim == 1:
        return single(theta)
    if theta.ndim == 2:
        return jax.vmap(single)(theta)
    if theta.ndim == 3:
        return jax.vmap(jax.vmap(single))(theta)
    raise ValueError(f"theta must have rank 1, 2, or 3; got shape {theta.shape}")


def derived_from_theta_matrix_jax(
    context,
    model_args,
    theta: jnp.ndarray,
    parameter_names: tuple[str, ...],
) -> jnp.ndarray:
    """Evaluate DSPS derived quantities for compact free-parameter vectors."""
    theta = jnp.asarray(theta, dtype=jnp.float32)
    if theta.shape[-1] != len(parameter_names):
        raise ValueError(
            "theta last dimension mismatch: "
            f"expected {len(parameter_names)}, got {theta.shape[-1]}"
        )

    def single(theta_row):
        params = theta_vector_to_model_param_dict(
            theta_row,
            parameter_names,
            getattr(context, "model_config", None),
        )
        result = run_dsps_model_jax_dynamic(context, model_args, params)
        return _jax_result_derived_array(result)

    if theta.ndim == 1:
        return single(theta)
    if theta.ndim == 2:
        return jax.vmap(single)(theta)
    if theta.ndim == 3:
        return jax.vmap(jax.vmap(single))(theta)
    raise ValueError(f"theta must have rank 1, 2, or 3; got shape {theta.shape}")
