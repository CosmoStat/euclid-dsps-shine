"""Global calibration nuisance parameters for DSPS photometry paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class GlobalSedScaleConfig:
    """Configuration for one global SED/flux scale per run."""

    enabled: bool = False
    mode: str = "disabled"
    initial_log_alpha: float = 0.0
    prior_sigma_log_alpha: float = 0.10
    trainable: bool = False


@dataclass(frozen=True)
class GlobalSedScaleState:
    """Trainable or fixed log-scale state."""

    log_alpha_sed: jnp.ndarray


def _flatten_global_sed_scale_state(state: GlobalSedScaleState):
    return (state.log_alpha_sed,), None


def _unflatten_global_sed_scale_state(_aux, children):
    return GlobalSedScaleState(log_alpha_sed=children[0])


jax.tree_util.register_pytree_node(
    GlobalSedScaleState,
    _flatten_global_sed_scale_state,
    _unflatten_global_sed_scale_state,
)


def global_sed_scale_config(config: dict[str, Any] | None) -> GlobalSedScaleConfig:
    """Resolve the top-level calibration config into a typed object."""
    block = dict(((config or {}).get("calibration", {}) or {}).get("global_sed_scale", {}) or {})
    enabled = bool(block.get("enabled", False))
    mode = str(block.get("mode", "learn_global" if enabled else "disabled"))
    if not enabled:
        mode = "disabled"
    return GlobalSedScaleConfig(
        enabled=enabled,
        mode=mode,
        initial_log_alpha=float(block.get("initial_log_alpha", 0.0)),
        prior_sigma_log_alpha=float(block.get("prior_sigma_log_alpha", 0.10)),
        trainable=bool(block.get("trainable", enabled and mode == "learn_global")),
    )


def make_global_sed_scale_state(config: dict[str, Any] | None) -> GlobalSedScaleState:
    """Create SED-scale state initialized from config."""
    cfg = global_sed_scale_config(config)
    return GlobalSedScaleState(
        log_alpha_sed=jnp.asarray(cfg.initial_log_alpha, dtype=jnp.float32)
    )


def alpha_from_log_alpha(log_alpha_sed: jnp.ndarray) -> jnp.ndarray:
    """Return ``alpha_sed = exp(log_alpha_sed)``."""
    return jnp.exp(log_alpha_sed)


def apply_global_sed_scale_to_sed(
    sed: jnp.ndarray,
    log_alpha_sed: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the global SED scale before filter integration."""
    return alpha_from_log_alpha(log_alpha_sed) * sed


def apply_global_sed_scale_to_flux(
    flux: jnp.ndarray,
    log_alpha_sed: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the global SED scale to model fluxes.

    For a single scale shared by all wavelengths and all filters, scaling the
    SED before filter integration and scaling the integrated model flux are
    mathematically equivalent. The DSPS paths in this repository often expose
    only integrated photometry, so this helper is the single application point.
    """
    return alpha_from_log_alpha(log_alpha_sed) * flux


def global_sed_scale_prior_penalty(
    log_alpha_sed: jnp.ndarray,
    prior_sigma_log_alpha: float,
) -> jnp.ndarray:
    """Gaussian prior penalty on ``log_alpha_sed``."""
    sigma = jnp.asarray(max(float(prior_sigma_log_alpha), 1.0e-12))
    return 0.5 * (log_alpha_sed / sigma) ** 2


def delta_mag_from_alpha(alpha_sed: float | np.ndarray) -> float | np.ndarray:
    """Return the global magnitude offset equivalent to ``alpha_sed``."""
    return -2.5 * np.log10(alpha_sed)


def log10_mass_alpha_corrected(
    log10_stellar_mass_raw: float | np.ndarray,
    alpha_sed: float | np.ndarray,
) -> float | np.ndarray:
    """Return the stellar mass constrained by ``alpha_sed * Mstar``."""
    return np.asarray(log10_stellar_mass_raw) + np.log10(alpha_sed)


def alpha_metadata(
    log_alpha_sed: float,
    prior_sigma_log_alpha: float = 0.10,
) -> dict[str, float | bool]:
    """Return JSON-friendly diagnostics for a global SED scale."""
    alpha = float(np.exp(float(log_alpha_sed)))
    delta_mag = float(delta_mag_from_alpha(alpha))
    sigma = max(float(prior_sigma_log_alpha), 1.0e-12)
    penalty = 0.5 * (float(log_alpha_sed) / sigma) ** 2
    return {
        "log_alpha_sed": float(log_alpha_sed),
        "alpha_sed": alpha,
        "delta_mag_global": delta_mag,
        "alpha_prior_penalty": float(penalty),
        "large_scale_warning": bool(abs(delta_mag) > 0.3),
    }
