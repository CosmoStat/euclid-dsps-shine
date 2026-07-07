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


@dataclass(frozen=True)
class PerBandFluxCalibrationConfig:
    """Configuration for one trainable multiplicative flux scale per band."""

    enabled: bool = False
    mode: str = "disabled"
    prior_sigma_log_alpha: float = 0.04605170185988091
    prior_sigma_mag: float = 0.05
    trainable: bool = False


@dataclass(frozen=True)
class PerBandFluxCalibrationState:
    """Trainable or fixed per-band log-scale state."""

    log_alpha_band: jnp.ndarray


def _flatten_global_sed_scale_state(state: GlobalSedScaleState):
    return (state.log_alpha_sed,), None


def _unflatten_global_sed_scale_state(_aux, children):
    return GlobalSedScaleState(log_alpha_sed=children[0])


def _flatten_per_band_flux_calibration_state(state: PerBandFluxCalibrationState):
    return (state.log_alpha_band,), None


def _unflatten_per_band_flux_calibration_state(_aux, children):
    return PerBandFluxCalibrationState(log_alpha_band=children[0])


jax.tree_util.register_pytree_node(
    GlobalSedScaleState,
    _flatten_global_sed_scale_state,
    _unflatten_global_sed_scale_state,
)
jax.tree_util.register_pytree_node(
    PerBandFluxCalibrationState,
    _flatten_per_band_flux_calibration_state,
    _unflatten_per_band_flux_calibration_state,
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


def per_band_flux_calibration_config(
    config: dict[str, Any] | None,
) -> PerBandFluxCalibrationConfig:
    """Resolve per-band flux calibration config into a typed object."""
    block = dict(
        ((config or {}).get("calibration", {}) or {}).get("per_band_zero_points", {})
        or {}
    )
    enabled = bool(block.get("enabled", False))
    mode = str(block.get("mode", "learn_per_band" if enabled else "disabled"))
    if not enabled:
        mode = "disabled"
    prior_sigma_mag = float(block.get("prior_sigma_mag", 0.05))
    if "prior_sigma_log_alpha" in block:
        prior_sigma_log_alpha = float(block["prior_sigma_log_alpha"])
        prior_sigma_mag = float(
            block.get("prior_sigma_mag", log_alpha_to_delta_mag(prior_sigma_log_alpha))
        )
    else:
        prior_sigma_log_alpha = float(delta_mag_to_log_alpha(-prior_sigma_mag))
    prior_sigma_log_alpha = max(abs(prior_sigma_log_alpha), 1.0e-12)
    prior_sigma_mag = max(abs(prior_sigma_mag), 1.0e-12)
    return PerBandFluxCalibrationConfig(
        enabled=enabled,
        mode=mode,
        prior_sigma_log_alpha=prior_sigma_log_alpha,
        prior_sigma_mag=prior_sigma_mag,
        trainable=bool(block.get("trainable", enabled and mode == "learn_per_band")),
    )


def make_per_band_flux_calibration_state(
    config: dict[str, Any] | None,
    band_names: tuple[str, ...],
) -> PerBandFluxCalibrationState | None:
    """Create per-band flux calibration state initialized from config."""
    cfg = per_band_flux_calibration_config(config)
    if not cfg.enabled:
        return None
    block = dict(
        ((config or {}).get("calibration", {}) or {}).get("per_band_zero_points", {})
        or {}
    )
    values = np.zeros(len(band_names), dtype=np.float32)
    offsets_mag = block.get("initial_offsets_mag") or {}
    if isinstance(offsets_mag, dict):
        for index, name in enumerate(band_names):
            if name in offsets_mag:
                values[index] = float(delta_mag_to_log_alpha(float(offsets_mag[name])))
    log_alpha = block.get("initial_log_alpha_band")
    if isinstance(log_alpha, dict):
        for index, name in enumerate(band_names):
            if name in log_alpha:
                values[index] = float(log_alpha[name])
    elif isinstance(log_alpha, (list, tuple)):
        arr = np.asarray(log_alpha, dtype=np.float32)
        if arr.shape != values.shape:
            raise ValueError(
                "calibration.per_band_zero_points.initial_log_alpha_band must "
                f"have length {len(band_names)}"
            )
        values = arr
    return PerBandFluxCalibrationState(
        log_alpha_band=jnp.asarray(values, dtype=jnp.float32)
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


def apply_per_band_flux_calibration_to_flux(
    flux: jnp.ndarray,
    log_alpha_band: jnp.ndarray,
) -> jnp.ndarray:
    """Apply one multiplicative calibration scale per output band."""
    return flux * jnp.exp(log_alpha_band)


def global_sed_scale_prior_penalty(
    log_alpha_sed: jnp.ndarray,
    prior_sigma_log_alpha: float,
) -> jnp.ndarray:
    """Gaussian prior penalty on ``log_alpha_sed``."""
    sigma = jnp.asarray(max(float(prior_sigma_log_alpha), 1.0e-12))
    return 0.5 * (log_alpha_sed / sigma) ** 2


def per_band_flux_calibration_prior_penalty(
    log_alpha_band: jnp.ndarray,
    prior_sigma_log_alpha: float,
) -> jnp.ndarray:
    """Gaussian prior penalty on per-band log flux scales."""
    sigma = jnp.asarray(max(float(prior_sigma_log_alpha), 1.0e-12))
    return 0.5 * jnp.sum((log_alpha_band / sigma) ** 2)


def delta_mag_to_log_alpha(delta_mag: float | np.ndarray) -> float | np.ndarray:
    """Return the flux log-scale equivalent to a magnitude offset."""
    return -np.asarray(delta_mag) * np.log(10.0) / 2.5


def log_alpha_to_delta_mag(log_alpha: float | np.ndarray) -> float | np.ndarray:
    """Return the magnitude offset equivalent to a flux log-scale."""
    return -2.5 * np.asarray(log_alpha) / np.log(10.0)


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


def per_band_flux_calibration_metadata(
    log_alpha_band: np.ndarray,
    band_names: tuple[str, ...],
    prior_sigma_log_alpha: float,
) -> dict[str, Any]:
    """Return JSON-friendly diagnostics for per-band flux calibration."""
    logs = np.asarray(log_alpha_band, dtype=float)
    if logs.shape != (len(band_names),):
        raise ValueError(
            "per-band calibration length does not match band names: "
            f"{logs.shape} vs {len(band_names)}"
        )
    sigma = max(float(prior_sigma_log_alpha), 1.0e-12)
    alpha = np.exp(logs)
    delta_mag = log_alpha_to_delta_mag(logs)
    bands = {
        str(name): {
            "log_alpha_band": float(logs[index]),
            "alpha_band": float(alpha[index]),
            "delta_mag_band": float(delta_mag[index]),
            "prior_penalty": float(0.5 * (logs[index] / sigma) ** 2),
        }
        for index, name in enumerate(band_names)
    }
    return {
        "bands": bands,
        "max_abs_delta_mag": float(np.max(np.abs(delta_mag))) if logs.size else 0.0,
        "mean_abs_delta_mag": float(np.mean(np.abs(delta_mag))) if logs.size else 0.0,
        "total_prior_penalty": float(0.5 * np.sum((logs / sigma) ** 2)),
        "large_scale_warning": bool(np.any(np.abs(delta_mag) > 0.3)),
    }
