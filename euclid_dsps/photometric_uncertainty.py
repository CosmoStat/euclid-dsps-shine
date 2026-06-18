"""Photometric flux-error and likelihood-sigma helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .photometry import magerr_to_fluxerr_fnu_cgs

DEFAULT_MIN_SIGMA_FNU_CGS = 1.0e-40


def normalize_flux_error_model(model: dict[str, Any] | None) -> dict[str, Any]:
    """Return a normalized flux-error model payload."""
    raw = dict(model or {})
    kind = str(raw.get("type", raw.get("kind", "magnitude_tolerance"))).lower()
    if kind in {"synthetic_snr_error", "synthetic_fractional_snr", "snr"}:
        kind = "fractional_snr"
    if kind in {"mag", "mag_tolerance", "model_tolerance_mag"}:
        kind = "magnitude_tolerance"
    if kind in {"fractional", "frac"}:
        kind = "fractional_floor"
    raw["type"] = kind
    raw.setdefault("min_sigma_fnu_cgs", DEFAULT_MIN_SIGMA_FNU_CGS)
    if kind == "fractional_snr":
        raw.setdefault("snr", 50.0)
    elif kind == "magnitude_tolerance":
        raw.setdefault("sigma_mag", 0.10)
    elif kind == "fractional_floor":
        raw.setdefault("fractional_error", 0.02)
        raw.setdefault("floor_fnu_cgs", 0.0)
    elif kind in {"none", "native"}:
        pass
    else:
        raise ValueError(
            "Unsupported flux error model type. Use 'fractional_snr', "
            "'magnitude_tolerance', 'fractional_floor', 'native', or 'none'."
        )
    return raw


def flux_error_from_model(flux_fnu_cgs: Any, model: dict[str, Any] | None) -> np.ndarray:
    """Compute per-object flux errors in ``fnu_cgs`` for a configured model."""
    flux = np.asarray(flux_fnu_cgs, dtype=float)
    cfg = normalize_flux_error_model(model)
    kind = str(cfg["type"])
    if kind == "none":
        return np.full_like(flux, np.nan, dtype=float)
    if kind == "native":
        raise ValueError("native flux-error model requires an explicit error column")
    if kind == "fractional_snr":
        snr = float(cfg.get("snr", 50.0))
        if not np.isfinite(snr) or snr <= 0.0:
            raise ValueError("fractional_snr error model requires snr > 0")
        error = np.abs(flux) / snr
    elif kind == "magnitude_tolerance":
        error = magerr_to_fluxerr_fnu_cgs(flux, float(cfg.get("sigma_mag", 0.10)))
    elif kind == "fractional_floor":
        frac = float(cfg.get("fractional_error", 0.02))
        floor = float(cfg.get("floor_fnu_cgs", 0.0))
        if frac < 0.0 or floor < 0.0:
            raise ValueError("fractional_floor requires non-negative frac and floor")
        error = np.sqrt((frac * np.abs(flux)) ** 2 + floor**2)
    else:  # pragma: no cover - guarded by normalize_flux_error_model
        raise ValueError(f"Unsupported flux error model: {kind}")
    min_sigma = float(cfg.get("min_sigma_fnu_cgs", DEFAULT_MIN_SIGMA_FNU_CGS))
    if min_sigma > 0.0:
        error = np.maximum(error, min_sigma)
    return np.asarray(error, dtype=float)


def flux_error_model_payload(model: dict[str, Any] | None) -> dict[str, Any]:
    """Return a manifest-friendly flux-error model description."""
    cfg = normalize_flux_error_model(model)
    kind = str(cfg["type"])
    payload: dict[str, Any] = {
        "type": kind,
        "native_error": kind == "native",
        "prepared_error_unit": "fnu_cgs",
    }
    if kind == "fractional_snr":
        payload.update(
            {
                "snr": float(cfg["snr"]),
                "description": "fluxerr_* columns are abs(flux) / snr.",
            }
        )
    elif kind == "magnitude_tolerance":
        payload.update(
            {
                "sigma_mag": float(cfg["sigma_mag"]),
                "description": (
                    "fluxerr_* columns are local AB-magnitude error propagation."
                ),
            }
        )
    elif kind == "fractional_floor":
        payload.update(
            {
                "fractional_error": float(cfg["fractional_error"]),
                "floor_fnu_cgs": float(cfg["floor_fnu_cgs"]),
                "description": (
                    "fluxerr_* columns are sqrt((fractional_error*flux)^2 + floor^2)."
                ),
            }
        )
    elif kind == "none":
        payload.update(
            {
                "native_error": False,
                "description": "No fluxerr_* columns are available.",
            }
        )
    return payload


def effective_flux_sigma(
    obs_flux: Any,
    obs_err: Any,
    *,
    model_flux: Any | None = None,
    error_floor_frac: float = 0.0,
    error_jitter: float = 0.0,
    floor_reference: str = "model",
) -> np.ndarray:
    """Return likelihood-space flux sigma in ``fnu_cgs``.

    ``floor_reference='model'`` matches the amortized JAX likelihood. Use
    ``'observed'`` for legacy standalone MAP outputs.
    """
    obs_flux_arr = np.asarray(obs_flux, dtype=float)
    obs_err_arr = np.asarray(obs_err, dtype=float)
    if model_flux is not None:
        model_flux_arr = np.asarray(model_flux, dtype=float)
        obs_flux_arr = np.broadcast_to(obs_flux_arr, model_flux_arr.shape)
        obs_err_arr = np.broadcast_to(obs_err_arr, model_flux_arr.shape)
    else:
        model_flux_arr = None
    reference = str(floor_reference).lower()
    if reference in {"model", "model_flux"}:
        if model_flux_arr is None:
            floor_flux = obs_flux_arr
        else:
            floor_flux = model_flux_arr
    elif reference in {"observed", "obs", "obs_flux"}:
        floor_flux = obs_flux_arr
    else:
        raise ValueError("floor_reference must be 'model' or 'observed'")
    sigma2 = obs_err_arr**2
    floor_frac = float(error_floor_frac)
    if floor_frac:
        sigma2 = sigma2 + (floor_frac * np.abs(floor_flux)) ** 2
    jitter = float(error_jitter)
    if jitter:
        sigma2 = sigma2 + jitter**2
    return np.sqrt(np.maximum(sigma2, 0.0))
