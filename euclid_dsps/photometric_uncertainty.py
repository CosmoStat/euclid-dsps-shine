"""Photometric flux-error and likelihood-sigma helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .photometry import abmag_to_fnu_cgs, magerr_to_fluxerr_fnu_cgs

DEFAULT_MIN_SIGMA_FNU_CGS = 1.0e-40
DEFAULT_PHOTERR_SIGMA_SYS_MAG = 0.005
DEFAULT_LSST_COADD_M5 = {
    "lsst_u": 26.1,
    "lsst_g": 27.4,
    "lsst_r": 27.5,
    "lsst_i": 26.8,
    "lsst_z": 26.1,
    "lsst_y": 24.9,
}
DEFAULT_LSST_GAMMA = {
    "lsst_u": 0.037,
    "lsst_g": 0.038,
    "lsst_r": 0.039,
    "lsst_i": 0.039,
    "lsst_z": 0.040,
    "lsst_y": 0.040,
}
DEFAULT_ROMAN_WFI_ONE_HOUR_POINT_M5 = {
    "roman_F062": 27.97,
    "roman_F087": 27.63,
    "roman_F106": 27.60,
    "roman_F129": 27.60,
    "roman_F146": 28.01,
    "roman_F158": 27.52,
    "roman_F184": 26.95,
    "roman_F213": 25.64,
}
DEFAULT_ROMAN_ETA = {
    "roman_F062": 0.95,
    "roman_F087": 0.95,
    "roman_F106": 0.95,
    "roman_F129": 0.95,
    "roman_F146": 0.95,
    "roman_F158": 0.95,
    "roman_F184": 0.95,
    "roman_F213": 0.95,
}


def default_m5_depth_error_model() -> dict[str, Any]:
    """Return the default LSST+Roman synthetic depth error model."""
    return {
        "type": "m5_depth",
        "m5": {
            **DEFAULT_LSST_COADD_M5,
            **DEFAULT_ROMAN_WFI_ONE_HOUR_POINT_M5,
        },
        "gamma": dict(DEFAULT_LSST_GAMMA),
        "eta": dict(DEFAULT_ROMAN_ETA),
        "default_eta": 1.0,
        "sigma_sys_mag": DEFAULT_PHOTERR_SIGMA_SYS_MAG,
        "depth_reference": (
            "LSST 10-year coadd for lsst_*; Roman WFI one-hour point-source "
            "for roman_* with eta=0.95 synthetic source-noise mixing; "
            "PhotErr-style sigmaSys=0.005 mag systematic floor."
        ),
    }


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
    elif kind in {"m5_depth", "depth_flux", "limiting_magnitude"}:
        raw["type"] = "m5_depth"
        raw.setdefault("default_eta", 1.0)
        raw.setdefault("sigma_sys_mag", DEFAULT_PHOTERR_SIGMA_SYS_MAG)
    elif kind in {"none", "native"}:
        pass
    else:
        raise ValueError(
            "Unsupported flux error model type. Use 'fractional_snr', "
            "'magnitude_tolerance', 'fractional_floor', 'm5_depth', "
            "'native', or 'none'."
        )
    return raw


def flux_error_from_model(
    flux_fnu_cgs: Any,
    model: dict[str, Any] | None,
    *,
    band_name: str | None = None,
) -> np.ndarray:
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
    elif kind == "m5_depth":
        error = _m5_depth_flux_error(flux, cfg, band_name=band_name)
    else:  # pragma: no cover - guarded by normalize_flux_error_model
        raise ValueError(f"Unsupported flux error model: {kind}")
    min_sigma = float(cfg.get("min_sigma_fnu_cgs", DEFAULT_MIN_SIGMA_FNU_CGS))
    if min_sigma > 0.0:
        error = np.maximum(error, min_sigma)
    return np.asarray(error, dtype=float)


def _m5_depth_flux_error(
    flux_fnu_cgs: np.ndarray,
    model: dict[str, Any],
    *,
    band_name: str | None,
) -> np.ndarray:
    """Return LSST/Roman-style synthetic errors from per-band 5-sigma depths."""
    m5 = _model_value_for_band(model, ("m5", "m5_by_band", "depth_m5"), band_name)
    f5 = np.asarray(abmag_to_fnu_cgs(float(m5)), dtype=float)
    gamma = _gamma_for_band(model, band_name)
    flux = np.abs(np.asarray(flux_fnu_cgs, dtype=float))
    sigma2 = (0.04 - gamma) * flux * f5 + gamma * f5**2
    sigma_sys_mag = _sigma_sys_mag_for_model(model)
    if sigma_sys_mag > 0.0:
        sys_frac = np.expm1(np.log(10.0) * sigma_sys_mag / 2.5)
        sigma2 = sigma2 + (sys_frac * flux) ** 2
    return np.sqrt(np.maximum(sigma2, 0.0))


def _sigma_sys_mag_for_model(model: dict[str, Any]) -> float:
    sigma_sys_mag = float(model.get("sigma_sys_mag", DEFAULT_PHOTERR_SIGMA_SYS_MAG))
    if not np.isfinite(sigma_sys_mag) or sigma_sys_mag < 0.0:
        raise ValueError("m5_depth requires sigma_sys_mag >= 0")
    return sigma_sys_mag


def _gamma_for_band(model: dict[str, Any], band_name: str | None) -> float:
    gamma = _optional_model_value_for_band(model, ("gamma", "gamma_by_band"), band_name)
    if gamma is None:
        eta = _optional_model_value_for_band(model, ("eta", "eta_by_band"), band_name)
        if eta is None:
            eta = float(model.get("default_eta", 1.0))
        gamma = 0.04 * float(eta)
    gamma_f = float(gamma)
    if not np.isfinite(gamma_f) or gamma_f < 0.0 or gamma_f > 0.04:
        raise ValueError("m5_depth requires 0 <= gamma <= 0.04")
    return gamma_f


def _model_value_for_band(
    model: dict[str, Any],
    keys: tuple[str, ...],
    band_name: str | None,
) -> float:
    value = _optional_model_value_for_band(model, keys, band_name)
    if value is None:
        raise ValueError(
            f"m5_depth requires one of {keys} as a scalar or a mapping for band {band_name!r}"
        )
    value_f = float(value)
    if not np.isfinite(value_f):
        raise ValueError(f"m5_depth value for band {band_name!r} must be finite")
    return value_f


def _optional_model_value_for_band(
    model: dict[str, Any],
    keys: tuple[str, ...],
    band_name: str | None,
) -> float | None:
    aliases = _band_aliases(band_name)
    for key in keys:
        if key not in model:
            continue
        value = model[key]
        if isinstance(value, dict):
            for alias in aliases:
                if alias in value:
                    return float(value[alias])
            if "default" in value:
                return float(value["default"])
            continue
        return float(value)
    return None


def _band_aliases(band_name: str | None) -> tuple[str, ...]:
    if band_name is None:
        return ()
    name = str(band_name)
    aliases = [name]
    lower = name.lower()
    aliases.append(lower)
    if lower.startswith("roman_"):
        suffix = name.split("_", 1)[1]
        aliases.extend([suffix, suffix.upper(), suffix.lower()])
    if lower.startswith("lsst_"):
        suffix = lower.removeprefix("lsst_")
        aliases.extend([suffix, suffix.upper()])
    return tuple(dict.fromkeys(aliases))


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
    elif kind == "m5_depth":
        payload.update(
            {
                "m5": cfg.get("m5", cfg.get("m5_by_band", cfg.get("depth_m5"))),
                "gamma": cfg.get("gamma", cfg.get("gamma_by_band")),
                "eta": cfg.get("eta", cfg.get("eta_by_band")),
                "default_eta": float(cfg.get("default_eta", 1.0)),
                "sigma_sys_mag": float(
                    cfg.get("sigma_sys_mag", DEFAULT_PHOTERR_SIGMA_SYS_MAG)
                ),
                "depth_reference": cfg.get("depth_reference"),
                "description": (
                    "fluxerr_* columns use per-band 5-sigma depths: "
                    "sigma_rand^2=(0.04-gamma)*abs(flux)*f5 + gamma*f5^2, "
                    "sigma^2=sigma_rand^2 + "
                    "(sys_frac*abs(flux))^2, with f5=fnu(m5) and "
                    "sys_frac=10^(sigma_sys_mag/2.5)-1."
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
