"""Photometric likelihood helpers.

Science inference should use flux space by default. Magnitude-space chi-square
is kept for legacy runs and diagnostics.
"""

from __future__ import annotations

import numpy as np

from .io import GalaxyObservation
from .model import ModelResult


def chi2_flux(
    model_flux,
    obs_flux=None,
    obs_flux_err=None,
    mask=None,
    floor_frac: float | None = None,
    jitter: float | None = None,
) -> float:
    """Return Gaussian chi-square in flux space.

    Invalid values, non-positive errors, and explicitly masked bands do not
    contribute. ``floor_frac`` and ``jitter`` inflate the effective uncertainty:

    ``sigma_eff^2 = sigma^2 + (floor_frac * obs_flux)^2 + jitter^2``.
    """
    if isinstance(model_flux, GalaxyObservation):
        if not isinstance(obs_flux, ModelResult):
            raise TypeError("chi2_flux(observation, result) needs a ModelResult")
        return _chi2_flux_observation(model_flux, obs_flux, floor_frac, jitter)

    model_arr = np.asarray(model_flux, dtype=float)
    obs_arr = np.asarray(obs_flux, dtype=float)
    err_arr = np.asarray(obs_flux_err, dtype=float)
    valid = _valid_flux_mask(model_arr, obs_arr, err_arr)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    sigma_eff = effective_flux_sigma(
        obs_arr, err_arr, floor_frac=floor_frac, jitter=jitter
    )
    valid &= np.isfinite(sigma_eff) & (sigma_eff > 0.0)
    if not np.any(valid):
        return 0.0
    chi = (model_arr[valid] - obs_arr[valid]) / sigma_eff[valid]
    return float(np.sum(chi**2))


def loglike_flux_gaussian(
    model_flux,
    obs_flux=None,
    obs_flux_err=None,
    mask=None,
    floor_frac: float | None = None,
    jitter: float | None = None,
) -> float:
    """Return Gaussian flux log likelihood without the normalization constant."""
    return -0.5 * chi2_flux(
        model_flux,
        obs_flux,
        obs_flux_err,
        mask=mask,
        floor_frac=floor_frac,
        jitter=jitter,
    )


def chi2_mag(
    model_mag,
    obs_mag=None,
    obs_mag_err=None,
    mask=None,
) -> float:
    """Legacy AB-magnitude chi-square.

    Supports both the old ``chi2_mag(observation, result)`` call pattern and the
    explicit array form ``chi2_mag(model_mag, obs_mag, obs_mag_err)``.
    """
    if isinstance(model_mag, GalaxyObservation):
        if not isinstance(obs_mag, ModelResult):
            raise TypeError("chi2_mag(observation, result) needs a ModelResult")
        return _chi2_mag_observation(model_mag, obs_mag)

    model_arr = np.asarray(model_mag, dtype=float)
    obs_arr = np.asarray(obs_mag, dtype=float)
    err_arr = np.asarray(obs_mag_err, dtype=float)
    valid = np.isfinite(model_arr) & np.isfinite(obs_arr) & np.isfinite(err_arr)
    valid &= err_arr > 0.0
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return 0.0
    chi = (obs_arr[valid] - model_arr[valid]) / err_arr[valid]
    return float(np.sum(chi**2))


def log_likelihood_mag(model_mag, obs_mag=None, obs_mag_err=None, mask=None) -> float:
    """Legacy Gaussian log likelihood in magnitude space."""
    return -0.5 * chi2_mag(model_mag, obs_mag, obs_mag_err, mask=mask)


def effective_flux_sigma(
    obs_flux,
    obs_flux_err,
    floor_frac: float | None = None,
    jitter: float | None = None,
):
    """Return effective flux uncertainty with optional fractional floor/jitter."""
    obs = np.asarray(obs_flux, dtype=float)
    err = np.asarray(obs_flux_err, dtype=float)
    floor_term = 0.0 if floor_frac is None else float(floor_frac) * obs
    jitter_term = 0.0 if jitter is None else float(jitter)
    sigma2 = err**2 + floor_term**2 + jitter_term**2
    with np.errstate(invalid="ignore"):
        return np.sqrt(sigma2)


def _valid_flux_mask(model_flux, obs_flux, obs_flux_err):
    return (
        np.isfinite(model_flux)
        & np.isfinite(obs_flux)
        & np.isfinite(obs_flux_err)
        & (obs_flux_err > 0.0)
    )


def _chi2_mag_observation(observation: GalaxyObservation, result: ModelResult) -> float:
    model = [result.photometry[band.name]["model_mag_ab"] for band in observation.bands]
    obs = [band.mag_ab for band in observation.bands]
    err = [band.sigma_mag for band in observation.bands]
    return chi2_mag(model, obs, err)


def _chi2_flux_observation(
    observation: GalaxyObservation,
    result: ModelResult,
    floor_frac: float | None,
    jitter: float | None,
) -> float:
    model = [
        result.photometry[band.name]["model_flux_fnu_cgs"] for band in observation.bands
    ]
    obs = [band.flux_fnu_cgs for band in observation.bands]
    err = [
        band.flux_error_fnu_cgs if band.flux_error_fnu_cgs is not None else np.nan
        for band in observation.bands
    ]
    return chi2_flux(model, obs, err, floor_frac=floor_frac, jitter=jitter)
