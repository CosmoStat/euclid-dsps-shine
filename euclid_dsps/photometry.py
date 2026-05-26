"""Photometric unit conversions.

AB magnitudes use the standard zero point ``0 AB = 3631 Jy``. The pipeline
keeps likelihood fluxes internally as ``Fnu`` in cgs units
``erg s^-1 cm^-2 Hz^-1`` because FS2 catalog flux columns and DSPS photometry
outputs are already represented in that convention.
"""

from __future__ import annotations

from typing import Any

import numpy as np

AB_ZEROPOINT_JY = 3631.0
JY_TO_FNU_CGS = 1.0e-23
FNU_CGS_TO_JY = 1.0 / JY_TO_FNU_CGS
AB_ZEROPOINT_FNU_CGS = AB_ZEROPOINT_JY * JY_TO_FNU_CGS
MAGERR_TO_FRAC_FLUXERR = np.log(10.0) / 2.5


def abmag_to_fnu_jy(mag: Any) -> Any:
    """Convert AB magnitude to ``Fnu`` in Jansky."""
    return AB_ZEROPOINT_JY * np.power(10.0, -0.4 * np.asarray(mag))


def fnu_jy_to_abmag(fnu_jy: Any) -> Any:
    """Convert ``Fnu`` in Jansky to AB magnitude.

    Non-positive fluxes return ``nan`` because AB magnitude is undefined there.
    Flux-space likelihoods should use the flux directly instead.
    """
    flux = np.asarray(fnu_jy)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = -2.5 * np.log10(flux / AB_ZEROPOINT_JY)
    return np.where(np.isfinite(flux) & (flux > 0.0), mag, np.nan)


def magerr_to_fluxerr_jy(mag: Any, mag_err: Any) -> Any:
    """Propagate AB magnitude error to a local ``Fnu`` error in Jansky."""
    return np.abs(abmag_to_fnu_jy(mag)) * MAGERR_TO_FRAC_FLUXERR * np.asarray(mag_err)


def fnu_cgs_to_abmag(flux_fnu_cgs: Any) -> Any:
    """Convert ``Fnu`` cgs to AB magnitude."""
    return fnu_jy_to_abmag(np.asarray(flux_fnu_cgs) * FNU_CGS_TO_JY)


def abmag_to_fnu_cgs(mag: Any) -> Any:
    """Convert AB magnitude to ``Fnu`` cgs."""
    return abmag_to_fnu_jy(mag) * JY_TO_FNU_CGS


def microjy_to_fnu_cgs(flux_microjy: Any) -> Any:
    """Convert microJansky to ``Fnu`` cgs."""
    return np.asarray(flux_microjy) * 1.0e-29


def microjy_to_abmag(flux_microjy: Any) -> Any:
    """Convert microJansky to AB magnitude."""
    return fnu_jy_to_abmag(np.asarray(flux_microjy) * 1.0e-6)


def magerr_to_fluxerr_fnu_cgs(flux_fnu_cgs: Any, mag_err: Any) -> Any:
    """Propagate local magnitude error to ``Fnu`` cgs error around a flux."""
    return np.abs(np.asarray(flux_fnu_cgs)) * MAGERR_TO_FRAC_FLUXERR * np.asarray(
        mag_err
    )


def fluxerr_fnu_cgs_to_magerr(
    flux_fnu_cgs: Any,
    flux_error_fnu_cgs: Any,
    floor: float | None = None,
    ceiling: float | None = None,
) -> Any:
    """Convert a flux-density uncertainty into a local AB-magnitude error."""
    flux = np.asarray(flux_fnu_cgs)
    error = np.asarray(flux_error_fnu_cgs)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.abs(error / flux) / MAGERR_TO_FRAC_FLUXERR
    sigma = np.where(
        np.isfinite(flux) & np.isfinite(error) & (flux > 0.0) & (error > 0.0),
        sigma,
        np.nan,
    )
    if floor is not None and np.isfinite(floor):
        sigma = np.maximum(sigma, float(floor))
    if ceiling is not None and np.isfinite(ceiling):
        sigma = np.minimum(sigma, float(ceiling))
    return sigma


def abmag_to_fnu_cgs_jax(mag: Any) -> Any:
    """JAX-compatible AB magnitude to ``Fnu`` cgs conversion."""
    import jax.numpy as jnp

    return AB_ZEROPOINT_FNU_CGS * jnp.power(10.0, -0.4 * mag)


def magerr_to_fluxerr_fnu_cgs_jax(flux_fnu_cgs: Any, mag_err: Any) -> Any:
    """JAX-compatible local mag-error to ``Fnu`` cgs error propagation."""
    import jax.numpy as jnp

    return jnp.abs(flux_fnu_cgs) * MAGERR_TO_FRAC_FLUXERR * mag_err
