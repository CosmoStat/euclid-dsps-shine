"""OpenUniverse data-side photon-rate photometry helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np

from euclid_dsps.photometry import AB_ZEROPOINT_FNU_CGS

from .filter_curves import OpenUniverseFilterCurve

PLANCK_ERG_S = 6.62607015e-27
LIGHT_SPEED_ANGSTROM_S = 2.99792458e18

SedFnuUnit = Literal["native", "fnu_cgs", "jy", "microjy", "nanojy"]

SED_FNU_TO_CGS_SCALE = {
    "native": 1.0,
    "fnu_cgs": 1.0,
    "jy": 1.0e-23,
    "microjy": 1.0e-29,
    "nanojy": 1.0e-32,
}


def photon_rate_from_fnu_sed(
    sed_wave_angstrom: np.ndarray,
    fnu: np.ndarray,
    filter_curve: OpenUniverseFilterCurve,
    *,
    fnu_unit: SedFnuUnit = "native",
    fnu_scale: float = 1.0,
) -> float:
    """Integrate an ``Fnu`` SED through a filter as photons/sec/cm^2.

    The integration is intentionally unnormalized by filter throughput because
    OpenUniverse flux columns are integrated photon rates, not AB flux-density
    averages. ``native`` means "use the numeric SED values as-is"; closure
    reports can then infer per-band calibration factors from the public flux
    table instead of silently asserting an unknown unit convention.
    """
    wave = np.asarray(sed_wave_angstrom, dtype=float)
    values = np.asarray(fnu, dtype=float)
    if wave.ndim != 1 or values.ndim != 1:
        raise ValueError("sed_wave_angstrom and fnu must be one-dimensional")
    if wave.shape[0] != values.shape[0]:
        raise ValueError(
            f"SED wavelength/value lengths differ: {wave.shape[0]} vs {values.shape[0]}"
        )
    scale = _fnu_unit_scale(fnu_unit) * float(fnu_scale)
    filter_wave = np.asarray(filter_curve.wave_angstrom, dtype=float)
    transmission = np.asarray(filter_curve.transmission, dtype=float)
    fnu_on_filter = np.interp(filter_wave, wave, values, left=0.0, right=0.0) * scale
    flambda = fnu_on_filter * LIGHT_SPEED_ANGSTROM_S / np.square(filter_wave)
    photon_density = (
        flambda
        * transmission
        * filter_wave
        / (PLANCK_ERG_S * LIGHT_SPEED_ANGSTROM_S)
    )
    rate = np.trapezoid(photon_density, filter_wave)
    if not np.isfinite(rate):
        return float("nan")
    return float(rate)


def photon_rates_from_fnu_sed(
    sed_wave_angstrom: np.ndarray,
    fnu: np.ndarray,
    filter_curves: dict[str, OpenUniverseFilterCurve],
    *,
    fnu_unit: SedFnuUnit = "native",
    fnu_scale: float = 1.0,
) -> dict[str, float]:
    """Integrate one SED through several OpenUniverse filter curves."""
    return {
        band: photon_rate_from_fnu_sed(
            sed_wave_angstrom,
            fnu,
            curve,
            fnu_unit=fnu_unit,
            fnu_scale=fnu_scale,
        )
        for band, curve in filter_curves.items()
    }


def ab0_photon_rate(
    filter_curve: OpenUniverseFilterCurve,
    *,
    n_wave: int = 2048,
) -> float:
    """Return the photon rate of a flat 0 AB source through one filter."""
    wave = np.linspace(
        float(np.nanmin(filter_curve.wave_angstrom)),
        float(np.nanmax(filter_curve.wave_angstrom)),
        int(n_wave),
    )
    fnu = np.full_like(wave, AB_ZEROPOINT_FNU_CGS, dtype=float)
    return photon_rate_from_fnu_sed(
        wave,
        fnu,
        filter_curve,
        fnu_unit="fnu_cgs",
    )


def photon_rate_to_fnu_cgs(
    photon_rate: np.ndarray,
    photon_rate_ab0: float,
) -> np.ndarray:
    """Convert integrated photon rate to equivalent AB ``Fnu`` cgs."""
    rate = np.asarray(photon_rate, dtype=float)
    reference = float(photon_rate_ab0)
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("photon_rate_ab0 must be finite and positive")
    return rate / reference * AB_ZEROPOINT_FNU_CGS


def _fnu_unit_scale(unit: str) -> float:
    normalized = str(unit).lower().strip()
    if normalized not in SED_FNU_TO_CGS_SCALE:
        raise ValueError(
            f"Unsupported SED Fnu unit {unit!r}; expected "
            f"{tuple(SED_FNU_TO_CGS_SCALE)}"
        )
    return float(SED_FNU_TO_CGS_SCALE[normalized])
