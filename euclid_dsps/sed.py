"""Photometry-anchored empirical SED diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .io import GalaxyObservation
from .model import ModelResult

L_SUN_ERG_PER_S = 3.828e33
MPC_CM = 3.0856775814913673e24
PC_CM = 3.0856775814913673e18
TEN_PC_CM = 10.0 * PC_CM
C_LIGHT_KM_PER_S = 299_792.458
PLANCK15_OM0 = 0.3075
PLANCK15_H0 = 67.74


@dataclass(frozen=True)
class EmpiricalSed:
    """Photometry-anchored pseudo-SED and comparison with one DSPS result."""

    points: pd.DataFrame
    continuous: pd.DataFrame
    summary: dict[str, float | int | str]


def luminosity_distance_cm(
    redshift: float,
    h0: float = PLANCK15_H0,
    om0: float = PLANCK15_OM0,
    n_grid: int = 2048,
) -> float:
    """Return flat-LambdaCDM luminosity distance in cm.

    This mirrors the DSPS default Planck15 cosmology closely enough for
    diagnostics without importing native DSPS during tests.
    """
    z = float(redshift)
    if not np.isfinite(z) or z < 0:
        return float("nan")
    if z == 0:
        return 0.0
    grid = np.linspace(0.0, z, max(int(n_grid), 8))
    ez = np.sqrt(om0 * (1.0 + grid) ** 3 + (1.0 - om0))
    comoving_mpc = (C_LIGHT_KM_PER_S / h0) * np.trapezoid(1.0 / ez, grid)
    return float((1.0 + z) * comoving_mpc * MPC_CM)


def observed_fnu_to_rest_lnu_lsun(
    flux_fnu_cgs: float,
    redshift: float,
    luminosity_distance: float | None = None,
) -> float:
    """Convert observed ``Fnu`` to rest-frame ``Lnu`` in ``Lsun/Hz``."""
    flux = float(flux_fnu_cgs)
    z = float(redshift)
    d_l = (
        luminosity_distance_cm(z)
        if luminosity_distance is None
        else luminosity_distance
    )
    if not np.isfinite(flux) or not np.isfinite(z) or not np.isfinite(d_l):
        return float("nan")
    return float(4.0 * np.pi * d_l**2 * (1.0 + z) * flux / L_SUN_ERG_PER_S)


def rest_lnu_lsun_to_observed_fnu(
    lnu_lsun: float,
    redshift: float,
    luminosity_distance: float | None = None,
) -> float:
    """Convert rest-frame ``Lnu`` in ``Lsun/Hz`` to observed ``Fnu`` cgs."""
    luminosity = float(lnu_lsun)
    z = float(redshift)
    d_l = (
        luminosity_distance_cm(z)
        if luminosity_distance is None
        else luminosity_distance
    )
    if (
        not np.isfinite(luminosity)
        or not np.isfinite(z)
        or not np.isfinite(d_l)
        or d_l <= 0
    ):
        return float("nan")
    return float(luminosity * L_SUN_ERG_PER_S / (4.0 * np.pi * d_l**2 * (1.0 + z)))


def rest_10pc_fnu_to_lnu_lsun(flux_fnu_cgs: float) -> float:
    """Convert rest-frame flux at 10 pc to ``Lnu`` in ``Lsun/Hz``."""
    flux = float(flux_fnu_cgs)
    if not np.isfinite(flux) or flux <= 0:
        return float("nan")
    return float(4.0 * np.pi * TEN_PC_CM**2 * flux / L_SUN_ERG_PER_S)


def reconstruct_empirical_sed(
    observation: GalaxyObservation,
    result: ModelResult,
    wave_grid: np.ndarray | None = None,
) -> EmpiricalSed:
    """Build a broad-band, photometry-anchored pseudo-SED.

    The reconstruction only interpolates the catalog broad-band luminosity
    points. It is not a template-level Flagship ground-truth SED.
    """
    points = empirical_sed_points(observation, result)
    wave = np.asarray(result.wave if wave_grid is None else wave_grid, dtype=float)
    empirical = interpolate_empirical_lnu(
        wave,
        points["rest_wavelength_angstrom"].to_numpy(dtype=float),
        points["inferred_rest_lnu_lsun_per_hz"].to_numpy(dtype=float),
    )
    dusted = np.asarray(result.dusted_rest_sed, dtype=float)
    intrinsic = np.asarray(result.rest_sed, dtype=float)
    continuous = pd.DataFrame(
        {
            "wave_angstrom": wave,
            "empirical_dusted_lnu_lsun_per_hz": empirical,
            "dsps_dusted_lnu_lsun_per_hz": dusted,
            "dsps_intrinsic_lnu_lsun_per_hz": intrinsic,
            "empirical_over_dsps_dusted": _safe_ratio(empirical, dusted),
            "log10_empirical_over_dsps_dusted": _safe_log10_ratio(empirical, dusted),
        }
    )
    return EmpiricalSed(
        points=points,
        continuous=continuous,
        summary=empirical_sed_summary(points),
    )


def empirical_sed_points(
    observation: GalaxyObservation, result: ModelResult
) -> pd.DataFrame:
    """Return rest-frame luminosity points inferred from catalog photometry."""
    z_obs = float(result.parameters.get("z_obs", np.nan))
    d_l = luminosity_distance_cm(z_obs)
    rows: list[dict[str, Any]] = []
    for band in observation.bands:
        model_band = result.photometry.get(band.name, {})
        wave_obs = float(model_band.get("effective_wavelength_angstrom", np.nan))
        rest_flux_column = rest_10pc_flux_column(band.column, observation.row)
        rest_flux = _row_float(observation.row, rest_flux_column)
        if rest_flux_column and np.isfinite(rest_flux) and rest_flux > 0:
            wave_rest = wave_obs
            inferred_lnu = rest_10pc_fnu_to_lnu_lsun(rest_flux)
            source_kind = "rest_10pc_flux"
            source_column = rest_flux_column
            source_flux = rest_flux
        else:
            wave_rest = wave_obs / (1.0 + z_obs) if np.isfinite(z_obs) else np.nan
            inferred_lnu = observed_fnu_to_rest_lnu_lsun(
                band.flux_fnu_cgs,
                z_obs,
                luminosity_distance=d_l,
            )
            source_kind = "observed_flux_luminosity_distance"
            source_column = band.column
            source_flux = float(band.flux_fnu_cgs)
        frac_sigma = magnitude_sigma_to_fractional_flux(band.sigma_mag)
        sigma_lnu = abs(inferred_lnu) * frac_sigma
        dsps_lnu = interpolate_lnu_at_wavelength(
            result.wave, result.dusted_rest_sed, wave_rest
        )
        dsps_intrinsic_lnu = interpolate_lnu_at_wavelength(
            result.wave, result.rest_sed, wave_rest
        )
        rows.append(
            {
                "band": band.name,
                "column": band.column,
                "observed_effective_wavelength_angstrom": wave_obs,
                "rest_wavelength_angstrom": wave_rest,
                "observed_flux_fnu_cgs": float(band.flux_fnu_cgs),
                "sed_flux_source_kind": source_kind,
                "sed_flux_source_column": source_column,
                "sed_flux_source_fnu_cgs": source_flux,
                "observed_mag_ab": float(band.mag_ab),
                "sigma_mag": float(band.sigma_mag),
                "inferred_rest_lnu_lsun_per_hz": inferred_lnu,
                "sigma_rest_lnu_lsun_per_hz": sigma_lnu,
                "sigma_log10_lnu": frac_sigma / np.log(10.0),
                "dsps_dusted_lnu_lsun_per_hz": dsps_lnu,
                "dsps_intrinsic_lnu_lsun_per_hz": dsps_intrinsic_lnu,
                "empirical_over_dsps_dusted": _safe_scalar_ratio(
                    inferred_lnu, dsps_lnu
                ),
                "log10_empirical_over_dsps_dusted": _safe_scalar_log10_ratio(
                    inferred_lnu, dsps_lnu
                ),
                "redshift": z_obs,
                "luminosity_distance_cm": d_l,
                "reconstruction_kind": "broadband_log_interpolated_pseudo_sed",
            }
        )
    return pd.DataFrame(rows).sort_values("rest_wavelength_angstrom")


def rest_10pc_flux_column(observed_flux_column: str, row: dict[str, Any]) -> str | None:
    """Return matching ``*_abs`` rest-frame flux column when available."""
    candidates = [f"{observed_flux_column}_abs"]
    for candidate in candidates:
        if candidate in row:
            return candidate
    return None


def _row_float(row: dict[str, Any], column: str | None) -> float:
    if not column:
        return float("nan")
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def interpolate_empirical_lnu(
    wave_angstrom: np.ndarray,
    point_wave_angstrom: np.ndarray,
    point_lnu_lsun_per_hz: np.ndarray,
) -> np.ndarray:
    """Log-log interpolation through broad-band luminosity points."""
    wave = np.asarray(wave_angstrom, dtype=float)
    point_wave = np.asarray(point_wave_angstrom, dtype=float)
    point_lnu = np.asarray(point_lnu_lsun_per_hz, dtype=float)
    output = np.full_like(wave, np.nan, dtype=float)
    mask = (
        np.isfinite(point_wave)
        & np.isfinite(point_lnu)
        & (point_wave > 0)
        & (point_lnu > 0)
    )
    if mask.sum() < 2:
        return output
    order = np.argsort(point_wave[mask])
    log_wave_points = np.log(point_wave[mask][order])
    log_lnu_points = np.log(point_lnu[mask][order])
    grid_mask = (
        np.isfinite(wave)
        & (wave > 0)
        & (wave >= point_wave[mask][order][0])
        & (wave <= point_wave[mask][order][-1])
    )
    output[grid_mask] = np.exp(
        np.interp(np.log(wave[grid_mask]), log_wave_points, log_lnu_points)
    )
    return output


def interpolate_lnu_at_wavelength(
    wave_angstrom: np.ndarray, lnu_lsun_per_hz: np.ndarray, target_wave: float
) -> float:
    """Interpolate an SED in log space at one positive wavelength."""
    wave = np.asarray(wave_angstrom, dtype=float)
    lnu = np.asarray(lnu_lsun_per_hz, dtype=float)
    target = float(target_wave)
    mask = np.isfinite(wave) & np.isfinite(lnu) & (wave > 0) & (lnu > 0)
    if mask.sum() < 2 or not np.isfinite(target) or target <= 0:
        return float("nan")
    order = np.argsort(wave[mask])
    wave_sorted = wave[mask][order]
    if target < wave_sorted[0] or target > wave_sorted[-1]:
        return float("nan")
    log_value = np.interp(
        np.log(target),
        np.log(wave_sorted),
        np.log(lnu[mask][order]),
    )
    return float(np.exp(log_value))


def magnitude_sigma_to_fractional_flux(sigma_mag: float) -> float:
    """Convert small AB-magnitude uncertainty to fractional flux uncertainty."""
    sigma = float(sigma_mag)
    if not np.isfinite(sigma) or sigma < 0:
        return float("nan")
    return float(np.log(10.0) * sigma / 2.5)


def empirical_sed_summary(points: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize empirical-vs-DSPS agreement at broad-band points."""
    residual = points["log10_empirical_over_dsps_dusted"].replace(
        [np.inf, -np.inf], np.nan
    )
    finite = residual.dropna()
    summary: dict[str, float | int | str] = {
        "n_photometric_points": int(len(points)),
        "n_finite_comparison_points": int(len(finite)),
        "reconstruction_kind": "broadband_log_interpolated_pseudo_sed",
        "warning": (
            "Pseudo-SED interpolates broad-band photometry only; it is not a "
            "template-level catalog ground truth."
        ),
    }
    if not finite.empty:
        summary.update(
            {
                "median_log10_empirical_over_dsps_dusted": float(finite.median()),
                "median_abs_log10_empirical_over_dsps_dusted": float(
                    finite.abs().median()
                ),
                "rms_log10_empirical_over_dsps_dusted": float(
                    np.sqrt(np.mean(finite.to_numpy(dtype=float) ** 2))
                ),
            }
        )
    return summary


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=np.isfinite(numerator)
        & np.isfinite(denominator)
        & (numerator > 0)
        & (denominator > 0),
    )


def _safe_log10_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    ratio = _safe_ratio(numerator, denominator)
    return np.where(ratio > 0, np.log10(ratio), np.nan)


def _safe_scalar_ratio(numerator: float, denominator: float) -> float:
    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or numerator <= 0
        or denominator <= 0
    ):
        return float("nan")
    return float(numerator / denominator)


def _safe_scalar_log10_ratio(numerator: float, denominator: float) -> float:
    ratio = _safe_scalar_ratio(numerator, denominator)
    return float(np.log10(ratio)) if ratio > 0 else float("nan")
