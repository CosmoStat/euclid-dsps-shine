"""Native DSPS model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from dsps import calc_obs_mag, calc_rest_sed_sfh_table_lognormal_mdf, load_ssp_templates
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
from dsps.dust.att_curves import _frac_transmission_from_k_lambda, sbl18_k_lambda

from .filters import FilterCurve
from .io import GalaxyObservation, abmag_to_flux_fnu_cgs


@dataclass
class DspsContext:
    ssp: Any
    filters: dict[str, FilterCurve]
    n_sfh_bins: int = 96


@dataclass(frozen=True)
class ModelResult:
    parameters: dict[str, float]
    derived: dict[str, float]
    wave: np.ndarray
    rest_sed: np.ndarray
    dusted_rest_sed: np.ndarray
    photometry: dict[str, dict[str, float]]


def load_context(ssp_path: str, filters: dict[str, FilterCurve], n_sfh_bins: int = 96) -> DspsContext:
    return DspsContext(ssp=load_ssp_templates(fn=ssp_path), filters=filters, n_sfh_bins=n_sfh_bins)


def parameters_for_row(
    base: dict[str, Any],
    parameter_columns: dict[str, str],
    row: dict[str, Any],
    redshift_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Merge fixed config parameters with optional per-row catalog overrides."""
    params = {key: float(value) for key, value in base.items()}
    for param_name, column in (parameter_columns or {}).items():
        if column in row and np.isfinite(row[column]):
            params[param_name] = float(row[column])
    params["z_obs"] = resolve_redshift(params, row, redshift_config or {})
    return params


def resolve_redshift(params: dict[str, float], row: dict[str, Any], redshift_config: dict[str, Any]) -> float:
    """Resolve the redshift used by DSPS from a catalog column or fixed fallback."""
    value = params.get("z_obs", redshift_config.get("fixed_value", 0.5))
    column = redshift_config.get("column")
    if column and column in row and np.isfinite(row[column]):
        value = float(row[column])
    elif np.isfinite(redshift_config.get("fixed_value", np.nan)):
        value = float(redshift_config["fixed_value"])

    z_min = float(redshift_config.get("min", 1.0e-4))
    z_max = float(redshift_config.get("max", 6.0))
    if not np.isfinite(value):
        value = z_min
    return float(np.clip(value, z_min, z_max))


def run_dsps_model(context: DspsContext, params: dict[str, float]) -> ModelResult:
    """Run DSPS from simple SFH/metallicity parameters to SED and photometry."""
    ssp = context.ssp
    z_obs = float(params["z_obs"])
    t_obs = float(np.ravel(np.asarray(age_at_z(z_obs, *DEFAULT_COSMOLOGY)))[0])

    gal_t_table = np.linspace(0.05, max(t_obs, 0.06), context.n_sfh_bins)
    gal_sfr_table = build_lognormal_sfh(
        gal_t_table=gal_t_table,
        log10_sfr=float(params["log10_sfr"]),
        sfh_t_peak=float(params["sfh_t_peak"]),
        sfh_tau=float(params["sfh_tau"]),
    )
    formed_mass = float(np.trapezoid(gal_sfr_table, gal_t_table) * 1.0e9)

    sed_info = calc_rest_sed_sfh_table_lognormal_mdf(
        gal_t_table,
        gal_sfr_table,
        float(params["log10_metallicity"]),
        float(params["metallicity_scatter"]),
        ssp.ssp_lgmet,
        ssp.ssp_lg_age_gyr,
        ssp.ssp_flux,
        t_obs,
    )
    rest_sed = np.asarray(sed_info.rest_sed, dtype=float)
    wave = np.asarray(ssp.ssp_wave, dtype=float)
    dusted_sed = apply_dust(wave, rest_sed, params)

    photometry: dict[str, dict[str, float]] = {}
    for name, curve in context.filters.items():
        mag = float(
            np.asarray(
                calc_obs_mag(
                    wave,
                    dusted_sed,
                    curve.wave,
                    curve.transmission,
                    z_obs,
                    *DEFAULT_COSMOLOGY,
                )
            )
        )
        photometry[name] = {
            "model_mag_ab": mag,
            "model_flux_fnu_cgs": abmag_to_flux_fnu_cgs(mag),
            "filter_source": curve.source,
            "effective_wavelength_angstrom": curve.effective_wavelength,
        }

    return ModelResult(
        parameters={key: float(value) for key, value in params.items()},
        derived={
            "t_obs_gyr": t_obs,
            "formed_mass_msun": formed_mass,
            "log10_formed_mass_msun": float(np.log10(formed_mass)) if formed_mass > 0 else float("nan"),
            "sfr_at_t_obs_msun_per_yr": float(gal_sfr_table[-1]),
        },
        wave=wave,
        rest_sed=rest_sed,
        dusted_rest_sed=dusted_sed,
        photometry=photometry,
    )


def build_lognormal_sfh(gal_t_table: np.ndarray, log10_sfr: float, sfh_t_peak: float, sfh_tau: float) -> np.ndarray:
    """Build a positive, smooth SFH in Msun/yr on cosmic-time bins."""
    amplitude = 10**log10_sfr
    t_peak = np.clip(sfh_t_peak, gal_t_table.min(), gal_t_table.max())
    tau = max(float(sfh_tau), 0.05)
    log_t = np.log(np.clip(gal_t_table, 1e-3, None))
    shape = np.exp(-0.5 * ((log_t - np.log(t_peak)) / tau) ** 2)
    shape = np.clip(shape, 1e-6, None)
    return amplitude * shape


def apply_dust(wave_angstrom: np.ndarray, rest_sed: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Apply a DSPS Salim+2018-style attenuation curve."""
    av = max(float(params.get("dust_av", 0.0)), 0.0)
    if av == 0.0:
        return rest_sed
    wave_micron = wave_angstrom / 10_000.0
    k_lambda = sbl18_k_lambda(
        wave_micron,
        0.0,
        float(params.get("dust_slope", -0.7)),
    )
    transmission = np.asarray(_frac_transmission_from_k_lambda(k_lambda, av), dtype=float)
    return rest_sed * transmission


def comparison_rows(observation: GalaxyObservation, result: ModelResult) -> list[dict[str, float | str]]:
    rows = []
    for observed in observation.bands:
        model = result.photometry[observed.name]
        residual = observed.mag_ab - model["model_mag_ab"]
        flux_ratio = model["model_flux_fnu_cgs"] / observed.flux_fnu_cgs if observed.flux_fnu_cgs > 0 else float("nan")
        rows.append(
            {
                "band": observed.name,
                "column": observed.column,
                "effective_wavelength_angstrom": model["effective_wavelength_angstrom"],
                "observed_flux_fnu_cgs": observed.flux_fnu_cgs,
                "observed_mag_ab": observed.mag_ab,
                "sigma_mag": observed.sigma_mag,
                "model_flux_fnu_cgs": model["model_flux_fnu_cgs"],
                "model_mag_ab": model["model_mag_ab"],
                "residual_mag_observed_minus_model": residual,
                "residual_mag_model_minus_observed": -residual,
                "flux_ratio_model_over_observed": flux_ratio,
                "fractional_flux_residual_model_minus_observed": flux_ratio - 1.0,
                "chi": residual / observed.sigma_mag,
                "filter_source": model["filter_source"],
            }
        )
    return rows
