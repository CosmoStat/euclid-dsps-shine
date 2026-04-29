"""Native DSPS model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jax
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


@dataclass(frozen=True)
class JaxModelResult:
    wave: jnp.ndarray
    rest_sed: jnp.ndarray
    dusted_rest_sed: jnp.ndarray
    model_mags: jnp.ndarray
    t_obs_gyr: jnp.ndarray
    formed_mass_msun: jnp.ndarray
    sfr_at_t_obs_msun_per_yr: jnp.ndarray


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
    jax_result = run_dsps_model_jax(context, params)
    rest_sed = np.asarray(jax_result.rest_sed, dtype=float)
    wave = np.asarray(jax_result.wave, dtype=float)
    dusted_sed = np.asarray(jax_result.dusted_rest_sed, dtype=float)
    model_mags = np.asarray(jax_result.model_mags, dtype=float)

    photometry: dict[str, dict[str, float]] = {}
    for (name, curve), mag in zip(context.filters.items(), model_mags):
        photometry[name] = {
            "model_mag_ab": float(mag),
            "model_flux_fnu_cgs": abmag_to_flux_fnu_cgs(float(mag)),
            "filter_source": curve.source,
            "effective_wavelength_angstrom": curve.effective_wavelength,
            "filter_wave_angstrom": curve.wave,
            "filter_transmission": curve.transmission,
        }

    return ModelResult(
        parameters={key: float(value) for key, value in params.items()},
        derived={
            "t_obs_gyr": float(jax_result.t_obs_gyr),
            "formed_mass_msun": float(jax_result.formed_mass_msun),
            "log10_formed_mass_msun": _safe_log10(float(jax_result.formed_mass_msun)),
            "sfr_at_t_obs_msun_per_yr": float(jax_result.sfr_at_t_obs_msun_per_yr),
        },
        wave=wave,
        rest_sed=rest_sed,
        dusted_rest_sed=dusted_sed,
        photometry=photometry,
    )


def run_dsps_model_jax(context: DspsContext, params: dict[str, Any]) -> JaxModelResult:
    """Pure-JAX DSPS forward model used by gradient-based fits."""
    ssp = context.ssp
    z_obs = jnp.asarray(params["z_obs"])
    t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]

    gal_t_table = jnp.linspace(0.05, jnp.maximum(t_obs, 0.06), context.n_sfh_bins)
    gal_sfr_table = build_lognormal_sfh_jax(
        gal_t_table=gal_t_table,
        log10_sfr=jnp.asarray(params["log10_sfr"]),
        sfh_t_peak=jnp.asarray(params["sfh_t_peak"]),
        sfh_tau=jnp.asarray(params["sfh_tau"]),
    )
    formed_mass = jnp.trapezoid(gal_sfr_table, gal_t_table) * 1.0e9

    sed_info = calc_rest_sed_sfh_table_lognormal_mdf(
        gal_t_table,
        gal_sfr_table,
        jnp.asarray(params["log10_metallicity"]),
        jnp.asarray(params["metallicity_scatter"]),
        jnp.asarray(ssp.ssp_lgmet),
        jnp.asarray(ssp.ssp_lg_age_gyr),
        jnp.asarray(ssp.ssp_flux),
        t_obs,
    )
    wave = jnp.asarray(ssp.ssp_wave)
    dusted_sed = apply_dust_jax(wave, sed_info.rest_sed, params)
    model_mags = predict_mags_jax(context, wave, dusted_sed, z_obs)
    return JaxModelResult(
        wave=wave,
        rest_sed=sed_info.rest_sed,
        dusted_rest_sed=dusted_sed,
        model_mags=model_mags,
        t_obs_gyr=t_obs,
        formed_mass_msun=formed_mass,
        sfr_at_t_obs_msun_per_yr=gal_sfr_table[-1],
    )


def predict_mags_jax(context: DspsContext, wave: jnp.ndarray, dusted_sed: jnp.ndarray, z_obs: jnp.ndarray) -> jnp.ndarray:
    """Predict configured apparent AB magnitudes with DSPS photometry kernels."""
    mags = []
    for curve in context.filters.values():
        mags.append(
            calc_obs_mag(
                wave,
                dusted_sed,
                jnp.asarray(curve.wave),
                jnp.asarray(curve.transmission),
                z_obs,
                *DEFAULT_COSMOLOGY,
            )
        )
    return jnp.stack(mags)


def model_mags_jax(context: DspsContext, params: dict[str, Any]) -> jnp.ndarray:
    """Return only model magnitudes for likelihood/gradient code."""
    return run_dsps_model_jax(context, params).model_mags


def predict_batch_mags(context: DspsContext, parameter_names: list[str], parameter_matrix: np.ndarray) -> np.ndarray:
    """Predict magnitudes for many parameter rows with one JAX-vmapped call."""

    def single(values):
        params = {name: values[index] for index, name in enumerate(parameter_names)}
        return model_mags_jax(context, params)

    predict = jax.jit(jax.vmap(single))
    return np.asarray(predict(jnp.asarray(parameter_matrix)))


def build_lognormal_sfh(gal_t_table: np.ndarray, log10_sfr: float, sfh_t_peak: float, sfh_tau: float) -> np.ndarray:
    """Build a positive, smooth SFH in Msun/yr on cosmic-time bins."""
    return np.asarray(build_lognormal_sfh_jax(gal_t_table, log10_sfr, sfh_t_peak, sfh_tau), dtype=float)


def build_lognormal_sfh_jax(
    gal_t_table: jnp.ndarray,
    log10_sfr: jnp.ndarray,
    sfh_t_peak: jnp.ndarray,
    sfh_tau: jnp.ndarray,
) -> jnp.ndarray:
    """JAX lognormal SFH in Msun/yr on cosmic-time bins."""
    amplitude = 10**log10_sfr
    t_peak = jnp.clip(sfh_t_peak, jnp.min(gal_t_table), jnp.max(gal_t_table))
    tau = jnp.maximum(sfh_tau, 0.05)
    log_t = jnp.log(jnp.clip(gal_t_table, 1.0e-3))
    shape = jnp.exp(-0.5 * ((log_t - jnp.log(t_peak)) / tau) ** 2)
    shape = jnp.clip(shape, 1.0e-6)
    return amplitude * shape


def apply_dust(wave_angstrom: np.ndarray, rest_sed: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Apply a DSPS Salim+2018-style attenuation curve."""
    return np.asarray(apply_dust_jax(wave_angstrom, rest_sed, params), dtype=float)


def apply_dust_jax(wave_angstrom: jnp.ndarray, rest_sed: jnp.ndarray, params: dict[str, Any]) -> jnp.ndarray:
    """Apply DSPS Salim+2018-style attenuation without leaving JAX."""
    av = jnp.maximum(jnp.asarray(params.get("dust_av", 0.0)), 0.0)
    wave_micron = jnp.asarray(wave_angstrom) / 10_000.0
    k_lambda = sbl18_k_lambda(
        wave_micron,
        0.0,
        jnp.asarray(params.get("dust_slope", -0.7)),
    )
    transmission = _frac_transmission_from_k_lambda(k_lambda, av)
    return jnp.asarray(rest_sed) * transmission


def _safe_log10(value: float) -> float:
    return float(np.log10(value)) if value > 0 else float("nan")



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
