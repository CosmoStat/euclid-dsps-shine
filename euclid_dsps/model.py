"""Native DSPS model wrapper."""

# ruff: noqa: I001, E402

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .jax_runtime import configure_jax_runtime

configure_jax_runtime()

import jax
import jax.numpy as jnp
import numpy as np

from .filters import FilterCurve
from .io import GalaxyObservation
from .photometry import abmag_to_fnu_cgs


@dataclass
class DspsContext:
    ssp: Any
    filters: dict[str, FilterCurve]
    n_sfh_bins: int = 96
    cosmos_dust_k_by_code: np.ndarray | None = None
    cosmos_dust_curve_names: tuple[str, ...] = ()
    ssp_wave_jax: Any | None = None
    ssp_lgmet_jax: Any | None = None
    ssp_lg_age_gyr_jax: Any | None = None
    ssp_flux_jax: Any | None = None
    ssp_emline_luminosity: np.ndarray | None = None
    ssp_emline_wave: np.ndarray | None = None
    ssp_emline_name: tuple[str, ...] = ()
    nebular_emission_mode: str = "ssp_flux"
    jax_filters: tuple[tuple[Any, Any], ...] = ()
    cosmos_dust_k_by_code_jax: Any | None = None


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
    sfr_at_obs_msun_per_yr: jnp.ndarray


@dataclass(frozen=True)
class BatchSedResult:
    """Batch DSPS SEDs and photometry from one JAX-vmapped call."""

    parameter_names: list[str]
    parameter_matrix: np.ndarray
    wave: np.ndarray
    rest_sed: np.ndarray
    dusted_rest_sed: np.ndarray
    model_mags: np.ndarray
    derived: dict[str, np.ndarray]


DERIVED_QUANTITY_NAMES = [
    "t_obs_gyr",
    "formed_mass_msun",
    "log10_formed_mass_msun",
    "sfr_at_obs_msun_per_yr",
    "log10_sfr_at_obs",
]


def load_context(
    ssp_path: str,
    filters: dict[str, FilterCurve],
    n_sfh_bins: int = 96,
    cosmos_config: dict[str, Any] | None = None,
    nebular_emission: str = "ssp_flux",
) -> DspsContext:
    from dsps import load_ssp_templates

    ssp = load_ssp_templates(fn=ssp_path)
    dust_k_by_code, dust_curve_names = _load_cosmos_dust_grid(ssp, cosmos_config)
    emline_luminosity, emline_wave, emline_name = _load_ssp_emline_data(ssp_path, ssp)
    return DspsContext(
        ssp=ssp,
        filters=filters,
        n_sfh_bins=n_sfh_bins,
        cosmos_dust_k_by_code=dust_k_by_code,
        cosmos_dust_curve_names=dust_curve_names,
        ssp_wave_jax=jnp.asarray(ssp.ssp_wave, dtype=jnp.float32),
        ssp_lgmet_jax=jnp.asarray(ssp.ssp_lgmet, dtype=jnp.float32),
        ssp_lg_age_gyr_jax=jnp.asarray(ssp.ssp_lg_age_gyr, dtype=jnp.float32),
        ssp_flux_jax=jnp.asarray(ssp.ssp_flux, dtype=jnp.float32),
        ssp_emline_luminosity=emline_luminosity,
        ssp_emline_wave=emline_wave,
        ssp_emline_name=emline_name,
        nebular_emission_mode=str(nebular_emission),
        jax_filters=tuple(
            (
                jnp.asarray(curve.wave, dtype=jnp.float32),
                jnp.asarray(curve.transmission, dtype=jnp.float32),
            )
            for curve in filters.values()
        ),
        cosmos_dust_k_by_code_jax=(
            None
            if dust_k_by_code is None
            else jnp.asarray(dust_k_by_code, dtype=jnp.float32)
        ),
    )


def _load_ssp_emline_data(
    ssp_path: str, ssp: Any
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[str, ...]]:
    luminosity = getattr(ssp, "ssp_emline_luminosity", None)
    wave = getattr(ssp, "ssp_emline_wave", None)
    names: tuple[str, ...] = ()
    if luminosity is not None:
        luminosity = np.asarray(luminosity, dtype=float)
    if wave is not None:
        wave = np.asarray(wave, dtype=float)
    try:
        import h5py

        with h5py.File(ssp_path, "r") as handle:
            if luminosity is None and "ssp_emline_luminosity" in handle:
                luminosity = np.asarray(handle["ssp_emline_luminosity"], dtype=float)
            if wave is None and "ssp_emline_wave" in handle:
                wave = np.asarray(handle["ssp_emline_wave"], dtype=float)
            if "ssp_emline_name" in handle:
                raw = np.asarray(handle["ssp_emline_name"])
                decoded = []
                for item in raw:
                    if isinstance(item, (bytes, np.bytes_)):
                        decoded.append(item.decode("utf-8", errors="replace"))
                    else:
                        decoded.append(str(item))
                names = tuple(decoded)
    except (OSError, ImportError, KeyError, TypeError):
        pass
    if luminosity is None:
        return None, wave, names
    n_lines = int(luminosity.shape[-1])
    if wave is not None and len(wave) != n_lines:
        wave = None
    if not names or len(names) != n_lines:
        names = tuple(f"line_{i:03d}" for i in range(n_lines))
    return luminosity, wave, names


def _load_cosmos_dust_grid(
    ssp: Any, cosmos_config: dict[str, Any] | None
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    if not cosmos_config or not bool(cosmos_config.get("use_cosmos_dust_in_dsps")):
        return None, ()

    from .cosmos import load_extinction_curves

    mapping, curves, _ = load_extinction_curves(cosmos_config)
    if not mapping:
        return None, ()
    max_code = max(int(code) for code in mapping)
    wave = np.asarray(ssp.ssp_wave, dtype=float)
    k_by_code = np.zeros((max_code + 1, len(wave)), dtype=float)
    names: list[str] = []
    for code in range(max_code + 1):
        curve_name = mapping.get(code, "none")
        names.append(curve_name)
        if curve_name == "none":
            continue
        curve = curves.get(curve_name)
        if curve is None:
            raise ValueError(
                f"COSMOS extinction curve {curve_name!r} is configured but not loaded."
            )
        k_by_code[code] = np.interp(
            wave,
            curve.wave_angstrom,
            curve.k_lambda,
            left=curve.k_lambda[0],
            right=curve.k_lambda[-1],
        )
    return k_by_code, tuple(names)


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
    params.update(redshift_prior_parameters(params["z_obs"], row, redshift_config or {}))
    return params


def resolve_redshift(
    params: dict[str, float], row: dict[str, Any], redshift_config: dict[str, Any]
) -> float:
    """Resolve DSPS redshift from configured initializer."""
    value = params.get("z_obs", redshift_config.get("fixed_value", 0.5))
    initial = str(redshift_config.get("initial", "catalog_column"))
    column = redshift_config.get("column")
    z_min = float(redshift_config.get("min", 1.0e-4))
    z_max = float(redshift_config.get("max", 6.0))

    if initial == "random_uniform":
        value = _random_uniform_redshift(row, redshift_config, z_min, z_max)
    elif (
        initial == "catalog_column"
        and column
        and column in row
        and np.isfinite(row[column])
    ):
        value = float(row[column])
    elif initial in {"catalog_column", "fixed"} and np.isfinite(
        redshift_config.get("fixed_value", np.nan)
    ):
        value = float(redshift_config["fixed_value"])

    if not np.isfinite(value):
        value = z_min
    return float(np.clip(value, z_min, z_max))


def redshift_prior_parameters(
    z_value: float, row: dict[str, Any], redshift_config: dict[str, Any]
) -> dict[str, float]:
    """Return row-level redshift prior metadata consumed by fit priors."""
    prior = redshift_config.get("prior_z") or {}
    if not isinstance(prior, dict) or str(prior.get("mode", "none")) != "gaussian":
        return {}
    sigma = float(prior.get("sigma", 0.35))
    if bool(prior.get("scale_with_1pz", True)):
        sigma *= 1.0 + max(float(z_value), 0.0)
    sigma = max(sigma, float(prior.get("sigma_min", 0.02)))
    return {
        "z_obs_prior_mu": float(z_value),
        "z_obs_prior_sigma": float(sigma),
    }


def _random_uniform_redshift(
    row: dict[str, Any], redshift_config: dict[str, Any], z_min: float, z_max: float
) -> float:
    seed = int(float(redshift_config.get("seed", 42)))
    payload = "|".join(
        f"{key}={row[key]}" for key in sorted(row) if np.isscalar(row[key])
    )
    digest = hashlib.blake2b(f"{seed}|{payload}".encode(), digest_size=8).digest()
    unit = int.from_bytes(digest, "big") / float(2**64 - 1)
    return z_min + unit * (z_max - z_min)


def run_dsps_model(context: DspsContext, params: dict[str, float]) -> ModelResult:
    """Run DSPS from simple SFH/metallicity parameters to SED and photometry."""
    jax_result = run_dsps_model_jax(context, params)
    rest_sed = np.asarray(jax_result.rest_sed, dtype=float)
    wave = np.asarray(jax_result.wave, dtype=float)
    dusted_sed = np.asarray(jax_result.dusted_rest_sed, dtype=float)
    model_mags = np.asarray(jax_result.model_mags, dtype=float)

    photometry: dict[str, dict[str, float]] = {}
    for (name, curve), mag in zip(context.filters.items(), model_mags, strict=True):
        photometry[name] = {
            "model_mag_ab": float(mag),
            "model_flux_fnu_cgs": float(abmag_to_fnu_cgs(float(mag))),
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
            "sfr_at_obs_msun_per_yr": float(jax_result.sfr_at_obs_msun_per_yr),
            "log10_sfr_at_obs": _safe_log10(float(jax_result.sfr_at_obs_msun_per_yr)),
        },
        wave=wave,
        rest_sed=rest_sed,
        dusted_rest_sed=dusted_sed,
        photometry=photometry,
    )


def run_dsps_model_jax(context: DspsContext, params: dict[str, Any]) -> JaxModelResult:
    """Pure-JAX DSPS forward model used by gradient-based fits."""
    from dsps import calc_rest_sed_sfh_table_lognormal_mdf
    from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z

    z_obs = jnp.asarray(params["z_obs"], dtype=jnp.float32)
    t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]

    gal_t_table = jnp.linspace(0.05, jnp.maximum(t_obs, 0.06), context.n_sfh_bins)
    gal_sfr_table = build_sfh_table_jax(gal_t_table, params)
    gal_sfr_table, formed_mass = normalize_sfh_mass_jax(
        gal_t_table, gal_sfr_table, params
    )

    sed_info = calc_rest_sed_sfh_table_lognormal_mdf(
        gal_t_table,
        gal_sfr_table,
        jnp.asarray(params["log10_metallicity"], dtype=jnp.float32),
        jnp.asarray(params["metallicity_scatter"], dtype=jnp.float32),
        _context_ssp_lgmet(context),
        _context_ssp_lg_age_gyr(context),
        _context_ssp_flux(context),
        t_obs,
    )
    wave = _context_ssp_wave(context)
    dusted_sed = apply_dust_jax(
        wave, sed_info.rest_sed, params, context.cosmos_dust_k_by_code_jax
    )
    model_mags = predict_mags_jax(context, wave, dusted_sed, z_obs)
    return JaxModelResult(
        wave=wave,
        rest_sed=sed_info.rest_sed,
        dusted_rest_sed=dusted_sed,
        model_mags=model_mags,
        t_obs_gyr=t_obs,
        formed_mass_msun=formed_mass,
        sfr_at_obs_msun_per_yr=gal_sfr_table[-1],
    )


def predict_mags_jax(
    context: DspsContext, wave: jnp.ndarray, dusted_sed: jnp.ndarray, z_obs: jnp.ndarray
) -> jnp.ndarray:
    """Predict configured apparent AB magnitudes with DSPS photometry kernels."""
    from dsps import calc_obs_mag
    from dsps.cosmology import DEFAULT_COSMOLOGY

    filter_arrays = context.jax_filters
    if not filter_arrays:
        filter_arrays = tuple(
            (
                jnp.asarray(curve.wave, dtype=jnp.float32),
                jnp.asarray(curve.transmission, dtype=jnp.float32),
            )
            for curve in context.filters.values()
        )
    mags = []
    for filter_wave, filter_transmission in filter_arrays:
        mags.append(
            calc_obs_mag(
                wave,
                dusted_sed,
                filter_wave,
                filter_transmission,
                z_obs,
                *DEFAULT_COSMOLOGY,
            )
        )
    return jnp.stack(mags)


def model_mags_jax(context: DspsContext, params: dict[str, Any]) -> jnp.ndarray:
    """Return only model magnitudes for likelihood/gradient code."""
    return run_dsps_model_jax(context, params).model_mags


def _context_ssp_wave(context: DspsContext) -> jnp.ndarray:
    if context.ssp_wave_jax is not None:
        return context.ssp_wave_jax
    return jnp.asarray(context.ssp.ssp_wave, dtype=jnp.float32)


def _context_ssp_lgmet(context: DspsContext) -> jnp.ndarray:
    if context.ssp_lgmet_jax is not None:
        return context.ssp_lgmet_jax
    return jnp.asarray(context.ssp.ssp_lgmet, dtype=jnp.float32)


def _context_ssp_lg_age_gyr(context: DspsContext) -> jnp.ndarray:
    if context.ssp_lg_age_gyr_jax is not None:
        return context.ssp_lg_age_gyr_jax
    return jnp.asarray(context.ssp.ssp_lg_age_gyr, dtype=jnp.float32)


def _context_ssp_flux(context: DspsContext) -> jnp.ndarray:
    if context.ssp_flux_jax is not None:
        return context.ssp_flux_jax
    return jnp.asarray(context.ssp.ssp_flux, dtype=jnp.float32)


_BATCH_PREDICT_CACHE = {}


def predict_batch_mags(
    context: DspsContext, parameter_names: list[str], parameter_matrix: np.ndarray
) -> np.ndarray:
    """Predict magnitudes for many parameter rows with one JAX-vmapped call."""
    cache_key = ("mags", id(context), tuple(parameter_names))
    if cache_key not in _BATCH_PREDICT_CACHE:

        def single(values):
            params = {name: values[index] for index, name in enumerate(parameter_names)}
            return model_mags_jax(context, params)

        _BATCH_PREDICT_CACHE[cache_key] = jax.jit(jax.vmap(single))

    predict = _BATCH_PREDICT_CACHE[cache_key]
    return np.asarray(predict(jnp.asarray(parameter_matrix, dtype=jnp.float32)))


def derived_quantities_jax(context: DspsContext, params: dict[str, Any]) -> jnp.ndarray:
    """Return derived quantities needed for scientifically comparable reports."""
    from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z

    z_obs = jnp.asarray(params["z_obs"], dtype=jnp.float32)
    t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]
    gal_t_table = jnp.linspace(0.05, jnp.maximum(t_obs, 0.06), context.n_sfh_bins)
    gal_sfr_table = build_sfh_table_jax(gal_t_table, params)
    gal_sfr_table, formed_mass = normalize_sfh_mass_jax(
        gal_t_table, gal_sfr_table, params
    )
    sfr_at_obs = gal_sfr_table[-1]
    return jnp.asarray(
        [
            t_obs,
            formed_mass,
            jnp.log10(jnp.maximum(formed_mass, 1.0e-300)),
            sfr_at_obs,
            jnp.log10(jnp.maximum(sfr_at_obs, 1.0e-300)),
        ]
    )


def predict_batch_derived(
    context: DspsContext, parameter_names: list[str], parameter_matrix: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute derived quantities for many fitted parameter rows."""
    cache_key = ("derived", id(context), tuple(parameter_names))
    if cache_key not in _BATCH_PREDICT_CACHE:

        def single(values):
            params = {name: values[index] for index, name in enumerate(parameter_names)}
            return derived_quantities_jax(context, params)

        _BATCH_PREDICT_CACHE[cache_key] = jax.jit(jax.vmap(single))

    predict = _BATCH_PREDICT_CACHE[cache_key]
    values = np.asarray(predict(jnp.asarray(parameter_matrix, dtype=jnp.float32)))
    return {name: values[:, index] for index, name in enumerate(DERIVED_QUANTITY_NAMES)}


def predict_batch_seds(
    context: DspsContext, parameter_names: list[str], parameter_matrix: np.ndarray
) -> BatchSedResult:
    """Predict rest SEDs, dusted rest SEDs, magnitudes, and derived quantities.

    This is the batch/GPU path used by COSMOS-template comparisons after MAP or
    population fits. It avoids one Python DSPS call per galaxy.
    """
    cache_key = ("seds", id(context), tuple(parameter_names))
    if cache_key not in _BATCH_PREDICT_CACHE:

        def single(values):
            params = {name: values[index] for index, name in enumerate(parameter_names)}
            result = run_dsps_model_jax(context, params)
            derived = jnp.asarray(
                [
                    result.t_obs_gyr,
                    result.formed_mass_msun,
                    jnp.log10(jnp.maximum(result.formed_mass_msun, 1.0e-300)),
                    result.sfr_at_obs_msun_per_yr,
                    jnp.log10(jnp.maximum(result.sfr_at_obs_msun_per_yr, 1.0e-300)),
                ]
            )
            return result.rest_sed, result.dusted_rest_sed, result.model_mags, derived

        _BATCH_PREDICT_CACHE[cache_key] = jax.jit(jax.vmap(single))

    predict = _BATCH_PREDICT_CACHE[cache_key]
    rest_sed, dusted_rest_sed, model_mags, derived_values = predict(
        jnp.asarray(parameter_matrix, dtype=jnp.float32)
    )
    derived_array = np.asarray(derived_values)
    return BatchSedResult(
        parameter_names=list(parameter_names),
        parameter_matrix=np.asarray(parameter_matrix, dtype=float),
        wave=np.asarray(_context_ssp_wave(context), dtype=float),
        rest_sed=np.asarray(rest_sed, dtype=float),
        dusted_rest_sed=np.asarray(dusted_rest_sed, dtype=float),
        model_mags=np.asarray(model_mags, dtype=float),
        derived={
            name: derived_array[:, index]
            for index, name in enumerate(DERIVED_QUANTITY_NAMES)
        },
    )


def build_lognormal_sfh(
    gal_t_table: np.ndarray,
    log10_sfr: float,
    sfh_t_peak: float,
    sfh_tau: float,
) -> np.ndarray:
    """Build a positive SFH in Msun/yr on cosmic-time bins."""
    return np.asarray(
        build_lognormal_sfh_jax(
            gal_t_table,
            log10_sfr,
            sfh_t_peak,
            sfh_tau,
        ),
        dtype=float,
    )


def build_sfh_table_jax(
    gal_t_table: jnp.ndarray, params: dict[str, Any]
) -> jnp.ndarray:
    """Build the simple production SFH table without leaving JAX."""
    return build_lognormal_sfh_jax(
        gal_t_table=gal_t_table,
        log10_sfr=jnp.asarray(params["log10_sfr"], dtype=jnp.float32),
        sfh_t_peak=jnp.asarray(params["sfh_t_peak"], dtype=jnp.float32),
        sfh_tau=jnp.asarray(params["sfh_tau"], dtype=jnp.float32),
    )


def build_lognormal_sfh_jax(
    gal_t_table: jnp.ndarray,
    log10_sfr: jnp.ndarray,
    sfh_t_peak: jnp.ndarray,
    sfh_tau: jnp.ndarray,
) -> jnp.ndarray:
    """JAX lognormal SFH used by production fits."""
    amplitude = 10**log10_sfr
    t_peak = jnp.clip(sfh_t_peak, jnp.min(gal_t_table), jnp.max(gal_t_table))
    tau = jnp.maximum(sfh_tau, 0.05)
    log_t = jnp.log(jnp.clip(gal_t_table, 1.0e-3))
    shape = jnp.exp(-0.5 * ((log_t - jnp.log(t_peak)) / tau) ** 2)
    shape = jnp.clip(shape, 1.0e-6)
    return jnp.clip(amplitude * shape, 1.0e-12, jnp.inf)


def normalize_sfh_mass_jax(
    gal_t_table: jnp.ndarray, gal_sfr_table: jnp.ndarray, params: dict[str, Any]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Optionally scale an SFH to a configured formed stellar mass.

    Without ``log10_formed_mass_msun`` this preserves the historical behavior,
    where ``log10_sfr`` is the SFH amplitude. With it, ``log10_sfr`` only sets
    the pre-normalization shape scale and the luminosity amplitude is controlled
    by the formed-mass parameter.
    """
    formed_mass = jnp.trapezoid(gal_sfr_table, gal_t_table) * 1.0e9
    if "log10_formed_mass_msun" not in params:
        return gal_sfr_table, formed_mass
    target_mass = 10.0 ** jnp.asarray(
        params["log10_formed_mass_msun"], dtype=jnp.float32
    )
    scale = target_mass / jnp.maximum(formed_mass, 1.0e-30)
    scaled_sfr = jnp.clip(gal_sfr_table * scale, 1.0e-12, jnp.inf)
    scaled_mass = jnp.trapezoid(scaled_sfr, gal_t_table) * 1.0e9
    return scaled_sfr, scaled_mass


def apply_dust(
    wave_angstrom: np.ndarray, rest_sed: np.ndarray, params: dict[str, float]
) -> np.ndarray:
    """Apply the configured attenuation model."""
    return np.asarray(apply_dust_jax(wave_angstrom, rest_sed, params), dtype=float)


def apply_dust_jax(
    wave_angstrom: jnp.ndarray,
    rest_sed: jnp.ndarray,
    params: dict[str, Any],
    cosmos_dust_k_by_code: np.ndarray | None = None,
) -> jnp.ndarray:
    """Apply COSMOS two-component dust when available, else DSPS Salim dust."""
    if cosmos_dust_k_by_code is not None:
        return apply_cosmos_two_component_dust_jax(
            rest_sed, params, cosmos_dust_k_by_code
        )
    return apply_salim_dust_jax(wave_angstrom, rest_sed, params)


def apply_salim_dust_jax(
    wave_angstrom: jnp.ndarray, rest_sed: jnp.ndarray, params: dict[str, Any]
) -> jnp.ndarray:
    """Apply DSPS Salim+2018-style attenuation without leaving JAX."""
    from dsps.dust.att_curves import _frac_transmission_from_k_lambda, sbl18_k_lambda

    av = jnp.maximum(jnp.asarray(params.get("dust_av", 0.0)), 0.0)
    wave_micron = jnp.asarray(wave_angstrom) / 10_000.0
    k_lambda = sbl18_k_lambda(
        wave_micron,
        0.0,
        jnp.asarray(params.get("dust_slope", -0.7)),
    )
    transmission = _frac_transmission_from_k_lambda(k_lambda, av)
    return jnp.asarray(rest_sed) * transmission


def apply_cosmos_two_component_dust_jax(
    rest_sed: jnp.ndarray, params: dict[str, Any], cosmos_dust_k_by_code: np.ndarray
) -> jnp.ndarray:
    """Apply the two COSMOS dust curves as a differentiable mixture."""
    k_grid = jnp.asarray(cosmos_dust_k_by_code)
    n_codes = k_grid.shape[0]
    code_1 = jnp.clip(
        jnp.rint(jnp.asarray(params.get("cosmos_ext_curve_1", 0.0))).astype(jnp.int32),
        0,
        n_codes - 1,
    )
    code_2 = jnp.clip(
        jnp.rint(jnp.asarray(params.get("cosmos_ext_curve_2", 0.0))).astype(jnp.int32),
        0,
        n_codes - 1,
    )
    ebv_1 = jnp.maximum(jnp.asarray(params.get("cosmos_ebv_1", 0.0)), 0.0)
    ebv_2 = jnp.maximum(jnp.asarray(params.get("cosmos_ebv_2", 0.0)), 0.0)
    frac_1 = jnp.maximum(jnp.asarray(params.get("cosmos_frac_1", 0.5)), 0.0)
    frac_2 = jnp.maximum(jnp.asarray(params.get("cosmos_frac_2", 0.5)), 0.0)
    frac_sum = frac_1 + frac_2
    frac_1 = jnp.where(frac_sum > 0.0, frac_1 / frac_sum, 0.5)
    frac_2 = jnp.where(frac_sum > 0.0, frac_2 / frac_sum, 0.5)
    trans_1 = 10.0 ** (-0.4 * ebv_1 * k_grid[code_1])
    trans_2 = 10.0 ** (-0.4 * ebv_2 * k_grid[code_2])
    return jnp.asarray(rest_sed) * (frac_1 * trans_1 + frac_2 * trans_2)


def _safe_log10(value: float) -> float:
    return float(np.log10(value)) if value > 0 else float("nan")


def comparison_rows(
    observation: GalaxyObservation, result: ModelResult
) -> list[dict[str, float | str]]:
    rows = []
    for observed in observation.bands:
        model = result.photometry[observed.name]
        residual = observed.mag_ab - model["model_mag_ab"]
        model_flux = float(model["model_flux_fnu_cgs"])
        observed_flux = float(observed.flux_fnu_cgs)
        flux_error = observed.flux_error_fnu_cgs
        if flux_error is None or not np.isfinite(flux_error) or flux_error <= 0:
            flux_error = abs(observed_flux) * np.log(10.0) * 0.4 * observed.sigma_mag
        flux_ratio = (
            model_flux / observed_flux
            if observed_flux > 0
            else float("nan")
        )
        chi_flux = (
            (model_flux - observed_flux) / flux_error
            if flux_error > 0
            else float("nan")
        )
        rows.append(
            {
                "band": observed.name,
                "column": observed.column,
                "effective_wavelength_angstrom": model["effective_wavelength_angstrom"],
                "observed_flux_fnu_cgs": observed_flux,
                "observed_flux_error_fnu_cgs": float(flux_error),
                "observed_mag_ab": observed.mag_ab,
                "sigma_mag": observed.sigma_mag,
                "model_flux_fnu_cgs": model_flux,
                "model_mag_ab": model["model_mag_ab"],
                "residual_mag_observed_minus_model": residual,
                "residual_mag_model_minus_observed": -residual,
                "flux_ratio_model_over_observed": flux_ratio,
                "fractional_flux_residual_model_minus_observed": flux_ratio - 1.0,
                "chi": residual / observed.sigma_mag,
                "chi_flux": chi_flux,
                "filter_source": model["filter_source"],
            }
        )
    return rows
