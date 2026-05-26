"""Native DSPS model wrapper."""

# ruff: noqa: I001, E402

from __future__ import annotations

import hashlib
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .jax_runtime import configure_jax_runtime

configure_jax_runtime()

import jax
import jax.numpy as jnp
import numpy as np

from .filters import FilterCurve
from .io import GalaxyObservation
from .parameters import POPCOSMOS_PARAMETER_NAMES
from .photometry import abmag_to_fnu_cgs


@dataclass
class DspsContext:
    ssp: Any
    filters: dict[str, FilterCurve]
    n_sfh_bins: int = 96
    z_sun: float = 0.0134
    model_config: dict[str, Any] | None = None
    cosmos_dust_k_by_code: np.ndarray | None = None
    cosmos_dust_curve_names: tuple[str, ...] = ()
    ssp_wave_jax: Any | None = None
    ssp_lgmet_jax: Any | None = None
    ssp_lg_age_gyr_jax: Any | None = None
    ssp_flux_jax: Any | None = None
    gas_lgmet_grid_jax: Any | None = None
    gas_lgu_grid_jax: Any | None = None
    ssp_flux_gas_grid_jax: Any | None = None
    agn_wave_jax: Any | None = None
    agn_tau_grid_jax: Any | None = None
    agn_template_grid_jax: Any | None = None
    ssp_emline_luminosity: np.ndarray | None = None
    ssp_emline_wave: np.ndarray | None = None
    ssp_emline_name: tuple[str, ...] = ()
    nebular_emission_mode: str = "ssp_flux"
    jax_filters: tuple[tuple[Any, Any], ...] = ()
    cosmos_dust_k_by_code_jax: Any | None = None


DYNAMIC_CONTEXT_FIELDS = (
    "ssp_wave_jax",
    "ssp_lgmet_jax",
    "ssp_lg_age_gyr_jax",
    "ssp_flux_jax",
    "gas_lgmet_grid_jax",
    "gas_lgu_grid_jax",
    "ssp_flux_gas_grid_jax",
    "agn_wave_jax",
    "agn_tau_grid_jax",
    "agn_template_grid_jax",
    "jax_filters",
    "cosmos_dust_k_by_code_jax",
)


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
    surviving_stellar_mass_msun: jnp.ndarray
    sfr_at_obs_msun_per_yr: jnp.ndarray
    sfr_bins_msun_per_yr: jnp.ndarray
    lookback_bin_edges_gyr: jnp.ndarray


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
    "surviving_stellar_mass_msun",
    "log10_surviving_stellar_mass_msun",
    "sfr_at_obs_msun_per_yr",
    "log10_sfr_at_obs",
    "sfr_bin_1",
    "sfr_bin_2",
    "sfr_bin_3",
    "sfr_bin_4",
    "sfr_bin_5",
    "sfr_bin_6",
    "sfr_bin_7",
    "lookback_bin_edge_0",
    "lookback_bin_edge_1",
    "lookback_bin_edge_2",
    "lookback_bin_edge_3",
    "lookback_bin_edge_4",
    "lookback_bin_edge_5",
    "lookback_bin_edge_6",
    "lookback_bin_edge_7",
]


def load_context(
    ssp_path: str,
    filters: dict[str, FilterCurve],
    n_sfh_bins: int = 96,
    cosmos_config: dict[str, Any] | None = None,
    nebular_emission: str = "ssp_flux",
    model_config: dict[str, Any] | None = None,
) -> DspsContext:
    from dsps import load_ssp_templates

    model_config = _normalized_model_config(model_config)
    ssp = load_ssp_templates(fn=ssp_path)
    dust_k_by_code, dust_curve_names = _load_cosmos_dust_grid(ssp, cosmos_config)
    emline_luminosity, emline_wave, emline_name = _load_ssp_emline_data(ssp_path, ssp)
    gas_lgmet_grid, gas_lgu_grid, ssp_flux_gas_grid = _load_optional_gas_grid(
        model_config
    )
    agn_wave, agn_tau_grid, agn_template_grid = _load_optional_agn_grid(model_config)
    return DspsContext(
        ssp=ssp,
        filters=filters,
        n_sfh_bins=n_sfh_bins,
        z_sun=float(model_config.get("z_sun", 0.0134)),
        model_config=model_config,
        cosmos_dust_k_by_code=dust_k_by_code,
        cosmos_dust_curve_names=dust_curve_names,
        ssp_wave_jax=jnp.asarray(ssp.ssp_wave, dtype=jnp.float32),
        ssp_lgmet_jax=jnp.asarray(ssp.ssp_lgmet, dtype=jnp.float32),
        ssp_lg_age_gyr_jax=jnp.asarray(ssp.ssp_lg_age_gyr, dtype=jnp.float32),
        ssp_flux_jax=jnp.asarray(ssp.ssp_flux, dtype=jnp.float32),
        gas_lgmet_grid_jax=(
            None
            if gas_lgmet_grid is None
            else jnp.asarray(gas_lgmet_grid, dtype=jnp.float32)
        ),
        gas_lgu_grid_jax=(
            None
            if gas_lgu_grid is None
            else jnp.asarray(gas_lgu_grid, dtype=jnp.float32)
        ),
        ssp_flux_gas_grid_jax=(
            None
            if ssp_flux_gas_grid is None
            else jnp.asarray(ssp_flux_gas_grid, dtype=jnp.float32)
        ),
        agn_wave_jax=(
            None if agn_wave is None else jnp.asarray(agn_wave, dtype=jnp.float32)
        ),
        agn_tau_grid_jax=(
            None
            if agn_tau_grid is None
            else jnp.asarray(agn_tau_grid, dtype=jnp.float32)
        ),
        agn_template_grid_jax=(
            None
            if agn_template_grid is None
            else jnp.asarray(agn_template_grid, dtype=jnp.float32)
        ),
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


def _normalized_model_config(model_config: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(model_config or {})
    sfh_model = str(config.get("sfh_model", "lognormal"))
    config.setdefault("sfh_model", sfh_model)
    config.setdefault(
        "stellar_metallicity_model",
        "single" if sfh_model == "popcosmos_bins" else "mdf",
    )
    config.setdefault(
        "dust_model",
        "charlot_fall" if sfh_model == "popcosmos_bins" else "legacy",
    )
    config.setdefault("igm_model", "none")
    config.setdefault("nebular_model", "fixed_ssp")
    config.setdefault("agn_model", "none")
    config.setdefault("z_sun", 0.0134)
    return config


def _load_optional_gas_grid(
    model_config: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if str(model_config.get("nebular_model", "fixed_ssp")) != "gas_grid":
        return None, None, None
    path = model_config.get("gas_grid_path")
    if not path:
        raise ValueError("model.nebular_model='gas_grid' requires model.gas_grid_path")
    return _load_gas_ssp_grid(path)


def _load_gas_ssp_grid(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_path = Path(path).expanduser()
    if not grid_path.exists():
        raise FileNotFoundError(f"Gas SSP grid file not found: {grid_path}")
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - pyproject requires h5py
        raise RuntimeError("h5py is required to load gas SSP grids") from exc

    required = (
        "ssp_wave",
        "ssp_lg_age_gyr",
        "ssp_lgmet",
        "gas_lgmet_grid",
        "gas_lgu_grid",
        "ssp_flux",
    )
    with h5py.File(grid_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(
                f"Gas SSP grid {grid_path} is missing datasets: {', '.join(missing)}"
            )
        gas_lgmet_grid = np.asarray(handle["gas_lgmet_grid"], dtype=float)
        gas_lgu_grid = np.asarray(handle["gas_lgu_grid"], dtype=float)
        ssp_wave = np.asarray(handle["ssp_wave"], dtype=float)
        ssp_lg_age_gyr = np.asarray(handle["ssp_lg_age_gyr"], dtype=float)
        ssp_lgmet = np.asarray(handle["ssp_lgmet"], dtype=float)
        ssp_flux = np.asarray(handle["ssp_flux"], dtype=np.float32)

    expected_ndim = 5
    if ssp_flux.ndim != expected_ndim:
        raise ValueError(
            "Gas SSP grid ssp_flux must have shape "
            "(n_gas_lgmet, n_gas_lgu, n_stellar_lgmet, n_age, n_wave)"
        )
    if ssp_flux.shape[0] != len(gas_lgmet_grid) or ssp_flux.shape[1] != len(
        gas_lgu_grid
    ):
        raise ValueError("Gas SSP grid axes do not match gas_lgmet_grid/gas_lgu_grid")
    expected_tail = (len(ssp_lgmet), len(ssp_lg_age_gyr), len(ssp_wave))
    if ssp_flux.shape[2:] != expected_tail:
        raise ValueError(
            "Gas SSP grid stellar metallicity/age/wavelength axes do not match "
            "ssp_flux trailing dimensions"
        )
    return gas_lgmet_grid, gas_lgu_grid, ssp_flux


def _load_optional_agn_grid(
    model_config: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if str(model_config.get("agn_model", "none")) != "template_grid":
        return None, None, None
    path = model_config.get("agn_template_path")
    if not path:
        raise ValueError(
            "model.agn_model='template_grid' requires model.agn_template_path"
        )
    return _load_agn_template_grid(path)


def _load_agn_template_grid(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_path = Path(path).expanduser()
    if not grid_path.exists():
        raise FileNotFoundError(f"AGN template grid file not found: {grid_path}")
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - pyproject requires h5py
        raise RuntimeError("h5py is required to load AGN template grids") from exc

    required = ("wave", "agn_tau_grid", "template_lnu_per_lbol")
    with h5py.File(grid_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(
                f"AGN template grid {grid_path} is missing datasets: {', '.join(missing)}"
            )
        wave = np.asarray(handle["wave"], dtype=float)
        tau_grid = np.asarray(handle["agn_tau_grid"], dtype=float)
        template_grid = np.asarray(handle["template_lnu_per_lbol"], dtype=float)

    if template_grid.ndim != 2 or template_grid.shape != (len(tau_grid), len(wave)):
        raise ValueError(
            "AGN template_lnu_per_lbol must have shape (n_agn_tau_grid, n_wave)"
        )
    return wave, tau_grid, template_grid


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
    params.update(
        redshift_prior_parameters(params["z_obs"], row, redshift_config or {})
    )
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

    derived_values = np.asarray(_jax_result_derived_array(jax_result), dtype=float)
    return ModelResult(
        parameters={key: float(value) for key, value in params.items()},
        derived={
            name: float(value)
            for name, value in zip(DERIVED_QUANTITY_NAMES, derived_values, strict=True)
        },
        wave=wave,
        rest_sed=rest_sed,
        dusted_rest_sed=dusted_sed,
        photometry=photometry,
    )


def run_dsps_model_jax(context: DspsContext, params: dict[str, Any]) -> JaxModelResult:
    """Pure-JAX DSPS forward model used by gradient-based fits."""
    model_config = _normalized_model_config(context.model_config)
    if str(model_config.get("sfh_model", "lognormal")) == "popcosmos_bins":
        return run_popcosmos_binned_model_jax(context, params)
    return run_lognormal_model_jax(context, params)


def run_lognormal_model_jax(
    context: DspsContext, params: dict[str, Any]
) -> JaxModelResult:
    """Legacy lognormal/MDF DSPS forward model."""
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
        surviving_stellar_mass_msun=jnp.asarray(jnp.nan, dtype=jnp.float32),
        sfr_at_obs_msun_per_yr=gal_sfr_table[-1],
        sfr_bins_msun_per_yr=jnp.full((7,), jnp.nan, dtype=jnp.float32),
        lookback_bin_edges_gyr=jnp.full((8,), jnp.nan, dtype=jnp.float32),
    )


def run_popcosmos_binned_model_jax(
    context: DspsContext, params: dict[str, Any]
) -> JaxModelResult:
    """PopCosmos-like seven-bin SFH forward model with age-dependent dust."""
    from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
    from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table

    model_config = _normalized_model_config(context.model_config)
    z_obs = jnp.asarray(params["z_obs"], dtype=jnp.float32)
    t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]
    t_start = jnp.minimum(jnp.asarray(0.001, dtype=jnp.float32), t_obs * 0.01)
    t_start = jnp.maximum(t_start, jnp.asarray(1.0e-5, dtype=jnp.float32))
    gal_t_table = jnp.linspace(
        t_start, jnp.maximum(t_obs, t_start * 1.01), context.n_sfh_bins
    )
    raw_sfr_table = build_popcosmos_sfh_table_jax(gal_t_table, t_obs, params)
    ssp_lg_age_gyr = _context_ssp_lg_age_gyr(context)
    gal_sfr_table, formed_mass, surviving_mass = normalize_sfh_to_stellar_mass_jax(
        gal_t_table,
        raw_sfr_table,
        ssp_lg_age_gyr,
        t_obs,
        params["log10_stellar_mass"],
    )
    sfr_bins_raw = logsfr_ratios_to_sfr_bins_jax(_popcosmos_dlog10_sfr(params))
    formed_raw = jnp.trapezoid(raw_sfr_table, gal_t_table) * 1.0e9
    formed_scale = formed_mass / jnp.maximum(formed_raw, 1.0e-30)
    sfr_bins = jnp.clip(sfr_bins_raw * formed_scale, 1.0e-30, jnp.inf)
    lookback_edges = build_popcosmos_lookback_bin_edges_jax(t_obs)

    ssp_flux_grid = _popcosmos_ssp_flux_grid(context, params, model_config)
    lgmet_abs = jnp.log10(jnp.asarray(context.z_sun, dtype=jnp.float32)) + jnp.asarray(
        params["log10_stellar_metallicity"], dtype=jnp.float32
    )
    ssp_flux_z = interpolate_ssp_stellar_metallicity_jax(
        _context_ssp_lgmet(context),
        ssp_flux_grid,
        lgmet_abs,
    )
    age_weights = calc_age_weights_from_sfh_table(
        gal_t_table,
        gal_sfr_table,
        ssp_lg_age_gyr,
        t_obs,
    )
    sed_by_age = jnp.clip(ssp_flux_z, 0.0, jnp.inf) * age_weights[:, None] * formed_mass
    intrinsic_stellar_sed = jnp.nan_to_num(
        sed_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
    )
    dusted_by_age = apply_charlot_fall_by_age_jax(
        _context_ssp_wave(context),
        ssp_lg_age_gyr,
        sed_by_age,
        params["tau2"],
        params["dust_index_n"],
        params["tau1_over_tau2"],
    )
    dusted_sed = jnp.nan_to_num(
        dusted_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
    )
    dusted_sed = add_agn_component_jax(
        context,
        _context_ssp_wave(context),
        intrinsic_stellar_sed,
        dusted_sed,
        params,
        model_config,
    )
    dusted_sed = apply_igm_transmission_jax(
        _context_ssp_wave(context),
        dusted_sed,
        z_obs,
        model_config,
    )
    model_mags = predict_mags_jax(
        context, _context_ssp_wave(context), dusted_sed, z_obs
    )
    return JaxModelResult(
        wave=_context_ssp_wave(context),
        rest_sed=intrinsic_stellar_sed,
        dusted_rest_sed=dusted_sed,
        model_mags=model_mags,
        t_obs_gyr=t_obs,
        formed_mass_msun=formed_mass,
        surviving_stellar_mass_msun=surviving_mass,
        sfr_at_obs_msun_per_yr=gal_sfr_table[-1],
        sfr_bins_msun_per_yr=sfr_bins,
        lookback_bin_edges_gyr=lookback_edges,
    )


def build_popcosmos_lookback_bin_edges_jax(t_obs: jnp.ndarray) -> jnp.ndarray:
    """Return PopCosmos-like seven-bin lookback edges in Gyr.

    For normal galaxy ages the first three edges are fixed at 0, 0.03, and
    0.10 Gyr, followed by three log-spaced intermediate edges, 0.85*t_obs, and
    t_obs. Very young universes switch to fixed fractions of t_obs so edges stay
    finite and strictly increasing.
    """
    t_safe = jnp.maximum(jnp.asarray(t_obs, dtype=jnp.float32), 1.0e-5)
    log_edges = jnp.logspace(
        jnp.log10(jnp.asarray(0.10, dtype=jnp.float32)),
        jnp.log10(jnp.maximum(0.85 * t_safe, 0.100001)),
        5,
    )
    nominal = jnp.concatenate(
        [
            jnp.asarray([0.0, 0.03], dtype=jnp.float32),
            log_edges,
            jnp.asarray([t_safe], dtype=jnp.float32),
        ]
    )
    fractional = t_safe * jnp.asarray(
        [0.0, 0.03, 0.10, 0.20, 0.35, 0.55, 0.85, 1.0],
        dtype=jnp.float32,
    )
    return jnp.where(t_safe > 0.13, nominal, fractional)


def logsfr_ratios_to_sfr_bins_jax(dlog10_sfr: jnp.ndarray) -> jnp.ndarray:
    """Convert six PopCosmos SFR log-ratios into seven relative SFR bins.

    The convention is dlog10_sfr_i = log10(SFR_i / SFR_{i+1}), ordered from
    youngest to oldest lookback bin. Thus positive dlog10_sfr_1 makes the
    youngest bin higher than the second bin.
    """
    ratios = jnp.asarray(dlog10_sfr, dtype=jnp.float32)
    older = jnp.cumprod(10.0 ** (-ratios))
    bins = jnp.concatenate([jnp.ones(1, dtype=jnp.float32), older])
    return jnp.clip(
        jnp.nan_to_num(bins, nan=1.0, posinf=1.0e30, neginf=1.0e-30), 1.0e-30, 1.0e30
    )


def build_popcosmos_sfh_table_jax(
    gal_t_table: jnp.ndarray, t_obs: jnp.ndarray, params: dict[str, Any]
) -> jnp.ndarray:
    """Build a seven-bin constant-SFR table on cosmic-time samples."""
    edges = build_popcosmos_lookback_bin_edges_jax(t_obs)
    sfr_bins = logsfr_ratios_to_sfr_bins_jax(_popcosmos_dlog10_sfr(params))
    lookback = jnp.clip(jnp.asarray(t_obs) - jnp.asarray(gal_t_table), 0.0, edges[-1])
    bin_index = jnp.searchsorted(edges, lookback, side="right") - 1
    bin_index = jnp.clip(bin_index, 0, 6)
    return jnp.clip(sfr_bins[bin_index], 1.0e-30, jnp.inf)


def normalize_sfh_to_stellar_mass_jax(
    gal_t_table: jnp.ndarray,
    gal_sfr_table: jnp.ndarray,
    ssp_lg_age_gyr: jnp.ndarray,
    t_obs: jnp.ndarray,
    log10_stellar_mass: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scale an SFH so the DSPS surviving mass matches log10_stellar_mass."""
    from dsps.imf.surviving_mstar import surviving_mstar
    from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table

    age_weights = calc_age_weights_from_sfh_table(
        gal_t_table,
        gal_sfr_table,
        ssp_lg_age_gyr,
        t_obs,
    )
    frac_surviving_by_age = surviving_mstar(ssp_lg_age_gyr + 9.0)
    mean_frac_surviving = jnp.sum(age_weights * frac_surviving_by_age)
    mean_frac_surviving = jnp.clip(mean_frac_surviving, 1.0e-4, 1.0)
    formed_mass = jnp.trapezoid(gal_sfr_table, gal_t_table) * 1.0e9
    target_surviving_mass = 10.0 ** jnp.asarray(log10_stellar_mass, dtype=jnp.float32)
    target_formed_mass = target_surviving_mass / mean_frac_surviving
    scale = target_formed_mass / jnp.maximum(formed_mass, 1.0e-30)
    scaled_sfr = jnp.clip(gal_sfr_table * scale, 1.0e-30, jnp.inf)
    scaled_formed_mass = jnp.trapezoid(scaled_sfr, gal_t_table) * 1.0e9
    scaled_surviving_mass = scaled_formed_mass * mean_frac_surviving
    return scaled_sfr, scaled_formed_mass, scaled_surviving_mass


def interpolate_ssp_stellar_metallicity_jax(
    ssp_lgmet: jnp.ndarray, ssp_flux: jnp.ndarray, lgmet_abs: jnp.ndarray
) -> jnp.ndarray:
    """Interpolate SSP flux to a single absolute log10 stellar metallicity."""
    return _interp_axis0_linear(
        jnp.asarray(ssp_lgmet, dtype=jnp.float32),
        jnp.asarray(ssp_flux, dtype=jnp.float32),
        jnp.asarray(lgmet_abs, dtype=jnp.float32),
    )


def apply_charlot_fall_by_age_jax(
    wave: jnp.ndarray,
    ssp_lg_age_gyr: jnp.ndarray,
    sed_by_age: jnp.ndarray,
    tau2: jnp.ndarray,
    dust_index_n: jnp.ndarray,
    tau1_over_tau2: jnp.ndarray,
    birth_cloud_slope: float = -1.0,
) -> jnp.ndarray:
    """Apply Charlot-Fall diffuse and birth-cloud attenuation by SSP age."""
    wave_safe = jnp.maximum(jnp.asarray(wave, dtype=jnp.float32), 1.0)
    tau2_safe = jnp.maximum(jnp.asarray(tau2, dtype=jnp.float32), 0.0)
    diffuse = tau2_safe * (wave_safe / 5500.0) ** jnp.asarray(
        dust_index_n, dtype=jnp.float32
    )
    tau1 = jnp.maximum(jnp.asarray(tau1_over_tau2, dtype=jnp.float32), 0.0) * tau2_safe
    birth = tau1 * (wave_safe / 5500.0) ** jnp.asarray(
        birth_cloud_slope, dtype=jnp.float32
    )
    age_gyr = 10.0 ** jnp.asarray(ssp_lg_age_gyr, dtype=jnp.float32)
    young = age_gyr <= 0.01
    old_trans = jnp.exp(-jnp.clip(diffuse, 0.0, 80.0))
    young_trans = jnp.exp(-jnp.clip(diffuse + birth, 0.0, 80.0))
    transmission = jnp.where(young[:, None], young_trans[None, :], old_trans[None, :])
    return jnp.asarray(sed_by_age, dtype=jnp.float32) * transmission


def apply_igm_transmission_jax(
    wave_rest: jnp.ndarray,
    rest_sed: jnp.ndarray,
    z_obs: jnp.ndarray,
    model_config: dict[str, Any] | None,
) -> jnp.ndarray:
    """Apply a stable JAX approximation to Madau95 IGM transmission."""
    mode = str(_normalized_model_config(model_config).get("igm_model", "none"))
    if mode == "none":
        return rest_sed
    if mode != "madau95_approx":
        raise ValueError(f"Unsupported model.igm_model: {mode}")
    wave = jnp.maximum(jnp.asarray(wave_rest, dtype=jnp.float32), 1.0)
    z = jnp.maximum(jnp.asarray(z_obs, dtype=jnp.float32), 0.0)
    below_lya = jnp.clip((1216.0 - wave) / 1216.0, 0.0, 1.0)
    below_limit = jnp.clip((912.0 - wave) / 912.0, 0.0, 1.0)
    tau_forest = 0.35 * z**1.6 * below_lya**1.2 * (1216.0 / wave) ** 0.7
    tau_continuum = 1.8 * z**2.0 * below_limit**1.5 * (912.0 / wave) ** 2.0
    transmission = jnp.exp(-jnp.clip(tau_forest + tau_continuum, 0.0, 80.0))
    return jnp.asarray(rest_sed, dtype=jnp.float32) * transmission


def interpolate_gas_ssp_grid_jax(
    context: DspsContext,
    log10_gas_metallicity: jnp.ndarray,
    log10_gas_ionization: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate a gas SSP grid to gas metallicity and ionization."""
    if (
        context.gas_lgmet_grid_jax is None
        or context.gas_lgu_grid_jax is None
        or context.ssp_flux_gas_grid_jax is None
    ):
        raise ValueError("model.nebular_model='gas_grid' requires a loaded gas grid")
    gas_z_lo, gas_z_hi, gas_z_weight = _interp_bracket(
        context.gas_lgmet_grid_jax,
        log10_gas_metallicity,
    )
    gas_u_lo, gas_u_hi, gas_u_weight = _interp_bracket(
        context.gas_lgu_grid_jax,
        log10_gas_ionization,
    )
    grid = jnp.asarray(context.ssp_flux_gas_grid_jax, dtype=jnp.float32)
    f00 = grid[gas_z_lo, gas_u_lo]
    f01 = grid[gas_z_lo, gas_u_hi]
    f10 = grid[gas_z_hi, gas_u_lo]
    f11 = grid[gas_z_hi, gas_u_hi]
    low_u = f00 * (1.0 - gas_u_weight) + f01 * gas_u_weight
    high_u = f10 * (1.0 - gas_u_weight) + f11 * gas_u_weight
    return low_u * (1.0 - gas_z_weight) + high_u * gas_z_weight


def _interp_bracket(
    x_grid: jnp.ndarray, x: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_grid = jnp.asarray(x_grid, dtype=jnp.float32)
    x_clipped = jnp.clip(jnp.asarray(x, dtype=jnp.float32), x_grid[0], x_grid[-1])
    hi = jnp.searchsorted(x_grid, x_clipped, side="right")
    hi = jnp.clip(hi, 1, x_grid.shape[0] - 1)
    lo = hi - 1
    x0 = x_grid[lo]
    x1 = x_grid[hi]
    weight = (x_clipped - x0) / jnp.maximum(x1 - x0, 1.0e-12)
    return lo, hi, weight


def add_agn_component_jax(
    context: DspsContext,
    wave: jnp.ndarray,
    intrinsic_stellar_sed: jnp.ndarray,
    dusted_sed: jnp.ndarray,
    params: dict[str, Any],
    model_config: dict[str, Any] | None,
) -> jnp.ndarray:
    """Add an AGN template grid component.

    The AGN amplitude uses a stable bolometric integral over the intrinsic
    stellar Lnu SED. This is an approximation until the exact FSPS/CLUMPY
    convention is audited.
    """
    config = _normalized_model_config(model_config)
    if str(config.get("agn_model", "none")) == "none":
        return dusted_sed
    if (
        context.agn_wave_jax is None
        or context.agn_tau_grid_jax is None
        or context.agn_template_grid_jax is None
    ):
        raise ValueError("model.agn_model='template_grid' requires a loaded AGN grid")
    tauagn = jnp.exp(jnp.asarray(params["ln_tauagn"], dtype=jnp.float32))
    fagn = jnp.exp(jnp.asarray(params["ln_fagn"], dtype=jnp.float32))
    template_tau = _interp_axis0_linear(
        context.agn_tau_grid_jax,
        context.agn_template_grid_jax,
        tauagn,
    )
    template = jnp.interp(
        jnp.asarray(wave, dtype=jnp.float32),
        context.agn_wave_jax,
        template_tau,
        left=0.0,
        right=0.0,
    )
    c_angstrom_per_s = 2.99792458e18
    wave_safe = jnp.maximum(jnp.asarray(wave, dtype=jnp.float32), 1.0)
    lbol_stellar = jnp.trapezoid(
        jnp.maximum(intrinsic_stellar_sed, 0.0) * c_angstrom_per_s / wave_safe**2,
        wave_safe,
    )
    agn_lnu = fagn * jnp.maximum(lbol_stellar, 0.0) * jnp.maximum(template, 0.0)
    return jnp.asarray(dusted_sed, dtype=jnp.float32) + agn_lnu


def _popcosmos_dlog10_sfr(params: dict[str, Any]) -> jnp.ndarray:
    return jnp.asarray(
        [params[name] for name in POPCOSMOS_PARAMETER_NAMES[2:8]],
        dtype=jnp.float32,
    )


def _popcosmos_ssp_flux_grid(
    context: DspsContext, params: dict[str, Any], model_config: dict[str, Any]
) -> jnp.ndarray:
    if str(model_config.get("nebular_model", "fixed_ssp")) == "gas_grid":
        return interpolate_gas_ssp_grid_jax(
            context,
            params["log10_gas_metallicity"],
            params["log10_gas_ionization"],
        )
    return _context_ssp_flux(context)


def _interp_axis0_linear(
    x_grid: jnp.ndarray, values: jnp.ndarray, x: jnp.ndarray
) -> jnp.ndarray:
    x_grid = jnp.asarray(x_grid, dtype=jnp.float32)
    values = jnp.asarray(values, dtype=jnp.float32)
    x_clipped = jnp.clip(jnp.asarray(x, dtype=jnp.float32), x_grid[0], x_grid[-1])
    hi = jnp.searchsorted(x_grid, x_clipped, side="right")
    hi = jnp.clip(hi, 1, x_grid.shape[0] - 1)
    lo = hi - 1
    x0 = x_grid[lo]
    x1 = x_grid[hi]
    weight = (x_clipped - x0) / jnp.maximum(x1 - x0, 1.0e-12)
    return values[lo] * (1.0 - weight) + values[hi] * weight


def _jax_result_derived_array(result: JaxModelResult) -> jnp.ndarray:
    formed = result.formed_mass_msun
    surviving = result.surviving_stellar_mass_msun
    sfr_obs = result.sfr_at_obs_msun_per_yr
    scalars = jnp.asarray(
        [
            result.t_obs_gyr,
            formed,
            jnp.log10(jnp.maximum(formed, 1.0e-300)),
            surviving,
            jnp.log10(jnp.maximum(surviving, 1.0e-300)),
            sfr_obs,
            jnp.log10(jnp.maximum(sfr_obs, 1.0e-300)),
        ],
        dtype=jnp.float32,
    )
    return jnp.concatenate(
        [
            scalars,
            jnp.asarray(result.sfr_bins_msun_per_yr, dtype=jnp.float32),
            jnp.asarray(result.lookback_bin_edges_gyr, dtype=jnp.float32),
        ]
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


def dynamic_model_args(context: DspsContext) -> tuple[Any, ...]:
    """Return JAX arrays that should be dynamic inputs to jitted model calls."""
    return tuple(getattr(context, field) for field in DYNAMIC_CONTEXT_FIELDS)


def bind_dynamic_model_args(
    context: DspsContext, args: tuple[Any, ...] | list[Any]
) -> DspsContext:
    """Return a shallow context copy with large JAX arrays supplied as arguments."""
    if len(args) != len(DYNAMIC_CONTEXT_FIELDS):
        raise ValueError(
            "dynamic model args length mismatch: "
            f"expected {len(DYNAMIC_CONTEXT_FIELDS)}, got {len(args)}"
        )
    bound = copy.copy(context)
    for field, value in zip(DYNAMIC_CONTEXT_FIELDS, args, strict=True):
        setattr(bound, field, value)
    return bound


def model_mags_jax_dynamic(
    context: DspsContext, args: tuple[Any, ...] | list[Any], params: dict[str, Any]
) -> jnp.ndarray:
    """Model magnitudes with large context arrays passed as dynamic JAX args."""
    return model_mags_jax(bind_dynamic_model_args(context, args), params)


def run_dsps_model_jax_dynamic(
    context: DspsContext, args: tuple[Any, ...] | list[Any], params: dict[str, Any]
) -> JaxModelResult:
    """Full model result with large context arrays passed as dynamic JAX args."""
    return run_dsps_model_jax(bind_dynamic_model_args(context, args), params)


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

        def single(values, model_args):
            params = {name: values[index] for index, name in enumerate(parameter_names)}
            return model_mags_jax_dynamic(context, model_args, params)

        def batch(values, model_args):
            return jax.vmap(lambda row: single(row, model_args))(values)

        _BATCH_PREDICT_CACHE[cache_key] = jax.jit(batch)

    predict = _BATCH_PREDICT_CACHE[cache_key]
    return np.asarray(
        predict(
            jnp.asarray(parameter_matrix, dtype=jnp.float32),
            dynamic_model_args(context),
        )
    )


def derived_quantities_jax(context: DspsContext, params: dict[str, Any]) -> jnp.ndarray:
    """Return derived quantities needed for scientifically comparable reports."""
    return _jax_result_derived_array(run_dsps_model_jax(context, params))


def predict_batch_derived(
    context: DspsContext, parameter_names: list[str], parameter_matrix: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute derived quantities for many fitted parameter rows."""
    cache_key = ("derived", id(context), tuple(parameter_names))
    if cache_key not in _BATCH_PREDICT_CACHE:

        def single(values, model_args):
            params = {name: values[index] for index, name in enumerate(parameter_names)}
            result = run_dsps_model_jax_dynamic(context, model_args, params)
            return _jax_result_derived_array(result)

        def batch(values, model_args):
            return jax.vmap(lambda row: single(row, model_args))(values)

        _BATCH_PREDICT_CACHE[cache_key] = jax.jit(batch)

    predict = _BATCH_PREDICT_CACHE[cache_key]
    values = np.asarray(
        predict(
            jnp.asarray(parameter_matrix, dtype=jnp.float32),
            dynamic_model_args(context),
        )
    )
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

        def single(values, model_args):
            params = {name: values[index] for index, name in enumerate(parameter_names)}
            result = run_dsps_model_jax_dynamic(context, model_args, params)
            derived = _jax_result_derived_array(result)
            return result.rest_sed, result.dusted_rest_sed, result.model_mags, derived

        def batch(values, model_args):
            return jax.vmap(lambda row: single(row, model_args))(values)

        _BATCH_PREDICT_CACHE[cache_key] = jax.jit(batch)

    predict = _BATCH_PREDICT_CACHE[cache_key]
    rest_sed, dusted_rest_sed, model_mags, derived_values = predict(
        jnp.asarray(parameter_matrix, dtype=jnp.float32),
        dynamic_model_args(context),
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
        flux_ratio = model_flux / observed_flux if observed_flux > 0 else float("nan")
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
