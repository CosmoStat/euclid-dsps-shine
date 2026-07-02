"""Native DSPS model wrapper."""

# ruff: noqa: I001, E402

from __future__ import annotations

import hashlib
import copy
import csv
import json
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
from .parameters import DIFFSTAR_FIXED_PARAMETER_DEFAULTS, POPCOSMOS_PARAMETER_NAMES
from .photometry import abmag_to_fnu_cgs


POPCOSMOS_Z_SUN = 0.0142
LEGACY_Z_SUN = 0.0134
_AXIS_RTOL = 0.0
_WAVE_ATOL = 1.0e-4
_LOG_AXIS_ATOL = 1.0e-6


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
    compressed_ssp_basis_jax: Any | None = None
    compressed_ssp_coeff_jax: Any | None = None
    compressed_ssp_scale_jax: Any | None = None
    ssp_surviving_mstar_jax: Any | None = None
    gas_lgmet_grid_jax: Any | None = None
    gas_lgu_grid_jax: Any | None = None
    ssp_flux_gas_grid_jax: Any | None = None
    compressed_gas_basis_jax: Any | None = None
    compressed_gas_coeff_jax: Any | None = None
    compressed_gas_scale_jax: Any | None = None
    agn_wave_jax: Any | None = None
    agn_fagn_grid_jax: Any | None = None
    agn_tau_grid_jax: Any | None = None
    agn_tage_grid_jax: Any | None = None
    agn_logzsol_grid_jax: Any | None = None
    agn_template_grid_jax: Any | None = None
    agn_component_lgmet_jax: Any | None = None
    agn_component_lg_age_gyr_jax: Any | None = None
    agn_component_grid_jax: Any | None = None
    compressed_agn_basis_jax: Any | None = None
    compressed_agn_coeff_jax: Any | None = None
    compressed_agn_scale_jax: Any | None = None
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
    "compressed_ssp_basis_jax",
    "compressed_ssp_coeff_jax",
    "compressed_ssp_scale_jax",
    "ssp_surviving_mstar_jax",
    "gas_lgmet_grid_jax",
    "gas_lgu_grid_jax",
    "ssp_flux_gas_grid_jax",
    "compressed_gas_basis_jax",
    "compressed_gas_coeff_jax",
    "compressed_gas_scale_jax",
    "agn_wave_jax",
    "agn_fagn_grid_jax",
    "agn_tau_grid_jax",
    "agn_tage_grid_jax",
    "agn_logzsol_grid_jax",
    "agn_template_grid_jax",
    "agn_component_lgmet_jax",
    "agn_component_lg_age_gyr_jax",
    "agn_component_grid_jax",
    "compressed_agn_basis_jax",
    "compressed_agn_coeff_jax",
    "compressed_agn_scale_jax",
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
    stellar_intrinsic_sed: jnp.ndarray | None = None
    stellar_dusted_sed: jnp.ndarray | None = None
    gas_sed: jnp.ndarray | None = None
    agn_sed: jnp.ndarray | None = None
    pre_igm_sed: jnp.ndarray | None = None
    post_igm_sed: jnp.ndarray | None = None


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
    _validate_popcosmos_ssp_metadata(ssp_path, model_config)
    compressed_ssp_basis, compressed_ssp_coeff, compressed_ssp_scale = (
        _load_optional_compressed_ssp_grid(model_config, ssp)
    )
    dust_k_by_code, dust_curve_names = _load_cosmos_dust_grid(ssp, cosmos_config)
    emline_luminosity, emline_wave, emline_name = _load_ssp_emline_data(ssp_path, ssp)
    surviving_mstar = _load_ssp_surviving_mstar(ssp_path, ssp)
    (
        gas_lgmet_grid,
        gas_lgu_grid,
        ssp_flux_gas_grid,
        compressed_gas_basis,
        compressed_gas_coeff,
        compressed_gas_scale,
    ) = _load_optional_gas_grid(model_config, ssp)
    (
        agn_wave,
        agn_fagn_grid,
        agn_tau_grid,
        agn_tage_grid,
        agn_logzsol_grid,
        agn_template_grid,
        agn_component_lgmet,
        agn_component_lg_age_gyr,
        agn_component_grid,
        compressed_agn_basis,
        compressed_agn_coeff,
        compressed_agn_scale,
    ) = _load_optional_agn_grid(model_config, ssp)
    return DspsContext(
        ssp=ssp,
        filters=filters,
        n_sfh_bins=n_sfh_bins,
        z_sun=float(model_config.get("z_sun", LEGACY_Z_SUN)),
        model_config=model_config,
        cosmos_dust_k_by_code=dust_k_by_code,
        cosmos_dust_curve_names=dust_curve_names,
        ssp_wave_jax=jnp.asarray(ssp.ssp_wave, dtype=jnp.float32),
        ssp_lgmet_jax=jnp.asarray(ssp.ssp_lgmet, dtype=jnp.float32),
        ssp_lg_age_gyr_jax=jnp.asarray(ssp.ssp_lg_age_gyr, dtype=jnp.float32),
        ssp_flux_jax=(
            None
            if str(model_config.get("ssp_model", "dense")) == "compressed_basis"
            else jnp.asarray(ssp.ssp_flux, dtype=jnp.float32)
        ),
        compressed_ssp_basis_jax=_jax_optional_array_preserve_float(
            compressed_ssp_basis,
            runtime_dtype=model_config.get("compressed_ssp_runtime_dtype"),
        ),
        compressed_ssp_coeff_jax=_jax_optional_array_preserve_float(
            compressed_ssp_coeff,
            runtime_dtype=model_config.get("compressed_ssp_runtime_dtype"),
        ),
        compressed_ssp_scale_jax=_jax_optional_array_preserve_float(
            compressed_ssp_scale,
            runtime_dtype=model_config.get("compressed_ssp_runtime_dtype"),
        ),
        ssp_surviving_mstar_jax=(
            None
            if surviving_mstar is None
            else jnp.asarray(surviving_mstar, dtype=jnp.float32)
        ),
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
        compressed_gas_basis_jax=(
            None
            if compressed_gas_basis is None
            else _jax_array_preserve_float(compressed_gas_basis)
        ),
        compressed_gas_coeff_jax=(
            None
            if compressed_gas_coeff is None
            else _jax_array_preserve_float(compressed_gas_coeff)
        ),
        compressed_gas_scale_jax=(
            None
            if compressed_gas_scale is None
            else jnp.asarray(compressed_gas_scale, dtype=jnp.float32)
        ),
        agn_wave_jax=(
            None if agn_wave is None else jnp.asarray(agn_wave, dtype=jnp.float32)
        ),
        agn_fagn_grid_jax=(
            None
            if agn_fagn_grid is None
            else jnp.asarray(agn_fagn_grid, dtype=jnp.float32)
        ),
        agn_tau_grid_jax=(
            None
            if agn_tau_grid is None
            else jnp.asarray(agn_tau_grid, dtype=jnp.float32)
        ),
        agn_tage_grid_jax=(
            None
            if agn_tage_grid is None
            else jnp.asarray(agn_tage_grid, dtype=jnp.float32)
        ),
        agn_logzsol_grid_jax=(
            None
            if agn_logzsol_grid is None
            else jnp.asarray(agn_logzsol_grid, dtype=jnp.float32)
        ),
        agn_template_grid_jax=(
            None
            if agn_template_grid is None
            else jnp.asarray(agn_template_grid, dtype=jnp.float32)
        ),
        agn_component_lgmet_jax=(
            None
            if agn_component_lgmet is None
            else jnp.asarray(agn_component_lgmet, dtype=jnp.float32)
        ),
        agn_component_lg_age_gyr_jax=(
            None
            if agn_component_lg_age_gyr is None
            else jnp.asarray(agn_component_lg_age_gyr, dtype=jnp.float32)
        ),
        agn_component_grid_jax=(
            None
            if agn_component_grid is None
            else jnp.asarray(agn_component_grid, dtype=jnp.float32)
        ),
        compressed_agn_basis_jax=(
            None
            if compressed_agn_basis is None
            else _jax_array_preserve_float(compressed_agn_basis)
        ),
        compressed_agn_coeff_jax=(
            None
            if compressed_agn_coeff is None
            else _jax_array_preserve_float(compressed_agn_coeff)
        ),
        compressed_agn_scale_jax=(
            None
            if compressed_agn_scale is None
            else jnp.asarray(compressed_agn_scale, dtype=jnp.float32)
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
    config.setdefault("ssp_model", "dense")
    popcosmos_like = _is_popcosmos_like_model_config(config)
    config.setdefault(
        "stellar_metallicity_model",
        "single" if popcosmos_like else "mdf",
    )
    config.setdefault(
        "dust_model",
        "charlot_fall_powerlaw" if popcosmos_like else "legacy",
    )
    config["dust_model"] = _normalize_model_dust_model(config["dust_model"])
    config.setdefault("igm_model", "none")
    config.setdefault("nebular_model", "fixed_ssp")
    config.setdefault("agn_model", "none")
    config.setdefault("agn_host_attenuation", "none")
    config.setdefault("agn_host_attenuation_scale", 1.0)
    config.setdefault("agn_igm_order", "pre_igm")
    config.setdefault("agn_baked_attenuation", "none")
    config.setdefault("agn_baked_dust_index", -0.7)
    config.setdefault("birth_cloud_slope", -1.0)
    config.setdefault("dust_tesc_logyr", 7.0)
    config.setdefault("dust1_index", -1.0)
    config.setdefault("emission_line_corrections", "none")
    config.setdefault("z_sun", POPCOSMOS_Z_SUN if popcosmos_like else LEGACY_Z_SUN)
    config.setdefault(
        "sfh_time_grid",
        "prospector_step" if sfh_model == "popcosmos_bins" else "linear",
    )
    return config


def _jax_array_preserve_float(
    value: np.ndarray,
    *,
    runtime_dtype: str | None = None,
) -> jnp.ndarray:
    """Copy float16/float32 compressed payloads without upcasting resident arrays."""
    array = np.asarray(value)
    if runtime_dtype:
        dtype = np.dtype(runtime_dtype)
        if dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
            raise ValueError(
                "Compressed runtime dtype must be 'float16' or 'float32', "
                f"got {runtime_dtype!r}"
            )
        return jnp.asarray(array, dtype=dtype)
    if array.dtype == np.float16:
        return jnp.asarray(array, dtype=jnp.float16)
    return jnp.asarray(array, dtype=jnp.float32)


def _jax_optional_array_preserve_float(
    value: np.ndarray | None,
    *,
    runtime_dtype: str | None = None,
) -> jnp.ndarray | None:
    if value is None:
        return None
    return _jax_array_preserve_float(value, runtime_dtype=runtime_dtype)


def _load_optional_compressed_ssp_grid(
    model_config: dict[str, Any], ssp: Any
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    ssp_model = str(model_config.get("ssp_model", "dense"))
    if ssp_model == "dense":
        return None, None, None
    if ssp_model != "compressed_basis":
        raise ValueError(f"Unsupported model.ssp_model: {ssp_model}")
    path = model_config.get("compressed_ssp_path")
    if not path:
        raise ValueError(
            "model.ssp_model='compressed_basis' requires model.compressed_ssp_path"
        )
    return _load_compressed_ssp_grid(path, reference_ssp=ssp, model_config=model_config)


def _load_compressed_ssp_grid(
    path: str | Path,
    reference_ssp: Any,
    model_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_path = Path(path).expanduser()
    if not grid_path.exists():
        raise FileNotFoundError(f"Compressed SSP grid file not found: {grid_path}")
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - pyproject requires h5py
        raise RuntimeError("h5py is required to load compressed SSP grids") from exc
    required = (
        "ssp_wave",
        "ssp_lg_age_gyr",
        "ssp_lgmet",
        "ssp_basis",
        "ssp_coeff",
    )
    with h5py.File(grid_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(
                f"Compressed SSP grid {grid_path} is missing datasets: "
                f"{', '.join(missing)}"
            )
        attrs = _hdf5_attrs(handle)
        wave = np.asarray(handle["ssp_wave"], dtype=float)
        lg_age = np.asarray(handle["ssp_lg_age_gyr"], dtype=float)
        lgmet = np.asarray(handle["ssp_lgmet"], dtype=float)
        basis = _read_float_dataset_preserve_dtype(handle["ssp_basis"])
        coeff = _read_float_dataset_preserve_dtype(handle["ssp_coeff"])
        scale = (
            np.asarray(handle["ssp_scale"], dtype=np.float32)
            if "ssp_scale" in handle
            else np.ones(coeff.shape[:-1], dtype=np.float32)
        )
    if basis.ndim != 2 or basis.shape[1] != len(wave):
        raise ValueError("Compressed SSP grid ssp_basis must have shape (n_basis, n_wave)")
    expected_coeff_shape = (len(lgmet), len(lg_age), basis.shape[0])
    if coeff.ndim != 3 or coeff.shape != expected_coeff_shape:
        raise ValueError(
            "Compressed SSP grid ssp_coeff must have shape "
            "(n_ssp_lgmet, n_ssp_lg_age_gyr, n_basis)"
        )
    if scale.shape != expected_coeff_shape[:-1]:
        raise ValueError(
            "Compressed SSP grid ssp_scale must have shape "
            "(n_ssp_lgmet, n_ssp_lg_age_gyr)"
        )
    _validate_compressed_ssp_reference_axes(grid_path, reference_ssp, wave, lg_age, lgmet)
    _validate_popcosmos_ssp_metadata(grid_path, model_config or {})
    _validate_compressed_asset_dtype_metadata(
        grid_path,
        attrs,
        {"ssp_basis": basis.dtype, "ssp_coeff": coeff.dtype, "ssp_scale": scale.dtype},
    )
    return basis, coeff, scale


def _read_float_dataset_preserve_dtype(dataset: Any) -> np.ndarray:
    dtype = np.dtype(dataset.dtype)
    if dtype == np.dtype(np.float16):
        return np.asarray(dataset, dtype=np.float16)
    if dtype == np.dtype(np.float32):
        return np.asarray(dataset, dtype=np.float32)
    return np.asarray(dataset, dtype=np.float32)


def _assert_axis_close(
    path: Path,
    name: str,
    expected: np.ndarray,
    actual: np.ndarray,
    atol: float,
) -> None:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.shape != actual.shape or not np.allclose(
        actual, expected, rtol=_AXIS_RTOL, atol=atol
    ):
        raise ValueError(
            f"Compressed SSP grid {path} axis {name} does not match the reference "
            f"SSP; expected shape {expected.shape}, got {actual.shape}; "
            f"tolerances are rtol={_AXIS_RTOL}, atol={atol}."
        )


def _validate_compressed_ssp_reference_axes(
    path: Path,
    reference_ssp: Any,
    wave: np.ndarray,
    lg_age: np.ndarray,
    lgmet: np.ndarray,
) -> None:
    _assert_axis_close(path, "ssp_wave", np.asarray(reference_ssp.ssp_wave), wave, _WAVE_ATOL)
    _assert_axis_close(
        path,
        "ssp_lg_age_gyr",
        np.asarray(reference_ssp.ssp_lg_age_gyr),
        lg_age,
        _LOG_AXIS_ATOL,
    )
    _assert_axis_close(
        path,
        "ssp_lgmet",
        np.asarray(reference_ssp.ssp_lgmet),
        lgmet,
        _LOG_AXIS_ATOL,
    )


def _validate_compressed_asset_dtype_metadata(
    path: Path, attrs: dict[str, Any], actual: dict[str, np.dtype]
) -> None:
    declared = _json_attr(attrs, "compressed_dtypes")
    if not declared:
        return
    mismatches = []
    for name, dtype in actual.items():
        expected = declared.get(name)
        if expected is not None and str(expected) != str(np.dtype(dtype)):
            mismatches.append(f"{name}: metadata={expected}, dataset={np.dtype(dtype)}")
    if mismatches:
        raise ValueError(
            f"Compressed asset {path} has inconsistent dtype metadata: "
            f"{'; '.join(mismatches)}"
        )


def _load_optional_gas_grid(
    model_config: dict[str, Any],
    ssp: Any | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    nebular_model = str(model_config.get("nebular_model", "fixed_ssp"))
    if nebular_model == "fixed_ssp":
        return None, None, None, None, None, None
    if nebular_model == "gas_grid":
        path = model_config.get("gas_grid_path")
        if not path:
            raise ValueError("model.nebular_model='gas_grid' requires model.gas_grid_path")
        gas_lgmet, gas_lgu, ssp_flux = _load_gas_ssp_grid(
            path, reference_ssp=ssp, model_config=model_config
        )
        return gas_lgmet, gas_lgu, ssp_flux, None, None, None
    if nebular_model == "compressed_gas_grid":
        path = model_config.get("compressed_gas_grid_path")
        if not path:
            raise ValueError(
                "model.nebular_model='compressed_gas_grid' requires "
                "model.compressed_gas_grid_path"
            )
        gas_lgmet, gas_lgu, basis, coeff, scale = _load_compressed_gas_ssp_grid(
            path, reference_ssp=ssp, model_config=model_config
        )
        return gas_lgmet, gas_lgu, None, basis, coeff, scale
    raise ValueError(f"Unsupported model.nebular_model: {nebular_model}")


def _load_gas_ssp_grid(
    path: str | Path,
    reference_ssp: Any | None = None,
    model_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_path = Path(path).expanduser()
    if not grid_path.exists():
        raise FileNotFoundError(f"Gas SSP grid file not found: {grid_path}")
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - pyproject requires h5py
        raise RuntimeError("h5py is required to load gas SSP grids") from exc

    required_axes = (
        "ssp_wave",
        "ssp_lg_age_gyr",
        "ssp_lgmet",
        "gas_lgmet_grid",
        "gas_lgu_grid",
    )
    enriched_required = (
        "nebular_continuum_flux",
        "line_flux_grid",
        "emline_wavelengths",
    )
    with h5py.File(grid_path, "r") as handle:
        missing = [key for key in required_axes if key not in handle]
        has_ssp_flux = "ssp_flux" in handle
        has_enriched_lines = all(key in handle for key in enriched_required)
        if not has_ssp_flux and not has_enriched_lines:
            missing.append("ssp_flux or enriched line datasets")
        if missing:
            raise ValueError(
                f"Gas SSP grid {grid_path} is missing datasets: {', '.join(missing)}"
            )
        attrs = _hdf5_attrs(handle)
        gas_lgmet_grid = np.asarray(handle["gas_lgmet_grid"], dtype=float)
        gas_lgu_grid = np.asarray(handle["gas_lgu_grid"], dtype=float)
        ssp_wave = np.asarray(handle["ssp_wave"], dtype=float)
        ssp_lg_age_gyr = np.asarray(handle["ssp_lg_age_gyr"], dtype=float)
        ssp_lgmet = np.asarray(handle["ssp_lgmet"], dtype=float)
        if _requires_popcosmos_table_corrections(model_config):
            if not has_enriched_lines:
                raise ValueError(
                    "model.emission_line_corrections='popcosmos_table' requires "
                    "gas grid datasets nebular_continuum_flux, line_flux_grid, "
                    "and emline_wavelengths"
                )
            ssp_flux = _corrected_enriched_gas_flux(handle, ssp_wave, model_config)
        elif has_ssp_flux:
            ssp_flux = np.asarray(handle["ssp_flux"], dtype=np.float32)
        else:
            ssp_flux = _corrected_enriched_gas_flux(handle, ssp_wave, model_config)

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
    _validate_gas_grid_reference_axes(
        grid_path,
        reference_ssp,
        ssp_wave,
        ssp_lg_age_gyr,
        ssp_lgmet,
        model_config,
    )
    _validate_popcosmos_gas_metadata(grid_path, attrs, model_config)
    return gas_lgmet_grid, gas_lgu_grid, ssp_flux


def _load_compressed_gas_ssp_grid(
    path: str | Path,
    reference_ssp: Any | None = None,
    model_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a low-rank gas SSP grid without materializing dense spectra."""
    grid_path = Path(path).expanduser()
    if not grid_path.exists():
        raise FileNotFoundError(f"Compressed gas SSP grid file not found: {grid_path}")
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - pyproject requires h5py
        raise RuntimeError("h5py is required to load compressed gas SSP grids") from exc

    required = (
        "ssp_wave",
        "ssp_lg_age_gyr",
        "ssp_lgmet",
        "gas_lgmet_grid",
        "gas_lgu_grid",
        "gas_basis",
        "gas_coeff",
    )
    with h5py.File(grid_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(
                f"Compressed gas SSP grid {grid_path} is missing datasets: "
                f"{', '.join(missing)}"
            )
        attrs = _hdf5_attrs(handle)
        gas_lgmet_grid = np.asarray(handle["gas_lgmet_grid"], dtype=float)
        gas_lgu_grid = np.asarray(handle["gas_lgu_grid"], dtype=float)
        ssp_wave = np.asarray(handle["ssp_wave"], dtype=float)
        ssp_lg_age_gyr = np.asarray(handle["ssp_lg_age_gyr"], dtype=float)
        ssp_lgmet = np.asarray(handle["ssp_lgmet"], dtype=float)
        basis = _read_float_dataset_preserve_dtype(handle["gas_basis"])
        coeff = _read_float_dataset_preserve_dtype(handle["gas_coeff"])
        scale = (
            np.asarray(handle["gas_scale"], dtype=np.float32)
            if "gas_scale" in handle
            else np.ones(coeff.shape[:-1], dtype=np.float32)
        )

    if basis.ndim != 2 or basis.shape[1] != len(ssp_wave):
        raise ValueError(
            "Compressed gas SSP grid gas_basis must have shape (n_basis, n_wave)"
        )
    expected_coeff_shape = (
        len(gas_lgmet_grid),
        len(gas_lgu_grid),
        len(ssp_lgmet),
        len(ssp_lg_age_gyr),
        basis.shape[0],
    )
    if coeff.ndim != 5 or coeff.shape != expected_coeff_shape:
        raise ValueError(
            "Compressed gas SSP grid gas_coeff must have shape "
            "(n_gas_lgmet, n_gas_lgu, n_ssp_lgmet, n_ssp_lg_age_gyr, n_basis)"
        )
    if scale.shape != expected_coeff_shape[:-1]:
        raise ValueError(
            "Compressed gas SSP grid gas_scale must have shape "
            "(n_gas_lgmet, n_gas_lgu, n_ssp_lgmet, n_ssp_lg_age_gyr)"
        )
    _validate_gas_grid_reference_axes(
        grid_path,
        reference_ssp,
        ssp_wave,
        ssp_lg_age_gyr,
        ssp_lgmet,
        model_config,
    )
    _validate_popcosmos_gas_metadata(grid_path, attrs, model_config)
    _validate_compressed_asset_dtype_metadata(
        grid_path,
        attrs,
        {"gas_basis": basis.dtype, "gas_coeff": coeff.dtype, "gas_scale": scale.dtype},
    )
    return gas_lgmet_grid, gas_lgu_grid, basis, coeff, scale


def _load_optional_agn_grid(
    model_config: dict[str, Any],
    reference_ssp: Any | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    agn_model = str(model_config.get("agn_model", "none"))
    if agn_model == "none":
        return None, None, None, None, None, None, None, None, None, None, None, None
    if agn_model == "fsps_component_grid":
        path = model_config.get("agn_component_grid_path")
        if not path:
            raise ValueError(
                "model.agn_model='fsps_component_grid' requires "
                "model.agn_component_grid_path"
            )
        wave, fagn, tau, lgmet, lg_age, grid = _load_agn_component_grid(
            path, model_config=model_config, reference_ssp=reference_ssp
        )
        return wave, fagn, tau, None, None, None, lgmet, lg_age, grid, None, None, None
    if agn_model == "compressed_fsps_component_grid":
        path = model_config.get("compressed_agn_component_grid_path")
        if not path:
            raise ValueError(
                "model.agn_model='compressed_fsps_component_grid' requires "
                "model.compressed_agn_component_grid_path"
            )
        wave, fagn, tau, lgmet, lg_age, basis, coeff, scale = (
            _load_compressed_agn_component_grid(
                path, model_config=model_config, reference_ssp=reference_ssp
            )
        )
        return wave, fagn, tau, None, None, None, lgmet, lg_age, None, basis, coeff, scale
    if agn_model != "template_grid":
        raise ValueError(f"Unsupported model.agn_model: {agn_model}")
    path = model_config.get("agn_template_path")
    if not path:
        raise ValueError(
            "model.agn_model='template_grid' requires model.agn_template_path"
        )
    wave, fagn, tau, tage, logzsol, grid = _load_agn_template_grid(
        path, model_config=model_config
    )
    return wave, fagn, tau, tage, logzsol, grid, None, None, None, None, None, None


def _load_agn_template_grid(
    path: str | Path,
    model_config: dict[str, Any] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
]:
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
        attrs = _hdf5_attrs(handle)
        wave = np.asarray(handle["wave"], dtype=float)
        fagn_grid = (
            np.asarray(handle["fagn_grid"], dtype=float)
            if "fagn_grid" in handle
            else (
                np.asarray(handle["fagn_normalization_grid"], dtype=float)
                if "fagn_normalization_grid" in handle
                else None
            )
        )
        tau_grid = np.asarray(handle["agn_tau_grid"], dtype=float)
        tage_grid = (
            np.asarray(handle["tage_gyr_grid"], dtype=float)
            if "tage_gyr_grid" in handle
            else None
        )
        logzsol_grid = (
            np.asarray(handle["stellar_logzsol_grid"], dtype=float)
            if "stellar_logzsol_grid" in handle
            else None
        )
        template_grid = np.asarray(handle["template_lnu_per_lbol"], dtype=float)

    if template_grid.ndim == 2 and template_grid.shape == (len(tau_grid), len(wave)):
        pass
    elif template_grid.ndim == 5:
        if fagn_grid is None or tage_grid is None or logzsol_grid is None:
            raise ValueError(
                "5D AGN template_lnu_per_lbol requires fagn_grid or "
                "fagn_normalization_grid, tage_gyr_grid, and stellar_logzsol_grid"
            )
        expected_shape = (
            len(fagn_grid),
            len(tau_grid),
            len(tage_grid),
            len(logzsol_grid),
            len(wave),
        )
        if template_grid.shape != expected_shape:
            raise ValueError(
                "5D AGN template_lnu_per_lbol must have shape "
                "(n_fagn_grid, n_agn_tau_grid, n_tage_gyr_grid, "
                "n_stellar_logzsol_grid, n_wave)"
            )
    else:
        raise ValueError(
            "AGN template_lnu_per_lbol must have shape (n_agn_tau_grid, n_wave) "
            "or (n_fagn_grid, n_agn_tau_grid, n_tage_gyr_grid, "
            "n_stellar_logzsol_grid, n_wave)"
        )
    _validate_popcosmos_agn_metadata(grid_path, attrs, model_config)
    return wave, fagn_grid, tau_grid, tage_grid, logzsol_grid, template_grid


def _load_agn_component_grid(
    path: str | Path,
    model_config: dict[str, Any] | None = None,
    reference_ssp: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load an FSPS-native AGN component SSP grid.

    Expected units are ``Lsun/Hz/Msun formed`` for
    ``agn_lnu_per_mformed[fagn, agn_tau, Zstar, age, wave]``.
    """
    grid_path = Path(path).expanduser()
    if not grid_path.exists():
        raise FileNotFoundError(f"AGN component grid file not found: {grid_path}")
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - pyproject requires h5py
        raise RuntimeError("h5py is required to load AGN component grids") from exc

    required = (
        "ssp_wave",
        "ssp_lg_age_gyr",
        "ssp_lgmet",
        "fagn_grid",
        "agn_tau_grid",
        "agn_lnu_per_mformed",
    )
    with h5py.File(grid_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(
                f"AGN component grid {grid_path} is missing datasets: "
                f"{', '.join(missing)}"
            )
        attrs = _hdf5_attrs(handle)
        wave = np.asarray(handle["ssp_wave"], dtype=float)
        lg_age = np.asarray(handle["ssp_lg_age_gyr"], dtype=float)
        lgmet = np.asarray(handle["ssp_lgmet"], dtype=float)
        fagn_grid = np.asarray(handle["fagn_grid"], dtype=float)
        tau_grid = np.asarray(handle["agn_tau_grid"], dtype=float)
        component = np.asarray(handle["agn_lnu_per_mformed"], dtype=np.float32)

    expected_shape = (
        len(fagn_grid),
        len(tau_grid),
        len(lgmet),
        len(lg_age),
        len(wave),
    )
    if component.ndim != 5 or component.shape != expected_shape:
        raise ValueError(
            "AGN component grid agn_lnu_per_mformed must have shape "
            "(n_fagn_grid, n_agn_tau_grid, n_ssp_lgmet, n_ssp_lg_age_gyr, n_wave)"
        )
    _validate_agn_component_reference_axes(
        grid_path,
        reference_ssp,
        wave,
        lg_age,
        lgmet,
    )
    _validate_popcosmos_agn_metadata(grid_path, attrs, model_config)
    return wave, fagn_grid, tau_grid, lgmet, lg_age, component


def _load_compressed_agn_component_grid(
    path: str | Path,
    model_config: dict[str, Any] | None = None,
    reference_ssp: Any | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load a low-rank FSPS-native AGN component grid.

    The compressed representation stores spectra as
    ``agn_scale * (agn_coeff @ agn_basis)`` and never materializes the dense
    ``(fagn, tau, Z, age, wave)`` tensor during context loading.
    """
    grid_path = Path(path).expanduser()
    if not grid_path.exists():
        raise FileNotFoundError(f"Compressed AGN component grid file not found: {grid_path}")
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - pyproject requires h5py
        raise RuntimeError(
            "h5py is required to load compressed AGN component grids"
        ) from exc

    required = (
        "ssp_wave",
        "ssp_lg_age_gyr",
        "ssp_lgmet",
        "fagn_grid",
        "agn_tau_grid",
        "agn_basis",
        "agn_coeff",
    )
    with h5py.File(grid_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(
                f"Compressed AGN component grid {grid_path} is missing datasets: "
                f"{', '.join(missing)}"
            )
        attrs = _hdf5_attrs(handle)
        wave = np.asarray(handle["ssp_wave"], dtype=float)
        lg_age = np.asarray(handle["ssp_lg_age_gyr"], dtype=float)
        lgmet = np.asarray(handle["ssp_lgmet"], dtype=float)
        fagn_grid = np.asarray(handle["fagn_grid"], dtype=float)
        tau_grid = np.asarray(handle["agn_tau_grid"], dtype=float)
        basis = _read_float_dataset_preserve_dtype(handle["agn_basis"])
        coeff = _read_float_dataset_preserve_dtype(handle["agn_coeff"])
        scale = (
            np.asarray(handle["agn_scale"], dtype=np.float32)
            if "agn_scale" in handle
            else np.ones(coeff.shape[:-1], dtype=np.float32)
        )
        fagn_handling = str(attrs.get("fagn_handling", "grid_interpolation"))

    if basis.ndim != 2 or basis.shape[1] != len(wave):
        raise ValueError(
            "Compressed AGN component grid agn_basis must have shape "
            "(n_basis, n_wave)"
        )
    if fagn_handling == "linear_runtime_multiplier":
        expected_coeff_shape = (
            len(tau_grid),
            len(lgmet),
            len(lg_age),
            basis.shape[0],
        )
        shape_message = "(n_agn_tau_grid, n_ssp_lgmet, n_ssp_lg_age_gyr, n_basis)"
        scale_shape_message = "(n_agn_tau_grid, n_ssp_lgmet, n_ssp_lg_age_gyr)"
    else:
        expected_coeff_shape = (
            len(fagn_grid),
            len(tau_grid),
            len(lgmet),
            len(lg_age),
            basis.shape[0],
        )
        shape_message = (
            "(n_fagn_grid, n_agn_tau_grid, n_ssp_lgmet, "
            "n_ssp_lg_age_gyr, n_basis)"
        )
        scale_shape_message = (
            "(n_fagn_grid, n_agn_tau_grid, n_ssp_lgmet, n_ssp_lg_age_gyr)"
        )
    if coeff.shape != expected_coeff_shape:
        raise ValueError(
            "Compressed AGN component grid agn_coeff must have shape "
            f"{shape_message}"
        )
    expected_scale_shape = expected_coeff_shape[:-1]
    if scale.shape != expected_scale_shape:
        raise ValueError(
            "Compressed AGN component grid agn_scale must have shape "
            f"{scale_shape_message}"
        )
    _validate_agn_component_reference_axes(
        grid_path,
        reference_ssp,
        wave,
        lg_age,
        lgmet,
    )
    _validate_popcosmos_agn_metadata(grid_path, attrs, model_config)
    _validate_compressed_asset_dtype_metadata(
        grid_path,
        attrs,
        {"agn_basis": basis.dtype, "agn_coeff": coeff.dtype, "agn_scale": scale.dtype},
    )
    return wave, fagn_grid, tau_grid, lgmet, lg_age, basis, coeff, scale


def _validate_agn_component_reference_axes(
    path: Path,
    reference_ssp: Any | None,
    wave: np.ndarray,
    lg_age: np.ndarray,
    lgmet: np.ndarray,
) -> None:
    if reference_ssp is None:
        return
    expected_axes = {
        "ssp_wave": np.asarray(reference_ssp.ssp_wave, dtype=float),
        "ssp_lg_age_gyr": np.asarray(reference_ssp.ssp_lg_age_gyr, dtype=float),
        "ssp_lgmet": np.asarray(reference_ssp.ssp_lgmet, dtype=float),
    }
    actual_axes = {
        "ssp_wave": np.asarray(wave, dtype=float),
        "ssp_lg_age_gyr": np.asarray(lg_age, dtype=float),
        "ssp_lgmet": np.asarray(lgmet, dtype=float),
    }
    tolerances = {
        "ssp_wave": _WAVE_ATOL,
        "ssp_lg_age_gyr": _LOG_AXIS_ATOL,
        "ssp_lgmet": _LOG_AXIS_ATOL,
    }
    for key, actual in actual_axes.items():
        expected = expected_axes[key]
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=_AXIS_RTOL, atol=tolerances[key]
        ):
            raise ValueError(
                f"AGN component grid {path} {key} axis is incompatible with "
                f"the base SSP axis. Expected shape {expected.shape}, got "
                f"{actual.shape}; tolerances are rtol={_AXIS_RTOL}, "
                f"atol={tolerances[key]}."
            )


def _normalize_model_dust_model(value: Any) -> str:
    name = str(value).strip().lower().replace("-", "_")
    aliases = {
        "charlot_fall": "charlot_fall_powerlaw",
        "charlot_fall_power_law": "charlot_fall_powerlaw",
    }
    return aliases.get(name, name)


def _is_popcosmos_like_model_config(model_config: dict[str, Any] | None) -> bool:
    config = model_config or {}
    return str(config.get("sfh_model", "lognormal")) in {
        "popcosmos_bins",
        "diffstar_reduced6",
        "diffsky_basic",
    }


def _hdf5_attrs(handle: Any) -> dict[str, Any]:
    return {str(key): _decode_hdf5_attr(value) for key, value in handle.attrs.items()}


def _decode_hdf5_attr(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_decode_hdf5_attr(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_attr(attrs: dict[str, Any], key: str) -> dict[str, Any]:
    raw = attrs.get(key)
    if raw in (None, ""):
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _float_attr(attrs: dict[str, Any], key: str) -> float | None:
    value = attrs.get(key)
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _imf_metadata(attrs: dict[str, Any]) -> tuple[float | None, str | None]:
    imf_type = _float_attr(attrs, "imf_type")
    imf_name = attrs.get("imf_name")
    if imf_name is not None:
        imf_name = str(imf_name).strip().lower()
    controls = _json_attr(attrs, "fsps_controls")
    if imf_type is None and "imf_type" in controls:
        try:
            imf_type = float(controls["imf_type"])
        except (TypeError, ValueError):
            imf_type = None
    if imf_name is None and "imf_name" in controls:
        imf_name = str(controls["imf_name"]).strip().lower()
    return imf_type, imf_name


def _validate_popcosmos_ssp_metadata(
    ssp_path: str | Path, model_config: dict[str, Any]
) -> None:
    if not _requires_popcosmos_asset_metadata(model_config):
        return
    path = Path(ssp_path).expanduser()
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            attrs = _hdf5_attrs(handle)
    except OSError as exc:
        raise ValueError(f"Could not read PopCosmos-like SSP metadata: {path}") from exc
    _validate_popcosmos_asset_name(path, "SSP")
    _validate_popcosmos_imf(path, attrs, "SSP")
    _validate_popcosmos_z_sun(path, attrs, model_config, "SSP")


def _validate_popcosmos_gas_metadata(
    path: Path, attrs: dict[str, Any], model_config: dict[str, Any] | None
) -> None:
    if not _requires_popcosmos_asset_metadata(model_config):
        return
    _validate_popcosmos_asset_name(path, "gas SSP grid")
    _validate_popcosmos_imf(path, attrs, "gas SSP grid")
    _validate_popcosmos_z_sun(path, attrs, model_config, "gas SSP grid")
    required_units = {
        "units_ssp_wave": "angstrom",
        "units_ssp_lg_age_gyr": "log10(age/gyr)",
        "units_ssp_lgmet": "log10(absolute stellar metallicity mass fraction)",
        "units_gas_lgmet_grid": "log10(zgas/zsun)",
        "units_gas_lgu_grid": "log10",
    }
    missing = []
    for key, needle in required_units.items():
        value = str(attrs.get(key, "")).strip().lower()
        if needle not in value:
            missing.append(key)
    if missing:
        raise ValueError(
            f"PopCosmos-like gas SSP grid {path} is missing required unit metadata: "
            f"{', '.join(missing)}"
        )


def _validate_popcosmos_agn_metadata(
    path: Path, attrs: dict[str, Any], model_config: dict[str, Any] | None
) -> None:
    if not _requires_popcosmos_asset_metadata(model_config):
        return
    _validate_popcosmos_asset_name(path, "AGN template grid")
    _validate_popcosmos_imf(path, attrs, "AGN template grid")
    _validate_popcosmos_z_sun(path, attrs, model_config, "AGN template grid")


def _validate_popcosmos_asset_name(path: Path, label: str) -> None:
    if "kroupa" in path.name.lower():
        raise ValueError(
            f"PopCosmos-like {label} path must not contain 'kroupa': {path}"
        )


def _requires_popcosmos_asset_metadata(model_config: dict[str, Any] | None) -> bool:
    config = model_config or {}
    metadata_policy = str(config.get("asset_metadata_policy", "strict")).lower()
    if metadata_policy in {"permissive", "skip", "none"}:
        return False
    return str(config.get("sfh_model", "lognormal")) in {
        "popcosmos_bins",
        "diffstar_reduced6",
    }


def _validate_popcosmos_imf(path: Path, attrs: dict[str, Any], label: str) -> None:
    imf_type, imf_name = _imf_metadata(attrs)
    if imf_type is None and imf_name is None:
        raise ValueError(
            f"PopCosmos-like {label} {path} is missing IMF metadata; expected "
            "imf_type=1 or imf_name='chabrier'."
        )
    errors = []
    if imf_type is not None and not np.isclose(imf_type, 1.0, rtol=0.0, atol=0.0):
        errors.append(f"imf_type={imf_type!r}")
    if imf_name is not None and imf_name != "chabrier":
        errors.append(f"imf_name={imf_name!r}")
    if not errors:
        return
    raise ValueError(
        f"PopCosmos-like {label} {path} must consistently declare a Chabrier IMF "
        f"(imf_type=1 when present and imf_name='chabrier' when present); found "
        f"imf_type={imf_type!r}, imf_name={imf_name!r}."
    )


def _validate_popcosmos_z_sun(
    path: Path, attrs: dict[str, Any], model_config: dict[str, Any] | None, label: str
) -> None:
    z_sun_attr = _float_attr(attrs, "z_sun")
    if z_sun_attr is None:
        raise ValueError(
            f"PopCosmos-like {label} {path} is missing z_sun metadata; expected "
            f"z_sun={float(_normalized_model_config(model_config).get('z_sun'))}."
        )
    z_sun_config = float(_normalized_model_config(model_config).get("z_sun"))
    if not np.isclose(z_sun_attr, z_sun_config, rtol=0.0, atol=1.0e-8):
        raise ValueError(
            f"PopCosmos-like {label} z_sun mismatch for {path}: config has "
            f"{z_sun_config}, HDF5 metadata has {z_sun_attr}."
        )


def _validate_gas_grid_reference_axes(
    path: Path,
    reference_ssp: Any | None,
    ssp_wave: np.ndarray,
    ssp_lg_age_gyr: np.ndarray,
    ssp_lgmet: np.ndarray,
    model_config: dict[str, Any] | None,
) -> None:
    if reference_ssp is None:
        return
    reference_axes = {
        "ssp_wave": np.asarray(reference_ssp.ssp_wave, dtype=float),
        "ssp_lg_age_gyr": np.asarray(reference_ssp.ssp_lg_age_gyr, dtype=float),
        "ssp_lgmet": np.asarray(reference_ssp.ssp_lgmet, dtype=float),
    }
    actual_axes = {
        "ssp_wave": np.asarray(ssp_wave, dtype=float),
        "ssp_lg_age_gyr": np.asarray(ssp_lg_age_gyr, dtype=float),
        "ssp_lgmet": np.asarray(ssp_lgmet, dtype=float),
    }
    tolerances = {
        "ssp_wave": _WAVE_ATOL,
        "ssp_lg_age_gyr": _LOG_AXIS_ATOL,
        "ssp_lgmet": _LOG_AXIS_ATOL,
    }
    for key, actual in actual_axes.items():
        expected = reference_axes[key]
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=_AXIS_RTOL, atol=tolerances[key]
        ):
            raise ValueError(
                f"Gas SSP grid {path} {key} axis is incompatible with the base SSP "
                f"axis. Expected shape {expected.shape}, got {actual.shape}; "
                f"tolerances are rtol={_AXIS_RTOL}, atol={tolerances[key]}."
            )
    if _is_popcosmos_like_model_config(model_config):
        # Axis units are validated from metadata separately for PopCosmos-like grids.
        return


def _requires_popcosmos_table_corrections(
    model_config: dict[str, Any] | None,
) -> bool:
    return (
        str(_normalized_model_config(model_config).get("emission_line_corrections"))
        == "popcosmos_table"
    )


def _corrected_enriched_gas_flux(
    handle: Any, ssp_wave: np.ndarray, model_config: dict[str, Any] | None
) -> np.ndarray:
    continuum = np.asarray(handle["nebular_continuum_flux"], dtype=np.float32)
    line_grid = np.asarray(handle["line_flux_grid"], dtype=np.float32)
    line_wave = np.asarray(handle["emline_wavelengths"], dtype=float)
    if line_grid.shape[:-1] != continuum.shape[:-1]:
        raise ValueError(
            "Gas SSP grid line_flux_grid must have shape "
            "(n_gas_lgmet, n_gas_lgu, n_stellar_lgmet, n_age, n_line)"
        )
    if continuum.ndim != 5 or line_grid.ndim != 5 or line_grid.shape[-1] != len(
        line_wave
    ):
        raise ValueError("Gas SSP grid enriched line datasets have incompatible shapes")
    line_names = _line_names_from_hdf5(handle, line_grid.shape[-1])
    multipliers = _line_correction_multipliers(line_names, model_config)
    corrected = np.array(continuum, copy=True)
    wave = np.asarray(ssp_wave, dtype=float)
    for line_index, line_wavelength in enumerate(line_wave):
        wave_index = int(np.argmin(np.abs(wave - float(line_wavelength))))
        corrected[..., wave_index] += (
            line_grid[..., line_index] * multipliers[line_index]
        )
    return np.asarray(corrected, dtype=np.float32)


def _line_names_from_hdf5(handle: Any, n_lines: int) -> tuple[str, ...]:
    for key in ("line_name", "line_names", "emline_names", "ssp_emline_name"):
        if key not in handle:
            continue
        raw = np.asarray(handle[key])
        names = []
        for item in raw:
            names.append(str(_decode_hdf5_attr(item)))
        if len(names) == n_lines:
            return tuple(names)
    return tuple(f"line_{index:03d}" for index in range(n_lines))


def _line_correction_multipliers(
    line_names: tuple[str, ...], model_config: dict[str, Any] | None
) -> np.ndarray:
    config = _normalized_model_config(model_config)
    if str(config.get("emission_line_corrections")) == "none":
        return np.ones(len(line_names), dtype=np.float32)
    if str(config.get("emission_line_corrections")) != "popcosmos_table":
        raise ValueError(
            f"Unsupported model.emission_line_corrections: "
            f"{config.get('emission_line_corrections')}"
        )
    path = config.get("emission_line_correction_path")
    if not path:
        raise ValueError(
            "model.emission_line_corrections='popcosmos_table' requires "
            "model.emission_line_correction_path"
        )
    corrections = _read_emission_line_correction_table(path)
    multipliers = np.ones(len(line_names), dtype=np.float32)
    missing = []
    for index, name in enumerate(line_names):
        if name not in corrections:
            missing.append(name)
            continue
        multipliers[index] = np.float32(1.0 + corrections[name])
    if missing:
        raise ValueError(
            "Emission-line correction table is missing lines present in the gas "
            f"grid: {', '.join(missing)}"
        )
    if not np.all(np.isfinite(multipliers)):
        raise ValueError("Emission-line correction multipliers contain non-finite values")
    return multipliers


def _read_emission_line_correction_table(path: str | Path) -> dict[str, float]:
    table_path = Path(path).expanduser()
    if not table_path.exists():
        raise FileNotFoundError(f"Emission-line correction table not found: {table_path}")
    required = {
        "line_name",
        "line_wavelength",
        "fractional_correction",
        "fractional_variance",
    }
    corrections: dict[str, float] = {}
    with table_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Emission-line correction table {table_path} must contain columns "
                f"{sorted(required)}"
            )
        for row in reader:
            name = str(row["line_name"]).strip()
            if not name:
                raise ValueError("Emission-line correction table has an empty line_name")
            for key in ("line_wavelength", "fractional_correction", "fractional_variance"):
                try:
                    value = float(row[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Emission-line correction table {table_path} has non-numeric "
                        f"{key} for line {name!r}"
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(
                        f"Emission-line correction table {table_path} has non-finite "
                        f"{key} for line {name!r}"
                    )
            corrections[name] = float(row["fractional_correction"])
    return corrections


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


def _load_ssp_surviving_mstar(
    ssp_path: str | Path,
    ssp: Any,
) -> np.ndarray | None:
    """Load optional FSPS surviving-mass fractions from an SSP HDF5."""
    path = Path(ssp_path).expanduser()
    candidates = (
        "ssp_surviving_mstar",
        "ssp_stellar_mass",
        "ssp_mstar_remaining",
    )
    try:
        import h5py
    except ImportError:  # pragma: no cover - pyproject requires h5py
        return None
    with h5py.File(path, "r") as handle:
        for name in candidates:
            if name not in handle:
                continue
            values = np.asarray(handle[name], dtype=np.float32)
            expected = (len(ssp.ssp_lgmet), len(ssp.ssp_lg_age_gyr))
            if values.shape != expected:
                raise ValueError(
                    f"SSP surviving-mass dataset {name!r} in {path} has shape "
                    f"{values.shape}; expected {expected}"
                )
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(
                    f"SSP surviving-mass dataset {name!r} in {path} must be "
                    "positive and finite"
                )
            return np.clip(values, 1.0e-4, 2.0)
    return None


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
    excluded = {
        "redshift",
        "redshiftHubble",
        "redshift_truth",
        "redshift_hubble_truth",
        str(redshift_config.get("column") or ""),
        str(redshift_config.get("truth_column") or ""),
    }
    id_keys = ("galaxy_id", "object_id", "source_id", "id", "row_index")
    id_payload = [
        f"{key}={row[key]}"
        for key in id_keys
        if key in row and np.isscalar(row[key]) and key not in excluded
    ]
    if id_payload:
        payload = "|".join(id_payload)
    else:
        payload = "|".join(
            f"{key}={row[key]}"
            for key in sorted(row)
            if key not in excluded and np.isscalar(row[key])
        )
    if not payload:
        payload = "no-stable-row-id"
    digest = hashlib.blake2b(f"{seed}|{payload}".encode(), digest_size=8).digest()
    unit = int.from_bytes(digest, "big") / float(2**64 - 1)
    return z_min + unit * (z_max - z_min)


def run_dsps_model(context: DspsContext, params: dict[str, float]) -> ModelResult:
    """Run DSPS from simple SFH/metallicity parameters to SED and photometry."""
    if float(
        gas_metallicity_constraint_penalty_jax(
            params, context.model_config, penalty=1.0
        )
    ) > 0.0:
        raise ValueError(
            "PopCosmos-like gas metallicity constraint violated: "
            "log10_gas_metallicity must be >= log10_stellar_metallicity "
            "(both in log10(Z/Zsun))."
        )
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
    if str(model_config.get("sfh_model", "lognormal")) == "diffstar_reduced6":
        return run_diffstar_reduced6_model_jax(context, params)
    if str(model_config.get("sfh_model", "lognormal")) == "diffsky_basic":
        return run_diffsky_basic_model_jax(context, params)
    return run_lognormal_model_jax(context, params)


def run_dsps_model_mags_jax(context: DspsContext, params: dict[str, Any]) -> jnp.ndarray:
    """Return model magnitudes without constructing diagnostic SED outputs."""
    model_config = _normalized_model_config(context.model_config)
    sfh_model = str(model_config.get("sfh_model", "lognormal"))
    if sfh_model == "popcosmos_bins":
        return run_popcosmos_binned_model_mags_jax(context, params)
    if sfh_model == "diffstar_reduced6":
        return run_diffstar_reduced6_model_mags_jax(context, params)
    if sfh_model == "diffsky_basic":
        return run_diffsky_basic_model_mags_jax(context, params)
    return run_lognormal_model_jax(context, params).model_mags


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
        stellar_intrinsic_sed=sed_info.rest_sed,
        stellar_dusted_sed=dusted_sed,
        gas_sed=jnp.zeros_like(dusted_sed),
        agn_sed=jnp.zeros_like(dusted_sed),
        pre_igm_sed=dusted_sed,
        post_igm_sed=dusted_sed,
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
    gal_t_table, raw_sfr_table = build_popcosmos_sfh_time_grid_jax(
        t_obs,
        params,
        context.n_sfh_bins,
        model_config,
    )
    ssp_lg_age_gyr = _context_ssp_lg_age_gyr(context)
    lgmet_abs = log10_stellar_metallicity_to_absolute_jax(
        params["log10_stellar_metallicity"], context.z_sun
    )
    frac_surviving_by_age = _context_surviving_mstar_by_age(context, lgmet_abs)
    if str(model_config.get("sfh_time_grid", "prospector_step")) == "prospector_step":
        age_weights = popcosmos_age_weights_jax(t_obs, params, ssp_lg_age_gyr)
        gal_sfr_table, formed_mass, surviving_mass = (
            normalize_sfh_to_stellar_mass_with_age_weights_jax(
                gal_t_table,
                raw_sfr_table,
                ssp_lg_age_gyr,
                age_weights,
                params["log10_stellar_mass"],
                frac_surviving_by_age,
            )
        )
    else:
        gal_sfr_table, formed_mass, surviving_mass = normalize_sfh_to_stellar_mass_jax(
            gal_t_table,
            raw_sfr_table,
            ssp_lg_age_gyr,
            t_obs,
            params["log10_stellar_mass"],
            frac_surviving_by_age,
        )
        age_weights = calc_age_weights_from_sfh_table(
            gal_t_table,
            gal_sfr_table,
            ssp_lg_age_gyr,
            t_obs,
        )
    sfr_bins_raw = logsfr_ratios_to_sfr_bins_jax(_popcosmos_dlog10_sfr(params))
    formed_raw = jnp.trapezoid(raw_sfr_table, gal_t_table) * 1.0e9
    formed_scale = formed_mass / jnp.maximum(formed_raw, 1.0e-30)
    sfr_bins = jnp.clip(sfr_bins_raw * formed_scale, 1.0e-30, jnp.inf)
    lookback_edges = build_popcosmos_lookback_bin_edges_jax(t_obs)

    stellar_ssp_flux_z = interpolate_context_ssp_stellar_metallicity_jax(
        context,
        lgmet_abs,
    )
    stellar_sed_by_age = (
        jnp.clip(stellar_ssp_flux_z, 0.0, jnp.inf) * age_weights[:, None] * formed_mass
    )
    stellar_intrinsic_sed = jnp.nan_to_num(
        stellar_sed_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
    )
    ssp_flux_z = interpolate_popcosmos_ssp_stellar_metallicity_jax(
        context,
        params,
        model_config,
        lgmet_abs,
    )
    sed_by_age = jnp.clip(ssp_flux_z, 0.0, jnp.inf) * age_weights[:, None] * formed_mass
    intrinsic_stellar_sed = jnp.nan_to_num(
        sed_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
    )
    gas_sed = jnp.nan_to_num(
        intrinsic_stellar_sed - stellar_intrinsic_sed,
        nan=0.0,
        posinf=1.0e30,
        neginf=-1.0e30,
    )
    stellar_dusted_by_age = apply_popcosmos_dust_by_age_jax(
        _context_ssp_wave(context),
        ssp_lg_age_gyr,
        stellar_sed_by_age,
        params["tau2"],
        params["dust_index_n"],
        params["tau1_over_tau2"],
        model_config,
    )
    stellar_dusted_sed = jnp.nan_to_num(
        stellar_dusted_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
    )
    dusted_by_age = apply_popcosmos_dust_by_age_jax(
        _context_ssp_wave(context),
        ssp_lg_age_gyr,
        sed_by_age,
        params["tau2"],
        params["dust_index_n"],
        params["tau1_over_tau2"],
        model_config,
    )
    dusted_sed = jnp.nan_to_num(
        dusted_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
    )
    agn_sed = agn_component_jax(
        context,
        _context_ssp_wave(context),
        intrinsic_stellar_sed,
        params,
        model_config,
        age_weights=age_weights,
        formed_mass=formed_mass,
        stellar_lgmet_abs=lgmet_abs,
        template_tage_gyr=t_obs,
        stellar_logzsol=params["log10_stellar_metallicity"],
    )
    pre_igm_sed, post_igm_sed = combine_agn_and_igm_jax(
        _context_ssp_wave(context),
        dusted_sed,
        agn_sed,
        z_obs,
        model_config,
    )
    model_mags = predict_mags_jax(
        context, _context_ssp_wave(context), post_igm_sed, z_obs
    )
    return JaxModelResult(
        wave=_context_ssp_wave(context),
        rest_sed=intrinsic_stellar_sed,
        dusted_rest_sed=post_igm_sed,
        model_mags=model_mags,
        t_obs_gyr=t_obs,
        formed_mass_msun=formed_mass,
        surviving_stellar_mass_msun=surviving_mass,
        sfr_at_obs_msun_per_yr=gal_sfr_table[-1],
        sfr_bins_msun_per_yr=sfr_bins,
        lookback_bin_edges_gyr=lookback_edges,
        stellar_intrinsic_sed=stellar_intrinsic_sed,
        stellar_dusted_sed=stellar_dusted_sed,
        gas_sed=gas_sed,
        agn_sed=agn_sed,
        pre_igm_sed=pre_igm_sed,
        post_igm_sed=post_igm_sed,
    )


def run_popcosmos_binned_model_mags_jax(
    context: DspsContext, params: dict[str, Any]
) -> jnp.ndarray:
    """Likelihood-only PopCosmos binned forward path."""
    from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
    from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table

    model_config = _normalized_model_config(context.model_config)
    z_obs = jnp.asarray(params["z_obs"], dtype=jnp.float32)
    t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]
    gal_t_table, raw_sfr_table = build_popcosmos_sfh_time_grid_jax(
        t_obs,
        params,
        context.n_sfh_bins,
        model_config,
    )
    ssp_lg_age_gyr = _context_ssp_lg_age_gyr(context)
    lgmet_abs = log10_stellar_metallicity_to_absolute_jax(
        params["log10_stellar_metallicity"], context.z_sun
    )
    frac_surviving_by_age = _context_surviving_mstar_by_age(context, lgmet_abs)
    if str(model_config.get("sfh_time_grid", "prospector_step")) == "prospector_step":
        age_weights = popcosmos_age_weights_jax(t_obs, params, ssp_lg_age_gyr)
        _, formed_mass, _ = normalize_sfh_to_stellar_mass_with_age_weights_jax(
            gal_t_table,
            raw_sfr_table,
            ssp_lg_age_gyr,
            age_weights,
            params["log10_stellar_mass"],
            frac_surviving_by_age,
        )
    else:
        gal_sfr_table, formed_mass, _ = normalize_sfh_to_stellar_mass_jax(
            gal_t_table,
            raw_sfr_table,
            ssp_lg_age_gyr,
            t_obs,
            params["log10_stellar_mass"],
            frac_surviving_by_age,
        )
        age_weights = calc_age_weights_from_sfh_table(
            gal_t_table,
            gal_sfr_table,
            ssp_lg_age_gyr,
            t_obs,
        )
    ssp_flux_z = interpolate_popcosmos_ssp_stellar_metallicity_jax(
        context,
        params,
        model_config,
        lgmet_abs,
    )
    sed_by_age = jnp.clip(ssp_flux_z, 0.0, jnp.inf) * age_weights[:, None] * formed_mass
    wave = _context_ssp_wave(context)
    dusted_by_age = apply_popcosmos_dust_by_age_jax(
        wave,
        ssp_lg_age_gyr,
        sed_by_age,
        params["tau2"],
        params["dust_index_n"],
        params["tau1_over_tau2"],
        model_config,
    )
    dusted_sed = jnp.nan_to_num(
        dusted_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
    )
    agn_model = str(model_config.get("agn_model", "none"))
    if agn_model == "none":
        agn_sed = jnp.zeros_like(dusted_sed)
    else:
        intrinsic_sed = (
            jnp.zeros_like(dusted_sed)
            if agn_model in {"fsps_component_grid", "compressed_fsps_component_grid"}
            else jnp.nan_to_num(
                sed_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
            )
        )
        agn_sed = agn_component_jax(
            context,
            wave,
            intrinsic_sed,
            params,
            model_config,
            age_weights=age_weights,
            formed_mass=formed_mass,
            stellar_lgmet_abs=lgmet_abs,
            template_tage_gyr=t_obs,
            stellar_logzsol=params["log10_stellar_metallicity"],
        )
    _, post_igm_sed = combine_agn_and_igm_jax(
        wave,
        dusted_sed,
        agn_sed,
        z_obs,
        model_config,
    )
    return predict_mags_jax(context, wave, post_igm_sed, z_obs)


def run_diffstar_reduced6_model_jax(
    context: DspsContext, params: dict[str, Any]
) -> JaxModelResult:
    """PopCosmos-like forward model using a six-free-parameter Diffstar SFH.

    The catalog/config interface does not currently provide halo assembly
    parameters, so Diffstar is evaluated with ``DEFAULT_MAH_PARAMS`` from
    diffmah. Gas, dust, IGM, AGN, metallicity, and GPU memory behavior follow
    the current PopCosmos FSPS-grid path.
    """
    from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
    from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table

    model_config = _normalized_model_config(context.model_config)
    if str(model_config.get("stellar_metallicity_model")) != "single":
        raise ValueError(
            "sfh_model='diffstar_reduced6' requires "
            "model.stellar_metallicity_model='single'."
        )
    if str(model_config.get("dust_model")) not in {
        "charlot_fall_powerlaw",
        "prospector_fsps",
    }:
        raise ValueError(
            "sfh_model='diffstar_reduced6' requires "
            "model.dust_model='charlot_fall_powerlaw' or 'prospector_fsps'."
        )

    z_obs = jnp.asarray(params["z_obs"], dtype=jnp.float32)
    t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]
    gal_t_table = jnp.linspace(0.05, jnp.maximum(t_obs, 0.06), context.n_sfh_bins)
    raw_sfr_table = build_diffstar_sfh_table_jax(gal_t_table, t_obs, params)
    ssp_lg_age_gyr = _context_ssp_lg_age_gyr(context)
    lgmet_abs = log10_stellar_metallicity_to_absolute_jax(
        params["log10_stellar_metallicity"], context.z_sun
    )
    frac_surviving_by_age = _context_surviving_mstar_by_age(context, lgmet_abs)
    gal_sfr_table, formed_mass, surviving_mass = normalize_sfh_to_stellar_mass_jax(
        gal_t_table,
        raw_sfr_table,
        ssp_lg_age_gyr,
        t_obs,
        params["log10_stellar_mass"],
        frac_surviving_by_age,
    )
    age_weights = calc_age_weights_from_sfh_table(
        gal_t_table,
        gal_sfr_table,
        ssp_lg_age_gyr,
        t_obs,
    )
    stellar_ssp_flux_z = interpolate_context_ssp_stellar_metallicity_jax(
        context,
        lgmet_abs,
    )
    stellar_sed_by_age = (
        jnp.clip(stellar_ssp_flux_z, 0.0, jnp.inf) * age_weights[:, None] * formed_mass
    )
    stellar_intrinsic_sed = jnp.nan_to_num(
        stellar_sed_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
    )
    ssp_flux_z = interpolate_popcosmos_ssp_stellar_metallicity_jax(
        context,
        params,
        model_config,
        lgmet_abs,
    )
    sed_by_age = jnp.clip(ssp_flux_z, 0.0, jnp.inf) * age_weights[:, None] * formed_mass
    intrinsic_stellar_sed = jnp.nan_to_num(
        sed_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
    )
    gas_sed = jnp.nan_to_num(
        intrinsic_stellar_sed - stellar_intrinsic_sed,
        nan=0.0,
        posinf=1.0e30,
        neginf=-1.0e30,
    )
    stellar_dusted_by_age = apply_popcosmos_dust_by_age_jax(
        _context_ssp_wave(context),
        ssp_lg_age_gyr,
        stellar_sed_by_age,
        params["tau2"],
        params["dust_index_n"],
        params["tau1_over_tau2"],
        model_config,
    )
    stellar_dusted_sed = jnp.nan_to_num(
        stellar_dusted_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
    )
    dusted_by_age = apply_popcosmos_dust_by_age_jax(
        _context_ssp_wave(context),
        ssp_lg_age_gyr,
        sed_by_age,
        params["tau2"],
        params["dust_index_n"],
        params["tau1_over_tau2"],
        model_config,
    )
    dusted_sed = jnp.nan_to_num(
        dusted_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
    )
    agn_sed = agn_component_jax(
        context,
        _context_ssp_wave(context),
        intrinsic_stellar_sed,
        params,
        model_config,
        age_weights=age_weights,
        formed_mass=formed_mass,
        stellar_lgmet_abs=lgmet_abs,
        template_tage_gyr=t_obs,
        stellar_logzsol=params["log10_stellar_metallicity"],
    )
    pre_igm_sed, post_igm_sed = combine_agn_and_igm_jax(
        _context_ssp_wave(context),
        dusted_sed,
        agn_sed,
        z_obs,
        model_config,
    )
    model_mags = predict_mags_jax(
        context, _context_ssp_wave(context), post_igm_sed, z_obs
    )
    return JaxModelResult(
        wave=_context_ssp_wave(context),
        rest_sed=intrinsic_stellar_sed,
        dusted_rest_sed=post_igm_sed,
        model_mags=model_mags,
        t_obs_gyr=t_obs,
        formed_mass_msun=formed_mass,
        surviving_stellar_mass_msun=surviving_mass,
        sfr_at_obs_msun_per_yr=gal_sfr_table[-1],
        sfr_bins_msun_per_yr=project_sfh_to_popcosmos_sfr_bins_jax(
            gal_t_table, gal_sfr_table, t_obs
        ),
        lookback_bin_edges_gyr=build_popcosmos_lookback_bin_edges_jax(t_obs),
        stellar_intrinsic_sed=stellar_intrinsic_sed,
        stellar_dusted_sed=stellar_dusted_sed,
        gas_sed=gas_sed,
        agn_sed=agn_sed,
        pre_igm_sed=pre_igm_sed,
        post_igm_sed=post_igm_sed,
    )


def run_diffstar_reduced6_model_mags_jax(
    context: DspsContext, params: dict[str, Any]
) -> jnp.ndarray:
    """Likelihood-only Diffstar reduced6 forward path."""
    from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
    from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table

    model_config = _normalized_model_config(context.model_config)
    z_obs = jnp.asarray(params["z_obs"], dtype=jnp.float32)
    t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]
    gal_t_table = jnp.linspace(0.05, jnp.maximum(t_obs, 0.06), context.n_sfh_bins)
    raw_sfr_table = build_diffstar_sfh_table_jax(gal_t_table, t_obs, params)
    ssp_lg_age_gyr = _context_ssp_lg_age_gyr(context)
    lgmet_abs = log10_stellar_metallicity_to_absolute_jax(
        params["log10_stellar_metallicity"], context.z_sun
    )
    frac_surviving_by_age = _context_surviving_mstar_by_age(context, lgmet_abs)
    gal_sfr_table, formed_mass, _ = normalize_sfh_to_stellar_mass_jax(
        gal_t_table,
        raw_sfr_table,
        ssp_lg_age_gyr,
        t_obs,
        params["log10_stellar_mass"],
        frac_surviving_by_age,
    )
    age_weights = calc_age_weights_from_sfh_table(
        gal_t_table,
        gal_sfr_table,
        ssp_lg_age_gyr,
        t_obs,
    )
    ssp_flux_z = interpolate_popcosmos_ssp_stellar_metallicity_jax(
        context,
        params,
        model_config,
        lgmet_abs,
    )
    sed_by_age = jnp.clip(ssp_flux_z, 0.0, jnp.inf) * age_weights[:, None] * formed_mass
    wave = _context_ssp_wave(context)
    dusted_by_age = apply_popcosmos_dust_by_age_jax(
        wave,
        ssp_lg_age_gyr,
        sed_by_age,
        params["tau2"],
        params["dust_index_n"],
        params["tau1_over_tau2"],
        model_config,
    )
    dusted_sed = jnp.nan_to_num(
        dusted_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
    )
    agn_model = str(model_config.get("agn_model", "none"))
    if agn_model == "none":
        agn_sed = jnp.zeros_like(dusted_sed)
    else:
        intrinsic_sed = (
            jnp.zeros_like(dusted_sed)
            if agn_model in {"fsps_component_grid", "compressed_fsps_component_grid"}
            else jnp.nan_to_num(
                sed_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
            )
        )
        agn_sed = agn_component_jax(
            context,
            wave,
            intrinsic_sed,
            params,
            model_config,
            age_weights=age_weights,
            formed_mass=formed_mass,
            stellar_lgmet_abs=lgmet_abs,
            template_tage_gyr=t_obs,
            stellar_logzsol=params["log10_stellar_metallicity"],
        )
    _, post_igm_sed = combine_agn_and_igm_jax(
        wave,
        dusted_sed,
        agn_sed,
        z_obs,
        model_config,
    )
    return predict_mags_jax(context, wave, post_igm_sed, z_obs)


def run_diffsky_basic_model_jax(
    context: DspsContext, params: dict[str, Any]
) -> JaxModelResult:
    """Diffsky-aligned forward model using Diffstar + Diffmah object parameters.

    This path is intentionally narrower than the PopCosmos-like full model:
    HLTDS exposes Diffstar, Diffmah, and dust latents, but not object-level gas
    metallicity/ionization or AGN latents. The first fitting target therefore
    keeps nebular gas and AGN out of the free parameterization instead of
    pretending those quantities are recoverable truths.
    """
    from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
    from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table

    model_config = _normalized_model_config(context.model_config)
    _validate_diffsky_basic_metallicity_model(model_config)

    z_obs = jnp.asarray(params["z_obs"], dtype=jnp.float32)
    t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]
    gal_t_table = jnp.linspace(0.05, jnp.maximum(t_obs, 0.06), context.n_sfh_bins)
    raw_sfr_table = build_diffsky_basic_sfh_table_jax(gal_t_table, t_obs, params)
    ssp_lg_age_gyr = _context_ssp_lg_age_gyr(context)
    lgmet_abs = log10_stellar_metallicity_to_absolute_jax(
        params["log10_stellar_metallicity"], context.z_sun
    )
    frac_surviving_by_age = _diffsky_basic_surviving_mstar_by_age_jax(
        context, model_config, lgmet_abs
    )
    gal_sfr_table, formed_mass, surviving_mass = normalize_sfh_to_stellar_mass_jax(
        gal_t_table,
        raw_sfr_table,
        ssp_lg_age_gyr,
        t_obs,
        params["log10_stellar_mass"],
        frac_surviving_by_age,
    )
    age_weights = calc_age_weights_from_sfh_table(
        gal_t_table,
        gal_sfr_table,
        ssp_lg_age_gyr,
        t_obs,
    )
    ssp_flux_z = diffsky_basic_ssp_flux_by_age_jax(context, model_config, lgmet_abs)
    sed_by_age = jnp.clip(ssp_flux_z, 0.0, jnp.inf) * age_weights[:, None] * formed_mass
    intrinsic_sed = jnp.nan_to_num(
        sed_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
    )
    tau2, dust_index_n, tau1_over_tau2 = diffsky_basic_dust_params_jax(params)
    wave = _context_ssp_wave(context)
    dusted_by_age = apply_popcosmos_dust_by_age_jax(
        wave,
        ssp_lg_age_gyr,
        sed_by_age,
        tau2,
        dust_index_n,
        tau1_over_tau2,
        model_config,
    )
    dusted_sed = jnp.nan_to_num(
        dusted_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
    )
    pre_igm_sed, post_igm_sed = combine_agn_and_igm_jax(
        wave,
        dusted_sed,
        jnp.zeros_like(dusted_sed),
        z_obs,
        model_config,
    )
    model_mags = predict_mags_jax(context, wave, post_igm_sed, z_obs)
    return JaxModelResult(
        wave=wave,
        rest_sed=intrinsic_sed,
        dusted_rest_sed=post_igm_sed,
        model_mags=model_mags,
        t_obs_gyr=t_obs,
        formed_mass_msun=formed_mass,
        surviving_stellar_mass_msun=surviving_mass,
        sfr_at_obs_msun_per_yr=gal_sfr_table[-1],
        sfr_bins_msun_per_yr=project_sfh_to_popcosmos_sfr_bins_jax(
            gal_t_table, gal_sfr_table, t_obs
        ),
        lookback_bin_edges_gyr=build_popcosmos_lookback_bin_edges_jax(t_obs),
        stellar_intrinsic_sed=intrinsic_sed,
        stellar_dusted_sed=dusted_sed,
        gas_sed=jnp.zeros_like(dusted_sed),
        agn_sed=jnp.zeros_like(dusted_sed),
        pre_igm_sed=pre_igm_sed,
        post_igm_sed=post_igm_sed,
    )


def run_diffsky_basic_model_mags_jax(
    context: DspsContext, params: dict[str, Any]
) -> jnp.ndarray:
    """Likelihood-only Diffsky basic forward path."""
    from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
    from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table

    model_config = _normalized_model_config(context.model_config)
    _validate_diffsky_basic_metallicity_model(model_config)
    z_obs = jnp.asarray(params["z_obs"], dtype=jnp.float32)
    t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]
    gal_t_table = jnp.linspace(0.05, jnp.maximum(t_obs, 0.06), context.n_sfh_bins)
    raw_sfr_table = build_diffsky_basic_sfh_table_jax(gal_t_table, t_obs, params)
    ssp_lg_age_gyr = _context_ssp_lg_age_gyr(context)
    lgmet_abs = log10_stellar_metallicity_to_absolute_jax(
        params["log10_stellar_metallicity"], context.z_sun
    )
    frac_surviving_by_age = _diffsky_basic_surviving_mstar_by_age_jax(
        context, model_config, lgmet_abs
    )
    gal_sfr_table, formed_mass, _ = normalize_sfh_to_stellar_mass_jax(
        gal_t_table,
        raw_sfr_table,
        ssp_lg_age_gyr,
        t_obs,
        params["log10_stellar_mass"],
        frac_surviving_by_age,
    )
    age_weights = calc_age_weights_from_sfh_table(
        gal_t_table,
        gal_sfr_table,
        ssp_lg_age_gyr,
        t_obs,
    )
    ssp_flux_z = diffsky_basic_ssp_flux_by_age_jax(context, model_config, lgmet_abs)
    sed_by_age = jnp.clip(ssp_flux_z, 0.0, jnp.inf) * age_weights[:, None] * formed_mass
    tau2, dust_index_n, tau1_over_tau2 = diffsky_basic_dust_params_jax(params)
    wave = _context_ssp_wave(context)
    dusted_by_age = apply_popcosmos_dust_by_age_jax(
        wave,
        ssp_lg_age_gyr,
        sed_by_age,
        tau2,
        dust_index_n,
        tau1_over_tau2,
        model_config,
    )
    dusted_sed = jnp.nan_to_num(
        dusted_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
    )
    _, post_igm_sed = combine_agn_and_igm_jax(
        wave,
        dusted_sed,
        jnp.zeros_like(dusted_sed),
        z_obs,
        model_config,
    )
    return predict_mags_jax(context, wave, post_igm_sed, z_obs)


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


def build_popcosmos_sfh_time_grid_jax(
    t_obs: jnp.ndarray,
    params: dict[str, Any],
    n_sfh_bins: int,
    model_config: dict[str, Any] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build the PopCosmos SFH table using the configured time-grid convention."""
    config = _normalized_model_config(model_config)
    mode = str(config.get("sfh_time_grid", "prospector_step"))
    if mode == "linear":
        t_start = jnp.minimum(jnp.asarray(0.001, dtype=jnp.float32), t_obs * 0.01)
        t_start = jnp.maximum(t_start, jnp.asarray(1.0e-5, dtype=jnp.float32))
        gal_t_table = jnp.linspace(
            t_start, jnp.maximum(t_obs, t_start * 1.01), int(n_sfh_bins)
        )
        return gal_t_table, build_popcosmos_sfh_table_jax(gal_t_table, t_obs, params)
    if mode == "prospector_step":
        return build_popcosmos_prospector_step_sfh_table_jax(t_obs, params)
    raise ValueError(f"Unsupported model.sfh_time_grid: {mode}")


def build_popcosmos_prospector_step_sfh_table_jax(
    t_obs: jnp.ndarray, params: dict[str, Any]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a Prospector/FastStepBasis-like step SFH table.

    Prospector converts log-age bins into two tabular-SFH samples around each
    bin edge using a small epsilon. Mirroring that convention removes the
    age-bin integration error from the DSPS benchmark without changing the
    PopCosmos seven-bin parameterization.
    """
    t_safe = jnp.maximum(jnp.asarray(t_obs, dtype=jnp.float32), 1.0e-5)
    edges = build_popcosmos_lookback_bin_edges_jax(t_safe)
    low = edges[:-1]
    high = edges[1:]
    low = low.at[0].set(jnp.maximum(low[0], jnp.asarray(1.0e-4, dtype=jnp.float32)))
    high = jnp.maximum(high, low + jnp.asarray(1.01e-3, dtype=jnp.float32))
    high = high.at[-1].set(jnp.maximum(high[-1], t_safe))
    age_edges = jnp.concatenate([low[:1], high])
    epsilon = jnp.asarray(1.0e-4, dtype=jnp.float32)
    age_points = jnp.sort(
        jnp.concatenate([age_edges * (1.0 - epsilon), age_edges * (1.0 + epsilon)])
    )[1:-1]
    age_points = jnp.minimum(age_points, t_safe * (1.0 - epsilon))
    gal_t_table = (t_safe - age_points)[::-1]
    sfr_bins = logsfr_ratios_to_sfr_bins_jax(_popcosmos_dlog10_sfr(params))
    sfr_table = jnp.repeat(sfr_bins, 2)[::-1]
    return (
        jnp.clip(gal_t_table, jnp.asarray(1.0e-3, dtype=jnp.float32), t_safe),
        jnp.clip(sfr_table, 1.0e-30, jnp.inf),
    )


def popcosmos_age_weights_jax(
    t_obs: jnp.ndarray,
    params: dict[str, Any],
    ssp_lg_age_gyr: jnp.ndarray,
) -> jnp.ndarray:
    """Integrate PopCosmos age bins directly onto the SSP age grid.

    DSPS' generic SFH-table age weighting interpolates a cumulative-mass table
    in log cosmic time. That is appropriate for smooth SFHs, but it distorts the
    sharp seven-bin PopCosmos step SFH. This direct overlap integral mirrors the
    mass-in-age-bin convention used by Prospector/FastStepBasis.
    """
    t_safe = jnp.maximum(jnp.asarray(t_obs, dtype=jnp.float32), 1.0e-5)
    pop_edges = build_popcosmos_lookback_bin_edges_jax(t_safe)
    pop_low = pop_edges[:-1]
    pop_high = pop_edges[1:]
    pop_low = pop_low.at[0].set(jnp.maximum(pop_low[0], jnp.asarray(1.0e-4)))
    pop_high = jnp.maximum(pop_high, pop_low + jnp.asarray(1.01e-3))
    pop_high = pop_high.at[-1].set(jnp.maximum(pop_high[-1], t_safe))

    sfr_bins = logsfr_ratios_to_sfr_bins_jax(_popcosmos_dlog10_sfr(params))
    ssp_edges = ssp_lg_age_bin_edges_jax(ssp_lg_age_gyr)
    ssp_low = 10.0 ** ssp_edges[:-1]
    ssp_high = 10.0 ** ssp_edges[1:]

    overlap = jnp.maximum(
        0.0,
        jnp.minimum(ssp_high[:, None], pop_high[None, :])
        - jnp.maximum(ssp_low[:, None], pop_low[None, :]),
    )
    mass_by_age = jnp.sum(overlap * sfr_bins[None, :], axis=1)
    total_mass = jnp.maximum(jnp.sum(mass_by_age), 1.0e-30)
    weights = mass_by_age / total_mass
    return jnp.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)


def ssp_lg_age_bin_edges_jax(ssp_lg_age_gyr: jnp.ndarray) -> jnp.ndarray:
    """Return log10(age/Gyr) bin edges around sorted SSP age centers."""
    ages = jnp.asarray(ssp_lg_age_gyr, dtype=jnp.float32)
    mid = 0.5 * (ages[1:] + ages[:-1])
    first = ages[0] - 0.5 * (ages[1] - ages[0])
    last = ages[-1] + 0.5 * (ages[-1] - ages[-2])
    return jnp.concatenate(
        [
            jnp.asarray([first], dtype=jnp.float32),
            mid,
            jnp.asarray([last], dtype=jnp.float32),
        ]
    )


def build_diffstar_sfh_table_jax(
    gal_t_table: jnp.ndarray,
    t_obs: jnp.ndarray,
    params: dict[str, Any],
) -> jnp.ndarray:
    """Evaluate a Diffstar SFH table using default Diffmah MAH parameters."""
    (
        calc_sfh_singlegal,
        default_diffstar_params,
        diffstar_params_cls,
        default_mah_params,
        fb,
    ) = _import_diffstar_api()
    diffstar_params = diffstar_params_cls(
        _jax_param(params, "diffstar_lgmcrit", default_diffstar_params.lgmcrit),
        _jax_param(
            params, "diffstar_lgy_at_mcrit", default_diffstar_params.lgy_at_mcrit
        ),
        _jax_param(params, "diffstar_indx_lo", default_diffstar_params.indx_lo),
        _jax_param(
            params,
            "diffstar_indx_hi",
            DIFFSTAR_FIXED_PARAMETER_DEFAULTS["diffstar_indx_hi"],
        ),
        _jax_param(params, "diffstar_lg_qt", default_diffstar_params.lg_qt),
        _jax_param(
            params,
            "diffstar_qlglgdt",
            DIFFSTAR_FIXED_PARAMETER_DEFAULTS["diffstar_qlglgdt"],
        ),
        _jax_param(params, "diffstar_lg_drop", default_diffstar_params.lg_drop),
        _jax_param(params, "diffstar_lg_rejuv", default_diffstar_params.lg_rejuv),
    )
    sfh = calc_sfh_singlegal(
        diffstar_params,
        default_mah_params,
        jnp.asarray(gal_t_table, dtype=jnp.float32),
        lgt0=jnp.log10(jnp.maximum(t_obs, 1.0e-6)),
        fb=fb,
    )
    return jnp.nan_to_num(jnp.clip(sfh, 1.0e-14, jnp.inf), nan=1.0e-14)


def build_diffsky_basic_sfh_table_jax(
    gal_t_table: jnp.ndarray,
    t_obs: jnp.ndarray,
    params: dict[str, Any],
) -> jnp.ndarray:
    """Evaluate Diffstar using per-object Diffstar and Diffmah parameters."""
    (
        calc_sfh_singlegal,
        default_diffstar_params,
        diffstar_params_cls,
        default_mah_params,
        fb,
    ) = _import_diffstar_api()
    diffstar_params = diffstar_params_cls(
        _jax_param(params, "diffstar_lgmcrit", default_diffstar_params.lgmcrit),
        _jax_param(
            params, "diffstar_lgy_at_mcrit", default_diffstar_params.lgy_at_mcrit
        ),
        _jax_param(params, "diffstar_indx_lo", default_diffstar_params.indx_lo),
        _jax_param(params, "diffstar_indx_hi", default_diffstar_params.indx_hi),
        _jax_param(params, "diffstar_lg_qt", default_diffstar_params.lg_qt),
        _jax_param(params, "diffstar_qlglgdt", default_diffstar_params.qlglgdt),
        _jax_param(params, "diffstar_lg_drop", default_diffstar_params.lg_drop),
        _jax_param(params, "diffstar_lg_rejuv", default_diffstar_params.lg_rejuv),
    )
    mah_params = type(default_mah_params)(
        _jax_param(params, "diffmah_logm0", default_mah_params.logm0),
        _jax_param(params, "diffmah_logtc", default_mah_params.logtc),
        _jax_param(params, "diffmah_early_index", default_mah_params.early_index),
        _jax_param(params, "diffmah_late_index", default_mah_params.late_index),
        _jax_param(params, "diffmah_t_peak", default_mah_params.t_peak),
    )
    sfh = calc_sfh_singlegal(
        diffstar_params,
        mah_params,
        jnp.asarray(gal_t_table, dtype=jnp.float32),
        lgt0=jnp.log10(jnp.maximum(t_obs, 1.0e-6)),
        fb=fb,
    )
    return jnp.nan_to_num(jnp.clip(sfh, 1.0e-14, jnp.inf), nan=1.0e-14)


def diffsky_basic_dust_params_jax(
    params: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Map HLTDS dust latents to the existing DSPS attenuation kernel.

    HLTDS exposes ``av`` and ``delta``. The current DSPS path expects optical
    depth at V band, a slope-like dust index, and a birth-cloud ratio. We use
    A_V = 1.086 * tau_V and set the birth-cloud component to zero by default,
    because no object-level birth-cloud latent is present in the prepared HLTDS
    table.
    """
    tau2 = jnp.asarray(params.get("dust_av", 0.0), dtype=jnp.float32) / jnp.asarray(
        1.086, dtype=jnp.float32
    )
    dust_index_n = jnp.asarray(params.get("dust_delta", 0.0), dtype=jnp.float32)
    tau1_over_tau2 = jnp.asarray(
        params.get("tau1_over_tau2", 0.0), dtype=jnp.float32
    )
    return (
        jnp.maximum(tau2, 0.0),
        dust_index_n,
        jnp.maximum(tau1_over_tau2, 0.0),
    )


def _import_diffstar_api():
    try:
        from diffmah.defaults import DEFAULT_MAH_PARAMS
        from diffstar import calc_sfh_singlegal
        from diffstar.defaults import DEFAULT_DIFFSTAR_PARAMS, FB, DiffstarParams
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "model.sfh_model='diffstar_reduced6' requires optional packages "
            "diffstar and diffmah. Install with "
            "`python -m pip install 'euclid-dsps-shine[diffstar]'`, or in this "
            "project's conda workflow with "
            "`conda run -n shine python -m pip install diffstar diffmah`."
        ) from exc
    return (
        calc_sfh_singlegal,
        DEFAULT_DIFFSTAR_PARAMS,
        DiffstarParams,
        DEFAULT_MAH_PARAMS,
        FB,
    )


def _jax_param(params: dict[str, Any], name: str, default: float) -> jnp.ndarray:
    return jnp.asarray(params.get(name, default), dtype=jnp.float32)


def project_sfh_to_popcosmos_sfr_bins_jax(
    gal_t_table: jnp.ndarray, gal_sfr_table: jnp.ndarray, t_obs: jnp.ndarray
) -> jnp.ndarray:
    """Project any SFH table into PopCosmos lookback-bin average SFRs."""
    t_table = jnp.asarray(gal_t_table, dtype=jnp.float32)
    sfr_table = jnp.asarray(gal_sfr_table, dtype=jnp.float32)
    dt = jnp.diff(t_table)
    dm = 0.5 * (sfr_table[1:] + sfr_table[:-1]) * dt
    cumulative = jnp.concatenate(
        [jnp.zeros(1, dtype=sfr_table.dtype), jnp.cumsum(dm)]
    )
    lookback_edges = build_popcosmos_lookback_bin_edges_jax(t_obs)
    t_high = jnp.asarray(t_obs, dtype=jnp.float32) - lookback_edges[:-1]
    t_low = jnp.asarray(t_obs, dtype=jnp.float32) - lookback_edges[1:]
    t_min = t_table[0]
    t_max = t_table[-1]
    t_low = jnp.clip(t_low, t_min, t_max)
    t_high = jnp.clip(t_high, t_min, t_max)
    mass = jnp.interp(t_high, t_table, cumulative) - jnp.interp(
        t_low, t_table, cumulative
    )
    width = jnp.maximum(t_high - t_low, 1.0e-6)
    return jnp.clip(mass / width, 1.0e-30, jnp.inf)


def project_sfh_to_popcosmos_dlogsfr_jax(
    gal_t_table: jnp.ndarray, gal_sfr_table: jnp.ndarray, t_obs: jnp.ndarray
) -> jnp.ndarray:
    """Project any SFH to six adjacent PopCosmos log-SFR ratios."""
    sfr_bins = project_sfh_to_popcosmos_sfr_bins_jax(
        gal_t_table, gal_sfr_table, t_obs
    )
    log_sfr = jnp.log10(jnp.maximum(sfr_bins, 1.0e-30))
    return log_sfr[:-1] - log_sfr[1:]


def normalize_sfh_to_stellar_mass_jax(
    gal_t_table: jnp.ndarray,
    gal_sfr_table: jnp.ndarray,
    ssp_lg_age_gyr: jnp.ndarray,
    t_obs: jnp.ndarray,
    log10_stellar_mass: jnp.ndarray,
    frac_surviving_by_age: jnp.ndarray | None = None,
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
    if frac_surviving_by_age is None:
        frac_surviving_by_age = surviving_mstar(ssp_lg_age_gyr + 9.0)
    else:
        frac_surviving_by_age = jnp.asarray(frac_surviving_by_age, dtype=jnp.float32)
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


def normalize_sfh_to_stellar_mass_with_age_weights_jax(
    gal_t_table: jnp.ndarray,
    gal_sfr_table: jnp.ndarray,
    ssp_lg_age_gyr: jnp.ndarray,
    age_weights: jnp.ndarray,
    log10_stellar_mass: jnp.ndarray,
    frac_surviving_by_age: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scale an SFH using precomputed SSP age weights."""
    from dsps.imf.surviving_mstar import surviving_mstar

    weights = jnp.asarray(age_weights, dtype=jnp.float32)
    weights = weights / jnp.maximum(jnp.sum(weights), 1.0e-30)
    if frac_surviving_by_age is None:
        frac_surviving_by_age = surviving_mstar(
            jnp.asarray(ssp_lg_age_gyr, dtype=jnp.float32) + 9.0
        )
    else:
        frac_surviving_by_age = jnp.asarray(frac_surviving_by_age, dtype=jnp.float32)
    mean_frac_surviving = jnp.sum(weights * frac_surviving_by_age)
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


def interpolate_context_ssp_stellar_metallicity_jax(
    context: DspsContext, lgmet_abs: jnp.ndarray
) -> jnp.ndarray:
    """Interpolate the base SSP, using compressed coefficients when configured."""
    if (
        context.compressed_ssp_basis_jax is None
        or context.compressed_ssp_coeff_jax is None
        or context.compressed_ssp_scale_jax is None
    ):
        return interpolate_ssp_stellar_metallicity_jax(
            _context_ssp_lgmet(context),
            _context_ssp_flux(context),
            lgmet_abs,
        )
    coeff = context.compressed_ssp_coeff_jax
    scale = context.compressed_ssp_scale_jax
    coeff_z = _interp_axis0_linear(
        _context_ssp_lgmet(context),
        coeff,
        jnp.asarray(lgmet_abs, dtype=jnp.float32),
    )
    scale_z = _interp_axis0_linear(
        _context_ssp_lgmet(context),
        scale,
        jnp.asarray(lgmet_abs, dtype=jnp.float32),
    )
    weighted_coeff = jnp.asarray(coeff_z, dtype=jnp.float32) * jnp.asarray(
        scale_z, dtype=jnp.float32
    )[:, None]
    basis = jnp.asarray(context.compressed_ssp_basis_jax, dtype=jnp.float32)
    reconstructed = jnp.einsum("ak,kw->aw", weighted_coeff, basis)
    return jnp.nan_to_num(reconstructed, nan=0.0, posinf=1.0e30, neginf=-1.0e30)


def lognormal_mdf_lgmet_weights_jax(
    ssp_lgmet: jnp.ndarray,
    lgmet_abs_median: jnp.ndarray,
    scatter_dex: jnp.ndarray,
) -> jnp.ndarray:
    """Return DSPS lognormal-MDF weights over the SSP metallicity grid."""
    from dsps.sed.metallicity_weights import calc_lgmet_weights_from_lognormal_mdf

    weights = calc_lgmet_weights_from_lognormal_mdf(
        jnp.asarray(lgmet_abs_median, dtype=jnp.float32),
        jnp.maximum(jnp.asarray(scatter_dex, dtype=jnp.float32), 1.0e-6),
        jnp.asarray(ssp_lgmet, dtype=jnp.float32),
    )
    weights = jnp.clip(jnp.asarray(weights, dtype=jnp.float32), 0.0, jnp.inf)
    return weights / jnp.maximum(jnp.sum(weights), 1.0e-30)


def context_ssp_lognormal_mdf_jax(
    context: DspsContext,
    lgmet_abs_median: jnp.ndarray,
    scatter_dex: jnp.ndarray,
) -> jnp.ndarray:
    """Return age-by-wave SSP flux integrated over a lognormal metallicity MDF."""
    weights = lognormal_mdf_lgmet_weights_jax(
        _context_ssp_lgmet(context), lgmet_abs_median, scatter_dex
    )
    if (
        context.compressed_ssp_basis_jax is None
        or context.compressed_ssp_coeff_jax is None
        or context.compressed_ssp_scale_jax is None
    ):
        flux = jnp.asarray(_context_ssp_flux(context), dtype=jnp.float32)
        return jnp.nan_to_num(
            jnp.einsum("m,maw->aw", weights, flux),
            nan=0.0,
            posinf=1.0e30,
            neginf=-1.0e30,
        )
    coeff = jnp.asarray(context.compressed_ssp_coeff_jax, dtype=jnp.float32)
    scale = jnp.asarray(context.compressed_ssp_scale_jax, dtype=jnp.float32)
    weighted_coeff = coeff * scale[..., None]
    coeff_mdf = jnp.einsum("m,mak->ak", weights, weighted_coeff)
    basis = jnp.asarray(context.compressed_ssp_basis_jax, dtype=jnp.float32)
    reconstructed = jnp.einsum("ak,kw->aw", coeff_mdf, basis)
    return jnp.nan_to_num(reconstructed, nan=0.0, posinf=1.0e30, neginf=-1.0e30)


def diffsky_basic_ssp_flux_by_age_jax(
    context: DspsContext,
    model_config: dict[str, Any],
    lgmet_abs: jnp.ndarray,
) -> jnp.ndarray:
    """Return Diffsky-basic SSP flux by age for the configured metallicity model."""
    metallicity_model = str(model_config.get("stellar_metallicity_model", "single"))
    if metallicity_model == "single":
        return interpolate_context_ssp_stellar_metallicity_jax(context, lgmet_abs)
    if metallicity_model == "lognormal_mdf_fixed_scatter":
        return context_ssp_lognormal_mdf_jax(
            context,
            lgmet_abs,
            jnp.asarray(model_config.get("stellar_metallicity_scatter_dex", 0.2)),
        )
    raise ValueError(
        "sfh_model='diffsky_basic' supports model.stellar_metallicity_model "
        "'single' or 'lognormal_mdf_fixed_scatter'."
    )


def _diffsky_basic_surviving_mstar_by_age_jax(
    context: DspsContext,
    model_config: dict[str, Any],
    lgmet_abs: jnp.ndarray,
) -> jnp.ndarray | None:
    """Return survival fractions by age for single-metallicity or MDF modes."""
    grid = _context_ssp_surviving_mstar(context)
    if grid is None:
        return None
    metallicity_model = str(model_config.get("stellar_metallicity_model", "single"))
    if metallicity_model == "single":
        return _interp_axis0_linear(
            _context_ssp_lgmet(context),
            grid,
            jnp.asarray(lgmet_abs, dtype=jnp.float32),
        )
    if metallicity_model == "lognormal_mdf_fixed_scatter":
        weights = lognormal_mdf_lgmet_weights_jax(
            _context_ssp_lgmet(context),
            lgmet_abs,
            jnp.asarray(model_config.get("stellar_metallicity_scatter_dex", 0.2)),
        )
        return jnp.nan_to_num(
            jnp.einsum("m,ma->a", weights, jnp.asarray(grid, dtype=jnp.float32)),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
    raise ValueError(
        "sfh_model='diffsky_basic' supports model.stellar_metallicity_model "
        "'single' or 'lognormal_mdf_fixed_scatter'."
    )


def _validate_diffsky_basic_metallicity_model(model_config: dict[str, Any]) -> None:
    metallicity_model = str(model_config.get("stellar_metallicity_model", "single"))
    if metallicity_model == "single":
        return
    if metallicity_model == "lognormal_mdf_fixed_scatter":
        scatter = float(model_config.get("stellar_metallicity_scatter_dex", 0.2))
        if not np.isfinite(scatter) or scatter <= 0.0:
            raise ValueError(
                "model.stellar_metallicity_scatter_dex must be positive for "
                "stellar_metallicity_model='lognormal_mdf_fixed_scatter'."
            )
        return
    raise ValueError(
        "sfh_model='diffsky_basic' supports model.stellar_metallicity_model "
        "'single' or 'lognormal_mdf_fixed_scatter'."
    )


def interpolate_popcosmos_ssp_stellar_metallicity_jax(
    context: DspsContext,
    params: dict[str, Any],
    model_config: dict[str, Any],
    lgmet_abs: jnp.ndarray,
) -> jnp.ndarray:
    """Return the SSP/gas grid at one stellar metallicity for PopCosmos paths."""
    nebular_model = str(model_config.get("nebular_model", "fixed_ssp"))
    if nebular_model == "compressed_gas_grid":
        return interpolate_compressed_gas_ssp_stellar_metallicity_jax(
            context,
            params["log10_gas_metallicity"],
            params["log10_gas_ionization"],
            lgmet_abs,
        )
    if nebular_model == "gas_grid":
        ssp_flux_grid = interpolate_gas_ssp_grid_jax(
            context,
            params["log10_gas_metallicity"],
            params["log10_gas_ionization"],
        )
        return interpolate_ssp_stellar_metallicity_jax(
            _context_ssp_lgmet(context),
            ssp_flux_grid,
            lgmet_abs,
        )
    return interpolate_context_ssp_stellar_metallicity_jax(context, lgmet_abs)


def log10_stellar_metallicity_to_absolute_jax(
    log10_stellar_metallicity: jnp.ndarray, z_sun: float
) -> jnp.ndarray:
    """Convert log10(Zstar/Zsun) to absolute log10(Zstar)."""
    return jnp.log10(jnp.asarray(z_sun, dtype=jnp.float32)) + jnp.asarray(
        log10_stellar_metallicity, dtype=jnp.float32
    )


def apply_popcosmos_dust_by_age_jax(
    wave: jnp.ndarray,
    ssp_lg_age_gyr: jnp.ndarray,
    sed_by_age: jnp.ndarray,
    tau2: jnp.ndarray,
    dust_index_n: jnp.ndarray,
    tau1_over_tau2: jnp.ndarray,
    model_config: dict[str, Any] | None,
) -> jnp.ndarray:
    """Apply the configured PopCosmos-like age-dependent attenuation model."""
    config = _normalized_model_config(model_config)
    mode = str(config.get("dust_model", "charlot_fall_powerlaw"))
    if mode == "charlot_fall_powerlaw":
        return apply_charlot_fall_by_age_jax(
            wave,
            ssp_lg_age_gyr,
            sed_by_age,
            tau2,
            dust_index_n,
            tau1_over_tau2,
            birth_cloud_slope=float(config.get("birth_cloud_slope", -1.0)),
        )
    if mode == "prospector_fsps":
        return apply_prospector_fsps_dust_by_age_jax(
            wave,
            ssp_lg_age_gyr,
            sed_by_age,
            tau2,
            dust_index_n,
            tau1_over_tau2,
            dust_tesc_logyr=float(config.get("dust_tesc_logyr", 7.0)),
            dust1_index=float(config.get("dust1_index", -1.0)),
        )
    raise ValueError(f"Unsupported model.dust_model: {mode}")


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


def apply_prospector_fsps_dust_by_age_jax(
    wave: jnp.ndarray,
    ssp_lg_age_gyr: jnp.ndarray,
    sed_by_age: jnp.ndarray,
    tau2: jnp.ndarray,
    dust_index_n: jnp.ndarray,
    tau1_over_tau2: jnp.ndarray,
    dust_tesc_logyr: float = 7.0,
    dust1_index: float = -1.0,
) -> jnp.ndarray:
    """Approximate Prospector/FSPS dust_type=4 plus birth-cloud attenuation.

    The diffuse component follows a Calzetti/Kriek-Conroy-like curve via DSPS'
    Noll+09 kernel, normalized so ``tau2`` is the V-band optical depth. The
    birth-cloud component is a V-band optical depth with FSPS-style power-law
    index ``dust1_index`` and applies only below ``dust_tesc_logyr``.
    """
    wave_safe = jnp.maximum(jnp.asarray(wave, dtype=jnp.float32), 1.0)
    tau2_safe = jnp.maximum(jnp.asarray(tau2, dtype=jnp.float32), 0.0)
    tau1 = jnp.maximum(jnp.asarray(tau1_over_tau2, dtype=jnp.float32), 0.0) * tau2_safe
    diffuse_shape = _prospector_fsps_diffuse_shape_jax(wave_safe, dust_index_n)
    diffuse = tau2_safe * diffuse_shape
    birth = tau1 * (wave_safe / 5500.0) ** jnp.asarray(
        dust1_index, dtype=jnp.float32
    )
    age_logyr = jnp.asarray(ssp_lg_age_gyr, dtype=jnp.float32) + 9.0
    young = age_logyr <= jnp.asarray(dust_tesc_logyr, dtype=jnp.float32)
    old_trans = jnp.exp(-jnp.clip(diffuse, 0.0, 80.0))
    young_trans = jnp.exp(-jnp.clip(diffuse + birth, 0.0, 80.0))
    transmission = jnp.where(young[:, None], young_trans[None, :], old_trans[None, :])
    return jnp.asarray(sed_by_age, dtype=jnp.float32) * transmission


def _prospector_fsps_diffuse_shape_jax(
    wave_angstrom: jnp.ndarray, dust_index_n: jnp.ndarray
) -> jnp.ndarray:
    wave = jnp.maximum(jnp.asarray(wave_angstrom, dtype=jnp.float32), 1.0)
    slope = jnp.asarray(dust_index_n, dtype=jnp.float32)
    calzetti = _fsps_calzetti_tau_shape_jax(wave)
    eb = 0.85 - 1.9 * slope
    lamuvb = jnp.asarray(2175.0, dtype=jnp.float32)
    dlam = jnp.asarray(350.0, dtype=jnp.float32)
    drude = eb * (wave * dlam) ** 2 / (
        (wave**2 - lamuvb**2) ** 2 + (wave * dlam) ** 2
    )
    shape = (calzetti + drude / 4.05) * (wave / 5500.0) ** slope
    return jnp.clip(jnp.nan_to_num(shape, nan=0.0, posinf=1.0e6, neginf=0.0), 0.0, 1.0e6)


def _fsps_calzetti_tau_shape_jax(wave_angstrom: jnp.ndarray) -> jnp.ndarray:
    """FSPS Calzetti optical-depth shape used inside dust_type=4."""
    wave = jnp.maximum(jnp.asarray(wave_angstrom, dtype=jnp.float32), 1.0)
    inv_micron = 10_000.0 / wave
    red = 1.17 * (-1.857 + 1.04 * inv_micron) + 1.78
    blue = 1.17 * (
        -2.156
        + 1.509 * inv_micron
        - 0.198 * inv_micron**2
        + 0.011 * inv_micron**3
    ) + 1.78
    cal00 = jnp.where(wave > 6300.0, red, blue) / (0.44 * 4.05)
    return jnp.maximum(cal00, 0.0)


def apply_igm_transmission_jax(
    wave_rest: jnp.ndarray,
    rest_sed: jnp.ndarray,
    z_obs: jnp.ndarray,
    model_config: dict[str, Any] | None,
) -> jnp.ndarray:
    """Apply the configured IGM transmission model."""
    mode = str(_normalized_model_config(model_config).get("igm_model", "none"))
    if mode == "none":
        return rest_sed
    if mode == "madau95_approx":
        wave = jnp.maximum(jnp.asarray(wave_rest, dtype=jnp.float32), 1.0)
        z = jnp.maximum(jnp.asarray(z_obs, dtype=jnp.float32), 0.0)
        below_lya = jnp.clip((1216.0 - wave) / 1216.0, 0.0, 1.0)
        below_limit = jnp.clip((912.0 - wave) / 912.0, 0.0, 1.0)
        tau_forest = 0.35 * z**1.6 * below_lya**1.2 * (1216.0 / wave) ** 0.7
        tau_continuum = 1.8 * z**2.0 * below_limit**1.5 * (912.0 / wave) ** 2.0
        transmission = jnp.exp(-jnp.clip(tau_forest + tau_continuum, 0.0, 80.0))
        return jnp.asarray(rest_sed, dtype=jnp.float32) * transmission
    if mode == "fsps_madau95":
        return jnp.asarray(rest_sed, dtype=jnp.float32) * fsps_madau95_igm_transmission_jax(
            wave_rest, z_obs
        )
    raise ValueError(f"Unsupported model.igm_model: {mode}")


def fsps_madau95_igm_transmission_jax(
    wave_rest: jnp.ndarray, z_obs: jnp.ndarray, factor: float = 1.0
) -> jnp.ndarray:
    """JAX port of FSPS ``igm_absorb.f90`` Madau95 transmission."""
    wave = jnp.maximum(jnp.asarray(wave_rest, dtype=jnp.float32), 1.0)
    z1 = 1.0 + jnp.maximum(jnp.asarray(z_obs, dtype=jnp.float32), 0.0)
    lobs = wave * z1
    lylim = jnp.asarray(911.75, dtype=jnp.float32)
    lyw = jnp.asarray(
        [
            1215.67,
            1025.72,
            972.537,
            949.743,
            937.803,
            930.748,
            926.226,
            923.150,
            920.963,
            919.352,
            918.129,
            917.181,
            916.429,
            915.824,
            915.329,
            914.919,
            914.576,
        ],
        dtype=jnp.float32,
    )
    lycoeff = jnp.asarray(
        [
            0.0036,
            0.0017,
            0.0011846,
            0.0009410,
            0.0007960,
            0.0006967,
            0.0006236,
            0.0005665,
            0.0005200,
            0.0004817,
            0.0004487,
            0.0004200,
            0.0003947,
            0.000372,
            0.000352,
            0.0003334,
            0.00031644,
        ],
        dtype=jnp.float32,
    )
    tau = jnp.zeros_like(wave)
    for index in range(17):
        line_wave = lyw[index]
        mask = wave <= line_wave
        term = lycoeff[index] * (lobs / line_wave) ** 3.46
        tau = tau + jnp.where(mask, term, 0.0)
        if index == 0:
            tau = tau + jnp.where(mask, 0.0017 * (lobs / line_wave) ** 1.68, 0.0)

    xc = lobs / lylim
    lyman_limit_mask = wave <= lylim
    tau_lyc = (
        0.25 * xc**3 * (z1**0.46 - xc**0.46)
        + 9.4 * xc**1.5 * (z1**0.18 - xc**0.18)
        - 0.7 * xc**3 * (xc ** (-1.32) - z1 ** (-1.32))
        - 0.023 * (z1**1.68 - xc**1.68)
    )
    tau = tau + jnp.where(lyman_limit_mask, tau_lyc, 0.0)
    max_index = jnp.argmax(tau)
    tau_max = tau[max_index]
    tau = jnp.where(jnp.arange(wave.shape[0]) <= max_index, tau_max, tau)
    return jnp.exp(-jnp.clip(tau * jnp.asarray(factor, dtype=jnp.float32), 0.0, 80.0))


def interpolate_gas_ssp_grid_jax(
    context: DspsContext,
    log10_gas_metallicity: jnp.ndarray,
    log10_gas_ionization: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate a gas SSP grid to gas metallicity and ionization."""
    if context.ssp_flux_gas_grid_jax is None:
        return interpolate_compressed_gas_ssp_grid_jax(
            context,
            log10_gas_metallicity,
            log10_gas_ionization,
        )
    if (
        context.gas_lgmet_grid_jax is None
        or context.gas_lgu_grid_jax is None
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


def interpolate_compressed_gas_ssp_grid_jax(
    context: DspsContext,
    log10_gas_metallicity: jnp.ndarray,
    log10_gas_ionization: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate a low-rank gas SSP grid and reconstruct only the selected grid."""
    if (
        context.gas_lgmet_grid_jax is None
        or context.gas_lgu_grid_jax is None
        or context.compressed_gas_basis_jax is None
        or context.compressed_gas_coeff_jax is None
        or context.compressed_gas_scale_jax is None
    ):
        raise ValueError(
            "model.nebular_model='compressed_gas_grid' requires a loaded "
            "compressed gas grid"
        )
    gas_z_lo, gas_z_hi, gas_z_weight = _interp_bracket(
        context.gas_lgmet_grid_jax,
        log10_gas_metallicity,
    )
    gas_u_lo, gas_u_hi, gas_u_weight = _interp_bracket(
        context.gas_lgu_grid_jax,
        log10_gas_ionization,
    )
    coeff = jnp.asarray(context.compressed_gas_coeff_jax)
    scale = jnp.asarray(context.compressed_gas_scale_jax, dtype=jnp.float32)
    basis = jnp.asarray(context.compressed_gas_basis_jax)

    wc00 = jnp.asarray(coeff[gas_z_lo, gas_u_lo], dtype=jnp.float32) * scale[
        gas_z_lo, gas_u_lo
    ][..., None]
    wc01 = jnp.asarray(coeff[gas_z_lo, gas_u_hi], dtype=jnp.float32) * scale[
        gas_z_lo, gas_u_hi
    ][..., None]
    wc10 = jnp.asarray(coeff[gas_z_hi, gas_u_lo], dtype=jnp.float32) * scale[
        gas_z_hi, gas_u_lo
    ][..., None]
    wc11 = jnp.asarray(coeff[gas_z_hi, gas_u_hi], dtype=jnp.float32) * scale[
        gas_z_hi, gas_u_hi
    ][..., None]
    low_u = wc00 * (1.0 - gas_u_weight) + wc01 * gas_u_weight
    high_u = wc10 * (1.0 - gas_u_weight) + wc11 * gas_u_weight
    weighted_coeff = low_u * (1.0 - gas_z_weight) + high_u * gas_z_weight
    reconstructed = jnp.einsum(
        "mak,kw->maw", weighted_coeff, jnp.asarray(basis, dtype=jnp.float32)
    )
    return jnp.nan_to_num(
        reconstructed,
        nan=0.0,
        posinf=1.0e30,
        neginf=-1.0e30,
    )


def interpolate_compressed_gas_ssp_stellar_metallicity_jax(
    context: DspsContext,
    log10_gas_metallicity: jnp.ndarray,
    log10_gas_ionization: jnp.ndarray,
    lgmet_abs: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate compressed gas and stellar-metallicity axes before expansion.

    This is algebraically equivalent to reconstructing the gas-selected
    ``[stellar_lgmet, age, wave]`` grid and then interpolating the stellar
    metallicity axis, but it avoids materializing that extra metallicity axis in
    vmapped galaxy batches.
    """
    if (
        context.gas_lgmet_grid_jax is None
        or context.gas_lgu_grid_jax is None
        or context.compressed_gas_basis_jax is None
        or context.compressed_gas_coeff_jax is None
        or context.compressed_gas_scale_jax is None
    ):
        raise ValueError(
            "model.nebular_model='compressed_gas_grid' requires a loaded "
            "compressed gas grid"
        )
    gas_z_lo, gas_z_hi, gas_z_weight = _interp_bracket(
        context.gas_lgmet_grid_jax,
        log10_gas_metallicity,
    )
    gas_u_lo, gas_u_hi, gas_u_weight = _interp_bracket(
        context.gas_lgu_grid_jax,
        log10_gas_ionization,
    )
    coeff = jnp.asarray(context.compressed_gas_coeff_jax)
    scale = jnp.asarray(context.compressed_gas_scale_jax, dtype=jnp.float32)

    wc00 = jnp.asarray(coeff[gas_z_lo, gas_u_lo], dtype=jnp.float32) * scale[
        gas_z_lo, gas_u_lo
    ][..., None]
    wc01 = jnp.asarray(coeff[gas_z_lo, gas_u_hi], dtype=jnp.float32) * scale[
        gas_z_lo, gas_u_hi
    ][..., None]
    wc10 = jnp.asarray(coeff[gas_z_hi, gas_u_lo], dtype=jnp.float32) * scale[
        gas_z_hi, gas_u_lo
    ][..., None]
    wc11 = jnp.asarray(coeff[gas_z_hi, gas_u_hi], dtype=jnp.float32) * scale[
        gas_z_hi, gas_u_hi
    ][..., None]
    low_u = wc00 * (1.0 - gas_u_weight) + wc01 * gas_u_weight
    high_u = wc10 * (1.0 - gas_u_weight) + wc11 * gas_u_weight
    weighted_coeff = low_u * (1.0 - gas_z_weight) + high_u * gas_z_weight
    coeff_z = _interp_axis0_linear(
        _context_ssp_lgmet(context),
        weighted_coeff,
        jnp.asarray(lgmet_abs, dtype=jnp.float32),
    )
    reconstructed = jnp.einsum(
        "ak,kw->aw",
        coeff_z,
        jnp.asarray(context.compressed_gas_basis_jax, dtype=jnp.float32),
    )
    return jnp.nan_to_num(
        reconstructed,
        nan=0.0,
        posinf=1.0e30,
        neginf=-1.0e30,
    )


def gas_metallicity_constraint_penalty_jax(
    params: dict[str, Any],
    model_config: dict[str, Any] | None,
    penalty: float = 1.0e30,
) -> jnp.ndarray:
    """Hard PopCosmos-like constraint enforcing log10(Zgas/Zsun) >= log10(Zstar/Zsun)."""
    config = _normalized_model_config(model_config)
    if (
        str(config.get("nebular_model", "fixed_ssp"))
        not in {"gas_grid", "compressed_gas_grid"}
        or not _is_popcosmos_like_model_config(config)
    ):
        return jnp.asarray(0.0, dtype=jnp.float32)
    gas = jnp.asarray(params["log10_gas_metallicity"], dtype=jnp.float32)
    stellar = jnp.asarray(params["log10_stellar_metallicity"], dtype=jnp.float32)
    valid = gas >= stellar
    return jnp.where(
        valid,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(penalty, dtype=jnp.float32),
    )


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
    agn_lnu = agn_component_jax(
        context,
        wave,
        intrinsic_stellar_sed,
        params,
        model_config,
    )
    return jnp.asarray(dusted_sed, dtype=jnp.float32) + agn_lnu


def agn_component_jax(
    context: DspsContext,
    wave: jnp.ndarray,
    intrinsic_stellar_sed: jnp.ndarray,
    params: dict[str, Any],
    model_config: dict[str, Any] | None,
    *,
    age_weights: jnp.ndarray | None = None,
    formed_mass: jnp.ndarray | float | None = None,
    stellar_lgmet_abs: jnp.ndarray | float | None = None,
    template_tage_gyr: jnp.ndarray | float | None = None,
    stellar_logzsol: jnp.ndarray | float | None = None,
) -> jnp.ndarray:
    """Return the DSPS AGN rest-frame Lnu component before IGM attenuation."""
    config = _normalized_model_config(model_config)
    wave = jnp.asarray(wave, dtype=jnp.float32)
    agn_model = str(config.get("agn_model", "none"))
    if agn_model == "none":
        return jnp.zeros_like(wave, dtype=jnp.float32)
    if agn_model == "fsps_component_grid":
        agn_lnu = agn_component_from_ssp_grid_jax(
            context,
            wave,
            params,
            config,
            age_weights=age_weights,
            formed_mass=formed_mass,
            stellar_lgmet_abs=stellar_lgmet_abs,
        )
        return jnp.nan_to_num(agn_lnu, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
    if agn_model == "compressed_fsps_component_grid":
        agn_lnu = agn_component_from_compressed_grid_jax(
            context,
            wave,
            params,
            config,
            age_weights=age_weights,
            formed_mass=formed_mass,
            stellar_lgmet_abs=stellar_lgmet_abs,
        )
        return jnp.nan_to_num(agn_lnu, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
    if agn_model != "template_grid":
        raise ValueError(f"Unsupported model.agn_model: {agn_model}")
    if (
        context.agn_wave_jax is None
        or context.agn_tau_grid_jax is None
        or context.agn_template_grid_jax is None
    ):
        raise ValueError("model.agn_model='template_grid' requires a loaded AGN grid")
    tauagn = jnp.exp(jnp.asarray(params["ln_tauagn"], dtype=jnp.float32))
    fagn = jnp.exp(jnp.asarray(params["ln_fagn"], dtype=jnp.float32))
    template_tau = _interpolate_agn_template_jax(
        context,
        fagn,
        tauagn,
        (
            jnp.asarray(template_tage_gyr, dtype=jnp.float32)
            if template_tage_gyr is not None
            else jnp.asarray(config.get("agn_template_tage_gyr", 1.0), dtype=jnp.float32)
        ),
        (
            jnp.asarray(stellar_logzsol, dtype=jnp.float32)
            if stellar_logzsol is not None
            else jnp.asarray(
                config.get("agn_template_stellar_logzsol", 0.0),
                dtype=jnp.float32,
            )
        ),
    )
    template = jnp.interp(
        wave,
        context.agn_wave_jax,
        template_tau,
        left=0.0,
        right=0.0,
    )
    c_angstrom_per_s = 2.99792458e18
    wave_safe = jnp.maximum(wave, 1.0)
    lbol_stellar = jnp.trapezoid(
        jnp.maximum(intrinsic_stellar_sed, 0.0) * c_angstrom_per_s / wave_safe**2,
        wave_safe,
    )
    agn_lnu = fagn * jnp.maximum(lbol_stellar, 0.0) * template
    agn_lnu = apply_agn_host_attenuation_jax(wave, agn_lnu, params, config)
    return jnp.nan_to_num(agn_lnu, nan=0.0, posinf=1.0e30, neginf=-1.0e30)


def agn_component_from_ssp_grid_jax(
    context: DspsContext,
    wave: jnp.ndarray,
    params: dict[str, Any],
    model_config: dict[str, Any],
    *,
    age_weights: jnp.ndarray | None,
    formed_mass: jnp.ndarray | float | None,
    stellar_lgmet_abs: jnp.ndarray | float | None,
) -> jnp.ndarray:
    """Convolve an FSPS-native AGN component SSP grid with the SFH weights."""
    if (
        context.agn_wave_jax is None
        or context.agn_fagn_grid_jax is None
        or context.agn_tau_grid_jax is None
        or context.agn_component_lgmet_jax is None
        or context.agn_component_lg_age_gyr_jax is None
        or context.agn_component_grid_jax is None
    ):
        raise ValueError(
            "model.agn_model='fsps_component_grid' requires a loaded AGN "
            "component grid"
        )
    if age_weights is None or formed_mass is None or stellar_lgmet_abs is None:
        raise ValueError(
            "model.agn_model='fsps_component_grid' requires age weights, formed "
            "mass, and absolute stellar metallicity from the SFH path"
        )
    fagn = jnp.exp(jnp.asarray(params["ln_fagn"], dtype=jnp.float32))
    tauagn = jnp.exp(jnp.asarray(params["ln_tauagn"], dtype=jnp.float32))
    grid = jnp.asarray(context.agn_component_grid_jax, dtype=jnp.float32)
    grid_fagn = _interp_axis0_linear(context.agn_fagn_grid_jax, grid, fagn)
    grid_tau = _interp_axis0_linear(context.agn_tau_grid_jax, grid_fagn, tauagn)
    grid_z = _interp_axis0_linear(
        context.agn_component_lgmet_jax,
        grid_tau,
        jnp.asarray(stellar_lgmet_abs, dtype=jnp.float32),
    )
    weights = jnp.asarray(age_weights, dtype=jnp.float32)[:, None]
    mass = jnp.asarray(formed_mass, dtype=jnp.float32)
    agn_native = jnp.sum(grid_z * weights * mass, axis=0)
    agn_lnu = jnp.interp(
        jnp.asarray(wave, dtype=jnp.float32),
        context.agn_wave_jax,
        agn_native,
        left=0.0,
        right=0.0,
    )
    return apply_agn_host_attenuation_jax(wave, agn_lnu, params, model_config)


def agn_component_from_compressed_grid_jax(
    context: DspsContext,
    wave: jnp.ndarray,
    params: dict[str, Any],
    model_config: dict[str, Any],
    *,
    age_weights: jnp.ndarray | None,
    formed_mass: jnp.ndarray | float | None,
    stellar_lgmet_abs: jnp.ndarray | float | None,
) -> jnp.ndarray:
    """Convolve a compressed FSPS-native AGN component grid with the SFH."""
    if (
        context.agn_wave_jax is None
        or context.agn_fagn_grid_jax is None
        or context.agn_tau_grid_jax is None
        or context.agn_component_lgmet_jax is None
        or context.agn_component_lg_age_gyr_jax is None
        or context.compressed_agn_basis_jax is None
        or context.compressed_agn_coeff_jax is None
    ):
        raise ValueError(
            "model.agn_model='compressed_fsps_component_grid' requires a "
            "loaded compressed AGN component grid"
        )
    if age_weights is None or formed_mass is None or stellar_lgmet_abs is None:
        raise ValueError(
            "model.agn_model='compressed_fsps_component_grid' requires age "
            "weights, formed mass, and absolute stellar metallicity from the "
            "SFH path"
        )
    fagn = jnp.exp(jnp.asarray(params["ln_fagn"], dtype=jnp.float32))
    tauagn = jnp.exp(jnp.asarray(params["ln_tauagn"], dtype=jnp.float32))
    coeff = jnp.asarray(context.compressed_agn_coeff_jax)
    fagn_is_factored = coeff.ndim == 4
    coeff_for_tau = (
        coeff
        if fagn_is_factored
        else _interp_axis0_linear(context.agn_fagn_grid_jax, coeff, fagn)
    )
    coeff_tau = _interp_axis0_linear(context.agn_tau_grid_jax, coeff_for_tau, tauagn)
    coeff_z = _interp_axis0_linear(
        context.agn_component_lgmet_jax,
        coeff_tau,
        jnp.asarray(stellar_lgmet_abs, dtype=jnp.float32),
    )
    if context.compressed_agn_scale_jax is None:
        scale_z = jnp.ones(coeff_z.shape[:-1], dtype=jnp.float32)
    else:
        scale = jnp.asarray(context.compressed_agn_scale_jax, dtype=jnp.float32)
        scale_for_tau = (
            scale
            if fagn_is_factored
            else _interp_axis0_linear(context.agn_fagn_grid_jax, scale, fagn)
        )
        scale_tau = _interp_axis0_linear(context.agn_tau_grid_jax, scale_for_tau, tauagn)
        scale_z = _interp_axis0_linear(
            context.agn_component_lgmet_jax,
            scale_tau,
            jnp.asarray(stellar_lgmet_abs, dtype=jnp.float32),
        )
    weights = jnp.asarray(age_weights, dtype=jnp.float32)[:, None]
    mass = jnp.asarray(formed_mass, dtype=jnp.float32)
    galaxy_coeff = jnp.sum(
        jnp.asarray(coeff_z, dtype=jnp.float32) * scale_z[:, None] * weights * mass,
        axis=0,
    )
    if fagn_is_factored:
        galaxy_coeff = galaxy_coeff * fagn
    basis = jnp.asarray(context.compressed_agn_basis_jax, dtype=jnp.float32)
    agn_native = jnp.matmul(galaxy_coeff, basis)
    agn_lnu = jnp.interp(
        jnp.asarray(wave, dtype=jnp.float32),
        context.agn_wave_jax,
        agn_native,
        left=0.0,
        right=0.0,
    )
    return apply_agn_host_attenuation_jax(wave, agn_lnu, params, model_config)


def _interpolate_agn_template_jax(
    context: DspsContext,
    fagn: jnp.ndarray,
    tauagn: jnp.ndarray,
    template_tage_gyr: jnp.ndarray,
    stellar_logzsol: jnp.ndarray,
) -> jnp.ndarray:
    template_grid = jnp.asarray(context.agn_template_grid_jax, dtype=jnp.float32)
    if template_grid.ndim == 2:
        return _interp_axis0_linear(
            context.agn_tau_grid_jax,
            template_grid,
            tauagn,
        )
    if (
        template_grid.ndim == 5
        and context.agn_fagn_grid_jax is not None
        and context.agn_tau_grid_jax is not None
        and context.agn_tage_grid_jax is not None
        and context.agn_logzsol_grid_jax is not None
    ):
        template_fagn = _interp_axis0_linear(
            context.agn_fagn_grid_jax,
            template_grid,
            fagn,
        )
        template_tau = _interp_axis0_linear(
            context.agn_tau_grid_jax,
            template_fagn,
            tauagn,
        )
        template_age = _interp_axis0_linear(
            context.agn_tage_grid_jax,
            template_tau,
            template_tage_gyr,
        )
        return _interp_axis0_linear(
            context.agn_logzsol_grid_jax,
            template_age,
            stellar_logzsol,
        )
    raise ValueError(
        "AGN template grid must be 2D legacy (agn_tau, wave) or 5D audit "
        "(fagn, agn_tau, tage_gyr, stellar_logzsol, wave)"
    )


def apply_agn_host_attenuation_jax(
    wave: jnp.ndarray,
    agn_lnu: jnp.ndarray,
    params: dict[str, Any],
    model_config: dict[str, Any] | None,
) -> jnp.ndarray:
    """Apply optional host attenuation to the AGN component for audit experiments."""
    config = _normalized_model_config(model_config)
    mode = str(config.get("agn_host_attenuation", "none"))
    if mode == "none":
        return jnp.asarray(agn_lnu, dtype=jnp.float32)
    wave_safe = jnp.maximum(jnp.asarray(wave, dtype=jnp.float32), 1.0)
    slope = jnp.asarray(params.get("dust_index_n", -0.7), dtype=jnp.float32)

    if mode == "fsps_diffuse_unit_tau":
        shape = _prospector_fsps_diffuse_shape_jax(wave_safe, slope)
        baked_mode = str(config.get("agn_baked_attenuation", "none"))
        if baked_mode == "none":
            optical_depth = shape
        elif baked_mode == "fsps_powerlaw_unit_tau":
            baked_shape = _fsps_powerlaw_unit_tau_shape_jax(
                wave_safe,
                jnp.asarray(config.get("agn_baked_dust_index", -0.7), dtype=jnp.float32),
            )
            optical_depth = shape - baked_shape
        else:
            raise ValueError(f"Unsupported model.agn_baked_attenuation: {baked_mode}")
        transmission = jnp.exp(-jnp.clip(optical_depth, -80.0, 80.0))
        return jnp.asarray(agn_lnu, dtype=jnp.float32) * transmission

    tau2 = jnp.maximum(jnp.asarray(params.get("tau2", 0.0), dtype=jnp.float32), 0.0)
    scale = jnp.maximum(
        jnp.asarray(config.get("agn_host_attenuation_scale", 1.0), dtype=jnp.float32),
        0.0,
    )
    if mode == "diffuse":
        shape = (wave_safe / 5500.0) ** slope
    elif mode == "prospector_fsps":
        shape = _prospector_fsps_diffuse_shape_jax(wave_safe, slope)
    else:
        raise ValueError(f"Unsupported model.agn_host_attenuation: {mode}")
    transmission = jnp.exp(-jnp.clip(scale * tau2 * shape, 0.0, 80.0))
    return jnp.asarray(agn_lnu, dtype=jnp.float32) * transmission


def _fsps_powerlaw_unit_tau_shape_jax(
    wave_angstrom: jnp.ndarray, dust_index: jnp.ndarray
) -> jnp.ndarray:
    """FSPS ``dust_type=0`` AGN attenuation shape at unit V-band optical depth."""
    wave = jnp.maximum(jnp.asarray(wave_angstrom, dtype=jnp.float32), 1.0)
    slope = jnp.asarray(dust_index, dtype=jnp.float32)
    shape = (wave / 5500.0) ** slope
    return jnp.clip(
        jnp.nan_to_num(shape, nan=0.0, posinf=1.0e6, neginf=0.0),
        0.0,
        1.0e6,
    )


def combine_agn_and_igm_jax(
    wave: jnp.ndarray,
    dusted_sed: jnp.ndarray,
    agn_sed: jnp.ndarray,
    z_obs: jnp.ndarray,
    model_config: dict[str, Any] | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Combine stellar/gas and AGN spectra with the configured IGM ordering."""
    config = _normalized_model_config(model_config)
    order = str(config.get("agn_igm_order", "pre_igm"))
    wave = jnp.asarray(wave, dtype=jnp.float32)
    dusted = jnp.asarray(dusted_sed, dtype=jnp.float32)
    agn = jnp.asarray(agn_sed, dtype=jnp.float32)
    if order == "pre_igm":
        pre_igm = dusted + agn
        post_igm = apply_igm_transmission_jax(wave, pre_igm, z_obs, config)
    elif order == "fsps_after_igm":
        pre_igm = dusted
        post_igm = apply_igm_transmission_jax(wave, dusted, z_obs, config) + agn
    else:
        raise ValueError(f"Unsupported model.agn_igm_order: {order}")
    return (
        jnp.nan_to_num(pre_igm, nan=0.0, posinf=1.0e30, neginf=-1.0e30),
        jnp.nan_to_num(post_igm, nan=0.0, posinf=1.0e30, neginf=-1.0e30),
    )


def _popcosmos_dlog10_sfr(params: dict[str, Any]) -> jnp.ndarray:
    return jnp.asarray(
        [params[name] for name in POPCOSMOS_PARAMETER_NAMES[2:8]],
        dtype=jnp.float32,
    )


def _popcosmos_ssp_flux_grid(
    context: DspsContext, params: dict[str, Any], model_config: dict[str, Any]
) -> jnp.ndarray:
    if str(model_config.get("nebular_model", "fixed_ssp")) in {
        "gas_grid",
        "compressed_gas_grid",
    }:
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
    values = jnp.asarray(values)
    if x_grid.shape[0] == 1:
        return jnp.asarray(values[0], dtype=jnp.float32)
    x_clipped = jnp.clip(jnp.asarray(x, dtype=jnp.float32), x_grid[0], x_grid[-1])
    hi = jnp.searchsorted(x_grid, x_clipped, side="right")
    hi = jnp.clip(hi, 1, x_grid.shape[0] - 1)
    lo = hi - 1
    x0 = x_grid[lo]
    x1 = x_grid[hi]
    weight = (x_clipped - x0) / jnp.maximum(x1 - x0, 1.0e-12)
    return jnp.asarray(values[lo], dtype=jnp.float32) * (1.0 - weight) + jnp.asarray(
        values[hi], dtype=jnp.float32
    ) * weight


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
    return run_dsps_model_mags_jax(context, params)


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
    if (
        context.compressed_ssp_basis_jax is not None
        or context.compressed_ssp_coeff_jax is not None
    ):
        raise ValueError(
            "model.ssp_model='compressed_basis' cannot be expanded through "
            "_context_ssp_flux; use metallicity-specific compressed SSP "
            "interpolation instead."
        )
    return jnp.asarray(context.ssp.ssp_flux, dtype=jnp.float32)


def _context_ssp_surviving_mstar(context: DspsContext) -> jnp.ndarray | None:
    if context.ssp_surviving_mstar_jax is not None:
        return jnp.asarray(context.ssp_surviving_mstar_jax, dtype=jnp.float32)
    if context.ssp is not None and hasattr(context.ssp, "ssp_surviving_mstar"):
        return jnp.asarray(context.ssp.ssp_surviving_mstar, dtype=jnp.float32)
    return None


def _context_surviving_mstar_by_age(
    context: DspsContext, lgmet_abs: jnp.ndarray
) -> jnp.ndarray | None:
    grid = _context_ssp_surviving_mstar(context)
    if grid is None:
        return None
    return _interp_axis0_linear(
        _context_ssp_lgmet(context),
        grid,
        jnp.asarray(lgmet_abs, dtype=jnp.float32),
    )


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
