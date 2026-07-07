from __future__ import annotations

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.config import load_config
from euclid_dsps.filters import FilterCurve
from euclid_dsps.io import BandObservation, GalaxyObservation
from euclid_dsps.model import (
    DspsContext,
    ModelResult,
    _load_gas_ssp_grid,
    agn_component_jax,
    apply_agn_host_attenuation_jax,
    apply_charlot_fall_by_age_jax,
    apply_cosmos_two_component_dust_jax,
    apply_igm_transmission_jax,
    apply_popcosmos_dust_by_age_jax,
    apply_prospector_fsps_dust_by_age_jax,
    build_diffstar_sfh_table_jax,
    build_lognormal_sfh,
    build_popcosmos_lookback_bin_edges_jax,
    build_popcosmos_prospector_step_sfh_table_jax,
    build_popcosmos_sfh_table_jax,
    build_popcosmos_sfh_time_grid_jax,
    combine_agn_and_igm_jax,
    comparison_rows,
    fsps_madau95_igm_transmission_jax,
    gas_metallicity_constraint_penalty_jax,
    interpolate_compressed_gas_ssp_grid_jax,
    interpolate_compressed_gas_ssp_stellar_metallicity_jax,
    interpolate_gas_ssp_grid_jax,
    interpolate_ssp_stellar_metallicity_jax,
    load_context,
    log10_stellar_metallicity_to_absolute_jax,
    logsfr_ratios_to_sfr_bins_jax,
    normalize_sfh_mass_jax,
    normalize_sfh_to_stellar_mass_jax,
    parameters_for_row,
    popcosmos_age_weights_jax,
    predict_batch_derived,
    predict_batch_mags,
    predict_batch_seds,
    project_sfh_to_popcosmos_dlogsfr_jax,
    project_sfh_to_popcosmos_sfr_bins_jax,
    run_dsps_model_jax,
    run_dsps_model_mags_jax,
)
from euclid_dsps.parameters import (
    DIFFSTAR_REDUCED6_PARAMETER_NAMES,
    POPCOSMOS_PARAMETER_NAMES,
)


def _synthetic_context(model_config: dict | None = None) -> DspsContext:
    wave = np.linspace(900.0, 12000.0, 96)
    filter_wave = np.linspace(4500.0, 8500.0, 128)
    filt = FilterCurve(
        name="wide",
        wave=filter_wave,
        transmission=np.ones_like(filter_wave),
        source="synthetic",
    )
    lgmet = np.log10(np.asarray([0.004, 0.0134, 0.03]))
    lg_age = np.linspace(-3.0, 1.05, 24)
    met_factor = np.linspace(0.8, 1.2, len(lgmet))[:, None, None]
    age_factor = np.linspace(1.4, 0.5, len(lg_age))[None, :, None]
    wave_factor = (1.0 + 0.2 * (wave / wave.max()))[None, None, :]
    ssp_flux = 1.0e-3 * met_factor * age_factor * wave_factor
    context = DspsContext(
        ssp=None,
        filters={"wide": filt},
        n_sfh_bins=96,
        z_sun=0.0134,
        model_config=model_config or {"sfh_model": "lognormal"},
        ssp_wave_jax=jnp.asarray(wave, dtype=jnp.float32),
        ssp_lgmet_jax=jnp.asarray(lgmet, dtype=jnp.float32),
        ssp_lg_age_gyr_jax=jnp.asarray(lg_age, dtype=jnp.float32),
        ssp_flux_jax=jnp.asarray(ssp_flux, dtype=jnp.float32),
        jax_filters=(
            (
                jnp.asarray(filter_wave, dtype=jnp.float32),
                jnp.ones(len(filter_wave), dtype=jnp.float32),
            ),
        ),
    )
    if model_config and model_config.get("nebular_model") == "gas_grid":
        gas_lgmet = np.asarray([-1.0, 0.0], dtype=float)
        gas_lgu = np.asarray([-3.0, -2.0], dtype=float)
        gas_grid = np.stack(
            [
                np.stack(
                    [
                        ssp_flux * (1.0 + 0.05 * i + 0.03 * j)
                        for j in range(len(gas_lgu))
                    ]
                )
                for i in range(len(gas_lgmet))
            ]
        )
        context.gas_lgmet_grid_jax = jnp.asarray(gas_lgmet, dtype=jnp.float32)
        context.gas_lgu_grid_jax = jnp.asarray(gas_lgu, dtype=jnp.float32)
        context.ssp_flux_gas_grid_jax = jnp.asarray(gas_grid, dtype=jnp.float32)
    if model_config and model_config.get("nebular_model") == "compressed_gas_grid":
        gas_lgmet = np.asarray([-1.0, 0.0], dtype=float)
        gas_lgu = np.asarray([-3.0, -2.0], dtype=float)
        basis = wave_factor.reshape(-1)[None, :].astype(np.float32)
        coeff = np.zeros(
            (len(gas_lgmet), len(gas_lgu), len(lgmet), len(lg_age), 1),
            dtype=np.float32,
        )
        met_1d = np.linspace(0.8, 1.2, len(lgmet))
        age_1d = np.linspace(1.4, 0.5, len(lg_age))
        for i in range(len(gas_lgmet)):
            for j in range(len(gas_lgu)):
                gas_factor = 1.0 + 0.05 * i + 0.03 * j
                coeff[i, j, :, :, 0] = (
                    1.0e-3 * gas_factor * met_1d[:, None] * age_1d[None, :]
                )
        context.gas_lgmet_grid_jax = jnp.asarray(gas_lgmet, dtype=jnp.float32)
        context.gas_lgu_grid_jax = jnp.asarray(gas_lgu, dtype=jnp.float32)
        context.compressed_gas_basis_jax = jnp.asarray(basis, dtype=jnp.float32)
        context.compressed_gas_coeff_jax = jnp.asarray(coeff, dtype=jnp.float32)
        context.compressed_gas_scale_jax = jnp.ones(coeff.shape[:-1], dtype=jnp.float32)
    if model_config and model_config.get("agn_model") == "template_grid":
        agn_tau = np.asarray([5.0, 10.0, 150.0], dtype=float)
        agn_template = np.stack(
            [
                (tau / 10.0) * 1.0e-12 * (1.0 + wave / wave.max())
                for tau in agn_tau
            ]
        )
        context.agn_wave_jax = jnp.asarray(wave, dtype=jnp.float32)
        context.agn_tau_grid_jax = jnp.asarray(agn_tau, dtype=jnp.float32)
        context.agn_template_grid_jax = jnp.asarray(agn_template, dtype=jnp.float32)
    if model_config and model_config.get("agn_model") == "fsps_component_grid":
        fagn_grid = np.asarray([1.0e-4, 1.0e-2], dtype=float)
        agn_tau = np.asarray([5.0, 20.0], dtype=float)
        component = np.zeros(
            (len(fagn_grid), len(agn_tau), len(lgmet), len(lg_age), len(wave)),
            dtype=np.float32,
        )
        for i, fagn in enumerate(fagn_grid):
            for j, tau in enumerate(agn_tau):
                component[i, j, :, :, :] = (
                    fagn
                    * tau
                    * 1.0e-7
                    * (1.0 + wave[None, None, :] / wave.max())
                    * np.linspace(1.4, 0.5, len(lg_age))[None, :, None]
                )
        context.agn_wave_jax = jnp.asarray(wave, dtype=jnp.float32)
        context.agn_fagn_grid_jax = jnp.asarray(fagn_grid, dtype=jnp.float32)
        context.agn_tau_grid_jax = jnp.asarray(agn_tau, dtype=jnp.float32)
        context.agn_component_lgmet_jax = jnp.asarray(lgmet, dtype=jnp.float32)
        context.agn_component_lg_age_gyr_jax = jnp.asarray(lg_age, dtype=jnp.float32)
        context.agn_component_grid_jax = jnp.asarray(component, dtype=jnp.float32)
    if model_config and model_config.get("agn_model") == "compressed_fsps_component_grid":
        fagn_grid = np.asarray([1.0e-4, 1.0e-2], dtype=float)
        agn_tau = np.asarray([5.0, 20.0], dtype=float)
        basis = (1.0 + wave / wave.max())[None, :].astype(np.float32)
        coeff = np.zeros(
            (len(fagn_grid), len(agn_tau), len(lgmet), len(lg_age), 1),
            dtype=np.float32,
        )
        age_factor_1d = np.linspace(1.4, 0.5, len(lg_age))
        for i, fagn in enumerate(fagn_grid):
            for j, tau in enumerate(agn_tau):
                coeff[i, j, :, :, 0] = fagn * tau * 1.0e-7 * age_factor_1d[None, :]
        context.agn_wave_jax = jnp.asarray(wave, dtype=jnp.float32)
        context.agn_fagn_grid_jax = jnp.asarray(fagn_grid, dtype=jnp.float32)
        context.agn_tau_grid_jax = jnp.asarray(agn_tau, dtype=jnp.float32)
        context.agn_component_lgmet_jax = jnp.asarray(lgmet, dtype=jnp.float32)
        context.agn_component_lg_age_gyr_jax = jnp.asarray(lg_age, dtype=jnp.float32)
        context.compressed_agn_basis_jax = jnp.asarray(basis, dtype=jnp.float32)
        context.compressed_agn_coeff_jax = jnp.asarray(coeff, dtype=jnp.float32)
        context.compressed_agn_scale_jax = jnp.ones(coeff.shape[:-1], dtype=jnp.float32)
    return context


def _popcosmos_params() -> dict[str, float]:
    values = {
        "z_obs": 0.5,
        "log10_stellar_mass": 10.0,
        "log10_stellar_metallicity": 0.0,
        "tau2": 0.2,
        "dust_index_n": -0.7,
        "tau1_over_tau2": 1.0,
        "log10_gas_metallicity": 0.0,
        "log10_gas_ionization": -2.4,
        "ln_fagn": -8.0,
        "ln_tauagn": np.log(10.0),
    }
    values.update({f"dlog10_sfr_{index}": 0.0 for index in range(1, 7)})
    return {name: values[name] for name in POPCOSMOS_PARAMETER_NAMES}


def _diffstar_params() -> dict[str, float]:
    values = {
        "z_obs": 0.5,
        "log10_stellar_mass": 10.0,
        "diffstar_lgmcrit": 12.0,
        "diffstar_lgy_at_mcrit": -10.0,
        "diffstar_indx_lo": 1.0,
        "diffstar_lg_qt": 1.0,
        "diffstar_lg_drop": -1.0,
        "diffstar_lg_rejuv": -0.2,
        "log10_stellar_metallicity": 0.0,
        "tau2": 0.2,
        "dust_index_n": -0.7,
        "tau1_over_tau2": 1.0,
        "log10_gas_metallicity": 0.0,
        "log10_gas_ionization": -2.4,
        "ln_fagn": -8.0,
        "ln_tauagn": np.log(10.0),
    }
    return {name: values[name] for name in DIFFSTAR_REDUCED6_PARAMETER_NAMES}


def _wide_filter(name: str = "wide", lo: float = 4500.0, hi: float = 8500.0) -> FilterCurve:
    wave = np.linspace(lo, hi, 64)
    return FilterCurve(
        name=name,
        wave=wave,
        transmission=np.ones_like(wave),
        source="synthetic",
    )


def _write_synthetic_ssp_hdf5(
    path,
    *,
    imf_type: int | None = 1,
    imf_name: str | None = "chabrier",
    z_sun: float | None = 0.0142,
    wave: np.ndarray | None = None,
    lg_age: np.ndarray | None = None,
    lgmet: np.ndarray | None = None,
    surviving_mstar: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wave = np.asarray(
        np.linspace(1000.0, 9000.0, 16) if wave is None else wave,
        dtype=np.float32,
    )
    lg_age = np.asarray(
        np.linspace(-3.0, 0.0, 5) if lg_age is None else lg_age,
        dtype=np.float32,
    )
    lgmet = np.asarray(
        np.log10([0.004, 0.0142, 0.03]) if lgmet is None else lgmet,
        dtype=np.float32,
    )
    flux = np.ones((len(lgmet), len(lg_age), len(wave)), dtype=np.float32) * 1.0e-3
    with h5py.File(path, "w") as handle:
        handle["ssp_wave"] = wave
        handle["ssp_lg_age_gyr"] = lg_age
        handle["ssp_lgmet"] = lgmet
        handle["ssp_flux"] = flux
        if surviving_mstar is not None:
            handle["ssp_surviving_mstar"] = np.asarray(
                surviving_mstar, dtype=np.float32
            )
        if imf_type is not None:
            handle.attrs["imf_type"] = imf_type
        if imf_name is not None:
            handle.attrs["imf_name"] = imf_name
        if z_sun is not None:
            handle.attrs["z_sun"] = z_sun
    return wave, lg_age, lgmet


def _write_synthetic_gas_hdf5(
    path,
    wave: np.ndarray,
    lg_age: np.ndarray,
    lgmet: np.ndarray,
    *,
    imf_type: int = 1,
    z_sun: float = 0.0142,
    include_units: bool = True,
    wave_offset: float = 0.0,
    enriched_line: bool = False,
) -> None:
    gas_lgmet = np.asarray([-1.0, 0.0], dtype=np.float32)
    gas_lgu = np.asarray([-3.0, -2.0], dtype=np.float32)
    axes_wave = np.asarray(wave + wave_offset, dtype=np.float32)
    flux_shape = (len(gas_lgmet), len(gas_lgu), len(lgmet), len(lg_age), len(wave))
    flux = np.ones(flux_shape, dtype=np.float32) * 1.0e-3
    with h5py.File(path, "w") as handle:
        handle["ssp_wave"] = axes_wave
        handle["ssp_lg_age_gyr"] = np.asarray(lg_age, dtype=np.float32)
        handle["ssp_lgmet"] = np.asarray(lgmet, dtype=np.float32)
        handle["gas_lgmet_grid"] = gas_lgmet
        handle["gas_lgu_grid"] = gas_lgu
        if enriched_line:
            continuum = np.ones(flux_shape, dtype=np.float32) * 1.0e-3
            line_grid = np.ones(flux_shape[:-1] + (1,), dtype=np.float32) * 5.0e-3
            handle["nebular_continuum_flux"] = continuum
            handle["line_flux_grid"] = line_grid
            handle["emline_wavelengths"] = np.asarray([5000.0], dtype=np.float32)
            handle["emline_names"] = np.asarray([b"SYNTH_5000"])
            raw = np.array(continuum, copy=True)
            raw[..., int(np.argmin(np.abs(axes_wave - 5000.0)))] += line_grid[..., 0]
            handle["ssp_flux"] = raw
        else:
            handle["ssp_flux"] = flux
        handle.attrs["imf_type"] = imf_type
        handle.attrs["imf_name"] = "chabrier" if imf_type == 1 else "kroupa"
        handle.attrs["z_sun"] = z_sun
        if include_units:
            handle.attrs["units_ssp_wave"] = "Angstrom"
            handle.attrs["units_ssp_lg_age_gyr"] = "log10(age/Gyr)"
            handle.attrs["units_ssp_lgmet"] = (
                "log10(absolute stellar metallicity mass fraction)"
            )
            handle.attrs["units_gas_lgmet_grid"] = "log10(Zgas/Zsun)"
            handle.attrs["units_gas_lgu_grid"] = "log10 ionization parameter U"


def _write_synthetic_compressed_gas_hdf5(
    path,
    wave: np.ndarray,
    lg_age: np.ndarray,
    lgmet: np.ndarray,
    *,
    imf_type: int = 1,
    z_sun: float = 0.0142,
) -> None:
    gas_lgmet = np.asarray([-1.0, 0.0], dtype=np.float32)
    gas_lgu = np.asarray([-3.0, -2.0], dtype=np.float32)
    basis = np.asarray([(wave / wave.max()) + 1.0], dtype=np.float32)
    coeff = np.ones(
        (len(gas_lgmet), len(gas_lgu), len(lgmet), len(lg_age), 1),
        dtype=np.float32,
    )
    scale = np.ones(coeff.shape[:-1], dtype=np.float32) * 1.0e-3
    with h5py.File(path, "w") as handle:
        handle["ssp_wave"] = np.asarray(wave, dtype=np.float32)
        handle["ssp_lg_age_gyr"] = np.asarray(lg_age, dtype=np.float32)
        handle["ssp_lgmet"] = np.asarray(lgmet, dtype=np.float32)
        handle["gas_lgmet_grid"] = gas_lgmet
        handle["gas_lgu_grid"] = gas_lgu
        handle["gas_basis"] = basis
        handle["gas_coeff"] = coeff
        handle["gas_scale"] = scale
        handle.attrs["asset_kind"] = "popcosmos_chabrier_compressed_gas_grid"
        handle.attrs["imf_type"] = imf_type
        handle.attrs["imf_name"] = "chabrier" if imf_type == 1 else "kroupa"
        handle.attrs["z_sun"] = z_sun
        handle.attrs["units_ssp_wave"] = "Angstrom"
        handle.attrs["units_ssp_lg_age_gyr"] = "log10(age/Gyr)"
        handle.attrs["units_ssp_lgmet"] = (
            "log10(absolute stellar metallicity mass fraction)"
        )
        handle.attrs["units_gas_lgmet_grid"] = "log10(Zgas/Zsun)"
        handle.attrs["units_gas_lgu_grid"] = "log10 ionization parameter U"


def _write_synthetic_compressed_ssp_hdf5(
    path,
    wave: np.ndarray,
    lg_age: np.ndarray,
    lgmet: np.ndarray,
    *,
    imf_type: int = 1,
    z_sun: float = 0.0142,
) -> None:
    basis = np.ones((1, len(wave)), dtype=np.float32)
    coeff = np.ones((len(lgmet), len(lg_age), 1), dtype=np.float16)
    scale = np.ones(coeff.shape[:-1], dtype=np.float32) * 1.0e-3
    with h5py.File(path, "w") as handle:
        handle["ssp_wave"] = np.asarray(wave, dtype=np.float32)
        handle["ssp_lg_age_gyr"] = np.asarray(lg_age, dtype=np.float32)
        handle["ssp_lgmet"] = np.asarray(lgmet, dtype=np.float32)
        handle["ssp_basis"] = basis
        handle["ssp_coeff"] = coeff
        handle["ssp_scale"] = scale
        handle.attrs["asset_kind"] = "popcosmos_chabrier_compressed_stellar_ssp"
        handle.attrs["imf_type"] = imf_type
        handle.attrs["imf_name"] = "chabrier" if imf_type == 1 else "kroupa"
        handle.attrs["z_sun"] = z_sun


def _noagn_params(params: dict[str, float]) -> dict[str, float]:
    return {
        name: value
        for name, value in params.items()
        if name not in {"ln_fagn", "ln_tauagn"}
    }


def test_lognormal_sfh_stays_positive_and_peak_controls_shape() -> None:
    time = np.linspace(0.1, 10.0, 128)

    early = build_lognormal_sfh(time, 0.0, 2.0, 0.6)
    late = build_lognormal_sfh(time, 0.0, 7.0, 0.6)

    assert np.all(early > 0.0)
    assert np.all(late > 0.0)
    assert early[np.argmin(np.abs(time - 2.0))] > late[
        np.argmin(np.abs(time - 2.0))
    ]
    assert late[np.argmin(np.abs(time - 7.0))] > early[
        np.argmin(np.abs(time - 7.0))
    ]


def test_popcosmos_bin_edges_are_increasing() -> None:
    for t_obs in (0.05, 0.2, 10.0):
        edges = np.asarray(build_popcosmos_lookback_bin_edges_jax(jnp.asarray(t_obs)))

        assert edges.shape == (8,)
        assert np.all(np.isfinite(edges))
        assert np.all(np.diff(edges) > 0.0)
        assert edges[0] == pytest.approx(0.0)
        assert edges[-1] == pytest.approx(t_obs)


def test_popcosmos_sfh_equal_ratios_constant_sfr_bins() -> None:
    params = _popcosmos_params()
    t_obs = jnp.asarray(10.0)
    edges = build_popcosmos_lookback_bin_edges_jax(t_obs)
    lookback_midpoints = 0.5 * (edges[:-1] + edges[1:])
    sfr = np.asarray(
        build_popcosmos_sfh_table_jax(t_obs - lookback_midpoints, t_obs, params)
    )

    assert np.asarray(logsfr_ratios_to_sfr_bins_jax(jnp.zeros(6))).tolist() == pytest.approx(
        [1.0] * 7
    )
    assert sfr.tolist() == pytest.approx([sfr[0]] * 7)
    assert np.trapezoid(
        np.asarray(build_popcosmos_sfh_table_jax(jnp.linspace(0.01, t_obs, 128), t_obs, params)),
        np.linspace(0.01, 10.0, 128),
    ) > 0.0


def test_popcosmos_sfh_ratios_follow_documented_sign() -> None:
    params = _popcosmos_params()
    params["dlog10_sfr_1"] = 1.0
    bins = np.asarray(logsfr_ratios_to_sfr_bins_jax(jnp.asarray([1.0, 0, 0, 0, 0, 0])))

    assert bins[0] == pytest.approx(1.0)
    assert bins[1] == pytest.approx(0.1)
    assert bins[0] > bins[1]

    t_obs = jnp.asarray(10.0)
    edges = build_popcosmos_lookback_bin_edges_jax(t_obs)
    lookback_midpoints = 0.5 * (edges[:2] + edges[1:3])
    sfr = np.asarray(
        build_popcosmos_sfh_table_jax(t_obs - lookback_midpoints, t_obs, params)
    )
    assert sfr[0] > sfr[1]


def test_popcosmos_prospector_step_sfh_table_matches_bin_convention() -> None:
    params = _popcosmos_params()
    params["dlog10_sfr_1"] = 1.0
    t_obs = jnp.asarray(10.0)

    gal_t, sfr = build_popcosmos_prospector_step_sfh_table_jax(t_obs, params)

    assert gal_t.shape == (14,)
    assert sfr.shape == (14,)
    assert np.all(np.diff(np.asarray(gal_t)) > 0.0)
    assert float(gal_t[-1]) < float(t_obs)
    assert float(sfr[-1]) > float(sfr[-3])


def test_popcosmos_age_weights_integrate_step_bins_directly() -> None:
    params = _popcosmos_params()
    params["dlog10_sfr_1"] = 1.0
    t_obs = jnp.asarray(10.0)
    ssp_lg_age = jnp.linspace(-4.0, 1.0, 256)

    weights = popcosmos_age_weights_jax(t_obs, params, ssp_lg_age)
    ages = 10.0 ** np.asarray(ssp_lg_age)
    weight_array = np.asarray(weights)

    assert weights.shape == ssp_lg_age.shape
    assert np.all(np.isfinite(weight_array))
    assert float(weights.sum()) == pytest.approx(1.0)
    assert weight_array[ages < 0.03].sum() > weight_array[
        (ages >= 0.03) & (ages < 0.10)
    ].sum()


def test_popcosmos_sfh_time_grid_can_preserve_legacy_linear_mode() -> None:
    params = _popcosmos_params()
    t_obs = jnp.asarray(10.0)

    gal_t, sfr = build_popcosmos_sfh_time_grid_jax(
        t_obs, params, 32, {"sfh_time_grid": "linear"}
    )

    assert gal_t.shape == (32,)
    assert sfr.shape == (32,)
    assert np.all(np.asarray(sfr) > 0.0)


def test_sfh_projection_to_popcosmos_bins_and_ratios() -> None:
    t_obs = jnp.asarray(10.0)
    time = jnp.linspace(0.01, t_obs, 256)
    rising_sfr = time

    bins = project_sfh_to_popcosmos_sfr_bins_jax(time, rising_sfr, t_obs)
    ratios = project_sfh_to_popcosmos_dlogsfr_jax(time, rising_sfr, t_obs)

    assert bins.shape == (7,)
    assert ratios.shape == (6,)
    assert np.all(np.isfinite(np.asarray(bins)))
    assert np.all(np.isfinite(np.asarray(ratios)))
    assert float(bins[0]) > float(bins[-1])


def test_stellar_mass_normalization_matches_surviving_mass() -> None:
    gal_t = jnp.linspace(0.01, 10.0, 256)
    sfr = jnp.ones_like(gal_t)
    ssp_lg_age = jnp.linspace(-3.0, 1.0, 32)

    scaled_sfr, formed_mass, surviving_mass = normalize_sfh_to_stellar_mass_jax(
        gal_t, sfr, ssp_lg_age, jnp.asarray(10.0), jnp.asarray(10.0)
    )

    assert np.all(np.asarray(scaled_sfr) > 0.0)
    assert float(surviving_mass) == pytest.approx(1.0e10, rel=2.0e-3)
    assert float(formed_mass) > float(surviving_mass)


def test_context_uses_fsps_surviving_mass_grid_when_available(tmp_path) -> None:
    ssp_path = tmp_path / "ssp_chabrier_mstar.h5"
    surviving = np.vstack(
        [
            np.linspace(0.9, 0.5, 5),
            np.linspace(0.8, 0.4, 5),
            np.linspace(0.7, 0.3, 5),
        ]
    )
    _write_synthetic_ssp_hdf5(ssp_path, surviving_mstar=surviving)

    context = load_context(
        str(ssp_path),
        {"wide": _wide_filter()},
        model_config={
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "dust_model": "prospector_fsps",
            "igm_model": "none",
            "nebular_model": "fixed_ssp",
            "agn_model": "none",
            "z_sun": 0.0142,
        },
    )

    assert context.ssp_surviving_mstar_jax is not None
    np.testing.assert_allclose(np.asarray(context.ssp_surviving_mstar_jax), surviving)


def test_single_stellar_metallicity_interpolation_shape_and_grad() -> None:
    lgmet = jnp.asarray([-2.0, -1.0, 0.0])
    ssp_flux = jnp.stack(
        [
            jnp.ones((4, 5)),
            2.0 * jnp.ones((4, 5)),
            4.0 * jnp.ones((4, 5)),
        ]
    )

    out = interpolate_ssp_stellar_metallicity_jax(lgmet, ssp_flux, jnp.asarray(-1.5))
    clipped_low = interpolate_ssp_stellar_metallicity_jax(
        lgmet, ssp_flux, jnp.asarray(-3.0)
    )
    clipped_high = interpolate_ssp_stellar_metallicity_jax(
        lgmet, ssp_flux, jnp.asarray(0.5)
    )
    grad = jax.grad(
        lambda value: jnp.sum(
            interpolate_ssp_stellar_metallicity_jax(lgmet, ssp_flux, value)
        )
    )(jnp.asarray(-1.5))

    assert out.shape == (4, 5)
    assert np.asarray(out).mean() == pytest.approx(1.5)
    np.testing.assert_allclose(np.asarray(clipped_low), np.ones((4, 5)))
    np.testing.assert_allclose(np.asarray(clipped_high), 4.0 * np.ones((4, 5)))
    assert np.isfinite(float(grad))


def test_popcosmos_metallicity_conversion_uses_configured_z_sun() -> None:
    value = log10_stellar_metallicity_to_absolute_jax(0.0, 0.0142)

    assert float(value) == pytest.approx(np.log10(0.0142))


def test_charlot_fall_birth_cloud_age_dependence() -> None:
    wave = jnp.asarray([100.0, 1500.0, 5500.0])
    ages = jnp.log10(jnp.asarray([0.005, 0.02]))
    sed_by_age = jnp.ones((2, 3))

    no_dust = apply_charlot_fall_by_age_jax(wave, ages, sed_by_age, 0.0, -0.7, 0.0)
    no_birth = apply_charlot_fall_by_age_jax(wave, ages, sed_by_age, 0.5, -0.7, 0.0)
    with_birth = apply_charlot_fall_by_age_jax(wave, ages, sed_by_age, 0.5, -0.7, 2.0)

    np.testing.assert_allclose(np.asarray(no_dust), np.ones((2, 3)))
    assert np.all(np.asarray(with_birth[0]) < np.asarray(no_birth[0]))
    np.testing.assert_allclose(np.asarray(with_birth[1]), np.asarray(no_birth[1]))
    assert np.all(np.isfinite(np.asarray(with_birth)))


def test_charlot_fall_powerlaw_wrapper_matches_previous_behavior() -> None:
    wave = jnp.asarray([100.0, 1500.0, 5500.0])
    ages = jnp.log10(jnp.asarray([0.005, 0.02]))
    sed_by_age = jnp.ones((2, 3))

    direct = apply_charlot_fall_by_age_jax(wave, ages, sed_by_age, 0.5, -0.7, 2.0)
    wrapped = apply_popcosmos_dust_by_age_jax(
        wave,
        ages,
        sed_by_age,
        0.5,
        -0.7,
        2.0,
        {"dust_model": "charlot_fall_powerlaw"},
    )

    np.testing.assert_allclose(np.asarray(wrapped), np.asarray(direct))


def test_prospector_fsps_dust_zero_tau_returns_intrinsic_sed() -> None:
    wave = jnp.asarray([100.0, 1500.0, 5500.0])
    ages = jnp.log10(jnp.asarray([0.005, 0.02]))
    sed_by_age = jnp.arange(1, 7, dtype=jnp.float32).reshape(2, 3)

    dusted = apply_prospector_fsps_dust_by_age_jax(
        wave, ages, sed_by_age, 0.0, -0.7, 0.0
    )

    np.testing.assert_allclose(np.asarray(dusted), np.asarray(sed_by_age))


def test_prospector_fsps_birth_cloud_affects_only_young_ages() -> None:
    wave = jnp.asarray([1500.0, 5500.0, 9000.0])
    ages = jnp.log10(jnp.asarray([0.005, 0.02]))
    sed_by_age = jnp.ones((2, 3))

    no_birth = apply_prospector_fsps_dust_by_age_jax(
        wave, ages, sed_by_age, 0.5, -0.7, 0.0
    )
    with_birth = apply_prospector_fsps_dust_by_age_jax(
        wave, ages, sed_by_age, 0.5, -0.7, 2.0
    )

    assert np.all(np.asarray(with_birth[0]) < np.asarray(no_birth[0]))
    np.testing.assert_allclose(np.asarray(with_birth[1]), np.asarray(no_birth[1]))


def test_prospector_fsps_dust_is_grad_compatible_and_finite_at_short_waves() -> None:
    wave = jnp.asarray([10.0, 100.0, 1500.0, 5500.0])
    ages = jnp.log10(jnp.asarray([0.005, 0.02]))
    sed_by_age = jnp.ones((2, 4))

    def objective(values: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(
            apply_prospector_fsps_dust_by_age_jax(
                wave, ages, sed_by_age, values[0], values[1], values[2]
            )
        )

    dusted = apply_prospector_fsps_dust_by_age_jax(
        wave, ages, sed_by_age, 0.5, -0.7, 1.0
    )
    grad = jax.grad(objective)(jnp.asarray([0.5, -0.7, 1.0]))

    assert np.all(np.isfinite(np.asarray(dusted)))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_igm_transmission_low_and_high_redshift() -> None:
    wave = jnp.asarray([800.0, 1000.0, 1300.0])
    sed = jnp.ones(3)

    none = apply_igm_transmission_jax(wave, sed, 6.0, {"igm_model": "none"})
    low = apply_igm_transmission_jax(wave, sed, 0.0, {"igm_model": "madau95_approx"})
    high = apply_igm_transmission_jax(wave, sed, 6.0, {"igm_model": "madau95_approx"})

    assert np.asarray(none).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert np.asarray(low).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert float(high[0]) < float(high[1]) < 1.0
    assert float(high[2]) == pytest.approx(1.0)


def test_fsps_madau95_igm_matches_expected_ordering() -> None:
    wave = jnp.asarray([800.0, 1000.0, 1300.0])
    transmission = fsps_madau95_igm_transmission_jax(wave, 6.0)

    assert np.all(np.asarray(transmission) > 0.0)
    assert float(transmission[0]) < float(transmission[1]) < 1.0
    assert float(transmission[2]) == pytest.approx(1.0)


def test_fsps_after_igm_order_leaves_agn_unattenuated_by_igm() -> None:
    wave = jnp.asarray([800.0, 1000.0, 1300.0], dtype=jnp.float32)
    stellar = jnp.ones(3, dtype=jnp.float32)
    agn = jnp.asarray([2.0, 2.0, 2.0], dtype=jnp.float32)
    config = {"igm_model": "fsps_madau95", "agn_igm_order": "fsps_after_igm"}

    pre_igm, post_igm = combine_agn_and_igm_jax(wave, stellar, agn, 6.0, config)
    stellar_only_post = apply_igm_transmission_jax(wave, stellar, 6.0, config)

    np.testing.assert_allclose(np.asarray(pre_igm), np.ones(3), rtol=1.0e-6)
    np.testing.assert_allclose(
        np.asarray(post_igm),
        np.asarray(stellar_only_post + agn),
        rtol=1.0e-6,
    )
    assert float(post_igm[0]) == pytest.approx(2.0)
    assert float(post_igm[0]) < 3.0


def test_pre_igm_order_attenuates_agn_and_stellar_together() -> None:
    wave = jnp.asarray([800.0, 1000.0, 1300.0], dtype=jnp.float32)
    stellar = jnp.ones(3, dtype=jnp.float32)
    agn = jnp.asarray([2.0, 2.0, 2.0], dtype=jnp.float32)
    config = {"igm_model": "fsps_madau95", "agn_igm_order": "pre_igm"}

    pre_igm, post_igm = combine_agn_and_igm_jax(wave, stellar, agn, 6.0, config)
    total_post = apply_igm_transmission_jax(wave, stellar + agn, 6.0, config)

    np.testing.assert_allclose(np.asarray(pre_igm), np.full(3, 3.0), rtol=1.0e-6)
    np.testing.assert_allclose(np.asarray(post_igm), np.asarray(total_post), rtol=1.0e-6)
    assert float(post_igm[0]) < 2.0


def test_gas_grid_loader_and_interpolation_from_synthetic_hdf5(tmp_path) -> None:
    path = tmp_path / "gas_grid.h5"
    flux = np.ones((2, 3, 4, 5, 6), dtype=float)
    flux[1] *= 2.0
    with h5py.File(path, "w") as handle:
        handle["ssp_wave"] = np.linspace(1000.0, 6000.0, 6)
        handle["ssp_lg_age_gyr"] = np.linspace(-3.0, 1.0, 5)
        handle["ssp_lgmet"] = np.linspace(-2.0, 0.0, 4)
        handle["gas_lgmet_grid"] = np.asarray([-1.0, 0.0])
        handle["gas_lgu_grid"] = np.asarray([-3.0, -2.0, -1.0])
        handle["ssp_flux"] = flux

    gas_lgmet, gas_lgu, loaded_flux = _load_gas_ssp_grid(path)
    context = DspsContext(
        ssp=None,
        filters={},
        gas_lgmet_grid_jax=jnp.asarray(gas_lgmet),
        gas_lgu_grid_jax=jnp.asarray(gas_lgu),
        ssp_flux_gas_grid_jax=jnp.asarray(loaded_flux),
    )
    interpolated = interpolate_gas_ssp_grid_jax(context, -0.5, -2.5)

    assert loaded_flux.shape == (2, 3, 4, 5, 6)
    assert interpolated.shape == (4, 5, 6)
    assert np.asarray(interpolated).mean() == pytest.approx(1.5)


def test_compressed_gas_grid_matches_synthetic_dense_interpolation() -> None:
    dense_context = _synthetic_context({"nebular_model": "gas_grid"})
    compressed_context = _synthetic_context({"nebular_model": "compressed_gas_grid"})

    dense = interpolate_gas_ssp_grid_jax(dense_context, -0.5, -2.5)
    compressed = interpolate_compressed_gas_ssp_grid_jax(
        compressed_context,
        -0.5,
        -2.5,
    )
    grad = jax.grad(
        lambda values: jnp.sum(
            interpolate_gas_ssp_grid_jax(compressed_context, values[0], values[1])
        )
    )(jnp.asarray([-0.5, -2.5]))

    np.testing.assert_allclose(np.asarray(compressed), np.asarray(dense), rtol=1.0e-6)
    assert compressed_context.ssp_flux_gas_grid_jax is None
    assert compressed_context.compressed_gas_coeff_jax is not None
    assert np.all(np.isfinite(np.asarray(grad)))


def test_compressed_gas_direct_stellar_metallicity_interp_matches_dense_path() -> None:
    context = _synthetic_context({"nebular_model": "compressed_gas_grid"})
    lgmet_abs = jnp.asarray(-2.0)
    full_grid = interpolate_compressed_gas_ssp_grid_jax(context, -0.5, -2.5)
    dense_path = interpolate_ssp_stellar_metallicity_jax(
        context.ssp_lgmet_jax,
        full_grid,
        lgmet_abs,
    )
    direct_path = interpolate_compressed_gas_ssp_stellar_metallicity_jax(
        context,
        -0.5,
        -2.5,
        lgmet_abs,
    )
    grad = jax.grad(
        lambda values: jnp.sum(
            interpolate_compressed_gas_ssp_stellar_metallicity_jax(
                context,
                values[0],
                values[1],
                values[2],
            )
        )
    )(jnp.asarray([-0.5, -2.5, -2.0]))

    assert direct_path.shape == (
        context.ssp_lg_age_gyr_jax.shape[0],
        context.ssp_wave_jax.shape[0],
    )
    np.testing.assert_allclose(
        np.asarray(direct_path),
        np.asarray(dense_path),
        rtol=1.0e-6,
    )
    assert np.all(np.isfinite(np.asarray(grad)))


def test_popcosmos_hdf5_imf_chabrier_metadata_passes(tmp_path) -> None:
    ssp_path = tmp_path / "ssp_chabrier.h5"
    _write_synthetic_ssp_hdf5(ssp_path, imf_type=1, imf_name="chabrier")

    context = load_context(
        str(ssp_path),
        {"wide": _wide_filter()},
        model_config={
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "nebular_model": "fixed_ssp",
            "agn_model": "none",
            "z_sun": 0.0142,
        },
    )

    assert context.z_sun == pytest.approx(0.0142)


def test_popcosmos_hdf5_imf_kroupa_metadata_fails(tmp_path) -> None:
    ssp_path = tmp_path / "ssp_chabrier_bad_imf_metadata.h5"
    _write_synthetic_ssp_hdf5(ssp_path, imf_type=2, imf_name="kroupa")

    with pytest.raises(ValueError, match="Chabrier IMF"):
        load_context(
            str(ssp_path),
            {"wide": _wide_filter()},
            model_config={
                "sfh_model": "popcosmos_bins",
                "stellar_metallicity_model": "single",
                "nebular_model": "fixed_ssp",
                "agn_model": "none",
                "z_sun": 0.0142,
            },
        )


@pytest.mark.parametrize(
    ("imf_type", "imf_name"),
    [
        (2, "chabrier"),
        (1, "kroupa"),
    ],
)
def test_popcosmos_hdf5_inconsistent_imf_metadata_fails(
    tmp_path, imf_type: int, imf_name: str
) -> None:
    ssp_path = tmp_path / "ssp_chabrier_inconsistent_imf_metadata.h5"
    _write_synthetic_ssp_hdf5(ssp_path, imf_type=imf_type, imf_name=imf_name)

    with pytest.raises(ValueError, match="consistently declare a Chabrier IMF"):
        load_context(
            str(ssp_path),
            {"wide": _wide_filter()},
            model_config={
                "sfh_model": "popcosmos_bins",
                "stellar_metallicity_model": "single",
                "nebular_model": "fixed_ssp",
                "agn_model": "none",
                "z_sun": 0.0142,
            },
        )


def test_legacy_lognormal_loads_asset_without_popcosmos_metadata(tmp_path) -> None:
    ssp_path = tmp_path / "legacy_ssp.h5"
    _write_synthetic_ssp_hdf5(ssp_path, imf_type=None, imf_name=None, z_sun=None)

    context = load_context(
        str(ssp_path),
        {"wide": _wide_filter()},
        model_config={"sfh_model": "lognormal"},
    )

    assert context.z_sun == pytest.approx(0.0134)


def test_popcosmos_z_sun_metadata_mismatch_fails(tmp_path) -> None:
    ssp_path = tmp_path / "ssp_chabrier.h5"
    _write_synthetic_ssp_hdf5(ssp_path, z_sun=0.0134)

    with pytest.raises(ValueError, match="z_sun mismatch"):
        load_context(
            str(ssp_path),
            {"wide": _wide_filter()},
            model_config={
                "sfh_model": "popcosmos_bins",
                "stellar_metallicity_model": "single",
                "nebular_model": "fixed_ssp",
                "agn_model": "none",
                "z_sun": 0.0142,
            },
        )


def test_popcosmos_gas_grid_axes_and_units_validation(tmp_path) -> None:
    ssp_path = tmp_path / "ssp_chabrier.h5"
    gas_path = tmp_path / "gas_chabrier.h5"
    wave, lg_age, lgmet = _write_synthetic_ssp_hdf5(ssp_path)
    _write_synthetic_gas_hdf5(gas_path, wave, lg_age, lgmet)

    context = load_context(
        str(ssp_path),
        {"wide": _wide_filter()},
        model_config={
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "nebular_model": "gas_grid",
            "gas_grid_path": str(gas_path),
            "agn_model": "none",
            "z_sun": 0.0142,
        },
    )

    assert context.ssp_flux_gas_grid_jax is not None


def test_load_context_compressed_gas_grid_does_not_load_dense_grid(tmp_path) -> None:
    ssp_path = tmp_path / "ssp_chabrier.h5"
    gas_path = tmp_path / "compressed_gas_chabrier.h5"
    wave, lg_age, lgmet = _write_synthetic_ssp_hdf5(ssp_path)
    _write_synthetic_compressed_gas_hdf5(gas_path, wave, lg_age, lgmet)

    context = load_context(
        str(ssp_path),
        {"wide": _wide_filter()},
        model_config={
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "nebular_model": "compressed_gas_grid",
            "compressed_gas_grid_path": str(gas_path),
            "agn_model": "none",
            "z_sun": 0.0142,
        },
    )

    assert context.ssp_flux_gas_grid_jax is None
    assert context.compressed_gas_basis_jax is not None
    assert context.compressed_gas_coeff_jax is not None
    assert context.compressed_gas_scale_jax is not None


def test_compressed_ssp_loads_without_resident_dense_ssp_flux(tmp_path) -> None:
    ssp_path = tmp_path / "ssp_chabrier.h5"
    compressed_path = tmp_path / "compressed_ssp_chabrier.h5"
    wave, lg_age, lgmet = _write_synthetic_ssp_hdf5(ssp_path)
    _write_synthetic_compressed_ssp_hdf5(compressed_path, wave, lg_age, lgmet)

    context = load_context(
        str(ssp_path),
        {"wide": _wide_filter()},
        model_config={
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "ssp_model": "compressed_basis",
            "compressed_ssp_path": str(compressed_path),
            "nebular_model": "fixed_ssp",
            "agn_model": "none",
            "z_sun": 0.0142,
        },
    )
    result = run_dsps_model_jax(context, _noagn_params(_popcosmos_params()))

    assert context.ssp_flux_jax is None
    assert context.compressed_ssp_coeff_jax.dtype == jnp.float16
    assert np.all(np.isfinite(np.asarray(result.model_mags)))


def test_compressed_ssp_runtime_dtype_can_upcast_coefficients(tmp_path) -> None:
    ssp_path = tmp_path / "ssp_chabrier.h5"
    compressed_path = tmp_path / "compressed_ssp_chabrier.h5"
    wave, lg_age, lgmet = _write_synthetic_ssp_hdf5(ssp_path)
    _write_synthetic_compressed_ssp_hdf5(compressed_path, wave, lg_age, lgmet)

    context = load_context(
        str(ssp_path),
        {"wide": _wide_filter()},
        model_config={
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "ssp_model": "compressed_basis",
            "compressed_ssp_path": str(compressed_path),
            "compressed_ssp_runtime_dtype": "float32",
            "nebular_model": "fixed_ssp",
            "agn_model": "none",
            "z_sun": 0.0142,
        },
    )

    assert context.ssp_flux_jax is None
    assert context.compressed_ssp_basis_jax.dtype == jnp.float32
    assert context.compressed_ssp_coeff_jax.dtype == jnp.float32
    assert context.compressed_ssp_scale_jax.dtype == jnp.float32


def test_popcosmos_gas_grid_axis_mismatch_fails(tmp_path) -> None:
    ssp_path = tmp_path / "ssp_chabrier.h5"
    gas_path = tmp_path / "gas_chabrier.h5"
    wave, lg_age, lgmet = _write_synthetic_ssp_hdf5(ssp_path)
    _write_synthetic_gas_hdf5(gas_path, wave, lg_age, lgmet, wave_offset=1.0)

    with pytest.raises(ValueError, match="axis is incompatible"):
        load_context(
            str(ssp_path),
            {"wide": _wide_filter()},
            model_config={
                "sfh_model": "popcosmos_bins",
                "stellar_metallicity_model": "single",
                "nebular_model": "gas_grid",
                "gas_grid_path": str(gas_path),
                "agn_model": "none",
                "z_sun": 0.0142,
            },
        )


def test_popcosmos_gas_grid_missing_units_fail(tmp_path) -> None:
    ssp_path = tmp_path / "ssp_chabrier.h5"
    gas_path = tmp_path / "gas_chabrier.h5"
    wave, lg_age, lgmet = _write_synthetic_ssp_hdf5(ssp_path)
    _write_synthetic_gas_hdf5(gas_path, wave, lg_age, lgmet, include_units=False)

    with pytest.raises(ValueError, match="unit metadata"):
        load_context(
            str(ssp_path),
            {"wide": _wide_filter()},
            model_config={
                "sfh_model": "popcosmos_bins",
                "stellar_metallicity_model": "single",
                "nebular_model": "gas_grid",
                "gas_grid_path": str(gas_path),
                "agn_model": "none",
                "z_sun": 0.0142,
            },
        )


def test_gas_stellar_metallicity_constraint_valid_invalid_and_grad() -> None:
    valid = {
        "log10_stellar_metallicity": -0.3,
        "log10_gas_metallicity": -0.2,
    }
    invalid = {
        "log10_stellar_metallicity": 0.0,
        "log10_gas_metallicity": -0.2,
    }
    model_config = {"sfh_model": "popcosmos_bins", "nebular_model": "gas_grid"}

    assert float(gas_metallicity_constraint_penalty_jax(valid, model_config)) == 0.0
    assert float(gas_metallicity_constraint_penalty_jax(invalid, model_config)) > 0.0
    assert (
        float(
            gas_metallicity_constraint_penalty_jax(
                valid,
                {"sfh_model": "popcosmos_bins", "nebular_model": "compressed_gas_grid"},
            )
        )
        == 0.0
    )
    grad = jax.grad(
        lambda values: gas_metallicity_constraint_penalty_jax(
            {
                "log10_stellar_metallicity": values[0],
                "log10_gas_metallicity": values[1],
            },
            model_config,
        )
    )(jnp.asarray([-0.3, -0.2]))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_gas_and_stellar_interpolation_is_grad_compatible() -> None:
    context = _synthetic_context({"nebular_model": "gas_grid"})

    def objective(values: jnp.ndarray) -> jnp.ndarray:
        gas_flux = interpolate_gas_ssp_grid_jax(context, values[0], values[1])
        stellar_flux = interpolate_ssp_stellar_metallicity_jax(
            context.ssp_lgmet_jax, gas_flux, values[2]
        )
        return jnp.sum(stellar_flux)

    grad = jax.grad(objective)(jnp.asarray([-0.5, -2.5, -1.8]))

    assert np.all(np.isfinite(np.asarray(grad)))


def test_emission_line_correction_changes_only_filter_covering_line(tmp_path) -> None:
    ssp_path = tmp_path / "ssp_chabrier.h5"
    gas_path = tmp_path / "gas_chabrier.h5"
    table_path = tmp_path / "line_corrections.csv"
    wave = np.asarray(
        [1000.0, 3000.0, 5000.0, 6000.0, 6500.0, 7000.0, 7600.0],
        dtype=np.float32,
    )
    wave, lg_age, lgmet = _write_synthetic_ssp_hdf5(ssp_path, wave=wave)
    _write_synthetic_gas_hdf5(gas_path, wave, lg_age, lgmet, enriched_line=True)
    table_path.write_text(
        "line_name,line_wavelength,fractional_correction,fractional_variance\n"
        "SYNTH_5000,5000,1.0,0.0\n",
        encoding="utf-8",
    )
    filters = {
        "line": _wide_filter("line", 7300.0, 7700.0),
        "off": _wide_filter("off", 10550.0, 11200.0),
    }
    base_model = {
        "sfh_model": "popcosmos_bins",
        "stellar_metallicity_model": "single",
        "dust_model": "prospector_fsps",
        "nebular_model": "gas_grid",
        "gas_grid_path": str(gas_path),
        "agn_model": "none",
        "z_sun": 0.0142,
        "emission_line_corrections": "none",
    }
    corrected_model = {
        **base_model,
        "emission_line_corrections": "popcosmos_table",
        "emission_line_correction_path": str(table_path),
    }
    params = _noagn_params(_popcosmos_params())
    params["tau2"] = 0.0
    params["tau1_over_tau2"] = 0.0
    raw = run_dsps_model_jax(
        load_context(str(ssp_path), filters, model_config=base_model), params
    ).model_mags
    corrected = run_dsps_model_jax(
        load_context(str(ssp_path), filters, model_config=corrected_model), params
    ).model_mags

    assert float(corrected[0]) < float(raw[0])
    assert float(corrected[1]) == pytest.approx(float(raw[1]), abs=1.0e-5)


def test_popcosmos_binned_complete_jax_grad_model_mags() -> None:
    context = _synthetic_context(
        {
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "dust_model": "charlot_fall",
            "igm_model": "none",
            "nebular_model": "gas_grid",
            "agn_model": "template_grid",
            "z_sun": 0.0134,
        }
    )
    params = _popcosmos_params()

    def objective(values: jnp.ndarray) -> jnp.ndarray:
        local = dict(params)
        local["log10_stellar_mass"] = values[0]
        local["tau2"] = values[1]
        return jnp.sum(run_dsps_model_jax(context, local).model_mags)

    mags = run_dsps_model_jax(context, params).model_mags
    grad = jax.grad(objective)(jnp.asarray([10.0, 0.2]))

    assert np.all(np.isfinite(np.asarray(mags)))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_popcosmos_binned_mags_only_matches_full_forward() -> None:
    context = _synthetic_context(
        {
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "dust_model": "prospector_fsps",
            "igm_model": "fsps_madau95",
            "nebular_model": "compressed_gas_grid",
            "agn_model": "compressed_fsps_component_grid",
            "agn_host_attenuation": "fsps_diffuse_unit_tau",
            "agn_igm_order": "fsps_after_igm",
            "z_sun": 0.0134,
        }
    )
    params = _popcosmos_params()

    full = run_dsps_model_jax(context, params).model_mags
    mags_only = run_dsps_model_mags_jax(context, params)
    grad = jax.grad(
        lambda values: jnp.sum(
            run_dsps_model_mags_jax(
                context,
                {
                    **params,
                    "log10_stellar_mass": values[0],
                    "tau2": values[1],
                },
            )
        )
    )(jnp.asarray([10.0, 0.2]))

    np.testing.assert_allclose(np.asarray(mags_only), np.asarray(full), rtol=1.0e-6)
    assert np.all(np.isfinite(np.asarray(grad)))


def test_diffstar_mags_only_matches_full_forward() -> None:
    pytest.importorskip("diffstar")
    context = _synthetic_context(
        {
            "sfh_model": "diffstar_reduced6",
            "stellar_metallicity_model": "single",
            "dust_model": "prospector_fsps",
            "igm_model": "none",
            "nebular_model": "compressed_gas_grid",
            "agn_model": "compressed_fsps_component_grid",
            "agn_host_attenuation": "none",
            "z_sun": 0.0134,
        }
    )
    params = _diffstar_params()

    full = run_dsps_model_jax(context, params).model_mags
    mags_only = run_dsps_model_mags_jax(context, params)

    np.testing.assert_allclose(np.asarray(mags_only), np.asarray(full), rtol=1.0e-6)


def test_hltds_dataset_config_loads_without_agn_or_gas_grid(tmp_path) -> None:
    config = load_config("configs/diffsky_dataset_hltds_04_14.yaml")
    ssp_path = tmp_path / "ssp_chabrier.h5"
    wave, lg_age, lgmet = _write_synthetic_ssp_hdf5(ssp_path)
    config["ssp_path"] = str(ssp_path)
    filters = {"wide": _wide_filter()}

    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"]["n_sfh_bins"]),
        model_config=config["model"],
    )
    result = run_dsps_model_jax(context, _noagn_params(_popcosmos_params()))

    assert context.agn_template_grid_jax is None
    assert np.all(np.isfinite(np.asarray(result.model_mags)))


def test_popcosmos_batch_prediction_paths_use_dynamic_context_args() -> None:
    context = _synthetic_context(
        {
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "dust_model": "charlot_fall",
            "igm_model": "none",
            "nebular_model": "gas_grid",
            "agn_model": "template_grid",
            "z_sun": 0.0134,
        }
    )
    params = _popcosmos_params()
    parameter_names = list(params)
    parameter_matrix = np.asarray(
        [[params[name] for name in parameter_names]], dtype=float
    )

    mags = predict_batch_mags(context, parameter_names, parameter_matrix)
    derived = predict_batch_derived(context, parameter_names, parameter_matrix)
    seds = predict_batch_seds(context, parameter_names, parameter_matrix)

    assert mags.shape == (1, 1)
    assert set(derived) >= {"formed_mass_msun", "surviving_stellar_mass_msun"}
    assert seds.rest_sed.shape == (1, len(context.ssp_wave_jax))
    assert seds.dusted_rest_sed.shape == (1, len(context.ssp_wave_jax))
    assert np.all(np.isfinite(mags))
    assert np.all(np.isfinite(seds.rest_sed))


def test_diffstar_sfh_positive_and_finite_when_installed() -> None:
    pytest.importorskip("diffstar")
    pytest.importorskip("diffmah")
    time = jnp.linspace(0.05, 5.0, 32)

    sfh = build_diffstar_sfh_table_jax(time, jnp.asarray(5.0), _diffstar_params())

    assert np.all(np.isfinite(np.asarray(sfh)))
    assert np.all(np.asarray(sfh) > 0.0)


def test_diffstar_complete_jax_grad_model_mags_when_installed() -> None:
    pytest.importorskip("diffstar")
    pytest.importorskip("diffmah")
    context = _synthetic_context(
        {
            "sfh_model": "diffstar_reduced6",
            "stellar_metallicity_model": "single",
            "dust_model": "charlot_fall",
            "igm_model": "none",
            "nebular_model": "gas_grid",
            "agn_model": "template_grid",
            "z_sun": 0.0134,
        }
    )
    params = _diffstar_params()

    def objective(values: jnp.ndarray) -> jnp.ndarray:
        local = dict(params)
        local["log10_stellar_mass"] = values[0]
        local["tau2"] = values[1]
        return jnp.sum(run_dsps_model_jax(context, local).model_mags)

    result = run_dsps_model_jax(context, params)
    grad = jax.grad(objective)(jnp.asarray([10.0, 0.2]))

    assert result.sfr_bins_msun_per_yr.shape == (7,)
    assert result.lookback_bin_edges_gyr.shape == (8,)
    assert np.all(np.isfinite(np.asarray(result.model_mags)))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_popcosmos_agn_component_outputs_are_separated() -> None:
    context = _synthetic_context(
        {
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "dust_model": "prospector_fsps",
            "igm_model": "none",
            "nebular_model": "fixed_ssp",
            "agn_model": "template_grid",
            "agn_host_attenuation": "none",
            "z_sun": 0.0134,
        }
    )
    params = _popcosmos_params()

    result = run_dsps_model_jax(context, params)

    assert result.stellar_intrinsic_sed is not None
    assert result.stellar_dusted_sed is not None
    assert result.gas_sed is not None
    assert result.agn_sed is not None
    assert result.pre_igm_sed is not None
    np.testing.assert_allclose(
        np.asarray(result.gas_sed),
        np.zeros_like(np.asarray(result.gas_sed)),
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(result.pre_igm_sed),
        np.asarray(result.dusted_rest_sed),
        rtol=1.0e-6,
    )
    assert np.nanmax(np.asarray(result.agn_sed)) > 0.0


def test_agn_host_attenuation_can_reduce_agn_component() -> None:
    base_config = {
        "sfh_model": "popcosmos_bins",
        "stellar_metallicity_model": "single",
        "dust_model": "prospector_fsps",
        "igm_model": "none",
        "nebular_model": "fixed_ssp",
        "agn_model": "template_grid",
        "z_sun": 0.0134,
    }
    params = _popcosmos_params()
    params["tau2"] = 1.0
    no_host = _synthetic_context({**base_config, "agn_host_attenuation": "none"})
    with_host = _synthetic_context(
        {**base_config, "agn_host_attenuation": "prospector_fsps"}
    )

    agn_no_host = np.asarray(run_dsps_model_jax(no_host, params).agn_sed)
    agn_with_host = np.asarray(run_dsps_model_jax(with_host, params).agn_sed)

    assert np.nanmax(agn_with_host) < np.nanmax(agn_no_host)


def test_agn_host_attenuation_scale_interpolates_between_none_and_full() -> None:
    base_config = {
        "sfh_model": "popcosmos_bins",
        "stellar_metallicity_model": "single",
        "dust_model": "prospector_fsps",
        "igm_model": "none",
        "nebular_model": "fixed_ssp",
        "agn_model": "template_grid",
        "agn_host_attenuation": "prospector_fsps",
        "z_sun": 0.0134,
    }
    params = _popcosmos_params()
    params["tau2"] = 1.0
    no_scale = _synthetic_context({**base_config, "agn_host_attenuation_scale": 0.0})
    half_scale = _synthetic_context({**base_config, "agn_host_attenuation_scale": 0.5})
    full_scale = _synthetic_context({**base_config, "agn_host_attenuation_scale": 1.0})

    agn_no_scale = np.asarray(run_dsps_model_jax(no_scale, params).agn_sed)
    agn_half_scale = np.asarray(run_dsps_model_jax(half_scale, params).agn_sed)
    agn_full_scale = np.asarray(run_dsps_model_jax(full_scale, params).agn_sed)

    assert np.nanmax(agn_no_scale) > np.nanmax(agn_half_scale)
    assert np.nanmax(agn_half_scale) > np.nanmax(agn_full_scale)


def test_fsps_unit_tau_agn_host_attenuation_is_tau2_independent() -> None:
    base_config = {
        "sfh_model": "popcosmos_bins",
        "stellar_metallicity_model": "single",
        "dust_model": "prospector_fsps",
        "igm_model": "none",
        "nebular_model": "fixed_ssp",
        "agn_model": "template_grid",
        "agn_host_attenuation": "fsps_diffuse_unit_tau",
        "z_sun": 0.0134,
    }
    context = _synthetic_context(base_config)
    params_low_tau = _popcosmos_params()
    params_high_tau = dict(params_low_tau)
    params_low_tau["tau2"] = 0.1
    params_high_tau["tau2"] = 3.0

    agn_low_tau = np.asarray(run_dsps_model_jax(context, params_low_tau).agn_sed)
    agn_high_tau = np.asarray(run_dsps_model_jax(context, params_high_tau).agn_sed)

    np.testing.assert_allclose(agn_low_tau, agn_high_tau, rtol=1.0e-6, atol=0.0)

    no_host = _synthetic_context({**base_config, "agn_host_attenuation": "none"})
    agn_no_host = np.asarray(run_dsps_model_jax(no_host, params_low_tau).agn_sed)

    assert np.nanmax(agn_low_tau) < np.nanmax(agn_no_host)


def test_fsps_unit_tau_can_replace_baked_powerlaw_agn_attenuation() -> None:
    wave = jnp.asarray([1500.0, 5500.0, 10000.0], dtype=jnp.float32)
    agn = jnp.ones(3, dtype=jnp.float32)
    params = {"dust_index_n": -0.7}
    target_only = apply_agn_host_attenuation_jax(
        wave,
        agn,
        params,
        {
            "agn_host_attenuation": "fsps_diffuse_unit_tau",
            "agn_baked_attenuation": "none",
        },
    )
    replace_baked = apply_agn_host_attenuation_jax(
        wave,
        agn,
        params,
        {
            "agn_host_attenuation": "fsps_diffuse_unit_tau",
            "agn_baked_attenuation": "fsps_powerlaw_unit_tau",
            "agn_baked_dust_index": -0.7,
        },
    )

    assert float(replace_baked[-1]) > float(target_only[-1])
    assert np.all(np.isfinite(np.asarray(replace_baked)))


def test_agn_component_interpolates_audit_template_axes() -> None:
    context = _synthetic_context(
        {
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "dust_model": "prospector_fsps",
            "igm_model": "none",
            "nebular_model": "fixed_ssp",
            "agn_model": "template_grid",
            "agn_host_attenuation": "none",
            "z_sun": 0.0134,
        }
    )
    wave = np.asarray(context.ssp_wave_jax)
    fagn_grid = np.asarray([1.0e-4, 1.0e-2], dtype=float)
    tau_grid = np.asarray([5.0, 20.0], dtype=float)
    age_grid = np.asarray([1.0, 5.0], dtype=float)
    logz_grid = np.asarray([-1.0, 0.0], dtype=float)
    template = np.zeros(
        (len(fagn_grid), len(tau_grid), len(age_grid), len(logz_grid), len(wave)),
        dtype=np.float32,
    )
    for i, _fagn in enumerate(fagn_grid):
        for j, _tau in enumerate(tau_grid):
            for k, _age in enumerate(age_grid):
                for m, _logz in enumerate(logz_grid):
                    template[i, j, k, m, :] = (
                        1.0e-12
                        * (1.0 + i + 0.1 * j + 0.01 * k + 0.001 * m)
                        * (1.0 + wave / wave.max())
                    )
    context.agn_fagn_grid_jax = jnp.asarray(fagn_grid, dtype=jnp.float32)
    context.agn_tau_grid_jax = jnp.asarray(tau_grid, dtype=jnp.float32)
    context.agn_tage_grid_jax = jnp.asarray(age_grid, dtype=jnp.float32)
    context.agn_logzsol_grid_jax = jnp.asarray(logz_grid, dtype=jnp.float32)
    context.agn_template_grid_jax = jnp.asarray(template, dtype=jnp.float32)
    params = _popcosmos_params()
    params["ln_fagn"] = np.log(1.0e-3)
    params["ln_tauagn"] = np.log(10.0)

    component = agn_component_jax(
        context,
        context.ssp_wave_jax,
        jnp.ones_like(context.ssp_wave_jax),
        params,
        context.model_config,
        template_tage_gyr=2.0,
        stellar_logzsol=-0.5,
    )

    assert component.shape == context.ssp_wave_jax.shape
    assert np.all(np.isfinite(np.asarray(component)))
    assert np.nanmax(np.asarray(component)) > 0.0


def test_fsps_component_grid_agn_scales_with_formed_mass() -> None:
    context = _synthetic_context(
        {
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "dust_model": "prospector_fsps",
            "igm_model": "none",
            "nebular_model": "fixed_ssp",
            "agn_model": "fsps_component_grid",
            "agn_host_attenuation": "none",
            "z_sun": 0.0134,
        }
    )
    params_low = _popcosmos_params()
    params_low["ln_fagn"] = np.log(1.0e-3)
    params_low["ln_tauagn"] = np.log(10.0)
    params_high = dict(params_low)
    params_high["log10_stellar_mass"] = params_low["log10_stellar_mass"] + 1.0

    low = np.asarray(run_dsps_model_jax(context, params_low).agn_sed)
    high = np.asarray(run_dsps_model_jax(context, params_high).agn_sed)

    assert np.nanmax(low) > 0.0
    np.testing.assert_allclose(high / low, np.full_like(low, 10.0), rtol=5.0e-4)


def test_compressed_fsps_component_grid_matches_synthetic_dense_path() -> None:
    dense_context = _synthetic_context(
        {
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "dust_model": "prospector_fsps",
            "igm_model": "none",
            "nebular_model": "fixed_ssp",
            "agn_model": "fsps_component_grid",
            "agn_host_attenuation": "none",
            "z_sun": 0.0134,
        }
    )
    compressed_context = _synthetic_context(
        {
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "dust_model": "prospector_fsps",
            "igm_model": "none",
            "nebular_model": "fixed_ssp",
            "agn_model": "compressed_fsps_component_grid",
            "agn_host_attenuation": "none",
            "z_sun": 0.0134,
        }
    )
    params = _popcosmos_params()
    params["ln_fagn"] = np.log(1.0e-3)
    params["ln_tauagn"] = np.log(10.0)

    dense = np.asarray(run_dsps_model_jax(dense_context, params).agn_sed)
    compressed = np.asarray(run_dsps_model_jax(compressed_context, params).agn_sed)

    assert compressed_context.agn_component_grid_jax is None
    assert compressed_context.compressed_agn_coeff_jax is not None
    np.testing.assert_allclose(compressed, dense, rtol=5.0e-5, atol=1.0e-8)


def test_load_context_compressed_agn_component_does_not_load_dense_grid(tmp_path) -> None:
    ssp_path = tmp_path / "ssp.h5"
    wave, lg_age, lgmet = _write_synthetic_ssp_hdf5(ssp_path)
    agn_path = tmp_path / "compressed_agn.h5"
    fagn = np.asarray([1.0e-4, 1.0e-2], dtype=np.float32)
    tau = np.asarray([5.0, 20.0], dtype=np.float32)
    basis = np.vstack(
        [
            1.0 + wave / wave.max(),
            0.2 + 0.1 * wave / wave.max(),
        ]
    ).astype(np.float32)
    coeff = np.ones((len(fagn), len(tau), len(lgmet), len(lg_age), 2), dtype=np.float32)
    scale = np.ones(coeff.shape[:-1], dtype=np.float32)
    with h5py.File(agn_path, "w") as handle:
        handle["ssp_wave"] = wave
        handle["ssp_lg_age_gyr"] = lg_age
        handle["ssp_lgmet"] = lgmet
        handle["fagn_grid"] = fagn
        handle["agn_tau_grid"] = tau
        handle["agn_basis"] = basis
        handle["agn_coeff"] = coeff
        handle["agn_scale"] = scale
        handle.attrs["imf_type"] = 1
        handle.attrs["imf_name"] = "chabrier"
        handle.attrs["z_sun"] = 0.0142

    context = load_context(
        str(ssp_path),
        {"wide": _wide_filter()},
        model_config={
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "nebular_model": "fixed_ssp",
            "agn_model": "compressed_fsps_component_grid",
            "compressed_agn_component_grid_path": str(agn_path),
            "z_sun": 0.0142,
        },
    )

    assert context.agn_component_grid_jax is None
    assert context.compressed_agn_basis_jax is not None
    assert context.compressed_agn_coeff_jax is not None
    assert context.compressed_agn_scale_jax is not None


def test_legacy_lognormal_config_still_runs() -> None:
    context = _synthetic_context({"sfh_model": "lognormal"})
    params = {
        "z_obs": 0.5,
        "log10_sfr": 0.0,
        "sfh_t_peak": 4.0,
        "sfh_tau": 0.6,
        "log10_metallicity": -2.0,
        "metallicity_scatter": 0.2,
        "dust_av": 0.1,
        "dust_slope": -0.7,
        "log10_formed_mass_msun": 10.0,
    }

    result = run_dsps_model_jax(context, params)

    assert np.all(np.isfinite(np.asarray(result.model_mags)))
    assert float(result.formed_mass_msun) == pytest.approx(1.0e10, rel=1.0e-5)


def test_cosmos_two_component_dust_mixes_configured_curves() -> None:
    rest_sed = np.ones(3)
    k_by_code = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0],
        ]
    )

    attenuated = np.asarray(
        apply_cosmos_two_component_dust_jax(
            rest_sed,
            {
                "cosmos_ext_curve_1": 1.0,
                "cosmos_ext_curve_2": 2.0,
                "cosmos_ebv_1": 0.5,
                "cosmos_ebv_2": 0.25,
                "cosmos_frac_1": 2.0,
                "cosmos_frac_2": 1.0,
            },
            k_by_code,
        )
    )

    expected = (2.0 / 3.0) * 10 ** (-0.4 * 0.5 * 2.0) + (1.0 / 3.0) * 10 ** (
        -0.4 * 0.25 * 4.0
    )
    assert attenuated.tolist() == pytest.approx([expected, expected, expected])


def test_sfh_can_be_normalized_to_formed_mass() -> None:
    time = np.linspace(0.1, 10.0, 128)
    sfr = build_lognormal_sfh(time, 0.0, 4.0, 0.6)

    scaled, formed_mass = normalize_sfh_mass_jax(
        time, sfr, {"log10_formed_mass_msun": 10.0}
    )

    assert float(formed_mass) == pytest.approx(1.0e10, rel=1.0e-5)
    assert np.trapezoid(np.asarray(scaled), time) * 1.0e9 == pytest.approx(
        1.0e10, rel=1.0e-5
    )


def test_parameters_for_row_adds_redshift_prior_sigma() -> None:
    params = parameters_for_row(
        {"z_obs": 0.5, "log10_formed_mass_msun": 10.0},
        {},
        {"phz_median": 1.0},
        {
            "initial": "catalog_column",
            "column": "phz_median",
            "fixed_value": 0.5,
            "min": 0.001,
            "max": 6.0,
            "prior_z": {
                "mode": "gaussian",
                "sigma": 0.25,
                "sigma_min": 0.05,
                "scale_with_1pz": True,
            },
        },
    )

    assert params["z_obs"] == pytest.approx(1.0)
    assert params["z_obs_prior_mu"] == pytest.approx(1.0)
    assert params["z_obs_prior_sigma"] == pytest.approx(0.5)


def test_random_uniform_redshift_uses_identifier_not_truth_redshift() -> None:
    redshift_config = {
        "initial": "random_uniform",
        "column": "redshift",
        "truth_column": "redshift",
        "fixed_value": 0.5,
        "min": 0.001,
        "max": 2.5,
        "seed": 42,
        "prior_z": {"mode": "none"},
    }

    params_a = parameters_for_row(
        {"z_obs": 0.5, "log10_stellar_mass": 8.0},
        {},
        {"galaxy_id": 123, "redshift": 0.2, "redshiftHubble": 0.21},
        redshift_config,
    )
    params_b = parameters_for_row(
        {"z_obs": 0.5, "log10_stellar_mass": 8.0},
        {},
        {"galaxy_id": 123, "redshift": 1.8, "redshiftHubble": 1.82},
        redshift_config,
    )
    params_c = parameters_for_row(
        {"z_obs": 0.5, "log10_stellar_mass": 8.0},
        {},
        {"galaxy_id": 456, "redshift": 0.2, "redshiftHubble": 0.21},
        redshift_config,
    )

    assert params_a["z_obs"] == pytest.approx(params_b["z_obs"])
    assert params_a["z_obs"] != pytest.approx(params_c["z_obs"])


def test_comparison_rows_include_flux_error_and_chi_flux() -> None:
    observation = GalaxyObservation(
        row_index=0,
        row={},
        bands=[
            BandObservation(
                name="vis",
                column="vis",
                flux_fnu_cgs=10.0,
                mag_ab=25.0,
                sigma_mag=0.1,
                flux_error_fnu_cgs=2.0,
            )
        ],
    )
    result = ModelResult(
        parameters={"z_obs": 0.5},
        derived={},
        wave=np.asarray([1000.0, 2000.0]),
        rest_sed=np.ones(2),
        dusted_rest_sed=np.ones(2),
        photometry={
            "vis": {
                "effective_wavelength_angstrom": 1500.0,
                "model_flux_fnu_cgs": 14.0,
                "model_mag_ab": 24.5,
                "filter_source": "test",
            }
        },
    )

    rows = comparison_rows(observation, result)

    assert rows[0]["observed_flux_error_fnu_cgs"] == pytest.approx(2.0)
    assert rows[0]["chi_flux"] == pytest.approx(2.0)
