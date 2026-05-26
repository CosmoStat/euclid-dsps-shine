from __future__ import annotations

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.filters import FilterCurve
from euclid_dsps.io import BandObservation, GalaxyObservation
from euclid_dsps.model import (
    DspsContext,
    ModelResult,
    _load_gas_ssp_grid,
    apply_charlot_fall_by_age_jax,
    apply_cosmos_two_component_dust_jax,
    apply_igm_transmission_jax,
    build_diffstar_sfh_table_jax,
    build_lognormal_sfh,
    build_popcosmos_lookback_bin_edges_jax,
    build_popcosmos_sfh_table_jax,
    comparison_rows,
    interpolate_gas_ssp_grid_jax,
    interpolate_ssp_stellar_metallicity_jax,
    logsfr_ratios_to_sfr_bins_jax,
    normalize_sfh_mass_jax,
    normalize_sfh_to_stellar_mass_jax,
    parameters_for_row,
    predict_batch_derived,
    predict_batch_mags,
    predict_batch_seds,
    project_sfh_to_popcosmos_dlogsfr_jax,
    project_sfh_to_popcosmos_sfr_bins_jax,
    run_dsps_model_jax,
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
    return context


def _popcosmos_params() -> dict[str, float]:
    values = {
        "z_obs": 0.5,
        "log10_stellar_mass": 10.0,
        "log10_stellar_metallicity": 0.0,
        "tau2": 0.2,
        "dust_index_n": -0.7,
        "tau1_over_tau2": 1.0,
        "log10_gas_metallicity": -0.2,
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
        "log10_gas_metallicity": -0.2,
        "log10_gas_ionization": -2.4,
        "ln_fagn": -8.0,
        "ln_tauagn": np.log(10.0),
    }
    return {name: values[name] for name in DIFFSTAR_REDUCED6_PARAMETER_NAMES}


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
