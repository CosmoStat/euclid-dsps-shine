from __future__ import annotations

import numpy as np
import pytest

from euclid_dsps.io import BandObservation, GalaxyObservation
from euclid_dsps.model import (
    ModelResult,
    apply_cosmos_two_component_dust_jax,
    build_lognormal_sfh,
    comparison_rows,
    normalize_sfh_mass_jax,
    parameters_for_row,
)


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
