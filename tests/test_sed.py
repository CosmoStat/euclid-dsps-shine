from __future__ import annotations

import math

import numpy as np
import pytest

from euclid_dsps.io import BandObservation, GalaxyObservation
from euclid_dsps.model import ModelResult
from euclid_dsps.reporting.forward import write_sed_comparison_outputs
from euclid_dsps.sed import (
    empirical_sed_points,
    interpolate_empirical_lnu,
    observed_fnu_to_rest_lnu_lsun,
    reconstruct_empirical_sed,
    rest_10pc_fnu_to_lnu_lsun,
    rest_lnu_lsun_to_observed_fnu,
)


def test_observed_flux_rest_luminosity_roundtrip() -> None:
    flux = rest_lnu_lsun_to_observed_fnu(3.0e-10, redshift=0.7)

    assert observed_fnu_to_rest_lnu_lsun(flux, redshift=0.7) == pytest.approx(3.0e-10)


def test_empirical_sed_points_compare_to_dsps_at_rest_wavelengths() -> None:
    observation, result = _toy_observation_and_result()

    points = empirical_sed_points(observation, result)

    assert points["rest_wavelength_angstrom"].tolist() == pytest.approx(
        [3000.0, 5000.0]
    )
    assert points["inferred_rest_lnu_lsun_per_hz"].tolist() == pytest.approx([2.0, 8.0])
    assert points["dsps_dusted_lnu_lsun_per_hz"].tolist() == pytest.approx([2.0, 8.0])
    assert points["log10_empirical_over_dsps_dusted"].tolist() == pytest.approx(
        [0.0, 0.0]
    )


def test_empirical_sed_points_prefer_rest_10pc_flux_when_present() -> None:
    observation, result = _toy_observation_and_result()
    observation = GalaxyObservation(
        row_index=observation.row_index,
        row={"z_phz": 0.5, "euclid_vis_abs": 1.0e-18},
        bands=observation.bands,
    )

    points = empirical_sed_points(observation, result)
    first = points.iloc[0]

    assert first["sed_flux_source_kind"] == "rest_10pc_flux"
    assert first["sed_flux_source_column"] == "euclid_vis_abs"
    assert first["rest_wavelength_angstrom"] == pytest.approx(4500.0)
    assert first["inferred_rest_lnu_lsun_per_hz"] == pytest.approx(
        rest_10pc_fnu_to_lnu_lsun(1.0e-18)
    )


def test_reconstruct_empirical_sed_interpolates_in_log_space() -> None:
    wave = np.asarray([3000.0, math.sqrt(3000.0 * 5000.0), 5000.0])
    empirical = interpolate_empirical_lnu(
        wave,
        np.asarray([3000.0, 5000.0]),
        np.asarray([2.0, 8.0]),
    )

    assert empirical[0] == pytest.approx(2.0)
    assert empirical[1] == pytest.approx(4.0)
    assert empirical[2] == pytest.approx(8.0)


def test_write_sed_comparison_outputs(tmp_path) -> None:
    observation, result = _toy_observation_and_result()

    empirical = write_sed_comparison_outputs(observation, result, tmp_path)

    assert empirical.summary["n_photometric_points"] == 2
    assert (tmp_path / "empirical_sed_points.csv").exists()
    assert (tmp_path / "empirical_sed.csv").exists()
    assert (tmp_path / "empirical_sed_summary.json").exists()
    assert (tmp_path / "sed_comparison.png").exists()


def test_reconstruct_empirical_sed_returns_continuous_comparison() -> None:
    observation, result = _toy_observation_and_result()

    empirical = reconstruct_empirical_sed(observation, result)

    assert "empirical_over_dsps_dusted" in empirical.continuous.columns
    finite = empirical.continuous["empirical_over_dsps_dusted"].dropna()
    assert not finite.empty
    assert finite.median() == pytest.approx(1.0)


def _toy_observation_and_result() -> tuple[GalaxyObservation, ModelResult]:
    z = 0.5
    lnu = [2.0, 8.0]
    observation = GalaxyObservation(
        row_index=3,
        row={"z_phz": z},
        bands=[
            BandObservation(
                name="euclid_vis",
                column="euclid_vis",
                flux_fnu_cgs=rest_lnu_lsun_to_observed_fnu(lnu[0], z),
                mag_ab=24.0,
                sigma_mag=0.05,
            ),
            BandObservation(
                name="euclid_nisp_h",
                column="euclid_nisp_h",
                flux_fnu_cgs=rest_lnu_lsun_to_observed_fnu(lnu[1], z),
                mag_ab=23.0,
                sigma_mag=0.05,
            ),
        ],
    )
    wave = np.asarray([2500.0, 3000.0, 4000.0, 5000.0, 5500.0])
    dusted = np.asarray([1.5, 2.0, 4.0, 8.0, 9.0])
    return observation, ModelResult(
        parameters={"z_obs": z},
        derived={},
        wave=wave,
        rest_sed=dusted * 1.2,
        dusted_rest_sed=dusted,
        photometry={
            "euclid_vis": {
                "model_mag_ab": 24.0,
                "model_flux_fnu_cgs": observation.bands[0].flux_fnu_cgs,
                "filter_source": "toy",
                "effective_wavelength_angstrom": 4500.0,
                "filter_wave_angstrom": np.asarray([4300.0, 4500.0, 4700.0]),
                "filter_transmission": np.asarray([0.0, 1.0, 0.0]),
            },
            "euclid_nisp_h": {
                "model_mag_ab": 23.0,
                "model_flux_fnu_cgs": observation.bands[1].flux_fnu_cgs,
                "filter_source": "toy",
                "effective_wavelength_angstrom": 7500.0,
                "filter_wave_angstrom": np.asarray([7300.0, 7500.0, 7700.0]),
                "filter_transmission": np.asarray([0.0, 1.0, 0.0]),
            },
        },
    )
