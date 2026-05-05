from __future__ import annotations

import math

import numpy as np
import pandas as pd

from euclid_dsps.io import (
    abmag_to_flux_fnu_cgs,
    build_observation,
    flux_fnu_cgs_to_abmag,
    load_row_indices,
    microjy_to_abmag,
    microjy_to_flux_fnu_cgs,
    truth_value_from_spec,
)


def test_ab_magnitude_flux_roundtrip() -> None:
    flux = abmag_to_flux_fnu_cgs(0.0)

    assert math.isclose(flux, 10 ** (-0.4 * 48.6))
    assert math.isclose(flux_fnu_cgs_to_abmag(flux), 0.0, abs_tol=1.0e-12)


def test_microjy_conversions() -> None:
    assert math.isclose(microjy_to_flux_fnu_cgs(1.0), 1.0e-29)
    assert math.isclose(microjy_to_abmag(1.0), 23.9)
    assert np.isnan(microjy_to_abmag(-1.0))


def test_truth_value_transform_scale_offset_and_log10() -> None:
    row = {"sfr": 100.0, "dust": 0.2}

    assert truth_value_from_spec(row, {"column": "sfr", "transform": "log10"}) == 2.0
    assert truth_value_from_spec(row, {"column": "dust", "scale": 4.05}) == 0.81
    assert truth_value_from_spec(row, {"column": "dust", "offset": 1.0}) == 1.2
    assert truth_value_from_spec(row, {"column": "missing"}) is None


def test_load_row_indices_deduplicates_and_sorts(tmp_path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("# comment\n7\n3\n7\n", encoding="utf-8")

    assert load_row_indices(path) == [3, 7]


def test_build_observation_supports_configured_units() -> None:
    row = pd.Series({"fnu": 1.0e-29, "mag": 23.0, "ujy": 1.0})
    bands = [
        {"name": "fnu", "column": "fnu", "units": "fnu_cgs", "sigma_mag": 0.1},
        {"name": "mag", "column": "mag", "units": "abmag", "sigma_mag": 0.2},
        {"name": "ujy", "column": "ujy", "units": "microjy", "sigma_mag": 0.3},
    ]

    obs = build_observation(11, row, bands)

    assert obs.row_index == 11
    assert [band.name for band in obs.bands] == ["fnu", "mag", "ujy"]
    assert math.isclose(obs.bands[1].flux_fnu_cgs, abmag_to_flux_fnu_cgs(23.0))
    assert math.isclose(obs.bands[2].mag_ab, 23.9)
