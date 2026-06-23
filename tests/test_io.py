from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from euclid_dsps.io import (
    abmag_to_flux_fnu_cgs,
    build_observation,
    flux_error_to_sigma_mag,
    flux_fnu_cgs_to_abmag,
    iter_catalog_batches,
    load_row_indices,
    microjy_to_abmag,
    microjy_to_flux_fnu_cgs,
    truth_value_from_spec,
)


def test_ab_magnitude_flux_roundtrip() -> None:
    flux = abmag_to_flux_fnu_cgs(0.0)

    assert flux == pytest.approx(3631.0e-23)
    assert math.isclose(flux_fnu_cgs_to_abmag(flux), 0.0, abs_tol=1.0e-12)


def test_microjy_conversions() -> None:
    assert math.isclose(microjy_to_flux_fnu_cgs(1.0), 1.0e-29)
    assert microjy_to_abmag(1.0) == pytest.approx(23.90006562228223)
    assert np.isnan(microjy_to_abmag(-1.0))


def test_truth_value_transform_scale_offset_and_log10() -> None:
    row = {"sfr": 100.0, "dust": 0.2}

    assert truth_value_from_spec(row, {"column": "sfr", "transform": "log10"}) == 2.0
    assert truth_value_from_spec(row, {"column": "dust", "scale": 4.05}) == 0.81
    assert truth_value_from_spec(row, {"column": "dust", "offset": 1.0}) == 1.2
    assert truth_value_from_spec(row, {"column": "missing"}) is None


def test_truth_value_converts_log_stellar_mass_h2_to_msun() -> None:
    row = {"log_stellar_mass": 10.0}

    value = truth_value_from_spec(
        row,
        {
            "column": "log_stellar_mass",
            "transform": "log_stellar_mass_h2_to_msun",
            "h": 0.67,
        },
    )

    assert value == pytest.approx(10.0 + 2.0 * np.log10(0.67))


def test_load_row_indices_deduplicates_and_sorts(tmp_path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("# comment\n7\n3\n7\n", encoding="utf-8")

    assert load_row_indices(path) == [3, 7]


def test_iter_catalog_batches_supports_start_index_and_limit(tmp_path) -> None:
    path = tmp_path / "catalog.parquet"
    pd.DataFrame({"value": np.arange(10)}).to_parquet(path)

    batches = list(
        iter_catalog_batches(
            path,
            columns=["value"],
            batch_size=4,
            start_index=3,
            limit=5,
        )
    )
    frame = pd.concat(batches)

    assert frame.index.tolist() == [3, 4, 5, 6, 7]
    assert frame["value"].tolist() == [3, 4, 5, 6, 7]


def test_iter_catalog_batches_coalesces_sparse_row_indices(tmp_path) -> None:
    path = tmp_path / "catalog.parquet"
    pd.DataFrame({"value": np.arange(20)}).to_parquet(path)

    batches = list(
        iter_catalog_batches(
            path,
            columns=["value"],
            batch_size=4,
            row_indices={1, 7, 13, 19},
        )
    )

    assert len(batches) == 1
    assert batches[0].index.tolist() == [1, 7, 13, 19]
    assert batches[0]["value"].tolist() == [1, 7, 13, 19]


def test_iter_catalog_batches_coalesced_row_indices_respect_limit(tmp_path) -> None:
    path = tmp_path / "catalog.parquet"
    pd.DataFrame({"value": np.arange(30)}).to_parquet(path)

    batches = list(
        iter_catalog_batches(
            path,
            columns=["value"],
            batch_size=4,
            limit=5,
            row_indices={1, 7, 13, 19, 23, 29},
        )
    )
    frame = pd.concat(batches)

    assert [len(batch) for batch in batches] == [4, 1]
    assert frame.index.tolist() == [1, 7, 13, 19, 23]
    assert frame["value"].tolist() == [1, 7, 13, 19, 23]


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
    assert obs.bands[2].mag_ab == pytest.approx(23.90006562228223)


def test_build_observation_uses_catalog_flux_error_for_sigma_mag() -> None:
    row = pd.Series({"flux": 2.0e-29, "flux_err": 2.0e-30})
    bands = [
        {
            "name": "euclid_vis",
            "column": "flux",
            "units": "fnu_cgs",
            "sigma_mag": 0.5,
            "error_column": "flux_err",
            "error_units": "fnu_cgs",
            "sigma_mag_floor": 0.01,
            "sigma_mag_ceiling": 0.3,
        }
    ]

    obs = build_observation(0, row, bands)

    expected = flux_error_to_sigma_mag(2.0e-29, 2.0e-30, floor=0.01, ceiling=0.3)
    assert obs.bands[0].sigma_mag == pytest.approx(expected)
    assert obs.bands[0].flux_error_fnu_cgs == pytest.approx(2.0e-30)
