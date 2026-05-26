from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from euclid_dsps.config import normalize_config, validate_catalog_columns
from euclid_dsps.io import read_catalog
from euclid_dsps.selection import select_galaxy_row
from euclid_dsps.workflows.eda import run_eda

FIXTURE = Path(__file__).parent / "data" / "synthetic_catalog.parquet"


def synthetic_config() -> dict:
    return normalize_config(
        {
            "catalog_path": str(FIXTURE),
            "ssp_path": "Data/ssp_data_fsps_v3.2_lgmet_age.h5",
            "selection": {"index": 1, "require_positive_flux": True},
            "eda": {"nrows": 3},
            "extra_columns": ["ra_gal", "dec_gal"],
            "redshift": {"column": "z_phz", "truth_column": "z_true"},
            "bands": [
                {
                    "name": "euclid_vis",
                    "column": "euclid_vis",
                    "units": "fnu_cgs",
                    "sigma_mag": 0.05,
                    "filter": {"kind": "tophat"},
                },
                {
                    "name": "euclid_nisp_y",
                    "column": "euclid_nisp_y",
                    "units": "fnu_cgs",
                    "sigma_mag": 0.05,
                    "filter": {"kind": "tophat"},
                },
            ],
            "truth": {
                "parameter_columns": {
                    "log10_metallicity": {
                        "column": "metallicity_true",
                        "offset": -10.61,
                    },
                    "log10_sfr_at_obs": "log_sfr_true",
                    "dust_av": {"column": "dust_ebv_true", "scale": 4.05},
                }
            },
        }
    )


def test_synthetic_fixture_schema_matches_config() -> None:
    config = synthetic_config()
    df = pd.read_parquet(FIXTURE)

    validate_catalog_columns(config, set(df.columns))


def test_read_catalog_derives_log10_metallicity_true() -> None:
    df = read_catalog(FIXTURE, columns=["metallicity_true"], nrows=2)

    assert df["log10_metallicity_true"].tolist() == pytest.approx([-2.41, -2.21])


def test_select_galaxy_row_uses_configured_index() -> None:
    df = pd.read_parquet(FIXTURE)

    row_index, row = select_galaxy_row(
        df,
        band_columns=["euclid_vis", "euclid_nisp_y"],
        index=1,
        require_positive_flux=True,
    )

    assert row_index == 1
    assert row["z_phz"] == 0.5


def test_select_galaxy_row_keeps_negative_flux_with_gaussian_policy() -> None:
    df = pd.DataFrame(
        {"band_a": [-1.0, 2.0], "band_b": [1.0, 2.0]},
        index=[10, 11],
    )

    row_index, row = select_galaxy_row(
        df,
        band_columns=["band_a", "band_b"],
        index=10,
        require_positive_flux=False,
        nondetection_policy="gaussian_flux",
    )

    assert row_index == 10
    assert row["band_a"] == -1.0


def test_run_eda_writes_expected_outputs(tmp_path) -> None:
    config = synthetic_config()
    out = tmp_path / "eda"

    run_eda(config, out)

    expected = {
        "catalog_schema.json",
        "catalog_stats.csv",
        "missing_values.csv",
        "flux_distributions.png",
        "color_distributions.png",
        "redshift_diagnostics.png",
        "physical_parameters.png",
    }
    assert expected.issubset({item.name for item in out.iterdir()})
