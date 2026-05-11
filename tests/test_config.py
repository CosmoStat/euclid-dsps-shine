from __future__ import annotations

import pytest

from euclid_dsps.config import (
    ConfigValidationError,
    normalize_config,
    validate_catalog_columns,
)
from euclid_dsps.io import required_catalog_columns


def minimal_config() -> dict:
    return {
        "catalog_path": "catalog.parquet",
        "ssp_path": "ssp.h5",
        "bands": [
            {
                "name": "euclid_vis",
                "column": "euclid_vis",
                "units": "fnu_cgs",
                "sigma_mag": 0.05,
                "filter": {"kind": "tophat"},
            }
        ],
        "redshift": {"column": "z_phz", "truth_column": "z_true"},
        "truth": {
            "parameter_columns": {"dust_av": {"column": "dust_ebv_true", "scale": 4.05}}
        },
        "extra_columns": ["ra_gal", "dec_gal"],
    }


def test_normalize_config_adds_defaults() -> None:
    config = normalize_config(minimal_config())

    assert config["model"]["n_sfh_bins"] == 96
    assert config["fit"]["method"] == "jax_adam"
    assert config["sample"]["sampler"] == "nuts"
    assert (
        config["redshift"]["fixed_value"]
        == config["model"]["fixed_parameters"]["z_obs"]
    )


def test_invalid_band_units_fail_fast() -> None:
    config = minimal_config()
    config["bands"][0]["units"] = "watts"

    with pytest.raises(ConfigValidationError, match="bands\\[0\\]\\.units"):
        normalize_config(config)


def test_invalid_fit_bounds_fail_fast() -> None:
    config = minimal_config()
    config["fit"] = {
        "free_parameters": {
            "dust_av": {"initial": 0.2, "bounds": [1.0, 0.0]},
        }
    }

    with pytest.raises(ConfigValidationError, match="bounds must be increasing"):
        normalize_config(config)


def test_required_catalog_columns_include_config_contract() -> None:
    config = normalize_config(minimal_config())

    assert required_catalog_columns(config) == [
        "dec_gal",
        "dust_ebv_true",
        "euclid_vis",
        "ra_gal",
        "z_phz",
        "z_true",
    ]


def test_validate_catalog_columns_reports_missing_columns() -> None:
    config = normalize_config(minimal_config())

    with pytest.raises(ConfigValidationError, match="dust_ebv_true"):
        validate_catalog_columns(config, {"euclid_vis", "z_phz", "z_true", "ra_gal"})
