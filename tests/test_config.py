from __future__ import annotations

from pathlib import Path

import pytest

from euclid_dsps.config import (
    ConfigValidationError,
    load_config,
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
    assert config["runtime"]["jax_platforms"] == "cpu"
    assert config["runtime"]["disable_jax_plugin_autoload"] is True
    assert config["runtime"]["require_gpu"] is False
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


def test_redshift_prior_interval_is_removed() -> None:
    config = minimal_config()
    config["redshift"]["prior_interval"] = {
        "min_column": "phz_min_70",
        "max_column": "phz_max_70",
        "probability": 0.70,
    }

    with pytest.raises(ConfigValidationError, match="redshift\\.prior_interval"):
        normalize_config(config)


def test_sample_priors_must_match_free_parameters() -> None:
    config = minimal_config()
    config["fit"] = {
        "free_parameters": {"z_obs": {"initial": "from_base", "bounds": [0.001, 6.0]}}
    }
    config["sample"] = {"priors": {"dust_av": {"type": "uniform"}}}

    with pytest.raises(ConfigValidationError, match="sample\\.priors\\.dust_av"):
        normalize_config(config)


def test_popcosmos_prior_set_is_reserved() -> None:
    config = minimal_config()
    config["prior_set"] = "popcosmos_like"

    with pytest.raises(ConfigValidationError, match="popcosmos_like"):
        normalize_config(config)


def test_validate_catalog_columns_reports_missing_columns() -> None:
    config = normalize_config(minimal_config())

    with pytest.raises(ConfigValidationError, match="dust_ebv_true"):
        validate_catalog_columns(config, {"euclid_vis", "z_phz", "z_true", "ra_gal"})


def test_legacy_euclid_config_was_moved_to_examples() -> None:
    assert Path("configs/legacy/fs2_phz1.yaml").exists()
    assert not Path("configs/fs2_phz1.yaml").exists()


def test_science_config_is_main_lsst_euclid_setup() -> None:
    config = load_config("configs/fs2_phz1_science.yaml")

    band_names = [band["name"] for band in config["bands"]]
    assert len(band_names) == 10
    assert {"euclid_vis", "euclid_nisp_h", "lsst_u", "lsst_y"}.issubset(band_names)
    assert config["cosmos_sed"]["observed_photometry_target_sets"] == [
        "continuum_internal_dust"
    ]
    assert config["cosmos_sed"]["use_cosmos_dust_in_dsps"] is True
    assert "dust_av" not in config["fit"]["free_parameters"]
    assert "log10_sfr" not in config["fit"]["free_parameters"]
    assert "log10_formed_mass_msun" in config["fit"]["free_parameters"]
    assert set(config["sample"]["priors"]) == set(config["fit"]["free_parameters"])
    assert config["fit"]["priors"]["z_obs"]["type"] == "uniform"
    assert "sfh_t_peak" in config["fit"]["free_parameters"]
    assert config["bands"][6]["error_column"].endswith("_error")
    assert config["redshift"]["initial"] == "random_uniform"
    assert config["redshift"]["column"] is None
    assert config["runtime"]["jax_platforms"] == "cuda"
    assert config["runtime"]["disable_jax_plugin_autoload"] is False
    assert config["runtime"]["require_gpu"] is True
    assert config["runtime"]["expected_gpu_name"] == "NVIDIA"


def test_science_config_expands_shorthands() -> None:
    config = load_config("configs/fs2_phz1_science.yaml")

    band_names = [band["name"] for band in config["bands"]]
    assert len(band_names) == 10
    assert band_names[:2] == ["lsst_u", "lsst_g"]
    assert config["runtime"]["require_gpu"] is True
    assert config["cosmos_sed"]["use_cosmos_dust_in_dsps"] is True
    assert config["model"]["parameter_columns"]["cosmos_ebv_1"] == "ebv_cosmos_1"
    assert "phz_min_70" not in config["extra_columns"]
    assert config["reporting"]["plot_ground_truth"] is True
    assert config["fit"]["priors"]["z_obs"]["type"] == "uniform"


def test_cosmos_sed_defaults_are_normalized() -> None:
    config = normalize_config(minimal_config())

    assert config["cosmos_sed"]["template_list"] == "COSMOS_MOD.list"
    assert config["cosmos_sed"]["filter_response_kind"] == "photon"
    assert config["cosmos_sed"]["component_fraction_policy"] == "strict"
    assert config["cosmos_sed"]["extinction"]["curves"][1] == "SMC_prevot"


def test_invalid_cosmos_sed_extinction_mapping_fails() -> None:
    config = minimal_config()
    config["cosmos_sed"] = {"extinction": {"curves": {"bad": "SMC_prevot"}}}

    with pytest.raises(ConfigValidationError, match="curves keys must be integers"):
        normalize_config(config)


def test_invalid_runtime_config_fails_fast() -> None:
    config = minimal_config()
    config["runtime"] = {"jax_platforms": "", "disable_jax_plugin_autoload": "yes"}

    with pytest.raises(ConfigValidationError, match="runtime\\.jax_platforms"):
        normalize_config(config)
