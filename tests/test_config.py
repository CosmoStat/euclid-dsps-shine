from __future__ import annotations

import pytest

from euclid_dsps.config import (
    ConfigValidationError,
    load_config,
    normalize_config,
    validate_catalog_columns,
)
from euclid_dsps.io import required_catalog_columns
from euclid_dsps.parameters import (
    DIFFSKY_BASIC_PARAMETER_NAMES,
    POPCOSMOS_PARAMETER_NAMES,
)
from euclid_dsps.semantics import is_comparable_fit_parameter


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
    assert config["sample"]["mclmc_progress_chunk_size"] == 16
    assert config["sample"]["mclmc_debug"] is False
    assert config["sample"]["posterior_predictive_batch_size"] == 512
    assert config["runtime"]["jax_platforms"] == "cpu"
    assert config["runtime"]["disable_jax_plugin_autoload"] is True
    assert config["runtime"]["require_gpu"] is False
    assert config["model"]["agn_host_attenuation_scale"] == 1.0
    assert config["model"]["agn_igm_order"] == "pre_igm"
    assert config["model"]["agn_baked_attenuation"] == "none"
    assert config["model"]["agn_baked_dust_index"] == -0.7
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


def test_invalid_agn_host_attenuation_scale_fails_fast() -> None:
    config = minimal_config()
    config["model"] = {"agn_host_attenuation_scale": -0.1}

    with pytest.raises(
        ConfigValidationError,
        match="model\\.agn_host_attenuation_scale",
    ):
        normalize_config(config)


def test_invalid_agn_igm_order_fails_fast() -> None:
    config = minimal_config()
    config["model"] = {"agn_igm_order": "after_the_fact"}

    with pytest.raises(ConfigValidationError, match="model\\.agn_igm_order"):
        normalize_config(config)


def test_invalid_agn_baked_attenuation_fails_fast() -> None:
    config = minimal_config()
    config["model"] = {"agn_baked_attenuation": "mystery_dust"}

    with pytest.raises(ConfigValidationError, match="model\\.agn_baked_attenuation"):
        normalize_config(config)


def test_fsps_unit_tau_agn_attenuation_rejects_scaled_mode() -> None:
    config = minimal_config()
    config["model"] = {
        "agn_host_attenuation": "fsps_diffuse_unit_tau",
        "agn_host_attenuation_scale": 0.5,
    }

    with pytest.raises(
        ConfigValidationError,
        match="agn_host_attenuation_scale.*fsps_diffuse_unit_tau",
    ):
        normalize_config(config)


def test_likelihood_space_defaults_to_flux() -> None:
    config = normalize_config(minimal_config())

    assert config["fit"]["likelihood_space"] == "flux"
    assert config["fit"]["photometric_likelihood"] == "gaussian"
    assert config["fit"]["student_t_dof"] == 2.0
    assert config["fit"]["flux_error_floor_frac"] == 0.0
    assert config["band_calibration"]["mode"] == "none"
    assert config["fit"]["band_calibration_offsets_mag"] == [0.0]
    assert config["nebular_emission"] == "ssp_flux"


def test_invalid_likelihood_space_fails_fast() -> None:
    config = minimal_config()
    config["fit"] = {
        "likelihood_space": "counts",
        "free_parameters": {"z_obs": {"initial": "from_base", "bounds": [0.001, 6.0]}},
    }

    with pytest.raises(ConfigValidationError, match="fit\\.likelihood_space"):
        normalize_config(config)


def test_student_t_photometric_likelihood_normalizes_aliases() -> None:
    config = minimal_config()
    config["fit"] = {
        "photometric_likelihood": "student-t",
        "student_t_dof": 2.0,
        "free_parameters": {"z_obs": {"initial": "from_base", "bounds": [0.001, 6.0]}},
    }

    normalized = normalize_config(config)

    assert normalized["fit"]["photometric_likelihood"] == "student_t"


def test_invalid_photometric_likelihood_fails_fast() -> None:
    config = minimal_config()
    config["fit"] = {
        "photometric_likelihood": "cauchy-ish",
        "free_parameters": {"z_obs": {"initial": "from_base", "bounds": [0.001, 6.0]}},
    }

    with pytest.raises(ConfigValidationError, match="fit\\.photometric_likelihood"):
        normalize_config(config)


def test_invalid_nebular_emission_mode_fails_fast() -> None:
    config = minimal_config()
    config["nebular_emission"] = "magic_lines"

    with pytest.raises(ConfigValidationError, match="nebular_emission"):
        normalize_config(config)


def test_gas_grid_required_when_gas_params_free() -> None:
    config = minimal_config()
    config["model"] = {
        "sfh_model": "popcosmos_bins",
        "stellar_metallicity_model": "single",
        "dust_model": "charlot_fall",
        "nebular_model": "fixed_ssp",
        "agn_model": "none",
    }
    config["fit"] = {
        "free_parameters": {
            "log10_gas_metallicity": {"initial": 0.0, "bounds": [-2.5, 0.5]},
        }
    }

    with pytest.raises(ConfigValidationError, match="model\\.nebular_model='gas_grid'"):
        normalize_config(config)


def test_compressed_gas_grid_accepts_gas_free_parameters() -> None:
    config = minimal_config()
    config["model"] = {
        "sfh_model": "popcosmos_bins",
        "stellar_metallicity_model": "single",
        "dust_model": "charlot_fall_powerlaw",
        "nebular_model": "compressed_gas_grid",
        "compressed_gas_grid_path": "Data/popcosmos_chabrier_gas_grid_basis_k64.h5",
        "agn_model": "none",
    }
    config["fit"] = {
        "free_parameters": {
            "log10_gas_metallicity": {"initial": 0.0, "bounds": [-2.5, 0.5]},
            "log10_gas_ionization": {"initial": -2.0, "bounds": [-4.0, -1.0]},
        }
    }

    normalized = normalize_config(config)

    assert normalized["model"]["nebular_model"] == "compressed_gas_grid"
    assert "log10_gas_metallicity" in normalized["fit"]["free_parameters"]
    assert "log10_gas_ionization" in normalized["fit"]["free_parameters"]


def test_compressed_gas_grid_requires_path() -> None:
    config = minimal_config()
    config["model"] = {
        "sfh_model": "popcosmos_bins",
        "stellar_metallicity_model": "single",
        "dust_model": "charlot_fall_powerlaw",
        "nebular_model": "compressed_gas_grid",
        "agn_model": "none",
    }

    with pytest.raises(ConfigValidationError, match="compressed_gas_grid_path"):
        normalize_config(config)


def test_agn_grid_required_when_agn_params_free() -> None:
    config = minimal_config()
    config["model"] = {
        "sfh_model": "popcosmos_bins",
        "stellar_metallicity_model": "single",
        "dust_model": "charlot_fall",
        "nebular_model": "fixed_ssp",
        "agn_model": "none",
    }
    config["fit"] = {
        "free_parameters": {
            "ln_fagn": {"initial": -8.0, "bounds": [-14.0, 1.0]},
        }
    }

    with pytest.raises(ConfigValidationError, match="active AGN model"):
        normalize_config(config)


def test_agn_none_rejects_ln_tauagn_free_parameter() -> None:
    config = minimal_config()
    config["model"] = {
        "sfh_model": "popcosmos_bins",
        "stellar_metallicity_model": "single",
        "dust_model": "charlot_fall_powerlaw",
        "nebular_model": "fixed_ssp",
        "agn_model": "none",
    }
    config["fit"] = {
        "free_parameters": {
            "ln_tauagn": {"initial": 2.3, "bounds": [1.6, 5.1]},
        }
    }

    with pytest.raises(ConfigValidationError, match="active AGN model"):
        normalize_config(config)


def test_fsps_component_grid_accepts_agn_free_parameters() -> None:
    config = minimal_config()
    config["model"] = {
        "sfh_model": "popcosmos_bins",
        "stellar_metallicity_model": "single",
        "dust_model": "charlot_fall_powerlaw",
        "nebular_model": "fixed_ssp",
        "agn_model": "fsps_component_grid",
        "agn_component_grid_path": "Data/popcosmos_chabrier_agn_component_ssp_grid.h5",
    }
    config["fit"] = {
        "free_parameters": {
            "ln_fagn": {"initial": -8.0, "bounds": [-14.0, 1.0]},
            "ln_tauagn": {"initial": 2.3, "bounds": [1.6, 5.1]},
        }
    }

    normalized = normalize_config(config)

    assert normalized["model"]["agn_model"] == "fsps_component_grid"
    assert "ln_fagn" in normalized["fit"]["free_parameters"]
    assert "ln_tauagn" in normalized["fit"]["free_parameters"]


def test_compressed_fsps_component_grid_accepts_agn_free_parameters() -> None:
    config = minimal_config()
    config["model"] = {
        "sfh_model": "popcosmos_bins",
        "stellar_metallicity_model": "single",
        "dust_model": "charlot_fall_powerlaw",
        "nebular_model": "fixed_ssp",
        "agn_model": "compressed_fsps_component_grid",
        "compressed_agn_component_grid_path": (
            "Data/popcosmos_chabrier_agn_component_basis_k32.h5"
        ),
    }
    config["fit"] = {
        "free_parameters": {
            "ln_fagn": {"initial": -8.0, "bounds": [-14.0, 1.0]},
            "ln_tauagn": {"initial": 2.3, "bounds": [1.6, 5.1]},
        }
    }

    normalized = normalize_config(config)

    assert normalized["model"]["agn_model"] == "compressed_fsps_component_grid"
    assert "ln_fagn" in normalized["fit"]["free_parameters"]
    assert "ln_tauagn" in normalized["fit"]["free_parameters"]


def test_compressed_fsps_component_grid_requires_path() -> None:
    config = minimal_config()
    config["model"] = {
        "sfh_model": "popcosmos_bins",
        "stellar_metallicity_model": "single",
        "dust_model": "charlot_fall_powerlaw",
        "nebular_model": "fixed_ssp",
        "agn_model": "compressed_fsps_component_grid",
    }

    with pytest.raises(
        ConfigValidationError, match="compressed_agn_component_grid_path"
    ):
        normalize_config(config)


def test_compressed_ssp_model_requires_path() -> None:
    config = minimal_config()
    config["model"] = {
        "sfh_model": "popcosmos_bins",
        "stellar_metallicity_model": "single",
        "ssp_model": "compressed_basis",
        "nebular_model": "fixed_ssp",
        "agn_model": "none",
    }

    with pytest.raises(ConfigValidationError, match="compressed_ssp_path"):
        normalize_config(config)


def test_compressed_ssp_model_accepts_path() -> None:
    config = minimal_config()
    config["model"] = {
        "sfh_model": "popcosmos_bins",
        "stellar_metallicity_model": "single",
        "ssp_model": "compressed_basis",
        "compressed_ssp_path": "Data/popcosmos_chabrier_stellar_ssp_basis_k64.h5",
        "nebular_model": "fixed_ssp",
        "agn_model": "none",
    }
    config["fit"] = {
        "free_parameters": {
            "z_obs": {"initial": 0.5, "bounds": [0.01, 2.0]},
        }
    }

    normalized = normalize_config(config)

    assert normalized["model"]["ssp_model"] == "compressed_basis"
    assert "compressed_ssp_path" in normalized["model"]


def test_popcosmos_binned_forbids_legacy_free_parameters() -> None:
    config = minimal_config()
    config["model"] = {
        "sfh_model": "popcosmos_bins",
        "stellar_metallicity_model": "single",
        "dust_model": "charlot_fall",
        "nebular_model": "fixed_ssp",
        "agn_model": "none",
    }
    config["fit"] = {
        "free_parameters": {
            "sfh_tau": {"initial": 0.6, "bounds": [0.08, 4.0]},
        }
    }

    with pytest.raises(ConfigValidationError, match="sfh_tau"):
        normalize_config(config)


def test_diffstar_forbids_binned_sfh_free_parameters() -> None:
    config = minimal_config()
    config["model"] = {
        "sfh_model": "diffstar_reduced6",
        "stellar_metallicity_model": "single",
        "dust_model": "charlot_fall",
        "nebular_model": "fixed_ssp",
        "agn_model": "none",
    }
    config["fit"] = {
        "free_parameters": {
            "dlog10_sfr_1": {"initial": 0.0, "bounds": [-3.0, 3.0]},
        }
    }

    with pytest.raises(ConfigValidationError, match="dlog10_sfr_1"):
        normalize_config(config)


def test_diffsky_basic_accepts_generated_truth_dust() -> None:
    config = minimal_config()
    config["model"] = {
        "sfh_model": "diffsky_basic",
        "stellar_metallicity_model": "single",
        "dust_model": "prospector_fsps",
        "nebular_model": "fixed_ssp",
        "agn_model": "none",
    }
    config["truth"] = {
        "redshift_column": "redshift_true",
        "parameter_columns": {
            "dust_av": {"column": "dust_av", "kind": "generated_truth"},
        },
    }
    config["fit"] = {
        "free_parameters": {
            "dust_av": {"initial": 0.5, "bounds": [0.0, 4.0]},
            "diffmah_logm0": {"initial": 12.0, "bounds": [11.0, 15.0]},
        }
    }

    normalized = normalize_config(config)

    assert "dust_av" in DIFFSKY_BASIC_PARAMETER_NAMES
    assert normalized["model"]["sfh_model"] == "diffsky_basic"
    assert is_comparable_fit_parameter(normalized, "dust_av") is True


def test_fixed_band_calibration_expands_to_fit_offsets() -> None:
    config = minimal_config()
    config["band_calibration"] = {
        "mode": "fixed_offsets",
        "offsets_mag": {"euclid_vis": 0.03},
    }

    normalized = normalize_config(config)

    assert normalized["fit"]["band_calibration_offsets_mag"] == pytest.approx([0.03])


def test_redshift_catalog_init_and_gaussian_prior_validate() -> None:
    config = minimal_config()
    config["redshift"] = {
        "initial": "catalog_column",
        "column": "z_phz",
        "truth_column": "z_true",
        "prior_z": {"mode": "gaussian", "sigma": 0.25, "sigma_min": 0.05},
    }

    normalized = normalize_config(config)

    assert normalized["redshift"]["initial"] == "catalog_column"
    assert normalized["redshift"]["prior_z"]["mode"] == "gaussian"


def test_redshift_multistart_is_removed() -> None:
    config = minimal_config()
    config["redshift"] = {
        "initial": "multi_start",
        "multi_start": {"values": [0.1, 1.0]},
    }

    with pytest.raises(ConfigValidationError, match="redshift\\.multi_start"):
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


def test_fs2_gpu_config_is_public_fs2_setup() -> None:
    config = load_config("configs/fs2_gpu.yaml")

    band_names = [band["name"] for band in config["bands"]]
    assert len(band_names) == 10
    assert {"euclid_vis", "euclid_nisp_h", "lsst_u", "lsst_y"}.issubset(band_names)
    assert tuple(config["fit"]["free_parameters"]) == POPCOSMOS_PARAMETER_NAMES
    assert config["model"]["sfh_model"] == "popcosmos_bins"
    assert config["model"]["sfh_time_grid"] == "prospector_step"
    assert config["model"]["dust_model"] == "prospector_fsps"
    assert config["model"]["igm_model"] == "fsps_madau95"
    assert config["runtime"]["jax_platforms"] == "cuda"
    assert config["runtime"]["require_gpu"] is True
    assert config["runtime"]["tf_gpu_allocator"] == "cuda_malloc_async"
    assert config["model"]["ssp_model"] == "compressed_basis"
    assert (
        config["model"]["compressed_ssp_path"]
        == "Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5"
    )
    assert config["model"]["nebular_model"] == "compressed_gas_grid"
    assert (
        config["model"]["compressed_gas_grid_path"]
        == "Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5"
    )
    assert (
        config["model"]["stellar_only_ssp_path"]
        == "Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5"
    )
    assert config["model"]["emission_line_corrections"] == "none"
    assert config["model"]["agn_model"] == "compressed_fsps_component_grid"
    assert (
        config["model"]["compressed_agn_component_grid_path"]
        == "Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5"
    )
    assert config["model"]["agn_igm_order"] == "fsps_after_igm"
    assert config["model"]["agn_baked_attenuation"] == "fsps_powerlaw_unit_tau"
    assert config["model"]["agn_baked_dust_index"] == -0.7
    assert "kroupa" not in config["ssp_path"]
    assert config["model"]["z_sun"] == 0.0142
    assert config["fit"]["likelihood_space"] == "flux"
    assert config["fit"]["photometric_likelihood"] == "student_t"
    assert config["fit"]["student_t_dof"] == 2.0
    assert config["fit"]["flux_error_floor_frac"] == 0.02
    assert config["selection"]["nondetection_policy"] == "gaussian_flux"
    assert (
        config["bands"][0]["error_column"] == "lsst_u_el_model3_ext_odonnell_ext_error"
    )
    assert config["bands"][0]["filter"]["path"] == "filters/LSST_LSST.u.dat"
    assert config["bands"][6]["error_column"].endswith("_error")
    assert config["bands"][6]["filter"]["path"] == "filters/Euclid_VIS.vis.dat"
    assert config["redshift"]["initial"] == "random_uniform"
    assert config["redshift"]["column"] is None
    assert config["redshift"]["prior_z"]["mode"] == "none"
    assert not any(
        column.startswith("phz") for column in required_catalog_columns(config)
    )
    assert "lsst_u_el_model3_ext_odonnell_ext_error" in config["extra_columns"]
    assert config["fit"]["free_parameters"]["ln_tauagn"]["initial"] == 2.302585
    assert config["fit"]["free_parameters"]["ln_tauagn"]["bounds"] == [
        1.609438,
        5.010635,
    ]
    assert config["fit"]["photometric_likelihood"] == "student_t"
    assert config["fit"]["student_t_dof"] == 2.0


def test_amortized_fs2_realnvp_config_extends_fs2_gpu() -> None:
    config = load_config("configs/amortized_fs2_realnvp.yaml")

    assert tuple(config["fit"]["free_parameters"]) == POPCOSMOS_PARAMETER_NAMES
    assert config["model"]["sfh_model"] == "popcosmos_bins"
    assert config["runtime"]["require_gpu"] is True
    assert len(config["bands"]) == 10
    assert config["amortized"]["latent"]["schema"] == "popcosmos_16"
    assert config["amortized"]["features"]["n_flux_bands"] == 10
    assert config["amortized"]["features"]["n_error_bands"] == 10
    assert config["amortized"]["features"]["flux_transform"] == "asinh"
    assert config["amortized"]["data"]["selection_mode"] == "stratified_redshift"
    assert config["amortized"]["data"]["stratified_strategy"] == "balanced"
    assert config["amortized"]["data"]["validation_fraction"] == 0.1
    assert config["amortized"]["prior"]["type"] == "realnvp"
    assert config["amortized"]["prior"]["train_jointly"] is True
    assert config["amortized"]["likelihood"]["type"] == "student_t"
    assert config["amortized"]["training"]["n_samples"] == 2
    assert config["amortized"]["training"]["kl_annealing_epochs"] == 20
    assert config["amortized"]["training"]["kl_weight_max"] == 0.5
    assert config["amortized"]["inference"]["prior_samples"] == 8192
    assert config["amortized"]["inference"]["decoder_sample_chunk_size"] == 1


def test_fit_trace_mode_validation() -> None:
    config = minimal_config()
    config["fit"] = {
        "trace_mode": "verbose",
        "free_parameters": {"z_obs": {"initial": 0.5, "bounds": [0.01, 2.0]}},
    }

    with pytest.raises(ConfigValidationError, match="trace_mode"):
        normalize_config(config)


def test_fit_jax_optimizer_options_validation() -> None:
    config = minimal_config()
    config["fit"] = {
        "scan_unroll": 0,
        "donate_optimizer_inputs": "yes",
        "remat_model_mags": "no",
        "batch_grad_mode": "global",
        "free_parameters": {"z_obs": {"initial": 0.5, "bounds": [0.01, 2.0]}},
    }

    with pytest.raises(ConfigValidationError) as exc:
        normalize_config(config)

    message = str(exc.value)
    assert "scan_unroll" in message
    assert "donate_optimizer_inputs" in message
    assert "remat_model_mags" in message
    assert "batch_grad_mode" in message


def test_diffsky_hltds_dataset_config_marks_debug_truth_contract() -> None:
    config = load_config("configs/diffsky_dataset_hltds_04_14.yaml")

    expected_free = (
        "z_obs",
        "log10_stellar_mass",
        "dlog10_sfr_1",
        "dlog10_sfr_2",
        "dlog10_sfr_3",
        "dlog10_sfr_4",
        "dlog10_sfr_5",
        "dlog10_sfr_6",
        "log10_stellar_metallicity",
        "tau2",
        "dust_index_n",
        "tau1_over_tau2",
    )

    assert config["model"]["sfh_model"] == "popcosmos_bins"
    assert config["model"]["nebular_model"] == "fixed_ssp"
    assert config["model"]["agn_model"] == "none"
    assert config["model"]["asset_metadata_policy"] == "permissive"
    assert config["runtime"]["require_gpu"] is True
    assert config["fit"]["likelihood_space"] == "flux"
    assert config["fit"]["photometric_likelihood"] == "student_t"
    assert config["fit"]["student_t_dof"] == 2.0
    assert config["fit"]["flux_error_floor_frac"] == 0.02
    assert len(config["bands"]) == 14
    assert {band["units"] for band in config["bands"]} == {"fnu_cgs"}
    assert all("error_column" in band for band in config["bands"])
    assert tuple(config["fit"]["free_parameters"]) == expected_free
    assert not any(
        name.startswith(("diffstar_", "diffmah_"))
        for name in config["fit"]["free_parameters"]
    )
    assert all(
        f"dlog10_sfr_{index}" in config["fit"]["free_parameters"]
        for index in range(1, 7)
    )
    truth_columns = config["truth"]["parameter_columns"]
    truth_by_kind = {
        kind: {name for name, spec in truth_columns.items() if spec.get("kind") == kind}
        for kind in {"direct_truth", "projected_generated_truth", "missing"}
    }
    assert truth_by_kind["direct_truth"] == {
        "z_obs",
        "log10_stellar_mass",
        "log10_sfr_at_obs",
        "log10_ssfr_at_obs",
    }
    assert truth_by_kind["projected_generated_truth"] == {
        "dlog10_sfr_1",
        "dlog10_sfr_2",
        "dlog10_sfr_3",
        "dlog10_sfr_4",
        "dlog10_sfr_5",
        "dlog10_sfr_6",
        "tau2",
        "dust_index_n",
    }
    assert truth_by_kind["missing"] == {
        "log10_stellar_metallicity",
        "tau1_over_tau2",
    }
    assert config["diffsky_dataset"]["name"].startswith("hltds_cosmos_260215")
    assert config["diffsky_dataset"]["redshift_subset"]["max"] == 0.35


def test_feniks_amortized_config_uses_full_diffsky_closure_schema() -> None:
    config = load_config("configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml")

    assert tuple(config["fit"]["free_parameters"]) == DIFFSKY_BASIC_PARAMETER_NAMES
    assert config["model"]["sfh_model"] == "diffsky_basic"
    assert config["amortized"]["latent"]["schema"] == "diffsky_dsps_closure_full"
    assert config["amortized"]["features"]["n_flux_bands"] == 14
    assert config["amortized"]["encoder"]["latent_dim"] == 18


def test_minimal_config_does_not_enable_legacy_cosmos_sed() -> None:
    config = normalize_config(minimal_config())

    assert "cosmos_sed" not in config


def test_invalid_runtime_config_fails_fast() -> None:
    config = minimal_config()
    config["runtime"] = {"jax_platforms": "", "disable_jax_plugin_autoload": "yes"}

    with pytest.raises(ConfigValidationError, match="runtime\\.jax_platforms"):
        normalize_config(config)
