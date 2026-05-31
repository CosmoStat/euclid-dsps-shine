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
    DIFFSTAR_REDUCED6_PARAMETER_NAMES,
    POPCOSMOS_PARAMETER_NAMES,
)


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

    with pytest.raises(ConfigValidationError, match="compressed_agn_component_grid_path"):
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


def test_popcosmos_binned_config_is_main_binned_setup() -> None:
    config = load_config("configs/popcosmos_binned.yaml")

    band_names = [band["name"] for band in config["bands"]]
    assert len(band_names) == 10
    assert {"euclid_vis", "euclid_nisp_h", "lsst_u", "lsst_y"}.issubset(band_names)
    assert tuple(config["fit"]["free_parameters"]) == POPCOSMOS_PARAMETER_NAMES
    assert config["model"]["sfh_model"] == "popcosmos_bins"
    assert config["model"]["sfh_time_grid"] == "prospector_step"
    assert config["model"]["dust_model"] == "prospector_fsps"
    assert config["model"]["igm_model"] == "fsps_madau95"
    assert config["model"]["nebular_model"] == "gas_grid"
    assert config["model"]["gas_grid_path"] == "Data/popcosmos_chabrier_gas_ssp_grid.h5"
    assert (
        config["model"]["stellar_only_ssp_path"]
        == "Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5"
    )
    assert config["model"]["emission_line_corrections"] == "none"
    assert config["model"]["agn_model"] == "fsps_component_grid"
    assert (
        config["model"]["agn_component_grid_path"]
        == "Data/popcosmos_chabrier_agn_component_ssp_grid.h5"
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
    assert config["bands"][0]["error_column"] == "lsst_u_el_model3_ext_odonnell_ext_error"
    assert config["bands"][0]["filter"]["path"] == "filters/LSST_LSST.u.dat"
    assert config["bands"][6]["error_column"].endswith("_error")
    assert config["bands"][6]["filter"]["path"] == "filters/Euclid_VIS.vis.dat"
    assert config["redshift"]["initial"] == "fixed"
    assert config["redshift"]["column"] is None
    assert config["redshift"]["prior_z"]["mode"] == "none"
    assert not any(
        column.startswith("phz") for column in required_catalog_columns(config)
    )
    assert config["runtime"]["jax_platforms"] == "auto"
    assert config["runtime"]["disable_jax_plugin_autoload"] is False
    assert config["runtime"]["require_gpu"] is False
    assert "lsst_u_el_model3_ext_odonnell_ext_error" in config["extra_columns"]
    assert config["fit"]["free_parameters"]["ln_tauagn"]["initial"] == 2.302585
    assert config["fit"]["free_parameters"]["ln_tauagn"]["bounds"] == [
        1.609438,
        5.010635,
    ]
    assert config["fit"]["photometric_likelihood"] == "student_t"
    assert config["fit"]["student_t_dof"] == 2.0


def test_popcosmos_binned_compressed_config_overrides_only_runtime_assets() -> None:
    dense = load_config("configs/popcosmos_binned.yaml")
    compressed = load_config("configs/popcosmos_binned_compressed.yaml")

    assert tuple(compressed["fit"]["free_parameters"]) == POPCOSMOS_PARAMETER_NAMES
    assert compressed["model"]["sfh_model"] == dense["model"]["sfh_model"]
    assert compressed["model"]["dust_model"] == dense["model"]["dust_model"]
    assert compressed["model"]["igm_model"] == dense["model"]["igm_model"]
    assert compressed["runtime"]["jax_platforms"] == "cuda"
    assert compressed["runtime"]["require_gpu"] is True
    assert compressed["runtime"]["tf_gpu_allocator"] == "cuda_malloc_async"
    assert compressed["fit"]["trace_mode"] == "optimizer"
    assert compressed["fit"]["trace_interval"] == 20
    assert compressed["fit"]["scan_unroll"] == 1
    assert compressed["fit"]["donate_optimizer_inputs"] is False
    assert compressed["fit"]["remat_model_mags"] is False
    assert compressed["fit"]["batch_grad_mode"] == "per_galaxy"
    assert compressed["model"]["ssp_model"] == "compressed_basis"
    assert (
        compressed["model"]["compressed_ssp_path"]
        == "Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5"
    )
    assert compressed["model"]["nebular_model"] == "compressed_gas_grid"
    assert (
        compressed["model"]["compressed_gas_grid_path"]
        == "Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5"
    )
    assert compressed["model"]["agn_model"] == "compressed_fsps_component_grid"
    assert (
        compressed["model"]["compressed_agn_component_grid_path"]
        == "Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5"
    )
    assert compressed["fit"]["photometric_likelihood"] == "student_t"


def test_popcosmos_diffstar_compressed_config_overrides_only_runtime_assets() -> None:
    dense = load_config("configs/popcosmos_diffstar.yaml")
    compressed = load_config("configs/popcosmos_diffstar_compressed.yaml")

    assert compressed["model"]["sfh_model"] == "diffstar_reduced6"
    assert compressed["model"]["sfh_model"] == dense["model"]["sfh_model"]
    assert compressed["model"]["dust_model"] == dense["model"]["dust_model"]
    assert compressed["model"]["igm_model"] == dense["model"]["igm_model"]
    assert compressed["runtime"]["jax_platforms"] == "cuda"
    assert compressed["runtime"]["require_gpu"] is True
    assert compressed["runtime"]["tf_gpu_allocator"] == "cuda_malloc_async"
    assert compressed["fit"]["trace_mode"] == "optimizer"
    assert compressed["fit"]["trace_interval"] == 20
    assert compressed["fit"]["batch_grad_mode"] == "per_galaxy"
    assert compressed["model"]["ssp_model"] == "compressed_basis"
    assert (
        compressed["model"]["compressed_ssp_path"]
        == "Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5"
    )
    assert compressed["model"]["nebular_model"] == "compressed_gas_grid"
    assert (
        compressed["model"]["compressed_gas_grid_path"]
        == "Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5"
    )
    assert compressed["model"]["agn_model"] == "compressed_fsps_component_grid"
    assert (
        compressed["model"]["compressed_agn_component_grid_path"]
        == "Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5"
    )
    assert compressed["fit"]["photometric_likelihood"] == "student_t"


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


def test_popcosmos_diffstar_config_combines_diffstar_with_gas_and_agn() -> None:
    config = load_config("configs/popcosmos_diffstar.yaml")

    assert tuple(config["fit"]["free_parameters"]) == DIFFSTAR_REDUCED6_PARAMETER_NAMES
    assert config["model"]["sfh_model"] == "diffstar_reduced6"
    assert config["model"]["stellar_metallicity_model"] == "single"
    assert config["model"]["dust_model"] == "prospector_fsps"
    assert config["model"]["igm_model"] == "fsps_madau95"
    assert config["model"]["nebular_model"] == "gas_grid"
    assert config["model"]["gas_grid_path"] == "Data/popcosmos_chabrier_gas_ssp_grid.h5"
    assert (
        config["model"]["stellar_only_ssp_path"]
        == "Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5"
    )
    assert config["model"]["emission_line_corrections"] == "none"
    assert config["model"]["agn_model"] == "fsps_component_grid"
    assert (
        config["model"]["agn_component_grid_path"]
        == "Data/popcosmos_chabrier_agn_component_ssp_grid.h5"
    )
    assert config["model"]["agn_igm_order"] == "fsps_after_igm"
    assert config["model"]["agn_baked_attenuation"] == "fsps_powerlaw_unit_tau"
    assert config["model"]["agn_baked_dust_index"] == -0.7
    assert "kroupa" not in config["ssp_path"]
    assert config["model"]["z_sun"] == 0.0142
    assert config["model"]["fixed_parameters"]["diffstar_indx_hi"] == -1.0
    assert config["model"]["fixed_parameters"]["diffstar_qlglgdt"] == -0.50725
    assert config["fit"]["free_parameters"]["ln_tauagn"]["bounds"] == [
        1.609438,
        5.010635,
    ]


def test_popcosmos_binned_noagn_config_is_fallback_setup() -> None:
    config = load_config("configs/popcosmos_binned_noagn.yaml")
    free_names = tuple(config["fit"]["free_parameters"])

    assert free_names == POPCOSMOS_PARAMETER_NAMES[:-2]
    assert "ln_fagn" not in free_names
    assert "ln_tauagn" not in free_names
    assert config["model"]["agn_model"] == "none"
    assert config["fit"]["photometric_likelihood"] == "student_t"
    assert config["model"]["z_sun"] == 0.0142
    assert config["model"]["sfh_time_grid"] == "prospector_step"
    assert config["model"]["igm_model"] == "fsps_madau95"
    assert (
        config["model"]["stellar_only_ssp_path"]
        == "Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5"
    )


def test_popcosmos_diffstar_noagn_config_is_fallback_comparison() -> None:
    config = load_config("configs/popcosmos_diffstar_noagn.yaml")
    free_names = tuple(config["fit"]["free_parameters"])

    assert free_names == DIFFSTAR_REDUCED6_PARAMETER_NAMES[:-2]
    assert "ln_fagn" not in free_names
    assert "ln_tauagn" not in free_names
    assert config["model"]["agn_model"] == "none"
    assert config["fit"]["photometric_likelihood"] == "student_t"
    assert config["model"]["z_sun"] == 0.0142
    assert config["model"]["igm_model"] == "fsps_madau95"
    assert (
        config["model"]["stellar_only_ssp_path"]
        == "Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5"
    )


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
