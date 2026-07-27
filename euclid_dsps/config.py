"""Configuration loading and defaults."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .parameters import (
    DIFFSKY_BASIC_PARAMETER_NAMES,
    DIFFSKY_TRUTH_BASIC_PARAMETER_NAMES,
    DIFFSTAR_REDUCED6_PARAMETER_NAMES,
    POPCOSMOS_PARAMETER_NAMES,
)
from .photometric_uncertainty import default_m5_depth_error_model


@dataclass(frozen=True)
class Paths:
    catalog: Path
    ssp: Path


DEFAULT_MODEL_PARAMETERS = {
    "z_obs": 0.5,
    "log10_sfr": 0.0,
    "sfh_t_peak": 4.0,
    "sfh_tau": 0.6,
    "log10_metallicity": -2.0,
    "metallicity_scatter": 0.2,
    "dust_av": 0.2,
    "dust_slope": -0.7,
    "cosmos_ebv_1": 0.0,
    "cosmos_ebv_2": 0.0,
    "cosmos_frac_1": 0.5,
    "cosmos_frac_2": 0.5,
    "cosmos_ext_curve_1": 0.0,
    "cosmos_ext_curve_2": 0.0,
    "log10_stellar_mass": 10.0,
    "dlog10_sfr_1": 0.0,
    "dlog10_sfr_2": 0.0,
    "dlog10_sfr_3": 0.0,
    "dlog10_sfr_4": 0.0,
    "dlog10_sfr_5": 0.0,
    "dlog10_sfr_6": 0.0,
    "log10_stellar_metallicity": 0.0,
    "tau2": 0.3,
    "dust_index_n": -0.7,
    "tau1_over_tau2": 1.0,
    "log10_gas_metallicity": 0.0,
    "log10_gas_ionization": -2.0,
    "ln_fagn": -8.0,
    "ln_tauagn": 2.302585,
    "diffstar_lgmcrit": 12.0,
    "diffstar_lgy_at_mcrit": -10.0,
    "diffstar_indx_lo": 1.0,
    "diffstar_indx_hi": -1.0,
    "diffstar_lg_qt": 1.0,
    "diffstar_qlglgdt": -0.50725,
    "diffstar_lg_drop": -1.01773,
    "diffstar_lg_rejuv": -0.212307,
    "diffmah_logm0": 12.0,
    "diffmah_logtc": 0.05,
    "diffmah_early_index": 2.6137643,
    "diffmah_late_index": 0.12692805,
    "diffmah_t_peak": 14.0,
    "dust_delta": 0.0,
}

DEFAULT_REDSHIFT_CONFIG = {
    "initial": "catalog_column",
    "column": None,
    "truth_column": None,
    "fixed_value": 0.5,
    "min": 1.0e-4,
    "max": 6.0,
    "seed": 42,
    "prior_z": {"mode": "none"},
}

SUPPORTED_PHOTOMETRY_UNITS = {
    "fnu_cgs",
    "abmag",
    "microjy",
    "ujy",
    "photon_per_sec_cm2",
}
SUPPORTED_FIT_METHODS = {"jax_adam", "jax_adam_vmap", "jax_bfgs"}
SUPPORTED_LIKELIHOOD_SPACES = {"flux", "mag"}
SUPPORTED_PHOTOMETRIC_LIKELIHOODS = {"gaussian", "student_t"}
SUPPORTED_FIT_TRACE_MODES = {"full", "optimizer", "none"}
SUPPORTED_FIT_BATCH_GRAD_MODES = {"per_galaxy", "sum"}
SUPPORTED_SAMPLERS = {"nuts", "hmc", "mclmc"}
SUPPORTED_CHAIN_METHODS = {"parallel", "sequential", "vectorized"}
SUPPORTED_SAMPLE_INIT_STRATEGIES = {
    "map",
    "config",
    "map_jitter",
    "random_uniform",
}
SUPPORTED_TRUTH_TRANSFORMS = {None, "linear", "log10", "log_stellar_mass_h2_to_msun"}
SUPPORTED_PRIOR_TYPES = {
    "uniform",
    "normal",
    "truncated_normal",
    "scaled_beta",
}
SUPPORTED_REPORTING_LEVELS = {"full", "light", "none"}
SUPPORTED_OUTPUT_FORMATS = {"csv", "parquet", "both"}
SUPPORTED_NONDETECTION_POLICIES = {"drop", "gaussian_flux", "upper_limit"}
SUPPORTED_BAND_CALIBRATION_MODES = {"none", "fixed_offsets"}
SUPPORTED_NEBULAR_EMISSION_MODES = {"none", "ssp_flux", "emline_table"}
SUPPORTED_SFH_MODELS = {
    "lognormal",
    "popcosmos_bins",
    "diffstar_reduced6",
    "diffsky_basic",
    "spline15d",
}
SUPPORTED_SSP_MODELS = {"dense", "compressed_basis"}
SUPPORTED_STELLAR_METALLICITY_MODELS = {
    "mdf",
    "single",
    "lognormal_mdf_fixed_scatter",
}
SUPPORTED_MODEL_DUST_MODELS = {
    "legacy",
    "charlot_fall",
    "charlot_fall_powerlaw",
    "prospector_fsps",
}
SUPPORTED_IGM_MODELS = {"none", "madau95_approx", "fsps_madau95"}
SUPPORTED_NEBULAR_MODELS = {"fixed_ssp", "gas_grid", "compressed_gas_grid"}
SUPPORTED_AGN_MODELS = {
    "none",
    "template_grid",
    "fsps_component_grid",
    "compressed_fsps_component_grid",
}
SUPPORTED_AGN_HOST_ATTENUATION_MODES = {
    "none",
    "diffuse",
    "prospector_fsps",
    "fsps_diffuse_unit_tau",
}
SUPPORTED_AGN_IGM_ORDERS = {"pre_igm", "fsps_after_igm"}
SUPPORTED_AGN_BAKED_ATTENUATION_MODES = {"none", "fsps_powerlaw_unit_tau"}
SUPPORTED_EMISSION_LINE_CORRECTIONS = {"none", "popcosmos_table"}
SUPPORTED_REDSHIFT_INITIALS = {
    "catalog_column",
    "fixed",
    "random_uniform",
}
SUPPORTED_REDSHIFT_PRIORS = {"none", "gaussian", "top_hat", "phz_interval"}

PRIOR_SETS = {
    "flat_debug": {
        "z_obs": {"type": "uniform"},
        "log10_formed_mass_msun": {"type": "uniform"},
        "sfh_t_peak": {"type": "uniform"},
        "sfh_tau": {"type": "uniform"},
        "log10_metallicity": {"type": "uniform"},
    },
    "weak_physical": {
        "z_obs": {"type": "uniform"},
        "log10_formed_mass_msun": {"type": "normal", "loc": 10.0, "scale": 1.5},
        "sfh_t_peak": {"type": "normal", "loc": 4.0, "scale": 3.0},
        "sfh_tau": {"type": "normal", "loc": 0.8, "scale": 0.8},
        "log10_metallicity": {"type": "normal", "loc": -2.4, "scale": 0.6},
    },
    "flat_diffstar_popcosmos": {
        "z_obs": {"type": "uniform"},
        "log10_stellar_mass": {"type": "uniform"},
        "diffstar_lgmcrit": {"type": "uniform"},
        "diffstar_lgy_at_mcrit": {"type": "uniform"},
        "diffstar_indx_lo": {"type": "uniform"},
        "diffstar_lg_qt": {"type": "uniform"},
        "diffstar_lg_drop": {"type": "uniform"},
        "diffstar_lg_rejuv": {"type": "uniform"},
        "log10_stellar_metallicity": {"type": "uniform"},
        "tau2": {"type": "uniform"},
        "dust_index_n": {"type": "uniform"},
        "tau1_over_tau2": {"type": "uniform"},
        "log10_gas_metallicity": {"type": "uniform"},
        "log10_gas_ionization": {"type": "uniform"},
        "ln_fagn": {"type": "uniform"},
        "ln_tauagn": {"type": "uniform"},
    },
    "flat_diffsky_basic": {
        "z_obs": {"type": "uniform"},
        "log10_stellar_mass": {"type": "uniform"},
        "diffstar_lgmcrit": {"type": "uniform"},
        "diffstar_lgy_at_mcrit": {"type": "uniform"},
        "diffstar_indx_lo": {"type": "uniform"},
        "diffstar_indx_hi": {"type": "uniform"},
        "diffstar_lg_qt": {"type": "uniform"},
        "diffstar_qlglgdt": {"type": "uniform"},
        "diffstar_lg_drop": {"type": "uniform"},
        "diffstar_lg_rejuv": {"type": "uniform"},
        "diffmah_logm0": {"type": "uniform"},
        "diffmah_logtc": {"type": "uniform"},
        "diffmah_early_index": {"type": "uniform"},
        "diffmah_late_index": {"type": "uniform"},
        "diffmah_t_peak": {"type": "uniform"},
        "log10_stellar_metallicity": {"type": "uniform"},
        "dust_av": {"type": "uniform"},
        "dust_delta": {"type": "uniform"},
    },
}

RUNTIME_PRESETS = {
    "auto": {
        "jax_platforms": "auto",
        "disable_jax_plugin_autoload": False,
        "xla_python_client_preallocate": False,
        "require_gpu": False,
    },
    "cpu": {
        "jax_platforms": "cpu",
        "disable_jax_plugin_autoload": True,
        "xla_python_client_preallocate": False,
        "require_gpu": False,
    },
    "gpu": {
        "jax_platforms": "cuda",
        "disable_jax_plugin_autoload": False,
        "xla_python_client_preallocate": False,
        "require_gpu": True,
        "expected_gpu_name": "NVIDIA",
        "tf_gpu_allocator": "cuda_malloc_async",
        "jax_compilation_cache_dir": "outputs/jax_cache",
        "jax_persistent_cache_min_compile_time_secs": 1.0,
    },
}

BAND_PRESETS = {
    "euclid_4": [
        {
            "name": "euclid_vis",
            "column": "euclid_vis",
            "units": "fnu_cgs",
            "sigma_mag": 0.05,
            "filter": {"path": "filters/Euclid_VIS.vis.dat"},
        },
        {
            "name": "euclid_nisp_y",
            "column": "euclid_nisp_y",
            "units": "fnu_cgs",
            "sigma_mag": 0.05,
            "filter": {"path": "filters/Euclid_NISP.Y.dat"},
        },
        {
            "name": "euclid_nisp_j",
            "column": "euclid_nisp_j",
            "units": "fnu_cgs",
            "sigma_mag": 0.05,
            "filter": {"path": "filters/Euclid_NISP.J.dat"},
        },
        {
            "name": "euclid_nisp_h",
            "column": "euclid_nisp_h",
            "units": "fnu_cgs",
            "sigma_mag": 0.05,
            "filter": {"path": "filters/Euclid_NISP.H.dat"},
        },
    ],
    "lsst_euclid_10": [
        {
            "name": "lsst_u",
            "column": "lsst_u",
            "units": "fnu_cgs",
            "sigma_mag": 0.05,
            "error_column": "lsst_u_el_model3_ext_odonnell_ext_error",
            "error_units": "fnu_cgs",
            "sigma_mag_floor": 0.01,
            "sigma_mag_ceiling": 0.5,
            "filter": {
                "kind": "ascii",
                "path": "filters/LSST_LSST.u.dat",
                "wave_unit": "angstrom",
            },
        },
        {
            "name": "lsst_g",
            "column": "lsst_g",
            "units": "fnu_cgs",
            "sigma_mag": 0.05,
            "error_column": "lsst_g_el_model3_ext_odonnell_ext_error",
            "error_units": "fnu_cgs",
            "sigma_mag_floor": 0.01,
            "sigma_mag_ceiling": 0.5,
            "filter": {
                "kind": "ascii",
                "path": "filters/LSST_LSST.g.dat",
                "wave_unit": "angstrom",
            },
        },
        {
            "name": "lsst_r",
            "column": "lsst_r",
            "units": "fnu_cgs",
            "sigma_mag": 0.05,
            "error_column": "lsst_r_el_model3_ext_odonnell_ext_error",
            "error_units": "fnu_cgs",
            "sigma_mag_floor": 0.01,
            "sigma_mag_ceiling": 0.5,
            "filter": {
                "kind": "ascii",
                "path": "filters/LSST_LSST.r.dat",
                "wave_unit": "angstrom",
            },
        },
        {
            "name": "lsst_i",
            "column": "lsst_i",
            "units": "fnu_cgs",
            "sigma_mag": 0.05,
            "error_column": "lsst_i_el_model3_ext_odonnell_ext_error",
            "error_units": "fnu_cgs",
            "sigma_mag_floor": 0.01,
            "sigma_mag_ceiling": 0.5,
            "filter": {
                "kind": "ascii",
                "path": "filters/LSST_LSST.i.dat",
                "wave_unit": "angstrom",
            },
        },
        {
            "name": "lsst_z",
            "column": "lsst_z",
            "units": "fnu_cgs",
            "sigma_mag": 0.05,
            "error_column": "lsst_z_el_model3_ext_odonnell_ext_error",
            "error_units": "fnu_cgs",
            "sigma_mag_floor": 0.01,
            "sigma_mag_ceiling": 0.5,
            "filter": {
                "kind": "ascii",
                "path": "filters/LSST_LSST.z.dat",
                "wave_unit": "angstrom",
            },
        },
        {
            "name": "lsst_y",
            "column": "lsst_y",
            "units": "fnu_cgs",
            "sigma_mag": 0.05,
            "error_column": "lsst_y_el_model3_ext_odonnell_ext_error",
            "error_units": "fnu_cgs",
            "sigma_mag_floor": 0.01,
            "sigma_mag_ceiling": 0.5,
            "filter": {
                "kind": "ascii",
                "path": "filters/LSST_LSST.y.dat",
                "wave_unit": "angstrom",
            },
        },
        {
            "name": "euclid_vis",
            "column": "euclid_vis",
            "units": "fnu_cgs",
            "sigma_mag": 0.05,
            "error_column": "euclid_vis_el_model3_ext_odonnell_ext_error",
            "error_units": "fnu_cgs",
            "sigma_mag_floor": 0.01,
            "sigma_mag_ceiling": 0.5,
            "filter": {
                "kind": "ascii",
                "path": "filters/Euclid_VIS.vis.dat",
                "wave_unit": "angstrom",
            },
        },
        {
            "name": "euclid_nisp_y",
            "column": "euclid_nisp_y",
            "units": "fnu_cgs",
            "sigma_mag": 0.05,
            "error_column": "euclid_nisp_y_el_model3_ext_odonnell_ext_error",
            "error_units": "fnu_cgs",
            "sigma_mag_floor": 0.01,
            "sigma_mag_ceiling": 0.5,
            "filter": {
                "kind": "ascii",
                "path": "filters/Euclid_NISP.Y.dat",
                "wave_unit": "angstrom",
            },
        },
        {
            "name": "euclid_nisp_j",
            "column": "euclid_nisp_j",
            "units": "fnu_cgs",
            "sigma_mag": 0.05,
            "error_column": "euclid_nisp_j_el_model3_ext_odonnell_ext_error",
            "error_units": "fnu_cgs",
            "sigma_mag_floor": 0.01,
            "sigma_mag_ceiling": 0.5,
            "filter": {
                "kind": "ascii",
                "path": "filters/Euclid_NISP.J.dat",
                "wave_unit": "angstrom",
            },
        },
        {
            "name": "euclid_nisp_h",
            "column": "euclid_nisp_h",
            "units": "fnu_cgs",
            "sigma_mag": 0.05,
            "error_column": "euclid_nisp_h_el_model3_ext_odonnell_ext_error",
            "error_units": "fnu_cgs",
            "sigma_mag_floor": 0.01,
            "sigma_mag_ceiling": 0.5,
            "filter": {
                "kind": "ascii",
                "path": "filters/Euclid_NISP.H.dat",
                "wave_unit": "angstrom",
            },
        },
    ],
}

DIFFSKY_HLTDS_FILTER_PATH = (
    "Data/diffsky/raw/hltds_cosmos_260215_04_14_2026/"
    "diffsky_hltds_cosmos_260215_04_14_2026_transmission_curves.hdf5"
)
DIFFSKY_HLTDS_LSST_ROMAN_BANDS = (
    "lsst_u",
    "lsst_g",
    "lsst_r",
    "lsst_i",
    "lsst_z",
    "lsst_y",
    "roman_F062",
    "roman_F087",
    "roman_F106",
    "roman_F129",
    "roman_F146",
    "roman_F158",
    "roman_F184",
    "roman_F213",
)
DIFFSKY_HLTDS_LSST_BANDS = tuple(
    band for band in DIFFSKY_HLTDS_LSST_ROMAN_BANDS if band.startswith("lsst_")
)
DIFFSKY_HLTDS_ROMAN_BANDS = tuple(
    band for band in DIFFSKY_HLTDS_LSST_ROMAN_BANDS if band.startswith("roman_")
)
EUCLID_COMPARISON_FILTERS = {
    "euclid_vis": "filters/Euclid_VIS.vis.dat",
    "euclid_nisp_y": "filters/Euclid_NISP.Y.dat",
    "euclid_nisp_j": "filters/Euclid_NISP.J.dat",
    "euclid_nisp_h": "filters/Euclid_NISP.H.dat",
}


def _diffsky_hltds_hdf5_band_config(band_name: str) -> dict[str, Any]:
    return {
        "name": band_name,
        "column": f"flux_{band_name}",
        "units": "fnu_cgs",
        "error_column": f"fluxerr_{band_name}",
        "error_units": "fnu_cgs",
        "sigma_mag": 0.05,
        "sigma_mag_floor": 0.005,
        "sigma_mag_ceiling": 0.5,
        "filter": {
            "kind": "hdf5_group",
            "path": DIFFSKY_HLTDS_FILTER_PATH,
            "group": band_name,
            "wave_dataset": "wave",
            "transmission_dataset": "transmission",
            "wave_unit": "angstrom",
        },
    }


def _euclid_comparison_band_config(band_name: str) -> dict[str, Any]:
    return {
        "name": band_name,
        "column": f"flux_{band_name}",
        "units": "fnu_cgs",
        "error_column": f"fluxerr_{band_name}",
        "error_units": "fnu_cgs",
        "sigma_mag": 0.05,
        "sigma_mag_floor": 0.005,
        "sigma_mag_ceiling": 0.5,
        "filter": {
            "kind": "ascii",
            "path": EUCLID_COMPARISON_FILTERS[band_name],
            "wave_unit": "angstrom",
        },
    }


BAND_PRESETS["diffsky_hltds_lsst_roman_14_fnu_cgs"] = [
    _diffsky_hltds_hdf5_band_config(band_name)
    for band_name in DIFFSKY_HLTDS_LSST_ROMAN_BANDS
]
BAND_PRESETS["diffsky_hltds_lsst_euclid_10_fnu_cgs"] = [
    *[
        _diffsky_hltds_hdf5_band_config(band_name)
        for band_name in DIFFSKY_HLTDS_LSST_BANDS
    ],
    *[
        _euclid_comparison_band_config(band_name)
        for band_name in EUCLID_COMPARISON_FILTERS
    ],
]
BAND_PRESETS["diffsky_hltds_lsst_euclid_roman_18_fnu_cgs"] = [
    *[
        _diffsky_hltds_hdf5_band_config(band_name)
        for band_name in DIFFSKY_HLTDS_LSST_BANDS
    ],
    *[
        _euclid_comparison_band_config(band_name)
        for band_name in EUCLID_COMPARISON_FILTERS
    ],
    *[
        _diffsky_hltds_hdf5_band_config(band_name)
        for band_name in DIFFSKY_HLTDS_ROMAN_BANDS
    ],
]
BAND_PRESETS["diffsky_hltds_lsst_roman_14_abmag_modelerr"] = [
    {
        "name": band_name,
        "column": f"mag_{band_name}",
        "units": "abmag",
        # HLTDS public shards expose generated magnitudes, not native survey
        # errors. Use an explicit flux-dependent synthetic error model.
        "sigma_mag": 0.10,
        "error_model": default_m5_depth_error_model(),
        "filter": {
            "kind": "hdf5_group",
            "path": DIFFSKY_HLTDS_FILTER_PATH,
            "group": band_name,
            "wave_dataset": "wave",
            "transmission_dataset": "transmission",
            "wave_unit": "angstrom",
        },
    }
    for band_name in DIFFSKY_HLTDS_LSST_ROMAN_BANDS
]

COLUMN_GROUPS = {
    "truth_basic": [
        "z_true_gal",
        "log_stellar_mass",
        "log_sfr_true",
        "metallicity_true",
        "dust_ebv_true",
    ],
    "phz_diagnostics": [
        "z_obs_gal",
        "redshift_step",
        "z_deepz",
        "phz_flags",
        "phz_min_70",
        "phz_max_70",
        "phz_min_90",
        "phz_max_90",
        "phz_min_95",
        "phz_max_95",
        "phz_mode_1_area",
        "phz_mode_2",
        "phz_mode_2_area",
    ],
    "cosmos_proxy": [
        "sed_cosmos_1",
        "sed_cosmos_2",
        "frac_cosmos_1",
        "frac_cosmos_2",
        "color_kind",
        "euclid_vis_abs",
        "euclid_nisp_y_abs",
        "euclid_nisp_j_abs",
        "euclid_nisp_h_abs",
        "lsst_u_abs",
        "lsst_g_abs",
        "lsst_r_abs",
        "lsst_i_abs",
        "lsst_z_abs",
        "lsst_y_abs",
        "ebv_cosmos_1",
        "ebv_cosmos_2",
        "ext_curve_cosmos_1",
        "ext_curve_cosmos_2",
        "mw_extinction",
    ],
    "photometry_errors": [
        "euclid_vis_el_model3_ext_odonnell_ext_error",
        "euclid_nisp_y_el_model3_ext_odonnell_ext_error",
        "euclid_nisp_j_el_model3_ext_odonnell_ext_error",
        "euclid_nisp_h_el_model3_ext_odonnell_ext_error",
        "lsst_u_el_model3_ext_odonnell_ext_error",
        "lsst_g_el_model3_ext_odonnell_ext_error",
        "lsst_r_el_model3_ext_odonnell_ext_error",
        "lsst_i_el_model3_ext_odonnell_ext_error",
        "lsst_z_el_model3_ext_odonnell_ext_error",
        "lsst_y_el_model3_ext_odonnell_ext_error",
    ],
    "emission_line_diagnostics": [
        "euclid_vis_el_model3_ext",
        "euclid_nisp_y_el_model3_ext",
        "euclid_nisp_j_el_model3_ext",
        "euclid_nisp_h_el_model3_ext",
        "lsst_u_el_model3_ext",
        "lsst_g_el_model3_ext",
        "lsst_r_el_model3_ext",
        "lsst_i_el_model3_ext",
        "lsst_z_el_model3_ext",
        "lsst_y_el_model3_ext",
        "euclid_vis_el_model3_ext_odonnell_ext",
        "euclid_nisp_y_el_model3_ext_odonnell_ext",
        "euclid_nisp_j_el_model3_ext_odonnell_ext",
        "euclid_nisp_h_el_model3_ext_odonnell_ext",
        "lsst_u_el_model3_ext_odonnell_ext",
        "lsst_g_el_model3_ext_odonnell_ext",
        "lsst_r_el_model3_ext_odonnell_ext",
        "lsst_i_el_model3_ext_odonnell_ext",
        "lsst_z_el_model3_ext_odonnell_ext",
        "lsst_y_el_model3_ext_odonnell_ext",
        "euclid_vis_el_model3_ext_odonnell_ext_error_realization",
        "euclid_nisp_y_el_model3_ext_odonnell_ext_error_realization",
        "euclid_nisp_j_el_model3_ext_odonnell_ext_error_realization",
        "euclid_nisp_h_el_model3_ext_odonnell_ext_error_realization",
        "lsst_u_el_model3_ext_odonnell_ext_error_realization",
        "lsst_g_el_model3_ext_odonnell_ext_error_realization",
        "lsst_r_el_model3_ext_odonnell_ext_error_realization",
        "lsst_i_el_model3_ext_odonnell_ext_error_realization",
        "lsst_z_el_model3_ext_odonnell_ext_error_realization",
        "lsst_y_el_model3_ext_odonnell_ext_error_realization",
    ],
    "morphology_halo": [
        "ra_gal",
        "dec_gal",
        "ra_mag_gal",
        "dec_mag_gal",
        "log_ml_r01",
        "abs_mag_r01",
        "log_luminosity_r01",
        "abs_mag_uv_unextincted",
        "bulge_fraction",
        "disk_r50",
        "bulge_r50",
        "eps1_gal",
        "eps2_gal",
        "disk_ellipticity",
        "bulge_ellipticity",
        "bulge_nsersic",
        "disk_nsersic",
        "lm_halo",
        "lmbound_halo",
        "r_halo",
        "conc_vir_halo",
        "rs_halo",
        "rvir_halo",
        "n_sats_halo",
        "num_p_halo",
    ],
}

DEFAULT_RUNTIME_CONFIG = {
    "jax_platforms": "cpu",
    "disable_jax_plugin_autoload": True,
    "xla_python_client_preallocate": False,
    "require_gpu": False,
    "expected_gpu_name": None,
    "tf_gpu_allocator": None,
    "jax_compilation_cache_dir": None,
    "jax_persistent_cache_min_compile_time_secs": 1.0,
}


class ConfigValidationError(ValueError):
    """Raised when a run configuration is internally inconsistent."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    config = _load_config_tree(Path(path).resolve(), seen=set())
    return normalize_config(config)


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fill lightweight defaults without hiding required paths."""
    config = _expand_config_shorthands(dict(config))
    config.setdefault("selection", {})
    config.setdefault("redshift", {})
    config.setdefault("model", {})
    config.setdefault("fit", {})
    config.setdefault("sample", {})
    config.setdefault("eda", {})
    config.setdefault("truth", {})
    config.setdefault("runtime", {})
    config.setdefault("reporting", {})
    config.setdefault("output", {})
    config.setdefault("extra_columns", [])
    config.setdefault("band_calibration", {})
    config.setdefault("calibration", {})
    config.setdefault("nebular_emission", "ssp_flux")

    raw_redshift = dict(config["redshift"] or {})
    redshift = dict(DEFAULT_REDSHIFT_CONFIG)
    redshift.update(raw_redshift)

    config["model"].setdefault("fixed_parameters", {})
    fixed = dict(DEFAULT_MODEL_PARAMETERS)
    fixed.update(config["model"]["fixed_parameters"] or {})
    if "fixed_value" in raw_redshift:
        fixed["z_obs"] = float(redshift["fixed_value"])
    else:
        redshift["fixed_value"] = float(fixed["z_obs"])
    config["redshift"] = redshift
    config["model"]["fixed_parameters"] = fixed
    config["model"].setdefault("parameter_columns", {})
    config["model"].setdefault("n_sfh_bins", 96)
    config["model"].setdefault("ssp_model", "dense")
    sfh_model = str(config["model"].get("sfh_model", "lognormal"))
    config["model"]["sfh_model"] = sfh_model
    config["model"].setdefault(
        "stellar_metallicity_model",
        "single" if _is_popcosmos_like_sfh(sfh_model) else "mdf",
    )
    popcosmos_like = _is_popcosmos_like_sfh(sfh_model)
    config["model"].setdefault(
        "dust_model",
        "charlot_fall_powerlaw" if popcosmos_like else "legacy",
    )
    config["model"]["dust_model"] = _normalize_model_dust_model(
        config["model"]["dust_model"]
    )
    config["model"].setdefault("igm_model", "none")
    config["model"].setdefault("nebular_model", "fixed_ssp")
    config["model"].setdefault("agn_model", "none")
    config["model"].setdefault("agn_host_attenuation", "none")
    config["model"].setdefault("agn_host_attenuation_scale", 1.0)
    config["model"].setdefault("agn_igm_order", "pre_igm")
    config["model"].setdefault("agn_baked_attenuation", "none")
    config["model"].setdefault("agn_baked_dust_index", -0.7)
    config["model"].setdefault("birth_cloud_slope", -1.0)
    config["model"].setdefault("dust_tesc_logyr", 7.0)
    config["model"].setdefault("dust1_index", -1.0)
    config["model"].setdefault("emission_line_corrections", "none")
    config["model"].setdefault("z_sun", 0.0142 if popcosmos_like else 0.0134)
    config["model"].setdefault(
        "sfh_time_grid",
        "prospector_step" if sfh_model == "popcosmos_bins" else "linear",
    )

    config["fit"].setdefault(
        "free_parameters",
        {
            "log10_sfr": {"initial": 0.0, "bounds": [-2.5, 3.0]},
            "dust_av": {"initial": 0.2, "bounds": [0.0, 2.5]},
            "log10_metallicity": {"initial": -2.0, "bounds": [-3.0, -1.0]},
        },
    )
    config["fit"].setdefault("method", "jax_adam")
    config["fit"].setdefault("likelihood_space", "flux")
    if "photometric_likelihood" not in config["fit"] and "likelihood" in config["fit"]:
        config["fit"]["photometric_likelihood"] = config["fit"]["likelihood"]
    config["fit"]["photometric_likelihood"] = _normalize_photometric_likelihood(
        config["fit"].get("photometric_likelihood", "gaussian")
    )
    config["fit"].setdefault("student_t_dof", 2.0)
    config["fit"].setdefault("flux_error_floor_frac", 0.0)
    config["fit"].setdefault("flux_error_jitter", 0.0)
    config["fit"].setdefault("maxiter", 80)
    config["fit"].setdefault("learning_rate", 0.1)
    config["fit"].setdefault("tolerance", 1.0e-5)
    config["fit"].setdefault("patience", 18)
    config["fit"].setdefault("prior_weight", 1.0)
    config["fit"].setdefault("trace_mode", "full")
    config["fit"].setdefault("trace_interval", 1)
    config["fit"].setdefault("scan_unroll", 1)
    config["fit"].setdefault("donate_optimizer_inputs", False)
    config["fit"].setdefault("remat_model_mags", False)
    config["fit"].setdefault("batch_grad_mode", "per_galaxy")
    config["fit"].setdefault("priors", {})
    config["band_calibration"] = dict(config["band_calibration"] or {})
    config["band_calibration"].setdefault("mode", "none")
    config["band_calibration"].setdefault("offsets_mag", {})
    config["band_calibration"].setdefault("flux_multipliers", {})
    _apply_band_calibration(config)
    config["calibration"] = dict(config["calibration"] or {})
    config["calibration"].setdefault("global_sed_scale", {})
    config["calibration"]["global_sed_scale"] = dict(
        config["calibration"]["global_sed_scale"] or {}
    )
    config["calibration"]["global_sed_scale"].setdefault("enabled", False)
    config["calibration"]["global_sed_scale"].setdefault("mode", "disabled")
    config["calibration"]["global_sed_scale"].setdefault(
        "parameterization", "log_alpha"
    )
    config["calibration"]["global_sed_scale"].setdefault("initial_log_alpha", 0.0)
    config["calibration"]["global_sed_scale"].setdefault("prior_sigma_log_alpha", 0.10)
    config["calibration"]["global_sed_scale"].setdefault("trainable", False)
    config["calibration"].setdefault("per_band_zero_points", {})
    config["calibration"]["per_band_zero_points"] = dict(
        config["calibration"]["per_band_zero_points"] or {}
    )
    config["calibration"]["per_band_zero_points"].setdefault("enabled", False)
    config["fit"]["population"] = dict(config["fit"].get("population") or {})
    config["fit"]["population"].setdefault("prior_weight", 1.0)
    config["fit"]["population"].setdefault("sigma_floor", 0.03)
    config["fit"]["population"].setdefault("hyper_mu_scale", 5.0)
    config["fit"]["population"].setdefault("relations", {})

    config["sample"] = dict(config["sample"] or {})
    config["sample"].setdefault("num_warmup", 100)
    config["sample"].setdefault("num_samples", 200)
    config["sample"].setdefault("num_chains", 1)
    config["sample"].setdefault("sampler", "nuts")
    config["sample"].setdefault("chain_method", "parallel")
    config["sample"].setdefault("target_accept_prob", 0.85)
    config["sample"].setdefault("max_tree_depth", 10)
    config["sample"].setdefault("num_steps", 8)
    config["sample"].setdefault("dense_mass", False)
    config["sample"].setdefault("jit_model_args", False)
    config["sample"].setdefault("seed", 42)
    config["sample"].setdefault("progress_bar", True)
    config["sample"].setdefault("init_from_map", True)
    if "init_strategy" not in config["sample"]:
        config["sample"]["init_strategy"] = (
            "map" if bool(config["sample"].get("init_from_map", True)) else "config"
        )
    if config["sample"]["init_strategy"] in {"config", "random_uniform"}:
        config["sample"]["init_from_map"] = False
    elif config["sample"]["init_strategy"] in {"map", "map_jitter"}:
        config["sample"]["init_from_map"] = True
    config["sample"].setdefault("init_jitter_scale", 0.25)
    config["sample"].setdefault("save_samples", True)
    config["sample"].setdefault("priors", {})
    config["sample"].setdefault("mclmc_l", "auto")
    config["sample"].setdefault("mclmc_step_size", "auto")
    config["sample"].setdefault("mclmc_inverse_mass_matrix", "identity")
    config["sample"].setdefault("mclmc_progress_chunk_size", 16)
    config["sample"].setdefault("mclmc_debug", False)
    config["sample"].setdefault("posterior_predictive_batch_size", 512)
    _apply_prior_set(config)
    _apply_redshift_prior(config)

    config["selection"].setdefault("index", None)
    config["selection"].setdefault("require_positive_flux", True)
    config["selection"].setdefault(
        "nondetection_policy",
        (
            "drop"
            if config["selection"].get("require_positive_flux", True)
            else "gaussian_flux"
        ),
    )
    config["selection"].setdefault("sort_by_flux", None)

    config["truth"].setdefault("redshift_column", redshift.get("truth_column"))
    config["truth"].setdefault("parameter_columns", {})

    runtime = dict(DEFAULT_RUNTIME_CONFIG)
    runtime.update(dict(config["runtime"] or {}))
    config["runtime"] = runtime

    config["reporting"] = dict(config["reporting"] or {})
    config["reporting"].setdefault("level", "full")
    config["reporting"].setdefault("save_sed_samples", 0)
    config["reporting"].setdefault("plot_filters", True)
    config["reporting"].setdefault("plot_ground_truth", False)
    config["output"] = dict(config["output"] or {})
    config["output"].setdefault("format", "both")
    config["output"].setdefault("verbose_benchmark", False)

    validate_config(config)
    return config


def _load_config_tree(path: Path, seen: set[Path]) -> dict[str, Any]:
    if path in seen:
        chain = " -> ".join(str(item) for item in [*seen, path])
        raise ConfigValidationError(f"Config extends cycle: {chain}")
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"Config must be a YAML mapping: {path}")

    seen = {*seen, path}
    extends = raw.pop("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    if not isinstance(extends, list):
        raise ConfigValidationError("extends must be a string or list of strings")

    merged: dict[str, Any] = {}
    for item in extends:
        if not isinstance(item, str) or not item:
            raise ConfigValidationError("extends entries must be non-empty strings")
        parent = Path(item)
        if not parent.is_absolute():
            parent = path.parent / parent
        merged = _deep_merge(merged, _load_config_tree(parent.resolve(), seen=seen))
    return _deep_merge(merged, raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and value.get("__replace__") is True:
            replacement = dict(value)
            replacement.pop("__replace__", None)
            merged[key] = replacement
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_photometric_likelihood(value: Any) -> str:
    name = str(value).strip().lower().replace("-", "_")
    aliases = {
        "chi2": "gaussian",
        "gaussian_chi2": "gaussian",
        "normal": "gaussian",
        "student": "student_t",
        "studentt": "student_t",
    }
    return aliases.get(name, name)


def _normalize_model_dust_model(value: Any) -> str:
    name = str(value).strip().lower().replace("-", "_")
    aliases = {
        "charlot_fall": "charlot_fall_powerlaw",
        "charlot_fall_power_law": "charlot_fall_powerlaw",
    }
    return aliases.get(name, name)


def _is_popcosmos_like_sfh(sfh_model: Any) -> bool:
    return str(sfh_model) in {
        "popcosmos_bins",
        "diffstar_reduced6",
        "diffsky_basic",
        "spline15d",
    }


def _expand_config_shorthands(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    runtime = config.get("runtime")
    runtime_preset = config.pop("runtime_preset", None)
    if isinstance(runtime, str):
        config["runtime"] = _named_preset(RUNTIME_PRESETS, runtime, "runtime")
    elif runtime_preset and "runtime" not in config:
        config["runtime"] = _named_preset(
            RUNTIME_PRESETS, str(runtime_preset), "runtime"
        )

    bands = config.get("bands")
    if isinstance(bands, str):
        config["bands"] = _named_preset(BAND_PRESETS, bands, "bands")

    groups = config.pop("column_groups", [])
    if isinstance(groups, str):
        groups = [groups]
    if groups is None:
        groups = []
    if not isinstance(groups, list):
        raise ConfigValidationError("column_groups must be a string or list")

    extra_columns = config.get("extra_columns", [])
    if isinstance(extra_columns, str):
        extra_columns = [extra_columns]
    if extra_columns is None:
        extra_columns = []
    if not isinstance(extra_columns, list):
        raise ConfigValidationError("extra_columns must be a string or list")

    expanded_columns: list[str] = []
    for item in [*groups, *extra_columns]:
        if not isinstance(item, str) or not item:
            raise ConfigValidationError(
                "column groups and extra columns must be strings"
            )
        if item in COLUMN_GROUPS:
            expanded_columns.extend(COLUMN_GROUPS[item])
        else:
            expanded_columns.append(item)
    config["extra_columns"] = sorted(dict.fromkeys(expanded_columns))

    return config


def _apply_prior_set(config: dict[str, Any]) -> None:
    prior_set = config.get("prior_set")
    if prior_set is None:
        return
    name = str(prior_set)
    if name == "popcosmos_like":
        raise ConfigValidationError(
            "prior_set='popcosmos_like' is reserved until exact POP-COSMOS "
            "parameter mapping and units are implemented."
        )
    if name not in PRIOR_SETS:
        expected = sorted([*PRIOR_SETS, "popcosmos_like"])
        raise ConfigValidationError(
            f"Unknown prior_set {name!r}; expected one of {expected}"
        )
    free = set(config["fit"]["free_parameters"])
    named_priors = {
        key: dict(value) for key, value in PRIOR_SETS[name].items() if key in free
    }
    fit_priors = dict(named_priors)
    fit_priors.update(config["fit"].get("priors") or {})
    sample_priors = dict(named_priors)
    sample_priors.update(config["sample"].get("priors") or {})
    config["fit"]["priors"] = fit_priors
    config["sample"]["priors"] = sample_priors


def _apply_redshift_prior(config: dict[str, Any]) -> None:
    prior = config.get("redshift", {}).get("prior_z") or {}
    if not isinstance(prior, dict) or str(prior.get("mode", "none")) != "gaussian":
        return
    free = set(config["fit"]["free_parameters"])
    if "z_obs" not in free:
        return
    z_prior = {
        "type": "normal",
        "loc": "from_base",
        "scale": "from_base",
        "scale_parameter": "z_obs_prior_sigma",
    }
    config["fit"].setdefault("priors", {})
    config["sample"].setdefault("priors", {})
    config["fit"]["priors"]["z_obs"] = {
        **config["fit"]["priors"].get("z_obs", {}),
        **z_prior,
    }
    config["sample"]["priors"]["z_obs"] = {
        **config["sample"]["priors"].get("z_obs", {}),
        **z_prior,
    }


def _apply_band_calibration(config: dict[str, Any]) -> None:
    calibration = config.get("band_calibration", {}) or {}
    mode = str(calibration.get("mode", "none"))
    offsets_mag = calibration.get("offsets_mag") or {}
    flux_multipliers = calibration.get("flux_multipliers") or {}
    values = []
    multipliers = []
    for band in config["bands"]:
        name = str(band["name"])
        offset = 0.0
        multiplier = 1.0
        if mode == "fixed_offsets":
            offset = float(offsets_mag.get(name, 0.0))
            multiplier = float(flux_multipliers.get(name, 1.0))
            if multiplier > 0.0:
                offset += -2.5 * math.log10(multiplier)
        values.append(offset)
        multipliers.append(multiplier)
    config["fit"]["band_calibration_offsets_mag"] = values
    config["fit"]["band_calibration_flux_multipliers"] = multipliers


def _named_preset(presets: dict[str, Any], name: str, label: str) -> Any:
    if name not in presets:
        raise ConfigValidationError(
            f"Unknown {label} preset {name!r}; expected one of {sorted(presets)}"
        )
    value = presets[name]
    if isinstance(value, dict):
        return _deep_merge({}, value)
    if isinstance(value, list):
        return [
            _deep_merge({}, item) if isinstance(item, dict) else item for item in value
        ]
    return value


def validate_config(config: dict[str, Any]) -> None:
    """Validate the normalized runtime configuration.

    Validation intentionally checks structure and scalar contracts only. It does
    not require local data files to exist, so CI can validate configs without
    shipping the private or large FS2 parquet files.
    """
    errors: list[str] = []
    _require_nonempty(config, "catalog_path", errors)
    _require_nonempty(config, "ssp_path", errors)
    _validate_bands(config.get("bands"), errors)
    _validate_selection(config.get("selection", {}), errors)
    _validate_redshift(config.get("redshift", {}), errors)
    _validate_model(config.get("model", {}), errors)
    _validate_fit(config.get("fit", {}), errors)
    _validate_model_fit_contract(config.get("model", {}), config.get("fit", {}), errors)
    _validate_nebular_emission(config.get("nebular_emission"), errors)
    _validate_sample(config.get("sample", {}), config.get("fit", {}), errors)
    _validate_truth(config.get("truth", {}), errors)
    _validate_runtime(config.get("runtime", {}), errors)
    _validate_reporting(config.get("reporting", {}), errors)
    _validate_output(config.get("output", {}), errors)
    _validate_band_calibration(config.get("band_calibration", {}), errors)
    _validate_calibration(config.get("calibration", {}), errors)
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise ConfigValidationError(f"Invalid configuration:\n{detail}")


def validate_catalog_columns(
    config: dict[str, Any], available_columns: set[str] | list[str] | tuple[str, ...]
) -> None:
    """Validate that every configured catalog column exists in a data source."""
    available = set(available_columns)
    missing = [
        column
        for column in _configured_catalog_columns(config)
        if column not in available
    ]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ConfigValidationError(f"Configured catalog columns are missing: {joined}")


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve paths relative to the current working directory or config dir."""
    p = Path(path)
    if p.is_absolute():
        return p
    if base_dir is None:
        return p.resolve()
    return (Path(base_dir) / p).resolve()


def _require_nonempty(config: dict[str, Any], key: str, errors: list[str]) -> None:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty path string")


def _validate_selection(selection: dict[str, Any], errors: list[str]) -> None:
    policy = str(selection.get("nondetection_policy", "drop"))
    if policy not in SUPPORTED_NONDETECTION_POLICIES:
        errors.append(
            "selection.nondetection_policy must be one of "
            f"{sorted(SUPPORTED_NONDETECTION_POLICIES)}"
        )
    if policy == "upper_limit":
        errors.append(
            "selection.nondetection_policy='upper_limit' is reserved but not implemented"
        )


def _validate_bands(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append("bands must be a non-empty list")
        return
    seen_names: set[str] = set()
    seen_columns: set[str] = set()
    for index, band in enumerate(value):
        if not isinstance(band, dict):
            errors.append(f"bands[{index}] must be a mapping")
            continue
        name = band.get("name")
        column = band.get("column")
        if not isinstance(name, str) or not name:
            errors.append(f"bands[{index}].name must be a non-empty string")
        elif name in seen_names:
            errors.append(f"bands[{index}].name duplicates {name!r}")
        else:
            seen_names.add(name)
        if not isinstance(column, str) or not column:
            errors.append(f"bands[{index}].column must be a non-empty string")
        elif column in seen_columns:
            errors.append(f"bands[{index}].column duplicates {column!r}")
        else:
            seen_columns.add(column)
        units = band.get("units", "fnu_cgs")
        if units not in SUPPORTED_PHOTOMETRY_UNITS:
            errors.append(
                f"bands[{index}].units must be one of {sorted(SUPPORTED_PHOTOMETRY_UNITS)}"
            )
        _positive_float(
            band.get("sigma_mag", 0.05), f"bands[{index}].sigma_mag", errors
        )
        filter_config = band.get("filter", {})
        if filter_config is not None and not isinstance(filter_config, dict):
            errors.append(f"bands[{index}].filter must be a mapping when provided")
        _optional_string(
            band.get("error_column"), f"bands[{index}].error_column", errors
        )
        error_units = band.get("error_units", units)
        if error_units not in SUPPORTED_PHOTOMETRY_UNITS:
            errors.append(
                f"bands[{index}].error_units must be one of {sorted(SUPPORTED_PHOTOMETRY_UNITS)}"
            )
        if band.get("sigma_mag_floor") is not None:
            _positive_float(
                band.get("sigma_mag_floor"), f"bands[{index}].sigma_mag_floor", errors
            )
        if band.get("sigma_mag_ceiling") is not None:
            _positive_float(
                band.get("sigma_mag_ceiling"),
                f"bands[{index}].sigma_mag_ceiling",
                errors,
            )


def _validate_redshift(redshift: dict[str, Any], errors: list[str]) -> None:
    for removed_key in ("prior_interval", "prior_intervals"):
        if removed_key in redshift:
            errors.append(f"redshift.{removed_key} was removed; fit z_obs directly")
    if "multi_start" in redshift:
        errors.append(
            "redshift.multi_start was removed; use posterior sampling for "
            "redshift inference"
        )
    initial = redshift.get("initial", "catalog_column")
    if initial not in SUPPORTED_REDSHIFT_INITIALS:
        errors.append(
            f"redshift.initial must be one of {sorted(SUPPORTED_REDSHIFT_INITIALS)}"
        )
    _optional_string(redshift.get("column"), "redshift.column", errors)
    _optional_string(redshift.get("truth_column"), "redshift.truth_column", errors)
    _finite_float(redshift.get("fixed_value"), "redshift.fixed_value", errors)
    _finite_float(redshift.get("seed", 42), "redshift.seed", errors)
    z_min = _finite_float(redshift.get("min"), "redshift.min", errors)
    z_max = _finite_float(redshift.get("max"), "redshift.max", errors)
    if z_min is not None and z_max is not None and z_min >= z_max:
        errors.append("redshift.min must be smaller than redshift.max")
    prior = redshift.get("prior_z", {"mode": "none"})
    if prior is None:
        prior = {"mode": "none"}
    if not isinstance(prior, dict):
        errors.append("redshift.prior_z must be a mapping")
    else:
        mode = str(prior.get("mode", "none"))
        if mode not in SUPPORTED_REDSHIFT_PRIORS:
            errors.append(
                f"redshift.prior_z.mode must be one of {sorted(SUPPORTED_REDSHIFT_PRIORS)}"
            )
        if mode == "gaussian":
            _positive_float(prior.get("sigma", 0.35), "redshift.prior_z.sigma", errors)
            _positive_float(
                prior.get("sigma_min", 0.02), "redshift.prior_z.sigma_min", errors
            )
        if mode in {"top_hat", "phz_interval"}:
            _positive_float(
                prior.get("penalty", 1.0e6), "redshift.prior_z.penalty", errors
            )
            _optional_string(
                prior.get("min_column"), "redshift.prior_z.min_column", errors
            )
            _optional_string(
                prior.get("max_column"), "redshift.prior_z.max_column", errors
            )


def _validate_model(model: dict[str, Any], errors: list[str]) -> None:
    n_sfh_bins = model.get("n_sfh_bins")
    if not isinstance(n_sfh_bins, int) or n_sfh_bins < 2:
        errors.append("model.n_sfh_bins must be an integer >= 2")
    sfh_model = str(model.get("sfh_model", "lognormal"))
    if sfh_model not in SUPPORTED_SFH_MODELS:
        errors.append(f"model.sfh_model must be one of {sorted(SUPPORTED_SFH_MODELS)}")
    ssp_model = str(model.get("ssp_model", "dense"))
    if ssp_model not in SUPPORTED_SSP_MODELS:
        errors.append(f"model.ssp_model must be one of {sorted(SUPPORTED_SSP_MODELS)}")
    stellar_model = str(model.get("stellar_metallicity_model", "mdf"))
    if stellar_model not in SUPPORTED_STELLAR_METALLICITY_MODELS:
        errors.append(
            "model.stellar_metallicity_model must be one of "
            f"{sorted(SUPPORTED_STELLAR_METALLICITY_MODELS)}"
        )
    dust_model = _normalize_model_dust_model(model.get("dust_model", "legacy"))
    if dust_model not in SUPPORTED_MODEL_DUST_MODELS:
        errors.append(
            f"model.dust_model must be one of {sorted(SUPPORTED_MODEL_DUST_MODELS)}"
        )
    igm_model = str(model.get("igm_model", "none"))
    if igm_model not in SUPPORTED_IGM_MODELS:
        errors.append(f"model.igm_model must be one of {sorted(SUPPORTED_IGM_MODELS)}")
    sfh_time_grid = str(model.get("sfh_time_grid", "linear"))
    if sfh_time_grid not in {"linear", "prospector_step"}:
        errors.append(
            "model.sfh_time_grid must be one of ['linear', 'prospector_step']"
        )
    nebular_model = str(model.get("nebular_model", "fixed_ssp"))
    if nebular_model not in SUPPORTED_NEBULAR_MODELS:
        errors.append(
            f"model.nebular_model must be one of {sorted(SUPPORTED_NEBULAR_MODELS)}"
        )
    agn_model = str(model.get("agn_model", "none"))
    if agn_model not in SUPPORTED_AGN_MODELS:
        errors.append(f"model.agn_model must be one of {sorted(SUPPORTED_AGN_MODELS)}")
    agn_host_attenuation = str(model.get("agn_host_attenuation", "none"))
    if agn_host_attenuation not in SUPPORTED_AGN_HOST_ATTENUATION_MODES:
        errors.append(
            "model.agn_host_attenuation must be one of "
            f"{sorted(SUPPORTED_AGN_HOST_ATTENUATION_MODES)}"
        )
    agn_host_attenuation_scale = _nonnegative_float(
        model.get("agn_host_attenuation_scale", 1.0),
        "model.agn_host_attenuation_scale",
        errors,
    )
    if (
        agn_host_attenuation == "fsps_diffuse_unit_tau"
        and agn_host_attenuation_scale is not None
        and not math.isclose(
            agn_host_attenuation_scale,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        errors.append(
            "model.agn_host_attenuation_scale must be 1.0 when "
            "model.agn_host_attenuation='fsps_diffuse_unit_tau'; this mode "
            "matches FSPS agn_dust.f90 unit-tau diffuse attenuation and does "
            "not use a fitted scale"
        )
    agn_igm_order = str(model.get("agn_igm_order", "pre_igm"))
    if agn_igm_order not in SUPPORTED_AGN_IGM_ORDERS:
        errors.append(
            "model.agn_igm_order must be one of " f"{sorted(SUPPORTED_AGN_IGM_ORDERS)}"
        )
    agn_baked_attenuation = str(model.get("agn_baked_attenuation", "none"))
    if agn_baked_attenuation not in SUPPORTED_AGN_BAKED_ATTENUATION_MODES:
        errors.append(
            "model.agn_baked_attenuation must be one of "
            f"{sorted(SUPPORTED_AGN_BAKED_ATTENUATION_MODES)}"
        )
    _finite_float(
        model.get("agn_baked_dust_index", -0.7),
        "model.agn_baked_dust_index",
        errors,
    )
    emission_line_corrections = str(model.get("emission_line_corrections", "none"))
    if emission_line_corrections not in SUPPORTED_EMISSION_LINE_CORRECTIONS:
        errors.append(
            "model.emission_line_corrections must be one of "
            f"{sorted(SUPPORTED_EMISSION_LINE_CORRECTIONS)}"
        )
    _finite_float(
        model.get("birth_cloud_slope", -1.0), "model.birth_cloud_slope", errors
    )
    _finite_float(model.get("dust_tesc_logyr", 7.0), "model.dust_tesc_logyr", errors)
    _finite_float(model.get("dust1_index", -1.0), "model.dust1_index", errors)
    _positive_float(model.get("z_sun", 0.0134), "model.z_sun", errors)
    _positive_float(
        model.get("stellar_metallicity_scatter_dex", 0.2),
        "model.stellar_metallicity_scatter_dex",
        errors,
    )
    if ssp_model == "compressed_basis":
        _require_model_path(
            model.get("compressed_ssp_path"), "model.compressed_ssp_path", errors
        )
    else:
        _optional_string(
            model.get("compressed_ssp_path"), "model.compressed_ssp_path", errors
        )
    if nebular_model == "gas_grid":
        _require_model_path(model.get("gas_grid_path"), "model.gas_grid_path", errors)
    else:
        _optional_string(model.get("gas_grid_path"), "model.gas_grid_path", errors)
    if nebular_model == "compressed_gas_grid":
        _require_model_path(
            model.get("compressed_gas_grid_path"),
            "model.compressed_gas_grid_path",
            errors,
        )
    else:
        _optional_string(
            model.get("compressed_gas_grid_path"),
            "model.compressed_gas_grid_path",
            errors,
        )
    _optional_string(
        model.get("stellar_only_ssp_path"), "model.stellar_only_ssp_path", errors
    )
    if agn_model == "template_grid":
        _require_model_path(
            model.get("agn_template_path"), "model.agn_template_path", errors
        )
    else:
        _optional_string(
            model.get("agn_template_path"), "model.agn_template_path", errors
        )
    if agn_model == "fsps_component_grid":
        _require_model_path(
            model.get("agn_component_grid_path"),
            "model.agn_component_grid_path",
            errors,
        )
    else:
        _optional_string(
            model.get("agn_component_grid_path"),
            "model.agn_component_grid_path",
            errors,
        )
    if agn_model == "compressed_fsps_component_grid":
        _require_model_path(
            model.get("compressed_agn_component_grid_path"),
            "model.compressed_agn_component_grid_path",
            errors,
        )
    else:
        _optional_string(
            model.get("compressed_agn_component_grid_path"),
            "model.compressed_agn_component_grid_path",
            errors,
        )
    if emission_line_corrections == "popcosmos_table":
        _require_model_path(
            model.get("emission_line_correction_path"),
            "model.emission_line_correction_path",
            errors,
        )
    else:
        _optional_string(
            model.get("emission_line_correction_path"),
            "model.emission_line_correction_path",
            errors,
        )
    fixed = model.get("fixed_parameters")
    if not isinstance(fixed, dict):
        errors.append("model.fixed_parameters must be a mapping")
    else:
        for name, value in fixed.items():
            _finite_float(value, f"model.fixed_parameters.{name}", errors)
    parameter_columns = model.get("parameter_columns", {})
    if not isinstance(parameter_columns, dict):
        errors.append("model.parameter_columns must be a mapping")
    else:
        for name, column in parameter_columns.items():
            if not isinstance(name, str) or not isinstance(column, str) or not column:
                errors.append("model.parameter_columns keys and values must be strings")


def _validate_model_fit_contract(
    model: dict[str, Any], fit: dict[str, Any], errors: list[str]
) -> None:
    free = fit.get("free_parameters", {})
    if not isinstance(free, dict):
        return
    free_names = set(free)
    sfh_model = str(model.get("sfh_model", "lognormal"))
    if sfh_model == "popcosmos_bins":
        _validate_popcosmos_free_parameters(model, free_names, errors)
    elif sfh_model == "diffstar_reduced6":
        _validate_diffstar_free_parameters(model, free_names, errors)
    elif sfh_model == "diffsky_basic":
        _validate_diffsky_basic_free_parameters(model, free_names, errors)
    elif sfh_model == "spline15d":
        _validate_spline15d_free_parameters(model, free_names, errors)
    elif sfh_model == "lognormal":
        _validate_lognormal_free_parameters(free_names, errors)


def _validate_popcosmos_free_parameters(
    model: dict[str, Any], free_names: set[str], errors: list[str]
) -> None:
    legacy_forbidden = {
        "sfh_t_peak",
        "sfh_tau",
        "log10_sfr",
        "metallicity_scatter",
    }
    forbidden_active = sorted(free_names & legacy_forbidden)
    for name in forbidden_active:
        errors.append(
            f"fit.free_parameters.{name} is ignored by model.sfh_model='popcosmos_bins'"
        )

    if str(model.get("stellar_metallicity_model", "single")) != "single":
        errors.append(
            "model.sfh_model='popcosmos_bins' requires "
            "model.stellar_metallicity_model='single'"
        )
    if _normalize_model_dust_model(
        model.get("dust_model", "charlot_fall_powerlaw")
    ) not in {"charlot_fall_powerlaw", "prospector_fsps"}:
        errors.append(
            "model.sfh_model='popcosmos_bins' requires "
            "model.dust_model='charlot_fall_powerlaw' or 'prospector_fsps'"
        )

    allowed = set(POPCOSMOS_PARAMETER_NAMES) | set(DIFFSKY_TRUTH_BASIC_PARAMETER_NAMES)
    nebular_model = str(model.get("nebular_model", "fixed_ssp"))
    agn_model = str(model.get("agn_model", "none"))
    gas_names = {"log10_gas_metallicity", "log10_gas_ionization"}
    agn_names = {"ln_fagn", "ln_tauagn"}
    if nebular_model not in {"gas_grid", "compressed_gas_grid"}:
        allowed -= gas_names
        for name in sorted(free_names & gas_names):
            errors.append(
                f"fit.free_parameters.{name} requires model.nebular_model='gas_grid' "
                "or 'compressed_gas_grid'"
            )
    if agn_model == "none":
        allowed -= agn_names
        for name in sorted(free_names & agn_names):
            errors.append(f"fit.free_parameters.{name} requires an active AGN model")
    unknown = sorted(free_names - allowed)
    for name in unknown:
        if (
            name not in legacy_forbidden
            and name not in gas_names
            and name not in agn_names
        ):
            errors.append(
                f"fit.free_parameters.{name} is not used by "
                "model.sfh_model='popcosmos_bins'"
            )


def _validate_diffstar_free_parameters(
    model: dict[str, Any], free_names: set[str], errors: list[str]
) -> None:
    legacy_forbidden = {
        "sfh_t_peak",
        "sfh_tau",
        "log10_sfr",
        "dlog10_sfr_1",
        "dlog10_sfr_2",
        "dlog10_sfr_3",
        "dlog10_sfr_4",
        "dlog10_sfr_5",
        "dlog10_sfr_6",
        "log10_metallicity",
        "metallicity_scatter",
    }
    forbidden_active = sorted(free_names & legacy_forbidden)
    for name in forbidden_active:
        errors.append(
            f"fit.free_parameters.{name} is ignored by "
            "model.sfh_model='diffstar_reduced6'"
        )

    if str(model.get("stellar_metallicity_model", "single")) != "single":
        errors.append(
            "model.sfh_model='diffstar_reduced6' requires "
            "model.stellar_metallicity_model='single'"
        )
    if _normalize_model_dust_model(
        model.get("dust_model", "charlot_fall_powerlaw")
    ) not in {"charlot_fall_powerlaw", "prospector_fsps"}:
        errors.append(
            "model.sfh_model='diffstar_reduced6' requires "
            "model.dust_model='charlot_fall_powerlaw' or 'prospector_fsps'"
        )

    allowed = set(DIFFSTAR_REDUCED6_PARAMETER_NAMES)
    nebular_model = str(model.get("nebular_model", "fixed_ssp"))
    agn_model = str(model.get("agn_model", "none"))
    gas_names = {"log10_gas_metallicity", "log10_gas_ionization"}
    agn_names = {"ln_fagn", "ln_tauagn"}
    if nebular_model not in {"gas_grid", "compressed_gas_grid"}:
        allowed -= gas_names
        for name in sorted(free_names & gas_names):
            errors.append(
                f"fit.free_parameters.{name} requires model.nebular_model='gas_grid' "
                "or 'compressed_gas_grid'"
            )
    if agn_model == "none":
        allowed -= agn_names
        for name in sorted(free_names & agn_names):
            errors.append(f"fit.free_parameters.{name} requires an active AGN model")
    unknown = sorted(free_names - allowed)
    for name in unknown:
        if (
            name not in legacy_forbidden
            and name not in gas_names
            and name not in agn_names
        ):
            errors.append(
                f"fit.free_parameters.{name} is not used by "
                "model.sfh_model='diffstar_reduced6'"
            )


def _validate_diffsky_basic_free_parameters(
    model: dict[str, Any], free_names: set[str], errors: list[str]
) -> None:
    legacy_forbidden = {
        "sfh_t_peak",
        "sfh_tau",
        "log10_sfr",
        "dlog10_sfr_1",
        "dlog10_sfr_2",
        "dlog10_sfr_3",
        "dlog10_sfr_4",
        "dlog10_sfr_5",
        "dlog10_sfr_6",
        "log10_metallicity",
        "metallicity_scatter",
        "tau2",
        "dust_index_n",
        "tau1_over_tau2",
        "log10_gas_metallicity",
        "log10_gas_ionization",
        "ln_fagn",
        "ln_tauagn",
    }
    for name in sorted(free_names & legacy_forbidden):
        errors.append(
            f"fit.free_parameters.{name} is not used by "
            "model.sfh_model='diffsky_basic'"
        )

    metallicity_model = str(model.get("stellar_metallicity_model", "single"))
    if metallicity_model not in {"single", "lognormal_mdf_fixed_scatter"}:
        errors.append(
            "model.sfh_model='diffsky_basic' requires "
            "model.stellar_metallicity_model='single' or "
            "'lognormal_mdf_fixed_scatter'"
        )
    if _normalize_model_dust_model(model.get("dust_model", "prospector_fsps")) not in {
        "charlot_fall_powerlaw",
        "prospector_fsps",
    }:
        errors.append(
            "model.sfh_model='diffsky_basic' requires "
            "model.dust_model='charlot_fall_powerlaw' or 'prospector_fsps'"
        )
    if str(model.get("nebular_model", "fixed_ssp")) != "fixed_ssp":
        errors.append(
            "model.sfh_model='diffsky_basic' currently requires "
            "model.nebular_model='fixed_ssp'; HLTDS does not expose gas "
            "metallicity/ionization latents in the prepared dataset"
        )
    if str(model.get("agn_model", "none")) != "none":
        errors.append(
            "model.sfh_model='diffsky_basic' currently requires "
            "model.agn_model='none'; AGN latents are not fit in this schema"
        )

    allowed = set(DIFFSKY_BASIC_PARAMETER_NAMES)
    unknown = sorted(free_names - allowed)
    for name in unknown:
        if name not in legacy_forbidden:
            errors.append(
                f"fit.free_parameters.{name} is not used by "
                "model.sfh_model='diffsky_basic'"
            )


def _validate_spline15d_free_parameters(
    model: dict[str, Any], free_names: set[str], errors: list[str]
) -> None:
    expected = {
        "z_obs",
        "log10_stellar_mass",
        "log10_stellar_metallicity",
        "dust_av",
        "dust_delta",
        *(f"sfh_dlog_sfr_{index:02d}" for index in range(1, 11)),
    }
    missing = sorted(expected - free_names)
    unknown = sorted(free_names - expected)
    if missing:
        errors.append(
            "model.sfh_model='spline15d' requires all 15 parameters; missing "
            + ", ".join(missing)
        )
    for name in unknown:
        errors.append(
            f"fit.free_parameters.{name} is not used by " "model.sfh_model='spline15d'"
        )
    metallicity_model = str(model.get("stellar_metallicity_model", "single"))
    if metallicity_model not in {"single", "lognormal_mdf_fixed_scatter"}:
        errors.append(
            "model.sfh_model='spline15d' requires "
            "model.stellar_metallicity_model='single' or "
            "'lognormal_mdf_fixed_scatter'"
        )
    if str(model.get("nebular_model", "fixed_ssp")) != "fixed_ssp":
        errors.append("model.sfh_model='spline15d' currently requires fixed_ssp")
    if str(model.get("agn_model", "none")) != "none":
        errors.append("model.sfh_model='spline15d' currently requires agn_model=none")


def _validate_lognormal_free_parameters(
    free_names: set[str], errors: list[str]
) -> None:
    allowed = {
        "z_obs",
        "log10_sfr",
        "sfh_t_peak",
        "sfh_tau",
        "log10_metallicity",
        "metallicity_scatter",
        "dust_av",
        "dust_slope",
        "log10_formed_mass_msun",
        "cosmos_ebv_1",
        "cosmos_ebv_2",
        "cosmos_frac_1",
        "cosmos_frac_2",
        "cosmos_ext_curve_1",
        "cosmos_ext_curve_2",
    }
    ignored = sorted(free_names - allowed)
    for name in ignored:
        errors.append(
            f"fit.free_parameters.{name} is not used by model.sfh_model='lognormal'"
        )


def _validate_nebular_emission(value: Any, errors: list[str]) -> None:
    if not isinstance(value, str):
        errors.append("nebular_emission must be a string")
        return
    if value not in SUPPORTED_NEBULAR_EMISSION_MODES:
        errors.append(
            "nebular_emission must be one of "
            f"{sorted(SUPPORTED_NEBULAR_EMISSION_MODES)}"
        )


def _validate_fit(fit: dict[str, Any], errors: list[str]) -> None:
    for removed_key in (
        "fast_warmstart_only",
        "fast_grid_search",
        "redshift_grid_size",
        "redshift_grid_width",
        "fast_grid_parameters",
        "fast_grid_prior_width",
    ):
        if removed_key in fit:
            errors.append(f"fit.{removed_key} was removed from the public workflow")
    method = str(fit.get("method", "jax_adam")).lower()
    if method not in SUPPORTED_FIT_METHODS:
        errors.append(f"fit.method must be one of {sorted(SUPPORTED_FIT_METHODS)}")
    likelihood_space = str(fit.get("likelihood_space", "flux")).lower()
    if likelihood_space not in SUPPORTED_LIKELIHOOD_SPACES:
        errors.append(
            "fit.likelihood_space must be one of "
            f"{sorted(SUPPORTED_LIKELIHOOD_SPACES)}"
        )
    photometric_likelihood = _normalize_photometric_likelihood(
        fit.get("photometric_likelihood", "gaussian")
    )
    if photometric_likelihood not in SUPPORTED_PHOTOMETRIC_LIKELIHOODS:
        errors.append(
            "fit.photometric_likelihood must be one of "
            f"{sorted(SUPPORTED_PHOTOMETRIC_LIKELIHOODS)}"
        )
    _positive_float(fit.get("student_t_dof", 2.0), "fit.student_t_dof", errors)
    _nonnegative_float(
        fit.get("flux_error_floor_frac", 0.0), "fit.flux_error_floor_frac", errors
    )
    _nonnegative_float(
        fit.get("flux_error_jitter", 0.0), "fit.flux_error_jitter", errors
    )
    _positive_int(fit.get("maxiter"), "fit.maxiter", errors)
    _positive_float(fit.get("learning_rate"), "fit.learning_rate", errors)
    _positive_float(fit.get("tolerance"), "fit.tolerance", errors)
    _positive_int(fit.get("patience"), "fit.patience", errors)
    _positive_float(fit.get("prior_weight", 1.0), "fit.prior_weight", errors)
    trace_mode = str(fit.get("trace_mode", "full")).lower()
    if trace_mode not in SUPPORTED_FIT_TRACE_MODES:
        errors.append(
            f"fit.trace_mode must be one of {sorted(SUPPORTED_FIT_TRACE_MODES)}"
        )
    _positive_int(fit.get("trace_interval", 1), "fit.trace_interval", errors)
    _positive_int(fit.get("scan_unroll", 1), "fit.scan_unroll", errors)
    for key in ("donate_optimizer_inputs", "remat_model_mags"):
        if not isinstance(fit.get(key, False), bool):
            errors.append(f"fit.{key} must be a boolean")
    batch_grad_mode = str(fit.get("batch_grad_mode", "per_galaxy")).lower()
    if batch_grad_mode not in SUPPORTED_FIT_BATCH_GRAD_MODES:
        errors.append(
            "fit.batch_grad_mode must be one of "
            f"{sorted(SUPPORTED_FIT_BATCH_GRAD_MODES)}"
        )
    _validate_population_config(fit.get("population", {}), errors)
    free = fit.get("free_parameters")
    if not isinstance(free, dict) or not free:
        errors.append("fit.free_parameters must be a non-empty mapping")
        return
    for name, spec in free.items():
        if not isinstance(spec, dict):
            errors.append(f"fit.free_parameters.{name} must be a mapping")
            continue
        bounds = spec.get("bounds")
        if not isinstance(bounds, list | tuple) or len(bounds) != 2:
            errors.append(f"fit.free_parameters.{name}.bounds must contain [min, max]")
            continue
        lower = _finite_float(
            bounds[0], f"fit.free_parameters.{name}.bounds[0]", errors
        )
        upper = _finite_float(
            bounds[1], f"fit.free_parameters.{name}.bounds[1]", errors
        )
        if lower is not None and upper is not None and lower >= upper:
            errors.append(f"fit.free_parameters.{name}.bounds must be increasing")
        initial = spec.get("initial", 0.0)
        if initial != "from_base":
            _finite_float(initial, f"fit.free_parameters.{name}.initial", errors)
    _validate_fit_priors(fit.get("priors", {}), free, errors)


def _validate_fit_priors(
    priors: Any, free_parameters: dict[str, Any], errors: list[str]
) -> None:
    if not isinstance(priors, dict):
        errors.append("fit.priors must be a mapping")
        return
    for name, spec in priors.items():
        if name not in free_parameters:
            errors.append(f"fit.priors.{name} must match a free parameter")
            continue
        if not isinstance(spec, dict):
            errors.append(f"fit.priors.{name} must be a mapping")
            continue
        prior_type = str(spec.get("type", "normal"))
        if prior_type not in SUPPORTED_PRIOR_TYPES:
            errors.append(
                f"fit.priors.{name}.type must be one of {sorted(SUPPORTED_PRIOR_TYPES)}"
            )
        if "loc" in spec and spec["loc"] != "from_base":
            _finite_float(spec["loc"], f"fit.priors.{name}.loc", errors)
        if "scale" in spec and spec["scale"] != "from_base":
            _positive_float(spec["scale"], f"fit.priors.{name}.scale", errors)
        if prior_type == "scaled_beta":
            _positive_float(spec.get("alpha", 1.0), f"fit.priors.{name}.alpha", errors)
            _positive_float(spec.get("beta", 1.0), f"fit.priors.{name}.beta", errors)


def _validate_population_config(population: Any, errors: list[str]) -> None:
    if not isinstance(population, dict):
        errors.append("fit.population must be a mapping")
        return
    _positive_float(
        population.get("prior_weight", 1.0), "fit.population.prior_weight", errors
    )
    _positive_float(
        population.get("sigma_floor", 0.03), "fit.population.sigma_floor", errors
    )
    _positive_float(
        population.get("hyper_mu_scale", 5.0), "fit.population.hyper_mu_scale", errors
    )
    relations = population.get("relations", {})
    if not isinstance(relations, dict):
        errors.append("fit.population.relations must be a mapping")
        return
    for target, spec in relations.items():
        if not isinstance(target, str) or not target:
            errors.append("fit.population.relations keys must be parameter names")
            continue
        if not isinstance(spec, dict):
            errors.append(f"fit.population.relations.{target} must be a mapping")
            continue
        _optional_string(
            spec.get("predictor"),
            f"fit.population.relations.{target}.predictor",
            errors,
        )
        for key in (
            "pivot",
            "intercept_initial",
            "slope_initial",
            "sigma_initial",
            "slope_scale",
        ):
            if key in spec and spec[key] != "median":
                label = f"fit.population.relations.{target}.{key}"
                if key in {"sigma_initial", "slope_scale"}:
                    _positive_float(spec[key], label, errors)
                else:
                    _finite_float(spec[key], label, errors)


def _validate_sample(
    sample: dict[str, Any], fit: dict[str, Any], errors: list[str]
) -> None:
    sampler = sample.get("sampler")
    if sampler not in SUPPORTED_SAMPLERS:
        errors.append(f"sample.sampler must be one of {sorted(SUPPORTED_SAMPLERS)}")
    init_strategy = sample.get("init_strategy")
    if init_strategy not in SUPPORTED_SAMPLE_INIT_STRATEGIES:
        errors.append(
            "sample.init_strategy must be one of "
            f"{sorted(SUPPORTED_SAMPLE_INIT_STRATEGIES)}"
        )
    chain_method = sample.get("chain_method")
    if chain_method not in SUPPORTED_CHAIN_METHODS:
        errors.append(
            f"sample.chain_method must be one of {sorted(SUPPORTED_CHAIN_METHODS)}"
        )
    for key in (
        "num_warmup",
        "num_samples",
        "num_chains",
        "max_tree_depth",
        "num_steps",
        "mclmc_progress_chunk_size",
        "posterior_predictive_batch_size",
    ):
        _positive_int(sample.get(key), f"sample.{key}", errors)
    if not isinstance(sample.get("mclmc_debug"), bool):
        errors.append("sample.mclmc_debug must be a boolean")
    target = _finite_float(
        sample.get("target_accept_prob"), "sample.target_accept_prob", errors
    )
    if target is not None and not 0.0 < target < 1.0:
        errors.append("sample.target_accept_prob must be between 0 and 1")
    jitter = _finite_float(
        sample.get("init_jitter_scale"), "sample.init_jitter_scale", errors
    )
    if jitter is not None and jitter < 0.0:
        errors.append("sample.init_jitter_scale must be >= 0")
    _finite_float(sample.get("seed"), "sample.seed", errors)
    free = fit.get("free_parameters", {})
    if isinstance(free, dict):
        _validate_sample_priors(sample.get("priors", {}), free, errors)
    else:
        _validate_sample_priors(sample.get("priors", {}), {}, errors)


def _validate_truth(truth: dict[str, Any], errors: list[str]) -> None:
    _optional_string(truth.get("redshift_column"), "truth.redshift_column", errors)
    specs = truth.get("parameter_columns", {})
    if not isinstance(specs, dict):
        errors.append("truth.parameter_columns must be a mapping")
        return
    for name, spec in specs.items():
        if isinstance(spec, str):
            continue
        if not isinstance(spec, dict):
            errors.append(f"truth.parameter_columns.{name} must be a string or mapping")
            continue
        _optional_string(
            spec.get("column"), f"truth.parameter_columns.{name}.column", errors
        )
        transform = spec.get("transform")
        if transform not in SUPPORTED_TRUTH_TRANSFORMS:
            errors.append(
                f"truth.parameter_columns.{name}.transform must be one of "
                f"{sorted(str(item) for item in SUPPORTED_TRUTH_TRANSFORMS)}"
            )
        _finite_float(
            spec.get("scale", 1.0), f"truth.parameter_columns.{name}.scale", errors
        )
        _finite_float(
            spec.get("offset", 0.0), f"truth.parameter_columns.{name}.offset", errors
        )
        if transform == "log_stellar_mass_h2_to_msun":
            _positive_float(spec.get("h"), f"truth.parameter_columns.{name}.h", errors)


def _validate_runtime(runtime: dict[str, Any], errors: list[str]) -> None:
    if not isinstance(runtime, dict):
        errors.append("runtime must be a mapping")
        return
    platforms = runtime.get("jax_platforms")
    if not isinstance(platforms, str) or not platforms.strip():
        errors.append("runtime.jax_platforms must be a non-empty string")
    for key in (
        "disable_jax_plugin_autoload",
        "xla_python_client_preallocate",
        "require_gpu",
    ):
        if not isinstance(runtime.get(key), bool):
            errors.append(f"runtime.{key} must be a boolean")
    _optional_string(
        runtime.get("expected_gpu_name"), "runtime.expected_gpu_name", errors
    )
    _optional_string(
        runtime.get("tf_gpu_allocator"), "runtime.tf_gpu_allocator", errors
    )
    _optional_string(
        runtime.get("jax_compilation_cache_dir"),
        "runtime.jax_compilation_cache_dir",
        errors,
    )
    cache_min = runtime.get("jax_persistent_cache_min_compile_time_secs")
    if cache_min is not None:
        _positive_float(
            cache_min, "runtime.jax_persistent_cache_min_compile_time_secs", errors
        )


def _validate_reporting(reporting: dict[str, Any], errors: list[str]) -> None:
    if not isinstance(reporting, dict):
        errors.append("reporting must be a mapping")
        return
    if reporting.get("level") not in SUPPORTED_REPORTING_LEVELS:
        errors.append(
            f"reporting.level must be one of {sorted(SUPPORTED_REPORTING_LEVELS)}"
        )
    _nonnegative_int(
        reporting.get("save_sed_samples"),
        "reporting.save_sed_samples",
        errors,
    )
    for key in ("plot_filters", "plot_ground_truth"):
        if not isinstance(reporting.get(key), bool):
            errors.append(f"reporting.{key} must be a boolean")


def _validate_output(output: dict[str, Any], errors: list[str]) -> None:
    if not isinstance(output, dict):
        errors.append("output must be a mapping")
        return
    if output.get("format") not in SUPPORTED_OUTPUT_FORMATS:
        errors.append(
            f"output.format must be one of {sorted(SUPPORTED_OUTPUT_FORMATS)}"
        )
    if not isinstance(output.get("verbose_benchmark"), bool):
        errors.append("output.verbose_benchmark must be a boolean")


def _validate_band_calibration(calibration: dict[str, Any], errors: list[str]) -> None:
    if not isinstance(calibration, dict):
        errors.append("band_calibration must be a mapping")
        return
    mode = str(calibration.get("mode", "none"))
    if mode not in SUPPORTED_BAND_CALIBRATION_MODES:
        errors.append(
            "band_calibration.mode must be one of "
            f"{sorted(SUPPORTED_BAND_CALIBRATION_MODES)}"
        )
    for key in ("offsets_mag", "flux_multipliers"):
        values = calibration.get(key, {})
        if values is None:
            continue
        if not isinstance(values, dict):
            errors.append(f"band_calibration.{key} must be a mapping")
            continue
        for band_name, value in values.items():
            if not isinstance(band_name, str) or not band_name:
                errors.append(f"band_calibration.{key} keys must be band names")
            _finite_float(value, f"band_calibration.{key}.{band_name}", errors)
            if key == "flux_multipliers":
                _positive_float(value, f"band_calibration.{key}.{band_name}", errors)


def _validate_calibration(calibration: dict[str, Any], errors: list[str]) -> None:
    if not isinstance(calibration, dict):
        errors.append("calibration must be a mapping")
        return
    global_scale = calibration.get("global_sed_scale", {})
    if not isinstance(global_scale, dict):
        errors.append("calibration.global_sed_scale must be a mapping")
        return
    for key in ("enabled", "trainable"):
        if not isinstance(global_scale.get(key, False), bool):
            errors.append(f"calibration.global_sed_scale.{key} must be a boolean")
    mode = str(global_scale.get("mode", "disabled"))
    allowed_modes = {"disabled", "fixed", "fit_global", "learn_global"}
    if mode not in allowed_modes:
        errors.append(
            "calibration.global_sed_scale.mode must be one of "
            f"{sorted(allowed_modes)}"
        )
    if str(global_scale.get("parameterization", "log_alpha")) != "log_alpha":
        errors.append(
            "calibration.global_sed_scale.parameterization must be 'log_alpha'"
        )
    _finite_float(
        global_scale.get("initial_log_alpha", 0.0),
        "calibration.global_sed_scale.initial_log_alpha",
        errors,
    )
    _positive_float(
        global_scale.get("prior_sigma_log_alpha", 0.10),
        "calibration.global_sed_scale.prior_sigma_log_alpha",
        errors,
    )
    per_band = calibration.get("per_band_zero_points", {})
    if not isinstance(per_band, dict):
        errors.append("calibration.per_band_zero_points must be a mapping")
        return
    if not isinstance(per_band.get("enabled", False), bool):
        errors.append("calibration.per_band_zero_points.enabled must be a boolean")


def _validate_sample_priors(
    priors: Any, free_parameters: dict[str, Any], errors: list[str]
) -> None:
    if not isinstance(priors, dict):
        errors.append("sample.priors must be a mapping")
        return
    for name, spec in priors.items():
        if name not in free_parameters:
            errors.append(f"sample.priors.{name} must match a free parameter")
            continue
        if not isinstance(spec, dict):
            errors.append(f"sample.priors.{name} must be a mapping")
            continue
        prior_type = str(spec.get("type", "truncated_normal"))
        if prior_type not in SUPPORTED_PRIOR_TYPES:
            errors.append(
                f"sample.priors.{name}.type must be one of "
                f"{sorted(SUPPORTED_PRIOR_TYPES)}"
            )
        if "loc" in spec and spec["loc"] != "from_base":
            _finite_float(spec["loc"], f"sample.priors.{name}.loc", errors)
        if "scale" in spec and spec["scale"] != "from_base":
            _positive_float(spec["scale"], f"sample.priors.{name}.scale", errors)
        if prior_type == "scaled_beta":
            _positive_float(
                spec.get("alpha", 1.0), f"sample.priors.{name}.alpha", errors
            )
            _positive_float(spec.get("beta", 1.0), f"sample.priors.{name}.beta", errors)


def _configured_catalog_columns(config: dict[str, Any]) -> set[str]:
    from .io import required_catalog_columns

    return set(required_catalog_columns(config))


def _optional_string(value: Any, label: str, errors: list[str]) -> None:
    if value is not None and not isinstance(value, str):
        errors.append(f"{label} must be a string or null")


def _require_model_path(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty path string")


def _finite_float(value: Any, label: str, errors: list[str]) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be numeric")
        return None
    if result != result or result in {float("inf"), float("-inf")}:
        errors.append(f"{label} must be finite")
        return None
    return result


def _positive_float(value: Any, label: str, errors: list[str]) -> float | None:
    result = _finite_float(value, label, errors)
    if result is not None and result <= 0.0:
        errors.append(f"{label} must be > 0")
    return result


def _nonnegative_float(value: Any, label: str, errors: list[str]) -> float | None:
    result = _finite_float(value, label, errors)
    if result is not None and result < 0.0:
        errors.append(f"{label} must be >= 0")
    return result


def _positive_int(value: Any, label: str, errors: list[str]) -> int | None:
    if not isinstance(value, int) or value <= 0:
        errors.append(f"{label} must be an integer > 0")
        return None
    return value


def _nonnegative_int(value: Any, label: str, errors: list[str]) -> int | None:
    if not isinstance(value, int) or value < 0:
        errors.append(f"{label} must be an integer >= 0")
        return None
    return value
