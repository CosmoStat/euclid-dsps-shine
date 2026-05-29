from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_against_fsps_prospector as benchmark  # noqa: E402


def test_fsps_prospector_benchmark_help_works_without_reference_packages() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_against_fsps_prospector.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "FSPS/Prospector" in result.stdout
    assert "--stellar-ssp" in result.stdout
    assert "--agn-template" in result.stdout
    assert "--agn-component-grid" in result.stdout
    assert "--agn-host-attenuation" in result.stdout
    assert "--agn-host-attenuation-scale" in result.stdout
    assert "--agn-igm-order" in result.stdout
    assert "--agn-baked-attenuation" in result.stdout
    assert "fsps_diffuse_unit_tau" in result.stdout
    assert "fsps_powerlaw_unit_tau" in result.stdout
    assert "fsps_after_igm" in result.stdout
    assert "--runtime" in result.stdout
    assert "--levels" in result.stdout
    assert "agn_component_only" in result.stdout
    assert "stellar_plus_dust_plus_agn" in result.stdout
    assert "stellar_plus_gas_plus_agn" in result.stdout


def test_fsps_prospector_benchmark_help_ignores_forced_cuda_env() -> None:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cuda"
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_against_fsps_prospector.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert "--stellar-ssp" in result.stdout


def test_benchmark_summary_reports_finite_counts_and_level_band_correlations() -> None:
    frame = pd.DataFrame(
        {
            "level": ["stellar_only"] * 5,
            "band": ["lsst_u"] * 5,
            "dsps_mag": [20.0, 21.0, np.inf, 22.0, 90.0],
            "reference_mag": [20.1, 21.1, 22.1, np.nan, 100.0],
            "delta_mag": [-0.1, -0.1, np.inf, np.nan, -10.0],
            "z_obs": [0.1, 0.2, 0.3, 0.4, 0.5],
            "tau2": [0.0, 0.1, 0.2, 0.3, 0.4],
            "dust_index_n": [-0.7] * 5,
            "log10_stellar_metallicity": [0.0] * 5,
            "log10_gas_metallicity": [0.0] * 5,
            "log10_gas_ionization": [-2.0] * 5,
            "log10_recent_sfr_msun_per_yr": [0.0, 0.1, 0.2, 0.3, 0.4],
        }
    )

    summary = benchmark.summarize_benchmark(frame)
    band = summary["levels"]["stellar_only"]["lsst_u"]

    assert band["n_total"] == 5
    assert band["n_finite_both"] == 3
    assert band["n_nonfinite_dsps"] == 1
    assert band["n_nonfinite_reference"] == 1
    assert band["n_nonfinite_delta"] == 2
    assert band["median_abs_delta_mag"] == 0.1
    assert band["n_effectively_faint_dsps"] == 1
    assert band["n_effectively_faint_reference"] == 1
    assert band["n_bright_finite_both"] == 2
    assert band["bright_p95_abs_delta_mag"] == 0.1
    assert "log10_recent_sfr_msun_per_yr" in summary["residual_correlations"]
    assert (
        "log10_recent_sfr_msun_per_yr"
        in summary["residual_correlations_by_level_band"]["stellar_only"]["lsst_u"]
    )


def test_benchmark_context_selection_uses_stellar_context_for_gas_free_levels() -> None:
    gas_context = object()
    stellar_context = object()

    assert benchmark._context_for_level("stellar_only", gas_context, stellar_context) is stellar_context
    assert benchmark._context_for_level("stellar_plus_dust", gas_context, stellar_context) is stellar_context
    assert benchmark._context_for_level("stellar_plus_agn", gas_context, stellar_context) is stellar_context
    assert (
        benchmark._context_for_level(
            "stellar_plus_dust_plus_agn", gas_context, stellar_context
        )
        is stellar_context
    )
    assert benchmark._context_for_level("agn_component_only", gas_context, stellar_context) is stellar_context
    assert benchmark._context_for_level("stellar_plus_gas", gas_context, stellar_context) is gas_context
    assert (
        benchmark._context_for_level(
            "stellar_plus_gas_plus_agn", gas_context, stellar_context
        )
        is gas_context
    )
    assert benchmark._context_for_level("full_noagn", gas_context, stellar_context) is gas_context
    assert benchmark._context_for_level("full_agn", gas_context, stellar_context) is gas_context


def test_benchmark_context_need_helpers_keep_component_only_gas_free() -> None:
    assert benchmark._needs_stellar_context(("agn_component_only",))
    assert not benchmark._needs_gas_context(("agn_component_only",))
    assert benchmark._needs_gas_context(("full_agn",))
    assert benchmark._needs_stellar_context(("stellar_plus_dust_plus_agn",))
    assert benchmark._needs_gas_context(("stellar_plus_gas_plus_agn",))


def test_benchmark_runtime_override_cpu_sets_platform() -> None:
    config = {"runtime": {"jax_platforms": "auto", "require_gpu": False}}

    out = benchmark._apply_runtime_override(config, "cpu")

    assert config["runtime"]["jax_platforms"] == "auto"
    assert out["runtime"]["jax_platforms"] == "cpu"
    assert out["runtime"]["disable_jax_plugin_autoload"] is True
    assert out["runtime"]["require_gpu"] is False


def test_benchmark_component_grid_axis_interpolation_is_strict() -> None:
    pairs = benchmark._axis_interp_pairs(
        np.asarray([0.0, 2.0]),
        1.0,
        "test_axis",
        "grid.h5",
    )

    assert pairs == ((0, 0.5), (1, 0.5))

    with pytest.raises(RuntimeError, match="does not cover sampled value"):
        benchmark._axis_interp_pairs(
            np.asarray([0.0, 2.0]),
            3.0,
            "test_axis",
            "grid.h5",
        )


def test_benchmark_model_config_without_agn_removes_component_path() -> None:
    model = {
        "agn_model": "fsps_component_grid",
        "agn_component_grid_path": "component.h5",
        "agn_template_path": "template.h5",
    }

    out = benchmark._model_config_without_agn(model)

    assert out["agn_model"] == "none"
    assert "agn_component_grid_path" not in out
    assert "agn_template_path" not in out


def test_benchmark_levels_add_agn_audit_for_agn_config() -> None:
    assert benchmark._benchmark_levels({"agn_model": "none"}, None) == benchmark.NOAGN_BENCHMARK_LEVELS

    levels = benchmark._benchmark_levels(
        {"agn_model": "template_grid", "agn_template_path": "agn.h5"}, None
    )

    assert levels == benchmark.AGN_AUDIT_LEVELS
    assert "stellar_plus_agn" in levels
    assert "stellar_plus_dust_plus_agn" in levels
    assert "stellar_plus_gas_plus_agn" in levels
    assert "full_agn" in levels

    requested = benchmark._benchmark_levels(
        {"agn_model": "template_grid", "agn_template_path": "agn.h5"},
        ["agn_component_only"],
    )
    assert requested == ("agn_component_only",)

    component_levels = benchmark._benchmark_levels(
        {"agn_model": "fsps_component_grid", "agn_component_grid_path": "agn_component.h5"},
        None,
    )
    assert component_levels == benchmark.AGN_AUDIT_LEVELS


def test_parameters_for_agn_levels_keep_agn_model() -> None:
    params = {
        "tau2": 0.5,
        "tau1_over_tau2": 2.0,
        "ln_fagn": -4.0,
        "ln_tauagn": 2.3,
    }
    model = {
        "agn_model": "template_grid",
        "agn_template_path": "agn.h5",
        "agn_host_attenuation": "fsps_diffuse_unit_tau",
    }

    stellar_params, stellar_model = benchmark._parameters_for_level(
        params, model, "stellar_plus_agn"
    )
    full_params, full_model = benchmark._parameters_for_level(params, model, "full_agn")

    assert stellar_model["agn_model"] == "template_grid"
    assert stellar_model["nebular_model"] == "fixed_ssp"
    assert stellar_model["agn_host_attenuation"] == "none"
    assert stellar_params["tau2"] == 0.0
    assert stellar_params["tau1_over_tau2"] == 0.0
    assert full_model["agn_model"] == "template_grid"
    assert full_model["agn_host_attenuation"] == "fsps_diffuse_unit_tau"
    assert full_params["tau2"] == 0.5

    dust_params, dust_model = benchmark._parameters_for_level(
        params, model, "stellar_plus_dust_plus_agn"
    )
    gas_params, gas_model = benchmark._parameters_for_level(
        params, model, "stellar_plus_gas_plus_agn"
    )
    assert dust_model["agn_model"] == "template_grid"
    assert dust_model["nebular_model"] == "fixed_ssp"
    assert dust_model["agn_host_attenuation"] == "fsps_diffuse_unit_tau"
    assert dust_params["tau2"] == 0.5
    assert gas_model["agn_model"] == "template_grid"
    assert gas_model["agn_host_attenuation"] == "none"
    assert gas_params["tau2"] == 0.0
    assert gas_params["tau1_over_tau2"] == 0.0

    component_params, component_model = benchmark._parameters_for_level(
        params, model, "agn_component_only"
    )
    assert component_model["agn_model"] == "template_grid"
    assert component_model["nebular_model"] == "fixed_ssp"
    assert component_model["igm_model"] == "none"
    assert component_model["agn_host_attenuation"] == "none"
    assert component_params["tau2"] == 0.0


def test_agn_component_row_uses_ratio_equivalent_delta_mag() -> None:
    row = benchmark._agn_component_row(
        0,
        "agn_component_only",
        1500.0,
        2.0,
        1.0,
        {"log10_recent_sfr_msun_per_yr": 0.0},
        {"ln_fagn": -4.0, "ln_tauagn": 2.3},
    )

    assert row["observable_type"] == "agn_component_sed"
    assert row["band"] == "rest_1500.0A"
    np.testing.assert_allclose(row["delta_mag"], -2.5 * np.log10(2.0))
    np.testing.assert_allclose(row["delta_flux_over_flux"], 1.0)


def test_prospector_params_maps_agn_parameters() -> None:
    params = {
        "z_obs": 0.5,
        "log10_stellar_mass": 10.0,
        "log10_stellar_metallicity": 0.0,
        "tau2": 0.0,
        "tau1_over_tau2": 0.0,
        "log10_gas_metallicity": 0.0,
        "log10_gas_ionization": -2.0,
        "ln_fagn": np.log(0.01),
        "ln_tauagn": np.log(20.0),
        **{f"dlog10_sfr_{index}": 0.0 for index in range(1, 7)},
    }

    out = benchmark._prospector_params(
        {"agn_model": "template_grid", "nebular_model": "fixed_ssp"}, params
    )

    assert out["add_dust_emission"] is True
    np.testing.assert_allclose(out["fagn"], 0.01)
    np.testing.assert_allclose(out["agn_tau"], 20.0)
