from __future__ import annotations

import pandas as pd
import pytest

from euclid_dsps.config import normalize_config
from euclid_dsps.reporting.core import (
    fit_objective_components,
    fit_parameter_audit,
    parameter_truth_metrics,
    redshift_attractor_summary,
    summarize_by_row,
)
from euclid_dsps.semantics import (
    active_parameters,
    inactive_parameters,
    is_comparable_fit_parameter,
)


def _config() -> dict:
    return normalize_config(
        {
            "catalog_path": "catalog.parquet",
            "ssp_path": "ssp.h5",
            "bands": [
                {
                    "name": "vis",
                    "column": "vis",
                    "units": "fnu_cgs",
                    "sigma_mag": 0.05,
                    "filter": {"kind": "tophat"},
                }
            ],
            "model": {
                "fixed_parameters": {
                    "z_obs": 0.5,
                    "log10_sfr": 0.0,
                    "sfh_t_peak": 4.0,
                    "sfh_tau": 0.6,
                    "log10_metallicity": -2.25,
                    "metallicity_scatter": 0.2,
                    "dust_av": 0.2,
                    "dust_slope": -0.7,
                },
                "parameter_columns": {
                    "z_obs_phz_min_70": "phz_min_70",
                    "z_obs_phz_max_70": "phz_max_70",
                    "z_obs_phz_min_90": "phz_min_90",
                    "z_obs_phz_max_90": "phz_max_90",
                    "z_obs_phz_min_95": "phz_min_95",
                    "z_obs_phz_max_95": "phz_max_95",
                },
            },
            "fit": {
                "free_parameters": {
                    "z_obs": {"initial": "from_base", "bounds": [0.0, 6.0]},
                    "log10_metallicity": {
                        "initial": -2.25,
                        "bounds": [-3.9, -1.6],
                    },
                },
                "priors": {
                    "z_obs": {"type": "uniform"},
                    "log10_metallicity": {
                        "type": "normal",
                        "loc": -2.25,
                        "scale": 1.0,
                    },
                },
            },
        }
    )


def _cosmos_dust_config() -> dict:
    config = _config()
    config["dust_model"] = "cosmos_proxy_fixed"
    return normalize_config(config)


def test_fit_parameter_audit_flags_constant_free_parameter() -> None:
    fits = pd.DataFrame(
        {
            "row_index": [0, 1],
            "fit_z_obs": [0.5, 0.6],
            "fit_log10_metallicity": [-2.25, -2.25],
            "fit_dust_av": [0.2, 0.2],
            "fit_t_obs_gyr": [8.0, 7.0],
        }
    )

    audit = fit_parameter_audit(fits, _config()).set_index("parameter")

    assert "constant_free_parameter" in audit.loc["log10_metallicity", "warning_flags"]
    assert audit.loc["dust_av", "source"] == "fixed"
    assert audit.loc["dust_av", "active_in_forward_model"]
    assert "not_inferred_column" in audit.loc["dust_av", "warning_flags"]
    assert audit.loc["t_obs_gyr", "source"] == "derived"


def test_cosmos_proxy_dust_is_inactive_not_inferred() -> None:
    config = _cosmos_dust_config()
    fits = pd.DataFrame(
        {
            "row_index": [0],
            "fit_dust_av": [0.2],
            "truth_dust_av": [0.4],
            "truth_kind_dust_av": ["proxy"],
            "fit_log10_metallicity": [-2.1],
            "truth_log10_metallicity": [-2.2],
            "truth_kind_log10_metallicity": ["proxy"],
        }
    )

    audit = fit_parameter_audit(fits, config).set_index("parameter")
    metrics = parameter_truth_metrics(fits, config=config)

    assert "dust_av" in inactive_parameters(config)
    assert "dust_av" not in active_parameters(config)
    assert audit.loc["dust_av", "source"] == "inactive_fixed"
    assert not audit.loc["dust_av", "active_in_forward_model"]
    assert "dust_av" not in metrics["parameter"].tolist()
    assert "log10_metallicity" in metrics["parameter"].tolist()


def test_dust_av_is_never_truth_comparable_even_when_free() -> None:
    config = _config()
    config["fit"]["free_parameters"]["dust_av"] = {
        "initial": 0.2,
        "bounds": [0.0, 2.5],
    }

    assert not is_comparable_fit_parameter(config, "dust_av")


def test_fit_objective_components_writes_photometric_and_prior_terms() -> None:
    fits = pd.DataFrame(
            {
                "row_index": [0],
                "fit_z_obs": [1.2],
                "fit_log10_metallicity": [-2.0],
            }
        )
    comparison = pd.DataFrame({"row_index": [0, 0], "chi": [2.0, -1.0]})

    components = fit_objective_components(fits, comparison, _config())

    assert components.loc[0, "photometric_chi2"] == 5.0
    assert components.loc[0, "photometric_objective"] == 5.0
    assert components.loc[0, "physical_gaussian_prior_penalty"] > 0.0
    assert components.loc[0, "approx_objective"] > 0.0


def test_fit_objective_components_uses_saved_student_t_objective() -> None:
    config = _config()
    config["fit"]["photometric_likelihood"] = "student_t"
    fits = pd.DataFrame(
        {
            "row_index": [0],
            "fit_z_obs": [1.2],
            "fit_log10_metallicity": [-2.0],
            "photometric_likelihood": ["student_t"],
        }
    )
    comparison = pd.DataFrame(
        {
            "row_index": [0, 0],
            "chi_likelihood": [10.0, 2.0],
            "photometric_objective_contribution": [11.79579055, 3.29583687],
        }
    )

    components = fit_objective_components(fits, comparison, config)

    assert components.loc[0, "photometric_chi2"] == pytest.approx(104.0)
    assert components.loc[0, "photometric_objective"] == pytest.approx(15.09162742)
    assert components.loc[0, "fit_quality_metric"] == "student_t_neg2loglike"


def test_summarize_by_row_uses_flux_chi_and_dof_reduced_chi2() -> None:
    comparison = pd.DataFrame(
        {
            "row_index": [0, 0, 0],
            "band": ["u", "g", "r"],
            "chi": [100.0, 100.0, 100.0],
            "chi_flux": [1.0, 2.0, 3.0],
            "n_valid_bands": [3, 3, 3],
            "n_free_effective": [1, 1, 1],
            "dof": [2, 2, 2],
            "residual_mag_model_minus_observed": [0.0, 0.0, 0.0],
            "flux_ratio_model_over_observed": [1.0, 1.0, 1.0],
        }
    )

    summary = summarize_by_row(comparison)

    assert summary.loc[0, "chi2"] == 14.0
    assert summary.loc[0, "chi2_per_band"] == 14.0 / 3.0
    assert summary.loc[0, "reduced_chi2_dof"] == 7.0
    assert summary.loc[0, "reduced_chi2"] == 7.0
    assert summary.loc[0, "fit_quality"] == 14.0
    assert summary.loc[0, "reduced_fit_quality"] == 7.0


def test_summarize_by_row_uses_student_t_objective_for_fit_quality() -> None:
    comparison = pd.DataFrame(
        {
            "row_index": [0, 0],
            "band": ["u", "g"],
            "chi_likelihood": [10.0, 2.0],
            "photometric_objective_contribution": [11.79579055, 3.29583687],
            "photometric_likelihood": ["student_t", "student_t"],
            "n_valid_bands": [2, 2],
            "n_free_effective": [1, 1],
            "dof": [1, 1],
            "residual_mag_model_minus_observed": [0.0, 0.0],
            "flux_ratio_model_over_observed": [1.0, 1.0],
        }
    )

    summary = summarize_by_row(comparison)

    assert summary.loc[0, "chi2"] == pytest.approx(104.0)
    assert summary.loc[0, "fit_quality"] == pytest.approx(15.09162742)
    assert summary.loc[0, "reduced_fit_quality"] == pytest.approx(15.09162742)
    assert summary.loc[0, "fit_quality_metric"] == "student_t_neg2loglike"


def test_redshift_attractor_summary_counts_repeated_fit_modes() -> None:
    by_row = pd.DataFrame(
        {
            "z_obs": [0.401, 0.402, 0.398, 0.85, 0.851],
            "redshift_truth": [0.1, 0.2, 0.3, 0.8, 0.9],
            "delta_z_obs_minus_truth": [0.301, 0.202, 0.098, 0.05, -0.049],
            "reduced_chi2": [1.0, 2.0, 1.5, 0.8, 0.9],
        }
    )

    summary = redshift_attractor_summary(by_row, min_count=2)

    assert summary.loc[0, "z_fit_bin"] == 0.4
    assert summary.loc[0, "n_galaxies"] == 3
    assert summary.loc[1, "z_fit_bin"] == pytest.approx(0.85)
