from __future__ import annotations

import pandas as pd

from euclid_dsps.config import normalize_config
from euclid_dsps.reporting.core import fit_objective_components, fit_parameter_audit


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
    assert "not_inferred_column" in audit.loc["dust_av", "warning_flags"]
    assert audit.loc["t_obs_gyr", "source"] == "derived"


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
    assert components.loc[0, "physical_gaussian_prior_penalty"] > 0.0
    assert components.loc[0, "approx_objective"] > 0.0
