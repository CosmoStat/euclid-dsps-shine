from __future__ import annotations

import json

import numpy as np
import pandas as pd

from euclid_dsps.amortized.diagnostics import (
    feature_diagnostics_frame,
    posterior_predictive_residual_frame,
    summarize_inference_outputs,
)


def test_posterior_predictive_residual_diagnostics(tmp_path) -> None:
    object_id = np.asarray([10, 11])
    obs = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    err = np.ones_like(obs) * 0.5
    mask = np.asarray([[True, True], [True, False]])
    model = np.asarray(
        [
            [[1.5, 2.0], [2.0, 5.0]],
            [[0.5, 3.0], [4.0, 6.0]],
        ]
    )
    bands = ("b0", "b1")

    residuals = posterior_predictive_residual_frame(
        object_id,
        obs,
        err,
        mask,
        model,
        bands,
    )
    assert residuals.loc[0, "chi_likelihood"] == -1.0
    assert residuals.loc[0, "residual_sigma"] == -1.0
    features = feature_diagnostics_frame(
        object_id,
        np.asarray([[1.0, -2.0, 0.1], [3.0, 4.0, -0.2]]),
        n_flux_bands=2,
    )
    summary = pd.DataFrame(
        {
            "object_id": object_id,
            "n_valid_bands": [2, 1],
            "photometric_loglike_mean": [1.0, -2.0],
            "posterior_predictive_chi2_median": [4.0, 9.0],
            "z_obs_q16": [0.1, 0.2],
            "z_obs_median": [0.2, 0.3],
            "z_obs_q84": [0.3, 0.4],
            "log10_stellar_mass_median": [9.0, 10.0],
        }
    )
    residuals.to_parquet(tmp_path / "posterior_predictive_residuals.parquet")
    features.to_parquet(tmp_path / "feature_diagnostics.parquet")
    pd.DataFrame(
        {
            "object_id": np.repeat(object_id, 2),
            "sample_id": [0, 1, 0, 1],
            "z_obs": [0.1, 0.2, 0.3, 0.4],
            "log10_stellar_mass": [9.0, 9.1, 10.0, 10.1],
            "tau2": [0.1, 0.2, 0.3, 0.4],
            "dust_index_n": [-0.5, -0.4, -0.3, -0.2],
            "log10_stellar_metallicity": [-1.0, -0.9, -0.8, -0.7],
            "ln_fagn": [-8.0, -7.0, -6.0, -5.0],
        }
    ).to_parquet(tmp_path / "posterior_samples.parquet")
    pd.DataFrame(
        {
            "sample_id": range(4),
            "logprior": [-16.0, -15.0, -14.0, -13.0],
            "z_obs": [0.15, 0.25, 0.35, 0.45],
            "log10_stellar_mass": [8.8, 9.0, 9.2, 9.4],
            "tau2": [0.1, 0.2, 0.3, 0.4],
            "dust_index_n": [-0.6, -0.5, -0.4, -0.3],
            "log10_stellar_metallicity": [-1.1, -1.0, -0.9, -0.8],
            "ln_fagn": [-9.0, -8.0, -7.0, -6.0],
        }
    ).to_parquet(tmp_path / "learned_prior_samples.parquet")
    summary.to_parquet(tmp_path / "posterior_summary.parquet")

    summarize_inference_outputs(tmp_path / "posterior_summary.parquet", tmp_path)

    payload = json.loads((tmp_path / "posterior_diagnostics_summary.json").read_text())
    residual_summary = pd.read_parquet(
        tmp_path / "posterior_predictive_residual_summary.parquet"
    )
    top = pd.read_parquet(tmp_path / "top_posterior_predictive_chi2.parquet")

    assert payload["n_objects"] == 2
    assert payload["residual_summary_rows"] == 4
    assert payload["learned_prior_rows"] == 4
    assert len(residual_summary) == 4
    assert top.iloc[0]["object_id"] == 11
    assert "worst_band" in top.columns
    assert (tmp_path / "learned_prior_summary.json").exists()
    assert (tmp_path / "learned_prior_logprob_hist.png").exists()
    assert (tmp_path / "posterior_corner.png").exists()
    assert (tmp_path / "learned_prior_corner.png").exists()
    assert (tmp_path / "posterior_vs_learned_prior_corner.png").exists()
    assert (tmp_path / "redshift_distribution_comparison.png").exists()


def test_catalog_proxy_diagnostics(tmp_path) -> None:
    catalog = pd.DataFrame(
        {
            "log_stellar_mass": [10.0, 11.0],
            "log_sfr_true": [0.1, 0.7],
        }
    )
    catalog_path = tmp_path / "catalog.parquet"
    catalog.to_parquet(catalog_path)
    summary = pd.DataFrame(
        {
            "object_id": [0, 1],
            "n_valid_bands": [10, 10],
            "posterior_predictive_chi2_median": [5.0, 30.0],
            "log10_stellar_mass_median": [9.7, 10.4],
        }
    )
    summary.to_parquet(tmp_path / "posterior_summary.parquet")

    config = {
        "catalog_path": str(catalog_path),
        "truth": {
            "parameter_columns": {
                "log10_formed_mass_msun": {
                    "column": "log_stellar_mass",
                    "transform": "log_stellar_mass_h2_to_msun",
                    "h": 0.73,
                },
                "log10_sfr_at_obs": "log_sfr_true",
            }
        },
    }

    summarize_inference_outputs(
        tmp_path / "posterior_summary.parquet",
        tmp_path,
        config=config,
        limit=2,
    )

    payload = json.loads((tmp_path / "posterior_diagnostics_summary.json").read_text())
    proxies = pd.read_parquet(tmp_path / "catalog_proxy_comparison.parquet")
    assert payload["catalog_proxy_comparison_rows"] == 2
    assert "catalog_log10_stellar_mass_proxy" in proxies
    assert "catalog_log10_sfr_at_obs_proxy" in proxies
    assert (
        tmp_path / "catalog_proxy_stellar_mass_comparison.png"
    ).exists()
    assert (tmp_path / "catalog_proxy_sfr_distribution.png").exists()
    assert (tmp_path / "catalog_proxy_mass_sfr_plane.png").exists()
