from __future__ import annotations

import json

import numpy as np
import pandas as pd

from euclid_dsps.amortized.diagnostics import (
    _read_posterior_samples_for_corner,
    _truth_parameter_frame,
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
    (tmp_path / "corner_truth_prior_posterior_map.png").write_bytes(b"stale")

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
    assert (tmp_path / "corner_full_latent_truth_prior_posterior.png").exists()
    assert not (tmp_path / "posterior_corner.png").exists()
    assert not (tmp_path / "learned_prior_corner.png").exists()
    assert not (tmp_path / "posterior_vs_learned_prior_corner.png").exists()
    assert not (tmp_path / "corner_truth_prior_posterior_map.png").exists()
    assert (tmp_path / "redshift_distribution_comparison.png").exists()


def test_corner_population_reads_sharded_posterior_samples(tmp_path) -> None:
    shard_dir = tmp_path / "posterior_samples"
    shard_dir.mkdir()
    pd.DataFrame({"object_id": [1, 2], "z_obs": [0.1, 0.2]}).to_parquet(
        shard_dir / "batch_000001.parquet"
    )
    pd.DataFrame({"object_id": [3, 4], "z_obs": [0.3, 0.4]}).to_parquet(
        shard_dir / "batch_000002.parquet"
    )

    frame = _read_posterior_samples_for_corner(tmp_path, max_rows=4)

    assert len(frame) == 4
    assert set(frame["object_id"]) == {1, 2, 3, 4}


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
    assert (tmp_path / "catalog_proxy_stellar_mass_comparison.png").exists()
    assert (tmp_path / "catalog_proxy_sfr_distribution.png").exists()
    assert (tmp_path / "catalog_proxy_mass_sfr_plane.png").exists()


def test_truth_frame_prefers_projected_truth_sibling(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "redshift_true": [0.1, 0.2],
            "logsm_true": [9.5, 10.5],
        }
    )
    raw_path = tmp_path / "catalog.parquet"
    raw.to_parquet(raw_path)
    projected = raw.assign(
        z_obs=[0.1, 0.2],
        log10_stellar_mass=[9.5, 10.5],
        dlog10_sfr_1=[0.3, -0.2],
        tau2=[0.4, 0.8],
    )
    projected.to_parquet(tmp_path / "catalog_projected_truth.parquet")
    summary = pd.DataFrame({"row_index": [0, 1], "object_id": [0, 1]})
    config = {
        "catalog_path": str(raw_path),
        "fit": {
            "free_parameters": {
                "z_obs": {},
                "log10_stellar_mass": {},
                "dlog10_sfr_1": {},
                "tau2": {},
            }
        },
        "truth": {"parameter_columns": {}},
    }

    truth = _truth_parameter_frame(summary, tmp_path, config=config)

    assert truth["dlog10_sfr_1"].tolist() == [0.3, -0.2]
    assert truth["tau2"].tolist() == [0.4, 0.8]
