from __future__ import annotations

import numpy as np
import pandas as pd

from euclid_dsps.amortized.collapse_gates import write_inference_collapse_gate
from euclid_dsps.amortized.truth_diagnostics import write_extended_truth_diagnostics


def test_extended_truth_diagnostics_include_sfr_and_ssfr(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    truth = pd.DataFrame(
        {
            "row_index": [0, 1],
            "object_id": [10, 11],
            "redshift_true": [0.1, 0.2],
            "logsm_true": [10.0, 10.5],
            "logsfr_true": [0.0, 0.3],
            "logssfr_true": [-10.0, -10.2],
            "dust_av": [0.2, 0.4],
            "dust_delta": [-0.7, -0.4],
        }
    )
    posterior = pd.DataFrame(
        {
            "row_index": [0, 1],
            "object_id": [10, 11],
            "z_obs_median": [0.11, 0.19],
            "log10_stellar_mass_median": [10.1, 10.4],
            "log10_stellar_mass_alpha_corrected": [10.1, 10.4],
            "log10_sfr_at_obs_median": [0.1, 0.2],
            "log10_sfr_at_obs_alpha_corrected": [0.1, 0.2],
            "log10_ssfr_at_obs_median": [-10.0, -10.2],
            "tau2_median": [0.18, 0.35],
            "dust_index_n_median": [-0.6, -0.5],
        }
    )
    truth.to_parquet(run / "inference_truth.parquet", index=False)
    posterior.to_parquet(run / "posterior_summary.parquet", index=False)

    outputs = write_extended_truth_diagnostics(run)
    assert "posterior_vs_truth_extended.csv" in outputs

    metrics = pd.read_csv(run / "posterior_vs_truth_extended.csv")
    assert "log10_sfr_at_obs_alpha_corrected" in set(metrics["parameter"])
    assert "log10_ssfr_at_obs" in set(metrics["parameter"])


def test_prior_truth_diagnostics_cover_all_15_dimensions_and_correlations(
    tmp_path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    rng = np.random.default_rng(4)
    names = (
        "z_obs",
        "log10_stellar_mass",
        "log10_stellar_metallicity",
        "dust_av",
        "dust_delta",
        *(f"sfh_dlog_sfr_{index:02d}" for index in range(1, 11)),
    )
    values = rng.normal(size=(128, len(names)))
    truth = pd.DataFrame(values, columns=names)
    prior = truth.sample(frac=1.0, random_state=3).reset_index(drop=True)
    truth.to_parquet(run / "inference_truth.parquet", index=False)
    prior.to_parquet(run / "learned_prior_samples.parquet", index=False)

    outputs = write_extended_truth_diagnostics(run)
    assert "prior_vs_truth_population.csv" in outputs
    assert "prior_vs_truth_correlations.csv" in outputs
    assert "prior_vs_truth_correlation_error.png" in outputs
    marginals = pd.read_csv(run / "prior_vs_truth_population.csv")
    correlations = pd.read_csv(run / "prior_vs_truth_correlations.csv")
    assert marginals.parameter.nunique() == 15
    assert len(correlations) == 105
    assert np.allclose(correlations.abs_delta, 0.0)

    gate = write_inference_collapse_gate(run)
    checks = {item["name"]: item for item in gate["checks"]}
    assert checks["prior_native_marginals_complete"]["status"] == "PASS"
    assert checks["prior_15d_spearman_mean_abs_delta"]["status"] == "PASS"
