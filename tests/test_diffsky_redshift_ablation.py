from __future__ import annotations

import json

import numpy as np
import pandas as pd

from euclid_dsps.diffsky_redshift_ablation import (
    redshift_metrics_from_samples,
    summarize_redshift_metrics,
    write_redshift_metrics_for_run,
)


def test_redshift_metrics_on_toy_posterior_are_correct() -> None:
    samples = pd.DataFrame(
        {
            "object_id": [1, 1, 1, 2, 2, 2],
            "z_obs": [0.9, 1.0, 1.1, 0.4, 0.5, 0.6],
        }
    )
    truth = pd.DataFrame(
        {
            "object_id": [1, 2],
            "redshift_true": [1.0, 0.5],
        }
    )

    object_metrics, summary = redshift_metrics_from_samples(samples, truth)

    assert np.allclose(object_metrics["delta_z"], [0.0, 0.0])
    assert summary["median_bias"] == 0.0
    assert summary["rmse"] == 0.0
    assert summary["outlier_fraction_0p15"] == 0.0
    assert summary["coverage_68"] == 1.0
    assert summary["coverage_95"] == 1.0


def test_redshift_summary_outlier_fraction() -> None:
    frame = pd.DataFrame(
        {
            "delta_z": [0.0, 0.2],
            "pit": [0.5, 1.0],
            "covered_68": [True, False],
            "covered_95": [True, True],
            "posterior_width_68": [0.1, 0.3],
        }
    )

    summary = summarize_redshift_metrics(frame)

    assert summary["n_objects"] == 2
    assert summary["outlier_fraction_0p15"] == 0.5
    assert summary["coverage_68"] == 0.5
    assert summary["coverage_95"] == 1.0


def test_write_redshift_metrics_includes_alpha_and_likelihood_metadata(tmp_path) -> None:
    dataset = tmp_path / "truth.parquet"
    pd.DataFrame(
        {
            "object_id": [1],
            "redshift_true": [0.5],
            "logsm_true": [10.0],
        }
    ).to_parquet(dataset, index=False)
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame(
        {
            "object_id": [1, 1],
            "z_obs": [0.45, 0.55],
            "log10_stellar_mass": [9.8, 9.9],
            "log10_stellar_mass_alpha_corrected": [10.0, 10.1],
        }
    ).to_parquet(run / "posterior_samples.parquet", index=False)
    (run / "inference_summary.json").write_text(
        json.dumps(
            {
                "global_sed_scale": {
                    "alpha_sed": 2.0,
                    "log_alpha_sed": 0.693147,
                    "delta_mag_global": -0.752575,
                    "alpha_prior_penalty": 24.0,
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "normalized_config.json").write_text(
        json.dumps(
            {
                "amortized": {
                    "likelihood": {
                        "type": "student_t",
                        "student_t_dof": 2.0,
                        "error_floor_frac": 0.02,
                        "error_jitter": 0.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    write_redshift_metrics_for_run(
        dataset_path=dataset,
        run_dir=run,
        label="toy",
    )

    photoz = pd.read_csv(run / "photoz_metrics.csv")
    posterior = pd.read_csv(run / "posterior_vs_truth_metrics.csv")
    assert photoz.loc[0, "alpha_sed"] == 2.0
    assert photoz.loc[0, "likelihood_type"] == "student_t"
    assert set(posterior["metric_name"]) >= {
        "mass_bias_raw",
        "mass_bias_alpha_corrected",
    }
