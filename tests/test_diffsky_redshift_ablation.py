from __future__ import annotations

import numpy as np
import pandas as pd

from euclid_dsps.diffsky_redshift_ablation import (
    redshift_metrics_from_samples,
    summarize_redshift_metrics,
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
