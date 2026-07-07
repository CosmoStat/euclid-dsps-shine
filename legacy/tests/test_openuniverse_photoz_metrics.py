from __future__ import annotations

import numpy as np

from euclid_dsps.openuniverse.diagnostics import compute_photoz_metrics, redshift_pit


def test_photoz_metrics_on_toy_posterior() -> None:
    truth = np.asarray([0.5, 1.0, 2.0])
    samples = np.asarray(
        [
            [0.48, 0.95, 2.1],
            [0.50, 1.00, 2.0],
            [0.52, 1.05, 1.9],
        ]
    )

    metrics = compute_photoz_metrics(samples, truth)
    pit = redshift_pit(samples, truth)

    assert metrics["n_objects"] == 3
    assert metrics["median_delta_z"] == 0.0
    assert metrics["outlier_fraction_015"] == 0.0
    assert metrics["coverage_68"] == 1.0
    assert pit.tolist() == [1 / 3, 1 / 3, 1 / 3]
