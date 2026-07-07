from __future__ import annotations

import numpy as np

from euclid_dsps.openuniverse.diagnostics import compute_prior_overlap_metrics


def test_prior_overlap_metrics_on_toy_samples() -> None:
    truth = np.asarray([0.0, 0.5, 1.0])
    posterior = np.asarray([0.0, 0.4, 0.6, 1.0])
    prior = np.linspace(-1.0, 2.0, 101)

    metrics = compute_prior_overlap_metrics(
        truth,
        posterior,
        prior,
        name="z",
    )

    assert metrics["parameter"] == "z"
    assert metrics["n_truth"] == 3
    assert 0.0 <= metrics["ks_truth_prior"] <= 1.0
    assert metrics["truth_outside_prior_95"] == 0.0
