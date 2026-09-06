from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from euclid_dsps.amortized.mira import FENIKS_SPLINE15D_PARAMETERS
from euclid_dsps.amortized.tarp import (
    _tarp_ecp,
    evaluate_feniks_tarp,
    randomized_finite_rank_uniform_ks,
    tarp_coverage_values,
)


def test_tarp_coverage_values_match_direct_numpy() -> None:
    truth = np.asarray([[0.1, 0.4], [0.8, 0.2]], dtype=np.float32)
    posterior = np.asarray(
        [
            [
                [[0.0, 0.3], [0.2, 0.5], [0.4, 0.2]],
                [[0.7, 0.1], [0.9, 0.3], [0.6, 0.4]],
            ],
            [
                [[0.2, 0.2], [0.3, 0.6], [0.5, 0.1]],
                [[0.6, 0.2], [1.0, 0.4], [0.5, 0.5]],
            ],
        ],
        dtype=np.float32,
    )
    references = np.asarray([[0.5, 0.5], [0.1, 0.4]], dtype=np.float32)
    actual = np.asarray(
        jax.device_get(tarp_coverage_values(truth, posterior, references))
    )
    expected = np.empty((2, 2), dtype=float)
    for model_index in range(2):
        for object_index in range(2):
            sample_distance = np.sum(
                (posterior[model_index, object_index] - references[object_index]) ** 2,
                axis=1,
            )
            truth_distance = np.sum(
                (truth[object_index] - references[object_index]) ** 2
            )
            expected[model_index, object_index] = np.mean(
                sample_distance < truth_distance
            )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_tarp_ecp_matches_histogram_definition() -> None:
    values = np.asarray([0.05, 0.15, 0.55, 0.95])
    ecp, alpha = _tarp_ecp(values, num_alpha_bins=4)
    assert np.array_equal(alpha, np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]))
    np.testing.assert_allclose(ecp, np.asarray([0.0, 0.5, 0.5, 0.75, 1.0]))


def test_tarp_ks_uses_randomized_finite_ranks_not_curve_ordinates() -> None:
    k = 8
    ranks = np.tile(np.arange(k + 1), 256)
    result = randomized_finite_rank_uniform_ks(
        ranks / float(k), posterior_samples=k, seed=19
    )
    assert result["ks_pvalue_method"] == "randomized_finite_rank_uniform"
    assert result["ks_statistic"] < 0.03


def test_tarp_randomized_rank_rejects_values_off_finite_grid() -> None:
    with np.testing.assert_raises_regex(ValueError, "expected K grid"):
        randomized_finite_rank_uniform_ks(
            np.asarray([0.13, 0.42]), posterior_samples=8, seed=3
        )


def test_feniks_tarp_workflow_writes_auditable_outputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(23)
    n_objects = 32
    n_samples = 16
    n_dimensions = len(FENIKS_SPLINE15D_PARAMETERS)
    object_ids = np.arange(30_000, 30_000 + n_objects)
    truth_values = rng.uniform(0.1, 0.9, size=(n_objects, n_dimensions))
    posterior_values = truth_values[:, None, :] + 0.25 * rng.normal(
        size=(n_objects, n_samples, n_dimensions)
    )
    truth = pd.DataFrame(
        truth_values,
        columns=FENIKS_SPLINE15D_PARAMETERS,
    )
    truth.insert(0, "row_index", np.arange(n_objects))
    truth.insert(0, "object_id", object_ids)
    inference = tmp_path / "inference"
    inference.mkdir()
    truth_path = inference / "inference_truth.parquet"
    truth.to_parquet(truth_path, index=False)
    samples = pd.DataFrame(
        posterior_values.reshape(n_objects * n_samples, n_dimensions),
        columns=FENIKS_SPLINE15D_PARAMETERS,
    )
    samples.insert(0, "sample_id", np.tile(np.arange(n_samples), n_objects))
    samples.insert(0, "row_index", np.repeat(np.arange(n_objects), n_samples))
    samples.insert(0, "object_id", np.repeat(object_ids, n_samples))
    samples.to_parquet(inference / "posterior_samples.parquet", index=False)

    out = tmp_path / "tarp"
    summary = evaluate_feniks_tarp(
        truth_path=truth_path,
        posterior_specs=[("seed2", inference), ("seed3", inference)],
        out_dir=out,
        num_alpha_bins=8,
        num_bootstrap=20,
        samples_per_object=n_samples,
        seed=123,
    )

    assert summary["status"] == "complete"
    assert summary["num_objects"] == n_objects
    assert summary["num_posterior_samples"] == n_samples
    assert summary["num_alpha_bins"] == 8
    assert summary["selected_sample_ids"] == list(range(n_samples))
    expected = {
        "DONE",
        "tarp_manifest.json",
        "tarp_summary.json",
        "tarp_summary.csv",
        "tarp_coverage.csv",
        "tarp_coverage.parquet",
        "tarp_coverage_values.parquet",
        "tarp_pairwise_differences.csv",
        "tarp_normalization.csv",
        "tarp_normalization_diagnostics.csv",
        "tarp_coverage.png",
    }
    assert expected <= {path.name for path in out.iterdir()}
    coverage = pd.read_parquet(out / "tarp_coverage.parquet")
    assert len(coverage) == 18 * 2 * 9
    assert coverage["ecp"].between(0.0, 1.0).all()
    values = pd.read_parquet(out / "tarp_coverage_values.parquet")
    assert len(values) == 18 * 2 * n_objects
    pairwise = pd.read_csv(out / "tarp_pairwise_differences.csv")
    assert len(pairwise) == 18
    np.testing.assert_allclose(pairwise["atc_a_minus_b"], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(pairwise["delta_mean"], 0.0, atol=1.0e-12)
    manifest = json.loads((out / "tarp_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["shared_random_references_across_models"] is True
    assert manifest["bootstrap_unit"] == "held_out_object"
