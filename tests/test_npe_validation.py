from __future__ import annotations

import importlib.util

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

HAS_EQUINOX = importlib.util.find_spec("equinox") is not None
pytestmark = pytest.mark.skipif(not HAS_EQUINOX, reason="equinox is not installed")

if HAS_EQUINOX:
    from euclid_dsps.amortized.features import FeatureStats
    from euclid_dsps.amortized.npe_validation import (
        assert_truth_free_columns,
        explicit_mixture_log_prob,
        flux_error_jacobian_sensitivity,
        held_out_band_predictive_gate,
        mask_held_out_bands,
        summarize_model_generated_rank_calibration,
        summarize_truth_free_joint_bank,
    )


def test_truth_columns_are_rejected_before_validation() -> None:
    with pytest.raises(ValueError, match="catalogue truth"):
        assert_truth_free_columns(["row_index", "redshift_true"])


def test_truth_free_joint_summary_separates_nonfinite_pareto(monkeypatch) -> None:
    import euclid_dsps.amortized.npe_validation as module

    calls = iter(
        [
            {
                "weight": np.array([0.5, 0.5]),
                "raw_ess": 2.0,
                "raw_ess_fraction": 1.0,
                "pareto_k": 0.8,
            },
            {
                "weight": np.array([0.9, 0.1]),
                "raw_ess": 1.22,
                "raw_ess_fraction": 0.61,
                "pareto_k": np.nan,
            },
        ]
    )
    monkeypatch.setattr(module, "normalized_importance_weights", lambda *_: next(calls))
    frame = pd.DataFrame(
        {
            "row_index": [0, 0, 1, 1],
            "sample_id": [0, 1, 0, 1],
            "logq": [0.0] * 4,
            "logprior": [0.0] * 4,
            "loglike": [0.0] * 4,
            "z_obs": [0.1, 0.2, 0.3, 0.4],
        }
    )
    summary, _ = summarize_truth_free_joint_bank(
        frame, parameter_names=("z_obs",), identity_column="row_index"
    )
    assert summary["pareto_k"]["nonfinite_fraction"] == 0.5
    assert summary["pareto_k"]["finite_gt_0p7_fraction"] == 1.0
    assert summary["pareto_k"]["gt_0p7_or_nonfinite_fraction"] == 1.0


def test_held_out_band_is_removed_from_features_and_likelihood_mask() -> None:
    stats = FeatureStats(
        flux_scale=jnp.ones(3),
        err_scale=jnp.ones(3),
        band_names=("a", "b", "c"),
        flux_transform="asinh",
        append_mask=True,
        error_epsilon=1e-6,
    )
    flux = jnp.ones((2, 3))
    err = jnp.ones((2, 3))
    mask = jnp.ones((2, 3), dtype=bool)
    features, conditioned = mask_held_out_bands(flux, err, mask, stats, [1])
    assert not bool(jnp.any(conditioned[:, 1]))
    assert bool(jnp.all(features[:, -3:][:, 1] == 0.0))


def test_flux_error_jacobian_reports_weak_coordinate() -> None:
    result = flux_error_jacobian_sensitivity(
        lambda x: jnp.asarray([2.0 * x[0], 0.0 * x[1]]),
        jnp.asarray([1.0, 2.0]),
        jnp.ones(2),
        jnp.ones(2, dtype=bool),
    )
    assert np.isclose(float(result["coordinate_norm"][0]), 2.0)
    assert bool(result["near_zero_coordinate"][1])


def test_defensive_mixture_uses_every_component_density() -> None:
    component = jnp.log(jnp.asarray([[0.2, 0.8], [0.9, 0.1]]))
    value = explicit_mixture_log_prob(component, jnp.asarray([0.25, 0.75]))
    expected = np.log([0.25 * 0.2 + 0.75 * 0.8, 0.25 * 0.9 + 0.75 * 0.1])
    assert np.allclose(np.asarray(value), expected)


def test_model_generated_rank_calibration_accepts_exchangeable_draws() -> None:
    rng = np.random.default_rng(4)
    joint = rng.normal(size=(65, 512, 2))
    result = summarize_model_generated_rank_calibration(
        joint[:-1],
        joint[-1],
        parameter_names=("a", "b"),
        seed=5,
        maximum_ks=0.10,
        maximum_coverage_ece=0.10,
    )
    assert result["status"] == "PASS"
    assert result["catalogue_truth_used"] is False


def test_held_out_gate_compares_to_model_generated_reference() -> None:
    reference = {
        "by_band": {
            "r": {
                "count": 100,
                "median_abs": 0.7,
                "rms": 1.0,
                "fraction_abs_gt_5": 0.01,
            }
        }
    }
    observed = {
        "by_band": {
            "r": {
                "count": 100,
                "median_abs": 0.8,
                "rms": 1.2,
                "fraction_abs_gt_5": 0.02,
            }
        }
    }
    assert held_out_band_predictive_gate(observed, reference)["status"] == "PASS"
