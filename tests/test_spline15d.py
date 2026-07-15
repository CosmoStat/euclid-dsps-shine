from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from euclid_dsps.prior_learning.spline15d import (
    DEFAULT_NORMALIZED_LOG_TIME_NODES,
    SFH_CONTRAST_NAMES,
    SPLINE15D_PARAMETER_NAMES,
    dequantize_normalized_zero_atoms,
    dequantize_spline_contrast_atoms,
    fit_affine_whitening,
    fit_asinh_transforms,
    forward_affine_whitening,
    forward_asinh_matrix,
    inverse_asinh_matrix,
    inverse_spline15d_flow_coordinates,
    reconstruct_relative_sfh_jax,
    spline_knot_times_jax,
    validate_normalized_log_time_nodes,
)
from euclid_dsps.prior_learning.spline15d_evaluation import (
    evaluate_sample_pair,
    novel_truth_mask,
    select_temperature,
    selection_payload,
)


def test_spline_node_validation() -> None:
    nodes = validate_normalized_log_time_nodes(DEFAULT_NORMALIZED_LOG_TIME_NODES)
    assert nodes.shape == (11,)
    with pytest.raises(ValueError, match="increasing"):
        validate_normalized_log_time_nodes(np.linspace(0.0, 1.0, 11)[::-1])


def test_dequantization_only_changes_exact_sfh_atoms() -> None:
    frame = pd.DataFrame(
        np.ones((6, len(SPLINE15D_PARAMETER_NAMES))),
        columns=SPLINE15D_PARAMETER_NAMES,
    )
    frame.insert(0, "object_id", np.arange(len(frame)))
    frame.loc[[0, 2, 5], SFH_CONTRAST_NAMES[3]] = 0.0
    result, counts = dequantize_spline_contrast_atoms(
        frame, half_width_dex=1.0e-4, seed=12
    )
    assert counts[SFH_CONTRAST_NAMES[3]] == 3
    changed = result[SFH_CONTRAST_NAMES[3]] != frame[SFH_CONTRAST_NAMES[3]]
    assert changed.to_numpy().tolist() == [True, False, True, False, False, True]
    assert np.max(np.abs(result.loc[changed, SFH_CONTRAST_NAMES[3]])) <= 1.0e-4
    np.testing.assert_array_equal(result["object_id"], frame["object_id"])


def test_asinh_transform_roundtrip() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.standard_t(df=3, size=(1000, len(SPLINE15D_PARAMETER_NAMES)))
    matrix[:, 0] = rng.exponential(size=len(matrix))
    transforms = fit_asinh_transforms(matrix, grid_size=33)
    normalized = forward_asinh_matrix(matrix, transforms)
    recovered = inverse_asinh_matrix(normalized, transforms)
    np.testing.assert_allclose(recovered, matrix, rtol=1.0e-11, atol=1.0e-11)
    assert np.isfinite(normalized).all()
    assert all(payload["lambda"] > 0.0 for payload in transforms.values())


def test_normalized_atom_dequantization_whitening_and_scientific_inverse() -> None:
    rng = np.random.default_rng(14)
    physical = rng.normal(size=(2000, len(SPLINE15D_PARAMETER_NAMES)))
    physical[:, 0] = rng.uniform(0.01, 5.0, len(physical))
    physical[:, 3] = rng.uniform(0.0, 2.0, len(physical))
    atom_index = SPLINE15D_PARAMETER_NAMES.index(SFH_CONTRAST_NAMES[2])
    physical[:500, atom_index] = 0.0
    transforms = fit_asinh_transforms(physical, grid_size=33)
    asinh_values = forward_asinh_matrix(physical, transforms)
    dequantized, counts = dequantize_normalized_zero_atoms(
        asinh_values,
        physical,
        half_width=0.05,
        seed=17,
    )
    assert counts[SFH_CONTRAST_NAMES[2]] == 500
    whitening = fit_affine_whitening(dequantized, covariance_jitter=1.0e-5)
    flow_values = forward_affine_whitening(dequantized, whitening)
    recovered = inverse_spline15d_flow_coordinates(
        flow_values,
        transforms=transforms,
        whitening=whitening,
        atom_half_width=0.05,
    )
    expected = physical.copy()
    for name in SFH_CONTRAST_NAMES:
        index = SPLINE15D_PARAMETER_NAMES.index(name)
        zero_normalized = -transforms[name]["center"] / transforms[name]["scale"]
        expected[
            np.abs(dequantized[:, index] - zero_normalized) <= 0.05,
            index,
        ] = 0.0
    np.testing.assert_allclose(recovered, expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(recovered[:500, atom_index], 0.0)
    covariance = np.cov(flow_values, rowvar=False)
    np.testing.assert_allclose(covariance, np.eye(15), atol=2.0e-3)


def test_relative_sfh_interpolates_all_knot_values() -> None:
    time = jnp.geomspace(0.05, 12.0, 400)
    contrasts = jnp.asarray([0.3, -0.2, 0.4, -0.1, 0.2, -0.3, 0.1, 0.2, -0.1, 0.05])
    knot_time = spline_knot_times_jax(
        time, jnp.asarray(DEFAULT_NORMALIZED_LOG_TIME_NODES)
    )
    reconstructed = reconstruct_relative_sfh_jax(knot_time, contrasts)
    expected = 10 ** jnp.concatenate((jnp.zeros(1), jnp.cumsum(contrasts)))
    np.testing.assert_allclose(reconstructed, expected, rtol=2.0e-5, atol=2.0e-5)
    assert bool(jnp.all(reconstructed > 0.0))


def test_novel_truth_mask_uses_fixed_15d_contract() -> None:
    train = pd.DataFrame(
        np.arange(45, dtype=float).reshape(3, 15),
        columns=SPLINE15D_PARAMETER_NAMES,
    )
    candidate = pd.concat(
        [train.iloc[[1]], train.iloc[[2]].assign(z_obs=-100.0)],
        ignore_index=True,
    )

    np.testing.assert_array_equal(novel_truth_mask(train, candidate), [False, True])


def test_selection_and_temperature_prefer_eligible_validation_candidate() -> None:
    rng = np.random.default_rng(8)
    truth = rng.normal(size=(1000, 15))
    metrics = evaluate_sample_pair(
        truth_theta=truth,
        truth_x=truth,
        prior_theta=truth.copy(),
        prior_x=truth.copy(),
    )
    payload = selection_payload(
        metrics,
        thresholds={
            "max_median_ks": 0.10,
            "max_max_ks": 0.20,
            "max_correlation_frobenius": 1.5,
            "min_base_std_mean": 0.8,
            "max_base_std_mean": 1.2,
            "max_normalized_tail_fraction": 0.002,
            "max_negative_fraction": 1.0,
        },
    )
    assert payload["eligible"]
    scan = pd.DataFrame(
        {
            "base_temperature": [0.1, 0.2],
            "eligible": [False, True],
            "metric": [0.01, 0.2],
        }
    )
    assert float(select_temperature(scan)["base_temperature"]) == 0.2
