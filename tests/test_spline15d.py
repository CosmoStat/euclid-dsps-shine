from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from euclid_dsps.prior_learning.spline15d import (
    DEFAULT_NORMALIZED_LOG_TIME_NODES,
    SFH_CONTRAST_NAMES,
    SPLINE15D_PARAMETER_NAMES,
    dequantize_spline_contrast_atoms,
    fit_asinh_transforms,
    forward_asinh_matrix,
    inverse_asinh_matrix,
    reconstruct_relative_sfh_jax,
    spline_knot_times_jax,
    validate_normalized_log_time_nodes,
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
