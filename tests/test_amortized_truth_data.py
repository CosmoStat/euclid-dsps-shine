from __future__ import annotations

import numpy as np
import pytest

from euclid_dsps.amortized.data import (
    _transform_truth_array,
    _truth_theta_batch,
)
from euclid_dsps.observation_arrays import PhotometryArrays


def test_truth_array_applies_configured_transforms() -> None:
    values = np.array([1.0, 10.0, 100.0])
    transformed = _transform_truth_array(
        values,
        {"transform": "log10", "scale": 2.0, "offset": -1.0},
    )
    assert np.allclose(transformed, [-1.0, 1.0, 3.0])

    mass = _transform_truth_array(
        np.array([10.0]),
        {"transform": "log_stellar_mass_h2_to_msun", "h": 0.7},
    )
    assert np.allclose(mass, 10.0 + 2.0 * np.log10(0.7))


def test_truth_batch_rejects_non_finite_supervision() -> None:
    arrays = PhotometryArrays(
        object_id=np.array([1, 2]),
        flux=np.ones((2, 1)),
        flux_err=np.ones((2, 1)),
        mask=np.ones((2, 1), dtype=bool),
        band_names=("b",),
        truth={"theta": np.array([0.5, np.nan])},
    )
    with pytest.raises(ValueError, match="Non-finite NPE truth value"):
        _truth_theta_batch(arrays, np.array([0, 1]), ("theta",))
