from __future__ import annotations

import numpy as np
import pytest

from euclid_dsps.openuniverse.noise import (
    add_band_snr_noise,
    add_depth_like_noise,
    add_fractional_snr_noise,
)


def test_fractional_snr_noise_shape_positive_error_and_reproducible() -> None:
    flux = np.full((500, 3), 100.0)

    noisy_a, err_a = add_fractional_snr_noise(flux, snr=50.0, seed=7)
    noisy_b, err_b = add_fractional_snr_noise(flux, snr=50.0, seed=7)

    assert noisy_a.shape == flux.shape
    assert err_a.shape == flux.shape
    assert np.all(err_a > 0.0)
    np.testing.assert_allclose(noisy_a, noisy_b)
    np.testing.assert_allclose(err_a, err_b)
    assert np.std(noisy_a - flux) == pytest.approx(2.0, rel=0.25)


def test_band_snr_noise_uses_per_band_snr() -> None:
    flux = np.full((100, 2), 100.0)

    _, err = add_band_snr_noise(
        flux,
        band_snr={"a": 10.0, "b": 50.0},
        band_names=("a", "b"),
        seed=1,
    )

    assert np.median(err[:, 0]) == pytest.approx(10.0)
    assert np.median(err[:, 1]) == pytest.approx(2.0)


def test_depth_like_noise_is_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="depth-like noise"):
        add_depth_like_noise(np.ones((2, 2)))
