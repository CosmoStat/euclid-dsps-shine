from __future__ import annotations

import numpy as np
import pytest

from euclid_dsps.likelihood import (
    chi2_flux,
    chi2_mag,
    loglike_flux_gaussian,
    student_t_flux_neg2loglike,
)


def test_chi2_flux_is_zero_for_matching_fluxes() -> None:
    flux = np.asarray([1.0, 2.0, 3.0])
    err = np.asarray([0.1, 0.2, 0.3])

    assert chi2_flux(flux, flux, err) == pytest.approx(0.0)
    assert loglike_flux_gaussian(flux, flux, err) == pytest.approx(0.0)


def test_chi2_flux_ignores_masked_bands() -> None:
    model = np.asarray([1.0, 100.0])
    obs = np.asarray([1.0, 0.0])
    err = np.asarray([0.1, 0.1])

    assert chi2_flux(model, obs, err, mask=[True, False]) == pytest.approx(0.0)


def test_chi2_flux_masks_invalid_errors() -> None:
    model = np.asarray([2.0, 3.0, 100.0])
    obs = np.asarray([1.0, 1.0, 1.0])
    err = np.asarray([1.0, np.nan, 0.0])

    assert chi2_flux(model, obs, err) == pytest.approx(1.0)


def test_chi2_flux_applies_floor_and_jitter() -> None:
    chi2 = chi2_flux([2.0], [1.0], [0.1], floor_frac=0.2, jitter=0.3)

    assert chi2 == pytest.approx(1.0 / (0.1**2 + 0.2**2 + 0.3**2))


def test_student_t_flux_neg2loglike_uses_two_dof_popcosmos_default() -> None:
    loss = student_t_flux_neg2loglike([3.0], [1.0], [1.0], dof=2.0)

    assert loss == pytest.approx(3.0 * np.log1p(4.0 / 2.0))


def test_chi2_mag_legacy_array_api() -> None:
    assert chi2_mag([20.0], [20.2], [0.1]) == pytest.approx(4.0)
