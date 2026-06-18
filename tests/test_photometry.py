from __future__ import annotations

import math

import pytest

from euclid_dsps.photometric_uncertainty import (
    effective_flux_sigma,
    flux_error_from_model,
)
from euclid_dsps.photometry import (
    abmag_to_fnu_jy,
    fnu_jy_to_abmag,
    magerr_to_fluxerr_jy,
)


def test_ab_zero_point_is_3631_jy() -> None:
    assert abmag_to_fnu_jy(0.0) == pytest.approx(3631.0)


def test_abmag_flux_round_trip() -> None:
    mag = 24.3
    assert fnu_jy_to_abmag(abmag_to_fnu_jy(mag)) == pytest.approx(mag)


def test_magerr_to_fluxerr_jy_uses_local_derivative() -> None:
    flux = abmag_to_fnu_jy(23.0)
    err = magerr_to_fluxerr_jy(23.0, 0.1)

    assert err == pytest.approx(flux * math.log(10.0) / 2.5 * 0.1)


def test_fractional_snr_flux_error_and_effective_sigma() -> None:
    flux = [2.0e-28, -4.0e-28]
    err = flux_error_from_model(flux, {"type": "fractional_snr", "snr": 50.0})

    assert err[0] == pytest.approx(4.0e-30)
    assert err[1] == pytest.approx(8.0e-30)

    sigma = effective_flux_sigma(
        flux,
        err,
        model_flux=[3.0e-28, -5.0e-28],
        error_floor_frac=0.02,
    )
    assert sigma[0] == pytest.approx(math.sqrt((4.0e-30) ** 2 + (6.0e-30) ** 2))
