from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.calibration import (
    alpha_from_log_alpha,
    apply_global_sed_scale_to_flux,
    apply_global_sed_scale_to_sed,
    apply_per_band_flux_calibration_to_flux,
    delta_mag_from_alpha,
    delta_mag_to_log_alpha,
    global_sed_scale_prior_penalty,
    log_alpha_to_delta_mag,
    per_band_flux_calibration_metadata,
)


def test_alpha_sed_one_leaves_sed_and_flux_unchanged() -> None:
    sed = jnp.asarray([1.0, 2.0, 3.0])
    flux = jnp.asarray([[4.0, 5.0]])

    assert jnp.allclose(apply_global_sed_scale_to_sed(sed, 0.0), sed)
    assert jnp.allclose(apply_global_sed_scale_to_flux(flux, 0.0), flux)


def test_alpha_sed_two_doubles_fluxes() -> None:
    flux = jnp.asarray([[4.0, 5.0]])
    log_alpha = jnp.log(jnp.asarray(2.0))

    assert jnp.allclose(alpha_from_log_alpha(log_alpha), 2.0)
    assert jnp.allclose(apply_global_sed_scale_to_flux(flux, log_alpha), 2.0 * flux)


def test_delta_mag_global_and_prior_penalty() -> None:
    assert np.isclose(delta_mag_from_alpha(2.0), -2.5 * np.log10(2.0))
    assert np.isclose(float(global_sed_scale_prior_penalty(0.0, 0.1)), 0.0)
    assert float(global_sed_scale_prior_penalty(jnp.log(jnp.asarray(2.0)), 0.1)) > 0.0


def test_per_band_flux_calibration_scales_each_band() -> None:
    flux = jnp.asarray([[1.0, 2.0, 4.0]])
    log_alpha = jnp.log(jnp.asarray([1.0, 2.0, 0.5]))

    scaled = apply_per_band_flux_calibration_to_flux(flux, log_alpha)

    assert jnp.allclose(scaled, jnp.asarray([[1.0, 4.0, 2.0]]))


def test_per_band_delta_mag_metadata_roundtrips() -> None:
    offsets_mag = np.asarray([0.05, -0.10])
    log_alpha = delta_mag_to_log_alpha(offsets_mag)
    payload = per_band_flux_calibration_metadata(
        log_alpha,
        ("lsst_u", "lsst_g"),
        prior_sigma_log_alpha=0.05,
    )

    assert np.allclose(log_alpha_to_delta_mag(log_alpha), offsets_mag)
    assert np.isclose(
        payload["bands"]["lsst_u"]["delta_mag_band"],
        offsets_mag[0],
    )
    assert payload["max_abs_delta_mag"] == pytest.approx(0.10)
