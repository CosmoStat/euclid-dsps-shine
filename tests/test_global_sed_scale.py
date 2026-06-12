from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from euclid_dsps.calibration import (
    alpha_from_log_alpha,
    apply_global_sed_scale_to_flux,
    apply_global_sed_scale_to_sed,
    delta_mag_from_alpha,
    global_sed_scale_prior_penalty,
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
