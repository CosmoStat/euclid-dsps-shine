from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from euclid_dsps.amortized.latent import latent_spec_from_config, theta_to_x, x_to_theta
from euclid_dsps.config import load_config
from euclid_dsps.parameters import POPCOSMOS_PARAMETER_NAMES


def test_latent_spec_uses_popcosmos_order() -> None:
    spec = latent_spec_from_config(load_config("configs/popcosmos_binned.yaml"))

    assert spec.names == POPCOSMOS_PARAMETER_NAMES
    assert spec.lower.shape == (16,)
    assert spec.upper.shape == (16,)


def test_x_theta_roundtrip_and_bounds() -> None:
    spec = latent_spec_from_config(load_config("configs/popcosmos_binned.yaml"))
    x = jnp.linspace(-2.0, 2.0, 16)

    theta = x_to_theta(x, spec)
    recovered = theta_to_x(theta, spec)

    assert theta.shape == (16,)
    assert jnp.all(theta >= spec.lower)
    assert jnp.all(theta <= spec.upper)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(x), atol=2.0e-5)


def test_latent_transform_supports_rank_two_and_three() -> None:
    spec = latent_spec_from_config(load_config("configs/popcosmos_binned.yaml"))

    assert x_to_theta(jnp.zeros((4, 16)), spec).shape == (4, 16)
    assert x_to_theta(jnp.zeros((2, 4, 16)), spec).shape == (2, 4, 16)


def test_gas_metallicity_constraint_is_satisfied() -> None:
    spec = latent_spec_from_config(load_config("configs/popcosmos_binned.yaml"))
    x = jnp.zeros((5, 16)).at[:, 12].set(-10.0)

    theta = x_to_theta(x, spec)

    stellar = theta[:, POPCOSMOS_PARAMETER_NAMES.index("log10_stellar_metallicity")]
    gas = theta[:, POPCOSMOS_PARAMETER_NAMES.index("log10_gas_metallicity")]
    assert jnp.all(gas >= stellar)
