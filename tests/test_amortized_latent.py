from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from euclid_dsps.amortized.latent import latent_spec_from_config, theta_to_x, x_to_theta
from euclid_dsps.config import load_config
from euclid_dsps.parameters import POPCOSMOS_PARAMETER_NAMES


def test_latent_spec_uses_popcosmos_order() -> None:
    spec = latent_spec_from_config(load_config("configs/fs2_gpu.yaml"))

    assert spec.names == POPCOSMOS_PARAMETER_NAMES
    assert spec.lower.shape == (16,)
    assert spec.upper.shape == (16,)


def test_x_theta_roundtrip_and_bounds() -> None:
    spec = latent_spec_from_config(load_config("configs/fs2_gpu.yaml"))
    x = jnp.linspace(-2.0, 2.0, 16)

    theta = x_to_theta(x, spec)
    recovered = theta_to_x(theta, spec)

    assert theta.shape == (16,)
    assert jnp.all(theta >= spec.lower)
    assert jnp.all(theta <= spec.upper)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(x), atol=2.0e-5)


def test_latent_transform_supports_rank_two_and_three() -> None:
    spec = latent_spec_from_config(load_config("configs/fs2_gpu.yaml"))

    assert x_to_theta(jnp.zeros((4, 16)), spec).shape == (4, 16)
    assert x_to_theta(jnp.zeros((2, 4, 16)), spec).shape == (2, 4, 16)


def test_diffsky_hltds_latent_schema_uses_configured_free_parameters() -> None:
    config = load_config("configs/amortized_diffsky_hltds_04_14_realnvp_gpu.yaml")
    spec = latent_spec_from_config(config)

    assert config["amortized"]["latent"]["schema"] == "diffsky_hltds_prior_v1"
    assert spec.names == tuple(config["fit"]["free_parameters"])
    assert spec.names == (
        "z_obs",
        "log10_stellar_mass",
        "dlog10_sfr_1",
        "dlog10_sfr_2",
        "dlog10_sfr_3",
        "dlog10_sfr_4",
        "dlog10_sfr_5",
        "dlog10_sfr_6",
        "log10_stellar_metallicity",
        "tau2",
        "dust_index_n",
        "tau1_over_tau2",
    )
    assert spec.lower.shape == (12,)
    assert spec.upper.shape == (12,)
    assert spec.normalization == "standardized_logit"
    assert spec.raw_center is not None
    assert spec.raw_scale is not None


def test_diffsky_supervised_prior_config_matches_truth_basic_schema() -> None:
    config = load_config("configs/amortized_diffsky_hltds_supervised_prior_gpu.yaml")
    spec = latent_spec_from_config(config)

    assert config["amortized"]["latent"]["schema"] == "diffsky_truth_basic"
    assert config["amortized"]["encoder"]["latent_dim"] == 5
    assert spec.names == (
        "z_obs",
        "log10_stellar_mass",
        "log10_ssfr_at_obs",
        "dust_av",
        "dust_delta",
    )
    np.testing.assert_allclose(
        np.asarray(spec.lower),
        np.asarray([0.001, 6.0, -14.5, 0.0, -2.5], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(spec.upper),
        np.asarray([0.5, 13.5, -7.0, 5.0, 1.0], dtype=np.float32),
    )


def test_gas_metallicity_constraint_is_satisfied() -> None:
    spec = latent_spec_from_config(load_config("configs/fs2_gpu.yaml"))
    x = jnp.zeros((5, 16)).at[:, 12].set(-10.0)

    theta = x_to_theta(x, spec)

    stellar = theta[:, POPCOSMOS_PARAMETER_NAMES.index("log10_stellar_metallicity")]
    gas = theta[:, POPCOSMOS_PARAMETER_NAMES.index("log10_gas_metallicity")]
    assert jnp.all(gas >= stellar)
