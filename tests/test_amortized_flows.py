from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import pytest

HAS_EQUINOX = importlib.util.find_spec("equinox") is not None
pytestmark = pytest.mark.skipif(
    not HAS_EQUINOX,
    reason="Equinox optional dependency is not installed",
)

if HAS_EQUINOX:
    from euclid_dsps.amortized.flows import RealNVPPrior, StandardNormalPrior


def test_realnvp_roundtrip_logprob_and_sample_shape() -> None:
    prior = RealNVPPrior(
        jax.random.PRNGKey(0),
        latent_dim=4,
        n_layers=4,
        hidden_size=8,
    )
    u = jnp.ones((3, 4)) * 0.2

    x, _ = prior.forward(u)
    recovered, _ = prior.inverse(x)
    logp = prior.log_prob(x)
    samples = prior.sample(jax.random.PRNGKey(1), (2, 3))

    assert recovered.shape == u.shape
    assert logp.shape == (3,)
    assert samples.shape == (2, 3, 4)
    assert jnp.all(jnp.isfinite(logp))
    assert jnp.allclose(recovered, u, atol=1.0e-5)


def test_realnvp_input_gradients_are_finite() -> None:
    prior = RealNVPPrior(
        jax.random.PRNGKey(0),
        latent_dim=4,
        n_layers=2,
        hidden_size=8,
    )

    grad = jax.grad(lambda x: -jnp.sum(prior.log_prob(x)))(jnp.ones((2, 4)))

    assert jnp.all(jnp.isfinite(grad))


def test_realnvp_identity_init_matches_standard_normal_log_prob() -> None:
    prior = RealNVPPrior(
        jax.random.PRNGKey(0),
        latent_dim=4,
        n_layers=4,
        hidden_size=8,
        init="identity",
        init_scale=0.0,
    )
    standard = StandardNormalPrior(latent_dim=4)
    x = jnp.asarray(
        [
            [0.0, 0.1, -0.2, 0.3],
            [1.0, -0.5, 0.25, 0.75],
        ],
        dtype=jnp.float32,
    )
    u, inverse_logdet = prior.inverse(x)
    recovered, forward_logdet = prior.forward(u)

    assert jnp.allclose(u, x, atol=1.0e-6)
    assert jnp.allclose(recovered, x, atol=1.0e-6)
    assert jnp.allclose(inverse_logdet, 0.0, atol=1.0e-6)
    assert jnp.allclose(forward_logdet, 0.0, atol=1.0e-6)
    assert jnp.allclose(prior.log_prob(x), standard.log_prob(x), atol=1.0e-6)
