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
    from euclid_dsps.amortized.encoder import GaussianEncoder


def test_gaussian_encoder_shapes_and_logq() -> None:
    encoder = GaussianEncoder(
        jax.random.PRNGKey(0),
        input_dim=20,
        latent_dim=16,
        hidden_sizes=(8, 8),
    )
    features = jnp.ones((5, 20))

    mean, log_std = encoder(features)
    samples, logq = encoder.sample_and_log_prob(jax.random.PRNGKey(1), features, 3)

    assert mean.shape == (5, 16)
    assert log_std.shape == (5, 16)
    assert samples.shape == (3, 5, 16)
    assert logq.shape == (3, 5)
    assert jnp.all(jnp.isfinite(logq))


def test_gaussian_encoder_gradients_are_finite() -> None:
    encoder = GaussianEncoder(
        jax.random.PRNGKey(0),
        input_dim=20,
        latent_dim=16,
        hidden_sizes=(8,),
    )
    features = jnp.ones((3, 20))

    grad = jax.grad(lambda x: jnp.sum(encoder(x)[0]))(features)

    assert jnp.all(jnp.isfinite(grad))
