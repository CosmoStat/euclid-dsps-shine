from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import pytest

HAS_DEPS = (
    importlib.util.find_spec("equinox") is not None
    and importlib.util.find_spec("optax") is not None
)
pytestmark = pytest.mark.skipif(
    not HAS_DEPS,
    reason="Equinox/Optax optional dependencies are not installed",
)

if HAS_DEPS:
    from euclid_dsps.amortized.data import PhotometryBatch
    from euclid_dsps.amortized.elbo import negative_elbo
    from euclid_dsps.amortized.latent import LatentSpec
    from euclid_dsps.amortized.train import build_amortized_model


def test_negative_elbo_mock_decoder_is_finite() -> None:
    config = {"amortized": {"encoder": {"hidden_sizes": [8]}, "prior": {"n_layers": 2}}}
    model = build_amortized_model(config, jax.random.PRNGKey(0))
    batch = PhotometryBatch(
        object_id=jnp.arange(4),
        flux=jnp.ones((4, 10)) * 1.0e-11,
        flux_err=jnp.ones((4, 10)) * 1.0e-12,
        mask=jnp.ones((4, 10), dtype=bool),
        features=jnp.ones((4, 20)),
    )
    spec = LatentSpec(tuple(f"p{i}" for i in range(16)), jnp.zeros(16), jnp.ones(16))
    decoder = {"weights": jnp.zeros((16, 10)), "bias": jnp.ones(10) * -25.0}

    loss, metrics = negative_elbo(
        model,
        batch,
        spec,
        None,
        None,
        spec.names,
        jax.random.PRNGKey(1),
        2,
        1.0,
        {"type": "student_t"},
        use_mock_decoder=True,
        mock_decoder_params=decoder,
    )

    assert jnp.isfinite(loss)
    assert jnp.isfinite(metrics["kl_mc_mean"])


def test_deterministic_reconstruction_objective_uses_encoder_mean_only() -> None:
    config = {"amortized": {"encoder": {"hidden_sizes": [8]}, "prior": {"n_layers": 2}}}
    model = build_amortized_model(config, jax.random.PRNGKey(2))
    batch = PhotometryBatch(
        object_id=jnp.arange(3),
        flux=jnp.ones((3, 10)) * 1.0e-11,
        flux_err=jnp.ones((3, 10)) * 1.0e-12,
        mask=jnp.ones((3, 10), dtype=bool),
        features=jnp.ones((3, 20)),
    )
    spec = LatentSpec(tuple(f"p{i}" for i in range(16)), jnp.zeros(16), jnp.ones(16))
    decoder = {"weights": jnp.zeros((16, 10)), "bias": jnp.ones(10) * -25.0}

    loss, metrics = negative_elbo(
        model,
        batch,
        spec,
        None,
        None,
        spec.names,
        jax.random.PRNGKey(3),
        8,
        1.0,
        {"type": "student_t"},
        objective_config={"mode": "deterministic_reconstruction"},
        use_mock_decoder=True,
        mock_decoder_params=decoder,
    )

    assert jnp.isfinite(loss)
    assert metrics["deterministic_reconstruction"] == 1.0
    assert metrics["effective_n_samples"] == 1.0
    assert metrics["kl_mc_mean"] == 0.0
