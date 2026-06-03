from __future__ import annotations

import importlib.util

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
    from euclid_dsps.amortized import infer as infer_mod
    from euclid_dsps.amortized.latent import LatentSpec


def test_model_flux_from_x_sample_chunks_splits_sample_axis(monkeypatch) -> None:
    calls = []

    def fake_model_flux_from_x(x, latent_spec, context, model_args, parameter_names):
        calls.append(tuple(x.shape))
        return jnp.ones(x.shape[:-1] + (10,), dtype=x.dtype) * len(calls)

    monkeypatch.setattr(infer_mod, "model_flux_from_x", fake_model_flux_from_x)
    spec = LatentSpec(
        names=tuple(f"p{i}" for i in range(16)),
        lower=jnp.zeros(16),
        upper=jnp.ones(16),
    )
    x = jnp.zeros((5, 2, 16), dtype=jnp.float32)

    flux = infer_mod._model_flux_from_x_sample_chunks(
        x,
        spec,
        None,
        None,
        spec.names,
        sample_chunk_size=2,
    )

    assert flux.shape == (5, 2, 10)
    assert calls == [(2, 2, 16), (2, 2, 16), (1, 2, 16)]
    assert jnp.all(flux[:2] == 1.0)
    assert jnp.all(flux[2:4] == 2.0)
    assert jnp.all(flux[4:] == 3.0)


def test_model_flux_from_x_sample_chunks_rejects_nonpositive_chunk() -> None:
    spec = LatentSpec(
        names=tuple(f"p{i}" for i in range(16)),
        lower=jnp.zeros(16),
        upper=jnp.ones(16),
    )
    with pytest.raises(ValueError, match="sample_chunk_size must be positive"):
        infer_mod._model_flux_from_x_sample_chunks(
            jnp.zeros((2, 1, 16)),
            spec,
            None,
            None,
            spec.names,
            sample_chunk_size=0,
        )
