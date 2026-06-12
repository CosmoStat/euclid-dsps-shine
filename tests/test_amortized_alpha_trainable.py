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
    import equinox as eqx
    import optax

    from euclid_dsps.amortized.data import PhotometryBatch
    from euclid_dsps.amortized.elbo import negative_elbo
    from euclid_dsps.amortized.latent import LatentSpec
    from euclid_dsps.amortized.train import (
        build_amortized_model,
        component_grad_norms,
        zero_sed_scale_grads,
    )


def _alpha_config(trainable: bool = True):
    return {
        "amortized": {
            "encoder": {"hidden_sizes": [8], "latent_dim": 16},
            "prior": {"n_layers": 2, "hidden_size": 8},
            "likelihood": {"type": "student_t", "student_t_dof": 2.0},
        },
        "calibration": {
            "global_sed_scale": {
                "enabled": True,
                "mode": "learn_global",
                "parameterization": "log_alpha",
                "initial_log_alpha": 0.0,
                "prior_sigma_log_alpha": 0.10,
                "trainable": trainable,
            },
            "per_band_zero_points": {"enabled": False},
        },
    }


def _loss_and_grads(config):
    model = build_amortized_model(config, jax.random.PRNGKey(0))
    batch = PhotometryBatch(
        object_id=jnp.arange(3),
        flux=jnp.ones((3, 10), dtype=jnp.float32) * 1.0e-10,
        flux_err=jnp.ones((3, 10), dtype=jnp.float32) * 1.0e-12,
        mask=jnp.ones((3, 10), dtype=bool),
        features=jnp.ones((3, 20), dtype=jnp.float32),
    )
    spec = LatentSpec(tuple(f"p{i}" for i in range(16)), jnp.zeros(16), jnp.ones(16))
    decoder = {"weights": jnp.zeros((16, 10)), "bias": jnp.ones(10) * -25.0}

    def loss_fn(candidate):
        return negative_elbo(
            candidate,
            batch,
            spec,
            None,
            None,
            spec.names,
            jax.random.PRNGKey(1),
            2,
            1.0,
            {"type": "student_t", "student_t_dof": 2.0},
            {"calibration": config["calibration"]},
            use_mock_decoder=True,
            mock_decoder_params=decoder,
        )

    return model, eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)


def test_log_alpha_sed_receives_gradients_when_enabled() -> None:
    _model, ((_loss, metrics), grads) = _loss_and_grads(_alpha_config(True))

    assert jnp.isfinite(metrics["alpha_prior_penalty"])
    assert component_grad_norms(grads)["alpha_grad_norm"] > 0.0


def test_frozen_alpha_gradients_can_be_zeroed() -> None:
    model, ((_loss, _metrics), grads) = _loss_and_grads(_alpha_config(False))
    zeroed = zero_sed_scale_grads(grads)
    optimizer = optax.adamw(1.0e-2)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    updates, _ = optimizer.update(
        zeroed,
        opt_state,
        eqx.filter(model, eqx.is_inexact_array),
    )
    updated = eqx.apply_updates(model, updates)

    assert component_grad_norms(grads)["alpha_grad_norm"] > 0.0
    assert component_grad_norms(zeroed)["alpha_grad_norm"] == 0.0
    assert jnp.allclose(model.sed_scale.log_alpha_sed, 0.0)
    assert jnp.allclose(updated.sed_scale.log_alpha_sed, model.sed_scale.log_alpha_sed)
