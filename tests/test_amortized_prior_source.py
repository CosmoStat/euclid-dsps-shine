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

    from euclid_dsps.amortized.elbo import AmortizedModel
    from euclid_dsps.amortized.flows import RealNVPPrior, StandardNormalPrior
    from euclid_dsps.amortized.train import zero_prior_grads


def _tree_leaves(tree):
    return [leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_inexact_array(leaf)]


def test_standard_normal_prior_baseline_shape_and_logprob() -> None:
    prior = StandardNormalPrior(latent_dim=3)
    x = jnp.zeros((5, 3), dtype=jnp.float32)

    logp = prior.log_prob(x)
    samples = prior.sample(jax.random.PRNGKey(0), 7)

    assert logp.shape == (5,)
    assert samples.shape == (7, 3)
    assert jnp.all(jnp.isfinite(logp))


def test_frozen_prior_gradients_do_not_update_prior() -> None:
    prior = RealNVPPrior(
        jax.random.PRNGKey(0),
        latent_dim=2,
        n_layers=2,
        hidden_size=8,
    )
    model = AmortizedModel(encoder=prior, prior=prior)

    def loss_fn(model):
        x = jnp.ones((4, 2), dtype=jnp.float32) * 0.2
        return -jnp.mean(model.prior.log_prob(x))

    grads = eqx.filter_grad(loss_fn)(model)
    grads = zero_prior_grads(grads)
    optimizer = optax.adamw(1.0e-2)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    updates, _ = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    updated = eqx.apply_updates(model, updates)

    for before, after in zip(_tree_leaves(model.prior), _tree_leaves(updated.prior), strict=True):
        assert jnp.allclose(before, after)


def test_joint_realnvp_prior_gradients_are_nonzero() -> None:
    prior = RealNVPPrior(
        jax.random.PRNGKey(0),
        latent_dim=2,
        n_layers=2,
        hidden_size=8,
    )

    def loss_fn(prior):
        x = jnp.ones((4, 2), dtype=jnp.float32) * 0.2
        return -jnp.mean(prior.log_prob(x))

    grads = eqx.filter_grad(loss_fn)(prior)
    norm = sum(jnp.sum(leaf**2) for leaf in _tree_leaves(grads))

    assert norm > 0.0
