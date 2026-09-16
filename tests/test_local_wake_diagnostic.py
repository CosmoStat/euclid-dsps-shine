from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.amortized.local_vi_diagnostic import (
    Budget,
    LocalParameters,
    log_prob,
    sample,
)
from euclid_dsps.amortized.local_wake_diagnostic import (
    make_wake_step,
    wake_batch,
    wake_loss,
)
from euclid_dsps.amortized.posterior import ConditionalFlowEncoder
from euclid_dsps.amortized.posterior_target import PosteriorTargetValues


def fixture(*, layers=0):
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(1),
        input_dim=6,
        latent_dim=2,
        hidden_sizes=(8,),
        activation="gelu",
        log_std_min=-4,
        log_std_max=2,
        initial_log_std=0,
        n_layers=layers,
        hidden_size=8,
        output_space="latent_x",
    )
    parameters = LocalParameters(
        jnp.zeros((1, 2), jnp.float32),
        jnp.zeros((1, 2), jnp.float32),
        encoder.layers,
    )
    return encoder, parameters, jnp.zeros((1, 4), jnp.float32)


def assert_tree_equal(left, right):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for a, b in zip(
        jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True
    ):
        np.testing.assert_array_equal(a, b)


def test_wake_batch_full_mixture_density_and_forward_only_budget():
    encoder, anchor, context = fixture(layers=2)
    local = eqx.tree_at(lambda p: p.mean, anchor, anchor.mean + 0.8)
    calls = []

    def target(x, observation):
        calls.append(x.shape[0])
        logtarget = log_prob(encoder, anchor, context, x)
        return PosteriorTargetValues(
            logtarget,
            logtarget,
            jnp.zeros_like(logtarget),
            jnp.ones(logtarget.shape, bool),
            x,
            x,
        )

    budget = Budget(60, 2048)
    x, logweights, info = wake_batch(
        encoder,
        local,
        anchor,
        context,
        None,
        target,
        budget,
        jax.random.PRNGKey(7),
        draws=2048,
    )
    lp = log_prob(encoder, anchor, context, x)
    lq = log_prob(encoder, local, context, x)
    expected = lp - (jnp.logaddexp(lp, lq) - np.log(2))
    np.testing.assert_allclose(logweights, expected, atol=1e-6)
    ratio = np.exp(np.asarray(logweights))
    assert ratio.max() <= 2.00001
    assert abs(ratio.mean() - 1) < 0.06
    assert sum(calls) == budget.forward == 2048
    assert max(calls) == 4 and budget.gradient == 0
    assert info["finite"] and info["proposal_log_prob_max_abs_error"] < 1e-6
    native_x, _ = sample(encoder, anchor, context, jax.random.PRNGKey(8), 1)
    assert x.dtype == native_x.dtype


def test_wake_loss_detaches_draws_and_weights_and_ignores_normalizer():
    encoder, parameters, context = fixture()
    x, _ = sample(encoder, parameters, context, jax.random.PRNGKey(2), 64)
    logweights = -((x[..., 0] - 1.0) ** 2)
    value = wake_loss(encoder, parameters, context, x, logweights)
    np.testing.assert_allclose(
        value,
        wake_loss(encoder, parameters, context, x, logweights + 500),
        rtol=1e-5,
    )
    gx, gw = jax.grad(
        lambda a, b: wake_loss(encoder, parameters, context, a, b), argnums=(0, 1)
    )(x, logweights)
    np.testing.assert_array_equal(gx, jnp.zeros_like(x))
    np.testing.assert_array_equal(gw, jnp.zeros_like(logweights))
    grads = eqx.filter_grad(lambda p: wake_loss(encoder, p, context, x, logweights))(
        parameters
    )
    assert np.linalg.norm(np.asarray(grads.mean)) > 0.1


def test_wake_gaussian_optimizer_recovers_dense_weighted_mle():
    encoder, initial, context = fixture()
    x, _ = sample(encoder, initial, context, jax.random.PRNGKey(3), 2048)
    logweights = -0.5 * jnp.sum(((x - jnp.array([0.5, -0.3])) / 0.7) ** 2, axis=-1)
    weights = np.asarray(jax.nn.softmax(logweights, axis=0))
    expected_mean = np.sum(weights[..., None] * np.asarray(x), axis=0)
    expected_std = np.sqrt(
        np.sum(weights[..., None] * (np.asarray(x) - expected_mean) ** 2, axis=0)
    )
    optimizer, step = make_wake_step(encoder, learning_rate=0.03)
    state = optimizer.init(eqx.filter(initial, eqx.is_inexact_array))
    parameters = initial
    for _ in range(200):
        parameters, state, info = step(parameters, state, context, x, logweights)
        assert bool(info["update_applied"]) and bool(info["finite"])
    np.testing.assert_allclose(parameters.mean, expected_mean, atol=2e-3)
    np.testing.assert_allclose(jnp.exp(parameters.log_std), expected_std, atol=2e-3)
    assert parameters.mean.dtype == initial.mean.dtype


@pytest.mark.parametrize("failure", ["ess", "max_weight", "nan", "inf_x"])
def test_rejected_batch_preserves_parameters_and_nonzero_adam_state(failure):
    encoder, initial, context = fixture(layers=2)
    x, _ = sample(encoder, initial, context, jax.random.PRNGKey(4), 64)
    optimizer, step = make_wake_step(encoder)
    state = optimizer.init(eqx.filter(initial, eqx.is_inexact_array))
    parameters = initial
    for _ in range(2):
        parameters, state, info = step(
            parameters, state, context, x + 0.5, jnp.zeros((64, 1))
        )
        assert bool(info["update_applied"])
    weights = jnp.zeros((64, 1))
    if failure == "ess":
        weights = weights.at[0].set(1000)
    elif failure == "max_weight":
        # ESS above 16 is possible with a single weight just above 0.2.
        weights = jnp.log(jnp.full((64, 1), 0.795 / 63).at[0].set(0.205))
    elif failure == "nan":
        weights = weights.at[0].set(jnp.nan)
    else:
        x = x.at[0, 0, 0].set(jnp.inf)
    before_parameters, before_state = parameters, state
    parameters, state, info = step(parameters, state, context, x, weights)
    assert not bool(info["update_applied"])
    assert bool(info["finite"]) == (failure in {"ess", "max_weight"})
    assert_tree_equal(parameters, before_parameters)
    assert_tree_equal(state, before_state)


def test_nonfinite_target_draw_is_preserved_not_dropped():
    encoder, parameters, context = fixture()

    def target(x, observation):
        value = jnp.zeros(x.shape[:-1]).at[0].set(jnp.nan)
        return PosteriorTargetValues(value, value, value, jnp.isfinite(value), x, x)

    x, weights, info = wake_batch(
        encoder,
        parameters,
        parameters,
        context,
        None,
        target,
        Budget(60, 17),
        jax.random.PRNGKey(5),
        draws=17,
    )
    assert len(x) == len(weights) == 17 and not info["finite"]
    assert np.count_nonzero(~np.isfinite(weights)) == 5


def test_wake_preconditions_and_shapes_are_explicit():
    encoder, parameters, context = fixture()
    with pytest.raises(ValueError, match="preconditions"):
        make_wake_step(encoder, maximum_weight=0)
    optimizer, step = make_wake_step(encoder)
    state = optimizer.init(eqx.filter(parameters, eqx.is_inexact_array))
    with pytest.raises(ValueError, match="shapes"):
        step(parameters, state, context, jnp.zeros((8, 1, 2)), jnp.zeros(8))
