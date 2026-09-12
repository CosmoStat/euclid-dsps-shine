"""Controlled encoder-only AVI experiments with a fixed population target.

The defensive proposal r and the learned posterior q are different densities.
Importance weights always use r; fitting and pathwise ELBO always use q.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from .config import require_amortized_dependencies
from .posterior import posterior_log_prob, sample_posterior
from .proposal_expressivity import IndependentFlowMixture, independent_mixture_log_prob

eqx, optax = require_amortized_dependencies()


@dataclass(frozen=True)
class Arm:
    name: str
    scratch: bool = False
    experts: int = 1
    elbo_weight: float = 0.0
    teacher_weight: float = 0.0
    neighbour_sigma: float = 0.0
    kind: str = "encoder"
    prior_initialization: str = "frozen"


ARMS = (
    Arm("A_repair"),
    Arm("B_experts", experts=4),
    Arm("C_scratch", scratch=True),
    Arm("D_scratch_experts", scratch=True, experts=4),
    Arm("E_experts_elbo", experts=4, elbo_weight=0.1),
    Arm("F_experts_teacher", experts=4, teacher_weight=0.1),
    Arm("G_experts_neighbours", experts=4, teacher_weight=0.1, neighbour_sigma=0.2),
)


def experts(candidate):
    return (
        candidate.experts
        if isinstance(candidate, IndependentFlowMixture)
        else (candidate,)
    )


def with_encoder(model, encoder):
    return eqx.tree_at(lambda m: m.encoder, model, encoder)


def log_prob(model, candidate, features, x):
    if isinstance(candidate, IndependentFlowMixture):
        return independent_mixture_log_prob(model, candidate, features, x)
    return posterior_log_prob(with_encoder(model, candidate), features, x)


def stratified_proposal(model, candidate, features, key, particles=128):
    """Fixed-count balance MIS, including a 1/8 frozen-prior component.

    Sample only K total particles, not K per expert. The exact denominator uses
    actual component counts, not the learned gate, and is identical for all draws.
    """
    es = experts(candidate)
    n_prior = particles // 8
    if particles < 8 or (particles - n_prior) % len(es):
        raise ValueError("particle budget must divide across expert components")
    per_expert = (particles - n_prior) // len(es)
    keys = jax.random.split(key, len(es) + 1)
    samples = [
        sample_posterior(with_encoder(model, e), k, features, per_expert).x
        for e, k in zip(es, keys[:-1], strict=True)
    ]
    b = features.shape[0]
    prior = model.prior.sample(keys[-1], n_prior * b).reshape(n_prior, b, -1)
    x = jnp.concatenate([*samples, prior], axis=0).astype(jnp.float64)
    terms = [
        posterior_log_prob(with_encoder(model, e), features, x)
        + jnp.log(per_expert / particles)
        for e in es
    ]
    terms.append(model.prior.log_prob(x) + jnp.log(n_prior / particles))
    logr = jax.scipy.special.logsumexp(jnp.stack(terms), axis=0)
    return jax.lax.stop_gradient(x), jax.lax.stop_gradient(logr)


def mixture_component_log_probs(model, candidate, features, x):
    """Return ``log pi_j(y) + log q_j(x|y)`` for every full-flow expert."""
    es = experts(candidate)
    component = jnp.stack(
        [posterior_log_prob(with_encoder(model, expert), features, x) for expert in es],
        axis=0,
    )
    if len(es) == 1:
        return component
    log_gate = jax.nn.log_softmax(candidate.logits(features), axis=-1)
    log_gate = jnp.moveaxis(log_gate, -1, 0)
    while log_gate.ndim < component.ndim:
        log_gate = jnp.expand_dims(log_gate, axis=1)
    return component + log_gate


def expert_responsibilities(model, candidate, features, x):
    """Exact categorical responsibilities for complete joint 15D draws."""
    terms = mixture_component_log_probs(model, candidate, features, x)
    return jax.nn.softmax(terms, axis=0)


def normalized_weights(logw):
    finite = jnp.isfinite(logw)
    valid = jnp.any(finite, axis=0) & ~jnp.any(
        jnp.isnan(logw) | jnp.isposinf(logw), axis=0
    )
    safe = jnp.where(finite, logw, -jnp.inf)
    safe = jnp.where(valid[None], safe, 0.0)
    weights = jax.nn.softmax(safe, axis=0)
    weights = jnp.where(valid[None], weights, 0.0)
    ess = jnp.where(valid, 1 / jnp.maximum(jnp.sum(weights**2, axis=0), 1e-300), 0)
    return weights, valid, ess


def weighted_nll(logq, weights, valid):
    # Zero-weight out-of-support entries must not produce 0 * -inf.
    term = jnp.where(weights > 0, logq, 0.0)
    per_object = -jnp.sum(weights * term, axis=0)
    return jnp.sum(jnp.where(valid, per_object, 0.0)) / jnp.maximum(jnp.sum(valid), 1)


def enumerated_elbo(model, candidate, features, key, target, draws=2):
    """Differentiate each continuous expert AND the categorical probabilities."""
    es = experts(candidate)
    keys = jax.random.split(key, len(es))
    probabilities = (
        jax.nn.softmax(candidate.logits(features), axis=-1).T
        if len(es) > 1
        else jnp.ones((1, features.shape[0]))
    )
    losses = []
    for expert, k in zip(es, keys, strict=True):
        x = sample_posterior(with_encoder(model, expert), k, features, draws).x
        # Rematerialize decoder activations during backward to bound GPU memory.
        lt = jax.checkpoint(target)(x)
        losses.append(jnp.mean(log_prob(model, candidate, features, x) - lt, axis=0))
    return jnp.mean(jnp.sum(probabilities * jnp.stack(losses), axis=0))


def tree_finite(tree):
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))


def make_parallel_steps(loss_fn, optimizer, *, devices):
    """Accumulate microbatch gradients, then average over actual GPU replicas.

    loss_fn(candidate, payload, key) returns loss and a fixed-size metric vector.
    Payload has [device, accumulation, local_object, ...] leading dimensions.
    """

    @partial(eqx.filter_pmap, axis_name="replica", devices=devices)
    def step(candidate, state, payload, key):
        keys = jax.random.split(key, jax.tree_util.tree_leaves(payload)[0].shape[0])
        grad0 = jax.tree_util.tree_map(
            jnp.zeros_like, eqx.filter(candidate, eqx.is_inexact_array)
        )

        def micro(carry, inputs):
            grads, loss_sum, metrics_sum = carry
            item, k = inputs
            (loss, metrics), grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                candidate, item, k
            )
            grads = jax.tree_util.tree_map(lambda a, b: a + b, grads, grad)
            return (grads, loss_sum + loss, metrics_sum + metrics), None

        (grads, loss, metrics), _ = jax.lax.scan(
            micro,
            (grad0, jnp.array(0.0, jnp.float64), jnp.zeros(4, jnp.float64)),
            (payload, keys),
        )
        n = keys.shape[0]
        grads = jax.lax.pmean(jax.tree_util.tree_map(lambda x: x / n, grads), "replica")
        loss, metrics = jax.lax.pmean((loss / n, metrics / n), "replica")
        updates, proposed_state = optimizer.update(
            grads, state, eqx.filter(candidate, eqx.is_inexact_array)
        )
        proposed = eqx.apply_updates(candidate, updates)
        finite = tree_finite(proposed) & tree_finite(grads) & jnp.isfinite(loss)
        # Reject nonfinite work consistently across all replicas, including moments.
        candidate = jax.tree_util.tree_map(
            lambda a, b: jnp.where(finite, a, b) if eqx.is_array(a) else a,
            proposed,
            candidate,
        )
        state = jax.tree_util.tree_map(
            lambda a, b: jnp.where(finite, a, b), proposed_state, state
        )
        return candidate, state, loss, metrics, finite

    return step
