"""Guarded local wake updates from fresh exact-density mixture draws.

The finite-sample self-normalized objective is an adaptation heuristic, not a
posterior reference. Nothing in this module updates the frozen physical target.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .local_vi_diagnostic import MixtureParameters, log_prob, sample


def wake_loss(encoder, parameters, context, x, logweights):
    """Weighted joint log-density loss, with samples and weights held fixed."""
    x = jax.lax.stop_gradient(x)
    weights = jax.lax.stop_gradient(jax.nn.softmax(logweights, axis=0))
    return -jnp.mean(
        jnp.sum(weights * log_prob(encoder, parameters, context, x), axis=0)
    )


def make_wake_step(
    encoder,
    *,
    learning_rate=1e-4,
    clip=5.0,
    minimum_ess=16.0,
    maximum_weight=0.2,
):
    """Return an optimizer and one guarded MLE update; no rejected-batch retry.

    ESS and maximum weight are predeclared optimizer preconditions only. A
    rejection preserves Adam moments and count as well as proposal parameters.
    """
    if not (
        np.isfinite(learning_rate)
        and learning_rate > 0
        and np.isfinite(clip)
        and clip > 0
        and np.isfinite(minimum_ess)
        and minimum_ess >= 1
        and np.isfinite(maximum_weight)
        and 0 < maximum_weight <= 1
    ):
        raise ValueError("invalid wake optimizer preconditions")
    optimizer = optax.chain(optax.clip_by_global_norm(clip), optax.adam(learning_rate))

    @eqx.filter_jit
    def step(parameters, state, context, x, logweights):
        if x.ndim != 3 or logweights.shape != x.shape[:-1] or x.shape[0] == 0:
            raise ValueError(
                "wake arrays must have shapes (draws, objects, dim)/(draws, objects)"
            )
        x = jax.lax.stop_gradient(x)
        logweights = jax.lax.stop_gradient(logweights)
        weights = jax.nn.softmax(logweights, axis=0)
        ess = jnp.min(1.0 / jnp.sum(weights**2, axis=0))
        max_weight = jnp.max(weights)
        value = wake_loss(encoder, parameters, context, x, logweights)
        finite = (
            jnp.all(jnp.isfinite(x))
            & jnp.all(jnp.isfinite(logweights))
            & jnp.isfinite(value)
            & jnp.isfinite(ess)
            & jnp.isfinite(max_weight)
        )
        eligible = finite & (ess >= minimum_ess) & (max_weight <= maximum_weight)
        parameter_arrays, parameter_static = eqx.partition(parameters, eqx.is_array)

        def update(_):
            def objective(p):
                return wake_loss(encoder, p, context, x, logweights)

            grads = eqx.filter_grad(objective)(parameters)
            updates, new_state = optimizer.update(grads, state, parameters)
            new_parameters = eqx.apply_updates(parameters, updates)
            arrays = jax.tree_util.tree_leaves(
                eqx.filter((new_parameters, new_state), eqx.is_inexact_array)
            )
            grad_norm, update_norm = (
                optax.global_norm(grads),
                optax.global_norm(updates),
            )
            valid = jnp.isfinite(grad_norm) & jnp.isfinite(update_norm)
            valid &= jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in arrays]))
            accepted_parameters, accepted_state = jax.lax.cond(
                valid,
                lambda _: (eqx.filter(new_parameters, eqx.is_array), new_state),
                lambda _: (parameter_arrays, state),
                operand=None,
            )
            return accepted_parameters, accepted_state, grad_norm, update_norm, valid

        def reject(_):
            norm_dtype = jnp.result_type(
                *[
                    a.dtype
                    for a in jax.tree_util.tree_leaves(
                        eqx.filter(parameters, eqx.is_inexact_array)
                    )
                ]
            )
            zero = jnp.asarray(0.0, dtype=norm_dtype)
            return parameter_arrays, state, zero, zero, finite

        new_parameters, new_state, grad_norm, update_norm, valid = jax.lax.cond(
            eligible, update, reject, operand=None
        )
        return (
            eqx.combine(new_parameters, parameter_static),
            new_state,
            dict(
                wake_loss=value,
                effective_sample_size=ess,
                max_weight=max_weight,
                update_eligible=eligible,
                update_applied=eligible & valid,
                finite=valid,
                raw_grad_norm=grad_norm,
                update_norm=jnp.where(eligible & valid, update_norm, 0.0),
            ),
        )

    return optimizer, step


def wake_batch(
    encoder,
    parameters,
    anchor,
    context,
    observation,
    target,
    budget,
    key,
    draws=256,
):
    """Fresh 50/50 joint-mixture batch with exact weights, decoder forward only."""
    if int(draws) != draws or draws < 1:
        raise ValueError("wake draws must be a positive integer")
    mixture = MixtureParameters(parameters, anchor)
    x, logproposal = sample(encoder, mixture, context, key, int(draws))
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError("wake batches require exactly one diagnostic object")
    explicit = jnp.logaddexp(
        log_prob(encoder, parameters, context, x),
        log_prob(encoder, anchor, context, x),
    ) - np.log(2.0)
    logtargets = []
    for start in range(0, int(draws), 4):
        count = min(4, int(draws) - start)
        budget.charge(count)
        values = target(jax.lax.stop_gradient(x[start : start + count]), observation)
        logtargets.append(np.asarray(jax.device_get(values.logtarget)))
    logtarget = jnp.asarray(np.concatenate(logtargets, axis=0))
    if logtarget.shape != logproposal.shape:
        raise ValueError("wake target and mixture log-density shapes differ")
    logweights = logtarget - logproposal
    finite = bool(
        jnp.all(jnp.isfinite(x))
        & jnp.all(jnp.isfinite(logproposal))
        & jnp.all(jnp.isfinite(logtarget))
        & jnp.all(jnp.isfinite(logweights))
    )
    parity = float(jnp.max(jnp.abs(logproposal - explicit)))
    if finite and (not np.isfinite(parity) or parity > 5e-4):
        raise ValueError("wake mixture density does not match its component densities")
    return (
        jax.lax.stop_gradient(x),
        jax.lax.stop_gradient(logweights),
        dict(
            draws=int(draws),
            finite=finite,
            proposal_log_prob_max_abs_error=parity,
            proposal="0.5 current_local + 0.5 frozen_amortized",
            scientific_promotion=False,
        ),
    )
