"""Opt-in finite-batch descent control for the local wake diagnostic."""

import equinox as eqx
import jax
import jax.numpy as jnp

from .local_vi_diagnostic import log_prob
from .local_wake_diagnostic import make_wake_step, wake_loss


def make_guarded_wake_step(encoder, *, armijo=1e-4, trials=12, **kwargs):
    if not 0 < armijo < 1 or not isinstance(trials, int) or trials < 1:
        raise ValueError("invalid backtracking settings")
    optimizer, proposal_step = make_wake_step(encoder, **kwargs)

    @eqx.filter_jit
    def step(parameters, state, context, x, logweights):
        proposed, proposed_state, metrics = proposal_step(
            parameters, state, context, x, logweights
        )
        before = metrics["wake_loss"]
        arrays, static = eqx.partition(parameters, eqx.is_inexact_array)
        delta = jax.tree.map(
            lambda a, b: b - a, arrays, eqx.filter(proposed, eqx.is_inexact_array)
        )

        def objective(p):
            return wake_loss(encoder, eqx.combine(p, static), context, x, logweights)

        grad = eqx.filter_grad(objective)(arrays)
        slope = sum(
            jnp.sum(g * d)
            for g, d in zip(jax.tree.leaves(grad), jax.tree.leaves(delta), strict=True)
        )
        eligible = metrics["update_applied"] & jnp.isfinite(slope) & (slope < 0)

        def search(i, carry):
            accepted, chosen, loss, scale, count = carry

            def evaluate(_):
                alpha = jnp.asarray(0.5, dtype=before.dtype) ** i
                candidate = jax.tree.map(lambda p, d: p + alpha * d, arrays, delta)
                value = objective(candidate)
                valid = (
                    jnp.isfinite(value)
                    & (value < before)
                    & (value <= before + armijo * alpha * slope)
                )
                selected = jax.tree.map(
                    lambda a, b: jnp.where(valid, a, b), candidate, chosen
                )
                return (
                    valid,
                    selected,
                    jnp.where(valid, value, loss),
                    jnp.where(valid, alpha, scale),
                    count + 1,
                )

            return jax.lax.cond(
                eligible & ~accepted, evaluate, lambda _: carry, operand=None
            )

        accepted, selected, after, scale, count = jax.lax.fori_loop(
            0,
            trials,
            search,
            (
                jnp.asarray(False),
                arrays,
                before,
                jnp.zeros_like(before),
                jnp.asarray(0),
            ),
        )
        new_state = jax.lax.cond(
            accepted, lambda _: proposed_state, lambda _: state, operand=None
        )
        result = eqx.combine(selected, static)
        shift = log_prob(encoder, result, context, x) - log_prob(
            encoder, parameters, context, x
        )
        metrics.update(
            update_applied=accepted,
            update_norm=metrics["update_norm"] * scale,
            wake_loss_after=after,
            accepted_scale=scale,
            line_search_evaluations=count,
            directional_derivative=slope,
            proposal_loss=wake_loss(encoder, proposed, context, x, logweights),
            batch_log_density_change_rms=jnp.sqrt(jnp.mean(shift**2)),
        )
        return result, new_state, metrics

    return optimizer, step
