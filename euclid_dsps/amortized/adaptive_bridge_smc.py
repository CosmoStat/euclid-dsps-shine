"""q-preconditioned adaptive bridge SMC for production E-steps.

The initial density is an exact defensive mixture ``r0``. The bridge is

``log gamma_beta = log r0 + beta * (log target - log r0)``.

Random-walk Metropolis mutation is performed in the standard-normal
coordinates of a frozen single-component conditional posterior. No gradient
through SMC particles, weights, resampling, or mutation is intended.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

LogDensity = Callable[[jnp.ndarray], jnp.ndarray]
Transport = Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]


@dataclass(frozen=True)
class AdaptiveBridgeSMCConfig:
    n_particles: int = 64
    target_conditional_ess_fraction: float = 0.75
    resample_ess_fraction: float = 0.50
    max_stages: int = 8
    steps_after_resample: int = 2
    final_steps_at_beta1: int = 1
    rw_scale: float = 0.60
    rw_adapt_target_acceptance: float = 0.30
    rw_adapt_rate: float = 1.0
    rw_scale_min: float = 0.15
    rw_scale_max: float = 1.00
    hard_final_ess_fraction: float = 0.30
    hard_min_mutation_acceptance: float = 0.10
    hard_min_ancestor_ess_fraction: float = 0.05
    hard_min_epsilon_squared_jump: float = 1.0e-4
    bisection_steps: int = 32


class AdaptiveBridgeSMCResult(NamedTuple):
    final_particles: jnp.ndarray
    final_normalized_weights: jnp.ndarray
    final_log_weights: jnp.ndarray
    beta_final: jnp.ndarray
    beta_path: jnp.ndarray
    conditional_ess_path: jnp.ndarray
    ess_path: jnp.ndarray
    resampled_path: jnp.ndarray
    mutation_acceptance_path: jnp.ndarray
    final_ess: jnp.ndarray
    final_max_weight: jnp.ndarray
    number_of_stages: jnp.ndarray
    number_of_resamples: jnp.ndarray
    mutation_acceptance: jnp.ndarray
    final_rw_scale: jnp.ndarray
    unique_ancestor_fraction: jnp.ndarray
    ancestor_ess: jnp.ndarray
    ancestor_ess_fraction: jnp.ndarray
    epsilon_squared_jump: jnp.ndarray
    median_epsilon_squared_jump: jnp.ndarray
    moved_particle_fraction: jnp.ndarray
    unchanged_from_ancestor_fraction: jnp.ndarray
    poor_acceptance: jnp.ndarray
    poor_ancestry: jnp.ndarray
    poor_movement: jnp.ndarray
    mixing_failure: jnp.ndarray
    hard_object_flag: jnp.ndarray
    finite_target_fraction: jnp.ndarray
    logZ_estimate: jnp.ndarray
    ancestor_ids: jnp.ndarray


def run_adaptive_bridge_smc(
    *,
    key: jax.Array,
    initial_particles: jnp.ndarray,
    log_r0_fn: LogDensity,
    log_target_fn: LogDensity,
    epsilon_to_x_fn: Transport,
    x_to_epsilon_fn: Transport,
    config: AdaptiveBridgeSMCConfig | None = None,
    initial_log_r0: jnp.ndarray | None = None,
    initial_log_target: jnp.ndarray | None = None,
) -> AdaptiveBridgeSMCResult:
    """Transport defensive proposal particles to an exact posterior target.

    All arrays use ``[particles, objects, latent]`` ordering. Density
    functions return ``[particles, objects]``. The function is JIT compatible
    and runs a fixed maximum number of stages; objects that do not reach
    ``beta=1`` are returned as hard rather than raising an exception.
    """
    cfg = config or AdaptiveBridgeSMCConfig()
    particles = jnp.asarray(initial_particles)
    _validate_inputs(particles, cfg)
    n_particles, n_objects, _ = particles.shape
    dtype = particles.dtype
    log_uniform = -jnp.log(jnp.asarray(n_particles, dtype=dtype))
    log_weights = jnp.full((n_particles, n_objects), log_uniform, dtype=dtype)
    density_shape = (n_particles, n_objects)
    log_r0 = (
        _finite_logdensity(log_r0_fn(particles))
        if initial_log_r0 is None
        else _finite_logdensity(jnp.asarray(initial_log_r0, dtype=dtype))
    )
    log_target = (
        log_target_fn(particles)
        if initial_log_target is None
        else jnp.asarray(initial_log_target, dtype=dtype)
    )
    if log_r0.shape != density_shape or log_target.shape != density_shape:
        raise ValueError(
            "cached bridge densities must have shape "
            f"{density_shape}, got {log_r0.shape} and {log_target.shape}"
        )
    beta = jnp.zeros((n_objects,), dtype=dtype)
    logz = jnp.zeros((n_objects,), dtype=dtype)
    ancestor_ids = jnp.broadcast_to(
        jnp.arange(n_particles, dtype=jnp.int32)[:, None],
        (n_particles, n_objects),
    )
    beta_path = jnp.zeros((cfg.max_stages + 1, n_objects), dtype=dtype)
    cess_path = jnp.full((cfg.max_stages, n_objects), jnp.nan, dtype=dtype)
    ess_path = jnp.full((cfg.max_stages, n_objects), jnp.nan, dtype=dtype)
    resampled_path = jnp.zeros((cfg.max_stages, n_objects), dtype=jnp.bool_)
    acceptance_path = jnp.full((cfg.max_stages, n_objects), jnp.nan, dtype=dtype)
    stages = jnp.zeros((n_objects,), dtype=jnp.int32)
    resamples = jnp.zeros((n_objects,), dtype=jnp.int32)
    accepted_total = jnp.zeros((n_objects,), dtype=dtype)
    proposed_total = jnp.zeros((n_objects,), dtype=dtype)
    ever_accepted = jnp.zeros((n_particles, n_objects), dtype=jnp.bool_)
    rw_scales = jnp.full((n_objects,), float(cfg.rw_scale), dtype=dtype)

    def stage_body(stage, state):
        (
            current_particles,
            current_log_r0,
            current_log_target,
            current_log_weights,
            current_beta,
            current_logz,
            current_ancestors,
            beta_history,
            cess_history,
            ess_history,
            resample_history,
            acceptance_history,
            stage_counts,
            resample_counts,
            accepted_counts,
            proposed_counts,
            current_rw_scales,
            current_ever_accepted,
        ) = state
        active = current_beta < 1.0 - 1.0e-6
        log_ratio = _finite_logdensity(current_log_target) - current_log_r0
        next_beta, conditional_ess = adaptive_next_beta(
            current_log_weights,
            log_ratio,
            current_beta,
            target_fraction=cfg.target_conditional_ess_fraction,
            bisection_steps=cfg.bisection_steps,
        )
        next_beta = jnp.where(active, next_beta, current_beta)
        delta = next_beta - current_beta
        increment = delta[None, :] * log_ratio
        log_norm = jax.scipy.special.logsumexp(
            current_log_weights + increment,
            axis=0,
        )
        updated_log_weights = current_log_weights + increment - log_norm[None, :]
        updated_log_weights = jnp.where(
            active[None, :], updated_log_weights, current_log_weights
        )
        updated_logz = current_logz + jnp.where(active, log_norm, 0.0)
        weights = jnp.exp(updated_log_weights)
        ess = 1.0 / jnp.sum(jnp.square(weights), axis=0)
        resample_mask = active & (ess < float(cfg.resample_ess_fraction) * n_particles)
        stage_key = jax.random.fold_in(key, stage)
        resample_key, mutation_key = jax.random.split(stage_key)
        (
            moved_particles,
            moved_ancestors,
            moved_log_weights,
            resample_indices,
        ) = _systematic_resample_selected(
            current_particles,
            current_ancestors,
            updated_log_weights,
            resample_mask,
            resample_key,
        )
        proposed_log_r0 = jnp.take_along_axis(
            current_log_r0,
            resample_indices,
            axis=0,
        )
        proposed_log_target = jnp.take_along_axis(
            current_log_target,
            resample_indices,
            axis=0,
        )
        moved_log_r0 = jnp.where(
            resample_mask[None, :], proposed_log_r0, current_log_r0
        )
        moved_log_target = jnp.where(
            resample_mask[None, :], proposed_log_target, current_log_target
        )
        resampled_ever_accepted = jnp.take_along_axis(
            current_ever_accepted,
            resample_indices,
            axis=0,
        )
        moved_ever_accepted = jnp.where(
            resample_mask[None, :],
            resampled_ever_accepted,
            current_ever_accepted,
        )
        stage_rw_scales = current_rw_scales
        stage_accepted = jnp.zeros((n_objects,), dtype=dtype)
        stage_proposed = jnp.zeros((n_objects,), dtype=dtype)
        for move_index in range(int(cfg.steps_after_resample)):
            move_key = jax.random.fold_in(mutation_key, move_index)
            (
                moved_particles,
                accepted,
                moved_log_r0,
                moved_log_target,
            ) = _epsilon_random_walk_mh_cached(
                key=move_key,
                particles=moved_particles,
                current_log_r0=moved_log_r0,
                current_log_target=moved_log_target,
                beta=next_beta,
                object_mask=resample_mask,
                log_r0_fn=log_r0_fn,
                log_target_fn=log_target_fn,
                epsilon_to_x_fn=epsilon_to_x_fn,
                x_to_epsilon_fn=x_to_epsilon_fn,
                rw_scale=stage_rw_scales,
            )
            stage_accepted += jnp.sum(accepted, axis=0)
            stage_proposed += n_particles * resample_mask.astype(dtype)
            moved_ever_accepted |= accepted
        reached_final = active & (next_beta >= 1.0 - 1.0e-6)
        for move_index in range(int(cfg.final_steps_at_beta1)):
            move_key = jax.random.fold_in(
                mutation_key,
                int(cfg.steps_after_resample) + move_index,
            )
            (
                moved_particles,
                accepted,
                moved_log_r0,
                moved_log_target,
            ) = _epsilon_random_walk_mh_cached(
                key=move_key,
                particles=moved_particles,
                current_log_r0=moved_log_r0,
                current_log_target=moved_log_target,
                beta=jnp.ones_like(next_beta),
                object_mask=reached_final,
                log_r0_fn=log_r0_fn,
                log_target_fn=log_target_fn,
                epsilon_to_x_fn=epsilon_to_x_fn,
                x_to_epsilon_fn=x_to_epsilon_fn,
                rw_scale=stage_rw_scales,
            )
            stage_accepted += jnp.sum(accepted, axis=0)
            stage_proposed += n_particles * reached_final.astype(dtype)
            moved_ever_accepted |= accepted
        stage_acceptance = jnp.where(
            stage_proposed > 0,
            stage_accepted / stage_proposed,
            jnp.nan,
        )
        current_rw_scales = _adapt_random_walk_scale(
            current_rw_scales,
            stage_acceptance,
            stage_proposed > 0,
            cfg,
        )
        beta_history = beta_history.at[stage + 1].set(next_beta)
        cess_history = cess_history.at[stage].set(
            jnp.where(active, conditional_ess, jnp.nan)
        )
        ess_history = ess_history.at[stage].set(jnp.where(active, ess, jnp.nan))
        resample_history = resample_history.at[stage].set(resample_mask)
        acceptance_history = acceptance_history.at[stage].set(stage_acceptance)
        return (
            moved_particles,
            moved_log_r0,
            moved_log_target,
            moved_log_weights,
            next_beta,
            updated_logz,
            moved_ancestors,
            beta_history,
            cess_history,
            ess_history,
            resample_history,
            acceptance_history,
            stage_counts + active.astype(jnp.int32),
            resample_counts + resample_mask.astype(jnp.int32),
            accepted_counts + stage_accepted,
            proposed_counts + stage_proposed,
            current_rw_scales,
            moved_ever_accepted,
        )

    state = (
        particles,
        log_r0,
        log_target,
        log_weights,
        beta,
        logz,
        ancestor_ids,
        beta_path,
        cess_path,
        ess_path,
        resampled_path,
        acceptance_path,
        stages,
        resamples,
        accepted_total,
        proposed_total,
        rw_scales,
        ever_accepted,
    )
    state = jax.lax.fori_loop(0, int(cfg.max_stages), stage_body, state)
    (
        particles,
        log_r0,
        log_target,
        log_weights,
        beta,
        logz,
        ancestor_ids,
        beta_path,
        cess_path,
        ess_path,
        resampled_path,
        acceptance_path,
        stages,
        resamples,
        accepted_total,
        proposed_total,
        rw_scales,
        ever_accepted,
    ) = state
    log_weights = jax.nn.log_softmax(log_weights, axis=0)
    weights = jnp.exp(log_weights)
    final_ess = 1.0 / jnp.sum(jnp.square(weights), axis=0)
    final_max_weight = jnp.max(weights, axis=0)
    finite_target_fraction = jnp.mean(jnp.isfinite(log_target), axis=0)
    acceptance = jnp.where(
        proposed_total > 0,
        accepted_total / proposed_total,
        jnp.nan,
    )
    sorted_ancestors = jnp.sort(ancestor_ids, axis=0)
    unique_ancestors = 1 + jnp.sum(
        sorted_ancestors[1:] != sorted_ancestors[:-1],
        axis=0,
    )
    unique_ancestor_fraction = unique_ancestors.astype(dtype) / n_particles
    ancestor_ess, ancestor_ess_fraction = ancestor_ess_from_ids(
        ancestor_ids,
        n_initial_ancestors=n_particles,
        dtype=dtype,
    )
    initial_ancestors = jnp.take_along_axis(
        initial_particles,
        ancestor_ids[..., None],
        axis=0,
    )
    final_epsilon, _final_logdet = x_to_epsilon_fn(particles)
    initial_epsilon, _initial_logdet = x_to_epsilon_fn(initial_ancestors)
    (
        epsilon_squared_jump,
        median_epsilon_squared_jump,
        moved_particle_fraction,
        unchanged_from_ancestor_fraction,
    ) = particle_movement_diagnostics(
        final_epsilon,
        initial_epsilon,
        ever_accepted,
    )
    (
        mixing_failure,
        poor_acceptance,
        poor_ancestry,
        poor_movement,
    ) = mixing_failure_mask(
        mutation_acceptance=acceptance,
        mutation_proposed=proposed_total > 0,
        ancestor_ess_fraction=ancestor_ess_fraction,
        epsilon_squared_jump=median_epsilon_squared_jump,
        min_mutation_acceptance=cfg.hard_min_mutation_acceptance,
        min_ancestor_ess_fraction=cfg.hard_min_ancestor_ess_fraction,
        min_epsilon_squared_jump=cfg.hard_min_epsilon_squared_jump,
    )
    hard = (
        (beta < 1.0 - 1.0e-6)
        | (final_ess / n_particles < float(cfg.hard_final_ess_fraction))
        | (finite_target_fraction <= 0.0)
        | mixing_failure
    )
    return AdaptiveBridgeSMCResult(
        final_particles=jax.lax.stop_gradient(particles),
        final_normalized_weights=jax.lax.stop_gradient(weights),
        final_log_weights=jax.lax.stop_gradient(log_weights),
        beta_final=jax.lax.stop_gradient(beta),
        beta_path=jax.lax.stop_gradient(beta_path),
        conditional_ess_path=jax.lax.stop_gradient(cess_path),
        ess_path=jax.lax.stop_gradient(ess_path),
        resampled_path=jax.lax.stop_gradient(resampled_path),
        mutation_acceptance_path=jax.lax.stop_gradient(acceptance_path),
        final_ess=jax.lax.stop_gradient(final_ess),
        final_max_weight=jax.lax.stop_gradient(final_max_weight),
        number_of_stages=jax.lax.stop_gradient(stages),
        number_of_resamples=jax.lax.stop_gradient(resamples),
        mutation_acceptance=jax.lax.stop_gradient(acceptance),
        final_rw_scale=jax.lax.stop_gradient(rw_scales),
        unique_ancestor_fraction=jax.lax.stop_gradient(unique_ancestor_fraction),
        ancestor_ess=jax.lax.stop_gradient(ancestor_ess),
        ancestor_ess_fraction=jax.lax.stop_gradient(ancestor_ess_fraction),
        epsilon_squared_jump=jax.lax.stop_gradient(epsilon_squared_jump),
        median_epsilon_squared_jump=jax.lax.stop_gradient(median_epsilon_squared_jump),
        moved_particle_fraction=jax.lax.stop_gradient(moved_particle_fraction),
        unchanged_from_ancestor_fraction=jax.lax.stop_gradient(
            unchanged_from_ancestor_fraction
        ),
        poor_acceptance=jax.lax.stop_gradient(poor_acceptance),
        poor_ancestry=jax.lax.stop_gradient(poor_ancestry),
        poor_movement=jax.lax.stop_gradient(poor_movement),
        mixing_failure=jax.lax.stop_gradient(mixing_failure),
        hard_object_flag=jax.lax.stop_gradient(hard),
        finite_target_fraction=jax.lax.stop_gradient(finite_target_fraction),
        logZ_estimate=jax.lax.stop_gradient(logz),
        ancestor_ids=jax.lax.stop_gradient(ancestor_ids),
    )


def particle_movement_diagnostics(
    final_epsilon: jnp.ndarray,
    ancestor_epsilon: jnp.ndarray,
    ever_accepted: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Summarize movement across particles without hiding cloned descendants."""
    final = jnp.asarray(final_epsilon)
    initial = jnp.asarray(ancestor_epsilon, dtype=final.dtype)
    accepted = jnp.asarray(ever_accepted, dtype=jnp.bool_)
    squared_jump = jnp.sum(jnp.square(final - initial), axis=-1)
    return (
        jnp.mean(squared_jump, axis=0),
        jnp.median(squared_jump, axis=0),
        jnp.mean(accepted.astype(final.dtype), axis=0),
        jnp.mean((squared_jump == 0.0).astype(final.dtype), axis=0),
    )


def adaptive_next_beta(
    log_weights: jnp.ndarray,
    log_ratio: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    target_fraction: float,
    bisection_steps: int = 32,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Choose the largest next beta whose conditional ESS meets the target."""
    target = float(target_fraction) * log_weights.shape[0]
    full_cess = conditional_ess(
        log_weights,
        (1.0 - beta)[None, :] * log_ratio,
    )
    lower = beta
    upper = jnp.ones_like(beta)

    def body(_index, bounds):
        low, high = bounds
        middle = 0.5 * (low + high)
        value = conditional_ess(
            log_weights,
            (middle - beta)[None, :] * log_ratio,
        )
        supported = value >= target
        return jnp.where(supported, middle, low), jnp.where(supported, high, middle)

    lower, _upper = jax.lax.fori_loop(
        0,
        int(bisection_steps),
        body,
        (lower, upper),
    )
    next_beta = jnp.where(full_cess >= target, 1.0, lower)
    next_cess = conditional_ess(
        log_weights,
        (next_beta - beta)[None, :] * log_ratio,
    )
    return jnp.minimum(next_beta, 1.0), next_cess


def conditional_ess(log_weights: jnp.ndarray, incremental_logweight: jnp.ndarray):
    """Return standard conditional ESS, equal to K for a zero increment."""
    log_weights = jax.nn.log_softmax(log_weights, axis=0)
    numerator = 2.0 * jax.scipy.special.logsumexp(
        log_weights + incremental_logweight,
        axis=0,
    )
    denominator = jax.scipy.special.logsumexp(
        log_weights + 2.0 * incremental_logweight,
        axis=0,
    )
    return log_weights.shape[0] * jnp.exp(numerator - denominator)


def epsilon_random_walk_mh(
    *,
    key: jax.Array,
    particles: jnp.ndarray,
    beta: jnp.ndarray,
    object_mask: jnp.ndarray,
    log_r0_fn: LogDensity,
    log_target_fn: LogDensity,
    epsilon_to_x_fn: Transport,
    x_to_epsilon_fn: Transport,
    rw_scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply an exact symmetric RW-MH move in frozen-q epsilon coordinates."""
    current_log_r0 = log_r0_fn(particles)
    current_log_target = log_target_fn(particles)
    moved, accepted, _log_r0, _log_target = _epsilon_random_walk_mh_cached(
        key=key,
        particles=particles,
        current_log_r0=current_log_r0,
        current_log_target=current_log_target,
        beta=beta,
        object_mask=object_mask,
        log_r0_fn=log_r0_fn,
        log_target_fn=log_target_fn,
        epsilon_to_x_fn=epsilon_to_x_fn,
        x_to_epsilon_fn=x_to_epsilon_fn,
        rw_scale=rw_scale,
    )
    return moved, accepted


def _epsilon_random_walk_mh_cached(
    *,
    key: jax.Array,
    particles: jnp.ndarray,
    current_log_r0: jnp.ndarray,
    current_log_target: jnp.ndarray,
    beta: jnp.ndarray,
    object_mask: jnp.ndarray,
    log_r0_fn: LogDensity,
    log_target_fn: LogDensity,
    epsilon_to_x_fn: Transport,
    x_to_epsilon_fn: Transport,
    rw_scale: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Cached exact RW-MH move, short-circuited when no object is active."""

    def move(_operand):
        return _epsilon_random_walk_mh_active(
            key=key,
            particles=particles,
            current_log_r0=current_log_r0,
            current_log_target=current_log_target,
            beta=beta,
            object_mask=object_mask,
            log_r0_fn=log_r0_fn,
            log_target_fn=log_target_fn,
            epsilon_to_x_fn=epsilon_to_x_fn,
            x_to_epsilon_fn=x_to_epsilon_fn,
            rw_scale=rw_scale,
        )

    def skip(_operand):
        return (
            particles,
            jnp.zeros(particles.shape[:2], dtype=jnp.bool_),
            current_log_r0,
            current_log_target,
        )

    return jax.lax.cond(jnp.any(object_mask), move, skip, operand=None)


def _epsilon_random_walk_mh_active(
    *,
    key: jax.Array,
    particles: jnp.ndarray,
    current_log_r0: jnp.ndarray,
    current_log_target: jnp.ndarray,
    beta: jnp.ndarray,
    object_mask: jnp.ndarray,
    log_r0_fn: LogDensity,
    log_target_fn: LogDensity,
    epsilon_to_x_fn: Transport,
    x_to_epsilon_fn: Transport,
    rw_scale: float | jnp.ndarray,
):
    proposal_key, accept_key = jax.random.split(key)
    epsilon, current_logdet = x_to_epsilon_fn(particles)
    scale = jnp.asarray(rw_scale, dtype=epsilon.dtype)
    if scale.ndim == 0:
        scale = jnp.broadcast_to(scale, (epsilon.shape[1],))
    proposal_epsilon = epsilon + scale[None, :, None] * jax.random.normal(
        proposal_key,
        epsilon.shape,
        dtype=epsilon.dtype,
    )
    proposal_x, proposal_logdet = epsilon_to_x_fn(proposal_epsilon)
    proposal_log_r0 = log_r0_fn(proposal_x)
    proposal_log_target = log_target_fn(proposal_x)
    current_log_gamma = _bridge_from_values(
        current_log_r0,
        current_log_target,
        beta,
    )
    proposal_log_gamma = _bridge_from_values(
        proposal_log_r0,
        proposal_log_target,
        beta,
    )
    log_acceptance = (
        proposal_log_gamma + proposal_logdet - current_log_gamma - current_logdet
    )
    log_uniform = jnp.log(
        jax.random.uniform(
            accept_key,
            log_acceptance.shape,
            minval=1.0e-7,
            maxval=1.0,
        )
    )
    accepted = (
        object_mask[None, :]
        & jnp.isfinite(log_acceptance)
        & (log_uniform < jnp.minimum(log_acceptance, 0.0))
    )
    return (
        jnp.where(accepted[..., None], proposal_x, particles),
        accepted,
        jnp.where(accepted, proposal_log_r0, current_log_r0),
        jnp.where(accepted, proposal_log_target, current_log_target),
    )


def _adapt_random_walk_scale(
    current_scale: jnp.ndarray,
    acceptance: jnp.ndarray,
    object_mask: jnp.ndarray,
    config: AdaptiveBridgeSMCConfig,
) -> jnp.ndarray:
    """Adapt per-object RW scales once between bridge stages."""
    safe_acceptance = jnp.where(
        object_mask,
        acceptance,
        jnp.asarray(config.rw_adapt_target_acceptance, dtype=acceptance.dtype),
    )
    log_multiplier = float(config.rw_adapt_rate) * (
        safe_acceptance - float(config.rw_adapt_target_acceptance)
    )
    proposed = jnp.clip(
        current_scale * jnp.exp(log_multiplier),
        float(config.rw_scale_min),
        float(config.rw_scale_max),
    )
    return jnp.where(object_mask, proposed, current_scale)


def ancestor_ess_from_ids(
    ancestor_ids: jnp.ndarray,
    *,
    n_initial_ancestors: int | None = None,
    dtype=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return genealogical ESS and its fraction of the initial particle count.

    For descendant frequencies ``p_a = n_a / K``, the diagnostic is
    ``1 / sum_a p_a**2``. Unlike a unique-ancestor count, it detects one
    dominant lineage among several surviving lineages.
    """
    ancestors = jnp.asarray(ancestor_ids, dtype=jnp.int32)
    if ancestors.ndim != 2:
        raise ValueError("ancestor_ids must have [particles, objects] shape")
    particle_count = int(ancestors.shape[0])
    initial_count = (
        particle_count if n_initial_ancestors is None else int(n_initial_ancestors)
    )
    if particle_count <= 0 or initial_count <= 0:
        raise ValueError("ancestor ESS requires positive particle counts")
    result_dtype = jnp.float32 if dtype is None else dtype
    counts = jnp.sum(
        jax.nn.one_hot(ancestors, initial_count, dtype=result_dtype),
        axis=0,
    )
    frequencies = counts / jnp.asarray(particle_count, dtype=result_dtype)
    ancestor_ess = 1.0 / jnp.sum(jnp.square(frequencies), axis=-1)
    return ancestor_ess, ancestor_ess / jnp.asarray(initial_count, dtype=result_dtype)


def mixing_failure_mask(
    *,
    mutation_acceptance: jnp.ndarray,
    mutation_proposed: jnp.ndarray,
    ancestor_ess_fraction: jnp.ndarray,
    epsilon_squared_jump: jnp.ndarray,
    min_mutation_acceptance: float,
    min_ancestor_ess_fraction: float,
    min_epsilon_squared_jump: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Classify mixing without treating low ancestry alone as a failure."""
    acceptance = jnp.asarray(mutation_acceptance)
    proposed = jnp.asarray(mutation_proposed, dtype=jnp.bool_)
    ancestry = jnp.asarray(ancestor_ess_fraction, dtype=acceptance.dtype)
    movement = jnp.asarray(epsilon_squared_jump, dtype=acceptance.dtype)
    poor_acceptance = proposed & (
        ~jnp.isfinite(acceptance) | (acceptance < float(min_mutation_acceptance))
    )
    poor_ancestry = ~jnp.isfinite(ancestry) | (
        ancestry < float(min_ancestor_ess_fraction)
    )
    poor_movement = ~jnp.isfinite(movement) | (
        movement < float(min_epsilon_squared_jump)
    )
    failure = poor_acceptance | (poor_ancestry & poor_movement)
    return failure, poor_acceptance, poor_ancestry, poor_movement


def bridge_logdensity(
    x: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    log_r0_fn: LogDensity,
    log_target_fn: LogDensity,
) -> jnp.ndarray:
    log_r0 = log_r0_fn(x)
    log_target = log_target_fn(x)
    return _bridge_from_values(log_r0, log_target, beta)


def _bridge_from_values(log_r0, log_target, beta):
    beta = beta[None, :]
    interpolated = log_r0 + beta * (_finite_logdensity(log_target) - log_r0)
    return jnp.where(
        beta <= 0.0,
        log_r0,
        jnp.where(jnp.isfinite(log_target), interpolated, -jnp.inf),
    )


def systematic_equal_weight_resample(
    key: jax.Array,
    particles: jnp.ndarray,
    normalized_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Return an equal-weight visualization/distillation draw for every object."""
    mask = jnp.ones((particles.shape[1],), dtype=jnp.bool_)
    ancestors = jnp.broadcast_to(
        jnp.arange(particles.shape[0], dtype=jnp.int32)[:, None],
        normalized_weights.shape,
    )
    resampled, _ancestors, _weights, _indices = _systematic_resample_selected(
        particles,
        ancestors,
        jnp.log(jnp.maximum(normalized_weights, 1.0e-30)),
        mask,
        key,
    )
    return resampled


def systematic_resample_to_count(
    key: jax.Array,
    particles: jnp.ndarray,
    normalized_weights: jnp.ndarray,
    output_count: int,
) -> jnp.ndarray:
    """Systematically resample a weighted posterior to a requested count."""
    count = int(output_count)
    if count <= 0:
        raise ValueError("output_count must be positive")
    particles = jnp.asarray(particles)
    weights = jnp.asarray(normalized_weights)
    if particles.ndim != 3 or weights.shape != particles.shape[:2]:
        raise ValueError("particles and normalized_weights shapes are inconsistent")
    n_objects = particles.shape[1]
    probability = weights / jnp.sum(weights, axis=0, keepdims=True)
    keys = jax.random.split(key, n_objects)

    def indices_one(draw_key, object_probability):
        start = jax.random.uniform(
            draw_key,
            (),
            minval=0.0,
            maxval=1.0 / count,
        )
        positions = start + jnp.arange(count) / count
        cumulative = jnp.cumsum(object_probability).at[-1].set(1.0)
        return jnp.searchsorted(cumulative, positions, side="right")

    indices = jax.vmap(indices_one)(keys, probability.T).T
    return jnp.take_along_axis(particles, indices[..., None], axis=0)


def _systematic_resample_selected(
    particles,
    ancestor_ids,
    log_weights,
    object_mask,
    key,
):
    n_particles, n_objects = log_weights.shape
    weights = jnp.exp(jax.nn.log_softmax(log_weights, axis=0))
    keys = jax.random.split(key, n_objects)

    def indices_one(draw_key, probability):
        start = jax.random.uniform(
            draw_key,
            (),
            minval=0.0,
            maxval=1.0 / n_particles,
        )
        positions = start + jnp.arange(n_particles) / n_particles
        cumulative = jnp.cumsum(probability).at[-1].set(1.0)
        return jnp.searchsorted(cumulative, positions, side="right")

    indices = jax.vmap(indices_one)(keys, weights.T).T
    proposed_particles = jnp.take_along_axis(
        particles,
        indices[..., None],
        axis=0,
    )
    proposed_ancestors = jnp.take_along_axis(ancestor_ids, indices, axis=0)
    particles = jnp.where(object_mask[None, :, None], proposed_particles, particles)
    ancestors = jnp.where(object_mask[None, :], proposed_ancestors, ancestor_ids)
    uniform = jnp.full_like(log_weights, -jnp.log(float(n_particles)))
    log_weights = jnp.where(object_mask[None, :], uniform, log_weights)
    return particles, ancestors, log_weights, indices


def _finite_logdensity(value):
    value = jnp.asarray(value)
    return jnp.where(jnp.isfinite(value), value, jnp.asarray(-1.0e30, value.dtype))


def _validate_inputs(particles, cfg):
    if particles.ndim != 3:
        raise ValueError("initial_particles must have shape [K, N, D]")
    if particles.shape[0] != int(cfg.n_particles):
        raise ValueError("initial particle count does not match config.n_particles")
    if particles.shape[0] < 2 or particles.shape[1] < 1:
        raise ValueError("adaptive bridge SMC requires K>=2 and N>=1")
    for name, value in (
        ("target_conditional_ess_fraction", cfg.target_conditional_ess_fraction),
        ("resample_ess_fraction", cfg.resample_ess_fraction),
        ("hard_final_ess_fraction", cfg.hard_final_ess_fraction),
    ):
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must lie in (0, 1]")
    if int(cfg.max_stages) < 1 or int(cfg.bisection_steps) < 1:
        raise ValueError("max_stages and bisection_steps must be positive")
    if int(cfg.steps_after_resample) < 0 or int(cfg.final_steps_at_beta1) < 0:
        raise ValueError("mutation step counts must be non-negative")
    if float(cfg.rw_scale) <= 0.0:
        raise ValueError("rw_scale must be positive")
    if not 0.0 < float(cfg.rw_adapt_target_acceptance) < 1.0:
        raise ValueError("rw_adapt_target_acceptance must lie in (0, 1)")
    if float(cfg.rw_adapt_rate) < 0.0:
        raise ValueError("rw_adapt_rate must be non-negative")
    if not 0.0 < float(cfg.rw_scale_min) <= float(cfg.rw_scale_max):
        raise ValueError("rw_scale bounds must satisfy 0 < min <= max")
    if not float(cfg.rw_scale_min) <= float(cfg.rw_scale) <= float(cfg.rw_scale_max):
        raise ValueError("rw_scale must lie within the configured bounds")
    if not 0.0 <= float(cfg.hard_min_mutation_acceptance) <= 1.0:
        raise ValueError("hard_min_mutation_acceptance must lie in [0, 1]")
    if not 0.0 <= float(cfg.hard_min_ancestor_ess_fraction) <= 1.0:
        raise ValueError("hard_min_ancestor_ess_fraction must lie in [0, 1]")
    if float(cfg.hard_min_epsilon_squared_jump) < 0.0:
        raise ValueError("hard_min_epsilon_squared_jump must be non-negative")
