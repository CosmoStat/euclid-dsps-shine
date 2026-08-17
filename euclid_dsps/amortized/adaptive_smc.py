"""Adaptive SMC bridge from an amortized proposal to an exact target.

The implementation keeps one annealing temperature per object. Intermediate
resampling and MALA rejuvenation transport joint latent particles towards the
target without pretending that a very large fixed proposal bank has adequate
support.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

LogDensity = Callable[..., jnp.ndarray]


@dataclass(frozen=True)
class AdaptiveSMCKernels:
    """Compiled kernels reusable across equal-shaped object batches."""

    evaluate: Callable
    resample: Callable
    mala_move: Callable


@dataclass(frozen=True)
class AdaptiveSMCResult:
    particles: jnp.ndarray
    weights: jnp.ndarray
    log_weights: jnp.ndarray
    proposal_logdensity: jnp.ndarray
    target_logdensity: jnp.ndarray
    log_evidence: jnp.ndarray
    final_beta: jnp.ndarray
    ancestor_ids: jnp.ndarray
    beta_from: np.ndarray
    beta_to: np.ndarray
    pre_resample_ess: np.ndarray
    resampled: np.ndarray
    mala_acceptance: np.ndarray


def run_adaptive_smc(
    *,
    key: jax.Array,
    initial_particles: jnp.ndarray,
    proposal_logdensity_fn: LogDensity,
    target_logdensity_fn: LogDensity,
    target_ess_fraction: float = 0.5,
    max_stages: int = 64,
    mala_steps: int = 2,
    mala_step_size: float = 0.02,
    density_args: tuple = (),
    kernels: AdaptiveSMCKernels | None = None,
) -> AdaptiveSMCResult:
    """Bridge ``q`` to ``p`` with adaptive temperatures and per-object weights.

    Parameters
    ----------
    initial_particles
        Joint proposal draws shaped ``[K, N, D]``.
    proposal_logdensity_fn, target_logdensity_fn
        Functions returning arrays shaped ``[K, N]``. The target must include
        both the population prior and the photometric likelihood.
    """
    particles = jnp.asarray(initial_particles, dtype=jnp.float32)
    if particles.ndim != 3:
        raise ValueError("initial_particles must have shape [K, N, D]")
    n_particles, n_objects, _latent_dim = particles.shape
    if n_particles < 2 or n_objects < 1:
        raise ValueError("adaptive SMC requires K >= 2 and N >= 1")
    if not 0.0 < float(target_ess_fraction) < 1.0:
        raise ValueError("target_ess_fraction must lie strictly between 0 and 1")
    if int(max_stages) < 1:
        raise ValueError("max_stages must be positive")
    if int(mala_steps) < 0 or float(mala_step_size) <= 0.0:
        raise ValueError("MALA steps must be non-negative and step size positive")

    if kernels is None:
        kernels = build_adaptive_smc_kernels(
            proposal_logdensity_fn=proposal_logdensity_fn,
            target_logdensity_fn=target_logdensity_fn,
            mala_step_size=mala_step_size,
        )
    proposal_logdensity, target_logdensity = kernels.evaluate(particles, *density_args)
    _validate_logdensity_shape(proposal_logdensity, n_particles, n_objects, "q")
    _validate_logdensity_shape(target_logdensity, n_particles, n_objects, "target")
    if not bool(np.all(np.isfinite(np.asarray(proposal_logdensity)))):
        raise ValueError("Initial proposal particles have non-finite log q")
    if bool(np.any(np.all(~np.isfinite(np.asarray(target_logdensity)), axis=0))):
        raise ValueError("At least one object has no finite target particle")

    beta = np.zeros(n_objects, dtype=np.float64)
    log_weights = jnp.zeros((n_particles, n_objects), dtype=particles.dtype)
    log_evidence = jnp.zeros((n_objects,), dtype=particles.dtype)
    ancestor_ids = jnp.broadcast_to(
        jnp.arange(n_particles, dtype=jnp.int32)[:, None],
        (n_particles, n_objects),
    )
    beta_from_history: list[np.ndarray] = []
    beta_to_history: list[np.ndarray] = []
    ess_history: list[np.ndarray] = []
    resampled_history: list[np.ndarray] = []
    acceptance_history: list[np.ndarray] = []

    completed = False
    for _stage in range(int(max_stages)):
        active = beta < 1.0 - 1.0e-7
        if not bool(np.any(active)):
            completed = True
            break
        proposal_logdensity, target_logdensity = kernels.evaluate(
            particles, *density_args
        )
        log_ratio = target_logdensity - proposal_logdensity
        next_beta = _adaptive_next_beta(
            np.asarray(log_weights),
            np.asarray(log_ratio),
            beta,
            target_ess=float(target_ess_fraction) * n_particles,
        )
        delta = jnp.asarray(next_beta - beta, dtype=particles.dtype)
        normalized_previous = jax.nn.log_softmax(log_weights, axis=0)
        increment = jax.scipy.special.logsumexp(
            normalized_previous + delta[None, :] * log_ratio,
            axis=0,
        )
        log_evidence = log_evidence + jnp.where(jnp.asarray(active), increment, 0.0)
        updated_log_weights = log_weights + delta[None, :] * log_ratio
        weights = jax.nn.softmax(updated_log_weights, axis=0)
        ess = 1.0 / jnp.sum(jnp.square(weights), axis=0)
        resample_mask = active & (next_beta < 1.0 - 1.0e-7)

        key, resample_key = jax.random.split(key)
        particles, ancestor_ids = kernels.resample(
            particles,
            ancestor_ids,
            weights,
            jnp.asarray(resample_mask),
            resample_key,
        )
        log_weights = jnp.where(
            jnp.asarray(resample_mask)[None, :],
            jnp.zeros_like(updated_log_weights),
            updated_log_weights,
        )

        stage_acceptance = []
        for _ in range(int(mala_steps)):
            key, proposal_key, accept_key = jax.random.split(key, 3)
            particles, accepted = kernels.mala_move(
                particles,
                jnp.asarray(next_beta, dtype=particles.dtype),
                jnp.asarray(active),
                proposal_key,
                accept_key,
                *density_args,
            )
            stage_acceptance.append(np.asarray(jnp.mean(accepted, axis=0)))

        beta_from_history.append(beta.copy())
        beta_to_history.append(next_beta.copy())
        ess_history.append(np.asarray(ess, dtype=np.float64))
        resampled_history.append(resample_mask.copy())
        acceptance_history.append(
            np.mean(stage_acceptance, axis=0)
            if stage_acceptance
            else np.ones(n_objects, dtype=np.float64)
        )
        beta = next_beta
    if not completed and bool(np.any(beta < 1.0 - 1.0e-7)):
        raise RuntimeError(
            "Adaptive SMC exhausted max_stages before every object reached beta=1"
        )

    proposal_logdensity, target_logdensity = kernels.evaluate(particles, *density_args)
    normalized_log_weights = jax.nn.log_softmax(log_weights, axis=0)
    return AdaptiveSMCResult(
        particles=particles,
        weights=jnp.exp(normalized_log_weights),
        log_weights=normalized_log_weights,
        proposal_logdensity=proposal_logdensity,
        target_logdensity=target_logdensity,
        log_evidence=log_evidence,
        final_beta=jnp.asarray(beta, dtype=particles.dtype),
        ancestor_ids=ancestor_ids,
        beta_from=np.stack(beta_from_history),
        beta_to=np.stack(beta_to_history),
        pre_resample_ess=np.stack(ess_history),
        resampled=np.stack(resampled_history),
        mala_acceptance=np.stack(acceptance_history),
    )


def build_adaptive_smc_kernels(
    *,
    proposal_logdensity_fn: LogDensity,
    target_logdensity_fn: LogDensity,
    mala_step_size: float,
) -> AdaptiveSMCKernels:
    """Build JIT kernels once so batches with the same shapes reuse compilation."""
    if float(mala_step_size) <= 0.0:
        raise ValueError("MALA step size must be positive")
    evaluate = jax.jit(
        lambda values, *args: (
            proposal_logdensity_fn(values, *args),
            target_logdensity_fn(values, *args),
        )
    )
    mala_move = jax.jit(
        lambda values, temperature, active_mask, proposal_key, accept_key, *args: (
            _mala_move(
                values,
                temperature,
                active_mask,
                proposal_logdensity_fn,
                target_logdensity_fn,
                float(mala_step_size),
                proposal_key,
                accept_key,
                args,
            )
        )
    )
    return AdaptiveSMCKernels(
        evaluate=evaluate,
        resample=jax.jit(_resample_selected_objects),
        mala_move=mala_move,
    )


def _adaptive_next_beta(log_weights, log_ratio, beta, *, target_ess):
    log_weights = np.asarray(log_weights, dtype=np.float64)
    log_ratio = np.asarray(log_ratio, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    result = beta.copy()
    for object_index, current in enumerate(beta):
        if current >= 1.0 - 1.0e-7:
            result[object_index] = 1.0
            continue
        ratio = log_ratio[:, object_index]
        weights = log_weights[:, object_index]
        finite = np.isfinite(ratio) & np.isfinite(weights)
        if not np.any(finite):
            raise ValueError(f"Object {object_index} has no finite SMC particle")
        ratio = np.where(finite, ratio, -np.inf)
        weights = np.where(finite, weights, -np.inf)
        if _ess(weights + (1.0 - current) * ratio) >= target_ess:
            result[object_index] = 1.0
            continue
        lower = current
        upper = 1.0
        for _ in range(32):
            middle = 0.5 * (lower + upper)
            candidate = weights + (middle - current) * ratio
            if _ess(candidate) < target_ess:
                upper = middle
            else:
                lower = middle
        result[object_index] = max(lower, current + 1.0e-6)
    return np.minimum(result, 1.0)


def _ess(log_weights: np.ndarray) -> float:
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        return 0.0
    safe = np.where(finite, log_weights, -np.inf)
    maximum = np.max(safe)
    weights = np.exp(safe - maximum)
    weights /= np.sum(weights)
    return float(1.0 / np.sum(np.square(weights)))


def _resample_selected_objects(particles, ancestor_ids, weights, resample_mask, key):
    n_particles, n_objects = weights.shape
    keys = jax.random.split(key, n_objects)
    indices = jax.vmap(
        lambda draw_key, probability: jax.random.choice(
            draw_key,
            n_particles,
            shape=(n_particles,),
            replace=True,
            p=probability,
        )
    )(keys, weights.T)
    gather = jnp.swapaxes(indices, 0, 1)
    proposed_particles = jnp.take_along_axis(particles, gather[..., None], axis=0)
    proposed_ancestors = jnp.take_along_axis(ancestor_ids, gather, axis=0)
    particles = jnp.where(resample_mask[None, :, None], proposed_particles, particles)
    ancestor_ids = jnp.where(resample_mask[None, :], proposed_ancestors, ancestor_ids)
    return particles, ancestor_ids


def _mala_move(
    particles,
    beta,
    active,
    proposal_logdensity_fn,
    target_logdensity_fn,
    step_size,
    proposal_key,
    accept_key,
    density_args,
):
    def tempered(values):
        logq = proposal_logdensity_fn(values, *density_args)
        logtarget = target_logdensity_fn(values, *density_args)
        return logq + beta[None, :] * (logtarget - logq)

    def scalar_target(values):
        values = jnp.nan_to_num(tempered(values), neginf=-1.0e30, posinf=1.0e30)
        return jnp.sum(values)

    gradient = jax.grad(scalar_target)(particles)
    scale = jnp.asarray(step_size, dtype=particles.dtype)
    mean_forward = particles + 0.5 * scale**2 * gradient
    proposal = mean_forward + scale * jax.random.normal(
        proposal_key, particles.shape, dtype=particles.dtype
    )
    reverse_gradient = jax.grad(scalar_target)(proposal)
    mean_reverse = proposal + 0.5 * scale**2 * reverse_gradient
    log_forward = -0.5 * jnp.sum(((proposal - mean_forward) / scale) ** 2, axis=-1)
    log_reverse = -0.5 * jnp.sum(((particles - mean_reverse) / scale) ** 2, axis=-1)
    log_acceptance = (
        tempered(proposal) - tempered(particles) + log_reverse - log_forward
    )
    log_uniform = jnp.log(
        jax.random.uniform(accept_key, log_acceptance.shape, minval=1.0e-7, maxval=1.0)
    )
    accepted = (
        active[None, :]
        & jnp.isfinite(log_acceptance)
        & (log_uniform < jnp.minimum(log_acceptance, 0.0))
    )
    return jnp.where(accepted[..., None], proposal, particles), accepted.astype(
        jnp.float32
    )


def _validate_logdensity_shape(value, n_particles, n_objects, label):
    if tuple(value.shape) != (int(n_particles), int(n_objects)):
        raise ValueError(
            f"{label} log-density must have shape {(n_particles, n_objects)}, "
            f"got {value.shape}"
        )
