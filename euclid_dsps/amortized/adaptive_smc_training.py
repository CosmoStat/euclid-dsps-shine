"""Self-supervised q distillation and parent-prior updates from adaptive SMC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .adaptive_bridge_smc import (
    AdaptiveBridgeSMCConfig,
    AdaptiveBridgeSMCResult,
    run_adaptive_bridge_smc,
    systematic_resample_to_count,
)
from .config import require_amortized_dependencies
from .posterior import (
    defensive_mixture_log_prob,
    posterior_log_prob,
    posterior_standard_base_to_x,
    posterior_x_to_standard_base,
    sample_posterior,
)
from .posterior_target import posterior_log_target

eqx, optax = require_amortized_dependencies()


@dataclass(frozen=True)
class AdaptiveSMCProposalConfig:
    """Exact initial mixture used by the production bridge."""

    posterior_unit_fraction: float = 0.70
    posterior_tempered_fraction: float = 0.20
    posterior_temperature: float = 1.50
    prior_fraction: float = 0.10

    def components(self) -> tuple[dict[str, float | str], ...]:
        return (
            {
                "source": "posterior",
                "temperature": 1.0,
                "fraction": self.posterior_unit_fraction,
            },
            {
                "source": "posterior",
                "temperature": self.posterior_temperature,
                "fraction": self.posterior_tempered_fraction,
            },
            {
                "source": "prior",
                "temperature": 1.0,
                "fraction": self.prior_fraction,
            },
        )

    def normalized_fractions(self, dtype=jnp.float32) -> jnp.ndarray:
        values = jnp.asarray(
            (
                self.posterior_unit_fraction,
                self.posterior_tempered_fraction,
                self.prior_fraction,
            ),
            dtype=dtype,
        )
        return values / jnp.sum(values)


class SMCPosteriorBatch(NamedTuple):
    """Fixed-K stopped posterior used by q and parent-prior objectives."""

    particles: jnp.ndarray
    normalized_weights: jnp.ndarray
    eligible: jnp.ndarray
    beta_final: jnp.ndarray
    final_ess: jnp.ndarray
    final_max_weight: jnp.ndarray
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
    logZ_estimate: jnp.ndarray
    fallback_attempted: jnp.ndarray
    fallback_succeeded: jnp.ndarray


class OrdinaryImportanceResult(NamedTuple):
    """Ordinary object-posterior IS diagnostics with no selection factors."""

    particles: jnp.ndarray
    normalized_weights: jnp.ndarray
    loglike: jnp.ndarray
    logprior: jnp.ndarray
    logproposal: jnp.ndarray
    logweight: jnp.ndarray
    ess: jnp.ndarray
    max_weight: jnp.ndarray
    logz_estimate: jnp.ndarray
    target_finite: jnp.ndarray
    accepted: jnp.ndarray


class OrdinaryImportanceDiagnostics(NamedTuple):
    normalized_weights: jnp.ndarray
    logweight: jnp.ndarray
    ess: jnp.ndarray
    max_weight: jnp.ndarray
    logz_estimate: jnp.ndarray
    target_finite: jnp.ndarray
    accepted: jnp.ndarray


class DistillationMetrics(NamedTuple):
    loss: jnp.ndarray
    eligible_count: jnp.ndarray
    cross_entropy: jnp.ndarray
    finite_fraction: jnp.ndarray


class PriorMstepTerms(NamedTuple):
    data_nll: jnp.ndarray
    prior_kl_old_new: jnp.ndarray
    eligible_count: jnp.ndarray
    finite_fraction: jnp.ndarray


class PriorDataTerms(NamedTuple):
    data_nll: jnp.ndarray
    eligible_count: jnp.ndarray
    finite_fraction: jnp.ndarray


class OptimizerStepMetrics(NamedTuple):
    loss: jnp.ndarray
    raw_grad_norm: jnp.ndarray
    clipped_grad_norm: jnp.ndarray
    grad_clipped: jnp.ndarray
    grads_finite: jnp.ndarray
    update_applied: jnp.ndarray


class PriorUpdateMetrics(NamedTuple):
    loss: jnp.ndarray
    data_nll: jnp.ndarray
    selection_log_alpha: jnp.ndarray
    selection_alpha: jnp.ndarray
    selection_alpha_mc_relative_error: jnp.ndarray
    selection_score_weight_ess: jnp.ndarray
    selection_maximum_score_weight: jnp.ndarray
    selection_score_weights_finite: jnp.ndarray
    prior_kl_loss_estimate: jnp.ndarray
    prior_kl_proposed: jnp.ndarray
    eligible_count: jnp.ndarray
    raw_grad_norm: jnp.ndarray
    clipped_grad_norm: jnp.ndarray
    grads_finite: jnp.ndarray
    component_diagnostics_evaluated: jnp.ndarray
    data_grad_norm: jnp.ndarray
    selection_grad_norm: jnp.ndarray
    trust_grad_norm: jnp.ndarray
    data_grads_finite: jnp.ndarray
    selection_grads_finite: jnp.ndarray
    trust_grads_finite: jnp.ndarray
    update_applied: jnp.ndarray
    rejection_code: jnp.ndarray


def snapshot_model(model):
    """Freeze all array leaves for one E-step without changing the model type."""
    return jax.tree_util.tree_map(
        lambda value: (
            jax.lax.stop_gradient(value) if eqx.is_inexact_array(value) else value
        ),
        model,
    )


def make_component_optimizer(
    *,
    learning_rate,
    gradient_clip_norm: float,
    weight_decay: float,
):
    """Build one optimizer for exactly one model component."""
    transforms = []
    if float(gradient_clip_norm) > 0.0:
        transforms.append(optax.clip_by_global_norm(float(gradient_clip_norm)))
    transforms.append(
        optax.adamw(
            learning_rate=(
                learning_rate if callable(learning_rate) else float(learning_rate)
            ),
            weight_decay=float(weight_decay),
        )
    )
    return optax.chain(*transforms)


def tree_l2_norm(tree) -> jnp.ndarray:
    leaves = [
        value
        for value in jax.tree_util.tree_leaves(tree)
        if value is not None and eqx.is_inexact_array(value)
    ]
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(value)) for value in leaves))


def tree_all_finite(tree) -> jnp.ndarray:
    leaves = [
        value
        for value in jax.tree_util.tree_leaves(tree)
        if value is not None and eqx.is_inexact_array(value)
    ]
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves]))


def run_model_adaptive_smc_e_step(
    *,
    model_snapshot,
    batch,
    latent_spec,
    context,
    model_args,
    parameter_names,
    likelihood_config,
    calibration_config,
    key: jax.Array,
    smc_config: AdaptiveBridgeSMCConfig,
    proposal_config: AdaptiveSMCProposalConfig | None = None,
    initial_particles: jnp.ndarray | None = None,
    initial_logproposal: jnp.ndarray | None = None,
    initial_logtarget: jnp.ndarray | None = None,
) -> AdaptiveBridgeSMCResult:
    """Run one exact-target bridge from a frozen q/prior snapshot."""
    proposal_cfg = proposal_config or AdaptiveSMCProposalConfig()
    proposal_key, smc_key = jax.random.split(key)
    proposal_x = (
        _sample_exact_initial_mixture(
            model_snapshot,
            proposal_key,
            batch.features,
            int(smc_config.n_particles),
            proposal_cfg,
        )
        if initial_particles is None
        else jax.lax.stop_gradient(jnp.asarray(initial_particles))
    )
    expected_shape = (
        int(smc_config.n_particles),
        int(batch.features.shape[0]),
        int(model_snapshot.prior.latent_dim),
    )
    if proposal_x.shape != expected_shape:
        raise ValueError(
            f"initial bridge particles must have shape {expected_shape}, "
            f"got {proposal_x.shape}"
        )
    fractions = proposal_cfg.normalized_fractions(proposal_x.dtype)

    def log_r0(values):
        component_values = jnp.stack(
            (
                posterior_log_prob(
                    model_snapshot,
                    batch.features,
                    values,
                    base_temperature=1.0,
                ),
                posterior_log_prob(
                    model_snapshot,
                    batch.features,
                    values,
                    base_temperature=proposal_cfg.posterior_temperature,
                ),
                model_snapshot.prior.log_prob(values),
            ),
            axis=0,
        )
        return defensive_mixture_log_prob(component_values, fractions)

    def log_target(values):
        return posterior_log_target(
            model_snapshot,
            values,
            batch,
            latent_spec,
            context,
            model_args,
            parameter_names,
            likelihood_config,
            calibration_config,
        ).logtarget

    def epsilon_to_x(epsilon):
        transformed = posterior_standard_base_to_x(
            model_snapshot,
            batch.features,
            epsilon,
        )
        return transformed.value, transformed.logabsdet_dx_depsilon

    def x_to_epsilon(values):
        transformed = posterior_x_to_standard_base(
            model_snapshot,
            batch.features,
            values,
        )
        return transformed.value, transformed.logabsdet_dx_depsilon

    return run_adaptive_bridge_smc(
        key=smc_key,
        initial_particles=proposal_x,
        log_r0_fn=log_r0,
        log_target_fn=log_target,
        epsilon_to_x_fn=epsilon_to_x,
        x_to_epsilon_fn=x_to_epsilon,
        config=smc_config,
        initial_log_r0=initial_logproposal,
        initial_log_target=initial_logtarget,
    )


def run_model_ordinary_importance(
    *,
    model_snapshot,
    batch,
    latent_spec,
    context,
    model_args,
    parameter_names,
    likelihood_config,
    calibration_config,
    key: jax.Array,
    n_particles: int = 64,
    proposal_config: AdaptiveSMCProposalConfig | None = None,
    minimum_ess_fraction: float = 0.10,
    maximum_weight: float = 0.80,
) -> OrdinaryImportanceResult:
    """Run the defensive ordinary-IS fast path for frozen q and prior snapshots."""
    if int(n_particles) <= 0:
        raise ValueError("ordinary importance sampling requires particles")
    if not 0.0 < float(minimum_ess_fraction) <= 1.0:
        raise ValueError("minimum_ess_fraction must be in (0, 1]")
    if not 0.0 < float(maximum_weight) <= 1.0:
        raise ValueError("maximum_weight must be in (0, 1]")
    proposal_cfg = proposal_config or AdaptiveSMCProposalConfig()
    particles = _sample_exact_initial_mixture(
        model_snapshot,
        key,
        batch.features,
        int(n_particles),
        proposal_cfg,
    )
    fractions = proposal_cfg.normalized_fractions(particles.dtype)
    component_values = jnp.stack(
        (
            posterior_log_prob(
                model_snapshot,
                batch.features,
                particles,
                base_temperature=1.0,
            ),
            posterior_log_prob(
                model_snapshot,
                batch.features,
                particles,
                base_temperature=proposal_cfg.posterior_temperature,
            ),
            model_snapshot.prior.log_prob(particles),
        ),
        axis=0,
    )
    logproposal = defensive_mixture_log_prob(component_values, fractions)
    target = posterior_log_target(
        model_snapshot,
        particles,
        batch,
        latent_spec,
        context,
        model_args,
        parameter_names,
        likelihood_config,
        calibration_config,
    )
    diagnostics = ordinary_importance_diagnostics(
        target.loglike,
        target.logprior,
        logproposal,
        minimum_ess_fraction=minimum_ess_fraction,
        maximum_weight=maximum_weight,
    )
    return OrdinaryImportanceResult(
        particles=jax.lax.stop_gradient(particles),
        normalized_weights=jax.lax.stop_gradient(diagnostics.normalized_weights),
        loglike=jax.lax.stop_gradient(target.loglike),
        logprior=jax.lax.stop_gradient(target.logprior),
        logproposal=jax.lax.stop_gradient(logproposal),
        logweight=jax.lax.stop_gradient(diagnostics.logweight),
        ess=jax.lax.stop_gradient(diagnostics.ess),
        max_weight=jax.lax.stop_gradient(diagnostics.max_weight),
        logz_estimate=jax.lax.stop_gradient(diagnostics.logz_estimate),
        target_finite=jax.lax.stop_gradient(diagnostics.target_finite),
        accepted=jax.lax.stop_gradient(diagnostics.accepted),
    )


def ordinary_importance_diagnostics(
    loglike: jnp.ndarray,
    logprior: jnp.ndarray,
    logproposal: jnp.ndarray,
    *,
    minimum_ess_fraction: float = 0.10,
    maximum_weight: float = 0.80,
) -> OrdinaryImportanceDiagnostics:
    """Return ordinary-IS weights using only likelihood, prior and proposal."""
    loglike = jnp.asarray(loglike)
    logprior = jnp.asarray(logprior, dtype=loglike.dtype)
    logproposal = jnp.asarray(logproposal, dtype=loglike.dtype)
    if loglike.shape != logprior.shape or loglike.shape != logproposal.shape:
        raise ValueError("ordinary importance log-density arrays must share shape")
    if loglike.ndim != 2:
        raise ValueError(
            "ordinary importance arrays must have shape [particles, objects]"
        )
    if not 0.0 < float(minimum_ess_fraction) <= 1.0:
        raise ValueError("minimum_ess_fraction must be in (0, 1]")
    if not 0.0 < float(maximum_weight) <= 1.0:
        raise ValueError("maximum_weight must be in (0, 1]")
    # This is deliberately the complete and only object-level weight contract.
    logweight = loglike + logprior - logproposal
    finite = jnp.isfinite(logweight)
    any_finite = jnp.any(finite, axis=0)
    safe = jnp.where(finite, logweight, -jnp.inf)
    safe = jnp.where(any_finite[None, :], safe, jnp.zeros_like(safe))
    normalized = jax.nn.softmax(safe, axis=0)
    normalized = jnp.where(any_finite[None, :], normalized, 0.0)
    ess = 1.0 / jnp.maximum(
        jnp.sum(jnp.square(normalized), axis=0),
        jnp.asarray(1.0e-30, dtype=normalized.dtype),
    )
    max_weight_value = jnp.max(normalized, axis=0)
    n_particles = int(logweight.shape[0])
    logz = jax.scipy.special.logsumexp(safe, axis=0) - jnp.log(
        jnp.asarray(int(n_particles), dtype=safe.dtype)
    )
    target_finite = any_finite & jnp.all(jnp.isfinite(logproposal), axis=0)
    accepted = target_finite
    accepted &= ess / float(n_particles) >= float(minimum_ess_fraction)
    accepted &= max_weight_value <= float(maximum_weight)
    return OrdinaryImportanceDiagnostics(
        normalized_weights=normalized,
        logweight=logweight,
        ess=ess,
        max_weight=max_weight_value,
        logz_estimate=logz,
        target_finite=target_finite,
        accepted=accepted,
    )


def _sample_exact_initial_mixture(
    model_snapshot,
    key: jax.Array,
    features: jnp.ndarray,
    n_particles: int,
    proposal_config: AdaptiveSMCProposalConfig,
) -> jnp.ndarray:
    """Draw iid particles from the exact nominal r0 mixture.

    Drawing a full candidate bank from every cheap q/prior component avoids
    rounding 64 particles into approximate stratified fractions. The selected
    particles therefore have exactly the nominal marginal density used by the
    bridge and its log-normalizer estimate.
    """
    component_key, unit_key, tempered_key, prior_key = jax.random.split(key, 4)
    count = int(n_particles)
    n_objects = int(features.shape[0])
    unit = sample_posterior(
        model_snapshot,
        unit_key,
        features,
        count,
        base_temperature=1.0,
    ).x
    tempered = sample_posterior(
        model_snapshot,
        tempered_key,
        features,
        count,
        base_temperature=proposal_config.posterior_temperature,
    ).x
    prior = model_snapshot.prior.sample(
        prior_key,
        count * n_objects,
    ).reshape(count, n_objects, -1)
    fractions = proposal_config.normalized_fractions(unit.dtype)
    candidates = jnp.stack((unit, tempered, prior), axis=0)
    selected, _component = _select_exact_mixture_candidates(
        component_key,
        candidates,
        fractions,
    )
    return jax.lax.stop_gradient(selected)


def _select_exact_mixture_candidates(key, candidates, fractions):
    candidates = jnp.asarray(candidates)
    fractions = jnp.asarray(fractions, dtype=candidates.dtype)
    if candidates.ndim < 3:
        raise ValueError("mixture candidates require component, draw and object axes")
    if int(candidates.shape[0]) != int(fractions.shape[0]):
        raise ValueError("candidate components and fractions do not match")
    draw_object_shape = candidates.shape[1:-1]
    component = jax.random.categorical(
        key,
        jnp.log(fractions),
        shape=draw_object_shape,
    )
    selected = jnp.take_along_axis(
        candidates,
        component[None, ..., None],
        axis=0,
    )[0]
    return selected, component


def primary_posterior_batch(result: AdaptiveBridgeSMCResult) -> SMCPosteriorBatch:
    """Convert a primary result to the fixed training contract."""
    hard = jnp.asarray(result.hard_object_flag, dtype=jnp.bool_)
    return SMCPosteriorBatch(
        particles=jax.lax.stop_gradient(result.final_particles),
        normalized_weights=jax.lax.stop_gradient(result.final_normalized_weights),
        eligible=jax.lax.stop_gradient(~hard),
        beta_final=jax.lax.stop_gradient(result.beta_final),
        final_ess=jax.lax.stop_gradient(result.final_ess),
        final_max_weight=jax.lax.stop_gradient(result.final_max_weight),
        mutation_acceptance=jax.lax.stop_gradient(result.mutation_acceptance),
        final_rw_scale=jax.lax.stop_gradient(result.final_rw_scale),
        unique_ancestor_fraction=jax.lax.stop_gradient(result.unique_ancestor_fraction),
        ancestor_ess=jax.lax.stop_gradient(result.ancestor_ess),
        ancestor_ess_fraction=jax.lax.stop_gradient(result.ancestor_ess_fraction),
        epsilon_squared_jump=jax.lax.stop_gradient(result.epsilon_squared_jump),
        median_epsilon_squared_jump=jax.lax.stop_gradient(
            result.median_epsilon_squared_jump
        ),
        moved_particle_fraction=jax.lax.stop_gradient(result.moved_particle_fraction),
        unchanged_from_ancestor_fraction=jax.lax.stop_gradient(
            result.unchanged_from_ancestor_fraction
        ),
        poor_acceptance=jax.lax.stop_gradient(result.poor_acceptance),
        poor_ancestry=jax.lax.stop_gradient(result.poor_ancestry),
        poor_movement=jax.lax.stop_gradient(result.poor_movement),
        mixing_failure=jax.lax.stop_gradient(result.mixing_failure),
        logZ_estimate=jax.lax.stop_gradient(result.logZ_estimate),
        fallback_attempted=jnp.zeros_like(hard),
        fallback_succeeded=jnp.zeros_like(hard),
    )


def merge_hard_fallback(
    *,
    key: jax.Array,
    primary: SMCPosteriorBatch,
    fallback: AdaptiveBridgeSMCResult,
    hard_object_indices: np.ndarray | jnp.ndarray,
) -> SMCPosteriorBatch:
    """Replace only hard primary objects with successful K=128 fallbacks."""
    indices = jnp.asarray(hard_object_indices, dtype=jnp.int32)
    if indices.ndim != 1:
        raise ValueError("hard_object_indices must be one-dimensional")
    if int(indices.size) != int(fallback.final_particles.shape[1]):
        raise ValueError("fallback object count does not match hard_object_indices")
    primary_count = int(primary.particles.shape[0])
    fallback_particles = systematic_resample_to_count(
        key,
        fallback.final_particles,
        fallback.final_normalized_weights,
        primary_count,
    )
    fallback_success = ~fallback.hard_object_flag
    replacement_mask = fallback_success[None, :, None]
    current_particles = jnp.take(primary.particles, indices, axis=1)
    replacement_particles = jnp.where(
        replacement_mask,
        fallback_particles,
        current_particles,
    )
    particles = primary.particles.at[:, indices, :].set(replacement_particles)
    uniform = jnp.full(
        (primary_count, int(indices.size)),
        1.0 / primary_count,
        dtype=primary.normalized_weights.dtype,
    )
    current_weights = jnp.take(primary.normalized_weights, indices, axis=1)
    replacement_weights = jnp.where(fallback_success[None, :], uniform, current_weights)
    weights = primary.normalized_weights.at[:, indices].set(replacement_weights)

    def replace_object_field(current, candidate):
        previous = jnp.take(current, indices, axis=0)
        replacement = jnp.where(fallback_success, candidate, previous)
        return current.at[indices].set(replacement)

    attempted = primary.fallback_attempted.at[indices].set(True)
    succeeded = primary.fallback_succeeded.at[indices].set(fallback_success)
    eligible = primary.eligible.at[indices].set(fallback_success)
    return SMCPosteriorBatch(
        particles=jax.lax.stop_gradient(particles),
        normalized_weights=jax.lax.stop_gradient(weights),
        eligible=jax.lax.stop_gradient(eligible),
        beta_final=jax.lax.stop_gradient(
            replace_object_field(primary.beta_final, fallback.beta_final)
        ),
        final_ess=jax.lax.stop_gradient(
            replace_object_field(
                primary.final_ess,
                jnp.where(fallback_success, float(primary_count), fallback.final_ess),
            )
        ),
        final_max_weight=jax.lax.stop_gradient(
            replace_object_field(
                primary.final_max_weight,
                jnp.where(
                    fallback_success,
                    1.0 / primary_count,
                    fallback.final_max_weight,
                ),
            )
        ),
        mutation_acceptance=jax.lax.stop_gradient(
            replace_object_field(
                primary.mutation_acceptance,
                fallback.mutation_acceptance,
            )
        ),
        final_rw_scale=jax.lax.stop_gradient(
            replace_object_field(primary.final_rw_scale, fallback.final_rw_scale)
        ),
        unique_ancestor_fraction=jax.lax.stop_gradient(
            replace_object_field(
                primary.unique_ancestor_fraction,
                fallback.unique_ancestor_fraction,
            )
        ),
        ancestor_ess=jax.lax.stop_gradient(
            replace_object_field(primary.ancestor_ess, fallback.ancestor_ess)
        ),
        ancestor_ess_fraction=jax.lax.stop_gradient(
            replace_object_field(
                primary.ancestor_ess_fraction,
                fallback.ancestor_ess_fraction,
            )
        ),
        epsilon_squared_jump=jax.lax.stop_gradient(
            replace_object_field(
                primary.epsilon_squared_jump,
                fallback.epsilon_squared_jump,
            )
        ),
        median_epsilon_squared_jump=jax.lax.stop_gradient(
            replace_object_field(
                primary.median_epsilon_squared_jump,
                fallback.median_epsilon_squared_jump,
            )
        ),
        moved_particle_fraction=jax.lax.stop_gradient(
            replace_object_field(
                primary.moved_particle_fraction,
                fallback.moved_particle_fraction,
            )
        ),
        unchanged_from_ancestor_fraction=jax.lax.stop_gradient(
            replace_object_field(
                primary.unchanged_from_ancestor_fraction,
                fallback.unchanged_from_ancestor_fraction,
            )
        ),
        poor_acceptance=jax.lax.stop_gradient(
            replace_object_field(
                primary.poor_acceptance,
                fallback.poor_acceptance,
            )
        ),
        poor_ancestry=jax.lax.stop_gradient(
            replace_object_field(primary.poor_ancestry, fallback.poor_ancestry)
        ),
        poor_movement=jax.lax.stop_gradient(
            replace_object_field(primary.poor_movement, fallback.poor_movement)
        ),
        mixing_failure=jax.lax.stop_gradient(
            replace_object_field(primary.mixing_failure, fallback.mixing_failure)
        ),
        logZ_estimate=jax.lax.stop_gradient(
            replace_object_field(primary.logZ_estimate, fallback.logZ_estimate)
        ),
        fallback_attempted=jax.lax.stop_gradient(attempted),
        fallback_succeeded=jax.lax.stop_gradient(succeeded),
    )


def smc_q_distillation_loss(
    model,
    features: jnp.ndarray,
    posterior: SMCPosteriorBatch,
) -> tuple[jnp.ndarray, DistillationMetrics]:
    """Stopped forward-KL cross entropy that trains q and nothing else."""
    numerator, count, finite_fraction = _smc_q_distillation_sums(
        model,
        features,
        posterior,
    )
    loss = numerator / jnp.maximum(count, 1.0)
    return loss, DistillationMetrics(
        loss=loss,
        eligible_count=count,
        cross_entropy=loss,
        finite_fraction=finite_fraction,
    )


def _smc_q_distillation_sums(model, features, posterior):
    particles = jax.lax.stop_gradient(posterior.particles)
    weights = jax.lax.stop_gradient(posterior.normalized_weights)
    eligible = jax.lax.stop_gradient(posterior.eligible)
    logq = posterior_log_prob(model, features, particles)
    finite = jnp.isfinite(logq)
    object_finite = jnp.all(finite, axis=0)
    usable = eligible & object_finite
    per_object = -jnp.sum(weights * jnp.where(finite, logq, 0.0), axis=0)
    count = jnp.sum(usable.astype(per_object.dtype))
    numerator = jnp.sum(jnp.where(usable, per_object, 0.0))
    return numerator, count, jnp.mean(finite.astype(per_object.dtype))


def apply_q_smc_update(
    *,
    model,
    optimizer,
    optimizer_state,
    features: jnp.ndarray,
    posterior: SMCPosteriorBatch,
    gradient_clip_norm: float,
):
    """Apply one encoder-only inclusive distillation update."""

    def objective(encoder):
        candidate = eqx.tree_at(lambda tree: tree.encoder, model, encoder)
        return smc_q_distillation_loss(candidate, features, posterior)

    (loss, distillation), grads = eqx.filter_value_and_grad(
        objective,
        has_aux=True,
    )(model.encoder)
    raw_norm = tree_l2_norm(grads)
    finite = jnp.isfinite(loss) & tree_all_finite(grads)
    finite &= distillation.eligible_count > 0.0
    safe_grads = jax.tree_util.tree_map(
        lambda value: (
            jnp.where(finite, value, jnp.zeros_like(value))
            if value is not None
            else None
        ),
        grads,
    )
    updates, proposed_state = optimizer.update(
        safe_grads,
        optimizer_state,
        eqx.filter(model.encoder, eqx.is_inexact_array),
    )
    proposed_encoder = eqx.apply_updates(model.encoder, updates)
    encoder = _select_tree(proposed_encoder, model.encoder, finite)
    optimizer_state = _select_tree(proposed_state, optimizer_state, finite)
    model = eqx.tree_at(lambda tree: tree.encoder, model, encoder)
    clipped_norm = jnp.minimum(raw_norm, float(gradient_clip_norm))
    return (
        model,
        optimizer_state,
        distillation,
        OptimizerStepMetrics(
            loss=loss,
            raw_grad_norm=raw_norm,
            clipped_grad_norm=clipped_norm,
            grad_clipped=raw_norm > float(gradient_clip_norm),
            grads_finite=tree_all_finite(grads),
            update_applied=finite,
        ),
    )


def apply_q_sleep_update(
    *,
    model,
    optimizer,
    optimizer_state,
    sleep_loss_fn,
    gradient_clip_norm: float,
):
    """Apply one encoder-only sleep/NPE update from a supplied exact loss."""

    def objective(encoder):
        candidate = eqx.tree_at(lambda tree: tree.encoder, model, encoder)
        return sleep_loss_fn(candidate)

    (loss, sleep_metrics), grads = eqx.filter_value_and_grad(
        objective,
        has_aux=True,
    )(model.encoder)
    raw_norm = tree_l2_norm(grads)
    finite = jnp.isfinite(loss) & tree_all_finite(grads)
    safe_grads = jax.tree_util.tree_map(
        lambda value: (
            jnp.where(finite, value, jnp.zeros_like(value))
            if value is not None
            else None
        ),
        grads,
    )
    updates, proposed_state = optimizer.update(
        safe_grads,
        optimizer_state,
        eqx.filter(model.encoder, eqx.is_inexact_array),
    )
    proposed_encoder = eqx.apply_updates(model.encoder, updates)
    encoder = _select_tree(proposed_encoder, model.encoder, finite)
    optimizer_state = _select_tree(proposed_state, optimizer_state, finite)
    model = eqx.tree_at(lambda tree: tree.encoder, model, encoder)
    clipped_norm = jnp.minimum(raw_norm, float(gradient_clip_norm))
    return (
        model,
        optimizer_state,
        sleep_metrics,
        OptimizerStepMetrics(
            loss=loss,
            raw_grad_norm=raw_norm,
            clipped_grad_norm=clipped_norm,
            grad_clipped=raw_norm > float(gradient_clip_norm),
            grads_finite=tree_all_finite(grads),
            update_applied=finite,
        ),
    )


def smc_prior_mstep_terms(
    current_prior,
    old_prior,
    posterior: SMCPosteriorBatch,
    trust_samples: jnp.ndarray,
) -> PriorMstepTerms:
    """Return stopped posterior data term and old-to-new prior KL estimate."""
    particles = jax.lax.stop_gradient(posterior.particles)
    weights = jax.lax.stop_gradient(posterior.normalized_weights)
    eligible = jax.lax.stop_gradient(posterior.eligible)
    logprior = current_prior.log_prob(particles)
    finite = jnp.isfinite(logprior)
    usable = eligible & jnp.all(finite, axis=0)
    per_object = -jnp.sum(weights * jnp.where(finite, logprior, 0.0), axis=0)
    count = jnp.sum(usable.astype(per_object.dtype))
    data_nll = jnp.sum(jnp.where(usable, per_object, 0.0)) / jnp.maximum(count, 1.0)
    stopped_samples = jax.lax.stop_gradient(trust_samples)
    old_log_prob = jax.lax.stop_gradient(old_prior.log_prob(stopped_samples))
    new_log_prob = current_prior.log_prob(stopped_samples)
    prior_kl = jnp.mean(old_log_prob - new_log_prob)
    return PriorMstepTerms(
        data_nll=data_nll,
        prior_kl_old_new=prior_kl,
        eligible_count=count,
        finite_fraction=jnp.mean(finite.astype(data_nll.dtype)),
    )


def batched_prior_data_mstep_terms(
    current_prior,
    posterior: SMCPosteriorBatch,
    *,
    object_batch_size: int = 128,
) -> PriorDataTerms:
    """Evaluate the exact prior data term with bounded hidden activations."""
    particles = jax.lax.stop_gradient(posterior.particles)
    weights = jax.lax.stop_gradient(posterior.normalized_weights)
    eligible = jax.lax.stop_gradient(posterior.eligible)
    particle_count, object_count, latent_dim = particles.shape
    batch_size = min(int(object_batch_size), int(object_count))
    if batch_size <= 0:
        raise ValueError("object_batch_size and posterior object count must be positive")
    batch_count = (int(object_count) + batch_size - 1) // batch_size
    padded_count = batch_count * batch_size
    padding = padded_count - int(object_count)
    particles = jnp.pad(particles, ((0, 0), (0, padding), (0, 0)))
    weights = jnp.pad(weights, ((0, 0), (0, padding)))
    eligible = jnp.pad(eligible, (0, padding), constant_values=False)
    valid = jnp.arange(padded_count) < int(object_count)
    particle_batches = jnp.swapaxes(
        particles.reshape(particle_count, batch_count, batch_size, latent_dim),
        0,
        1,
    )
    weight_batches = jnp.swapaxes(
        weights.reshape(particle_count, batch_count, batch_size),
        0,
        1,
    )
    eligible_batches = eligible.reshape(batch_count, batch_size)
    valid_batches = valid.reshape(batch_count, batch_size)
    value_dtype = current_prior.log_prob(
        jnp.zeros((1, 1, latent_dim), dtype=particles.dtype)
    ).dtype

    def accumulate(carry, inputs):
        data_sum, usable_count, finite_count, evaluated_count = carry
        particle_batch, weight_batch, eligible_batch, valid_batch = inputs
        logprior = current_prior.log_prob(particle_batch)
        finite = jnp.isfinite(logprior)
        usable = eligible_batch & valid_batch & jnp.all(finite, axis=0)
        per_object = -jnp.sum(
            weight_batch * jnp.where(finite, logprior, 0.0),
            axis=0,
        )
        data_sum += jnp.sum(jnp.where(usable, per_object, 0.0))
        usable_count += jnp.sum(usable.astype(value_dtype))
        finite_count += jnp.sum(
            (finite & valid_batch[None, :]).astype(value_dtype)
        )
        evaluated_count += jnp.asarray(particle_count, dtype=value_dtype) * jnp.sum(
            valid_batch.astype(value_dtype)
        )
        return (data_sum, usable_count, finite_count, evaluated_count), None

    initial = tuple(jnp.asarray(0.0, dtype=value_dtype) for _ in range(4))
    (data_sum, usable_count, finite_count, evaluated_count), _ = jax.lax.scan(
        jax.checkpoint(accumulate),
        initial,
        (particle_batches, weight_batches, eligible_batches, valid_batches),
    )
    return PriorDataTerms(
        data_nll=data_sum / jnp.maximum(usable_count, 1.0),
        eligible_count=usable_count,
        finite_fraction=finite_count / jnp.maximum(evaluated_count, 1.0),
    )


def apply_prior_macro_update(
    *,
    model,
    prior_snapshot,
    optimizer,
    optimizer_state,
    posterior: SMCPosteriorBatch,
    trust_key: jax.Array,
    selection_key: jax.Array,
    selection_log_alpha_fn,
    trust_samples: int,
    trust_strength: float,
    max_kl_per_dimension: float,
    max_alpha_mc_relative_error: float,
    gradient_clip_norm: float,
):
    """Apply one prior-only macro M-step with selection and a hard trust gate.

    ``selection_log_alpha_fn(candidate_model, key)`` must return the
    differentiable ``log(alpha_eta)`` and a metric mapping. Selection never
    enters the object-level particle weights.
    """
    old_samples = jax.lax.stop_gradient(
        prior_snapshot.prior.sample(trust_key, int(trust_samples))
    )

    def data_objective(candidate_prior):
        terms = batched_prior_data_mstep_terms(
            candidate_prior,
            posterior,
        )
        return terms.data_nll, terms

    def selection_objective(candidate_prior):
        candidate_model = eqx.tree_at(
            lambda tree: tree.prior,
            model,
            candidate_prior,
        )
        return selection_log_alpha_fn(candidate_model, selection_key)

    def trust_objective(candidate_prior):
        stopped_samples = jax.lax.stop_gradient(old_samples)
        old_log_prob = jax.lax.stop_gradient(
            prior_snapshot.prior.log_prob(stopped_samples)
        )
        return jnp.mean(old_log_prob - candidate_prior.log_prob(stopped_samples))

    (_data_nll, terms), data_grads = eqx.filter_value_and_grad(
        data_objective,
        has_aux=True,
    )(model.prior)
    jax.block_until_ready((terms, data_grads))
    (log_alpha, selection_metrics), selection_grads = eqx.filter_value_and_grad(
        selection_objective,
        has_aux=True,
    )(model.prior)
    jax.block_until_ready((log_alpha, selection_metrics, selection_grads))
    trust_value, trust_grads = eqx.filter_value_and_grad(trust_objective)(model.prior)
    jax.block_until_ready((trust_value, trust_grads))
    grads = jax.tree_util.tree_map(
        lambda data, selection, trust: (
            data + selection + float(trust_strength) * trust
            if data is not None
            else None
        ),
        data_grads,
        selection_grads,
        trust_grads,
    )
    terms = PriorMstepTerms(
        data_nll=terms.data_nll,
        prior_kl_old_new=trust_value,
        eligible_count=terms.eligible_count,
        finite_fraction=terms.finite_fraction,
    )
    loss = terms.data_nll + log_alpha + float(trust_strength) * trust_value
    raw_norm = tree_l2_norm(grads)
    gradients_finite = tree_all_finite(grads)
    diagnostic_dtype = jnp.asarray(loss).dtype
    component_evaluated = True
    data_grad_norm = tree_l2_norm(data_grads)
    selection_grad_norm = tree_l2_norm(selection_grads)
    trust_grad_norm = tree_l2_norm(trust_grads)
    data_grads_finite = tree_all_finite(data_grads)
    selection_grads_finite = tree_all_finite(selection_grads)
    trust_grads_finite = tree_all_finite(trust_grads)
    alpha_relative_error = jnp.asarray(
        selection_metrics["selection/alpha_mc_relative_error"],
        dtype=loss.dtype,
    )
    pre_update_ok = jnp.isfinite(loss) & gradients_finite
    pre_update_ok &= terms.eligible_count > 0.0
    pre_update_ok &= jnp.isfinite(log_alpha)
    pre_update_ok &= alpha_relative_error <= float(max_alpha_mc_relative_error)
    safe_grads = jax.tree_util.tree_map(
        lambda value: (
            jnp.where(pre_update_ok, value, jnp.zeros_like(value))
            if value is not None
            else None
        ),
        grads,
    )
    updates, proposed_state = optimizer.update(
        safe_grads,
        optimizer_state,
        eqx.filter(model.prior, eqx.is_inexact_array),
    )
    proposed_prior = eqx.apply_updates(model.prior, updates)
    old_log_prob = jax.lax.stop_gradient(prior_snapshot.prior.log_prob(old_samples))
    proposed_kl = jnp.mean(old_log_prob - proposed_prior.log_prob(old_samples))
    dimension = int(model.prior.latent_dim)
    trust_ok = jnp.isfinite(proposed_kl)
    trust_ok &= jnp.abs(proposed_kl) <= float(max_kl_per_dimension) * dimension
    apply_update = pre_update_ok & trust_ok
    prior = _select_tree(proposed_prior, model.prior, apply_update)
    optimizer_state = _select_tree(
        proposed_state,
        optimizer_state,
        apply_update,
    )
    model = eqx.tree_at(lambda tree: tree.prior, model, prior)
    rejection_code = jnp.where(
        ~jnp.isfinite(loss) | ~gradients_finite,
        1,
        jnp.where(
            terms.eligible_count <= 0.0,
            2,
            jnp.where(
                alpha_relative_error > float(max_alpha_mc_relative_error),
                3,
                jnp.where(~trust_ok, 4, 0),
            ),
        ),
    )
    return (
        model,
        optimizer_state,
        PriorUpdateMetrics(
            loss=loss,
            data_nll=terms.data_nll,
            selection_log_alpha=log_alpha,
            selection_alpha=jnp.exp(log_alpha),
            selection_alpha_mc_relative_error=alpha_relative_error,
            selection_score_weight_ess=jnp.asarray(
                selection_metrics.get("selection/score_weight_ess", jnp.nan),
                dtype=diagnostic_dtype,
            ),
            selection_maximum_score_weight=jnp.asarray(
                selection_metrics.get("selection/maximum_score_weight", jnp.nan),
                dtype=diagnostic_dtype,
            ),
            selection_score_weights_finite=jnp.asarray(
                selection_metrics.get("selection/score_weights_finite", 0.0),
                dtype=diagnostic_dtype,
            ),
            prior_kl_loss_estimate=terms.prior_kl_old_new,
            prior_kl_proposed=proposed_kl,
            eligible_count=terms.eligible_count,
            raw_grad_norm=raw_norm,
            clipped_grad_norm=jnp.minimum(raw_norm, float(gradient_clip_norm)),
            grads_finite=gradients_finite,
            component_diagnostics_evaluated=jnp.asarray(component_evaluated),
            data_grad_norm=jnp.asarray(data_grad_norm, dtype=diagnostic_dtype),
            selection_grad_norm=jnp.asarray(
                selection_grad_norm, dtype=diagnostic_dtype
            ),
            trust_grad_norm=jnp.asarray(trust_grad_norm, dtype=diagnostic_dtype),
            data_grads_finite=jnp.asarray(data_grads_finite),
            selection_grads_finite=jnp.asarray(selection_grads_finite),
            trust_grads_finite=jnp.asarray(trust_grads_finite),
            update_applied=apply_update,
            rejection_code=jnp.asarray(rejection_code, dtype=jnp.int32),
        ),
    )


def make_pmap_prior_macro_step(
    *,
    optimizer,
    selection_log_beta_fn,
    total_selection_samples: int,
    total_trust_samples: int,
    trust_strength: float,
    max_kl_per_dimension: float,
    max_alpha_mc_relative_error: float,
    gradient_clip_norm: float,
    n_devices: int,
):
    """Build a data-parallel prior M-step with one global centered score."""
    if int(n_devices) <= 0:
        raise ValueError("prior pmap requires at least one device")
    if int(total_selection_samples) % int(n_devices):
        raise ValueError("selection samples must be divisible by local devices")
    if int(total_trust_samples) % int(n_devices):
        raise ValueError("trust samples must be divisible by local devices")
    local_selection_samples = int(total_selection_samples) // int(n_devices)
    local_trust_samples = int(total_trust_samples) // int(n_devices)
    array_axis = eqx.if_array(0)

    @eqx.filter_pmap(
        axis_name="devices",
        in_axes=(
            array_axis,
            array_axis,
            array_axis,
            array_axis,
            array_axis,
            array_axis,
        ),
        out_axes=(array_axis, array_axis, array_axis),
    )
    def step(
        model,
        prior_snapshot,
        optimizer_state,
        posterior,
        trust_key,
        selection_key,
    ):
        old_samples = jax.lax.stop_gradient(
            prior_snapshot.prior.sample(trust_key, local_trust_samples)
        )

        def data_objective(candidate_prior):
            particles = jax.lax.stop_gradient(posterior.particles)
            weights = jax.lax.stop_gradient(posterior.normalized_weights)
            eligible = jax.lax.stop_gradient(posterior.eligible)
            logprior = candidate_prior.log_prob(particles)
            finite = jnp.isfinite(logprior)
            usable = eligible & jnp.all(finite, axis=0)
            per_object = -jnp.sum(weights * jnp.where(finite, logprior, 0.0), axis=0)
            local_count = jnp.sum(usable.astype(per_object.dtype))
            local_numerator = jnp.sum(jnp.where(usable, per_object, 0.0))
            global_count = jax.lax.psum(local_count, axis_name="devices")
            global_numerator = jax.lax.psum(local_numerator, axis_name="devices")
            finite_fraction = jax.lax.pmean(
                jnp.mean(finite.astype(per_object.dtype)), axis_name="devices"
            )
            return global_numerator / jnp.maximum(global_count, 1.0), (
                global_count,
                finite_fraction,
            )

        def selection_objective(candidate_prior):
            candidate_model = eqx.tree_at(
                lambda tree: tree.prior, model, candidate_prior
            )
            samples = jax.lax.stop_gradient(
                candidate_prior.sample(selection_key, local_selection_samples)
            )
            log_beta = jax.lax.stop_gradient(
                selection_log_beta_fn(candidate_model, samples)
            )
            beta = jnp.where(jnp.isfinite(log_beta), jnp.exp(log_beta), 0.0)
            beta_sum = jax.lax.psum(jnp.sum(beta), axis_name="devices")
            beta_sum_square = jax.lax.psum(
                jnp.sum(jnp.square(beta)), axis_name="devices"
            )
            sample_count = jax.lax.psum(
                jnp.asarray(local_selection_samples, dtype=beta.dtype),
                axis_name="devices",
            )
            safe_sum = jnp.maximum(beta_sum, jnp.asarray(1.0e-30, dtype=beta.dtype))
            normalized = jax.lax.stop_gradient(beta / safe_sum)
            centered = jax.lax.stop_gradient(normalized - 1.0 / sample_count)
            local_score = jnp.sum(centered * candidate_prior.log_prob(samples))
            score = jax.lax.psum(local_score, axis_name="devices")
            log_alpha = jnp.log(safe_sum / sample_count)
            value = jax.lax.stop_gradient(log_alpha - score) + score
            alpha = beta_sum / sample_count
            variance = jnp.maximum(
                (beta_sum_square - beta_sum**2 / sample_count)
                / jnp.maximum(sample_count - 1.0, 1.0),
                0.0,
            )
            mc_error = jnp.sqrt(variance / sample_count)
            score_ess = beta_sum**2 / jnp.maximum(beta_sum_square, 1.0e-30)
            maximum_weight = jax.lax.pmax(jnp.max(normalized), axis_name="devices")
            weights_finite = jax.lax.pmin(
                jnp.all(jnp.isfinite(normalized)).astype(jnp.int32),
                axis_name="devices",
            ).astype(jnp.bool_)
            metrics = {
                "log_alpha": log_alpha,
                "alpha": alpha,
                "alpha_mc_relative_error": mc_error
                / jnp.maximum(alpha, jnp.asarray(1.0e-12, dtype=alpha.dtype)),
                "score_weight_ess": score_ess,
                "maximum_score_weight": maximum_weight,
                "score_weights_finite": weights_finite,
            }
            return value, metrics

        def trust_objective(candidate_prior):
            old_log_prob = jax.lax.stop_gradient(
                prior_snapshot.prior.log_prob(old_samples)
            )
            local_kl = jnp.mean(old_log_prob - candidate_prior.log_prob(old_samples))
            return jax.lax.pmean(local_kl, axis_name="devices")

        (data_nll, data_aux), data_grads = eqx.filter_value_and_grad(
            data_objective, has_aux=True
        )(model.prior)
        (selection_value, selection_metrics), selection_grads = (
            eqx.filter_value_and_grad(selection_objective, has_aux=True)(model.prior)
        )
        trust_value, trust_grads = eqx.filter_value_and_grad(trust_objective)(
            model.prior
        )
        # Reverse-mode differentiation leaves one data-dependent contribution on
        # each replica. Average those contributions so replicated parameters stay
        # synchronized without multiplying the effective learning rate by the
        # device count.
        data_grads = _pmean_tree(data_grads, "devices")
        selection_grads = _pmean_tree(selection_grads, "devices")
        trust_grads = _pmean_tree(trust_grads, "devices")
        grads = jax.tree_util.tree_map(
            lambda data, selection, trust: (
                (data + selection + float(trust_strength) * trust)
                if data is not None
                else None
            ),
            data_grads,
            selection_grads,
            trust_grads,
        )
        loss = data_nll + selection_value + float(trust_strength) * trust_value
        raw_norm = tree_l2_norm(grads)
        gradients_finite = tree_all_finite(grads)
        data_grad_norm = tree_l2_norm(data_grads)
        selection_grad_norm = tree_l2_norm(selection_grads)
        trust_grad_norm = tree_l2_norm(trust_grads)
        alpha_relative_error = selection_metrics["alpha_mc_relative_error"]
        global_count, _finite_fraction = data_aux
        pre_update_ok = jnp.isfinite(loss) & gradients_finite
        pre_update_ok &= global_count > 0.0
        pre_update_ok &= jnp.isfinite(selection_metrics["log_alpha"])
        pre_update_ok &= alpha_relative_error <= float(max_alpha_mc_relative_error)
        pre_update_ok &= selection_metrics["score_weights_finite"]
        safe_grads = jax.tree_util.tree_map(
            lambda value: (
                jnp.where(pre_update_ok, value, jnp.zeros_like(value))
                if value is not None
                else None
            ),
            grads,
        )
        updates, proposed_state = optimizer.update(
            safe_grads,
            optimizer_state,
            eqx.filter(model.prior, eqx.is_inexact_array),
        )
        proposed_prior = eqx.apply_updates(model.prior, updates)
        old_log_prob = jax.lax.stop_gradient(prior_snapshot.prior.log_prob(old_samples))
        local_proposed_kl = jnp.mean(
            old_log_prob - proposed_prior.log_prob(old_samples)
        )
        proposed_kl = jax.lax.pmean(local_proposed_kl, axis_name="devices")
        dimension = int(model.prior.latent_dim)
        trust_ok = jnp.isfinite(proposed_kl)
        trust_ok &= jnp.abs(proposed_kl) <= float(max_kl_per_dimension) * dimension
        apply_update = jax.lax.pmin(
            (pre_update_ok & trust_ok).astype(jnp.int32), axis_name="devices"
        ).astype(jnp.bool_)
        prior = _select_tree(proposed_prior, model.prior, apply_update)
        optimizer_state = _select_tree(proposed_state, optimizer_state, apply_update)
        model = eqx.tree_at(lambda tree: tree.prior, model, prior)
        rejection_code = jnp.where(
            ~jnp.isfinite(loss) | ~gradients_finite,
            1,
            jnp.where(
                global_count <= 0.0,
                2,
                jnp.where(
                    alpha_relative_error > float(max_alpha_mc_relative_error),
                    3,
                    jnp.where(~trust_ok, 4, 0),
                ),
            ),
        )
        metrics = PriorUpdateMetrics(
            loss=loss,
            data_nll=data_nll,
            selection_log_alpha=selection_metrics["log_alpha"],
            selection_alpha=selection_metrics["alpha"],
            selection_alpha_mc_relative_error=alpha_relative_error,
            selection_score_weight_ess=selection_metrics["score_weight_ess"],
            selection_maximum_score_weight=selection_metrics["maximum_score_weight"],
            selection_score_weights_finite=selection_metrics["score_weights_finite"],
            prior_kl_loss_estimate=trust_value,
            prior_kl_proposed=proposed_kl,
            eligible_count=global_count,
            raw_grad_norm=raw_norm,
            clipped_grad_norm=jnp.minimum(raw_norm, float(gradient_clip_norm)),
            grads_finite=gradients_finite,
            component_diagnostics_evaluated=jnp.asarray(True),
            data_grad_norm=data_grad_norm,
            selection_grad_norm=selection_grad_norm,
            trust_grad_norm=trust_grad_norm,
            data_grads_finite=tree_all_finite(data_grads),
            selection_grads_finite=tree_all_finite(selection_grads),
            trust_grads_finite=tree_all_finite(trust_grads),
            update_applied=apply_update,
            rejection_code=jnp.asarray(rejection_code, dtype=jnp.int32),
        )
        return model, optimizer_state, metrics

    return step


def _pmean_tree(tree, axis_name: str):
    return jax.tree_util.tree_map(
        lambda value: (
            jax.lax.pmean(value, axis_name=axis_name) if value is not None else None
        ),
        tree,
    )


def _select_tree(proposed, current, condition):
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(condition, new, old) if eqx.is_array(new) else new,
        proposed,
        current,
    )


def q_only_importance_diagnostics(
    *,
    model_snapshot,
    batch,
    latent_spec,
    context,
    model_args,
    parameter_names,
    likelihood_config,
    calibration_config,
    key: jax.Array,
    n_particles: int,
) -> dict[str, jnp.ndarray]:
    """Measure whether raw q has become a usable target proposal."""
    from .posterior import sample_posterior

    proposal = sample_posterior(
        model_snapshot,
        key,
        batch.features,
        int(n_particles),
    )
    target = posterior_log_target(
        model_snapshot,
        proposal.x,
        batch,
        latent_spec,
        context,
        model_args,
        parameter_names,
        likelihood_config,
        calibration_config,
    )
    finite = jnp.isfinite(target.logtarget) & jnp.isfinite(proposal.logq)
    valid = jnp.any(finite, axis=0)
    safe = jnp.where(finite, target.logtarget - proposal.logq, -jnp.inf)
    safe = jnp.where(valid[None, :], safe, jnp.zeros_like(safe))
    weights = jax.nn.softmax(safe, axis=0)
    ess = 1.0 / jnp.sum(jnp.square(weights), axis=0)
    return {
        "ess_fraction": ess / float(n_particles),
        "max_weight": jnp.max(weights, axis=0),
        "valid": valid,
    }


def make_pmap_e_step(
    *,
    latent_spec,
    context,
    model_args,
    parameter_names,
    likelihood_config,
    calibration_config,
    smc_config: AdaptiveBridgeSMCConfig,
    proposal_config: AdaptiveSMCProposalConfig,
):
    """Build an object-sharded E-step with no cross-device communication."""
    array_axis = eqx.if_array(0)

    @eqx.filter_pmap(
        in_axes=(array_axis, array_axis, array_axis),
        out_axes=array_axis,
    )
    def step(model_snapshot, batch, key):
        frozen_snapshot = snapshot_model(model_snapshot)
        return run_model_adaptive_smc_e_step(
            model_snapshot=frozen_snapshot,
            batch=batch,
            latent_spec=latent_spec,
            context=context,
            model_args=model_args,
            parameter_names=parameter_names,
            likelihood_config=likelihood_config,
            calibration_config=calibration_config,
            key=key,
            smc_config=smc_config,
            proposal_config=proposal_config,
        )

    return step


def make_pmap_ordinary_importance_step(
    *,
    latent_spec,
    context,
    model_args,
    parameter_names,
    likelihood_config,
    calibration_config,
    proposal_config: AdaptiveSMCProposalConfig,
    minimum_ess_fraction: float = 0.10,
    maximum_weight: float = 0.80,
):
    """Build an object-sharded ordinary-IS fast path without collectives."""
    array_axis = eqx.if_array(0)

    @eqx.filter_pmap(
        in_axes=(array_axis, array_axis, array_axis),
        out_axes=array_axis,
    )
    def step(model_snapshot, batch, key):
        return run_model_ordinary_importance(
            model_snapshot=snapshot_model(model_snapshot),
            batch=batch,
            latent_spec=latent_spec,
            context=context,
            model_args=model_args,
            parameter_names=parameter_names,
            likelihood_config=likelihood_config,
            calibration_config=calibration_config,
            key=key,
            n_particles=64,
            proposal_config=proposal_config,
            minimum_ess_fraction=minimum_ess_fraction,
            maximum_weight=maximum_weight,
        )

    return step


def make_pmap_continuation_e_step(
    *,
    latent_spec,
    context,
    model_args,
    parameter_names,
    likelihood_config,
    calibration_config,
    smc_config: AdaptiveBridgeSMCConfig,
    proposal_config: AdaptiveSMCProposalConfig,
):
    """Build a pmap SMC step continuing cached ordinary-IS K64 particles."""
    if int(smc_config.n_particles) != 64:
        raise ValueError("cached ordinary-IS continuation requires K=64")
    array_axis = eqx.if_array(0)

    @eqx.filter_pmap(
        in_axes=(array_axis, array_axis, array_axis, 0, 0, 0),
        out_axes=array_axis,
    )
    def step(
        model_snapshot,
        batch,
        key,
        initial_particles,
        initial_logproposal,
        initial_logtarget,
    ):
        return run_model_adaptive_smc_e_step(
            model_snapshot=snapshot_model(model_snapshot),
            batch=batch,
            latent_spec=latent_spec,
            context=context,
            model_args=model_args,
            parameter_names=parameter_names,
            likelihood_config=likelihood_config,
            calibration_config=calibration_config,
            key=key,
            smc_config=smc_config,
            proposal_config=proposal_config,
            initial_particles=initial_particles,
            initial_logproposal=initial_logproposal,
            initial_logtarget=initial_logtarget,
        )

    return step


def make_pmap_q_smc_step(
    *,
    optimizer,
    gradient_clip_norm: float,
):
    """Build an all-reduced encoder-only SMC distillation update."""
    array_axis = eqx.if_array(0)

    @eqx.filter_pmap(
        axis_name="devices",
        in_axes=(array_axis, array_axis, array_axis, array_axis),
        out_axes=(array_axis, array_axis, array_axis, array_axis),
    )
    def step(model, optimizer_state, features, posterior):
        def local_numerator(encoder):
            candidate = eqx.tree_at(lambda tree: tree.encoder, model, encoder)
            numerator, count, finite_fraction = _smc_q_distillation_sums(
                candidate,
                features,
                posterior,
            )
            return numerator, (count, finite_fraction)

        (numerator, auxiliary), grads = eqx.filter_value_and_grad(
            local_numerator,
            has_aux=True,
        )(model.encoder)
        count, finite_fraction = auxiliary
        global_count = jax.lax.psum(count, axis_name="devices")
        global_numerator = jax.lax.psum(numerator, axis_name="devices")
        grads = jax.tree_util.tree_map(
            lambda value: (
                jax.lax.psum(value, axis_name="devices")
                / jnp.maximum(global_count, 1.0)
                if value is not None
                else None
            ),
            grads,
        )
        loss = global_numerator / jnp.maximum(global_count, 1.0)
        raw_norm = tree_l2_norm(grads)
        gradients_finite = tree_all_finite(grads)
        apply_update = jax.lax.pmin(
            (jnp.isfinite(loss) & gradients_finite & (global_count > 0.0)).astype(
                jnp.int32
            ),
            axis_name="devices",
        ).astype(jnp.bool_)
        safe_grads = jax.tree_util.tree_map(
            lambda value: (
                jnp.where(apply_update, value, jnp.zeros_like(value))
                if value is not None
                else None
            ),
            grads,
        )
        updates, proposed_state = optimizer.update(
            safe_grads,
            optimizer_state,
            eqx.filter(model.encoder, eqx.is_inexact_array),
        )
        proposed_encoder = eqx.apply_updates(model.encoder, updates)
        encoder = _select_tree(proposed_encoder, model.encoder, apply_update)
        optimizer_state = _select_tree(proposed_state, optimizer_state, apply_update)
        model = eqx.tree_at(lambda tree: tree.encoder, model, encoder)
        metrics = DistillationMetrics(
            loss=loss,
            eligible_count=global_count,
            cross_entropy=loss,
            finite_fraction=jax.lax.pmean(finite_fraction, axis_name="devices"),
        )
        step_metrics = OptimizerStepMetrics(
            loss=loss,
            raw_grad_norm=raw_norm,
            clipped_grad_norm=jnp.minimum(raw_norm, float(gradient_clip_norm)),
            grad_clipped=raw_norm > float(gradient_clip_norm),
            grads_finite=gradients_finite,
            update_applied=apply_update,
        )
        return model, optimizer_state, metrics, step_metrics

    return step


def make_pmap_q_sleep_step(
    *,
    optimizer,
    sleep_loss_fn,
    gradient_clip_norm: float,
):
    """Build an all-reduced q-only observed-covariate sleep update."""
    array_axis = eqx.if_array(0)

    @eqx.filter_pmap(
        axis_name="devices",
        in_axes=(array_axis, array_axis, array_axis, array_axis),
        out_axes=(array_axis, array_axis, array_axis, array_axis),
    )
    def step(model, optimizer_state, batch, key):
        def local_numerator(encoder):
            candidate = eqx.tree_at(lambda tree: tree.encoder, model, encoder)
            loss, metrics = sleep_loss_fn(candidate, batch, key)
            selected = jnp.maximum(
                jnp.asarray(
                    metrics["sleep_selection_selected_count"],
                    dtype=loss.dtype,
                ),
                0.0,
            )
            return loss * selected, (loss, metrics, selected)

        (numerator, auxiliary), grads = eqx.filter_value_and_grad(
            local_numerator,
            has_aux=True,
        )(model.encoder)
        _local_loss, sleep_metrics, selected = auxiliary
        global_count = jax.lax.psum(selected, axis_name="devices")
        global_numerator = jax.lax.psum(numerator, axis_name="devices")
        grads = jax.tree_util.tree_map(
            lambda value: (
                jax.lax.psum(value, axis_name="devices")
                / jnp.maximum(global_count, 1.0)
                if value is not None
                else None
            ),
            grads,
        )
        loss = global_numerator / jnp.maximum(global_count, 1.0)
        raw_norm = tree_l2_norm(grads)
        gradients_finite = tree_all_finite(grads)
        apply_update = jax.lax.pmin(
            (jnp.isfinite(loss) & gradients_finite & (global_count > 0.0)).astype(
                jnp.int32
            ),
            axis_name="devices",
        ).astype(jnp.bool_)
        safe_grads = jax.tree_util.tree_map(
            lambda value: (
                jnp.where(apply_update, value, jnp.zeros_like(value))
                if value is not None
                else None
            ),
            grads,
        )
        updates, proposed_state = optimizer.update(
            safe_grads,
            optimizer_state,
            eqx.filter(model.encoder, eqx.is_inexact_array),
        )
        proposed_encoder = eqx.apply_updates(model.encoder, updates)
        encoder = _select_tree(proposed_encoder, model.encoder, apply_update)
        optimizer_state = _select_tree(proposed_state, optimizer_state, apply_update)
        model = eqx.tree_at(lambda tree: tree.encoder, model, encoder)
        sleep_metrics = jax.tree_util.tree_map(
            lambda value: jax.lax.pmean(value, axis_name="devices"),
            sleep_metrics,
        )
        step_metrics = OptimizerStepMetrics(
            loss=loss,
            raw_grad_norm=raw_norm,
            clipped_grad_norm=jnp.minimum(raw_norm, float(gradient_clip_norm)),
            grad_clipped=raw_norm > float(gradient_clip_norm),
            grads_finite=gradients_finite,
            update_applied=apply_update,
        )
        return model, optimizer_state, sleep_metrics, step_metrics

    return step
