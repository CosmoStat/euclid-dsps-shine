"""Numerical primitives for Selection-Corrected Defensive RWS.

The module deliberately contains no catalogue truth access and no SMC kernel.
Object-level weights always target ``p(y | x) p_eta(x)``. Survey selection is
handled only by the separate population-prior normalization term.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .posterior import (
    defensive_mixture_log_prob,
    defensive_posterior_proposal,
    posterior_entropy_diagnostics,
    posterior_log_prob,
)
from .posterior_target import posterior_log_target

C0_SCOPE_STATEMENT = (
    "We infer the parent distribution within the predefined FENIKS "
    "refinement and catalogue-support domain, while explicitly correcting "
    "the additional observed r<27.5 selection."
)


@dataclass(frozen=True)
class SCDrwsSchedule:
    warmup_epochs: int = 60
    joint_epochs: int = 120
    sleep_epochs_per_cycle: int = 3
    wake_epochs_per_cycle: int = 1
    log_std_floor_start: float = -1.5
    log_std_floor_end: float = -4.0
    log_std_floor_end_epoch: int = 100
    flow_scale_clamp_start: float = 0.15
    flow_scale_clamp_end_epoch: int = 60
    q_weight_temperature_start: float = 0.50
    q_weight_temperature_end: float = 1.00
    q_weight_temperature_wake_updates: int = 80

    @property
    def total_epochs(self) -> int:
        return int(self.warmup_epochs) + int(self.joint_epochs)


@dataclass(frozen=True)
class DefensiveMixture:
    components: tuple[tuple[str, float, float], ...]

    def as_config(self) -> tuple[dict[str, float | str], ...]:
        return tuple(
            {"source": source, "temperature": temperature, "fraction": fraction}
            for source, temperature, fraction in self.components
        )

    def normalized(self) -> DefensiveMixture:
        total = sum(float(item[2]) for item in self.components)
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("defensive-mixture fractions must have a finite sum")
        return DefensiveMixture(
            tuple(
                (str(source), float(temperature), float(fraction) / total)
                for source, temperature, fraction in self.components
            )
        )

    def realized(self, n_particles: int) -> DefensiveMixture:
        """Return the deterministic-mixture fractions actually drawn at fixed K."""
        normalized = self.normalized()
        count = int(n_particles)
        if count <= 0:
            raise ValueError("n_particles must be positive")
        requested = np.asarray(
            [component[2] for component in normalized.components], dtype=float
        )
        raw = requested * count
        allocated = np.floor(raw).astype(np.int64)
        missing = count - int(np.sum(allocated))
        if missing:
            order = np.argsort(-(raw - allocated), kind="stable")
            allocated[order[:missing]] += 1
        return DefensiveMixture(
            tuple(
                (source, temperature, float(component_count) / count)
                for (source, temperature, _fraction), component_count in zip(
                    normalized.components, allocated, strict=True
                )
                if component_count > 0
            )
        )


WARMUP_PROPOSAL = DefensiveMixture(
    (
        ("posterior", 1.0, 0.55),
        ("posterior", 1.5, 0.30),
        ("posterior", 2.5, 0.10),
        ("prior", 1.0, 0.05),
    )
)
JOINT_PROPOSAL = DefensiveMixture(
    (
        ("posterior", 1.0, 0.50),
        ("posterior", 1.5, 0.30),
        ("posterior", 2.5, 0.15),
        ("prior", 1.0, 0.05),
    )
)
HARD_EXPANSION_PROPOSAL = DefensiveMixture(
    (
        ("posterior", 1.5, 0.45),
        ("posterior", 2.5, 0.45),
        ("prior", 1.0, 0.10),
    )
)


class ImportanceDiagnostics(NamedTuple):
    normalized_weights: jnp.ndarray
    logweight: jnp.ndarray
    ess: jnp.ndarray
    ess_fraction: jnp.ndarray
    max_weight: jnp.ndarray
    finite: jnp.ndarray
    hard: jnp.ndarray


class DefensiveImportanceBatch(NamedTuple):
    particles: jnp.ndarray
    logproposal: jnp.ndarray
    diagnostics: ImportanceDiagnostics


class AdaptiveMISBatch(NamedTuple):
    first: DefensiveImportanceBatch
    expanded_particles: jnp.ndarray
    expanded_logproposal: jnp.ndarray
    expanded_diagnostics: ImportanceDiagnostics
    expanded_object: jnp.ndarray
    unresolved: jnp.ndarray


@dataclass(frozen=True)
class PriorSupportGate:
    accepted: bool
    reason: str
    finite_objects: int
    median_ess_fraction: float
    median_max_weight: float
    unresolved_fraction: float


class SCDrwsStepMetrics(NamedTuple):
    loss: jnp.ndarray
    raw_grad_norm: jnp.ndarray
    clipped_grad_norm: jnp.ndarray
    grad_clipped: jnp.ndarray
    grads_finite: jnp.ndarray
    update_applied: jnp.ndarray


def _linear_schedule(
    step: int | float | jnp.ndarray,
    *,
    start: float,
    end: float,
    first_step: int,
    last_step: int,
):
    if int(last_step) <= int(first_step):
        raise ValueError("schedule last_step must exceed first_step")
    value = jnp.asarray(step, dtype=jnp.float32)
    fraction = (value - float(first_step)) / float(last_step - first_step)
    fraction = jnp.clip(fraction, 0.0, 1.0)
    return jnp.asarray(start, dtype=value.dtype) + fraction * float(end - start)


def phase_for_epoch(epoch: int, schedule: SCDrwsSchedule) -> str:
    if not 1 <= int(epoch) <= schedule.total_epochs:
        raise ValueError("epoch is outside the SC-DRWS schedule")
    return "robust_warmup" if int(epoch) <= schedule.warmup_epochs else "joint_gaussian"


def update_kind_for_epoch(epoch: int, schedule: SCDrwsSchedule) -> str:
    cycle = int(schedule.sleep_epochs_per_cycle) + int(schedule.wake_epochs_per_cycle)
    if cycle <= 0 or int(schedule.wake_epochs_per_cycle) != 1:
        raise ValueError("SC-DRWS currently requires one wake epoch per cycle")
    return "wake" if int(epoch) % cycle == 0 else "sleep"


def log_std_floor(epoch: int | jnp.ndarray, schedule: SCDrwsSchedule):
    return _linear_schedule(
        epoch,
        start=schedule.log_std_floor_start,
        end=schedule.log_std_floor_end,
        first_step=1,
        last_step=schedule.log_std_floor_end_epoch,
    )


def flow_scale_clamp(
    epoch: int | jnp.ndarray,
    *,
    final_value: float,
    schedule: SCDrwsSchedule,
):
    return _linear_schedule(
        epoch,
        start=schedule.flow_scale_clamp_start,
        end=float(final_value),
        first_step=1,
        last_step=schedule.flow_scale_clamp_end_epoch,
    )


def q_weight_temperature(
    wake_update: int | jnp.ndarray,
    schedule: SCDrwsSchedule,
):
    return _linear_schedule(
        wake_update,
        start=schedule.q_weight_temperature_start,
        end=schedule.q_weight_temperature_end,
        first_step=1,
        last_step=schedule.q_weight_temperature_wake_updates,
    )


def entropy_penalty_factor(joint_epoch: int, schedule: SCDrwsSchedule) -> float:
    """Return a factor that is exactly zero for the final 25% of Phase B."""
    if not 1 <= int(joint_epoch) <= int(schedule.joint_epochs):
        raise ValueError("joint_epoch is outside Phase B")
    progress = (int(joint_epoch) - 1) / max(int(schedule.joint_epochs) - 1, 1)
    if progress <= 0.50:
        return 1.0
    if progress >= 0.75:
        return 0.0
    return float((0.75 - progress) / 0.25)


def deterministic_multiple_mixture(
    first: DefensiveMixture,
    additional: DefensiveMixture,
    *,
    first_count: int,
    additional_count: int,
) -> DefensiveMixture:
    """Combine deterministic component counts into one complete MIS density."""
    total = int(first_count) + int(additional_count)
    if min(int(first_count), int(additional_count), total) <= 0:
        raise ValueError("deterministic MIS budgets must be positive")
    merged: dict[tuple[str, float], float] = {}
    for budget, proposal in (
        (int(first_count), first.normalized()),
        (int(additional_count), additional.normalized()),
    ):
        for source, temperature, fraction in proposal.components:
            key = (source, float(temperature) if source == "posterior" else 1.0)
            merged[key] = merged.get(key, 0.0) + budget / total * float(fraction)
    return DefensiveMixture(
        tuple(
            (source, temperature, fraction)
            for (source, temperature), fraction in sorted(merged.items())
        )
    ).normalized()


def complete_mixture_log_prob(
    model: Any,
    features: jnp.ndarray,
    particles: jnp.ndarray,
    mixture: DefensiveMixture,
    *,
    log_std_floor_value=None,
    flow_scale_clamp_value=None,
) -> jnp.ndarray:
    values = []
    normalized = mixture.normalized()
    for source, temperature, _fraction in normalized.components:
        if source == "posterior":
            value = posterior_log_prob(
                model,
                features,
                particles,
                base_temperature=float(temperature),
                log_std_floor=log_std_floor_value,
                flow_scale_clamp=flow_scale_clamp_value,
            )
        elif source == "prior":
            value = model.prior.log_prob(particles)
        else:
            raise ValueError(f"unknown proposal source {source}")
        values.append(value)
    fractions = jnp.asarray(
        [item[2] for item in normalized.components], dtype=particles.dtype
    )
    return defensive_mixture_log_prob(jnp.stack(values, axis=0), fractions)


def importance_diagnostics(
    logtarget: jnp.ndarray,
    logproposal: jnp.ndarray,
    *,
    minimum_ess_fraction: float = 0.05,
    maximum_weight: float = 0.90,
) -> ImportanceDiagnostics:
    """Normalize object weights without any survey-selection factor."""
    logweight = jnp.asarray(logtarget) - jnp.asarray(logproposal)
    finite_particle = jnp.isfinite(logweight)
    finite_object = jnp.any(finite_particle, axis=0)
    safe = jnp.where(finite_particle, logweight, -jnp.inf)
    safe = jnp.where(finite_object[None, ...], safe, jnp.zeros_like(safe))
    weights = jax.nn.softmax(safe, axis=0)
    ess = 1.0 / jnp.sum(jnp.square(weights), axis=0)
    ess_fraction = ess / float(logweight.shape[0])
    max_weight_value = jnp.max(weights, axis=0)
    hard = (~finite_object) | (ess_fraction < float(minimum_ess_fraction))
    hard |= max_weight_value > float(maximum_weight)
    return ImportanceDiagnostics(
        normalized_weights=jax.lax.stop_gradient(weights),
        logweight=jax.lax.stop_gradient(logweight),
        ess=jax.lax.stop_gradient(ess),
        ess_fraction=jax.lax.stop_gradient(ess_fraction),
        max_weight=jax.lax.stop_gradient(max_weight_value),
        finite=jax.lax.stop_gradient(finite_object),
        hard=jax.lax.stop_gradient(hard),
    )


def run_defensive_importance(
    *,
    model_snapshot: Any,
    features: jnp.ndarray,
    key: jax.Array,
    n_particles: int,
    proposal: DefensiveMixture,
    logtarget_fn: Callable[[jnp.ndarray], jnp.ndarray],
    log_std_floor_value=None,
    flow_scale_clamp_value=None,
    minimum_ess_fraction: float = 0.05,
    maximum_weight: float = 0.90,
) -> DefensiveImportanceBatch:
    draw = defensive_posterior_proposal(
        model_snapshot,
        key,
        features,
        int(n_particles),
        proposal.as_config(),
        antithetic=True,
        log_std_floor=log_std_floor_value,
        flow_scale_clamp=flow_scale_clamp_value,
    )
    target = logtarget_fn(draw.x)
    diagnostics = importance_diagnostics(
        target,
        draw.logproposal,
        minimum_ess_fraction=minimum_ess_fraction,
        maximum_weight=maximum_weight,
    )
    return DefensiveImportanceBatch(
        particles=jax.lax.stop_gradient(draw.x),
        logproposal=jax.lax.stop_gradient(draw.logproposal),
        diagnostics=diagnostics,
    )


def expand_defensive_importance(
    *,
    model_snapshot: Any,
    features: jnp.ndarray,
    key: jax.Array,
    first: DefensiveImportanceBatch,
    first_proposal: DefensiveMixture,
    additional_proposal: DefensiveMixture,
    additional_particles: int,
    logtarget_fn: Callable[[jnp.ndarray], jnp.ndarray],
    log_std_floor_value=None,
    flow_scale_clamp_value=None,
    minimum_ess_fraction: float = 0.05,
    maximum_weight: float = 0.90,
) -> AdaptiveMISBatch:
    """Expand a batch already restricted to hard objects to K<=512."""
    first_count = int(first.particles.shape[0])
    extra_count = int(additional_particles)
    if first_count + extra_count > 512:
        raise ValueError("SC-DRWS hard-object budget cannot exceed K=512")
    additional = defensive_posterior_proposal(
        model_snapshot,
        key,
        features,
        extra_count,
        additional_proposal.as_config(),
        antithetic=True,
        log_std_floor=log_std_floor_value,
        flow_scale_clamp=flow_scale_clamp_value,
    )
    particles = jnp.concatenate((first.particles, additional.x), axis=0)
    complete = deterministic_multiple_mixture(
        first_proposal.realized(first_count),
        additional_proposal.realized(extra_count),
        first_count=first_count,
        additional_count=extra_count,
    )
    logproposal = complete_mixture_log_prob(
        model_snapshot,
        features,
        particles,
        complete,
        log_std_floor_value=log_std_floor_value,
        flow_scale_clamp_value=flow_scale_clamp_value,
    )
    diagnostics = importance_diagnostics(
        logtarget_fn(particles),
        logproposal,
        minimum_ess_fraction=minimum_ess_fraction,
        maximum_weight=maximum_weight,
    )
    expanded = jnp.ones_like(diagnostics.hard, dtype=jnp.bool_)
    return AdaptiveMISBatch(
        first=first,
        expanded_particles=jax.lax.stop_gradient(particles),
        expanded_logproposal=jax.lax.stop_gradient(logproposal),
        expanded_diagnostics=diagnostics,
        expanded_object=expanded,
        unresolved=jax.lax.stop_gradient(diagnostics.hard),
    )


def tempered_q_weights(
    exact_normalized_weights: jnp.ndarray,
    tau: float | jnp.ndarray,
) -> jnp.ndarray:
    weights = jax.lax.stop_gradient(jnp.asarray(exact_normalized_weights))
    temperature = jnp.asarray(tau, dtype=weights.dtype)
    if weights.ndim < 1:
        raise ValueError("importance weights require a particle axis")
    log_weight = jnp.where(weights > 0.0, jnp.log(weights), -jnp.inf)
    return jax.nn.softmax(temperature * log_weight, axis=0)


def q_wake_loss(
    model: Any,
    features: jnp.ndarray,
    particles: jnp.ndarray,
    exact_normalized_weights: jnp.ndarray,
    object_mask: jnp.ndarray,
    *,
    tau: float | jnp.ndarray,
    log_std_floor_value=None,
    flow_scale_clamp_value=None,
) -> jnp.ndarray:
    weights = tempered_q_weights(exact_normalized_weights, tau)
    logq = posterior_log_prob(
        model,
        features,
        jax.lax.stop_gradient(particles),
        log_std_floor=log_std_floor_value,
        flow_scale_clamp=flow_scale_clamp_value,
    )
    finite = jnp.all(jnp.isfinite(logq), axis=0)
    usable = jax.lax.stop_gradient(jnp.asarray(object_mask, dtype=bool)) & finite
    per_object = -jnp.sum(weights * jnp.where(jnp.isfinite(logq), logq, 0.0), axis=0)
    count = jnp.maximum(jnp.sum(usable.astype(per_object.dtype)), 1.0)
    return jnp.sum(jnp.where(usable, per_object, 0.0)) / count


def entropy_floor_loss(
    model: Any,
    features: jnp.ndarray,
    key: jax.Array,
    *,
    reference_entropy: float | jnp.ndarray,
    margin: float,
    strength: float,
    factor: float | jnp.ndarray,
    log_std_floor_value=None,
    flow_scale_clamp_value=None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    diagnostics = posterior_entropy_diagnostics(
        model,
        features,
        key,
        n_samples=2,
        log_std_floor=log_std_floor_value,
        flow_scale_clamp=flow_scale_clamp_value,
    )
    entropy = diagnostics["posterior_full_entropy_mc"]
    threshold = jnp.asarray(reference_entropy, dtype=entropy.dtype) - float(margin)
    deficit = jnp.maximum(threshold - entropy, 0.0)
    penalty = float(strength) * jnp.asarray(factor, dtype=entropy.dtype) * deficit**2
    return penalty, diagnostics


def prior_data_loss(
    prior: Any,
    particles: jnp.ndarray,
    exact_normalized_weights: jnp.ndarray,
    eligible: jnp.ndarray,
) -> jnp.ndarray:
    logprior = prior.log_prob(jax.lax.stop_gradient(particles))
    finite = jnp.all(jnp.isfinite(logprior), axis=0)
    usable = jax.lax.stop_gradient(jnp.asarray(eligible, dtype=bool)) & finite
    weights = jax.lax.stop_gradient(exact_normalized_weights)
    per_object = -jnp.sum(
        weights * jnp.where(jnp.isfinite(logprior), logprior, 0.0), axis=0
    )
    count = jnp.maximum(jnp.sum(usable.astype(per_object.dtype)), 1.0)
    return jnp.sum(jnp.where(usable, per_object, 0.0)) / count


def prior_support_gate(
    *,
    ess_fraction: Sequence[float] | np.ndarray,
    max_weight: Sequence[float] | np.ndarray,
    finite: Sequence[bool] | np.ndarray,
    unresolved: Sequence[bool] | np.ndarray,
    minimum_finite_objects: int,
    minimum_median_ess_fraction: float,
    maximum_median_weight: float,
    maximum_unresolved_fraction: float,
) -> PriorSupportGate:
    finite_values = np.asarray(finite, dtype=bool)
    unresolved_values = np.asarray(unresolved, dtype=bool)
    usable = finite_values & ~unresolved_values
    count = int(np.sum(usable))
    ess = np.asarray(ess_fraction, dtype=float)
    maximum = np.asarray(max_weight, dtype=float)
    median_ess = float(np.nanmedian(ess[usable])) if count else float("nan")
    median_weight = float(np.nanmedian(maximum[usable])) if count else float("nan")
    unresolved_fraction = float(np.mean(unresolved_values)) if len(unresolved_values) else 1.0
    checks = (
        (count >= int(minimum_finite_objects), "too_few_finite_objects"),
        (
            np.isfinite(median_ess)
            and median_ess >= float(minimum_median_ess_fraction),
            "median_ess_below_gate",
        ),
        (
            np.isfinite(median_weight)
            and median_weight <= float(maximum_median_weight),
            "median_max_weight_above_gate",
        ),
        (
            unresolved_fraction <= float(maximum_unresolved_fraction),
            "unresolved_fraction_above_gate",
        ),
    )
    reason = next((name for passed, name in checks if not passed), "accepted")
    return PriorSupportGate(
        accepted=reason == "accepted",
        reason=reason,
        finite_objects=count,
        median_ess_fraction=median_ess,
        median_max_weight=median_weight,
        unresolved_fraction=unresolved_fraction,
    )


def parent_to_selected_weights(beta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Derive selected-population weights from one parent sample set."""
    values = jnp.where(jnp.isfinite(beta) & (beta >= 0.0), beta, 0.0)
    alpha = jnp.mean(values)
    normalized = values / jnp.maximum(jnp.sum(values), jnp.asarray(1.0e-30))
    return normalized, alpha


def contains_selection_in_object_weights(source: str) -> bool:
    """Static regression helper used by config/receipt validation."""
    normalized = "".join(str(source).lower().split())
    forbidden = ("+log_beta", "+logbeta", "*beta", "+log_alpha", "+logalpha")
    return any(token in normalized for token in forbidden)


def _tree_l2_norm(tree) -> jnp.ndarray:
    leaves = [
        value
        for value in jax.tree_util.tree_leaves(tree)
        if value is not None and hasattr(value, "dtype")
    ]
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(value)) for value in leaves))


def _tree_all_finite(tree) -> jnp.ndarray:
    leaves = [
        value
        for value in jax.tree_util.tree_leaves(tree)
        if value is not None and hasattr(value, "dtype")
    ]
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves]))


def _pmean_tree(tree, axis_name: str):
    return jax.tree_util.tree_map(
        lambda value: (
            jax.lax.pmean(value, axis_name=axis_name)
            if value is not None and hasattr(value, "dtype")
            else value
        ),
        tree,
    )


def _select_tree(proposed, current, condition):
    return jax.tree_util.tree_map(
        lambda new, old: (
            jnp.where(condition, new, old)
            if hasattr(new, "dtype")
            else new
        ),
        proposed,
        current,
    )


def make_pmap_sc_drws_importance_step(
    *,
    latent_spec,
    context,
    model_args,
    parameter_names,
    likelihood_config,
    calibration_config,
    n_particles: int,
    proposal: DefensiveMixture,
    minimum_ess_fraction: float,
    maximum_weight: float,
):
    """Build an object-sharded ordinary-IW step with a fixed K shape."""
    from .config import require_equinox

    eqx = require_equinox()
    array_axis = eqx.if_array(0)

    @eqx.filter_pmap(
        axis_name="devices",
        in_axes=(array_axis, array_axis, array_axis, None, None),
        out_axes=array_axis,
    )
    def step(model_snapshot, batch, key, floor, clamp):
        frozen = jax.tree_util.tree_map(
            lambda value: (
                jax.lax.stop_gradient(value)
                if hasattr(value, "dtype")
                else value
            ),
            model_snapshot,
        )

        def logtarget(values):
            return posterior_log_target(
                frozen,
                values,
                batch,
                latent_spec,
                context,
                model_args,
                parameter_names,
                likelihood_config,
                calibration_config,
            ).logtarget

        return run_defensive_importance(
            model_snapshot=frozen,
            features=batch.features,
            key=key,
            n_particles=int(n_particles),
            proposal=proposal,
            logtarget_fn=logtarget,
            log_std_floor_value=floor,
            flow_scale_clamp_value=clamp,
            minimum_ess_fraction=minimum_ess_fraction,
            maximum_weight=maximum_weight,
        )

    return step


def make_pmap_sc_drws_expansion_step(
    *,
    latent_spec,
    context,
    model_args,
    parameter_names,
    likelihood_config,
    calibration_config,
    first_proposal: DefensiveMixture,
    additional_proposal: DefensiveMixture,
    additional_particles: int,
    minimum_ess_fraction: float,
    maximum_weight: float,
):
    """Build the hard-only K128-to-K512 deterministic-MIS step."""
    from .config import require_equinox

    eqx = require_equinox()
    array_axis = eqx.if_array(0)

    @eqx.filter_pmap(
        axis_name="devices",
        in_axes=(array_axis, array_axis, array_axis, array_axis, None, None),
        out_axes=array_axis,
    )
    def step(model_snapshot, batch, key, first, floor, clamp):
        frozen = jax.tree_util.tree_map(
            lambda value: (
                jax.lax.stop_gradient(value)
                if hasattr(value, "dtype")
                else value
            ),
            model_snapshot,
        )

        def logtarget(values):
            return posterior_log_target(
                frozen,
                values,
                batch,
                latent_spec,
                context,
                model_args,
                parameter_names,
                likelihood_config,
                calibration_config,
            ).logtarget

        return expand_defensive_importance(
            model_snapshot=frozen,
            features=batch.features,
            key=key,
            first=first,
            first_proposal=first_proposal,
            additional_proposal=additional_proposal,
            additional_particles=int(additional_particles),
            logtarget_fn=logtarget,
            log_std_floor_value=floor,
            flow_scale_clamp_value=clamp,
            minimum_ess_fraction=minimum_ess_fraction,
            maximum_weight=maximum_weight,
        )

    return step


def make_pmap_sc_drws_q_step(
    *,
    optimizer,
    gradient_clip_norm: float,
):
    """Build an encoder-only stopped RWS update with an entropy floor."""
    from .config import require_equinox

    eqx = require_equinox()
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
            array_axis,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        out_axes=(array_axis, array_axis, array_axis, array_axis),
    )
    def step(
        model,
        optimizer_state,
        features,
        particles,
        exact_weights,
        object_mask,
        key,
        tau,
        floor,
        clamp,
        reference_entropy,
        entropy_margin,
        entropy_strength,
        entropy_factor,
    ):
        def objective(encoder):
            candidate = eqx.tree_at(lambda tree: tree.encoder, model, encoder)
            wake = q_wake_loss(
                candidate,
                features,
                particles,
                exact_weights,
                object_mask,
                tau=tau,
                log_std_floor_value=floor,
                flow_scale_clamp_value=clamp,
            )
            penalty, entropy = entropy_floor_loss(
                candidate,
                features,
                key,
                reference_entropy=reference_entropy,
                margin=entropy_margin,
                strength=entropy_strength,
                factor=entropy_factor,
                log_std_floor_value=floor,
                flow_scale_clamp_value=clamp,
            )
            return wake + penalty, (wake, penalty, entropy)

        (loss, auxiliary), grads = eqx.filter_value_and_grad(
            objective, has_aux=True
        )(model.encoder)
        grads = _pmean_tree(grads, "devices")
        loss = jax.lax.pmean(loss, "devices")
        raw_norm = _tree_l2_norm(grads)
        finite = jax.lax.pmin(
            (jnp.isfinite(loss) & _tree_all_finite(grads)).astype(jnp.int32),
            "devices",
        ).astype(bool)
        safe_grads = jax.tree_util.tree_map(
            lambda value: (
                jnp.where(finite, value, jnp.zeros_like(value))
                if value is not None and hasattr(value, "dtype")
                else value
            ),
            grads,
        )
        updates, proposed_state = optimizer.update(
            safe_grads,
            optimizer_state,
            eqx.filter(model.encoder, eqx.is_inexact_array),
        )
        proposed = eqx.apply_updates(model.encoder, updates)
        encoder = _select_tree(proposed, model.encoder, finite)
        optimizer_state = _select_tree(proposed_state, optimizer_state, finite)
        model = eqx.tree_at(lambda tree: tree.encoder, model, encoder)
        metrics = SCDrwsStepMetrics(
            loss=loss,
            raw_grad_norm=raw_norm,
            clipped_grad_norm=jnp.minimum(raw_norm, float(gradient_clip_norm)),
            grad_clipped=raw_norm > float(gradient_clip_norm),
            grads_finite=finite,
            update_applied=finite,
        )
        wake, penalty, entropy = auxiliary
        details = {
            "wake_loss": jax.lax.pmean(wake, "devices"),
            "entropy_floor_penalty": jax.lax.pmean(penalty, "devices"),
            **jax.tree_util.tree_map(
                lambda value: jax.lax.pmean(value, "devices"), entropy
            ),
        }
        return model, optimizer_state, metrics, details

    return step


def make_pmap_sc_drws_sleep_step(
    *,
    optimizer,
    latent_spec,
    context,
    model_args,
    parameter_names,
    likelihood_config,
    calibration_config,
    objective_config,
    gradient_clip_norm: float,
):
    """Build a q-only selected-sleep step with dynamic variance controls."""
    from .config import require_equinox
    from .train import _model_generated_sleep_loss

    eqx = require_equinox()
    array_axis = eqx.if_array(0)

    @eqx.filter_pmap(
        axis_name="devices",
        in_axes=(array_axis, array_axis, array_axis, array_axis, None, None),
        out_axes=(array_axis, array_axis, array_axis, array_axis),
    )
    def step(model, optimizer_state, batch, key, floor, clamp):
        def objective(encoder):
            candidate = eqx.tree_at(lambda tree: tree.encoder, model, encoder)
            return _model_generated_sleep_loss(
                candidate,
                batch,
                latent_spec,
                context,
                model_args,
                parameter_names,
                key,
                likelihood_config,
                calibration_config,
                objective_config,
                log_std_floor=floor,
                flow_scale_clamp=clamp,
            )

        (loss, details), grads = eqx.filter_value_and_grad(
            objective, has_aux=True
        )(model.encoder)
        grads = _pmean_tree(grads, "devices")
        loss = jax.lax.pmean(loss, "devices")
        raw_norm = _tree_l2_norm(grads)
        finite = jax.lax.pmin(
            (jnp.isfinite(loss) & _tree_all_finite(grads)).astype(jnp.int32),
            "devices",
        ).astype(bool)
        safe_grads = jax.tree_util.tree_map(
            lambda value: (
                jnp.where(finite, value, jnp.zeros_like(value))
                if value is not None and hasattr(value, "dtype")
                else value
            ),
            grads,
        )
        updates, proposed_state = optimizer.update(
            safe_grads,
            optimizer_state,
            eqx.filter(model.encoder, eqx.is_inexact_array),
        )
        proposed = eqx.apply_updates(model.encoder, updates)
        encoder = _select_tree(proposed, model.encoder, finite)
        optimizer_state = _select_tree(proposed_state, optimizer_state, finite)
        model = eqx.tree_at(lambda tree: tree.encoder, model, encoder)
        metrics = SCDrwsStepMetrics(
            loss=loss,
            raw_grad_norm=raw_norm,
            clipped_grad_norm=jnp.minimum(raw_norm, float(gradient_clip_norm)),
            grad_clipped=raw_norm > float(gradient_clip_norm),
            grads_finite=finite,
            update_applied=finite,
        )
        details = jax.tree_util.tree_map(
            lambda value: jax.lax.pmean(value, "devices"), details
        )
        return model, optimizer_state, metrics, details

    return step
