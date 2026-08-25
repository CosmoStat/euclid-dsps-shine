"""Ordinary-IS plus adaptive-SMC hierarchy for frozen object posteriors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .adaptive_bridge_smc import AdaptiveBridgeSMCConfig, AdaptiveBridgeSMCResult
from .adaptive_smc_training import (
    AdaptiveSMCProposalConfig,
    OrdinaryImportanceResult,
    make_pmap_continuation_e_step,
    make_pmap_e_step,
    make_pmap_ordinary_importance_step,
    run_model_adaptive_smc_e_step,
    run_model_ordinary_importance,
    snapshot_model,
)
from .posterior_bank import PosteriorBankShard
from .sc_asmc_em import HierarchyDispatch, dispatch_posterior_hierarchy


@dataclass(frozen=True)
class ModelHierarchyResult:
    ordinary: OrdinaryImportanceResult
    primary_indices: np.ndarray
    primary: AdaptiveBridgeSMCResult | None
    fallback_indices: np.ndarray
    fallback: AdaptiveBridgeSMCResult | None
    extended_indices: np.ndarray
    extended: AdaptiveBridgeSMCResult | None
    dispatch: HierarchyDispatch


@dataclass(frozen=True)
class PmapHierarchyKernels:
    ordinary: Any
    primary: Any
    fallback: Any
    extended: Any
    devices: tuple[Any, ...]
    primary_batch_size: int
    fallback_batch_size: int
    extended_batch_size: int


def build_pmap_hierarchy_kernels(
    *,
    latent_spec: Any,
    context: Any,
    model_args: Any,
    parameter_names: tuple[str, ...],
    likelihood_config: dict[str, Any],
    calibration_config: dict[str, Any],
    primary_config: AdaptiveBridgeSMCConfig,
    fallback_config: AdaptiveBridgeSMCConfig,
    extended_config: AdaptiveBridgeSMCConfig,
    proposal_config: AdaptiveSMCProposalConfig,
    minimum_is_ess_fraction: float = 0.10,
    maximum_is_weight: float = 0.80,
    primary_batch_size: int = 64,
    fallback_batch_size: int = 32,
    extended_batch_size: int = 16,
    devices: tuple[Any, ...] | None = None,
) -> PmapHierarchyKernels:
    """Compile-shape contract for object-independent local-device scaling."""
    _validate_hierarchy_configs(primary_config, fallback_config, extended_config)
    local_devices = tuple(devices or jax.local_devices())
    if not local_devices:
        raise RuntimeError("no local JAX device is available")
    for name, size in {
        "primary": primary_batch_size,
        "fallback": fallback_batch_size,
        "extended": extended_batch_size,
    }.items():
        if int(size) <= 0 or int(size) % len(local_devices):
            raise ValueError(
                f"{name} fixed batch size must be positive and divisible by devices"
            )
    common = dict(
        latent_spec=latent_spec,
        context=context,
        model_args=model_args,
        parameter_names=parameter_names,
        likelihood_config=likelihood_config,
        calibration_config=calibration_config,
        proposal_config=proposal_config,
    )
    return PmapHierarchyKernels(
        ordinary=make_pmap_ordinary_importance_step(
            **common,
            minimum_ess_fraction=minimum_is_ess_fraction,
            maximum_weight=maximum_is_weight,
        ),
        primary=make_pmap_continuation_e_step(
            **common,
            smc_config=primary_config,
        ),
        fallback=make_pmap_e_step(**common, smc_config=fallback_config),
        extended=make_pmap_e_step(**common, smc_config=extended_config),
        devices=local_devices,
        primary_batch_size=int(primary_batch_size),
        fallback_batch_size=int(fallback_batch_size),
        extended_batch_size=int(extended_batch_size),
    )


def run_pmap_model_hierarchical_e_step(
    *,
    model_snapshot: Any,
    batch: Any,
    key: jax.Array,
    kernels: PmapHierarchyKernels,
) -> ModelHierarchyResult:
    """Run one object batch across every local device with selective levels."""
    n_objects = int(batch.features.shape[0])
    if n_objects <= 0 or n_objects > kernels.primary_batch_size:
        raise ValueError(
            "hierarchical pmap batch must contain 1..primary_batch_size objects"
        )
    frozen = snapshot_model(model_snapshot)
    replicated = _replicate_model(frozen, kernels.devices)
    ordinary_batch, _ = _pad_object_batch(batch, kernels.primary_batch_size)
    ordinary_sharded = _shard_object_batch(ordinary_batch, len(kernels.devices))
    ordinary_keys = jax.random.split(jax.random.fold_in(key, 1), len(kernels.devices))
    ordinary = _slice_ordinary_result(
        _unshard_ordinary_result(
            kernels.ordinary(replicated, ordinary_sharded, ordinary_keys)
        ),
        n_objects,
    )
    direct = np.asarray(jax.device_get(ordinary.accepted), dtype=bool)
    primary_indices = np.flatnonzero(~direct).astype(np.int64)
    primary = None
    primary_succeeded = np.zeros(n_objects, dtype=bool)
    if len(primary_indices):
        primary = _run_selective_pmap_level(
            model_replicated=replicated,
            batch=batch,
            selected_indices=primary_indices,
            key=jax.random.fold_in(key, 2),
            step=kernels.primary,
            fixed_batch_size=kernels.primary_batch_size,
            n_devices=len(kernels.devices),
            initial_particles=jnp.take(
                ordinary.particles, jnp.asarray(primary_indices), axis=1
            ),
            initial_logproposal=jnp.take(
                ordinary.logproposal, jnp.asarray(primary_indices), axis=1
            ),
            initial_logtarget=jnp.take(
                ordinary.loglike + ordinary.logprior,
                jnp.asarray(primary_indices),
                axis=1,
            ),
        )
        primary_succeeded[primary_indices] = ~np.asarray(
            jax.device_get(primary.hard_object_flag), dtype=bool
        )
    fallback_indices = primary_indices[~primary_succeeded[primary_indices]]
    fallback = None
    fallback_succeeded = np.zeros(n_objects, dtype=bool)
    if len(fallback_indices):
        fallback = _run_selective_pmap_level(
            model_replicated=replicated,
            batch=batch,
            selected_indices=fallback_indices,
            key=jax.random.fold_in(key, 3),
            step=kernels.fallback,
            fixed_batch_size=kernels.fallback_batch_size,
            n_devices=len(kernels.devices),
        )
        fallback_succeeded[fallback_indices] = ~np.asarray(
            jax.device_get(fallback.hard_object_flag), dtype=bool
        )
    extended_indices = fallback_indices[~fallback_succeeded[fallback_indices]]
    extended = None
    extended_succeeded = np.zeros(n_objects, dtype=bool)
    if len(extended_indices):
        extended = _run_selective_pmap_level(
            model_replicated=replicated,
            batch=batch,
            selected_indices=extended_indices,
            key=jax.random.fold_in(key, 4),
            step=kernels.extended,
            fixed_batch_size=kernels.extended_batch_size,
            n_devices=len(kernels.devices),
        )
        extended_succeeded[extended_indices] = ~np.asarray(
            jax.device_get(extended.hard_object_flag), dtype=bool
        )
    dispatch = dispatch_posterior_hierarchy(
        direct,
        primary_succeeded,
        fallback_succeeded,
        extended_succeeded,
    )
    return ModelHierarchyResult(
        ordinary=ordinary,
        primary_indices=primary_indices,
        primary=primary,
        fallback_indices=fallback_indices,
        fallback=fallback,
        extended_indices=extended_indices,
        extended=extended,
        dispatch=dispatch,
    )


def compiled_hierarchy_memory_analysis(
    *,
    model_snapshot: Any,
    batch: Any,
    key: jax.Array,
    kernels: PmapHierarchyKernels,
) -> dict[str, dict[str, int]]:
    """Compile every fixed hierarchy kernel and return per-device buffer plans."""
    frozen = snapshot_model(model_snapshot)
    replicated = _replicate_model(frozen, kernels.devices)
    n_devices = len(kernels.devices)
    latent_dim = int(frozen.prior.latent_dim)

    def fixed_inputs(size: int, fold: int) -> tuple[Any, jax.Array]:
        count = min(int(batch.features.shape[0]), int(size))
        selected = _take_object_batch(batch, np.arange(count, dtype=np.int64))
        padded, _ = _pad_object_batch(selected, int(size))
        return (
            _shard_object_batch(padded, n_devices),
            jax.random.split(jax.random.fold_in(key, fold), n_devices),
        )

    ordinary_batch, ordinary_keys = fixed_inputs(kernels.primary_batch_size, 11)
    local_primary = kernels.primary_batch_size // n_devices
    initial_particles = jnp.zeros(
        (n_devices, 64, local_primary, latent_dim), dtype=jnp.float32
    )
    initial_logproposal = jnp.zeros((n_devices, 64, local_primary), dtype=jnp.float32)
    initial_logtarget = jnp.zeros_like(initial_logproposal)
    fallback_batch, fallback_keys = fixed_inputs(kernels.fallback_batch_size, 12)
    extended_batch, extended_keys = fixed_inputs(kernels.extended_batch_size, 13)
    compiled = {
        "ordinary_k64": kernels.ordinary.lower(
            replicated, ordinary_batch, ordinary_keys
        ).compile(),
        "primary_k64": kernels.primary.lower(
            replicated,
            ordinary_batch,
            ordinary_keys,
            initial_particles,
            initial_logproposal,
            initial_logtarget,
        ).compile(),
        "fallback_k128": kernels.fallback.lower(
            replicated, fallback_batch, fallback_keys
        ).compile(),
        "extended_k128": kernels.extended.lower(
            replicated, extended_batch, extended_keys
        ).compile(),
    }
    return {
        name: _compiled_memory_record(executable)
        for name, executable in compiled.items()
    }


def _compiled_memory_record(executable: Any) -> dict[str, int]:
    compiled = getattr(executable, "compiled", executable)
    stats = compiled.memory_analysis()
    fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "peak_memory_in_bytes",
    )
    return {name: int(getattr(stats, name) or 0) for name in fields}


def run_model_hierarchical_e_step(
    *,
    model_snapshot: Any,
    batch: Any,
    latent_spec: Any,
    context: Any,
    model_args: Any,
    parameter_names: tuple[str, ...],
    likelihood_config: dict[str, Any],
    calibration_config: dict[str, Any],
    key: jax.Array,
    primary_config: AdaptiveBridgeSMCConfig,
    fallback_config: AdaptiveBridgeSMCConfig,
    extended_config: AdaptiveBridgeSMCConfig,
    proposal_config: AdaptiveSMCProposalConfig | None = None,
    minimum_is_ess_fraction: float = 0.10,
    maximum_is_weight: float = 0.80,
) -> ModelHierarchyResult:
    """Run the hierarchy while freezing q/prior and limiting expensive fallbacks."""
    _validate_hierarchy_configs(primary_config, fallback_config, extended_config)
    frozen = snapshot_model(model_snapshot)
    is_key, primary_key, fallback_key, extended_key = jax.random.split(key, 4)
    ordinary = run_model_ordinary_importance(
        model_snapshot=frozen,
        batch=batch,
        latent_spec=latent_spec,
        context=context,
        model_args=model_args,
        parameter_names=parameter_names,
        likelihood_config=likelihood_config,
        calibration_config=calibration_config,
        key=is_key,
        n_particles=64,
        proposal_config=proposal_config,
        minimum_ess_fraction=minimum_is_ess_fraction,
        maximum_weight=maximum_is_weight,
    )
    n_objects = int(batch.features.shape[0])
    direct = np.asarray(jax.device_get(ordinary.accepted), dtype=bool)
    primary_indices = np.flatnonzero(~direct).astype(np.int64)
    primary = None
    primary_succeeded = np.zeros(n_objects, dtype=bool)
    if len(primary_indices):
        primary_batch = _take_object_batch(batch, primary_indices)
        primary = run_model_adaptive_smc_e_step(
            model_snapshot=frozen,
            batch=primary_batch,
            latent_spec=latent_spec,
            context=context,
            model_args=model_args,
            parameter_names=parameter_names,
            likelihood_config=likelihood_config,
            calibration_config=calibration_config,
            key=primary_key,
            smc_config=primary_config,
            proposal_config=proposal_config,
            initial_particles=jnp.take(
                ordinary.particles, jnp.asarray(primary_indices), axis=1
            ),
            initial_logproposal=jnp.take(
                ordinary.logproposal, jnp.asarray(primary_indices), axis=1
            ),
            initial_logtarget=jnp.take(
                ordinary.loglike + ordinary.logprior,
                jnp.asarray(primary_indices),
                axis=1,
            ),
        )
        local_success = ~np.asarray(
            jax.device_get(primary.hard_object_flag), dtype=bool
        )
        primary_succeeded[primary_indices] = local_success
    fallback_indices = primary_indices[~primary_succeeded[primary_indices]]
    fallback = None
    fallback_succeeded = np.zeros(n_objects, dtype=bool)
    if len(fallback_indices):
        fallback = run_model_adaptive_smc_e_step(
            model_snapshot=frozen,
            batch=_take_object_batch(batch, fallback_indices),
            latent_spec=latent_spec,
            context=context,
            model_args=model_args,
            parameter_names=parameter_names,
            likelihood_config=likelihood_config,
            calibration_config=calibration_config,
            key=fallback_key,
            smc_config=fallback_config,
            proposal_config=proposal_config,
        )
        local_success = ~np.asarray(
            jax.device_get(fallback.hard_object_flag), dtype=bool
        )
        fallback_succeeded[fallback_indices] = local_success
    extended_indices = fallback_indices[~fallback_succeeded[fallback_indices]]
    extended = None
    extended_succeeded = np.zeros(n_objects, dtype=bool)
    if len(extended_indices):
        extended = run_model_adaptive_smc_e_step(
            model_snapshot=frozen,
            batch=_take_object_batch(batch, extended_indices),
            latent_spec=latent_spec,
            context=context,
            model_args=model_args,
            parameter_names=parameter_names,
            likelihood_config=likelihood_config,
            calibration_config=calibration_config,
            key=extended_key,
            smc_config=extended_config,
            proposal_config=proposal_config,
        )
        local_success = ~np.asarray(
            jax.device_get(extended.hard_object_flag), dtype=bool
        )
        extended_succeeded[extended_indices] = local_success
    dispatch = dispatch_posterior_hierarchy(
        direct,
        primary_succeeded,
        fallback_succeeded,
        extended_succeeded,
    )
    return ModelHierarchyResult(
        ordinary=ordinary,
        primary_indices=primary_indices,
        primary=primary,
        fallback_indices=fallback_indices,
        fallback=fallback,
        extended_indices=extended_indices,
        extended=extended,
        dispatch=dispatch,
    )


def hierarchy_result_to_bank_shard(
    result: ModelHierarchyResult,
    *,
    model_snapshot: Any,
    row_index: np.ndarray,
    object_id: np.ndarray,
    features: np.ndarray | None,
    feature_reference: str | None = None,
    particle_capacity: int = 128,
    primary_config: AdaptiveBridgeSMCConfig,
    fallback_config: AdaptiveBridgeSMCConfig,
    extended_config: AdaptiveBridgeSMCConfig,
) -> PosteriorBankShard:
    """Pack variable-cost hierarchy outputs into one validated padded shard."""
    rows = np.asarray(row_index, dtype=np.int64)
    identifiers = np.asarray(object_id, dtype=str)
    n_objects = len(rows)
    latent_dim = int(result.ordinary.particles.shape[-1])
    capacity = int(particle_capacity)
    if capacity < 128:
        raise ValueError("hierarchical posterior bank requires capacity at least 128")
    particles = np.zeros((n_objects, capacity, latent_dim), dtype=np.float32)
    weights = np.zeros((n_objects, capacity), dtype=np.float64)
    source_logprior = np.zeros((n_objects, capacity), dtype=np.float64)
    counts = np.full(n_objects, 64, dtype=np.int16)
    ess = np.asarray(jax.device_get(result.ordinary.ess), dtype=np.float64)
    maximum = np.asarray(jax.device_get(result.ordinary.max_weight), dtype=np.float64)
    beta = np.ones(n_objects, dtype=np.float32)
    logz = np.asarray(jax.device_get(result.ordinary.logz_estimate), dtype=np.float64)
    stages = np.zeros(n_objects, dtype=np.int16)
    acceptance = np.full(n_objects, np.nan, dtype=np.float32)
    ancestor_ess = np.full(n_objects, 64.0, dtype=np.float32)
    unique_ancestor = np.ones(n_objects, dtype=np.float32)
    movement = np.zeros(n_objects, dtype=np.float32)
    moved_fraction = np.zeros(n_objects, dtype=np.float32)
    evaluations = np.full(n_objects, 64, dtype=np.int32)
    ordinary_particles = np.asarray(jax.device_get(result.ordinary.particles))
    ordinary_weights = np.asarray(jax.device_get(result.ordinary.normalized_weights))
    ordinary_logprior = np.asarray(jax.device_get(result.ordinary.logprior))
    particles[:, :64] = ordinary_particles.transpose(1, 0, 2)
    weights[:, :64] = ordinary_weights.T
    source_logprior[:, :64] = ordinary_logprior.T

    if result.primary is not None:
        evaluations[result.primary_indices] += _smc_mutation_evaluations(
            result.primary, primary_config, include_initial=False
        )
        _fill_smc_objects(
            model_snapshot,
            result.primary,
            result.primary_indices[
                ~np.asarray(jax.device_get(result.primary.hard_object_flag), dtype=bool)
            ],
            np.flatnonzero(
                ~np.asarray(jax.device_get(result.primary.hard_object_flag), dtype=bool)
            ),
            particles,
            weights,
            source_logprior,
            counts,
            ess,
            maximum,
            beta,
            logz,
            stages,
            acceptance,
            ancestor_ess,
            unique_ancestor,
            movement,
            moved_fraction,
        )
    if result.fallback is not None:
        evaluations[result.fallback_indices] += _smc_mutation_evaluations(
            result.fallback, fallback_config, include_initial=True
        )
        _fill_smc_objects(
            model_snapshot,
            result.fallback,
            result.fallback_indices[
                ~np.asarray(
                    jax.device_get(result.fallback.hard_object_flag), dtype=bool
                )
            ],
            np.flatnonzero(
                ~np.asarray(
                    jax.device_get(result.fallback.hard_object_flag), dtype=bool
                )
            ),
            particles,
            weights,
            source_logprior,
            counts,
            ess,
            maximum,
            beta,
            logz,
            stages,
            acceptance,
            ancestor_ess,
            unique_ancestor,
            movement,
            moved_fraction,
        )
    if result.extended is not None:
        evaluations[result.extended_indices] += _smc_mutation_evaluations(
            result.extended, extended_config, include_initial=True
        )
        _fill_smc_objects(
            model_snapshot,
            result.extended,
            result.extended_indices,
            np.arange(len(result.extended_indices), dtype=np.int64),
            particles,
            weights,
            source_logprior,
            counts,
            ess,
            maximum,
            beta,
            logz,
            stages,
            acceptance,
            ancestor_ess,
            unique_ancestor,
            movement,
            moved_fraction,
        )
    shard = PosteriorBankShard(
        row_index=rows,
        object_id=identifiers,
        method=np.asarray(result.dispatch.method, dtype=np.int8),
        particles=particles,
        normalized_weights=weights,
        source_logprior=source_logprior,
        particle_count=counts,
        ess=ess,
        max_weight=maximum,
        beta_final=beta,
        logz=logz,
        stage_count=stages,
        acceptance=acceptance,
        ancestor_ess=ancestor_ess,
        unique_ancestor_fraction=unique_ancestor,
        movement_squared=movement,
        moved_particle_fraction=moved_fraction,
        dsps_evaluations=evaluations,
        resolved=np.asarray(result.dispatch.resolved, dtype=bool),
        features=None if features is None else np.asarray(features, dtype=np.float32),
        feature_reference=feature_reference,
    )
    shard.validate()
    return shard


def _take_object_batch(batch: Any, indices: np.ndarray) -> Any:
    selected = jnp.asarray(indices, dtype=jnp.int32)
    return jax.tree_util.tree_map(
        lambda value: jnp.take(value, selected, axis=0), batch
    )


def _replicate_model(model: Any, devices: tuple[Any, ...]) -> Any:
    count = len(devices)

    def replicate(value):
        if not hasattr(value, "shape"):
            return value
        host = np.asarray(jax.device_get(value))
        return jnp.asarray(np.broadcast_to(host, (count, *host.shape)).copy())

    return jax.tree_util.tree_map(replicate, model)


def _pad_object_batch(batch: Any, target_count: int) -> tuple[Any, int]:
    count = int(batch.features.shape[0])
    if count <= 0 or count > int(target_count):
        raise ValueError("invalid object batch size for fixed-shape padding")
    indices = np.arange(int(target_count), dtype=np.int64) % count
    return _take_object_batch(batch, indices), count


def _shard_object_batch(batch: Any, n_devices: int) -> Any:
    objects = int(batch.features.shape[0])
    if objects % int(n_devices):
        raise ValueError("fixed object batch is not divisible by local devices")
    local = objects // int(n_devices)
    return jax.tree_util.tree_map(
        lambda value: jnp.asarray(value).reshape(
            int(n_devices), local, *jnp.asarray(value).shape[1:]
        ),
        batch,
    )


def _shard_particle_objects(value: jnp.ndarray, n_devices: int) -> np.ndarray:
    # Selective continuation starts from a previous pmap result. Materialize it
    # on the host so its replicated NamedSharding cannot leak into the next pmap.
    array = np.asarray(jax.device_get(value))
    particles, objects = array.shape[:2]
    if objects % int(n_devices):
        raise ValueError("particle objects are not divisible by local devices")
    local = objects // int(n_devices)
    axes = (1, 0, 2, *range(3, array.ndim + 1))
    return np.ascontiguousarray(
        array.reshape(
            particles, int(n_devices), local, *array.shape[2:]
        ).transpose(axes)
    )


def _run_selective_pmap_level(
    *,
    model_replicated: Any,
    batch: Any,
    selected_indices: np.ndarray,
    key: jax.Array,
    step: Any,
    fixed_batch_size: int,
    n_devices: int,
    initial_particles: jnp.ndarray | None = None,
    initial_logproposal: jnp.ndarray | None = None,
    initial_logtarget: jnp.ndarray | None = None,
) -> AdaptiveBridgeSMCResult:
    results = []
    for chunk_index, start in enumerate(
        range(0, len(selected_indices), int(fixed_batch_size))
    ):
        indices = selected_indices[start : start + int(fixed_batch_size)]
        selected = _take_object_batch(batch, indices)
        padded, actual = _pad_object_batch(selected, int(fixed_batch_size))
        sharded = _shard_object_batch(padded, int(n_devices))
        keys = jax.random.split(jax.random.fold_in(key, chunk_index), int(n_devices))
        if initial_particles is None:
            sharded_result = step(model_replicated, sharded, keys)
        else:
            local_slice = slice(start, start + actual)
            particles = _pad_particle_objects(
                initial_particles[:, local_slice], int(fixed_batch_size)
            )
            logproposal = _pad_particle_objects(
                initial_logproposal[:, local_slice], int(fixed_batch_size)
            )
            logtarget = _pad_particle_objects(
                initial_logtarget[:, local_slice], int(fixed_batch_size)
            )
            sharded_result = step(
                model_replicated,
                sharded,
                keys,
                _shard_particle_objects(particles, int(n_devices)),
                _shard_particle_objects(logproposal, int(n_devices)),
                _shard_particle_objects(logtarget, int(n_devices)),
            )
        results.append(_slice_smc_result(_unshard_smc_result(sharded_result), actual))
    return _concat_smc_results(results)


def _pad_particle_objects(value: jnp.ndarray, target_count: int) -> jnp.ndarray:
    array = jnp.asarray(value)
    count = int(array.shape[1])
    indices = jnp.arange(int(target_count), dtype=jnp.int32) % count
    return jnp.take(array, indices, axis=1)


def _unshard_ordinary_result(
    result: OrdinaryImportanceResult,
) -> OrdinaryImportanceResult:
    def particle_object(value):
        array = jnp.asarray(value)
        devices, particles, local = array.shape[:3]
        trailing = array.shape[3:]
        axes = (1, 0, 2, *range(3, array.ndim))
        return array.transpose(axes).reshape(particles, devices * local, *trailing)

    def objects(value):
        return jnp.asarray(value).reshape(-1)

    return OrdinaryImportanceResult(
        particles=particle_object(result.particles),
        normalized_weights=particle_object(result.normalized_weights),
        loglike=particle_object(result.loglike),
        logprior=particle_object(result.logprior),
        logproposal=particle_object(result.logproposal),
        logweight=particle_object(result.logweight),
        ess=objects(result.ess),
        max_weight=objects(result.max_weight),
        logz_estimate=objects(result.logz_estimate),
        target_finite=objects(result.target_finite),
        accepted=objects(result.accepted),
    )


def _slice_ordinary_result(
    result: OrdinaryImportanceResult,
    count: int,
) -> OrdinaryImportanceResult:
    return OrdinaryImportanceResult(
        *(
            value[:, : int(count)] if index < 6 else value[: int(count)]
            for index, value in enumerate(result)
        )
    )


def _unshard_smc_result(result: AdaptiveBridgeSMCResult) -> AdaptiveBridgeSMCResult:
    particle_fields = {
        "final_particles",
        "final_normalized_weights",
        "final_log_weights",
        "ancestor_ids",
    }
    path_fields = {
        "beta_path",
        "conditional_ess_path",
        "ess_path",
        "resampled_path",
        "mutation_acceptance_path",
    }
    values = []
    for name, value in zip(result._fields, result, strict=True):
        array = jnp.asarray(value)
        if name == "final_particles":
            devices, particles, local, latent = array.shape
            converted = array.transpose(1, 0, 2, 3).reshape(
                particles, devices * local, latent
            )
        elif name in particle_fields:
            devices, particles, local = array.shape
            converted = array.transpose(1, 0, 2).reshape(particles, devices * local)
        elif name in path_fields:
            devices, stages, local = array.shape
            converted = array.transpose(1, 0, 2).reshape(stages, devices * local)
        else:
            converted = array.reshape(-1)
        values.append(converted)
    return AdaptiveBridgeSMCResult(*values)


def _slice_smc_result(
    result: AdaptiveBridgeSMCResult,
    count: int,
) -> AdaptiveBridgeSMCResult:
    particle_fields = {
        "final_particles",
        "final_normalized_weights",
        "final_log_weights",
        "ancestor_ids",
    }
    path_fields = {
        "beta_path",
        "conditional_ess_path",
        "ess_path",
        "resampled_path",
        "mutation_acceptance_path",
    }
    return AdaptiveBridgeSMCResult(
        *(
            value[:, : int(count)]
            if name in particle_fields | path_fields
            else value[: int(count)]
            for name, value in zip(result._fields, result, strict=True)
        )
    )


def _concat_smc_results(
    results: list[AdaptiveBridgeSMCResult],
) -> AdaptiveBridgeSMCResult:
    if not results:
        raise ValueError("cannot concatenate an empty SMC result list")
    particle_or_path = {
        "final_particles",
        "final_normalized_weights",
        "final_log_weights",
        "ancestor_ids",
        "beta_path",
        "conditional_ess_path",
        "ess_path",
        "resampled_path",
        "mutation_acceptance_path",
    }
    return AdaptiveBridgeSMCResult(
        *(
            jnp.concatenate(
                [getattr(result, name) for result in results],
                axis=1 if name in particle_or_path else 0,
            )
            for name in results[0]._fields
        )
    )


def _validate_hierarchy_configs(primary, fallback, extended) -> None:
    expected = ((primary, 64, 16), (fallback, 128, 32), (extended, 128, 48))
    for config, particles, max_stages in expected:
        if int(config.n_particles) != particles or int(config.max_stages) != max_stages:
            raise ValueError(
                "SC-ASMC-EM hierarchy requires K64/stage16, K128/stage32, "
                "and hard-only K128/stage48"
            )


def _smc_mutation_evaluations(
    result: AdaptiveBridgeSMCResult,
    config: AdaptiveBridgeSMCConfig,
    *,
    include_initial: bool,
) -> np.ndarray:
    resamples = np.asarray(jax.device_get(result.number_of_resamples), dtype=np.int64)
    reached = np.asarray(jax.device_get(result.beta_final)) >= 1.0 - 1.0e-6
    count = int(config.n_particles) * (
        resamples * int(config.steps_after_resample)
        + reached.astype(np.int64) * int(config.final_steps_at_beta1)
    )
    if include_initial:
        count += int(config.n_particles)
    return count.astype(np.int32)


def _fill_smc_objects(
    model_snapshot: Any,
    result: AdaptiveBridgeSMCResult,
    global_indices: np.ndarray,
    local_indices: np.ndarray,
    particles: np.ndarray,
    weights: np.ndarray,
    source_logprior: np.ndarray,
    counts: np.ndarray,
    ess: np.ndarray,
    maximum: np.ndarray,
    beta: np.ndarray,
    logz: np.ndarray,
    stages: np.ndarray,
    acceptance: np.ndarray,
    ancestor_ess: np.ndarray,
    unique_ancestor: np.ndarray,
    movement: np.ndarray,
    moved_fraction: np.ndarray,
) -> None:
    if not len(global_indices):
        return
    local = jnp.asarray(local_indices, dtype=jnp.int32)
    value = jnp.take(result.final_particles, local, axis=1)
    normalized = jnp.take(result.final_normalized_weights, local, axis=1)
    logprior = model_snapshot.prior.log_prob(value)
    value_np = np.asarray(jax.device_get(value))
    weight_np = np.asarray(jax.device_get(normalized))
    logprior_np = np.asarray(jax.device_get(logprior))
    count = int(value_np.shape[0])
    particles[global_indices, :count] = value_np.transpose(1, 0, 2)
    weights[global_indices, :] = 0.0
    weights[global_indices, :count] = weight_np.T
    source_logprior[global_indices, :] = 0.0
    source_logprior[global_indices, :count] = logprior_np.T
    counts[global_indices] = count
    fields = {
        "ess": (ess, result.final_ess),
        "maximum": (maximum, result.final_max_weight),
        "beta": (beta, result.beta_final),
        "logz": (logz, result.logZ_estimate),
        "stages": (stages, result.number_of_stages),
        "acceptance": (acceptance, result.mutation_acceptance),
        "ancestor_ess": (ancestor_ess, result.ancestor_ess),
        "unique_ancestor": (unique_ancestor, result.unique_ancestor_fraction),
        "movement": (movement, result.median_epsilon_squared_jump),
        "moved_fraction": (moved_fraction, result.moved_particle_fraction),
    }
    for destination, source in fields.values():
        values = np.asarray(jax.device_get(jnp.take(source, local, axis=0)))
        destination[global_indices] = values
