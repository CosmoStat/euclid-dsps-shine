"""Production trainer for self-supervised adaptive-SMC wake learning."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.filters import load_filters
from euclid_dsps.io import ensure_dir, write_json
from euclid_dsps.model import dynamic_model_args, load_context

from .adaptive_bridge_smc import AdaptiveBridgeSMCConfig, AdaptiveBridgeSMCResult
from .adaptive_smc_training import (
    AdaptiveSMCProposalConfig,
    PriorUpdateMetrics,
    SMCPosteriorBatch,
    apply_prior_macro_update,
    make_component_optimizer,
    make_pmap_e_step,
    make_pmap_q_sleep_step,
    make_pmap_q_smc_step,
    merge_hard_fallback,
    primary_posterior_batch,
    q_only_importance_diagnostics,
    smc_q_distillation_loss,
    snapshot_model,
    tree_all_finite,
    tree_l2_norm,
)
from .config import amortized_config, require_amortized_dependencies
from .data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from .features import (
    compute_feature_stats,
    write_feature_stats,
)
from .latent import (
    latent_spec_hash,
    latent_spec_to_jsonable,
    write_latent_prior_geometry,
)
from .posterior import posterior_entropy_diagnostics
from .train import (
    JitLatentSpec,
    LossBatch,
    _estimate_selection_log_alpha,
    _latent_spec_for_amortized_config,
    _loss_batch,
    _model_generated_sleep_loss,
    _pad_epoch_order_for_data_parallel,
    _replicate_tree,
    _selection_correction_runtime_config,
    _shard_loss_batch,
    _sleep_runtime_config,
    _unreplicate_tree,
    build_amortized_model,
    build_training_split,
    save_checkpoint,
    write_training_split_artifacts,
)

eqx, _optax = require_amortized_dependencies()


@dataclass(frozen=True)
class AdaptiveTrainingConfig:
    bootstrap_sleep_epochs: int = 12
    observed_sweeps: int = 3
    micro_batch_size: int = 128
    prior_macro_objects: int = 512
    min_prior_macro_objects: int = 128
    fallback_batch_size: int = 32
    sleep_replay_every_smc_updates: int = 4
    q_sleep_learning_rate: float = 5.0e-5
    q_smc_learning_rate: float = 2.0e-5
    prior_learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-6
    gradient_clip_norm: float = 5.0
    trust_strength: float = 0.2
    trust_samples: int = 512
    max_prior_kl_per_dimension: float = 0.05
    max_alpha_mc_relative_error: float = 0.15
    validation_objects: int = 32
    validation_q_is_particles: int = 64
    hard_fraction_fail: float = 0.30
    seed: int = 260821


class AdaptiveTrainingState(NamedTuple):
    model: Any
    q_sleep_optimizer_state: Any
    q_smc_optimizer_state: Any
    prior_optimizer_state: Any
    bootstrap_epoch: jnp.ndarray
    observed_sweep: jnp.ndarray
    smc_updates: jnp.ndarray
    prior_updates: jnp.ndarray
    random_key: jax.Array


@dataclass(frozen=True)
class RuntimeBundle:
    config: dict[str, Any]
    latent_spec: Any
    jit_latent_spec: JitLatentSpec
    context: Any
    model_args: Any
    parameter_names: tuple[str, ...]
    likelihood_config: dict[str, Any]
    calibration_config: dict[str, Any]
    sleep_objective_config: dict[str, Any]
    selection_objective_config: dict[str, Any]
    feature_stats: Any
    train_arrays: Any
    validation_arrays: Any
    split: Any


def adaptive_training_config(config: dict[str, Any]) -> AdaptiveTrainingConfig:
    cfg = amortized_config(config)
    raw = dict(cfg["training"].get("adaptive_smc", {}) or {})
    return AdaptiveTrainingConfig(
        bootstrap_sleep_epochs=int(raw.get("bootstrap_sleep_epochs", 12)),
        observed_sweeps=int(raw.get("observed_sweeps", 3)),
        micro_batch_size=int(raw.get("micro_batch_size", 128)),
        prior_macro_objects=int(raw.get("prior_macro_objects", 512)),
        min_prior_macro_objects=int(raw.get("min_prior_macro_objects", 128)),
        fallback_batch_size=int(raw.get("fallback_batch_size", 32)),
        sleep_replay_every_smc_updates=int(
            raw.get("sleep_replay_every_smc_updates", 4)
        ),
        q_sleep_learning_rate=float(raw.get("q_sleep_learning_rate", 5.0e-5)),
        q_smc_learning_rate=float(raw.get("q_smc_learning_rate", 2.0e-5)),
        prior_learning_rate=float(raw.get("prior_learning_rate", 1.0e-5)),
        weight_decay=float(raw.get("weight_decay", 1.0e-6)),
        gradient_clip_norm=float(raw.get("gradient_clip_norm", 5.0)),
        trust_strength=float(raw.get("trust_strength", 0.2)),
        trust_samples=int(raw.get("trust_samples", 512)),
        max_prior_kl_per_dimension=float(
            raw.get("max_prior_kl_per_dimension", 0.05)
        ),
        max_alpha_mc_relative_error=float(
            raw.get("max_alpha_mc_relative_error", 0.15)
        ),
        validation_objects=int(raw.get("validation_objects", 32)),
        validation_q_is_particles=int(raw.get("validation_q_is_particles", 64)),
        hard_fraction_fail=float(raw.get("hard_fraction_fail", 0.30)),
        seed=int(cfg["training"].get("seed", raw.get("seed", 260821))),
    )


def adaptive_smc_configs(config: dict[str, Any]):
    objective = amortized_config(config)["objective"]
    raw = dict(objective.get("adaptive_smc", {}) or {})
    proposal_raw = dict(raw.get("initial_proposal", {}) or {})
    primary = AdaptiveBridgeSMCConfig(
        n_particles=int(raw.get("n_particles", 64)),
        target_conditional_ess_fraction=float(raw.get("target_conditional_ess", 0.75)),
        resample_ess_fraction=float(raw.get("resample_ess_fraction", 0.50)),
        max_stages=int(raw.get("max_stages", 8)),
        steps_after_resample=int(raw.get("steps_after_resample", 2)),
        final_steps_at_beta1=int(raw.get("final_steps_at_beta1", 1)),
        rw_scale=float(raw.get("rw_scale", 0.60)),
        hard_final_ess_fraction=float(raw.get("hard_final_ess_fraction", 0.30)),
        hard_min_mutation_acceptance=float(
            raw.get("hard_min_mutation_acceptance", 0.05)
        ),
        bisection_steps=int(raw.get("bisection_steps", 32)),
    )
    fallback_raw = dict(raw.get("hard_fallback", {}) or {})
    fallback = AdaptiveBridgeSMCConfig(
        n_particles=int(fallback_raw.get("n_particles", 128)),
        target_conditional_ess_fraction=float(
            fallback_raw.get(
                "target_conditional_ess",
                primary.target_conditional_ess_fraction,
            )
        ),
        resample_ess_fraction=float(
            fallback_raw.get("resample_ess_fraction", primary.resample_ess_fraction)
        ),
        max_stages=int(fallback_raw.get("max_stages", 12)),
        steps_after_resample=int(
            fallback_raw.get("steps_after_resample", primary.steps_after_resample)
        ),
        final_steps_at_beta1=int(
            fallback_raw.get("final_steps_at_beta1", primary.final_steps_at_beta1)
        ),
        rw_scale=float(fallback_raw.get("rw_scale", primary.rw_scale)),
        hard_final_ess_fraction=float(
            fallback_raw.get(
                "hard_final_ess_fraction",
                primary.hard_final_ess_fraction,
            )
        ),
        hard_min_mutation_acceptance=float(
            fallback_raw.get(
                "hard_min_mutation_acceptance",
                primary.hard_min_mutation_acceptance,
            )
        ),
        bisection_steps=int(
            fallback_raw.get("bisection_steps", primary.bisection_steps)
        ),
    )
    proposal = AdaptiveSMCProposalConfig(
        posterior_unit_fraction=float(
            proposal_raw.get("posterior_unit_fraction", 0.70)
        ),
        posterior_tempered_fraction=float(
            proposal_raw.get("posterior_tempered_fraction", 0.20)
        ),
        posterior_temperature=float(
            proposal_raw.get("posterior_temperature", 1.50)
        ),
        prior_fraction=float(proposal_raw.get("prior_fraction", 0.10)),
    )
    _validate_runtime_configs(primary, fallback, proposal)
    return primary, fallback, proposal


def _validate_runtime_configs(primary, fallback, proposal):
    if primary.n_particles != 64:
        raise ValueError("production adaptive SMC requires primary K=64")
    if fallback.n_particles != 128:
        raise ValueError("production hard fallback requires K=128")
    fractions = np.asarray(proposal.normalized_fractions())
    if not np.allclose(fractions, [0.70, 0.20, 0.10], atol=1.0e-8):
        raise ValueError("production r0 must be exactly 0.70 q1 + 0.20 q1.5 + 0.10 p")
    if not np.isclose(proposal.posterior_temperature, 1.5):
        raise ValueError("production defensive temperature must be 1.5")


def _config_without_truth(config: dict[str, Any]) -> dict[str, Any]:
    runtime = copy.deepcopy(config)
    runtime["truth"] = {"parameter_columns": {}}
    return runtime


def prepare_adaptive_training_runtime(
    config: dict[str, Any],
    out: Path,
    *,
    train_indices_file: str | Path,
    validation_indices_file: str | Path,
) -> RuntimeBundle:
    """Load only observed photometry and fixed physical-model assets."""
    runtime_config = _config_without_truth(config)
    cfg = amortized_config(runtime_config)
    split = build_training_split(
        runtime_config,
        limit=None,
        seed=int(cfg["training"].get("seed", 260821)),
        train_indices_file=train_indices_file,
        validation_indices_file=validation_indices_file,
    )
    write_training_split_artifacts(out, split)
    catalog_batch_size = int(cfg["data"].get("catalog_batch_size", 10_000))
    train_arrays = load_photometry_arrays_from_config(
        runtime_config,
        batch_size=catalog_batch_size,
        row_indices=split.train_indices,
    )
    validation_arrays = load_photometry_arrays_from_config(
        runtime_config,
        batch_size=catalog_batch_size,
        row_indices=split.validation_indices,
    )
    if train_arrays.truth or validation_arrays.truth:
        raise RuntimeError("production adaptive-SMC training loaded truth columns")
    feature_stats = compute_feature_stats(
        train_arrays.flux,
        train_arrays.flux_err,
        train_arrays.mask,
        band_names=train_arrays.band_names,
        flux_transform=str(cfg["features"].get("flux_transform", "asinh")),
    )
    write_feature_stats(out / "feature_stats.json", feature_stats)
    filters = load_filters(runtime_config["bands"])
    context = load_context(
        runtime_config["ssp_path"],
        filters,
        n_sfh_bins=int(runtime_config["model"].get("n_sfh_bins", 96)),
        cosmos_config=runtime_config.get("cosmos_sed"),
        nebular_emission=runtime_config.get("nebular_emission", "ssp_flux"),
        model_config=runtime_config.get("model"),
    )
    model_args = dynamic_model_args(context)
    latent_spec = _latent_spec_for_amortized_config(runtime_config)
    if latent_spec.normalization != "standardized_logit":
        raise ValueError(
            "strict self-supervised production requires bounds-based "
            "standardized_logit coordinates"
        )
    if cfg["latent"].get("normalization_checkpoint"):
        raise ValueError("truth-derived normalization checkpoints are forbidden")
    if str(cfg["prior"].get("source")) not in {"joint_realnvp", "realnvp"}:
        raise ValueError("production parent prior must be a broad joint_realnvp")
    if cfg["prior"].get("checkpoint"):
        raise ValueError("production parent prior must not load a truth-trained checkpoint")
    latent_payload = latent_spec_to_jsonable(latent_spec)
    latent_payload.update(
        {
            "normalization_hash": latent_spec_hash(latent_spec),
            "coordinate_information_source": "fit_bounds_and_fit_initials_only",
            "population_density_initialization": "identity_realnvp_standard_normal",
            "truth_used": False,
        }
    )
    write_json(out / "effective_latent_spec.json", latent_payload)
    write_latent_prior_geometry(
        runtime_config,
        out,
        n_samples=int(cfg["latent"].get("geometry_samples", 20_000)),
        seed=int(cfg["training"].get("seed", 260821)),
    )
    jit_latent_spec = JitLatentSpec(
        names=latent_spec.names,
        lower=latent_spec.lower,
        upper=latent_spec.upper,
        raw_center=latent_spec.raw_center,
        raw_scale=latent_spec.raw_scale,
        normalization=latent_spec.normalization,
        transform_family=latent_spec.transform_family,
        transform_location=latent_spec.transform_location,
        transform_lambda=latent_spec.transform_lambda,
    )
    sleep_runtime = _sleep_runtime_config(runtime_config, feature_stats)
    if str(sleep_runtime.get("error_model")) != "observed_catalog":
        raise ValueError("production sleep must use observed_catalog errors")
    selection_runtime = _selection_correction_runtime_config(
        runtime_config,
        feature_stats,
    )
    if not selection_runtime.get("enabled"):
        raise ValueError("production parent prior requires selection correction")
    sleep_objective = {
        "sleep": sleep_runtime,
        "selection_correction": selection_runtime,
        "prior_train_jointly": True,
    }
    selection_objective = {
        "selection_correction": selection_runtime,
        "prior_train_jointly": True,
    }
    return RuntimeBundle(
        config=runtime_config,
        latent_spec=latent_spec,
        jit_latent_spec=jit_latent_spec,
        context=context,
        model_args=model_args,
        parameter_names=latent_spec.names,
        likelihood_config=dict(cfg["likelihood"]),
        calibration_config={
            "calibration": runtime_config.get("calibration", {}) or {}
        },
        sleep_objective_config=sleep_objective,
        selection_objective_config=selection_objective,
        feature_stats=feature_stats,
        train_arrays=train_arrays,
        validation_arrays=validation_arrays,
        split=split,
    )


def _loss_batch_take(batch: LossBatch, indices: np.ndarray) -> LossBatch:
    selected = jnp.asarray(indices, dtype=jnp.int32)
    return LossBatch(*(jnp.take(value, selected, axis=0) for value in batch))


def _pad_loss_batch(batch: LossBatch, target_count: int) -> tuple[LossBatch, int]:
    count = int(batch.flux.shape[0])
    if count <= 0 or count > int(target_count):
        raise ValueError("invalid fallback batch size")
    pad = int(target_count) - count
    if pad == 0:
        return batch, count
    indices = jnp.concatenate(
        (
            jnp.arange(count, dtype=jnp.int32),
            jnp.arange(pad, dtype=jnp.int32) % count,
        )
    )
    return _loss_batch_take(batch, np.asarray(indices)), count


def _unshard_smc_result(result: AdaptiveBridgeSMCResult) -> AdaptiveBridgeSMCResult:
    def particles(value):
        array = jnp.asarray(value)
        devices, count, local_objects, latent = array.shape
        return array.transpose(1, 0, 2, 3).reshape(
            count, devices * local_objects, latent
        )

    def particle_object(value):
        array = jnp.asarray(value)
        devices, count, local_objects = array.shape
        return array.transpose(1, 0, 2).reshape(count, devices * local_objects)

    def objects(value):
        array = jnp.asarray(value)
        return array.reshape(-1)

    def paths(value):
        array = jnp.asarray(value)
        devices, stages, local_objects = array.shape
        return array.transpose(1, 0, 2).reshape(stages, devices * local_objects)

    return AdaptiveBridgeSMCResult(
        final_particles=particles(result.final_particles),
        final_normalized_weights=particle_object(result.final_normalized_weights),
        final_log_weights=particle_object(result.final_log_weights),
        beta_final=objects(result.beta_final),
        beta_path=paths(result.beta_path),
        conditional_ess_path=paths(result.conditional_ess_path),
        ess_path=paths(result.ess_path),
        resampled_path=paths(result.resampled_path),
        mutation_acceptance_path=paths(result.mutation_acceptance_path),
        final_ess=objects(result.final_ess),
        final_max_weight=objects(result.final_max_weight),
        number_of_stages=objects(result.number_of_stages),
        number_of_resamples=objects(result.number_of_resamples),
        mutation_acceptance=objects(result.mutation_acceptance),
        hard_object_flag=objects(result.hard_object_flag),
        finite_target_fraction=objects(result.finite_target_fraction),
        logZ_estimate=objects(result.logZ_estimate),
        ancestor_ids=particle_object(result.ancestor_ids),
    )


def _slice_smc_result(result: AdaptiveBridgeSMCResult, count: int):
    count = int(count)
    return AdaptiveBridgeSMCResult(
        final_particles=result.final_particles[:, :count],
        final_normalized_weights=result.final_normalized_weights[:, :count],
        final_log_weights=result.final_log_weights[:, :count],
        beta_final=result.beta_final[:count],
        beta_path=result.beta_path[:, :count],
        conditional_ess_path=result.conditional_ess_path[:, :count],
        ess_path=result.ess_path[:, :count],
        resampled_path=result.resampled_path[:, :count],
        mutation_acceptance_path=result.mutation_acceptance_path[:, :count],
        final_ess=result.final_ess[:count],
        final_max_weight=result.final_max_weight[:count],
        number_of_stages=result.number_of_stages[:count],
        number_of_resamples=result.number_of_resamples[:count],
        mutation_acceptance=result.mutation_acceptance[:count],
        hard_object_flag=result.hard_object_flag[:count],
        finite_target_fraction=result.finite_target_fraction[:count],
        logZ_estimate=result.logZ_estimate[:count],
        ancestor_ids=result.ancestor_ids[:, :count],
    )


def _shard_posterior(posterior: SMCPosteriorBatch, n_devices: int):
    devices = int(n_devices)
    n_objects = int(posterior.eligible.shape[0])
    if n_objects % devices:
        raise ValueError("posterior object count must be divisible by devices")
    local = n_objects // devices

    def particle_object(value):
        value = jnp.asarray(value)
        return value.reshape(value.shape[0], devices, local, *value.shape[2:]).transpose(
            1, 0, 2, *range(3, value.ndim + 1)
        )

    def objects(value):
        value = jnp.asarray(value)
        return value.reshape(devices, local, *value.shape[1:])

    return SMCPosteriorBatch(
        particles=particle_object(posterior.particles),
        normalized_weights=particle_object(posterior.normalized_weights),
        eligible=objects(posterior.eligible),
        beta_final=objects(posterior.beta_final),
        final_ess=objects(posterior.final_ess),
        final_max_weight=objects(posterior.final_max_weight),
        mutation_acceptance=objects(posterior.mutation_acceptance),
        logZ_estimate=objects(posterior.logZ_estimate),
        fallback_attempted=objects(posterior.fallback_attempted),
        fallback_succeeded=objects(posterior.fallback_succeeded),
    )


def _concat_posteriors(values: list[SMCPosteriorBatch]) -> SMCPosteriorBatch:
    return SMCPosteriorBatch(
        *(jnp.concatenate([getattr(item, field) for item in values], axis=1 if field in {"particles", "normalized_weights"} else 0)
          for field in SMCPosteriorBatch._fields)
    )


def _replicate_model_for_pmap(model, devices: tuple[Any, ...]):
    """Build a fresh leading device axis without retaining an old mesh.

    Indexing a ``filter_pmap`` result can leave its leaves committed to a
    replicated named mesh. Broadcasting those leaves directly preserves that
    mesh and conflicts with the internal axis of the next ``filter_pmap``.
    A host round trip deliberately removes the stale sharding annotation before
    the model is replicated again after a prior macro-update.
    """
    n_devices = len(tuple(devices))

    def replicate_leaf(leaf):
        if not eqx.is_array(leaf):
            return leaf
        host = np.asarray(jax.device_get(leaf))
        replicated = np.broadcast_to(host, (n_devices, *host.shape)).copy()
        return jnp.asarray(replicated)

    return jax.tree_util.tree_map(replicate_leaf, model)


def _run_training_e_step(
    *,
    model_replicated,
    batch: LossBatch,
    real_object_mask: np.ndarray,
    row_indices: np.ndarray,
    primary_step,
    fallback_step,
    primary_key,
    fallback_key,
    n_devices: int,
    fallback_batch_size: int,
    hard_rows: list[dict[str, Any]],
    sweep: int,
    batch_index: int,
):
    sharded_batch = _shard_loss_batch(batch, n_devices)
    primary_keys = jax.random.split(primary_key, n_devices)
    primary_sharded = primary_step(
        model_replicated,
        sharded_batch,
        primary_keys,
    )
    primary_result = _unshard_smc_result(primary_sharded)
    posterior = primary_posterior_batch(primary_result)
    real_mask = jnp.asarray(real_object_mask, dtype=jnp.bool_)
    posterior = posterior._replace(eligible=posterior.eligible & real_mask)
    hard_indices = np.flatnonzero(
        np.asarray(primary_result.hard_object_flag) & np.asarray(real_object_mask)
    )
    if not len(hard_indices):
        return posterior, primary_result
    if int(fallback_batch_size) % int(n_devices):
        raise ValueError("fallback_batch_size must be divisible by device count")
    for chunk_index, start in enumerate(
        range(0, len(hard_indices), int(fallback_batch_size))
    ):
        selected_indices = hard_indices[start : start + int(fallback_batch_size)]
        selected_batch = _loss_batch_take(batch, selected_indices)
        padded_batch, actual_count = _pad_loss_batch(
            selected_batch,
            int(fallback_batch_size),
        )
        sharded_fallback_batch = _shard_loss_batch(padded_batch, n_devices)
        chunk_key = jax.random.fold_in(fallback_key, chunk_index)
        fallback_keys = jax.random.split(chunk_key, n_devices)
        fallback_sharded = fallback_step(
            model_replicated,
            sharded_fallback_batch,
            fallback_keys,
        )
        fallback_result = _slice_smc_result(
            _unshard_smc_result(fallback_sharded),
            actual_count,
        )
        actual_indices = selected_indices[:actual_count]
        posterior = merge_hard_fallback(
            key=jax.random.fold_in(chunk_key, 991),
            primary=posterior,
            fallback=fallback_result,
            hard_object_indices=actual_indices,
        )
        for local_index, object_index in enumerate(actual_indices):
            hard_rows.append(
                {
                    "sweep": int(sweep),
                    "batch": int(batch_index),
                    "row_index": int(row_indices[object_index]),
                    "primary_beta_final": float(
                        np.asarray(primary_result.beta_final[object_index])
                    ),
                    "primary_ess_fraction": float(
                        np.asarray(primary_result.final_ess[object_index])
                        / primary_result.final_particles.shape[0]
                    ),
                    "fallback_beta_final": float(
                        np.asarray(fallback_result.beta_final[local_index])
                    ),
                    "fallback_ess_fraction": float(
                        np.asarray(fallback_result.final_ess[local_index])
                        / fallback_result.final_particles.shape[0]
                    ),
                    "fallback_acceptance": float(
                        np.asarray(fallback_result.mutation_acceptance[local_index])
                    ),
                    "fallback_succeeded": bool(
                        not np.asarray(fallback_result.hard_object_flag[local_index])
                    ),
                }
            )
    return posterior, primary_result


def _posterior_summary(posterior: SMCPosteriorBatch) -> dict[str, float]:
    eligible = np.asarray(posterior.eligible, dtype=bool)
    count = max(int(np.sum(eligible)), 1)

    def eligible_mean(value):
        array = np.asarray(value, dtype=float)
        return float(np.sum(np.where(eligible, array, 0.0)) / count)

    acceptance = np.asarray(posterior.mutation_acceptance, dtype=float)
    finite_acceptance = eligible & np.isfinite(acceptance)
    weights = np.asarray(posterior.normalized_weights, dtype=float)
    ess_fraction = np.asarray(posterior.final_ess, dtype=float) / float(
        posterior.particles.shape[0]
    )
    max_weight = np.asarray(posterior.final_max_weight, dtype=float)
    weight_entropy = -np.sum(
        np.where(weights > 0.0, weights * np.log(np.maximum(weights, 1.0e-300)), 0.0),
        axis=0,
    )
    return {
        "objects": int(eligible.size),
        "eligible_objects": int(np.sum(eligible)),
        "eligible_fraction": float(np.mean(eligible)),
        "hard_fraction_after_fallback": float(np.mean(~eligible)),
        "median_beta_final": float(np.median(np.asarray(posterior.beta_final))),
        "median_final_ess_fraction": float(
            np.median(
                np.asarray(posterior.final_ess)
                / float(posterior.particles.shape[0])
            )
        ),
        "mean_final_max_weight": eligible_mean(posterior.final_max_weight),
        "median_weight_entropy": (
            float(np.median(weight_entropy[eligible]))
            if np.any(eligible)
            else float("nan")
        ),
        "fraction_ess_below_0p1": (
            float(np.mean(ess_fraction[eligible] < 0.10))
            if np.any(eligible)
            else 1.0
        ),
        "fraction_max_weight_above_0p8": (
            float(np.mean(max_weight[eligible] > 0.80))
            if np.any(eligible)
            else 1.0
        ),
        "median_mutation_acceptance": (
            float(np.median(acceptance[finite_acceptance]))
            if np.any(finite_acceptance)
            else float("nan")
        ),
        "fallback_attempt_fraction": float(
            np.mean(np.asarray(posterior.fallback_attempted, dtype=bool))
        ),
        "fallback_success_fraction": float(
            np.mean(np.asarray(posterior.fallback_succeeded, dtype=bool))
        ),
    }


def save_adaptive_training_state(
    path: str | Path,
    state: AdaptiveTrainingState,
    *,
    config: dict[str, Any],
    latent_spec,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, state)
    write_json(
        path.with_suffix(path.suffix + ".json"),
        {
            "latent_spec_hash": latent_spec_hash(latent_spec),
            "bootstrap_epoch": int(np.asarray(state.bootstrap_epoch)),
            "observed_sweep": int(np.asarray(state.observed_sweep)),
            "smc_updates": int(np.asarray(state.smc_updates)),
            "prior_updates": int(np.asarray(state.prior_updates)),
            "random_key": np.asarray(state.random_key, dtype=np.uint32).tolist(),
            "optimizer_contract": {
                "q_sleep": "independent_adamw",
                "q_smc": "independent_adamw",
                "prior": "independent_adamw",
            },
            "truth_used": False,
            "adaptive_training": asdict(adaptive_training_config(config)),
        },
    )


def load_adaptive_training_state(
    path: str | Path,
    template: AdaptiveTrainingState,
    *,
    latent_spec,
) -> AdaptiveTrainingState:
    path = Path(path)
    sidecar = json.loads(
        path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8")
    )
    if sidecar.get("latent_spec_hash") != latent_spec_hash(latent_spec):
        raise ValueError("adaptive training state latent spec mismatch")
    return eqx.tree_deserialise_leaves(path, template)


def _make_sleep_loss_fn(runtime: RuntimeBundle):
    def loss(model, batch, key):
        return _model_generated_sleep_loss(
            model,
            batch,
            runtime.jit_latent_spec,
            runtime.context,
            runtime.model_args,
            runtime.parameter_names,
            key,
            runtime.likelihood_config,
            runtime.calibration_config,
            runtime.sleep_objective_config,
        )

    return loss


def _make_selection_log_alpha_fn(runtime: RuntimeBundle):
    def selection(model, key):
        return _estimate_selection_log_alpha(
            model,
            runtime.jit_latent_spec,
            runtime.context,
            runtime.model_args,
            runtime.parameter_names,
            key,
            runtime.calibration_config,
            runtime.selection_objective_config,
        )

    return selection


def _selection_alpha_gradient_preflight(runtime: RuntimeBundle, model, key):
    objective = copy.deepcopy(runtime.selection_objective_config)
    selection = objective["selection_correction"]
    count = int(selection.get("gradient_preflight_samples", 64))
    selection["n_prior_samples"] = count
    selection["prior_sample_batch_size"] = min(
        count,
        int(selection.get("prior_sample_batch_size", count)),
    )

    def log_alpha_for_prior(prior):
        candidate = eqx.tree_at(lambda tree: tree.prior, model, prior)
        log_alpha, _metrics = _estimate_selection_log_alpha(
            candidate,
            runtime.jit_latent_spec,
            runtime.context,
            runtime.model_args,
            runtime.parameter_names,
            key,
            runtime.calibration_config,
            objective,
        )
        return log_alpha

    log_alpha, gradient = eqx.filter_value_and_grad(log_alpha_for_prior)(model.prior)
    gradient_norm = tree_l2_norm(gradient)
    return {
        "samples": count,
        "log_alpha": float(np.asarray(log_alpha)),
        "gradient_norm": float(np.asarray(gradient_norm)),
        "finite": bool(
            np.asarray(jnp.isfinite(log_alpha) & tree_all_finite(gradient))
        ),
        "nonzero": bool(np.asarray(gradient_norm > 0.0)),
    }


def _validation_batch(runtime: RuntimeBundle, requested: int, n_devices: int):
    count = min(int(requested), int(runtime.validation_arrays.flux.shape[0]))
    if count <= 0:
        raise ValueError("adaptive SMC validation requires held-out observations")
    target = ((count + n_devices - 1) // n_devices) * n_devices
    order = np.arange(count, dtype=np.int64)
    if target > count:
        order = np.concatenate([order, np.arange(target - count) % count])
    photometry = next(
        iter_photometry_batches_from_arrays(
            runtime.validation_arrays,
            batch_size=target,
            feature_stats=runtime.feature_stats,
            order=order,
            truth_names=None,
        )
    )
    return (
        _loss_batch(photometry),
        np.arange(target) < count,
        np.asarray(photometry.row_index, dtype=np.int64),
    )


def _run_validation(
    *,
    model,
    model_replicated,
    runtime: RuntimeBundle,
    training_config: AdaptiveTrainingConfig,
    primary_step,
    fallback_step,
    n_devices: int,
    key: jax.Array,
    sweep: int,
) -> tuple[dict[str, Any], SMCPosteriorBatch]:
    batch, real_mask, row_indices = _validation_batch(
        runtime,
        training_config.validation_objects,
        n_devices,
    )
    primary_key, fallback_key, is_key, alpha_key, entropy_key = jax.random.split(
        key, 5
    )
    validation_hard_rows: list[dict[str, Any]] = []
    posterior, _primary = _run_training_e_step(
        model_replicated=model_replicated,
        batch=batch,
        real_object_mask=real_mask,
        row_indices=row_indices,
        primary_step=primary_step,
        fallback_step=fallback_step,
        primary_key=primary_key,
        fallback_key=fallback_key,
        n_devices=n_devices,
        fallback_batch_size=training_config.fallback_batch_size,
        hard_rows=validation_hard_rows,
        sweep=sweep,
        batch_index=-1,
    )
    q_cross_entropy, q_metrics = smc_q_distillation_loss(
        model,
        batch.features,
        posterior,
    )
    q_is = q_only_importance_diagnostics(
        model_snapshot=snapshot_model(model),
        batch=batch,
        latent_spec=runtime.jit_latent_spec,
        context=runtime.context,
        model_args=runtime.model_args,
        parameter_names=runtime.parameter_names,
        likelihood_config=runtime.likelihood_config,
        calibration_config=runtime.calibration_config,
        key=is_key,
        n_particles=training_config.validation_q_is_particles,
    )
    log_alpha, selection_metrics = _make_selection_log_alpha_fn(runtime)(
        model,
        alpha_key,
    )
    entropy = posterior_entropy_diagnostics(
        model,
        batch.features,
        entropy_key,
        n_samples=4,
    )
    eligible = np.asarray(posterior.eligible) & np.asarray(real_mask)
    logz = np.asarray(posterior.logZ_estimate, dtype=float)
    q_is_valid = np.asarray(q_is["valid"]) & np.asarray(real_mask)
    q_is_ess = np.asarray(q_is["ess_fraction"], dtype=float)
    summary = _posterior_summary(posterior._replace(eligible=jnp.asarray(eligible)))
    summary.update(
        {
            "sweep": int(sweep),
            "validation_smc_cross_entropy": float(np.asarray(q_cross_entropy)),
            "validation_smc_eligible_count": float(
                np.asarray(q_metrics.eligible_count)
            ),
            "validation_q_is_ess_fraction": (
                float(np.median(q_is_ess[q_is_valid]))
                if np.any(q_is_valid)
                else 0.0
            ),
            "validation_q_is_max_weight": (
                float(np.median(np.asarray(q_is["max_weight"])[q_is_valid]))
                if np.any(q_is_valid)
                else 1.0
            ),
            "validation_selected_log_evidence": (
                float(np.mean(logz[eligible]) - np.asarray(log_alpha))
                if np.any(eligible)
                else float("nan")
            ),
            "selection_alpha": float(
                np.asarray(selection_metrics["selection/alpha"])
            ),
            "selection_log_alpha": float(np.asarray(log_alpha)),
            "selection_alpha_mc_relative_error": float(
                np.asarray(
                    selection_metrics["selection/alpha_mc_relative_error"]
                )
            ),
            "posterior_full_entropy_mc": float(
                np.asarray(entropy["posterior_full_entropy_mc"])
            ),
            "posterior_base_entropy": float(
                np.asarray(entropy["posterior_base_entropy"])
            ),
            "posterior_residual_logdet_mean": float(
                np.asarray(entropy["posterior_residual_logdet_mean"])
            ),
            "posterior_residual_logdet_q05": float(
                np.asarray(entropy["posterior_residual_logdet_q05"])
            ),
            "posterior_residual_logdet_q95": float(
                np.asarray(entropy["posterior_residual_logdet_q95"])
            ),
            "truth_used": False,
        }
    )
    return summary, posterior


def train_feniks_adaptive_smc(
    config: dict[str, Any],
    out_dir: str | Path,
    *,
    train_indices_file: str | Path,
    validation_indices_file: str | Path,
    smoke: bool = False,
    resume_state: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run bootstrap sleep followed by exact-target observed SMC sweeps."""
    out = ensure_dir(out_dir)
    if (out / "training_receipt.json").exists():
        raise FileExistsError(f"adaptive training output already complete: {out}")
    write_json(out / "normalized_config.json", _config_without_truth(config))
    runtime = prepare_adaptive_training_runtime(
        config,
        out,
        train_indices_file=train_indices_file,
        validation_indices_file=validation_indices_file,
    )
    training = adaptive_training_config(runtime.config)
    if smoke:
        training = replace(
            training,
            bootstrap_sleep_epochs=min(training.bootstrap_sleep_epochs, 12),
            observed_sweeps=1,
            micro_batch_size=min(training.micro_batch_size, 32),
            prior_macro_objects=min(training.prior_macro_objects, 64),
            min_prior_macro_objects=min(training.min_prior_macro_objects, 32),
            validation_objects=min(training.validation_objects, 8),
        )
        selection = runtime.selection_objective_config["selection_correction"]
        selection["n_prior_samples"] = min(
            int(selection["n_prior_samples"]), 512
        )
        selection["prior_sample_batch_size"] = min(
            int(selection["prior_sample_batch_size"]), 128
        )
    primary_config, fallback_config, proposal_config = adaptive_smc_configs(
        runtime.config
    )
    devices = tuple(jax.local_devices())
    n_devices = len(devices)
    if n_devices < 1:
        raise RuntimeError("no local JAX device is available")
    if training.micro_batch_size % n_devices:
        raise ValueError("micro_batch_size must be divisible by local device count")
    if training.fallback_batch_size % n_devices:
        raise ValueError("fallback_batch_size must be divisible by local device count")
    _log(
        verbose,
        "[adaptive-smc-train] "
        f"backend={jax.default_backend()} devices={n_devices} smoke={int(smoke)} ",
    )
    _log(
        verbose,
        "[adaptive-smc-train] "
        f"train={runtime.train_arrays.flux.shape[0]} "
        f"validation={runtime.validation_arrays.flux.shape[0]} "
        f"bootstrap={training.bootstrap_sleep_epochs} "
        f"observed_sweeps={training.observed_sweeps} "
        f"micro_batch={training.micro_batch_size}",
    )
    key = jax.random.PRNGKey(int(training.seed))
    key, model_key = jax.random.split(key)
    model = build_amortized_model(
        runtime.config,
        model_key,
        latent_spec=runtime.latent_spec,
    )
    q_sleep_optimizer = make_component_optimizer(
        learning_rate=training.q_sleep_learning_rate,
        gradient_clip_norm=training.gradient_clip_norm,
        weight_decay=training.weight_decay,
    )
    q_smc_optimizer = make_component_optimizer(
        learning_rate=training.q_smc_learning_rate,
        gradient_clip_norm=training.gradient_clip_norm,
        weight_decay=training.weight_decay,
    )
    prior_optimizer = make_component_optimizer(
        learning_rate=training.prior_learning_rate,
        gradient_clip_norm=training.gradient_clip_norm,
        weight_decay=training.weight_decay,
    )
    q_sleep_state = q_sleep_optimizer.init(
        eqx.filter(model.encoder, eqx.is_inexact_array)
    )
    q_smc_state = q_smc_optimizer.init(eqx.filter(model.encoder, eqx.is_inexact_array))
    prior_state = prior_optimizer.init(eqx.filter(model.prior, eqx.is_inexact_array))
    bootstrap_start = 1
    observed_start = 1
    smc_update_count = 0
    prior_update_count = 0
    template_state = AdaptiveTrainingState(
        model=model,
        q_sleep_optimizer_state=q_sleep_state,
        q_smc_optimizer_state=q_smc_state,
        prior_optimizer_state=prior_state,
        bootstrap_epoch=jnp.asarray(0, dtype=jnp.int32),
        observed_sweep=jnp.asarray(0, dtype=jnp.int32),
        smc_updates=jnp.asarray(0, dtype=jnp.int32),
        prior_updates=jnp.asarray(0, dtype=jnp.int32),
        random_key=key,
    )
    if resume_state is not None:
        restored = load_adaptive_training_state(
            resume_state,
            template_state,
            latent_spec=runtime.latent_spec,
        )
        model = restored.model
        q_sleep_state = restored.q_sleep_optimizer_state
        q_smc_state = restored.q_smc_optimizer_state
        prior_state = restored.prior_optimizer_state
        bootstrap_start = int(np.asarray(restored.bootstrap_epoch)) + 1
        observed_start = int(np.asarray(restored.observed_sweep)) + 1
        smc_update_count = int(np.asarray(restored.smc_updates))
        prior_update_count = int(np.asarray(restored.prior_updates))
        key = restored.random_key
        _log(verbose, f"[adaptive-smc-train] resumed state={resume_state}")
    alpha_preflight_key = jax.random.fold_in(key, 4242)
    alpha_preflight = _selection_alpha_gradient_preflight(
        runtime,
        model,
        alpha_preflight_key,
    )
    write_json(out / "selection_gradient_preflight.json", alpha_preflight)
    _log(
        verbose,
        "[adaptive-smc-train] selection gradient preflight "
        f"finite={int(alpha_preflight['finite'])} "
        f"norm={alpha_preflight['gradient_norm']:.6g}",
    )
    q_sleep_state_replicated = _replicate_tree(q_sleep_state, devices)
    q_smc_state_replicated = _replicate_tree(q_smc_state, devices)
    model_replicated = _replicate_model_for_pmap(model, devices)
    primary_step = make_pmap_e_step(
        latent_spec=runtime.jit_latent_spec,
        context=runtime.context,
        model_args=runtime.model_args,
        parameter_names=runtime.parameter_names,
        likelihood_config=runtime.likelihood_config,
        calibration_config=runtime.calibration_config,
        smc_config=primary_config,
        proposal_config=proposal_config,
    )
    fallback_step = make_pmap_e_step(
        latent_spec=runtime.jit_latent_spec,
        context=runtime.context,
        model_args=runtime.model_args,
        parameter_names=runtime.parameter_names,
        likelihood_config=runtime.likelihood_config,
        calibration_config=runtime.calibration_config,
        smc_config=fallback_config,
        proposal_config=proposal_config,
    )
    q_smc_step = make_pmap_q_smc_step(
        optimizer=q_smc_optimizer,
        gradient_clip_norm=training.gradient_clip_norm,
    )
    q_sleep_step = make_pmap_q_sleep_step(
        optimizer=q_sleep_optimizer,
        sleep_loss_fn=_make_sleep_loss_fn(runtime),
        gradient_clip_norm=training.gradient_clip_norm,
    )
    selection_fn = _make_selection_log_alpha_fn(runtime)
    log_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    best_cross_entropy = float("inf")
    best_epoch_label = "initial"
    start_time = time.time()

    for epoch in range(bootstrap_start, training.bootstrap_sleep_epochs + 1):
        epoch_rng = np.random.default_rng(int(training.seed) + 1000 + epoch)
        order = epoch_rng.permutation(runtime.train_arrays.flux.shape[0])
        order, pad_count = _pad_epoch_order_for_data_parallel(
            order,
            global_batch_size=training.micro_batch_size,
            enabled=True,
            rng=epoch_rng,
        )
        n_batches = len(order) // training.micro_batch_size
        for batch_index, photometry in enumerate(
            iter_photometry_batches_from_arrays(
                runtime.train_arrays,
                batch_size=training.micro_batch_size,
                feature_stats=runtime.feature_stats,
                order=order,
                truth_names=None,
            )
        ):
            key, step_key = jax.random.split(key)
            loss_batch = _loss_batch(photometry)
            sharded_batch = _shard_loss_batch(loss_batch, n_devices)
            model_replicated, q_sleep_state_replicated, sleep_metrics, step_metrics = (
                q_sleep_step(
                    model_replicated,
                    q_sleep_state_replicated,
                    sharded_batch,
                    jax.random.split(step_key, n_devices),
                )
            )
            row = {
                "phase": "bootstrap_sleep",
                "epoch": epoch,
                "batch": batch_index,
                "loss": _replicated_scalar(step_metrics.loss),
                "raw_grad_norm": _replicated_scalar(step_metrics.raw_grad_norm),
                "clipped_grad_norm": _replicated_scalar(
                    step_metrics.clipped_grad_norm
                ),
                "grad_clipped": _replicated_bool(step_metrics.grad_clipped),
                "update_applied": _replicated_bool(step_metrics.update_applied),
                "selection_acceptance": _replicated_scalar(
                    sleep_metrics["sleep_selection_acceptance_fraction"]
                ),
            }
            log_rows.append(row)
            _log(
                verbose,
                "[adaptive-smc-train] "
                f"bootstrap={epoch}/{training.bootstrap_sleep_epochs} "
                f"batch={batch_index + 1}/{n_batches} "
                f"sleep_nll={row['loss']:.5f} update={int(row['update_applied'])}",
            )
        model = _unreplicate_tree(model_replicated)
        q_sleep_state = _unreplicate_tree(q_sleep_state_replicated)
        q_smc_state = _unreplicate_tree(q_smc_state_replicated)
        state = AdaptiveTrainingState(
            model=model,
            q_sleep_optimizer_state=q_sleep_state,
            q_smc_optimizer_state=q_smc_state,
            prior_optimizer_state=prior_state,
            bootstrap_epoch=jnp.asarray(epoch, dtype=jnp.int32),
            observed_sweep=jnp.asarray(observed_start - 1, dtype=jnp.int32),
            smc_updates=jnp.asarray(smc_update_count, dtype=jnp.int32),
            prior_updates=jnp.asarray(prior_update_count, dtype=jnp.int32),
            random_key=key,
        )
        save_adaptive_training_state(
            out / "checkpoints" / "training_state_last.eqx",
            state,
            config=runtime.config,
            latent_spec=runtime.latent_spec,
        )
        if pad_count:
            _log(verbose, f"[adaptive-smc-train] bootstrap padding={pad_count}")

    for sweep in range(observed_start, training.observed_sweeps + 1):
        sweep_rng = np.random.default_rng(int(training.seed) + 2000 + sweep)
        order = sweep_rng.permutation(runtime.train_arrays.flux.shape[0])
        original_count = len(order)
        order, pad_count = _pad_epoch_order_for_data_parallel(
            order,
            global_batch_size=training.micro_batch_size,
            enabled=True,
            rng=sweep_rng,
        )
        real_order_mask = np.arange(len(order)) < original_count
        macro_values: list[SMCPosteriorBatch] = []
        macro_snapshot = None
        macro_object_count = 0
        n_batches = len(order) // training.micro_batch_size
        for batch_index, photometry in enumerate(
            iter_photometry_batches_from_arrays(
                runtime.train_arrays,
                batch_size=training.micro_batch_size,
                feature_stats=runtime.feature_stats,
                order=order,
                truth_names=None,
            )
        ):
            start = batch_index * training.micro_batch_size
            stop = start + training.micro_batch_size
            real_mask = real_order_mask[start:stop]
            batch = _loss_batch(photometry)
            key, primary_key, fallback_key = jax.random.split(key, 3)
            posterior, _primary = _run_training_e_step(
                model_replicated=model_replicated,
                batch=batch,
                real_object_mask=real_mask,
                row_indices=np.asarray(photometry.row_index, dtype=np.int64),
                primary_step=primary_step,
                fallback_step=fallback_step,
                primary_key=primary_key,
                fallback_key=fallback_key,
                n_devices=n_devices,
                fallback_batch_size=training.fallback_batch_size,
                hard_rows=hard_rows,
                sweep=sweep,
                batch_index=batch_index,
            )
            summary = _posterior_summary(posterior)
            sharded_batch = _shard_loss_batch(batch, n_devices)
            sharded_posterior = _shard_posterior(posterior, n_devices)
            model_replicated, q_smc_state_replicated, q_metrics, q_step_metrics = (
                q_smc_step(
                    model_replicated,
                    q_smc_state_replicated,
                    sharded_batch.features,
                    sharded_posterior,
                )
            )
            smc_update_count += int(_replicated_bool(q_step_metrics.update_applied))
            row = {
                "phase": "observed_smc",
                "sweep": sweep,
                "batch": batch_index,
                **summary,
                "q_cross_entropy": _replicated_scalar(q_metrics.cross_entropy),
                "q_update_applied": _replicated_bool(
                    q_step_metrics.update_applied
                ),
                "q_raw_grad_norm": _replicated_scalar(
                    q_step_metrics.raw_grad_norm
                ),
                "q_clipped_grad_norm": _replicated_scalar(
                    q_step_metrics.clipped_grad_norm
                ),
                "q_grad_clipped": _replicated_bool(q_step_metrics.grad_clipped),
            }
            log_rows.append(row)
            if macro_snapshot is None:
                macro_snapshot = snapshot_model(_unreplicate_tree(model_replicated))
            macro_values.append(posterior)
            macro_object_count += int(np.sum(real_mask))
            replay_every = training.sleep_replay_every_smc_updates
            if replay_every > 0 and smc_update_count % replay_every == 0:
                key, replay_key = jax.random.split(key)
                (
                    model_replicated,
                    q_sleep_state_replicated,
                    _sleep_metrics,
                    replay_metrics,
                ) = q_sleep_step(
                    model_replicated,
                    q_sleep_state_replicated,
                    sharded_batch,
                    jax.random.split(replay_key, n_devices),
                )
                log_rows.append(
                    {
                        "phase": "sleep_replay",
                        "sweep": sweep,
                        "batch": batch_index,
                        "loss": _replicated_scalar(replay_metrics.loss),
                        "raw_grad_norm": _replicated_scalar(
                            replay_metrics.raw_grad_norm
                        ),
                        "clipped_grad_norm": _replicated_scalar(
                            replay_metrics.clipped_grad_norm
                        ),
                        "grad_clipped": _replicated_bool(
                            replay_metrics.grad_clipped
                        ),
                        "update_applied": _replicated_bool(
                            replay_metrics.update_applied
                        ),
                    }
                )
            if macro_object_count >= training.prior_macro_objects:
                (
                    model_replicated,
                    prior_state,
                    macro_metrics,
                    key,
                ) = _apply_prior_macro(
                    model_replicated=model_replicated,
                    prior_snapshot=macro_snapshot,
                    prior_state=prior_state,
                    prior_optimizer=prior_optimizer,
                    posterior=_concat_posteriors(macro_values),
                    selection_fn=selection_fn,
                    training=training,
                    devices=devices,
                    key=key,
                )
                prior_update_count += int(macro_metrics["update_applied"])
                macro_metrics.update(
                    {"sweep": sweep, "batch": batch_index, "phase": "prior_macro"}
                )
                prior_rows.append(macro_metrics)
                macro_values = []
                macro_snapshot = None
                macro_object_count = 0
            _log(
                verbose,
                "[adaptive-smc-train] "
                f"sweep={sweep}/{training.observed_sweeps} "
                f"batch={batch_index + 1}/{n_batches} "
                f"beta={summary['median_beta_final']:.3f} "
                f"ESS={summary['median_final_ess_fraction']:.3f} "
                f"hard={summary['hard_fraction_after_fallback']:.3f} "
                f"q_ce={row['q_cross_entropy']:.4f}",
            )
        if macro_values and macro_object_count >= training.min_prior_macro_objects:
            (
                model_replicated,
                prior_state,
                macro_metrics,
                key,
            ) = _apply_prior_macro(
                model_replicated=model_replicated,
                prior_snapshot=macro_snapshot,
                prior_state=prior_state,
                prior_optimizer=prior_optimizer,
                posterior=_concat_posteriors(macro_values),
                selection_fn=selection_fn,
                training=training,
                devices=devices,
                key=key,
            )
            prior_update_count += int(macro_metrics["update_applied"])
            macro_metrics.update(
                {"sweep": sweep, "batch": n_batches, "phase": "prior_macro_tail"}
            )
            prior_rows.append(macro_metrics)
        model = _unreplicate_tree(model_replicated)
        key, validation_key = jax.random.split(key)
        validation, _validation_posterior = _run_validation(
            model=model,
            model_replicated=model_replicated,
            runtime=runtime,
            training_config=training,
            primary_step=primary_step,
            fallback_step=fallback_step,
            n_devices=n_devices,
            key=validation_key,
            sweep=sweep,
        )
        validation_rows.append(validation)
        validation_ce = float(validation["validation_smc_cross_entropy"])
        if np.isfinite(validation_ce) and validation_ce < best_cross_entropy:
            best_cross_entropy = validation_ce
            best_epoch_label = f"observed_sweep_{sweep}"
            save_checkpoint(
                out / "checkpoints" / "best.eqx",
                model,
                config=runtime.config,
                latent_spec=runtime.latent_spec,
                feature_stats=runtime.feature_stats,
                epoch=training.bootstrap_sleep_epochs + sweep,
                metric=validation_ce,
                metric_name="validation_smc_cross_entropy",
            )
        save_checkpoint(
            out / "checkpoints" / "last.eqx",
            model,
            config=runtime.config,
            latent_spec=runtime.latent_spec,
            feature_stats=runtime.feature_stats,
            epoch=training.bootstrap_sleep_epochs + sweep,
            metric=validation_ce,
            metric_name="validation_smc_cross_entropy",
        )
        q_sleep_state = _unreplicate_tree(q_sleep_state_replicated)
        q_smc_state = _unreplicate_tree(q_smc_state_replicated)
        state = AdaptiveTrainingState(
            model=model,
            q_sleep_optimizer_state=q_sleep_state,
            q_smc_optimizer_state=q_smc_state,
            prior_optimizer_state=prior_state,
            bootstrap_epoch=jnp.asarray(
                training.bootstrap_sleep_epochs, dtype=jnp.int32
            ),
            observed_sweep=jnp.asarray(sweep, dtype=jnp.int32),
            smc_updates=jnp.asarray(smc_update_count, dtype=jnp.int32),
            prior_updates=jnp.asarray(prior_update_count, dtype=jnp.int32),
            random_key=key,
        )
        save_adaptive_training_state(
            out / "checkpoints" / "training_state_last.eqx",
            state,
            config=runtime.config,
            latent_spec=runtime.latent_spec,
        )
        _write_progress_tables(out, log_rows, validation_rows, prior_rows, hard_rows)
        _log(
            verbose,
            "[adaptive-smc-train] validation "
            f"sweep={sweep} CE={validation_ce:.5f} "
            f"qIS={validation['validation_q_is_ess_fraction']:.4f} "
            f"evidence={validation['validation_selected_log_evidence']:.5f}",
        )
        if pad_count:
            _log(verbose, f"[adaptive-smc-train] observed padding={pad_count}")

    if not validation_rows:
        model = _unreplicate_tree(model_replicated)
        key, validation_key = jax.random.split(key)
        validation, _ = _run_validation(
            model=model,
            model_replicated=model_replicated,
            runtime=runtime,
            training_config=training,
            primary_step=primary_step,
            fallback_step=fallback_step,
            n_devices=n_devices,
            key=validation_key,
            sweep=0,
        )
        validation_rows.append(validation)
    receipt = _final_training_receipt(
        runtime=runtime,
        training=training,
        primary=primary_config,
        fallback=fallback_config,
        proposal=proposal_config,
        validation_rows=validation_rows,
        prior_rows=prior_rows,
        log_rows=log_rows,
        best_cross_entropy=best_cross_entropy,
        best_epoch_label=best_epoch_label,
        elapsed_seconds=time.time() - start_time,
        smoke=smoke,
        alpha_preflight=alpha_preflight,
    )
    _write_progress_tables(out, log_rows, validation_rows, prior_rows, hard_rows)
    write_json(out / "training_receipt.json", receipt)
    _log(
        verbose,
        f"[adaptive-smc-train] complete status={receipt['status']} -> {out}",
    )
    return receipt


def _apply_prior_macro(
    *,
    model_replicated,
    prior_snapshot,
    prior_state,
    prior_optimizer,
    posterior,
    selection_fn,
    training,
    devices,
    key,
):
    model = _unreplicate_tree(model_replicated)
    key, trust_key, selection_key = jax.random.split(key, 3)
    model, prior_state, metrics = apply_prior_macro_update(
        model=model,
        prior_snapshot=prior_snapshot,
        optimizer=prior_optimizer,
        optimizer_state=prior_state,
        posterior=posterior,
        trust_key=trust_key,
        selection_key=selection_key,
        selection_log_alpha_fn=selection_fn,
        trust_samples=training.trust_samples,
        trust_strength=training.trust_strength,
        max_kl_per_dimension=training.max_prior_kl_per_dimension,
        max_alpha_mc_relative_error=training.max_alpha_mc_relative_error,
        gradient_clip_norm=training.gradient_clip_norm,
    )
    payload = {
        field: _host_scalar(getattr(metrics, field))
        for field in PriorUpdateMetrics._fields
    }
    payload["update_applied"] = bool(payload["update_applied"])
    payload["grads_finite"] = bool(payload["grads_finite"])
    payload["rejection_code"] = int(payload["rejection_code"])
    return _replicate_model_for_pmap(model, devices), prior_state, payload, key


def _write_progress_tables(out, log_rows, validation_rows, prior_rows, hard_rows):
    pd.DataFrame(log_rows).to_csv(out / "adaptive_training_log.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(
        out / "adaptive_validation_log.csv", index=False
    )
    pd.DataFrame(prior_rows).to_csv(out / "prior_macro_log.csv", index=False)
    pd.DataFrame(hard_rows).to_csv(out / "hard_object_queue.csv", index=False)
    progress = {
        "training_rows": len(log_rows),
        "validation_rows": len(validation_rows),
        "prior_macro_rows": len(prior_rows),
        "hard_queue_rows": len(hard_rows),
        "last_validation": validation_rows[-1] if validation_rows else None,
        "last_prior_macro": prior_rows[-1] if prior_rows else None,
    }
    write_json(out / "training_progress.json", progress)


def _final_training_receipt(
    *,
    runtime,
    training,
    primary,
    fallback,
    proposal,
    validation_rows,
    prior_rows,
    log_rows,
    best_cross_entropy,
    best_epoch_label,
    elapsed_seconds,
    smoke,
    alpha_preflight,
):
    final = validation_rows[-1]
    applied_prior = [row for row in prior_rows if bool(row.get("update_applied"))]
    alpha_finite = bool(np.isfinite(final["selection_alpha"]))
    alpha_error_ok = (
        float(final["selection_alpha_mc_relative_error"])
        <= training.max_alpha_mc_relative_error
    )
    acceptance = float(final["median_mutation_acceptance"])
    entropy_values = np.asarray(
        [row["posterior_full_entropy_mc"] for row in validation_rows],
        dtype=float,
    )
    entropy_drop = float(entropy_values[0] - entropy_values[-1])
    entropy_drop_limit = 0.5 * len(runtime.parameter_names)
    q_clipped = [
        bool(row.get("q_grad_clipped", row.get("grad_clipped")))
        for row in log_rows
        if row.get("q_grad_clipped", row.get("grad_clipped")) is not None
    ]
    prior_clipped = [
        float(row["raw_grad_norm"]) > training.gradient_clip_norm
        for row in prior_rows
        if np.isfinite(row.get("raw_grad_norm", np.nan))
    ]
    checks = {
        "no_truth_in_training": True,
        "canonical_target_reused": True,
        "median_beta_final_is_one": bool(
            np.isclose(float(final["median_beta_final"]), 1.0, atol=1.0e-6)
        ),
        "median_final_ess_fraction_gt_0p3": bool(
            float(final["median_final_ess_fraction"]) > 0.30
        ),
        "hard_fraction_lt_0p3": bool(
            float(final["hard_fraction_after_fallback"])
            < training.hard_fraction_fail
        ),
        "mutation_acceptance_reasonable": bool(
            np.isfinite(acceptance) and 0.05 <= acceptance <= 0.80
        ),
        "validation_smc_cross_entropy_finite": bool(
            np.isfinite(final["validation_smc_cross_entropy"])
        ),
        "posterior_full_entropy_no_systematic_collapse": bool(
            np.isfinite(entropy_values).all() and entropy_drop <= entropy_drop_limit
        ),
        "prior_update_nonzero": bool(applied_prior),
        "prior_kl_finite": bool(
            applied_prior
            and all(np.isfinite(row["prior_kl_proposed"]) for row in applied_prior)
        ),
        "alpha_finite": alpha_finite,
        "alpha_gradient_preflight_finite_nonzero": bool(
            alpha_preflight["finite"] and alpha_preflight["nonzero"]
        ),
        "alpha_mc_relative_error_acceptable": alpha_error_ok,
        "all_applied_prior_gradients_finite": bool(
            applied_prior and all(row["grads_finite"] for row in applied_prior)
        ),
        "q_smc_update_nonzero": any(
            row.get("phase") == "observed_smc" and row.get("q_update_applied")
            for row in log_rows
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "contract": (
            "no-truth selected-catalog training; q receives sleep and final-SMC "
            "inclusive distillation only; parent prior receives final-SMC M-step "
            "+ log(alpha_eta) + KL trust region"
        ),
        "smoke": bool(smoke),
        "training": asdict(training),
        "primary_smc": asdict(primary),
        "fallback_smc": asdict(fallback),
        "r0": {
            "definition": "0.70 q_T1 + 0.20 q_T1.5 + 0.10 p_eta_snapshot",
            **asdict(proposal),
        },
        "train_objects": int(runtime.train_arrays.flux.shape[0]),
        "validation_objects": int(runtime.validation_arrays.flux.shape[0]),
        "latent_normalization": runtime.latent_spec.normalization,
        "normalization_information_source": "fit_bounds_and_fit_initials_only",
        "population_prior_initialization": "identity_realnvp_standard_normal",
        "truth_used_for_training_or_selection": False,
        "selection_gradient_preflight": alpha_preflight,
        "best_validation_smc_cross_entropy": float(best_cross_entropy),
        "best_checkpoint_label": best_epoch_label,
        "posterior_full_entropy_mc_first": float(entropy_values[0]),
        "posterior_full_entropy_mc_last": float(entropy_values[-1]),
        "posterior_full_entropy_mc_drop": entropy_drop,
        "posterior_full_entropy_mc_drop_limit": entropy_drop_limit,
        "prior_macro_updates_attempted": len(prior_rows),
        "prior_macro_updates_applied": len(applied_prior),
        "q_gradient_clipped_fraction": (
            float(np.mean(q_clipped)) if q_clipped else None
        ),
        "prior_gradient_clipped_fraction": (
            float(np.mean(prior_clipped)) if prior_clipped else None
        ),
        "final_validation": final,
        "elapsed_seconds": float(elapsed_seconds),
        "next_action": (
            "RUN_BIG_TRAINING"
            if smoke and all(checks.values())
            else (
                "RUN_EXACT_POSTERIOR_VALIDATION"
                if (not smoke and all(checks.values()))
                else "STOP_AND_REVIEW_FAILED_SMC_GATE"
            )
        ),
    }


def _replicated_scalar(value) -> float:
    array = np.asarray(jax.device_get(value))
    return float(array.reshape(-1)[0])


def _replicated_bool(value) -> bool:
    array = np.asarray(jax.device_get(value))
    return bool(array.reshape(-1)[0])


def _host_scalar(value):
    array = np.asarray(jax.device_get(value))
    scalar = array.reshape(-1)[0]
    if np.issubdtype(array.dtype, np.bool_):
        return bool(scalar)
    if np.issubdtype(array.dtype, np.integer):
        return int(scalar)
    return float(scalar)


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)
