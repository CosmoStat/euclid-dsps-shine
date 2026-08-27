"""End-to-end truth-free trainer for FENIKS SC-DRWS."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.io import ensure_dir, write_json

from .adaptive_smc_trainer import (
    RuntimeBundle,
    _loss_batch_take,
    _make_selection_log_alpha_fn,
    _pad_loss_batch,
    _replicate_model_for_pmap,
    prepare_adaptive_training_runtime,
)
from .adaptive_smc_training import (
    apply_prior_macro_update,
    make_component_optimizer,
    snapshot_model,
)
from .config import amortized_config, require_amortized_dependencies
from .data import iter_photometry_batches_from_arrays
from .features import feature_stats_hash
from .latent import latent_spec_hash, latent_spec_to_jsonable
from .posterior import posterior_entropy_diagnostics
from .sc_asmc_training import update_ema_encoder
from .sc_drws import (
    C0_SCOPE_STATEMENT,
    HARD_EXPANSION_PROPOSAL,
    JOINT_PROPOSAL,
    WARMUP_PROPOSAL,
    DefensiveImportanceBatch,
    ImportanceDiagnostics,
    SCDrwsSchedule,
    entropy_penalty_factor,
    flow_scale_clamp,
    log_std_floor,
    make_pmap_sc_drws_expansion_step,
    make_pmap_sc_drws_importance_step,
    make_pmap_sc_drws_q_step,
    make_pmap_sc_drws_sleep_step,
    phase_for_epoch,
    prior_support_gate,
    q_weight_temperature,
    update_kind_for_epoch,
)
from .train import LossBatch, _loss_batch, build_amortized_model

eqx, optax = require_amortized_dependencies()


class SCDrwsTrainingState(NamedTuple):
    model: Any
    ema_encoder: Any
    q_warmup_optimizer_state: Any
    q_joint_optimizer_state: Any
    prior_optimizer_state: Any
    epoch: jnp.ndarray
    wake_updates: jnp.ndarray
    prior_updates: jnp.ndarray
    random_key: jax.Array
    reference_entropy: jnp.ndarray


class RWSPosteriorBatch(NamedTuple):
    """Stopped weighted joint draws consumed by the shared prior objective."""

    particles: jnp.ndarray
    normalized_weights: jnp.ndarray
    eligible: jnp.ndarray


def _raw_sc_drws(config: dict[str, Any]) -> dict[str, Any]:
    return dict((config.get("amortized", {}) or {}).get("sc_drws", {}) or {})


def validate_sc_drws_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = amortized_config(config)
    raw = _raw_sc_drws(config)
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(raw.get("acronym") == "SC-DRWS", "workflow acronym must be SC-DRWS")
    require(
        raw.get("c0_scope_statement") == C0_SCOPE_STATEMENT,
        "C0 scope statement must match the publication contract verbatim",
    )
    require(raw.get("truth_allowed") is False, "SC-DRWS training must forbid truth")
    require(
        raw.get("production_entrypoint") == "scripts/train_feniks_sc_drws.py",
        "SC-DRWS must use its dedicated production trainer",
    )
    require(
        raw.get("legacy_generic_rws_trainer_allowed") is False,
        "legacy generic K8 RWS must be disabled",
    )
    require(
        not (config.get("truth", {}) or {}).get("parameter_columns", {}),
        "truth.parameter_columns must be empty",
    )
    require(
        cfg["inference"].get("write_truth_snapshot") is False,
        "SC-DRWS inference must disable truth snapshots",
    )
    require(
        cfg["inference"].get("write_truth_diagnostics") is False,
        "SC-DRWS inference must disable truth diagnostics",
    )
    require(cfg["latent"]["normalization"] == "bounded_mixed_warp", "bounded_mixed_warp is required")
    require(not cfg["latent"].get("normalization_checkpoint"), "truth-fitted normalization is forbidden")
    require(cfg["prior"]["source"] == "joint_realnvp", "parent prior must be joint_realnvp")
    require(not cfg["prior"].get("checkpoint"), "parent prior must start from scratch")
    require(cfg["prior"].get("init") == "identity", "parent prior must use identity initialization")
    require(float(cfg["encoder"]["initial_log_std"]) == 0.25, "initial_log_std must equal +0.25")
    require(float(cfg["encoder"]["flow_init_scale"]) == 0.0, "conditional flow must initialize as identity")
    require(raw.get("phase_a", {}).get("likelihood") == "student_t", "Phase A must use Student-t")
    require(float(raw.get("phase_a", {}).get("student_t_dof", 0.0)) == 2.0, "Phase A dof must equal 2")
    require(raw.get("phase_a", {}).get("prior_frozen") is True, "Phase A prior must be frozen")
    require(raw.get("phase_b", {}).get("likelihood") == "gaussian", "Phase B must use Gaussian likelihood")
    require(raw.get("phase_b", {}).get("prior_frozen") is False, "Phase B prior must be trainable")
    hard = raw.get("hard_mis", {}) or {}
    require(int(hard.get("first_particles", 0)) == 128, "hard MIS first pass must use K=128")
    require(int(hard.get("additional_particles", 0)) == 384, "hard MIS expansion must add 384")
    require(int(hard.get("maximum_particles", 0)) == 512, "hard MIS maximum must be K=512")
    require(float(raw.get("ema", {}).get("decay", 0.0)) in {0.99, 0.995}, "EMA decay must be 0.99 or 0.995")
    selection = cfg["objective"].get("selection_correction", {}) or {}
    sleep_selection = cfg["objective"].get("sleep", {}).get("selection", {}) or {}
    require(float(selection.get("max_mag_ab", 0.0)) == 29.0, "selection correction must use r<29.0")
    require(float(sleep_selection.get("max_mag_ab", 0.0)) == 29.0, "sleep must use observed r<29.0")
    require(selection.get("gradient_estimator") == "score_function", "log alpha must use the existing score-function gradient")
    require(not (config.get("calibration", {}) or {}).get("per_band_zero_points", {}).get("trainable", False), "zero points must remain fixed")
    if errors:
        raise ValueError("invalid SC-DRWS config:\n- " + "\n- ".join(errors))
    return {"status": "PASS", "errors": []}


def schedule_from_config(config: dict[str, Any]) -> SCDrwsSchedule:
    raw = _raw_sc_drws(config)
    values = raw.get("schedule", {}) or {}
    return SCDrwsSchedule(
        warmup_epochs=int(raw["phase_a"]["epochs"]),
        joint_epochs=int(raw["phase_b"]["epochs"]),
        sleep_epochs_per_cycle=int(values["sleep_epochs_per_cycle"]),
        wake_epochs_per_cycle=int(values["wake_epochs_per_cycle"]),
        log_std_floor_start=float(values["log_std_floor_start"]),
        log_std_floor_end=float(values["log_std_floor_end"]),
        log_std_floor_end_epoch=int(values["log_std_floor_end_epoch"]),
        flow_scale_clamp_start=float(values["flow_scale_clamp_start"]),
        flow_scale_clamp_end_epoch=int(values["flow_scale_clamp_end_epoch"]),
        q_weight_temperature_start=float(values["q_weight_temperature_start"]),
        q_weight_temperature_end=float(values["q_weight_temperature_end"]),
        q_weight_temperature_wake_updates=int(values["q_weight_temperature_wake_updates"]),
    )


def train_feniks_sc_drws(
    config: dict[str, Any],
    *,
    out_dir: str | Path,
    train_indices_file: str | Path,
    validation_indices_file: str | Path,
    manifest_file: str | Path,
    resume_state: str | Path | None = None,
    smoke: bool = False,
    require_full_dataset: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train q and the parent prior without reading any latent truth columns."""
    validate_sc_drws_config(config)
    out = ensure_dir(out_dir)
    manifest = _validate_manifest(
        manifest_file,
        train_indices_file=train_indices_file,
        validation_indices_file=validation_indices_file,
        require_full_dataset=require_full_dataset,
    )
    runtime = prepare_adaptive_training_runtime(
        config,
        out,
        train_indices_file=train_indices_file,
        validation_indices_file=validation_indices_file,
    )
    if runtime.train_arrays.truth or runtime.validation_arrays.truth:
        raise RuntimeError("SC-DRWS runtime loaded truth")
    schedule = schedule_from_config(config)
    if smoke:
        schedule = replace(schedule, warmup_epochs=4, joint_epochs=4)
    cfg = amortized_config(config)
    raw = _raw_sc_drws(config)
    optimizer_cfg = raw["optimizer"]
    prior_cfg = raw["prior_update"]
    entropy_cfg = raw["entropy"]
    hard_cfg = raw["hard_mis"]
    final_flow_clamp = float(cfg["encoder"]["flow_scale_clamp"])
    seed = int(cfg["training"].get("seed", 260827))
    key = jax.random.PRNGKey(seed)
    model = build_amortized_model(
        config,
        jax.random.fold_in(key, 0),
        latent_spec=runtime.latent_spec,
    )
    q_warmup_optimizer = make_component_optimizer(
        learning_rate=float(optimizer_cfg["q_warmup_peak_learning_rate"]),
        gradient_clip_norm=float(optimizer_cfg["q_gradient_clip_norm"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    q_joint_optimizer = make_component_optimizer(
        learning_rate=float(optimizer_cfg["q_joint_learning_rate"]),
        gradient_clip_norm=float(optimizer_cfg["q_gradient_clip_norm"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    prior_optimizer = make_component_optimizer(
        learning_rate=float(optimizer_cfg["prior_learning_rate"]),
        gradient_clip_norm=float(optimizer_cfg["prior_gradient_clip_norm"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    q_warmup_state = q_warmup_optimizer.init(eqx.filter(model.encoder, eqx.is_inexact_array))
    q_joint_state = q_joint_optimizer.init(eqx.filter(model.encoder, eqx.is_inexact_array))
    prior_state = prior_optimizer.init(eqx.filter(model.prior, eqx.is_inexact_array))
    state = SCDrwsTrainingState(
        model=model,
        ema_encoder=model.encoder,
        q_warmup_optimizer_state=q_warmup_state,
        q_joint_optimizer_state=q_joint_state,
        prior_optimizer_state=prior_state,
        epoch=jnp.asarray(0, dtype=jnp.int32),
        wake_updates=jnp.asarray(0, dtype=jnp.int32),
        prior_updates=jnp.asarray(0, dtype=jnp.int32),
        random_key=key,
        reference_entropy=jnp.asarray(jnp.nan, dtype=jnp.float32),
    )
    if resume_state is not None:
        state = _load_state(
            resume_state,
            state,
            config=config,
            runtime=runtime,
        )
    model = state.model
    ema_encoder = state.ema_encoder
    q_warmup_state = state.q_warmup_optimizer_state
    q_joint_state = state.q_joint_optimizer_state
    prior_state = state.prior_optimizer_state
    key = state.random_key
    wake_updates = int(np.asarray(state.wake_updates))
    prior_updates = int(np.asarray(state.prior_updates))
    reference_entropy = float(np.asarray(state.reference_entropy))
    start_epoch = int(np.asarray(state.epoch)) + 1
    devices = tuple(jax.local_devices())
    n_devices = len(devices)
    if n_devices <= 0:
        raise RuntimeError("SC-DRWS requires at least one JAX device")
    global_batch = int(cfg["training"].get("jax_batch_size", 64))
    if smoke:
        global_batch = min(global_batch, max(n_devices, 16))
    if global_batch % n_devices:
        raise ValueError("SC-DRWS batch size must be divisible by local devices")
    hard_batch = int(raw["performance"].get("hard_objects_per_gpu_initial", 4)) * n_devices
    phase_a_likelihood = {
        **runtime.likelihood_config,
        "type": "student_t",
        "student_t_dof": 2.0,
    }
    phase_b_likelihood = {**runtime.likelihood_config, "type": "gaussian"}
    warmup_importance = make_pmap_sc_drws_importance_step(
        latent_spec=runtime.jit_latent_spec,
        context=runtime.context,
        model_args=runtime.model_args,
        parameter_names=runtime.parameter_names,
        likelihood_config=phase_a_likelihood,
        calibration_config=runtime.calibration_config,
        n_particles=64,
        proposal=WARMUP_PROPOSAL,
        minimum_ess_fraction=float(hard_cfg["minimum_ess_fraction"]),
        maximum_weight=float(hard_cfg["maximum_weight"]),
    )
    joint_importance = make_pmap_sc_drws_importance_step(
        latent_spec=runtime.jit_latent_spec,
        context=runtime.context,
        model_args=runtime.model_args,
        parameter_names=runtime.parameter_names,
        likelihood_config=phase_b_likelihood,
        calibration_config=runtime.calibration_config,
        n_particles=128,
        proposal=JOINT_PROPOSAL,
        minimum_ess_fraction=float(hard_cfg["minimum_ess_fraction"]),
        maximum_weight=float(hard_cfg["maximum_weight"]),
    )
    expansion_step = make_pmap_sc_drws_expansion_step(
        latent_spec=runtime.jit_latent_spec,
        context=runtime.context,
        model_args=runtime.model_args,
        parameter_names=runtime.parameter_names,
        likelihood_config=phase_b_likelihood,
        calibration_config=runtime.calibration_config,
        first_proposal=JOINT_PROPOSAL,
        additional_proposal=HARD_EXPANSION_PROPOSAL,
        additional_particles=384,
        minimum_ess_fraction=float(hard_cfg["minimum_ess_fraction"]),
        maximum_weight=float(hard_cfg["maximum_weight"]),
    )
    sleep_a_step = make_pmap_sc_drws_sleep_step(
        optimizer=q_warmup_optimizer,
        latent_spec=runtime.jit_latent_spec,
        context=runtime.context,
        model_args=runtime.model_args,
        parameter_names=runtime.parameter_names,
        likelihood_config=phase_a_likelihood,
        calibration_config=runtime.calibration_config,
        objective_config=runtime.sleep_objective_config,
        gradient_clip_norm=float(optimizer_cfg["q_gradient_clip_norm"]),
    )
    sleep_b_step = make_pmap_sc_drws_sleep_step(
        optimizer=q_joint_optimizer,
        latent_spec=runtime.jit_latent_spec,
        context=runtime.context,
        model_args=runtime.model_args,
        parameter_names=runtime.parameter_names,
        likelihood_config=phase_b_likelihood,
        calibration_config=runtime.calibration_config,
        objective_config=runtime.sleep_objective_config,
        gradient_clip_norm=float(optimizer_cfg["q_gradient_clip_norm"]),
    )
    q_a_step = make_pmap_sc_drws_q_step(
        optimizer=q_warmup_optimizer,
        gradient_clip_norm=float(optimizer_cfg["q_gradient_clip_norm"]),
    )
    q_b_step = make_pmap_sc_drws_q_step(
        optimizer=q_joint_optimizer,
        gradient_clip_norm=float(optimizer_cfg["q_gradient_clip_norm"]),
    )
    selection_fn = _make_selection_log_alpha_fn(runtime)
    selection_preflight = _selection_preflight(model, selection_fn, key)
    write_json(out / "selection_gradient_preflight.json", selection_preflight)
    if not selection_preflight["finite_nonzero"]:
        raise RuntimeError("SC-DRWS selection score-function gradient preflight failed")
    model_replicated = _replicate_model_for_pmap(model, devices)
    ema_replicated = _replicate_model_for_pmap(ema_encoder, devices)
    q_warmup_replicated = _replicate_model_for_pmap(q_warmup_state, devices)
    q_joint_replicated = _replicate_model_for_pmap(q_joint_state, devices)
    training_log = out / "sc_drws_training_log.csv"
    prior_log = out / "sc_drws_prior_log.csv"
    hard_log = out / "sc_drws_hard_log.csv"
    start_time = time.time()
    for epoch in range(start_epoch, schedule.total_epochs + 1):
        phase = phase_for_epoch(epoch, schedule)
        update_kind = update_kind_for_epoch(epoch, schedule)
        floor = float(np.asarray(log_std_floor(epoch, schedule)))
        clamp = float(
            np.asarray(
                flow_scale_clamp(
                    epoch,
                    final_value=final_flow_clamp,
                    schedule=schedule,
                )
            )
        )
        phase_a = phase == "robust_warmup"
        q_optimizer_state_replicated = (
            q_warmup_replicated if phase_a else q_joint_replicated
        )
        rng = np.random.default_rng(seed + epoch)
        order = _padded_epoch_order(
            len(runtime.train_arrays.flux), global_batch, rng
        )
        epoch_rows: list[dict[str, Any]] = []
        hard_rows: list[dict[str, Any]] = []
        prior_particles: list[np.ndarray] = []
        prior_weights: list[np.ndarray] = []
        prior_ess: list[np.ndarray] = []
        prior_max_weight: list[np.ndarray] = []
        prior_finite: list[np.ndarray] = []
        prior_unresolved: list[np.ndarray] = []
        wake_prior_snapshot = (
            snapshot_model(model)
            if update_kind == "wake" and not phase_a
            else None
        )
        for batch_index, photometry in enumerate(
            iter_photometry_batches_from_arrays(
                runtime.train_arrays,
                batch_size=global_batch,
                feature_stats=runtime.feature_stats,
                order=order,
                truth_names=None,
            )
        ):
            batch_start_time = time.time()
            key, step_key = jax.random.split(key)
            batch = _loss_batch(photometry)
            sharded_batch = _shard_loss_batch(batch, n_devices)
            if update_kind == "sleep":
                sleep_step = sleep_a_step if phase_a else sleep_b_step
                model_replicated, q_optimizer_state_replicated, metrics, details = (
                    sleep_step(
                        model_replicated,
                        q_optimizer_state_replicated,
                        sharded_batch,
                        jax.random.split(step_key, n_devices),
                        floor,
                        clamp,
                    )
                )
                ema_replicated = update_ema_encoder(
                    ema_replicated,
                    model_replicated.encoder,
                    decay=float(raw["ema"]["decay"]),
                )
                row = {
                    "epoch": epoch,
                    "phase": phase,
                    "update_kind": "sleep",
                    "batch": batch_index,
                    "loss": _scalar(metrics.loss),
                    "q_raw_grad_norm": _scalar(metrics.raw_grad_norm),
                    "q_grad_clipped": bool(_scalar(metrics.grad_clipped)),
                    "q_grads_finite": bool(_scalar(metrics.grads_finite)),
                    "q_update_applied": bool(_scalar(metrics.update_applied)),
                    "log_std_floor": floor,
                    "flow_scale_clamp": clamp,
                    "q_weight_temperature": np.nan,
                    "base_entropy": _metric(details, "posterior_base_entropy"),
                    "flow_residual_logdet": _metric(
                        details, "posterior_residual_logdet_mean"
                    ),
                    "full_entropy": _metric(
                        details, "posterior_full_entropy_mc"
                    ),
                    "entropy_floor_penalty": 0.0,
                    "first_pass_ess_fraction": np.nan,
                    "expanded_ess_fraction": np.nan,
                    "max_weight": np.nan,
                    "expansion_fraction": 0.0,
                    "unresolved_fraction": 0.0,
                    "estimated_dsps_evaluations": int(
                        global_batch
                        * int(
                            runtime.sleep_objective_config.get("sleep", {}).get(
                                "selection_candidate_factor", 1
                            )
                        )
                    ),
                }
            else:
                wake_updates += 1
                tau = float(np.asarray(q_weight_temperature(wake_updates, schedule)))
                importance_step = warmup_importance if phase_a else joint_importance
                first_sharded = importance_step(
                    model_replicated,
                    sharded_batch,
                    jax.random.split(step_key, n_devices),
                    floor,
                    clamp,
                )
                first = _unshard_importance(first_sharded)
                packed = _pack_first_pass(first, maximum_particles=64 if phase_a else 512)
                hard_indices = (
                    np.empty(0, dtype=np.int64)
                    if phase_a
                    else np.flatnonzero(np.asarray(first.diagnostics.hard))
                )
                if len(hard_indices):
                    packed, hard_batch_rows = _expand_hard_objects(
                        model_replicated=model_replicated,
                        batch=batch,
                        first=first,
                        hard_indices=hard_indices,
                        expansion_step=expansion_step,
                        key=jax.random.fold_in(step_key, 991),
                        floor=floor,
                        clamp=clamp,
                        n_devices=n_devices,
                        hard_batch_size=hard_batch,
                        packed=packed,
                    )
                    hard_rows.extend(
                        {
                            "epoch": epoch,
                            "batch": batch_index,
                            "row_index": int(
                                np.asarray(photometry.row_index)[
                                    item["batch_object"]
                                ]
                            ),
                            **item,
                        }
                        for item in hard_batch_rows
                    )
                joint_epoch = epoch - schedule.warmup_epochs
                entropy_factor = (
                    0.0
                    if phase_a or not np.isfinite(reference_entropy)
                    else entropy_penalty_factor(joint_epoch, schedule)
                )
                packed_sharded = _shard_packed(packed, n_devices)
                q_step = q_a_step if phase_a else q_b_step
                model_replicated, q_optimizer_state_replicated, metrics, details = q_step(
                    model_replicated,
                    q_optimizer_state_replicated,
                    _shard_features(batch.features, n_devices),
                    packed_sharded[0],
                    packed_sharded[1],
                    packed_sharded[2],
                    jax.random.split(jax.random.fold_in(step_key, 992), n_devices),
                    tau,
                    floor,
                    clamp,
                    reference_entropy if np.isfinite(reference_entropy) else 0.0,
                    float(entropy_cfg["margin"]),
                    float(entropy_cfg["penalty_strength"]),
                    entropy_factor,
                )
                ema_replicated = update_ema_encoder(
                    ema_replicated,
                    model_replicated.encoder,
                    decay=float(raw["ema"]["decay"]),
                )
                if not phase_a:
                    _append_prior_reservoir(
                        packed,
                        prior_particles,
                        prior_weights,
                        prior_ess,
                        prior_max_weight,
                        prior_finite,
                        prior_unresolved,
                    )
                row = {
                    "epoch": epoch,
                    "phase": phase,
                    "update_kind": "wake",
                    "batch": batch_index,
                    "loss": _scalar(metrics.loss),
                    "q_raw_grad_norm": _scalar(metrics.raw_grad_norm),
                    "q_grad_clipped": bool(_scalar(metrics.grad_clipped)),
                    "q_grads_finite": bool(_scalar(metrics.grads_finite)),
                    "q_update_applied": bool(_scalar(metrics.update_applied)),
                    "log_std_floor": floor,
                    "flow_scale_clamp": clamp,
                    "q_weight_temperature": tau,
                    "base_entropy": _metric(details, "posterior_base_entropy"),
                    "flow_residual_logdet": _metric(
                        details, "posterior_residual_logdet_mean"
                    ),
                    "full_entropy": _metric(
                        details, "posterior_full_entropy_mc"
                    ),
                    "entropy_floor_penalty": _metric(
                        details, "entropy_floor_penalty"
                    ),
                    "first_pass_ess_fraction": float(
                        np.nanmedian(np.asarray(first.diagnostics.ess_fraction))
                    ),
                    "expanded_ess_fraction": float(
                        np.nanmedian(np.asarray(packed["ess_fraction"]))
                    ),
                    "max_weight": float(
                        np.nanmedian(np.asarray(packed["max_weight"]))
                    ),
                    "expansion_fraction": float(
                        np.mean(np.asarray(packed["expanded"]))
                    ),
                    "unresolved_fraction": float(
                        np.mean(np.asarray(packed["unresolved"]))
                    ),
                    "estimated_dsps_evaluations": int(
                        global_batch * (64 if phase_a else 128)
                        + len(hard_indices) * (0 if phase_a else 384)
                    ),
                }
            row["batch_elapsed_seconds"] = time.time() - batch_start_time
            row["estimated_dsps_evaluations_per_second"] = (
                row["estimated_dsps_evaluations"]
                / max(row["batch_elapsed_seconds"], 1.0e-9)
            )
            epoch_rows.append(row)
            if verbose:
                print(
                    "[sc-drws] "
                    f"epoch={epoch}/{schedule.total_epochs} phase={phase} "
                    f"kind={update_kind} batch={batch_index + 1}/{len(order) // global_batch} "
                    f"loss={row['loss']:.5f} ess={row['expanded_ess_fraction']:.4f} "
                    f"hard={row['expansion_fraction']:.3f} unresolved={row['unresolved_fraction']:.3f}",
                    flush=True,
                )
        if phase_a:
            q_warmup_replicated = q_optimizer_state_replicated
        else:
            q_joint_replicated = q_optimizer_state_replicated
        _append_csv(training_log, epoch_rows)
        _append_csv(hard_log, hard_rows)
        model = _unreplicate(model_replicated)
        ema_encoder = _unreplicate(ema_replicated)
        q_warmup_state = _unreplicate(q_warmup_replicated)
        q_joint_state = _unreplicate(q_joint_replicated)
        if epoch == schedule.warmup_epochs:
            reference_entropy = _heldout_entropy_reference(
                eqx.tree_at(lambda tree: tree.encoder, model, ema_encoder),
                runtime,
                key=jax.random.fold_in(key, 771),
            )
            write_json(
                out / "warmup_entropy_reference.json",
                {
                    "median_full_entropy": reference_entropy,
                    "epoch": epoch,
                    "truth_used": False,
                },
            )
        if update_kind == "wake" and not phase_a and prior_particles:
            assert wake_prior_snapshot is not None
            all_particles = np.concatenate(prior_particles, axis=1)
            all_weights = np.concatenate(prior_weights, axis=1)
            all_ess = np.concatenate(prior_ess)
            all_max_weight = np.concatenate(prior_max_weight)
            all_finite = np.concatenate(prior_finite)
            all_unresolved = np.concatenate(prior_unresolved)
            macro_size = int(prior_cfg["macro_objects"])
            prior_rows = []
            gates = []
            macro_key = jax.random.fold_in(key, epoch + 8800)
            for macro_index, (start, stop) in enumerate(
                _macro_slices(
                    all_particles.shape[1],
                    macro_size,
                    int(prior_cfg["minimum_finite_objects"]),
                )
            ):
                gate, prior_state, model, prior_updates, rows = (
                    _apply_prior_updates(
                        model=model,
                        optimizer=prior_optimizer,
                        optimizer_state=prior_state,
                        selection_fn=selection_fn,
                        particles=all_particles[:, start:stop],
                        weights=all_weights[:, start:stop],
                        ess_fraction=all_ess[start:stop],
                        max_weight=all_max_weight[start:stop],
                        finite=all_finite[start:stop],
                        unresolved=all_unresolved[start:stop],
                        prior_cfg=prior_cfg,
                        key=macro_key,
                        prior_updates=prior_updates,
                        epoch=epoch,
                        macro_index=macro_index,
                        prior_snapshot=wake_prior_snapshot,
                    )
                )
                gates.append(gate)
                prior_rows.extend(rows)
            _append_csv(prior_log, prior_rows)
            if verbose:
                print(
                    "[sc-drws] prior "
                    f"epoch={epoch} macros={len(gates)} "
                    f"accepted={sum(item.accepted for item in gates)} "
                    f"updates={prior_updates}",
                    flush=True,
                )
            model_replicated = _replicate_model_for_pmap(model, devices)
            ema_replicated = _replicate_model_for_pmap(ema_encoder, devices)
            q_warmup_replicated = _replicate_model_for_pmap(q_warmup_state, devices)
            q_joint_replicated = _replicate_model_for_pmap(q_joint_state, devices)
        checkpoint_every = int(raw["checkpoint"]["every_epochs"])
        if epoch % checkpoint_every == 0 or epoch == schedule.total_epochs:
            _save_components(
                out / "checkpoints" / f"epoch_{epoch:04d}",
                model=model,
                ema_encoder=ema_encoder,
                config=config,
                runtime=runtime,
                epoch=epoch,
                reference_entropy=reference_entropy,
            )
        state = SCDrwsTrainingState(
            model=model,
            ema_encoder=ema_encoder,
            q_warmup_optimizer_state=q_warmup_state,
            q_joint_optimizer_state=q_joint_state,
            prior_optimizer_state=prior_state,
            epoch=jnp.asarray(epoch, dtype=jnp.int32),
            wake_updates=jnp.asarray(wake_updates, dtype=jnp.int32),
            prior_updates=jnp.asarray(prior_updates, dtype=jnp.int32),
            random_key=key,
            reference_entropy=jnp.asarray(reference_entropy, dtype=jnp.float32),
        )
        _save_state(
            out / "states" / "latest.eqx",
            state,
            config=config,
            runtime=runtime,
        )
    final_components = _save_components(
        out / "checkpoints" / "final",
        model=model,
        ema_encoder=ema_encoder,
        config=config,
        runtime=runtime,
        epoch=schedule.total_epochs,
        reference_entropy=reference_entropy,
    )
    final_log_alpha, final_selection_metrics = selection_fn(
        model, jax.random.fold_in(key, 99001)
    )
    receipt = {
        "status": "TRAINING_COMPLETE_PENDING_SUPPORT_SELECTION",
        "workflow": "Selection-Corrected Defensive Reweighted Wake-Sleep",
        "acronym": "SC-DRWS",
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "target_population": "p_eta(theta | C0)",
        "selected_population": "beta(theta) p_eta(theta | C0) / alpha_eta",
        "observed_selection": "A = 1[m_r_observed < 29.0]",
        "upstream_true_space_selection": "conditioned_as_C0_not_inverted",
        "truth_used_for_training_validation_or_checkpoint_selection": False,
        "selection_in_object_weights": False,
        "phase_schedule": asdict(schedule),
        "wake_updates": wake_updates,
        "prior_updates": prior_updates,
        "selection_log_alpha": float(np.asarray(final_log_alpha)),
        "selection_alpha": float(np.asarray(jnp.exp(final_log_alpha))),
        "selection_alpha_mc_relative_error": float(
            np.asarray(
                final_selection_metrics.get(
                    "selection/alpha_mc_relative_error", jnp.nan
                )
            )
        ),
        "reference_entropy": reference_entropy,
        "raw_q_checkpoint": final_components["raw_q"],
        "ema_q_checkpoint": final_components["ema_q"],
        "prior_checkpoint": final_components["prior"],
        "raw_model_checkpoint": final_components["raw_model"],
        "ema_model_checkpoint": final_components["ema_model"],
        "raw_and_ema_require_independent_k2048_evaluation": True,
        "manifest_sha256": _sha256(Path(manifest_file)),
        "selected_training_rows": int(len(runtime.train_arrays.flux)),
        "full_dataset_contract_required": bool(require_full_dataset),
        "full_dataset_expected_rows": int(
            manifest["final_full_dataset_contract"]["expected_rows"]
        ),
        "elapsed_seconds": time.time() - start_time,
    }
    write_json(out / "training_receipt.json", receipt)
    (out / "TRAINING_COMPLETE").touch()
    return receipt


def _selection_preflight(model, selection_fn, key) -> dict[str, Any]:
    def objective(prior):
        candidate = eqx.tree_at(lambda tree: tree.prior, model, prior)
        return selection_fn(candidate, key)[0]

    value, grads = eqx.filter_value_and_grad(objective)(model.prior)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grads) if eqx.is_array(leaf)]
    finite = all(bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in leaves)
    norm = float(np.sqrt(sum(np.sum(np.asarray(leaf) ** 2) for leaf in leaves)))
    return {
        "gradient_estimator": "score_function",
        "log_alpha": float(np.asarray(value)),
        "gradient_norm": norm,
        "finite_nonzero": bool(finite and np.isfinite(norm) and norm > 0.0),
        "truth_used": False,
    }


def _validate_manifest(
    path,
    *,
    train_indices_file,
    validation_indices_file,
    require_full_dataset,
):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("c0_scope_statement") != C0_SCOPE_STATEMENT:
        raise ValueError("manifest C0 scope statement mismatch")
    if payload.get("truth_used_for_training_or_checkpoint_selection") is not False:
        raise ValueError("manifest is not truth-free")
    if float(payload["selection"]["max_mag_ab"]) != 29.0:
        raise ValueError("SC-DRWS manifest must use observed r<29.0")
    if float(payload["selection"]["configured_train_retained_fraction"]) < 0.90:
        raise ValueError("SC-DRWS manifest retains less than 90% of C0")
    records = payload.get("manifests", {})

    def validate_indices(candidate, *, required_label=None):
        candidate = Path(candidate)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        matches = [
            (label, record)
            for label, record in records.items()
            if Path(record["path"]).name == candidate.name
        ]
        if len(matches) != 1:
            raise ValueError(f"manifest has no unique record for {candidate.name}")
        label, record = matches[0]
        if required_label is not None and label != required_label:
            raise ValueError(
                f"expected manifest {required_label}, received {label}"
            )
        if record["sha256"] != _sha256(candidate):
            raise ValueError(f"manifest hash mismatch for {candidate.name}")
        values = np.load(candidate, allow_pickle=False)
        if len(values) != int(record["count"]):
            raise ValueError(f"manifest row-count mismatch for {candidate.name}")
        return values

    indices = validate_indices(
        train_indices_file,
        required_label="full_train" if require_full_dataset else None,
    )
    validate_indices(validation_indices_file, required_label="validation")
    if require_full_dataset:
        expected = int(payload["final_full_dataset_contract"]["expected_rows"])
        if len(indices) != expected:
            raise ValueError(
                f"full SC-DRWS run has {len(indices)} rows, expected {expected}"
            )
    return payload


def _padded_epoch_order(size: int, batch_size: int, rng) -> np.ndarray:
    order = rng.permutation(int(size))
    missing = (-len(order)) % int(batch_size)
    if missing:
        order = np.concatenate((order, rng.choice(order, missing, replace=True)))
    return order


def _shard_loss_batch(batch: LossBatch, devices: int) -> LossBatch:
    objects = int(batch.features.shape[0])
    local = objects // int(devices)
    if local * int(devices) != objects:
        raise ValueError("loss batch is not divisible by devices")
    return LossBatch(
        *(jnp.asarray(value).reshape(devices, local, *value.shape[1:]) for value in batch)
    )


def _shard_features(features, devices):
    values = jnp.asarray(features)
    return values.reshape(devices, values.shape[0] // devices, values.shape[-1])


def _unshard_particle_object(value):
    array = np.asarray(jax.device_get(value))
    devices, particles, local = array.shape[:3]
    trailing = array.shape[3:]
    return array.transpose(1, 0, 2, *range(3, array.ndim)).reshape(
        particles, devices * local, *trailing
    )


def _unshard_objects(value):
    return np.asarray(jax.device_get(value)).reshape(-1)


def _unshard_diagnostics(value: ImportanceDiagnostics) -> ImportanceDiagnostics:
    return ImportanceDiagnostics(
        normalized_weights=jnp.asarray(_unshard_particle_object(value.normalized_weights)),
        logweight=jnp.asarray(_unshard_particle_object(value.logweight)),
        ess=jnp.asarray(_unshard_objects(value.ess)),
        ess_fraction=jnp.asarray(_unshard_objects(value.ess_fraction)),
        max_weight=jnp.asarray(_unshard_objects(value.max_weight)),
        finite=jnp.asarray(_unshard_objects(value.finite)),
        hard=jnp.asarray(_unshard_objects(value.hard)),
    )


def _unshard_importance(value: DefensiveImportanceBatch) -> DefensiveImportanceBatch:
    return DefensiveImportanceBatch(
        particles=jnp.asarray(_unshard_particle_object(value.particles)),
        logproposal=jnp.asarray(_unshard_particle_object(value.logproposal)),
        diagnostics=_unshard_diagnostics(value.diagnostics),
    )


def _shard_importance(value: DefensiveImportanceBatch, devices: int):
    objects = value.particles.shape[1]
    local = objects // devices

    def particle(array):
        values = jnp.asarray(array)
        return values.reshape(values.shape[0], devices, local, *values.shape[2:]).transpose(
            1, 0, 2, *range(3, values.ndim + 1)
        )

    def object_array(array):
        values = jnp.asarray(array)
        return values.reshape(devices, local, *values.shape[1:])

    return DefensiveImportanceBatch(
        particles=particle(value.particles),
        logproposal=particle(value.logproposal),
        diagnostics=ImportanceDiagnostics(
            normalized_weights=particle(value.diagnostics.normalized_weights),
            logweight=particle(value.diagnostics.logweight),
            ess=object_array(value.diagnostics.ess),
            ess_fraction=object_array(value.diagnostics.ess_fraction),
            max_weight=object_array(value.diagnostics.max_weight),
            finite=object_array(value.diagnostics.finite),
            hard=object_array(value.diagnostics.hard),
        ),
    )


def _pack_first_pass(first, *, maximum_particles):
    particles, objects, latent = first.particles.shape
    output_particles = np.zeros((maximum_particles, objects, latent), dtype=np.float32)
    output_weights = np.zeros((maximum_particles, objects), dtype=np.float32)
    output_particles[:particles] = np.asarray(first.particles)
    output_weights[:particles] = np.asarray(first.diagnostics.normalized_weights)
    return {
        "particles": output_particles,
        "weights": output_weights,
        "q_eligible": np.array(first.diagnostics.finite, dtype=bool, copy=True),
        "prior_eligible": np.array(
            first.diagnostics.finite & ~first.diagnostics.hard,
            dtype=bool,
            copy=True,
        ),
        "finite": np.array(first.diagnostics.finite, dtype=bool, copy=True),
        "ess_fraction": np.array(
            first.diagnostics.ess_fraction, dtype=float, copy=True
        ),
        "max_weight": np.array(
            first.diagnostics.max_weight, dtype=float, copy=True
        ),
        "expanded": np.zeros(objects, dtype=bool),
        "unresolved": np.array(first.diagnostics.hard, dtype=bool, copy=True),
    }


def _expand_hard_objects(
    *, model_replicated, batch, first, hard_indices, expansion_step, key,
    floor, clamp, n_devices, hard_batch_size, packed,
):
    rows = []
    for start in range(0, len(hard_indices), hard_batch_size):
        selected = np.asarray(hard_indices[start : start + hard_batch_size], dtype=np.int64)
        selected_batch = _loss_batch_take(batch, selected)
        selected_first = DefensiveImportanceBatch(
            particles=jnp.take(first.particles, selected, axis=1),
            logproposal=jnp.take(first.logproposal, selected, axis=1),
            diagnostics=ImportanceDiagnostics(
                normalized_weights=jnp.take(first.diagnostics.normalized_weights, selected, axis=1),
                logweight=jnp.take(first.diagnostics.logweight, selected, axis=1),
                ess=jnp.take(first.diagnostics.ess, selected),
                ess_fraction=jnp.take(first.diagnostics.ess_fraction, selected),
                max_weight=jnp.take(first.diagnostics.max_weight, selected),
                finite=jnp.take(first.diagnostics.finite, selected),
                hard=jnp.take(first.diagnostics.hard, selected),
            ),
        )
        target = max(n_devices, int(np.ceil(len(selected) / n_devices)) * n_devices)
        pad_indices = np.arange(target) % len(selected)
        padded_batch, _ = _pad_loss_batch(selected_batch, target)
        selected_first = DefensiveImportanceBatch(
            particles=jnp.take(selected_first.particles, pad_indices, axis=1),
            logproposal=jnp.take(selected_first.logproposal, pad_indices, axis=1),
            diagnostics=ImportanceDiagnostics(
                normalized_weights=jnp.take(selected_first.diagnostics.normalized_weights, pad_indices, axis=1),
                logweight=jnp.take(selected_first.diagnostics.logweight, pad_indices, axis=1),
                ess=jnp.take(selected_first.diagnostics.ess, pad_indices),
                ess_fraction=jnp.take(selected_first.diagnostics.ess_fraction, pad_indices),
                max_weight=jnp.take(selected_first.diagnostics.max_weight, pad_indices),
                finite=jnp.take(selected_first.diagnostics.finite, pad_indices),
                hard=jnp.take(selected_first.diagnostics.hard, pad_indices),
            ),
        )
        result = expansion_step(
            model_replicated,
            _shard_loss_batch(padded_batch, n_devices),
            jax.random.split(jax.random.fold_in(key, start), n_devices),
            _shard_importance(selected_first, n_devices),
            floor,
            clamp,
        )
        particles = _unshard_particle_object(result.expanded_particles)[:, : len(selected)]
        diagnostics = _unshard_diagnostics(result.expanded_diagnostics)
        weights = np.asarray(diagnostics.normalized_weights)[:, : len(selected)]
        packed["particles"][:, selected] = particles
        packed["weights"][:, selected] = weights
        packed["ess_fraction"][selected] = np.asarray(diagnostics.ess_fraction)[: len(selected)]
        packed["max_weight"][selected] = np.asarray(diagnostics.max_weight)[: len(selected)]
        packed["finite"][selected] = np.asarray(diagnostics.finite)[: len(selected)]
        packed["expanded"][selected] = True
        packed["unresolved"][selected] = np.asarray(diagnostics.hard)[: len(selected)]
        packed["prior_eligible"][selected] = (
            np.asarray(diagnostics.finite)[: len(selected)]
            & ~np.asarray(diagnostics.hard)[: len(selected)]
        )
        for index, row_index in enumerate(selected):
            rows.append(
                {
                    "batch_object": int(row_index),
                    "first_ess_fraction": float(np.asarray(first.diagnostics.ess_fraction)[row_index]),
                    "first_max_weight": float(np.asarray(first.diagnostics.max_weight)[row_index]),
                    "expanded_ess_fraction": float(np.asarray(diagnostics.ess_fraction)[index]),
                    "expanded_max_weight": float(np.asarray(diagnostics.max_weight)[index]),
                    "unresolved": bool(np.asarray(diagnostics.hard)[index]),
                }
            )
    return packed, rows


def _shard_packed(packed, devices):
    particles = jnp.asarray(packed["particles"])
    weights = jnp.asarray(packed["weights"])
    objects = particles.shape[1]
    local = objects // devices
    particle_sharded = particles.reshape(
        particles.shape[0], devices, local, particles.shape[-1]
    ).transpose(1, 0, 2, 3)
    weight_sharded = weights.reshape(weights.shape[0], devices, local).transpose(1, 0, 2)
    mask_sharded = jnp.asarray(packed["q_eligible"]).reshape(devices, local)
    return particle_sharded, weight_sharded, mask_sharded


def _append_prior_reservoir(packed, *lists):
    particle_list, weight_list, ess_list, max_list, finite_list, unresolved_list = lists
    particle_list.append(np.asarray(packed["particles"]))
    weight_list.append(np.asarray(packed["weights"]))
    ess_list.append(np.asarray(packed["ess_fraction"]))
    max_list.append(np.asarray(packed["max_weight"]))
    finite_list.append(np.asarray(packed["finite"]))
    unresolved_list.append(np.asarray(packed["unresolved"]))


def _macro_slices(total: int, size: int, minimum_tail: int):
    if min(int(total), int(size), int(minimum_tail)) <= 0:
        raise ValueError("prior macro dimensions must be positive")
    starts = list(range(0, int(total), int(size)))
    if len(starts) > 1 and int(total) - starts[-1] < int(minimum_tail):
        starts.pop()
    return [
        (start, starts[index + 1] if index + 1 < len(starts) else int(total))
        for index, start in enumerate(starts)
    ]


def _posterior_for_prior(particles, weights, eligible):
    return RWSPosteriorBatch(
        particles=jnp.asarray(particles),
        normalized_weights=jnp.asarray(weights),
        eligible=jnp.asarray(eligible),
    )


def _apply_prior_updates(
    *, model, optimizer, optimizer_state, selection_fn, particles, weights,
    ess_fraction, max_weight, finite, unresolved, prior_cfg, key,
    prior_updates, epoch, macro_index=0, prior_snapshot=None,
):
    gate = prior_support_gate(
        ess_fraction=ess_fraction,
        max_weight=max_weight,
        finite=finite,
        unresolved=unresolved,
        minimum_finite_objects=int(prior_cfg["minimum_finite_objects"]),
        minimum_median_ess_fraction=float(prior_cfg["minimum_median_ess_fraction"]),
        maximum_median_weight=float(prior_cfg["maximum_median_weight"]),
        maximum_unresolved_fraction=float(prior_cfg.get("maximum_unresolved_fraction", 0.02)),
    )
    rows = []
    eligible = np.asarray(finite) & ~np.asarray(unresolved)
    posterior = _posterior_for_prior(particles, weights, eligible)
    if not gate.accepted:
        rows.append(
            {
                "epoch": epoch,
                "macro": macro_index,
                "step": 0,
                "gate_accepted": False,
                "rejection_reason": gate.reason,
                "eligible_objects": gate.finite_objects,
                "median_ess_fraction": gate.median_ess_fraction,
                "median_max_weight": gate.median_max_weight,
                "unresolved_fraction": gate.unresolved_fraction,
                "loss": np.nan,
                "data_nll": np.nan,
                "log_alpha": np.nan,
                "alpha": np.nan,
                "selection_gradient_finite": False,
                "data_gradient_finite": False,
                "trust_gradient_finite": False,
                "proposed_kl": np.nan,
                "update_applied": False,
            }
        )
        return gate, optimizer_state, model, prior_updates, rows
    e_step_prior_snapshot = (
        snapshot_model(model) if prior_snapshot is None else prior_snapshot
    )
    trust_key = jax.random.fold_in(key, 0)
    selection_key = jax.random.fold_in(key, 1)
    for step in range(int(prior_cfg["updates_per_macro"])):
        model, optimizer_state, metrics = apply_prior_macro_update(
            model=model,
            prior_snapshot=e_step_prior_snapshot,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            posterior=posterior,
            trust_key=trust_key,
            selection_key=selection_key,
            selection_log_alpha_fn=selection_fn,
            trust_samples=int(prior_cfg["trust_samples"]),
            trust_strength=float(prior_cfg["trust_strength"]),
            max_kl_per_dimension=float(prior_cfg["maximum_kl_per_dimension"]),
            max_alpha_mc_relative_error=float(prior_cfg["maximum_alpha_mc_relative_error"]),
            gradient_clip_norm=5.0,
        )
        applied = bool(np.asarray(metrics.update_applied))
        prior_updates += int(applied)
        rows.append({
            "epoch": epoch, "macro": macro_index,
            "step": step + 1, "gate_accepted": True,
            "rejection_reason": int(np.asarray(metrics.rejection_code)),
            "eligible_objects": gate.finite_objects,
            "median_ess_fraction": gate.median_ess_fraction,
            "median_max_weight": gate.median_max_weight,
            "unresolved_fraction": gate.unresolved_fraction,
            "loss": float(np.asarray(metrics.loss)),
            "data_nll": float(np.asarray(metrics.data_nll)),
            "log_alpha": float(np.asarray(metrics.selection_log_alpha)),
            "alpha": float(np.asarray(metrics.selection_alpha)),
            "selection_gradient_finite": bool(np.asarray(metrics.selection_grads_finite)),
            "data_gradient_finite": bool(np.asarray(metrics.data_grads_finite)),
            "trust_gradient_finite": bool(np.asarray(metrics.trust_grads_finite)),
            "proposed_kl": float(np.asarray(metrics.prior_kl_proposed)),
            "update_applied": applied,
        })
        if not applied:
            break
    return gate, optimizer_state, model, prior_updates, rows


def _heldout_entropy_reference(model, runtime: RuntimeBundle, *, key):
    values = []
    batch_size = min(128, len(runtime.validation_arrays.flux))
    for index, batch in enumerate(iter_photometry_batches_from_arrays(
        runtime.validation_arrays,
        batch_size=batch_size,
        feature_stats=runtime.feature_stats,
        truth_names=None,
    )):
        metrics = posterior_entropy_diagnostics(
            model, batch.features, jax.random.fold_in(key, index), n_samples=4
        )
        values.append(float(np.asarray(metrics["posterior_full_entropy_mc"])))
    return float(np.median(values))


def _save_components(path, *, model, ema_encoder, config, runtime, epoch, reference_entropy):
    path.mkdir(parents=True, exist_ok=True)
    values = {
        "raw_q": model.encoder,
        "ema_q": ema_encoder,
        "prior": model.prior,
        "raw_model": model,
        "ema_model": eqx.tree_at(lambda tree: tree.encoder, model, ema_encoder),
    }
    records = {}
    for name, value in values.items():
        destination = path / f"{name}.eqx"
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        eqx.tree_serialise_leaves(temporary, value)
        os.replace(temporary, destination)
        sidecar = {
            "component": name,
            "epoch": int(epoch),
            "sha256": _sha256(destination),
            "config_hash": _config_hash(config),
            "latent_spec": latent_spec_to_jsonable(runtime.latent_spec),
            "latent_spec_hash": latent_spec_hash(runtime.latent_spec),
            "latent_transform_hash": latent_spec_hash(runtime.latent_spec),
            "feature_stats_hash": feature_stats_hash(runtime.feature_stats),
            "c0_scope_statement": C0_SCOPE_STATEMENT,
            "selection": "observed r<29.0",
            "truth_used": False,
            "reference_entropy": float(reference_entropy),
            "support_selection_status": "pending_independent_k2048_evaluation",
        }
        write_json(destination.with_suffix(".eqx.json"), sidecar)
        records[name] = {"path": str(destination.resolve()), **sidecar}
    return records


def _save_state(path, state, *, config, runtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    eqx.tree_serialise_leaves(temporary, state)
    os.replace(temporary, path)
    write_json(path.with_suffix(".eqx.json"), {
        "sha256": _sha256(path),
        "config_hash": _config_hash(config),
        "latent_transform_hash": latent_spec_hash(runtime.latent_spec),
        "feature_stats_hash": feature_stats_hash(runtime.feature_stats),
        "epoch": int(np.asarray(state.epoch)),
        "truth_used": False,
    })


def _load_state(path, template, *, config, runtime):
    path = Path(path)
    sidecar = json.loads(path.with_suffix(".eqx.json").read_text())
    checks = (
        sidecar.get("sha256") == _sha256(path),
        sidecar.get("config_hash") == _config_hash(config),
        sidecar.get("latent_transform_hash") == latent_spec_hash(runtime.latent_spec),
        sidecar.get("feature_stats_hash") == feature_stats_hash(runtime.feature_stats),
        sidecar.get("truth_used") is False,
    )
    if not all(checks):
        raise ValueError("SC-DRWS resume state provenance mismatch")
    return eqx.tree_deserialise_leaves(path, template)


def _config_hash(config):
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unreplicate(tree):
    return jax.tree_util.tree_map(
        lambda value: value[0] if eqx.is_array(value) else value, tree
    )


def _scalar(value):
    array = np.asarray(jax.device_get(value))
    return float(array.reshape(-1)[0])


def _metric(mapping, name):
    return _scalar(mapping[name]) if name in mapping else float("nan")


def _append_csv(path, rows):
    if not rows:
        return
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
