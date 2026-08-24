"""Training primitives for the staged FENIKS SC-ASMC-EM workflow."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.io import ensure_dir, write_json

from .adaptive_smc_trainer import (
    RuntimeBundle,
    _make_sleep_loss_fn,
    prepare_adaptive_training_runtime,
)
from .adaptive_smc_training import make_pmap_q_sleep_step
from .config import require_amortized_dependencies
from .data import iter_photometry_batches_from_arrays
from .features import feature_stats_hash
from .latent import latent_spec_hash
from .posterior import posterior_entropy_diagnostics
from .posterior_bank import C0_SCOPE_STATEMENT, sha256_file
from .sc_asmc_config import (
    sc_asmc_em_config_hash,
    sc_asmc_em_schedule,
    validate_sc_asmc_em_config,
)
from .train import LossBatch, _loss_batch, build_amortized_model

eqx, optax = require_amortized_dependencies()


class SleepTrainingState(NamedTuple):
    model: Any
    ema_encoder: Any
    optimizer_state: Any
    epoch: jnp.ndarray
    optimizer_step: jnp.ndarray
    random_key: jax.Array
    best_validation_nll: jnp.ndarray
    best_epoch: jnp.ndarray


def prepare_sc_runtime(
    config: dict[str, Any],
    out_dir: str | Path,
    *,
    feature_train_rows: str | Path,
    heldout_rows: str | Path,
) -> RuntimeBundle:
    """Prepare DSPS and observed arrays under the final no-truth contract."""
    validate_sc_asmc_em_config(config)
    runtime = prepare_adaptive_training_runtime(
        config,
        ensure_dir(out_dir),
        train_indices_file=feature_train_rows,
        validation_indices_file=heldout_rows,
    )
    if runtime.train_arrays.truth or runtime.validation_arrays.truth:
        raise RuntimeError("SC-ASMC-EM runtime loaded truth")
    if runtime.latent_spec.normalization != "bounded_mixed_warp":
        raise ValueError("SC-ASMC-EM runtime requires bounded_mixed_warp")
    return runtime


def initialize_sc_model(
    config: dict[str, Any],
    runtime: RuntimeBundle,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Create q and prior once from config, with no checkpoint load."""
    validate_sc_asmc_em_config(config)
    out = ensure_dir(out_dir)
    marker = out / "initialization_receipt.json"
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        validate_component_checkpoint_record(payload["q_initial"], runtime)
        validate_component_checkpoint_record(payload["prior_p0"], runtime)
        return payload
    seed = int(
        (config.get("amortized", {}) or {}).get("training", {}).get("seed", 260824)
    )
    model = build_amortized_model(
        config,
        _initial_model_key(seed),
        latent_spec=runtime.latent_spec,
    )
    base = out / "components"
    q0 = save_component_checkpoint(
        base / "q_initial.eqx",
        model.encoder,
        component="q",
        config=config,
        runtime=runtime,
        phase="from_scratch_initialization",
    )
    p0 = save_component_checkpoint(
        base / "p0.eqx",
        model.prior,
        component="prior",
        config=config,
        runtime=runtime,
        phase="from_scratch_initialization",
    )
    payload = {
        "status": "complete",
        "phase": "from_scratch_initialization",
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "truth_used": False,
        "warm_started": False,
        "previous_checkpoints_used": [],
        "q_initial": q0,
        "prior_p0": p0,
        "prior_base": "Normal(0,I)",
        "prior_flow_initialization": "identity with zero final layers",
        "q_flow_initialization": "identity with zero final layers",
        "q_base_initial_log_std": 0.0,
    }
    write_json(marker, payload)
    return payload


def train_sleep_bootstrap(
    config: dict[str, Any],
    runtime: RuntimeBundle,
    out_dir: str | Path,
    *,
    resume_state: str | Path | None = None,
    smoke: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train q only on selected Gaussian sleep pairs with p0 frozen."""
    validation = validate_sc_asmc_em_config(config)
    schedule_cfg = sc_asmc_em_schedule(config)
    raw = dict(
        ((config.get("amortized", {}) or {}).get("sc_asmc_em", {}) or {}).get(
            "sleep", {}
        )
        or {}
    )
    out = ensure_dir(out_dir)
    receipt_path = out / "sleep_receipt.json"
    if receipt_path.is_file():
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_component_checkpoint(
            payload["q_ema_checkpoint"], payload["q_ema_sha256"], runtime
        )
        validate_component_checkpoint(
            payload["q_raw_checkpoint"], payload["q_raw_sha256"], runtime
        )
        return payload
    devices = tuple(jax.local_devices())
    n_devices = len(devices)
    if n_devices <= 0:
        raise RuntimeError("sleep bootstrap requires a JAX device")
    batch_size = int(raw.get("batch_size", 128))
    if smoke:
        batch_size = max(n_devices, min(batch_size, 32))
    if batch_size % n_devices:
        raise ValueError("sleep batch_size must be divisible by local device count")
    epochs = 1 if smoke else int(schedule_cfg.sleep_epochs)
    updates_per_epoch = int(np.ceil(len(runtime.train_arrays.flux) / batch_size))
    if smoke:
        updates_per_epoch = min(updates_per_epoch, 2)
    total_steps = max(epochs * updates_per_epoch, 1)
    warmup_steps = max(
        1, int(np.ceil(float(raw.get("warmup_fraction", 0.05)) * total_steps))
    )
    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(raw.get("peak_learning_rate", 2.0e-4)),
        warmup_steps=min(warmup_steps, max(total_steps - 1, 1)),
        decay_steps=max(total_steps, 2),
        end_value=float(raw.get("final_learning_rate", 2.0e-5)),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(raw.get("gradient_clip_norm", 10.0))),
        optax.adamw(
            learning_rate=learning_rate,
            weight_decay=float(raw.get("weight_decay", 1.0e-6)),
        ),
    )
    seed = int(
        (config.get("amortized", {}) or {}).get("training", {}).get("seed", 260824)
    )
    key = jax.random.fold_in(jax.random.PRNGKey(seed), 1)
    model_key = _initial_model_key(seed)
    model = build_amortized_model(config, model_key, latent_spec=runtime.latent_spec)
    initial_prior_hash = tree_semantic_hash(model.prior)
    optimizer_state = optimizer.init(eqx.filter(model.encoder, eqx.is_inexact_array))
    state = SleepTrainingState(
        model=model,
        ema_encoder=model.encoder,
        optimizer_state=optimizer_state,
        epoch=jnp.asarray(0, dtype=jnp.int32),
        optimizer_step=jnp.asarray(0, dtype=jnp.int32),
        random_key=key,
        best_validation_nll=jnp.asarray(jnp.inf, dtype=jnp.float32),
        best_epoch=jnp.asarray(-1, dtype=jnp.int32),
    )
    if resume_state is not None:
        state = load_sleep_state(
            resume_state,
            state,
            config=config,
            runtime=runtime,
        )
    start_epoch = int(np.asarray(state.epoch)) + 1
    model_replicated = _replicate_tree(state.model, devices)
    ema_replicated = _replicate_tree(state.ema_encoder, devices)
    optimizer_replicated = _replicate_tree(state.optimizer_state, devices)
    key = state.random_key
    best_nll = float(np.asarray(state.best_validation_nll))
    best_epoch = int(np.asarray(state.best_epoch))
    optimizer_step = int(np.asarray(state.optimizer_step))
    step_fn = make_pmap_q_sleep_step(
        optimizer=optimizer,
        sleep_loss_fn=_make_sleep_loss_fn(runtime),
        gradient_clip_norm=float(raw.get("gradient_clip_norm", 10.0)),
    )
    log_path = out / "sleep_training.csv"
    log_rows: list[dict[str, Any]] = []
    all_gradients_finite = True
    ema_decay = float(schedule_cfg.ema_decay)
    initial_entropy = _observed_entropy(
        state.model,
        runtime,
        key=jax.random.fold_in(key, 900),
    )
    for epoch in range(start_epoch, epochs + 1):
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(runtime.train_arrays.flux))
        target_count = updates_per_epoch * batch_size
        if len(order) > target_count:
            order = order[:target_count]
        elif len(order) < target_count:
            order = np.concatenate(
                (order, rng.choice(order, size=target_count - len(order), replace=True))
            )
        for batch_index, photometry in enumerate(
            iter_photometry_batches_from_arrays(
                runtime.train_arrays,
                batch_size=batch_size,
                feature_stats=runtime.feature_stats,
                order=order,
                truth_names=None,
            )
        ):
            key, step_key = jax.random.split(key)
            sharded_batch = _shard_loss_batch(_loss_batch(photometry), n_devices)
            model_replicated, optimizer_replicated, metrics, step_metrics = step_fn(
                model_replicated,
                optimizer_replicated,
                sharded_batch,
                jax.random.split(step_key, n_devices),
            )
            ema_replicated = update_ema_encoder(
                ema_replicated,
                model_replicated.encoder,
                decay=ema_decay,
            )
            optimizer_step += 1
            gradients_finite = bool(
                np.asarray(jax.device_get(step_metrics.grads_finite[0]))
            )
            update_applied = bool(
                np.asarray(jax.device_get(step_metrics.update_applied[0]))
            )
            all_gradients_finite &= gradients_finite and update_applied
            row = {
                "epoch": epoch,
                "batch": batch_index,
                "optimizer_step": optimizer_step,
                "sleep_nll": float(np.asarray(jax.device_get(step_metrics.loss[0]))),
                "raw_gradient_norm": float(
                    np.asarray(jax.device_get(step_metrics.raw_grad_norm[0]))
                ),
                "clipped_gradient_norm": float(
                    np.asarray(jax.device_get(step_metrics.clipped_grad_norm[0]))
                ),
                "gradient_finite": gradients_finite,
                "update_applied": update_applied,
                "selection_acceptance": float(
                    np.asarray(
                        jax.device_get(
                            metrics["sleep_selection_acceptance_fraction"][0]
                        )
                    )
                ),
                "learning_rate": float(np.asarray(learning_rate(optimizer_step - 1))),
            }
            log_rows.append(row)
            if verbose:
                print(
                    "[sc-asmc][sleep] "
                    f"epoch={epoch}/{epochs} batch={batch_index + 1}/{updates_per_epoch} "
                    f"nll={row['sleep_nll']:.5f} finite={int(gradients_finite)}",
                    flush=True,
                )
        model = _unreplicate_tree(model_replicated)
        ema_encoder = _unreplicate_tree(ema_replicated)
        ema_model = eqx.tree_at(lambda tree: tree.encoder, model, ema_encoder)
        validation_metrics = evaluate_sleep_validation(
            ema_model,
            runtime,
            key=jax.random.fold_in(key, 1000 + epoch),
            batch_size=batch_size,
            entropy_samples=4,
        )
        entropy_drop = (
            initial_entropy["full_entropy"] - validation_metrics["full_entropy"]
        )
        entropy_gate = bool(
            np.isfinite(validation_metrics["full_entropy"])
            and validation_metrics["full_entropy"]
            >= float(raw.get("minimum_full_entropy", -20.0))
            and entropy_drop <= float(raw.get("maximum_entropy_drop", 30.0))
            and validation_metrics["minimum_log_std"] > -3.999
        )
        validation_ok = bool(
            np.isfinite(validation_metrics["sleep_nll"])
            and validation_metrics["finite_logq"]
            and entropy_gate
            and all_gradients_finite
        )
        if validation_ok and validation_metrics["sleep_nll"] < best_nll:
            best_nll = float(validation_metrics["sleep_nll"])
            best_epoch = epoch
            save_component_checkpoint(
                out / "components" / "q_sleep_best_ema.eqx",
                ema_encoder,
                component="q_ema",
                config=config,
                runtime=runtime,
                phase="sleep_bootstrap",
                extra={"epoch": epoch, **validation_metrics},
            )
        save_component_checkpoint(
            out / "components" / "q_sleep_latest_raw.eqx",
            model.encoder,
            component="q_raw",
            config=config,
            runtime=runtime,
            phase="sleep_bootstrap",
            extra={"epoch": epoch, **validation_metrics},
        )
        save_component_checkpoint(
            out / "components" / "q_sleep_latest_ema.eqx",
            ema_encoder,
            component="q_ema",
            config=config,
            runtime=runtime,
            phase="sleep_bootstrap",
            extra={"epoch": epoch, **validation_metrics},
        )
        state = SleepTrainingState(
            model=model,
            ema_encoder=ema_encoder,
            optimizer_state=_unreplicate_tree(optimizer_replicated),
            epoch=jnp.asarray(epoch, dtype=jnp.int32),
            optimizer_step=jnp.asarray(optimizer_step, dtype=jnp.int32),
            random_key=key,
            best_validation_nll=jnp.asarray(best_nll, dtype=jnp.float32),
            best_epoch=jnp.asarray(best_epoch, dtype=jnp.int32),
        )
        save_sleep_state(
            out / "states" / "sleep_latest.eqx",
            state,
            config=config,
            runtime=runtime,
        )
        _append_csv(log_path, log_rows)
        log_rows.clear()
    final_model = _unreplicate_tree(model_replicated)
    if best_epoch < 0:
        raise RuntimeError(
            "sleep bootstrap produced no checkpoint satisfying finite/entropy gates"
        )
    if tree_semantic_hash(final_model.prior) != initial_prior_hash:
        raise RuntimeError("p0 changed during the q-only sleep bootstrap")
    best_path = out / "components" / "q_sleep_best_ema.eqx"
    payload = {
        "status": "PASS",
        "phase": "sleep_bootstrap",
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "truth_used": False,
        "prior_frozen": True,
        "q_trained_from_scratch": True,
        "epochs": epochs,
        "optimizer_steps": optimizer_step,
        "best_epoch": best_epoch,
        "best_held_out_sleep_nll": best_nll,
        "q_ema_checkpoint": str(best_path.resolve()),
        "q_ema_sha256": sha256_file(best_path),
        "q_raw_checkpoint": str(
            (out / "components" / "q_sleep_latest_raw.eqx").resolve()
        ),
        "q_raw_sha256": sha256_file(out / "components" / "q_sleep_latest_raw.eqx"),
        "p0_semantic_hash_before": initial_prior_hash,
        "p0_semantic_hash_after": tree_semantic_hash(final_model.prior),
        "all_gradients_finite": all_gradients_finite,
        "initial_observed_entropy": initial_entropy,
        "model_input_dim": int(validation["input_dim"]),
        "schedule": asdict(schedule_cfg),
    }
    write_json(receipt_path, payload)
    (out / "SLEEP_PASS").touch()
    return payload


def evaluate_sleep_validation(
    model: Any,
    runtime: RuntimeBundle,
    *,
    key: jax.Array,
    batch_size: int,
    entropy_samples: int,
) -> dict[str, Any]:
    numerator = 0.0
    count = 0.0
    finite_logq = True
    first_batch = None
    sleep_loss = _make_sleep_loss_fn(runtime)
    for batch_index, photometry in enumerate(
        iter_photometry_batches_from_arrays(
            runtime.validation_arrays,
            batch_size=int(batch_size),
            feature_stats=runtime.feature_stats,
            truth_names=None,
        )
    ):
        batch = _loss_batch(photometry)
        if first_batch is None:
            first_batch = batch
        loss, metrics = sleep_loss(
            model,
            batch,
            jax.random.fold_in(key, batch_index),
        )
        selected = float(np.asarray(metrics["sleep_selection_selected_count"]))
        numerator += float(np.asarray(loss)) * selected
        count += selected
        finite_logq &= bool(np.asarray(metrics["finite_fraction"]) == 1.0)
    if first_batch is None or count <= 0.0:
        raise RuntimeError("held-out sleep validation generated no selected pair")
    entropy = posterior_entropy_diagnostics(
        model,
        first_batch.features,
        jax.random.fold_in(key, 99_001),
        n_samples=int(entropy_samples),
    )
    _mean, log_std = model.encoder(first_batch.features)
    return {
        "sleep_nll": numerator / count,
        "selected_pairs": count,
        "finite_logq": finite_logq,
        "base_entropy": float(np.asarray(entropy["posterior_base_entropy"])),
        "flow_residual_logdet": float(
            np.asarray(entropy["posterior_residual_logdet_mean"])
        ),
        "full_entropy": float(np.asarray(entropy["posterior_full_entropy_mc"])),
        "mean_log_std": float(np.asarray(jnp.mean(log_std))),
        "minimum_log_std": float(np.asarray(jnp.min(log_std))),
        "maximum_log_std": float(np.asarray(jnp.max(log_std))),
    }


def load_sc_model(
    config: dict[str, Any],
    runtime: RuntimeBundle,
    *,
    q_checkpoint: str | Path,
    prior_checkpoint: str | Path,
) -> Any:
    validate_component_checkpoint(q_checkpoint, sha256_file(q_checkpoint), runtime)
    validate_component_checkpoint(
        prior_checkpoint,
        sha256_file(prior_checkpoint),
        runtime,
    )
    seed = int(
        (config.get("amortized", {}) or {}).get("training", {}).get("seed", 260824)
    )
    template = build_amortized_model(
        config,
        _initial_model_key(seed),
        latent_spec=runtime.latent_spec,
    )
    q = eqx.tree_deserialise_leaves(q_checkpoint, template.encoder)
    prior = eqx.tree_deserialise_leaves(prior_checkpoint, template.prior)
    model = eqx.tree_at(lambda tree: tree.encoder, template, q)
    return eqx.tree_at(lambda tree: tree.prior, model, prior)


def save_component_checkpoint(
    path: str | Path,
    component_value: Any,
    *,
    component: str,
    config: dict[str, Any],
    runtime: RuntimeBundle,
    phase: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    eqx.tree_serialise_leaves(temporary, component_value)
    os.replace(temporary, destination)
    sidecar = {
        "component": component,
        "phase": phase,
        "sha256": sha256_file(destination),
        "workflow_config_hash": sc_asmc_em_config_hash(config),
        "latent_transform_hash": latent_spec_hash(runtime.latent_spec),
        "feature_stats_hash": feature_stats_hash(runtime.feature_stats),
        "truth_used": False,
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        **(extra or {}),
    }
    write_json(destination.with_suffix(destination.suffix + ".json"), sidecar)
    return {"path": str(destination.resolve()), **sidecar}


def validate_component_checkpoint_record(
    record: dict[str, Any],
    runtime: RuntimeBundle,
) -> None:
    """Validate a component record against the current no-truth runtime."""
    validate_component_checkpoint(record["path"], record["sha256"], runtime)


def validate_component_checkpoint(
    path: str | Path,
    expected_sha256: str,
    runtime: RuntimeBundle,
) -> None:
    """Fail closed when a resumed component or its semantic sidecar changed."""
    checkpoint = Path(path)
    if not checkpoint.is_file() or sha256_file(checkpoint) != str(expected_sha256):
        raise ValueError("component checkpoint hash mismatch during resume")
    sidecar_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("sha256") != str(expected_sha256):
        raise ValueError("component checkpoint sidecar hash mismatch")
    if sidecar.get("workflow_config_hash") != sc_asmc_em_config_hash(runtime.config):
        raise ValueError("component checkpoint workflow configuration mismatch")
    if sidecar.get("latent_transform_hash") != latent_spec_hash(runtime.latent_spec):
        raise ValueError("component checkpoint latent transform mismatch")
    if sidecar.get("feature_stats_hash") != feature_stats_hash(runtime.feature_stats):
        raise ValueError("component checkpoint feature statistics mismatch")
    if sidecar.get("truth_used") is not False:
        raise ValueError("component checkpoint lacks a no-truth receipt")


def save_sleep_state(
    path: str | Path,
    state: SleepTrainingState,
    *,
    config: dict[str, Any],
    runtime: RuntimeBundle,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    eqx.tree_serialise_leaves(temporary, state)
    os.replace(temporary, destination)
    write_json(
        destination.with_suffix(destination.suffix + ".json"),
        {
            "epoch": int(np.asarray(state.epoch)),
            "optimizer_step": int(np.asarray(state.optimizer_step)),
            "sha256": sha256_file(destination),
            "workflow_config_hash": sc_asmc_em_config_hash(config),
            "latent_transform_hash": latent_spec_hash(runtime.latent_spec),
            "feature_stats_hash": feature_stats_hash(runtime.feature_stats),
            "truth_used": False,
            "config_contract": validate_sc_asmc_em_config(config)["status"],
        },
    )


def load_sleep_state(
    path: str | Path,
    template: SleepTrainingState,
    *,
    config: dict[str, Any],
    runtime: RuntimeBundle,
) -> SleepTrainingState:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sidecar_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("sha256") != sha256_file(checkpoint):
        raise ValueError("sleep resume state hash mismatch")
    if sidecar.get("workflow_config_hash") != sc_asmc_em_config_hash(config):
        raise ValueError("sleep resume workflow configuration mismatch")
    if sidecar["latent_transform_hash"] != latent_spec_hash(runtime.latent_spec):
        raise ValueError("sleep resume latent transform hash mismatch")
    if sidecar["feature_stats_hash"] != feature_stats_hash(runtime.feature_stats):
        raise ValueError("sleep resume feature stats hash mismatch")
    if sidecar.get("truth_used") is not False:
        raise ValueError("sleep resume state lacks a no-truth receipt")
    validate_sc_asmc_em_config(config)
    return eqx.tree_deserialise_leaves(checkpoint, template)


def update_ema_encoder(ema: Any, current: Any, *, decay: float) -> Any:
    if not 0.0 <= float(decay) < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    return jax.tree_util.tree_map(
        lambda old, new: (
            float(decay) * old + (1.0 - float(decay)) * new
            if eqx.is_inexact_array(old)
            else old
        ),
        ema,
        current,
    )


def tree_semantic_hash(tree: Any) -> str:
    import hashlib

    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        if not eqx.is_array(leaf):
            continue
        array = np.ascontiguousarray(jax.device_get(leaf))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _observed_entropy(
    model: Any,
    runtime: RuntimeBundle,
    *,
    key: jax.Array,
) -> dict[str, float]:
    batch = next(
        iter_photometry_batches_from_arrays(
            runtime.validation_arrays,
            batch_size=min(128, len(runtime.validation_arrays.flux)),
            feature_stats=runtime.feature_stats,
            truth_names=None,
        )
    )
    metrics = posterior_entropy_diagnostics(
        model,
        batch.features,
        key,
        n_samples=4,
    )
    return {
        "base_entropy": float(np.asarray(metrics["posterior_base_entropy"])),
        "flow_residual_logdet": float(
            np.asarray(metrics["posterior_residual_logdet_mean"])
        ),
        "full_entropy": float(np.asarray(metrics["posterior_full_entropy_mc"])),
    }


def _replicate_tree(tree: Any, devices: tuple[Any, ...]) -> Any:
    count = len(devices)

    def replicate(value):
        if not eqx.is_array(value):
            return value
        host = np.asarray(jax.device_get(value))
        return jnp.asarray(np.broadcast_to(host, (count, *host.shape)).copy())

    return jax.tree_util.tree_map(replicate, tree)


def _unreplicate_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda value: value[0] if eqx.is_array(value) else value,
        tree,
    )


def _shard_loss_batch(batch: LossBatch, n_devices: int) -> LossBatch:
    objects = int(batch.features.shape[0])
    if objects % int(n_devices):
        raise ValueError("sleep loss batch is not divisible by devices")
    local = objects // int(n_devices)
    return LossBatch(
        *(
            jnp.asarray(value).reshape(
                int(n_devices), local, *jnp.asarray(value).shape[1:]
            )
            for value in batch
        )
    )


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _initial_model_key(seed: int) -> jax.Array:
    return jax.random.fold_in(jax.random.PRNGKey(int(seed)), 0)
