"""Full-bank q distillation with Gaussian sleep replay and EMA selection."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.io import ensure_dir, write_json

from .adaptive_smc_trainer import _make_sleep_loss_fn
from .adaptive_smc_training import (
    SMCPosteriorBatch,
    make_pmap_q_sleep_step,
    make_pmap_q_smc_step,
    smc_q_distillation_loss,
)
from .config import require_amortized_dependencies
from .data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from .features import feature_stats_hash
from .hierarchical_e_step import (
    build_pmap_hierarchy_kernels,
    run_pmap_model_hierarchical_e_step,
)
from .latent import latent_spec_hash
from .posterior import posterior_entropy_diagnostics
from .posterior_bank import (
    C0_SCOPE_STATEMENT,
    read_posterior_bank_shard,
    sha256_file,
    validate_posterior_bank_manifest_provenance,
)
from .sc_asmc_config import (
    sc_asmc_em_config_hash,
    sc_asmc_em_hierarchy,
    validate_sc_asmc_em_config,
)
from .sc_asmc_mstep import _posterior_from_bank_shard
from .sc_asmc_training import (
    RuntimeBundle,
    load_sc_model,
    save_component_checkpoint,
    tree_semantic_hash,
    update_ema_encoder,
    validate_component_checkpoint,
)
from .train import LossBatch, _loss_batch

eqx, optax = require_amortized_dependencies()


def distill_q_from_full_bank(
    config: dict[str, Any],
    runtime: RuntimeBundle,
    *,
    input_bank_manifest: str | Path,
    heldout_rows: np.ndarray,
    q_checkpoint: str | Path,
    prior_checkpoint: str | Path,
    out_dir: str | Path,
    iteration: int,
    seed: int,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    resume: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train q against stopped bank weights while keeping p_eta fixed."""
    validate_sc_asmc_em_config(config)
    if int(iteration) not in {0, 1, 2}:
        raise ValueError("q distillation iteration must be active-bootstrap, 1, or 2")
    output = ensure_dir(out_dir)
    label = "active_bootstrap" if int(iteration) == 0 else f"em{int(iteration)}"
    receipt_path = output / f"q_distillation_{label}_receipt.json"
    bank_path = Path(input_bank_manifest).resolve()
    manifest = json.loads(bank_path.read_text(encoding="utf-8"))
    input_hashes = {
        "input_bank_manifest_sha256": sha256_file(bank_path),
        "input_q_checkpoint_sha256": sha256_file(q_checkpoint),
        "input_prior_checkpoint_sha256": sha256_file(prior_checkpoint),
    }
    validate_posterior_bank_manifest_provenance(
        manifest,
        expected_fields={
            "dataset_hash": sha256_file(config["catalog_path"]),
            "workflow_config_hash": sc_asmc_em_config_hash(config),
            "q_ema_hash": input_hashes["input_q_checkpoint_sha256"],
            "prior_checkpoint_hash": input_hashes["input_prior_checkpoint_sha256"],
            "latent_transform_hash": latent_spec_hash(runtime.latent_spec),
            "feature_stats_hash": feature_stats_hash(runtime.feature_stats),
        },
    )
    if receipt_path.is_file() and resume:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if any(payload.get(name) != value for name, value in input_hashes.items()):
            raise ValueError("q distillation resume inputs changed")
        validate_component_checkpoint(
            payload["q_ema_checkpoint"], payload["q_ema_sha256"], runtime
        )
        validate_component_checkpoint(
            payload["q_raw_checkpoint"], payload["q_raw_sha256"], runtime
        )
        return payload
    raw = dict(
        ((config.get("amortized", {}) or {}).get("sc_asmc_em", {}) or {}).get(
            "q_distillation", {}
        )
        or {}
    )
    epochs = int(epochs_override or raw.get("epochs", 3))
    if not 1 <= epochs <= 5:
        raise ValueError("q distillation epochs must be in [1, 5]")
    batch_size = int(
        batch_size_override
        if batch_size_override is not None
        else raw.get("distinct_objects_per_batch", 128)
    )
    devices = tuple(jax.local_devices())
    if batch_size % len(devices):
        raise ValueError("q distillation batch must be divisible by local devices")
    heldout_set = set(np.asarray(heldout_rows, dtype=np.int64).tolist())
    training_rows = set(_manifest_rows(manifest).tolist()) - heldout_set
    if not training_rows or not heldout_set:
        raise ValueError("q distillation requires train and held-out bank objects")
    raw_bank_update_count = _bank_batch_count(
        manifest,
        batch_size=batch_size,
        include_rows=training_rows,
    )
    ratio_padding_updates = (-raw_bank_update_count) % 3
    bank_update_count = raw_bank_update_count + ratio_padding_updates
    updates_per_epoch = bank_update_count + int(np.ceil(bank_update_count / 3.0))
    total_steps = max(epochs * updates_per_epoch, 1)
    warmup = max(1, int(np.ceil(total_steps * float(raw.get("warmup_fraction", 0.05)))))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(raw.get("learning_rate", 2.0e-5)),
        warmup_steps=min(warmup, max(total_steps - 1, 1)),
        decay_steps=max(total_steps, 2),
        end_value=float(raw.get("final_learning_rate", 2.0e-6)),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(raw.get("gradient_clip_norm", 20.0))),
        optax.adamw(schedule, weight_decay=1.0e-6),
    )
    model = load_sc_model(
        config,
        runtime,
        q_checkpoint=q_checkpoint,
        prior_checkpoint=prior_checkpoint,
    )
    prior_hash = tree_semantic_hash(model.prior)
    optimizer_state = optimizer.init(eqx.filter(model.encoder, eqx.is_inexact_array))
    model_replicated = _replicate_tree(model, devices)
    ema_replicated = _replicate_tree(model.encoder, devices)
    optimizer_replicated = _replicate_tree(optimizer_state, devices)
    bank_step = make_pmap_q_smc_step(
        optimizer=optimizer,
        gradient_clip_norm=float(raw.get("gradient_clip_norm", 20.0)),
    )
    sleep_step = make_pmap_q_sleep_step(
        optimizer=optimizer,
        sleep_loss_fn=_make_sleep_loss_fn(runtime),
        gradient_clip_norm=float(raw.get("gradient_clip_norm", 20.0)),
    )
    best_score = float("inf")
    best_epoch = -1
    optimizer_step = 0
    log_rows = []
    ema_decay = float(raw.get("ema_decay", 0.999))
    for epoch in range(1, epochs + 1):
        bank_updates_since_sleep = 0
        sleep_replays = 0
        batches = _ratio_padded_bank_batches(
            iter_bank_feature_batches(
                manifest,
                include_rows=training_rows,
                batch_size=batch_size,
                shuffle_seed=int(seed) + epoch,
            ),
            ratio=3,
        )
        for batch_index, (features, posterior, ratio_padding) in enumerate(batches):
            padded_features, padded_posterior = _pad_q_batch(
                features, posterior, batch_size
            )
            sharded_features = _shard_features(padded_features, len(devices))
            sharded_posterior = _shard_posterior(padded_posterior, len(devices))
            model_replicated, optimizer_replicated, metrics, step_metrics = bank_step(
                model_replicated,
                optimizer_replicated,
                sharded_features,
                sharded_posterior,
            )
            ema_replicated = update_ema_encoder(
                ema_replicated,
                model_replicated.encoder,
                decay=ema_decay,
            )
            optimizer_step += 1
            bank_updates_since_sleep += 1
            log_rows.append(
                _q_step_row(
                    step_metrics,
                    epoch=epoch,
                    batch=batch_index,
                    optimizer_step=optimizer_step,
                    update_kind=(
                        "posterior_bank_ratio_padding"
                        if ratio_padding
                        else "posterior_bank"
                    ),
                    learning_rate=float(np.asarray(schedule(optimizer_step - 1))),
                )
            )
            if bank_updates_since_sleep == 3:
                sleep_batch = _sample_observed_sleep_batch(
                    runtime,
                    batch_size=batch_size,
                    seed=int(seed) + 10_000 * epoch + sleep_replays,
                )
                (
                    model_replicated,
                    optimizer_replicated,
                    _sleep_metrics,
                    sleep_metrics,
                ) = sleep_step(
                    model_replicated,
                    optimizer_replicated,
                    _shard_loss_batch(sleep_batch, len(devices)),
                    jax.random.split(
                        jax.random.fold_in(jax.random.PRNGKey(seed), optimizer_step),
                        len(devices),
                    ),
                )
                ema_replicated = update_ema_encoder(
                    ema_replicated,
                    model_replicated.encoder,
                    decay=ema_decay,
                )
                optimizer_step += 1
                sleep_replays += 1
                bank_updates_since_sleep = 0
                log_rows.append(
                    _q_step_row(
                        sleep_metrics,
                        epoch=epoch,
                        batch=sleep_replays - 1,
                        optimizer_step=optimizer_step,
                        update_kind="sleep_replay",
                        learning_rate=float(np.asarray(schedule(optimizer_step - 1))),
                    )
                )
        if bank_updates_since_sleep:
            raise AssertionError("internal 3:1 q-distillation scheduler error")
        model = _unreplicate_tree(model_replicated)
        ema = _unreplicate_tree(ema_replicated)
        ema_model = eqx.tree_at(lambda tree: tree.encoder, model, ema)
        validation = evaluate_q_distillation_checkpoint(
            config,
            runtime,
            manifest,
            model=ema_model,
            heldout_rows=heldout_set,
            seed=int(seed) + epoch,
        )
        if validation["checkpoint_gate"] and validation["selection_score"] < best_score:
            best_score = float(validation["selection_score"])
            best_epoch = epoch
            save_component_checkpoint(
                output / "components" / f"q_{label}_best_ema.eqx",
                ema,
                component="q_ema",
                config=config,
                runtime=runtime,
                phase=f"q_distillation_{label}",
                extra={"epoch": epoch, **validation},
            )
        save_component_checkpoint(
            output / "components" / f"q_{label}_latest_raw.eqx",
            model.encoder,
            component="q_raw",
            config=config,
            runtime=runtime,
            phase=f"q_distillation_{label}",
            extra={"epoch": epoch, **validation},
        )
        save_component_checkpoint(
            output / "components" / f"q_{label}_latest_ema.eqx",
            ema,
            component="q_ema",
            config=config,
            runtime=runtime,
            phase=f"q_distillation_{label}",
            extra={"epoch": epoch, **validation},
        )
        if verbose:
            print(
                "[sc-asmc][q-distill] "
                f"iteration={iteration} epoch={epoch}/{epochs} "
                f"CE={validation['heldout_cross_entropy']:.5f} "
                f"stages={validation['median_standard_smc_stages']:.2f} "
                f"hard={validation['standard_smc_hard_fraction']:.3f} "
                f"entropy={validation['posterior_full_entropy']:.3f}",
                flush=True,
            )
    if best_epoch < 0:
        raise RuntimeError("q distillation found no finite noncollapsed checkpoint")
    final_model = _unreplicate_tree(model_replicated)
    if tree_semantic_hash(final_model.prior) != prior_hash:
        raise RuntimeError("prior changed during q-only full-bank distillation")
    _write_jsonl(output / f"q_distillation_{label}.jsonl", log_rows)
    best = output / "components" / f"q_{label}_best_ema.eqx"
    raw_checkpoint = output / "components" / f"q_{label}_latest_raw.eqx"
    payload = {
        "status": "PASS",
        "phase": f"q_distillation_{label}",
        "iteration": int(iteration),
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "truth_used": False,
        "posterior_bank_frozen": True,
        "prior_frozen": True,
        "trainable_components": ["q"],
        "loss": ("-mean_i sum_k stop(w_ik) log q_psi(x_ik|y_i) + lambda_sleep J_sleep"),
        "posterior_bank_to_sleep_update_ratio": "3:1",
        "posterior_bank_updates_per_epoch": bank_update_count,
        "ratio_padding_updates_per_epoch": ratio_padding_updates,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "optimizer_steps": optimizer_step,
        "q_ema_checkpoint": str(best.resolve()),
        "q_ema_sha256": sha256_file(best),
        "q_raw_checkpoint": str(raw_checkpoint.resolve()),
        "q_raw_sha256": sha256_file(raw_checkpoint),
        **input_hashes,
        "prior_semantic_hash_before": prior_hash,
        "prior_semantic_hash_after": tree_semantic_hash(final_model.prior),
    }
    write_json(receipt_path, payload)
    return payload


def evaluate_q_distillation_checkpoint(
    config: dict[str, Any],
    runtime: RuntimeBundle,
    manifest: dict[str, Any],
    *,
    model: Any,
    heldout_rows: set[int],
    seed: int,
) -> dict[str, Any]:
    numerator = 0.0
    objects = 0.0
    first_features = None
    for features, posterior in iter_bank_feature_batches(
        manifest,
        include_rows=heldout_rows,
        batch_size=128,
        shuffle_seed=0,
    ):
        loss, metrics = smc_q_distillation_loss(model, features, posterior)
        count = float(np.asarray(metrics.eligible_count))
        numerator += float(np.asarray(loss)) * count
        objects += count
        if first_features is None:
            first_features = features
    if objects <= 0 or first_features is None:
        raise RuntimeError("q validation has no held-out resolved bank object")
    cross_entropy = numerator / objects
    entropy = posterior_entropy_diagnostics(
        model,
        first_features,
        jax.random.PRNGKey(seed),
        n_samples=4,
    )
    diagnostic_rows = np.asarray(sorted(heldout_rows)[:32], dtype=np.int64)
    arrays = load_photometry_arrays_from_config(
        runtime.config,
        batch_size=10_000,
        row_indices=diagnostic_rows,
    )
    batch = next(
        iter_photometry_batches_from_arrays(
            arrays,
            batch_size=len(diagnostic_rows),
            feature_stats=runtime.feature_stats,
            truth_names=None,
        )
    )
    hierarchy = sc_asmc_em_hierarchy(config)
    devices = tuple(jax.local_devices())
    fixed = max(32, len(devices))
    if fixed % len(devices):
        fixed += len(devices) - fixed % len(devices)
    kernels = build_pmap_hierarchy_kernels(
        latent_spec=runtime.jit_latent_spec,
        context=runtime.context,
        model_args=runtime.model_args,
        parameter_names=runtime.parameter_names,
        likelihood_config=runtime.likelihood_config,
        calibration_config=runtime.calibration_config,
        primary_config=hierarchy.primary,
        fallback_config=hierarchy.fallback,
        extended_config=hierarchy.extended,
        proposal_config=hierarchy.proposal,
        minimum_is_ess_fraction=hierarchy.minimum_is_ess_fraction,
        maximum_is_weight=hierarchy.maximum_is_weight,
        primary_batch_size=fixed,
        fallback_batch_size=max(len(devices), 16),
        extended_batch_size=max(len(devices), 16),
        devices=devices,
    )
    result = run_pmap_model_hierarchical_e_step(
        model_snapshot=model,
        batch=_loss_batch(batch),
        key=jax.random.fold_in(jax.random.PRNGKey(seed), 80_000),
        kernels=kernels,
    )
    median_stages, hard_fraction = _standard_smc_checkpoint_diagnostics(
        result,
        n_objects=len(diagnostic_rows),
    )
    full_entropy = float(np.asarray(entropy["posterior_full_entropy_mc"]))
    selection_score = cross_entropy + 0.05 * median_stages + 10.0 * hard_fraction
    gate = bool(
        np.isfinite(selection_score)
        and np.isfinite(full_entropy)
        and full_entropy > -20.0
        and hard_fraction <= 0.30
    )
    return {
        "heldout_cross_entropy": cross_entropy,
        "heldout_objects": objects,
        "median_standard_smc_stages": median_stages,
        "standard_smc_hard_fraction": hard_fraction,
        "raw_is_median_ess_fraction": float(
            np.median(np.asarray(result.ordinary.ess)) / 64.0
        ),
        "posterior_base_entropy": float(np.asarray(entropy["posterior_base_entropy"])),
        "flow_residual_logdet": float(
            np.asarray(entropy["posterior_residual_logdet_mean"])
        ),
        "posterior_full_entropy": full_entropy,
        "selection_score": selection_score,
        "checkpoint_gate": gate,
    }


def _standard_smc_checkpoint_diagnostics(
    result: Any,
    *,
    n_objects: int,
) -> tuple[float, float]:
    """Measure q quality after primary/fallback, before extended hard-only SMC."""
    count = int(n_objects)
    stages = np.zeros(count, dtype=np.float64)
    attempted = np.zeros(count, dtype=bool)
    if result.primary is not None:
        indices = np.asarray(result.primary_indices, dtype=np.int64)
        attempted[indices] = True
        stages[indices] = np.asarray(
            jax.device_get(result.primary.number_of_stages), dtype=np.float64
        )
    if result.fallback is not None:
        indices = np.asarray(result.fallback_indices, dtype=np.int64)
        attempted[indices] = True
        stages[indices] = np.asarray(
            jax.device_get(result.fallback.number_of_stages), dtype=np.float64
        )
    median = float(np.median(stages[attempted])) if np.any(attempted) else 0.0
    hard_fraction = float(len(np.asarray(result.extended_indices)) / max(count, 1))
    return median, hard_fraction


def iter_bank_feature_batches(
    manifest: dict[str, Any],
    *,
    include_rows: set[int],
    batch_size: int,
    shuffle_seed: int,
) -> Iterator[tuple[jnp.ndarray, SMCPosteriorBatch]]:
    records = list(manifest["shards"])
    rng = np.random.default_rng(int(shuffle_seed))
    rng.shuffle(records)
    for record in records:
        shard = read_posterior_bank_shard(record["path"])
        if shard.features is None:
            raise ValueError("q distillation requires inline bank features")
        selected = np.flatnonzero(
            np.asarray(
                [int(row) in include_rows for row in shard.row_index], dtype=bool
            )
            & np.asarray(shard.resolved, dtype=bool)
        )
        rng.shuffle(selected)
        for start in range(0, len(selected), int(batch_size)):
            indices = selected[start : start + int(batch_size)]
            if not len(indices):
                continue
            yield (
                jnp.asarray(shard.features[indices]),
                _posterior_from_bank_shard(shard, indices),
            )


def _pad_q_batch(
    features: jnp.ndarray,
    posterior: SMCPosteriorBatch,
    target: int,
) -> tuple[jnp.ndarray, SMCPosteriorBatch]:
    count = int(features.shape[0])
    if count > int(target):
        raise ValueError("q batch exceeds fixed target")
    indices = jnp.arange(int(target), dtype=jnp.int32) % count
    padded_features = jnp.take(features, indices, axis=0)
    fields = []
    for name, value in zip(posterior._fields, posterior, strict=True):
        axis = 1 if name in {"particles", "normalized_weights"} else 0
        fields.append(jnp.take(value, indices, axis=axis))
    padded = SMCPosteriorBatch(*fields)
    real = jnp.arange(int(target)) < count
    padded = padded._replace(eligible=padded.eligible & real)
    return padded_features, padded


def _shard_features(features: jnp.ndarray, n_devices: int) -> jnp.ndarray:
    return features.reshape(int(n_devices), features.shape[0] // int(n_devices), -1)


def _shard_posterior(
    posterior: SMCPosteriorBatch,
    n_devices: int,
) -> SMCPosteriorBatch:
    objects = int(posterior.eligible.shape[0])
    local = objects // int(n_devices)
    values = []
    for name, value in zip(posterior._fields, posterior, strict=True):
        array = jnp.asarray(value)
        if name in {"particles", "normalized_weights"}:
            axes = (1, 0, 2, *range(3, array.ndim + 1))
            converted = array.reshape(
                array.shape[0], int(n_devices), local, *array.shape[2:]
            ).transpose(axes)
        else:
            converted = array.reshape(int(n_devices), local, *array.shape[1:])
        values.append(converted)
    return SMCPosteriorBatch(*values)


def _sample_observed_sleep_batch(
    runtime: RuntimeBundle,
    *,
    batch_size: int,
    seed: int,
) -> LossBatch:
    rng = np.random.default_rng(int(seed))
    order = rng.choice(
        len(runtime.train_arrays.flux), size=int(batch_size), replace=True
    )
    batch = next(
        iter_photometry_batches_from_arrays(
            runtime.train_arrays,
            batch_size=int(batch_size),
            feature_stats=runtime.feature_stats,
            order=order,
            truth_names=None,
        )
    )
    return _loss_batch(batch)


def _shard_loss_batch(batch: LossBatch, n_devices: int) -> LossBatch:
    local = int(batch.features.shape[0]) // int(n_devices)
    return LossBatch(
        *(value.reshape(int(n_devices), local, *value.shape[1:]) for value in batch)
    )


def _bank_batch_count(
    manifest: dict[str, Any],
    *,
    batch_size: int,
    include_rows: set[int],
) -> int:
    """Count streamed resolved batches without loading particle arrays together."""
    count = 0
    for record in manifest["shards"]:
        shard = read_posterior_bank_shard(record["path"])
        selected = np.asarray(shard.resolved, dtype=bool) & np.asarray(
            [int(row) in include_rows for row in shard.row_index], dtype=bool
        )
        count += int(np.ceil(np.sum(selected) / int(batch_size)))
    if count <= 0:
        raise ValueError("q distillation has no resolved training batch")
    return count


def _ratio_padded_bank_batches(
    batches: Iterator[tuple[jnp.ndarray, SMCPosteriorBatch]],
    *,
    ratio: int,
) -> Iterator[tuple[jnp.ndarray, SMCPosteriorBatch, bool]]:
    """Repeat at most ``ratio-1`` streamed batches to close an exact replay group."""
    if int(ratio) <= 0:
        raise ValueError("posterior-to-sleep update ratio must be positive")
    saved: list[tuple[jnp.ndarray, SMCPosteriorBatch]] = []
    count = 0
    for batch in batches:
        if len(saved) < int(ratio) - 1:
            saved.append(batch)
        count += 1
        yield (*batch, False)
    if count <= 0:
        raise ValueError("q distillation has no posterior-bank batch")
    padding = (-count) % int(ratio)
    for index in range(padding):
        yield (*saved[index % len(saved)], True)


def _manifest_rows(manifest: dict[str, Any]) -> np.ndarray:
    values = []
    for record in manifest["shards"]:
        values.append(read_posterior_bank_shard(record["path"]).row_index)
    return np.concatenate(values).astype(np.int64)


def _q_step_row(
    metrics: Any,
    *,
    epoch: int,
    batch: int,
    optimizer_step: int,
    update_kind: str,
    learning_rate: float,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "batch": int(batch),
        "optimizer_step": int(optimizer_step),
        "update_kind": update_kind,
        "loss": float(np.asarray(jax.device_get(metrics.loss[0]))),
        "raw_gradient_norm": float(
            np.asarray(jax.device_get(metrics.raw_grad_norm[0]))
        ),
        "clipped_gradient_norm": float(
            np.asarray(jax.device_get(metrics.clipped_grad_norm[0]))
        ),
        "gradients_finite": bool(np.asarray(jax.device_get(metrics.grads_finite[0]))),
        "update_applied": bool(np.asarray(jax.device_get(metrics.update_applied[0]))),
        "learning_rate": float(learning_rate),
    }


def _replicate_tree(tree: Any, devices: tuple[Any, ...]) -> Any:
    count = len(devices)
    return jax.tree_util.tree_map(
        lambda value: (
            jnp.asarray(
                np.broadcast_to(
                    np.asarray(jax.device_get(value)),
                    (count, *value.shape),
                ).copy()
            )
            if eqx.is_array(value)
            else value
        ),
        tree,
    )


def _unreplicate_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda value: value[0] if eqx.is_array(value) else value,
        tree,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
