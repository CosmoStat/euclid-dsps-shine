"""Selection-corrected prior M-step over a frozen posterior bank."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.io import ensure_dir, write_json

from .adaptive_smc_trainer import _make_selection_log_alpha_fn
from .adaptive_smc_training import (
    SMCPosteriorBatch,
    make_component_optimizer,
    make_pmap_prior_macro_step,
    smc_prior_mstep_terms,
    snapshot_model,
)
from .config import require_amortized_dependencies
from .features import feature_stats_hash
from .latent import latent_spec_hash
from .posterior_bank import (
    C0_SCOPE_STATEMENT,
    POSTERIOR_METHOD_CODES,
    PosteriorBankShard,
    read_posterior_bank_shard,
    sha256_file,
    validate_posterior_bank_manifest_provenance,
)
from .sc_asmc_config import sc_asmc_em_config_hash, validate_sc_asmc_em_config
from .sc_asmc_training import (
    RuntimeBundle,
    _replicate_tree,
    _unreplicate_tree,
    load_sc_model,
    save_component_checkpoint,
    tree_semantic_hash,
    validate_component_checkpoint,
)
from .train import _selection_log_beta_from_prior_samples

eqx, _optax = require_amortized_dependencies()


def run_prior_mstep(
    config: dict[str, Any],
    runtime: RuntimeBundle,
    *,
    input_bank_manifest: str | Path,
    heldout_rows: np.ndarray,
    q_checkpoint: str | Path,
    old_prior_checkpoint: str | Path,
    out_dir: str | Path,
    iteration: int,
    seed: int,
    resume: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Choose an update count on held-out rows, then refit using every row."""
    validate_sc_asmc_em_config(config)
    if int(iteration) not in {1, 2}:
        raise ValueError("prior M-step iteration must be one or two")
    output = ensure_dir(out_dir)
    receipt_path = output / f"prior_mstep_{int(iteration)}_receipt.json"
    bank_path = Path(input_bank_manifest).resolve()
    bank_manifest = json.loads(bank_path.read_text(encoding="utf-8"))
    input_hashes = {
        "input_bank_manifest_sha256": sha256_file(bank_path),
        "input_q_checkpoint_sha256": sha256_file(q_checkpoint),
        "input_prior_checkpoint_sha256": sha256_file(old_prior_checkpoint),
    }
    validate_posterior_bank_manifest_provenance(
        bank_manifest,
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
            raise ValueError("prior M-step resume inputs changed")
        validate_component_checkpoint(
            payload["prior_checkpoint"], payload["prior_sha256"], runtime
        )
        return payload
    old_model = load_sc_model(
        config,
        runtime,
        q_checkpoint=q_checkpoint,
        prior_checkpoint=old_prior_checkpoint,
    )
    q_hash_before = tree_semantic_hash(old_model.encoder)
    prior_snapshot = snapshot_model(old_model)
    raw = dict(
        ((config.get("amortized", {}) or {}).get("sc_asmc_em", {}) or {}).get(
            "prior_mstep", {}
        )
        or {}
    )
    max_steps = int(raw.get("max_steps", 100))
    min_steps = int(raw.get("min_steps", 50))
    evaluation_every = int(raw.get("evaluation_every", 5))
    patience = int(raw.get("early_stopping_patience_steps", 20))
    macro_objects = int(raw.get("macro_objects", 512))
    learning_rate = float(raw.get("learning_rate", 1.0e-5))
    gradient_clip = float(raw.get("gradient_clip_norm", 5.0))
    optimizer = make_component_optimizer(
        learning_rate=learning_rate,
        gradient_clip_norm=gradient_clip,
        weight_decay=0.0,
    )
    selection_fn = _make_selection_log_alpha_fn(runtime)
    selection_config = runtime.selection_objective_config["selection_correction"]
    devices = tuple(jax.local_devices())

    def selection_log_beta(candidate_model, samples):
        return _selection_log_beta_from_prior_samples(
            candidate_model,
            samples,
            runtime.jit_latent_spec,
            runtime.context,
            runtime.model_args,
            runtime.parameter_names,
            runtime.calibration_config,
            selection_config,
        )

    pmap_step = make_pmap_prior_macro_step(
        optimizer=optimizer,
        selection_log_beta_fn=selection_log_beta,
        total_selection_samples=int(selection_config["n_prior_samples"]),
        total_trust_samples=int(raw.get("trust_samples", 1024)),
        trust_strength=float(raw.get("trust_strength", 0.2)),
        max_kl_per_dimension=float(raw.get("max_kl_per_dimension", 0.05)),
        max_alpha_mc_relative_error=float(raw.get("max_alpha_mc_relative_error", 0.15)),
        gradient_clip_norm=gradient_clip,
        n_devices=len(devices),
    )
    heldout = set(np.asarray(heldout_rows, dtype=np.int64).tolist())
    all_rows = _manifest_rows(bank_manifest)
    train_rows = set(all_rows.tolist()) - heldout
    if not train_rows or not heldout:
        raise ValueError("prior M-step requires non-empty train and held-out rows")

    pilot_model = old_model
    pilot_state = optimizer.init(eqx.filter(pilot_model.prior, eqx.is_inexact_array))
    pilot_replicated = _replicate_tree(pilot_model, devices)
    pilot_state_replicated = _replicate_tree(pilot_state, devices)
    snapshot_replicated = _replicate_tree(prior_snapshot, devices)
    pilot_log: list[dict[str, Any]] = []
    best_score = float("inf")
    best_step = -1
    step = 0
    epoch = 0
    while step < max_steps:
        for posterior in iter_bank_macro_batches(
            bank_manifest,
            include_rows=train_rows,
            macro_objects=macro_objects,
            shuffle_seed=int(seed) + epoch,
        ):
            step += 1
            pilot_replicated, pilot_state_replicated, replicated_metrics = pmap_step(
                pilot_replicated,
                snapshot_replicated,
                pilot_state_replicated,
                _shard_prior_posterior(posterior, len(devices)),
                jax.random.split(
                    jax.random.fold_in(jax.random.PRNGKey(seed), 10_000 + step),
                    len(devices),
                ),
                jax.random.split(
                    jax.random.fold_in(jax.random.PRNGKey(seed), 20_000 + step),
                    len(devices),
                ),
            )
            pilot_model = _unreplicate_tree(pilot_replicated)
            metrics = _unreplicate_tree(replicated_metrics)
            row = _prior_metrics_row(metrics, step=step, stage="pilot")
            if step % evaluation_every == 0 or step == max_steps:
                score = evaluate_selected_catalog_score(
                    pilot_model,
                    prior_snapshot,
                    bank_manifest,
                    include_rows=heldout,
                    selection_fn=selection_fn,
                    key=jax.random.fold_in(jax.random.PRNGKey(seed), 30_000 + step),
                )
                row.update(score)
                if (
                    step >= min_steps
                    and score["heldout_selected_catalog_score"] < best_score
                    and bool(row["update_applied"])
                ):
                    best_score = float(score["heldout_selected_catalog_score"])
                    best_step = step
            pilot_log.append(row)
            if verbose:
                print(
                    "[sc-asmc][prior-mstep] "
                    f"iteration={iteration} pilot_step={step}/{max_steps} "
                    f"loss={row['loss']:.5f} alpha={row['alpha']:.5f} "
                    f"kl/d={row['proposed_kl_per_dimension']:.4g} "
                    f"applied={int(row['update_applied'])}",
                    flush=True,
                )
            if step >= max_steps:
                break
            if best_step >= min_steps and step - best_step >= patience:
                break
        if step >= max_steps or (
            best_step >= min_steps and step - best_step >= patience
        ):
            break
        epoch += 1
    if best_step < min_steps:
        raise RuntimeError(
            "prior M-step found no finite held-out checkpoint after minimum steps"
        )

    # Replay the selected step count from the immutable old prior using every
    # resolved catalogue object. This removes held-out selection bias from the
    # final p_eta while preserving the early-stopped optimization budget.
    model = old_model
    optimizer_state = optimizer.init(eqx.filter(model.prior, eqx.is_inexact_array))
    model_replicated = _replicate_tree(model, devices)
    optimizer_state_replicated = _replicate_tree(optimizer_state, devices)
    final_log: list[dict[str, Any]] = []
    applied = 0
    replay_step = 0
    epoch = 0
    all_row_set = set(all_rows.tolist())
    resolved_objects = _resolved_row_count(bank_manifest, all_row_set)
    minimum_full_pass_steps = int(np.ceil(resolved_objects / macro_objects))
    replay_target_steps = max(best_step, minimum_full_pass_steps)
    if replay_target_steps > max_steps:
        raise RuntimeError(
            "prior M-step max_steps is too small to use every resolved object"
        )
    while replay_step < replay_target_steps:
        for posterior in iter_bank_macro_batches(
            bank_manifest,
            include_rows=all_row_set,
            macro_objects=macro_objects,
            shuffle_seed=int(seed) + 100_000 + epoch,
        ):
            replay_step += 1
            model_replicated, optimizer_state_replicated, replicated_metrics = (
                pmap_step(
                    model_replicated,
                    snapshot_replicated,
                    optimizer_state_replicated,
                    _shard_prior_posterior(posterior, len(devices)),
                    jax.random.split(
                        jax.random.fold_in(
                            jax.random.PRNGKey(seed), 110_000 + replay_step
                        ),
                        len(devices),
                    ),
                    jax.random.split(
                        jax.random.fold_in(
                            jax.random.PRNGKey(seed), 120_000 + replay_step
                        ),
                        len(devices),
                    ),
                )
            )
            model = _unreplicate_tree(model_replicated)
            metrics = _unreplicate_tree(replicated_metrics)
            row = _prior_metrics_row(metrics, step=replay_step, stage="all_data_replay")
            applied += int(row["update_applied"])
            final_log.append(row)
            if replay_step >= replay_target_steps:
                break
        epoch += 1
    if applied <= 0:
        raise RuntimeError("prior M-step replay applied no trusted update")
    if tree_semantic_hash(model.encoder) != q_hash_before:
        raise RuntimeError("q changed during the prior-only M-step")
    checkpoint = output / "components" / f"p{int(iteration)}.eqx"
    checkpoint_record = save_component_checkpoint(
        checkpoint,
        model.prior,
        component=f"prior_p{int(iteration)}",
        config=config,
        runtime=runtime,
        phase=f"prior_mstep_{int(iteration)}",
        extra={
            "selected_pilot_step": best_step,
            "heldout_selected_catalog_score": best_score,
            "applied_all_data_updates": applied,
        },
    )
    _write_jsonl(output / f"prior_mstep_{int(iteration)}_pilot.jsonl", pilot_log)
    _write_jsonl(output / f"prior_mstep_{int(iteration)}_all_data.jsonl", final_log)
    last = final_log[-1]
    payload = {
        "status": "PASS",
        "phase": f"prior_mstep_{int(iteration)}",
        "iteration": int(iteration),
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "truth_used": False,
        "posterior_bank_frozen": True,
        "q_frozen": True,
        "trainable_components": ["prior"],
        "objective": (
            "-mean_i sum_k stop(w_ik) log p_eta(x_ik) + log(alpha_eta) "
            "+ lambda_trust KL(p_eta_old || p_eta)"
        ),
        "score_gradient": ("sum_k (normalized_beta_k - 1/M) grad log p_eta(x_k)"),
        "data_parallel_devices": len(devices),
        "selection_score_normalization": "global across all local pmap devices",
        "selected_pilot_step": best_step,
        "all_data_replay_steps": replay_target_steps,
        "heldout_selected_catalog_score": best_score,
        "all_resolved_rows_used_in_final_replay": True,
        "applied_all_data_updates": applied,
        "prior_checkpoint": checkpoint_record["path"],
        "prior_sha256": checkpoint_record["sha256"],
        **input_hashes,
        "q_semantic_hash_before": q_hash_before,
        "q_semantic_hash_after": tree_semantic_hash(model.encoder),
        "final_diagnostics": last,
    }
    write_json(receipt_path, payload)
    return payload


def iter_bank_macro_batches(
    manifest: dict[str, Any],
    *,
    include_rows: set[int],
    macro_objects: int,
    shuffle_seed: int,
) -> Iterator[SMCPosteriorBatch]:
    """Stream shards and yield object mini-batches without global concatenation."""
    records = list(manifest["shards"])
    rng = np.random.default_rng(int(shuffle_seed))
    rng.shuffle(records)
    pending: list[SMCPosteriorBatch] = []
    pending_count = 0
    for record in records:
        shard = read_posterior_bank_shard(record["path"])
        selected = np.asarray(
            [int(row) in include_rows for row in shard.row_index], dtype=bool
        )
        selected &= np.asarray(shard.resolved, dtype=bool)
        indices = np.flatnonzero(selected)
        if not len(indices):
            continue
        rng.shuffle(indices)
        for start in range(0, len(indices), int(macro_objects)):
            value = _posterior_from_bank_shard(
                shard,
                indices[start : start + int(macro_objects)],
            )
            pending.append(value)
            pending_count += int(value.eligible.shape[0])
            if pending_count >= int(macro_objects):
                combined = _concat_posteriors(pending)
                yield _take_posterior(combined, np.arange(int(macro_objects)))
                remainder = np.arange(int(macro_objects), pending_count)
                pending = (
                    [_take_posterior(combined, remainder)] if len(remainder) else []
                )
                pending_count = int(len(remainder))
    if pending:
        yield _concat_posteriors(pending)


def evaluate_selected_catalog_score(
    model: Any,
    prior_snapshot: Any,
    manifest: dict[str, Any],
    *,
    include_rows: set[int],
    selection_fn: Any,
    key: jax.Array,
) -> dict[str, float]:
    numerator = 0.0
    count = 0.0
    for batch in iter_bank_macro_batches(
        manifest,
        include_rows=include_rows,
        macro_objects=512,
        shuffle_seed=0,
    ):
        samples = prior_snapshot.prior.sample(
            jax.random.fold_in(key, int(count) + 1), 256
        )
        terms = smc_prior_mstep_terms(
            model.prior,
            prior_snapshot.prior,
            batch,
            samples,
        )
        objects = float(np.asarray(terms.eligible_count))
        numerator += float(np.asarray(terms.data_nll)) * objects
        count += objects
    if count <= 0.0:
        raise RuntimeError("held-out selected-catalog score has no resolved object")
    log_alpha, metrics = selection_fn(model, key)
    data_nll = numerator / count
    value = data_nll + float(np.asarray(log_alpha))
    return {
        "heldout_data_nll": data_nll,
        "heldout_log_alpha": float(np.asarray(log_alpha)),
        "heldout_alpha": float(np.exp(np.asarray(log_alpha))),
        "heldout_alpha_mc_relative_error": float(
            np.asarray(metrics["selection/alpha_mc_relative_error"])
        ),
        "heldout_selected_catalog_score": value,
    }


def _posterior_from_bank_shard(
    shard: PosteriorBankShard,
    indices: np.ndarray,
) -> SMCPosteriorBatch:
    selected = np.asarray(indices, dtype=np.int64)
    particles = jnp.asarray(shard.particles[selected]).transpose(1, 0, 2)
    weights = jnp.asarray(shard.normalized_weights[selected]).T
    resolved = jnp.asarray(shard.resolved[selected])
    method = np.asarray(shard.method[selected])
    hard = ~resolved
    zeros = jnp.zeros(len(selected), dtype=jnp.float32)
    return SMCPosteriorBatch(
        particles=particles,
        normalized_weights=weights,
        eligible=resolved,
        beta_final=jnp.asarray(shard.beta_final[selected]),
        final_ess=jnp.asarray(shard.ess[selected]),
        final_max_weight=jnp.asarray(shard.max_weight[selected]),
        mutation_acceptance=jnp.asarray(shard.acceptance[selected]),
        final_rw_scale=zeros,
        unique_ancestor_fraction=jnp.asarray(shard.unique_ancestor_fraction[selected]),
        ancestor_ess=jnp.asarray(shard.ancestor_ess[selected]),
        ancestor_ess_fraction=jnp.asarray(shard.ancestor_ess[selected])
        / jnp.maximum(jnp.asarray(shard.particle_count[selected]), 1),
        epsilon_squared_jump=jnp.asarray(shard.movement_squared[selected]),
        median_epsilon_squared_jump=jnp.asarray(shard.movement_squared[selected]),
        moved_particle_fraction=jnp.asarray(shard.moved_particle_fraction[selected]),
        unchanged_from_ancestor_fraction=1.0
        - jnp.asarray(shard.moved_particle_fraction[selected]),
        poor_acceptance=hard,
        poor_ancestry=hard,
        poor_movement=hard,
        mixing_failure=hard,
        logZ_estimate=jnp.asarray(shard.logz[selected]),
        fallback_attempted=jnp.asarray(
            method >= POSTERIOR_METHOD_CODES["fallback SMC"]
        ),
        fallback_succeeded=jnp.asarray(
            np.isin(
                method,
                [
                    POSTERIOR_METHOD_CODES["fallback SMC"],
                    POSTERIOR_METHOD_CODES["extended SMC"],
                ],
            )
        ),
    )


def _take_posterior(
    posterior: SMCPosteriorBatch,
    indices: np.ndarray,
) -> SMCPosteriorBatch:
    selected = jnp.asarray(indices, dtype=jnp.int32)
    return SMCPosteriorBatch(
        *(
            jnp.take(
                value,
                selected,
                axis=1 if name in {"particles", "normalized_weights"} else 0,
            )
            for name, value in zip(posterior._fields, posterior, strict=True)
        )
    )


def _concat_posteriors(values: list[SMCPosteriorBatch]) -> SMCPosteriorBatch:
    return SMCPosteriorBatch(
        *(
            jnp.concatenate(
                [getattr(value, name) for value in values],
                axis=1 if name in {"particles", "normalized_weights"} else 0,
            )
            for name in values[0]._fields
        )
    )


def _shard_prior_posterior(
    posterior: SMCPosteriorBatch,
    n_devices: int,
) -> SMCPosteriorBatch:
    """Pad the object axis and expose an explicit local-device axis."""
    objects = int(posterior.eligible.shape[0])
    target = int(np.ceil(objects / int(n_devices))) * int(n_devices)
    indices = np.arange(target, dtype=np.int64) % objects
    padded = _take_posterior(posterior, indices)
    padded = padded._replace(eligible=padded.eligible & (jnp.arange(target) < objects))
    local = target // int(n_devices)
    fields = []
    for name, value in zip(padded._fields, padded, strict=True):
        array = jnp.asarray(value)
        if name in {"particles", "normalized_weights"}:
            axes = (1, 0, 2, *range(3, array.ndim + 1))
            sharded = array.reshape(
                array.shape[0], int(n_devices), local, *array.shape[2:]
            ).transpose(axes)
        else:
            sharded = array.reshape(int(n_devices), local, *array.shape[1:])
        fields.append(sharded)
    return SMCPosteriorBatch(*fields)


def _manifest_rows(manifest: dict[str, Any]) -> np.ndarray:
    rows = []
    for record in manifest["shards"]:
        shard = read_posterior_bank_shard(record["path"])
        rows.append(np.asarray(shard.row_index, dtype=np.int64))
    values = np.concatenate(rows)
    if len(np.unique(values)) != len(values):
        raise ValueError("posterior bank has duplicate rows")
    return values


def _resolved_row_count(manifest: dict[str, Any], include_rows: set[int]) -> int:
    count = 0
    for record in manifest["shards"]:
        shard = read_posterior_bank_shard(record["path"])
        count += int(
            np.sum(
                np.asarray(shard.resolved, dtype=bool)
                & np.asarray(
                    [int(row) in include_rows for row in shard.row_index],
                    dtype=bool,
                )
            )
        )
    return count


def _prior_metrics_row(metrics: Any, *, step: int, stage: str) -> dict[str, Any]:
    dimension = 15.0
    return {
        "stage": stage,
        "step": int(step),
        "loss": float(np.asarray(metrics.loss)),
        "data_nll": float(np.asarray(metrics.data_nll)),
        "log_alpha": float(np.asarray(metrics.selection_log_alpha)),
        "alpha": float(np.asarray(metrics.selection_alpha)),
        "alpha_mc_relative_error": float(
            np.asarray(metrics.selection_alpha_mc_relative_error)
        ),
        "score_weight_ess": float(np.asarray(metrics.selection_score_weight_ess)),
        "maximum_score_weight": float(
            np.asarray(metrics.selection_maximum_score_weight)
        ),
        "score_weights_finite": bool(
            np.asarray(metrics.selection_score_weights_finite)
        ),
        "score_gradient_norm": float(np.asarray(metrics.selection_grad_norm)),
        "data_gradient_norm": float(np.asarray(metrics.data_grad_norm)),
        "trust_gradient_norm": float(np.asarray(metrics.trust_grad_norm)),
        "gradient_norm": float(np.asarray(metrics.raw_grad_norm)),
        "gradients_finite": bool(np.asarray(metrics.grads_finite)),
        "score_gradient_finite": bool(np.asarray(metrics.selection_grads_finite)),
        "proposed_kl": float(np.asarray(metrics.prior_kl_proposed)),
        "proposed_kl_per_dimension": float(
            np.asarray(metrics.prior_kl_proposed) / dimension
        ),
        "update_applied": bool(np.asarray(metrics.update_applied)),
        "rejection_code": int(np.asarray(metrics.rejection_code)),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
