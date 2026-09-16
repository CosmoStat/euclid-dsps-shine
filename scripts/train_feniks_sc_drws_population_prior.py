#!/usr/bin/env python3
"""Run the fixed-bank, selection-corrected parent-prior M-step on four GPUs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import wasserstein_distance

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.amortized.features import read_feature_stats
from euclid_dsps.amortized.latent import latent_spec_from_config, x_to_theta
from euclid_dsps.amortized.population_vem import (
    fixed_reference_selection_terms,
    iter_array_bank_shards,
    make_pmap_fixed_reference_prior_step,
    require_git_commit,
    resolve_manifest_config,
    sha256_file,
)
from euclid_dsps.amortized.train import load_checkpoint, save_checkpoint
from euclid_dsps.config import load_config

eqx, optax = require_amortized_dependencies()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _replicate_tree(tree, devices):
    count = len(devices)
    return jax.tree_util.tree_map(
        lambda value: (
            jnp.broadcast_to(value, (count, *value.shape))
            if eqx.is_array(value)
            else value
        ),
        tree,
    )


def _unreplicate_tree(tree):
    return jax.tree_util.tree_map(
        lambda value: value[0] if eqx.is_array(value) else value,
        tree,
    )


def _scalar(value) -> float:
    return float(np.asarray(jax.device_get(value)).reshape(-1)[0])


def _copy_checkpoint(source: Path, destination: Path) -> None:
    source_sidecar = source.with_suffix(source.suffix + ".json")
    if not source_sidecar.is_file():
        raise FileNotFoundError(source_sidecar)
    shutil.copy2(source, destination)
    shutil.copy2(
        source_sidecar,
        destination.with_suffix(destination.suffix + ".json"),
    )


def _validated_history(pass_root: Path, requested_passes: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing_seen = False
    for pass_index in range(1, int(requested_passes) + 1):
        receipt_path = pass_root / f"pass_{pass_index:02d}.json"
        if not receipt_path.is_file():
            missing_seen = True
            continue
        if missing_seen:
            raise ValueError("prior M-step receipts are not a contiguous prefix")
        record = _read_json(receipt_path)
        if (
            record.get("status") != "complete"
            or int(record.get("pass", -1)) != pass_index
            or record.get("truth_used") is not False
        ):
            raise ValueError(f"invalid prior M-step receipt: {receipt_path}")
        for path_key, hash_key in (
            ("checkpoint", "checkpoint_sha256"),
            ("checkpoint_sidecar", "checkpoint_sidecar_sha256"),
            ("optimizer_state", "optimizer_state_sha256"),
        ):
            path = Path(record[path_key])
            if not path.is_file() or sha256_file(path) != record[hash_key]:
                raise ValueError(f"prior M-step resume provenance mismatch: {path}")
        records.append(record)
    return records


def _load_q(path: Path) -> np.ndarray:
    parts = list(iter_array_bank_shards(path))
    return np.concatenate([np.asarray(part["x"], dtype=np.float32) for part in parts])


def _load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = list(iter_array_bank_shards(path))
    return (
        np.concatenate([part["x"] for part in parts]).astype(np.float32),
        np.concatenate([part["log_p_reference"] for part in parts]).astype(np.float32),
        np.concatenate([part["log_beta"] for part in parts]).astype(np.float64),
    )


def _validation_metrics(
    prior,
    source_prior,
    q_validation: np.ndarray,
    reference_x: np.ndarray,
    reference_log_prob: np.ndarray,
    reference_log_beta: np.ndarray,
    latent_spec,
) -> dict[str, float]:
    q_log_prob_parts = []
    for start in range(0, len(q_validation), 256):
        values = prior.log_prob(jnp.asarray(q_validation[start : start + 256]))
        q_log_prob_parts.append(np.asarray(jax.device_get(values)))
    q_log_prob = np.concatenate(q_log_prob_parts)
    if not np.all(np.isfinite(q_log_prob)):
        raise ValueError("candidate prior is non-finite on q validation draws")
    data_nll = float(-np.mean(q_log_prob))
    candidate_reference = np.asarray(
        jax.device_get(prior.log_prob(jnp.asarray(reference_x)))
    )
    terms = fixed_reference_selection_terms(
        jnp.asarray(candidate_reference),
        jnp.asarray(reference_log_prob),
        jnp.asarray(reference_log_beta),
    )
    if not bool(terms.finite):
        raise ValueError("candidate prior has no finite selection-reference support")
    source_reference = np.asarray(
        jax.device_get(source_prior.log_prob(jnp.asarray(reference_x)))
    )
    source_to_candidate_kl = float(np.mean(source_reference - candidate_reference))
    log_weight = reference_log_beta + candidate_reference - reference_log_prob
    finite = np.isfinite(log_weight)
    if not np.any(finite):
        raise ValueError("candidate prior has no finite selected-reference weight")
    shifted = np.where(finite, log_weight - np.max(log_weight[finite]), -np.inf)
    selected_weights = np.exp(shifted)
    selected_weights /= selected_weights.sum()
    q_theta = np.asarray(
        jax.device_get(x_to_theta(jnp.asarray(q_validation), latent_spec))
    )
    reference_theta = np.asarray(
        jax.device_get(x_to_theta(jnp.asarray(reference_x), latent_spec))
    )
    q_flat = q_theta.reshape(-1, q_theta.shape[-1])
    q_weights = np.full(len(q_flat), 1.0 / len(q_flat))
    normalized_w1 = []
    for dimension in range(q_flat.shape[-1]):
        scale = max(
            float(np.quantile(q_flat[:, dimension], 0.75))
            - float(np.quantile(q_flat[:, dimension], 0.25)),
            1.0e-6,
        )
        distance = wasserstein_distance(
            q_flat[:, dimension],
            reference_theta[:, dimension],
            u_weights=q_weights,
            v_weights=selected_weights,
        )
        normalized_w1.append(float(distance / scale))
    return {
        "selected_validation_objective": data_nll + float(terms.log_alpha),
        "q_data_nll": data_nll,
        "log_alpha": float(terms.log_alpha),
        "alpha": float(terms.alpha),
        "reference_ess_fraction": float(terms.ess_fraction),
        "alpha_relative_mc_error": float(terms.relative_mc_error),
        "maximum_reference_weight": float(terms.maximum_normalized_weight),
        "source_to_candidate_kl": source_to_candidate_kl,
        "selected_prior_vs_q_median_w1_over_q_iqr": float(np.median(normalized_w1)),
        "selected_prior_vs_q_redshift_w1_over_q_iqr": normalized_w1[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=8)
    parser.add_argument("--objects-per-step", type=int, default=1024)
    parser.add_argument("--peak-learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--final-learning-rate", type=float, default=2.0e-6)
    parser.add_argument("--trust-strength", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=266000)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = _read_json(root / "RUN_MANIFEST.json")
    repo = Path(__file__).resolve().parents[1]
    require_git_commit(repo, manifest["code_commit"])
    stage1 = _read_json(root / "STAGE1_PASS.json")
    if stage1.get("status") != "PASS":
        raise ValueError("prior M-step requires a passing stage-1 receipt")
    complete_path = root / "prior" / "PRIOR_MSTEP_COMPLETE.json"
    if complete_path.is_file():
        complete = _read_json(complete_path)
        checkpoint = Path(complete.get("checkpoint", ""))
        if (
            complete.get("status") != "COMPLETE"
            or complete.get("truth_used") is not False
            or not checkpoint.is_file()
            or sha256_file(checkpoint) != complete.get("checkpoint_sha256")
        ):
            raise ValueError("existing prior M-step receipt is invalid")
        print(f"[population-vem-prior] already complete: {complete_path}")
        return

    devices = tuple(jax.local_devices())
    if len(devices) != 4:
        raise RuntimeError(
            f"population prior M-step requires exactly four local GPUs, got {devices}"
        )
    if int(args.objects_per_step) % len(devices):
        raise ValueError("objects_per_step must be divisible by local GPU count")
    config = load_config(resolve_manifest_config(manifest, "config", repo))
    latent_spec = latent_spec_from_config(config)
    feature_stats_path = Path(manifest["frozen_source"]["feature_stats"])
    feature_stats = read_feature_stats(feature_stats_path)
    source_checkpoint = Path(manifest["frozen_source"]["checkpoint"])
    if sha256_file(source_checkpoint) != manifest["frozen_source"]["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint changed before prior M-step")
    source_sidecar = Path(manifest["frozen_source"]["checkpoint_sidecar"])
    if (
        sha256_file(source_sidecar)
        != manifest["frozen_source"]["checkpoint_sidecar_sha256"]
    ):
        raise ValueError("frozen checkpoint sidecar changed before prior M-step")
    source_model = load_checkpoint(source_checkpoint, config)
    q_fit = _load_q(root / "banks" / "q_fit" / "bank_manifest.json")
    q_validation = _load_q(root / "banks" / "q_validation" / "bank_manifest.json")
    reference_x, reference_log_prob, reference_log_beta = _load_reference(
        root / "banks" / "selection_reference" / "bank_manifest.json"
    )
    if len(q_fit) != int(manifest["banks"]["q_fit"]["objects"]):
        raise ValueError("q-fit bank object count mismatch")
    if len(q_validation) != int(manifest["banks"]["q_validation"]["objects"]):
        raise ValueError("q-validation bank object count mismatch")
    if len(reference_x) != int(manifest["banks"]["selection_reference"]["samples"]):
        raise ValueError("selection-reference sample count mismatch")
    if len(reference_x) % len(devices):
        raise ValueError("selection reference count must divide across GPUs")

    prior_root = root / "prior"
    checkpoint_root = prior_root / "checkpoints"
    state_root = prior_root / "states"
    pass_root = prior_root / "passes"
    for path in (checkpoint_root, state_root, pass_root):
        path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(feature_stats_path, prior_root / "feature_stats.json")

    steps_per_pass = int(math.ceil(len(q_fit) / int(args.objects_per_step)))
    total_steps = int(args.passes) * steps_per_pass
    schedule = optax.cosine_decay_schedule(
        init_value=float(args.peak_learning_rate),
        decay_steps=max(total_steps, 1),
        alpha=float(args.final_learning_rate) / float(args.peak_learning_rate),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adamw(schedule, weight_decay=1.0e-6),
    )

    completed = _validated_history(pass_root, int(args.passes))
    if completed:
        latest = completed[-1]
        candidate_model = load_checkpoint(latest["checkpoint"], config)
        state_template = optimizer.init(
            eqx.filter(candidate_model.prior, eqx.is_inexact_array)
        )
        optimizer_state = eqx.tree_deserialise_leaves(
            latest["optimizer_state"], state_template
        )
        first_pass = int(latest["pass"]) + 1
        print(
            f"[population-vem-prior] resuming after pass {latest['pass']}",
            flush=True,
        )
    else:
        candidate_model = source_model
        optimizer_state = optimizer.init(
            eqx.filter(candidate_model.prior, eqx.is_inexact_array)
        )
        first_pass = 1

    history = completed[:]
    baseline = _validation_metrics(
        source_model.prior,
        source_model.prior,
        q_validation,
        reference_x,
        reference_log_prob,
        reference_log_beta,
        latent_spec,
    )
    _write_json(prior_root / "BASELINE.json", baseline)
    candidates = [
        (float(baseline["selected_validation_objective"]), 0, source_checkpoint)
    ]
    candidates.extend(
        (
            float(record["validation"]["selected_validation_objective"]),
            int(record["pass"]),
            Path(record["checkpoint"]),
        )
        for record in history
    )
    best_score, best_pass, best_source = min(candidates, key=lambda item: item[0])
    best_checkpoint = checkpoint_root / "best.eqx"
    if not best_checkpoint.is_file() or sha256_file(best_checkpoint) != sha256_file(
        best_source
    ):
        _copy_checkpoint(best_source, best_checkpoint)
    stale_passes = int(history[-1]["pass"]) - best_pass if history else 0

    step = make_pmap_fixed_reference_prior_step(
        optimizer=optimizer,
        minimum_reference_ess_fraction=0.10,
        maximum_alpha_relative_mc_error=0.05,
        maximum_kl_per_dimension=0.03,
    )
    replicated_prior = _replicate_tree(candidate_model.prior, devices)
    replicated_source = _replicate_tree(source_model.prior, devices)
    replicated_optimizer = _replicate_tree(optimizer_state, devices)
    references_per_device = len(reference_x) // len(devices)
    sharded_reference_x = jnp.asarray(reference_x).reshape(
        len(devices), references_per_device, reference_x.shape[-1]
    )
    sharded_reference_log_prob = jnp.asarray(reference_log_prob).reshape(
        len(devices), references_per_device
    )
    sharded_reference_log_beta = jnp.asarray(reference_log_beta).reshape(
        len(devices), references_per_device
    )
    for pass_index in range(first_pass, int(args.passes) + 1):
        rng = np.random.default_rng(int(args.seed) + pass_index)
        order = rng.permutation(len(q_fit))
        pass_updates = []
        for batch_index, start in enumerate(
            range(0, len(order), int(args.objects_per_step)), start=1
        ):
            indices = order[start : start + int(args.objects_per_step)]
            if len(indices) < int(args.objects_per_step):
                padding = rng.choice(
                    order,
                    size=int(args.objects_per_step) - len(indices),
                    replace=True,
                )
                indices = np.concatenate((indices, padding))
            batch = q_fit[indices]
            local_objects = int(args.objects_per_step) // len(devices)
            sharded_q = jnp.asarray(batch).reshape(
                len(devices), local_objects, batch.shape[1], batch.shape[2]
            )
            valid = jnp.ones((len(devices), local_objects), dtype=jnp.bool_)
            replicated_prior, replicated_optimizer, metrics = step(
                replicated_prior,
                replicated_source,
                replicated_optimizer,
                sharded_q,
                valid,
                sharded_reference_x,
                sharded_reference_log_prob,
                sharded_reference_log_beta,
                jnp.full((len(devices),), float(args.trust_strength)),
            )
            record = {
                "pass": pass_index,
                "batch": batch_index,
                **{name: _scalar(value) for name, value in metrics._asdict().items()},
            }
            pass_updates.append(record)
            print(
                "[population-vem-prior] "
                f"pass={pass_index}/{args.passes} "
                f"step={batch_index}/{steps_per_pass} "
                f"loss={record['loss']:.5f} alpha={record['alpha']:.5f} "
                f"refESS={record['reference_ess_fraction']:.3f} "
                f"applied={int(record['update_applied'])}",
                flush=True,
            )
        candidate_prior = _unreplicate_tree(replicated_prior)
        optimizer_state = _unreplicate_tree(replicated_optimizer)
        candidate_model = eqx.tree_at(
            lambda model: model.prior, candidate_model, candidate_prior
        )
        validation = _validation_metrics(
            candidate_prior,
            source_model.prior,
            q_validation,
            reference_x,
            reference_log_prob,
            reference_log_beta,
            latent_spec,
        )
        checkpoint_path = checkpoint_root / f"pass_{pass_index:02d}.eqx"
        save_checkpoint(
            checkpoint_path,
            candidate_model,
            config=config,
            latent_spec=latent_spec,
            feature_stats=feature_stats,
            epoch=int(manifest["frozen_source"]["epoch"]),
            metric=validation["selected_validation_objective"],
            metric_name="fixed_q_selected_validation_objective",
        )
        optimizer_path = state_root / f"pass_{pass_index:02d}.opt.eqx"
        eqx.tree_serialise_leaves(optimizer_path, optimizer_state)
        pass_receipt = {
            "status": "complete",
            "pass": pass_index,
            "updates": len(pass_updates),
            "applied_updates": int(
                sum(record["update_applied"] > 0.5 for record in pass_updates)
            ),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_sidecar": str(
                checkpoint_path.with_suffix(".eqx.json").resolve()
            ),
            "checkpoint_sidecar_sha256": sha256_file(
                checkpoint_path.with_suffix(".eqx.json")
            ),
            "optimizer_state": str(optimizer_path.resolve()),
            "optimizer_state_sha256": sha256_file(optimizer_path),
            "validation": validation,
            "truth_used": False,
            "dsps_calls_inside_optimizer": 0,
        }
        _write_json(pass_root / f"pass_{pass_index:02d}.json", pass_receipt)
        history.append(pass_receipt)
        score = validation["selected_validation_objective"]
        if score < best_score - 1.0e-4:
            best_score = score
            best_pass = pass_index
            stale_passes = 0
            _copy_checkpoint(checkpoint_path, best_checkpoint)
        else:
            stale_passes += 1
        _write_json(
            prior_root / "PROGRESS.json",
            {
                "status": "running",
                "completed_passes": pass_index,
                "requested_passes": int(args.passes),
                "best_pass": best_pass,
                "best_score": best_score,
                "stale_passes": stale_passes,
                "latest_validation": validation,
            },
        )
        if pass_index >= 3 and stale_passes >= int(args.patience):
            print(
                f"[population-vem-prior] early stop after pass {pass_index}",
                flush=True,
            )
            break

    if not best_checkpoint.is_file():
        raise RuntimeError("prior M-step produced no valid checkpoint")
    best_model = load_checkpoint(best_checkpoint, config)
    final_validation = _validation_metrics(
        best_model.prior,
        source_model.prior,
        q_validation,
        reference_x,
        reference_log_prob,
        reference_log_beta,
        latent_spec,
    )
    receipt = {
        "status": "COMPLETE",
        "stage": 2,
        "source_epoch": int(manifest["frozen_source"]["epoch"]),
        "vem_iteration": int(manifest.get("iteration", 1)),
        "checkpoint": str(best_checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(best_checkpoint),
        "checkpoint_sidecar": str(best_checkpoint.with_suffix(".eqx.json").resolve()),
        "checkpoint_sidecar_sha256": sha256_file(
            best_checkpoint.with_suffix(".eqx.json")
        ),
        "best_pass": best_pass,
        "improved_over_source": bool(best_pass > 0),
        "passes_completed": len(history),
        "updates_applied": int(sum(record["applied_updates"] for record in history)),
        "baseline": baseline,
        "final_validation": final_validation,
        "truth_used": False,
        "dsps_calls_inside_optimizer": 0,
        "posterior_role": "frozen approximate q bank with equal object weights",
        "target": "p_eta(theta|C0) with log alpha_eta selection correction",
    }
    _write_json(complete_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
