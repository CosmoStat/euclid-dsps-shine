#!/usr/bin/env python3
"""Fit selected and inverse-selection parent flows on fixed joint q draws."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.amortized.features import read_feature_stats
from euclid_dsps.amortized.latent import latent_spec_from_config
from euclid_dsps.amortized.population_projection import (
    inverse_selection_weights,
    make_pmap_weighted_density_step,
    require_projection_runtime_commit,
)
from euclid_dsps.amortized.population_vem import (
    iter_array_bank_shards,
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


def _load_q(path: Path) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(part["x"], dtype=np.float32).reshape(-1, part["x"].shape[-1])
            for part in iter_array_bank_shards(path)
        ]
    )


def _load_beta_target(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    parts = list(iter_array_bank_shards(path))
    x = np.concatenate(
        [
            np.asarray(part["x"], dtype=np.float32).reshape(-1, part["x"].shape[-1])
            for part in parts
        ]
    )
    log_beta = np.concatenate(
        [np.asarray(part["log_beta"], dtype=np.float64).reshape(-1) for part in parts]
    )
    weights, diagnostics = inverse_selection_weights(log_beta)
    return x, weights, diagnostics


def _validation_nll(prior, x: np.ndarray, weights: np.ndarray) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(x), 65536):
        stop = min(start + 65536, len(x))
        log_prob = np.asarray(
            jax.device_get(prior.log_prob(jnp.asarray(x[start:stop]))),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(log_prob)):
            raise ValueError("projection flow is non-finite on validation draws")
        total += float(np.sum(weights[start:stop] * -log_prob))
        count += stop - start
    return total / count


def _model_with_prior(model, prior):
    return eqx.tree_at(lambda item: item.prior, model, prior)


def _fit_target(
    *,
    name: str,
    initial_prior,
    source_model,
    train_x: np.ndarray,
    train_weights: np.ndarray,
    validation_x: np.ndarray,
    validation_weights: np.ndarray,
    output: Path,
    config: dict[str, Any],
    latent_spec,
    feature_stats,
    feature_stats_path: Path,
    devices,
    passes: int,
    samples_per_step: int,
    peak_learning_rate: float,
    final_learning_rate: float,
    patience: int,
    seed: int,
    progress_path: Path,
) -> tuple[Any, dict[str, Any]]:
    if len(train_x) != len(train_weights) or len(validation_x) != len(
        validation_weights
    ):
        raise ValueError(f"{name} target arrays have incompatible lengths")
    if int(samples_per_step) % len(devices):
        raise ValueError("samples_per_step must divide across local GPUs")
    steps_per_pass = int(math.ceil(len(train_x) / int(samples_per_step)))
    schedule = optax.cosine_decay_schedule(
        init_value=float(peak_learning_rate),
        decay_steps=max(int(passes) * steps_per_pass, 1),
        alpha=float(final_learning_rate) / float(peak_learning_rate),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adamw(schedule, weight_decay=1.0e-6),
    )
    optimizer_state = optimizer.init(eqx.filter(initial_prior, eqx.is_inexact_array))
    prior = _replicate_tree(initial_prior, devices)
    optimizer_state = _replicate_tree(optimizer_state, devices)
    step = make_pmap_weighted_density_step(optimizer=optimizer)
    rng = np.random.default_rng(int(seed))
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_prior = initial_prior
    stale = 0
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(feature_stats_path, output / "feature_stats.json")
    checkpoint = output / "checkpoints" / "best.eqx"

    for pass_index in range(1, int(passes) + 1):
        order = rng.permutation(len(train_x))
        losses = []
        applied = 0
        for start in range(0, len(order), int(samples_per_step)):
            indices = order[start : start + int(samples_per_step)]
            batch_x = np.zeros(
                (int(samples_per_step), train_x.shape[-1]), dtype=np.float32
            )
            batch_weight = np.zeros(int(samples_per_step), dtype=np.float32)
            batch_valid = np.zeros(int(samples_per_step), dtype=bool)
            batch_x[: len(indices)] = train_x[indices]
            batch_weight[: len(indices)] = train_weights[indices]
            batch_valid[: len(indices)] = True
            per_device = int(samples_per_step) // len(devices)
            prior, optimizer_state, metrics = step(
                prior,
                optimizer_state,
                jnp.asarray(batch_x.reshape(len(devices), per_device, -1)),
                jnp.asarray(batch_weight.reshape(len(devices), per_device)),
                jnp.asarray(batch_valid.reshape(len(devices), per_device)),
            )
            losses.append(_scalar(metrics.loss))
            applied += int(_scalar(metrics.update_applied) > 0.5)
        candidate = _unreplicate_tree(prior)
        validation_nll = _validation_nll(candidate, validation_x, validation_weights)
        record = {
            "pass": pass_index,
            "training_loss": float(np.mean(losses)),
            "validation_weighted_nll": validation_nll,
            "updates_applied": applied,
            "updates_expected": steps_per_pass,
        }
        history.append(record)
        if validation_nll < best_score - 1.0e-4:
            best_score = validation_nll
            best_prior = candidate
            stale = 0
            save_checkpoint(
                checkpoint,
                _model_with_prior(source_model, candidate),
                config=config,
                latent_spec=latent_spec,
                feature_stats=feature_stats,
                epoch=pass_index,
                metric=validation_nll,
                metric_name="weighted_validation_nll",
            )
        else:
            stale += 1
        _write_json(
            progress_path,
            {
                "status": "running",
                "target": name,
                "pass": pass_index,
                "passes_requested": int(passes),
                "best_validation_weighted_nll": best_score,
                "stale_passes": stale,
                "history": history,
            },
        )
        print(
            f"[population-projection-fit] target={name} pass={pass_index}/{passes} "
            f"train={record['training_loss']:.6f} validation={validation_nll:.6f} "
            f"applied={applied}/{steps_per_pass}",
            flush=True,
        )
        if pass_index >= 5 and stale >= int(patience):
            break
    if not checkpoint.is_file():
        raise RuntimeError(f"{name} projection produced no checkpoint")
    return best_prior, {
        "status": "complete",
        "target": name,
        "training_samples": int(len(train_x)),
        "validation_samples": int(len(validation_x)),
        "passes_completed": len(history),
        "best_validation_weighted_nll": best_score,
        "checkpoint": str(checkpoint),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=16)
    parser.add_argument("--samples-per-step", type=int, default=32768)
    parser.add_argument("--peak-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--final-learning-rate", type=float, default=2.0e-6)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=269100)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = _read_json(root / "RUN_MANIFEST.json")
    repo = Path(__file__).resolve().parents[1]
    runtime_provenance = require_projection_runtime_commit(root, manifest, repo)
    beta_receipt = _read_json(root / "BETA_TARGET_COMPLETE.json")
    if (
        beta_receipt.get("status") != "PASS"
        or beta_receipt.get("truth_used") is not False
    ):
        raise ValueError("flow fitting requires a passing truth-free beta target")
    complete_path = root / "PROJECTION_FIT_COMPLETE.json"
    if complete_path.is_file():
        existing = _read_json(complete_path)
        if existing.get("status") != "COMPLETE":
            raise ValueError("existing projection-fit receipt is invalid")
        print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
        return

    devices = tuple(jax.local_devices())
    if len(devices) != 4:
        raise RuntimeError(
            f"population projection requires exactly four local GPUs, got {devices}"
        )
    config = load_config(resolve_manifest_config(manifest, "config", repo))
    latent_spec = latent_spec_from_config(config)
    checkpoint = Path(manifest["source"]["checkpoint"])
    if sha256_file(checkpoint) != manifest["source"]["checkpoint_sha256"]:
        raise ValueError("source checkpoint changed before flow projection")
    source_model = load_checkpoint(checkpoint, config)
    feature_stats_path = Path(manifest["source"]["feature_stats"])
    feature_stats = read_feature_stats(feature_stats_path)

    q_fit = _load_q(Path(manifest["q_banks"]["fit"]["manifest"]))
    q_validation = _load_q(Path(manifest["q_banks"]["validation"]["manifest"]))
    selected_fit_weights = np.ones(len(q_fit), dtype=np.float32)
    selected_validation_weights = np.ones(len(q_validation), dtype=np.float32)
    parent_fit_x, parent_fit_weights, parent_fit_diagnostics = _load_beta_target(
        root / "banks" / "beta_fit" / "bank_manifest.json"
    )
    parent_validation_x, parent_validation_weights, parent_validation_diagnostics = (
        _load_beta_target(root / "banks" / "beta_validation" / "bank_manifest.json")
    )

    final_fit = root / "fit"
    if final_fit.exists():
        raise FileExistsError(f"incomplete projection fit already exists: {final_fit}")
    attempt = root / f".fit-attempt-{os.environ.get('SLURM_JOB_ID', 'local')}"
    if attempt.exists():
        shutil.rmtree(attempt)
    attempt.mkdir(parents=True)
    selected_prior, selected = _fit_target(
        name="selected_q_aggregate",
        initial_prior=source_model.prior,
        source_model=source_model,
        train_x=q_fit,
        train_weights=selected_fit_weights,
        validation_x=q_validation,
        validation_weights=selected_validation_weights,
        output=attempt / "selected",
        config=config,
        latent_spec=latent_spec,
        feature_stats=feature_stats,
        feature_stats_path=feature_stats_path,
        devices=devices,
        passes=int(args.passes),
        samples_per_step=int(args.samples_per_step),
        peak_learning_rate=float(args.peak_learning_rate),
        final_learning_rate=float(args.final_learning_rate),
        patience=int(args.patience),
        seed=int(args.seed),
        progress_path=root / "FIT_PROGRESS.json",
    )
    _ = selected_prior
    parent_prior, parent = _fit_target(
        name="parent_inverse_beta_q",
        initial_prior=source_model.prior,
        source_model=source_model,
        train_x=parent_fit_x,
        train_weights=parent_fit_weights,
        validation_x=parent_validation_x,
        validation_weights=parent_validation_weights,
        output=attempt / "parent",
        config=config,
        latent_spec=latent_spec,
        feature_stats=feature_stats,
        feature_stats_path=feature_stats_path,
        devices=devices,
        passes=int(args.passes),
        samples_per_step=int(args.samples_per_step),
        peak_learning_rate=float(args.peak_learning_rate),
        final_learning_rate=float(args.final_learning_rate),
        patience=int(args.patience),
        seed=int(args.seed) + 1,
        progress_path=root / "FIT_PROGRESS.json",
    )
    _ = parent_prior
    os.replace(attempt, final_fit)
    for record in (selected, parent):
        record["checkpoint"] = str(
            final_fit / Path(record["checkpoint"]).relative_to(attempt)
        )
        checkpoint_path = Path(record["checkpoint"])
        record["checkpoint_sha256"] = sha256_file(checkpoint_path)
        record["checkpoint_sidecar"] = str(checkpoint_path.with_suffix(".eqx.json"))
        record["checkpoint_sidecar_sha256"] = sha256_file(
            checkpoint_path.with_suffix(".eqx.json")
        )
    receipt = {
        "status": "COMPLETE",
        "stage": "direct_distribution_projection",
        "selected": selected,
        "parent": parent,
        "parent_fit_inverse_selection": parent_fit_diagnostics,
        "parent_validation_inverse_selection": parent_validation_diagnostics,
        "truth_used": False,
        "point_estimates_used": False,
        "dsps_calls_inside_optimizer": 0,
        "checkpoint_selection": "held-out weighted density only",
        "runtime_provenance": runtime_provenance,
    }
    _write_json(complete_path, receipt)
    _write_json(
        root / "FIT_PROGRESS.json",
        {"status": "complete", "selected": selected, "parent": parent},
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
