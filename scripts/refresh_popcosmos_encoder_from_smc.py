#!/usr/bin/env python3
"""Distill two weighted SMC banks into the Pop-COSMOS encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from euclid_dsps.amortized.config import (
    amortized_config,
    require_amortized_dependencies,
)
from euclid_dsps.amortized.data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from euclid_dsps.amortized.features import read_feature_stats
from euclid_dsps.amortized.latent import latent_spec_hash
from euclid_dsps.amortized.posterior import posterior_mixture_diagnostics
from euclid_dsps.amortized.proposal_refresh import (
    expand_conditional_flow_base,
    refresh_encoder_from_weighted_particles,
)
from euclid_dsps.amortized.train import (
    _latent_spec_for_amortized_config,
    architecture_summary,
    build_amortized_model,
    load_checkpoint,
)
from euclid_dsps.config import load_config

eqx, _optax = require_amortized_dependencies()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--bank", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--object-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=260817)
    parser.add_argument("--mixture-mean-offset", type=float, default=0.05)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_config_path = args.source_config or args.config
    for path in (
        args.config,
        source_config_path,
        args.dataset,
        args.checkpoint,
        args.feature_stats,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if len(args.bank) < 1:
        raise ValueError("At least one weighted SMC bank is required")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected GPU backend, got {jax.default_backend()}")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite refresh output: {args.out}")
    args.out.mkdir(parents=True)
    (args.out / "checkpoints").mkdir()

    config = load_config(args.config)
    source_config = load_config(source_config_path)
    config["catalog_path"] = str(args.dataset)
    source_config["catalog_path"] = str(args.dataset)
    latent_spec = _latent_spec_for_amortized_config(config)
    source_latent_spec = _latent_spec_for_amortized_config(source_config)
    if latent_spec_hash(source_latent_spec) != latent_spec_hash(latent_spec):
        raise ValueError("source and target configs use different latent contracts")
    row_indices, particles, weights = _load_banks(args.bank, latent_spec.names)
    feature_stats = read_feature_stats(args.feature_stats)
    arrays = load_photometry_arrays_from_config(
        config, batch_size=10_000, row_indices=row_indices
    )
    if arrays.row_index is None:
        raise ValueError("Selected catalog does not expose row indices")
    position = {int(value): index for index, value in enumerate(arrays.row_index)}
    order = np.asarray([position[int(value)] for value in row_indices], dtype=int)
    batch = next(
        iter_photometry_batches_from_arrays(
            arrays,
            batch_size=len(row_indices),
            feature_stats=feature_stats,
            order=order,
        )
    )
    if not np.array_equal(np.asarray(batch.row_index), row_indices):
        raise RuntimeError("Feature cohort/order does not match the SMC banks")
    checkpoint_model = load_checkpoint(args.checkpoint, source_config)
    source_components = int(
        amortized_config(source_config)["encoder"].get("base_components", 1)
    )
    target_components = int(
        amortized_config(config)["encoder"].get("base_components", 1)
    )
    if source_components == target_components:
        source_model = checkpoint_model
        initialization = "checkpoint_exact"
    else:
        if source_components != 1 or target_components < 2:
            raise ValueError(
                "only controlled expansion from one to multiple base components "
                "is supported"
            )
        target_model = build_amortized_model(
            config,
            jax.random.PRNGKey(args.seed),
            latent_spec=latent_spec,
        )
        source_model = expand_conditional_flow_base(
            checkpoint_model,
            target_model,
            mean_offset=args.mixture_mean_offset,
        )
        initialization = "expanded_from_unimodal_checkpoint"
    result = refresh_encoder_from_weighted_particles(
        source_model,
        features=batch.features,
        particles=particles,
        weights=weights,
        epochs=args.epochs,
        object_batch_size=args.object_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    mixture = posterior_mixture_diagnostics(result.model, batch.features)
    mixture_diagnostics = {
        "components": int(np.asarray(jax.device_get(mixture["components"]))),
        "mean_entropy": float(np.asarray(jax.device_get(mixture["entropy"]))),
        "mean_max_weight": float(np.asarray(jax.device_get(mixture["max_weight"]))),
    }
    prior_unchanged = _trees_equal(source_model.prior, result.model.prior)
    if not prior_unchanged:
        raise RuntimeError("Population prior changed during encoder-only refresh")
    checkpoint = args.out / "checkpoints" / "best.eqx"
    eqx.tree_serialise_leaves(checkpoint, result.model)
    source_sidecar = Path(str(args.checkpoint) + ".json")
    sidecar = json.loads(source_sidecar.read_text()) if source_sidecar.is_file() else {}
    sidecar["amortized"] = amortized_config(config)
    sidecar["architecture"] = architecture_summary(config)
    sidecar["posthoc_smc_encoder_refresh"] = {
        "source_checkpoint": _receipt(args.checkpoint),
        "source_config": _receipt(source_config_path),
        "target_config": _receipt(args.config),
        "weighted_banks": [_directory_receipt(path) for path in args.bank],
        "prior_frozen_exactly": True,
        "source_base_components": source_components,
        "target_base_components": target_components,
        "initialization": initialization,
        "mixture_mean_offset": (
            float(args.mixture_mean_offset) if target_components > 1 else None
        ),
        "mixture_diagnostics": mixture_diagnostics,
        "best_epoch": result.best_epoch,
        "initial_validation_weighted_nll": result.initial_validation_nll,
        "best_validation_weighted_nll": result.best_validation_nll,
    }
    Path(str(checkpoint) + ".json").write_text(
        json.dumps(sidecar, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    shutil.copy2(args.feature_stats, args.out / "feature_stats.json")
    np.save(args.out / "row_indices.npy", row_indices)
    np.save(args.out / "train_object_positions.npy", result.train_indices)
    np.save(args.out / "validation_object_positions.npy", result.validation_indices)
    pd.DataFrame(result.history).to_csv(args.out / "refresh_history.csv", index=False)
    summary = {
        "status": "complete",
        "algorithm": "weighted joint posterior distillation into q_x(latent_x|features)",
        "n_objects": int(len(row_indices)),
        "particles_per_object": int(particles.shape[0]),
        "n_banks": int(len(args.bank)),
        "source_base_components": source_components,
        "target_base_components": target_components,
        "initialization": initialization,
        "mixture_mean_offset": (
            float(args.mixture_mean_offset) if target_components > 1 else None
        ),
        "mixture_diagnostics": mixture_diagnostics,
        "prior_frozen_exactly": prior_unchanged,
        "initial_validation_weighted_nll": result.initial_validation_nll,
        "best_validation_weighted_nll": result.best_validation_nll,
        "validation_weighted_nll_delta": float(
            result.best_validation_nll - result.initial_validation_nll
        ),
        "best_epoch": result.best_epoch,
        "refresh_gate": (
            "PASS"
            if result.best_epoch > 0
            and result.best_validation_nll < result.initial_validation_nll
            else "FAIL"
        ),
        "distribution_contract": "joint SMC particles with per-object normalized weights; no posterior medians",
        "inputs": {
            "source_config": _receipt(source_config_path),
            "target_config": _receipt(args.config),
            "dataset": _receipt(args.dataset),
            "checkpoint": _receipt(args.checkpoint),
            "feature_stats": _receipt(args.feature_stats),
            "banks": [_directory_receipt(path) for path in args.bank],
        },
    }
    (args.out / "refresh_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out / "DONE").touch()
    print(
        f"[proposal-refresh] complete gate={summary['refresh_gate']} -> {args.out}",
        flush=True,
    )


def _load_banks(paths, parameter_names):
    all_particles = []
    all_weights = []
    expected_rows = None
    for root in paths:
        files = _particle_files(root)
        if not files or not (root / "DONE").is_file():
            raise FileNotFoundError(f"Incomplete weighted SMC bank: {root}")
        frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        frame = frame.sort_values(["row_index", "sample_id"]).reset_index(drop=True)
        counts = frame.groupby("row_index", sort=True).size()
        if counts.nunique() != 1:
            raise ValueError(f"Unequal particles per object in {root}")
        rows = counts.index.to_numpy(dtype=np.int64)
        if expected_rows is None:
            expected_rows = rows
        elif not np.array_equal(expected_rows, rows):
            raise ValueError(f"SMC bank cohort mismatch in {root}")
        n_objects = len(rows)
        n_samples = int(counts.iloc[0])
        x_columns = [f"latent_x_{name}" for name in parameter_names]
        missing = sorted(set([*x_columns, "smc_weight"]) - set(frame.columns))
        if missing:
            raise ValueError(f"SMC bank missing columns {missing}: {root}")
        x = frame[x_columns].to_numpy(np.float32).reshape(n_objects, n_samples, -1)
        weight = (
            frame["smc_weight"]
            .to_numpy(np.float32, copy=True)
            .reshape(n_objects, n_samples)
        )
        weight /= np.sum(weight, axis=1, keepdims=True)
        all_particles.append(np.swapaxes(x, 0, 1))
        all_weights.append(np.swapaxes(weight, 0, 1) / len(paths))
    return (
        expected_rows,
        np.concatenate(all_particles, axis=0),
        np.concatenate(all_weights, axis=0),
    )


def _trees_equal(left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return False
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _receipt(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest}


def _directory_receipt(path):
    path = Path(path)
    files = _particle_files(path)
    return {
        "path": str(path),
        "done": (path / "DONE").is_file(),
        "particle_files": [_receipt(file) for file in files],
    }


def _particle_files(path):
    path = Path(path)
    direct = sorted((path / "weighted_particles").glob("batch_*.parquet"))
    if direct:
        return direct
    return sorted(path.glob("shard_*/weighted_particles/batch_*.parquet"))


if __name__ == "__main__":
    main()
