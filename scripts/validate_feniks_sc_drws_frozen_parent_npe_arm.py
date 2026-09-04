#!/usr/bin/env python3
"""Certify one pure-sleep NPE arm and its frozen population prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.amortized.population_vem import sha256_file
from euclid_dsps.amortized.train import load_checkpoint
from euclid_dsps.config import load_config

eqx, _optax = require_amortized_dependencies()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _equal_trees(left, right) -> bool:
    left_leaves = jax.tree_util.tree_leaves(eqx.filter(left, eqx.is_array))
    right_leaves = jax.tree_util.tree_leaves(eqx.filter(right, eqx.is_array))
    return len(left_leaves) == len(right_leaves) and all(
        np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def validate(*, root: Path, arm: str, initial_checkpoint: Path) -> dict:
    manifest = _read_json(root / "RUN_MANIFEST.json")
    train = root / "arms" / arm / "train"
    summary = _read_json(train / "training_summary.json")
    normalized = _read_json(train / "normalized_config.json")
    history = pd.read_csv(train / "training_log.csv")
    fit = history.loc[history["split"].eq("train")]
    validation = history.loc[history["split"].eq("validation")]
    checkpoint = train / "checkpoints/best.eqx"
    config = load_config(manifest["config"]["path"])
    initial = load_checkpoint(initial_checkpoint, config)
    trained = load_checkpoint(checkpoint, config)
    checks = {
        "summary_complete": int(summary.get("epochs", -1))
        == int(manifest["training"]["epochs"]),
        "objective_is_pure_sleep": summary.get("objective_mode")
        == "reweighted_wake_sleep",
        "prior_not_trainable": summary.get("prior_train_jointly") is False,
        "truth_absent_from_config": not bool(
            (normalized.get("truth", {}) or {}).get("parameter_columns")
        ),
        "all_train_rows_are_encoder_sleep": bool(
            not fit.empty and fit["update_phase"].eq("encoder_sleep").all()
        ),
        "no_wake_rows": bool(
            "wake_active" in fit
            and pd.to_numeric(fit["wake_active"], errors="coerce").fillna(0).eq(0).all()
        ),
        "validation_is_sleep": bool(
            not validation.empty
            and pd.to_numeric(
                validation["sleep_active"], errors="coerce"
            ).eq(1).all()
            and np.isfinite(
                pd.to_numeric(validation["sleep_nll"], errors="coerce")
            ).all()
        ),
        "prior_gradients_are_zero": bool(
            np.allclose(
                pd.to_numeric(fit["prior_raw_grad_norm"], errors="coerce"),
                0.0,
                atol=1.0e-10,
            )
        ),
        "all_updates_finite": bool(
            pd.to_numeric(fit["loss_finite"], errors="coerce").eq(1).all()
            and pd.to_numeric(fit["grads_finite"], errors="coerce").eq(1).all()
        ),
        "prior_bitwise_unchanged": _equal_trees(initial.prior, trained.prior),
        "encoder_changed": not _equal_trees(initial.encoder, trained.encoder),
        "best_validation_sleep_nll_finite": bool(
            summary.get("best_checkpoint_metric") == "validation_sleep_nll"
            and np.isfinite(float(summary.get("best_loss", np.nan)))
        ),
        "cohorts_match": bool(
            int(summary.get("train_rows", -1))
            == int(manifest["cohorts"]["train"]["objects"])
            and int(summary.get("validation_rows", -1))
            == int(manifest["cohorts"]["validation"]["objects"])
        ),
        "four_gpu_pmap": bool(
            summary.get("data_parallel", {}).get("effective") == "pmap"
            and int(summary.get("data_parallel", {}).get("local_device_count", -1))
            == 4
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "arm": arm,
        "checks": checks,
        "validation_sleep_nll": float(summary["best_loss"]),
        "best_checkpoint_epoch": int(summary["best_checkpoint_epoch"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_sidecar": str(checkpoint.with_suffix(".eqx.json")),
        "checkpoint_sidecar_sha256": sha256_file(
            checkpoint.with_suffix(".eqx.json")
        ),
        "initial_checkpoint": str(initial_checkpoint.resolve()),
        "prior_bitwise_unchanged": checks["prior_bitwise_unchanged"],
        "truth_used": False,
    }
    receipt = root / "arms" / arm / "ARM_COMPLETE.json"
    receipt.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if payload["status"] != "PASS":
        raise SystemExit(2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("warm_start", "scratch_encoder"), required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(**vars(args)), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
