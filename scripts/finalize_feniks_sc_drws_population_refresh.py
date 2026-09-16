#!/usr/bin/env python3
"""Certify that the short AVI refresh changed q but not the learned prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.amortized.population_vem import (
    require_git_commit,
    resolve_manifest_config,
    sha256_file,
)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--refresh", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    refresh = args.refresh.resolve()
    manifest = _read_json(root / "RUN_MANIFEST.json")
    repo = Path(__file__).resolve().parents[1]
    require_git_commit(repo, manifest["code_commit"])
    prior_receipt = _read_json(root / "prior" / "PRIOR_MSTEP_COMPLETE.json")
    prior_checkpoint = Path(prior_receipt.get("checkpoint", ""))
    if (
        prior_receipt.get("status") != "COMPLETE"
        or not prior_checkpoint.is_file()
        or sha256_file(prior_checkpoint) != prior_receipt.get("checkpoint_sha256")
    ):
        raise ValueError("AVI refresh source-prior receipt is invalid")
    summary = _read_json(refresh / "training_summary.json")
    normalized_config = _read_json(refresh / "normalized_config.json")
    if int(summary.get("epochs", -1)) != 2:
        raise ValueError("AVI refresh did not complete exactly two epochs")
    if summary.get("objective_mode") != "stochastic_elbo":
        raise ValueError("AVI refresh did not use stochastic ELBO")
    if summary.get("prior_train_jointly") is not False:
        raise ValueError("AVI refresh did not freeze the population prior")
    if (
        Path(summary.get("initial_checkpoint", "")).resolve()
        != prior_checkpoint.resolve()
    ):
        raise ValueError("AVI refresh did not start from the selected prior checkpoint")
    if (
        Path(summary.get("fixed_feature_stats", "")).resolve()
        != Path(manifest["frozen_source"]["feature_stats"]).resolve()
    ):
        raise ValueError("AVI refresh changed the frozen feature statistics")
    if int(summary.get("n_samples", -1)) != 2:
        raise ValueError("AVI refresh did not use exactly two draws per gradient")
    if int(summary.get("train_rows", -1)) != int(manifest["banks"]["q_fit"]["objects"]):
        raise ValueError("AVI refresh training cohort changed")
    if int(summary.get("validation_rows", -1)) != int(
        manifest["banks"]["q_validation"]["objects"]
    ):
        raise ValueError("AVI refresh validation cohort changed")
    parallel = summary.get("data_parallel", {})
    if (
        parallel.get("effective") != "pmap"
        or int(parallel.get("local_device_count", -1)) != 4
    ):
        raise ValueError("AVI refresh did not use the required four-GPU pmap")
    if (summary.get("selection_correction", {}) or {}).get("enabled"):
        raise ValueError("selection correction was unexpectedly active in q refresh")
    if (normalized_config.get("truth", {}) or {}).get("parameter_columns"):
        raise ValueError("truth entered the AVI refresh config")
    checkpoint = refresh / "checkpoints" / "best.eqx"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = load_config(resolve_manifest_config(manifest, "config", repo))
    source_model = load_checkpoint(prior_receipt["checkpoint"], config)
    refreshed_model = load_checkpoint(checkpoint, config)
    if not _equal_trees(source_model.prior, refreshed_model.prior):
        raise ValueError("prior parameters changed during prior-frozen AVI refresh")
    if _equal_trees(source_model.encoder, refreshed_model.encoder):
        raise ValueError("encoder did not change during AVI refresh")
    receipt = {
        "status": "COMPLETE",
        "stage": 3,
        "method": "two_epoch_prior_frozen_stochastic_elbo",
        "posterior_role": "approximate AVI proposal; no importance correction",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_sidecar": str(checkpoint.with_suffix(".eqx.json")),
        "checkpoint_sidecar_sha256": sha256_file(checkpoint.with_suffix(".eqx.json")),
        "source_prior_checkpoint": prior_receipt["checkpoint"],
        "source_prior_checkpoint_sha256": prior_receipt["checkpoint_sha256"],
        "prior_bitwise_unchanged": True,
        "encoder_changed": True,
        "truth_used": False,
        "epochs": 2,
        "selected_checkpoint_epoch": int(summary["best_checkpoint_epoch"]),
        "selected_checkpoint_metric": str(summary["best_checkpoint_metric"]),
        "draws_per_object_per_gradient": int(summary["n_samples"]),
        "local_gpu_count": int(parallel["local_device_count"]),
    }
    (refresh / "Q_REFRESH_COMPLETE.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
