#!/usr/bin/env python3
"""Certify one topology-corrected frozen-parent NPE training arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.amortized.latent import latent_spec_from_config
from euclid_dsps.amortized.population_vem import sha256_file
from euclid_dsps.amortized.posterior import conditional_flow_topology
from euclid_dsps.amortized.train import load_checkpoint
from euclid_dsps.config import load_config

eqx, _optax = require_amortized_dependencies()


def _same_arrays(left, right) -> bool:
    a = jax.tree_util.tree_leaves(eqx.filter(left, eqx.is_array))
    b = jax.tree_util.tree_leaves(eqx.filter(right, eqx.is_array))
    return len(a) == len(b) and all(
        np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
        for x, y in zip(a, b, strict=True)
    )


def finalize(*, root: Path, arm: str) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    arm = str(arm).upper()
    if arm not in {"B", "C"}:
        raise ValueError("arm must be B or C")
    config_path = root / "arms" / arm / "runtime_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = load_config(config_path)
    source_config = load_config(manifest["source"]["config"])
    source = load_checkpoint(Path(manifest["source"]["checkpoint"]), source_config)
    train = root / "arms" / arm / "train"
    checkpoint = train / "checkpoints/best.eqx"
    sidecar = checkpoint.with_suffix(".eqx.json")
    summary_path = train / "training_summary.json"
    metrics_path = train / "training_log.csv"
    for path in (checkpoint, sidecar, summary_path, metrics_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    model = load_checkpoint(checkpoint, config)
    if not _same_arrays(source.prior, model.prior):
        raise ValueError("frozen parent changed during posterior training")
    names = tuple(latent_spec_from_config(config).names)
    topology = conditional_flow_topology(model.encoder, coordinate_names=names)
    if int(topology["minimum_transform_count"]) < 2:
        raise ValueError("trained posterior violates the topology coverage contract")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    train_rows = metrics[metrics["split"] == "train"]
    if train_rows.empty or not np.all(train_rows["update_applied"].to_numpy() > 0.5):
        raise ValueError("posterior arm contains missing or rejected updates")
    observed = dict(config["amortized"]["objective"]["observed_elbo"])
    if arm == "B" and observed.get("enabled"):
        raise ValueError("arm B unexpectedly enabled observed reverse KL")
    if arm == "C" and not observed.get("enabled"):
        raise ValueError("arm C did not enable observed reverse KL")
    receipt = {
        "status": "COMPLETE",
        "arm": arm,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_sidecar": str(sidecar.resolve()),
        "checkpoint_sidecar_sha256": sha256_file(sidecar),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "feature_stats": str((train / "feature_stats.json").resolve()),
        "feature_stats_sha256": sha256_file(train / "feature_stats.json"),
        "topology": topology,
        "prior_bitwise_unchanged": True,
        "observed_elbo": observed,
        "decoder_budget": summary["decoder_budget"],
        "best_checkpoint_metric": summary["best_checkpoint_metric"],
        "best_checkpoint_value": summary["best_loss"],
        "updates": int(len(train_rows)),
        "truth_used_for_training_or_checkpoint_selection": False,
        "scientific_promotion": False,
    }
    path = root / "arms" / arm / "ARM_COMPLETE.json"
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("B", "C"), required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
