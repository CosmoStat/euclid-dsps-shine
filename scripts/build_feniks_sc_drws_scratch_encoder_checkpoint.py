#!/usr/bin/env python3
"""Build a random-q checkpoint while preserving a frozen projected parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from euclid_dsps.amortized.elbo import AmortizedModel
from euclid_dsps.amortized.features import read_feature_stats, write_feature_stats
from euclid_dsps.amortized.latent import latent_spec_from_config
from euclid_dsps.amortized.population_vem import sha256_file
from euclid_dsps.amortized.train import (
    build_amortized_model,
    load_checkpoint,
    save_checkpoint,
)
from euclid_dsps.config import load_config


def build(
    *,
    config_path: Path,
    parent_checkpoint: Path,
    feature_stats_path: Path,
    out: Path,
    seed: int,
) -> dict[str, object]:
    config_path = config_path.resolve()
    parent_checkpoint = parent_checkpoint.resolve()
    feature_stats_path = feature_stats_path.resolve()
    out = out.resolve()
    if out.exists():
        receipt = out / "SCRATCH_INITIAL_COMPLETE.json"
        if receipt.is_file():
            return json.loads(receipt.read_text(encoding="utf-8"))
        raise FileExistsError(out)
    config = load_config(config_path)
    latent_spec = latent_spec_from_config(config)
    projected = load_checkpoint(parent_checkpoint, config)
    random_model = build_amortized_model(
        config,
        jax.random.PRNGKey(int(seed)),
        latent_spec=latent_spec,
    )
    model = AmortizedModel(
        encoder=random_model.encoder,
        prior=projected.prior,
        sed_scale=projected.sed_scale,
        band_calibration=projected.band_calibration,
    )
    out.mkdir(parents=True, exist_ok=False)
    stats = read_feature_stats(feature_stats_path)
    write_feature_stats(out / "feature_stats.json", stats)
    checkpoint = out / "scratch_encoder_projected_parent.eqx"
    save_checkpoint(
        checkpoint,
        model,
        config=config,
        latent_spec=latent_spec,
        feature_stats=stats,
        epoch=0,
        metric=0.0,
        metric_name="random_encoder_frozen_projected_parent",
    )
    receipt = {
        "status": "COMPLETE",
        "role": "random encoder plus bitwise projected parent",
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_sidecar": str(checkpoint.with_suffix(".eqx.json")),
        "checkpoint_sidecar_sha256": sha256_file(
            checkpoint.with_suffix(".eqx.json")
        ),
        "source_parent_checkpoint": str(parent_checkpoint),
        "source_parent_checkpoint_sha256": sha256_file(parent_checkpoint),
        "truth_used": False,
    }
    (out / "SCRATCH_INITIAL_COMPLETE.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    payload = build(
        config_path=args.config,
        parent_checkpoint=args.parent_checkpoint,
        feature_stats_path=args.feature_stats,
        out=args.out,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
