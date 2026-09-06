#!/usr/bin/env python3
"""Rebuild q topology while preserving the frozen parent and photometry trunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.amortized.elbo import AmortizedModel
from euclid_dsps.amortized.features import read_feature_stats, write_feature_stats
from euclid_dsps.amortized.latent import latent_spec_from_config
from euclid_dsps.amortized.population_vem import sha256_file
from euclid_dsps.amortized.posterior import (
    ConditionalFlowEncoder,
    conditional_flow_topology,
    posterior_log_prob,
    sample_posterior,
    transfer_residual_photometry_trunk,
)
from euclid_dsps.amortized.train import (
    build_amortized_model,
    load_checkpoint,
    save_checkpoint,
)
from euclid_dsps.config import load_config

eqx, _optax = require_amortized_dependencies()


def _equal_array_trees(left, right) -> bool:
    left_leaves = jax.tree_util.tree_leaves(eqx.filter(left, eqx.is_array))
    right_leaves = jax.tree_util.tree_leaves(eqx.filter(right, eqx.is_array))
    return len(left_leaves) == len(right_leaves) and all(
        np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def build(
    *,
    source_config_path: Path,
    target_config_path: Path,
    source_checkpoint: Path,
    feature_stats_path: Path,
    out: Path,
    seed: int,
    minimum_transform_count: int = 2,
) -> dict[str, object]:
    """Create and certify a topology-corrected initialization checkpoint."""
    paths = [
        source_config_path,
        target_config_path,
        source_checkpoint,
        feature_stats_path,
    ]
    source_config_path, target_config_path, source_checkpoint, feature_stats_path = (
        path.resolve() for path in paths
    )
    out = out.resolve()
    if out.exists():
        receipt = out / "TOPOLOGY_CORRECTED_INITIAL_COMPLETE.json"
        if receipt.is_file():
            return json.loads(receipt.read_text(encoding="utf-8"))
        raise FileExistsError(out)

    source_config = load_config(source_config_path)
    target_config = load_config(target_config_path)
    source = load_checkpoint(source_checkpoint, source_config)
    target_spec = latent_spec_from_config(target_config)
    target_random = build_amortized_model(
        target_config,
        jax.random.PRNGKey(int(seed)),
        latent_spec=target_spec,
    )
    if not isinstance(source.encoder, ConditionalFlowEncoder) or not isinstance(
        target_random.encoder, ConditionalFlowEncoder
    ):
        raise TypeError("source and target checkpoints must use conditional flows")

    source_topology = conditional_flow_topology(
        source.encoder, coordinate_names=tuple(target_spec.names)
    )
    target_encoder = transfer_residual_photometry_trunk(
        source.encoder, target_random.encoder
    )
    target_topology = conditional_flow_topology(
        target_encoder, coordinate_names=tuple(target_spec.names)
    )
    if target_topology["minimum_transform_count"] < int(minimum_transform_count):
        raise ValueError(
            "rebuilt posterior topology does not meet the requested coverage: "
            f"{target_topology['transform_counts']}"
        )
    if source_topology["fingerprint_sha256"] == target_topology["fingerprint_sha256"]:
        raise ValueError("rebuilt posterior retained the historical topology")

    model = AmortizedModel(
        encoder=target_encoder,
        prior=source.prior,
        sed_scale=source.sed_scale,
        band_calibration=source.band_calibration,
    )
    if not _equal_array_trees(source.prior, model.prior):
        raise RuntimeError("frozen parent changed before serialization")

    out.mkdir(parents=True, exist_ok=False)
    stats = read_feature_stats(feature_stats_path)
    write_feature_stats(out / "feature_stats.json", stats)
    checkpoint = out / "topology_corrected_frozen_parent.eqx"
    save_checkpoint(
        checkpoint,
        model,
        config=target_config,
        latent_spec=target_spec,
        feature_stats=stats,
        epoch=0,
        metric=0.0,
        metric_name="topology_corrected_initialization",
    )
    restored = load_checkpoint(checkpoint, target_config)
    restored_topology = conditional_flow_topology(
        restored.encoder, coordinate_names=tuple(target_spec.names)
    )
    if restored_topology["fingerprint_sha256"] != target_topology["fingerprint_sha256"]:
        raise RuntimeError("posterior topology changed after checkpoint reload")
    if not _equal_array_trees(source.prior, restored.prior):
        raise RuntimeError("frozen parent changed after checkpoint reload")

    input_dim = int(restored.encoder.base.input_dim)
    features = jnp.zeros((2, input_dim), dtype=jnp.float32)
    draw = sample_posterior(restored, jax.random.PRNGKey(int(seed) + 1), features, 4)
    rescored = posterior_log_prob(restored, features, draw.x)
    agreement = float(np.max(np.abs(np.asarray(draw.logq) - np.asarray(rescored))))
    if not np.isfinite(agreement) or agreement > 1.0e-5:
        raise RuntimeError(f"sample/log_prob disagreement after reload: {agreement}")

    receipt = {
        "status": "COMPLETE",
        "truth_used": False,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_config": str(source_config_path),
        "target_config": str(target_config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_sidecar_sha256": sha256_file(checkpoint.with_suffix(".eqx.json")),
        "transfer_contract": (
            "input_projection+residual_blocks+representation_projection only; "
            "base heads, context head, coupling nets, masks and permutations rebuilt"
        ),
        "source_topology": source_topology,
        "target_topology": target_topology,
        "prior_bitwise_unchanged": True,
        "sample_log_prob_max_abs_error": agreement,
    }
    (out / "TOPOLOGY_CORRECTED_INITIAL_COMPLETE.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260906)
    parser.add_argument("--minimum-transform-count", type=int, default=2)
    args = parser.parse_args()
    payload = build(
        source_config_path=args.source_config,
        target_config_path=args.target_config,
        source_checkpoint=args.source_checkpoint,
        feature_stats_path=args.feature_stats,
        out=args.out,
        seed=args.seed,
        minimum_transform_count=args.minimum_transform_count,
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
