#!/usr/bin/env python3
"""Fit one population-flow architecture to fixed selected and parent targets."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import jax
import numpy as np

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.amortized.features import read_feature_stats
from euclid_dsps.amortized.flows import (
    assert_flow_integrity,
    flow_coordinate_transform_counts,
)
from euclid_dsps.amortized.latent import latent_spec_from_config
from euclid_dsps.amortized.population_projection import (
    require_projection_runtime_commit,
)
from euclid_dsps.amortized.population_vem import resolve_manifest_config, sha256_file
from euclid_dsps.amortized.train import build_prior_from_config, load_checkpoint
from euclid_dsps.config import load_config

try:
    from scripts.train_feniks_sc_drws_population_projection import (
        _fit_target,
        _load_beta_target,
        _load_q,
        _write_json,
    )
except ModuleNotFoundError:
    from train_feniks_sc_drws_population_projection import (
        _fit_target,
        _load_beta_target,
        _load_q,
        _write_json,
    )

eqx, _optax = require_amortized_dependencies()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relocate_record(record: dict[str, Any], attempt: Path, final: Path) -> None:
    record["checkpoint"] = str(final / Path(record["checkpoint"]).relative_to(attempt))
    checkpoint = Path(record["checkpoint"])
    record["checkpoint_sha256"] = sha256_file(checkpoint)
    record["checkpoint_sidecar"] = str(checkpoint.with_suffix(".eqx.json"))
    record["checkpoint_sidecar_sha256"] = sha256_file(
        checkpoint.with_suffix(".eqx.json")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--samples-per-step", type=int, default=32768)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = _read_json(root / "RUN_MANIFEST.json")
    repo = Path(__file__).resolve().parents[1]
    runtime_provenance = require_projection_runtime_commit(
        root, manifest, repo, stage="fit"
    )
    benchmark = manifest["architecture_benchmark"]
    candidates = benchmark["trained_candidates"]
    if not 0 <= int(args.candidate_index) < len(candidates):
        raise ValueError("candidate index is outside the prepared benchmark")
    candidate = candidates[int(args.candidate_index)]
    candidate_root = root / "candidates" / candidate["name"]
    receipt_path = candidate_root / "FIT_COMPLETE.json"
    if receipt_path.is_file():
        print(receipt_path.read_text(encoding="utf-8"), flush=True)
        return

    config_path = Path(candidate["config"])
    if sha256_file(config_path) != candidate["config_sha256"]:
        raise ValueError("candidate config SHA256 mismatch")
    config = load_config(config_path)
    latent_spec = latent_spec_from_config(config)
    base_config = load_config(resolve_manifest_config(manifest, "config", repo))
    source_checkpoint = Path(manifest["source"]["checkpoint"])
    if sha256_file(source_checkpoint) != manifest["source"]["checkpoint_sha256"]:
        raise ValueError("frozen source checkpoint changed")
    source_model = load_checkpoint(source_checkpoint, base_config)
    feature_stats_path = Path(manifest["source"]["feature_stats"])
    if sha256_file(feature_stats_path) != manifest["source"]["feature_stats_sha256"]:
        raise ValueError("frozen feature statistics changed")
    feature_stats = read_feature_stats(feature_stats_path)

    devices = tuple(jax.local_devices())
    if len(devices) != 4:
        raise RuntimeError(
            f"architecture fitting requires four local GPUs, got {devices}"
        )
    q_fit = _load_q(Path(manifest["q_banks"]["fit"]["manifest"]))
    q_validation = _load_q(Path(manifest["q_banks"]["validation"]["manifest"]))
    parent_fit_x, parent_fit_weights, parent_fit_diagnostics = _load_beta_target(
        root / "banks" / "beta_fit" / "bank_manifest.json"
    )
    parent_validation_x, parent_validation_weights, parent_validation_diagnostics = (
        _load_beta_target(root / "banks" / "beta_validation" / "bank_manifest.json")
    )
    selected_fit_weights = np.ones(len(q_fit), dtype=np.float32)
    selected_validation_weights = np.ones(len(q_validation), dtype=np.float32)

    request = manifest["request"]
    seed = 281000 + 100 * int(args.candidate_index)
    selected_initial = build_prior_from_config(
        config,
        jax.random.PRNGKey(seed),
        latent_dim=q_fit.shape[-1],
        active_spec=latent_spec,
    )
    parent_initial = build_prior_from_config(
        config,
        jax.random.PRNGKey(seed + 1),
        latent_dim=q_fit.shape[-1],
        active_spec=latent_spec,
    )
    transform_counts = flow_coordinate_transform_counts(selected_initial)
    if min(transform_counts) <= 0:
        missing = [
            latent_spec.names[index]
            for index, count in enumerate(transform_counts)
            if count <= 0
        ]
        raise ValueError(
            f"candidate coupling topology leaves coordinates untransformed: {missing}"
        )
    candidate_root.mkdir(parents=True, exist_ok=True)
    attempt = candidate_root / f".fit-attempt-{os.environ.get('SLURM_JOB_ID', 'local')}"
    if attempt.exists():
        shutil.rmtree(attempt)
    attempt.mkdir()
    selected_prior, selected = _fit_target(
        name="selected_q_aggregate",
        initial_prior=selected_initial,
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
        passes=int(request["passes"]),
        samples_per_step=int(args.samples_per_step),
        peak_learning_rate=float(request["peak_learning_rate"]),
        final_learning_rate=float(request["final_learning_rate"]),
        patience=int(request["patience"]),
        seed=seed,
        progress_path=candidate_root / "FIT_PROGRESS.json",
        retain_initial_if_best=False,
    )
    parent_prior, parent = _fit_target(
        name="parent_inverse_beta_q",
        initial_prior=parent_initial,
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
        passes=int(request["passes"]),
        samples_per_step=int(args.samples_per_step),
        peak_learning_rate=float(request["peak_learning_rate"]),
        final_learning_rate=float(request["final_learning_rate"]),
        patience=int(request["patience"]),
        seed=seed + 1,
        progress_path=candidate_root / "FIT_PROGRESS.json",
        retain_initial_if_best=False,
    )
    selected_integrity = assert_flow_integrity(
        selected_prior, context=f"{candidate['name']} selected", sample_count=128
    )
    parent_integrity = assert_flow_integrity(
        parent_prior, context=f"{candidate['name']} parent", sample_count=128
    )
    final = candidate_root / "fit"
    if final.exists():
        raise FileExistsError(final)
    os.replace(attempt, final)
    for record in (selected, parent):
        _relocate_record(record, attempt, final)
        record["config"] = str(config_path.resolve())
        record["config_sha256"] = candidate["config_sha256"]
    receipt = {
        "status": "COMPLETE",
        "candidate": candidate["name"],
        "label": candidate["label"],
        "architecture": candidate["prior"],
        "selected": selected,
        "parent": parent,
        "selected_integrity": selected_integrity,
        "parent_integrity": parent_integrity,
        "parent_fit_inverse_selection": parent_fit_diagnostics,
        "parent_validation_inverse_selection": parent_validation_diagnostics,
        "steps_per_pass_selected": int(
            math.ceil(len(q_fit) / int(args.samples_per_step))
        ),
        "truth_used": False,
        "point_estimates_used": False,
        "new_posterior_inference": False,
        "checkpoint_selection": "held-out weighted density only",
        "coordinate_transform_counts": list(transform_counts),
        "all_coordinates_receive_active_transform": True,
        "runtime_provenance": runtime_provenance,
    }
    _write_json(receipt_path, receipt)
    _write_json(
        candidate_root / "FIT_PROGRESS.json",
        {"status": "complete", "selected": selected, "parent": parent},
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
