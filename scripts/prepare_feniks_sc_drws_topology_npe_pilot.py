#!/usr/bin/env python3
"""Prepare immutable inputs for the topology-corrected frozen-parent pilot."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import jax
import numpy as np
import yaml

from euclid_dsps.amortized.data import load_photometry_arrays_from_config
from euclid_dsps.amortized.latent import latent_spec_from_config
from euclid_dsps.amortized.population_vem import canonical_json_sha256, sha256_file
from euclid_dsps.amortized.posterior import (
    ConditionalFlowEncoder,
    conditional_flow_topology,
)
from euclid_dsps.amortized.train import build_amortized_model, load_checkpoint
from euclid_dsps.config import load_config


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _file(path: str | Path) -> Path:
    value = Path(path).resolve()
    if not value.is_file() or value.stat().st_size <= 0:
        raise FileNotFoundError(value)
    return value


def _representative_observed_rows(
    config: dict[str, Any],
    validation_rows: np.ndarray,
    *,
    count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = dict(config)
    arrays = load_photometry_arrays_from_config(
        config,
        batch_size=4096,
        row_indices=np.asarray(validation_rows, dtype=np.int64),
    )
    if count <= 0 or count > len(validation_rows):
        raise ValueError("validation pilot count must lie within the validation split")
    band_names = tuple(arrays.band_names)
    r_index = band_names.index("lsst_r")
    valid_snr = np.where(arrays.mask, np.abs(arrays.flux) / arrays.flux_err, np.nan)
    median_snr = np.nanmedian(valid_snr, axis=1)
    mask_count = np.sum(arrays.mask, axis=1)
    r_flux = np.asarray(arrays.flux[:, r_index], dtype=float)
    finite = np.isfinite(r_flux) & np.isfinite(median_snr)
    if int(np.sum(finite)) < count:
        raise ValueError("too few finite observed validation rows for the pilot")

    def quantile_bin(value: np.ndarray, bins: int) -> np.ndarray:
        edges = np.unique(
            np.nanquantile(value[finite], np.linspace(0.0, 1.0, bins + 1))
        )
        if len(edges) <= 2:
            return np.zeros(len(value), dtype=np.int64)
        return np.digitize(value, edges[1:-1], right=False).astype(np.int64)

    r_bin = quantile_bin(r_flux, 4)
    snr_bin = quantile_bin(median_snr, 4)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for position in np.flatnonzero(finite):
        key = (int(r_bin[position]), int(snr_bin[position]), int(mask_count[position]))
        groups.setdefault(key, []).append(int(position))
    selected: list[int] = []
    ordered_groups = [groups[key] for key in sorted(groups)]
    round_index = 0
    while len(selected) < count:
        progressed = False
        for group in ordered_groups:
            if round_index < len(group) and len(selected) < count:
                selected.append(group[round_index])
                progressed = True
        if not progressed:
            break
        round_index += 1
    if len(selected) != count:
        raise RuntimeError("observed-only stratification did not fill the pilot cohort")
    selected = sorted(selected, key=lambda index: int(arrays.row_index[index]))
    rows = np.asarray(arrays.row_index, dtype=np.int64)[selected]
    return rows, {
        "method": "round_robin_observed_r_flux_x_snr_x_mask_count_strata",
        "truth_used": False,
        "objects": int(len(rows)),
        "r_flux_range": [
            float(np.min(r_flux[selected])),
            float(np.max(r_flux[selected])),
        ],
        "median_snr_range": [
            float(np.min(median_snr[selected])),
            float(np.max(median_snr[selected])),
        ],
        "mask_count_range": [
            int(np.min(mask_count[selected])),
            int(np.max(mask_count[selected])),
        ],
    }


def prepare(
    *,
    source_root: Path,
    topology_config_path: Path,
    elbo_config_path: Path,
    out: Path,
    repo: Path,
    validation_objects: int,
    support_objects: int,
    seed: int,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    out = out.resolve()
    repo = repo.resolve()
    winner_path = _file(source_root / "NPE_WINNER_FROZEN.json")
    manifest_path = _file(source_root / "RUN_MANIFEST.json")
    winner = _read(winner_path)
    source_manifest = _read(manifest_path)
    if (
        winner.get("status") != "FROZEN"
        or winner.get("truth_used_for_training_or_checkpoint_selection") is not False
        or winner.get("prior_bitwise_unchanged") is not True
    ):
        raise ValueError("source NPE winner is not a truth-free frozen-parent model")
    source_checkpoint = _file(winner["checkpoint"])
    source_config_path = _file(winner["config"])
    feature_stats = _file(winner["feature_stats"])
    for path, expected in (
        (source_checkpoint, winner["checkpoint_sha256"]),
        (source_config_path, winner["config_sha256"]),
        (feature_stats, winner["feature_stats_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"source artifact SHA256 mismatch: {path}")
    dataset = _file(source_manifest["dataset"]["train"])
    train_rows_path = _file(source_manifest["cohorts"]["train"]["path"])
    validation_rows_path = _file(source_manifest["cohorts"]["validation"]["path"])
    train_rows = np.load(train_rows_path, allow_pickle=False).astype(np.int64)
    validation_rows = np.load(validation_rows_path, allow_pickle=False).astype(np.int64)
    topology_config_path = _file(topology_config_path)
    elbo_config_path = _file(elbo_config_path)
    source_config = load_config(source_config_path)
    topology_config = load_config(topology_config_path)
    elbo_config = load_config(elbo_config_path)
    for label, config in (("B", topology_config), ("C", elbo_config)):
        if (config.get("truth", {}) or {}).get("parameter_columns"):
            raise ValueError(f"arm {label} exposes catalogue truth")
        config["catalog_path"] = str(dataset)
    if topology_config["amortized"]["encoder"]["flow_permutation"] != "indexed_roll":
        raise ValueError("arm B must use indexed_roll")
    if topology_config["amortized"]["objective"]["observed_elbo"]["enabled"]:
        raise ValueError("arm B must remain pure sleep")
    if not elbo_config["amortized"]["objective"]["observed_elbo"]["enabled"]:
        raise ValueError("arm C must enable the observed reverse-KL term")
    source_model = load_checkpoint(source_checkpoint, source_config)
    if not isinstance(source_model.encoder, ConditionalFlowEncoder):
        raise TypeError("source encoder is not a conditional flow")
    names = tuple(latent_spec_from_config(source_config).names)
    source_topology = conditional_flow_topology(
        source_model.encoder, coordinate_names=names
    )
    target_template = build_amortized_model(
        topology_config,
        jax.random.PRNGKey(int(seed)),
        latent_spec=latent_spec_from_config(topology_config),
    )
    target_topology = conditional_flow_topology(
        target_template.encoder,
        coordinate_names=names,
    )
    if int(target_topology["minimum_transform_count"]) < 2:
        raise ValueError("target topology leaves a coordinate under-transformed")
    topology_config["catalog_path"] = str(dataset)
    elbo_config["catalog_path"] = str(dataset)
    validation_pilot, cohort_audit = _representative_observed_rows(
        topology_config,
        validation_rows,
        count=int(validation_objects),
    )
    if support_objects <= 0 or support_objects > validation_objects:
        raise ValueError("support objects must lie within the validation pilot")
    support_positions = np.rint(
        np.linspace(0, validation_objects - 1, support_objects)
    ).astype(np.int64)
    support_pilot = validation_pilot[support_positions]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()

    request = {
        "code_commit": commit,
        "source_winner_sha256": sha256_file(winner_path),
        "topology_config_sha256": canonical_json_sha256(topology_config),
        "elbo_config_sha256": canonical_json_sha256(elbo_config),
        "validation_objects": int(validation_objects),
        "support_objects": int(support_objects),
        "seed": int(seed),
    }
    if out.exists():
        current = out / "RUN_MANIFEST.json"
        if current.is_file() and _read(current).get("request") == request:
            return _read(current)
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        manifests = staging / "manifests"
        manifests.mkdir(parents=True)
        np.save(manifests / "train_indices.npy", train_rows, allow_pickle=False)
        np.save(
            manifests / "validation_indices.npy", validation_rows, allow_pickle=False
        )
        np.save(
            manifests / "validation_pilot.npy", validation_pilot, allow_pickle=False
        )
        np.save(manifests / "support_pilot.npy", support_pilot, allow_pickle=False)
        shutil.copy2(feature_stats, manifests / "feature_stats.json")
        for name, config in (
            ("arm_b.yaml", topology_config),
            ("arm_c.yaml", elbo_config),
        ):
            (manifests / name).write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
        manifest = {
            "status": "PREPARED",
            "schema_version": 1,
            "method": "topology_corrected_sleep_plus_observed_elbo_v1",
            "code_commit": commit,
            "request": request,
            "stages": [
                "1_rebuild_and_certify_topology",
                "2_corrected_topology_pure_sleep",
                "3_corrected_topology_sleep_plus_observed_elbo",
                "4_matched_truth_free_validation",
                "5_fail_closed_population_vi_gate",
            ],
            "source": {
                "root": str(source_root),
                "winner_receipt": str(winner_path),
                "winner_receipt_sha256": sha256_file(winner_path),
                "checkpoint": str(source_checkpoint),
                "checkpoint_sha256": sha256_file(source_checkpoint),
                "config": str(source_config_path),
                "config_sha256": sha256_file(source_config_path),
                "feature_stats": str((out / "manifests/feature_stats.json").resolve()),
                "feature_stats_sha256": sha256_file(manifests / "feature_stats.json"),
                "topology": source_topology,
            },
            "target_topology": target_topology,
            "dataset": {"path": str(dataset), "sha256": sha256_file(dataset)},
            "cohorts": {
                "train": {
                    "path": str((out / "manifests/train_indices.npy").resolve()),
                    "objects": int(len(train_rows)),
                },
                "validation": {
                    "path": str((out / "manifests/validation_indices.npy").resolve()),
                    "objects": int(len(validation_rows)),
                },
                "validation_pilot": {
                    "path": str((out / "manifests/validation_pilot.npy").resolve()),
                    **cohort_audit,
                },
                "support_pilot": {
                    "path": str((out / "manifests/support_pilot.npy").resolve()),
                    "objects": int(len(support_pilot)),
                    "truth_used": False,
                },
            },
            "configs": {
                "B": str((out / "manifests/arm_b.yaml").resolve()),
                "C": str((out / "manifests/arm_c.yaml").resolve()),
            },
            "budgets": {
                "B_epochs": 8,
                "C_epochs": 4,
                "sleep_cache_candidates": 131072,
                "tracking_draws": 256,
                "support_draws": 1024,
                "population_vi_submitted": False,
            },
            "truth_boundary": {
                "catalogue_truth_used_for_training": False,
                "catalogue_truth_used_for_cohort_selection": False,
                "catalogue_truth_used_for_checkpoint_selection": False,
                "catalogue_truth_used_for_stage4": False,
            },
            "scientific_promotion": False,
        }
        (staging / "RUN_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        shutil.move(str(staging), str(out))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--topology-config",
        type=Path,
        default=Path(
            "configs/experiments/feniks_sc_drws_r29_frozen_parent_topology_sleep_npe.yaml"
        ),
    )
    parser.add_argument(
        "--elbo-config",
        type=Path,
        default=Path(
            "configs/experiments/feniks_sc_drws_r29_frozen_parent_topology_sleep_elbo_npe.yaml"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--validation-objects", type=int, default=256)
    parser.add_argument("--support-objects", type=int, default=128)
    parser.add_argument("--seed", type=int, default=260906)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
