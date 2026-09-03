#!/usr/bin/env python3
"""Freeze the cohorts and provenance for the five-stage population VEM run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from euclid_dsps.amortized.latent import latent_spec_from_config, latent_spec_hash
from euclid_dsps.amortized.population_vem import (
    canonical_json_sha256,
    sha256_file,
)
from euclid_dsps.config import load_config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return path.resolve()


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _repo_relative(path: Path, repo: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError as exc:
        raise ValueError(
            f"workflow config must live inside the repository: {path}"
        ) from exc


def _validate_rows(rows: np.ndarray, total: int, label: str) -> np.ndarray:
    values = np.asarray(rows, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{label} is empty")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{label} contains duplicate rows")
    if int(values.min()) < 0 or int(values.max()) >= int(total):
        raise ValueError(f"{label} contains rows outside [0, {total})")
    return values


def _save_rows(
    staging: Path,
    final_root: Path,
    label: str,
    rows: np.ndarray,
    shards: int,
) -> dict[str, Any]:
    records = []
    directory = staging / "manifests" / label
    directory.mkdir(parents=True, exist_ok=True)
    for shard, values in enumerate(np.array_split(rows, int(shards))):
        temporary_path = directory / f"shard_{shard:05d}.npy"
        np.save(temporary_path, np.asarray(values, dtype=np.int64), allow_pickle=False)
        final_path = final_root / "manifests" / label / temporary_path.name
        records.append(
            {
                "shard_id": shard,
                "path": str(final_path.resolve()),
                "objects": int(len(values)),
                "sha256": sha256_file(temporary_path),
            }
        )
    cohort_path = staging / "manifests" / f"{label}.npy"
    np.save(cohort_path, np.asarray(rows, dtype=np.int64), allow_pickle=False)
    return {
        "objects": int(len(rows)),
        "shards": int(shards),
        "cohort_path": str((final_root / "manifests" / f"{label}.npy").resolve()),
        "cohort_sha256": sha256_file(cohort_path),
        "records": records,
    }


def _validate_selection_manifest(
    source: dict[str, Any],
    *,
    train_sha256: str,
    test_sha256: str,
    train_indices_sha256: str,
    test_indices_sha256: str,
    train_objects: int,
    test_objects: int,
) -> None:
    checks = (
        source.get("status") == "complete",
        source.get("truth_used_for_training_or_checkpoint_selection") is False,
        float(source.get("selection", {}).get("max_mag_ab", -1.0)) == 29.0,
        source.get("catalogs", {}).get("train", {}).get("sha256") == train_sha256,
        source.get("catalogs", {}).get("test", {}).get("sha256") == test_sha256,
        source.get("manifests", {}).get("full_train", {}).get("sha256")
        == train_indices_sha256,
        source.get("manifests", {}).get("full_test", {}).get("sha256")
        == test_indices_sha256,
        int(source.get("manifests", {}).get("full_train", {}).get("count", -1))
        == train_objects,
        int(source.get("manifests", {}).get("full_test", {}).get("count", -1))
        == test_objects,
    )
    if not all(checks):
        raise ValueError(
            "r<29 selection manifest does not match the supplied catalogues/indices"
        )


def _resolve_source(
    *,
    freeze_receipt: Path | None,
    source_vem_root: Path | None,
    source_variant: str,
) -> dict[str, Any]:
    if (freeze_receipt is None) == (source_vem_root is None):
        raise ValueError(
            "supply exactly one of --freeze-receipt or --source-vem-root"
        )
    if source_vem_root is None:
        receipt_path = _require_file(freeze_receipt)
        freeze = _read_json(receipt_path)
        if freeze.get("status") != "FROZEN" or int(freeze.get("epoch", -1)) != 160:
            raise ValueError("population VEM requires the immutable epoch-160 receipt")
        if freeze.get("truth_used_for_training_or_checkpoint_selection") is not False:
            raise ValueError("epoch-160 source checkpoint is not certified truth-free")
        component = freeze["components"][f"{source_variant}_model"]
        checkpoint = _require_file(Path(component["path"]))
        if sha256_file(checkpoint) != component["sha256"]:
            raise ValueError("frozen source checkpoint SHA256 mismatch")
        checkpoint_sidecar = _require_file(Path(component["sidecar"]))
        feature_stats = _require_file(Path(freeze["feature_stats"]["path"]))
        if sha256_file(feature_stats) != freeze["feature_stats"]["sha256"]:
            raise ValueError("frozen feature-stat SHA256 mismatch")
        return {
            "epoch": 160,
            "iteration": 1,
            "source_iteration": 0,
            "variant": source_variant,
            "checkpoint": checkpoint,
            "checkpoint_sha256": component["sha256"],
            "checkpoint_sidecar": checkpoint_sidecar,
            "feature_stats": feature_stats,
            "feature_stats_sha256": freeze["feature_stats"]["sha256"],
            "latent_transform_sha256": freeze["latent_transform_hash"],
            "source_receipt": receipt_path,
            "source_receipt_sha256": sha256_file(receipt_path),
            "parent_vem_root": None,
        }

    parent_root = source_vem_root.resolve()
    parent_manifest_path = _require_file(parent_root / "RUN_MANIFEST.json")
    parent_complete_path = _require_file(parent_root / "POPULATION_VEM_COMPLETE.json")
    refresh_path = _require_file(
        parent_root / "q_refresh" / "Q_REFRESH_COMPLETE.json"
    )
    parent_manifest = _read_json(parent_manifest_path)
    parent_complete = _read_json(parent_complete_path)
    refresh = _read_json(refresh_path)
    if parent_complete.get("status") not in {
        "DIAGNOSTIC_COMPLETE",
        "POPULATION_TARGET_PASS",
    }:
        raise ValueError("source population-VEM run is not complete")
    if (
        parent_complete.get("truth_used_for_training_or_checkpoint_selection")
        is not False
    ):
        raise ValueError("source population-VEM run is not certified truth-free")
    if (
        refresh.get("status") != "COMPLETE"
        or refresh.get("truth_used") is not False
        or refresh.get("prior_bitwise_unchanged") is not True
    ):
        raise ValueError("source q refresh is not a certified prior-frozen checkpoint")
    checkpoint = _require_file(Path(refresh["checkpoint"]))
    checkpoint_sidecar = _require_file(Path(refresh["checkpoint_sidecar"]))
    if sha256_file(checkpoint) != refresh["checkpoint_sha256"]:
        raise ValueError("source q-refresh checkpoint SHA256 mismatch")
    if sha256_file(checkpoint_sidecar) != refresh["checkpoint_sidecar_sha256"]:
        raise ValueError("source q-refresh sidecar SHA256 mismatch")
    parent_source = parent_manifest["frozen_source"]
    feature_stats = _require_file(Path(parent_source["feature_stats"]))
    if sha256_file(feature_stats) != parent_source["feature_stats_sha256"]:
        raise ValueError("source population-VEM feature-stat SHA256 mismatch")
    parent_iteration = int(parent_manifest.get("iteration", 1))
    return {
        "epoch": int(parent_source["epoch"]),
        "iteration": parent_iteration + 1,
        "source_iteration": parent_iteration,
        "variant": "vem_refresh",
        "checkpoint": checkpoint,
        "checkpoint_sha256": refresh["checkpoint_sha256"],
        "checkpoint_sidecar": checkpoint_sidecar,
        "feature_stats": feature_stats,
        "feature_stats_sha256": parent_source["feature_stats_sha256"],
        "latent_transform_sha256": parent_source["latent_transform_sha256"],
        "source_receipt": refresh_path,
        "source_receipt_sha256": sha256_file(refresh_path),
        "parent_vem_root": parent_root,
        "parent_manifest": parent_manifest_path,
        "parent_manifest_sha256": sha256_file(parent_manifest_path),
        "parent_complete_receipt": parent_complete_path,
        "parent_complete_receipt_sha256": sha256_file(parent_complete_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--truth-config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--freeze-receipt", type=Path)
    source.add_argument("--source-vem-root", type=Path)
    parser.add_argument("--train-catalog", type=Path, required=True)
    parser.add_argument("--test-catalog", type=Path, required=True)
    parser.add_argument("--train-indices", type=Path, required=True)
    parser.add_argument("--test-indices", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260903)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--source-variant", choices=("raw", "ema"), default="raw")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    config_path = _require_file(args.config)
    truth_config_path = _require_file(args.truth_config)
    train_catalog = _require_file(args.train_catalog)
    test_catalog = _require_file(args.test_catalog)
    train_indices_path = _require_file(args.train_indices)
    test_indices_path = _require_file(args.test_indices)
    selection_manifest_path = _require_file(args.selection_manifest)
    out = args.out.resolve()
    if not 0.0 < float(args.validation_fraction) < 0.5:
        raise ValueError("validation_fraction must lie in (0, 0.5)")

    source_info = _resolve_source(
        freeze_receipt=args.freeze_receipt,
        source_vem_root=args.source_vem_root,
        source_variant=args.source_variant,
    )

    config = load_config(config_path)
    latent_hash = latent_spec_hash(latent_spec_from_config(config))
    if latent_hash != source_info["latent_transform_sha256"]:
        raise ValueError("active config does not match the frozen latent transform")
    if source_info["parent_vem_root"] is not None:
        parent_manifest = _read_json(source_info["parent_manifest"])
        if parent_manifest.get("config", {}).get(
            "resolved_sha256"
        ) != canonical_json_sha256(config):
            raise ValueError("active config does not match the source population-VEM run")
    truth_config = load_config(truth_config_path)
    truth_names = tuple(
        (truth_config.get("truth", {}) or {}).get("parameter_columns", {})
    )
    latent_names = tuple(latent_spec_from_config(config).names)
    if truth_names != latent_names:
        raise ValueError("truth-only audit mapping does not match latent ordering")

    train_total = int(pq.ParquetFile(train_catalog).metadata.num_rows)
    test_total = int(pq.ParquetFile(test_catalog).metadata.num_rows)
    selected_train = _validate_rows(
        np.load(train_indices_path, allow_pickle=False),
        train_total,
        "selected training cohort",
    )
    selected_test = _validate_rows(
        np.load(test_indices_path, allow_pickle=False),
        test_total,
        "selected test cohort",
    )
    train_sha = sha256_file(train_catalog)
    test_sha = sha256_file(test_catalog)
    train_indices_sha = sha256_file(train_indices_path)
    test_indices_sha = sha256_file(test_indices_path)
    _validate_selection_manifest(
        _read_json(selection_manifest_path),
        train_sha256=train_sha,
        test_sha256=test_sha,
        train_indices_sha256=train_indices_sha,
        test_indices_sha256=test_indices_sha,
        train_objects=len(selected_train),
        test_objects=len(selected_test),
    )
    if source_info["parent_vem_root"] is not None:
        parent_manifest = _read_json(source_info["parent_manifest"])
        for name, digest in (("train", train_sha), ("test", test_sha)):
            if parent_manifest.get("datasets", {}).get(name, {}).get("sha256") != digest:
                raise ValueError(
                    f"{name} catalogue does not match the source population-VEM run"
                )
    generator = np.random.default_rng(int(args.seed))
    order = generator.permutation(len(selected_train))
    validation_count = max(
        512, int(round(float(args.validation_fraction) * len(selected_train)))
    )
    validation_count = min(validation_count, len(selected_train) - 1)
    validation_rows = np.sort(selected_train[order[:validation_count]])
    fit_rows = np.sort(selected_train[order[validation_count:]])
    audit_rows = np.arange(train_total, dtype=np.int64)

    requested = {
        "source_checkpoint_sha256": source_info["checkpoint_sha256"],
        "source_variant": source_info["variant"],
        "source_iteration": source_info["iteration"],
        "train_indices_sha256": train_indices_sha,
        "test_indices_sha256": test_indices_sha,
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "seed": int(args.seed),
    }
    if out.exists():
        manifest_path = out / "RUN_MANIFEST.json"
        if manifest_path.is_file():
            existing = _read_json(manifest_path)
            if existing.get("request") == requested:
                print(f"[population-vem] immutable manifest already exists: {out}")
                return
        raise FileExistsError(f"population-VEM output already exists: {out}")

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        q_fit = _save_rows(staging, out, "q_fit", fit_rows, 16)
        q_validation = _save_rows(staging, out, "q_validation", validation_rows, 4)
        selection_audit = _save_rows(staging, out, "selection_audit", audit_rows, 8)
        q_evaluation = _save_rows(
            staging, out, "q_evaluation", np.sort(selected_test), 8
        )
        manifest = {
            "status": "PREPARED",
            "schema_version": 1,
            "iteration": source_info["iteration"],
            "method": "selection_corrected_population_vem_fixed_reference_v1",
            "scientific_steps": [
                "selection_audit_and_frozen_q_reference_banks",
                "selection_corrected_prior_only_mstep",
                "two_epoch_prior_frozen_avi_refresh",
                "low_draw_full_selected_test_inference",
                "population_and_individual_closure",
            ],
            "request": requested,
            "code_commit": _git_commit(repo),
            "config": {
                "path": str(config_path),
                "repo_relative_path": _repo_relative(config_path, repo),
                "sha256": sha256_file(config_path),
                "resolved_sha256": canonical_json_sha256(config),
            },
            "truth_config": {
                "path": str(truth_config_path),
                "repo_relative_path": _repo_relative(truth_config_path, repo),
                "sha256": sha256_file(truth_config_path),
                "role": "isolated selection audit and final closure only",
            },
            "frozen_source": {
                "epoch": source_info["epoch"],
                "iteration": source_info["source_iteration"],
                "variant": source_info["variant"],
                "checkpoint": str(source_info["checkpoint"]),
                "checkpoint_sha256": source_info["checkpoint_sha256"],
                "checkpoint_sidecar": str(source_info["checkpoint_sidecar"]),
                "checkpoint_sidecar_sha256": sha256_file(
                    source_info["checkpoint_sidecar"]
                ),
                "feature_stats": str(source_info["feature_stats"]),
                "feature_stats_sha256": source_info["feature_stats_sha256"],
                "latent_transform_sha256": latent_hash,
                "truth_used": False,
                "source_receipt": str(source_info["source_receipt"]),
                "source_receipt_sha256": source_info["source_receipt_sha256"],
                "freeze_receipt": (
                    str(source_info["source_receipt"])
                    if source_info["parent_vem_root"] is None
                    else None
                ),
                "freeze_receipt_sha256": (
                    source_info["source_receipt_sha256"]
                    if source_info["parent_vem_root"] is None
                    else None
                ),
                "parent_vem_root": (
                    str(source_info["parent_vem_root"])
                    if source_info["parent_vem_root"] is not None
                    else None
                ),
            },
            "datasets": {
                "train": {
                    "path": str(train_catalog),
                    "sha256": train_sha,
                    "c0_objects": train_total,
                    "selected_objects": int(len(selected_train)),
                },
                "test": {
                    "path": str(test_catalog),
                    "sha256": test_sha,
                    "c0_objects": test_total,
                    "selected_objects": int(len(selected_test)),
                    "role": "untouched until final evaluation",
                },
            },
            "selection_manifest": {
                "path": str(selection_manifest_path),
                "sha256": sha256_file(selection_manifest_path),
                "event": "A=1[m_r_observed<29.0]",
                "truth_used": False,
            },
            "banks": {
                "q_fit": {**q_fit, "draws_per_object": 32, "truth_used": False},
                "q_validation": {
                    **q_validation,
                    "draws_per_object": 64,
                    "truth_used": False,
                },
                "selection_reference": {
                    "samples": 16384,
                    "shards": 8,
                    "samples_per_shard": 2048,
                    "truth_used": False,
                },
                "selection_audit": {
                    **selection_audit,
                    "truth_used": True,
                    "role": "frozen closure-only beta audit",
                },
                "q_evaluation": {
                    **q_evaluation,
                    "draws_per_object": 32,
                    "truth_used": False,
                },
                "prior_evaluation": {
                    "samples": 16384,
                    "shards": 8,
                    "samples_per_shard": 2048,
                    "truth_used": False,
                },
            },
            "memory_contract": {
                "bank_task_gpus": 1,
                "prior_mstep_gpus": 4,
                "avi_refresh_gpus": 4,
                "q_draws_fit": 32,
                "q_draws_validation": 64,
                "q_draws_final_test": 32,
                "no_dsps_inside_prior_optimizer": True,
            },
            "truth_boundary": {
                "training": False,
                "q_banks": False,
                "prior_mstep": False,
                "avi_refresh": False,
                "selection_audit": True,
                "final_closure": True,
            },
        }
        manifest_path = staging / "RUN_MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        f"[population-vem] prepared fit={len(fit_rows)} "
        f"validation={len(validation_rows)} test={len(selected_test)} root={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
