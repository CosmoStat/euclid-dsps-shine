#!/usr/bin/env python3
"""Freeze provenance for direct selected and parent population projections."""

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

from euclid_dsps.amortized.population_vem import (
    canonical_json_sha256,
    is_array_bank_shard_complete,
    sha256_file,
)
from euclid_dsps.config import load_config


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _repo_record(path: Path, repo: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "repo_relative_path": str(resolved.relative_to(repo.resolve())),
        "sha256": sha256_file(resolved),
        "resolved_sha256": canonical_json_sha256(load_config(resolved)),
    }


def _validate_source_bank(
    root: Path,
    source_manifest: dict[str, Any],
    name: str,
    kind: str,
) -> dict[str, Any]:
    path = root / "banks" / name / "bank_manifest.json"
    bank = _read_json(path)
    contract = bank.get("contract", {})
    expected = source_manifest["banks"][name]
    if (
        bank.get("status") != "complete"
        or contract.get("kind") != kind
        or contract.get("truth_used") is not False
        or contract.get("checkpoint_sha256")
        != source_manifest["frozen_source"]["checkpoint_sha256"]
        or int(contract.get("draws_per_object", -1))
        != int(expected["draws_per_object"])
        or int(bank.get("shard_count", -1)) != int(expected["shards"])
    ):
        raise ValueError(f"invalid source q bank contract: {path}")
    objects = 0
    for record in bank["shards"]:
        shard = Path(record["path"])
        if not is_array_bank_shard_complete(shard, validate_arrays=True):
            raise ValueError(f"incomplete source q shard: {shard}")
        if sha256_file(shard / "arrays.npz") != record["arrays_sha256"]:
            raise ValueError(f"source q shard SHA256 mismatch: {shard}")
        objects += int(record["arrays"]["row_index"]["shape"][0])
    if objects != int(expected["objects"]):
        raise ValueError(f"source q bank object count mismatch: {name}")
    cohort = Path(expected["cohort_path"])
    if sha256_file(cohort) != expected["cohort_sha256"]:
        raise ValueError(f"source q cohort SHA256 mismatch: {cohort}")
    return {
        "name": name,
        "kind": kind,
        "manifest": str(path.resolve()),
        "manifest_sha256": sha256_file(path),
        "objects": objects,
        "draws_per_object": int(contract["draws_per_object"]),
        "shards": int(bank["shard_count"]),
        "cohort_path": str(cohort.resolve()),
        "cohort_sha256": expected["cohort_sha256"],
        "contract": contract,
    }


def _posterior_records(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no posterior parquet shards under {path}")
    return [
        {"path": str(file.resolve()), "sha256": sha256_file(file)} for file in files
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--truth-config", type=Path, required=True)
    parser.add_argument("--source-bank-vem-root", type=Path, required=True)
    parser.add_argument("--calibration-vem-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    source_root = args.source_bank_vem_root.resolve()
    calibration_root = args.calibration_vem_root.resolve()
    out = args.out.resolve()
    source_manifest_path = source_root / "RUN_MANIFEST.json"
    source_complete_path = source_root / "POPULATION_VEM_COMPLETE.json"
    source_manifest = _read_json(source_manifest_path)
    source_complete = _read_json(source_complete_path)
    if source_complete.get("status") not in {
        "DIAGNOSTIC_COMPLETE",
        "POPULATION_TARGET_PASS",
    }:
        raise ValueError("source-bank VEM run is not complete")
    if (
        source_complete.get("truth_used_for_training_or_checkpoint_selection")
        is not False
    ):
        raise ValueError("source-bank VEM run is not truth-free")

    q_fit = _validate_source_bank(source_root, source_manifest, "q_fit", "q_train")
    q_validation = _validate_source_bank(
        source_root, source_manifest, "q_validation", "q_validation"
    )
    selected_train = int(source_manifest["datasets"]["train"]["selected_objects"])
    if q_fit["objects"] + q_validation["objects"] != selected_train:
        raise ValueError("source q banks do not cover every selected training object")
    fit_rows = np.load(q_fit["cohort_path"], allow_pickle=False)
    validation_rows = np.load(q_validation["cohort_path"], allow_pickle=False)
    if np.intersect1d(fit_rows, validation_rows).size:
        raise ValueError("source q fit and validation cohorts overlap")

    calibration_manifest_path = calibration_root / "RUN_MANIFEST.json"
    calibration_complete_path = calibration_root / "POPULATION_VEM_COMPLETE.json"
    calibration_refresh_path = (
        calibration_root / "q_refresh" / "Q_REFRESH_COMPLETE.json"
    )
    calibration_manifest = _read_json(calibration_manifest_path)
    calibration_complete = _read_json(calibration_complete_path)
    calibration_refresh = _read_json(calibration_refresh_path)
    if calibration_complete.get("status") not in {
        "DIAGNOSTIC_COMPLETE",
        "POPULATION_TARGET_PASS",
    }:
        raise ValueError("calibration VEM run is not complete")
    if (
        calibration_complete.get("truth_used_for_training_or_checkpoint_selection")
        is not False
    ):
        raise ValueError("calibration VEM run is not truth-free before closure")
    for dataset in ("train", "test"):
        if (
            calibration_manifest["datasets"][dataset]["sha256"]
            != source_manifest["datasets"][dataset]["sha256"]
        ):
            raise ValueError(
                f"source-bank and calibration VEM use different {dataset} data"
            )
    source_checkpoint_sha = source_manifest["frozen_source"]["checkpoint_sha256"]
    if calibration_refresh.get("checkpoint_sha256") != source_checkpoint_sha:
        raise ValueError(
            "source q banks and independent calibration posterior use different q"
        )
    calibration_q_bank = _read_json(
        calibration_root / "banks" / "q_evaluation" / "bank_manifest.json"
    )
    if (
        calibration_q_bank.get("contract", {}).get("checkpoint_sha256")
        != source_checkpoint_sha
    ):
        raise ValueError("independent q-evaluation bank has a different checkpoint")

    truth_path = calibration_root / "evaluation" / "selected_test_truth.parquet"
    posterior_path = calibration_root / "evaluation" / "posterior_q"
    mira_path = calibration_root / "evaluation" / "mira" / "mira_summary.json"
    tarp_path = calibration_root / "evaluation" / "tarp" / "tarp_summary.json"
    for path in (truth_path, mira_path, tarp_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    posterior_files = _posterior_records(posterior_path)
    expected_test = int(calibration_complete["test_objects"])
    if expected_test != int(source_manifest["datasets"]["test"]["selected_objects"]):
        raise ValueError("source and calibration selected-test counts differ")

    config_record = _repo_record(args.config, repo)
    truth_config_record = _repo_record(args.truth_config, repo)
    if config_record["resolved_sha256"] != source_manifest["config"]["resolved_sha256"]:
        raise ValueError("active config differs from the source q-bank config")
    if truth_config_record["sha256"] != source_manifest["truth_config"]["sha256"]:
        raise ValueError("active truth-closure config differs from the source run")

    request = {
        "source_bank_vem_root": str(source_root),
        "source_bank_manifest_sha256": sha256_file(source_manifest_path),
        "calibration_vem_root": str(calibration_root),
        "calibration_receipt_sha256": sha256_file(calibration_complete_path),
        "source_checkpoint_sha256": source_checkpoint_sha,
        "fit_draws_for_beta": 8,
        "validation_draws_for_beta": 16,
    }
    if out.exists():
        existing_path = out / "RUN_MANIFEST.json"
        if (
            existing_path.is_file()
            and _read_json(existing_path).get("request") == request
        ):
            print(f"[population-projection] immutable manifest already exists: {out}")
            return
        raise FileExistsError(f"population-projection output already exists: {out}")

    manifest = {
        "status": "PREPARED",
        "schema_version": 1,
        "method": "direct_joint_q_and_inverse_selection_population_projection_v1",
        "code_commit": _git_commit(repo),
        "request": request,
        "config": config_record,
        "truth_config": {
            **truth_config_record,
            "role": "final frozen closure and posterior PIT only",
        },
        "source": {
            "bank_vem_root": str(source_root),
            "bank_manifest": str(source_manifest_path.resolve()),
            "bank_manifest_sha256": sha256_file(source_manifest_path),
            "bank_complete_receipt": str(source_complete_path.resolve()),
            "bank_complete_receipt_sha256": sha256_file(source_complete_path),
            "calibration_vem_root": str(calibration_root),
            "calibration_manifest": str(calibration_manifest_path.resolve()),
            "calibration_manifest_sha256": sha256_file(calibration_manifest_path),
            "calibration_complete_receipt": str(calibration_complete_path.resolve()),
            "calibration_complete_receipt_sha256": sha256_file(
                calibration_complete_path
            ),
            "checkpoint": source_manifest["frozen_source"]["checkpoint"],
            "checkpoint_sha256": source_checkpoint_sha,
            "checkpoint_sidecar": source_manifest["frozen_source"][
                "checkpoint_sidecar"
            ],
            "checkpoint_sidecar_sha256": source_manifest["frozen_source"][
                "checkpoint_sidecar_sha256"
            ],
            "feature_stats": source_manifest["frozen_source"]["feature_stats"],
            "feature_stats_sha256": source_manifest["frozen_source"][
                "feature_stats_sha256"
            ],
            "latent_transform_sha256": source_manifest["frozen_source"][
                "latent_transform_sha256"
            ],
            "q_identity": "VEM-1 prior-frozen q; stored in VEM-2 initial banks",
        },
        "q_banks": {"fit": q_fit, "validation": q_validation},
        "datasets": source_manifest["datasets"],
        "selection": source_manifest["selection_manifest"],
        "independent_posterior_calibration": {
            "objects": expected_test,
            "draws_per_object": int(calibration_complete["q_draws_per_object"]),
            "truth": str(truth_path.resolve()),
            "truth_sha256": sha256_file(truth_path),
            "posterior": str(posterior_path.resolve()),
            "posterior_files": posterior_files,
            "mira": str(mira_path.resolve()),
            "mira_sha256": sha256_file(mira_path),
            "tarp": str(tarp_path.resolve()),
            "tarp_sha256": sha256_file(tarp_path),
        },
        "projection_targets": {
            "selected": "object-equal mixture of all joint q draws",
            "parent": "same joint q draws weighted proportionally to 1 / beta(theta)",
        },
        "truth_boundary": {
            "beta_cache": False,
            "selected_flow_fit": False,
            "parent_flow_fit": False,
            "checkpoint_selection": False,
            "posterior_calibration": True,
            "final_population_closure": True,
        },
        "resources": {
            "beta_tasks": q_fit["shards"] + q_validation["shards"],
            "fit_draws_for_beta": 8,
            "validation_draws_for_beta": 16,
            "flow_fit_gpus": 4,
            "new_posterior_inference": False,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        (staging / "RUN_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        "[population-projection] prepared "
        f"fit={q_fit['objects']} validation={q_validation['objects']} "
        f"independent_test={expected_test} root={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
