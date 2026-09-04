#!/usr/bin/env python3
"""Prepare immutable inputs for a frozen-parent pure-sleep NPE comparison."""

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
import yaml

from euclid_dsps.amortized.latent import latent_spec_from_config
from euclid_dsps.amortized.population_vem import canonical_json_sha256, sha256_file
from euclid_dsps.config import load_config


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: str | Path) -> Path:
    value = Path(path).resolve()
    if not value.is_file() or value.stat().st_size <= 0:
        raise FileNotFoundError(value)
    return value


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def prepare(
    *,
    benchmark_root: Path,
    config_path: Path,
    out: Path,
    repo: Path,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    benchmark_root = benchmark_root.resolve()
    config_path = _require_file(config_path)
    out = out.resolve()
    repo = repo.resolve()
    if epochs < 4:
        raise ValueError("pure-sleep NPE requires at least four epochs")

    benchmark_manifest_path = _require_file(benchmark_root / "RUN_MANIFEST.json")
    winner_path = _require_file(benchmark_root / "TRUTH_FREE_ARCHITECTURE_WINNER.json")
    fit_path = _require_file(benchmark_root / "PROJECTION_FIT_COMPLETE.json")
    closure_path = _require_file(benchmark_root / "POPULATION_PROJECTION_COMPLETE.json")
    benchmark = _read_json(benchmark_manifest_path)
    winner = _read_json(winner_path)
    fit = _read_json(fit_path)
    closure = _read_json(closure_path)
    if (
        winner.get("status") != "WINNER_SELECTED"
        or winner.get("winner") != "realnvp_wide"
        or winner.get("truth_used") is not False
        or winner.get("winner_passes_all_truth_free_distribution_gates") is not True
        or winner.get("winner_passes_nll_non_regression_gate") is not True
    ):
        raise ValueError("frozen-parent NPE requires the passing realnvp_wide winner")
    if fit.get("status") != "COMPLETE" or fit.get("truth_used") is not False:
        raise ValueError("projected-parent fit is not complete and truth-free")
    if closure.get("status") != "DIAGNOSTIC_COMPLETE":
        raise ValueError("projected-parent closure is incomplete")

    parent = fit["parent"]
    parent_checkpoint = _require_file(parent["checkpoint"])
    parent_sidecar = _require_file(parent["checkpoint_sidecar"])
    parent_config_path = _require_file(parent["config"])
    feature_stats = _require_file(benchmark["source"]["feature_stats"])
    train_catalog = _require_file(benchmark["datasets"]["train"]["path"])
    fit_rows = _require_file(benchmark["q_banks"]["fit"]["cohort_path"])
    validation_rows = _require_file(
        benchmark["q_banks"]["validation"]["cohort_path"]
    )
    for path, expected, label in (
        (parent_checkpoint, parent["checkpoint_sha256"], "parent checkpoint"),
        (parent_sidecar, parent["checkpoint_sidecar_sha256"], "parent sidecar"),
        (parent_config_path, parent["config_sha256"], "parent config"),
        (
            feature_stats,
            benchmark["source"]["feature_stats_sha256"],
            "feature statistics",
        ),
        (train_catalog, benchmark["datasets"]["train"]["sha256"], "train catalogue"),
        (fit_rows, benchmark["q_banks"]["fit"]["cohort_sha256"], "fit rows"),
        (
            validation_rows,
            benchmark["q_banks"]["validation"]["cohort_sha256"],
            "validation rows",
        ),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} SHA256 mismatch")

    config = load_config(config_path)
    amortized = config["amortized"]
    if (config.get("truth", {}) or {}).get("parameter_columns"):
        raise ValueError("pure-sleep NPE config exposes catalogue truth")
    if amortized["objective"]["mode"] != "reweighted_wake_sleep":
        raise ValueError("pure-sleep NPE requires reweighted_wake_sleep mode")
    if amortized["prior"].get("train_jointly") is not False:
        raise ValueError("pure-sleep NPE prior must be frozen")
    sleep = amortized["objective"]["sleep"]
    if int(sleep["selection"]["candidate_factor"]) != 2:
        raise ValueError("time-bounded NPE requires candidate_factor=2")
    parent_config = load_config(parent_config_path)
    if tuple(latent_spec_from_config(config).names) != tuple(
        latent_spec_from_config(parent_config).names
    ):
        raise ValueError("NPE and projected-parent latent parameter order differ")

    request = {
        "benchmark_root": str(benchmark_root),
        "benchmark_manifest_sha256": sha256_file(benchmark_manifest_path),
        "winner_receipt_sha256": sha256_file(winner_path),
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "config_resolved_sha256": canonical_json_sha256(config),
        "epochs": int(epochs),
        "seed": int(seed),
        "arms": ["warm_start", "scratch_encoder"],
    }
    if out.exists():
        existing = out / "RUN_MANIFEST.json"
        if existing.is_file() and _read_json(existing).get("request") == request:
            return _read_json(existing)
        raise FileExistsError(f"frozen-parent NPE output exists: {out}")

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        manifests = staging / "manifests"
        manifests.mkdir(parents=True)
        for name, source in (
            ("train_indices.npy", fit_rows),
            ("validation_indices.npy", validation_rows),
            ("feature_stats.json", feature_stats),
        ):
            shutil.copy2(source, manifests / name)
        resolved_config = manifests / "training_config.yaml"
        resolved_config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        train_count = int(len(np.load(fit_rows, allow_pickle=False)))
        validation_count = int(len(np.load(validation_rows, allow_pickle=False)))
        manifest = {
            "status": "PREPARED",
            "schema_version": 1,
            "method": "frozen_projected_parent_pure_sleep_npe_v1",
            "scientific_steps": [
                "1_provenance_and_support_preflight",
                "2_current_q_independent_test_baseline",
                "3_parallel_warm_and_scratch_pure_sleep_npe",
                "4_matched_full_test_and_k1024_winner_evaluation",
                "5_frozen_posterior_and_population_closure",
            ],
            "request": request,
            "code_commit": _git_commit(repo),
            "benchmark": {
                "root": str(benchmark_root),
                "manifest": str(benchmark_manifest_path),
                "manifest_sha256": sha256_file(benchmark_manifest_path),
                "winner_receipt": str(winner_path),
                "winner_receipt_sha256": sha256_file(winner_path),
                "closure_receipt": str(closure_path),
                "closure_receipt_sha256": sha256_file(closure_path),
            },
            "config": {
                "source_path": str(config_path),
                "source_sha256": sha256_file(config_path),
                "path": str((out / "manifests/training_config.yaml").resolve()),
                "sha256": sha256_file(resolved_config),
                "resolved_sha256": canonical_json_sha256(config),
            },
            "frozen_parent": {
                "checkpoint": str(parent_checkpoint),
                "checkpoint_sha256": parent["checkpoint_sha256"],
                "checkpoint_sidecar": str(parent_sidecar),
                "checkpoint_sidecar_sha256": parent[
                    "checkpoint_sidecar_sha256"
                ],
                "feature_stats": str((out / "manifests/feature_stats.json").resolve()),
                "feature_stats_sha256": sha256_file(
                    manifests / "feature_stats.json"
                ),
            },
            "dataset": {
                "train": str(train_catalog),
                "train_sha256": benchmark["datasets"]["train"]["sha256"],
            },
            "cohorts": {
                "train": {
                    "path": str((out / "manifests/train_indices.npy").resolve()),
                    "sha256": sha256_file(manifests / "train_indices.npy"),
                    "objects": train_count,
                },
                "validation": {
                    "path": str(
                        (out / "manifests/validation_indices.npy").resolve()
                    ),
                    "sha256": sha256_file(manifests / "validation_indices.npy"),
                    "objects": validation_count,
                },
            },
            "training": {
                "epochs": int(epochs),
                "batch_size": 1024,
                "jax_batch_size": 256,
                "gpus_per_arm": 4,
                "candidate_factor": 2,
                "checkpoint_selection": "fixed-seed validation_sleep_nll",
            },
            "truth_boundary": {
                "training": False,
                "checkpoint_selection": False,
                "stage4_support": False,
                "stage5_frozen_closure": True,
            },
        }
        (staging / "RUN_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "STAGE1_PASS.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "stage": 1,
                    "parent_checkpoint_sha256": parent["checkpoint_sha256"],
                    "train_objects": train_count,
                    "validation_objects": validation_count,
                    "truth_used": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=260904)
    args = parser.parse_args()
    payload = prepare(
        benchmark_root=args.benchmark_root,
        config_path=args.config,
        out=args.out,
        repo=Path(__file__).resolve().parents[1],
        epochs=args.epochs,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
