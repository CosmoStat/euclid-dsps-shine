#!/usr/bin/env python3
"""Prepare an immutable truth-free population-flow architecture benchmark."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from euclid_dsps.amortized.latent import latent_spec_from_config
from euclid_dsps.amortized.population_projection_benchmark import (
    BASELINE_NAME,
    CORE_PARAMETER_NAMES,
    TRAINED_CANDIDATES,
    TRUTH_FREE_TOLERANCES,
    config_for_candidate,
)
from euclid_dsps.amortized.population_vem import resolve_manifest_config, sha256_file
from euclid_dsps.config import load_config


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def prepare_benchmark(
    *,
    source_root: Path,
    out: Path,
    repo: Path,
    passes: int,
    patience: int,
    peak_learning_rate: float,
    final_learning_rate: float,
    prior_samples: int,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    out = out.resolve()
    repo = repo.resolve()
    if source_root == out:
        raise ValueError("architecture benchmark output must differ from source")
    if passes <= 0 or patience <= 0 or prior_samples < 4096:
        raise ValueError("passes/patience must be positive and prior_samples >= 4096")
    if not 0.0 < final_learning_rate <= peak_learning_rate:
        raise ValueError("learning rates must satisfy 0 < final <= peak")

    source_manifest_path = source_root / "RUN_MANIFEST.json"
    source_fit_path = source_root / "PROJECTION_FIT_COMPLETE.json"
    source_beta_path = source_root / "BETA_TARGET_COMPLETE.json"
    source_final_path = source_root / "POPULATION_PROJECTION_COMPLETE.json"
    source_manifest = _read_json(source_manifest_path)
    source_fit = _read_json(source_fit_path)
    source_beta = _read_json(source_beta_path)
    source_final = _read_json(source_final_path)
    if (
        source_fit.get("status") != "COMPLETE"
        or source_fit.get("truth_used") is not False
    ):
        raise ValueError("source projection fit is not complete and truth-free")
    if (
        source_beta.get("status") != "PASS"
        or source_beta.get("truth_used") is not False
    ):
        raise ValueError("source beta target is not a passing truth-free bank")
    if source_final.get("status") != "DIAGNOSTIC_COMPLETE":
        raise ValueError("source population projection is not complete")
    for name in ("beta_fit", "beta_validation"):
        path = source_root / "banks" / name / "bank_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)

    base_config_path = resolve_manifest_config(source_manifest, "config", repo)
    base_config = load_config(base_config_path)
    names = tuple(latent_spec_from_config(base_config).names)
    if names[:5] != CORE_PARAMETER_NAMES:
        raise ValueError(f"unexpected physical core order: {names[:5]}")
    code_commit = _git_commit(repo)
    request = {
        "source_projection_root": str(source_root),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_fit_sha256": sha256_file(source_fit_path),
        "source_beta_sha256": sha256_file(source_beta_path),
        "trained_candidates": [item["name"] for item in TRAINED_CANDIDATES],
        "baseline": BASELINE_NAME,
        "passes": int(passes),
        "patience": int(patience),
        "peak_learning_rate": float(peak_learning_rate),
        "final_learning_rate": float(final_learning_rate),
        "prior_samples": int(prior_samples),
    }
    if out.exists():
        existing = out / "RUN_MANIFEST.json"
        if existing.is_file() and _read_json(existing).get("request") == request:
            print(f"[projection-benchmark] already prepared: {out}")
            return _read_json(existing)
        raise FileExistsError(f"population projection benchmark exists: {out}")

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        candidate_records = []
        config_dir = staging / "candidate_configs"
        config_dir.mkdir(parents=True)
        for index, candidate in enumerate(TRAINED_CANDIDATES):
            config = config_for_candidate(base_config, candidate)
            staged_path = config_dir / f"{candidate['name']}.yaml"
            staged_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            candidate_records.append(
                {
                    "index": index,
                    "name": candidate["name"],
                    "label": candidate["label"],
                    "prior": candidate["prior"],
                    "config": str(
                        (out / "candidate_configs" / staged_path.name).resolve()
                    ),
                    "config_sha256": sha256_file(staged_path),
                }
            )

        manifest = copy.deepcopy(source_manifest)
        manifest.update(
            {
                "status": "PREPARED",
                "schema_version": 3,
                "method": "truth_free_population_projection_architecture_benchmark_v1",
                "code_commit": code_commit,
                "request": request,
                "architecture_benchmark": {
                    "source_projection_root": str(source_root),
                    "source_manifest": str(source_manifest_path.resolve()),
                    "source_manifest_sha256": sha256_file(source_manifest_path),
                    "source_fit_receipt": str(source_fit_path.resolve()),
                    "source_fit_receipt_sha256": sha256_file(source_fit_path),
                    "source_beta_receipt": str(source_beta_path.resolve()),
                    "source_beta_receipt_sha256": sha256_file(source_beta_path),
                    "baseline": {
                        "name": BASELINE_NAME,
                        "label": "Saturated source RealNVP",
                        "config": str(Path(base_config_path).resolve()),
                        "selected": source_fit["selected"],
                        "parent": source_fit["parent"],
                    },
                    "trained_candidates": candidate_records,
                    "candidate_selection": (
                        "lexicographic truth-free validation: worst normalized core/redshift "
                        "CDF gate, mean core CDF, then mean selected/parent weighted NLL"
                    ),
                    "tolerances": TRUTH_FREE_TOLERANCES,
                    "sfh_used_for_architecture_selection": False,
                    "redshift_median_gate_used": False,
                    "truth_used_for_fit_or_selection": False,
                    "closure_runs_after_winner_freeze": True,
                },
                "resources": {
                    "fit_array_tasks": len(TRAINED_CANDIDATES),
                    "fit_gpus_per_task": 4,
                    "validation_array_tasks": len(TRAINED_CANDIDATES) + 1,
                    "validation_gpus_per_task": 1,
                    "closure_gpus": 1,
                    "new_posterior_inference": False,
                    "q_banks_reused": True,
                    "beta_banks_reused": True,
                },
            }
        )
        (staging / "RUN_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(source_beta_path, staging / "BETA_TARGET_COMPLETE.json")
        shutil.copy2(source_fit_path, staging / "SOURCE_PROJECTION_FIT_COMPLETE.json")
        banks = staging / "banks"
        banks.mkdir()
        for name in ("beta_fit", "beta_validation"):
            (banks / name).symlink_to(
                (source_root / "banks" / name).resolve(), target_is_directory=True
            )
        os.replace(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        f"[projection-benchmark] prepared candidates={len(TRAINED_CANDIDATES)} "
        f"prior_samples={prior_samples} root={out}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--peak-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--final-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--prior-samples", type=int, default=32768)
    args = parser.parse_args()
    prepare_benchmark(
        source_root=args.source_root,
        out=args.out,
        repo=Path(__file__).resolve().parents[1],
        passes=args.passes,
        patience=args.patience,
        peak_learning_rate=args.peak_learning_rate,
        final_learning_rate=args.final_learning_rate,
        prior_samples=args.prior_samples,
    )


if __name__ == "__main__":
    main()
