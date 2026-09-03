#!/usr/bin/env python3
"""Prepare an immutable truth-free continuation of population-flow fitting."""

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

from euclid_dsps.amortized.population_vem import sha256_file


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _checkpoint_record(record: dict[str, Any], target: str) -> dict[str, Any]:
    checkpoint = Path(record["checkpoint"]).resolve()
    sidecar = Path(record["checkpoint_sidecar"]).resolve()
    if record.get("target") != target:
        raise ValueError(f"unexpected source projection target: {record.get('target')}")
    if sha256_file(checkpoint) != record.get("checkpoint_sha256"):
        raise ValueError(f"source projection checkpoint SHA256 mismatch: {checkpoint}")
    if sha256_file(sidecar) != record.get("checkpoint_sidecar_sha256"):
        raise ValueError(f"source projection sidecar SHA256 mismatch: {sidecar}")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "checkpoint_sidecar": str(sidecar),
        "checkpoint_sidecar_sha256": record["checkpoint_sidecar_sha256"],
        "source_best_validation_weighted_nll": float(
            record["best_validation_weighted_nll"]
        ),
        "source_passes_completed": int(record["passes_completed"]),
    }


def prepare_continuation(
    *,
    source_root: Path,
    out: Path,
    repo: Path,
    passes: int,
    patience: int,
    peak_learning_rate: float,
    final_learning_rate: float,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    out = out.resolve()
    repo = repo.resolve()
    if source_root == out:
        raise ValueError("continuation output must differ from its source root")
    if passes <= 0 or patience <= 0:
        raise ValueError("continuation passes and patience must be positive")
    if not 0.0 < final_learning_rate <= peak_learning_rate:
        raise ValueError("continuation learning rates must satisfy 0 < final <= peak")

    manifest_path = source_root / "RUN_MANIFEST.json"
    fit_path = source_root / "PROJECTION_FIT_COMPLETE.json"
    beta_path = source_root / "BETA_TARGET_COMPLETE.json"
    source_manifest = _read_json(manifest_path)
    source_fit = _read_json(fit_path)
    beta = _read_json(beta_path)
    if source_fit.get("status") != "COMPLETE":
        raise ValueError("source population projection fit is not complete")
    if (
        source_fit.get("truth_used") is not False
        or source_fit.get("point_estimates_used") is not False
        or source_fit.get("checkpoint_selection") != "held-out weighted density only"
    ):
        raise ValueError(
            "source projection checkpoints are not truth-free density fits"
        )
    if beta.get("status") != "PASS" or beta.get("truth_used") is not False:
        raise ValueError("source beta target is not a passing truth-free receipt")

    initial = {
        "selected": _checkpoint_record(source_fit["selected"], "selected_q_aggregate"),
        "parent": _checkpoint_record(source_fit["parent"], "parent_inverse_beta_q"),
    }
    beta_banks = {}
    for name in ("beta_fit", "beta_validation"):
        path = source_root / "banks" / name
        manifest = path / "bank_manifest.json"
        if not path.is_dir() or not manifest.is_file():
            raise FileNotFoundError(path)
        beta_banks[name] = {
            "path": str(path.resolve()),
            "manifest_sha256": sha256_file(manifest),
        }

    request = {
        "source_projection_root": str(source_root),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_fit_receipt_sha256": sha256_file(fit_path),
        "source_beta_receipt_sha256": sha256_file(beta_path),
        "passes": int(passes),
        "patience": int(patience),
        "peak_learning_rate": float(peak_learning_rate),
        "final_learning_rate": float(final_learning_rate),
    }
    if out.exists():
        existing = out / "RUN_MANIFEST.json"
        if existing.is_file() and _read_json(existing).get("request") == request:
            print(f"[population-projection-continuation] already prepared: {out}")
            return _read_json(existing)
        raise FileExistsError(f"population-projection continuation exists: {out}")

    continuation = {
        "source_projection_root": str(source_root),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_fit_receipt": str(fit_path.resolve()),
        "source_fit_receipt_sha256": sha256_file(fit_path),
        "initial": initial,
        "beta_banks": beta_banks,
        "optimizer_state_reused": False,
        "checkpoint_initialization_only": True,
        "truth_used": False,
    }
    manifest = copy.deepcopy(source_manifest)
    manifest.update(
        {
            "status": "PREPARED",
            "schema_version": 2,
            "method": (
                "direct_joint_q_and_inverse_selection_population_projection_"
                "continuation_v1"
            ),
            "code_commit": _git_commit(repo),
            "request": request,
            "continuation": continuation,
        }
    )
    manifest["resources"] = {
        **manifest.get("resources", {}),
        "beta_tasks": 0,
        "flow_fit_gpus": 4,
        "continuation_passes": int(passes),
        "continuation_patience": int(patience),
        "new_posterior_inference": False,
        "beta_banks_reused": True,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        (staging / "RUN_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(beta_path, staging / "BETA_TARGET_COMPLETE.json")
        shutil.copy2(fit_path, staging / "SOURCE_PROJECTION_FIT_COMPLETE.json")
        banks = staging / "banks"
        banks.mkdir()
        for name, record in beta_banks.items():
            (banks / name).symlink_to(record["path"], target_is_directory=True)
        os.replace(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        "[population-projection-continuation] prepared "
        f"passes={passes} patience={patience} root={out}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=48)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--peak-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--final-learning-rate", type=float, default=5.0e-7)
    args = parser.parse_args()
    prepare_continuation(
        source_root=args.source_root,
        out=args.out,
        repo=Path(__file__).resolve().parents[1],
        passes=args.passes,
        patience=args.patience,
        peak_learning_rate=args.peak_learning_rate,
        final_learning_rate=args.final_learning_rate,
    )


if __name__ == "__main__":
    main()
