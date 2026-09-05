#!/usr/bin/env python3
"""Freeze the best truth-free pure-sleep NPE arm."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import jax
import numpy as np

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.amortized.population_vem import require_git_commit, sha256_file
from euclid_dsps.amortized.train import load_checkpoint
from euclid_dsps.config import load_config

eqx, _optax = require_amortized_dependencies()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _equal_trees(left, right) -> bool:
    left_leaves = jax.tree_util.tree_leaves(eqx.filter(left, eqx.is_array))
    right_leaves = jax.tree_util.tree_leaves(eqx.filter(right, eqx.is_array))
    return len(left_leaves) == len(right_leaves) and all(
        np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _runtime_provenance(
    *, root: Path, repo: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    manifest_commit = str(manifest["code_commit"])
    raw_authorization = os.environ.get("NPE_GATE_RECOVERY_RECEIPT")
    if not raw_authorization:
        actual = require_git_commit(repo, manifest_commit)
        return {
            "mode": "manifest",
            "manifest_code_commit": manifest_commit,
            "finalizer_code_commit": actual,
            "authorized_arm_code_commits": [manifest_commit],
        }

    authorization_path = Path(raw_authorization).resolve()
    authorization = _read_json(authorization_path)
    actual = require_git_commit(repo, str(authorization["finalizer_code_commit"]))
    arm_receipts = {
        name: root / "arms" / name / "ARM_COMPLETE.json"
        for name in manifest["request"]["arms"]
    }
    arm_payloads = {name: _read_json(path) for name, path in arm_receipts.items()}
    arm_commits = {
        name: str(payload.get("runtime_code_commit", manifest_commit))
        for name, payload in arm_payloads.items()
    }
    checks = {
        "status": authorization.get("status") == "AUTHORIZED",
        "scope": authorization.get("scope")
        == "gate_finalizer_only_no_git_binary",
        "root": Path(authorization.get("npe_root", "")).resolve() == root,
        "manifest_sha256": authorization.get("manifest_sha256")
        == sha256_file(root / "RUN_MANIFEST.json"),
        "manifest_code_commit": authorization.get("manifest_code_commit")
        == manifest_commit,
        "warm_receipt_sha256": authorization.get("warm_receipt_sha256")
        == sha256_file(arm_receipts["warm_start"]),
        "scratch_receipt_sha256": authorization.get("scratch_receipt_sha256")
        == sha256_file(arm_receipts["scratch_encoder"]),
        "warm_runtime_code_commit": authorization.get(
            "warm_runtime_code_commit"
        )
        == arm_commits["warm_start"],
        "scratch_runtime_code_commit": authorization.get(
            "scratch_runtime_code_commit"
        )
        == arm_commits["scratch_encoder"],
        "failed_gate_job": str(authorization.get("failed_gate_job"))
        == str(os.environ.get("NPE_FAILED_GATE_JOB")),
        "training_reused": authorization.get("training_reused") is True,
        "baseline_reused": authorization.get("baseline_reused") is True,
        "truth_used": authorization.get("truth_used") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"invalid frozen-NPE gate recovery: {failed}")
    return {
        "mode": "authorized_gate_finalizer_recovery",
        "manifest_code_commit": manifest_commit,
        "finalizer_code_commit": actual,
        "authorization": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "failed_gate_job": str(authorization["failed_gate_job"]),
        "authorized_arm_code_commits": sorted(set(arm_commits.values())),
    }


def freeze(*, root: Path, repo: Path) -> dict[str, Any]:
    root = root.resolve()
    repo = repo.resolve()
    manifest = _read_json(root / "RUN_MANIFEST.json")
    provenance = _runtime_provenance(root=root, repo=repo, manifest=manifest)
    manifest_commit = manifest["code_commit"]
    final_path = root / "NPE_WINNER_FROZEN.json"
    if final_path.is_file():
        return _read_json(final_path)

    arms = []
    for name in manifest["request"]["arms"]:
        receipt = _read_json(root / "arms" / name / "ARM_COMPLETE.json")
        if (
            receipt.get("status") != "PASS"
            or receipt.get("truth_used") is not False
            or receipt.get("prior_bitwise_unchanged") is not True
        ):
            raise ValueError(f"arm is not eligible: {name}")
        arms.append(receipt)
    allowed_arm_commits = set(provenance["authorized_arm_code_commits"])
    for receipt in arms:
        arm_commit = receipt.get("runtime_code_commit", manifest_commit)
        if arm_commit not in allowed_arm_commits:
            raise ValueError(
                f"unrecognized runtime commit for {receipt['arm']}: {arm_commit}"
            )
    arms.sort(key=lambda row: (float(row["validation_sleep_nll"]), row["arm"]))
    winner = arms[0]

    config_path = Path(manifest["config"]["path"])
    config = load_config(config_path)
    parent = load_checkpoint(manifest["frozen_parent"]["checkpoint"], config)
    selected = load_checkpoint(winner["checkpoint"], config)
    if not _equal_trees(parent.prior, selected.prior):
        raise ValueError("selected NPE prior differs from the frozen projected parent")

    frozen = root / "winner"
    if frozen.exists():
        raise FileExistsError(frozen)
    frozen.mkdir(parents=True)
    checkpoint = frozen / "model.eqx"
    sidecar = frozen / "model.eqx.json"
    frozen_config = frozen / "config.yaml"
    feature_stats = frozen / "feature_stats.json"
    shutil.copy2(winner["checkpoint"], checkpoint)
    shutil.copy2(winner["checkpoint_sidecar"], sidecar)
    shutil.copy2(config_path, frozen_config)
    shutil.copy2(manifest["frozen_parent"]["feature_stats"], feature_stats)
    payload = {
        "status": "FROZEN",
        "stage": 3,
        "method": manifest["method"],
        "selected_arm": winner["arm"],
        "selection_metric": "fixed_seed_validation_sleep_nll",
        "validation_sleep_nll": float(winner["validation_sleep_nll"]),
        "arms": {
            row["arm"]: {
                "validation_sleep_nll": float(row["validation_sleep_nll"]),
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
            for row in arms
        },
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_sidecar": str(sidecar.resolve()),
        "checkpoint_sidecar_sha256": sha256_file(sidecar),
        "config": str(frozen_config.resolve()),
        "config_sha256": sha256_file(frozen_config),
        "feature_stats": str(feature_stats.resolve()),
        "feature_stats_sha256": sha256_file(feature_stats),
        "frozen_parent_checkpoint_sha256": manifest["frozen_parent"][
            "checkpoint_sha256"
        ],
        "prior_bitwise_unchanged": True,
        "runtime_provenance": {
            **provenance,
            "arm_code_commits": {
                row["arm"]: row.get("runtime_code_commit", manifest_commit)
                for row in arms
            },
        },
        "truth_used_for_training_or_checkpoint_selection": False,
        "scientific_promotion": False,
    }
    final_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze(
        root=args.root, repo=Path(__file__).resolve().parents[1]
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
