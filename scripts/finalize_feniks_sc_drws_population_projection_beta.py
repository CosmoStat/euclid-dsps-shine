#!/usr/bin/env python3
"""Merge beta caches and gate the inverse-selection parent target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from euclid_dsps.amortized.population_projection import inverse_selection_weights
from euclid_dsps.amortized.population_vem import (
    iter_array_bank_shards,
    merge_array_bank_shards,
    require_git_commit,
    sha256_file,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _diagnostics(path: Path) -> dict[str, Any]:
    values = np.concatenate(
        [
            np.asarray(part["log_beta"], dtype=np.float64).reshape(-1)
            for part in iter_array_bank_shards(path)
        ]
    )
    _weights, diagnostics = inverse_selection_weights(values)
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--minimum-ess-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-weight", type=float, default=0.01)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = _read_json(root / "RUN_MANIFEST.json")
    require_git_commit(Path(__file__).resolve().parents[1], manifest["code_commit"])
    receipt_path = root / "BETA_TARGET_COMPLETE.json"
    if receipt_path.is_file():
        existing = _read_json(receipt_path)
        if existing.get("status") not in {"PASS", "FAIL"}:
            raise ValueError("existing beta-target receipt is invalid")
        print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
        if existing["status"] != "PASS":
            raise SystemExit(1)
        return

    manifests: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for split, bank_name in (("fit", "beta_fit"), ("validation", "beta_validation")):
        source = manifest["q_banks"][split]
        expected_rows = np.load(source["cohort_path"], allow_pickle=False)
        merged = merge_array_bank_shards(
            root / "banks" / bank_name,
            expected_shards=int(source["shards"]),
            expected_row_indices=expected_rows,
        )
        expected_kind = f"q_beta_{split}"
        contract = merged["contract"]
        if (
            contract["kind"] != expected_kind
            or contract["truth_used"] is not False
            or contract["checkpoint_sha256"] != manifest["source"]["checkpoint_sha256"]
            or contract["selection_event"] != manifest["selection"]["event"]
        ):
            raise ValueError(f"invalid merged beta-{split} bank contract")
        manifest_path = root / "banks" / bank_name / "bank_manifest.json"
        manifests[split] = {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        }
        diagnostics[split] = _diagnostics(manifest_path)

    thresholds = {
        "minimum_ess_fraction": float(args.minimum_ess_fraction),
        "maximum_normalized_weight": float(args.maximum_weight),
    }
    fit = diagnostics["fit"]
    validation = diagnostics["validation"]
    passed = bool(
        fit["ess_fraction"] >= thresholds["minimum_ess_fraction"]
        and validation["ess_fraction"] >= thresholds["minimum_ess_fraction"]
        and fit["maximum_normalized_weight"] <= thresholds["maximum_normalized_weight"]
        and validation["maximum_normalized_weight"]
        <= thresholds["maximum_normalized_weight"]
    )
    receipt = {
        "status": "PASS" if passed else "FAIL",
        "stage": "inverse_selection_target",
        "method": "stable joint-draw weights proportional to 1 / beta(theta)",
        "fit": fit,
        "validation": validation,
        "thresholds": thresholds,
        "bank_manifests": manifests,
        "truth_used": False,
        "point_estimates_used": False,
    }
    _write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
