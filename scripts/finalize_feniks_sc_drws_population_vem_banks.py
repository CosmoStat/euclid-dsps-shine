#!/usr/bin/env python3
"""Validate initial population-VEM banks and close the truth boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from euclid_dsps.amortized.population_vem import (
    fixed_reference_selection_terms,
    iter_array_bank_shards,
    merge_array_bank_shards,
    require_git_commit,
    selection_calibration_summary,
    sha256_file,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _rows(manifest: dict[str, Any], bank: str) -> np.ndarray:
    path = Path(manifest["banks"][bank]["cohort_path"])
    if sha256_file(path) != manifest["banks"][bank]["cohort_sha256"]:
        raise ValueError(f"cohort SHA256 mismatch: {path}")
    return np.load(path, allow_pickle=False).astype(np.int64)


def _validate_contract(
    manifest: dict[str, Any], bank: str, bank_manifest: dict[str, Any]
) -> None:
    contract = bank_manifest["contract"]
    expected_kind = {
        "q_fit": "q_train",
        "q_validation": "q_validation",
        "selection_reference": "selection_reference",
        "selection_audit": "selection_audit",
    }[bank]
    expected_dataset = manifest["datasets"]["train"]["sha256"]
    expected_draws = manifest["banks"][bank].get("draws_per_object")
    checks = (
        contract["kind"] == expected_kind,
        contract["dataset_sha256"] == expected_dataset,
        contract["checkpoint_sha256"] == manifest["frozen_source"]["checkpoint_sha256"],
        contract["latent_transform_sha256"]
        == manifest["frozen_source"]["latent_transform_sha256"],
        contract["code_commit"] == manifest["code_commit"],
        contract["truth_used"] == (bank == "selection_audit"),
        contract.get("draws_per_object") == expected_draws,
    )
    if not all(checks):
        raise ValueError(f"{bank} bank contract does not match the run manifest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = _read_json(root / "RUN_MANIFEST.json")
    require_git_commit(Path(__file__).resolve().parents[1], manifest["code_commit"])
    complete_path = root / "STAGE1_PASS.json"
    if complete_path.is_file():
        complete = _read_json(complete_path)
        if complete.get("status") == "PASS":
            print(f"[population-vem] stage 1 already complete: {complete_path}")
            return

    bank_manifests = {}
    for bank in ("q_fit", "q_validation", "selection_audit"):
        bank_manifests[bank] = merge_array_bank_shards(
            root / "banks" / bank,
            expected_shards=int(manifest["banks"][bank]["shards"]),
            expected_row_indices=_rows(manifest, bank),
        )
    bank_manifests["selection_reference"] = merge_array_bank_shards(
        root / "banks" / "selection_reference",
        expected_shards=int(manifest["banks"]["selection_reference"]["shards"]),
    )
    for bank, bank_manifest in bank_manifests.items():
        _validate_contract(manifest, bank, bank_manifest)

    audit_parts = list(
        iter_array_bank_shards(
            root / "banks" / "selection_audit" / "bank_manifest.json"
        )
    )
    audit = selection_calibration_summary(
        np.concatenate([part["beta"] for part in audit_parts]),
        np.concatenate([part["selected"] for part in audit_parts]),
        np.concatenate([part["redshift"] for part in audit_parts]),
        probability_bins=10,
        redshift_bins=10,
        minimum_redshift_bin_objects=500,
        maximum_global_error=0.03,
        maximum_ece=0.05,
        maximum_redshift_bin_error=0.10,
    )
    _write_json(root / "selection_audit" / "SELECTION_CALIBRATION.json", audit)

    reference_parts = list(
        iter_array_bank_shards(
            root / "banks" / "selection_reference" / "bank_manifest.json"
        )
    )
    reference_log_prob = np.concatenate(
        [part["log_p_reference"] for part in reference_parts]
    )
    reference_log_beta = np.concatenate([part["log_beta"] for part in reference_parts])
    expected_reference_samples = int(
        manifest["banks"]["selection_reference"]["samples"]
    )
    if len(reference_log_prob) != expected_reference_samples:
        raise ValueError(
            "selection-reference sample count mismatch: "
            f"expected={expected_reference_samples}, actual={len(reference_log_prob)}"
        )
    terms = fixed_reference_selection_terms(
        reference_log_prob,
        reference_log_prob,
        reference_log_beta,
    )
    reference = {
        "status": "PASS",
        "samples": int(len(reference_log_prob)),
        "alpha": float(terms.alpha),
        "log_alpha": float(terms.log_alpha),
        "ess": float(terms.ess),
        "ess_fraction": float(terms.ess_fraction),
        "relative_mc_error": float(terms.relative_mc_error),
        "maximum_normalized_weight": float(terms.maximum_normalized_weight),
        "finite": bool(terms.finite),
        "minimum_ess_fraction": 0.10,
        "maximum_relative_mc_error": 0.03,
        "truth_used": False,
    }
    if (
        not reference["finite"]
        or reference["ess_fraction"] < reference["minimum_ess_fraction"]
        or reference["relative_mc_error"] > reference["maximum_relative_mc_error"]
    ):
        reference["status"] = "FAIL"
    _write_json(root / "selection_reference" / "REFERENCE_SUPPORT.json", reference)
    jax.clear_caches()

    passed = audit["status"] == "PASS" and reference["status"] == "PASS"
    receipt = {
        "status": "PASS" if passed else "FAIL",
        "stage": 1,
        "method": manifest["method"],
        "selection_audit": str(
            (root / "selection_audit" / "SELECTION_CALIBRATION.json").resolve()
        ),
        "selection_reference": str(
            (root / "selection_reference" / "REFERENCE_SUPPORT.json").resolve()
        ),
        "bank_manifests": {
            name: str((root / "banks" / name / "bank_manifest.json").resolve())
            for name in bank_manifests
        },
        "q_fit_objects": int(manifest["banks"]["q_fit"]["objects"]),
        "q_validation_objects": int(manifest["banks"]["q_validation"]["objects"]),
        "truth_boundary": {
            "audit_truth_consumed_only_by": "SELECTION_CALIBRATION.json",
            "prior_mstep_reads_audit_arrays": False,
            "q_banks_truth_used": False,
        },
    }
    _write_json(root / "STAGE1_PASS.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit("population-VEM stage 1 failed its beta/reference gate")


if __name__ == "__main__":
    main()
