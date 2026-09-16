#!/usr/bin/env python3
"""Freeze a truth-free topology NPE winner and gate population VI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from euclid_dsps.amortized.population_vem import sha256_file
from euclid_dsps.amortized.population_vi import require_population_vi_gate


def _read(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _predictive_score(internal: dict) -> float:
    records = internal["held_out_band"]["bands"].values()
    values = []
    for row in records:
        values.extend(
            [
                max(float(row["median_abs_excess"]), 0.0),
                abs(float(np.log(max(row["rms_ratio"], 1.0e-12)))),
                max(float(row["fraction_abs_gt_5_excess"]), 0.0),
            ]
        )
    return float(np.mean(values))


def finalize(*, root: Path) -> dict:
    root = root.resolve()
    manifest = _read(root / "RUN_MANIFEST.json")
    validation = {
        arm: _read(root / "validation" / arm / "VALIDATION_COMPLETE.json")
        for arm in ("A", "B", "C")
    }
    arms = {arm: _read(root / "arms" / arm / "ARM_COMPLETE.json") for arm in ("B", "C")}
    candidates = {}
    for arm in ("B", "C"):
        result = validation[arm]
        support = result["support"]["technical_gate"]
        internal = result["internal"]
        checks = {
            "coordinate_topology": int(arms[arm]["topology"]["minimum_transform_count"])
            >= 2,
            "prior_unchanged": arms[arm]["prior_bitwise_unchanged"] is True,
            "support": support["status"] == "PASS",
            "held_out_band": internal["held_out_band"]["status"] == "PASS",
            "model_generated_calibration": internal["model_generated_calibration"][
                "status"
            ]
            == "PASS",
        }
        candidates[arm] = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "predictive_score": _predictive_score(internal),
            "support": result["support"]["support"],
            "held_out_band": internal["held_out_band"],
            "model_generated_calibration": internal["model_generated_calibration"],
            "decoder_budget": arms[arm]["decoder_budget"],
        }
    passing = [arm for arm in ("B", "C") if candidates[arm]["status"] == "PASS"]
    winner = (
        min(passing, key=lambda arm: candidates[arm]["predictive_score"])
        if passing
        else None
    )
    winner_receipt = None
    gate_payload = {
        "status": "BLOCKED",
        "truth_used": False,
        "reason": "no topology-corrected arm passed all truth-free prerequisites",
        "scientific_promotion": False,
    }
    if winner is not None:
        record = arms[winner]
        winner_receipt = {
            "status": "FROZEN",
            "selected_arm": winner,
            "selection_metric": "predeclared truth-free gates then held-out predictive score",
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "checkpoint_sidecar": record["checkpoint_sidecar"],
            "checkpoint_sidecar_sha256": record["checkpoint_sidecar_sha256"],
            "config": record["config"],
            "config_sha256": record["config_sha256"],
            "feature_stats": record["feature_stats"],
            "feature_stats_sha256": record["feature_stats_sha256"],
            "topology": record["topology"],
            "prior_bitwise_unchanged": True,
            "truth_used_for_training_or_checkpoint_selection": False,
            "scientific_promotion": False,
        }
        (root / "NPE_TOPOLOGY_WINNER_FROZEN.json").write_text(
            json.dumps(winner_receipt, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        gate_input = {
            "truth_used": False,
            "technical_gate": validation[winner]["support"]["technical_gate"],
            "held_out_band": validation[winner]["internal"]["held_out_band"],
            "model_generated_calibration": validation[winner]["internal"][
                "model_generated_calibration"
            ],
        }
        try:
            gate_payload = require_population_vi_gate(gate_input)
        except ValueError as error:
            gate_payload = {
                "status": "BLOCKED",
                "truth_used": False,
                "reason": str(error),
                "scientific_promotion": False,
            }
        if gate_payload["status"] == "PASS":
            gate_payload.update(
                {
                    "posterior_winner": winner,
                    "posterior_checkpoint_sha256": record["checkpoint_sha256"],
                    "population_objective": (
                        "sum_i log integral L_i p_phi - sum_i log alpha_phi"
                    ),
                    "all_objects_required": True,
                    "good_ess_filter_forbidden": True,
                    "population_training_started": False,
                }
            )
            (root / "POPULATION_VI_READY.json").write_text(
                json.dumps(gate_payload, indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
    payload = {
        "status": "DIAGNOSTIC_COMPLETE",
        "method": manifest["method"],
        "baseline_A": {
            "topology": manifest["source"]["topology"],
            "validation": validation["A"],
        },
        "candidates": candidates,
        "winner": winner,
        "winner_receipt": winner_receipt,
        "population_vi_gate": gate_payload,
        "population_training_started": False,
        "truth_used": False,
        "scientific_promotion": False,
        "interpretation": (
            "technical completion and population readiness are distinct from "
            "scientific posterior validation"
        ),
    }
    path = root / "TOPOLOGY_NPE_PILOT_COMPLETE.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    payload["receipt_sha256"] = sha256_file(path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
