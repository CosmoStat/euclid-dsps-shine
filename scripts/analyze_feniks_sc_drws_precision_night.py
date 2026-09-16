#!/usr/bin/env python3
"""CPU-only readback of existing precision-night results; never modifies gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value):
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def calibration_summary(value):
    n = value["objects"]
    m = len(value["parameters"])
    if n < 1 or m < 1:
        raise ValueError("empty calibration diagnostic")
    return {
        "status_as_recorded": value["status"],
        "objects": n,
        "directions": m,
        "max_ks": value["maximum_coordinate_pit_ks"],
        "max_ece": value["maximum_coordinate_coverage_ece"],
        "recorded_thresholds": value["thresholds"],
        "simultaneous_95pct_dkw_bound": math.sqrt(math.log(2 * m / 0.05) / (2 * n)),
        "interpretation": "Descriptive finite-N uniform-rank bound via union bound across directions, assuming independent objects. Not a replacement threshold, p-value, or permission to promote.",
    }


def heldout_summary(value):
    observed = value["held_out_observed_residuals"]["by_band"]
    reference = value["held_out_model_generated_reference"]["by_band"]
    bands = {}
    for band, gate in value["held_out_band"]["bands"].items():
        obs, ref = observed[band], reference[band]
        ratio = obs["rms"] / max(ref["rms"], 1e-12)
        if not np.isclose(ratio, gate["rms_ratio"], rtol=1e-6, atol=1e-12):
            raise ValueError(f"held-out RMS ratio mismatch: {band}")
        bands[band] = {
            "observed": obs,
            "model_generated_q_reference": ref,
            "observed_to_reference_rms": ratio,
            "reference_dominated_warning": ratio < 0.05,
        }
    return {
        "status_as_recorded": value["held_out_band"]["status"],
        "bands": bands,
        "masked_sbc": value.get("masked_model_generated_calibration", {}).get("status"),
        "masked_projection_sbc": value.get(
            "masked_joint_projection_calibration", {}
        ).get("status"),
        "interpretation": "Reference includes q approximation error on simulated inputs. A small RMS ratio is not a posterior validation. Warning is descriptive; historical gates unchanged.",
    }


def analyze(root):
    root = Path(root)
    inventory = json.loads((root / "NIGHT_ARTIFACTS.json").read_text())
    checked_files = {}

    def checked(name):
        path = root / name
        digest = sha256(path)
        if inventory.get(name) != digest:
            raise ValueError(f"changed or unrecorded artifact: {name}")
        checked_files[name] = digest
        return json.loads(path.read_text())

    final = checked("NIGHT_FINAL.json")
    if final["status"] != "PRECISION_NIGHT_DIAGNOSTIC_COMPLETE":
        raise ValueError("night incomplete")
    results = {}
    for arm in "ABC":
        summary = final["summaries"][arm]
        internal = checked(
            f"validation/{arm}/internal/INTERNAL_TRUTH_FREE_VALIDATION.json"
        )
        results[arm] = dict(
            support=summary["support"],
            technical_gate=summary["technical_gate"],
            residuals=summary.get("photometric_residuals", {}).get("all_bands"),
            calibration=calibration_summary(internal["model_generated_calibration"]),
            projection_calibration=calibration_summary(
                internal["joint_projection_calibration"]
            ),
            heldout=heldout_summary(internal),
        )
        if arm != "A":
            train = checked(f"arms/{arm}/train/training_summary.json")
            results[arm]["training"] = {
                key: train.get(key)
                for key in (
                    "best_checkpoint_metric",
                    "best_loss",
                    "best_epoch",
                    "decoder_budget",
                    "objective_component_gradient_audit",
                )
            }
            path = root / f"arms/{arm}/train/training_log.csv"
            checked_files[str(path.relative_to(root))] = sha256(path)
            logs = pd.read_csv(path)
            numeric = logs.select_dtypes(include="number")
            results[arm]["training"]["numeric_log_columns"] = list(numeric.columns)
            results[arm]["training"]["last_rows"] = logs.tail(6).to_dict("records")
    return finite(
        dict(
            status="READBACK_COMPLETE",
            arms=results,
            checked_files=checked_files,
            csv_hash_scope="training CSV hashes recorded now; not covered by original night inventory",
            scientific_promotion=False,
            population_training_started=False,
            truth_used=False,
            decoder_calls=0,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve prior readback; choose a new --out")
    result = analyze(args.root)
    args.out.mkdir(parents=True)
    (args.out / "NIGHT_READBACK.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Precision Night Readback",
        "",
        "Existing artifacts only; no promotion or decoder calls.",
        "",
    ]
    for arm, data in result["arms"].items():
        s = data["support"]
        lines += [
            f"## Arm {arm}",
            "",
            f"Support: {data['technical_gate']['status']}; median ESS/K={s['raw_ess']['fraction_median']:.5f}.",
        ]
        for band, info in data["heldout"]["bands"].items():
            lines.append(
                f"- {band}: observed RMS={info['observed']['rms']:.5g}; simulated-q reference RMS={info['model_generated_q_reference']['rms']:.5g}; reference-dominated warning={info['reference_dominated_warning']}."
            )
        lines += [f"- Marginal calibration: {data['calibration']}", ""]
    text = "\n".join(lines) + "\n"
    (args.out / "README.md").write_text(text)
    print(text)
    print("Full losses, gradients, reference tails:", args.out / "NIGHT_READBACK.json")


if __name__ == "__main__":
    main()
