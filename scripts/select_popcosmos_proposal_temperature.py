#!/usr/bin/env python3
"""Select an exact proposal temperature by ordinary-IS support diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--temperatures", required=True)
    parser.add_argument("--probe-samples", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temperatures = tuple(_parse_temperatures(args.temperatures))
    candidates = [_baseline_candidate(args.refresh_root, args.probe_samples)]
    candidates.extend(
        _scan_candidate(args.scan_root, value, args.probe_samples)
        for value in temperatures
    )
    n_objects = {int(item["n_objects"]) for item in candidates}
    n_draws = {int(item["n_joint_draws"]) for item in candidates}
    if len(n_objects) != 1 or len(n_draws) != 1:
        raise ValueError("temperature candidates do not share one fixed probe contract")
    if n_draws != {next(iter(n_objects)) * int(args.probe_samples)}:
        raise ValueError(
            "temperature candidates do not contain the requested K per object"
        )

    eligible = [item for item in candidates if item["support_status"] == "PASS"]
    selected = (
        max(
            eligible,
            key=lambda item: (
                float(item["median_raw_ess_fraction"]),
                -float(item["fraction_pareto_k_gt_0p7"]),
            ),
        )
        if eligible
        else None
    )
    baseline = candidates[0]
    payload = {
        "status": "complete",
        "selection_status": "PASS" if selected is not None else "FAIL",
        "selected_posterior_base_temperature": (
            float(selected["posterior_base_temperature"])
            if selected is not None
            else None
        ),
        "selection_rule": (
            "require the ordinary-IS support gate, then maximize median raw ESS "
            "fraction and break ties by the fraction of Pareto-k values above 0.7"
        ),
        "cohort_role": "proposal-temperature tuning on one frozen disjoint probe",
        "spectroscopy_used": False,
        "confirmation_required": selected is not None,
        "ready_for_empirical_bayes": False,
        "baseline": {
            "median_raw_ess_fraction": baseline["median_raw_ess_fraction"],
            "fraction_pareto_k_gt_0p7": baseline["fraction_pareto_k_gt_0p7"],
        },
        "candidates": candidates,
    }
    args.scan_root.mkdir(parents=True, exist_ok=True)
    (args.scan_root / "temperature_selection.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(candidates).to_csv(
        args.scan_root / "temperature_candidates.csv", index=False
    )
    (args.scan_root / "DONE").touch()
    print(json.dumps(payload, indent=2, allow_nan=False))


def _parse_temperatures(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise ValueError(
            "temperatures must be a non-empty CSV of positive finite values"
        )
    if len(set(result)) != len(result) or 1.0 in result:
        raise ValueError("scan temperatures must be unique and exclude baseline 1.0")
    return result


def _temperature_slug(value: float) -> str:
    return f"{value:.8g}".replace(".", "p")


def _baseline_candidate(refresh_root: Path, probe_samples: int) -> dict:
    root = refresh_root / f"moderate_k{probe_samples}_importance"
    return _candidate_from_importance(root, temperature=1.0, source="baseline")


def _scan_candidate(scan_root: Path, temperature: float, probe_samples: int) -> dict:
    root = (
        scan_root
        / f"temperature_{_temperature_slug(temperature)}"
        / f"importance_k{probe_samples}"
    )
    candidate = _candidate_from_importance(
        root, temperature=temperature, source="temperature_scan"
    )
    inference_summary = (
        root.parent / f"proposal_k{probe_samples}" / "inference_summary.json"
    )
    inference = json.loads(inference_summary.read_text())
    recorded = float(inference["posterior_base_temperature"])
    if recorded != float(temperature):
        raise ValueError(
            f"recorded proposal temperature mismatch: {recorded} != {temperature}"
        )
    candidate["inference_summary"] = str(inference_summary)
    return candidate


def _candidate_from_importance(root: Path, *, temperature: float, source: str) -> dict:
    summary_path = root / "importance_summary.json"
    gate_path = root / "support_gate.json"
    if (
        not (root / "DONE").is_file()
        or not summary_path.is_file()
        or not gate_path.is_file()
    ):
        raise FileNotFoundError(f"incomplete importance candidate: {root}")
    summary = json.loads(summary_path.read_text())
    gate = json.loads(gate_path.read_text())
    if summary.get("inputs", {}).get("truth") is not None:
        raise ValueError(f"temperature tuning candidate used truth: {root}")
    return {
        "source": source,
        "posterior_base_temperature": float(temperature),
        "support_status": str(gate["status"]),
        "n_objects": int(summary["n_objects"]),
        "n_joint_draws": int(summary["n_joint_draws"]),
        "median_raw_ess_fraction": float(summary["median_raw_ess_fraction"]),
        "median_psis_ess_fraction": float(summary["median_psis_ess_fraction"]),
        "fraction_pareto_k_gt_0p7": float(summary["fraction_pareto_k_gt_0p7"]),
        "fraction_pareto_k_gt_1": float(summary["fraction_pareto_k_gt_1"]),
        "importance_summary": str(summary_path),
    }


if __name__ == "__main__":
    main()
