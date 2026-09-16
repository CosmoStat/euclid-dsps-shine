#!/usr/bin/env python3
"""Select a defensive proposal candidate by ordinary-IS support diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-importance", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--tail-temperatures", required=True)
    parser.add_argument("--tail-fractions", required=True)
    parser.add_argument("--probe-samples", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temperatures = _parse_csv(args.tail_temperatures, lower=1.0)
    fractions = _parse_csv(args.tail_fractions, lower=0.0, upper=1.0)
    baseline = _baseline_candidate(args.baseline_importance)
    candidates = [
        _defensive_candidate(args.scan_root, temperature, fraction)
        for temperature in temperatures
        for fraction in fractions
    ]
    expected_objects = int(baseline["n_objects"])
    expected_draws = expected_objects * int(args.probe_samples)
    if int(baseline["n_joint_draws"]) != expected_draws:
        raise ValueError("baseline importance result does not contain requested K")
    for candidate in candidates:
        if int(candidate["n_objects"]) != expected_objects:
            raise ValueError("defensive candidates do not share the baseline cohort")
        if int(candidate["n_joint_draws"]) != expected_draws:
            raise ValueError("defensive candidate does not contain requested K")
        if float(candidate["min_median_raw_ess_fraction"]) != float(
            baseline["min_median_raw_ess_fraction"]
        ) or float(candidate["max_fraction_pareto_k_gt_0p7"]) != float(
            baseline["max_fraction_pareto_k_gt_0p7"]
        ):
            raise ValueError("defensive candidate support thresholds differ")

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
    payload = {
        "status": "complete",
        "selection_status": "PASS" if selected is not None else "FAIL",
        "selected_tail_temperature": (
            float(selected["tail_temperature"]) if selected is not None else None
        ),
        "selected_requested_tail_fraction": (
            float(selected["requested_tail_fraction"]) if selected is not None else None
        ),
        "selected_realized_tail_fraction": (
            float(selected["realized_tail_fraction"]) if selected is not None else None
        ),
        "selection_rule": (
            "require the ordinary-IS support gate, then maximize median raw ESS "
            "fraction and break ties by the fraction of Pareto-k values above 0.7"
        ),
        "cohort_role": "defensive-proposal tuning on one frozen disjoint probe",
        "spectroscopy_used": False,
        "confirmation_required": selected is not None,
        "ready_for_empirical_bayes": False,
        "ordinary_importance_thresholds": {
            "min_median_raw_ess_fraction": baseline["min_median_raw_ess_fraction"],
            "max_fraction_pareto_k_gt_0p7": baseline["max_fraction_pareto_k_gt_0p7"],
        },
        "baseline": baseline,
        "candidates": candidates,
    }
    args.scan_root.mkdir(parents=True, exist_ok=True)
    (args.scan_root / "defensive_selection.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(candidates).to_csv(
        args.scan_root / "defensive_candidates.csv", index=False
    )
    (args.scan_root / "DONE").touch()
    print(json.dumps(payload, indent=2, allow_nan=False))


def _parse_csv(value: str, *, lower: float, upper: float | None = None) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    invalid = not result or any(
        not math.isfinite(item)
        or item <= lower
        or (upper is not None and item >= upper)
        for item in result
    )
    if invalid or len(set(result)) != len(result):
        interval = f"({lower}, {upper})" if upper is not None else f"({lower}, inf)"
        raise ValueError(f"values must be a unique non-empty CSV within {interval}")
    return result


def _slug(value: float) -> str:
    return f"{value:.8g}".replace(".", "p")


def _baseline_candidate(path: Path) -> dict[str, object]:
    summary = json.loads(path.read_text())
    if summary.get("inputs", {}).get("truth") is not None:
        raise ValueError("baseline importance candidate used truth")
    support = summary["support_gate"]
    return {
        "source": "unit_temperature_baseline",
        "support_status": str(support["status"]),
        "min_median_raw_ess_fraction": float(support["min_median_raw_ess_fraction"]),
        "max_fraction_pareto_k_gt_0p7": float(support["max_fraction_pareto_k_gt_0p7"]),
        "n_objects": int(summary["n_objects"]),
        "n_joint_draws": int(summary["n_joint_draws"]),
        "median_raw_ess_fraction": float(summary["median_raw_ess_fraction"]),
        "median_psis_ess_fraction": float(summary["median_psis_ess_fraction"]),
        "fraction_pareto_k_gt_0p7": float(summary["fraction_pareto_k_gt_0p7"]),
        "fraction_pareto_k_gt_1": float(summary["fraction_pareto_k_gt_1"]),
        "importance_summary": str(path),
    }


def _defensive_candidate(
    scan_root: Path, temperature: float, fraction: float
) -> dict[str, object]:
    root = (
        scan_root
        / f"tail_temperature_{_slug(temperature)}"
        / f"epsilon_{_slug(fraction)}"
    )
    summary_path = root / "importance_summary.json"
    gate_path = root / "support_gate.json"
    if (
        not (root / "DONE").is_file()
        or not summary_path.is_file()
        or not gate_path.is_file()
    ):
        raise FileNotFoundError(f"incomplete defensive candidate: {root}")
    summary = json.loads(summary_path.read_text())
    gate = json.loads(gate_path.read_text())
    allocation = summary["allocation"]
    if float(summary["tail_temperature"]) != float(temperature):
        raise ValueError(f"tail temperature mismatch in {summary_path}")
    if float(allocation["requested_tail_fraction"]) != float(fraction):
        raise ValueError(f"tail fraction mismatch in {summary_path}")
    if summary.get("spectroscopy_used") is not False:
        raise ValueError(f"defensive tuning spectroscopy contract missing: {root}")
    return {
        "source": "defensive_scan",
        "tail_temperature": float(temperature),
        "requested_tail_fraction": float(fraction),
        "realized_tail_fraction": float(allocation["realized_tail_fraction"]),
        "support_status": str(gate["status"]),
        "min_median_raw_ess_fraction": float(gate["min_median_raw_ess_fraction"]),
        "max_fraction_pareto_k_gt_0p7": float(gate["max_fraction_pareto_k_gt_0p7"]),
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
