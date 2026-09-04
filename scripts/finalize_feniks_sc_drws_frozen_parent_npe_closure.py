#!/usr/bin/env python3
"""Compare frozen-parent NPE against the current-q matched baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from euclid_dsps.amortized.population_vem import sha256_file


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _diagnostic(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(root / "RUN_MANIFEST.json")
    receipt = _read_json(root / "INDIVIDUAL_POSTERIOR_DIAGNOSTIC_COMPLETE.json")
    if receipt.get("status") != "DIAGNOSTIC_COMPLETE":
        raise ValueError(f"posterior diagnostic is incomplete: {root}")
    return manifest, receipt


def _calibration(receipt: dict[str, Any], model: str = "q") -> dict[str, float]:
    row = receipt["redshift_calibration"][model]
    return {
        "pit_ks_uniform": float(row["pit_ks_uniform"]),
        "coverage_ece": float(row["coverage_ece"]),
        "coverage_68": float(row["coverage_68"]),
        "coverage_95": float(row["coverage_95"]),
    }


def _support(receipt: dict[str, Any]) -> dict[str, Any]:
    row = receipt["projected_parent_support"]
    return {
        "status": row["status"],
        "median_raw_ess": float(row["median_raw_ess"]),
        "median_raw_ess_fraction": float(row["median_raw_ess_fraction"]),
        "fraction_pareto_k_gt_0p7": float(row["fraction_pareto_k_gt_0p7"]),
        "p90_max_raw_weight": float(row["p90_max_raw_weight"]),
    }


def _delta(after: float, before: float) -> float:
    return float(after - before)


def _write_plot(
    path: Path,
    *,
    baseline_cal: dict[str, float],
    npe_cal: dict[str, float],
    baseline_support: dict[str, Any],
    npe_support: dict[str, Any],
) -> None:
    labels = ("current q", "sleep-NPE q")
    colors = ("#777777", "#0072B2")
    figure, axes = plt.subplots(2, 2, figsize=(9.8, 7.4), constrained_layout=True)
    plots = (
        (
            axes[0, 0],
            (baseline_cal["pit_ks_uniform"], npe_cal["pit_ks_uniform"]),
            "Redshift PIT KS (lower is better)",
            0.05,
        ),
        (
            axes[0, 1],
            (baseline_cal["coverage_ece"], npe_cal["coverage_ece"]),
            "Redshift coverage ECE (lower is better)",
            0.05,
        ),
        (
            axes[1, 0],
            (
                baseline_support["median_raw_ess"],
                npe_support["median_raw_ess"],
            ),
            "K=1024 median importance ESS (higher is better)",
            51.2,
        ),
        (
            axes[1, 1],
            (
                baseline_support["fraction_pareto_k_gt_0p7"],
                npe_support["fraction_pareto_k_gt_0p7"],
            ),
            "Fraction Pareto k > 0.7 (lower is better)",
            0.20,
        ),
    )
    for axis, values, title, threshold in plots:
        bars = axis.bar(labels, values, color=colors, width=0.62)
        axis.axhline(threshold, color="#D55E00", linestyle="--", linewidth=1.2)
        axis.set_title(title, fontsize=10)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        axis.set_ylim(0.0, max(max(values) * 1.22, threshold * 1.25, 1.0e-3))
    figure.suptitle("Matched independent-test posterior validation")
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def finalize(
    *,
    root: Path,
    baseline_root: Path,
    full_root: Path,
    support_root: Path,
) -> dict[str, Any]:
    root = root.resolve()
    final_path = root / "FROZEN_NPE_EXPERIMENT_COMPLETE.json"
    if final_path.is_file():
        return _read_json(final_path)
    baseline_root = baseline_root.resolve()
    baseline_full_manifest, baseline_full = _diagnostic(
        baseline_root / "full_test_k256"
    )
    baseline_support_manifest, baseline_support_receipt = _diagnostic(
        baseline_root / "support_k1024"
    )
    npe_full_manifest, npe_full = _diagnostic(full_root.resolve())
    npe_support_manifest, npe_support_receipt = _diagnostic(support_root.resolve())
    winner_path = root / "NPE_WINNER_FROZEN.json"
    winner = _read_json(winner_path)
    if winner.get("status") != "FROZEN":
        raise ValueError("NPE winner is not frozen")

    for label, left, right in (
        ("full-test", baseline_full_manifest, npe_full_manifest),
        ("support", baseline_support_manifest, npe_support_manifest),
    ):
        if left["cohort"]["sha256"] != right["cohort"]["sha256"]:
            raise ValueError(f"{label} baseline and NPE cohorts differ")
        if left["inference"]["posterior_draws_per_object"] != right["inference"][
            "posterior_draws_per_object"
        ]:
            raise ValueError(f"{label} baseline and NPE draw counts differ")
    if npe_full_manifest["model"].get("freeze_receipt", {}).get("sha256") != sha256_file(
        winner_path
    ):
        raise ValueError("stage-4 inference did not use the frozen NPE winner")

    baseline_cal = _calibration(baseline_full)
    npe_cal = _calibration(npe_full)
    baseline_support = _support(baseline_support_receipt)
    npe_support = _support(npe_support_receipt)
    calibration_pass = (
        npe_cal["pit_ks_uniform"] <= 0.05
        and npe_cal["coverage_ece"] <= 0.05
    )
    support_pass = npe_support["status"] == "PASS"
    improvement = {
        "pit_ks_uniform": _delta(
            npe_cal["pit_ks_uniform"], baseline_cal["pit_ks_uniform"]
        ),
        "coverage_ece": _delta(
            npe_cal["coverage_ece"], baseline_cal["coverage_ece"]
        ),
        "coverage_68_absolute_error": _delta(
            abs(npe_cal["coverage_68"] - 0.68),
            abs(baseline_cal["coverage_68"] - 0.68),
        ),
        "coverage_95_absolute_error": _delta(
            abs(npe_cal["coverage_95"] - 0.95),
            abs(baseline_cal["coverage_95"] - 0.95),
        ),
        "median_raw_ess": _delta(
            npe_support["median_raw_ess"], baseline_support["median_raw_ess"]
        ),
        "fraction_pareto_k_gt_0p7": _delta(
            npe_support["fraction_pareto_k_gt_0p7"],
            baseline_support["fraction_pareto_k_gt_0p7"],
        ),
    }
    plots = root / "evaluation"
    plots.mkdir(parents=True, exist_ok=True)
    comparison_plot = plots / "matched_posterior_comparison.png"
    _write_plot(
        comparison_plot,
        baseline_cal=baseline_cal,
        npe_cal=npe_cal,
        baseline_support=baseline_support,
        npe_support=npe_support,
    )
    payload = {
        "status": (
            "POSTERIOR_TARGET_PASS"
            if calibration_pass and support_pass
            else "DIAGNOSTIC_COMPLETE"
        ),
        "stage": 5,
        "method": "matched_frozen_parent_sleep_npe_independent_test_closure_v1",
        "winner": {
            "arm": winner["selected_arm"],
            "checkpoint_sha256": winner["checkpoint_sha256"],
            "validation_sleep_nll": winner["validation_sleep_nll"],
        },
        "objects": {"full_test_k256": 4706, "support_test_k1024": 512},
        "baseline": {
            "redshift_q_calibration": baseline_cal,
            "k1024_projected_parent_support": baseline_support,
        },
        "sleep_npe": {
            "redshift_q_calibration": npe_cal,
            "k1024_projected_parent_support": npe_support,
        },
        "delta_sleep_npe_minus_baseline": improvement,
        "gates": {
            "posterior_redshift_calibration": calibration_pass,
            "individual_importance_support": support_pass,
            "thresholds": {
                "redshift_pit_ks_max": 0.05,
                "redshift_coverage_ece_max": 0.05,
                "median_ess_fraction_min": 0.05,
                "fraction_pareto_k_gt_0p7_max": 0.20,
                "p90_max_raw_weight_max": 0.80,
            },
        },
        "population_distributions": npe_full["population_distributions"],
        "artifacts": {
            "comparison_plot": str(comparison_plot.resolve()),
            "baseline_full": baseline_full["artifacts"],
            "baseline_support": baseline_support_receipt["artifacts"],
            "sleep_npe_full": npe_full["artifacts"],
            "sleep_npe_support": npe_support_receipt["artifacts"],
        },
        "point_estimates_used": False,
        "truth_used_for_training_or_checkpoint_selection": False,
        "truth_used_for_frozen_closure": True,
        "scientific_promotion": bool(calibration_pass and support_pass),
        "interpretation": (
            "Population-distribution agreement and conditional posterior "
            "calibration are separate gates. PSIS-IW remains diagnostic when "
            "the matched K=1024 support gate fails."
        ),
    }
    final_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(**vars(args)), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
