#!/usr/bin/env python3
"""Apply explicit q/IS/NUTS support gates to the final exact cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _support(values: pd.DataFrame, prefix: str) -> dict[str, object]:
    ess = pd.to_numeric(values[f"{prefix}_raw_ess_fraction"], errors="coerce")
    pareto = pd.to_numeric(values[f"{prefix}_pareto_k"], errors="coerce")
    median_ess = float(np.nanmedian(ess))
    bad_pareto = float(np.mean(pareto > 0.7))
    return {
        "median_raw_ess_fraction": median_ess,
        "fraction_pareto_k_gt_0p7": bad_pareto,
        "fraction_pareto_k_gt_1": float(np.mean(pareto > 1.0)),
        "status": ("PASS" if median_ess >= 0.05 and bad_pareto <= 0.20 else "FAIL"),
    }


def validate(*, root: Path) -> dict[str, object]:
    scoreboard = pd.read_parquet(root / "scoreboard.parquet")
    summary = json.loads((root / "benchmark_summary.json").read_text())
    raw = _support(scoreboard, "importance")
    defensive = _support(scoreboard, "defensive_importance")
    population_summary = json.loads(
        (root / "population/population_summary.json").read_text()
    )
    population_comparisons = pd.read_csv(
        root / "population/population_comparisons.csv"
    ).set_index("comparison")
    parent_comparison = population_comparisons.loc["parent_prior_vs_parent_truth"]
    selected_comparison = population_comparisons.loc[
        "forward_selected_prior_vs_selected_truth"
    ]
    bounds = []
    geometry = []
    cohort = pd.read_parquet(root / "cohort.parquet")
    for item in cohort.itertuples(index=False):
        galaxy = (
            root
            / "galaxies"
            / (f"{int(item.order):02d}_{item.example_key}_row{int(item.row_index)}")
        )
        bounds.append(json.loads((galaxy / "fit_bounds_diagnostics.json").read_text()))
        geometry.append(
            json.loads((galaxy / "posterior_geometry_diagnostics.json").read_text())
        )
    nuts_outside = np.asarray(
        [item["nuts"]["fraction_of_samples_outside_fit_bounds"] for item in bounds],
        dtype=float,
    )
    q_max_ratio = np.asarray(
        [item["encoder"]["generalized_variance_ratio_max"] for item in geometry],
        dtype=float,
    )
    defensive_max_ratio = np.asarray(
        [
            item["defensive_encoder"]["generalized_variance_ratio_max"]
            for item in geometry
        ],
        dtype=float,
    )
    nuts_convergence = all(
        bool(summary.get(key, False)) for key in summary if key.startswith("all_nuts_")
    )
    checks = {
        "all_nuts_samples_inside_shared_fit_bounds": bool(np.all(nuts_outside == 0.0)),
        "nuts_rhat_gate": nuts_convergence,
        "raw_q_is_support": raw["status"] == "PASS",
        "defensive_is_support": defensive["status"] == "PASS",
        "parent_prior_population_closure": bool(
            parent_comparison["median_quantile_l1_over_truth_q90_width"] <= 0.50
            and parent_comparison["correlation_rmse"] <= 0.35
            and parent_comparison["min_std_ratio"] >= 0.20
        ),
        "forward_selected_prior_population_closure": bool(
            selected_comparison["median_quantile_l1_over_truth_q90_width"] <= 0.50
            and selected_comparison["correlation_rmse"] <= 0.35
            and selected_comparison["min_std_ratio"] >= 0.20
        ),
        "parent_prior_physical_support": bool(
            population_summary["selection"]["prior_physical_valid_fraction"] >= 0.99
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "raw_q_importance": raw,
        "defensive_importance": defensive,
        "geometry": {
            "median_q_generalized_variance_ratio_max": float(np.median(q_max_ratio)),
            "fraction_q_generalized_variance_ratio_max_ge_2": float(
                np.mean(q_max_ratio >= 2.0)
            ),
            "median_defensive_generalized_variance_ratio_max": float(
                np.median(defensive_max_ratio)
            ),
        },
        "target_support": {
            "maximum_nuts_fraction_outside_fit_bounds": float(np.max(nuts_outside))
        },
        "population": {
            "selection_alpha_mc": population_summary["selection"]["alpha_mc"],
            "parent_prior_vs_parent_truth": parent_comparison.to_dict(),
            "forward_selected_prior_vs_selected_truth": selected_comparison.to_dict(),
            "catalog_inferred_vs_selected_truth": population_comparisons.loc[
                "catalog_inferred_vs_selected_truth"
            ].to_dict(),
        },
        "ready_for_production": bool(all(checks.values())),
        "next_action": (
            "PROMOTE_PARENT_PRIOR_AND_AMORTIZED_POSTERIOR"
            if all(checks.values())
            else "STOP_REVIEW_EXACT_POSTERIOR_DIAGNOSTICS"
        ),
    }
    (root / "parentprior_exact_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    payload = validate(**vars(args))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
