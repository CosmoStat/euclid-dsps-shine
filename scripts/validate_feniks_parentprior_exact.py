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


def _calibration(root: Path, *, include_adaptive_smc: bool) -> dict[str, object]:
    tarp = pd.read_csv(root / "calibration/tarp/tarp_summary.csv")
    mira = pd.read_csv(root / "calibration/mira/mira_scores.csv")
    models = ["q", "q_is", "defensive_is"]
    if include_adaptive_smc:
        models.append("adaptive_smc")
    models.append("nuts")
    tarp = tarp.loc[tarp["group"].eq("full_15d")].set_index("model")
    mira = mira.loc[mira["group"].eq("full_15d")].set_index("model")
    missing = sorted(set(models) - set(tarp.index)) + sorted(
        set(models) - set(mira.index)
    )
    if missing:
        raise ValueError(f"missing full-15D calibration rows: {missing}")

    thresholds = {
        "tarp_coverage_rmse_margin_over_nuts": 0.10,
        "tarp_max_abs_error_margin_over_nuts": 0.15,
        "mira_minimum_absolute_margin": 0.10,
        "mira_theoretical_sigma_multiplier": 3.0,
    }
    nuts_tarp_rmse = float(tarp.loc["nuts", "coverage_rmse"])
    nuts_tarp_max = float(tarp.loc["nuts", "coverage_max_abs_error"])
    nuts_mira_delta = abs(float(mira.loc["nuts", "delta_from_ideal"]))
    theoretical_sigma = float(mira.loc["nuts", "theoretical_sigma"])
    mira_margin = max(
        thresholds["mira_minimum_absolute_margin"],
        thresholds["mira_theoretical_sigma_multiplier"] * theoretical_sigma,
    )
    checks: dict[str, bool] = {}
    metrics: dict[str, dict[str, float]] = {}
    for model in models:
        tarp_rmse = float(tarp.loc[model, "coverage_rmse"])
        tarp_max = float(tarp.loc[model, "coverage_max_abs_error"])
        mira_score = float(mira.loc[model, "score"])
        mira_delta = float(mira.loc[model, "delta_from_ideal"])
        metrics[model] = {
            "tarp_coverage_rmse": tarp_rmse,
            "tarp_coverage_max_abs_error": tarp_max,
            "mira_score": mira_score,
            "mira_delta_from_ideal": mira_delta,
        }
        if model == "nuts":
            continue
        checks[f"{model}_tarp_close_to_nuts"] = bool(
            tarp_rmse
            <= nuts_tarp_rmse
            + thresholds["tarp_coverage_rmse_margin_over_nuts"]
            and tarp_max
            <= nuts_tarp_max
            + thresholds["tarp_max_abs_error_margin_over_nuts"]
        )
        checks[f"{model}_mira_close_to_nuts"] = bool(
            abs(mira_delta) <= nuts_mira_delta + mira_margin
        )
    return {
        "checks": checks,
        "metrics": metrics,
        "thresholds": thresholds,
        "mira_effective_margin": mira_margin,
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
    adaptive_available = bool(
        "adaptive_smc_eligible" in scoreboard
        and all("adaptive_smc" in item for item in geometry)
    )
    adaptive_checks: dict[str, bool] = {}
    adaptive_payload: dict[str, object] = {"available": adaptive_available}
    adaptive_geometry: dict[str, float] = {}
    central_payload: dict[str, object] = {}
    if adaptive_available:
        adaptive_max_ratio = np.asarray(
            [
                item["adaptive_smc"]["generalized_variance_ratio_max"]
                for item in geometry
            ],
            dtype=float,
        )
        adaptive_eligible = scoreboard["adaptive_smc_eligible"].astype(bool).to_numpy()
        adaptive_beta = pd.to_numeric(
            scoreboard["adaptive_smc_beta_final"], errors="coerce"
        ).to_numpy(dtype=float)
        adaptive_ess = pd.to_numeric(
            scoreboard["adaptive_smc_final_ess_fraction"], errors="coerce"
        ).to_numpy(dtype=float)
        agreement = pd.read_parquet(root / "posterior_agreement.parquet")
        adaptive_agreement = agreement.loc[agreement["method"].eq("Adaptive SMC")]
        adaptive_mean_offset = pd.to_numeric(
            adaptive_agreement["nuts_standardized_mean_offset"], errors="coerce"
        ).to_numpy(dtype=float)
        adaptive_width_ratio = pd.to_numeric(
            adaptive_agreement["std_ratio_to_nuts"], errors="coerce"
        ).to_numpy(dtype=float)
        central = pd.read_csv(root / "calibration/central_coverage.csv")
        central_summary = central.groupby("method")[["coverage_68", "coverage_95"]].mean()
        central_required = {"encoder", "adaptive_smc", "nuts"}
        if not central_required <= set(central_summary.index):
            raise ValueError("central coverage is missing q, adaptive SMC, or NUTS")
        q_coverage_delta = np.abs(
            central_summary.loc["encoder"].to_numpy(dtype=float)
            - central_summary.loc["nuts"].to_numpy(dtype=float)
        )
        adaptive_coverage_delta = np.abs(
            central_summary.loc["adaptive_smc"].to_numpy(dtype=float)
            - central_summary.loc["nuts"].to_numpy(dtype=float)
        )
        adaptive_checks = {
            "adaptive_smc_reaches_target": bool(
                np.mean(adaptive_eligible) >= 0.70
                and np.mean(np.isclose(adaptive_beta, 1.0, atol=1.0e-6)) >= 0.70
                and np.nanmedian(adaptive_ess) >= 0.30
            ),
            "adaptive_smc_matches_nuts_means": bool(
                np.isfinite(adaptive_mean_offset).all()
                and np.median(np.abs(adaptive_mean_offset)) <= 0.50
            ),
            "adaptive_smc_matches_nuts_widths": bool(
                np.isfinite(adaptive_width_ratio).all()
                and 0.70 <= np.median(adaptive_width_ratio) <= 1.30
            ),
            "adaptive_smc_covariance_mass_covering": bool(
                np.median(adaptive_max_ratio) < 2.0
                and np.mean(adaptive_max_ratio >= 2.0) <= 0.20
            ),
            "q_central_coverage_close_to_nuts": bool(
                np.max(q_coverage_delta) <= 0.20
            ),
            "adaptive_smc_central_coverage_close_to_nuts": bool(
                np.max(adaptive_coverage_delta) <= 0.20
            ),
        }
        adaptive_geometry = {
            "median_adaptive_smc_generalized_variance_ratio_max": float(
                np.median(adaptive_max_ratio)
            ),
            "fraction_adaptive_smc_generalized_variance_ratio_max_ge_2": float(
                np.mean(adaptive_max_ratio >= 2.0)
            ),
        }
        adaptive_payload = {
            "available": True,
            "eligible_fraction": float(np.mean(adaptive_eligible)),
            "fraction_beta_final_one": float(
                np.mean(np.isclose(adaptive_beta, 1.0, atol=1.0e-6))
            ),
            "median_final_ess_fraction": float(np.nanmedian(adaptive_ess)),
            "median_absolute_nuts_standardized_mean_offset": float(
                np.median(np.abs(adaptive_mean_offset))
            ),
            "median_width_ratio_to_nuts": float(np.median(adaptive_width_ratio)),
            "central_coverage_68": float(
                central_summary.loc["adaptive_smc", "coverage_68"]
            ),
            "central_coverage_95": float(
                central_summary.loc["adaptive_smc", "coverage_95"]
            ),
        }
        central_payload = {
            "q_central_coverage": {
                "coverage_68": float(central_summary.loc["encoder", "coverage_68"]),
                "coverage_95": float(central_summary.loc["encoder", "coverage_95"]),
            },
            "nuts_central_coverage": {
                "coverage_68": float(central_summary.loc["nuts", "coverage_68"]),
                "coverage_95": float(central_summary.loc["nuts", "coverage_95"]),
            },
        }
    calibration = _calibration(root, include_adaptive_smc=adaptive_available)
    nuts_convergence = all(
        bool(summary.get(key, False)) for key in summary if key.startswith("all_nuts_")
    )
    checks = {
        "all_nuts_samples_inside_shared_fit_bounds": bool(np.all(nuts_outside == 0.0)),
        "nuts_rhat_gate": nuts_convergence,
        "raw_q_is_support": raw["status"] == "PASS",
        "defensive_is_support": defensive["status"] == "PASS",
        "q_covariance_mass_covering": bool(
            np.median(q_max_ratio) < 2.0 and np.mean(q_max_ratio >= 2.0) <= 0.20
        ),
        "defensive_covariance_mass_covering": bool(
            np.median(defensive_max_ratio) < 2.0
            and np.mean(defensive_max_ratio >= 2.0) <= 0.20
        ),
        **adaptive_checks,
        **calibration["checks"],
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
        "parent_prior_identifiable_mass": bool(
            population_summary["selection"].get(
                "fraction_prior_mass_beta_lt_1e-3", 0.0
            )
            <= 0.50
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
            "fraction_defensive_generalized_variance_ratio_max_ge_2": float(
                np.mean(defensive_max_ratio >= 2.0)
            ),
            **adaptive_geometry,
        },
        "adaptive_smc": adaptive_payload,
        **central_payload,
        "calibration": calibration,
        "target_support": {
            "maximum_nuts_fraction_outside_fit_bounds": float(np.max(nuts_outside))
        },
        "population": {
            "selection_alpha_mc": population_summary["selection"]["alpha_mc"],
            "fraction_prior_mass_beta_lt_1e-3": population_summary[
                "selection"
            ].get("fraction_prior_mass_beta_lt_1e-3"),
            "fraction_prior_mass_beta_lt_1e-2": population_summary[
                "selection"
            ].get("fraction_prior_mass_beta_lt_1e-2"),
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
