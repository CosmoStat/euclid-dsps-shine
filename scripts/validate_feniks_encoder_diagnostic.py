#!/usr/bin/env python3
"""Summarize the two-domain exact benchmark without promoting the prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = ("observed_catalog", "sleep_synthetic")
METHODS = ("q", "q_is", "defensive_is", "nuts")


def _support(frame: pd.DataFrame, prefix: str) -> dict[str, object]:
    ess = pd.to_numeric(frame[f"{prefix}_raw_ess_fraction"], errors="coerce")
    pareto = pd.to_numeric(frame[f"{prefix}_pareto_k"], errors="coerce")
    finite = np.isfinite(ess) & np.isfinite(pareto)
    median_ess = float(np.nanmedian(ess))
    bad_k = float(np.mean(pareto > 0.7))
    return {
        "objects": int(len(frame)),
        "finite_fraction": float(np.mean(finite)),
        "median_raw_ess_fraction": median_ess,
        "fraction_pareto_k_gt_0p7": bad_k,
        "fraction_pareto_k_gt_1": float(np.mean(pareto > 1.0)),
        "status": (
            "PASS"
            if bool(np.all(finite)) and median_ess >= 0.05 and bad_k <= 0.20
            else "FAIL"
        ),
    }


def _calibration(root: Path, domain: str) -> dict[str, object]:
    base = root / f"calibration_{domain}"
    tarp = pd.read_csv(base / "tarp/tarp_summary.csv")
    mira = pd.read_csv(base / "mira/mira_scores.csv")
    tarp = tarp.loc[tarp["group"].eq("full_15d")].set_index("model")
    mira = mira.loc[mira["group"].eq("full_15d")].set_index("model")
    missing = sorted(set(METHODS) - set(tarp.index)) + sorted(
        set(METHODS) - set(mira.index)
    )
    if missing:
        raise ValueError(f"missing full-15D calibration rows for {domain}: {missing}")
    return {
        method: {
            "tarp_coverage_rmse": float(tarp.loc[method, "coverage_rmse"]),
            "tarp_coverage_max_abs_error": float(
                tarp.loc[method, "coverage_max_abs_error"]
            ),
            "mira_score": float(mira.loc[method, "score"]),
            "mira_delta_from_ideal": float(mira.loc[method, "delta_from_ideal"]),
        }
        for method in METHODS
    }


def _galaxy_dir(root: Path, item) -> Path:
    return (
        root
        / "galaxies"
        / (f"{int(item.order):02d}_{item.example_key}_row{int(item.row_index)}")
    )


def _domain_summary(
    root: Path, cohort: pd.DataFrame, scoreboard: pd.DataFrame, domain: str
):
    selected = cohort.loc[cohort["domain"].astype(str).eq(domain)]
    scores = scoreboard.loc[scoreboard["domain"].astype(str).eq(domain)]
    q_ratios = []
    defensive_ratios = []
    nuts_outside = []
    for item in selected.itertuples(index=False):
        galaxy = _galaxy_dir(root, item)
        bounds = json.loads((galaxy / "fit_bounds_diagnostics.json").read_text())
        geometry = json.loads(
            (galaxy / "posterior_geometry_diagnostics.json").read_text()
        )
        nuts_outside.append(bounds["nuts"]["fraction_of_samples_outside_fit_bounds"])
        q_ratios.append(geometry["encoder"]["generalized_variance_ratio_max"])
        defensive_ratios.append(
            geometry["defensive_encoder"]["generalized_variance_ratio_max"]
        )
    q_ratios = np.asarray(q_ratios, dtype=float)
    defensive_ratios = np.asarray(defensive_ratios, dtype=float)
    nuts_outside = np.asarray(nuts_outside, dtype=float)
    nuts_rhat = pd.to_numeric(scores["nuts_max_rhat"], errors="coerce").to_numpy()
    agreement = pd.read_parquet(root / "posterior_agreement.parquet")
    agreement = agreement.loc[agreement["domain"].astype(str).eq(domain)]
    agreement_summary = {}
    for method in ("Encoder", "Encoder + IS", "Defensive + IS"):
        method_rows = agreement.loc[agreement["method"].eq(method)]
        agreement_summary[method] = {
            "median_wasserstein_to_nuts_in_nuts_std": float(
                np.nanmedian(method_rows["wasserstein_to_nuts_in_nuts_std"])
            ),
            "median_std_ratio_to_nuts": float(
                np.nanmedian(method_rows["std_ratio_to_nuts"])
            ),
            "median_absolute_nuts_standardized_mean_offset": float(
                np.nanmedian(np.abs(method_rows["nuts_standardized_mean_offset"]))
            ),
        }
    return {
        "objects": int(len(selected)),
        "q_only_importance": _support(scores, "importance"),
        "defensive_importance": _support(scores, "defensive_importance"),
        "nuts": {
            "maximum_rhat": float(np.nanmax(nuts_rhat)),
            "all_rhat_le_1p01": bool(
                np.isfinite(nuts_rhat).all() and np.all(nuts_rhat <= 1.01)
            ),
            "maximum_fraction_outside_shared_fit_bounds": float(
                np.nanmax(nuts_outside)
            ),
        },
        "geometry": {
            "median_q_generalized_variance_ratio_max": float(np.median(q_ratios)),
            "fraction_q_generalized_variance_ratio_max_ge_2": float(
                np.mean(q_ratios >= 2.0)
            ),
            "median_defensive_generalized_variance_ratio_max": float(
                np.median(defensive_ratios)
            ),
            "fraction_defensive_generalized_variance_ratio_max_ge_2": float(
                np.mean(defensive_ratios >= 2.0)
            ),
        },
        "agreement_with_nuts": agreement_summary,
        "truth_closure": _calibration(root, domain),
    }


def _diagnosis(domains: dict[str, dict[str, object]]) -> tuple[str, str]:
    sleep = domains["sleep_synthetic"]
    observed = domains["observed_catalog"]
    sleep_q = sleep["q_only_importance"]["status"] == "PASS"
    observed_q = observed["q_only_importance"]["status"] == "PASS"
    sleep_def = sleep["defensive_importance"]["status"] == "PASS"
    observed_def = observed["defensive_importance"]["status"] == "PASS"
    if sleep_q and observed_q:
        return "RAW_Q_SUPPORT_RECOVERED", "CONFIRM_ON_A_NEW_DISJOINT_COHORT"
    if sleep_def and observed_def:
        return "DEFENSIVE_IS_SUPPORT_RECOVERED", "CONFIRM_DEFENSIVE_IS_DISJOINT"
    if (sleep_q or sleep_def) and not (observed_q or observed_def):
        return (
            "SIMULATION_TO_OBSERVATION_GAP",
            "ALIGN_SLEEP_SIMULATOR_WITH_OBSERVED_PHOTOMETRY",
        )
    if not (sleep_q or sleep_def):
        return (
            "SLEEP_POSTERIOR_NOT_MASS_COVERING",
            "FIX_SLEEP_TARGET_OR_OPTIMIZATION_BEFORE_PRIOR_WAKE",
        )
    return "OBSERVED_SUPPORT_PARTIAL", "STRATIFY_FAILURES_AND_REFINE_PROPOSAL"


def validate(*, root: Path) -> dict[str, object]:
    contract = json.loads((root / "contract.json").read_text())
    if contract.get("analysis_contract") != "ENCODER_DIAGNOSTIC_ONLY":
        raise ValueError("benchmark is not marked ENCODER_DIAGNOSTIC_ONLY")
    cohort = pd.read_parquet(root / "cohort.parquet")
    scoreboard = pd.read_parquet(root / "scoreboard.parquet")
    if set(cohort["domain"].astype(str)) != set(DOMAINS):
        raise ValueError("diagnostic cohort must contain both configured domains")
    domains = {
        domain: _domain_summary(root, cohort, scoreboard, domain) for domain in DOMAINS
    }
    diagnosis, next_action = _diagnosis(domains)
    nuts_valid = all(
        value["nuts"]["all_rhat_le_1p01"]
        and value["nuts"]["maximum_fraction_outside_shared_fit_bounds"] == 0.0
        for value in domains.values()
    )
    payload = {
        "status": "complete" if nuts_valid else "INVALID_EXACT_REFERENCE",
        "analysis_contract": "ENCODER_DIAGNOSTIC_ONLY",
        "scientific_diagnosis": diagnosis,
        "domains": domains,
        "nuts_reference_valid": nuts_valid,
        "prior_updated_by_training": False,
        "ready_for_prior_promotion": False,
        "ready_for_production": False,
        "truth_role": "FENIKS closure diagnostics only; never training or checkpoint selection",
        "next_action": next_action if nuts_valid else "FIX_NUTS_REFERENCE_FIRST",
    }
    (root / "encoder_diagnostic_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
