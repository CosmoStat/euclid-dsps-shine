#!/usr/bin/env python3
"""Aggregate Phase 0/1 posterior experiments and gate Phase 2 promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from evaluate_popcosmos_proposal_architecture import (
    PHASE0_CANDIDATES,
    PHASE1_CANDIDATES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase1-seeds", default="260820,260821,260822")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.phase1_seeds.split(","))
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Phase 1 requires at least two distinct seeds")
    phase0 = {
        name: _read_summary(args.root / "phase0" / name / "seed_260820")
        for name in PHASE0_CANDIDATES
    }
    phase1 = {
        name: [
            _read_summary(args.root / "phase1" / name / f"seed_{seed}")
            for seed in seeds
        ]
        for name in PHASE1_CANDIDATES
    }
    phase0_decision = _phase0_decision(phase0)
    phase1_decision = _phase1_decision(phase1)
    # Phase 0 localizes the likely gap but is not itself a production gate: a
    # finite-particle KDE can be inconclusive in 15 dimensions. A stable direct
    # Phase 1 ordinary-IS PASS is stronger evidence and controls promotion.
    promote = bool(phase1_decision["selection_status"] == "PASS")
    selected = phase1_decision["selected_candidate"] if promote else None
    summary = {
        "status": "complete",
        "experiment_contract": (
            "purely self-supervised weighted-SMC posterior repair; no truth labels; "
            "prior, likelihood, selection, features, cohorts and support gates frozen"
        ),
        "phase0": phase0_decision,
        "phase1": phase1_decision,
        "phase2_promotion": {
            "status": "PASS" if promote else "FAIL",
            "selected_architecture": selected,
            "allowed_objectives": (
                ["fixed_bank_inclusive_kl", "smc_wake", "selected_sleep_plus_smc_wake"]
                if promote
                else []
            ),
            "alpha2_allowed": False,
            "independent_confirmation_required": bool(promote),
            "next_action": (
                "RUN_PHASE2_TOP_ARCHITECTURE_OBJECTIVES"
                if promote
                else phase1_decision["next_action"]
            ),
        },
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "architecture_selection.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.root / "DONE").touch()
    print(
        "[proposal-architecture-select] "
        f"phase0={phase0_decision['diagnostic_status']} "
        f"phase1={phase1_decision['selection_status']} "
        f"phase2={'PASS' if promote else 'BLOCKED'} selected={selected}",
        flush=True,
    )


def _read_summary(root: Path) -> dict:
    path = root / "candidate_summary.json"
    if not (root / "DONE").is_file() or not path.is_file():
        raise FileNotFoundError(f"incomplete architecture candidate: {root}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"candidate did not complete: {root}")
    return payload


def _validation(summary):
    return summary["ordinary_is"]["validation"]


def _phase0_decision(candidates):
    metrics = {
        name: {
            "support_status": _validation(value)["support_status"],
            "median_raw_ess_fraction": _validation(value)["median_raw_ess_fraction"],
            "fraction_pareto_k_gt_0p7": _validation(value)["fraction_pareto_k_gt_0p7"],
            "weighted_smc_b_nll": value["validation_weighted_smc_b_nll"],
            "sliced_wasserstein": value["validation_geometry_medians"][
                "sliced_wasserstein"
            ],
            "nearest_cover_ratio": value["validation_geometry_medians"][
                "nearest_cover_ratio"
            ],
        }
        for name, value in candidates.items()
    }
    current = metrics["current_compressed"]
    oracle = metrics["oracle_kde"]
    free = metrics["free_context_rqspline"]
    direct = metrics["direct_context_realnvp"]
    oracle_materially_better = bool(
        oracle["median_raw_ess_fraction"] >= 2.0 * current["median_raw_ess_fraction"]
        and oracle["fraction_pareto_k_gt_0p7"]
        <= current["fraction_pareto_k_gt_0p7"] - 0.1
    )
    free_materially_better = bool(
        free["median_raw_ess_fraction"] >= 1.5 * current["median_raw_ess_fraction"]
        and free["fraction_pareto_k_gt_0p7"]
        <= current["fraction_pareto_k_gt_0p7"] - 0.05
    )
    direct_materially_better = bool(
        direct["median_raw_ess_fraction"] >= 1.25 * current["median_raw_ess_fraction"]
        and direct["fraction_pareto_k_gt_0p7"]
        <= current["fraction_pareto_k_gt_0p7"] - 0.05
    )
    if oracle["support_status"] != "PASS" and not oracle_materially_better:
        status = "KDE_ORACLE_INCONCLUSIVE"
        interpretation = (
            "Even the per-object nonparametric oracle does not cover the target; "
            "inspect target geometry, latent coordinates and SMC/KDE resolution."
        )
    elif free["support_status"] != "PASS" and not free_materially_better:
        status = "SHARED_FLOW_CAPACITY_GAP"
        interpretation = (
            "The local oracle improves support but a shared flow with free object "
            "contexts does not; shared transform capacity is the leading bottleneck."
        )
    elif direct["support_status"] != "PASS" and not direct_materially_better:
        status = "AMORTIZATION_OR_FEATURE_GAP"
        interpretation = (
            "Free object contexts outperform observed-photometry conditioning; "
            "the remaining gap is amortization, features or context optimization."
        )
    else:
        status = "CURRENT_CONTEXT_BOTTLENECK_EVIDENCE"
        interpretation = (
            "Direct photometry context materially improves the current compressed "
            "context, supporting Phase 1 architecture selection."
        )
    return {
        "diagnostic_status": status,
        "interpretation": interpretation,
        "checks": {
            "oracle_materially_better_than_current": oracle_materially_better,
            "free_context_materially_better_than_current": free_materially_better,
            "direct_context_materially_better_than_current": direct_materially_better,
        },
        "candidates": metrics,
    }


def _phase1_decision(candidates):
    aggregate = []
    for name, runs in candidates.items():
        validation = [_validation(run) for run in runs]
        ess = np.asarray(
            [value["median_raw_ess_fraction"] for value in validation], dtype=float
        )
        bad_k = np.asarray(
            [value["fraction_pareto_k_gt_0p7"] for value in validation], dtype=float
        )
        nll = np.asarray(
            [value["validation_weighted_smc_b_nll"] for value in runs], dtype=float
        )
        passes = [value["support_status"] == "PASS" for value in validation]
        stable = bool(
            np.all(np.isfinite(ess))
            and np.min(ess) >= 0.05
            and np.max(bad_k) <= 0.2
            and np.std(ess) / max(float(np.mean(ess)), 1.0e-12) <= 0.25
        )
        aggregate.append(
            {
                "candidate": name,
                "seeds": [int(run["seed"]) for run in runs],
                "all_seed_support_pass": bool(all(passes)),
                "seed_stability_pass": stable,
                "median_seed_ess_fraction": float(np.median(ess)),
                "worst_seed_ess_fraction": float(np.min(ess)),
                "median_seed_bad_pareto_fraction": float(np.median(bad_k)),
                "worst_seed_bad_pareto_fraction": float(np.max(bad_k)),
                "mean_validation_weighted_nll": float(np.mean(nll)),
                "eligible": bool(all(passes) and stable),
            }
        )
    eligible = [value for value in aggregate if value["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda value: (
            -value["median_seed_ess_fraction"],
            value["median_seed_bad_pareto_fraction"],
            value["mean_validation_weighted_nll"],
        ),
    )
    selected = ranked[0]["candidate"] if ranked else None
    best_diagnostic = sorted(
        aggregate,
        key=lambda value: (
            -value["median_seed_ess_fraction"],
            value["median_seed_bad_pareto_fraction"],
        ),
    )[0]["candidate"]
    return {
        "selection_status": "PASS" if selected else "FAIL",
        "selection_rule": (
            "require every seed to pass median ESS >= 0.05 and bad Pareto-k "
            "fraction <= 0.20, plus ESS coefficient of variation <= 0.25; then "
            "rank by ESS, Pareto tail and held-out weighted NLL"
        ),
        "selected_candidate": selected,
        "best_diagnostic_candidate": best_diagnostic,
        "candidates": aggregate,
        "next_action": (
            "PROMOTE_TOP_ARCHITECTURE_TO_PHASE2"
            if selected
            else "STOP_AND_REVIEW_PHASE0_GAP_BEFORE_NEW_OBJECTIVES"
        ),
    }


if __name__ == "__main__":
    main()
