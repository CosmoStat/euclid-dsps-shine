#!/usr/bin/env python3
"""Aggregate warm-start adapter diagnostics across controlled random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from evaluate_popcosmos_proposal_adapter import ADAPTER_CANDIDATES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", default="260820,260821,260822")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("adapter benchmark requires distinct random seeds")
    runs = {
        candidate: [_read(args.root / candidate / f"seed_{seed}") for seed in seeds]
        for candidate in ADAPTER_CANDIDATES
    }
    decision = adapter_decision(runs)
    payload = {
        "status": "complete",
        "experiment_contract": (
            "zero-initialized context residuals on the frozen current encoder; "
            "SMC-A fit, SMC-B validation, no catalog truth"
        ),
        **decision,
    }
    (args.root / "adapter_selection.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.root / "DONE").touch()
    print(
        "[proposal-adapter-select] "
        f"diagnosis={decision['diagnosis']} "
        f"recommended={decision['recommended_representation']} "
        f"next={decision['next_action']}",
        flush=True,
    )


def adapter_decision(runs):
    aggregate = {
        candidate: _aggregate(candidate_runs)
        for candidate, candidate_runs in runs.items()
    }
    current = aggregate["current_compressed"]
    for name, value in aggregate.items():
        if name == "current_compressed":
            value["material_gain_vs_current"] = False
            continue
        value["median_ess_ratio_vs_current"] = value["median_seed_ess_fraction"] / max(
            current["median_seed_ess_fraction"], 1.0e-12
        )
        value["median_bad_k_delta_vs_current"] = (
            value["median_seed_bad_pareto_fraction"]
            - current["median_seed_bad_pareto_fraction"]
        )
        value["mean_nll_delta_vs_current"] = (
            value["mean_validation_weighted_nll"]
            - current["mean_validation_weighted_nll"]
        )
        value["material_gain_vs_current"] = bool(
            value["exact_warm_start_pass"]
            and value["source_freeze_pass"]
            and value["mean_nll_delta_vs_current"] <= -0.02
            and (
                value["median_ess_ratio_vs_current"] >= 1.1
                or value["median_bad_k_delta_vs_current"] <= -0.05
                or value["mean_sliced_wasserstein"]
                <= 0.95 * current["mean_sliced_wasserstein"]
            )
        )
    free_gain = aggregate["free_context_adapter"]["material_gain_vs_current"]
    direct_gain = aggregate["direct_photometry_adapter"]["material_gain_vs_current"]
    token_gain = aggregate["band_token_adapter"]["material_gain_vs_current"]
    if direct_gain or token_gain:
        choices = [
            aggregate[name]
            for name in ("direct_photometry_adapter", "band_token_adapter")
            if aggregate[name]["material_gain_vs_current"]
        ]
        best = sorted(
            choices,
            key=lambda value: (
                -value["median_seed_ess_fraction"],
                value["median_seed_bad_pareto_fraction"],
                value["mean_validation_weighted_nll"],
            ),
        )[0]["candidate"]
        diagnosis = "OBSERVED_CONTEXT_RESIDUAL_HELPS"
        next_action = "SCALE_SELECTED_CONTEXT_WITH_SELFSUP_PRETRAINING_AND_SMC_WAKE"
    elif free_gain:
        best = "free_context_adapter"
        diagnosis = "FREE_CONTEXT_ONLY_GAIN_AMORTIZATION_GAP"
        next_action = "SCALE_TEACHER_OBJECTS_BEFORE_SELECTING_CONTEXT_ARCHITECTURE"
    else:
        best = None
        diagnosis = "NO_ADAPTER_GAIN_ON_SMALL_PANEL"
        next_action = "SCALE_DATA_WITH_CURRENT_WARM_START_BEFORE_ARCHITECTURE_DECISION"
    calibrated = [
        value["candidate"]
        for value in aggregate.values()
        if value["all_seed_support_pass"]
    ]
    return {
        "diagnosis": diagnosis,
        "recommended_representation": best,
        "calibrated_candidates": calibrated,
        "ready_for_production": bool(calibrated),
        "next_action": next_action,
        "candidates": list(aggregate.values()),
    }


def _aggregate(runs):
    validation = [run["ordinary_is"]["validation"] for run in runs]
    ess = np.asarray(
        [value["median_raw_ess_fraction"] for value in validation], dtype=float
    )
    bad_k = np.asarray(
        [value["fraction_pareto_k_gt_0p7"] for value in validation], dtype=float
    )
    nll = np.asarray(
        [value["validation_weighted_smc_b_nll"] for value in runs], dtype=float
    )
    sliced = np.asarray(
        [value["validation_geometry_medians"]["sliced_wasserstein"] for value in runs],
        dtype=float,
    )
    clone_error = np.asarray(
        [value["warm_start_contract"]["initial_max_abs_logq_error"] for value in runs]
    )
    source_change = np.asarray(
        [
            value["warm_start_contract"]["frozen_source_encoder_max_abs_change"]
            for value in runs
        ]
    )
    return {
        "candidate": runs[0]["candidate"],
        "seeds": [int(value["seed"]) for value in runs],
        "proposal_samples_per_object": int(runs[0]["proposal_samples_per_object"]),
        "exact_warm_start_pass": bool(np.max(clone_error) <= 2.0e-5),
        "source_freeze_pass": bool(np.max(source_change) == 0.0),
        "all_seed_support_pass": bool(
            all(value["support_status"] == "PASS" for value in validation)
        ),
        "median_seed_ess_fraction": float(np.median(ess)),
        "median_seed_bad_pareto_fraction": float(np.median(bad_k)),
        "mean_validation_weighted_nll": float(np.mean(nll)),
        "mean_sliced_wasserstein": float(np.mean(sliced)),
    }


def _read(root: Path):
    path = root / "adapter_summary.json"
    if not (root / "DONE").is_file() or not path.is_file():
        raise FileNotFoundError(f"incomplete adapter candidate: {root}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"adapter candidate failed: {root}")
    return payload


if __name__ == "__main__":
    main()
