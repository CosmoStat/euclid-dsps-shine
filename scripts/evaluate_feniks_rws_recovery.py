#!/usr/bin/env python3
"""Summarize and gate the truth-free FENIKS RWS recovery workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATES = ("historical_4x128", "current_residual_6x256")
SEEDS = (260826, 260827)


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _predictive_files(root: Path) -> list[Path]:
    monolithic = root / "posterior_predictive_flux.parquet"
    if monolithic.is_file():
        return [monolithic]
    files = sorted((root / "posterior_predictive_flux").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(root / "posterior_predictive_flux")
    return files


def support_tail_metrics(root: Path) -> dict[str, object]:
    path = root / "importance_diagnostics.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(
        path,
        columns=[
            "n_proposal_samples",
            "n_finite_logweights",
            "raw_ess_fraction",
            "max_raw_weight",
        ],
    )
    if frame.empty:
        raise ValueError("importance diagnostics are empty")
    ess = pd.to_numeric(frame["raw_ess_fraction"], errors="coerce").to_numpy()
    max_weight = pd.to_numeric(frame["max_raw_weight"], errors="coerce").to_numpy()
    finite_fraction = (
        pd.to_numeric(frame["n_finite_logweights"], errors="coerce").to_numpy()
        / pd.to_numeric(frame["n_proposal_samples"], errors="coerce").to_numpy()
    )
    metrics = {
        "p10_raw_ess_fraction": float(np.nanquantile(ess, 0.10)),
        "fraction_raw_ess_below_0p01": float(np.nanmean(ess < 0.01)),
        "p90_max_raw_weight": float(np.nanquantile(max_weight, 0.90)),
        "minimum_finite_logweight_fraction": float(np.nanmin(finite_fraction)),
    }
    metrics["thresholds"] = {
        "minimum_p10_raw_ess_fraction": 0.01,
        "maximum_fraction_raw_ess_below_0p01": 0.10,
        "maximum_p90_max_raw_weight": 0.50,
        "minimum_finite_logweight_fraction": 0.99,
    }
    metrics["status"] = (
        "PASS"
        if metrics["p10_raw_ess_fraction"] >= 0.01
        and metrics["fraction_raw_ess_below_0p01"] <= 0.10
        and metrics["p90_max_raw_weight"] <= 0.50
        and metrics["minimum_finite_logweight_fraction"] >= 0.99
        else "FAIL"
    )
    return metrics


def predictive_metrics(root: Path, *, out: Path) -> dict[str, object]:
    compact_path = root / "posterior_predictive_residual_summary.parquet"
    if not compact_path.is_file():
        raise FileNotFoundError(compact_path)
    compact = pd.read_parquet(
        compact_path,
        columns=[
            "row_index",
            "band",
            "obs_flux_fnu_cgs",
            "obs_err_fnu_cgs",
            "valid",
        ],
    ).drop_duplicates(["row_index", "band"])
    compact = compact.loc[
        compact["valid"].astype(bool)
        & np.isfinite(compact["obs_flux_fnu_cgs"])
        & np.isfinite(compact["obs_err_fnu_cgs"])
        & (compact["obs_err_fnu_cgs"] > 0.0)
    ]
    rows: list[pd.DataFrame] = []
    for path in _predictive_files(root):
        frame = pd.read_parquet(
            path,
            columns=["row_index", "sample_id", "band", "model_flux_fnu_cgs"],
        )
        merged = frame.merge(
            compact,
            on=["row_index", "band"],
            how="inner",
            validate="many_to_one",
        )
        merged["normalized_residual"] = (
            merged["obs_flux_fnu_cgs"] - merged["model_flux_fnu_cgs"]
        ) / merged["obs_err_fnu_cgs"]
        rows.append(
            merged[["row_index", "sample_id", "band", "normalized_residual"]]
        )
    residuals = pd.concat(rows, ignore_index=True)
    residuals = residuals.loc[np.isfinite(residuals["normalized_residual"])]
    if residuals.empty:
        raise ValueError("posterior predictive residual table is empty")
    by_band = (
        residuals.groupby("band", sort=True)["normalized_residual"]
        .agg(
            draws="size",
            mean="mean",
            median="median",
            rms=lambda values: float(np.sqrt(np.mean(np.square(values)))),
        )
        .reset_index()
    )
    by_band["abs_median"] = np.abs(by_band["median"])
    out.mkdir(parents=True, exist_ok=True)
    by_band.to_csv(out / "predictive_residuals_by_band.csv", index=False)
    metrics = {
        "objects": int(residuals["row_index"].nunique()),
        "draws": int(len(residuals)),
        "median_band_rms": float(by_band["rms"].median()),
        "maximum_band_rms": float(by_band["rms"].max()),
        "median_absolute_band_bias": float(by_band["abs_median"].median()),
        "maximum_absolute_band_bias": float(by_band["abs_median"].max()),
    }
    metrics["status"] = (
        "PASS"
        if metrics["median_band_rms"] <= 2.0
        and metrics["maximum_band_rms"] <= 4.0
        and metrics["median_absolute_band_bias"] <= 1.0
        else "FAIL"
    )
    metrics["thresholds"] = {
        "maximum_median_band_rms": 2.0,
        "maximum_worst_band_rms": 4.0,
        "maximum_median_absolute_band_bias": 1.0,
    }
    (out / "predictive_gate.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def summarize_candidate(
    *,
    candidate: str,
    seed: int,
    phase: str,
    config: Path,
    train: Path,
    importance: Path,
    predictive: Path,
    out: Path,
    smoke: bool = False,
) -> dict[str, object]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate}")
    training = _read_json(train / "training_summary.json")
    support = _read_json(importance / "importance_summary.json")
    support_tail = support_tail_metrics(importance)
    predictive_gate = predictive_metrics(predictive, out=out)
    training_pass = bool(
        int(training.get("updates_applied", 0)) > 0
        and int(training.get("wake_updates", 0)) > 0
        and int(training.get("updates_skipped", 0)) == 0
        and (train / "checkpoints" / "best.eqx").is_file()
    )
    scientific_pass = bool(
        training_pass
        and support["support_gate"]["status"] == "PASS"
        and support_tail["status"] == "PASS"
        and predictive_gate["status"] == "PASS"
    )
    status = "SMOKE_PASS" if smoke and training_pass else (
        "PASS" if scientific_pass else "FAIL"
    )
    payload = {
        "status": status,
        "phase": phase,
        "candidate": candidate,
        "seed": int(seed),
        "config": str(config),
        "checkpoint": str((train / "checkpoints" / "best.eqx").resolve()),
        "feature_stats": str((train / "feature_stats.json").resolve()),
        "training": {
            "status": "PASS" if training_pass else "FAIL",
            "train_rows": int(training["train_rows"]),
            "validation_rows": int(training["validation_rows"]),
            "epochs": int(training["epochs"]),
            "updates_applied": int(training["updates_applied"]),
            "updates_skipped": int(training["updates_skipped"]),
            "sleep_updates": int(training["sleep_updates"]),
            "wake_updates": int(training["wake_updates"]),
            "wake_ess_fraction_mean": float(training["wake_ess_fraction_mean"]),
            "best_checkpoint_epoch": int(training["best_checkpoint_epoch"]),
        },
        "exact_gaussian_ordinary_iw": {
            "status": support["support_gate"]["status"],
            "objects": int(support["n_objects"]),
            "draws": int(support["n_joint_draws"]),
            "median_raw_ess_fraction": float(support["median_raw_ess_fraction"]),
            "fraction_pareto_k_gt_0p7": float(
                support["fraction_pareto_k_gt_0p7"]
            ),
            "fraction_pareto_k_gt_1": float(support["fraction_pareto_k_gt_1"]),
            "tail_gate": support_tail,
        },
        "exact_gaussian_posterior_predictive": predictive_gate,
        "truth_used_for_training_or_checkpoint_selection": False,
        "promotion_contract": (
            "Both seeds must pass ordinary IW under the exact Gaussian target and "
            "the dense posterior-predictive gate. Truth is reserved for closure "
            "after candidate selection."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{phase}_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _candidate_runs(root: Path, candidate: str, phase: str) -> list[dict[str, object]]:
    return [
        _read_json(root / candidate / f"seed_{seed}" / f"{phase}_summary.json")
        for seed in SEEDS
    ]


def select_pilot(root: Path) -> dict[str, object]:
    candidates = []
    for candidate in CANDIDATES:
        runs = _candidate_runs(root, candidate, "pilot")
        candidates.append(
            {
                "candidate": candidate,
                "all_seed_pass": all(run["status"] == "PASS" for run in runs),
                "median_seed_ess_fraction": float(
                    np.median(
                        [
                            run["exact_gaussian_ordinary_iw"][
                                "median_raw_ess_fraction"
                            ]
                            for run in runs
                        ]
                    )
                ),
                "worst_seed_pareto_fraction": float(
                    np.max(
                        [
                            run["exact_gaussian_ordinary_iw"][
                                "fraction_pareto_k_gt_0p7"
                            ]
                            for run in runs
                        ]
                    )
                ),
                "median_seed_predictive_rms": float(
                    np.median(
                        [
                            run["exact_gaussian_posterior_predictive"][
                                "median_band_rms"
                            ]
                            for run in runs
                        ]
                    )
                ),
                "runs": runs,
            }
        )
    eligible = [value for value in candidates if value["all_seed_pass"]]
    eligible.sort(
        key=lambda value: (
            -value["median_seed_ess_fraction"],
            value["worst_seed_pareto_fraction"],
            value["median_seed_predictive_rms"],
        )
    )
    selected = eligible[0]["candidate"] if eligible else None
    payload = {
        "status": "PASS" if selected else "FAIL",
        "selected_candidate": selected,
        "ready_for_independent_2000_object_confirmation": bool(selected),
        "truth_used_for_training_or_checkpoint_selection": False,
        "candidates": candidates,
    }
    name = "PILOT_PASS.json" if selected else "PILOT_FAIL.json"
    (root / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def finalize_confirmation(root: Path) -> dict[str, object]:
    pilot = _read_json(root / "PILOT_PASS.json")
    selected = str(pilot["selected_candidate"])
    runs = _candidate_runs(root, selected, "confirmation")
    passed = all(run["status"] == "PASS" for run in runs)
    payload = {
        "status": "PASS" if passed else "FAIL",
        "selected_candidate": selected,
        "selected_checkpoints": [run["checkpoint"] for run in runs],
        "selected_feature_stats": [run["feature_stats"] for run in runs],
        "pilot_receipt": str((root / "PILOT_PASS.json").resolve()),
        "confirmation_objects_per_seed": [
            run["exact_gaussian_ordinary_iw"]["objects"] for run in runs
        ],
        "ready_for_smc_diversity_benchmark": bool(passed),
        "ready_for_population_prior_update": False,
        "ready_for_full_catalogue": False,
        "truth_used_for_training_or_checkpoint_selection": False,
        "next_action": (
            "RUN_FIXED_COHORT_MALA_SMC_DIVERSITY_BENCHMARK"
            if passed
            else "STOP_AND_REVIEW_RWS_SUPPORT_DIAGNOSTICS"
        ),
        "runs": runs,
    }
    name = "RWS_RECOVERY_PASS.json" if passed else "RWS_RECOVERY_FAIL.json"
    (root / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--candidate", required=True, choices=CANDIDATES)
    summarize.add_argument("--seed", type=int, required=True)
    summarize.add_argument("--phase", choices=("pilot", "confirmation"), required=True)
    summarize.add_argument("--config", type=Path, required=True)
    summarize.add_argument("--train", type=Path, required=True)
    summarize.add_argument("--importance", type=Path, required=True)
    summarize.add_argument("--predictive", type=Path, required=True)
    summarize.add_argument("--out", type=Path, required=True)
    summarize.add_argument("--smoke", action="store_true")
    pilot = sub.add_parser("select-pilot")
    pilot.add_argument("--root", type=Path, required=True)
    final = sub.add_parser("finalize-confirmation")
    final.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    values = vars(args)
    command = values.pop("command")
    if command == "summarize":
        payload = summarize_candidate(**values)
    elif command == "select-pilot":
        payload = select_pilot(**values)
    else:
        payload = finalize_confirmation(**values)
    print(json.dumps(payload, indent=2), flush=True)
    if command != "summarize" and payload["status"] == "FAIL":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
