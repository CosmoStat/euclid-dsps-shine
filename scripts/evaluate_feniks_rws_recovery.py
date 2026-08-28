#!/usr/bin/env python3
"""Summarize and gate the truth-free FENIKS RWS recovery workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.config import load_config

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


def truth_free_inference_contract(root: Path) -> dict[str, object]:
    summary_path = root / "inference_summary.json"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    snapshot = root / "inference_truth.parquet"
    passed = bool(
        summary.get("truth_snapshot_enabled") is False
        and summary.get("truth_diagnostics_enabled") is False
        and summary.get("truth_used_for_inference_or_checkpoint_selection") is False
        and int(summary.get("truth_snapshot_rows", -1)) == 0
        and not snapshot.exists()
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "inference_summary": str(summary_path),
        "truth_snapshot_present": snapshot.exists(),
        "truth_snapshot_enabled": summary.get("truth_snapshot_enabled"),
        "truth_diagnostics_enabled": summary.get("truth_diagnostics_enabled"),
        "truth_used_for_inference_or_checkpoint_selection": summary.get(
            "truth_used_for_inference_or_checkpoint_selection"
        ),
    }


def _importance_source_inference_root(summary: dict[str, object]) -> Path | None:
    inputs = summary.get("inputs", {})
    if not isinstance(inputs, dict):
        return None
    receipt = inputs.get("posterior_inference_summary")
    if not isinstance(receipt, dict) or not receipt.get("path"):
        return None
    return Path(str(receipt["path"])).parent


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


def _training_metrics(
    train: Path,
    *,
    maximum_entropy_drop: float = 3.0,
    maximum_unresolved_fraction: float = 0.02,
) -> dict[str, object]:
    """Read either the SC-DRWS receipt or the legacy diagnostic summary."""
    receipt_path = train / "training_receipt.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path)
        log_path = train / "sc_drws_training_log.csv"
        frame = pd.read_csv(log_path) if log_path.is_file() else pd.DataFrame()
        final_entropy = float("nan")
        if not frame.empty and "full_entropy" in frame:
            values = pd.to_numeric(frame["full_entropy"], errors="coerce")
            final_epoch = frame.loc[values.notna(), "epoch"].max()
            final_entropy = float(
                pd.to_numeric(
                    frame.loc[frame["epoch"] == final_epoch, "full_entropy"],
                    errors="coerce",
                ).median()
            )
        reference = float(receipt.get("reference_entropy", np.nan))
        entropy_ok = bool(
            np.isfinite(final_entropy)
            and np.isfinite(reference)
            and final_entropy >= reference - float(maximum_entropy_drop)
        )
        wake = frame.loc[frame.get("update_kind") == "wake"] if not frame.empty else frame
        final_unresolved = float("nan")
        if not wake.empty:
            last_wake_epoch = wake["epoch"].max()
            final_unresolved = float(
                pd.to_numeric(
                    wake.loc[
                        wake["epoch"] == last_wake_epoch, "unresolved_fraction"
                    ],
                    errors="coerce",
                ).mean()
            )
        unresolved_ok = bool(
            np.isfinite(final_unresolved)
            and final_unresolved <= float(maximum_unresolved_fraction)
        )
        q_gradients_finite = bool(
            not frame.empty
            and "q_grads_finite" in frame
            and frame["q_grads_finite"].astype(bool).all()
        )
        receipt_complete = bool(
            receipt.get("status")
            == "TRAINING_COMPLETE_PENDING_SUPPORT_SELECTION"
        )
        truth_free_training = bool(
            receipt.get(
                "truth_used_for_training_validation_or_checkpoint_selection"
            )
            is False
        )
        wake_path_exercised = bool(
            not wake.empty and int(receipt.get("wake_updates", 0)) > 0
        )
        hard_mis_path_exercised = bool(
            not wake.empty
            and "expansion_fraction" in wake
            and (
                pd.to_numeric(wake["expansion_fraction"], errors="coerce") > 0.0
            ).any()
        )
        prior_path = train / "sc_drws_prior_log.csv"
        prior_frame = pd.read_csv(prior_path) if prior_path.is_file() else pd.DataFrame()
        prior_gate_evaluated = bool(
            not prior_frame.empty
            and "gate_accepted" in prior_frame
            and "update_applied" in prior_frame
        )
        applied_prior = (
            prior_frame.loc[prior_frame["update_applied"].astype(bool)]
            if not prior_frame.empty and "update_applied" in prior_frame
            else pd.DataFrame()
        )
        prior_gradients_finite = bool(
            not applied_prior.empty
            and all(
                applied_prior[name].astype(bool).all()
                for name in (
                    "selection_gradient_finite",
                    "data_gradient_finite",
                    "trust_gradient_finite",
                )
            )
        )
        selection_finite = bool(
            np.isfinite(float(receipt.get("selection_log_alpha", np.nan)))
            and np.isfinite(float(receipt.get("selection_alpha", np.nan)))
        )
        technical_smoke_pass = bool(
            receipt_complete
            and truth_free_training
            and wake_path_exercised
            and hard_mis_path_exercised
            and prior_gate_evaluated
            and selection_finite
            and np.isfinite(final_entropy)
            and q_gradients_finite
        )
        training_pass = bool(
            receipt_complete
            and truth_free_training
            and wake_path_exercised
            and int(receipt.get("prior_updates", 0)) > 0
            and selection_finite
            and entropy_ok
            and unresolved_ok
            and q_gradients_finite
            and prior_gradients_finite
        )
        return {
            "status": "PASS" if training_pass else "FAIL",
            "technical_smoke_status": (
                "PASS" if technical_smoke_pass else "FAIL"
            ),
            "train_rows": int(receipt["selected_training_rows"]),
            "validation_rows": None,
            "epochs": int(receipt["phase_schedule"]["total_epochs"])
            if "total_epochs" in receipt["phase_schedule"]
            else int(receipt["phase_schedule"]["warmup_epochs"])
            + int(receipt["phase_schedule"]["joint_epochs"]),
            "updates_applied": int(receipt.get("wake_updates", 0)),
            "updates_skipped": 0,
            "sleep_updates": int((frame["update_kind"] == "sleep").sum())
            if not frame.empty
            else 0,
            "wake_updates": int(receipt.get("wake_updates", 0)),
            "wake_ess_fraction_mean": float(
                pd.to_numeric(frame.get("expanded_ess_fraction"), errors="coerce").mean()
            )
            if not frame.empty
            else float("nan"),
            "best_checkpoint_epoch": int(
                receipt["phase_schedule"]["warmup_epochs"]
                + receipt["phase_schedule"]["joint_epochs"]
            ),
            "selection_log_alpha": float(receipt["selection_log_alpha"]),
            "selection_alpha": float(receipt["selection_alpha"]),
            "reference_entropy": reference,
            "final_entropy": final_entropy,
            "entropy_not_collapsed": entropy_ok,
            "final_unresolved_fraction": final_unresolved,
            "unresolved_fraction_below_gate": unresolved_ok,
            "q_gradients_finite": q_gradients_finite,
            "truth_free_training_contract": truth_free_training,
            "wake_path_exercised": wake_path_exercised,
            "hard_mis_path_exercised": hard_mis_path_exercised,
            "prior_gate_evaluated": prior_gate_evaluated,
            "selection_finite": selection_finite,
            "q_gradient_clipped_fraction": float(
                frame["q_grad_clipped"].astype(bool).mean()
            )
            if not frame.empty and "q_grad_clipped" in frame
            else float("nan"),
            "all_applied_prior_component_gradients_finite": prior_gradients_finite,
            "raw_model_checkpoint": receipt["raw_model_checkpoint"]["path"],
            "ema_model_checkpoint": receipt["ema_model_checkpoint"]["path"],
            "feature_stats": str((train / "feature_stats.json").resolve()),
        }
    training = _read_json(train / "training_summary.json")
    checkpoint = train / "checkpoints" / "best.eqx"
    training_pass = bool(
        int(training.get("updates_applied", 0)) > 0
        and int(training.get("wake_updates", 0)) > 0
        and int(training.get("updates_skipped", 0)) == 0
        and checkpoint.is_file()
    )
    return {
        "status": "PASS" if training_pass else "FAIL",
        "technical_smoke_status": "PASS" if training_pass else "FAIL",
        **{
            key: training[key]
            for key in (
                "train_rows",
                "validation_rows",
                "epochs",
                "updates_applied",
                "updates_skipped",
                "sleep_updates",
                "wake_updates",
                "wake_ess_fraction_mean",
                "best_checkpoint_epoch",
            )
        },
        "selection_log_alpha": 0.0,
        "raw_model_checkpoint": str(checkpoint.resolve()),
        "ema_model_checkpoint": str(checkpoint.resolve()),
        "feature_stats": str((train / "feature_stats.json").resolve()),
    }


def _variant_metrics(
    *, label: str, importance: Path, predictive: Path, out: Path, log_alpha: float
) -> dict[str, object]:
    support = _read_json(importance / "importance_summary.json")
    support_gate = support.get("support_gate", {})
    if not isinstance(support_gate, dict):
        support_gate = {}
    mean_log_evidence_is = float(
        support_gate.get(
            "mean_log_evidence_is",
            support.get("mean_log_evidence_is", np.nan),
        )
    )
    support_tail = support_tail_metrics(importance)
    predictive_gate = predictive_metrics(predictive, out=out / label)
    support_source = _importance_source_inference_root(support)
    support_truth_contract = (
        truth_free_inference_contract(support_source)
        if support_source is not None
        else {"status": "FAIL", "reason": "missing source inference receipt"}
    )
    predictive_truth_contract = truth_free_inference_contract(predictive)
    support_values = (
        support.get("median_raw_ess_fraction", np.nan),
        support.get("fraction_pareto_k_gt_0p7", np.nan),
        support.get("fraction_pareto_k_gt_1", np.nan),
        mean_log_evidence_is,
        support_tail.get("p10_raw_ess_fraction", np.nan),
        support_tail.get("fraction_raw_ess_below_0p01", np.nan),
        support_tail.get("p90_max_raw_weight", np.nan),
        support_tail.get("minimum_finite_logweight_fraction", np.nan),
    )
    predictive_values = (
        predictive_gate.get("median_band_rms", np.nan),
        predictive_gate.get("maximum_band_rms", np.nan),
        predictive_gate.get("median_absolute_band_bias", np.nan),
        predictive_gate.get("maximum_absolute_band_bias", np.nan),
    )
    technical_pass = bool(
        support.get("status") == "complete"
        and int(support.get("n_objects", 0)) > 0
        and int(support.get("n_joint_draws", 0)) > 0
        and all(np.isfinite(float(value)) for value in support_values)
        and int(predictive_gate.get("objects", 0)) > 0
        and int(predictive_gate.get("draws", 0)) > 0
        and all(np.isfinite(float(value)) for value in predictive_values)
        and support_truth_contract["status"] == "PASS"
        and predictive_truth_contract["status"] == "PASS"
    )
    passed = bool(
        support_gate.get("status") == "PASS"
        and support_tail["status"] == "PASS"
        and predictive_gate["status"] == "PASS"
        and support_truth_contract["status"] == "PASS"
        and predictive_truth_contract["status"] == "PASS"
        and np.isfinite(mean_log_evidence_is)
    )
    return {
        "label": label,
        "status": "PASS" if passed else "FAIL",
        "technical_status": "PASS" if technical_pass else "FAIL",
        "selection_corrected_exact_gaussian_iw_score": float(
            mean_log_evidence_is
        )
        - float(log_alpha),
        "exact_gaussian_ordinary_iw": {
            "status": support_gate.get("status"),
            "objects": int(support["n_objects"]),
            "draws": int(support["n_joint_draws"]),
            "median_raw_ess_fraction": float(support["median_raw_ess_fraction"]),
            "fraction_pareto_k_gt_0p7": float(support["fraction_pareto_k_gt_0p7"]),
            "fraction_pareto_k_gt_1": float(support["fraction_pareto_k_gt_1"]),
            "mean_log_evidence_is": mean_log_evidence_is,
            "tail_gate": support_tail,
        },
        "exact_gaussian_posterior_predictive": predictive_gate,
        "truth_free_inference_contract": {
            "support": support_truth_contract,
            "predictive": predictive_truth_contract,
        },
    }


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
    ema_importance: Path | None = None,
    ema_predictive: Path | None = None,
    out: Path,
    smoke: bool = False,
) -> dict[str, object]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate}")
    loaded_config = load_config(config)
    checkpoint_config = (
        (loaded_config.get("amortized", {}) or {})
        .get("sc_drws", {})
        .get("checkpoint", {})
    )
    training = _training_metrics(
        train,
        maximum_entropy_drop=float(
            checkpoint_config.get("maximum_entropy_drop", 3.0)
        ),
        maximum_unresolved_fraction=float(
            ((loaded_config.get("amortized", {}) or {}).get("sc_drws", {}) or {})
            .get("hard_mis", {})
            .get("maximum_unresolved_fraction", 0.02)
        ),
    )
    variants = [
        _variant_metrics(
            label="raw",
            importance=importance,
            predictive=predictive,
            out=out,
            log_alpha=float(training["selection_log_alpha"]),
        )
    ]
    if ema_importance is not None and ema_predictive is not None:
        variants.append(
            _variant_metrics(
                label="ema",
                importance=ema_importance,
                predictive=ema_predictive,
                out=out,
                log_alpha=float(training["selection_log_alpha"]),
            )
        )
    eligible = [variant for variant in variants if variant["status"] == "PASS"]
    eligible.sort(
        key=lambda value: -value["selection_corrected_exact_gaussian_iw_score"]
    )
    selected = eligible[0] if eligible else None
    training_pass = training["status"] == "PASS"
    scientific_pass = bool(training_pass and selected is not None)
    smoke_technical_pass = bool(
        training.get("technical_smoke_status") == "PASS"
        and all(variant["technical_status"] == "PASS" for variant in variants)
    )
    if smoke:
        status = "SMOKE_PASS" if smoke_technical_pass else "FAIL"
    else:
        status = "PASS" if scientific_pass else "FAIL"
    payload = {
        "status": status,
        "phase": phase,
        "candidate": candidate,
        "seed": int(seed),
        "config": str(config),
        "checkpoint_variant": selected["label"] if selected else None,
        "checkpoint": (
            training[f"{selected['label']}_model_checkpoint"] if selected else None
        ),
        "feature_stats": training["feature_stats"],
        "training": training,
        "variants": variants,
        "smoke_contract": {
            "status": "PASS" if smoke_technical_pass else "FAIL",
            "purpose": (
                "Runtime, numerical, hard-MIS, prior-gate, raw/EMA evaluation, "
                "and no-truth contract only. Scientific support and PPC "
                "thresholds remain promotion gates for the 512-object pilot."
            ),
            "scientific_thresholds_are_diagnostic_only": bool(smoke),
        },
        "exact_gaussian_ordinary_iw": (
            selected["exact_gaussian_ordinary_iw"] if selected else variants[0]["exact_gaussian_ordinary_iw"]
        ),
        "exact_gaussian_posterior_predictive": (
            selected["exact_gaussian_posterior_predictive"] if selected else variants[0]["exact_gaussian_posterior_predictive"]
        ),
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
        "ready_for_smc_diversity_benchmark": False,
        "ready_for_population_prior_update": bool(passed),
        "ready_for_full_catalogue": bool(passed),
        "truth_used_for_training_or_checkpoint_selection": False,
        "next_action": (
            "LAUNCH_SELECTED_ARCHITECTURE_TWO_SEED_FULL_DATASET_SC_DRWS"
            if passed
            else "STOP_AND_REVIEW_SC_DRWS_SUPPORT_OR_PPC_DIAGNOSTICS"
        ),
        "runs": runs,
    }
    name = "RWS_RECOVERY_PASS.json" if passed else "RWS_RECOVERY_FAIL.json"
    (root / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def finalize_full(root: Path) -> dict[str, object]:
    promotion = _read_json(root / "RWS_RECOVERY_PASS.json")
    selected = str(promotion["selected_candidate"])
    runs = [
        _read_json(root / "full" / selected / f"seed_{seed}" / "full_summary.json")
        for seed in SEEDS
    ]
    passed = all(run["status"] == "PASS" for run in runs)
    eligible = [run for run in runs if run["status"] == "PASS"]
    eligible.sort(
        key=lambda run: -max(
            variant["selection_corrected_exact_gaussian_iw_score"]
            for variant in run["variants"]
            if variant["status"] == "PASS"
        )
    )
    chosen = eligible[0] if eligible else None
    payload = {
        "status": "PASS" if passed else "FAIL",
        "selected_candidate": selected,
        "selected_seed": int(chosen["seed"]) if chosen else None,
        "selected_checkpoint": chosen["checkpoint"] if chosen else None,
        "selected_feature_stats": chosen["feature_stats"] if chosen else None,
        "both_final_seeds_passed": bool(passed),
        "ready_for_four_shard_catalogue_inference": bool(passed),
        "truth_used_for_training_or_checkpoint_selection": False,
        "runs": runs,
    }
    name = "FULL_TRAIN_PASS.json" if passed else "FULL_TRAIN_FAIL.json"
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
    summarize.add_argument(
        "--phase", choices=("pilot", "confirmation", "full"), required=True
    )
    summarize.add_argument("--config", type=Path, required=True)
    summarize.add_argument("--train", type=Path, required=True)
    summarize.add_argument("--importance", type=Path, required=True)
    summarize.add_argument("--predictive", type=Path, required=True)
    summarize.add_argument("--ema-importance", type=Path)
    summarize.add_argument("--ema-predictive", type=Path)
    summarize.add_argument("--out", type=Path, required=True)
    summarize.add_argument("--smoke", action="store_true")
    pilot = sub.add_parser("select-pilot")
    pilot.add_argument("--root", type=Path, required=True)
    final = sub.add_parser("finalize-confirmation")
    final.add_argument("--root", type=Path, required=True)
    full = sub.add_parser("finalize-full")
    full.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    values = vars(args)
    command = values.pop("command")
    if command == "summarize":
        payload = summarize_candidate(**values)
    elif command == "select-pilot":
        payload = select_pilot(**values)
    elif command == "finalize-confirmation":
        payload = finalize_confirmation(**values)
    else:
        payload = finalize_full(**values)
    print(json.dumps(payload, indent=2), flush=True)
    if command != "summarize" and payload["status"] == "FAIL":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
