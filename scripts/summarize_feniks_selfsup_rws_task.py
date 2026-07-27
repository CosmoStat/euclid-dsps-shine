#!/usr/bin/env python3
"""Summarize the self-supervised RWS experiment array."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from summarize_feniks_conditional_posterior_task import (
    _first_row,
    _float,
    _read_json,
    posterior_coverage,
)
from summarize_feniks_mode_covering_task import (
    MODE_KEY_PLOTS,
    _series_mean,
    _write_photometry_fit_examples,
    _write_plots,
)

LABELS = (
    "fixed_ref_rws_k4_gaussian",
    "selfsup_rws_k4_weighted_prior",
    "selfsup_rws_sleep3_wake1_k4",
    "selfsup_rws_sleep3_wake1_k8",
)

DESCRIPTIONS = {
    "fixed_ref_rws_k4_gaussian": (
        "Frozen reference prior; model-generated sleep 3:1; real-data RWS wake K=4"
    ),
    "selfsup_rws_k4_weighted_prior": (
        "Identity-initialized learned RealNVP prior; every epoch is shared RWS wake K=4"
    ),
    "selfsup_rws_sleep3_wake1_k4": (
        "Learned RealNVP prior; three physical sleep epochs then one shared RWS wake K=4"
    ),
    "selfsup_rws_sleep3_wake1_k8": (
        "Learned RealNVP prior; three physical sleep epochs then one shared RWS wake K=8"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--inference", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--aggregate-root", type=Path)
    parser.add_argument("--expected", type=int, default=len(LABELS))
    args = parser.parse_args()
    if args.aggregate_root is not None:
        aggregate(args.aggregate_root, args.expected)
        return
    required = (args.label, args.config, args.train, args.inference, args.out)
    if any(value is None for value in required):
        parser.error(
            "task mode requires --label, --config, --train, --inference, --out"
        )
    summarize_task(args.label, args.config, args.train, args.inference, args.out)


def summarize_task(
    label: str, config: Path, train: Path, inference: Path, out: Path
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    training = _read_json(train / "training_summary.json")
    latent = _read_json(train / "effective_latent_spec.json")
    diagnostics = _read_json(inference / "posterior_diagnostics_summary.json")
    gate = _read_json(inference / "collapse_gate.json")
    log = pd.read_csv(train / "training_log.csv")
    rows = log.loc[log.get("split", "train") == "train"].copy()
    phase_by_epoch = rows.groupby("epoch").update_phase.first()
    wake_rows = rows.loc[rows.update_phase.isin(("encoder_wake", "joint_wake"))]
    sleep_rows = rows.loc[rows.update_phase == "encoder_sleep"]
    prior_rows = rows.loc[rows.update_phase == "joint_wake"]
    coverage = posterior_coverage(inference)
    coverage_path = out.parent / "posterior_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    photoz = _first_row(inference / "photoz_metrics.csv")
    prior_population = (
        pd.read_csv(inference / "prior_vs_truth_population.csv")
        if (inference / "prior_vs_truth_population.csv").is_file()
        else pd.DataFrame()
    )
    prior_correlations = (
        pd.read_csv(inference / "prior_vs_truth_correlations.csv")
        if (inference / "prior_vs_truth_correlations.csv").is_file()
        else pd.DataFrame()
    )
    _write_photometry_fit_examples(inference)
    missing_plots = [
        name for name in MODE_KEY_PLOTS if not (inference / name).is_file()
    ]
    if missing_plots:
        raise FileNotFoundError("Missing required plots: " + ", ".join(missing_plots))
    elapsed = float(training.get("elapsed_time_s", np.nan))
    payload = {
        "label": label,
        "description": DESCRIPTIONS[label],
        "config": str(config),
        "elapsed_time_s": elapsed,
        "encoder_epochs": int(len(phase_by_epoch)),
        "sleep_epochs": int(sleep_rows.epoch.nunique()),
        "wake_epochs": int(wake_rows.epoch.nunique()),
        "prior_update_epochs": int(prior_rows.epoch.nunique()),
        "seconds_per_encoder_epoch": elapsed / len(phase_by_epoch),
        "best_checkpoint_epoch": training.get("best_checkpoint_epoch"),
        "updates_skipped": int(training.get("updates_skipped", 0)),
        "wake_updates": int(
            wake_rows.get("update_applied", pd.Series(dtype=float)).sum()
        ),
        "sleep_updates": int(
            sleep_rows.get("update_applied", pd.Series(dtype=float)).sum()
        ),
        "wake_ess_fraction_mean": _series_mean(wake_rows, "wake_ess_fraction_mean"),
        "wake_weight_max_mean": _series_mean(wake_rows, "wake_weight_max_mean"),
        "wake_all_nonfinite_fraction": _series_mean(
            wake_rows, "wake_all_nonfinite_fraction"
        ),
        "wake_physical_valid_fraction": _series_mean(
            wake_rows, "wake_physical_valid_fraction"
        ),
        "sleep_physical_valid_fraction": _series_mean(
            sleep_rows, "sleep_physical_valid_fraction"
        ),
        "sleep_noise_abs_median": _series_mean(sleep_rows, "sleep_noise_abs_median"),
        "sleep_noise_abs_q90": _series_mean(sleep_rows, "sleep_noise_abs_q90"),
        "posterior_mixture_entropy": _series_mean(rows, "posterior_mixture_entropy"),
        "posterior_mixture_max_weight": _series_mean(
            rows, "posterior_mixture_max_weight"
        ),
        "smc_stage_ess_mean": _series_mean(wake_rows, "smc_stage_ess_mean"),
        "smc_mala_acceptance_mean": _series_mean(wake_rows, "smc_mala_acceptance_mean"),
        "prior_mstep_nll": _series_mean(prior_rows, "prior_mstep_nll"),
        "normalization": latent.get("normalization"),
        "normalization_hash": latent.get("normalization_hash"),
        "normalization_checkpoint": latent.get("normalization_checkpoint"),
        "prior_source": latent.get("prior_source"),
        "collapse_gate_status": gate.get("status"),
        "collapse_gate_failures": int(gate.get("n_fail", 0)),
        "coverage_68_median": float(coverage.coverage_68.median()),
        "coverage_95_median": float(coverage.coverage_95.median()),
        "coverage_error": float(
            np.median(np.abs(coverage.coverage_68 - 0.68))
            + np.median(np.abs(coverage.coverage_95 - 0.95))
        ),
        "median_parameter_rmse": float(coverage.rmse.median()),
        "photoz_rmse": _float(photoz.get("rmse")),
        "photoz_bias": _float(photoz.get("median_bias")),
        "photoz_coverage_68": _float(photoz.get("coverage_68")),
        "median_posterior_predictive_chi2": _float(
            diagnostics.get("median_posterior_predictive_chi2")
        ),
        "prior_15d_mean_quantile_l1_iqr": _series_mean(
            prior_population, "quantile_l1_iqr"
        ),
        "prior_15d_mean_spearman_abs_delta": _series_mean(
            prior_correlations, "abs_delta"
        ),
        "n_objects": int(diagnostics.get("n_objects", 0)),
        "coverage_csv": str(coverage_path),
        "key_plots": [str(inference / name) for name in MODE_KEY_PLOTS],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def aggregate(root: Path, expected: int) -> None:
    complete = [label for label in LABELS if (root / label / "DONE").is_file()]
    if len(complete) != expected:
        print(f"[aggregate] waiting: complete={len(complete)} expected={expected}")
        return
    out = root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(
        [_read_json(root / label / "metrics.json") for label in LABELS]
    )
    metrics.to_csv(out / "experiment_metrics.csv", index=False)
    coverage_parts = []
    for label in LABELS:
        frame = pd.read_csv(root / label / "posterior_coverage.csv")
        frame.insert(0, "label", label)
        coverage_parts.append(frame)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    coverage.to_csv(out / "coverage_by_parameter.csv", index=False)
    _write_plots(metrics, coverage, out)
    _write_report(metrics, out / "README.md")
    print(f"[aggregate] complete: {out}")


def _write_report(metrics: pd.DataFrame, path: Path) -> None:
    columns = (
        "label",
        "sleep_epochs",
        "wake_epochs",
        "prior_update_epochs",
        "seconds_per_encoder_epoch",
        "wake_ess_fraction_mean",
        "wake_physical_valid_fraction",
        "coverage_68_median",
        "coverage_95_median",
        "median_parameter_rmse",
        "photoz_rmse",
        "median_posterior_predictive_chi2",
        "collapse_gate_status",
    )
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in metrics.loc[:, columns].itertuples(index=False, name=None):
        table.append("| " + " | ".join(str(value) for value in row) + " |")
    hashes = sorted(set(metrics.normalization_hash.dropna().astype(str)))
    lines = [
        "# Self-supervised RWS prior and posterior matrix",
        "",
        "Truth columns are excluded from training and used only for held-out closure diagnostics.",
        "The synthetic likelihood is Gaussian with the catalog flux errors and no extra floor.",
        f"Common normalization hashes: {', '.join(hashes)}",
        "",
        *table,
        "",
        "## Experiment definitions",
        "",
        *[f"- **{label}**: {DESCRIPTIONS[label]}" for label in LABELS],
        "",
        "## Photometry fits and corner plots",
        "",
    ]
    for label in LABELS:
        lines.extend(
            [
                f"### {label}",
                "",
                f"- [Full corner](../{label}/inference/{MODE_KEY_PLOTS[0]})",
                f"- [Normalized residuals](../{label}/inference/{MODE_KEY_PLOTS[1]})",
                f"- [Residuals by band](../{label}/inference/{MODE_KEY_PLOTS[2]})",
                f"- [Worst-fit chi2](../{label}/inference/{MODE_KEY_PLOTS[3]})",
                f"- [Worst photometry fit panels](../{label}/inference/{MODE_KEY_PLOTS[4]})",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
