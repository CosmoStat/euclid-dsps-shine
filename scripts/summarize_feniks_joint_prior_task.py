#!/usr/bin/env python3
"""Summarize the independent posterior/prior experiment matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from summarize_feniks_conditional_posterior_task import (
    KEY_PLOTS,
    _first_row,
    _float,
    _read_json,
    posterior_coverage,
)

LABELS = (
    "ind_frozen_rqspline",
    "ind_joint",
    "ind_vem1",
    "ind_vem4",
    "ind_vem4_hybrid",
    "ind_vem4_oracle",
)

DESCRIPTIONS = {
    "ind_frozen_rqspline": "AVI; independent RealNVP q; frozen pretrained RQ-spline p",
    "ind_joint": "AVI; independent RealNVP q and RealNVP p updated simultaneously",
    "ind_vem1": "AVI; variational EM with 1 q epoch then 1 prior M-step",
    "ind_vem4": "AVI; variational EM with 4 q epochs then 1 prior M-step",
    "ind_vem4_hybrid": "VEM 4:1; ELBO + 50 supervised NPE loss on q",
    "ind_vem4_oracle": "VEM 4:1 hybrid; plus truth NLL on p (synthetic oracle)",
}


def main() -> None:
    parser = argparse.ArgumentParser()
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
        parser.error("task mode requires --label, --config, --train, --inference, --out")
    summarize_task(args.label, args.config, args.train, args.inference, args.out)


def summarize_task(
    label: str, config: Path, train: Path, inference: Path, out: Path
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    training = _read_json(train / "training_summary.json")
    diagnostics = _read_json(inference / "posterior_diagnostics_summary.json")
    gate = _read_json(inference / "collapse_gate.json")
    log = pd.read_csv(train / "training_log.csv")
    phase_rows = log.loc[log.get("split", "train") == "train"].copy()
    phase_by_epoch = phase_rows.groupby("epoch").update_phase.first()
    encoder_epochs = int((phase_by_epoch != "prior").sum())
    prior_epochs = int((phase_by_epoch == "prior").sum())
    coverage = posterior_coverage(inference)
    coverage_path = out.parent / "posterior_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    photoz = _first_row(inference / "photoz_metrics.csv")
    missing_plots = [name for name in KEY_PLOTS if not (inference / name).is_file()]
    if missing_plots:
        raise FileNotFoundError("Missing required plots: " + ", ".join(missing_plots))
    elapsed = float(training.get("elapsed_time_s", np.nan))
    payload = {
        "label": label,
        "description": DESCRIPTIONS[label],
        "config": str(config),
        "elapsed_time_s": elapsed,
        "encoder_epochs": encoder_epochs,
        "prior_epochs": prior_epochs,
        "seconds_per_encoder_epoch": elapsed / encoder_epochs if encoder_epochs else None,
        "best_checkpoint_epoch": training.get("best_checkpoint_epoch"),
        "best_validation_metric": training.get("best_loss"),
        "updates_skipped": int(training.get("updates_skipped", 0)),
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
        "n_objects": int(diagnostics.get("n_objects", 0)),
        "coverage_csv": str(coverage_path),
        "key_plots": [str(inference / name) for name in KEY_PLOTS],
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
    write_plots(metrics, coverage, out)
    write_report(metrics, out / "README.md")
    print(f"[aggregate] complete: {out}")


def write_plots(metrics: pd.DataFrame, coverage: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    stable = (metrics.updates_skipped == 0) & (metrics.collapse_gate_failures == 0)
    colors = np.where(stable, "#177245", "#b33a3a")
    ax.scatter(metrics.seconds_per_encoder_epoch, metrics.coverage_error, c=colors, s=70)
    for row in metrics.itertuples():
        ax.annotate(row.label, (row.seconds_per_encoder_epoch, row.coverage_error), fontsize=8)
    ax.set_xlabel("wall seconds per encoder epoch (includes prior M-steps)")
    ax.set_ylabel("median absolute coverage error (68% + 95%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "speed_vs_coverage.png", dpi=180)
    plt.close(fig)

    parameters = list(dict.fromkeys(coverage.parameter.tolist()))
    x = np.arange(len(parameters))
    fig, axes = plt.subplots(2, 1, figsize=(max(11, 0.7 * len(parameters)), 8), sharex=True)
    for label, group in coverage.groupby("label", sort=False):
        aligned = group.set_index("parameter").reindex(parameters)
        axes[0].plot(x, aligned.coverage_68, marker="o", ms=3, label=label)
        axes[1].plot(x, aligned.coverage_95, marker="o", ms=3, label=label)
    axes[0].axhline(0.68, color="black", ls="--", lw=1)
    axes[1].axhline(0.95, color="black", ls="--", lw=1)
    axes[0].set_ylabel("coverage 68%")
    axes[1].set_ylabel("coverage 95%")
    axes[1].set_xticks(x, parameters, rotation=65, ha="right")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "coverage_by_parameter.png", dpi=180)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, path: Path) -> None:
    columns = (
        "label",
        "encoder_epochs",
        "prior_epochs",
        "seconds_per_encoder_epoch",
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
    lines = [
        "# Independent posterior and learned-prior matrix",
        "",
        "No winner is selected automatically: the oracle job uses synthetic truth and must not be compared as a deployable real-data method.",
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
                f"- [Full truth/prior/posterior corner](../{label}/inference/{KEY_PLOTS[0]})",
                f"- [Normalized photometric residuals](../{label}/inference/{KEY_PLOTS[1]})",
                f"- [Photometric residuals by band](../{label}/inference/{KEY_PLOTS[2]})",
                f"- [Worst photometric fits](../{label}/inference/{KEY_PLOTS[3]})",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
