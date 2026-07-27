#!/usr/bin/env python3
"""Summarize the common-15D and mode-covering experiment array."""

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
    "common15d_vem4_elbo_k1",
    "frozen_ref_elbo_k2_antithetic",
    "frozen_ref_periodic_wake_k4",
    "common15d_vem4_periodic_wake_k4",
)

PHOTOMETRY_FIT_PLOT = "posterior_predictive_fit_examples.png"
MODE_KEY_PLOTS = (*KEY_PLOTS, PHOTOMETRY_FIT_PLOT)

DESCRIPTIONS = {
    "common15d_vem4_elbo_k1": (
        "Common 15D transform; learned RealNVP prior; VEM 4:1; reverse-KL K=1"
    ),
    "frozen_ref_elbo_k2_antithetic": (
        "Common 15D transform; frozen reference RealNVP prior; antithetic ELBO K=2"
    ),
    "frozen_ref_periodic_wake_k4": (
        "Frozen reference RealNVP prior; ELBO plus periodic tempered wake K=4"
    ),
    "common15d_vem4_periodic_wake_k4": (
        "Common 15D transform; learned RealNVP prior; VEM 4:1; periodic wake K=4"
    ),
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
    phase_rows = log.loc[log.get("split", "train") == "train"].copy()
    phase_by_epoch = phase_rows.groupby("epoch").update_phase.first()
    encoder_epochs = int((phase_by_epoch != "prior").sum())
    prior_epochs = int((phase_by_epoch == "prior").sum())
    wake_rows = phase_rows.loc[phase_rows.update_phase == "encoder_wake"]
    coverage = posterior_coverage(inference)
    coverage_path = out.parent / "posterior_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    photoz = _first_row(inference / "photoz_metrics.csv")
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
        "encoder_epochs": encoder_epochs,
        "prior_epochs": prior_epochs,
        "seconds_per_encoder_epoch": (
            elapsed / encoder_epochs if encoder_epochs else None
        ),
        "best_checkpoint_epoch": training.get("best_checkpoint_epoch"),
        "updates_skipped": int(training.get("updates_skipped", 0)),
        "wake_epochs": int(wake_rows.epoch.nunique()),
        "wake_updates": int(
            wake_rows.get("update_applied", pd.Series(dtype=float)).sum()
        ),
        "wake_ess_fraction_mean": _series_mean(wake_rows, "wake_ess_fraction_mean"),
        "wake_weight_max_mean": _series_mean(wake_rows, "wake_weight_max_mean"),
        "wake_all_nonfinite_fraction": _series_mean(
            wake_rows, "wake_all_nonfinite_fraction"
        ),
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
        "n_objects": int(diagnostics.get("n_objects", 0)),
        "coverage_csv": str(coverage_path),
        "key_plots": [str(inference / name) for name in MODE_KEY_PLOTS],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _series_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def _write_photometry_fit_examples(inference: Path, *, n_objects: int = 6) -> None:
    """Plot compact observed-versus-posterior photometry for the worst fits."""
    top_path = inference / "top_posterior_predictive_chi2.csv"
    residual_path = inference / "posterior_predictive_residual_summary.parquet"
    if not top_path.is_file() or not residual_path.is_file():
        raise FileNotFoundError(
            "Photometry fit examples require top chi2 and residual-summary tables"
        )
    top = pd.read_csv(top_path, dtype={"object_id": str}).head(int(n_objects))
    residual = pd.read_parquet(residual_path)
    required = {
        "object_id",
        "band",
        "obs_flux_fnu_cgs",
        "obs_err_fnu_cgs",
        "model_flux_q16",
        "model_flux_median",
        "model_flux_q84",
    }
    missing = sorted(required.difference(residual.columns))
    if missing:
        raise ValueError(f"Residual summary missing fit-plot columns: {missing}")
    if top.empty:
        raise ValueError("Cannot plot photometry fits from an empty top-chi2 table")

    object_ids = top["object_id"].tolist()
    n_cols = 2
    n_rows = int(np.ceil(len(object_ids) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8.5 * n_cols, 3.4 * n_rows),
        squeeze=False,
    )
    for axis, object_id in zip(axes.ravel(), object_ids, strict=False):
        rows = residual.loc[residual["object_id"].astype(str) == str(object_id)].copy()
        if rows.empty:
            raise ValueError(f"No residual summary rows for object_id={object_id}")
        x = np.arange(len(rows))
        obs = rows["obs_flux_fnu_cgs"].to_numpy(float) * 1.0e32
        obs_err = rows["obs_err_fnu_cgs"].to_numpy(float) * 1.0e32
        model = rows["model_flux_median"].to_numpy(float) * 1.0e32
        model_lo = rows["model_flux_q16"].to_numpy(float) * 1.0e32
        model_hi = rows["model_flux_q84"].to_numpy(float) * 1.0e32
        axis.errorbar(
            x,
            obs,
            yerr=obs_err,
            fmt="o",
            ms=4,
            capsize=2,
            color="black",
            label="observed",
        )
        axis.plot(x, model, "o-", ms=3, lw=1.1, color="tab:blue", label="posterior")
        axis.fill_between(x, model_lo, model_hi, color="tab:blue", alpha=0.2)
        axis.axhline(0.0, color="0.5", lw=0.7)
        axis.set_xticks(x, rows["band"].astype(str), rotation=50, ha="right")
        chi2_row = top.loc[top["object_id"] == object_id]
        chi2 = (
            float(chi2_row["posterior_predictive_chi2_median"].iloc[0])
            if "posterior_predictive_chi2_median" in chi2_row
            else float("nan")
        )
        axis.set_title(f"object {object_id}; median chi2={chi2:.2f}")
        axis.set_ylabel("flux density [nJy]")
        axis.grid(alpha=0.2)
    for axis in axes.ravel()[len(object_ids) :]:
        axis.axis("off")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Worst posterior-predictive photometry fits", y=0.998)
    fig.tight_layout()
    fig.savefig(inference / PHOTOMETRY_FIT_PLOT, dpi=180)
    plt.close(fig)


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


def _write_plots(metrics: pd.DataFrame, coverage: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(metrics.seconds_per_encoder_epoch, metrics.coverage_error, s=70)
    for row in metrics.itertuples():
        ax.annotate(
            row.label, (row.seconds_per_encoder_epoch, row.coverage_error), fontsize=8
        )
    ax.set_xlabel("wall seconds per encoder epoch (including prior M-steps)")
    ax.set_ylabel("median absolute coverage error (68% + 95%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "speed_vs_coverage.png", dpi=180)
    plt.close(fig)

    parameters = list(dict.fromkeys(coverage.parameter.tolist()))
    x = np.arange(len(parameters))
    fig, axes = plt.subplots(
        2, 1, figsize=(max(11, 0.7 * len(parameters)), 8), sharex=True
    )
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


def _write_report(metrics: pd.DataFrame, path: Path) -> None:
    columns = (
        "label",
        "encoder_epochs",
        "prior_epochs",
        "seconds_per_encoder_epoch",
        "wake_ess_fraction_mean",
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
        "# Common-15D and mode-covering posterior matrix",
        "",
        "Training is unsupervised: truth columns are used only after training for held-out diagnostics.",
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
