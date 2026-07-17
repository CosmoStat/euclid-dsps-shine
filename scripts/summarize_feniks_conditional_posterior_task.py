#!/usr/bin/env python3
"""Summarize and aggregate the conditional-posterior experiment matrix."""

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

LABELS = (
    "avi_gaussian_x",
    "avi_gaussian_u",
    "avi_realnvp",
    "avi_rqspline",
    "npe_gaussian_x",
    "npe_gaussian_u",
    "npe_realnvp",
    "npe_rqspline",
)
KEY_PLOTS = (
    "corner_full_latent_truth_prior_posterior.png",
    "posterior_predictive_normalized_residual_hist.png",
    "posterior_predictive_residuals_by_band.png",
    "top_posterior_predictive_chi2.png",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--inference", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--aggregate-root", type=Path)
    parser.add_argument("--expected", type=int, default=8)
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
    diagnostics = _read_json(inference / "posterior_diagnostics_summary.json")
    coverage = posterior_coverage(inference)
    coverage_path = out.parent / "posterior_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    photoz = _first_row(inference / "photoz_metrics.csv")
    missing_plots = [name for name in KEY_PLOTS if not (inference / name).is_file()]
    if missing_plots:
        raise FileNotFoundError(
            "Missing required inference plots: " + ", ".join(missing_plots)
        )
    elapsed = float(training.get("elapsed_time_s", np.nan))
    epochs = int(training.get("epochs", 0))
    payload = {
        "label": label,
        "config": str(config),
        "objective": "npe" if label.startswith("npe_") else "avi",
        "posterior_family": label.split("_", 1)[1],
        "elapsed_time_s": elapsed,
        "seconds_per_epoch": elapsed / epochs if epochs else None,
        "best_checkpoint_epoch": training.get("best_checkpoint_epoch"),
        "best_validation_metric": training.get("best_loss"),
        "updates_skipped": int(training.get("updates_skipped", 0)),
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


def posterior_coverage(inference: Path) -> pd.DataFrame:
    summary = pd.read_parquet(inference / "posterior_summary.parquet")
    truth = pd.read_parquet(inference / "inference_truth.parquet")
    joined = truth.merge(
        summary, on="object_id", suffixes=("_truth", ""), validate="one_to_one"
    )
    tail_quantiles = _posterior_tail_quantiles(inference, summary)
    rows = []
    for column in summary.columns:
        if not column.endswith("_median"):
            continue
        name = column[: -len("_median")]
        truth_name = name if name in joined else f"{name}_truth"
        required = (f"{name}_q16", f"{name}_q84")
        if truth_name not in joined or any(item not in joined for item in required):
            continue
        target = joined[truth_name].to_numpy(float)
        median = joined[column].to_numpy(float)
        finite = np.isfinite(target) & np.isfinite(median)
        if not finite.any():
            continue
        target, median = target[finite], median[finite]
        q16 = joined[required[0]].to_numpy(float)[finite]
        q84 = joined[required[1]].to_numpy(float)[finite]
        tails = tail_quantiles.get(name)
        if tails is None:
            q025, q975 = q16, q84
        else:
            aligned = joined[["object_id"]].merge(
                tails, on="object_id", how="left", validate="one_to_one"
            )
            q025 = aligned["q025"].to_numpy(float)[finite]
            q975 = aligned["q975"].to_numpy(float)[finite]
        rows.append(
            {
                "parameter": name,
                "n_objects": int(len(target)),
                "bias": float(np.mean(median - target)),
                "rmse": float(np.sqrt(np.mean((median - target) ** 2))),
                "coverage_68": float(np.mean((target >= q16) & (target <= q84))),
                "coverage_95": float(np.mean((target >= q025) & (target <= q975))),
                "width_68_median": float(np.median(q84 - q16)),
            }
        )
    if not rows:
        raise ValueError(f"No truth/posterior parameter pairs found in {inference}")
    return pd.DataFrame(rows)


def _posterior_tail_quantiles(
    inference: Path, summary: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    shard_paths = sorted((inference / "posterior_samples").glob("batch_*.parquet"))
    if not shard_paths:
        return {}
    names = [
        column[: -len("_median")] for column in summary if column.endswith("_median")
    ]
    available = set(pd.read_parquet(shard_paths[0]).columns)
    names = [name for name in names if name in available]
    if not names:
        return {}
    samples = pd.concat(
        [pd.read_parquet(path, columns=["object_id", *names]) for path in shard_paths],
        ignore_index=True,
    )
    result = {}
    for name in names:
        quantiles = samples.groupby("object_id", sort=False)[name].quantile(
            [0.025, 0.975]
        )
        frame = quantiles.unstack().reset_index()
        frame.columns = ["object_id", "q025", "q975"]
        result[name] = frame
    return result


def aggregate(root: Path, expected: int) -> None:
    complete = [label for label in LABELS if (root / label / "DONE").is_file()]
    if len(complete) != expected:
        print(
            f"[aggregate] waiting for tasks: complete={len(complete)} expected={expected}"
        )
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
    write_comparison_plots(metrics, coverage, out)
    selected = select_model(metrics)
    (out / "selection.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(root, metrics, selected, out / "README.md")
    print(f"[aggregate] complete: {out}")


def write_comparison_plots(
    metrics: pd.DataFrame, coverage: pd.DataFrame, out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for objective, group in metrics.groupby("objective"):
        ax.scatter(
            group.seconds_per_epoch, group.coverage_error, s=65, label=objective.upper()
        )
        for row in group.itertuples():
            ax.annotate(
                row.posterior_family,
                (row.seconds_per_epoch, row.coverage_error),
                fontsize=8,
            )
    ax.set_xlabel("seconds per epoch")
    ax.set_ylabel("median |coverage68-0.68| + |coverage95-0.95|")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "speed_vs_coverage.png", dpi=180)
    plt.close(fig)

    parameters = list(dict.fromkeys(coverage.parameter.tolist()))
    x = np.arange(len(parameters))
    fig, axes = plt.subplots(
        2, 1, figsize=(max(11, len(parameters) * 0.7), 8), sharex=True
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
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "coverage_by_parameter.png", dpi=180)
    plt.close(fig)


def select_model(metrics: pd.DataFrame) -> dict:
    baseline = metrics.loc[metrics.label == "avi_gaussian_x"].iloc[0]
    eligible = metrics[
        (metrics.updates_skipped == 0)
        & np.isfinite(metrics.coverage_error)
        & (metrics.seconds_per_epoch <= 1.2 * baseline.seconds_per_epoch)
    ].copy()
    if eligible.empty:
        eligible = metrics[np.isfinite(metrics.coverage_error)].copy()
    winner = eligible.sort_values(["coverage_error", "median_parameter_rmse"]).iloc[0]
    return {
        "selected_label": str(winner.label),
        "selection_rule": "minimum coverage error, then median parameter RMSE, under 20% AVI-baseline epoch slowdown",
        "baseline_seconds_per_epoch": float(baseline.seconds_per_epoch),
        "selected_seconds_per_epoch": float(winner.seconds_per_epoch),
        "selected_coverage_error": float(winner.coverage_error),
    }


def write_report(root: Path, metrics: pd.DataFrame, selected: dict, path: Path) -> None:
    columns = (
        "label",
        "seconds_per_epoch",
        "coverage_68_median",
        "coverage_95_median",
        "median_parameter_rmse",
        "photoz_rmse",
        "median_posterior_predictive_chi2",
    )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    table = [header, separator]
    for row in metrics.loc[:, columns].itertuples(index=False, name=None):
        table.append("| " + " | ".join(str(value) for value in row) + " |")
    lines = [
        "# FENIKS conditional posterior matrix",
        "",
        f"Selected model: **{selected['selected_label']}**.",
        "",
        "## Summary",
        "",
        *table,
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_row(path: Path) -> dict:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    return {} if frame.empty else frame.iloc[0].to_dict()


def _float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


if __name__ == "__main__":
    main()
