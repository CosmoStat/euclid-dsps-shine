#!/usr/bin/env python3
"""Evaluate paired PIT/rank and central coverage from dense redshift posteriors."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.amortized.mira import (
    _read_dense_posterior,
    _read_truth,
    parse_posterior_spec,
    resolve_posterior_input,
    resolve_truth_path,
)

DEFAULT_LEVELS = tuple(sorted({*np.linspace(0.05, 0.95, 19).tolist(), 0.68, 0.95}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--posterior", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--truth-column", default="redshift_true")
    parser.add_argument("--samples-per-object", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=260806)
    parser.add_argument("--expected-objects", type=int)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def finite_rank_pit(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return midpoint randomized-rank PIT values for finite posterior draws."""
    less = np.sum(samples < truth[:, None], axis=1)
    equal = np.sum(samples == truth[:, None], axis=1)
    return (less + 0.5 * equal + 0.5) / (samples.shape[1] + 1.0)


def uniform_ks(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n_values = len(ordered)
    upper = np.max(np.arange(1, n_values + 1) / n_values - ordered)
    lower = np.max(ordered - np.arange(0, n_values) / n_values)
    return float(max(upper, lower))


def calibration_arrays(
    samples: np.ndarray,
    truth: np.ndarray,
    levels: np.ndarray,
) -> dict[str, np.ndarray]:
    lower_q = (1.0 - levels) / 2.0
    upper_q = 1.0 - lower_q
    lower = np.quantile(samples, lower_q, axis=1).T
    upper = np.quantile(samples, upper_q, axis=1).T
    covered = (truth[:, None] >= lower) & (truth[:, None] <= upper)
    return {
        "pit": finite_rank_pit(samples, truth),
        "covered": covered,
        "width": upper - lower,
    }


def scalar_metrics(
    arrays: dict[str, np.ndarray], levels: np.ndarray, indices: np.ndarray | None = None
) -> dict[str, float]:
    if indices is None:
        pit = arrays["pit"]
        covered = arrays["covered"]
        width = arrays["width"]
    else:
        pit = arrays["pit"][indices]
        covered = arrays["covered"][indices]
        width = arrays["width"][indices]
    empirical = covered.mean(axis=0)
    index68 = int(np.flatnonzero(np.isclose(levels, 0.68))[0])
    index95 = int(np.flatnonzero(np.isclose(levels, 0.95))[0])
    return {
        "pit_mean": float(np.mean(pit)),
        "pit_variance": float(np.var(pit)),
        "pit_ks_uniform": uniform_ks(pit),
        "coverage_ece": float(np.mean(np.abs(empirical - levels))),
        "coverage_68": float(empirical[index68]),
        "coverage_95": float(empirical[index95]),
        "median_width_68": float(np.median(width[:, index68])),
        "median_width_95": float(np.median(width[:, index95])),
    }


def _bootstrap_metrics(
    model_arrays: dict[str, dict[str, np.ndarray]],
    levels: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_bootstrap <= 0:
        return pd.DataFrame(), pd.DataFrame()
    names = tuple(model_arrays)
    n_objects = len(next(iter(model_arrays.values()))["pit"])
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    by_draw: dict[tuple[int, str], dict[str, float]] = {}
    for draw in range(n_bootstrap):
        indices = rng.integers(0, n_objects, size=n_objects)
        for name in names:
            metrics = scalar_metrics(model_arrays[name], levels, indices)
            by_draw[(draw, name)] = metrics
            rows.extend(
                {
                    "bootstrap": draw,
                    "model": name,
                    "metric": metric,
                    "value": value,
                }
                for metric, value in metrics.items()
            )
    intervals = []
    frame = pd.DataFrame(rows)
    for name in names:
        for metric in frame["metric"].unique():
            values = frame.loc[
                (frame["model"] == name) & (frame["metric"] == metric), "value"
            ].to_numpy(float)
            low, high = np.quantile(values, [0.025, 0.975])
            intervals.append(
                {
                    "left_model": name,
                    "right_model": "",
                    "metric": metric,
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                }
            )
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for metric in frame["metric"].unique():
                delta = np.asarray(
                    [
                        by_draw[(draw, left)][metric] - by_draw[(draw, right)][metric]
                        for draw in range(n_bootstrap)
                    ]
                )
                low, high = np.quantile(delta, [0.025, 0.975])
                intervals.append(
                    {
                        "left_model": left,
                        "right_model": right,
                        "metric": metric,
                        "ci95_low": float(low),
                        "ci95_high": float(high),
                    }
                )
    return frame, pd.DataFrame(intervals)


def _write_plot(
    pit: pd.DataFrame,
    coverage: pd.DataFrame,
    summary: pd.DataFrame,
    out: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"rws26": "#0072B2", "rws24": "#D55E00", "popcosmos": "#222222"}
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)
    bins = np.linspace(0.0, 1.0, 21)
    for model, frame in pit.groupby("model", sort=False):
        axes[0].hist(
            frame["pit"],
            bins=bins,
            histtype="step",
            density=True,
            lw=1.8,
            label=model,
            color=colors.get(model),
        )
    axes[0].axhline(1.0, color="0.4", ls="--", lw=1.0)
    axes[0].set(xlabel="finite-sample rank PIT", ylabel="density", title="PIT / rank")
    axes[0].legend(frameon=False)

    for model, frame in coverage.groupby("model", sort=False):
        axes[1].plot(
            frame["nominal_coverage"],
            frame["empirical_coverage"],
            marker="o",
            ms=3,
            label=model,
            color=colors.get(model),
        )
    axes[1].plot([0, 1], [0, 1], color="0.4", ls="--", lw=1.0)
    axes[1].set(
        xlabel="nominal central coverage",
        ylabel="empirical coverage",
        title="Coverage calibration",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    x = np.arange(len(summary))
    axes[2].bar(
        x - 0.18,
        summary["coverage_68"],
        width=0.36,
        label="68%",
        color="#56B4E9",
    )
    axes[2].bar(
        x + 0.18,
        summary["coverage_95"],
        width=0.36,
        label="95%",
        color="#009E73",
    )
    axes[2].axhline(0.68, color="#56B4E9", ls="--", lw=1.0)
    axes[2].axhline(0.95, color="#009E73", ls="--", lw=1.0)
    axes[2].set_xticks(x, summary["model"], rotation=15)
    axes[2].set(ylabel="empirical coverage", title="Central intervals", ylim=(0, 1))
    axes[2].legend(frameon=False)
    fig.suptitle(
        f"Same-object redshift posterior calibration (N={pit.object_id.nunique():,})"
    )
    fig.savefig(out / "redshift_pit_coverage.png", dpi=220)
    fig.savefig(out / "redshift_pit_coverage.pdf")
    plt.close(fig)


def evaluate(
    *,
    truth_path: Path,
    posterior_specs: list[tuple[str, Path]],
    out: Path,
    truth_column: str,
    samples_per_object: int | None,
    bootstrap: int,
    seed: int,
    expected_objects: int | None,
) -> dict[str, Any]:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out}")
    truth_file = resolve_truth_path(truth_path)
    truth = _read_truth(
        truth_file,
        ("z_obs",),
        limit=None,
        truth_column_map={"z_obs": truth_column},
        drop_nonfinite=True,
    )
    if expected_objects is not None and len(truth) != expected_objects:
        raise ValueError(
            f"Expected {expected_objects} truth objects, found {len(truth)}"
        )
    inputs = [resolve_posterior_input(name, path) for name, path in posterior_specs]
    dense = [
        _read_dense_posterior(
            item,
            truth,
            ("z_obs",),
            samples_per_object=samples_per_object,
            require_exact_object_set=False,
        )
        for item in inputs
    ]
    if len({model.values.shape[1] for model in dense}) != 1:
        raise ValueError("All models must use the same posterior sample count")

    levels = np.asarray(DEFAULT_LEVELS, dtype=np.float64)
    z_true = truth["z_obs"].to_numpy(np.float64)
    model_arrays: dict[str, dict[str, np.ndarray]] = {}
    pit_rows = []
    coverage_rows = []
    summary_rows = []
    for model in dense:
        samples = model.values[:, :, 0].astype(np.float64)
        arrays = calibration_arrays(samples, z_true, levels)
        model_arrays[model.name] = arrays
        metrics = scalar_metrics(arrays, levels)
        summary_rows.append(
            {
                "model": model.name,
                "n_objects": len(truth),
                "samples_per_object": samples.shape[1],
                **metrics,
            }
        )
        pit_rows.extend(
            {
                "model": model.name,
                "object_id": object_id,
                "row_index": row_index,
                "redshift_true": truth_value,
                "pit": pit_value,
            }
            for object_id, row_index, truth_value, pit_value in zip(
                truth["object_id"],
                truth.get("row_index", pd.Series(range(len(truth)))),
                z_true,
                arrays["pit"],
                strict=True,
            )
        )
        empirical = arrays["covered"].mean(axis=0)
        coverage_rows.extend(
            {
                "model": model.name,
                "nominal_coverage": float(level),
                "empirical_coverage": float(value),
                "difference": float(value - level),
            }
            for level, value in zip(levels, empirical, strict=True)
        )

    out.mkdir(parents=True, exist_ok=False)
    pit_frame = pd.DataFrame(pit_rows)
    coverage_frame = pd.DataFrame(coverage_rows)
    summary_frame = pd.DataFrame(summary_rows)
    bootstrap_frame, intervals = _bootstrap_metrics(
        model_arrays, levels, n_bootstrap=bootstrap, seed=seed
    )
    pit_frame.to_parquet(out / "redshift_pit_values.parquet", index=False)
    coverage_frame.to_csv(out / "redshift_coverage_curve.csv", index=False)
    summary_frame.to_csv(out / "redshift_calibration_summary.csv", index=False)
    if not bootstrap_frame.empty:
        bootstrap_frame.to_parquet(
            out / "redshift_calibration_bootstrap.parquet", index=False
        )
        intervals.to_csv(
            out / "redshift_calibration_bootstrap_intervals.csv", index=False
        )
    _write_plot(pit_frame, coverage_frame, summary_frame, out)

    payload = {
        "status": "complete",
        "scope": "redshift_only_same_public_specz_objects",
        "n_objects": int(len(truth)),
        "samples_per_object": int(dense[0].values.shape[1]),
        "pit_contract": "(n_less + 0.5*n_equal + 0.5) / (n_samples + 1)",
        "coverage_contract": "central equal-tail empirical coverage",
        "bootstrap": {"paired_objects": True, "n_resamples": bootstrap, "seed": seed},
        "models": {row["model"]: row for row in summary_rows},
        "inputs": {
            "truth": {"path": str(truth_file), "sha256": _sha256(truth_file)},
            "posteriors": {
                item.name: [
                    {"path": str(path), "sha256": _sha256(path)} for path in item.files
                ]
                for item in inputs
            },
        },
        "git_commit": _git_sha(),
    }
    (out / "redshift_calibration_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out / "DONE").touch()
    return payload


def main() -> None:
    args = parse_args()
    samples = None if args.samples_per_object == 0 else args.samples_per_object
    summary = evaluate(
        truth_path=args.truth,
        posterior_specs=[parse_posterior_spec(value) for value in args.posterior],
        out=args.out,
        truth_column=args.truth_column,
        samples_per_object=samples,
        bootstrap=args.bootstrap,
        seed=args.seed,
        expected_objects=args.expected_objects,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[redshift-calibration] complete -> {args.out}")


if __name__ == "__main__":
    main()
