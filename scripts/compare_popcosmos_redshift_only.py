#!/usr/bin/env python3
"""Build the frozen same-object RWS/Pop-COSMOS redshift benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_COLUMNS = {
    "rws26": ("z_obs_q16", "z_obs_median", "z_obs_q84"),
    "rws24": ("z_obs_q16", "z_obs_median", "z_obs_q84"),
    "popcosmos": ("z_pc_160", "z_pc_500", "z_pc_840"),
}
METRIC_NAMES = (
    "median_bias",
    "nmad",
    "rmse",
    "outlier_fraction_0p15",
    "coverage_68",
    "median_interval_width_68",
)
PAIRWISE_COMPARISONS = (
    ("rws26", "rws24"),
    ("rws26", "popcosmos"),
    ("rws24", "popcosmos"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rws26", type=Path, required=True)
    parser.add_argument("--rws24", type=Path, required=True)
    parser.add_argument("--popcosmos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-evaluation", type=int, default=5_000)
    parser.add_argument("--expected-specz", type=int, default=1_395)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=260805)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(int(value)).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_rws(path: Path, method: str) -> pd.DataFrame:
    required = [
        "object_id",
        "row_index",
        "redshift_true",
        "z_obs_q16",
        "z_obs_median",
        "z_obs_q84",
    ]
    frame = pd.read_parquet(path, columns=required)
    if frame["object_id"].duplicated().any():
        raise ValueError(f"{method} contains duplicate object_id values")
    rename = {
        "redshift_true": f"redshift_true_{method}",
        "row_index": f"row_index_{method}",
        "z_obs_q16": f"{method}_q16",
        "z_obs_median": f"{method}_median",
        "z_obs_q84": f"{method}_q84",
    }
    return frame.rename(columns=rename)


def build_paired_table(
    rws26_path: Path,
    rws24_path: Path,
    popcosmos_path: Path,
) -> pd.DataFrame:
    """Join the two RWS variants and public Pop-COSMOS summaries by exact ID."""
    rws26 = _read_rws(rws26_path, "rws26")
    rws24 = _read_rws(rws24_path, "rws24")
    if not np.array_equal(rws26["object_id"], rws24["object_id"]):
        raise RuntimeError("RWS-26 and RWS-24 object ordering differs")
    if not np.array_equal(rws26["row_index_rws26"], rws24["row_index_rws24"]):
        raise RuntimeError("RWS-26 and RWS-24 row_index ordering differs")
    truth26 = rws26["redshift_true_rws26"].to_numpy(float)
    truth24 = rws24["redshift_true_rws24"].to_numpy(float)
    if not np.array_equal(truth26, truth24, equal_nan=True):
        raise RuntimeError("RWS variants do not carry identical spectroscopy")

    reference = pd.read_csv(
        popcosmos_path,
        sep=r"\s+",
        usecols=[
            "INDEX_COSMOS",
            "MAGCUT_r",
            "XRAY",
            "z_SPEC",
            "z_SPECSOURCE",
            "z_pc_160",
            "z_pc_500",
            "z_pc_840",
        ],
        na_values=["-99", "None"],
        low_memory=False,
    ).rename(columns={"INDEX_COSMOS": "object_id"})
    reference["object_id"] = pd.to_numeric(
        reference["object_id"], errors="raise"
    ).astype(np.int64)
    if reference["object_id"].duplicated().any():
        raise ValueError("Pop-COSMOS summary contains duplicate INDEX_COSMOS values")

    paired = rws26.merge(rws24, on="object_id", how="inner", validate="one_to_one")
    paired = paired.merge(reference, on="object_id", how="left", validate="one_to_one")
    if len(paired) != len(rws26):
        raise RuntimeError("Exact-ID join changed the RWS evaluation row count")
    missing = paired["z_pc_500"].isna()
    if missing.any():
        raise RuntimeError(
            f"Pop-COSMOS summaries are missing for {int(missing.sum())} RWS objects"
        )
    if not ((paired["MAGCUT_r"] == "Y") & (paired["XRAY"] == "N")).all():
        raise RuntimeError("Evaluation contains objects outside the A24 r<25 non-X-ray cut")

    paired["row_index"] = paired.pop("row_index_rws26")
    paired = paired.drop(columns=["row_index_rws24"])
    paired["redshift_true"] = paired.pop("redshift_true_rws26")
    paired = paired.drop(columns=["redshift_true_rws24"])
    paired = paired.rename(
        columns={
            "z_pc_160": "popcosmos_q16",
            "z_pc_500": "popcosmos_median",
            "z_pc_840": "popcosmos_q84",
            "z_SPEC": "popcosmos_specz_flag",
            "z_SPECSOURCE": "popcosmos_specz_source_flag",
        }
    )
    paired["has_public_specz"] = np.isfinite(
        paired["redshift_true"].to_numpy(float)
    ) & (paired["redshift_true"].to_numpy(float) >= 0.0)
    return paired


def redshift_metrics(frame: pd.DataFrame, method: str) -> dict[str, float | int]:
    truth = frame["redshift_true"].to_numpy(float)
    median = frame[f"{method}_median"].to_numpy(float)
    q16 = frame[f"{method}_q16"].to_numpy(float)
    q84 = frame[f"{method}_q84"].to_numpy(float)
    valid = np.isfinite(truth) & (truth >= 0.0) & np.isfinite(median)
    if not valid.any():
        raise ValueError(f"No finite spectroscopy for {method}")
    dz = (median[valid] - truth[valid]) / (1.0 + truth[valid])
    center = float(np.median(dz))
    interval = np.isfinite(q16[valid]) & np.isfinite(q84[valid])
    return {
        "n_spec": int(valid.sum()),
        "median_bias": center,
        "nmad": float(1.48 * np.median(np.abs(dz - center))),
        "rmse": float(np.sqrt(np.mean(dz**2))),
        "outlier_fraction_0p15": float(np.mean(np.abs(dz) > 0.15)),
        "coverage_68": float(
            np.mean(
                (truth[valid][interval] >= q16[valid][interval])
                & (truth[valid][interval] <= q84[valid][interval])
            )
        ),
        "median_interval_width_68": float(
            np.median(q84[valid][interval] - q16[valid][interval])
        ),
    }


def paired_bootstrap(
    frame: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, dict[str, float]]],
    dict[str, dict[str, dict[str, float]]],
]:
    """Return paired method differences using shared object resamples."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(int(seed))
    method_draws = {
        method: {metric: np.empty(n_resamples) for metric in METRIC_NAMES}
        for method in METHOD_COLUMNS
    }
    for draw in range(n_resamples):
        indices = rng.integers(0, len(frame), size=len(frame))
        sample = frame.iloc[indices]
        for method in METHOD_COLUMNS:
            metrics = redshift_metrics(sample, method)
            for metric in METRIC_NAMES:
                method_draws[method][metric][draw] = float(metrics[metric])

    method_intervals = {
        method: {
            metric: {
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
            }
            for metric, values in metric_draws.items()
        }
        for method, metric_draws in method_draws.items()
    }
    rows: list[dict[str, float | str | int]] = []
    intervals: dict[str, dict[str, dict[str, float]]] = {}
    for left, right in PAIRWISE_COMPARISONS:
        pair = f"{left}_minus_{right}"
        intervals[pair] = {}
        for metric in METRIC_NAMES:
            delta = method_draws[left][metric] - method_draws[right][metric]
            low, high = np.quantile(delta, [0.025, 0.975])
            point = redshift_metrics(frame, left)[metric] - redshift_metrics(
                frame, right
            )[metric]
            values = {
                "estimate": float(point),
                "ci95_low": float(low),
                "ci95_high": float(high),
            }
            intervals[pair][metric] = values
            rows.append(
                {
                    "left_method": left,
                    "right_method": right,
                    "metric": metric,
                    "difference_left_minus_right": float(point),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "n_objects": int(len(frame)),
                }
            )
    return pd.DataFrame(rows), intervals, method_intervals


def write_comparison_figure(
    frame: pd.DataFrame,
    metrics: dict[str, dict[str, float | int]],
    method_intervals: dict[str, dict[str, dict[str, float]]],
    out: Path,
) -> None:
    """Write the primary same-cohort redshift comparison figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = ("rws26", "rws24", "popcosmos")
    labels = {
        "rws26": "RWS, 26 bands",
        "rws24": "RWS, 24 bands",
        "popcosmos": "Pop-COSMOS",
    }
    colors = {
        "rws26": "#0072B2",
        "rws24": "#D55E00",
        "popcosmos": "#333333",
    }
    truth = frame["redshift_true"].to_numpy(float)
    finite_truth = truth[np.isfinite(truth)]
    upper = max(4.0, float(np.quantile(finite_truth, 0.995)))
    upper = min(5.5, np.ceil(upper * 2.0) / 2.0)

    fig = plt.figure(figsize=(13.2, 7.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.05, 0.8))
    for index, method in enumerate(methods):
        ax = fig.add_subplot(grid[0, index])
        prediction = frame[f"{method}_median"].to_numpy(float)
        valid = np.isfinite(truth) & np.isfinite(prediction)
        ax.scatter(
            truth[valid],
            prediction[valid],
            s=8,
            alpha=0.28,
            color=colors[method],
            edgecolors="none",
            rasterized=True,
        )
        ax.plot([0.0, upper], [0.0, upper], color="black", lw=1.0, ls="--")
        ax.set_xlim(0.0, upper)
        ax.set_ylim(0.0, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(labels[method], fontsize=12)
        ax.set_xlabel(r"$z_{\rm spec}$")
        if index == 0:
            ax.set_ylabel(r"posterior median $z$")
        values = metrics[method]
        ax.text(
            0.04,
            0.96,
            (
                f"NMAD = {values['nmad']:.3f}\n"
                f"outliers = {100.0 * values['outlier_fraction_0p15']:.1f}%"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        ax.grid(alpha=0.18)

    error_ax = fig.add_subplot(grid[1, :2])
    bins = np.linspace(-0.35, 0.35, 100)
    for method in methods:
        prediction = frame[f"{method}_median"].to_numpy(float)
        dz = (prediction - truth) / (1.0 + truth)
        error_ax.hist(
            dz[np.isfinite(dz)],
            bins=bins,
            density=True,
            histtype="step",
            lw=1.8,
            color=colors[method],
            label=labels[method],
        )
    error_ax.axvline(0.0, color="black", lw=1.0)
    error_ax.axvline(-0.15, color="0.5", lw=0.9, ls="--")
    error_ax.axvline(0.15, color="0.5", lw=0.9, ls="--")
    error_ax.set_xlabel(r"$(z_{\rm med}-z_{\rm spec})/(1+z_{\rm spec})$")
    error_ax.set_ylabel("density")
    error_ax.set_yscale("log")
    error_ax.set_ylim(bottom=0.08)
    error_ax.legend(frameon=False, ncol=3, fontsize=9)
    error_ax.grid(alpha=0.18)

    metric_ax = fig.add_subplot(grid[1, 2])
    displayed_metrics = (
        ("nmad", "NMAD"),
        ("outlier_fraction_0p15", "outlier fraction"),
        ("rmse", "RMSE"),
    )
    offsets = {"rws26": -0.18, "rws24": 0.0, "popcosmos": 0.18}
    for method in methods:
        for row, (metric, _label) in enumerate(displayed_metrics):
            value = float(metrics[method][metric])
            interval = method_intervals[method][metric]
            metric_ax.errorbar(
                value,
                row + offsets[method],
                xerr=np.asarray(
                    [
                        [value - interval["ci95_low"]],
                        [interval["ci95_high"] - value],
                    ]
                ),
                fmt="o",
                ms=5,
                capsize=2,
                color=colors[method],
                label=labels[method] if row == 0 else None,
            )
    metric_ax.set_yticks(
        np.arange(len(displayed_metrics)),
        [label for _metric, label in displayed_metrics],
    )
    metric_ax.invert_yaxis()
    metric_ax.set_xlabel("metric value with paired-bootstrap 95% CI")
    metric_ax.grid(axis="x", alpha=0.18)
    metric_ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(
        f"COSMOS2020 matched redshift benchmark (N_spec = {len(frame):,})",
        fontsize=14,
    )
    fig.savefig(out / "redshift_method_comparison.png", dpi=220)
    fig.savefig(out / "redshift_method_comparison.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    inputs = {
        "rws26": args.rws26,
        "rws24": args.rws24,
        "popcosmos": args.popcosmos,
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    paired = build_paired_table(args.rws26, args.rws24, args.popcosmos)
    if len(paired) != args.expected_evaluation:
        raise RuntimeError(
            f"Expected {args.expected_evaluation} evaluation objects, got {len(paired)}"
        )
    spec = paired.loc[paired["has_public_specz"]].reset_index(drop=True)
    if len(spec) != args.expected_specz:
        raise RuntimeError(
            f"Expected {args.expected_specz} public-specz objects, got {len(spec)}"
        )

    metrics = {
        method: redshift_metrics(spec, method) for method in METHOD_COLUMNS
    }
    differences, intervals, method_intervals = paired_bootstrap(
        spec,
        n_resamples=args.bootstrap,
        seed=args.bootstrap_seed,
    )
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    paired.to_parquet(out / "paired_evaluation_objects.parquet", index=False)
    spec.to_parquet(out / "paired_public_specz_objects.parquet", index=False)
    metric_rows = []
    for method, values in metrics.items():
        row = {"method": method, **values}
        for metric, interval in method_intervals[method].items():
            row[f"{metric}_ci95_low"] = interval["ci95_low"]
            row[f"{metric}_ci95_high"] = interval["ci95_high"]
        metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(out / "redshift_method_metrics.csv", index=False)
    differences.to_csv(out / "redshift_paired_bootstrap_differences.csv", index=False)

    provenance = {
        "status": "complete",
        "comparison": "redshift_only_same_object",
        "join_key": "Farmer object_id / Pop-COSMOS INDEX_COSMOS",
        "evaluation_rows": int(len(paired)),
        "public_specz_rows": int(len(spec)),
        "evaluation_object_ids_sha256": _ids_sha256(paired["object_id"].to_numpy()),
        "specz_object_ids_sha256": _ids_sha256(spec["object_id"].to_numpy()),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "bootstrap": {
            "paired": True,
            "n_resamples": int(args.bootstrap),
            "seed": int(args.bootstrap_seed),
            "confidence_level": 0.95,
        },
        "metric_contract": {
            "normalized_error": "(z_median - z_spec) / (1 + z_spec)",
            "nmad_scale": 1.48,
            "outlier_threshold": 0.15,
            "interval": "marginal q16-q84",
        },
        "spectroscopy": {
            "truth": "public COSMOS DR1.1 Confidence_level >= 50",
            "popcosmos_z_SPEC": "availability flag only; never used as numeric truth",
        },
    }
    (out / "provenance_manifest.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    summary = {
        **provenance,
        "metrics": metrics,
        "method_bootstrap_intervals": method_intervals,
        "paired_bootstrap_differences": intervals,
        "scope": (
            "All methods are evaluated on the same reproducible public-specz "
            "intersection. Published Pop-COSMOS 12,014-object headline metrics "
            "are not treated as same-cohort results."
        ),
    }
    (out / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_comparison_figure(spec, metrics, method_intervals, out)
    (out / "DONE").touch()
    print(
        "[cosmos-redshift-comparison] "
        f"evaluation={len(paired)} specz={len(spec)} bootstrap={args.bootstrap} -> {out}"
    )


if __name__ == "__main__":
    main()
