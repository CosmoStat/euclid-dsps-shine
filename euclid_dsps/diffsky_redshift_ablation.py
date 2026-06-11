"""Redshift posterior metrics and Diffsky ablation reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir


def redshift_metrics_from_samples(
    posterior_samples: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    z_parameter: str = "z_obs",
    truth_column: str = "redshift_true",
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compute per-object and aggregate redshift posterior metrics."""
    required = {"object_id", z_parameter}
    if not required <= set(posterior_samples.columns):
        missing = ", ".join(sorted(required - set(posterior_samples.columns)))
        raise ValueError(f"posterior_samples missing required columns: {missing}")
    if "object_id" not in truth or truth_column not in truth:
        raise ValueError(f"truth must contain object_id and {truth_column}")
    rows = []
    truth_lookup = truth.set_index("object_id")[truth_column]
    for object_id, group in posterior_samples.groupby("object_id", sort=False):
        if object_id not in truth_lookup:
            continue
        z_true = float(truth_lookup.loc[object_id])
        samples = pd.to_numeric(group[z_parameter], errors="coerce").to_numpy(dtype=float)
        samples = samples[np.isfinite(samples)]
        if samples.size == 0 or not np.isfinite(z_true):
            continue
        q16, q50, q84 = np.quantile(samples, [0.16, 0.5, 0.84])
        q025, q975 = np.quantile(samples, [0.025, 0.975])
        delta = (q50 - z_true) / (1.0 + z_true)
        rows.append(
            {
                "object_id": object_id,
                "z_true": z_true,
                "z_pred_median": float(q50),
                "z_q16": float(q16),
                "z_q84": float(q84),
                "z_q025": float(q025),
                "z_q975": float(q975),
                "delta_z": float(delta),
                "pit": float(np.mean(samples <= z_true)),
                "covered_68": bool(q16 <= z_true <= q84),
                "covered_95": bool(q025 <= z_true <= q975),
                "posterior_width_68": float(q84 - q16),
                "posterior_width_95": float(q975 - q025),
                "n_samples": int(samples.size),
            }
        )
    frame = pd.DataFrame(rows)
    return frame, summarize_redshift_metrics(frame)


def summarize_redshift_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {
            "n_objects": 0,
            "median_bias": float("nan"),
            "sigma_mad": float("nan"),
            "rmse": float("nan"),
            "outlier_fraction_0p15": float("nan"),
            "pit_mean": float("nan"),
            "coverage_68": float("nan"),
            "coverage_95": float("nan"),
            "posterior_width_68_median": float("nan"),
        }
    delta = frame["delta_z"].to_numpy(dtype=float)
    finite = np.isfinite(delta)
    delta = delta[finite]
    if delta.size == 0:
        return summarize_redshift_metrics(pd.DataFrame())
    med = float(np.median(delta))
    return {
        "n_objects": int(delta.size),
        "median_bias": med,
        "sigma_mad": float(1.4826 * np.median(np.abs(delta - med))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "outlier_fraction_0p15": float(np.mean(np.abs(delta) > 0.15)),
        "pit_mean": float(np.nanmean(frame["pit"])),
        "coverage_68": float(np.nanmean(frame["covered_68"].astype(float))),
        "coverage_95": float(np.nanmean(frame["covered_95"].astype(float))),
        "posterior_width_68_median": float(
            np.nanmedian(frame["posterior_width_68"])
        ),
    }


def posterior_truth_metrics(
    posterior_samples: pd.DataFrame,
    truth: pd.DataFrame,
    parameter_pairs: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    """Summarize posterior median recovery for configured parameter/truth pairs."""
    rows = []
    for parameter, truth_column in parameter_pairs:
        if parameter not in posterior_samples or truth_column not in truth:
            continue
        med = (
            posterior_samples.groupby("object_id", sort=False)[parameter]
            .median()
            .rename("posterior_median")
            .reset_index()
        )
        joined = med.merge(
            truth[["object_id", truth_column]],
            on="object_id",
            how="inner",
        )
        if joined.empty:
            continue
        y_pred = joined["posterior_median"].to_numpy(dtype=float)
        y_true = joined[truth_column].to_numpy(dtype=float)
        finite = np.isfinite(y_pred) & np.isfinite(y_true)
        if not finite.any():
            continue
        residual = y_pred[finite] - y_true[finite]
        rows.append(
            {
                "parameter": parameter,
                "truth_column": truth_column,
                "n_objects": int(finite.sum()),
                "bias": float(np.mean(residual)),
                "median_bias": float(np.median(residual)),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "sigma_mad": float(
                    1.4826 * np.median(np.abs(residual - np.median(residual)))
                ),
            }
        )
    return pd.DataFrame(rows)


def write_redshift_metrics_for_run(
    *,
    dataset_path: str | Path,
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    label: str = "run",
) -> dict[str, Path]:
    """Compute redshift and posterior-vs-truth metrics for one inference run."""
    run = Path(run_dir)
    out = ensure_dir(out_dir or run)
    samples = _read_table(run / "posterior_samples.parquet")
    truth_cols = ["object_id", "redshift_true", "logsm_true", "logsfr_true", "logssfr_true"]
    truth = _read_existing_columns(dataset_path, truth_cols)
    object_metrics, summary = redshift_metrics_from_samples(samples, truth)
    object_path = out / "photoz_object_metrics.csv"
    object_metrics.to_csv(object_path, index=False)
    summary_frame = pd.DataFrame([{**summary, "label": label}])
    photoz_path = out / "photoz_metrics.csv"
    summary_frame.to_csv(photoz_path, index=False)
    pairs = (
        ("z_obs", "redshift_true"),
        ("log10_stellar_mass", "logsm_true"),
        ("log10_sfr_at_obs", "logsfr_true"),
        ("log10_ssfr_at_obs", "logssfr_true"),
    )
    posterior_metrics = posterior_truth_metrics(samples, truth, pairs)
    posterior_path = out / "posterior_vs_truth_metrics.csv"
    posterior_metrics.to_csv(posterior_path, index=False)
    return {
        "photoz_object_metrics": object_path,
        "photoz_metrics": photoz_path,
        "posterior_vs_truth_metrics": posterior_path,
    }


def run_redshift_ablation(
    *,
    dataset_path: str | Path,
    runs: list[tuple[str, Path]],
    out_dir: str | Path,
) -> Path:
    """Compare redshift metrics across multiple inference/closure runs."""
    out = ensure_dir(out_dir)
    rows = []
    object_frames = []
    truth = _read_existing_columns(dataset_path, ["object_id", "redshift_true"])
    for label, run_dir in runs:
        samples_path = run_dir / "posterior_samples.parquet"
        if not samples_path.exists():
            continue
        samples = _read_table(samples_path)
        object_metrics, summary = redshift_metrics_from_samples(samples, truth)
        rows.append({"label": label, "run_dir": str(run_dir), **summary})
        if not object_metrics.empty:
            object_metrics.insert(0, "label", label)
            object_frames.append(object_metrics)
    summary = pd.DataFrame(rows)
    summary_path = out / "redshift_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    objects = pd.concat(object_frames, ignore_index=True) if object_frames else pd.DataFrame()
    if not objects.empty:
        objects.to_csv(out / "redshift_ablation_object_metrics.csv", index=False)
        _write_redshift_plots(objects, out)
    report = out / "redshift_ablation_report.md"
    _write_ablation_report(report, summary)
    return report


def parse_run_specs(specs: list[str]) -> list[tuple[str, Path]]:
    runs = []
    for spec in specs:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            path = spec
            label = Path(path).name
        runs.append((label, Path(path)))
    return runs


def _read_existing_columns(path: str | Path, columns: list[str]) -> pd.DataFrame:
    available = pd.read_parquet(path, columns=None).columns
    selected = [column for column in columns if column in available]
    return pd.read_parquet(path, columns=selected)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_redshift_plots(frame: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, group in frame.groupby("label", sort=False):
        ax.scatter(group["z_true"], group["z_pred_median"], s=8, alpha=0.5, label=label)
    lo = float(np.nanmin(frame[["z_true", "z_pred_median"]].to_numpy()))
    hi = float(np.nanmax(frame[["z_true", "z_pred_median"]].to_numpy()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("z_true")
    ax.set_ylabel("z_posterior_median")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "z_pred_vs_z_true.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, group in frame.groupby("label", sort=False):
        ax.hist(group["pit"], bins=20, histtype="step", density=True, label=label)
    ax.set_xlabel("PIT")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "pit_histogram.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, group in frame.groupby("label", sort=False):
        ax.hist(group["delta_z"], bins=40, histtype="step", density=True, label=label)
    ax.set_xlabel("delta_z")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "delta_z_histogram.png", dpi=160)
    plt.close(fig)


def _write_ablation_report(path: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Diffsky Redshift Ablation",
        "",
        "This report compares posterior redshift accuracy and calibration.",
        "",
        "## Summary",
        "",
        _markdown_table(summary) if not summary.empty else "_No runs scored._",
        "",
        "Median accuracy alone is insufficient; use PIT, coverage, and posterior "
        "widths to evaluate calibration.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for col in frame.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}" if np.isfinite(value) else "nan")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
