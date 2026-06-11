"""Prior/truth overlap diagnostics for amortized Diffsky runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir, write_json


_COMPARISONS = (
    ("z_obs", "redshift_true", "redshift"),
    ("log10_stellar_mass", "logsm_true", "stellar_mass"),
)


def write_diffsky_prior_overlap_report(
    *,
    dataset_path: str | Path,
    run_dir: str | Path,
    out_dir: str | Path,
    config: dict[str, Any],
    max_objects: int | None = None,
) -> Path:
    """Compare truth, posterior aggregate, and learned RealNVP prior samples."""
    dataset_path = Path(dataset_path)
    run_dir = Path(run_dir)
    out = ensure_dir(out_dir)
    truth = _read_truth(dataset_path, max_objects=max_objects)
    posterior_samples = _read_optional_table(run_dir / "posterior_samples.parquet")
    posterior_summary = _read_optional_table(run_dir / "posterior_summary.parquet")
    prior = _read_optional_table(run_dir / "learned_prior_samples.parquet")
    rows = []
    for param, truth_col, label in _COMPARISONS:
        rows.extend(
            _comparison_rows(
                label=label,
                parameter=param,
                truth_column=truth_col,
                truth=truth,
                posterior_samples=posterior_samples,
                posterior_summary=posterior_summary,
                prior=prior,
            )
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "prior_overlap_metrics.csv", index=False)
    _write_overlap_plots(out, truth, posterior_samples, prior)
    summary = {
        "dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "n_truth_rows": int(len(truth)),
        "n_posterior_sample_rows": int(len(posterior_samples)),
        "n_prior_rows": int(len(prior)),
        "comparisons": [dict(row) for row in rows],
        "notes": [
            "Only directly comparable parameters are scored here.",
            "logsfr_true is not compared to dlog10_sfr_* directly; use a derived DSPS SFR diagnostic before making SFR recovery claims.",
        ],
    }
    write_json(out / "prior_overlap_summary.json", summary)
    report_path = out / "prior_overlap_report.md"
    _write_markdown(report_path, metrics, summary)
    return report_path


def _read_truth(dataset_path: Path, *, max_objects: int | None) -> pd.DataFrame:
    columns = ["object_id", "redshift_true", "logsm_true", "logsfr_true"]
    available = pd.read_parquet(dataset_path, columns=None).columns
    selected = [column for column in columns if column in available]
    frame = pd.read_parquet(dataset_path, columns=selected)
    if max_objects is not None:
        frame = frame.head(int(max_objects))
    return frame


def _read_optional_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _comparison_rows(
    *,
    label: str,
    parameter: str,
    truth_column: str,
    truth: pd.DataFrame,
    posterior_samples: pd.DataFrame,
    posterior_summary: pd.DataFrame,
    prior: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if truth_column not in truth:
        return rows
    truth_values = _finite(truth[truth_column])
    if truth_values.size == 0:
        return rows
    rows.append(_distribution_row(label, parameter, "truth", truth_values, truth_values))
    if parameter in prior:
        rows.append(
            _distribution_row(
                label,
                parameter,
                "learned_prior",
                _finite(prior[parameter]),
                truth_values,
            )
        )
    if parameter in posterior_samples:
        rows.append(
            _distribution_row(
                label,
                parameter,
                "posterior_aggregate",
                _finite(posterior_samples[parameter]),
                truth_values,
            )
        )
    median_col = f"{parameter}_median"
    if median_col in posterior_summary and "object_id" in posterior_summary and "object_id" in truth:
        joined = posterior_summary[["object_id", median_col]].merge(
            truth[["object_id", truth_column]],
            on="object_id",
            how="inner",
        )
        if not joined.empty:
            y_true = joined[truth_column].to_numpy(dtype=float)
            y_pred = joined[median_col].to_numpy(dtype=float)
            finite = np.isfinite(y_true) & np.isfinite(y_pred)
            if finite.any():
                residual = y_pred[finite] - y_true[finite]
                rows.append(
                    {
                        "label": label,
                        "parameter": parameter,
                        "source": "posterior_median_recovery",
                        "n": int(finite.sum()),
                        "mean": float(np.mean(y_pred[finite])),
                        "std": float(np.std(y_pred[finite])),
                        "median": float(np.median(y_pred[finite])),
                        "truth_mean": float(np.mean(y_true[finite])),
                        "bias": float(np.mean(residual)),
                        "median_bias": float(np.median(residual)),
                        "rmse": float(np.sqrt(np.mean(residual**2))),
                        "sigma_mad": float(
                            1.4826 * np.median(np.abs(residual - np.median(residual)))
                        ),
                        "ks_to_truth": np.nan,
                        "wasserstein_to_truth": np.nan,
                    }
                )
    return rows


def _distribution_row(
    label: str,
    parameter: str,
    source: str,
    values: np.ndarray,
    truth_values: np.ndarray,
) -> dict[str, Any]:
    return {
        "label": label,
        "parameter": parameter,
        "source": source,
        "n": int(values.size),
        "mean": _safe_stat(np.mean, values),
        "std": _safe_stat(np.std, values),
        "median": _safe_stat(np.median, values),
        "truth_mean": _safe_stat(np.mean, truth_values),
        "bias": float(_safe_stat(np.mean, values) - _safe_stat(np.mean, truth_values)),
        "median_bias": float(
            _safe_stat(np.median, values) - _safe_stat(np.median, truth_values)
        ),
        "rmse": np.nan,
        "sigma_mad": np.nan,
        "ks_to_truth": _ks_2samp(values, truth_values),
        "wasserstein_to_truth": _wasserstein(values, truth_values),
    }


def _finite(series: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    return values[np.isfinite(values)]


def _safe_stat(fn, values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(fn(values))


def _ks_2samp(values: np.ndarray, truth: np.ndarray) -> float:
    if values.size == 0 or truth.size == 0:
        return float("nan")
    try:
        from scipy.stats import ks_2samp

        return float(ks_2samp(values, truth).statistic)
    except Exception:
        grid = np.sort(np.unique(np.concatenate([values, truth])))
        if grid.size == 0:
            return float("nan")
        cdf_values = np.searchsorted(np.sort(values), grid, side="right") / values.size
        cdf_truth = np.searchsorted(np.sort(truth), grid, side="right") / truth.size
        return float(np.max(np.abs(cdf_values - cdf_truth)))


def _wasserstein(values: np.ndarray, truth: np.ndarray) -> float:
    if values.size == 0 or truth.size == 0:
        return float("nan")
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(values, truth))
    except Exception:
        return float("nan")


def _write_overlap_plots(
    out: Path,
    truth: pd.DataFrame,
    posterior_samples: pd.DataFrame,
    prior: pd.DataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for parameter, truth_col, label in _COMPARISONS:
        if truth_col not in truth:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        _hist(ax, truth[truth_col], "truth", "#111827")
        if parameter in posterior_samples:
            _hist(ax, posterior_samples[parameter], "q_agg", "#2563eb")
        if parameter in prior:
            _hist(ax, prior[parameter], "learned prior", "#dc2626")
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / f"{label}_distribution_overlap.png", dpi=170)
        plt.close(fig)


def _hist(ax, values, label: str, color: str) -> None:
    values = _finite(values)
    if values.size == 0:
        return
    ax.hist(values, bins=50, density=True, histtype="step", linewidth=1.8, label=label, color=color)


def _write_markdown(path: Path, metrics: pd.DataFrame, summary: dict[str, Any]) -> None:
    lines = [
        "# Diffsky Amortized Prior Overlap",
        "",
        f"- dataset: `{summary['dataset_path']}`",
        f"- run: `{summary['run_dir']}`",
        f"- truth rows: {summary['n_truth_rows']}",
        f"- posterior sample rows: {summary['n_posterior_sample_rows']}",
        f"- learned prior rows: {summary['n_prior_rows']}",
        "",
        "## Metrics",
        "",
        _markdown_table(metrics) if not metrics.empty else "_No metrics._",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in summary["notes"])
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(col) for col in frame.columns]
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in frame.iterrows():
        values = []
        for col in frame.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}" if np.isfinite(value) else "nan")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)
