"""Prior/truth overlap diagnostics for amortized Diffsky runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir, write_json

_COMPARISONS = (
    ("z_obs", "redshift_true", "redshift"),
    ("log10_stellar_mass", "logsm_true", "stellar_mass"),
    ("log10_sfr_at_obs", "logsfr_true", "log_sfr_at_obs"),
    ("log10_ssfr_at_obs", "logssfr_true", "log_ssfr_at_obs"),
    ("dust_av", "dust_av", "dust_av"),
    ("dust_delta", "dust_delta", "dust_delta"),
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
    prior = _read_optional_table(run_dir / "learned_or_loaded_prior_samples.parquet")
    if prior.empty:
        prior = _read_optional_table(run_dir / "learned_prior_samples.parquet")
    rows = []
    for param, truth_col, label in _available_comparisons(truth, posterior_samples, prior):
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
            "Only directly comparable or derived-compatible parameters are scored here.",
            "logsfr_true is never compared to raw dlog10_sfr_* parameters.",
            "Generated Diffstar/Diffmah truth columns are population diagnostics, not photometric recoveries by themselves.",
        ],
    }
    write_json(out / "prior_overlap_summary.json", summary)
    report_path = out / "prior_overlap_report.md"
    _write_markdown(report_path, metrics, summary)
    population_report = out / "population_realism_report.md"
    _write_markdown(population_report, metrics, summary)
    return report_path


def _read_truth(dataset_path: Path, *, max_objects: int | None) -> pd.DataFrame:
    frame = pd.read_parquet(dataset_path)
    if max_objects is not None:
        frame = frame.head(int(max_objects))
    return frame


def _available_comparisons(
    truth: pd.DataFrame,
    posterior_samples: pd.DataFrame,
    prior: pd.DataFrame,
) -> tuple[tuple[str, str, str], ...]:
    comparisons = list(_COMPARISONS)
    for prefix in ("diffstar_", "diffmah_", "burst_"):
        for column in truth.columns:
            if column.startswith(prefix):
                comparisons.append((column, column, column))
    available = []
    for parameter, truth_col, label in comparisons:
        if truth_col not in truth:
            continue
        if parameter.startswith("dlog10_sfr_"):
            continue
        if parameter in posterior_samples or parameter in prior:
            available.append((parameter, truth_col, label))
        elif parameter in {"z_obs", "log10_stellar_mass"}:
            available.append((parameter, truth_col, label))
    return tuple(dict.fromkeys(available))


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
    comparisons = _available_comparisons(truth, posterior_samples, prior)
    for parameter, truth_col, label in comparisons:
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
    _write_z_logm_logsfr_plot(out, truth, prior, plt)
    _write_corner_plot(out / "truth_vs_prior_corner.png", truth, prior)
    _write_corner_plot(out / "truth_vs_qagg_corner.png", truth, posterior_samples)


def _write_z_logm_logsfr_plot(out: Path, truth: pd.DataFrame, prior: pd.DataFrame, plt) -> None:
    pairs = [
        ("redshift_true", "z_obs", "z"),
        ("logsm_true", "log10_stellar_mass", "logM"),
        ("logsfr_true", "log10_sfr_at_obs", "logSFR"),
    ]
    available = [(t, p, label) for t, p, label in pairs if t in truth and p in prior]
    if len(available) < 2:
        return
    x_truth, x_prior, xlabel = available[0]
    fig, axes = plt.subplots(1, len(available) - 1, figsize=(5 * (len(available) - 1), 4))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (y_truth, y_prior, ylabel) in zip(axes_arr, available[1:], strict=True):
        ax.scatter(truth[x_truth], truth[y_truth], s=6, alpha=0.35, label="truth")
        ax.scatter(prior[x_prior], prior[y_prior], s=6, alpha=0.35, label="prior")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    axes_arr[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "truth_vs_prior_z_logm_logsfr.png", dpi=170)
    plt.close(fig)


def _write_corner_plot(path: Path, truth: pd.DataFrame, other: pd.DataFrame) -> None:
    try:
        import corner
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pairs = [
        ("redshift_true", "z_obs", "z"),
        ("logsm_true", "log10_stellar_mass", "logM"),
        ("logsfr_true", "log10_sfr_at_obs", "logSFR"),
        ("dust_av", "dust_av", "dust_av"),
        ("dust_delta", "dust_delta", "dust_delta"),
    ]
    truth_cols = [truth_col for truth_col, other_col, _ in pairs if truth_col in truth and other_col in other]
    other_cols = [other_col for truth_col, other_col, _ in pairs if truth_col in truth and other_col in other]
    labels = [label for truth_col, other_col, label in pairs if truth_col in truth and other_col in other]
    if len(truth_cols) < 2:
        return
    truth_values = truth[truth_cols].dropna().to_numpy(dtype=float)
    other_values = other[other_cols].dropna().to_numpy(dtype=float)
    if truth_values.shape[0] <= truth_values.shape[1] or other_values.shape[0] <= other_values.shape[1]:
        return
    try:
        fig = corner.corner(truth_values, labels=labels, color="C0")
        corner.corner(other_values, fig=fig, labels=labels, color="C1")
    except Exception:
        return
    fig.savefig(path, dpi=170)
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
