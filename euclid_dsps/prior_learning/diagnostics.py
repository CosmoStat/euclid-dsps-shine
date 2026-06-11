"""Diagnostics for supervised truth-prior learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir, write_json


def distribution_metrics_frame(
    truth: pd.DataFrame,
    prior: pd.DataFrame,
    parameter_names: tuple[str, ...],
) -> pd.DataFrame:
    """Return per-parameter truth-vs-prior distribution metrics."""
    rows = []
    for name in parameter_names:
        if name not in truth or name not in prior:
            continue
        t = _finite_array(truth[name])
        p = _finite_array(prior[name])
        if t.size == 0 or p.size == 0:
            continue
        rows.append(
            {
                "parameter": name,
                "truth_n": int(t.size),
                "prior_n": int(p.size),
                "truth_mean": float(np.mean(t)),
                "prior_mean": float(np.mean(p)),
                "mean_residual": float(np.mean(p) - np.mean(t)),
                "truth_std": float(np.std(t)),
                "prior_std": float(np.std(p)),
                "std_residual": float(np.std(p) - np.std(t)),
                "truth_median": float(np.median(t)),
                "prior_median": float(np.median(p)),
                "median_residual": float(np.median(p) - np.median(t)),
                "ks_distance": float(ks_distance(t, p)),
                "wasserstein_distance": float(wasserstein_distance(t, p)),
            }
        )
    return pd.DataFrame(rows)


def write_supervised_prior_diagnostics(
    *,
    truth: pd.DataFrame,
    prior: pd.DataFrame,
    parameter_names: tuple[str, ...],
    out_dir: str | Path,
    summary: dict[str, Any],
) -> dict[str, str]:
    """Write CSV/JSON/Markdown diagnostics and best-effort plots."""
    out = ensure_dir(out_dir)
    metrics = distribution_metrics_frame(truth, prior, parameter_names)
    metrics_path = out / "prior_vs_truth_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    payload = {
        **summary,
        "n_parameters": int(len(parameter_names)),
        "median_ks_distance": _safe_median(metrics.get("ks_distance")),
        "median_wasserstein_distance": _safe_median(metrics.get("wasserstein_distance")),
    }
    write_json(out / "supervised_prior_summary.json", payload)
    report_path = out / "supervised_prior_vs_truth_report.md"
    write_supervised_prior_report(
        metrics=metrics,
        summary=payload,
        report_path=report_path,
    )
    outputs = {
        "metrics": str(metrics_path),
        "summary": str(out / "supervised_prior_summary.json"),
        "report": str(report_path),
    }
    outputs.update(
        _write_plots(
            truth=truth,
            prior=prior,
            parameter_names=parameter_names,
            out_dir=out,
        )
    )
    return outputs


def write_supervised_prior_report(
    *,
    metrics: pd.DataFrame,
    summary: dict[str, Any],
    report_path: str | Path,
) -> Path:
    """Write the supervised prior truth-comparison report."""
    path = Path(report_path)
    lines = [
        "# Supervised Diffsky Prior vs Truth",
        "",
        "This report evaluates `p_beta(theta_true)` learned directly from truth "
        "parameters. It does not evaluate photometric posterior inference.",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(summary):
        lines.append(f"- `{key}`: {summary[key]}")
    lines.extend(["", "## Distribution Metrics", ""])
    if metrics.empty:
        lines.append("_No metrics were computed._")
    else:
        lines.append(_frame_to_markdown(metrics))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A good prior match is a population diagnostic, not a photometric fit.",
            "- Physical recovery claims still require same-parameter forward closure and posterior calibration.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov distance without SciPy."""
    a = np.sort(_finite_array(a))
    b = np.sort(_finite_array(b))
    if a.size == 0 or b.size == 0:
        return float("nan")
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / a.size
    cdf_b = np.searchsorted(b, values, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def wasserstein_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Approximate 1D Wasserstein distance by matching quantiles."""
    a = np.sort(_finite_array(a))
    b = np.sort(_finite_array(b))
    if a.size == 0 or b.size == 0:
        return float("nan")
    q = np.linspace(0.0, 1.0, max(a.size, b.size), endpoint=True)
    aq = np.quantile(a, q)
    bq = np.quantile(b, q)
    return float(np.mean(np.abs(aq - bq)))


def _write_plots(
    *,
    truth: pd.DataFrame,
    prior: pd.DataFrame,
    parameter_names: tuple[str, ...],
    out_dir: Path,
) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}
    outputs = {}
    names = [name for name in parameter_names if name in truth and name in prior]
    if names:
        ncols = min(3, len(names))
        nrows = int(np.ceil(len(names) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, name in zip(axes_arr, names, strict=False):
            ax.hist(_finite_array(truth[name]), bins=40, histtype="step", density=True, label="truth")
            ax.hist(_finite_array(prior[name]), bins=40, histtype="step", density=True, label="prior")
            ax.set_title(name)
        for ax in axes_arr[len(names):]:
            ax.axis("off")
        axes_arr[0].legend()
        fig.tight_layout()
        path = out_dir / "truth_vs_prior_histograms.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs["histograms"] = str(path)
    pair_names = _pair_plot_names(names)
    if len(pair_names) >= 2:
        fig, axes = plt.subplots(1, len(pair_names) - 1, figsize=(4 * (len(pair_names) - 1), 4))
        axes_arr = np.asarray(axes).reshape(-1)
        xname = pair_names[0]
        for ax, yname in zip(axes_arr, pair_names[1:], strict=True):
            ax.scatter(truth[xname], truth[yname], s=4, alpha=0.25, label="truth")
            ax.scatter(prior[xname], prior[yname], s=4, alpha=0.25, label="prior")
            ax.set_xlabel(xname)
            ax.set_ylabel(yname)
        axes_arr[0].legend()
        fig.tight_layout()
        path = out_dir / "truth_vs_prior_z_logm_logsfr.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs["pair_z_logm_logsfr"] = str(path)
    try:
        import corner
    except Exception:
        return outputs
    corner_names = names[: min(len(names), 8)]
    if len(corner_names) >= 2:
        truth_values = truth[corner_names].to_numpy(dtype=float)
        prior_values = prior[corner_names].to_numpy(dtype=float)
        fig = corner.corner(truth_values, labels=corner_names, color="C0", hist_kwargs={"density": True})
        corner.corner(prior_values, fig=fig, labels=corner_names, color="C1", hist_kwargs={"density": True})
        path = out_dir / "truth_vs_prior_corner.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs["corner"] = str(path)
    return outputs


def _pair_plot_names(names: list[str]) -> list[str]:
    wanted = ["z_obs", "log10_stellar_mass", "log10_sfr_at_obs", "log10_ssfr_at_obs"]
    return [name for name in wanted if name in names]


def _finite_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _safe_median(values) -> float | None:
    if values is None:
        return None
    arr = _finite_array(values)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _frame_to_markdown(frame: pd.DataFrame, max_rows: int = 80) -> str:
    sample = frame.head(max_rows)
    try:
        return sample.to_markdown(index=False)
    except ImportError:
        columns = [str(column) for column in sample.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for _, row in sample.iterrows():
            lines.append("| " + " | ".join(_markdown_cell(row[col]) for col in sample.columns) + " |")
        return "\n".join(lines)


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")
