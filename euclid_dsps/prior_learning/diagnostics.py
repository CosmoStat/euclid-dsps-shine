"""Diagnostics for supervised truth-prior learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir, write_json

FENIKS_FULL_18D_PARAMETER_ORDER = [
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
    "diffstar_lgmcrit",
    "diffstar_lgy_at_mcrit",
    "diffstar_indx_lo",
    "diffstar_indx_hi",
    "diffstar_lg_qt",
    "diffstar_qlglgdt",
    "diffstar_lg_drop",
    "diffstar_lg_rejuv",
    "diffmah_logm0",
    "diffmah_logtc",
    "diffmah_early_index",
    "diffmah_late_index",
    "diffmah_t_peak",
]

FENIKS_USEFUL_PARAMETER_ORDER = [
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
    "log10_sfr_at_obs",
    "log10_ssfr_at_obs",
    "diffstar_lgmcrit",
    "diffstar_lgy_at_mcrit",
    "diffmah_logm0",
    "diffmah_t_peak",
]


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
    max_corner_rows: int = 4000,
) -> dict[str, str]:
    """Write CSV/JSON/Markdown diagnostics and best-effort plots."""
    out = ensure_dir(out_dir)
    metrics = distribution_metrics_frame(truth, prior, parameter_names)
    metrics_path = out / "prior_vs_truth_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    corr_payload = correlation_metrics(truth, prior, parameter_names)
    write_json(out / "prior_vs_truth_correlation_metrics.json", corr_payload)
    conditional = conditional_metrics_frame(truth, prior)
    conditional_path = out / "prior_vs_truth_conditional_metrics.csv"
    conditional.to_csv(conditional_path, index=False)
    multivariate = multivariate_distance_metrics(truth, prior, parameter_names)
    payload = {
        **summary,
        "n_parameters": int(len(parameter_names)),
        "median_ks_distance": _safe_median(metrics.get("ks_distance")),
        "median_wasserstein_distance": _safe_median(metrics.get("wasserstein_distance")),
        "correlation_frobenius_error": corr_payload["frobenius_error"],
        "correlation_max_abs_error": corr_payload["max_abs_error"],
        "sliced_wasserstein_distance": multivariate["sliced_wasserstein_distance"],
        "energy_distance": multivariate["energy_distance"],
        "prior_evaluated_after_physical_resampling": True,
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
        "correlation_metrics": str(out / "prior_vs_truth_correlation_metrics.json"),
        "conditional_metrics": str(conditional_path),
        "summary": str(out / "supervised_prior_summary.json"),
        "report": str(report_path),
    }
    outputs.update(
        _write_plots(
            truth=truth,
            prior=prior,
            parameter_names=parameter_names,
            out_dir=out,
            max_corner_rows=max_corner_rows,
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


def correlation_metrics(
    truth: pd.DataFrame,
    prior: pd.DataFrame,
    parameter_names: tuple[str, ...],
) -> dict[str, Any]:
    """Compare true and generated parameter correlation matrices."""
    names = [name for name in parameter_names if name in truth and name in prior]
    if len(names) < 2:
        return {
            "parameters": names,
            "frobenius_error": None,
            "max_abs_error": None,
            "truth_correlation": [],
            "prior_correlation": [],
            "correlation_error": [],
        }
    t = _finite_matrix(truth[names])
    p = _finite_matrix(prior[names])
    n = min(len(t), len(p))
    if n < 3:
        return {
            "parameters": names,
            "frobenius_error": None,
            "max_abs_error": None,
            "truth_correlation": [],
            "prior_correlation": [],
            "correlation_error": [],
        }
    t_corr = np.nan_to_num(np.corrcoef(t[:n], rowvar=False), nan=0.0)
    p_corr = np.nan_to_num(np.corrcoef(p[:n], rowvar=False), nan=0.0)
    err = p_corr - t_corr
    return {
        "parameters": names,
        "frobenius_error": float(np.linalg.norm(err)),
        "max_abs_error": float(np.max(np.abs(err))),
        "truth_correlation": t_corr.tolist(),
        "prior_correlation": p_corr.tolist(),
        "correlation_error": err.tolist(),
    }


def multivariate_distance_metrics(
    truth: pd.DataFrame,
    prior: pd.DataFrame,
    parameter_names: tuple[str, ...],
    *,
    n_projections: int = 64,
    max_rows: int = 4096,
) -> dict[str, float | None]:
    """Return lightweight multivariate distribution distances."""
    names = [name for name in parameter_names if name in truth and name in prior]
    if len(names) < 2:
        return {"sliced_wasserstein_distance": None, "energy_distance": None}
    t = _finite_matrix(truth[names])
    p = _finite_matrix(prior[names])
    n = min(len(t), len(p), int(max_rows))
    if n < 3:
        return {"sliced_wasserstein_distance": None, "energy_distance": None}
    rng = np.random.default_rng(260617)
    t = t[rng.choice(len(t), size=n, replace=False)]
    p = p[rng.choice(len(p), size=n, replace=False)]
    center = np.mean(t, axis=0)
    scale = np.std(t, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    tz = (t - center) / scale
    pz = (p - center) / scale
    dirs = rng.normal(size=(int(n_projections), len(names)))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1.0e-12)
    sw = [wasserstein_distance(tz @ direction, pz @ direction) for direction in dirs]
    return {
        "sliced_wasserstein_distance": float(np.mean(sw)),
        "energy_distance": float(_energy_distance(tz, pz)),
    }


def conditional_metrics_frame(truth: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """Compare simple conditional slopes for important parameter pairs."""
    pairs = [
        ("log10_stellar_mass", "log10_stellar_metallicity"),
        ("log10_stellar_mass", "log10_ssfr_at_obs"),
        ("log10_stellar_mass", "dust_av"),
        ("log10_stellar_mass", "diffstar_lgmcrit"),
        ("z_obs", "diffstar_lgmcrit"),
        ("log10_stellar_mass", "diffmah_logm0"),
        ("z_obs", "diffmah_logm0"),
    ]
    rows = []
    for xname, yname in pairs:
        if xname not in truth or yname not in truth or xname not in prior or yname not in prior:
            continue
        truth_slope = _linear_slope(truth[xname], truth[yname])
        prior_slope = _linear_slope(prior[xname], prior[yname])
        rows.append(
            {
                "x": xname,
                "y": yname,
                "truth_slope": truth_slope,
                "prior_slope": prior_slope,
                "slope_residual": prior_slope - truth_slope,
            }
        )
    return pd.DataFrame(rows)


def _write_plots(
    *,
    truth: pd.DataFrame,
    prior: pd.DataFrame,
    parameter_names: tuple[str, ...],
    out_dir: Path,
    max_corner_rows: int,
) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}
    outputs = {}
    names = [name for name in parameter_names if name in truth and name in prior]
    skipped_plot_parameters = _constant_or_invalid_plot_names(names, truth, prior)
    if skipped_plot_parameters:
        skipped_path = out_dir / "plot_skipped_parameters.json"
        write_json(
            skipped_path,
            {
                "reason": "constant_or_nonfinite_distribution",
                "parameters": skipped_plot_parameters,
            },
        )
        outputs["plot_skipped_parameters"] = str(skipped_path)
    if names:
        ncols = min(3, len(names))
        nrows = int(np.ceil(len(names) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, name in zip(axes_arr, names, strict=False):
            ax.hist(
                _finite_array(truth[name]),
                bins=40,
                histtype="step",
                density=True,
                label="truth",
            )
            ax.hist(
                _finite_array(prior[name]),
                bins=40,
                histtype="step",
                density=True,
                label="prior",
            )
            ax.set_title(name)
        for ax in axes_arr[len(names) :]:
            ax.axis("off")
        axes_arr[0].legend()
        fig.tight_layout()
        path = out_dir / "truth_vs_prior_histograms.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs["histograms"] = str(path)
    pair_names = _pair_plot_names(names)
    if len(pair_names) >= 2:
        fig, axes = plt.subplots(
            1,
            len(pair_names) - 1,
            figsize=(4 * (len(pair_names) - 1), 4),
        )
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
    corner_metadata: list[dict[str, Any]] = []
    corner_names = [
        name
        for name in names
        if _has_dynamic_range(truth[name]) and _has_dynamic_range(prior[name])
    ][: min(len(names), 8)]
    if len(corner_names) >= 2:
        truth_values = _cap_matrix_rows(
            truth[corner_names].to_numpy(dtype=float),
            max_rows=max_corner_rows,
        )
        prior_values = _cap_matrix_rows(
            prior[corner_names].to_numpy(dtype=float),
            max_rows=max_corner_rows,
        )
        fig = corner.corner(
            truth_values,
            labels=corner_names,
            color="C0",
            hist_kwargs={"density": True},
        )
        corner.corner(
            prior_values,
            fig=fig,
            labels=corner_names,
            color="C1",
            hist_kwargs={"density": True},
        )
        path = out_dir / "truth_vs_prior_corner.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs["corner"] = str(path)
        corner_metadata.append(
            {
                "plot": path.name,
                "kind": "legacy_first8",
                "plotted_columns": ",".join(corner_names),
                "n_columns": int(len(corner_names)),
                "written": True,
                "reason": "",
            }
        )
    for kind, wanted, filename in [
        (
            "full18",
            FENIKS_FULL_18D_PARAMETER_ORDER,
            "corner_truth_vs_prior_full18.png",
        ),
        (
            "useful",
            FENIKS_USEFUL_PARAMETER_ORDER,
            "corner_truth_vs_prior_useful.png",
        ),
    ]:
        path, metadata = _write_truth_prior_corner(
            corner,
            truth=truth,
            prior=prior,
            wanted=wanted,
            out_dir=out_dir,
            filename=filename,
            kind=kind,
            max_rows=max_corner_rows,
        )
        corner_metadata.append(metadata)
        if path is not None:
            outputs[f"corner_{kind}"] = str(path)
    if corner_metadata:
        meta_path = out_dir / "corner_plot_metadata.csv"
        pd.DataFrame(corner_metadata).to_csv(meta_path, index=False)
        outputs["corner_plot_metadata"] = str(meta_path)
    return outputs


def _write_truth_prior_corner(
    corner,
    *,
    truth: pd.DataFrame,
    prior: pd.DataFrame,
    wanted: list[str],
    out_dir: Path,
    filename: str,
    kind: str,
    max_rows: int,
) -> tuple[Path | None, dict[str, Any]]:
    names = [
        name
        for name in wanted
        if name in truth
        and name in prior
        and _has_dynamic_range(truth[name])
        and _has_dynamic_range(prior[name])
    ]
    skipped = [name for name in wanted if name not in names]
    if len(names) < 2:
        return None, {
            "plot": filename,
            "kind": kind,
            "plotted_columns": ",".join(names),
            "skipped_columns": ",".join(skipped),
            "n_columns": int(len(names)),
            "written": False,
            "reason": "fewer_than_two_dynamic_columns",
        }
    values_truth = _cap_matrix_rows(_finite_matrix(truth[names]), max_rows=max_rows)
    values_prior = _cap_matrix_rows(_finite_matrix(prior[names]), max_rows=max_rows)
    if len(values_truth) == 0 or len(values_prior) == 0:
        return None, {
            "plot": filename,
            "kind": kind,
            "plotted_columns": ",".join(names),
            "skipped_columns": ",".join(skipped),
            "n_columns": int(len(names)),
            "written": False,
            "reason": "no_finite_rows",
        }
    fig = corner.corner(
        values_truth,
        labels=names,
        color="C0",
        hist_kwargs={"density": True},
    )
    corner.corner(
        values_prior,
        fig=fig,
        labels=names,
        color="C1",
        hist_kwargs={"density": True},
    )
    path = out_dir / filename
    fig.savefig(path, dpi=150)
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return path, {
        "plot": filename,
        "kind": kind,
        "plotted_columns": ",".join(names),
        "skipped_columns": ",".join(skipped),
        "n_columns": int(len(names)),
        "truth_finite_rows": int(len(values_truth)),
        "prior_finite_rows": int(len(values_prior)),
        "max_corner_rows": int(max_rows),
        "written": True,
        "reason": "",
    }


def _pair_plot_names(names: list[str]) -> list[str]:
    wanted = ["z_obs", "log10_stellar_mass", "log10_sfr_at_obs", "log10_ssfr_at_obs"]
    return [name for name in wanted if name in names]


def _finite_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _finite_matrix(frame: pd.DataFrame) -> np.ndarray:
    arr = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr).all(axis=1)]


def _cap_matrix_rows(values: np.ndarray, *, max_rows: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        values = np.reshape(values, (len(values), -1))
    values = values[np.isfinite(values).all(axis=1)]
    max_rows = max(int(max_rows), 1)
    if len(values) <= max_rows:
        return values
    rng = np.random.default_rng(260617)
    indices = np.sort(rng.choice(len(values), size=max_rows, replace=False))
    return values[indices]


def _energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b), 2048)
    if n < 2:
        return float("nan")
    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)
    d_ab = _mean_pairwise_distance(a, b)
    d_aa = _mean_pairwise_distance(a, a)
    d_bb = _mean_pairwise_distance(b, b)
    return float(2.0 * d_ab - d_aa - d_bb)


def _mean_pairwise_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a[:, None, :] - b[None, :, :]
    return float(np.mean(np.sqrt(np.sum(diff * diff, axis=-1))))


def _linear_slope(x, y) -> float:
    xarr = np.asarray(x, dtype=float)
    yarr = np.asarray(y, dtype=float)
    mask = np.isfinite(xarr) & np.isfinite(yarr)
    if mask.sum() < 3 or np.std(xarr[mask]) == 0.0:
        return float("nan")
    cov = np.cov(xarr[mask], yarr[mask], ddof=0)
    return float(cov[0, 1] / cov[0, 0])


def _has_dynamic_range(values) -> bool:
    arr = _finite_array(values)
    if arr.size < 2:
        return False
    return bool(np.nanmax(arr) > np.nanmin(arr))


def _constant_or_invalid_plot_names(
    names: list[str],
    truth: pd.DataFrame,
    prior: pd.DataFrame,
) -> list[str]:
    skipped = []
    for name in names:
        if not _has_dynamic_range(truth[name]) or not _has_dynamic_range(prior[name]):
            skipped.append(name)
    return skipped


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
