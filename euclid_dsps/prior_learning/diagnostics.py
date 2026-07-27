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
    quality_gate = prior_quality_gate(
        metrics=metrics,
        correlation_payload=corr_payload,
        multivariate=multivariate,
    )
    payload = {
        **summary,
        "n_parameters": int(len(parameter_names)),
        "median_ks_distance": _safe_median(metrics.get("ks_distance")),
        "median_wasserstein_distance": _safe_median(
            metrics.get("wasserstein_distance")
        ),
        "correlation_frobenius_error": corr_payload["frobenius_error"],
        "correlation_max_abs_error": corr_payload["max_abs_error"],
        "sliced_wasserstein_distance": multivariate["sliced_wasserstein_distance"],
        "energy_distance": multivariate["energy_distance"],
        "prior_quality_gate": quality_gate,
        "prior_quality_gate_status": quality_gate["status"],
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
        if key == "prior_quality_gate":
            continue
        lines.append(f"- `{key}`: {summary[key]}")
    gate = summary.get("prior_quality_gate")
    if isinstance(gate, dict):
        lines.extend(["", "## Prior Quality Gate", ""])
        lines.append(f"- `status`: {gate.get('status')}")
        bad_checks = [
            check
            for check in gate.get("checks", [])
            if isinstance(check, dict) and check.get("status") != "PASS"
        ]
        if bad_checks:
            lines.extend(
                [
                    "",
                    "| check | status | value | warn | fail |",
                    "| --- | --- | ---: | ---: | ---: |",
                ]
            )
            for check in bad_checks:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(check.get("name")),
                            str(check.get("status")),
                            _markdown_cell(check.get("value")),
                            _markdown_cell(check.get("warn")),
                            _markdown_cell(check.get("fail")),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("- All quality checks passed.")
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
            "- The NLL on truth latents is not sufficient: generated samples must "
            "also match the truth population after mapping back to physical parameters.",
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


def prior_quality_gate(
    *,
    metrics: pd.DataFrame,
    correlation_payload: dict[str, Any],
    multivariate: dict[str, float | None],
) -> dict[str, Any]:
    """Return a conservative pass/warn/fail gate for truth-vs-prior samples."""
    if metrics.empty:
        checks = [
            _quality_check(
                "metrics_nonempty",
                "FAIL",
                value=0,
                fail=1,
                message="No comparable truth/prior parameters were found.",
            )
        ]
        return {
            "status": "FAIL",
            "checks": checks,
            "interpretation": _quality_gate_interpretation("FAIL"),
        }

    ks = pd.to_numeric(metrics["ks_distance"], errors="coerce").dropna()
    wasserstein = pd.to_numeric(
        metrics["wasserstein_distance"],
        errors="coerce",
    ).dropna()
    n_params = max(int(len(ks)), 1)
    frac_ks_gt_0p5 = float((ks > 0.5).sum() / n_params) if len(ks) else float("nan")
    frac_ks_gt_0p8 = float((ks > 0.8).sum() / n_params) if len(ks) else float("nan")
    max_abs_median_delta = float(
        pd.to_numeric(metrics["median_residual"], errors="coerce").abs().max()
    )
    checks = [
        _threshold_check(
            "median_ks_distance",
            float(ks.median()) if len(ks) else float("nan"),
            warn=0.2,
            fail=0.5,
        ),
        _threshold_check(
            "max_ks_distance",
            float(ks.max()) if len(ks) else float("nan"),
            warn=0.4,
            fail=0.8,
        ),
        _threshold_check(
            "fraction_parameters_ks_gt_0p5",
            frac_ks_gt_0p5,
            warn=0.2,
            fail=0.5,
        ),
        _threshold_check(
            "fraction_parameters_ks_gt_0p8",
            frac_ks_gt_0p8,
            warn=0.05,
            fail=0.25,
        ),
        _threshold_check(
            "max_abs_median_residual",
            max_abs_median_delta,
            warn=1.0,
            fail=3.0,
        ),
    ]
    corr_frob = correlation_payload.get("frobenius_error")
    if corr_frob is not None:
        checks.append(
            _threshold_check(
                "correlation_frobenius_error",
                float(corr_frob),
                warn=1.0,
                fail=3.0,
            )
        )
    if len(wasserstein):
        checks.append(
            _threshold_check(
                "median_wasserstein_distance",
                float(wasserstein.median()),
                warn=0.5,
                fail=2.0,
            )
        )
    status = _quality_status(checks)
    return {
        "status": status,
        "checks": checks,
        "n_parameters_checked": int(len(ks)),
        "worst_ks_parameters": metrics.sort_values("ks_distance", ascending=False)
        .head(5)["parameter"]
        .astype(str)
        .tolist(),
        "sliced_wasserstein_distance": multivariate.get("sliced_wasserstein_distance"),
        "energy_distance": multivariate.get("energy_distance"),
        "interpretation": _quality_gate_interpretation(status),
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
        if (
            xname not in truth
            or yname not in truth
            or xname not in prior
            or yname not in prior
        ):
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
    corner_metadata: list[dict[str, Any]] = []
    path, metadata = _write_truth_prior_corner(
        plt,
        truth=truth,
        prior=prior,
        wanted=names[: min(len(names), 8)],
        out_dir=out_dir,
        filename="truth_vs_prior_corner.png",
        kind="legacy_first8",
        max_rows=max_corner_rows,
    )
    corner_metadata.append(metadata)
    if path is not None:
        outputs["corner"] = str(path)
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
            plt,
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
    plt,
    *,
    truth: pd.DataFrame,
    prior: pd.DataFrame,
    wanted: list[str],
    out_dir: Path,
    filename: str,
    kind: str,
    max_rows: int,
) -> tuple[Path | None, dict[str, Any]]:
    available = [name for name in wanted if name in truth and name in prior]
    missing = [name for name in wanted if name not in available]
    names, values_truth, values_prior, ranges, skipped_dynamic = (
        _prepare_truth_prior_corner_values(
            truth=truth,
            prior=prior,
            names=available,
            max_rows=max_rows,
        )
    )
    skipped = missing + skipped_dynamic
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
    fig = _truth_prior_corner_like_figure(
        plt,
        truth_values=values_truth,
        prior_values=values_prior,
        names=names,
        ranges=ranges,
    )
    path = out_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
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


def _prepare_truth_prior_corner_values(
    *,
    truth: pd.DataFrame,
    prior: pd.DataFrame,
    names: list[str],
    max_rows: int,
) -> tuple[list[str], np.ndarray, np.ndarray, list[tuple[float, float]], list[str]]:
    """Return finite/capped matrices and plot ranges for robust corner-like plots."""
    if not names:
        return [], np.empty((0, 0)), np.empty((0, 0)), [], []
    values_truth = _cap_matrix_rows(_finite_matrix(truth[names]), max_rows=max_rows)
    values_prior = _cap_matrix_rows(_finite_matrix(prior[names]), max_rows=max_rows)
    if len(values_truth) == 0 or len(values_prior) == 0:
        return names, values_truth, values_prior, [], []

    kept: list[str] = []
    kept_indices: list[int] = []
    ranges: list[tuple[float, float]] = []
    skipped: list[str] = []
    for index, name in enumerate(names):
        merged = np.concatenate([values_truth[:, index], values_prior[:, index]])
        merged = merged[np.isfinite(merged)]
        if merged.size < 2 or not bool(np.nanmax(merged) > np.nanmin(merged)):
            skipped.append(name)
            continue
        lo, hi = np.nanquantile(merged, [0.005, 0.995])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo = float(np.nanmin(merged))
            hi = float(np.nanmax(merged))
        if lo >= hi:
            skipped.append(name)
            continue
        pad = max(0.05 * (hi - lo), 1.0e-6)
        kept.append(name)
        kept_indices.append(index)
        ranges.append((float(lo - pad), float(hi + pad)))

    if not kept_indices:
        return [], values_truth[:, :0], values_prior[:, :0], [], skipped
    return (
        kept,
        values_truth[:, kept_indices],
        values_prior[:, kept_indices],
        ranges,
        skipped,
    )


def _truth_prior_corner_like_figure(
    plt,
    *,
    truth_values: np.ndarray,
    prior_values: np.ndarray,
    names: list[str],
    ranges: list[tuple[float, float]],
):
    n_columns = len(names)
    figsize = max(5.0, 1.45 * n_columns)
    fig, axes = plt.subplots(n_columns, n_columns, figsize=(figsize, figsize))
    axes_arr = np.asarray(axes).reshape(n_columns, n_columns)
    truth_plot = _subsample_rows_for_plot(np.asarray(truth_values, dtype=float))
    prior_plot = _subsample_rows_for_plot(np.asarray(prior_values, dtype=float))
    point_size = 1.5 if n_columns > 10 else 3.0
    alpha = 0.12 if n_columns > 10 else 0.20

    for row, y_name in enumerate(names):
        for col, x_name in enumerate(names):
            ax = axes_arr[row, col]
            if row < col:
                ax.axis("off")
                continue
            x_range = ranges[col]
            if row == col:
                _plot_distribution_hist(
                    ax,
                    truth_values[:, col],
                    value_range=x_range,
                    color="C0",
                    label="truth" if row == 0 else "",
                )
                _plot_distribution_hist(
                    ax,
                    prior_values[:, col],
                    value_range=x_range,
                    color="C1",
                    label="prior" if row == 0 else "",
                    linestyle="--",
                )
            else:
                ax.scatter(
                    truth_plot[:, col],
                    truth_plot[:, row],
                    s=point_size,
                    alpha=alpha,
                    color="C0",
                    rasterized=True,
                    linewidths=0,
                )
                ax.scatter(
                    prior_plot[:, col],
                    prior_plot[:, row],
                    s=point_size,
                    alpha=alpha,
                    color="C1",
                    rasterized=True,
                    linewidths=0,
                )
                ax.set_xlim(x_range)
                ax.set_ylim(ranges[row])
            if row == n_columns - 1:
                ax.set_xlabel(x_name, fontsize=6)
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(y_name, fontsize=6)
            elif col != 0:
                ax.set_yticklabels([])
            ax.tick_params(axis="both", labelsize=5, length=2)
    if n_columns:
        axes_arr[0, 0].legend(frameon=False, fontsize=6)
    fig.tight_layout(pad=0.15)
    return fig


def _plot_distribution_hist(
    ax,
    values,
    *,
    value_range: tuple[float, float],
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    ax.hist(
        arr,
        bins=36,
        range=value_range,
        density=True,
        histtype="step",
        color=color,
        linestyle=linestyle,
        linewidth=1.0,
        label=label,
    )


def _subsample_rows_for_plot(values: np.ndarray, max_rows: int = 1500) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= int(max_rows):
        return values
    rng = np.random.default_rng(260617)
    indices = np.sort(rng.choice(len(values), size=int(max_rows), replace=False))
    return values[indices]


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


def _threshold_check(
    name: str,
    value: float,
    *,
    warn: float,
    fail: float,
) -> dict[str, Any]:
    if not np.isfinite(value):
        return _quality_check(name, "FAIL", value=value, warn=warn, fail=fail)
    if value >= fail:
        status = "FAIL"
    elif value >= warn:
        status = "WARN"
    else:
        status = "PASS"
    return _quality_check(name, status, value=value, warn=warn, fail=fail)


def _quality_check(
    name: str,
    status: str,
    *,
    value: float | int | None = None,
    warn: float | None = None,
    fail: float | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "status": status,
        "value": value,
        "warn": warn,
        "fail": fail,
    }
    if message:
        payload["message"] = message
    return payload


def _quality_status(checks: list[dict[str, Any]]) -> str:
    if any(check["status"] == "FAIL" for check in checks):
        return "FAIL"
    if any(check["status"] == "WARN" for check in checks):
        return "WARN"
    return "PASS"


def _quality_gate_interpretation(status: str) -> str:
    if status == "PASS":
        return "Prior samples broadly match the held-out truth population diagnostics."
    if status == "WARN":
        return (
            "Prior samples show measurable population mismatch; inspect plots before "
            "using this checkpoint downstream."
        )
    return (
        "Prior samples do not match the truth population well enough for downstream "
        "scientific inference, even if the truth-latent NLL improved."
    )


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
            lines.append(
                "| "
                + " | ".join(_markdown_cell(row[col]) for col in sample.columns)
                + " |"
            )
        return "\n".join(lines)


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")
