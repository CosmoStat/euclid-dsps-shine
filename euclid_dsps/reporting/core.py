"""EDA and run reporting."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

from ..io import GalaxyObservation, ensure_dir, write_json
from ..model import ModelResult, comparison_rows


def configure_plot_style() -> None:
    """Apply project-wide, publication-style matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.alpha": 0.22,
            "grid.linewidth": 0.45,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "axes.prop_cycle": cycler(
                color=[
                    "#2F5D8C",
                    "#B85C38",
                    "#3F7F5F",
                    "#7A4E8A",
                    "#8C6A2F",
                    "#4F6F7A",
                ]
            ),
        }
    )


configure_plot_style()


def write_eda_outputs(
    df: pd.DataFrame,
    band_configs: list[dict[str, Any]],
    out_dir: str | Path,
    redshift_config: dict[str, Any] | None = None,
) -> None:
    out = ensure_dir(out_dir)
    schema = [{"name": col, "dtype": str(dtype)} for col, dtype in df.dtypes.items()]
    write_json(out / "catalog_schema.json", schema)
    df.describe(include="all").transpose().to_csv(out / "catalog_stats.csv")
    missing = df.isna().sum().rename("missing_count").to_frame()
    missing["missing_fraction"] = missing["missing_count"] / max(len(df), 1)
    missing.to_csv(out / "missing_values.csv")

    band_columns = [
        band["column"] for band in band_configs if band["column"] in df.columns
    ]
    if band_columns:
        plot_flux_distributions(df, band_columns, out / "flux_distributions.png")
        plot_color_distributions(df, band_columns, out / "color_distributions.png")
    if redshift_config:
        plot_redshift_distributions(
            df, redshift_config, out / "redshift_diagnostics.png"
        )
    plot_physical_parameters_distributions(df, out / "physical_parameters.png")


def write_run_outputs(
    observation: GalaxyObservation, result: ModelResult, out_dir: str | Path
) -> pd.DataFrame:
    out = ensure_dir(out_dir)
    write_json(out / "selected_galaxy.json", observation)
    write_json(
        out / "model_parameters.json",
        {"parameters": result.parameters, "derived": result.derived},
    )

    sed = pd.DataFrame(
        {
            "wave_angstrom": result.wave,
            "rest_sed_lsun_per_hz": result.rest_sed,
            "dusted_rest_sed_lsun_per_hz": result.dusted_rest_sed,
        }
    )
    sed.to_csv(out / "sed.csv", index=False)

    comparison = pd.DataFrame(comparison_rows(observation, result))
    comparison.to_csv(out / "photometry_comparison.csv", index=False)

    plot_sed(result, out / "sed.png")
    plot_photometry_comparison(comparison, out / "photometry_comparison.png")
    return comparison


def write_fit_outputs(fit_result: Any, out_dir: str | Path) -> None:
    out = ensure_dir(out_dir)
    write_json(
        out / "fit_result.json",
        {
            "success": fit_result.success,
            "message": fit_result.message,
            "best_parameters": fit_result.best_parameters,
            "chi2": fit_result.chi2,
            "n_bands": fit_result.n_bands,
            "gradient_norm": fit_result.gradient_norm,
        },
    )
    pd.DataFrame(fit_result.trace).to_csv(out / "fit_trace.csv", index=False)
    plot_fit_trace(pd.DataFrame(fit_result.trace), out / "fit_trace.png")


def write_mcmc_outputs(
    mcmc_result: Any,
    out_dir: str | Path,
    truth_values: dict[str, Any] | None = None,
) -> None:
    out = ensure_dir(out_dir)
    samples = pd.DataFrame(mcmc_result.samples)
    samples.to_csv(out / "posterior_samples.csv", index=False)
    derived = pd.DataFrame(getattr(mcmc_result, "derived_samples", {}))
    if not derived.empty:
        derived.to_csv(out / "posterior_derived_samples.csv", index=False)
    pd.DataFrame(mcmc_result.summary).to_csv(out / "posterior_summary.csv", index=False)
    write_json(out / "mcmc_diagnostics.json", mcmc_result.diagnostics)
    if truth_values:
        write_json(out / "posterior_truth_values.json", truth_values)
    write_posterior_predictive(mcmc_result, out / "posterior_predictive_photometry.csv")
    plot_mcmc_traces(samples, out / "posterior_trace.png")
    plot_corner(samples, out / "posterior_corner.png")
    comparable = posterior_comparable_frame(samples, derived, truth_values or {})
    if not comparable.empty:
        comparable.to_csv(out / "posterior_comparable_samples.csv", index=False)
        plot_corner_with_truth(
            comparable,
            truth_values or {},
            out / "posterior_corner_with_truth.png",
        )
    plot_posterior_predictive(mcmc_result, out / "posterior_predictive_photometry.png")


def write_mcmc_batch_outputs(
    summary: pd.DataFrame,
    predictive: pd.DataFrame,
    diagnostics: pd.DataFrame,
    out_dir: str | Path,
) -> None:
    out = ensure_dir(out_dir)
    summary_payload = {
        "n_galaxies": (
            int(summary["row_index"].nunique()) if "row_index" in summary else 0
        ),
        "n_parameters": (
            int(summary["parameter"].nunique()) if "parameter" in summary else 0
        ),
    }
    if "n_divergent" in diagnostics:
        summary_payload["n_divergent"] = int(diagnostics["n_divergent"].sum())
    if "residual_mag_median_model_minus_observed" in predictive:
        summary_payload["median_abs_posterior_residual_mag"] = float(
            predictive["residual_mag_median_model_minus_observed"].abs().median()
        )
    write_json(out / "batch_mcmc_summary.json", summary_payload)
    plot_batch_posterior_intervals(
        summary, out / "batch_posterior_parameter_intervals.png"
    )
    plot_batch_posterior_predictive(predictive, out / "batch_posterior_predictive.png")
    plot_batch_mcmc_diagnostics(diagnostics, out / "batch_mcmc_diagnostics.png")


def write_population_corner_outputs(
    fits: pd.DataFrame, free_parameters: list[str], out_dir: str | Path
) -> None:
    """Write population-level MAP point-estimate distributions."""
    out = ensure_dir(out_dir)
    params = _fit_parameter_frame(fits, free_parameters)
    if params.empty:
        return
    paired_params, paired_truth = paired_fit_truth_frames(fits)
    params.to_csv(out / "population_map_parameters.csv", index=False)
    params.describe(percentiles=[0.05, 0.16, 0.5, 0.84, 0.95]).transpose().to_csv(
        out / "population_map_parameter_summary.csv"
    )
    plot_corner(params, out / "population_corner_parameters.png")
    plot_population_parameter_histograms(
        params, out / "population_parameter_distributions.png"
    )
    if not paired_truth.empty:
        paired_truth.to_csv(out / "population_truth_parameters.csv", index=False)
        paired_truth.describe(
            percentiles=[0.05, 0.16, 0.5, 0.84, 0.95]
        ).transpose().to_csv(out / "population_truth_parameter_summary.csv")
        metrics = parameter_truth_metrics(fits)
        if not metrics.empty:
            metrics.to_csv(out / "population_parameter_truth_metrics.csv", index=False)
        plot_corner_overlay(
            paired_params,
            paired_truth,
            out / "population_corner_parameters_with_truth.png",
        )
        plot_population_parameter_histograms(
            paired_params,
            out / "population_parameter_distributions_with_truth.png",
            truth=paired_truth,
        )


def plot_population_parameter_histograms(
    params: pd.DataFrame, path: str | Path, truth: pd.DataFrame | None = None
) -> None:
    if params.empty:
        return
    columns = list(params.columns)
    fig, axes = plt.subplots(len(columns), 1, figsize=(8, max(2.2 * len(columns), 3)))
    axes = np.atleast_1d(axes)
    for ax, column in zip(axes, columns, strict=True):
        values = params[column].replace([np.inf, -np.inf], np.nan).dropna()
        truth_values = (
            truth[column].replace([np.inf, -np.inf], np.nan).dropna()
            if truth is not None and column in truth
            else pd.Series(dtype=float)
        )
        bins = 50
        if not values.empty and not truth_values.empty:
            combined = pd.concat([values, truth_values])
            lo, hi = float(combined.min()), float(combined.max())
            if lo < hi:
                bins = np.linspace(lo, hi, 51)
        if not values.empty:
            ax.hist(
                values,
                bins=bins,
                histtype="stepfilled",
                alpha=0.55,
                color="#8fbcd4",
                label="inferred",
            )
            ax.axvline(values.median(), color="black", lw=1, label="median")
        if not truth_values.empty:
            ax.hist(
                truth_values,
                bins=bins,
                histtype="step",
                lw=1.7,
                color="#e6a0a8",
                label=_truth_legend_label(column),
            )
            ax.axvline(truth_values.median(), color="#9f5360", lw=1, alpha=0.8)
        ax.set_xlabel(_parameter_display_label(column))
        ax.set_ylabel("galaxies")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def parameter_truth_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired inferred-vs-truth parameter errors."""
    rows = []
    pairs: list[tuple[str, str, str]] = []
    if {"z_obs", "redshift_truth"}.issubset(frame.columns):
        pairs.append(("z_obs", "z_obs", "redshift_truth"))
    for col in frame.columns:
        if col.startswith("truth_"):
            parameter = col[6:]
            fit_col = f"fit_{parameter}"
            if fit_col in frame.columns:
                pairs.append((parameter, fit_col, col))

    for parameter, fit_col, truth_col in pairs:
        work = frame[[fit_col, truth_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if work.empty:
            continue
        delta = work[fit_col] - work[truth_col]
        corr = (
            float(work[fit_col].corr(work[truth_col]))
            if len(work) > 1
            and work[fit_col].nunique() > 1
            and work[truth_col].nunique() > 1
            else float("nan")
        )
        rows.append(
            {
                "parameter": parameter,
                "comparison_kind": _truth_kind_from_frame(frame, parameter),
                "fit_column": fit_col,
                "truth_column": truth_col,
                "n": int(len(work)),
                "bias_mean": float(delta.mean()),
                "bias_median": float(delta.median()),
                "mae": float(delta.abs().mean()),
                "median_abs_error": float(delta.abs().median()),
                "rmse": float(np.sqrt(np.mean(delta**2))),
                "std_delta": float(delta.std()),
                "correlation": corr,
                "truth_median": float(work[truth_col].median()),
                "inferred_median": float(work[fit_col].median()),
            }
        )
    return pd.DataFrame(rows)


def write_trace_truth_outputs(
    trace: pd.DataFrame, out_dir: str | Path, label: str, make_plots: bool = True
) -> None:
    if trace.empty or "truth_mse" not in trace:
        return
    metric_values = trace["truth_mse"].replace([np.inf, -np.inf], np.nan).dropna()
    if metric_values.empty:
        return
    out = ensure_dir(out_dir)
    trace_truth_summary(trace).to_csv(
        out / f"{label}_trace_truth_summary.csv", index=False
    )
    if make_plots:
        plot_trace_truth_metrics(trace, out / f"{label}_trace_truth.png")


def trace_truth_summary(trace: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        col
        for col in trace.columns
        if col in {"truth_mse", "truth_rmse", "truth_mae"}
        or col.startswith("truth_mse_")
    ]
    if not metric_columns:
        return pd.DataFrame()
    group_names = ["chunk_index"] if "chunk_index" in trace.columns else []
    grouped = (
        trace.groupby(group_names, dropna=False) if group_names else [(None, trace)]
    )
    rows = []
    for key, group in grouped:
        row: dict[str, float | int] = {}
        if group_names:
            if isinstance(key, tuple):
                key = key[0]
            row["chunk_index"] = int(key)
        group = group.sort_values("iteration") if "iteration" in group else group
        for col in metric_columns:
            values = group[col].replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            row[f"final_{col}"] = float(values.iloc[-1])
            row[f"min_{col}"] = float(values.min())
            row[f"final_minus_min_{col}"] = float(values.iloc[-1] - values.min())
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def plot_trace_truth_metrics(trace: pd.DataFrame, path: str | Path) -> None:
    if trace.empty or "truth_rmse" not in trace:
        return
    truth = trace["truth_rmse"].replace([np.inf, -np.inf], np.nan).dropna()
    if truth.empty:
        return
    loss_col = (
        "mean_chi2_or_loss"
        if "mean_chi2_or_loss" in trace
        else "chi2" if "chi2" in trace else None
    )
    param_cols = [col for col in trace.columns if col.startswith("truth_mse_")]
    ncols = 3 if param_cols else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    axes = np.atleast_1d(axes)

    if loss_col:
        _plot_trace_metric(trace, loss_col, axes[0], label=loss_col, logy=True)
        axes[0].set_ylabel(loss_col)
    else:
        axes[0].axis("off")

    _plot_trace_metric(
        trace,
        "truth_rmse",
        axes[1],
        label="configured truth/proxy RMSE",
        logy=True,
    )
    axes[1].set_ylabel("diagnostic RMSE")
    axes[1].set_title("not optimized")

    if param_cols:
        grouped = _trace_group_mean(trace, param_cols)
        x = grouped["iteration"] if "iteration" in grouped else np.arange(len(grouped))
        for col in param_cols:
            values = np.sqrt(
                grouped[col].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            )
            if np.isfinite(values).any():
                parameter = col.removeprefix("truth_mse_")
                axes[2].plot(
                    x,
                    values,
                    lw=1.4,
                    label=_parameter_display_label(parameter),
                )
        axes[2].set_xlabel("iteration")
        axes[2].set_ylabel("per-parameter diagnostic RMSE")
        axes[2].set_title("photometry objective can diverge from truth")
        axes[2].set_yscale("log")
        axes[2].grid(alpha=0.25)
        axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_trace_metric(
    trace: pd.DataFrame, column: str, ax: plt.Axes, label: str, logy: bool = False
) -> None:
    if "chunk_index" in trace:
        for _, group in trace.groupby("chunk_index"):
            group = group.sort_values("iteration")
            ax.plot(
                group["iteration"],
                group[column],
                color="#8fbcd4",
                alpha=0.18,
                lw=0.8,
            )
    grouped = _trace_group_mean(trace, [column])
    x = grouped["iteration"] if "iteration" in grouped else np.arange(len(grouped))
    ax.plot(x, grouped[column], color="black", lw=1.6, label=label)
    ax.set_xlabel("iteration")
    if logy and (grouped[column].dropna() > 0).any():
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def _trace_group_mean(trace: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [col for col in columns if col in trace]
    if "iteration" not in trace:
        return trace[available].reset_index(drop=True)
    return (
        trace[["iteration", *available]]
        .replace([np.inf, -np.inf], np.nan)
        .groupby("iteration", as_index=False)
        .mean(numeric_only=True)
    )


def write_batch_outputs(
    comparison: pd.DataFrame,
    out_dir: str | Path,
    label: str = "batch",
    reporting_level: str = "full",
) -> None:
    """Write aggregate tables and plots for multi-galaxy runs."""
    out = ensure_dir(out_dir)
    error_rows = (
        comparison[comparison["error"].notna()]
        if "error" in comparison
        else pd.DataFrame()
    )
    valid = (
        comparison[comparison["band"].notna()].copy()
        if "band" in comparison
        else pd.DataFrame()
    )

    if not error_rows.empty:
        error_rows.to_csv(out / f"{label}_errors.csv", index=False)

    if valid.empty:
        write_json(
            out / f"{label}_summary.json",
            {"n_valid_rows": 0, "n_errors": int(len(error_rows))},
        )
        return

    by_band = summarize_by_band(valid)
    by_row = summarize_by_row(valid)
    by_band.to_csv(out / f"{label}_summary_by_band.csv")
    by_row.to_csv(out / f"{label}_summary_by_galaxy.csv")

    summary = {
        "n_valid_comparisons": int(len(valid)),
        "n_valid_galaxies": int(by_row.shape[0]),
        "n_error_rows": int(len(error_rows)),
        "median_chi2": float(by_row["chi2"].median()),
        "median_reduced_chi2": float(by_row["reduced_chi2"].median()),
        "median_abs_residual_mag": float(
            valid["residual_mag_model_minus_observed"].abs().median()
        ),
    }
    if "delta_z_obs_minus_truth" in by_row:
        dz = by_row["delta_z_obs_minus_truth"].dropna()
        if not dz.empty:
            summary["median_delta_z_obs_minus_truth"] = float(dz.median())
            summary["mad_delta_z_obs_minus_truth"] = float(
                (dz - dz.median()).abs().median()
            )
    truth_metrics = parameter_truth_metrics(by_row.reset_index())
    if not truth_metrics.empty:
        truth_metrics.to_csv(out / f"{label}_truth_metrics.csv", index=False)
    residual_properties = residuals_by_property(by_row.reset_index())
    if not residual_properties.empty:
        residual_properties.to_csv(
            out / f"{label}_residuals_by_property.csv", index=False
        )
    write_json(out / f"{label}_summary.json", summary)

    if str(reporting_level).lower() != "full":
        return

    plot_batch_dashboard(valid, by_row, out / f"{label}_dashboard.png")
    plot_batch_residuals_by_band(valid, out / f"{label}_residuals_by_band.png")
    plot_batch_observed_vs_model(valid, out / f"{label}_observed_vs_model.png")
    plot_batch_redshift_truth(by_row, out / f"{label}_redshift_truth.png")
    plot_batch_parameter_truth(by_row, out / f"{label}_parameter_truth.png")
    plot_residuals_by_property(
        by_row.reset_index(), out / f"{label}_residuals_by_property.png"
    )
    plot_population_bias_heatmap(
        by_row.reset_index(), out / f"{label}_bias_heatmap.png"
    )


def write_fit_diagnostic_outputs(
    fits: pd.DataFrame,
    comparison: pd.DataFrame,
    config: dict[str, Any],
    out_dir: str | Path,
    label: str = "batch_fit",
    hyperparameters: pd.DataFrame | None = None,
) -> None:
    """Write fit audit tables that protect scientific interpretation."""
    out = ensure_dir(out_dir)
    audit = fit_parameter_audit(fits, config)
    if not audit.empty:
        audit.to_csv(out / f"{label}_parameter_audit.csv", index=False)
    components = fit_objective_components(fits, comparison, config, hyperparameters)
    if not components.empty:
        components.to_csv(out / f"{label}_objective_components.csv", index=False)


def fit_parameter_audit(fits: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Summarize whether reported fit columns were truly inferred."""
    if fits.empty:
        return pd.DataFrame()
    free = config.get("fit", {}).get("free_parameters", {}) or {}
    fixed = config.get("model", {}).get("fixed_parameters", {}) or {}
    injected = config.get("model", {}).get("parameter_columns", {}) or {}
    derived = {
        "t_obs_gyr",
        "formed_mass_msun",
        "log10_formed_mass_msun",
        "sfr_at_obs_msun_per_yr",
        "log10_sfr_at_obs",
    }
    rows = []
    for column in sorted(c for c in fits.columns if c.startswith("fit_")):
        name = column[4:]
        values = pd.to_numeric(fits[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        finite = values.dropna()
        source = _fit_parameter_source(name, free, fixed, injected, derived)
        warning_flags = _fit_parameter_warning_flags(name, source, finite, free)
        row: dict[str, Any] = {
            "parameter": name,
            "fit_column": column,
            "source": source,
            "is_free": bool(name in free),
            "is_fixed": bool(name in fixed and name not in free),
            "is_row_injected": bool(name in injected and name not in free),
            "is_derived": bool(name in derived),
            "n": int(len(finite)),
            "n_unique": int(finite.nunique()) if not finite.empty else 0,
            "warning_flags": ",".join(warning_flags),
        }
        if not finite.empty:
            row.update(
                {
                    "min": float(finite.min()),
                    "p16": float(finite.quantile(0.16)),
                    "median": float(finite.median()),
                    "p84": float(finite.quantile(0.84)),
                    "max": float(finite.max()),
                    "std": float(finite.std()) if len(finite) > 1 else 0.0,
                }
            )
        if name in free:
            spec = free[name] or {}
            row["initial"] = spec.get("initial")
            bounds = spec.get("bounds")
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                low = float(bounds[0])
                high = float(bounds[1])
                span = high - low
                row["lower_bound"] = low
                row["upper_bound"] = high
                if not finite.empty and span > 0:
                    row["fraction_near_lower_1pct"] = float(
                        ((finite - low).abs() <= 0.01 * span).mean()
                    )
                    row["fraction_near_upper_1pct"] = float(
                        ((high - finite).abs() <= 0.01 * span).mean()
                    )
        if name in injected:
            row["source_column"] = injected[name]
        rows.append(row)
    return pd.DataFrame(rows)


def fit_objective_components(
    fits: pd.DataFrame,
    comparison: pd.DataFrame,
    config: dict[str, Any],
    hyperparameters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Post-hoc objective decomposition from saved MAP rows."""
    if fits.empty:
        return pd.DataFrame()
    rows = fits.copy()
    out = pd.DataFrame({"row_index": rows["row_index"].astype(int)})
    if "chunk_index" in rows:
        out["chunk_index"] = rows["chunk_index"]
    out["photometric_chi2"] = _photometric_chi2_by_row(rows, comparison)
    physical = _physical_prior_components(rows, config)
    for name, values in physical.items():
        out[name] = values
    if hyperparameters is not None and not hyperparameters.empty:
        population = _population_prior_components(rows, hyperparameters)
        for name, values in population.items():
            out[name] = values
    for column in [
        "physical_gaussian_prior_penalty",
        "physical_beta_prior_penalty",
        "phz_prior_penalty",
        "population_gaussian_prior_penalty",
        "population_relation_prior_penalty",
    ]:
        if column not in out:
            out[column] = 0.0
    out["physical_prior_penalty"] = (
        out["physical_gaussian_prior_penalty"]
        + out["physical_beta_prior_penalty"]
        + out["phz_prior_penalty"]
    )
    out["population_prior_penalty"] = (
        out["population_gaussian_prior_penalty"]
        + out["population_relation_prior_penalty"]
    )
    out["approx_objective"] = (
        0.5 * out["photometric_chi2"]
        + out["physical_prior_penalty"]
        + out["population_prior_penalty"]
    )
    return out


def _fit_parameter_source(
    name: str,
    free: dict[str, Any],
    fixed: dict[str, Any],
    injected: dict[str, Any],
    derived: set[str],
) -> str:
    if name in free:
        return "free"
    if name in derived:
        return "derived"
    if name in injected:
        return "row_injected"
    if name in fixed:
        return "fixed"
    if name.startswith("z_obs_phz_") or name.endswith("_prior_sigma"):
        return "prior_context"
    return "reported_not_configured"


def _fit_parameter_warning_flags(
    name: str, source: str, finite: pd.Series, free: dict[str, Any]
) -> list[str]:
    flags: list[str] = []
    if source != "free" and source not in {"derived"}:
        flags.append("not_inferred_column")
    if source == "free" and len(finite) > 1 and finite.nunique() <= 1:
        flags.append("constant_free_parameter")
    spec = free.get(name) or {}
    bounds = spec.get("bounds")
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2 and not finite.empty:
        low = float(bounds[0])
        high = float(bounds[1])
        span = high - low
        if span > 0:
            if ((finite - low).abs() <= 0.01 * span).mean() > 0.1:
                flags.append("near_lower_bound_population")
            if ((high - finite).abs() <= 0.01 * span).mean() > 0.1:
                flags.append("near_upper_bound_population")
    return flags


def _photometric_chi2_by_row(fits: pd.DataFrame, comparison: pd.DataFrame) -> pd.Series:
    if not comparison.empty and {"row_index", "chi"}.issubset(comparison.columns):
        grouped = comparison.assign(
            _chi2=pd.to_numeric(comparison["chi"], errors="coerce") ** 2
        )
        chi2 = grouped.groupby("row_index")["_chi2"].sum()
        return fits["row_index"].map(chi2).fillna(fits.get("chi2", 0.0)).astype(float)
    if "chi2" in fits:
        return pd.to_numeric(fits["chi2"], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(fits)), index=fits.index)


def _physical_prior_components(
    fits: pd.DataFrame, config: dict[str, Any]
) -> dict[str, pd.Series]:
    priors = config.get("fit", {}).get("priors", {}) or {}
    free = config.get("fit", {}).get("free_parameters", {}) or {}
    gaussian = pd.Series(np.zeros(len(fits)), index=fits.index, dtype=float)
    beta = pd.Series(np.zeros(len(fits)), index=fits.index, dtype=float)
    phz = pd.Series(np.zeros(len(fits)), index=fits.index, dtype=float)
    for name, spec in priors.items():
        if name not in free:
            continue
        values = _fit_values(fits, name)
        if values is None:
            continue
        prior_type = str((spec or {}).get("type", "normal"))
        if prior_type in {"normal", "truncated_normal"}:
            loc = _prior_loc(fits, name, spec)
            scale = max(float((spec or {}).get("scale", 1.0)), 1.0e-6)
            gaussian += 0.5 * ((values - loc) / scale) ** 2 + np.log(scale)
        elif prior_type == "scaled_beta":
            bounds = free.get(name, {}).get("bounds", [0.0, 1.0])
            low, high = float(bounds[0]), float(bounds[1])
            scaled = ((values - low) / max(high - low, 1.0e-12)).clip(
                1.0e-6, 1 - 1.0e-6
            )
            alpha = max(float((spec or {}).get("alpha", 1.0)), 1.0e-6)
            beta_param = max(float((spec or {}).get("beta", 1.0)), 1.0e-6)
            beta += -(
                (alpha - 1.0) * np.log(scaled) + (beta_param - 1.0) * np.log1p(-scaled)
            )
        elif prior_type == "phz_interval" and name == "z_obs":
            phz += _phz_interval_penalty_numpy(values, fits, spec)
    return {
        "physical_gaussian_prior_penalty": gaussian,
        "physical_beta_prior_penalty": beta,
        "phz_prior_penalty": phz,
    }


def _population_prior_components(
    fits: pd.DataFrame, hyperparameters: pd.DataFrame
) -> dict[str, pd.Series]:
    gaussian = pd.Series(np.zeros(len(fits)), index=fits.index, dtype=float)
    relation = pd.Series(np.zeros(len(fits)), index=fits.index, dtype=float)
    relation_targets = set(
        hyperparameters.loc[
            hyperparameters.get("kind", pd.Series(dtype=str)).eq("relation"),
            "target_parameter",
        ]
        .dropna()
        .astype(str)
    )
    chunk_values = (
        fits["chunk_index"] if "chunk_index" in fits else pd.Series(0, index=fits.index)
    )
    for _, row in hyperparameters.iterrows():
        chunk_mask = chunk_values.eq(row.get("chunk_index", 0))
        if not chunk_mask.any():
            continue
        kind = row.get("kind")
        if kind == "gaussian":
            parameter = str(row.get("parameter"))
            if parameter in relation_targets:
                continue
            values = _fit_values(fits.loc[chunk_mask], parameter)
            if values is None:
                continue
            mu = float(row.get("population_mu", 0.0))
            sigma = max(float(row.get("population_sigma", 1.0)), 1.0e-6)
            gaussian.loc[chunk_mask] += 0.5 * ((values - mu) / sigma) ** 2 + np.log(
                sigma
            )
        elif kind == "relation":
            target = str(row.get("target_parameter"))
            predictor = str(row.get("predictor_parameter"))
            target_values = _fit_values(fits.loc[chunk_mask], target)
            predictor_values = _fit_values(fits.loc[chunk_mask], predictor)
            if target_values is None or predictor_values is None:
                continue
            pivot = float(row.get("population_pivot", 0.0))
            intercept = float(row.get("population_intercept", 0.0))
            slope = float(row.get("population_slope", 0.0))
            sigma = max(float(row.get("population_sigma", 1.0)), 1.0e-6)
            loc = intercept + slope * (predictor_values - pivot)
            relation.loc[chunk_mask] += 0.5 * (
                (target_values - loc) / sigma
            ) ** 2 + np.log(sigma)
    return {
        "population_gaussian_prior_penalty": gaussian,
        "population_relation_prior_penalty": relation,
    }


def _fit_values(fits: pd.DataFrame, parameter: str) -> pd.Series | None:
    for column in (f"fit_{parameter}", f"param_{parameter}", parameter):
        if column in fits:
            return pd.to_numeric(fits[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    return None


def _prior_loc(fits: pd.DataFrame, name: str, spec: dict[str, Any]) -> pd.Series:
    loc = spec.get("loc", 0.0)
    if loc == "from_base":
        values = _fit_values(fits, name)
        return (
            values
            if values is not None
            else pd.Series(np.zeros(len(fits)), index=fits.index)
        )
    return pd.Series(float(loc), index=fits.index)


def _phz_interval_penalty_numpy(
    z: pd.Series, fits: pd.DataFrame, spec: dict[str, Any]
) -> pd.Series:
    def column(name: str, fallback: str) -> pd.Series:
        param = str(spec.get(name, fallback))
        values = _fit_values(fits, param)
        if values is None:
            return pd.Series(np.nan, index=fits.index)
        return values

    min70 = column("min_70_parameter", "z_obs_phz_min_70")
    max70 = column("max_70_parameter", "z_obs_phz_max_70")
    min90 = column("min_90_parameter", "z_obs_phz_min_90")
    max90 = column("max_90_parameter", "z_obs_phz_max_90")
    min95 = column("min_95_parameter", "z_obs_phz_min_95")
    max95 = column("max_95_parameter", "z_obs_phz_max_95")
    valid = (
        min70.notna()
        & max70.notna()
        & min90.notna()
        & max90.notna()
        & min95.notna()
        & max95.notna()
        & (max70 > min70)
        & (max90 > min90)
        & (max95 > min95)
    )
    d70 = np.maximum(np.maximum(min70 - z, z - max70), 0.0)
    d90 = np.maximum(np.maximum(min90 - z, z - max90), 0.0)
    d95 = np.maximum(np.maximum(min95 - z, z - max95), 0.0)
    width70 = np.where(z < min70, min70 - min90, max90 - max70)
    width90 = np.where(z < min90, min90 - min95, max95 - max90)
    width70 = np.maximum(width70, 1.0e-4)
    width90 = np.maximum(width90, 1.0e-4)
    tail = max(float(spec.get("tail_scale", 0.05)), 1.0e-4)
    penalty_70_90 = 0.5 * (d70 / width70) ** 2
    penalty_90_95 = 0.5 + (d90 / width90) ** 2
    penalty_tail = 1.5 + 2.0 * (d95 / tail) ** 2
    penalty = np.where(
        d70 <= 0.0,
        0.0,
        np.where(
            d90 <= 0.0, penalty_70_90, np.where(d95 <= 0.0, penalty_90_95, penalty_tail)
        ),
    )
    penalty = pd.Series(penalty, index=fits.index).where(valid, 0.0)
    return penalty * max(float(spec.get("weight", 1.0)), 0.0)


def plot_population_bias_heatmap(by_row: pd.DataFrame, path: str | Path) -> None:
    """Plot a heatmap of reduced chi2 in the Redshift-Mass plane."""
    z_col = "redshift_truth" if "redshift_truth" in by_row else "z_obs"
    m_col = (
        "catalog_log_stellar_mass"
        if "catalog_log_stellar_mass" in by_row
        else "fit_log10_formed_mass_msun"
    )

    if z_col not in by_row or m_col not in by_row or "reduced_chi2" not in by_row:
        return

    work = (
        by_row[[z_col, m_col, "reduced_chi2"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(work) < 10:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        # Create 2D bins
        z_bins = np.linspace(work[z_col].min(), work[z_col].max(), 12)
        m_bins = np.linspace(work[m_col].min(), work[m_col].max(), 12)

        stats = (
            pd.DataFrame(
                {
                    "z_bin": pd.cut(work[z_col], bins=z_bins),
                    "m_bin": pd.cut(work[m_col], bins=m_bins),
                    "chi2": work["reduced_chi2"],
                }
            )
            .groupby(["z_bin", "m_bin"], observed=True)["chi2"]
            .median()
            .unstack()
        )

        im = ax.pcolormesh(
            m_bins,
            z_bins,
            stats.to_numpy(),
            cmap="viridis",
            shading="flat",
            norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=10.0),
        )
        fig.colorbar(im, ax=ax, label="median reduced chi2")
        ax.set_xlabel(_parameter_display_label(m_col))
        ax.set_ylabel(_parameter_display_label(z_col))
        ax.set_title("Population Fit Quality Map")
    except Exception:
        ax.text(0.5, 0.5, "Heatmap failed (insufficient coverage)", ha="center")

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def summarize_by_band(valid: pd.DataFrame) -> pd.DataFrame:
    return valid.groupby("band").agg(
        n=("row_index", "count"),
        effective_wavelength_angstrom=("effective_wavelength_angstrom", "median"),
        mean_residual_mag=("residual_mag_model_minus_observed", "mean"),
        median_residual_mag=("residual_mag_model_minus_observed", "median"),
        std_residual_mag=("residual_mag_model_minus_observed", "std"),
        rms_residual_mag=(
            "residual_mag_model_minus_observed",
            lambda x: float(np.sqrt(np.nanmean(x**2))),
        ),
        mean_abs_residual_mag=(
            "residual_mag_model_minus_observed",
            lambda x: float(np.nanmean(np.abs(x))),
        ),
        median_flux_ratio=("flux_ratio_model_over_observed", "median"),
        mean_chi=("chi", "mean"),
    )


def summarize_by_row(valid: pd.DataFrame) -> pd.DataFrame:
    context_columns = [
        col
        for col in valid.columns
        if col in {"z_obs", "redshift_truth", "delta_z_obs_minus_truth"}
        or col in {"z_obs_source", "redshift_truth_source"}
        or col.startswith("param_")
        or col.startswith("fit_")
        or col.startswith("truth_")
        or col.startswith("delta_")
        or col.startswith("catalog_")
    ]
    aggregations: dict[str, tuple[str, Any]] = {
        "n_bands": ("band", "count"),
        "chi2": ("chi", lambda x: float(np.nansum(x**2))),
        "mean_residual_mag": ("residual_mag_model_minus_observed", "mean"),
        "median_residual_mag": ("residual_mag_model_minus_observed", "median"),
        "rms_residual_mag": (
            "residual_mag_model_minus_observed",
            lambda x: float(np.sqrt(np.nanmean(x**2))),
        ),
        "mean_abs_residual_mag": (
            "residual_mag_model_minus_observed",
            lambda x: float(np.nanmean(np.abs(x))),
        ),
    }
    for col in context_columns:
        aggregations[col] = (col, "first")
    by_row = valid.groupby("row_index").agg(**aggregations)
    derived_columns = {"reduced_chi2": by_row["chi2"] / by_row["n_bands"].clip(lower=1)}

    for col in list(by_row.columns):
        if col.startswith("truth_"):
            param_name = col[6:]
            fit_col = f"fit_{param_name}"
            if fit_col in by_row.columns:
                derived_columns[f"delta_fit_{param_name}_minus_truth"] = (
                    by_row[fit_col] - by_row[col]
                )

    by_row = pd.concat([by_row, pd.DataFrame(derived_columns)], axis=1).copy()

    return by_row


def residuals_by_property(by_row: pd.DataFrame) -> pd.DataFrame:
    """Summarize row residuals against catalog/fit properties."""
    rows: list[dict[str, Any]] = []
    property_specs = [
        ("redshift_truth", "redshift"),
        ("z_obs", "redshift"),
        ("z_true_gal", "truth redshift gal"),
        ("catalog_log_stellar_mass", "catalog log stellar mass"),
        ("fit_log10_formed_mass_msun", "fit log formed mass"),
        ("fit_log10_sfr_at_obs", "fit log sfr at obs"),
        ("catalog_color_kind", "color_kind"),
    ]
    for column, label in property_specs:
        if column not in by_row:
            continue
        values = by_row[column]
        if label == "color_kind":
            groups = values.astype("string")
        else:
            numeric = pd.to_numeric(values, errors="coerce")
            finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
            if finite.nunique() < 2:
                continue
            try:
                # Use a mix of fixed and quantile bins for better scientific grouping
                if label == "redshift":
                    groups = pd.cut(numeric, bins=[0, 0.5, 1.0, 1.5, 2.5, 4.0, 6.0])
                else:
                    groups = pd.qcut(
                        numeric, q=min(6, finite.nunique()), duplicates="drop"
                    )
            except ValueError:
                continue
        work = by_row.assign(_group=groups)
        for group, subset in work.groupby("_group", dropna=True):
            row = _residual_property_row(subset, label, column, str(group))
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def _residual_property_row(
    subset: pd.DataFrame, property_name: str, source_column: str, group_label: str
) -> dict[str, Any]:
    residual = subset["mean_residual_mag"].replace([np.inf, -np.inf], np.nan).dropna()
    reduced = subset["reduced_chi2"].replace([np.inf, -np.inf], np.nan).dropna()
    if residual.empty:
        return {}
    row: dict[str, Any] = {
        "property": property_name,
        "source_column": source_column,
        "group": group_label,
        "n_galaxies": int(len(residual)),
        "mean_residual_mag": float(residual.mean()),
        "median_residual_mag": float(residual.median()),
        "mean_abs_residual_mag": float(residual.abs().mean()),
        "median_abs_residual_mag": float(residual.abs().median()),
    }
    if not reduced.empty:
        row["median_reduced_chi2"] = float(reduced.median())
    return row


def plot_flux_distributions(
    df: pd.DataFrame, columns: list[str], path: str | Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in columns:
        values = df[col].to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            ax.hist(np.log10(values), bins=60, histtype="step", lw=1.4, label=col)
    ax.set_xlabel("log10 flux [Fnu cgs]")
    ax.set_ylabel("count")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_color_distributions(
    df: pd.DataFrame, columns: list[str], path: str | Path
) -> None:
    if len(columns) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for left, right in zip(columns[:-1], columns[1:], strict=True):
        a = df[left].to_numpy(dtype=float)
        b = df[right].to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if mask.any():
            color = -2.5 * np.log10(a[mask] / b[mask])
            ax.hist(color, bins=60, histtype="step", lw=1.2, label=f"{left}-{right}")
    ax.set_xlabel("AB color [mag]")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_redshift_distributions(
    df: pd.DataFrame, redshift_config: dict[str, Any], path: str | Path
) -> None:
    z_col = redshift_config.get("column")
    truth_col = redshift_config.get("truth_column")
    if not z_col or z_col not in df:
        return

    fig, axes = plt.subplots(1, 2 if truth_col in df else 1, figsize=(10, 4))
    axes = np.atleast_1d(axes)
    z = df[z_col].to_numpy(dtype=float)
    z = z[np.isfinite(z)]
    axes[0].hist(z, bins=70, histtype="stepfilled", alpha=0.65, label=z_col)
    if truth_col in df:
        zt = df[truth_col].to_numpy(dtype=float)
        zt = zt[np.isfinite(zt)]
        axes[0].hist(zt, bins=70, histtype="step", lw=1.4, label=truth_col)
    axes[0].set_xlabel("redshift")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    if truth_col in df:
        work = df[[z_col, truth_col]].dropna()
        delta = work[z_col] - work[truth_col]
        axes[1].hist(delta, bins=80, histtype="stepfilled", alpha=0.7)
        axes[1].axvline(0, color="black", lw=1)
        axes[1].set_xlabel(f"{z_col} - {truth_col}")
        axes[1].set_ylabel("count")
        axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_physical_parameters_distributions(df: pd.DataFrame, path: str | Path) -> None:
    params_map = {
        "log10_metallicity_true": "Ground Truth log10(Z)",
        "sfr_true": "Ground Truth SFR [Msun/yr]",
        "log_sfr_true": "Ground Truth log10(SFR)",
        "dust_ebv_true": "Ground Truth E(B-V)",
    }
    valid_params = [p for p in params_map if p in df.columns]
    if not valid_params:
        return

    fig, axes = plt.subplots(1, len(valid_params), figsize=(4 * len(valid_params), 4))
    axes = np.atleast_1d(axes)

    for ax, param in zip(axes, valid_params, strict=True):
        val = df[param].to_numpy(dtype=float)
        val = val[np.isfinite(val)]
        ax.hist(val, bins=70, histtype="stepfilled", alpha=0.7)
        ax.set_xlabel(params_map[param])
        ax.set_ylabel("galaxies")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_sed(result: ModelResult, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    mask = (
        np.isfinite(result.wave)
        & np.isfinite(result.dusted_rest_sed)
        & (result.wave >= 800)
        & (result.wave <= 30_000)
        & (result.rest_sed > 0)
        & (result.dusted_rest_sed > 0)
    )
    ax.plot(
        result.wave[mask],
        result.rest_sed[mask],
        label="Intrinsic rest SED",
        lw=1.1,
        alpha=0.65,
    )
    ax.plot(
        result.wave[mask],
        result.dusted_rest_sed[mask],
        label="Dust-attenuated rest SED",
        lw=1.4,
    )
    z_obs = result.parameters.get("z_obs", np.nan)
    ymin, ymax = _positive_axis_limits(result.dusted_rest_sed[mask])
    for band, values in result.photometry.items():
        wave_filter_obs = np.asarray(
            values.get("filter_wave_angstrom", []), dtype=float
        )
        transmission = np.asarray(values.get("filter_transmission", []), dtype=float)
        passband_mask = (
            np.isfinite(wave_filter_obs)
            & np.isfinite(transmission)
            & (transmission > 0)
        )
        if passband_mask.any() and np.isfinite(z_obs):
            wave_filter_rest = wave_filter_obs[passband_mask] / (1.0 + z_obs)
            scaled = ymin * (ymax / ymin) ** (
                0.06
                + 0.16
                * transmission[passband_mask]
                / np.nanmax(transmission[passband_mask])
            )
            ax.fill_between(
                wave_filter_rest,
                ymin,
                scaled,
                alpha=0.16,
                lw=0,
                label=f"{band.replace('euclid_', '')} passband",
            )
        wave_rest = values["effective_wavelength_angstrom"] / (1.0 + z_obs)
        if np.isfinite(wave_rest) and 800 <= wave_rest <= 30_000:
            ax.axvline(wave_rest, color="black", lw=0.7, alpha=0.18)
            ax.text(
                wave_rest,
                0.98,
                band.replace("euclid_", ""),
                rotation=90,
                va="top",
                ha="right",
                transform=ax.get_xaxis_transform(),
                fontsize=7,
            )
    ax.set_xlabel("rest-frame wavelength [Angstrom]")
    ax.set_ylabel("Lsun / Hz")
    ax.set_yscale("log")
    ax.set_title(f"DSPS rest SED, z={z_obs:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _positive_axis_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        return 1.0, 10.0
    return float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99.5))


def plot_photometry_comparison(comparison: pd.DataFrame, path: str | Path) -> None:
    work = comparison.sort_values("effective_wavelength_angstrom").reset_index(
        drop=True
    )
    x = work["effective_wavelength_angstrom"].to_numpy(dtype=float) / 10_000.0

    fig, (ax_mag, ax_resid) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax_mag.errorbar(
        x,
        work["observed_mag_ab"],
        yerr=work["sigma_mag"],
        fmt="o",
        ms=6,
        capsize=3,
        label="Simulated catalog",
    )
    ax_mag.plot(x, work["model_mag_ab"], marker="s", lw=1.4, label="DSPS model")
    for xi, yi, label in zip(x, work["observed_mag_ab"], work["band"], strict=True):
        ax_mag.annotate(
            label.replace("euclid_", ""),
            (xi, yi),
            textcoords="offset points",
            xytext=(3, 5),
            fontsize=8,
        )
    ax_mag.set_ylabel("AB magnitude")
    ax_mag.invert_yaxis()
    ax_mag.legend(fontsize=8)
    ax_mag.grid(alpha=0.25)

    ax_resid.axhline(0, color="black", lw=1)
    ax_resid.errorbar(
        x,
        work["residual_mag_model_minus_observed"],
        yerr=work["sigma_mag"],
        fmt="o",
        capsize=3,
    )
    ax_resid.set_xlabel("observed-frame effective wavelength [micron]")
    ax_resid.set_ylabel("model - obs [mag]")
    ax_resid.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_fit_trace(trace: pd.DataFrame, path: str | Path) -> None:
    if trace.empty:
        return
    y_col = (
        "chi2"
        if "chi2" in trace
        else "mean_chi2_or_loss" if "mean_chi2_or_loss" in trace else None
    )
    if y_col is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(trace)), trace[y_col], lw=1)
    ax.set_xlabel("evaluation")
    ax.set_ylabel(y_col)
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_posterior_predictive(mcmc_result: Any, path: str | Path) -> None:
    rows = []
    mags = np.asarray(mcmc_result.posterior_model_mags)
    for band_index, band in enumerate(mcmc_result.band_names):
        values = mags[:, band_index]
        rows.append(
            {
                "band": band,
                "observed_mag_ab": float(mcmc_result.observed_mag[band_index]),
                "sigma_mag": float(mcmc_result.sigma_mag[band_index]),
                "model_mag_q05": float(np.quantile(values, 0.05)),
                "model_mag_q16": float(np.quantile(values, 0.16)),
                "model_mag_median": float(np.quantile(values, 0.50)),
                "model_mag_q84": float(np.quantile(values, 0.84)),
                "model_mag_q95": float(np.quantile(values, 0.95)),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_mcmc_traces(samples: pd.DataFrame, path: str | Path) -> None:
    if samples.empty:
        return
    fig, axes = plt.subplots(
        len(samples.columns),
        1,
        figsize=(8, max(2.2 * len(samples.columns), 3)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, samples.columns, strict=True):
        ax.plot(samples[col].to_numpy(dtype=float), lw=0.8)
        ax.set_ylabel(col)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("posterior sample")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_corner(samples: pd.DataFrame, path: str | Path) -> None:
    if samples.empty:
        return
    varying = samples.loc[:, samples.nunique(dropna=True) > 1]
    if varying.shape[1] < 2:
        return
    if varying.shape[0] <= varying.shape[1]:
        return
    try:
        import corner
    except ImportError:
        return
    try:
        fig = corner.corner(
            varying.to_numpy(dtype=float),
            labels=list(varying.columns),
            show_titles=True,
        )
    except (AssertionError, ValueError):
        return
    fig.savefig(path, dpi=170)
    plt.close(fig)


def posterior_comparable_frame(
    samples: pd.DataFrame,
    derived: pd.DataFrame,
    truth_values: dict[str, Any],
) -> pd.DataFrame:
    """Build posterior columns that have like-for-like truth/proxy values."""
    data = {}
    if "truth_log10_sfr_at_obs" in truth_values and "log10_sfr_at_obs" in derived:
        data["log10_sfr_at_obs"] = derived["log10_sfr_at_obs"].to_numpy(dtype=float)
    if (
        "truth_log10_formed_mass_msun" in truth_values
        and "log10_formed_mass_msun" in derived
    ):
        data["log10_formed_mass_msun"] = derived["log10_formed_mass_msun"].to_numpy(
            dtype=float
        )
    if "truth_dust_av" in truth_values and "dust_av" in samples:
        data["dust_av"] = samples["dust_av"].to_numpy(dtype=float)
    if "truth_log10_metallicity" in truth_values and "log10_metallicity" in samples:
        data["log10_metallicity"] = samples["log10_metallicity"].to_numpy(dtype=float)
    for column in samples.columns:
        truth_key = f"truth_{column}"
        if truth_key in truth_values and column not in data:
            data[column] = samples[column].to_numpy(dtype=float)
    return pd.DataFrame(data)


def plot_corner_with_truth(
    samples: pd.DataFrame,
    truth_values: dict[str, Any],
    path: str | Path,
) -> None:
    if samples.empty:
        return
    varying = samples.loc[:, samples.nunique(dropna=True) > 1]
    if varying.shape[1] < 2 or varying.shape[0] <= varying.shape[1]:
        return
    truth_by_column = {}
    for column in varying.columns:
        value = truth_values.get(f"truth_{column}")
        if value is not None and np.isfinite(value):
            truth_by_column[column] = float(value)
    ranges = [
        _corner_range_with_truth(
            varying[column].to_numpy(dtype=float), truth_by_column.get(column)
        )
        for column in varying.columns
    ]
    try:
        import corner
    except ImportError:
        return
    try:
        fig = corner.corner(
            varying.to_numpy(dtype=float),
            labels=[_posterior_display_label(name) for name in varying.columns],
            show_titles=True,
            range=ranges,
            title_kwargs={"fontsize": 11},
        )
    except (AssertionError, ValueError):
        return
    _annotate_corner_truth(fig, list(varying.columns), truth_by_column, truth_values)
    fig.suptitle(
        "Posterior with catalog truth/proxy markers",
        y=0.995,
        fontsize=13,
    )
    fig.subplots_adjust(top=0.92)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _corner_range_with_truth(
    values: np.ndarray, truth_value: float | None
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if truth_value is not None and np.isfinite(truth_value):
        lo = min(lo, float(truth_value))
        hi = max(hi, float(truth_value))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (0.0, 1.0)
    if lo == hi:
        pad = max(abs(lo) * 0.05, 1.0e-3)
    else:
        pad = max((hi - lo) * 0.08, 1.0e-3)
    return (lo - pad, hi + pad)


def _annotate_corner_truth(
    fig: Any,
    columns: list[str],
    truth_by_column: dict[str, float],
    truth_values: dict[str, Any],
) -> None:
    n_dim = len(columns)
    axes = np.asarray(fig.axes).reshape((n_dim, n_dim))
    truth_color = "#d94f70"
    for row, y_name in enumerate(columns):
        y_truth = truth_by_column.get(y_name)
        for col, x_name in enumerate(columns):
            if row < col:
                continue
            ax = axes[row, col]
            x_truth = truth_by_column.get(x_name)
            if x_truth is not None:
                ax.axvline(x_truth, color=truth_color, lw=2.0, ls="--", zorder=8)
            if row > col and y_truth is not None:
                ax.axhline(y_truth, color=truth_color, lw=2.0, ls="--", zorder=8)
            if row == col and x_truth is not None:
                ax.text(
                    0.98,
                    0.86,
                    _truth_axis_label(x_name, x_truth, truth_values),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.5,
                    color=truth_color,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": truth_color,
                        "alpha": 0.85,
                        "pad": 2,
                    },
                )


def _truth_axis_label(
    parameter: str, value: float, truth_values: dict[str, Any]
) -> str:
    source = truth_values.get(f"truth_source_{parameter}")
    kind = truth_values.get(f"truth_kind_{parameter}", "direct")
    prefix = "proxy" if kind == "proxy" else "truth"
    if source:
        return f"{prefix} = {value:.3g} ({source})"
    return f"{prefix} = {value:.3g}"


def _posterior_display_label(parameter: str) -> str:
    labels = {
        "z_obs": "DSPS redshift",
        "log10_sfr_at_obs": "DSPS log10 SFR(t_obs)",
        "log10_sfr": "DSPS log10 SFH amplitude",
        "log10_formed_mass_msun": "DSPS log10 formed mass",
        "dust_av": "DSPS dust A_V",
        "log10_metallicity": "DSPS log10 stellar Z",
    }
    return labels.get(parameter, parameter)


def plot_posterior_predictive(mcmc_result: Any, path: str | Path) -> None:
    mags = np.asarray(mcmc_result.posterior_model_mags)
    x = np.arange(len(mcmc_result.band_names))
    med = np.quantile(mags, 0.50, axis=0)
    lo = np.quantile(mags, 0.16, axis=0)
    hi = np.quantile(mags, 0.84, axis=0)
    obs = np.asarray(mcmc_result.observed_mag, dtype=float)
    sig = np.asarray(mcmc_result.sigma_mag, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(x, obs, yerr=sig, fmt="o", capsize=3, label="catalog")
    ax.errorbar(
        x, med, yerr=[med - lo, hi - med], fmt="s", capsize=3, label="posterior model"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [name.replace("euclid_", "") for name in mcmc_result.band_names], rotation=20
    )
    ax.set_ylabel("AB magnitude")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_batch_posterior_intervals(summary: pd.DataFrame, path: str | Path) -> None:
    if summary.empty or not {"row_index", "parameter", "median", "q16", "q84"}.issubset(
        summary.columns
    ):
        return
    parameters = list(summary["parameter"].drop_duplicates())
    fig, axes = plt.subplots(
        len(parameters), 1, figsize=(9, max(2.4 * len(parameters), 3)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, parameter in zip(axes, parameters, strict=True):
        work = summary[summary["parameter"] == parameter].sort_values("row_index")
        x = work["row_index"].to_numpy(dtype=float)
        median = work["median"].to_numpy(dtype=float)
        q16 = work["q16"].to_numpy(dtype=float)
        q84 = work["q84"].to_numpy(dtype=float)
        ax.errorbar(
            x, median, yerr=[median - q16, q84 - median], fmt="o", ms=4, capsize=2
        )
        ax.set_ylabel(parameter)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("catalog row index")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_batch_posterior_predictive(predictive: pd.DataFrame, path: str | Path) -> None:
    if predictive.empty or "residual_mag_median_model_minus_observed" not in predictive:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for band in ordered_bands(predictive):
        work = predictive[predictive["band"] == band]
        ax.scatter(
            work["observed_mag_ab"],
            work["residual_mag_median_model_minus_observed"],
            s=14,
            alpha=0.65,
            label=band.replace("euclid_", ""),
        )
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("observed AB magnitude")
    ax.set_ylabel("posterior median model - observed [mag]")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_batch_mcmc_diagnostics(diagnostics: pd.DataFrame, path: str | Path) -> None:
    if diagnostics.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if "n_divergent" in diagnostics:
        axes[0].bar(diagnostics["row_index"].astype(str), diagnostics["n_divergent"])
        axes[0].set_ylabel("divergences")
    else:
        axes[0].axis("off")
    if "mean_accept_prob" in diagnostics:
        axes[1].bar(
            diagnostics["row_index"].astype(str), diagnostics["mean_accept_prob"]
        )
        axes[1].axhline(0.8, color="black", lw=1, alpha=0.4)
        axes[1].set_ylabel("mean accept prob")
        axes[1].set_ylim(0, 1)
    else:
        axes[1].axis("off")
    for ax in axes:
        ax.set_xlabel("catalog row index")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_workflow_comparison(
    map_fits: pd.DataFrame,
    population_fits: pd.DataFrame,
    hmc_summary: pd.DataFrame,
    hmc_diagnostics: pd.DataFrame,
    free_parameters: list[str],
    out_dir: str | Path,
    hmc_samples: pd.DataFrame | None = None,
) -> None:
    out = ensure_dir(out_dir)
    map_pop_params = workflow_parameter_comparison(
        map_fits, population_fits, free_parameters
    )
    map_pop_fits = workflow_fit_comparison(map_fits, population_fits)
    hmc_compare = workflow_hmc_comparison(
        map_fits, population_fits, hmc_summary, free_parameters
    )

    map_pop_params.to_csv(out / "map_vs_population_parameters.csv", index=False)
    map_pop_fits.to_csv(out / "map_vs_population_fit_quality.csv", index=False)
    if not hmc_compare.empty:
        hmc_compare.to_csv(out / "hmc_vs_map_population.csv", index=False)
    if not hmc_diagnostics.empty:
        hmc_diagnostics.to_csv(out / "hmc_diagnostics.csv", index=False)

    summary = {
        "n_map_galaxies": (
            int(map_fits["row_index"].nunique()) if "row_index" in map_fits else 0
        ),
        "n_population_galaxies": (
            int(population_fits["row_index"].nunique())
            if "row_index" in population_fits
            else 0
        ),
        "n_hmc_galaxies": (
            int(hmc_summary["row_index"].nunique()) if "row_index" in hmc_summary else 0
        ),
    }
    if "delta_chi2_population_minus_map" in map_pop_fits:
        summary["median_delta_chi2_population_minus_map"] = float(
            map_pop_fits["delta_chi2_population_minus_map"].median()
        )
    if "n_divergent" in hmc_diagnostics:
        summary["hmc_total_divergences"] = int(hmc_diagnostics["n_divergent"].sum())
    write_json(out / "workflow_comparison_summary.json", summary)

    plot_map_population_parameters(
        map_pop_params, out / "map_vs_population_parameters.png"
    )
    plot_map_population_chi2(map_pop_fits, out / "map_vs_population_chi2.png")
    plot_hmc_map_population(hmc_compare, out / "hmc_vs_map_population.png")
    plot_workflow_parameter_corners(
        map_fits,
        population_fits,
        hmc_summary,
        hmc_samples,
        free_parameters,
        out,
    )


def workflow_parameter_comparison(
    map_fits: pd.DataFrame, population_fits: pd.DataFrame, free_parameters: list[str]
) -> pd.DataFrame:
    rows = []
    merged = map_fits.merge(
        population_fits, on="row_index", suffixes=("_map", "_population")
    )
    for _, row in merged.iterrows():
        for parameter in free_parameters:
            map_col = f"fit_{parameter}_map"
            pop_col = f"fit_{parameter}_population"
            if map_col in row and pop_col in row:
                rows.append(
                    {
                        "row_index": int(row["row_index"]),
                        "parameter": parameter,
                        "map_value": float(row[map_col]),
                        "population_value": float(row[pop_col]),
                        "delta_population_minus_map": float(
                            row[pop_col] - row[map_col]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def workflow_fit_comparison(
    map_fits: pd.DataFrame, population_fits: pd.DataFrame
) -> pd.DataFrame:
    cols = ["row_index", "chi2", "reduced_chi2", "gradient_norm"]
    left = map_fits[[col for col in cols if col in map_fits]].copy()
    right = population_fits[[col for col in cols if col in population_fits]].copy()
    merged = left.merge(right, on="row_index", suffixes=("_map", "_population"))
    if {"chi2_map", "chi2_population"}.issubset(merged.columns):
        merged["delta_chi2_population_minus_map"] = (
            merged["chi2_population"] - merged["chi2_map"]
        )
    if {"reduced_chi2_map", "reduced_chi2_population"}.issubset(merged.columns):
        merged["delta_reduced_chi2_population_minus_map"] = (
            merged["reduced_chi2_population"] - merged["reduced_chi2_map"]
        )
    return merged


def workflow_hmc_comparison(
    map_fits: pd.DataFrame,
    population_fits: pd.DataFrame,
    hmc_summary: pd.DataFrame,
    free_parameters: list[str],
) -> pd.DataFrame:
    if hmc_summary.empty:
        return pd.DataFrame()
    rows = []
    map_by_row = map_fits.drop_duplicates("row_index").set_index("row_index")
    pop_by_row = population_fits.drop_duplicates("row_index").set_index("row_index")
    for _, row in hmc_summary.iterrows():
        parameter = row.get("parameter")
        row_index = int(row["row_index"])
        if parameter not in free_parameters or row_index not in map_by_row.index:
            continue
        map_value = float(map_by_row.loc[row_index, f"fit_{parameter}"])
        population_value = (
            float(pop_by_row.loc[row_index, f"fit_{parameter}"])
            if row_index in pop_by_row.index
            else np.nan
        )
        median = float(row["median"])
        rows.append(
            {
                "row_index": row_index,
                "parameter": parameter,
                "hmc_q16": float(row["q16"]),
                "hmc_median": median,
                "hmc_q84": float(row["q84"]),
                "map_value": map_value,
                "population_value": population_value,
                "delta_map_minus_hmc_median": map_value - median,
                "delta_population_minus_hmc_median": population_value - median,
            }
        )
    return pd.DataFrame(rows)


def plot_workflow_parameter_corners(
    map_fits: pd.DataFrame,
    population_fits: pd.DataFrame,
    hmc_summary: pd.DataFrame,
    hmc_samples: pd.DataFrame | None,
    free_parameters: list[str],
    out: Path,
) -> None:
    plot_corner(
        _fit_parameter_frame(map_fits, free_parameters),
        out / "corner_map_parameters.png",
    )
    plot_corner(
        _fit_parameter_frame(population_fits, free_parameters),
        out / "corner_population_parameters.png",
    )

    if not hmc_summary.empty:
        hmc_medians = hmc_summary.pivot(
            index="row_index", columns="parameter", values="median"
        )
        hmc_medians = hmc_medians[
            [name for name in free_parameters if name in hmc_medians.columns]
        ]
        plot_corner(
            hmc_medians.reset_index(drop=True), out / "corner_hmc_posterior_medians.png"
        )

    if hmc_samples is None or hmc_samples.empty:
        return

    sample_columns = [name for name in free_parameters if name in hmc_samples.columns]
    if not sample_columns:
        return
    plot_corner(hmc_samples[sample_columns], out / "corner_hmc_all_samples.png")
    individual_dir = ensure_dir(out / "hmc_individual_corners")
    for row_index, group in hmc_samples.groupby("row_index"):
        plot_corner(
            group[sample_columns],
            individual_dir / f"corner_hmc_row_{int(row_index)}.png",
        )


def _fit_parameter_frame(
    fits: pd.DataFrame, free_parameters: list[str]
) -> pd.DataFrame:
    data = {}
    for parameter in free_parameters:
        col = f"fit_{parameter}"
        if col in fits:
            data[parameter] = fits[col].to_numpy(dtype=float)
    return pd.DataFrame(data)


def _truth_parameter_frame(
    fits: pd.DataFrame, free_parameters: list[str]
) -> pd.DataFrame:
    data = {}
    for parameter in free_parameters:
        col = f"truth_{parameter}"
        if col in fits:
            data[parameter] = fits[col].to_numpy(dtype=float)
    return pd.DataFrame(data)


def paired_fit_truth_frames(fits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned frames for parameters with both ``fit_`` and ``truth_`` columns."""
    inferred = {}
    truth = {}
    for col in fits.columns:
        if not col.startswith("truth_"):
            continue
        parameter = col[6:]
        fit_col = f"fit_{parameter}"
        if fit_col in fits:
            inferred[parameter] = fits[fit_col].to_numpy(dtype=float)
            truth[parameter] = fits[col].to_numpy(dtype=float)
    return pd.DataFrame(inferred), pd.DataFrame(truth)


def plot_corner_overlay(
    inferred: pd.DataFrame, truth: pd.DataFrame, path: str | Path
) -> None:
    common = [col for col in inferred.columns if col in truth.columns]
    if len(common) < 2:
        return
    inferred = inferred[common].replace([np.inf, -np.inf], np.nan)
    truth = truth[common].replace([np.inf, -np.inf], np.nan)
    valid = inferred.notna().all(axis=1) & truth.notna().all(axis=1)
    inferred = inferred.loc[valid]
    truth = truth.loc[valid]
    if inferred.empty or truth.empty:
        return
    varying = [
        col
        for col in common
        if pd.concat([inferred[col], truth[col]]).nunique(dropna=True) > 1
    ]
    if len(varying) < 2 or len(inferred) <= len(varying) or len(truth) <= len(varying):
        return
    inferred = inferred[varying]
    truth = truth[varying]
    ranges = []
    for col in varying:
        combined = pd.concat([inferred[col], truth[col]])
        lo, hi = float(combined.min()), float(combined.max())
        pad = 0.03 * (hi - lo)
        ranges.append((lo - pad, hi + pad))
    try:
        import corner
    except ImportError:
        return

    inferred_color = "#8fbcd4"
    truth_color = "#e6a0a8"
    try:
        fig = corner.corner(
            inferred.to_numpy(dtype=float),
            labels=[_parameter_display_label(name) for name in varying],
            range=ranges,
            show_titles=True,
            color=inferred_color,
            plot_datapoints=True,
            fill_contours=True,
            hist_kwargs={
                "density": True,
                "color": inferred_color,
                "alpha": 0.55,
            },
            contour_kwargs={"colors": inferred_color, "linewidths": 1.0},
        )
        corner.corner(
            truth.to_numpy(dtype=float),
            labels=[_parameter_display_label(name) for name in varying],
            fig=fig,
            range=ranges,
            show_titles=False,
            color=truth_color,
            plot_datapoints=False,
            fill_contours=False,
            hist_kwargs={
                "density": True,
                "histtype": "step",
                "linewidth": 1.6,
                "color": truth_color,
            },
            contour_kwargs={"colors": truth_color, "linewidths": 1.4},
        )
    except (AssertionError, ValueError):
        return
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], color=inferred_color, lw=3, label="inferred"),
            Line2D([0], [0], color=truth_color, lw=3, label="truth/proxy"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=10,
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_map_population_parameters(comparison: pd.DataFrame, path: str | Path) -> None:
    if comparison.empty:
        return
    parameters = list(comparison["parameter"].drop_duplicates())
    fig, axes = plt.subplots(
        1, len(parameters), figsize=(4 * len(parameters), 4), squeeze=False
    )
    for ax, parameter in zip(axes[0], parameters, strict=True):
        work = comparison[comparison["parameter"] == parameter]
        ax.scatter(work["map_value"], work["population_value"], s=12, alpha=0.5)
        values = (
            pd.concat([work["map_value"], work["population_value"]])
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if not values.empty:
            lo, hi = float(values.min()), float(values.max())
            ax.plot([lo, hi], [lo, hi], color="black", lw=1)
        ax.set_title(parameter)
        ax.set_xlabel("independent MAP")
        ax.set_ylabel("population MAP")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_map_population_chi2(comparison: pd.DataFrame, path: str | Path) -> None:
    if comparison.empty or not {"reduced_chi2_map", "reduced_chi2_population"}.issubset(
        comparison.columns
    ):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(
        comparison["reduced_chi2_map"],
        comparison["reduced_chi2_population"],
        s=12,
        alpha=0.5,
    )
    values = (
        pd.concat(
            [comparison["reduced_chi2_map"], comparison["reduced_chi2_population"]]
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if not values.empty:
        lo, hi = float(values.min()), float(values.max())
        axes[0].plot([lo, hi], [lo, hi], color="black", lw=1)
    axes[0].set_xlabel("independent MAP reduced chi2")
    axes[0].set_ylabel("population MAP reduced chi2")
    axes[0].grid(alpha=0.2)
    if "delta_reduced_chi2_population_minus_map" in comparison:
        axes[1].hist(
            comparison["delta_reduced_chi2_population_minus_map"].dropna(),
            bins=50,
            alpha=0.75,
        )
        axes[1].axvline(0, color="black", lw=1)
    axes[1].set_xlabel("population - independent reduced chi2")
    axes[1].set_ylabel("galaxies")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_hmc_map_population(comparison: pd.DataFrame, path: str | Path) -> None:
    if comparison.empty:
        return
    parameters = list(comparison["parameter"].drop_duplicates())
    fig, axes = plt.subplots(
        len(parameters), 1, figsize=(9, max(2.6 * len(parameters), 3)), sharex=False
    )
    axes = np.atleast_1d(axes)
    for ax, parameter in zip(axes, parameters, strict=True):
        work = (
            comparison[comparison["parameter"] == parameter]
            .sort_values("row_index")
            .reset_index(drop=True)
        )
        x = np.arange(len(work))
        median = work["hmc_median"].to_numpy(dtype=float)
        q16 = work["hmc_q16"].to_numpy(dtype=float)
        q84 = work["hmc_q84"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            median,
            yerr=[median - q16, q84 - median],
            fmt="o",
            ms=4,
            capsize=2,
            label="HMC",
        )
        ax.scatter(x, work["map_value"], marker="x", s=35, label="MAP")
        if "population_value" in work:
            ax.scatter(
                x,
                work["population_value"],
                marker="s",
                s=20,
                facecolors="none",
                edgecolors="tab:orange",
                label="population",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(work["row_index"].astype(str), rotation=30)
        ax.set_ylabel(parameter)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("catalog row index")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_batch_dashboard(
    valid: pd.DataFrame, by_row: pd.DataFrame, path: str | Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_residual_boxplot(valid, axes[0, 0])
    plot_observed_model_scatter(valid, axes[0, 1])

    reduced = by_row["reduced_chi2"].replace([np.inf, -np.inf], np.nan).dropna()
    if not reduced.empty:
        axes[1, 0].hist(np.log10(reduced + 1.0e-12), bins=50, alpha=0.75)
    axes[1, 0].set_xlabel("log10 reduced chi2")
    axes[1, 0].set_ylabel("galaxies")
    axes[1, 0].grid(alpha=0.2)

    if {"z_obs", "redshift_truth"}.issubset(by_row.columns):
        plot_redshift_scatter(by_row, axes[1, 1])
    elif "z_obs" in by_row.columns:
        axes[1, 1].scatter(
            by_row["z_obs"], by_row["mean_residual_mag"], s=8, alpha=0.35
        )
        axes[1, 1].axhline(0, color="black", lw=1)
        axes[1, 1].set_xlabel("z used by DSPS")
        axes[1, 1].set_ylabel("mean residual [mag]")
        axes[1, 1].grid(alpha=0.2)
    else:
        axes[1, 1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_batch_residuals_by_band(valid: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_residual_boxplot(valid, ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_batch_observed_vs_model(valid: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_observed_model_scatter(valid, ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_batch_redshift_truth(by_row: pd.DataFrame, path: str | Path) -> None:
    if not {"z_obs", "redshift_truth"}.issubset(by_row.columns):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_redshift_scatter(by_row, axes[0])
    dz = by_row["delta_z_obs_minus_truth"].replace([np.inf, -np.inf], np.nan).dropna()
    axes[1].hist(dz, bins=60, alpha=0.75)
    axes[1].axvline(0, color="black", lw=1)

    fit_label, truth_label = _redshift_axis_labels(by_row)
    label = f"{fit_label} - {truth_label}"
    axes[1].set_xlabel(label)
    axes[1].set_ylabel("galaxies")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_batch_parameter_truth(by_row: pd.DataFrame, path: str | Path) -> None:
    # Find all parameters that have both fit and truth
    params = []
    for col in by_row.columns:
        if col.startswith("truth_"):
            param_name = col[6:]
            if f"fit_{param_name}" in by_row.columns:
                params.append(param_name)

    if not params:
        return

    fig, axes = plt.subplots(len(params), 2, figsize=(10, 4 * len(params)))
    axes = np.atleast_2d(axes)

    for i, param in enumerate(params):
        truth_col = f"truth_{param}"
        fit_col = f"fit_{param}"
        delta_col = f"delta_fit_{param}_minus_truth"

        work = (
            by_row[[truth_col, fit_col, delta_col]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if work.empty:
            continue

        ax_scatter = axes[i, 0]
        ax_hist = axes[i, 1]

        ax_scatter.scatter(work[truth_col], work[fit_col], s=8, alpha=0.35)
        lo, hi = (
            min(work[truth_col].min(), work[fit_col].min()),
            max(work[truth_col].max(), work[fit_col].max()),
        )
        ax_scatter.plot([lo, hi], [lo, hi], color="black", lw=1)
        ax_scatter.set_xlabel(
            f"{_truth_legend_label(param)} {_parameter_display_label(param)}"
        )
        ax_scatter.set_ylabel(f"inferred {_parameter_display_label(param)}")
        ax_scatter.grid(alpha=0.2)

        ax_hist.hist(work[delta_col], bins=60, alpha=0.75)
        ax_hist.axvline(0, color="black", lw=1)
        ax_hist.set_xlabel(
            f"inferred {_parameter_display_label(param)} - {_truth_legend_label(param)}"
        )
        ax_hist.set_ylabel("galaxies")
        ax_hist.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_residuals_by_property(by_row: pd.DataFrame, path: str | Path) -> None:
    specs = [
        ("redshift_truth", "truth redshift"),
        ("z_obs", "photo-z / fitted redshift"),
        ("catalog_log_stellar_mass", "catalog log stellar mass"),
        ("fit_log10_formed_mass_msun", "fit log formed mass"),
    ]
    available = [(column, label) for column, label in specs if column in by_row]
    has_color = "catalog_color_kind" in by_row
    if not available and not has_color:
        return
    n_panels = len(available) + int(has_color)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(8, max(2.8 * n_panels, 3)), squeeze=False
    )
    axes_flat = axes[:, 0]
    panel = 0
    for column, label in available:
        work = by_row[[column, "mean_residual_mag"]].replace([np.inf, -np.inf], np.nan)
        work = work.dropna()
        if work.empty:
            axes_flat[panel].axis("off")
            panel += 1
            continue
        axes_flat[panel].scatter(
            work[column], work["mean_residual_mag"], s=10, alpha=0.35
        )
        _plot_running_median(
            work[column].to_numpy(dtype=float),
            work["mean_residual_mag"].to_numpy(dtype=float),
            axes_flat[panel],
        )
        axes_flat[panel].axhline(0, color="black", lw=1)
        axes_flat[panel].set_xlabel(label)
        axes_flat[panel].set_ylabel("mean model - obs [mag]")
        axes_flat[panel].grid(alpha=0.2)
        panel += 1
    if has_color:
        work = by_row[["catalog_color_kind", "mean_residual_mag"]].dropna()
        if work.empty:
            axes_flat[panel].axis("off")
        else:
            groups = [
                group["mean_residual_mag"].to_numpy(dtype=float)
                for _, group in work.groupby("catalog_color_kind")
            ]
            labels = [str(key) for key in work.groupby("catalog_color_kind").groups]
            axes_flat[panel].boxplot(groups, labels=labels, showfliers=False)
            axes_flat[panel].axhline(0, color="black", lw=1)
            axes_flat[panel].set_xlabel("catalog color_kind")
            axes_flat[panel].set_ylabel("mean model - obs [mag]")
            axes_flat[panel].grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_running_median(x: np.ndarray, y: np.ndarray, ax: plt.Axes) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 8:
        return
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    bins = np.array_split(np.arange(len(x)), min(8, len(x)))
    centers = []
    medians = []
    for item in bins:
        if len(item) == 0:
            continue
        centers.append(float(np.median(x[item])))
        medians.append(float(np.median(y[item])))
    ax.plot(centers, medians, color="black", lw=1.4)


def plot_residual_boxplot(valid: pd.DataFrame, ax: plt.Axes) -> None:
    bands = ordered_bands(valid)
    data = [
        valid.loc[valid["band"] == band, "residual_mag_model_minus_observed"].dropna()
        for band in bands
    ]
    ax.boxplot(data, labels=bands, showfliers=False)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("model - simulated [mag]")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.2, axis="y")


def plot_observed_model_scatter(valid: pd.DataFrame, ax: plt.Axes) -> None:
    sample = (
        valid.sample(min(len(valid), 5000), random_state=1)
        if len(valid) > 5000
        else valid
    )
    for band in ordered_bands(sample):
        work = sample[sample["band"] == band]
        ax.scatter(
            work["observed_mag_ab"], work["model_mag_ab"], s=8, alpha=0.35, label=band
        )
    values = (
        pd.concat([sample["observed_mag_ab"], sample["model_mag_ab"]])
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if not values.empty:
        lo, hi = float(values.min()), float(values.max())
        ax.plot([lo, hi], [lo, hi], color="black", lw=1)
    ax.set_xlabel("simulated AB mag")
    ax.set_ylabel("DSPS AB mag")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)


def plot_redshift_scatter(by_row: pd.DataFrame, ax: plt.Axes) -> None:
    work = (
        by_row[["redshift_truth", "z_obs"]].replace([np.inf, -np.inf], np.nan).dropna()
    )
    if len(work) > 5000:
        work = work.sample(5000, random_state=2)
    ax.scatter(work["redshift_truth"], work["z_obs"], s=8, alpha=0.35)
    if not work.empty:
        lo = float(min(work["redshift_truth"].min(), work["z_obs"].min()))
        hi = float(max(work["redshift_truth"].max(), work["z_obs"].max()))
        ax.plot([lo, hi], [lo, hi], color="black", lw=1)
    fit_label, truth_label = _redshift_axis_labels(by_row)
    ax.set_xlabel(truth_label)
    ax.set_ylabel(fit_label)
    ax.grid(alpha=0.2)


def _redshift_axis_labels(by_row: pd.DataFrame) -> tuple[str, str]:
    truth_source = _first_string(by_row, "redshift_truth_source") or "z_true"
    fit_source = _first_string(by_row, "z_obs_source") or "z_phz"
    fit_label = (
        "Inferred Redshift (Fit)"
        if "fit_z_obs" in by_row.columns
        else f"Input {fit_source} to DSPS"
    )
    return fit_label, f"Ground Truth ({truth_source})"


def _first_string(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return str(values.iloc[0])


def _truth_kind_from_frame(frame: pd.DataFrame, parameter: str) -> str:
    column = f"truth_kind_{parameter}"
    value = _first_string(frame, column)
    if value:
        return value
    return "proxy" if parameter in {"dust_av", "log10_metallicity"} else "direct"


def _truth_legend_label(parameter: str) -> str:
    kind = " (proxy)" if parameter in {"dust_av", "log10_metallicity"} else ""
    return f"Ground Truth{kind}"


def _parameter_display_label(parameter: str) -> str:
    labels = {
        "z_obs": "Redshift",
        "log10_sfr_at_obs": "log10 SFR [Msun/yr]",
        "log10_sfr": "log10 SFH Amplitude",
        "log10_formed_mass_msun": "log10 formed mass [Msun]",
        "dust_av": "Dust Av [mag]",
        "log10_metallicity": "log10 Metallicity [absolute log10(Z)]",
    }
    return labels.get(parameter, parameter)


def ordered_bands(df: pd.DataFrame) -> list[str]:
    if "effective_wavelength_angstrom" not in df:
        return [str(band) for band in df["band"].drop_duplicates().tolist()]
    order = (
        df.groupby("band")["effective_wavelength_angstrom"]
        .median()
        .sort_values()
        .index.tolist()
    )
    return [str(band) for band in order]
