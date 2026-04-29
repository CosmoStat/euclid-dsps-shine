"""EDA and run reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import GalaxyObservation, ensure_dir, write_json
from .model import ModelResult, comparison_rows


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

    band_columns = [band["column"] for band in band_configs if band["column"] in df.columns]
    if band_columns:
        plot_flux_distributions(df, band_columns, out / "flux_distributions.png")
        plot_color_distributions(df, band_columns, out / "color_distributions.png")
    if redshift_config:
        plot_redshift_distributions(df, redshift_config, out / "redshift_diagnostics.png")


def write_run_outputs(observation: GalaxyObservation, result: ModelResult, out_dir: str | Path) -> pd.DataFrame:
    out = ensure_dir(out_dir)
    write_json(out / "selected_galaxy.json", observation)
    write_json(out / "model_parameters.json", {"parameters": result.parameters, "derived": result.derived})

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
        },
    )
    pd.DataFrame(fit_result.trace).to_csv(out / "fit_trace.csv", index=False)
    plot_fit_trace(pd.DataFrame(fit_result.trace), out / "fit_trace.png")


def write_batch_outputs(comparison: pd.DataFrame, out_dir: str | Path, label: str = "batch") -> None:
    """Write aggregate tables and plots for multi-galaxy runs."""
    out = ensure_dir(out_dir)
    error_rows = comparison[comparison["error"].notna()] if "error" in comparison else pd.DataFrame()
    valid = comparison[comparison["band"].notna()].copy() if "band" in comparison else pd.DataFrame()

    if not error_rows.empty:
        error_rows.to_csv(out / f"{label}_errors.csv", index=False)

    if valid.empty:
        write_json(out / f"{label}_summary.json", {"n_valid_rows": 0, "n_errors": int(len(error_rows))})
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
        "median_abs_residual_mag": float(valid["residual_mag_model_minus_observed"].abs().median()),
    }
    if "delta_z_obs_minus_truth" in by_row:
        dz = by_row["delta_z_obs_minus_truth"].dropna()
        if not dz.empty:
            summary["median_delta_z_obs_minus_truth"] = float(dz.median())
            summary["mad_delta_z_obs_minus_truth"] = float((dz - dz.median()).abs().median())
    write_json(out / f"{label}_summary.json", summary)

    plot_batch_dashboard(valid, by_row, out / f"{label}_dashboard.png")
    plot_batch_residuals_by_band(valid, out / f"{label}_residuals_by_band.png")
    plot_batch_observed_vs_model(valid, out / f"{label}_observed_vs_model.png")
    plot_batch_redshift_truth(by_row, out / f"{label}_redshift_truth.png")


def summarize_by_band(valid: pd.DataFrame) -> pd.DataFrame:
    return valid.groupby("band").agg(
        n=("row_index", "count"),
        effective_wavelength_angstrom=("effective_wavelength_angstrom", "median"),
        mean_residual_mag=("residual_mag_model_minus_observed", "mean"),
        median_residual_mag=("residual_mag_model_minus_observed", "median"),
        std_residual_mag=("residual_mag_model_minus_observed", "std"),
        rms_residual_mag=("residual_mag_model_minus_observed", lambda x: float(np.sqrt(np.nanmean(x**2)))),
        mean_abs_residual_mag=("residual_mag_model_minus_observed", lambda x: float(np.nanmean(np.abs(x)))),
        median_flux_ratio=("flux_ratio_model_over_observed", "median"),
        mean_chi=("chi", "mean"),
    )


def summarize_by_row(valid: pd.DataFrame) -> pd.DataFrame:
    context_columns = [
        col
        for col in valid.columns
        if col in {"z_obs", "redshift_truth", "delta_z_obs_minus_truth"}
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
        "rms_residual_mag": ("residual_mag_model_minus_observed", lambda x: float(np.sqrt(np.nanmean(x**2)))),
        "mean_abs_residual_mag": ("residual_mag_model_minus_observed", lambda x: float(np.nanmean(np.abs(x)))),
    }
    for col in context_columns:
        aggregations[col] = (col, "first")
    by_row = valid.groupby("row_index").agg(**aggregations)
    by_row["reduced_chi2"] = by_row["chi2"] / by_row["n_bands"].clip(lower=1)
    return by_row


def plot_flux_distributions(df: pd.DataFrame, columns: list[str], path: str | Path) -> None:
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


def plot_color_distributions(df: pd.DataFrame, columns: list[str], path: str | Path) -> None:
    if len(columns) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for left, right in zip(columns[:-1], columns[1:]):
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


def plot_redshift_distributions(df: pd.DataFrame, redshift_config: dict[str, Any], path: str | Path) -> None:
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
    ax.plot(result.wave[mask], result.rest_sed[mask], label="Intrinsic rest SED", lw=1.1, alpha=0.65)
    ax.plot(result.wave[mask], result.dusted_rest_sed[mask], label="Dust-attenuated rest SED", lw=1.4)
    z_obs = result.parameters.get("z_obs", np.nan)
    for band, values in result.photometry.items():
        wave_rest = values["effective_wavelength_angstrom"] / (1.0 + z_obs)
        if np.isfinite(wave_rest) and 800 <= wave_rest <= 30_000:
            ax.axvline(wave_rest, color="black", lw=0.7, alpha=0.18)
            ax.text(wave_rest, 0.98, band.replace("euclid_", ""), rotation=90, va="top", ha="right", transform=ax.get_xaxis_transform(), fontsize=7)
    ax.set_xlabel("rest-frame wavelength [Angstrom]")
    ax.set_ylabel("Lsun / Hz")
    ax.set_yscale("log")
    ax.set_title(f"DSPS rest SED, z={z_obs:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_photometry_comparison(comparison: pd.DataFrame, path: str | Path) -> None:
    work = comparison.sort_values("effective_wavelength_angstrom").reset_index(drop=True)
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
    for xi, yi, label in zip(x, work["observed_mag_ab"], work["band"]):
        ax_mag.annotate(label.replace("euclid_", ""), (xi, yi), textcoords="offset points", xytext=(3, 5), fontsize=8)
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
    if trace.empty or "chi2" not in trace:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(trace)), trace["chi2"], lw=1)
    ax.set_xlabel("evaluation")
    ax.set_ylabel("chi2")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_batch_dashboard(valid: pd.DataFrame, by_row: pd.DataFrame, path: str | Path) -> None:
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
        axes[1, 1].scatter(by_row["z_obs"], by_row["mean_residual_mag"], s=8, alpha=0.35)
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
    axes[1].set_xlabel("z used - z truth")
    axes[1].set_ylabel("galaxies")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_residual_boxplot(valid: pd.DataFrame, ax: plt.Axes) -> None:
    bands = ordered_bands(valid)
    data = [valid.loc[valid["band"] == band, "residual_mag_model_minus_observed"].dropna() for band in bands]
    ax.boxplot(data, labels=bands, showfliers=False)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("model - simulated [mag]")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.2, axis="y")


def plot_observed_model_scatter(valid: pd.DataFrame, ax: plt.Axes) -> None:
    sample = valid.sample(min(len(valid), 5000), random_state=1) if len(valid) > 5000 else valid
    for band in ordered_bands(sample):
        work = sample[sample["band"] == band]
        ax.scatter(work["observed_mag_ab"], work["model_mag_ab"], s=8, alpha=0.35, label=band)
    values = pd.concat([sample["observed_mag_ab"], sample["model_mag_ab"]]).replace([np.inf, -np.inf], np.nan).dropna()
    if not values.empty:
        lo, hi = float(values.min()), float(values.max())
        ax.plot([lo, hi], [lo, hi], color="black", lw=1)
    ax.set_xlabel("simulated AB mag")
    ax.set_ylabel("DSPS AB mag")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)


def plot_redshift_scatter(by_row: pd.DataFrame, ax: plt.Axes) -> None:
    work = by_row[["redshift_truth", "z_obs"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) > 5000:
        work = work.sample(5000, random_state=2)
    ax.scatter(work["redshift_truth"], work["z_obs"], s=8, alpha=0.35)
    if not work.empty:
        lo = float(min(work["redshift_truth"].min(), work["z_obs"].min()))
        hi = float(max(work["redshift_truth"].max(), work["z_obs"].max()))
        ax.plot([lo, hi], [lo, hi], color="black", lw=1)
    ax.set_xlabel("truth redshift")
    ax.set_ylabel("redshift used by DSPS")
    ax.grid(alpha=0.2)


def ordered_bands(df: pd.DataFrame) -> list[str]:
    order = (
        df.groupby("band")["effective_wavelength_angstrom"]
        .median()
        .sort_values()
        .index.tolist()
    )
    return [str(band) for band in order]
