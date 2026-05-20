"""Reporting helpers for COSMOS-template pseudo-SED diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..cosmos import CosmosSedResult, flambda_10pc_to_lnu_lsun
from ..io import ensure_dir
from .core import configure_plot_style

configure_plot_style()


def plot_cosmos_sed_example(result: CosmosSedResult, path: str | Path) -> None:
    """Plot one reconstructed COSMOS proxy SED."""
    wave = result.wave_angstrom
    scaled = flambda_10pc_to_lnu_lsun(wave, result.flambda_scaled)
    unscaled = flambda_10pc_to_lnu_lsun(wave, result.flambda_unscaled * result.alpha)
    mask = (
        np.isfinite(wave)
        & np.isfinite(scaled)
        & (wave >= 800.0)
        & (wave <= 50_000.0)
        & (scaled > 0)
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    if mask.any():
        ax.plot(wave[mask], scaled[mask], lw=1.5, label=r"COSMOS proxy $L_\nu$")
    raw_mask = mask & np.isfinite(unscaled) & (unscaled > 0)
    if raw_mask.any():
        ax.plot(
            wave[raw_mask],
            unscaled[raw_mask],
            lw=0.9,
            ls="--",
            alpha=0.65,
            label=r"template shape $\times\alpha$",
        )
    for band, target in result.catalog_abs_fluxes.items():
        synth = result.synthetic_abs_fluxes_after[band]
        ax.scatter([], [], label=f"{band}: synth/catalog={synth / target:.3f}")
    title = (
        f"COSMOS proxy SED row {result.row_index}, "
        f"templates {result.diagnostics['sed_cosmos_1']}/"
        f"{result.diagnostics['sed_cosmos_2']}"
    )
    ax.set_title(title)
    ax.set_xlabel(r"rest-frame wavelength [$\AA$]")
    ax.set_ylabel(r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_cosmos_sed_sample_set(
    results: list[CosmosSedResult],
    path: str | Path,
    max_seds: int = 12,
    comparisons: list[pd.DataFrame] | None = None,
) -> None:
    """Plot sampled SEDs as row-wise COSMOS-vs-DSPS grids when available."""
    selected = results[: max(int(max_seds), 1)]
    if not selected:
        return
    comparison_by_row = _comparison_by_row(comparisons or [])
    n_rows = len(selected)
    has_comparison = any(result.row_index in comparison_by_row for result in selected)
    n_cols = 2 if has_comparison else 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(9.8 if has_comparison else 6.8, max(2.2 * n_rows, 3.0)),
        squeeze=False,
        sharex=False,
    )
    for row_number, result in enumerate(selected):
        ax_sed = axes[row_number, 0]
        comparison = comparison_by_row.get(result.row_index)
        if comparison is not None and not comparison.empty:
            _plot_sample_comparison_panel(ax_sed, axes[row_number, 1], comparison)
        else:
            _plot_sample_cosmos_only_panel(ax_sed, result)
            if has_comparison:
                axes[row_number, 1].axis("off")
        ax_sed.set_title(_sample_title(result), fontsize=9)
    suffix = (
        "COSMOS proxy vs inferred DSPS SEDs"
        if has_comparison
        else "COSMOS proxy SED sample"
    )
    fig.suptitle(
        (
            f"{suffix}, n={len(selected)}\n"
            r"color_kind: 0=red, 1=green, 2=blue; "
            r"$f_1$=normalized COSMOS component-1 fraction"
        ),
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _comparison_by_row(comparisons: list[pd.DataFrame]) -> dict[int, pd.DataFrame]:
    by_row = {}
    for comparison in comparisons:
        if comparison.empty or "row_index" not in comparison:
            continue
        row_index = int(comparison["row_index"].iloc[0])
        by_row[row_index] = comparison
    return by_row


def _plot_sample_comparison_panel(
    ax_sed: plt.Axes, ax_resid: plt.Axes, comparison: pd.DataFrame
) -> None:
    work = comparison.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "wave_angstrom",
            "cosmos_proxy_lnu_lsun_per_hz",
            "dsps_scaled_lnu_lsun_per_hz",
            "log10_dsps_scaled_minus_cosmos",
        ]
    )
    if work.empty:
        ax_sed.axis("off")
        ax_resid.axis("off")
        return
    wave = work["wave_angstrom"].to_numpy(dtype=float)
    cosmos = work["cosmos_proxy_lnu_lsun_per_hz"].to_numpy(dtype=float)
    dsps = work["dsps_scaled_lnu_lsun_per_hz"].to_numpy(dtype=float)
    residual = work["log10_dsps_scaled_minus_cosmos"].to_numpy(dtype=float)
    mask = (
        np.isfinite(wave)
        & np.isfinite(cosmos)
        & np.isfinite(dsps)
        & (wave >= 800.0)
        & (wave <= 50_000.0)
        & (cosmos > 0)
        & (dsps > 0)
    )
    if not mask.any():
        ax_sed.axis("off")
        ax_resid.axis("off")
        return
    ax_sed.plot(wave[mask], cosmos[mask], lw=1.15, label="COSMOS proxy")
    ax_sed.plot(wave[mask], dsps[mask], lw=1.05, ls="--", label="inferred DSPS")
    ax_sed.set_xscale("log")
    ax_sed.set_yscale("log")
    ax_sed.set_ylabel(r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]")
    ax_sed.legend(fontsize=7, loc="best")

    resid_mask = mask & np.isfinite(residual)
    ax_resid.axhline(0.0, color="black", lw=0.8)
    ax_resid.plot(wave[resid_mask], residual[resid_mask], lw=0.9, color="#B85C38")
    ax_resid.set_xscale("log")
    ax_resid.set_xlabel(r"rest-frame wavelength [$\AA$]")
    ax_resid.set_ylabel(r"$\Delta\log_{10}L_\nu$")


def _plot_sample_cosmos_only_panel(ax: plt.Axes, result: CosmosSedResult) -> None:
    wave = result.wave_angstrom
    lnu = flambda_10pc_to_lnu_lsun(wave, result.flambda_scaled)
    mask = (
        np.isfinite(wave)
        & np.isfinite(lnu)
        & (wave >= 800.0)
        & (wave <= 50_000.0)
        & (lnu > 0)
    )
    if not mask.any():
        ax.axis("off")
        return
    ax.plot(wave[mask], lnu[mask], lw=1.05, label="COSMOS proxy")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"rest-frame wavelength [$\AA$]")
    ax.set_ylabel(r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]")
    ax.legend(fontsize=7)


def _sample_title(result: CosmosSedResult) -> str:
    color_kind = result.diagnostics.get("color_kind")
    f1 = result.diagnostics.get("frac_cosmos_1_used")
    try:
        f1_text = f"{float(f1):.2f}"
    except (TypeError, ValueError):
        f1_text = "nan"
    return f"row {result.row_index}, " f"color_kind={color_kind}, " rf"$f_1$={f1_text}"


def plot_template_pair_heatmap(diagnostics: pd.DataFrame, path: str | Path) -> None:
    """Plot frequency of ``sed_cosmos_1`` / ``sed_cosmos_2`` template pairs."""
    if diagnostics.empty or not {"sed_cosmos_1", "sed_cosmos_2"}.issubset(diagnostics):
        return
    work = diagnostics[["sed_cosmos_1", "sed_cosmos_2"]].dropna().astype(int)
    if work.empty:
        return
    grid = pd.crosstab(work["sed_cosmos_1"], work["sed_cosmos_2"])
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(grid.to_numpy(dtype=float), origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(grid.columns)))
    ax.set_xticklabels(grid.columns, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(grid.index)))
    ax.set_yticklabels(grid.index, fontsize=7)
    ax.set_xlabel("sed_cosmos_2")
    ax.set_ylabel("sed_cosmos_1")
    ax.set_title("COSMOS template-pair counts")
    fig.colorbar(image, ax=ax, label="galaxies")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_fraction_diagnostics(diagnostics: pd.DataFrame, path: str | Path) -> None:
    """Plot component fraction distribution and alpha relation."""
    needed = {"frac_cosmos_1_used", "frac_cosmos_2_used", "alpha"}
    if diagnostics.empty or not needed.issubset(diagnostics):
        return
    work = diagnostics.replace([np.inf, -np.inf], np.nan).dropna(subset=list(needed))
    if work.empty:
        return
    fig, (ax_hist, ax_alpha) = plt.subplots(1, 2, figsize=(9.0, 4.0))
    ax_hist.hist(
        work["frac_cosmos_1_used"],
        bins=30,
        histtype="stepfilled",
        alpha=0.65,
        label=r"$f_1$",
    )
    ax_hist.hist(
        work["frac_cosmos_2_used"],
        bins=30,
        histtype="step",
        lw=1.5,
        label=r"$f_2$",
    )
    ax_hist.set_xlabel("normalized component fraction")
    ax_hist.set_ylabel("galaxies")
    ax_hist.legend()

    color = work["color_kind"] if "color_kind" in work else None
    scatter = ax_alpha.scatter(
        work["frac_cosmos_1_used"],
        work["alpha"],
        c=color,
        s=22,
        alpha=0.78,
    )
    if color is not None:
        fig.colorbar(scatter, ax=ax_alpha, label="color_kind")
    ax_alpha.set_yscale("log")
    ax_alpha.set_xlabel(r"$f_1$")
    ax_alpha.set_ylabel(r"normalization $\alpha$")
    ax_alpha.set_title("Template mix vs normalization")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_synthetic_vs_catalog_abs_flux(frame: pd.DataFrame, path: str | Path) -> None:
    """Plot synthetic normalized absolute flux versus catalog absolute flux."""
    if frame.empty:
        return
    work = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["catalog_abs_flux_fnu_cgs", "synthetic_abs_flux_after_scaling_fnu_cgs"]
    )
    work = work[
        (work["catalog_abs_flux_fnu_cgs"] > 0)
        & (work["synthetic_abs_flux_after_scaling_fnu_cgs"] > 0)
    ]
    if work.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for band, group in work.groupby("band"):
        ax.scatter(
            group["catalog_abs_flux_fnu_cgs"],
            group["synthetic_abs_flux_after_scaling_fnu_cgs"],
            s=24,
            alpha=0.75,
            label=band.replace("euclid_", ""),
        )
    lo = min(
        work["catalog_abs_flux_fnu_cgs"].min(),
        work["synthetic_abs_flux_after_scaling_fnu_cgs"].min(),
    )
    hi = max(
        work["catalog_abs_flux_fnu_cgs"].max(),
        work["synthetic_abs_flux_after_scaling_fnu_cgs"].max(),
    )
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, alpha=0.75)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"catalog $F_\nu$ at 10 pc")
    ax.set_ylabel(r"synthetic proxy $F_\nu$ at 10 pc")
    ax.set_title("Euclid absolute-flux normalization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_cosmos_dsps_rest_comparison(
    comparison: pd.DataFrame, path: str | Path, row_index: int
) -> None:
    """Plot COSMOS proxy and DSPS rest SED comparison for one row."""
    if comparison.empty:
        return
    work = comparison.replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return
    wave = work["wave_angstrom"].to_numpy(dtype=float)
    fig, (ax_sed, ax_resid) = plt.subplots(
        2,
        1,
        figsize=(8.4, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.06},
    )
    ax_sed.plot(
        wave,
        work["cosmos_proxy_lnu_lsun_per_hz"],
        lw=1.55,
        label=r"COSMOS proxy $L_\nu$",
    )
    ax_sed.plot(
        wave,
        work["dsps_scaled_lnu_lsun_per_hz"],
        lw=1.25,
        label=r"DSPS attenuated $L_\nu$, scaled",
    )
    ax_sed.set_yscale("log")
    ax_sed.set_xscale("log")
    ax_sed.set_ylabel(r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]")
    ax_sed.set_title(f"Rest-frame SED comparison, row {row_index}")
    ax_sed.legend()

    ax_resid.axhline(0.0, color="black", lw=0.9)
    ax_resid.plot(
        wave,
        work["log10_dsps_scaled_minus_cosmos"],
        lw=1.0,
        color="#B85C38",
    )
    ax_resid.set_xscale("log")
    ax_resid.set_xlabel(r"rest-frame wavelength [$\AA$]")
    ax_resid.set_ylabel(r"$\Delta\log_{10}L_\nu$")
    fig.subplots_adjust(hspace=0.08)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_branch1_metric_summary(metrics: pd.DataFrame, path: str | Path) -> None:
    """Plot rest-frame SED RMS residuals by color kind."""
    if metrics.empty or "rms_log_sed_residual" not in metrics:
        return
    work = metrics.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["rms_log_sed_residual"]
    )
    if work.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    if "color_kind" in work:
        groups = [
            group["rms_log_sed_residual"].to_numpy(dtype=float)
            for _, group in work.groupby("color_kind")
        ]
        labels = [str(key) for key, _ in work.groupby("color_kind")]
        ax.boxplot(groups, labels=labels, showfliers=False)
        ax.set_xlabel("color_kind")
    else:
        ax.hist(work["rms_log_sed_residual"], bins=30)
        ax.set_xlabel(r"RMS $\Delta\log_{10}L_\nu$")
    ax.set_ylabel("galaxies")
    ax.set_title("COSMOS proxy vs DSPS rest-SED residuals")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_rest_color_residuals(metrics: pd.DataFrame, path: str | Path) -> None:
    """Plot DSPS-COSMOS Euclid rest-color residuals."""
    columns = [col for col in metrics.columns if col.startswith("rest_color_residual_")]
    if metrics.empty or not columns:
        return
    work = metrics.replace([np.inf, -np.inf], np.nan)
    fig, axes = plt.subplots(
        len(columns),
        1,
        figsize=(7.4, max(2.3 * len(columns), 3.0)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    z_column = "z_true_gal" if "z_true_gal" in work else "z_true"
    x = work[z_column] if z_column in work else np.arange(len(work))
    color = work["color_kind"] if "color_kind" in work else None
    for ax, column in zip(axes, columns, strict=True):
        mask = np.isfinite(work[column]) & np.isfinite(x)
        scatter = ax.scatter(
            x[mask],
            work.loc[mask, column],
            c=color[mask] if color is not None else None,
            s=24,
            alpha=0.75,
        )
        ax.axhline(0.0, color="black", lw=0.9)
        ax.set_ylabel(column.replace("rest_color_residual_", "").replace("_mag", ""))
        if color is not None:
            fig.colorbar(scatter, ax=ax, label="color_kind")
    axes[-1].set_xlabel(z_column if z_column in work else "sample index")
    fig.suptitle("DSPS - COSMOS rest-color residuals [mag]", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_branch1_residual_heatmap(metrics: pd.DataFrame, path: str | Path) -> None:
    """Plot median rest-SED RMS residual by redshift bin and color kind."""
    z_column = "z_true_gal" if "z_true_gal" in metrics else "z_true"
    needed = {"rms_log_sed_residual", z_column, "color_kind"}
    if metrics.empty or not needed.issubset(metrics):
        return
    work = metrics.replace([np.inf, -np.inf], np.nan).dropna(subset=list(needed))
    if work.empty:
        return
    work = work.copy()
    work["z_bin"] = pd.cut(
        pd.to_numeric(work[z_column], errors="coerce"),
        bins=[0.0, 0.5, 1.0, 1.5, 2.5, 4.5, np.inf],
        include_lowest=True,
    ).astype(str)
    pivot = work.pivot_table(
        index="color_kind",
        columns="z_bin",
        values="rms_log_sed_residual",
        aggfunc="median",
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(item) for item in pivot.index])
    ax.set_xlabel(f"{z_column} bin")
    ax.set_ylabel("color_kind")
    ax.set_title(r"Median RMS $\Delta\log_{10}L_\nu$ by population")
    fig.colorbar(image, ax=ax, label=r"median RMS $\Delta\log_{10}L_\nu$")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_worst_sed_grid(
    metrics: pd.DataFrame, comparison: pd.DataFrame, path: str | Path, n: int = 16
) -> None:
    """Plot the worst COSMOS-vs-DSPS SED overlays by RMS log residual."""
    if metrics.empty or comparison.empty or "rms_log_sed_residual" not in metrics:
        return
    worst = (
        metrics.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["rms_log_sed_residual", "row_index"])
        .nlargest(int(n), "rms_log_sed_residual")
    )
    if worst.empty:
        return
    rows = [int(value) for value in worst["row_index"].tolist()]
    n_panels = len(rows)
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13.0, max(2.8 * n_rows, 3.0)),
        squeeze=False,
    )
    metric_by_row = worst.set_index("row_index").to_dict(orient="index")
    for ax, row_index in zip(axes.ravel(), rows, strict=False):
        subset = comparison[comparison["row_index"] == row_index]
        _plot_worst_sed_panel(ax, subset, metric_by_row.get(row_index, {}), row_index)
    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")
    fig.suptitle("Worst COSMOS proxy vs inferred DSPS SEDs", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_worst_sed_panel(
    ax: plt.Axes, comparison: pd.DataFrame, metrics: dict[str, float], row_index: int
) -> None:
    work = comparison.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "wave_angstrom",
            "cosmos_proxy_lnu_lsun_per_hz",
            "dsps_scaled_lnu_lsun_per_hz",
        ]
    )
    if work.empty:
        ax.axis("off")
        return
    wave = work["wave_angstrom"].to_numpy(dtype=float)
    cosmos = work["cosmos_proxy_lnu_lsun_per_hz"].to_numpy(dtype=float)
    dsps = work["dsps_scaled_lnu_lsun_per_hz"].to_numpy(dtype=float)
    mask = (
        np.isfinite(wave)
        & np.isfinite(cosmos)
        & np.isfinite(dsps)
        & (wave >= 800.0)
        & (wave <= 50_000.0)
        & (cosmos > 0)
        & (dsps > 0)
    )
    if not mask.any():
        ax.axis("off")
        return
    ax.plot(wave[mask], cosmos[mask], lw=0.9, label="COSMOS")
    ax.plot(wave[mask], dsps[mask], lw=0.9, ls="--", label="DSPS")
    ax.set_xscale("log")
    ax.set_yscale("log")
    rms = metrics.get("rms_log_sed_residual", np.nan)
    color_kind = metrics.get("color_kind", np.nan)
    ax.set_title(f"row {row_index}, RMS={rms:.2f}, color_kind={color_kind}", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlabel(r"$\lambda_\mathrm{rest}$ [$\AA$]", fontsize=8)
    ax.set_ylabel(r"$L_\nu$", fontsize=8)


def plot_observed_flux_residuals(frame: pd.DataFrame, path: str | Path) -> None:
    """Plot branch-2 residuals by band and target set with robust clipping."""
    if frame.empty or "relative_flux_residual" not in frame:
        return
    work = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["relative_flux_residual"]
    )
    if work.empty:
        return
    target_sets = list(work["target_set"].dropna().unique())
    fig, axes = plt.subplots(
        len(target_sets),
        1,
        figsize=(8.0, max(2.5 * len(target_sets), 3.0)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for ax, target_set in zip(axes, target_sets, strict=True):
        subset = work[work["target_set"] == target_set]
        bands = list(subset["band"].dropna().unique())
        if target_set == "noisy_observation" and "chi" in subset:
            value_column = "chi"
            ylabel = r"clipped $(F_\mathrm{model}-F_\mathrm{obs})/\sigma_F$"
            values = [
                _clip_for_boxplot(
                    subset.loc[subset["band"] == band, value_column].to_numpy(
                        dtype=float
                    ),
                    hard=(-8.0, 8.0),
                )
                for band in bands
            ]
        else:
            ylabel = r"clipped $(F_\mathrm{model}-F_\mathrm{obs})/F_\mathrm{obs}$"
            values = [
                _clip_for_boxplot(
                    subset.loc[
                        subset["band"] == band, "relative_flux_residual"
                    ].to_numpy(dtype=float),
                    hard=(-2.0, 2.0),
                )
                for band in bands
            ]
        ax.axhline(0.0, color="black", lw=0.9)
        ax.boxplot(
            values,
            labels=[_short_band_label(band) for band in bands],
            showfliers=False,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(target_set)
    axes[-1].set_xlabel("band")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_population_validation_summary(summary: pd.DataFrame, path: str | Path) -> None:
    """Plot grouped population-validation medians from summary CSV rows."""
    if summary.empty or not {"grouping", "group", "median", "count"}.issubset(summary):
        return
    work = summary.replace([np.inf, -np.inf], np.nan).dropna(subset=["median"])
    if work.empty:
        return
    preferred = [
        "color_kind",
        "z_bin",
        "apparent_mag_bin",
        "log_sfr_bin",
        "metallicity_bin",
        "stellar_mass_bin",
        "template_pair",
        "dust_curve_pair",
    ]
    groupings = [item for item in preferred if item in set(work["grouping"])]
    groupings = groupings[:6]
    if not groupings:
        return
    n_cols = 2
    n_rows = int(np.ceil(len(groupings) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12.0, max(3.0 * n_rows, 3.2)),
        squeeze=False,
    )
    for ax, grouping in zip(axes.ravel(), groupings, strict=False):
        subset = work[work["grouping"] == grouping].copy()
        subset = subset.sort_values("count", ascending=False).head(12)
        subset = subset.sort_values("median")
        labels = subset["group"].astype(str).to_list()
        values = subset["median"].to_numpy(dtype=float)
        ax.barh(np.arange(len(values)), values, color="#4C78A8", alpha=0.82)
        ax.axvline(0.0, color="black", lw=0.8)
        ax.set_yticks(np.arange(len(values)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(grouping)
        ax.set_xlabel("median metric")
    for ax in axes.ravel()[len(groupings) :]:
        ax.axis("off")
    fig.suptitle("Population-level validation summary", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _clip_for_boxplot(values: np.ndarray, hard: tuple[float, float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return values
    lo, hi = np.nanpercentile(values, [1.0, 99.0])
    lo = max(float(lo), hard[0])
    hi = min(float(hi), hard[1])
    if lo >= hi:
        lo, hi = hard
    return np.clip(values, lo, hi)


def _short_band_label(band: str) -> str:
    return (
        band.replace("euclid_", "")
        .replace("lsst_", "")
        .replace("nisp_", "NISP ")
        .upper()
    )


def write_cosmos_output_manifest(out_dir: str | Path, files: list[str]) -> None:
    """Write lightweight manifest for generated COSMOS SED artifacts."""
    out = ensure_dir(out_dir)
    pd.DataFrame({"file": files}).to_csv(out / "cosmos_sed_manifest.csv", index=False)
