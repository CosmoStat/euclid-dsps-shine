"""Build a paired report comparing two MAP fit runs."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BAND_ORDER = [
    "lsst_u",
    "lsst_g",
    "lsst_r",
    "lsst_i",
    "lsst_z",
    "lsst_y",
    "euclid_vis",
    "euclid_nisp_y",
    "euclid_nisp_j",
    "euclid_nisp_h",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian-run", required=True, type=Path)
    parser.add_argument("--student-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    gaussian = load_run(args.gaussian_run, "chi2")
    student = load_run(args.student_run, "student_t")
    paired = paired_galaxies(gaussian, student)
    band = paired_bands(gaussian, student)
    summary = summary_metrics(gaussian, student, paired)
    runtime = runtime_metrics(gaussian, student)

    paired.to_csv(out / "galaxy_by_galaxy.csv", index=False)
    band.to_csv(out / "band_comparison.csv", index=False)
    summary.to_csv(out / "summary_metrics.csv", index=False)
    runtime.to_csv(out / "runtime_comparison.csv", index=False)

    write_top_tables(paired, out)
    plot_dashboard(paired, band, runtime, out / "comparison_dashboard.png")
    plot_redshift(paired, out / "paired_redshift_comparison.png")
    plot_fit_quality(paired, out / "paired_fit_quality.png")
    plot_galaxy_deltas(paired, out / "galaxy_by_galaxy_deltas.png")
    plot_band_comparison(band, out / "band_residuals_comparison.png")
    plot_runtime(runtime, out / "runtime_comparison.png")
    write_report(args.gaussian_run, args.student_run, paired, band, summary, runtime, out)


def load_run(root: Path, label: str) -> dict[str, Any]:
    required = [
        "batch_fit_results.parquet",
        "batch_fit_photometry_comparison.parquet",
        "batch_fit_summary_by_galaxy.csv",
        "batch_fit_summary_by_band.csv",
        "batch_fit_summary.json",
        "batch_fit_performance_by_batch.csv",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"{root} is missing {missing}")
    return {
        "label": label,
        "root": root,
        "fits": pd.read_parquet(root / "batch_fit_results.parquet"),
        "photometry": pd.read_parquet(root / "batch_fit_photometry_comparison.parquet"),
        "by_galaxy": pd.read_csv(root / "batch_fit_summary_by_galaxy.csv"),
        "by_band": pd.read_csv(root / "batch_fit_summary_by_band.csv"),
        "summary": json.loads((root / "batch_fit_summary.json").read_text()),
        "performance": pd.read_csv(root / "batch_fit_performance_by_batch.csv"),
    }


def paired_galaxies(
    gaussian: dict[str, Any], student: dict[str, Any]
) -> pd.DataFrame:
    g = row_frame(gaussian, "chi2")
    s = row_frame(student, "student_t")
    paired = g.merge(s, on="row_index", how="inner", validate="one_to_one")
    if "z_truth_chi2" in paired and "z_truth_student_t" in paired:
        paired["z_truth"] = paired["z_truth_chi2"].combine_first(
            paired["z_truth_student_t"]
        )
    elif "z_truth_chi2" in paired:
        paired["z_truth"] = paired["z_truth_chi2"]
    else:
        paired["z_truth"] = np.nan
    for label in ("chi2", "student_t"):
        paired[f"delta_z_{label}"] = paired[f"fit_z_obs_{label}"] - paired["z_truth"]
        paired[f"abs_delta_z_{label}"] = paired[f"delta_z_{label}"].abs()
        paired[f"abs_delta_z_over_1pz_{label}"] = paired[f"abs_delta_z_{label}"] / (
            1.0 + paired["z_truth"].clip(lower=0.0)
        )
        paired[f"catastrophic_z_{label}"] = (
            paired[f"abs_delta_z_{label}"] > 0.15 * (1.0 + paired["z_truth"])
        )
    paired["delta_abs_z_student_minus_chi2"] = (
        paired["abs_delta_z_student_t"] - paired["abs_delta_z_chi2"]
    )
    paired["delta_abs_z_norm_student_minus_chi2"] = (
        paired["abs_delta_z_over_1pz_student_t"]
        - paired["abs_delta_z_over_1pz_chi2"]
    )
    paired["delta_reduced_gaussian_chi2_student_minus_chi2"] = (
        paired["reduced_chi2_student_t"] - paired["reduced_chi2_chi2"]
    )
    paired["ratio_reduced_gaussian_chi2_student_over_chi2"] = (
        paired["reduced_chi2_student_t"]
        / paired["reduced_chi2_chi2"].replace(0.0, np.nan)
    )
    paired["delta_mean_abs_mag_residual_student_minus_chi2"] = (
        paired["mean_abs_residual_mag_student_t"]
        - paired["mean_abs_residual_mag_chi2"]
    )
    paired["student_t_better_redshift"] = (
        paired["delta_abs_z_student_minus_chi2"] < 0.0
    )
    paired["student_t_better_residual"] = (
        paired["delta_mean_abs_mag_residual_student_minus_chi2"] < 0.0
    )
    paired["student_t_better_gaussian_chi2"] = (
        paired["delta_reduced_gaussian_chi2_student_minus_chi2"] < 0.0
    )
    return paired.sort_values("row_index").reset_index(drop=True)


def row_frame(run: dict[str, Any], suffix: str) -> pd.DataFrame:
    fits = run["fits"].copy()
    by_row = run["by_galaxy"].copy()
    if "row_index" not in by_row:
        by_row = by_row.rename(columns={by_row.columns[0]: "row_index"})
    cols_fit = [
        "row_index",
        "success",
        "fit_z_obs",
        "redshift_truth",
        "catalog_z_true_gal",
        "chi2",
        "reduced_chi2",
        "fit_quality",
        "reduced_fit_quality",
        "fit_quality_metric",
        "photometric_likelihood",
        "gradient_norm",
        "n_valid_bands",
        "dof",
    ]
    cols_row = [
        "row_index",
        "mean_residual_mag",
        "median_residual_mag",
        "rms_residual_mag",
        "mean_abs_residual_mag",
        "chi2_per_band",
        "fit_quality_per_band",
    ]
    left = fits[[col for col in cols_fit if col in fits]].copy()
    right = by_row[[col for col in cols_row if col in by_row]].copy()
    frame = left.merge(right, on="row_index", how="left")
    truth_cols = [col for col in ("redshift_truth", "catalog_z_true_gal") if col in frame]
    if truth_cols:
        frame["z_truth"] = frame[truth_cols[0]]
        for col in truth_cols[1:]:
            frame["z_truth"] = frame["z_truth"].combine_first(frame[col])
    else:
        frame["z_truth"] = np.nan
    frame = frame.drop(columns=[col for col in ("redshift_truth", "catalog_z_true_gal") if col in frame])
    return frame.rename(
        columns={col: f"{col}_{suffix}" for col in frame.columns if col != "row_index"}
    )


def paired_bands(gaussian: dict[str, Any], student: dict[str, Any]) -> pd.DataFrame:
    g = band_frame(gaussian, "chi2")
    s = band_frame(student, "student_t")
    band = g.merge(s, on="band", how="inner", validate="one_to_one")
    band["delta_mean_abs_residual_student_minus_chi2"] = (
        band["mean_abs_residual_mag_student_t"] - band["mean_abs_residual_mag_chi2"]
    )
    band["delta_rms_residual_student_minus_chi2"] = (
        band["rms_residual_mag_student_t"] - band["rms_residual_mag_chi2"]
    )
    band["delta_mean_chi_likelihood_student_minus_chi2"] = (
        band["mean_chi_likelihood_student_t"] - band["mean_chi_likelihood_chi2"]
    )
    band["_order"] = band["band"].map({name: i for i, name in enumerate(BAND_ORDER)})
    return band.sort_values(["_order", "band"]).drop(columns="_order").reset_index(
        drop=True
    )


def band_frame(run: dict[str, Any], suffix: str) -> pd.DataFrame:
    by_band = run["by_band"].copy()
    cols = [
        "band",
        "n",
        "mean_residual_mag",
        "median_residual_mag",
        "rms_residual_mag",
        "mean_abs_residual_mag",
        "median_flux_ratio",
        "mean_chi",
        "mean_chi_likelihood",
        "mean_photometric_objective_contribution",
    ]
    frame = by_band[[col for col in cols if col in by_band]].copy()
    return frame.rename(
        columns={col: f"{col}_{suffix}" for col in frame.columns if col != "band"}
    )


def summary_metrics(
    gaussian: dict[str, Any], student: dict[str, Any], paired: pd.DataFrame
) -> pd.DataFrame:
    rows = [
        run_summary_row(gaussian, paired, "chi2"),
        run_summary_row(student, paired, "student_t"),
    ]
    return pd.DataFrame(rows)


def run_summary_row(
    run: dict[str, Any], paired: pd.DataFrame, suffix: str
) -> dict[str, Any]:
    summary = run["summary"]
    return {
        "run": suffix,
        "path": str(run["root"]),
        "n_galaxies": int(len(run["fits"])),
        "success_fraction": float(pd.to_numeric(run["fits"]["success"]).mean()),
        "photometric_likelihood": summary.get("fit_quality_metric", suffix),
        "median_reduced_gaussian_chi2": median(paired[f"reduced_chi2_{suffix}"]),
        "median_reduced_fit_quality": median(paired[f"reduced_fit_quality_{suffix}"]),
        "median_abs_delta_z": median(paired[f"abs_delta_z_{suffix}"]),
        "median_abs_delta_z_over_1pz": median(
            paired[f"abs_delta_z_over_1pz_{suffix}"]
        ),
        "catastrophic_z_fraction_0p15_1pz": float(
            paired[f"catastrophic_z_{suffix}"].mean()
        ),
        "median_mean_abs_mag_residual": median(
            paired[f"mean_abs_residual_mag_{suffix}"]
        ),
        "median_rms_mag_residual": median(paired[f"rms_residual_mag_{suffix}"]),
        "median_gradient_norm": median(paired[f"gradient_norm_{suffix}"]),
    }


def runtime_metrics(
    gaussian: dict[str, Any], student: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    for run in (gaussian, student):
        perf = run["performance"].copy()
        post_compile = perf.iloc[1:] if len(perf) > 1 else perf
        rows.append(
            {
                "run": run["label"],
                "n_batches": int(len(perf)),
                "n_galaxies": int(perf["n_rows"].sum()),
                "first_batch_seconds_per_galaxy": float(
                    perf["seconds_per_galaxy"].iloc[0]
                ),
                "median_seconds_per_galaxy_all_batches": median(
                    perf["seconds_per_galaxy"]
                ),
                "median_seconds_per_galaxy_after_first_batch": median(
                    post_compile["seconds_per_galaxy"]
                ),
                "total_elapsed_fit_chunks_seconds": float(
                    perf["elapsed_seconds_sum"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def write_top_tables(paired: pd.DataFrame, out: Path) -> None:
    columns = [
        "row_index",
        "z_truth",
        "fit_z_obs_chi2",
        "fit_z_obs_student_t",
        "abs_delta_z_chi2",
        "abs_delta_z_student_t",
        "delta_abs_z_student_minus_chi2",
        "reduced_chi2_chi2",
        "reduced_chi2_student_t",
        "mean_abs_residual_mag_chi2",
        "mean_abs_residual_mag_student_t",
    ]
    paired.nsmallest(30, "delta_abs_z_student_minus_chi2")[columns].to_csv(
        out / "top30_student_t_redshift_improvements.csv", index=False
    )
    paired.nlargest(30, "delta_abs_z_student_minus_chi2")[columns].to_csv(
        out / "top30_student_t_redshift_degradations.csv", index=False
    )
    paired.nsmallest(30, "delta_reduced_gaussian_chi2_student_minus_chi2")[
        columns
    ].to_csv(out / "top30_student_t_chi2_improvements.csv", index=False)


def plot_dashboard(
    paired: pd.DataFrame, band: pd.DataFrame, runtime: pd.DataFrame, path: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ax = axes[0, 0]
    bins = np.linspace(
        0.0,
        float(
            np.nanpercentile(
                pd.concat(
                    [
                        paired["abs_delta_z_over_1pz_chi2"],
                        paired["abs_delta_z_over_1pz_student_t"],
                    ]
                ),
                98,
            )
        ),
        50,
    )
    ax.hist(paired["abs_delta_z_over_1pz_chi2"], bins=bins, alpha=0.55, label="chi2")
    ax.hist(
        paired["abs_delta_z_over_1pz_student_t"],
        bins=bins,
        alpha=0.55,
        label="student-t",
    )
    ax.set_xlabel("|dz| / (1 + z_true)")
    ax.set_ylabel("galaxies")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    scatter_log(
        ax,
        paired["reduced_chi2_chi2"],
        paired["reduced_chi2_student_t"],
        "chi2 run reduced Gaussian chi2",
        "student-t run reduced Gaussian chi2",
    )

    ax = axes[0, 2]
    ax.hist(
        paired["delta_abs_z_student_minus_chi2"],
        bins=60,
        alpha=0.75,
        color="tab:green",
    )
    ax.axvline(0.0, color="black", lw=1)
    ax.set_xlabel("student-t minus chi2 |dz|")
    ax.set_ylabel("galaxies")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.scatter(
        paired["delta_abs_z_student_minus_chi2"],
        paired["delta_mean_abs_mag_residual_student_minus_chi2"],
        s=12,
        alpha=0.45,
    )
    ax.axhline(0.0, color="black", lw=1)
    ax.axvline(0.0, color="black", lw=1)
    ax.set_xlabel("student-t minus chi2 |dz|")
    ax.set_ylabel("student-t minus chi2 mean |mag residual|")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    x = np.arange(len(band))
    ax.bar(x, band["delta_mean_abs_residual_student_minus_chi2"], color="tab:orange")
    ax.axhline(0.0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(band["band"], rotation=35, ha="right")
    ax.set_ylabel("student-t minus chi2 mean |mag residual|")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1, 2]
    ax.bar(
        runtime["run"],
        runtime["median_seconds_per_galaxy_after_first_batch"],
        color=["tab:blue", "tab:orange"],
    )
    ax.set_ylabel("median seconds / galaxy after first batch")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Gaussian chi2 vs Student-t(2) paired comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_redshift(paired: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    z = paired["z_truth"]
    lo = float(np.nanmin(z))
    hi = float(np.nanmax(z))
    for ax, label, color in [
        (axes[0], "chi2", "tab:blue"),
        (axes[1], "student_t", "tab:orange"),
    ]:
        ax.scatter(z, paired[f"fit_z_obs_{label}"], s=12, alpha=0.45, color=color)
        ax.plot([lo, hi], [lo, hi], color="black", lw=1)
        ax.set_xlabel("truth z")
        ax.set_ylabel(f"fit z ({label})")
        ax.grid(alpha=0.25)
    axes[2].scatter(
        paired["abs_delta_z_chi2"],
        paired["abs_delta_z_student_t"],
        s=12,
        alpha=0.45,
    )
    maxv = float(
        np.nanpercentile(
            pd.concat([paired["abs_delta_z_chi2"], paired["abs_delta_z_student_t"]]),
            99,
        )
    )
    axes[2].plot([0, maxv], [0, maxv], color="black", lw=1)
    axes[2].set_xlabel("chi2 |dz|")
    axes[2].set_ylabel("student-t |dz|")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_fit_quality(paired: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    scatter_log(
        axes[0],
        paired["reduced_chi2_chi2"],
        paired["reduced_chi2_student_t"],
        "chi2 run reduced Gaussian chi2",
        "student-t run reduced Gaussian chi2",
    )
    scatter_log(
        axes[1],
        paired["mean_abs_residual_mag_chi2"],
        paired["mean_abs_residual_mag_student_t"],
        "chi2 mean |mag residual|",
        "student-t mean |mag residual|",
    )
    axes[2].hist(
        np.log10(paired["ratio_reduced_gaussian_chi2_student_over_chi2"]),
        bins=60,
        alpha=0.75,
    )
    axes[2].axvline(0.0, color="black", lw=1)
    axes[2].set_xlabel("log10(student-t reduced chi2 / chi2-run reduced chi2)")
    axes[2].set_ylabel("galaxies")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_galaxy_deltas(paired: pd.DataFrame, path: Path) -> None:
    work = paired.sort_values("delta_abs_z_student_minus_chi2").reset_index(drop=True)
    x = np.arange(len(work))
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    metrics = [
        ("delta_abs_z_student_minus_chi2", "delta |dz|"),
        ("delta_mean_abs_mag_residual_student_minus_chi2", "delta mean |mag residual|"),
        ("delta_reduced_gaussian_chi2_student_minus_chi2", "delta reduced Gaussian chi2"),
    ]
    for ax, (column, label) in zip(axes, metrics, strict=True):
        ax.plot(x, work[column], lw=1)
        ax.axhline(0.0, color="black", lw=1)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("galaxies sorted by redshift improvement")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_band_comparison(band: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(band))
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(
        x - width / 2,
        band["mean_abs_residual_mag_chi2"],
        width=width,
        label="chi2",
    )
    axes[0].bar(
        x + width / 2,
        band["mean_abs_residual_mag_student_t"],
        width=width,
        label="student-t",
    )
    axes[0].set_ylabel("mean |mag residual|")
    axes[0].legend()
    axes[0].grid(alpha=0.25, axis="y")
    axes[1].bar(x, band["delta_mean_abs_residual_student_minus_chi2"], color="tab:green")
    axes[1].axhline(0.0, color="black", lw=1)
    axes[1].set_ylabel("student-t minus chi2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(band["band"], rotation=35, ha="right")
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_runtime(runtime: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(
        runtime["run"],
        runtime["first_batch_seconds_per_galaxy"],
        color=["tab:blue", "tab:orange"],
    )
    axes[0].set_ylabel("first batch seconds / galaxy")
    axes[0].grid(alpha=0.25, axis="y")
    axes[1].bar(
        runtime["run"],
        runtime["median_seconds_per_galaxy_after_first_batch"],
        color=["tab:blue", "tab:orange"],
    )
    axes[1].set_ylabel("median seconds / galaxy after first batch")
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    gaussian_root: Path,
    student_root: Path,
    paired: pd.DataFrame,
    band: pd.DataFrame,
    summary: pd.DataFrame,
    runtime: pd.DataFrame,
    out: Path,
) -> None:
    g = summary.set_index("run").loc["chi2"]
    s = summary.set_index("run").loc["student_t"]
    n = len(paired)
    redshift_better = 100.0 * paired["student_t_better_redshift"].mean()
    residual_better = 100.0 * paired["student_t_better_residual"].mean()
    chi2_better = 100.0 * paired["student_t_better_gaussian_chi2"].mean()
    best_bands = band.nsmallest(3, "delta_mean_abs_residual_student_minus_chi2")
    worst_bands = band.nlargest(3, "delta_mean_abs_residual_student_minus_chi2")
    text = f"""# Gaussian chi2 vs Student-t(2) MAP comparison

Inputs:

- Gaussian chi2 run: `{gaussian_root}`
- Student-t run: `{student_root}`
- Paired galaxies: {n}

## Executive summary

The Student-t(2) run improves the paired redshift and photometry diagnostics on
this 512-galaxy sample. The two runs are compared galaxy by galaxy using the
same `row_index` values.

| metric | chi2 | Student-t(2) | Student-t minus chi2 |
|---|---:|---:|---:|
| median reduced Gaussian chi2 | {g['median_reduced_gaussian_chi2']:.4g} | {s['median_reduced_gaussian_chi2']:.4g} | {s['median_reduced_gaussian_chi2'] - g['median_reduced_gaussian_chi2']:.4g} |
| median |dz| | {g['median_abs_delta_z']:.4g} | {s['median_abs_delta_z']:.4g} | {s['median_abs_delta_z'] - g['median_abs_delta_z']:.4g} |
| median |dz|/(1+z) | {g['median_abs_delta_z_over_1pz']:.4g} | {s['median_abs_delta_z_over_1pz']:.4g} | {s['median_abs_delta_z_over_1pz'] - g['median_abs_delta_z_over_1pz']:.4g} |
| catastrophic fraction, |dz| > 0.15(1+z) | {g['catastrophic_z_fraction_0p15_1pz']:.3%} | {s['catastrophic_z_fraction_0p15_1pz']:.3%} | {s['catastrophic_z_fraction_0p15_1pz'] - g['catastrophic_z_fraction_0p15_1pz']:.3%} |
| median mean |mag residual| | {g['median_mean_abs_mag_residual']:.4g} | {s['median_mean_abs_mag_residual']:.4g} | {s['median_mean_abs_mag_residual'] - g['median_mean_abs_mag_residual']:.4g} |

Galaxy-by-galaxy fractions:

- Student-t has smaller absolute redshift error for {redshift_better:.1f}% of paired galaxies.
- Student-t has smaller mean absolute magnitude residual for {residual_better:.1f}% of paired galaxies.
- Student-t has smaller final Gaussian reduced chi2 for {chi2_better:.1f}% of paired galaxies.

Runtime:

| run | first batch s/gal | median s/gal after first batch |
|---|---:|---:|
| chi2 | {runtime.set_index('run').loc['chi2', 'first_batch_seconds_per_galaxy']:.3f} | {runtime.set_index('run').loc['chi2', 'median_seconds_per_galaxy_after_first_batch']:.3f} |
| Student-t(2) | {runtime.set_index('run').loc['student_t', 'first_batch_seconds_per_galaxy']:.3f} | {runtime.set_index('run').loc['student_t', 'median_seconds_per_galaxy_after_first_batch']:.3f} |

## Band-level residuals

Best three bands for Student-t in mean absolute magnitude residual:

{markdown_table(best_bands[['band', 'delta_mean_abs_residual_student_minus_chi2']])}

Worst three bands for Student-t in mean absolute magnitude residual:

{markdown_table(worst_bands[['band', 'delta_mean_abs_residual_student_minus_chi2']])}

## Generated figures

- `comparison_dashboard.png`: overall paired dashboard.
- `paired_redshift_comparison.png`: truth-vs-fit and paired |dz| comparison.
- `paired_fit_quality.png`: final Gaussian chi2 and residual comparisons.
- `galaxy_by_galaxy_deltas.png`: per-galaxy deltas sorted by redshift improvement.
- `band_residuals_comparison.png`: per-band photometric residual comparison.
- `runtime_comparison.png`: first-batch and post-compilation timing.

## Generated tables

- `galaxy_by_galaxy.csv`: full paired table by `row_index`.
- `top30_student_t_redshift_improvements.csv`
- `top30_student_t_redshift_degradations.csv`
- `top30_student_t_chi2_improvements.csv`
- `band_comparison.csv`
- `summary_metrics.csv`
- `runtime_comparison.csv`

## Caveats

- The Student-t objective and Gaussian chi2 are not on the same likelihood scale.
  The paired Gaussian chi2 columns evaluate both final solutions with the same
  Gaussian diagnostic.
- Runtime includes one first batch with compilation and data movement; use the
  post-first-batch median for steady-state comparison.
- This remains the repository's PopCosmos-like DSPS/FSPS setup, not a full audit
  of the original POP-COSMOS population model.
"""
    (out / "report.md").write_text(text)


def scatter_log(ax, x, y, xlabel: str, ylabel: str) -> None:
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    ax.scatter(x[finite], y[finite], s=12, alpha=0.45)
    if finite.any():
        lo = float(min(np.nanpercentile(x[finite], 1), np.nanpercentile(y[finite], 1)))
        hi = float(max(np.nanpercentile(x[finite], 99), np.nanpercentile(y[finite], 99)))
        ax.plot([lo, hi], [lo, hi], color="black", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(numeric.median())


def markdown_table(frame: pd.DataFrame) -> str:
    rows = []
    columns = list(frame.columns)
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.5g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
