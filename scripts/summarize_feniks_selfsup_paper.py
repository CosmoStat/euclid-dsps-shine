#!/usr/bin/env python3
"""Summarize and plot the final self-supervised paper array."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import summarize_feniks_selfsup_rws_task as base

LABELS = (
    "rws_k8_t2_seed2",
    "rws_k8_t2_seed3",
    "fixed_prior_rws_k8_t2",
    "avi_joint_t2",
)

DESCRIPTIONS = {
    "rws_k8_t2_seed2": (
        "Learned RealNVP prior; Student-t2 sleep/wake RWS K=8; replication seed 2"
    ),
    "rws_k8_t2_seed3": (
        "Learned RealNVP prior; Student-t2 sleep/wake RWS K=8; replication seed 3"
    ),
    "fixed_prior_rws_k8_t2": (
        "Frozen reference RealNVP prior; Student-t2 sleep/wake RWS K=8"
    ),
    "avi_joint_t2": "Learned RealNVP prior; Student-t2 stochastic ELBO with reverse KL",
}

_base_aggregate = base.aggregate


def aggregate(root: Path, expected: int) -> None:
    _base_aggregate(root, expected)
    out = root / "comparison"
    metrics_path = out / "experiment_metrics.csv"
    if not metrics_path.is_file():
        return
    metrics = pd.read_csv(metrics_path)
    _plot_scientific_scoreboard(metrics, out)
    _plot_jacobian_spectra(root, out)
    _plot_photoz_by_redshift(root, out)
    _plot_prior_marginal_errors(root, out)
    report = out / "README.md"
    text = report.read_text(encoding="utf-8").replace(
        "The synthetic likelihood is Gaussian with the catalog flux errors and no extra floor.",
        "The likelihood is Student-t with two degrees of freedom, catalog flux-error scales, and no added floor.",
    )
    text += (
        "\n## Cross-run publication diagnostics\n\n"
        "- [Posterior coverage by parameter](coverage_by_parameter.png)\n"
        "- [Scientific scoreboard](scientific_scoreboard.png)\n"
        "- [Flow-aware Jacobian spectra](jacobian_spectrum_comparison.png)\n"
        "- [Photo-z performance by redshift](photoz_by_redshift.png)\n"
        "- [Prior marginal errors by parameter](prior_marginal_error_by_parameter.png)\n"
        "\nThe Jacobian Lens point and covariance are estimated from samples after "
        "the complete conditional flow. The autoencoder Lens uses the differentiable "
        "flow push-forward of the encoder base mean.\n"
    )
    report.write_text(text, encoding="utf-8")


def _plot_scientific_scoreboard(metrics: pd.DataFrame, out: Path) -> None:
    columns = (
        ("photoz_rmse", "photo-z RMSE", False),
        ("coverage_error", "coverage error", False),
        ("prior_15d_mean_quantile_l1_iqr", "prior marginal error", False),
        ("prior_15d_mean_spearman_abs_delta", "prior correlation error", False),
        ("median_posterior_predictive_chi2", "median photometric chi2", False),
        ("wake_ess_fraction_mean", "wake ESS fraction", True),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    labels = metrics["label"].astype(str).tolist()
    colors = ("#266dd3", "#37a169", "#d97706", "#9b51e0")
    for ax, (column, title, higher_better) in zip(axes.flat, columns, strict=True):
        values = pd.to_numeric(metrics.get(column), errors="coerce")
        ax.bar(np.arange(len(labels)), values, color=colors)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.text(
            0.98,
            0.95,
            "higher is better" if higher_better else "lower is better",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out / "scientific_scoreboard.png", dpi=190)
    plt.close(fig)


def _plot_jacobian_spectra(root: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    found = False
    for label in LABELS:
        path = root / label / "jacobian_lens" / "singular_values.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        median = frame.groupby("direction_index").singular_value.median()
        ax.semilogy(median.index, median.values, marker="o", ms=3, label=label)
        found = True
    if not found:
        plt.close(fig)
        raise FileNotFoundError("No finalized Jacobian singular-value tables")
    ax.set_xlabel("latent direction index")
    ax.set_ylabel("median likelihood-weighted singular value")
    ax.set_title("Flow-aware DSPS Jacobian spectra")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "jacobian_spectrum_comparison.png", dpi=190)
    plt.close(fig)


def _plot_photoz_by_redshift(root: Path, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.2), sharex=True)
    for label in LABELS:
        path = root / label / "inference" / "photoz_metrics_by_redshift_bin.csv"
        frame = pd.read_csv(path)
        center = 0.5 * (frame["z_bin_lower"] + frame["z_bin_upper"])
        axes[0].plot(center, frame["rmse"], marker="o", ms=3, label=label)
        axes[1].plot(center, frame["coverage_68"], marker="o", ms=3, label=label)
    axes[1].axhline(0.68, color="black", ls="--", lw=1)
    axes[0].set_ylabel("normalized photo-z RMSE")
    axes[1].set_ylabel("68% coverage")
    axes[1].set_xlabel("true redshift")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "photoz_by_redshift.png", dpi=190)
    plt.close(fig)


def _plot_prior_marginal_errors(root: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    parameters = None
    for label in LABELS:
        path = root / label / "inference" / "prior_vs_truth_population.csv"
        frame = pd.read_csv(path)
        if parameters is None:
            parameters = frame["parameter"].astype(str).tolist()
        aligned = frame.set_index("parameter").reindex(parameters)
        ax.plot(
            np.arange(len(parameters)),
            aligned["quantile_l1_iqr"],
            marker="o",
            ms=3,
            label=label,
        )
    if parameters is None:
        plt.close(fig)
        raise FileNotFoundError("No prior population diagnostic tables")
    ax.set_xticks(np.arange(len(parameters)), parameters, rotation=65, ha="right")
    ax.set_ylabel("prior quantile L1 / truth IQR")
    ax.set_title("Population-prior marginal error")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "prior_marginal_error_by_parameter.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    base.LABELS = LABELS
    base.DESCRIPTIONS = DESCRIPTIONS
    base.aggregate = aggregate
    base.main()
