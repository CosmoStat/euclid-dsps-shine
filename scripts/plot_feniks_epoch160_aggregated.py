#!/usr/bin/env python3
"""Plot aggregate and individual diagnostics from a complete epoch-160 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

from euclid_dsps.amortized.mira import FENIKS_SPLINE15D_PARAMETERS  # noqa: E402

PARAMETERS = tuple(FENIKS_SPLINE15D_PARAMETERS)
PHYSICAL_PARAMETERS = PARAMETERS[:5]
PARAMETER_LABELS = {
    "z_obs": r"$z$",
    "log10_stellar_mass": r"$\log_{10} M_\star$",
    "log10_stellar_metallicity": r"$\log_{10} Z_\star$",
    "dust_av": r"$A_V$",
    "dust_delta": r"$\delta_{\rm dust}$",
    **{
        f"sfh_dlog_sfr_{index:02d}": rf"$\Delta\log{{\rm SFR}}_{{{index}}}$"
        for index in range(1, 11)
    },
}
SHORT_PARAMETER_LABELS = [
    "z",
    "log M*",
    "log Z*",
    "Av",
    "dust d",
    *[f"dSFR{i}" for i in range(1, 11)],
]

INK = "#17202A"
MUTED = "#64748B"
GRID = "#CBD5E1"
TRUTH = "#111827"
RAW_Q = "#0F766E"
EMA_Q = "#2563EB"
RAW_IW = "#D97706"
EMA_IW = "#DC2626"
PARENT = "#6B7280"
SELECTED = "#7C3AED"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titleweight": "semibold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.35,
            "grid.linewidth": 0.7,
            "font.size": 10,
            "legend.frameon": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _validate_dense_bank(
    frame: pd.DataFrame,
    *,
    expected_rows: set[int],
    samples_per_object: int,
    label: str,
) -> None:
    required = {"row_index", "sample_id", *PARAMETERS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    actual_rows = set(frame["row_index"].astype(int).unique().tolist())
    if actual_rows != expected_rows:
        raise ValueError(
            f"{label} row identities differ: "
            f"missing={len(expected_rows - actual_rows)} extra={len(actual_rows - expected_rows)}"
        )
    counts = frame.groupby("row_index", sort=False)["sample_id"].size()
    if not counts.eq(int(samples_per_object)).all():
        raise ValueError(f"{label} is not object-equal at {samples_per_object} draws")
    if frame.duplicated(["row_index", "sample_id"]).any():
        raise ValueError(f"{label} contains duplicate joint-draw identities")


def _select_panel_rows(row_indices: list[int], count: int) -> list[int]:
    if count <= 0 or count > len(row_indices):
        raise ValueError("individual count must be within the available panel size")
    positions = np.rint(np.linspace(0, len(row_indices) - 1, count)).astype(int)
    selected = [int(row_indices[position]) for position in positions]
    if len(selected) != len(set(selected)):
        raise ValueError("deterministic individual selection produced duplicates")
    return selected


def _finite(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _plot_limits(
    arrays: list[np.ndarray], *, tail: float = 0.002
) -> tuple[float, float]:
    quantiles = []
    for values in arrays:
        finite = _finite(values)
        if finite.size:
            quantiles.append(np.quantile(finite, [tail, 1.0 - tail]))
    if not quantiles:
        raise ValueError("cannot determine plot limits from empty arrays")
    bounds = np.asarray(quantiles)
    lower = float(np.min(bounds[:, 0]))
    upper = float(np.max(bounds[:, 1]))
    if not np.isfinite(lower + upper) or upper <= lower:
        delta = max(abs(lower), 1.0) * 0.05
        return lower - delta, upper + delta
    padding = 0.035 * (upper - lower)
    return lower - padding, upper + padding


def _density_curve(
    values: np.ndarray | pd.Series,
    limits: tuple[float, float],
    *,
    bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    finite = _finite(values)
    finite = finite[(finite >= limits[0]) & (finite <= limits[1])]
    if finite.size < 2:
        raise ValueError("at least two finite values are required for a density")
    hist, edges = np.histogram(finite, bins=bins, range=limits, density=True)
    radius = 4
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / 1.25) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(hist, kernel, mode="same")
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, smooth


def _draw_density(
    ax: plt.Axes,
    values: np.ndarray | pd.Series,
    limits: tuple[float, float],
    *,
    color: str,
    label: str,
    linewidth: float = 1.7,
    linestyle: str = "-",
    alpha: float = 1.0,
    fill_alpha: float = 0.0,
) -> None:
    x, y = _density_curve(values, limits)
    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
    )
    if fill_alpha:
        ax.fill_between(x, 0.0, y, color=color, alpha=fill_alpha, linewidth=0)


def _finish_density_axis(ax: plt.Axes, parameter: str) -> None:
    ax.set_title(PARAMETER_LABELS[parameter], fontsize=11, pad=5)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)


def _save_png_pdf(fig: plt.Figure, out: Path, stem: str) -> list[Path]:
    paths = [out / f"{stem}.png", out / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=210)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def _plot_selected_marginals(
    out: Path,
    truth: pd.DataFrame,
    banks: dict[str, pd.DataFrame],
    support: dict[str, dict[str, Any]],
) -> list[Path]:
    fig, axes = plt.subplots(3, 5, figsize=(15.2, 9.2))
    for ax, parameter in zip(axes.flat, PARAMETERS, strict=True):
        limits = _plot_limits(
            [
                truth[parameter].to_numpy(),
                *(bank[parameter].to_numpy() for bank in banks.values()),
            ]
        )
        _draw_density(
            ax,
            truth[parameter],
            limits,
            color=TRUTH,
            label="Selected-test truth",
            linewidth=2.2,
            fill_alpha=0.07,
        )
        _draw_density(
            ax,
            banks["raw_q"][parameter],
            limits,
            color=RAW_Q,
            label="Raw q",
            linestyle="--",
            alpha=0.85,
        )
        _draw_density(
            ax,
            banks["ema_q"][parameter],
            limits,
            color=EMA_Q,
            label="EMA q",
            linewidth=1.9,
        )
        _draw_density(
            ax,
            banks["raw_iw"][parameter],
            limits,
            color=RAW_IW,
            label="Raw IW resamples",
            linestyle=":",
            alpha=0.8,
        )
        _draw_density(
            ax,
            banks["ema_iw"][parameter],
            limits,
            color=EMA_IW,
            label="EMA IW resamples",
            linewidth=1.6,
        )
        ax.set_xlim(limits)
        _finish_density_axis(ax, parameter)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(
        "Epoch 160: aggregate selected-catalogue posterior recovery",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    raw_ess = float(support["raw"]["median_raw_ess"])
    ema_ess = float(support["ema"]["median_raw_ess"])
    fig.text(
        0.5,
        0.958,
        (
            "4,706 independent selected-test objects; 32 joint draws/object. "
            f"IW is diagnostic: held-out K=1024 median ESS = {raw_ess:.2f} raw, "
            f"{ema_ess:.2f} EMA."
        ),
        ha="center",
        va="top",
        fontsize=9.5,
        color=MUTED,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.932),
        ncol=5,
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89), h_pad=0.7, w_pad=0.4)
    return _save_png_pdf(fig, out, "population_selected_marginals")


def _plot_prior_recovery(
    out: Path,
    c0_truth: pd.DataFrame,
    selected_truth: pd.DataFrame,
    parent: np.ndarray,
    selected: np.ndarray,
) -> list[Path]:
    fig, axes = plt.subplots(3, 5, figsize=(15.2, 9.2))
    for index, (ax, parameter) in enumerate(zip(axes.flat, PARAMETERS, strict=True)):
        limits = _plot_limits(
            [
                c0_truth[parameter].to_numpy(),
                selected_truth[parameter].to_numpy(),
                parent[:, index],
                selected[:, index],
            ]
        )
        _draw_density(
            ax,
            c0_truth[parameter],
            limits,
            color=TRUTH,
            label="C0 truth",
            linewidth=2.1,
            fill_alpha=0.05,
        )
        _draw_density(
            ax,
            parent[:, index],
            limits,
            color=PARENT,
            label="Learned parent prior",
            linewidth=1.9,
        )
        _draw_density(
            ax,
            selected_truth[parameter],
            limits,
            color=RAW_Q,
            label="Selected truth",
            linestyle="--",
            linewidth=2.0,
        )
        _draw_density(
            ax,
            selected[:, index],
            limits,
            color=SELECTED,
            label="Selection-weighted prior",
            linewidth=1.8,
        )
        ax.set_xlim(limits)
        _finish_density_axis(ax, parameter)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(
        "Epoch 160: parent and observed-selected population priors",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.958,
        (
            "Parent target: p(theta | C0). Selected target: "
            "beta(theta) p(theta | C0) / alpha. The two targets are not merged."
        ),
        ha="center",
        va="top",
        fontsize=9.5,
        color=MUTED,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.932),
        ncol=4,
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89), h_pad=0.7, w_pad=0.4)
    return _save_png_pdf(fig, out, "population_prior_recovery")


def _plot_recovery_metrics(out: Path, recovery: pd.DataFrame) -> list[Path]:
    model_order = [
        "raw_q",
        "ema_q",
        "raw_iw",
        "ema_iw",
        "learned_parent_prior",
        "beta_weighted_selected_prior",
    ]
    model_labels = [
        "Raw q",
        "EMA q",
        "Raw IW",
        "EMA IW",
        "Parent prior",
        "Selected prior",
    ]
    index = pd.MultiIndex.from_product(
        [model_order, PARAMETERS], names=["model", "parameter"]
    )
    frame = recovery.set_index(["model", "parameter"]).reindex(index)
    if frame[["wasserstein_over_truth_iqr", "std_ratio"]].isna().any().any():
        raise ValueError("population recovery table lacks the expected model grid")
    wasserstein = frame["wasserstein_over_truth_iqr"].to_numpy().reshape(6, 15)
    std_ratio = frame["std_ratio"].to_numpy().reshape(6, 15)

    fig, axes = plt.subplots(2, 1, figsize=(15.5, 7.5), constrained_layout=True)
    first = axes[0].imshow(
        np.maximum(wasserstein, 1.0e-3),
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=0.03, vmax=max(30.0, float(np.nanmax(wasserstein)))),
    )
    axes[0].set_title("Marginal Wasserstein distance / truth IQR")
    fig.colorbar(first, ax=axes[0], label="W1 / IQR", pad=0.01)
    log_std = np.log2(np.maximum(std_ratio, 1.0e-4))
    limit = max(2.0, float(np.nanpercentile(np.abs(log_std), 95)))
    second = axes[1].imshow(
        log_std,
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axes[1].set_title("Aggregate scale recovery")
    fig.colorbar(second, ax=axes[1], label="log2(model std / truth std)", pad=0.01)
    for ax in axes:
        ax.set_yticks(np.arange(len(model_labels)), model_labels)
        ax.set_xticks(
            np.arange(len(PARAMETERS)), SHORT_PARAMETER_LABELS, rotation=45, ha="right"
        )
        ax.grid(False)
    fig.suptitle(
        "Population recovery metrics across all aggregate banks",
        fontsize=16,
        fontweight="bold",
    )
    return _save_png_pdf(fig, out, "population_recovery_metrics")


def _plot_correlation_residuals(
    out: Path,
    truth: pd.DataFrame,
    correlations: dict[str, np.ndarray],
) -> list[Path]:
    truth_correlation = np.corrcoef(
        truth.loc[:, PARAMETERS].to_numpy(float), rowvar=False
    )
    order = ["raw_q", "ema_q", "raw_iw", "ema_iw"]
    titles = ["Raw q", "EMA q", "Raw IW", "EMA IW"]
    residuals = [np.asarray(correlations[name]) - truth_correlation for name in order]
    limit = max(0.1, float(np.nanpercentile(np.abs(np.stack(residuals)), 99)))
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 10.8), constrained_layout=True)
    image = None
    for index, (ax, residual, title) in enumerate(
        zip(axes.flat, residuals, titles, strict=True)
    ):
        image = ax.imshow(
            residual,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.grid(False)
        ax.set_xticks(np.arange(15), SHORT_PARAMETER_LABELS, rotation=90, fontsize=7)
        ax.set_yticks(np.arange(15), SHORT_PARAMETER_LABELS, fontsize=7)
        if index % 2:
            ax.set_yticklabels([])
        if index < 2:
            ax.set_xticklabels([])
    assert image is not None
    fig.colorbar(
        image, ax=axes, label="aggregate correlation - selected truth", shrink=0.8
    )
    fig.suptitle(
        "Residual correlation structure of the selected population",
        fontsize=16,
        fontweight="bold",
    )
    return _save_png_pdf(fig, out, "population_correlation_residuals")


def _ecdf(values: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(_finite(values))
    if x.size == 0:
        raise ValueError("cannot compute ECDF from empty values")
    return x, np.arange(1, x.size + 1, dtype=np.float64) / x.size


def _plot_support(
    out: Path,
    diagnostics: dict[str, pd.DataFrame],
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.4), constrained_layout=True)
    specs = [
        ("raw_ess_fraction", "ESS / K", 0.05, True),
        ("max_raw_weight", "Maximum normalized weight", 0.50, False),
        ("pareto_k", "Pareto k", 0.70, False),
    ]
    colors = {"raw": RAW_Q, "ema": EMA_Q}
    for ax, (column, label, threshold, log_x) in zip(axes, specs, strict=True):
        for variant in ("raw", "ema"):
            values = diagnostics[variant][column]
            x, y = _ecdf(values)
            median = float(np.nanmedian(values))
            ax.plot(
                x,
                y,
                color=colors[variant],
                linewidth=2.2,
                label=f"{variant.upper()} (med={median:.3g})",
            )
        ax.axvline(
            threshold,
            color=EMA_IW,
            linestyle="--",
            linewidth=1.4,
            label=f"target {threshold:g}",
        )
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Empirical CDF")
        ax.set_ylim(0.0, 1.01)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        "Held-out ordinary-IW support at K=1024 (512 objects)",
        fontsize=15,
        fontweight="bold",
    )
    return _save_png_pdf(fig, out, "heldout_importance_support")


def _individual_inputs(
    frame: pd.DataFrame, row_index: int, parameter: str
) -> np.ndarray:
    return frame.loc[frame["row_index"].eq(row_index), parameter].to_numpy(float)


def _draw_individual_axis(
    ax: plt.Axes,
    *,
    parameter: str,
    parameter_index: int,
    row_index: int,
    parent: np.ndarray,
    raw_q: pd.DataFrame,
    ema_q: pd.DataFrame,
    ema_iw: pd.DataFrame,
    truth: float,
) -> None:
    raw_values = _individual_inputs(raw_q, row_index, parameter)
    ema_values = _individual_inputs(ema_q, row_index, parameter)
    iw_values = _individual_inputs(ema_iw, row_index, parameter)
    prior_values = parent[:, parameter_index]
    limits = _plot_limits(
        [prior_values, raw_values, ema_values, iw_values, np.asarray([truth])],
        tail=0.005,
    )
    _draw_density(
        ax,
        prior_values,
        limits,
        color=PARENT,
        label="Parent prior",
        linewidth=1.4,
        fill_alpha=0.08,
    )
    _draw_density(
        ax,
        raw_values,
        limits,
        color=RAW_Q,
        label="Raw q (256)",
        linestyle="--",
        linewidth=1.4,
    )
    _draw_density(
        ax,
        ema_values,
        limits,
        color=EMA_Q,
        label="EMA q (256)",
        linewidth=2.0,
        fill_alpha=0.10,
    )
    ymin, ymax = ax.get_ylim()
    rug_height = ymax * 0.055
    ax.vlines(
        iw_values,
        0.0,
        rug_height,
        color=EMA_IW,
        alpha=0.35,
        linewidth=0.8,
        label="EMA IW resamples (32)",
    )
    ax.axvline(truth, color=TRUTH, linewidth=2.0, label="Truth", zorder=8)
    ax.set_xlim(limits)
    _finish_density_axis(ax, parameter)


def _plot_individuals(
    out: Path,
    selected_rows: list[int],
    truth: pd.DataFrame,
    parent: np.ndarray,
    panels: dict[str, pd.DataFrame],
    manifest_rows: list[int],
) -> tuple[list[Path], pd.DataFrame]:
    truth_by_row = truth.set_index("row_index", drop=False)
    selection_records = []
    for row in selected_rows:
        record = truth_by_row.loc[row]
        object_id = int(record["object_id"])
        selection_records.append(
            {
                "row_index": row,
                "object_id": object_id,
                "observed_r_flux_quantile_rank": manifest_rows.index(row),
                "observed_r_flux_quantile_fraction": manifest_rows.index(row)
                / (len(manifest_rows) - 1),
                **{f"truth_{name}": float(record[name]) for name in PARAMETERS},
            }
        )
    selection = pd.DataFrame(selection_records)
    selection_path = out / "individual_selection.csv"
    selection.to_csv(selection_path, index=False)

    fig, axes = plt.subplots(
        len(selected_rows),
        len(PHYSICAL_PARAMETERS),
        figsize=(16.0, 2.35 * len(selected_rows)),
    )
    axes = np.asarray(axes).reshape(len(selected_rows), len(PHYSICAL_PARAMETERS))
    for row_number, row_index in enumerate(selected_rows):
        truth_row = truth_by_row.loc[row_index]
        for column, parameter in enumerate(PHYSICAL_PARAMETERS):
            ax = axes[row_number, column]
            _draw_individual_axis(
                ax,
                parameter=parameter,
                parameter_index=PARAMETERS.index(parameter),
                row_index=row_index,
                parent=parent,
                raw_q=panels["raw_q"],
                ema_q=panels["ema_q"],
                ema_iw=panels["ema_iw"],
                truth=float(truth_row[parameter]),
            )
            if column == 0:
                ax.set_ylabel(
                    f"row {row_index}\nobject {int(truth_row['object_id'])}",
                    fontsize=8,
                    color=INK,
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "Individual posterior examples across observed r-flux quantiles",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=5,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.945), h_pad=0.8, w_pad=0.5)
    figure_paths = _save_png_pdf(fig, out, "individual_posteriors_physical5d")

    multipage = out / "individual_posteriors_full15d.pdf"
    with PdfPages(multipage) as pdf:
        for row_index in selected_rows:
            truth_row = truth_by_row.loc[row_index]
            page, page_axes = plt.subplots(3, 5, figsize=(15.2, 9.2))
            for ax, parameter in zip(page_axes.flat, PARAMETERS, strict=True):
                _draw_individual_axis(
                    ax,
                    parameter=parameter,
                    parameter_index=PARAMETERS.index(parameter),
                    row_index=row_index,
                    parent=parent,
                    raw_q=panels["raw_q"],
                    ema_q=panels["ema_q"],
                    ema_iw=panels["ema_iw"],
                    truth=float(truth_row[parameter]),
                )
            page_handles, page_labels = page_axes.flat[0].get_legend_handles_labels()
            page.legend(
                page_handles,
                page_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.955),
                ncol=5,
                fontsize=9,
            )
            page.suptitle(
                f"row {row_index} | object {int(truth_row['object_id'])} | 15D posterior",
                fontsize=15,
                fontweight="bold",
                y=0.995,
            )
            page.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), h_pad=0.7, w_pad=0.4)
            pdf.savefig(page)
            plt.close(page)
    return [*figure_paths, multipage, selection_path], selection


def generate(root: Path, out: Path, *, individual_count: int = 6) -> dict[str, Any]:
    root = root.resolve()
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output: {out}")
    completion_path = root / "EPOCH160_EVALUATION_COMPLETE.json"
    completion = _read_json(completion_path)
    if (
        completion.get("status") != "DIAGNOSTIC_COMPLETE"
        or int(completion.get("epoch", -1)) != 160
        or completion.get("truth_used_for_training_or_checkpoint_selection")
        is not False
        or int(completion.get("catalogue_objects", -1)) != 4706
    ):
        raise ValueError("input is not the complete truth-safe epoch-160 bundle")

    source_paths = {
        "completion": completion_path,
        "selected_truth": root / "population/catalogue_selected_truth.parquet",
        "c0_truth": root / "population/test_population_truth_C0.parquet",
        "prior": root / "population/prior/parent_and_selected_prior.npz",
        "population_recovery": root / "population/population_recovery.csv",
        "posterior_correlations": root
        / "population/posterior_aggregate_correlations.npz",
        "individual_manifest": root / "population/individual_panels/manifest.json",
        **{
            f"aggregate_{name}": root / f"population/posterior_aggregate/{name}.parquet"
            for name in ("raw_q", "ema_q", "raw_iw", "ema_iw")
        },
        **{
            f"individual_{name}": root / f"population/individual_panels/{name}.parquet"
            for name in ("raw_q", "ema_q", "raw_iw", "ema_iw")
        },
        **{
            f"support_{variant}": root
            / f"heldout/{variant}_importance_diagnostics.parquet"
            for variant in ("raw", "ema")
        },
        **{
            f"support_summary_{variant}": root
            / f"heldout/{variant}_support_summary.json"
            for variant in ("raw", "ema")
        },
    }
    source_records = {name: _file_record(path) for name, path in source_paths.items()}

    selected_truth = pd.read_parquet(source_paths["selected_truth"])
    c0_truth = pd.read_parquet(source_paths["c0_truth"])
    if (
        selected_truth["row_index"].nunique() != 4706
        or c0_truth["row_index"].nunique() != 5000
    ):
        raise ValueError("truth cohort size differs from the frozen epoch-160 contract")
    expected_rows = set(selected_truth["row_index"].astype(int).tolist())
    banks = {
        name: pd.read_parquet(source_paths[f"aggregate_{name}"])
        for name in ("raw_q", "ema_q", "raw_iw", "ema_iw")
    }
    for name, frame in banks.items():
        _validate_dense_bank(
            frame,
            expected_rows=expected_rows,
            samples_per_object=32,
            label=f"aggregate {name}",
        )

    individual_manifest = _read_json(source_paths["individual_manifest"])
    manifest_rows = [int(value) for value in individual_manifest["row_indices"]]
    panel_rows = set(manifest_rows)
    panels = {
        name: pd.read_parquet(source_paths[f"individual_{name}"])
        for name in ("raw_q", "ema_q", "raw_iw", "ema_iw")
    }
    for name, frame in panels.items():
        _validate_dense_bank(
            frame,
            expected_rows=panel_rows,
            samples_per_object=256 if name.endswith("_q") else 32,
            label=f"individual {name}",
        )
    selected_rows = _select_panel_rows(manifest_rows, individual_count)

    with np.load(source_paths["prior"], allow_pickle=False) as arrays:
        parent = np.asarray(arrays["theta"], dtype=np.float64)
        selected_prior = np.asarray(arrays["selected_theta"], dtype=np.float64)
    if parent.shape[1:] != (15,) or selected_prior.shape[1:] != (15,):
        raise ValueError("prior sample geometry differs from the 15D contract")
    support_summaries = {
        variant: _read_json(source_paths[f"support_summary_{variant}"])
        for variant in ("raw", "ema")
    }
    support_diagnostics = {
        variant: pd.read_parquet(source_paths[f"support_{variant}"])
        for variant in ("raw", "ema")
    }
    if any(len(frame) != 512 for frame in support_diagnostics.values()):
        raise ValueError("support diagnostics do not contain the 512 held-out objects")

    staging = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    _configure_plotting()
    generated = []
    generated.extend(
        _plot_selected_marginals(staging, selected_truth, banks, support_summaries)
    )
    generated.extend(
        _plot_prior_recovery(staging, c0_truth, selected_truth, parent, selected_prior)
    )
    recovery = pd.read_csv(source_paths["population_recovery"])
    generated.extend(_plot_recovery_metrics(staging, recovery))
    with np.load(source_paths["posterior_correlations"], allow_pickle=False) as arrays:
        correlations = {name: np.asarray(arrays[name]) for name in arrays.files}
    generated.extend(_plot_correlation_residuals(staging, selected_truth, correlations))
    generated.extend(_plot_support(staging, support_diagnostics))
    individual_files, selection = _plot_individuals(
        staging,
        selected_rows,
        selected_truth,
        parent,
        panels,
        manifest_rows,
    )
    generated.extend(individual_files)

    artifact_records = {
        path.relative_to(staging).as_posix(): {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in generated
    }
    manifest = {
        "status": "COMPLETE",
        "epoch": 160,
        "git_sha": _git_sha(),
        "source_root": str(root),
        "output_root": str(out),
        "catalogue_cohort": completion["catalogue_cohort"],
        "catalogue_objects": 4706,
        "population_draws_per_object": 32,
        "population_joint_draws_per_model": 4706 * 32,
        "heldout_objects": 512,
        "heldout_importance_draws_per_object": 1024,
        "individual_objects": int(len(selection)),
        "individual_q_draws_per_object": 256,
        "individual_iw_resamples_per_object": 32,
        "parameters": list(PARAMETERS),
        "individual_row_indices": selected_rows,
        "contracts": {
            "population_aggregation": (
                "object-equal mixture of dense joint draws; no posterior is replaced "
                "by a point estimate"
            ),
            "parent_prior": "p_eta(theta | C0)",
            "selected_prior": "beta(theta) p_eta(theta | C0) / alpha_eta",
            "individual_prior": "learned parent prior p_eta(theta | C0)",
            "truth_role": "post-freeze visualization only",
            "iw_warning": (
                "IW curves and rugs are diagnostic resamples; the held-out K=1024 "
                "median ESS is approximately six effective samples per object"
            ),
            "axis_range": "pooled robust 0.2%-99.8% range (0.5%-99.5% for individuals)",
        },
        "support": support_summaries,
        "sources": source_records,
        "artifacts": artifact_records,
    }
    manifest_path = staging / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(staging, out)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/feniks_epoch160_complete"),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--individual-count", type=int, default=6)
    args = parser.parse_args()
    out = args.out or args.root / "aggregated_figures_v1"
    result = generate(args.root, out, individual_count=args.individual_count)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[epoch160-plots] complete -> {out}")


if __name__ == "__main__":
    main()
