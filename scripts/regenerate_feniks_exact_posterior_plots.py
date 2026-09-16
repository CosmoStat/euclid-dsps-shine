#!/usr/bin/env python3
"""Regenerate pairwise FENIKS posterior plots from saved artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

KEY_PARAMETERS = (
    "z_obs",
    "log10_stellar_mass",
    "dust_av",
    "dust_delta",
    "sfh_dlog_sfr_01",
)

PLOT_VARIANTS = (
    ("prior_encoder", None),
    ("prior_encoder_nuts", "nuts"),
    ("prior_encoder_map", "map"),
    ("prior_encoder_is", "is"),
)

TRACE_VARIANTS = (
    ("prior_encoder_nuts", None),
    ("prior_encoder_nuts_map", "map"),
    ("prior_encoder_nuts_is", "is"),
)

COLORS = {
    "prior": "#6F706A",
    "encoder": "#0072B2",
    "nuts": "#D55E00",
    "map": "#E69F00",
    "is": "#009E73",
    "truth": "#171717",
}

CHAIN_COLORS = ("#D55E00", "#CC79A7", "#7A5AF8", "#A0522D")

DISPLAY_NAMES = {
    "z_obs": "redshift",
    "log10_stellar_mass": "log stellar mass",
    "log10_stellar_metallicity": "log metallicity",
    "dust_av": "dust A_V",
    "dust_delta": "dust slope",
    **{f"sfh_dlog_sfr_{index:02d}": f"SFH dlog SFR {index}" for index in range(1, 11)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--prior-samples", type=Path, required=True)
    parser.add_argument(
        "--out-name",
        default="publication_plots_pairwise_v1",
        help="Output directory created inside each root.",
    )
    parser.add_argument("--max-corner-samples", type=int, default=2500)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_corner_samples < 200:
        raise ValueError("--max-corner-samples must be at least 200")
    if args.dpi < 100:
        raise ValueError("--dpi must be at least 100")
    prior_path = args.prior_samples.resolve()
    if not prior_path.is_file():
        raise FileNotFoundError(prior_path)
    prior = pd.read_parquet(prior_path)
    summaries = []
    for root in args.root:
        summaries.append(
            regenerate_root(
                root.resolve(),
                prior,
                prior_path=prior_path,
                out_name=args.out_name,
                max_corner_samples=int(args.max_corner_samples),
                dpi=int(args.dpi),
                overwrite=bool(args.overwrite),
            )
        )
    print(json.dumps(summaries, indent=2, sort_keys=True, allow_nan=False))


def regenerate_root(
    root: Path,
    prior: pd.DataFrame,
    *,
    prior_path: Path,
    out_name: str,
    max_corner_samples: int,
    dpi: int,
    overwrite: bool,
) -> dict[str, Any]:
    _require(root / "DONE")
    contract_path = _require(root / "big_run_contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_prior_label = _expected_prior_label(contract)
    _validate_prior_identity(prior_path, expected_prior_label)
    cohort = pd.read_parquet(_require(root / "cohort.parquet"))
    if len(cohort) != 2:
        raise ValueError(f"Expected two galaxies in {root}, got {len(cohort)}")

    out = root / out_name
    if out.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    prior_copy = out / "learned_prior_samples.parquet"
    shutil.copy2(prior_path, prior_copy)

    generated: list[Path] = [prior_copy]
    galaxy_summaries = []
    for item in cohort.itertuples(index=False):
        galaxy = (
            root
            / "galaxies"
            / f"{int(item.order):02d}_{item.example_key}_row{int(item.row_index)}"
        )
        galaxy_out = out / galaxy.name
        galaxy_out.mkdir(parents=True, exist_ok=True)
        summary, paths = regenerate_galaxy(
            galaxy,
            galaxy_out,
            prior,
            item=item,
            max_corner_samples=max_corner_samples,
            dpi=dpi,
        )
        galaxy_summaries.append(summary)
        generated.extend(paths)

    readme = out / "README.md"
    readme.write_text(
        _readme_text(root, expected_prior_label, galaxy_summaries),
        encoding="utf-8",
    )
    generated.append(readme)
    manifest_path = out / "plot_manifest.json"
    manifest = {
        "schema_version": 1,
        "source_root": str(root),
        "source_code_commit": contract.get("code_commit"),
        "plot_code_commit": _git_commit(),
        "expected_prior_label": expected_prior_label,
        "prior_source": str(prior_path),
        "prior_source_sha256": _sha256(prior_path),
        "prior_rows": int(len(prior)),
        "variants": [name for name, _method in PLOT_VARIANTS],
        "trace_variants": [name for name, _method in TRACE_VARIANTS],
        "galaxies": galaxy_summaries,
        "style": {
            "colors": COLORS,
            "corner_levels": [0.68, 0.95],
            "axis_range_quantiles": {
                "prior": [0.01, 0.99],
                "posterior_methods": [0.005, 0.995],
            },
            "dpi": dpi,
            "max_corner_samples": max_corner_samples,
        },
        "generated_files": [],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    manifest["generated_files"] = [
        {
            "path": str(path.relative_to(out)),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for path in sorted(generated)
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "root": str(root),
        "output": str(out),
        "galaxies": len(galaxy_summaries),
        "generated_files": len(generated),
    }


def regenerate_galaxy(
    galaxy: Path,
    out: Path,
    prior: pd.DataFrame,
    *,
    item: Any,
    max_corner_samples: int,
    dpi: int,
) -> tuple[dict[str, Any], list[Path]]:
    manifest = json.loads(
        _require(galaxy / "prepare_manifest.json").read_text(encoding="utf-8")
    )
    names = tuple(manifest["latent_spec"]["names"])
    if len(names) != 15 or not set(KEY_PARAMETERS).issubset(names):
        raise ValueError(f"Unexpected latent parameter contract in {galaxy}")
    _require_columns(prior, names, "learned prior")
    frames = {
        "prior": prior,
        "encoder": pd.read_parquet(_require(galaxy / "encoder_samples.parquet")),
        "is": pd.read_parquet(
            _require(galaxy / "importance_resampled_samples.parquet")
        ),
        "nuts": pd.read_parquet(_require(galaxy / "nuts/samples.parquet")),
        "map": pd.read_parquet(_require(galaxy / "map_solutions.parquet")),
        "truth": pd.read_parquet(_require(galaxy / "truth.parquet")),
    }
    for method in ("encoder", "is", "nuts", "map", "truth"):
        _require_columns(frames[method], names, method)
    if frames["truth"].shape[0] != 1:
        raise ValueError(f"Expected one truth row in {galaxy}")
    map_best = (
        frames["map"]
        .loc[np.isfinite(frames["map"]["objective"])]
        .sort_values("objective")
        .iloc[0]
    )
    truth = frames["truth"].iloc[0]
    diagnostics = pd.read_parquet(
        _require(galaxy / "nuts/diagnostics.parquet")
    ).set_index("parameter")
    diagnostics_json = json.loads(
        _require(galaxy / "nuts/diagnostics.json").read_text(encoding="utf-8")
    )
    ranges = _parameter_ranges(
        names,
        prior=frames["prior"],
        posterior_frames=(frames["encoder"], frames["is"], frames["nuts"]),
        points=(truth, map_best),
        lower=manifest["latent_spec"].get("lower"),
        upper=manifest["latent_spec"].get("upper"),
    )
    ranges_path = out / "plot_ranges.json"
    ranges_path.write_text(
        json.dumps(ranges, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    title = (
        f"{str(item.example_key).replace('_', ' ').title()} galaxy | "
        f"catalog row {int(item.row_index)} | object {item.object_id}"
    )
    nuts_warning = (
        f"NUTS not converged: max R-hat {float(diagnostics_json['max_rhat']):.2f}, "
        f"min bulk ESS {float(diagnostics_json['min_bulk_ess']):.1f}"
    )
    generated = [ranges_path]
    subsets = (("key5", KEY_PARAMETERS), ("full15", names))
    for subset_name, subset in subsets:
        for variant, comparison in PLOT_VARIANTS:
            stem = f"corner_{subset_name}_{variant}"
            figure = _corner_figure(
                frames,
                subset,
                ranges=ranges,
                truth=truth,
                map_best=map_best,
                comparison=comparison,
                title=title,
                nuts_warning=nuts_warning,
                max_samples=max_corner_samples,
            )
            generated.extend(_save_figure(figure, out / stem, dpi=dpi))

            stem = f"marginals_{subset_name}_{variant}"
            figure = _marginal_figure(
                frames,
                subset,
                ranges=ranges,
                truth=truth,
                map_best=map_best,
                comparison=comparison,
                title=title,
                nuts_warning=nuts_warning,
            )
            generated.extend(_save_figure(figure, out / stem, dpi=dpi))

        for variant, comparison in TRACE_VARIANTS:
            stem = f"mcmc_trace_{subset_name}_{variant}"
            figure = _trace_figure(
                frames,
                subset,
                ranges=ranges,
                truth=truth,
                map_best=map_best,
                diagnostics=diagnostics,
                comparison=comparison,
                title=title,
                nuts_warning=nuts_warning,
            )
            generated.extend(_save_figure(figure, out / stem, dpi=dpi))

    source_paths = {
        "encoder": galaxy / "encoder_samples.parquet",
        "importance_resampled": galaxy / "importance_resampled_samples.parquet",
        "map": galaxy / "map_solutions.parquet",
        "nuts": galaxy / "nuts/samples.parquet",
        "nuts_diagnostics": galaxy / "nuts/diagnostics.parquet",
        "truth": galaxy / "truth.parquet",
    }
    summary = {
        "galaxy": galaxy.name,
        "row_index": int(item.row_index),
        "object_id": str(item.object_id),
        "parameters": list(names),
        "nuts_converged": bool(
            diagnostics_json.get("passes_rhat_1_01")
            and diagnostics_json.get("passes_bulk_ess_400")
            and diagnostics_json.get("passes_tail_ess_400")
        ),
        "nuts_max_rhat": float(diagnostics_json["max_rhat"]),
        "nuts_min_bulk_ess": float(diagnostics_json["min_bulk_ess"]),
        "source_artifacts": {
            name: {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            for name, path in source_paths.items()
        },
        "generated_files": [
            str(path.relative_to(out.parent)) for path in sorted(generated)
        ],
    }
    galaxy_manifest = out / "plot_manifest.json"
    galaxy_manifest.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    generated.append(galaxy_manifest)
    return summary, generated


def _corner_figure(
    frames: dict[str, pd.DataFrame],
    parameters: tuple[str, ...],
    *,
    ranges: dict[str, list[float]],
    truth: pd.Series,
    map_best: pd.Series,
    comparison: str | None,
    title: str,
    nuts_warning: str,
    max_samples: int,
):
    labels = [DISPLAY_NAMES.get(name, name) for name in parameters]
    plot_ranges = [tuple(ranges[name]) for name in parameters]
    figure = None
    layers = (
        ("prior", COLORS["prior"], "--", 1.25),
        ("encoder", COLORS["encoder"], "-", 1.8),
    )
    if comparison in {"nuts", "is"}:
        layers += ((comparison, COLORS[comparison], "-", 1.65),)
    for method, color, linestyle, linewidth in layers:
        values = _subsample(
            frames[method].loc[:, parameters].to_numpy(dtype=float),
            max_samples,
        )
        figure = corner.corner(
            values,
            labels=labels,
            range=plot_ranges,
            bins=34,
            smooth=0.9,
            smooth1d=0.9,
            fig=figure,
            color=color,
            plot_datapoints=False,
            plot_density=True,
            fill_contours=False,
            levels=(0.68, 0.95),
            hist_kwargs={
                "linewidth": linewidth,
                "linestyle": linestyle,
            },
            contour_kwargs={
                "linewidths": linewidth,
                "linestyles": linestyle,
            },
            quiet=True,
        )
    ndim = len(parameters)
    axes = np.asarray(figure.axes).reshape((ndim, ndim))
    truth_values = truth.loc[list(parameters)].to_numpy(dtype=float)
    map_values = map_best.loc[list(parameters)].to_numpy(dtype=float)
    for row in range(ndim):
        diagonal = axes[row, row]
        diagonal.axvline(
            truth_values[row], color=COLORS["truth"], linewidth=1.2, zorder=8
        )
        if comparison == "map":
            diagonal.axvline(
                map_values[row],
                color=COLORS["map"],
                linewidth=1.5,
                linestyle="-.",
                zorder=9,
            )
        for col in range(row):
            axis = axes[row, col]
            axis.scatter(
                truth_values[col],
                truth_values[row],
                marker="*",
                color=COLORS["truth"],
                edgecolor="white",
                linewidth=0.35,
                s=34 if ndim <= 5 else 18,
                zorder=10,
            )
            if comparison == "map":
                axis.scatter(
                    map_values[col],
                    map_values[row],
                    marker="D",
                    facecolor=COLORS["map"],
                    edgecolor="white",
                    linewidth=0.4,
                    s=24 if ndim <= 5 else 12,
                    zorder=10,
                )
    legend = _comparison_legend(comparison, include_nuts_chains=False)
    figure.legend(
        handles=legend,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        fontsize=9 if ndim <= 5 else 7,
    )
    figure.suptitle(title, x=0.06, y=0.995, ha="left", fontsize=14)
    if comparison == "nuts":
        figure.text(
            0.06,
            0.974,
            nuts_warning,
            ha="left",
            va="top",
            color="#A33A2B",
            fontsize=9,
            weight="semibold",
        )
    for axis in figure.axes:
        axis.tick_params(labelsize=7 if ndim > 5 else 9, colors="#3E403B")
        for spine in axis.spines.values():
            spine.set_color("#C8CAC4")
    figure.subplots_adjust(top=0.95, right=0.985)
    return figure


def _marginal_figure(
    frames: dict[str, pd.DataFrame],
    parameters: tuple[str, ...],
    *,
    ranges: dict[str, list[float]],
    truth: pd.Series,
    map_best: pd.Series,
    comparison: str | None,
    title: str,
    nuts_warning: str,
):
    ncols = 3 if len(parameters) > 5 else 2
    nrows = math.ceil(len(parameters) / ncols)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 2.75 * nrows + 1.0),
        squeeze=False,
    )
    for axis, parameter in zip(axes.ravel(), parameters, strict=False):
        low, high = ranges[parameter]
        for method, linestyle, linewidth in (
            ("prior", "--", 1.4),
            ("encoder", "-", 2.0),
        ):
            grid, density = _density_curve(
                frames[method][parameter].to_numpy(dtype=float),
                low,
                high,
            )
            axis.plot(
                grid,
                density,
                color=COLORS[method],
                linestyle=linestyle,
                linewidth=linewidth,
            )
            if method == "encoder":
                axis.fill_between(grid, density, color=COLORS[method], alpha=0.10)
        if comparison in {"nuts", "is"}:
            grid, density = _density_curve(
                frames[comparison][parameter].to_numpy(dtype=float),
                low,
                high,
            )
            axis.plot(
                grid,
                density,
                color=COLORS[comparison],
                linewidth=1.7,
            )
        axis.axvline(
            float(truth[parameter]),
            color=COLORS["truth"],
            linewidth=1.2,
        )
        if comparison == "map":
            axis.axvline(
                float(map_best[parameter]),
                color=COLORS["map"],
                linewidth=1.6,
                linestyle="-.",
            )
        axis.set_xlim(low, high)
        axis.set_title(
            DISPLAY_NAMES.get(parameter, parameter),
            loc="left",
            fontsize=10,
            weight="semibold",
        )
        axis.set_yticks([])
        axis.grid(axis="x", color="#E8E9E5", linewidth=0.7)
        axis.tick_params(axis="x", labelsize=8)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#B9BBB5")
    for axis in axes.ravel()[len(parameters) :]:
        axis.axis("off")
    figure.legend(
        handles=_comparison_legend(comparison, include_nuts_chains=False),
        loc="upper right",
        bbox_to_anchor=(0.98, 0.985),
        frameon=False,
        ncol=2,
        fontsize=9,
    )
    figure.suptitle(title, x=0.04, y=0.995, ha="left", fontsize=14)
    if comparison == "nuts":
        figure.text(
            0.04,
            0.966,
            nuts_warning,
            ha="left",
            va="top",
            color="#A33A2B",
            fontsize=9,
            weight="semibold",
        )
    figure.tight_layout(rect=(0.025, 0.02, 0.99, 0.94))
    return figure


def _trace_figure(
    frames: dict[str, pd.DataFrame],
    parameters: tuple[str, ...],
    *,
    ranges: dict[str, list[float]],
    truth: pd.Series,
    map_best: pd.Series,
    diagnostics: pd.DataFrame,
    comparison: str | None,
    title: str,
    nuts_warning: str,
):
    ncols = 3 if len(parameters) > 5 else 1
    nrows = math.ceil(len(parameters) / ncols)
    figure_width = 5.4 * ncols if ncols > 1 else 10.0
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figure_width, 2.0 * nrows + 1.3),
        squeeze=False,
        sharex=True,
    )
    nuts = frames["nuts"]
    for axis, parameter in zip(axes.ravel(), parameters, strict=False):
        prior_q = np.quantile(_finite(frames["prior"][parameter]), (0.16, 0.5, 0.84))
        encoder_q = np.quantile(
            _finite(frames["encoder"][parameter]), (0.16, 0.5, 0.84)
        )
        xmin = float(nuts["draw"].min())
        xmax = float(nuts["draw"].max())
        axis.axhspan(prior_q[0], prior_q[2], color=COLORS["prior"], alpha=0.08)
        axis.axhline(
            prior_q[1],
            color=COLORS["prior"],
            linestyle="--",
            linewidth=1.0,
        )
        axis.axhspan(encoder_q[0], encoder_q[2], color=COLORS["encoder"], alpha=0.10)
        axis.axhline(
            encoder_q[1],
            color=COLORS["encoder"],
            linewidth=1.1,
        )
        if comparison == "is":
            importance_q = np.quantile(
                _finite(frames["is"][parameter]), (0.16, 0.5, 0.84)
            )
            axis.axhspan(
                importance_q[0],
                importance_q[2],
                color=COLORS["is"],
                alpha=0.08,
            )
            axis.axhline(importance_q[1], color=COLORS["is"], linewidth=1.1)
        if comparison == "map":
            axis.axhline(
                float(map_best[parameter]),
                color=COLORS["map"],
                linestyle="-.",
                linewidth=1.4,
            )
        axis.axhline(
            float(truth[parameter]),
            color=COLORS["truth"],
            linewidth=1.0,
            zorder=7,
        )
        for chain_index, (chain, group) in enumerate(nuts.groupby("chain", sort=True)):
            axis.plot(
                group["draw"],
                group[parameter],
                color=CHAIN_COLORS[chain_index % len(CHAIN_COLORS)],
                linewidth=0.55,
                alpha=0.72,
                label=f"chain {int(chain) + 1}",
            )
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(*ranges[parameter])
        axis.set_title(
            DISPLAY_NAMES.get(parameter, parameter),
            loc="left",
            fontsize=9,
            weight="semibold",
        )
        rhat = float(diagnostics.loc[parameter, "rhat"])
        axis.text(
            0.99,
            0.93,
            f"R-hat {rhat:.2f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color="#A33A2B" if rhat > 1.01 else "#406343",
            fontsize=8,
            weight="semibold",
        )
        axis.grid(color="#E8E9E5", linewidth=0.6)
        axis.tick_params(labelsize=7)
        for spine in axis.spines.values():
            spine.set_color("#C8CAC4")
    for axis in axes.ravel()[len(parameters) :]:
        axis.axis("off")
    for axis in axes[-1]:
        if axis.axison:
            axis.set_xlabel("stored NUTS draw", fontsize=8)
    figure.legend(
        handles=_comparison_legend(comparison, include_nuts_chains=True),
        loc="upper left",
        bbox_to_anchor=(0.03, 0.952),
        frameon=False,
        ncol=5,
        fontsize=8,
    )
    figure.suptitle(
        f"{title} | NUTS chain traces",
        x=0.03,
        y=0.995,
        ha="left",
        fontsize=14,
    )
    figure.text(
        0.03,
        0.966,
        nuts_warning,
        ha="left",
        va="top",
        color="#A33A2B",
        fontsize=9,
        weight="semibold",
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.90))
    return figure


def _comparison_legend(
    comparison: str | None,
    *,
    include_nuts_chains: bool,
) -> list[Any]:
    handles: list[Any] = [
        Line2D(
            [0],
            [0],
            color=COLORS["prior"],
            linestyle="--",
            linewidth=1.5,
            label="Learned prior",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["encoder"],
            linewidth=2.0,
            label="Encoder posterior",
        ),
    ]
    if include_nuts_chains:
        handles.append(
            Line2D(
                [0],
                [0],
                color=CHAIN_COLORS[0],
                linewidth=1.2,
                label="NUTS chains",
            )
        )
    elif comparison == "nuts":
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS["nuts"],
                linewidth=1.8,
                label="NUTS (not converged)",
            )
        )
    if comparison == "map":
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS["map"],
                linestyle="-.",
                linewidth=1.6,
                marker="D",
                markersize=5,
                label="MAP",
            )
        )
    if comparison == "is":
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS["is"],
                linewidth=1.7,
                label="Encoder + IS",
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            color=COLORS["truth"],
            linewidth=1.2,
            marker="*",
            markersize=7,
            label="Truth",
        )
    )
    return handles


def _parameter_ranges(
    names: tuple[str, ...],
    *,
    prior: pd.DataFrame,
    posterior_frames: tuple[pd.DataFrame, ...],
    points: tuple[pd.Series, ...],
    lower: list[float] | None,
    upper: list[float] | None,
) -> dict[str, list[float]]:
    result = {}
    for index, name in enumerate(names):
        prior_values = _finite(prior[name])
        low, high = np.quantile(prior_values, (0.01, 0.99))
        for frame in posterior_frames:
            values = _finite(frame[name])
            frame_low, frame_high = np.quantile(values, (0.005, 0.995))
            low = min(low, frame_low)
            high = max(high, frame_high)
        for point in points:
            value = float(point[name])
            if math.isfinite(value):
                low = min(low, value)
                high = max(high, value)
        span = high - low
        if not math.isfinite(span) or span <= 0:
            center = float(low) if math.isfinite(low) else 0.0
            low, high = center - 0.5, center + 0.5
        else:
            low -= 0.045 * span
            high += 0.045 * span
        if lower is not None:
            low = max(low, float(lower[index]))
        if upper is not None:
            high = min(high, float(upper[index]))
        if high <= low:
            raise ValueError(f"Invalid display range for {name}: {low}, {high}")
        result[name] = [float(low), float(high)]
    return result


def _density_curve(
    values: np.ndarray,
    low: float,
    high: float,
    *,
    bins: int = 90,
) -> tuple[np.ndarray, np.ndarray]:
    values = _finite(values)
    clipped = values[(values >= low) & (values <= high)]
    if len(clipped) < 5:
        clipped = values
    density, edges = np.histogram(
        clipped,
        bins=bins,
        range=(low, high),
        density=True,
    )
    density = gaussian_filter1d(density.astype(float), sigma=1.35)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density


def _subsample(values: np.ndarray, size: int) -> np.ndarray:
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    if len(values) <= size:
        return values
    indices = np.linspace(0, len(values) - 1, size, dtype=int)
    return values[indices]


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = array[np.isfinite(array)]
    if result.size == 0:
        raise ValueError("No finite values available for plotting")
    return result


def _expected_prior_label(contract: dict[str, Any]) -> str:
    checkpoint = Path(str(contract.get("checkpoint", "")))
    if len(checkpoint.parents) < 3:
        raise ValueError("Cannot derive learned-prior label from run contract")
    label = checkpoint.parents[2].name
    if not label:
        raise ValueError("Empty learned-prior label in run contract")
    return label


def _validate_prior_identity(path: Path, expected_label: str) -> None:
    if expected_label not in path.parts:
        raise ValueError(f"Prior samples do not belong to {expected_label}: {path}")


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing parameter columns: {missing}")


def _save_figure(figure, stem: Path, *, dpi: int) -> list[Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [png, pdf]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _readme_text(
    root: Path,
    prior_label: str,
    galaxies: list[dict[str, Any]],
) -> str:
    lines = [
        "# FENIKS pairwise posterior figures",
        "",
        f"Source run: `{root}`",
        f"Learned prior: `{prior_label}`",
        "",
        "Every corner and marginal figure contains the learned prior and the",
        "encoder posterior. Separate variants add NUTS, MAP, or encoder plus",
        "importance sampling. Truth is retained as a point reference.",
        "",
        "NUTS warning: these chains are technically complete but not converged.",
        "Do not use their combined samples as a reference posterior.",
        "",
        "## Galaxies",
        "",
    ]
    for galaxy in galaxies:
        lines.append(
            f"- `{galaxy['galaxy']}`: max R-hat "
            f"{galaxy['nuts_max_rhat']:.3f}, min bulk ESS "
            f"{galaxy['nuts_min_bulk_ess']:.1f}"
        )
    lines.extend(
        [
            "",
            "The parquet inputs, display ranges, style constants, hashes, and",
            "generated-file inventory are recorded in `plot_manifest.json`.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
