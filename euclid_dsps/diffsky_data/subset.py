"""Prepared Diffsky subset builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from euclid_dsps.photometric_uncertainty import (
    flux_error_from_model,
    flux_error_model_payload,
    normalize_flux_error_model,
)

DEFAULT_TRUTH_COLUMNS = (
    "redshift_true",
    "logsm_true",
    "logssfr_true",
    "logsfr_true",
    "logmp_true",
    "central_true",
)


def build_redshift_subset(
    *,
    dataset_path: Path,
    output_path: Path,
    redshift_column: str = "redshift_true",
    redshift_min: float = 0.0,
    redshift_max: float = 0.5,
    max_objects: int | None = None,
    seed: int = 42,
    error_model: dict[str, Any] | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Write a redshift-filtered prepared parquet plus manifest/diagnostics."""
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    frame = pd.read_parquet(dataset_path)
    if redshift_column not in frame:
        raise ValueError(f"{dataset_path} does not contain {redshift_column!r}")
    z = pd.to_numeric(frame[redshift_column], errors="coerce").to_numpy(dtype=float)
    selected = np.isfinite(z) & (z >= float(redshift_min)) & (z <= float(redshift_max))
    subset = frame.loc[selected].copy()
    if max_objects is not None and len(subset) > int(max_objects):
        rng = np.random.default_rng(int(seed))
        take = rng.choice(len(subset), size=int(max_objects), replace=False)
        subset = subset.iloc[np.sort(take)].copy()
    subset.reset_index(drop=True, inplace=True)
    if "object_id" in subset:
        subset["source_object_id"] = subset["object_id"].to_numpy()
    subset["object_id"] = np.arange(len(subset), dtype=np.int64)
    if error_model is not None:
        _materialize_flux_errors(subset, error_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(output_path, index=False)
    report = _subset_report(
        source=dataset_path,
        output=output_path,
        frame=subset,
        redshift_column=redshift_column,
        redshift_min=redshift_min,
        redshift_max=redshift_max,
        max_objects=max_objects,
        error_model=error_model,
    )
    manifest_path = output_path.with_suffix(".manifest.yaml")
    manifest_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_schema_json(subset, report, output_path.with_suffix(".schema.json"))
    _write_truth_distribution_table(subset, output_path.with_suffix(".truth_summary.csv"))
    _write_markdown_report(report, output_path.with_suffix(".report.md"))
    _write_truth_report(subset, report, output_path.with_suffix(".truth_report.md"))
    if make_plots:
        _write_distribution_plots(subset, output_path)
    return report


def _materialize_flux_errors(frame: pd.DataFrame, model: dict[str, Any]) -> None:
    cfg = normalize_flux_error_model(model)
    for column in list(frame.columns):
        if not column.startswith("flux_") or column.startswith("fluxerr_"):
            continue
        band = column.removeprefix("flux_")
        frame[f"fluxerr_{band}"] = flux_error_from_model(
            frame[column].to_numpy(),
            cfg,
            band_name=band,
        )


def _subset_report(
    *,
    source: Path,
    output: Path,
    frame: pd.DataFrame,
    redshift_column: str,
    redshift_min: float,
    redshift_max: float,
    max_objects: int | None,
    error_model: dict[str, Any] | None,
) -> dict[str, Any]:
    z = pd.to_numeric(frame[redshift_column], errors="coerce").to_numpy(dtype=float)
    z_finite = z[np.isfinite(z)]
    return {
        "type": "diffsky_redshift_subset",
        "source_dataset": str(source),
        "output_path": str(output),
        "n_objects": int(len(frame)),
        "redshift_filter": {
            "column": redshift_column,
            "min": float(redshift_min),
            "max": float(redshift_max),
            "inclusive": True,
        },
        "max_objects": None if max_objects is None else int(max_objects),
        "redshift_summary": _series_summary(z_finite),
        "truth_columns": [column for column in DEFAULT_TRUTH_COLUMNS if column in frame],
        "band_names": [
            column.removeprefix("flux_")
            for column in frame.columns
            if column.startswith("flux_") and not column.startswith("fluxerr_")
        ],
        "error_model": (
            "preserved"
            if error_model is None
            else flux_error_model_payload(error_model)
        ),
    }


def _write_truth_distribution_table(frame: pd.DataFrame, path: Path) -> None:
    rows = []
    for column in DEFAULT_TRUTH_COLUMNS:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        rows.append({"column": column, **_series_summary(values)})
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_schema_json(frame: pd.DataFrame, report: dict[str, Any], path: Path) -> None:
    band_names = list(report["band_names"])
    schema = {
        "latent_schema": "diffsky_redshift_subset",
        "object_id_column": "object_id",
        "source_object_id_column": "source_object_id" if "source_object_id" in frame else None,
        "source_dataset": report["source_dataset"],
        "band_names": band_names,
        "photometry": {
            "prepared_flux_unit": "fnu_cgs",
            "flux_columns": [f"flux_{band}" for band in band_names],
            "flux_error_columns": [
                f"fluxerr_{band}" for band in band_names if f"fluxerr_{band}" in frame
            ],
            "error_model": report["error_model"],
        },
        "truth": {
            "columns": list(report["truth_columns"]),
            "redshift": report["redshift_filter"]["column"],
        },
        "parameters": _parameter_summaries(frame),
    }
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def _parameter_summaries(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    columns = [
        column
        for column in frame.columns
        if column.endswith("_true")
        or column.startswith(("diffmah_", "diffstar_", "dust_", "burst_"))
    ]
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        summary = _series_summary(values)
        if int(summary["n_finite"]) == 0:
            continue
        rows.append({"name": column, "column": column, **summary})
    return rows


def _series_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n_finite": 0,
            "min": None,
            "p16": None,
            "median": None,
            "p84": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "n_finite": int(values.size),
        "min": float(np.min(values)),
        "p16": float(np.quantile(values, 0.16)),
        "median": float(np.median(values)),
        "p84": float(np.quantile(values, 0.84)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _write_markdown_report(report: dict[str, Any], path: Path) -> None:
    redshift = report["redshift_summary"]
    lines = [
        "# Diffsky Redshift Subset Report",
        "",
        f"- Source: `{report['source_dataset']}`",
        f"- Output: `{report['output_path']}`",
        f"- Objects: `{report['n_objects']}`",
        f"- Redshift column: `{report['redshift_filter']['column']}`",
        f"- Redshift range: `{report['redshift_filter']['min']}` to `{report['redshift_filter']['max']}`",
        "",
        "## Redshift Summary",
        "",
        f"- finite: `{redshift['n_finite']}`",
        f"- median: `{redshift['median']}`",
        f"- p16/p84: `{redshift['p16']}` / `{redshift['p84']}`",
        f"- min/max: `{redshift['min']}` / `{redshift['max']}`",
        "",
        "## Error Model",
        "",
        f"```json\n{json.dumps(report['error_model'], indent=2)}\n```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_truth_report(frame: pd.DataFrame, report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Diffsky Subset Truth Report",
        "",
        f"- objects: {len(frame)}",
        f"- bands: {', '.join(report['band_names'])}",
        f"- source: `{report['source_dataset']}`",
        f"- output: `{report['output_path']}`",
        "",
        "## Truth Columns",
        "",
    ]
    truth_columns = [column for column in DEFAULT_TRUTH_COLUMNS if column in frame]
    if not truth_columns:
        lines.append("_None._")
    for column in truth_columns:
        summary = _series_summary(
            pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        )
        lines.append(
            "- "
            f"`{column}`: n={summary['n_finite']}, "
            f"median={summary['median']}, "
            f"p16/p84={summary['p16']}/{summary['p84']}, "
            f"min/max={summary['min']}/{summary['max']}"
        )
    lines.extend(
        [
            "",
            "## Error Model",
            "",
            f"```json\n{json.dumps(report['error_model'], indent=2)}\n```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_distribution_plots(frame: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    plot_dir = output_path.with_suffix("")
    plot_dir.mkdir(parents=True, exist_ok=True)
    if "redshift_true" in frame:
        fig, ax = plt.subplots(figsize=(7, 4))
        values = pd.to_numeric(frame["redshift_true"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        ax.hist(values, bins=80, histtype="step", density=True, lw=1.8)
        ax.set_xlabel("redshift_true")
        ax.set_ylabel("density")
        ax.set_title("Diffsky subset redshift distribution")
        fig.tight_layout()
        fig.savefig(plot_dir / "redshift_true_distribution.png", dpi=150)
        plt.close(fig)
    truth_cols = [
        column for column in DEFAULT_TRUTH_COLUMNS if column in frame and column != "redshift_true"
    ]
    if truth_cols:
        n_cols = min(3, len(truth_cols))
        n_rows = int(np.ceil(len(truth_cols) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.0 * n_cols, 3.0 * n_rows),
            squeeze=False,
        )
        for ax, column in zip(axes.ravel(), truth_cols, strict=False):
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                ax.hist(values, bins=50, histtype="stepfilled", alpha=0.55)
            ax.set_title(column)
        for ax in axes.ravel()[len(truth_cols) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(plot_dir / "truth_distributions.png", dpi=150)
        plt.close(fig)
    _write_flux_error_diagnostics(frame, plot_dir, plt)


def _write_flux_error_diagnostics(frame: pd.DataFrame, plot_dir: Path, plt: Any) -> None:
    bands = [
        column.removeprefix("flux_")
        for column in frame.columns
        if column.startswith("flux_")
        and not column.startswith("fluxerr_")
        and f"fluxerr_{column.removeprefix('flux_')}" in frame
    ]
    if not bands:
        return
    rows = []
    frac_by_band = []
    snr_by_band = []
    scatter_payload = []
    rng = np.random.default_rng(12345)
    for band in bands:
        flux = pd.to_numeric(frame[f"flux_{band}"], errors="coerce").to_numpy(dtype=float)
        err = pd.to_numeric(frame[f"fluxerr_{band}"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(flux) & np.isfinite(err) & (np.abs(flux) > 0.0) & (err > 0.0)
        if not np.any(ok):
            continue
        abs_flux = np.abs(flux[ok])
        err = err[ok]
        frac = err / abs_flux
        snr = abs_flux / err
        rows.append(
            {
                "band": band,
                "n": int(abs_flux.size),
                "flux_median": float(np.median(abs_flux)),
                "fluxerr_median": float(np.median(err)),
                "fracerr_median": float(np.median(frac)),
                "fracerr_p16": float(np.quantile(frac, 0.16)),
                "fracerr_p84": float(np.quantile(frac, 0.84)),
                "snr_median": float(np.median(snr)),
                "snr_p16": float(np.quantile(snr, 0.16)),
                "snr_p84": float(np.quantile(snr, 0.84)),
            }
        )
        frac_by_band.append(np.log10(frac))
        snr_by_band.append(np.log10(snr))
        take_n = min(3000, abs_flux.size)
        take = rng.choice(abs_flux.size, size=take_n, replace=False)
        scatter_payload.append((band, np.log10(abs_flux[take]), np.log10(err[take])))
    if not rows:
        return
    pd.DataFrame(rows).to_csv(plot_dir / "flux_error_summary.csv", index=False)
    labels = [row["band"] for row in rows]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(labels)), 4.5))
    ax.boxplot(frac_by_band, labels=labels, showfliers=False)
    ax.set_ylabel("log10(fluxerr / abs(flux))")
    ax.set_title("Synthetic fractional flux error by band")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plot_dir / "flux_fractional_error_by_band.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(labels)), 4.5))
    ax.boxplot(snr_by_band, labels=labels, showfliers=False)
    ax.set_ylabel("log10(abs(flux) / fluxerr)")
    ax.set_title("Synthetic catalog SNR by band")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plot_dir / "flux_snr_by_band.png", dpi=150)
    plt.close(fig)
    n_cols = min(4, len(scatter_payload))
    n_rows = int(np.ceil(len(scatter_payload) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )
    for ax, (band, log_flux, log_err) in zip(axes.ravel(), scatter_payload, strict=False):
        ax.scatter(log_flux, log_err, s=2, alpha=0.25, rasterized=True)
        ax.set_title(band)
        ax.set_xlabel("log10(abs(flux))")
        ax.set_ylabel("log10(fluxerr)")
    for ax in axes.ravel()[len(scatter_payload) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(plot_dir / "flux_vs_fluxerr_by_band.png", dpi=150)
    plt.close(fig)
