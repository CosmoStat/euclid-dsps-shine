"""Population diagnostics for generated Diffsky/FENIKS closure catalogs."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir
from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES

from .config import SPLIT_ORDER
from .photometry import GROUND_TRUTH_COLUMNS
from .reference_comparison import compare_synthetic_closure_to_reference

DEFAULT_DIAGNOSTIC_BANDS = (
    "lsst_u",
    "lsst_g",
    "lsst_r",
    "lsst_i",
    "lsst_z",
    "lsst_y",
    "roman_F062",
    "roman_F087",
    "roman_F106",
    "roman_F129",
    "roman_F146",
    "roman_F158",
    "roman_F184",
    "roman_F213",
)

PHYSICAL_COLUMNS = (
    ("redshift", "redshift_true"),
    ("logsm", "logsm_true"),
    ("logsfr", "logsfr_true"),
    ("logssfr", "logssfr_true"),
    ("logmp", "logmp_true"),
    ("logmp0", "logmp0_true"),
    ("central", "central_true"),
    ("lgmet_abs_median", "lgmet_abs_median_true"),
    ("metallicity_scatter_dex", "metallicity_scatter_dex"),
)

COLOR_PAIRS = (
    ("lsst_u", "lsst_g"),
    ("lsst_g", "lsst_r"),
    ("lsst_r", "lsst_i"),
    ("lsst_i", "lsst_z"),
    ("lsst_z", "lsst_y"),
    ("roman_F062", "roman_F087"),
    ("roman_F087", "roman_F106"),
    ("roman_F106", "roman_F158"),
    ("roman_F158", "roman_F213"),
)

PROPOSAL_COMPARE_COLUMNS = (
    ("redshift", "redshift_true"),
    ("logsm", "logsm_true"),
    ("logsfr", "logsfr_true"),
    ("logssfr", "logssfr_true"),
    ("logmp", "logmp_true"),
    ("logmp0", "logmp0_true"),
    ("dust_av", "dust_av_true"),
    ("dust_delta", "dust_delta_true"),
    ("log10_stellar_metallicity", "log10_stellar_metallicity_true"),
)


def run_generation_population_diagnostics(
    config: dict[str, Any],
    *,
    dataset_dir: str | Path,
    smoke: bool = False,
) -> Path | None:
    """Run configured post-generation diagnostics and return summary path."""
    diagnostics_cfg = dict(
        (config.get("synthetic_diffsky", {}) or {}).get("diagnostics", {}) or {}
    )
    if not bool(diagnostics_cfg.get("enabled", False)):
        return None
    dataset_dir = Path(dataset_dir)
    out = ensure_dir(dataset_dir / str(diagnostics_cfg.get("output_subdir", "diagnostics/population")))
    bands = tuple(str(band["name"]) for band in config.get("bands", [])) or DEFAULT_DIAGNOSTIC_BANDS
    frames = _load_split_frames(dataset_dir)
    if not frames:
        raise ValueError(f"No split parquets found under {dataset_dir}")
    all_frame = pd.concat(frames.values(), ignore_index=True)
    sample_seed = int(diagnostics_cfg.get("seed", 260617))
    max_rows = int(_runtime_value(diagnostics_cfg, "max_rows", 50_000, smoke=smoke))
    max_plot_rows = int(
        _runtime_value(diagnostics_cfg, "max_plot_rows", 20_000, smoke=smoke)
    )
    corner_max_rows = int(
        _runtime_value(diagnostics_cfg, "corner_max_rows", 5_000, smoke=smoke)
    )
    sampled = _sample_frame(all_frame, max_rows, sample_seed)
    plot_sample = _sample_frame(all_frame, max_plot_rows, sample_seed + 1)
    corner_sample = _sample_frame(all_frame, corner_max_rows, sample_seed + 2)

    parameter_stats = _parameter_stats(frames, all_frame)
    parameter_stats_path = out / "parameter_stats.csv"
    parameter_stats.to_csv(parameter_stats_path, index=False)

    photometry_stats = _photometry_stats(frames, all_frame, bands)
    photometry_stats_path = out / "photometry_stats.csv"
    photometry_stats.to_csv(photometry_stats_path, index=False)

    error_model_stats = _error_model_stats(frames, all_frame, bands)
    error_model_stats_path = out / "error_model_stats.csv"
    error_model_stats.to_csv(error_model_stats_path, index=False)

    color_stats = _color_stats(frames, all_frame, bands)
    color_stats_path = out / "color_stats.csv"
    color_stats.to_csv(color_stats_path, index=False)

    correlations = _correlation_payload(sampled)
    correlation_path = out / "correlation_matrices.json"
    _write_json(correlation_path, correlations)

    proposal_metrics = _proposal_final_metrics(dataset_dir, frames)
    proposal_metrics_path = out / "proposal_vs_final_metrics.csv"
    proposal_metrics.to_csv(proposal_metrics_path, index=False)

    plots = {}
    if bool(diagnostics_cfg.get("make_plots", True)):
        plots = _write_plots(
            out=ensure_dir(out / "plots"),
            frame=plot_sample,
            corner_frame=corner_sample,
            bands=bands,
            make_corner=bool(diagnostics_cfg.get("make_corner", True)),
            max_corner_parameters=int(diagnostics_cfg.get("max_corner_parameters", 18)),
        )

    reference_outputs: dict[str, str] = {}
    reference_path = diagnostics_cfg.get("reference_dataset")
    if reference_path:
        reference_outputs = _run_reference_comparison(
            config,
            dataset_dir=dataset_dir,
            all_frame=all_frame,
            reference_path=Path(reference_path),
            diagnostics_cfg=diagnostics_cfg,
            out=ensure_dir(out / "reference_comparison"),
            bands=bands,
            seed=sample_seed,
            smoke=smoke,
        )
    for reference_cfg in list(diagnostics_cfg.get("reference_datasets", []) or []):
        if not isinstance(reference_cfg, dict) or not reference_cfg.get("path"):
            continue
        name = str(reference_cfg.get("name", Path(str(reference_cfg["path"])).stem))
        merged_cfg = dict(diagnostics_cfg)
        merged_cfg.update(reference_cfg)
        outputs = _run_reference_comparison(
            config,
            dataset_dir=dataset_dir,
            all_frame=all_frame,
            reference_path=Path(str(reference_cfg["path"])),
            diagnostics_cfg=merged_cfg,
            out=ensure_dir(out / f"{name}_comparison"),
            bands=tuple(reference_cfg.get("bands", bands) or bands),
            seed=sample_seed,
            smoke=smoke,
        )
        reference_outputs[name] = outputs

    summary = _summary_payload(
        config=config,
        dataset_dir=dataset_dir,
        out=out,
        all_frame=all_frame,
        frames=frames,
        parameter_stats=parameter_stats,
        photometry_stats=photometry_stats,
        error_model_stats=error_model_stats,
        color_stats=color_stats,
        proposal_metrics=proposal_metrics,
        plots=plots,
        reference_outputs=reference_outputs,
        smoke=smoke,
    )
    summary_path = out / "population_diagnostics_summary.json"
    _write_json(summary_path, summary)
    (out / "report.md").write_text(_markdown_report(summary), encoding="utf-8")
    return summary_path


def _load_split_frames(dataset_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLIT_ORDER:
        path = dataset_dir / f"{split}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            if "split" not in frame.columns:
                frame = frame.copy()
                frame["split"] = split
            frames[split] = frame
    return frames


def _parameter_stats(
    frames: dict[str, pd.DataFrame],
    all_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    named_frames = {**frames, "all": all_frame}
    columns = [
        (name, GROUND_TRUTH_COLUMNS[name])
        for name in DIFFSKY_BASIC_PARAMETER_NAMES
        if GROUND_TRUTH_COLUMNS[name] in all_frame.columns
    ]
    columns.extend((name, column) for name, column in PHYSICAL_COLUMNS if column in all_frame)
    seen: set[str] = set()
    for label, column in columns:
        if column in seen:
            continue
        seen.add(column)
        for split, frame in named_frames.items():
            rows.append(_stats_row(split, "parameter", label, column, frame[column]))
    return pd.DataFrame(rows)


def _photometry_stats(
    frames: dict[str, pd.DataFrame],
    all_frame: pd.DataFrame,
    bands: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    named_frames = {**frames, "all": all_frame}
    for split, frame in named_frames.items():
        for band in bands:
            for kind, column in (
                ("mag_true", f"mag_true_{band}"),
                ("flux_true", f"flux_true_{band}"),
                ("flux", f"flux_{band}"),
                ("fluxerr", f"fluxerr_{band}"),
                ("mask", f"mask_{band}"),
            ):
                if column in frame.columns:
                    rows.append(_stats_row(split, kind, band, column, frame[column]))
            snr = _ratio_from_columns(frame, f"flux_{band}", f"fluxerr_{band}")
            rows.append(_array_stats_row(split, "snr", band, f"flux_{band}/fluxerr_{band}", snr))
    return pd.DataFrame(rows)


def _color_stats(
    frames: dict[str, pd.DataFrame],
    all_frame: pd.DataFrame,
    bands: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    named_frames = {**frames, "all": all_frame}
    for split, frame in named_frames.items():
        for left, right in COLOR_PAIRS:
            if left not in bands or right not in bands:
                continue
            color = _difference_from_columns(frame, f"mag_true_{left}", f"mag_true_{right}")
            rows.append(
                _array_stats_row(
                    split,
                    "color_true",
                    f"{left}-{right}",
                    f"mag_true_{left}-mag_true_{right}",
                    color,
                )
            )
    return pd.DataFrame(rows)


def _error_model_stats(
    frames: dict[str, pd.DataFrame],
    all_frame: pd.DataFrame,
    bands: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    named_frames = {**frames, "all": all_frame}
    for split, frame in named_frames.items():
        for band in bands:
            required = {
                f"flux_true_{band}",
                f"flux_{band}",
                f"fluxerr_{band}",
                f"mag_true_{band}",
            }
            if not required <= set(frame.columns):
                continue
            flux_true = pd.to_numeric(frame[f"flux_true_{band}"], errors="coerce").to_numpy(float)
            flux = pd.to_numeric(frame[f"flux_{band}"], errors="coerce").to_numpy(float)
            fluxerr = pd.to_numeric(frame[f"fluxerr_{band}"], errors="coerce").to_numpy(float)
            mag = pd.to_numeric(frame[f"mag_true_{band}"], errors="coerce").to_numpy(float)
            finite = (
                np.isfinite(flux_true)
                & np.isfinite(flux)
                & np.isfinite(fluxerr)
                & np.isfinite(mag)
                & (fluxerr > 0.0)
            )
            residual = (flux[finite] - flux_true[finite]) / fluxerr[finite]
            snr = flux[finite] / fluxerr[finite]
            true_snr = flux_true[finite] / fluxerr[finite]
            rows.append(
                {
                    "split": split,
                    "band": band,
                    "n": int(np.count_nonzero(finite)),
                    "error_model_applied": bool(np.count_nonzero(finite) == len(frame)),
                    "residual_mean": float(np.mean(residual)) if residual.size else np.nan,
                    "residual_std": float(np.std(residual)) if residual.size else np.nan,
                    "residual_q01": float(np.quantile(residual, 0.01)) if residual.size else np.nan,
                    "residual_q99": float(np.quantile(residual, 0.99)) if residual.size else np.nan,
                    "fluxerr_median": float(np.median(fluxerr[finite])) if np.any(finite) else np.nan,
                    "fluxerr_q16": float(np.quantile(fluxerr[finite], 0.16)) if np.any(finite) else np.nan,
                    "fluxerr_q84": float(np.quantile(fluxerr[finite], 0.84)) if np.any(finite) else np.nan,
                    "snr_median": float(np.median(snr)) if snr.size else np.nan,
                    "true_snr_median": float(np.median(true_snr)) if true_snr.size else np.nan,
                    "mag_median": float(np.median(mag[finite])) if np.any(finite) else np.nan,
                    "negative_noisy_flux_fraction": (
                        float(np.mean(flux[finite] < 0.0)) if np.any(finite) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _proposal_final_metrics(
    dataset_dir: Path,
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, final in frames.items():
        proposal_dir = dataset_dir / "proposals" / split
        proposals = _read_proposals(proposal_dir)
        if proposals is None or "galaxy_weight" not in proposals.columns:
            continue
        weights = pd.to_numeric(proposals["galaxy_weight"], errors="coerce").to_numpy(float)
        for label, column in PROPOSAL_COMPARE_COLUMNS:
            if column not in proposals.columns or column not in final.columns:
                continue
            prop = pd.to_numeric(proposals[column], errors="coerce").to_numpy(float)
            fin = pd.to_numeric(final[column], errors="coerce").to_numpy(float)
            finite = np.isfinite(prop) & np.isfinite(weights) & (weights > 0.0)
            prop = prop[finite]
            prop_w = weights[finite]
            fin = _finite_array(fin)
            row: dict[str, Any] = {
                "split": split,
                "quantity": label,
                "column": column,
                "proposal_n": int(prop.size),
                "final_n": int(fin.size),
                "proposal_weight_sum": float(np.sum(prop_w)) if prop_w.size else np.nan,
                "weighted_proposal_mean": _weighted_mean(prop, prop_w),
                "weighted_proposal_q05": _weighted_quantile(prop, prop_w, 0.05),
                "weighted_proposal_median": _weighted_quantile(prop, prop_w, 0.50),
                "weighted_proposal_q95": _weighted_quantile(prop, prop_w, 0.95),
                "final_mean": float(np.mean(fin)) if fin.size else np.nan,
                "final_median": float(np.median(fin)) if fin.size else np.nan,
                "ks_unweighted": _ks_distance(prop, fin),
                "wasserstein_quantile": _wasserstein_quantile(prop, fin),
            }
            if label == "logsm":
                for threshold in (5.0, 7.0, 8.0, 9.0, 10.0):
                    row[f"weighted_proposal_fraction_ge_{threshold:g}"] = (
                        float(np.sum(prop_w[prop >= threshold]) / np.sum(prop_w))
                        if prop_w.size and np.sum(prop_w) > 0.0
                        else np.nan
                    )
                    row[f"final_fraction_ge_{threshold:g}"] = (
                        float(np.mean(fin >= threshold)) if fin.size else np.nan
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def _correlation_payload(frame: pd.DataFrame) -> dict[str, Any]:
    truth_columns = [
        GROUND_TRUTH_COLUMNS[name]
        for name in DIFFSKY_BASIC_PARAMETER_NAMES
        if GROUND_TRUTH_COLUMNS[name] in frame.columns
    ]
    if len(truth_columns) < 2:
        return {"status": "insufficient_columns", "columns": truth_columns}
    matrix = frame[truth_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    finite = np.isfinite(matrix).all(axis=1)
    matrix = matrix[finite]
    dynamic = np.std(matrix, axis=0) > 0.0 if len(matrix) else np.zeros(len(truth_columns), dtype=bool)
    columns = [column for column, keep in zip(truth_columns, dynamic, strict=True) if keep]
    matrix = matrix[:, dynamic] if len(matrix) else matrix
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return {
            "status": "insufficient_rows_or_dynamic_range",
            "columns": columns,
            "n_rows": int(matrix.shape[0]),
        }
    corr = np.corrcoef(matrix, rowvar=False)
    return {
        "status": "ok",
        "columns": columns,
        "n_rows": int(matrix.shape[0]),
        "correlation": corr,
    }


def _write_plots(
    *,
    out: Path,
    frame: pd.DataFrame,
    corner_frame: pd.DataFrame,
    bands: Sequence[str],
    make_corner: bool,
    max_corner_parameters: int,
) -> dict[str, str]:
    try:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(ensure_dir(Path(tempfile.gettempdir()) / "euclid_dsps_matplotlib")),
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}
    outputs: dict[str, str] = {}
    latent_columns = [
        (name, GROUND_TRUTH_COLUMNS[name])
        for name in DIFFSKY_BASIC_PARAMETER_NAMES
        if GROUND_TRUTH_COLUMNS[name] in frame.columns
    ]
    path = _hist_grid(
        plt,
        out / "truth_parameter_histograms.png",
        frame,
        latent_columns,
        bins=45,
    )
    if path:
        outputs["truth_parameter_histograms"] = str(path)
    physical_columns = [
        (name, column) for name, column in PHYSICAL_COLUMNS if column in frame.columns
    ]
    path = _hist_grid(plt, out / "physical_diagnostic_histograms.png", frame, physical_columns, bins=45)
    if path:
        outputs["physical_diagnostic_histograms"] = str(path)
    mag_columns = [(band, f"mag_true_{band}") for band in bands if f"mag_true_{band}" in frame.columns]
    path = _hist_grid(plt, out / "magnitude_histograms.png", frame, mag_columns, bins=45)
    if path:
        outputs["magnitude_histograms"] = str(path)
    color_frame = _color_frame(frame, bands)
    path = _hist_grid(
        plt,
        out / "color_histograms.png",
        color_frame,
        [(column, column) for column in color_frame.columns],
        bins=45,
    )
    if path:
        outputs["color_histograms"] = str(path)
    path = _band_summary_plot(plt, out / "photometry_band_summary.png", frame, bands)
    if path:
        outputs["photometry_band_summary"] = str(path)
    error_paths = _write_error_model_plots(plt, out, frame, bands)
    outputs.update(error_paths)
    path = _mass_redshift_plot(plt, out / "mass_redshift_sfr_dust.png", frame)
    if path:
        outputs["mass_redshift_sfr_dust"] = str(path)
    if make_corner:
        corner_outputs = _write_corner_plots(
            out=out,
            frame=corner_frame,
            latent_columns=latent_columns,
            max_corner_parameters=max_corner_parameters,
        )
        outputs.update(corner_outputs)
    plt.close("all")
    return outputs


def _hist_grid(
    plt: Any,
    path: Path,
    frame: pd.DataFrame,
    columns: Sequence[tuple[str, str]],
    *,
    bins: int,
) -> Path | None:
    columns = [(label, column) for label, column in columns if column in frame.columns]
    if not columns:
        return None
    ncols = min(4, len(columns))
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (label, column) in zip(axes_arr, columns, strict=False):
        values = _finite_numeric(frame[column])
        if values.size:
            ax.hist(values, bins=min(bins, max(8, values.size // 3)), histtype="stepfilled", alpha=0.65)
        ax.set_title(label, fontsize=9)
    for ax in axes_arr[len(columns):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _band_summary_plot(
    plt: Any,
    path: Path,
    frame: pd.DataFrame,
    bands: Sequence[str],
) -> Path | None:
    rows = []
    for band in bands:
        mag_col = f"mag_true_{band}"
        flux_col = f"flux_{band}"
        err_col = f"fluxerr_{band}"
        if mag_col not in frame:
            continue
        mag = _finite_numeric(frame[mag_col])
        snr = _ratio_from_columns(frame, flux_col, err_col)
        rows.append(
            {
                "band": band,
                "mag_median": float(np.median(mag)) if mag.size else np.nan,
                "mag_q16": float(np.quantile(mag, 0.16)) if mag.size else np.nan,
                "mag_q84": float(np.quantile(mag, 0.84)) if mag.size else np.nan,
                "snr_median": float(np.median(snr)) if snr.size else np.nan,
            }
        )
    if not rows:
        return None
    table = pd.DataFrame(rows)
    x = np.arange(len(table))
    fig, axes = plt.subplots(2, 1, figsize=(max(8.0, 0.55 * len(table)), 6.5), sharex=True)
    axes[0].errorbar(
        x,
        table["mag_median"],
        yerr=[
            table["mag_median"] - table["mag_q16"],
            table["mag_q84"] - table["mag_median"],
        ],
        fmt="o",
    )
    axes[0].invert_yaxis()
    axes[0].set_ylabel("mag_true median")
    axes[1].plot(x, table["snr_median"], marker="o")
    axes[1].set_ylabel("median S/N")
    axes[1].set_xticks(x, table["band"], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_error_model_plots(
    plt: Any,
    out: Path,
    frame: pd.DataFrame,
    bands: Sequence[str],
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    summary_rows = []
    residuals_by_band: dict[str, np.ndarray] = {}
    for band in bands:
        cols = [f"flux_true_{band}", f"flux_{band}", f"fluxerr_{band}", f"mag_true_{band}"]
        if not set(cols) <= set(frame.columns):
            continue
        flux_true = pd.to_numeric(frame[cols[0]], errors="coerce").to_numpy(float)
        flux = pd.to_numeric(frame[cols[1]], errors="coerce").to_numpy(float)
        fluxerr = pd.to_numeric(frame[cols[2]], errors="coerce").to_numpy(float)
        mag = pd.to_numeric(frame[cols[3]], errors="coerce").to_numpy(float)
        finite = (
            np.isfinite(flux_true)
            & np.isfinite(flux)
            & np.isfinite(fluxerr)
            & np.isfinite(mag)
            & (fluxerr > 0.0)
        )
        if not np.any(finite):
            continue
        residual = _finite_array((flux[finite] - flux_true[finite]) / fluxerr[finite])
        residuals_by_band[band] = residual
        summary_rows.append(
            {
                "band": band,
                "fluxerr_median": float(np.median(fluxerr[finite])),
                "fluxerr_q16": float(np.quantile(fluxerr[finite], 0.16)),
                "fluxerr_q84": float(np.quantile(fluxerr[finite], 0.84)),
                "true_snr_median": float(np.median(flux_true[finite] / fluxerr[finite])),
                "mag_median": float(np.median(mag[finite])),
            }
        )
    if summary_rows:
        table = pd.DataFrame(summary_rows)
        x = np.arange(len(table))
        fig, axes = plt.subplots(3, 1, figsize=(max(8.0, 0.55 * len(table)), 8.0), sharex=True)
        axes[0].errorbar(
            x,
            table["fluxerr_median"],
            yerr=[
                table["fluxerr_median"] - table["fluxerr_q16"],
                table["fluxerr_q84"] - table["fluxerr_median"],
            ],
            fmt="o",
        )
        axes[0].set_ylabel("fluxerr fnu_cgs")
        axes[0].set_yscale("log")
        axes[1].plot(x, table["true_snr_median"], marker="o")
        axes[1].set_ylabel("median true S/N")
        axes[1].set_yscale("symlog", linthresh=1.0)
        axes[2].plot(x, table["mag_median"], marker="o")
        axes[2].invert_yaxis()
        axes[2].set_ylabel("median mag_true")
        axes[2].set_xticks(x, table["band"], rotation=45, ha="right")
        fig.tight_layout()
        path = out / "error_model_band_summary.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs["error_model_band_summary"] = str(path)
    if residuals_by_band:
        ncols = 4
        nrows = int(np.ceil(len(residuals_by_band) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows))
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, (band, residual) in zip(axes_arr, residuals_by_band.items(), strict=False):
            if residual.size:
                ax.hist(residual, bins=min(50, max(8, residual.size // 2)), histtype="stepfilled", alpha=0.7)
                ax.axvline(0.0, color="black", lw=0.8)
            ax.set_title(band, fontsize=9)
            ax.set_xlabel("(flux - flux_true) / fluxerr")
        for ax in axes_arr[len(residuals_by_band):]:
            ax.axis("off")
        fig.tight_layout()
        path = out / "normalized_noise_residual_histograms.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs["normalized_noise_residual_histograms"] = str(path)
    rows = []
    for band in bands:
        mag_col = f"mag_true_{band}"
        err_col = f"fluxerr_{band}"
        if mag_col not in frame.columns or err_col not in frame.columns:
            continue
        mag = pd.to_numeric(frame[mag_col], errors="coerce").to_numpy(float)
        err = pd.to_numeric(frame[err_col], errors="coerce").to_numpy(float)
        finite = np.isfinite(mag) & np.isfinite(err) & (err > 0.0)
        if np.any(finite):
            rows.append((band, mag[finite], err[finite]))
    if rows:
        ncols = 4
        nrows = int(np.ceil(len(rows) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows))
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, (band, mag, err) in zip(axes_arr, rows, strict=False):
            ax.scatter(mag, err, s=3, alpha=0.2)
            ax.set_title(band, fontsize=9)
            ax.set_xlabel("mag_true")
            ax.set_ylabel("fluxerr")
            ax.set_yscale("log")
        for ax in axes_arr[len(rows):]:
            ax.axis("off")
        fig.tight_layout()
        path = out / "fluxerr_vs_mag_true.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs["fluxerr_vs_mag_true"] = str(path)
    return outputs


def _mass_redshift_plot(plt: Any, path: Path, frame: pd.DataFrame) -> Path | None:
    needed = {"redshift_true", "logsm_true"}
    if not needed <= set(frame.columns):
        return None
    z = _finite_numeric(frame["redshift_true"])
    mass = _finite_numeric(frame["logsm_true"])
    n = min(z.size, mass.size)
    if n < 2:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    axes[0].scatter(z[:n], mass[:n], s=4, alpha=0.25)
    axes[0].set_xlabel("redshift_true")
    axes[0].set_ylabel("logsm_true")
    if "logssfr_true" in frame:
        y = _finite_numeric(frame["logssfr_true"])
        n2 = min(z.size, y.size)
        axes[1].scatter(z[:n2], y[:n2], s=4, alpha=0.25)
    axes[1].set_xlabel("redshift_true")
    axes[1].set_ylabel("logssfr_true")
    if "dust_av_true" in frame:
        y = _finite_numeric(frame["dust_av_true"])
        n3 = min(mass.size, y.size)
        axes[2].scatter(mass[:n3], y[:n3], s=4, alpha=0.25)
    axes[2].set_xlabel("logsm_true")
    axes[2].set_ylabel("dust_av_true")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_corner_plots(
    *,
    out: Path,
    frame: pd.DataFrame,
    latent_columns: Sequence[tuple[str, str]],
    max_corner_parameters: int,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    columns = [
        (label, column)
        for label, column in latent_columns[: int(max_corner_parameters)]
        if column in frame.columns and _has_dynamic_range(frame[column])
    ]
    core_names = (
        "redshift_true",
        "logsm_true",
        "log10_stellar_metallicity_true",
        "dust_av_true",
        "dust_delta_true",
        "logsfr_true",
        "logssfr_true",
        "diffmah_logm0_true",
    )
    core = [
        (label, column)
        for label, column in columns
        if column in core_names or label in {"z_obs", "log10_stellar_mass"}
    ]
    core.extend(
        (name, name)
        for name in ("logsfr_true", "logssfr_true")
        if name in frame.columns and _has_dynamic_range(frame[name])
    )
    outputs.update(_corner_or_scatter_matrix(out, frame, core[:8], "corner_core_truths"))
    outputs.update(_corner_or_scatter_matrix(out, frame, columns, "corner_18_truths"))
    return outputs


def _corner_or_scatter_matrix(
    out: Path,
    frame: pd.DataFrame,
    columns: Sequence[tuple[str, str]],
    name: str,
) -> dict[str, str]:
    columns = list(dict(columns).items())
    if len(columns) < 2:
        return {}
    labels = [label for label, _ in columns]
    data = frame[[column for _, column in columns]].apply(pd.to_numeric, errors="coerce")
    matrix = data.to_numpy(dtype=float)
    finite = np.isfinite(matrix).all(axis=1)
    matrix = matrix[finite]
    if matrix.shape[0] < 3:
        return {}
    if matrix.shape[0] < 100:
        return _scatter_matrix(out, matrix, labels, name)
    try:
        import corner

        fig = corner.corner(
            matrix,
            labels=labels,
            plot_datapoints=False,
            fill_contours=True,
            hist_kwargs={"density": True},
        )
        path = out / f"{name}.png"
        fig.savefig(path, dpi=150)
        import matplotlib.pyplot as plt

        plt.close(fig)
        return {name: str(path)}
    except Exception:
        return _scatter_matrix(out, matrix, labels, name)


def _scatter_matrix(
    out: Path,
    matrix: np.ndarray,
    labels: Sequence[str],
    name: str,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    n = len(labels)
    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(matrix[:, j], bins=35, histtype="stepfilled", alpha=0.65)
            elif i > j:
                ax.scatter(matrix[:, j], matrix[:, i], s=2, alpha=0.15)
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=7)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=7)
            ax.tick_params(labelsize=6)
    fig.tight_layout()
    path = out / f"{name}_scatter_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {f"{name}_scatter_matrix": str(path)}


def _run_reference_comparison(
    config: dict[str, Any],
    *,
    dataset_dir: Path,
    all_frame: pd.DataFrame,
    reference_path: Path,
    diagnostics_cfg: dict[str, Any],
    out: Path,
    bands: Sequence[str],
    seed: int,
    smoke: bool,
) -> dict[str, str]:
    if not reference_path.exists():
        return {"status": "missing_reference", "reference_path": str(reference_path)}
    synthetic_path = dataset_dir / "all_50k.parquet"
    temp_path: Path | None = None
    if bool(diagnostics_cfg.get("reference_overlap_only", True)):
        z_min = float((config.get("synthetic_diffsky", {}) or {}).get("z_min", 0.0))
        reference_z_max = diagnostics_cfg.get("reference_z_max")
        if reference_z_max is None:
            reference = pd.read_parquet(reference_path, columns=["redshift_true"])
            reference_z_max = float(np.nanmax(reference["redshift_true"].to_numpy(float)))
        overlap = all_frame[
            (pd.to_numeric(all_frame["redshift_true"], errors="coerce") >= z_min)
            & (pd.to_numeric(all_frame["redshift_true"], errors="coerce") <= float(reference_z_max))
        ]
        temp_path = out / "synthetic_overlap_reference_z.parquet"
        overlap.to_parquet(temp_path, index=False)
        synthetic_path = temp_path
    outputs = compare_synthetic_closure_to_reference(
        synthetic_path=synthetic_path,
        reference_path=reference_path,
        out_dir=out,
        proposal_dir=dataset_dir / "proposals",
        bands=bands,
        max_reference=diagnostics_cfg.get("reference_max_rows"),
        seed=seed,
        plots=bool(diagnostics_cfg.get("make_reference_plots", False)) and not smoke,
        reference_kind=str(diagnostics_cfg.get("reference_kind", "auto")),
    )
    return {key: str(value) for key, value in outputs.items()}


def _summary_payload(
    *,
    config: dict[str, Any],
    dataset_dir: Path,
    out: Path,
    all_frame: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    parameter_stats: pd.DataFrame,
    photometry_stats: pd.DataFrame,
    error_model_stats: pd.DataFrame,
    color_stats: pd.DataFrame,
    proposal_metrics: pd.DataFrame,
    plots: dict[str, str],
    reference_outputs: dict[str, str],
    smoke: bool,
) -> dict[str, Any]:
    z = _finite_numeric(all_frame["redshift_true"]) if "redshift_true" in all_frame else np.asarray([])
    logsm = _finite_numeric(all_frame["logsm_true"]) if "logsm_true" in all_frame else np.asarray([])
    mass_fractions = {
        f"logsm_ge_{threshold:g}": float(np.mean(logsm >= threshold)) if logsm.size else np.nan
        for threshold in (5.0, 7.0, 8.0, 9.0, 10.0)
    }
    bands = tuple(str(band["name"]) for band in config.get("bands", [])) or DEFAULT_DIAGNOSTIC_BANDS
    all_error = (
        error_model_stats[error_model_stats["split"] == "all"]
        if "split" in error_model_stats
        else pd.DataFrame()
    )
    residual_summary = []
    for _, row in all_error.iterrows():
        residual_summary.append(
            {
                "band": str(row.get("band", "")),
                "n": int(row.get("n", 0)),
                "residual_mean": float(row.get("residual_mean", np.nan)),
                "residual_std": float(row.get("residual_std", np.nan)),
                "negative_noisy_flux_fraction": float(
                    row.get("negative_noisy_flux_fraction", np.nan)
                ),
            }
        )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_dir": str(dataset_dir),
        "diagnostics_dir": str(out),
        "smoke": bool(smoke),
        "n_rows": int(len(all_frame)),
        "split_rows": {split: int(len(frame)) for split, frame in frames.items()},
        "configured_redshift_range": {
            "z_min": float((config.get("synthetic_diffsky", {}) or {}).get("z_min", np.nan)),
            "z_max": float((config.get("synthetic_diffsky", {}) or {}).get("z_max", np.nan)),
        },
        "realized_redshift_range": {
            "min": float(np.min(z)) if z.size else np.nan,
            "median": float(np.median(z)) if z.size else np.nan,
            "max": float(np.max(z)) if z.size else np.nan,
        },
        "mass_fractions": mass_fractions,
        "bands": list(bands),
        "parameter_stats": str(out / "parameter_stats.csv"),
        "photometry_stats": str(out / "photometry_stats.csv"),
        "error_model_stats": str(out / "error_model_stats.csv"),
        "color_stats": str(out / "color_stats.csv"),
        "proposal_vs_final_metrics": str(out / "proposal_vs_final_metrics.csv"),
        "correlation_matrices": str(out / "correlation_matrices.json"),
        "n_parameter_rows": int(len(parameter_stats)),
        "n_photometry_rows": int(len(photometry_stats)),
        "n_error_model_rows": int(len(error_model_stats)),
        "n_color_rows": int(len(color_stats)),
        "n_proposal_metric_rows": int(len(proposal_metrics)),
        "error_model_residual_summary": residual_summary,
        "plots": plots,
        "reference_comparison": reference_outputs,
        "interpretation": (
            "Diagnostics summarize the generated DSPS-closure population. "
            "Reference comparisons are sanity checks of population/photometry distributions, "
            "not exact closure comparisons, because reference photometry can come from a "
            "different Diffsky/OpenUniverse forward model."
        ),
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    zconf = summary["configured_redshift_range"]
    zreal = summary["realized_redshift_range"]
    lines = [
        "# Synthetic Diffsky/FENIKS Population Diagnostics",
        "",
        f"- Dataset: `{summary['dataset_dir']}`",
        f"- Rows: {summary['n_rows']}",
        f"- Configured redshift range: {zconf['z_min']} to {zconf['z_max']}",
        f"- Realized redshift range: {zreal['min']:.6g} to {zreal['max']:.6g}",
        "",
        "## Mass Fractions",
        "",
    ]
    for key, value in summary["mass_fractions"].items():
        lines.append(f"- `{key}`: {_format_value(value)}")
    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            f"- `{summary['parameter_stats']}`",
            f"- `{summary['photometry_stats']}`",
            f"- `{summary['error_model_stats']}`",
            f"- `{summary['color_stats']}`",
            f"- `{summary['proposal_vs_final_metrics']}`",
            f"- `{summary['correlation_matrices']}`",
            "",
            "## Error Model",
            "",
        ]
    )
    if summary.get("error_model_residual_summary"):
        for row in summary["error_model_residual_summary"]:
            lines.append(
                "- "
                f"`{row['band']}`: residual mean={_format_value(row['residual_mean'])}, "
                f"std={_format_value(row['residual_std'])}, "
                f"negative noisy flux fraction="
                f"{_format_value(row['negative_noisy_flux_fraction'])}"
            )
    else:
        lines.append("- No error-model residual rows were available.")
    lines.extend(
        [
            "",
            "## Plots",
            "",
        ]
    )
    if summary["plots"]:
        for key, path in summary["plots"].items():
            lines.append(f"- `{key}`: `{path}`")
    else:
        lines.append("- No plots written; matplotlib/corner may be unavailable or plotting disabled.")
    if summary["reference_comparison"]:
        lines.extend(["", "## Reference Comparison", ""])
        for key, path in summary["reference_comparison"].items():
            lines.append(f"- `{key}`: `{path}`")
    lines.extend(["", summary["interpretation"], ""])
    return "\n".join(lines)


def _stats_row(
    split: str,
    group: str,
    quantity: str,
    column: str,
    values: pd.Series,
) -> dict[str, Any]:
    return _array_stats_row(split, group, quantity, column, _finite_numeric(values))


def _array_stats_row(
    split: str,
    group: str,
    quantity: str,
    column: str,
    values: Iterable[float] | np.ndarray,
) -> dict[str, Any]:
    arr = _finite_array(values)
    row: dict[str, Any] = {
        "split": split,
        "group": group,
        "quantity": quantity,
        "column": column,
        "n": int(arr.size),
    }
    row.update(_stats_payload(arr))
    return row


def _stats_payload(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q01": np.nan,
            "q05": np.nan,
            "q16": np.nan,
            "median": np.nan,
            "q84": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "max": np.nan,
        }
    qs = np.quantile(values, [0.01, 0.05, 0.16, 0.5, 0.84, 0.95, 0.99])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q01": float(qs[0]),
        "q05": float(qs[1]),
        "q16": float(qs[2]),
        "median": float(qs[3]),
        "q84": float(qs[4]),
        "q95": float(qs[5]),
        "q99": float(qs[6]),
        "max": float(np.max(values)),
    }


def _read_proposals(proposal_dir: Path) -> pd.DataFrame | None:
    paths = sorted(proposal_dir.glob("*.parquet"))
    if not paths:
        return None
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _sample_frame(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    return frame.sample(n=int(max_rows), random_state=int(seed)).reset_index(drop=True)


def _color_frame(frame: pd.DataFrame, bands: Sequence[str]) -> pd.DataFrame:
    data = {}
    for left, right in COLOR_PAIRS:
        if left in bands and right in bands:
            values = _difference_from_columns(frame, f"mag_true_{left}", f"mag_true_{right}")
            if values.size:
                data[f"{left}-{right}"] = pd.Series(values)
    return pd.DataFrame(data)


def _finite_numeric(values: pd.Series | Iterable[float] | np.ndarray) -> np.ndarray:
    if isinstance(values, pd.Series):
        if pd.api.types.is_bool_dtype(values):
            arr = values.astype(float).to_numpy()
        else:
            arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
    return _finite_array(arr)


def _finite_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _ratio_from_columns(frame: pd.DataFrame, numerator: str, denominator: str) -> np.ndarray:
    if numerator not in frame.columns or denominator not in frame.columns:
        return np.asarray([], dtype=float)
    num = pd.to_numeric(frame[numerator], errors="coerce").to_numpy(dtype=float)
    den = pd.to_numeric(frame[denominator], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(num) & np.isfinite(den)
    if not np.any(finite):
        return np.asarray([], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _finite_array(num[finite] / den[finite])


def _difference_from_columns(frame: pd.DataFrame, left: str, right: str) -> np.ndarray:
    if left not in frame.columns or right not in frame.columns:
        return np.asarray([], dtype=float)
    left_values = pd.to_numeric(frame[left], errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(frame[right], errors="coerce").to_numpy(dtype=float)
    return _finite_array(left_values - right_values)


def _has_dynamic_range(values: pd.Series) -> bool:
    arr = _finite_numeric(values)
    return bool(arr.size >= 3 and np.nanmax(arr) > np.nanmin(arr))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or weights.size == 0 or np.sum(weights) <= 0.0:
        return np.nan
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.size == 0 or weights.size == 0 or np.sum(weights) <= 0.0:
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(values[np.searchsorted(cumulative, float(quantile), side="left")])


def _ks_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(_finite_array(left))
    right = np.sort(_finite_array(right))
    if left.size == 0 or right.size == 0:
        return np.nan
    values = np.sort(np.concatenate([left, right]))
    left_cdf = np.searchsorted(left, values, side="right") / left.size
    right_cdf = np.searchsorted(right, values, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _wasserstein_quantile(left: np.ndarray, right: np.ndarray, n_grid: int = 1001) -> float:
    left = _finite_array(left)
    right = _finite_array(right)
    if left.size == 0 or right.size == 0:
        return np.nan
    q = np.linspace(0.0, 1.0, int(n_grid))
    return float(np.mean(np.abs(np.quantile(left, q) - np.quantile(right, q))))


def _runtime_value(
    cfg: dict[str, Any],
    name: str,
    default: Any,
    *,
    smoke: bool,
) -> Any:
    if smoke and f"smoke_{name}" in cfg:
        return cfg[f"smoke_{name}"]
    return cfg.get(name, default)


def _format_value(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.6g}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=False, allow_nan=False),
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value
