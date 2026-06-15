"""Diagnostics for FS2 amortized inference outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.io import (
    iter_catalog_batches,
    truth_column_from_spec,
    truth_value_from_spec,
    write_json,
)

_CORNER_PARAMETER_ORDER = [
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "tau2",
    "dust_index_n",
    "tau1_over_tau2",
    "ln_fagn",
    "ln_tauagn",
    "log10_gas_metallicity",
    "log10_gas_ionization",
    "z_obs",
    "dlog10_sfr_6",
    "dlog10_sfr_5",
    "dlog10_sfr_4",
    "dlog10_sfr_3",
    "dlog10_sfr_2",
    "dlog10_sfr_1",
]

_PARAMETER_LABELS = {
    "z_obs": "z",
    "log10_stellar_mass": "log M",
    "dlog10_sfr_1": "dSFR1",
    "dlog10_sfr_2": "dSFR2",
    "dlog10_sfr_3": "dSFR3",
    "dlog10_sfr_4": "dSFR4",
    "dlog10_sfr_5": "dSFR5",
    "dlog10_sfr_6": "dSFR6",
    "log10_stellar_metallicity": "log Z*",
    "tau2": "tau2",
    "dust_index_n": "n",
    "tau1_over_tau2": "tau1/tau2",
    "log10_gas_metallicity": "log Zgas",
    "log10_gas_ionization": "log Ugas",
    "ln_fagn": "ln fAGN",
    "ln_tauagn": "ln tauAGN",
}

_CATALOG_PROXY_SPECS = {
    "catalog_log10_stellar_mass_proxy": (
        "log10_formed_mass_msun",
        "log10 stellar mass catalog proxy",
    ),
    "catalog_log10_sfr_at_obs_proxy": (
        "log10_sfr_at_obs",
        "log10 SFR catalog proxy",
    ),
    "catalog_log10_metallicity_proxy": (
        "log10_metallicity",
        "log10 metallicity catalog proxy",
    ),
    "catalog_dust_av_proxy": (
        "dust_av",
        "dust Av catalog proxy",
    ),
}


def write_training_diagnostics(log_path: str | Path, out_dir: str | Path) -> list[str]:
    """Write epoch-level training diagnostic plots from ``training_log.csv``."""
    log_path = Path(log_path)
    out = Path(out_dir)
    if not log_path.exists():
        return []
    frame = pd.read_csv(log_path)
    if frame.empty:
        return []
    written = []
    _prepare_matplotlib_cache(out)
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting fallback
        write_json(
            out / "training_diagnostics_summary.json",
            {"plots": [], "reason": "matplotlib_unavailable"},
        )
        return []

    epoch_summary = _write_training_epoch_summary(frame, out)
    for column in [
        "loss",
        "negative_loglike",
        "kl_mc_mean",
        "logprior_mean",
        "logq_mean",
        "residual_rms",
        "finite_fraction",
        "encoder_grad_norm",
        "prior_grad_norm",
        "joint_grad_norm",
    ]:
        if column not in epoch_summary:
            continue
        path = _write_epoch_metric_plot(epoch_summary, column, out, plt)
        if path is None:
            continue
        written.append(path.name)
    path = _write_training_overview_plot(epoch_summary, out, plt)
    if path is not None:
        written.append(path.name)
    bin_path = out / "validation_redshift_bin_metrics.csv"
    if bin_path.exists():
        bins = pd.read_csv(bin_path)
        written.extend(path.name for path in _write_redshift_bin_plots(bins, out, plt))
    write_json(
        out / "training_diagnostics_summary.json",
        {
            "plots": written,
            "n_rows": int(len(frame)),
            "epoch_summary_rows": int(len(epoch_summary)),
            "x_axis": "epoch",
        },
    )
    return written


def _write_training_epoch_summary(frame: pd.DataFrame, out: Path) -> pd.DataFrame:
    metrics = [
        "loss",
        "negative_loglike",
        "kl_mc_mean",
        "logprior_mean",
        "logq_mean",
        "residual_rms",
        "finite_fraction",
        "encoder_grad_norm",
        "prior_grad_norm",
        "joint_grad_norm",
    ]
    metrics = [metric for metric in metrics if metric in frame]
    if "split" not in frame:
        frame = frame.copy()
        frame["split"] = "train"
    rows = []
    for (split, epoch), group in frame.groupby(["split", "epoch"], sort=True):
        row: dict[str, float | int | str] = {
            "split": str(split),
            "epoch": int(epoch),
            "n_rows": int(len(group)),
            "n_objects": int(group["n_objects"].sum()) if "n_objects" in group else 0,
            "kl_weight": (
                float(np.nanmean(group["kl_weight"])) if "kl_weight" in group else np.nan
            ),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                row[metric] = np.nan
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_q16"] = np.nan
                row[f"{metric}_q84"] = np.nan
                continue
            row[metric] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.nanmedian(values))
            row[f"{metric}_q16"] = float(np.nanquantile(values, 0.16))
            row[f"{metric}_q84"] = float(np.nanquantile(values, 0.84))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "training_epoch_summary.csv", index=False)
    return summary


def _write_epoch_metric_plot(
    summary: pd.DataFrame,
    metric: str,
    out: Path,
    plt,
) -> Path | None:
    if summary.empty or metric not in summary:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    colors = {"train": "#2a9fd6", "validation": "#ef476f"}
    for split, group in summary.groupby("split", sort=False):
        if _is_gradient_metric(metric) and str(split) != "train":
            continue
        group = group.sort_values("epoch")
        x = group["epoch"].to_numpy(dtype=float)
        y = group[metric].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            continue
        color = colors.get(str(split), None)
        label = str(split)
        ax.plot(
            x[finite],
            y[finite],
            marker="o",
            ms=3.2,
            lw=1.8,
            color=color,
            label=label,
        )
        q16_col = f"{metric}_q16"
        q84_col = f"{metric}_q84"
        if q16_col in group and q84_col in group and len(group) > 1:
            q16 = group[q16_col].to_numpy(dtype=float)
            q84 = group[q84_col].to_numpy(dtype=float)
            band_finite = finite & np.isfinite(q16) & np.isfinite(q84)
            if band_finite.any():
                ax.fill_between(
                    x[band_finite],
                    q16[band_finite],
                    q84[band_finite],
                    color=color,
                    alpha=0.12,
                    linewidth=0.0,
                )
    if not ax.has_data():
        plt.close(fig)
        return None
    if _use_log_y(metric, summary):
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(f"{_metric_label(metric)} by epoch")
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out / f"{metric}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_training_overview_plot(
    summary: pd.DataFrame,
    out: Path,
    plt,
) -> Path | None:
    metrics = ["loss", "negative_loglike", "kl_mc_mean", "logprior_mean"]
    metrics = [metric for metric in metrics if metric in summary]
    if not metrics:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics, strict=False):
        for split, group in summary.groupby("split", sort=False):
            group = group.sort_values("epoch")
            color = "#2a9fd6" if str(split) == "train" else "#ef476f"
            ax.plot(
                group["epoch"].to_numpy(dtype=float),
                group[metric].to_numpy(dtype=float),
                marker="o",
                ms=3,
                lw=1.6,
                color=color,
                label=str(split),
            )
        ax.set_title(_metric_label(metric))
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.22, linewidth=0.7)
    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Training history by epoch", y=0.995)
    fig.tight_layout()
    path = out / "training_history_overview.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_redshift_bin_plots(
    bins: pd.DataFrame,
    out: Path,
    plt,
) -> list[Path]:
    if bins.empty or not {"epoch", "z_bin"} <= set(bins):
        return []
    written = []
    for metric, filename, label, log10_values in [
        ("loss", "validation_loss_by_redshift_bin.png", "validation loss", False),
        (
            "negative_loglike",
            "validation_negative_loglike_by_redshift_bin.png",
            "validation negative loglike",
            False,
        ),
        (
            "posterior_predictive_chi2",
            "validation_chi2_by_redshift_bin.png",
            "log10 median posterior predictive chi2",
            True,
        ),
        ("kl_mc_mean", "validation_kl_by_redshift_bin.png", "validation KL MC", False),
    ]:
        if metric not in bins:
            continue
        path = _write_redshift_bin_heatmap(
            bins,
            metric,
            out / filename,
            plt,
            label=label,
            log10_values=log10_values,
        )
        if path is not None:
            written.append(path)
    return written


def _write_redshift_bin_heatmap(
    bins: pd.DataFrame,
    metric: str,
    path: Path,
    plt,
    *,
    label: str,
    log10_values: bool,
) -> Path | None:
    pivot = bins.pivot(index="z_bin", columns="epoch", values=metric)
    if pivot.empty:
        return None
    values = pivot.to_numpy(dtype=float)
    if log10_values:
        values = np.log10(np.maximum(values, 1.0e-12))
    if not np.isfinite(values).any():
        return None
    fig, ax = plt.subplots(figsize=(9, 4.8))
    image = ax.imshow(values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index], fontsize=8)
    epochs = pivot.columns.to_numpy(dtype=int)
    tick_count = min(8, len(epochs))
    tick_positions = np.linspace(0, len(epochs) - 1, tick_count, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(epochs[index]) for index in tick_positions])
    ax.set_xlabel("epoch")
    ax.set_ylabel("z_true_gal bin")
    ax.set_title(label)
    fig.colorbar(image, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _metric_label(metric: str) -> str:
    return {
        "loss": "negative ELBO",
        "negative_loglike": "negative log likelihood",
        "kl_mc_mean": "KL MC mean",
        "logprior_mean": "mean log p_beta(x)",
        "logq_mean": "mean log q_psi(x|f,err)",
        "residual_rms": "RMS residual",
        "finite_fraction": "finite fraction",
        "encoder_grad_norm": "encoder grad norm",
        "prior_grad_norm": "RealNVP prior grad norm",
        "joint_grad_norm": "joint grad norm",
    }.get(metric, metric)


def _is_gradient_metric(metric: str) -> bool:
    return metric in {"encoder_grad_norm", "prior_grad_norm", "joint_grad_norm"}


def _use_log_y(metric: str, summary: pd.DataFrame) -> bool:
    if metric not in {"encoder_grad_norm", "prior_grad_norm", "joint_grad_norm"}:
        return False
    values = summary[metric].to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    return values.size > 0


def posterior_predictive_residual_frame(
    object_id,
    obs_flux,
    obs_err,
    mask,
    model_flux,
    band_names: tuple[str, ...],
) -> pd.DataFrame:
    """Return long-form normalized posterior predictive residual rows."""
    object_id = np.asarray(object_id)
    obs_flux = np.asarray(obs_flux, dtype=float)
    obs_err = np.asarray(obs_err, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    model_flux = np.asarray(model_flux, dtype=float)
    if model_flux.ndim != 3:
        raise ValueError(f"model_flux must be [K,N,B], got {model_flux.shape}")
    rows = []
    n_samples, n_objects, n_bands = model_flux.shape
    for sample_id in range(n_samples):
        for object_index in range(n_objects):
            for band_index in range(n_bands):
                err = obs_err[object_index, band_index]
                residual = np.nan
                if mask[object_index, band_index] and np.isfinite(err) and err > 0.0:
                    residual = (
                        model_flux[sample_id, object_index, band_index]
                        - obs_flux[object_index, band_index]
                    ) / err
                rows.append(
                    {
                        "object_id": object_id[object_index],
                        "sample_id": int(sample_id),
                        "band": band_names[band_index],
                        "obs_flux_fnu_cgs": float(obs_flux[object_index, band_index]),
                        "obs_err_fnu_cgs": float(obs_err[object_index, band_index]),
                        "model_flux_fnu_cgs": float(
                            model_flux[sample_id, object_index, band_index]
                        ),
                        "residual_sigma": float(residual),
                        "valid": bool(mask[object_index, band_index]),
                    }
                )
    return pd.DataFrame(rows)


def posterior_predictive_residual_summary_frame(
    object_id,
    obs_flux,
    obs_err,
    mask,
    model_flux,
    band_names: tuple[str, ...],
) -> pd.DataFrame:
    """Return one posterior predictive residual summary row per object-band."""
    object_id = np.asarray(object_id)
    obs_flux = np.asarray(obs_flux, dtype=float)
    obs_err = np.asarray(obs_err, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    model_flux = np.asarray(model_flux, dtype=float)
    if model_flux.ndim != 3:
        raise ValueError(f"model_flux must be [K,N,B], got {model_flux.shape}")
    rows = []
    _n_samples, n_objects, n_bands = model_flux.shape
    residual = np.full_like(model_flux, np.nan, dtype=float)
    valid_err = np.isfinite(obs_err) & (obs_err > 0.0) & mask
    residual[:, valid_err] = (
        model_flux[:, valid_err] - obs_flux[None, :, :][:, valid_err]
    ) / obs_err[None, :, :][:, valid_err]
    for object_index in range(n_objects):
        for band_index in range(n_bands):
            rows.append(
                {
                    "object_id": object_id[object_index],
                    "band": band_names[band_index],
                    "obs_flux_fnu_cgs": float(obs_flux[object_index, band_index]),
                    "obs_err_fnu_cgs": float(obs_err[object_index, band_index]),
                    "model_flux_q16": float(
                        _nanquantile_or_nan(model_flux[:, object_index, band_index], 0.16)
                    ),
                    "model_flux_median": float(
                        np.nanmedian(model_flux[:, object_index, band_index])
                    ),
                    "model_flux_q84": float(
                        _nanquantile_or_nan(model_flux[:, object_index, band_index], 0.84)
                    ),
                    "residual_sigma_q16": float(
                        _nanquantile_or_nan(residual[:, object_index, band_index], 0.16)
                    ),
                    "residual_sigma_median": float(
                        np.nanmedian(residual[:, object_index, band_index])
                    ),
                    "residual_sigma_q84": float(
                        _nanquantile_or_nan(residual[:, object_index, band_index], 0.84)
                    ),
                    "valid": bool(mask[object_index, band_index]),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["abs_residual_sigma_median"] = frame["residual_sigma_median"].abs()
    return frame


def feature_diagnostics_frame(
    object_id,
    features,
    *,
    n_flux_bands: int = 10,
) -> pd.DataFrame:
    """Return per-object feature magnitude diagnostics."""
    object_id = np.asarray(object_id)
    features = np.asarray(features, dtype=float)
    flux_features = features[:, :n_flux_bands]
    err_features = features[:, n_flux_bands:]
    return pd.DataFrame(
        {
            "object_id": object_id,
            "feature_max_abs": np.nanmax(np.abs(features), axis=1),
            "flux_feature_max_abs": np.nanmax(np.abs(flux_features), axis=1),
            "err_feature_max_abs": np.nanmax(np.abs(err_features), axis=1),
        }
    )


def summarize_inference_outputs(
    summary_path: str | Path,
    out_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
    limit: int | None = None,
) -> None:
    """Write posterior predictive diagnostics from inference parquet outputs."""
    summary_path = Path(summary_path)
    out = Path(out_dir)
    if not summary_path.exists():
        return
    frame = pd.read_parquet(summary_path)
    residual_summary = _write_residual_summary(out)
    top_chi2 = _write_top_chi2(frame, residual_summary, out)
    redshift = _write_redshift_comparison(frame, out, config=config, limit=limit)
    catalog_proxy = _write_catalog_proxy_comparison(
        frame,
        out,
        config=config,
        limit=limit,
    )
    prior_summary = _write_learned_prior_summary(out)
    pit_summary = _write_redshift_pit(redshift, out)
    plots = _write_inference_plots(
        frame,
        residual_summary,
        top_chi2,
        redshift,
        catalog_proxy,
        out,
    )
    payload = {
        "n_objects": int(len(frame)),
        "median_valid_bands": (
            float(frame["n_valid_bands"].median())
            if "n_valid_bands" in frame and not frame.empty
            else None
        ),
        "median_posterior_predictive_chi2": (
            float(frame["posterior_predictive_chi2_median"].median())
            if "posterior_predictive_chi2_median" in frame and not frame.empty
            else None
        ),
        "residual_summary_rows": int(len(residual_summary)),
        "top_chi2_rows": int(len(top_chi2)),
        "redshift_comparison_rows": int(len(redshift)),
        "catalog_proxy_comparison_rows": int(len(catalog_proxy)),
        "learned_prior_rows": int(prior_summary.get("n_samples", 0)),
        "redshift_pit": pit_summary,
        "plots": plots,
    }
    if not residual_summary.empty:
        band_stats = (
            residual_summary.assign(
                abs_residual_median=lambda df: df["residual_sigma_median"].abs()
            )
            .groupby("band")["abs_residual_median"]
            .median()
            .sort_index()
        )
        payload["median_abs_residual_sigma_by_band"] = {
            str(band): float(value) for band, value in band_stats.items()
        }
    write_json(Path(out_dir) / "posterior_diagnostics_summary.json", payload)


def _write_residual_summary(out: Path) -> pd.DataFrame:
    summary_path = out / "posterior_predictive_residual_summary.parquet"
    if summary_path.exists():
        return pd.read_parquet(summary_path)
    residual_path = out / "posterior_predictive_residuals.parquet"
    if not residual_path.exists():
        return pd.DataFrame()
    residuals = pd.read_parquet(residual_path)
    if residuals.empty:
        return pd.DataFrame()
    grouped = residuals.groupby(["object_id", "band"], sort=False)
    summary = grouped.agg(
        obs_flux_fnu_cgs=("obs_flux_fnu_cgs", "first"),
        obs_err_fnu_cgs=("obs_err_fnu_cgs", "first"),
        model_flux_q16=("model_flux_fnu_cgs", lambda x: _nanquantile_or_nan(x, 0.16)),
        model_flux_median=("model_flux_fnu_cgs", "median"),
        model_flux_q84=("model_flux_fnu_cgs", lambda x: _nanquantile_or_nan(x, 0.84)),
        residual_sigma_q16=(
            "residual_sigma",
            lambda x: _nanquantile_or_nan(x, 0.16),
        ),
        residual_sigma_median=("residual_sigma", "median"),
        residual_sigma_q84=(
            "residual_sigma",
            lambda x: _nanquantile_or_nan(x, 0.84),
        ),
        valid=("valid", "first"),
    ).reset_index()
    summary["abs_residual_sigma_median"] = summary["residual_sigma_median"].abs()
    summary.to_parquet(out / "posterior_predictive_residual_summary.parquet", index=False)
    return summary


def _write_top_chi2(
    summary: pd.DataFrame,
    residual_summary: pd.DataFrame,
    out: Path,
    *,
    top_n: int = 50,
) -> pd.DataFrame:
    if summary.empty or "posterior_predictive_chi2_median" not in summary:
        return pd.DataFrame()
    top = summary.sort_values("posterior_predictive_chi2_median", ascending=False).head(
        top_n
    )
    top = top.copy()
    if not residual_summary.empty:
        idx = residual_summary.groupby("object_id")[
            "abs_residual_sigma_median"
        ].idxmax()
        worst = residual_summary.loc[
            idx,
            [
                "object_id",
                "band",
                "residual_sigma_median",
                "obs_flux_fnu_cgs",
                "obs_err_fnu_cgs",
                "model_flux_median",
            ],
        ].rename(
            columns={
                "band": "worst_band",
                "residual_sigma_median": "worst_band_residual_sigma_median",
            }
        )
        top = top.merge(worst, on="object_id", how="left")
    columns = [
        column
        for column in [
            "object_id",
            "photometric_loglike_mean",
            "posterior_predictive_chi2_median",
            "worst_band",
            "worst_band_residual_sigma_median",
            "obs_flux_fnu_cgs",
            "obs_err_fnu_cgs",
            "model_flux_median",
            "z_obs_median",
            "z_obs_q16",
            "z_obs_q84",
            "log10_stellar_mass_median",
        ]
        if column in top
    ]
    top = top[columns]
    top.to_parquet(out / "top_posterior_predictive_chi2.parquet", index=False)
    top.to_csv(out / "top_posterior_predictive_chi2.csv", index=False)
    return top


def _write_redshift_comparison(
    summary: pd.DataFrame,
    out: Path,
    *,
    config: dict[str, Any] | None,
    limit: int | None,
) -> pd.DataFrame:
    if config is None or summary.empty or "z_obs_median" not in summary:
        return pd.DataFrame()
    candidates = ["z_true_gal", "z_obs_gal", "z_true", "z_phz", "phz_median"]
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(config["catalog_path"]).schema.names)
    except Exception:
        available = set()
    columns = [column for column in candidates if column in available]
    if not columns:
        return pd.DataFrame()
    frames = []
    for batch in iter_catalog_batches(
        config["catalog_path"],
        columns=columns,
        batch_size=10_000,
        limit=limit,
    ):
        work = batch.copy()
        work["object_id"] = work.index.to_numpy()
        frames.append(work)
    if not frames:
        return pd.DataFrame()
    proxies = pd.concat(frames, axis=0, ignore_index=True)
    comparison = summary.merge(proxies, on="object_id", how="left")
    for column in columns:
        comparison[f"delta_z_obs_median_minus_{column}"] = (
            comparison["z_obs_median"] - comparison[column]
        )
    comparison.to_parquet(out / "redshift_comparison.parquet", index=False)
    return comparison


def _write_catalog_proxy_comparison(
    summary: pd.DataFrame,
    out: Path,
    *,
    config: dict[str, Any] | None,
    limit: int | None,
) -> pd.DataFrame:
    if config is None or summary.empty:
        return pd.DataFrame()
    truth_specs = dict((config.get("truth", {}) or {}).get("parameter_columns") or {})
    specs = {
        output_name: truth_specs[truth_key]
        for output_name, (truth_key, _) in _CATALOG_PROXY_SPECS.items()
        if truth_key in truth_specs
    }
    if not specs:
        return pd.DataFrame()
    columns = []
    for spec in specs.values():
        column = truth_column_from_spec(spec)
        if column is not None:
            columns.append(column)
    if not columns:
        return pd.DataFrame()
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(config["catalog_path"]).schema.names)
    except Exception:
        available = set()
    columns = sorted({column for column in columns if column in available})
    if not columns:
        return pd.DataFrame()
    frames = []
    for batch in iter_catalog_batches(
        config["catalog_path"],
        columns=columns,
        batch_size=10_000,
        limit=limit,
    ):
        rows = []
        for row_index, row in batch.iterrows():
            row_dict = row.to_dict()
            out_row = {"object_id": row_index}
            for output_name, spec in specs.items():
                out_row[output_name] = truth_value_from_spec(row_dict, spec)
            rows.append(out_row)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    proxies = pd.concat(frames, axis=0, ignore_index=True)
    comparison = summary.merge(proxies, on="object_id", how="left")
    if (
        "log10_stellar_mass_median" in comparison
        and "catalog_log10_stellar_mass_proxy" in comparison
    ):
        comparison["delta_log10_stellar_mass_median_minus_catalog_proxy"] = (
            comparison["log10_stellar_mass_median"]
            - comparison["catalog_log10_stellar_mass_proxy"]
        )
    comparison.to_parquet(out / "catalog_proxy_comparison.parquet", index=False)
    comparison.to_csv(out / "catalog_proxy_comparison.csv", index=False)
    return comparison


def _write_learned_prior_summary(out: Path) -> dict[str, Any]:
    prior_path = out / "learned_prior_samples.parquet"
    if not prior_path.exists():
        return {}
    prior = pd.read_parquet(prior_path)
    if prior.empty:
        return {"n_samples": 0}
    payload: dict[str, Any] = {"n_samples": int(len(prior))}
    for column in _corner_columns(prior):
        values = prior[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        payload[column] = {
            "q01": float(np.nanquantile(values, 0.01)),
            "q16": float(np.nanquantile(values, 0.16)),
            "median": float(np.nanquantile(values, 0.50)),
            "q84": float(np.nanquantile(values, 0.84)),
            "q99": float(np.nanquantile(values, 0.99)),
        }
    write_json(out / "learned_prior_summary.json", payload)
    return payload


def _write_redshift_pit(redshift: pd.DataFrame, out: Path) -> dict[str, Any]:
    samples_path = out / "posterior_samples.parquet"
    if redshift.empty or not samples_path.exists():
        return {}
    truth_col = next(
        (
            column
            for column in ["z_true_gal", "z_obs_gal", "z_true", "z_phz", "phz_median"]
            if column in redshift
        ),
        None,
    )
    if truth_col is None:
        return {}
    samples = pd.read_parquet(samples_path, columns=["object_id", "z_obs"])
    merged = samples.merge(
        redshift[["object_id", truth_col, "posterior_predictive_chi2_median"]],
        on="object_id",
        how="inner",
    )
    if merged.empty:
        return {}
    rows = []
    for object_id, group in merged.groupby("object_id", sort=False):
        truth = float(group[truth_col].iloc[0])
        z_values = group["z_obs"].to_numpy(dtype=float)
        z_values = z_values[np.isfinite(z_values)]
        if not np.isfinite(truth) or z_values.size == 0:
            continue
        rows.append(
            {
                "object_id": object_id,
                "truth_column": truth_col,
                "z_reference": truth,
                "pit": float(np.mean(z_values < truth)),
                "posterior_predictive_chi2_median": float(
                    group["posterior_predictive_chi2_median"].iloc[0]
                ),
            }
        )
    pit = pd.DataFrame(rows)
    if pit.empty:
        return {}
    pit.to_parquet(out / "redshift_pit.parquet", index=False)
    values = np.sort(pit["pit"].to_numpy(dtype=float))
    n_values = len(values)
    empirical = np.arange(1, n_values + 1, dtype=float) / float(n_values)
    ks = float(np.max(np.abs(empirical - values))) if n_values else float("nan")
    return {
        "truth_column": truth_col,
        "n_objects": int(n_values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "ks_uniform": ks,
        "frac_lt_0p05": float(np.mean(values < 0.05)),
        "frac_gt_0p95": float(np.mean(values > 0.95)),
    }


def _write_inference_plots(
    summary: pd.DataFrame,
    residual_summary: pd.DataFrame,
    top_chi2: pd.DataFrame,
    redshift: pd.DataFrame,
    catalog_proxy: pd.DataFrame,
    out: Path,
) -> list[str]:
    try:
        _prepare_matplotlib_cache(out)
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting fallback
        return []

    written: list[str] = []
    if not residual_summary.empty:
        bands = list(dict.fromkeys(residual_summary["band"].astype(str)))
        data = [
            residual_summary.loc[
                residual_summary["band"].astype(str) == band,
                "residual_sigma_median",
            ].to_numpy()
            for band in bands
        ]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.axhline(0.0, color="black", lw=1.0, alpha=0.5)
        ax.boxplot(data, tick_labels=bands, showfliers=False)
        ax.set_ylabel("median posterior predictive residual (sigma)")
        ax.set_title("Posterior predictive residuals by band")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = out / "posterior_predictive_residuals_by_band.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)

        fig, ax = plt.subplots(figsize=(7, 4))
        values = residual_summary["abs_residual_sigma_median"].to_numpy()
        ax.hist(values[np.isfinite(values)], bins=40)
        ax.set_xlabel("|median residual| (sigma)")
        ax.set_ylabel("object-band count")
        ax.set_title("Normalized posterior predictive residuals")
        fig.tight_layout()
        path = out / "posterior_predictive_abs_residual_hist.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)

    if not top_chi2.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = top_chi2["object_id"].astype(str).head(20)
        values = top_chi2["posterior_predictive_chi2_median"].head(20)
        ax.bar(labels, values)
        ax.set_xlabel("object_id")
        ax.set_ylabel("median posterior predictive chi2")
        ax.set_title("Top posterior predictive chi2 objects")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = out / "top_posterior_predictive_chi2.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)

    feature_path = out / "feature_diagnostics.parquet"
    if feature_path.exists():
        features = pd.read_parquet(feature_path)
        if not features.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(features["feature_max_abs"].to_numpy(), bins=40)
            ax.set_xlabel("max |encoder feature| per object")
            ax.set_ylabel("object count")
            ax.set_title("Encoder feature scale diagnostics")
            fig.tight_layout()
            path = out / "feature_max_abs_hist.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path.name)

    if not redshift.empty:
        truth_col = next(
            (
                column
                for column in ["z_true_gal", "z_obs_gal", "z_true", "z_phz", "phz_median"]
                if column in redshift
            ),
            None,
        )
        if truth_col is not None:
            fig, ax = plt.subplots(figsize=(5, 5))
            x = redshift[truth_col].to_numpy(dtype=float)
            y = redshift["z_obs_median"].to_numpy(dtype=float)
            yerr = np.vstack(
                [
                    y - redshift["z_obs_q16"].to_numpy(dtype=float),
                    redshift["z_obs_q84"].to_numpy(dtype=float) - y,
                ]
            )
            ax.errorbar(x, y, yerr=yerr, fmt="o", ms=3, alpha=0.5, lw=0.7)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.any():
                lo = float(min(np.nanmin(x[finite]), np.nanmin(y[finite])))
                hi = float(max(np.nanmax(x[finite]), np.nanmax(y[finite])))
                ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, alpha=0.5)
            ax.set_xlabel(truth_col)
            ax.set_ylabel("posterior z_obs median")
            ax.set_title("Redshift posterior vs catalog proxy")
            fig.tight_layout()
            path = out / "z_obs_median_vs_catalog_proxy.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path.name)

    samples_path = out / "posterior_samples.parquet"
    if samples_path.exists():
        samples = pd.read_parquet(samples_path)
        prior = _read_learned_prior(out)
        path = _write_corner_plot(
            samples,
            out,
            plt,
            filename="posterior_corner.png",
            title="Aggregate amortized posterior",
            color="#2a9fd6",
            label="posterior q_psi",
        )
        if path is not None:
            written.append(path.name)
        if prior is not None and not prior.empty:
            path = _write_learned_prior_logprob_plot(prior, out, plt)
            if path is not None:
                written.append(path.name)
            path = _write_corner_plot(
                prior,
                out,
                plt,
                filename="learned_prior_corner.png",
                title="Learned RealNVP prior",
                color="#ef476f",
                label="learned prior p_beta",
            )
            if path is not None:
                written.append(path.name)
            path = _write_corner_plot(
                samples,
                out,
                plt,
                comparison=prior,
                filename="posterior_vs_learned_prior_corner.png",
                title="Aggregate posterior vs learned RealNVP prior",
                color="#2a9fd6",
                comparison_color="#ef476f",
                label="posterior q_psi",
                comparison_label="learned prior p_beta",
            )
            if path is not None:
                written.append(path.name)
        path = _write_redshift_distribution_plot(samples, redshift, prior, out, plt)
        if path is not None:
            written.append(path.name)
        path = _write_redshift_pit_plot(out, plt)
        if path is not None:
            written.append(path.name)

    if not catalog_proxy.empty:
        prior = _read_learned_prior(out)
        samples_path = out / "posterior_samples.parquet"
        posterior_samples = (
            pd.read_parquet(samples_path) if samples_path.exists() else pd.DataFrame()
        )
        for path in _write_catalog_proxy_plots(
            summary,
            posterior_samples,
            prior,
            catalog_proxy,
            out,
            plt,
        ):
            written.append(path.name)

    if not summary.empty and "posterior_predictive_chi2_median" in summary:
        fig, ax = plt.subplots(figsize=(7, 4))
        values = summary["posterior_predictive_chi2_median"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            ax.hist(np.log10(np.maximum(values, 1.0e-12)), bins=40)
        ax.set_xlabel("log10 median posterior predictive chi2")
        ax.set_ylabel("object count")
        ax.set_title("Posterior predictive chi2")
        fig.tight_layout()
        path = out / "posterior_predictive_chi2_hist.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)

    return written


def _read_learned_prior(out: Path) -> pd.DataFrame | None:
    prior_path = out / "learned_prior_samples.parquet"
    if not prior_path.exists():
        return None
    prior = pd.read_parquet(prior_path)
    return prior if not prior.empty else None


def _write_catalog_proxy_plots(
    summary: pd.DataFrame,
    posterior_samples: pd.DataFrame,
    prior: pd.DataFrame | None,
    catalog_proxy: pd.DataFrame,
    out: Path,
    plt,
) -> list[Path]:
    written = []
    if "catalog_log10_stellar_mass_proxy" in catalog_proxy:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        series = [
            (
                catalog_proxy["catalog_log10_stellar_mass_proxy"].to_numpy(dtype=float),
                "FS2 catalog proxy",
                "#222222",
                "step",
                1.0,
            )
        ]
        if "log10_stellar_mass_median" in summary:
            series.append(
                (
                    summary["log10_stellar_mass_median"].to_numpy(dtype=float),
                    "posterior median",
                    "#2a9fd6",
                    "stepfilled",
                    0.35,
                )
            )
        if prior is not None and "log10_stellar_mass" in prior:
            series.append(
                (
                    prior["log10_stellar_mass"].to_numpy(dtype=float),
                    "learned prior",
                    "#ef476f",
                    "step",
                    1.0,
                )
            )
        series = [
            (values[np.isfinite(values)], label, color, histtype, alpha)
            for values, label, color, histtype, alpha in series
            if np.isfinite(values).any()
        ]
        if series:
            merged = np.concatenate([values for values, *_ in series])
            lo, hi = np.nanquantile(merged, [0.01, 0.99])
            bins = np.linspace(float(lo), float(hi), 42)
            for values, label, color, histtype, alpha in series:
                ax.hist(
                    values,
                    bins=bins,
                    density=True,
                    histtype=histtype,
                    color=color,
                    alpha=alpha,
                    label=label,
                    lw=1.2,
                )
            ax.set_xlabel("log10 stellar mass")
            ax.set_ylabel("density")
            ax.set_title("Stellar mass distribution vs FS2 catalog proxy")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            path = out / "catalog_proxy_stellar_mass_comparison.png"
            fig.savefig(path, dpi=150)
            written.append(path)
        plt.close(fig)

    if "delta_log10_stellar_mass_median_minus_catalog_proxy" in catalog_proxy:
        values = catalog_proxy[
            "delta_log10_stellar_mass_median_minus_catalog_proxy"
        ].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.axvline(0.0, color="black", lw=1.0, alpha=0.6)
            ax.hist(values, bins=40, histtype="stepfilled", color="#2a9fd6", alpha=0.35)
            ax.set_xlabel("posterior median - FS2 catalog proxy")
            ax.set_ylabel("object count")
            ax.set_title("Stellar mass proxy residual")
            fig.tight_layout()
            path = out / "catalog_proxy_stellar_mass_residual.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)

    if "catalog_log10_sfr_at_obs_proxy" in catalog_proxy:
        values = catalog_proxy["catalog_log10_sfr_at_obs_proxy"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(values, bins=40, histtype="step", color="#222222", lw=1.2)
            ax.set_xlabel("log10 SFR catalog proxy")
            ax.set_ylabel("object count")
            ax.set_title("FS2 catalog SFR proxy distribution")
            fig.tight_layout()
            path = out / "catalog_proxy_sfr_distribution.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)

    if {
        "catalog_log10_stellar_mass_proxy",
        "catalog_log10_sfr_at_obs_proxy",
    }.issubset(catalog_proxy.columns):
        x = catalog_proxy["catalog_log10_stellar_mass_proxy"].to_numpy(dtype=float)
        y = catalog_proxy["catalog_log10_sfr_at_obs_proxy"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.any():
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            color_values = None
            if "posterior_predictive_chi2_median" in catalog_proxy:
                chi2 = catalog_proxy["posterior_predictive_chi2_median"].to_numpy(
                    dtype=float
                )
                if np.isfinite(chi2).any():
                    color_values = np.log10(np.maximum(chi2, 1.0e-12))
            if color_values is not None:
                scatter = ax.scatter(
                    x[finite],
                    y[finite],
                    c=color_values[finite],
                    s=12,
                    alpha=0.65,
                    cmap="viridis",
                )
                fig.colorbar(scatter, ax=ax, label="log10 posterior predictive chi2")
            else:
                ax.scatter(x[finite], y[finite], s=12, alpha=0.65, color="#222222")
            ax.set_xlabel("log10 stellar mass catalog proxy")
            ax.set_ylabel("log10 SFR catalog proxy")
            ax.set_title("FS2 catalog proxy mass-SFR plane")
            fig.tight_layout()
            path = out / "catalog_proxy_mass_sfr_plane.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)

    if not posterior_samples.empty and "log10_stellar_mass" in posterior_samples:
        # Aggregate posterior mass is useful to compare with posterior medians;
        # it should not be interpreted as an independent population prior.
        values = posterior_samples["log10_stellar_mass"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(values, bins=45, density=True, histtype="step", color="#2a9fd6")
            ax.set_xlabel("log10 stellar mass")
            ax.set_ylabel("density")
            ax.set_title("Aggregate posterior stellar mass samples")
            fig.tight_layout()
            path = out / "aggregate_posterior_stellar_mass_samples.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)

    return written


def _write_corner_plot(
    samples: pd.DataFrame,
    out: Path,
    plt,
    *,
    comparison: pd.DataFrame | None = None,
    filename: str,
    title: str,
    color: str,
    label: str,
    comparison_color: str = "#ef476f",
    comparison_label: str = "comparison",
) -> Path | None:
    columns = _corner_columns(samples)
    if comparison is not None:
        columns = [column for column in columns if column in comparison]
    if len(columns) < 2 or samples.empty:
        return None
    work = _finite_sample(samples, columns, max_rows=4_000)
    if work.empty:
        return None
    comparison_work = None
    if comparison is not None:
        comparison_work = _finite_sample(comparison, columns, max_rows=4_000)
        if comparison_work.empty:
            comparison_work = None
    ranges = _corner_ranges(work, comparison_work, columns)
    n_columns = len(columns)
    fig, axes = plt.subplots(
        n_columns,
        n_columns,
        figsize=(1.55 * n_columns, 1.55 * n_columns),
    )
    for row, y_col in enumerate(columns):
        for col, x_col in enumerate(columns):
            ax = axes[row, col]
            if row == col:
                _plot_1d_hist(
                    ax,
                    work[x_col].to_numpy(dtype=float),
                    ranges[x_col],
                    color=color,
                    label=label,
                )
                if comparison_work is not None:
                    _plot_1d_hist(
                        ax,
                        comparison_work[x_col].to_numpy(dtype=float),
                        ranges[x_col],
                        color=comparison_color,
                        label=comparison_label,
                    )
                q16, q50, q84 = np.nanquantile(
                    work[x_col].to_numpy(dtype=float), [0.16, 0.50, 0.84]
                )
                ax.axvline(q50, color=color, lw=0.9, alpha=0.8)
                ax.set_title(
                    f"{_label(x_col)}={q50:.2g}+{q84 - q50:.2g}/-{q50 - q16:.2g}",
                    fontsize=6,
                    color=color,
                )
            elif row > col:
                _plot_2d_contours(
                    ax,
                    work[x_col].to_numpy(dtype=float),
                    work[y_col].to_numpy(dtype=float),
                    ranges[x_col],
                    ranges[y_col],
                    color=color,
                    linewidth=0.85,
                )
                if comparison_work is not None:
                    _plot_2d_contours(
                        ax,
                        comparison_work[x_col].to_numpy(dtype=float),
                        comparison_work[y_col].to_numpy(dtype=float),
                        ranges[x_col],
                        ranges[y_col],
                        color=comparison_color,
                        linewidth=0.85,
                    )
            else:
                ax.axis("off")
                continue
            if row == n_columns - 1:
                ax.set_xlabel(_label(x_col), fontsize=7)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(_label(y_col), fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.set_xlim(*ranges[x_col])
            if row > col:
                ax.set_ylim(*ranges[y_col])
            ax.tick_params(labelsize=5)
    if comparison_work is not None:
        legend_ax = axes[0, min(1, n_columns - 1)]
        legend_ax.plot([], [], color=color, label=label)
        legend_ax.plot([], [], color=comparison_color, label=comparison_label)
        legend_ax.legend(frameon=False, fontsize=7, loc="center")
    fig.suptitle(title, y=1.002)
    fig.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_redshift_distribution_plot(
    posterior_samples: pd.DataFrame,
    redshift: pd.DataFrame,
    prior: pd.DataFrame | None,
    out: Path,
    plt,
) -> Path | None:
    if "z_obs" not in posterior_samples:
        return None
    posterior_summary_path = out / "posterior_summary.parquet"
    if posterior_summary_path.exists():
        summary = pd.read_parquet(posterior_summary_path, columns=["z_obs_median"])
        posterior_z = summary["z_obs_median"].to_numpy(dtype=float)
    else:
        posterior_z = posterior_samples.groupby("object_id")["z_obs"].median().to_numpy()
    series = [
        (posterior_z[np.isfinite(posterior_z)], "posterior median", "#2a9fd6")
    ]
    if prior is not None and "z_obs" in prior:
        prior_z = prior["z_obs"].to_numpy(dtype=float)
        series.append((prior_z[np.isfinite(prior_z)], "learned prior", "#ef476f"))
    for column, color in [
        ("z_true_gal", "#222222"),
        ("z_phz", "#3a7d44"),
        ("phz_median", "#8f6bb8"),
    ]:
        if column in redshift:
            z = redshift[column].to_numpy(dtype=float)
            series.append((z[np.isfinite(z)], column, color))
    series = [(value, label, color) for value, label, color in series if len(value)]
    if not series:
        return None
    high = max(float(np.nanquantile(value, 0.995)) for value, _, _ in series)
    high = max(1.0, min(6.0, high))
    bins = np.linspace(0.0, high, 45)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for value, label, color in series:
        histtype = "stepfilled" if label == "posterior median" else "step"
        alpha = 0.35 if label == "posterior median" else 0.95
        ax.hist(value, bins=bins, histtype=histtype, color=color, alpha=alpha, label=label)
    ax.set_xlabel("z")
    ax.set_ylabel("count")
    ax.set_title("Redshift distribution comparison")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out / "redshift_distribution_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_learned_prior_logprob_plot(
    prior: pd.DataFrame,
    out: Path,
    plt,
) -> Path | None:
    if "logprior" not in prior:
        return None
    values = prior["logprior"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=50, histtype="stepfilled", color="#ef476f", alpha=0.35)
    ax.axvline(np.nanmedian(values), color="#ef476f", lw=1.2)
    ax.set_xlabel("log p_beta(x)")
    ax.set_ylabel("prior sample count")
    ax.set_title("Learned RealNVP prior density")
    fig.tight_layout()
    path = out / "learned_prior_logprob_hist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_redshift_pit_plot(out: Path, plt) -> Path | None:
    pit_path = out / "redshift_pit.parquet"
    if not pit_path.exists():
        return None
    pit = pd.read_parquet(pit_path)
    if pit.empty or "pit" not in pit:
        return None
    values = pit["pit"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    bins = np.linspace(0.0, 1.0, 21)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(values, bins=bins, histtype="step", color="#ef476f", lw=1.5, label="amortized")
    ax.axhline(values.size / (len(bins) - 1), color="#2a9fd6", lw=1.2, label="uniform")
    ax.set_xlabel("P(z < z_ref)")
    ax.set_ylabel("count")
    ax.set_title("Redshift PIT diagnostic")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out / "redshift_pit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _corner_columns(samples: pd.DataFrame) -> list[str]:
    return [column for column in _CORNER_PARAMETER_ORDER if column in samples]


def _finite_sample(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    max_rows: int,
) -> pd.DataFrame:
    work = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) > max_rows:
        work = work.sample(max_rows, random_state=0)
    return work


def _corner_ranges(
    work: pd.DataFrame,
    comparison: pd.DataFrame | None,
    columns: list[str],
) -> dict[str, tuple[float, float]]:
    ranges = {}
    for column in columns:
        values = [work[column].to_numpy(dtype=float)]
        if comparison is not None and column in comparison:
            values.append(comparison[column].to_numpy(dtype=float))
        merged = np.concatenate(values)
        merged = merged[np.isfinite(merged)]
        if merged.size == 0:
            ranges[column] = (0.0, 1.0)
            continue
        lo, hi = np.nanquantile(merged, [0.005, 0.995])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.nanmin(merged)), float(np.nanmax(merged))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        pad = 0.05 * (hi - lo)
        ranges[column] = (float(lo - pad), float(hi + pad))
    return ranges


def _plot_1d_hist(ax, values, value_range, *, color: str, label: str) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    ax.hist(
        values,
        bins=36,
        range=value_range,
        density=True,
        histtype="step",
        color=color,
        lw=1.0,
        label=label,
    )


def _plot_2d_contours(
    ax,
    x_values,
    y_values,
    x_range,
    y_range,
    *,
    color: str,
    linewidth: float,
) -> None:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if finite.sum() < 10:
        return
    hist, x_edges, y_edges = np.histogram2d(
        x_values[finite],
        y_values[finite],
        bins=36,
        range=[x_range, y_range],
    )
    if not np.any(hist > 0.0):
        return
    levels = _credible_contour_levels(hist, probs=(0.95, 0.68))
    if not levels:
        return
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    ax.contour(
        x_centers,
        y_centers,
        hist.T,
        levels=levels,
        colors=color,
        linewidths=linewidth,
    )


def _credible_contour_levels(
    hist: np.ndarray,
    *,
    probs: tuple[float, ...],
) -> list[float]:
    flat = np.asarray(hist, dtype=float).ravel()
    flat = flat[np.isfinite(flat) & (flat > 0.0)]
    if flat.size == 0:
        return []
    ordered = np.sort(flat)[::-1]
    cdf = np.cumsum(ordered)
    cdf = cdf / cdf[-1]
    levels = []
    for prob in probs:
        index = min(int(np.searchsorted(cdf, prob, side="left")), len(ordered) - 1)
        levels.append(float(ordered[index]))
    levels = sorted(set(level for level in levels if level > 0.0))
    return levels


def _label(column: str) -> str:
    return _PARAMETER_LABELS.get(column, column)


def _nanquantile_or_nan(values, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.nanquantile(values, quantile))


def _prepare_matplotlib_cache(out: Path) -> None:
    cache_dir = Path(out) / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
