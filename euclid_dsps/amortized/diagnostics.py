"""Diagnostics for FS2 amortized inference outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.amortized.catalog_identity import (
    available_columns,
    truth_columns_from_config,
)
from euclid_dsps.io import (
    iter_catalog_batches,
    truth_column_from_spec,
    truth_value_from_spec,
    write_json,
)
from euclid_dsps.photometric_uncertainty import effective_flux_sigma

FENIKS_FULL_18D_PARAMETER_ORDER = [
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
    "diffstar_lgmcrit",
    "diffstar_lgy_at_mcrit",
    "diffstar_indx_lo",
    "diffstar_indx_hi",
    "diffstar_lg_qt",
    "diffstar_qlglgdt",
    "diffstar_lg_drop",
    "diffstar_lg_rejuv",
    "diffmah_logm0",
    "diffmah_logtc",
    "diffmah_early_index",
    "diffmah_late_index",
    "diffmah_t_peak",
]

FENIKS_USEFUL_PARAMETER_ORDER = [
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
    "log10_sfr_at_obs",
    "log10_ssfr_at_obs",
    "diffstar_lgmcrit",
    "diffstar_lgy_at_mcrit",
    "diffmah_logm0",
    "diffmah_t_peak",
]

_LEGACY_POPCOSMOS_PARAMETER_ORDER = [
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

_CORNER_PARAMETER_ORDER = [
    *FENIKS_FULL_18D_PARAMETER_ORDER,
    *[
        name
        for name in _LEGACY_POPCOSMOS_PARAMETER_ORDER
        if name not in FENIKS_FULL_18D_PARAMETER_ORDER
    ],
]

_PARAMETER_LABELS = {
    "z_obs": "z",
    "log10_stellar_mass": "log M",
    "log10_stellar_metallicity": "log Z*",
    "dust_av": "A_V",
    "dust_delta": "dust delta",
    "log10_sfr_at_obs": "log SFR",
    "log10_ssfr_at_obs": "log sSFR",
    "diffstar_lgmcrit": "DS lgmcrit",
    "diffstar_lgy_at_mcrit": "DS lgy",
    "diffstar_indx_lo": "DS idx lo",
    "diffstar_indx_hi": "DS idx hi",
    "diffstar_lg_qt": "DS lg qt",
    "diffstar_qlglgdt": "DS q dt",
    "diffstar_lg_drop": "DS lg drop",
    "diffstar_lg_rejuv": "DS lg rejuv",
    "diffmah_logm0": "DM log M0",
    "diffmah_logtc": "DM log tc",
    "diffmah_early_index": "DM early",
    "diffmah_late_index": "DM late",
    "diffmah_t_peak": "DM tpeak",
    "dlog10_sfr_1": "dSFR1",
    "dlog10_sfr_2": "dSFR2",
    "dlog10_sfr_3": "dSFR3",
    "dlog10_sfr_4": "dSFR4",
    "dlog10_sfr_5": "dSFR5",
    "dlog10_sfr_6": "dSFR6",
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
        "likelihood_temperature",
        "entropy_floor_penalty",
        "posterior_entropy_mean",
        "posterior_min_log_std",
        "posterior_median_log_std",
        "logprior_mean",
        "logq_mean",
        "residual_rms",
        "flux_residual_rms",
        "finite_fraction",
        "encoder_grad_norm",
        "prior_grad_norm",
        "band_alpha_grad_norm",
        "joint_grad_norm",
        "band_alpha_prior_penalty",
        "max_abs_band_delta_mag",
        "wake_nll",
        "wake_ess_mean",
        "wake_ess_fraction_mean",
        "wake_weight_max_mean",
        "wake_weight_entropy_mean",
        "wake_all_nonfinite_fraction",
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
        "likelihood_temperature",
        "entropy_floor_penalty",
        "posterior_entropy_mean",
        "posterior_min_log_std",
        "posterior_median_log_std",
        "logprior_mean",
        "logq_mean",
        "residual_rms",
        "flux_residual_rms",
        "finite_fraction",
        "encoder_grad_norm",
        "prior_grad_norm",
        "band_alpha_grad_norm",
        "joint_grad_norm",
        "band_alpha_prior_penalty",
        "max_abs_band_delta_mag",
        "wake_nll",
        "wake_ess_mean",
        "wake_ess_fraction_mean",
        "wake_weight_max_mean",
        "wake_weight_entropy_mean",
        "wake_all_nonfinite_fraction",
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
                float(np.nanmean(group["kl_weight"]))
                if "kl_weight" in group
                else np.nan
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
        "band_alpha_grad_norm": "per-band calibration grad norm",
        "joint_grad_norm": "joint grad norm",
        "band_alpha_prior_penalty": "per-band calibration prior penalty",
        "max_abs_band_delta_mag": "max |band offset| (mag)",
    }.get(metric, metric)


def _is_gradient_metric(metric: str) -> bool:
    return metric in {
        "encoder_grad_norm",
        "prior_grad_norm",
        "band_alpha_grad_norm",
        "joint_grad_norm",
    }


def _use_log_y(metric: str, summary: pd.DataFrame) -> bool:
    if metric not in {
        "encoder_grad_norm",
        "prior_grad_norm",
        "band_alpha_grad_norm",
        "joint_grad_norm",
        "band_alpha_prior_penalty",
    }:
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
    *,
    row_index=None,
    likelihood_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return long-form likelihood-normalized posterior predictive residual rows."""
    object_id = np.asarray(object_id)
    row_index = _optional_row_index(row_index, object_id)
    obs_flux = np.asarray(obs_flux, dtype=float)
    obs_err = np.asarray(obs_err, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    model_flux = np.asarray(model_flux, dtype=float)
    if model_flux.ndim != 3:
        raise ValueError(f"model_flux must be [K,N,B], got {model_flux.shape}")
    likelihood_config = likelihood_config or {}
    sigma_eff = effective_flux_sigma(
        obs_flux[None, :, :],
        obs_err[None, :, :],
        model_flux=model_flux,
        error_floor_frac=float(likelihood_config.get("error_floor_frac", 0.0)),
        error_jitter=float(likelihood_config.get("error_jitter", 0.0)),
        floor_reference=str(likelihood_config.get("error_floor_reference", "model")),
    )
    rows = []
    n_samples, n_objects, n_bands = model_flux.shape
    for sample_id in range(n_samples):
        for object_index in range(n_objects):
            for band_index in range(n_bands):
                err = obs_err[object_index, band_index]
                sigma = sigma_eff[sample_id, object_index, band_index]
                raw_residual = (
                    obs_flux[object_index, band_index]
                    - model_flux[sample_id, object_index, band_index]
                )
                abs_obs_flux = max(abs(obs_flux[object_index, band_index]), 1.0e-300)
                residual = np.nan
                raw_residual_sigma = np.nan
                snr_proxy = np.nan
                err_over_abs_flux = np.nan
                if mask[object_index, band_index]:
                    if np.isfinite(sigma) and sigma > 0.0:
                        residual = raw_residual / sigma
                    if np.isfinite(err) and err > 0.0:
                        raw_residual_sigma = raw_residual / err
                        snr_proxy = abs_obs_flux / err
                        err_over_abs_flux = err / abs_obs_flux
                rows.append(
                    {
                        "object_id": object_id[object_index],
                        **_row_index_value(row_index, object_index),
                        "sample_id": int(sample_id),
                        "band": band_names[band_index],
                        "obs_flux_fnu_cgs": float(obs_flux[object_index, band_index]),
                        "obs_err_fnu_cgs": float(obs_err[object_index, band_index]),
                        "model_flux_fnu_cgs": float(
                            model_flux[sample_id, object_index, band_index]
                        ),
                        "sigma_eff_fnu_cgs": float(sigma),
                        "snr_proxy": float(snr_proxy),
                        "obs_err_over_abs_flux": float(err_over_abs_flux),
                        "flux_residual_obs_minus_model_fnu_cgs": float(raw_residual),
                        "abs_flux_residual_fnu_cgs": float(abs(raw_residual)),
                        "abs_flux_residual_over_abs_flux": float(
                            abs(raw_residual) / abs_obs_flux
                        ),
                        "chi_likelihood": float(residual),
                        "residual_sigma": float(residual),
                        "raw_residual_sigma": float(raw_residual_sigma),
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
    *,
    row_index=None,
    likelihood_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return one likelihood-normalized residual summary row per object-band."""
    object_id = np.asarray(object_id)
    row_index = _optional_row_index(row_index, object_id)
    obs_flux = np.asarray(obs_flux, dtype=float)
    obs_err = np.asarray(obs_err, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    model_flux = np.asarray(model_flux, dtype=float)
    if model_flux.ndim != 3:
        raise ValueError(f"model_flux must be [K,N,B], got {model_flux.shape}")
    likelihood_config = likelihood_config or {}
    rows = []
    _n_samples, n_objects, n_bands = model_flux.shape
    sigma_eff = effective_flux_sigma(
        obs_flux[None, :, :],
        obs_err[None, :, :],
        model_flux=model_flux,
        error_floor_frac=float(likelihood_config.get("error_floor_frac", 0.0)),
        error_jitter=float(likelihood_config.get("error_jitter", 0.0)),
        floor_reference=str(likelihood_config.get("error_floor_reference", "model")),
    )
    flux_residual = obs_flux[None, :, :] - model_flux
    abs_flux_residual = np.abs(flux_residual)
    abs_obs_flux = np.maximum(np.abs(obs_flux), 1.0e-300)
    abs_flux_residual_over_abs_flux = abs_flux_residual / abs_obs_flux[None, :, :]
    residual = np.full_like(model_flux, np.nan, dtype=float)
    valid_sigma = np.isfinite(sigma_eff) & (sigma_eff > 0.0) & mask[None, :, :]
    residual[valid_sigma] = flux_residual[valid_sigma] / sigma_eff[valid_sigma]
    raw_residual = np.full_like(model_flux, np.nan, dtype=float)
    valid_err = np.isfinite(obs_err) & (obs_err > 0.0) & mask
    raw_residual[:, valid_err] = (
        flux_residual[:, valid_err] / obs_err[None, :, :][:, valid_err]
    )
    for object_index in range(n_objects):
        for band_index in range(n_bands):
            rows.append(
                {
                    "object_id": object_id[object_index],
                    **_row_index_value(row_index, object_index),
                    "band": band_names[band_index],
                    "obs_flux_fnu_cgs": float(obs_flux[object_index, band_index]),
                    "obs_err_fnu_cgs": float(obs_err[object_index, band_index]),
                    "snr_proxy": float(
                        abs_obs_flux[object_index, band_index]
                        / obs_err[object_index, band_index]
                    )
                    if obs_err[object_index, band_index] > 0.0
                    else float("nan"),
                    "obs_err_over_abs_flux": float(
                        obs_err[object_index, band_index]
                        / abs_obs_flux[object_index, band_index]
                    ),
                    "model_flux_q16": float(
                        _nanquantile_or_nan(
                            model_flux[:, object_index, band_index], 0.16
                        )
                    ),
                    "model_flux_median": float(
                        np.nanmedian(model_flux[:, object_index, band_index])
                    ),
                    "model_flux_q84": float(
                        _nanquantile_or_nan(
                            model_flux[:, object_index, band_index], 0.84
                        )
                    ),
                    "sigma_eff_q16": float(
                        _nanquantile_or_nan(
                            sigma_eff[:, object_index, band_index], 0.16
                        )
                    ),
                    "sigma_eff_median": float(
                        np.nanmedian(sigma_eff[:, object_index, band_index])
                    ),
                    "sigma_eff_q84": float(
                        _nanquantile_or_nan(
                            sigma_eff[:, object_index, band_index], 0.84
                        )
                    ),
                    "flux_residual_obs_minus_model_q16": float(
                        _nanquantile_or_nan(
                            flux_residual[:, object_index, band_index], 0.16
                        )
                    ),
                    "flux_residual_obs_minus_model_median": float(
                        np.nanmedian(flux_residual[:, object_index, band_index])
                    ),
                    "flux_residual_obs_minus_model_q84": float(
                        _nanquantile_or_nan(
                            flux_residual[:, object_index, band_index], 0.84
                        )
                    ),
                    "abs_flux_residual_median": float(
                        np.nanmedian(abs_flux_residual[:, object_index, band_index])
                    ),
                    "abs_flux_residual_over_abs_flux_median": float(
                        np.nanmedian(
                            abs_flux_residual_over_abs_flux[:, object_index, band_index]
                        )
                    ),
                    "chi_likelihood_q16": float(
                        _nanquantile_or_nan(residual[:, object_index, band_index], 0.16)
                    ),
                    "chi_likelihood_median": float(
                        np.nanmedian(residual[:, object_index, band_index])
                    ),
                    "chi_likelihood_q84": float(
                        _nanquantile_or_nan(residual[:, object_index, band_index], 0.84)
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
                    "raw_residual_sigma_median": float(
                        np.nanmedian(raw_residual[:, object_index, band_index])
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
    row_index=None,
    n_flux_bands: int = 10,
) -> pd.DataFrame:
    """Return per-object feature magnitude diagnostics."""
    object_id = np.asarray(object_id)
    row_index = _optional_row_index(row_index, object_id)
    features = np.asarray(features, dtype=float)
    flux_features = features[:, :n_flux_bands]
    err_features = features[:, n_flux_bands:]
    data = {
        "object_id": object_id,
        "feature_max_abs": np.nanmax(np.abs(features), axis=1),
        "flux_feature_max_abs": np.nanmax(np.abs(flux_features), axis=1),
        "err_feature_max_abs": np.nanmax(np.abs(err_features), axis=1),
    }
    if row_index is not None:
        data["row_index"] = row_index
    return pd.DataFrame(data)


def _optional_row_index(row_index, object_id: np.ndarray) -> np.ndarray | None:
    if row_index is None:
        return None
    values = np.asarray(row_index, dtype=np.int64)
    if values.shape[0] != np.asarray(object_id).shape[0]:
        raise ValueError(
            "row_index length must match object_id length: "
            f"{values.shape[0]} vs {np.asarray(object_id).shape[0]}"
        )
    return values


def _row_index_value(row_index: np.ndarray | None, object_index: int) -> dict[str, int]:
    if row_index is None:
        return {}
    return {"row_index": int(row_index[object_index])}


def summarize_inference_outputs(
    summary_path: str | Path,
    out_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
    limit: int | None = None,
    row_indices: np.ndarray | None = None,
) -> None:
    """Write posterior predictive diagnostics from inference parquet outputs."""
    summary_path = Path(summary_path)
    out = Path(out_dir)
    if not summary_path.exists():
        return
    _remove_redundant_corner_artifacts(out)
    frame = pd.read_parquet(summary_path)
    residual_summary = _write_residual_summary(out)
    top_chi2 = _write_top_chi2(frame, residual_summary, out)
    redshift = _write_redshift_comparison(
        frame,
        out,
        config=config,
        limit=limit,
        row_indices=row_indices,
    )
    catalog_proxy = _write_catalog_proxy_comparison(
        frame,
        out,
        config=config,
        limit=limit,
        row_indices=row_indices,
    )
    prior_summary = _write_learned_prior_summary(out)
    pit_summary = _write_redshift_pit(redshift, out)
    residual_tail_summary = _write_residual_tail_summary(residual_summary, out)
    plots = _write_inference_plots(
        frame,
        residual_summary,
        top_chi2,
        redshift,
        catalog_proxy,
        out,
        config=config,
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
        "normalized_residual_tail_rows": int(len(residual_tail_summary)),
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


_REDUNDANT_CORNER_ARTIFACTS = (
    "corner_full_latent_posterior_medians.png",
    "corner_full_latent_posterior_medians_columns.csv",
    "corner_truth_prior_posterior_map.png",
    "posterior_corner.png",
    "learned_prior_corner.png",
    "posterior_vs_learned_prior_corner.png",
)


def _remove_redundant_corner_artifacts(out: Path) -> None:
    """Keep one canonical 15D truth/prior/posterior corner per inference run."""
    for name in _REDUNDANT_CORNER_ARTIFACTS:
        path = out / name
        if path.exists():
            path.unlink()


def _write_full_latent_truth_prior_posterior_corner(
    summary: pd.DataFrame,
    out: Path,
    plt,
    *,
    config: dict[str, Any] | None,
) -> Path | None:
    posterior, posterior_label = _full_latent_posterior_frame(out, config=config)
    if posterior.empty:
        return None
    truth = _truth_parameter_frame(summary, out, config=config)
    prior = _read_learned_prior(out)
    return _write_multi_overlay_corner_plot(
        posterior,
        out,
        plt,
        truth=truth,
        prior=prior,
        filename="corner_full_latent_truth_prior_posterior.png",
        title="Full latent truth / prior / posterior",
        posterior_label=posterior_label,
        config=config,
    )


def _full_latent_posterior_frame(
    out: Path,
    *,
    config: dict[str, Any] | None,
) -> tuple[pd.DataFrame, str]:
    samples = _read_posterior_samples_for_corner(out)
    if not samples.empty:
        columns = _corner_columns_for_config(samples, config)
        if len(columns) >= 2:
            return samples[columns], "aggregate posterior samples"
    return pd.DataFrame(), "posterior samples unavailable"


def _read_posterior_samples_for_corner(
    out: Path,
    *,
    max_rows: int = 50_000,
) -> pd.DataFrame:
    """Read a deterministic population sample from dense or sharded outputs."""
    dense = out / "posterior_samples.parquet"
    if dense.exists():
        frame = pd.read_parquet(dense)
        if len(frame) > max_rows:
            frame = frame.sample(n=max_rows, random_state=0)
        return frame
    shards = sorted((out / "posterior_samples").glob("batch_*.parquet"))
    if not shards:
        return pd.DataFrame()
    rows_per_shard = max(1, int(np.ceil(max_rows / len(shards))))
    frames = []
    for index, path in enumerate(shards):
        frame = pd.read_parquet(path)
        if len(frame) > rows_per_shard:
            frame = frame.sample(n=rows_per_shard, random_state=index)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) > max_rows:
        combined = combined.sample(n=max_rows, random_state=0)
    return combined


def _truth_parameter_frame(
    summary: pd.DataFrame,
    out: Path,
    *,
    config: dict[str, Any] | None,
) -> pd.DataFrame:
    truth_path = out / "inference_truth.parquet"
    if config is None:
        return pd.DataFrame()
    truth = pd.read_parquet(truth_path) if truth_path.exists() else pd.DataFrame()
    catalog_truth = _catalog_truth_frame(summary, config=config)
    truth = _combine_truth_frames(truth, catalog_truth)
    if truth.empty:
        return pd.DataFrame()
    truth_specs = dict((config.get("truth", {}) or {}).get("parameter_columns") or {})
    names = _configured_free_parameters(config)
    if not names:
        names = [column for column in _CORNER_PARAMETER_ORDER if column in truth]
    columns: dict[str, pd.Series] = {}
    for name in names:
        spec = truth_specs.get(name)
        if isinstance(spec, dict) and str(spec.get("kind", "")).lower() == "missing":
            continue
        series = _truth_parameter_series(truth, name, spec)
        if series is not None:
            columns[name] = series
    if len(columns) < 2:
        return pd.DataFrame()
    return pd.DataFrame(columns)


def _catalog_truth_frame(
    summary: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> pd.DataFrame:
    if summary.empty or "row_index" not in summary:
        return pd.DataFrame()
    catalog_path = config.get("catalog_path")
    if not catalog_path:
        return pd.DataFrame()
    path = Path(str(catalog_path))
    if not path.is_absolute():
        path = Path.cwd() / path
    path = _prefer_projected_truth_catalog(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        available = available_columns(path)
    except Exception:
        return pd.DataFrame()
    names = _configured_free_parameters(config)
    fallback_columns = [
        column for name in names for column in _truth_fallback_columns(name)
    ]
    requested_columns = []
    for column in [*truth_columns_from_config(config), *fallback_columns]:
        if column in available:
            requested_columns.append(column)
        elif column == "log10_metallicity_true" and "metallicity_true" in available:
            requested_columns.append(column)
    requested_columns = sorted(set(requested_columns))
    if not requested_columns:
        return pd.DataFrame()
    row_indices = (
        pd.to_numeric(summary["row_index"], errors="coerce")
        .dropna()
        .astype(np.int64)
        .drop_duplicates()
        .to_numpy()
    )
    if row_indices.size == 0:
        return pd.DataFrame()
    frames = []
    for batch in iter_catalog_batches(
        path,
        columns=requested_columns,
        batch_size=10_000,
        row_indices=set(row_indices.tolist()),
    ):
        frames.append(batch.reset_index(names="row_index"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _prefer_projected_truth_catalog(path: Path) -> Path:
    if path.stem.endswith("_projected_truth"):
        return path
    candidate = path.with_name(f"{path.stem}_projected_truth{path.suffix}")
    return candidate if candidate.exists() else path


def _combine_truth_frames(
    primary: pd.DataFrame,
    fallback: pd.DataFrame,
) -> pd.DataFrame:
    if primary.empty:
        return fallback
    if fallback.empty:
        return primary
    if "row_index" not in primary or "row_index" not in fallback:
        return primary
    merged = primary.merge(
        fallback,
        on="row_index",
        how="outer",
        suffixes=("", "__catalog"),
    )
    for column in list(merged.columns):
        if not column.endswith("__catalog"):
            continue
        base = column[: -len("__catalog")]
        if base in merged:
            merged[base] = merged[base].combine_first(merged[column])
            merged = merged.drop(columns=[column])
        else:
            merged = merged.rename(columns={column: base})
    return merged


def _truth_parameter_series(
    truth: pd.DataFrame,
    name: str,
    spec: Any,
) -> pd.Series | None:
    spec_column = truth_column_from_spec(spec)
    candidates = []
    if spec_column:
        candidates.append(spec_column)
    candidates.extend(_truth_fallback_columns(name))
    for column in dict.fromkeys(candidates):
        if column not in truth:
            continue
        return _apply_truth_spec_to_series(truth[column], spec)
    if name == "tau2" and "dust_av" in truth:
        return pd.to_numeric(truth["dust_av"], errors="coerce") / 1.086
    if name == "dust_index_n" and "dust_delta" in truth:
        return pd.to_numeric(truth["dust_delta"], errors="coerce")
    return None


def _truth_fallback_columns(name: str) -> list[str]:
    fallback = {
        "z_obs": ["z_obs", "redshift_true", "z_true", "z_true_gal", "z_obs_gal"],
        "log10_stellar_mass": [
            "log10_stellar_mass",
            "logsm_true",
            "log10_stellar_mass_true",
        ],
        "log10_stellar_metallicity": [
            "log10_stellar_metallicity",
            "log10_metallicity",
            "log10_metallicity_true",
        ],
        "tau2": ["tau2"],
        "dust_index_n": ["dust_index_n", "dust_delta"],
        "tau1_over_tau2": ["tau1_over_tau2"],
    }
    if name.startswith("dlog10_sfr_"):
        return [name]
    return fallback.get(name, [name])


def _apply_truth_spec_to_series(series: pd.Series, spec: Any) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if not isinstance(spec, dict):
        return values
    arr = values.to_numpy(dtype=float)
    transform = spec.get("transform")
    if transform == "log10":
        arr = np.where(arr > 0.0, np.log10(arr), np.nan)
    elif transform == "log_stellar_mass_h2_to_msun":
        h = float(spec.get("h"))
        if np.isfinite(h) and h > 0.0:
            arr = arr + 2.0 * np.log10(h)
        else:
            arr = np.full_like(arr, np.nan, dtype=float)
    elif transform not in {None, "linear"}:
        rows = [{truth_column_from_spec(spec): value} for value in values]
        arr = np.asarray(
            [truth_value_from_spec(row, spec) for row in rows],
            dtype=float,
        )
        return pd.Series(arr, index=series.index)
    arr = arr * float(spec.get("scale", 1.0))
    arr = arr + float(spec.get("offset", 0.0))
    return pd.Series(arr, index=series.index)


def _residual_value_column(frame: pd.DataFrame) -> str | None:
    for column in ("chi_likelihood_median", "residual_sigma_median"):
        if column in frame:
            return column
    return None


def _write_residual_tail_summary(
    residual_summary: pd.DataFrame, out: Path
) -> pd.DataFrame:
    column = _residual_value_column(residual_summary)
    if residual_summary.empty or column is None:
        return pd.DataFrame()
    rows = [_residual_tail_row(residual_summary[column], band="__all__")]
    for band, group in residual_summary.groupby("band", sort=True):
        rows.append(_residual_tail_row(group[column], band=str(band)))
    table = pd.DataFrame(rows)
    table.to_csv(
        out / "posterior_predictive_normalized_residual_tails.csv", index=False
    )
    table.to_parquet(
        out / "posterior_predictive_normalized_residual_tails.parquet",
        index=False,
    )
    return table


def _residual_tail_row(values: pd.Series, *, band: str) -> dict[str, float | int | str]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "band": band,
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "p16": np.nan,
            "p84": np.nan,
            "frac_abs_gt_3": np.nan,
            "frac_abs_gt_5": np.nan,
        }
    return {
        "band": band,
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p16": float(np.quantile(arr, 0.16)),
        "p84": float(np.quantile(arr, 0.84)),
        "frac_abs_gt_3": float(np.mean(np.abs(arr) > 3.0)),
        "frac_abs_gt_5": float(np.mean(np.abs(arr) > 5.0)),
    }


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
    residual_column = (
        "chi_likelihood" if "chi_likelihood" in residuals else "residual_sigma"
    )
    grouped = residuals.groupby(["object_id", "band"], sort=False)
    agg_kwargs = {
        "obs_flux_fnu_cgs": ("obs_flux_fnu_cgs", "first"),
        "obs_err_fnu_cgs": ("obs_err_fnu_cgs", "first"),
        "model_flux_q16": (
            "model_flux_fnu_cgs",
            lambda x: _nanquantile_or_nan(x, 0.16),
        ),
        "model_flux_median": ("model_flux_fnu_cgs", "median"),
        "model_flux_q84": (
            "model_flux_fnu_cgs",
            lambda x: _nanquantile_or_nan(x, 0.84),
        ),
        "chi_likelihood_q16": (
            residual_column,
            lambda x: _nanquantile_or_nan(x, 0.16),
        ),
        "chi_likelihood_median": (residual_column, "median"),
        "chi_likelihood_q84": (
            residual_column,
            lambda x: _nanquantile_or_nan(x, 0.84),
        ),
        "residual_sigma_q16": (
            residual_column,
            lambda x: _nanquantile_or_nan(x, 0.16),
        ),
        "residual_sigma_median": (residual_column, "median"),
        "residual_sigma_q84": (
            residual_column,
            lambda x: _nanquantile_or_nan(x, 0.84),
        ),
        "valid": ("valid", "first"),
    }
    if "sigma_eff_fnu_cgs" in residuals:
        agg_kwargs.update(
            {
                "sigma_eff_q16": (
                    "sigma_eff_fnu_cgs",
                    lambda x: _nanquantile_or_nan(x, 0.16),
                ),
                "sigma_eff_median": ("sigma_eff_fnu_cgs", "median"),
                "sigma_eff_q84": (
                    "sigma_eff_fnu_cgs",
                    lambda x: _nanquantile_or_nan(x, 0.84),
                ),
            }
        )
    if "flux_residual_obs_minus_model_fnu_cgs" in residuals:
        agg_kwargs.update(
            {
                "flux_residual_obs_minus_model_q16": (
                    "flux_residual_obs_minus_model_fnu_cgs",
                    lambda x: _nanquantile_or_nan(x, 0.16),
                ),
                "flux_residual_obs_minus_model_median": (
                    "flux_residual_obs_minus_model_fnu_cgs",
                    "median",
                ),
                "flux_residual_obs_minus_model_q84": (
                    "flux_residual_obs_minus_model_fnu_cgs",
                    lambda x: _nanquantile_or_nan(x, 0.84),
                ),
            }
        )
    if "raw_residual_sigma" in residuals:
        agg_kwargs["raw_residual_sigma_median"] = ("raw_residual_sigma", "median")
    summary = grouped.agg(**agg_kwargs).reset_index()
    summary["abs_residual_sigma_median"] = summary["residual_sigma_median"].abs()
    summary.to_parquet(
        out / "posterior_predictive_residual_summary.parquet", index=False
    )
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
    row_indices: np.ndarray | None,
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
        row_indices=(
            None
            if row_indices is None
            else set(np.asarray(row_indices, dtype=np.int64).tolist())
        ),
    ):
        work = batch.copy()
        work["row_index"] = work.index.to_numpy()
        work["object_id"] = work.index.to_numpy()
        frames.append(work)
    if not frames:
        return pd.DataFrame()
    proxies = pd.concat(frames, axis=0, ignore_index=True)
    identity = _summary_identity_column(summary, proxies)
    comparison = summary.merge(proxies, on=identity, how="left")
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
    row_indices: np.ndarray | None,
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
        row_indices=(
            None
            if row_indices is None
            else set(np.asarray(row_indices, dtype=np.int64).tolist())
        ),
    ):
        rows = []
        for row_index, row in batch.iterrows():
            row_dict = row.to_dict()
            out_row = {"row_index": row_index, "object_id": row_index}
            for output_name, spec in specs.items():
                out_row[output_name] = truth_value_from_spec(row_dict, spec)
            rows.append(out_row)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    proxies = pd.concat(frames, axis=0, ignore_index=True)
    identity = _summary_identity_column(summary, proxies)
    comparison = summary.merge(proxies, on=identity, how="left")
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


def _summary_identity_column(summary: pd.DataFrame, proxies: pd.DataFrame) -> str:
    if "row_index" in summary and "row_index" in proxies:
        return "row_index"
    return "object_id"


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
    *,
    config: dict[str, Any] | None = None,
) -> list[str]:
    try:
        _prepare_matplotlib_cache(out)
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting fallback
        return []

    written: list[str] = []
    path = _write_full_latent_truth_prior_posterior_corner(
        summary,
        out,
        plt,
        config=config,
    )
    if path is not None:
        written.append(path.name)
    if not residual_summary.empty:
        residual_column = _residual_value_column(residual_summary)
        if residual_column is None:
            residual_column = "residual_sigma_median"
        bands = list(dict.fromkeys(residual_summary["band"].astype(str)))
        data = [
            residual_summary.loc[
                residual_summary["band"].astype(str) == band,
                residual_column,
            ].to_numpy()
            for band in bands
        ]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.axhline(0.0, color="black", lw=1.0, alpha=0.5)
        ax.axhline(-3.0, color="tab:red", lw=1.0, alpha=0.6, ls="--")
        ax.axhline(3.0, color="tab:red", lw=1.0, alpha=0.6, ls="--")
        ax.boxplot(data, tick_labels=bands, showfliers=False)
        ax.set_ylabel("median (flux_in - flux_out) / sigma_eff")
        ax.set_title("Likelihood-normalized posterior predictive residuals by band")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = out / "posterior_predictive_residuals_by_band.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)

        fig, ax = plt.subplots(figsize=(7, 4))
        values = pd.to_numeric(
            residual_summary[residual_column], errors="coerce"
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            ax.hist(values, bins=60, density=True, alpha=0.65, label="posterior median")
            x = np.linspace(
                min(-6.0, float(np.nanpercentile(values, 1.0))),
                max(6.0, float(np.nanpercentile(values, 99.0))),
                400,
            )
            y = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
            ax.plot(x, y, color="black", lw=1.2, label="N(0,1)")
        ax.axvline(-3.0, color="tab:red", lw=1.0, ls="--")
        ax.axvline(3.0, color="tab:red", lw=1.0, ls="--")
        ax.axvline(0.0, color="black", lw=1.0, alpha=0.5)
        ax.set_xlabel("(flux_in - flux_out) / sigma_eff")
        ax.set_ylabel("density")
        ax.set_title("Likelihood-normalized posterior predictive residuals")
        ax.legend(loc="best")
        fig.tight_layout()
        path = out / "posterior_predictive_normalized_residual_hist.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)

        n_bands = len(bands)
        if n_bands:
            n_cols = min(4, n_bands)
            n_rows = int(np.ceil(n_bands / n_cols))
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(3.4 * n_cols, 2.6 * n_rows),
                squeeze=False,
            )
            for ax, band, values_for_band in zip(
                axes.ravel(),
                bands,
                data,
                strict=False,
            ):
                band_values = np.asarray(values_for_band, dtype=float)
                band_values = band_values[np.isfinite(band_values)]
                if band_values.size:
                    ax.hist(band_values, bins=40, density=True, alpha=0.65)
                    x = np.linspace(-6.0, 6.0, 240)
                    y = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
                    ax.plot(x, y, color="black", lw=1.0)
                ax.axvline(-3.0, color="tab:red", lw=0.9, ls="--")
                ax.axvline(3.0, color="tab:red", lw=0.9, ls="--")
                ax.axvline(0.0, color="black", lw=0.8, alpha=0.5)
                ax.set_xlim(-8.0, 8.0)
                ax.set_title(band)
            for ax in axes.ravel()[n_bands:]:
                ax.axis("off")
            fig.suptitle("Likelihood-normalized residuals by band", y=0.995)
            fig.tight_layout()
            path = out / "posterior_predictive_normalized_residual_hist_by_band.png"
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
                for column in [
                    "z_true_gal",
                    "z_obs_gal",
                    "z_true",
                    "z_phz",
                    "phz_median",
                ]
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
        if prior is not None and not prior.empty:
            path = _write_learned_prior_logprob_plot(prior, out, plt)
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
    for filename in (
        "learned_prior_samples.parquet",
        "learned_or_loaded_prior_samples.parquet",
    ):
        prior_path = out / filename
        if not prior_path.exists():
            continue
        prior = pd.read_parquet(prior_path)
        if not prior.empty:
            return prior
    return None


def _write_multi_overlay_corner_plot(
    posterior: pd.DataFrame,
    out: Path,
    plt,
    *,
    truth: pd.DataFrame | None,
    prior: pd.DataFrame | None,
    filename: str,
    title: str,
    posterior_label: str,
    config: dict[str, Any] | None,
) -> Path | None:
    columns = _corner_columns_for_config(posterior, config)
    if len(columns) < 2 or posterior.empty:
        return None
    frames = [
        {
            "key": "posterior",
            "label": posterior_label,
            "frame": _finite_sample_partial(posterior, columns, max_rows=4_000),
            "color": "#0072b2",
            "linestyle": "-",
            "linewidth": 1.15,
            "fill_alpha": 0.16,
            "contour_alpha": 0.95,
            "points": False,
        },
    ]
    if truth is not None and not truth.empty:
        frames.append(
            {
                "key": "truth",
                "label": "truth / projected truth",
                "frame": _finite_sample_partial(truth, columns, max_rows=4_000),
                "color": "#d55e00",
                "linestyle": "--",
                "linewidth": 1.45,
                "fill_alpha": 0.0,
                "contour_alpha": 0.98,
                "points": True,
            }
        )
    if prior is not None and not prior.empty:
        frames.append(
            {
                "key": "prior",
                "label": "learned prior",
                "frame": _finite_sample_partial(prior, columns, max_rows=4_000),
                "color": "#111827",
                "linestyle": ":",
                "linewidth": 1.25,
                "fill_alpha": 0.0,
                "contour_alpha": 0.88,
                "points": False,
            }
        )
    frames = [item for item in frames if not item["frame"].empty]
    if not frames or frames[0]["key"] != "posterior":
        return None
    columns = [
        column for column in columns if any(column in item["frame"] for item in frames)
    ]
    if len(columns) < 2:
        return None
    ranges = _corner_ranges_multi(
        [item["frame"] for item in frames],
        columns,
    )
    n_columns = len(columns)
    fig, axes = plt.subplots(
        n_columns,
        n_columns,
        figsize=(1.75 * n_columns, 1.75 * n_columns),
    )
    for row, y_col in enumerate(columns):
        for col, x_col in enumerate(columns):
            ax = axes[row, col]
            if row == col:
                for item in frames:
                    frame = item["frame"]
                    if x_col not in frame:
                        continue
                    finite_values = (
                        pd.to_numeric(frame[x_col], errors="coerce")
                        .dropna()
                        .to_numpy(dtype=float)
                    )
                    if item["key"] == "truth" and finite_values.size == 1:
                        ax.axvline(
                            finite_values[0],
                            color=item["color"],
                            linestyle=item["linestyle"],
                            linewidth=item["linewidth"],
                            label=item["label"],
                        )
                        continue
                    _plot_1d_hist(
                        ax,
                        finite_values,
                        ranges[x_col],
                        color=item["color"],
                        label=item["label"],
                        linestyle=item["linestyle"],
                        linewidth=item["linewidth"],
                        fill_alpha=item["fill_alpha"],
                    )
                posterior_frame = frames[0]["frame"]
                if x_col in posterior_frame:
                    values = posterior_frame[x_col].to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    if values.size:
                        q16, q50, q84 = np.nanquantile(values, [0.16, 0.50, 0.84])
                        ax.axvline(q50, color=frames[0]["color"], lw=0.9, alpha=0.8)
                        ax.set_title(
                            f"{_label(x_col)}={q50:.2g}"
                            f"+{q84 - q50:.2g}/-{q50 - q16:.2g}",
                            fontsize=6,
                            color=frames[0]["color"],
                        )
            elif row > col:
                for item in frames:
                    frame = item["frame"]
                    if x_col not in frame or y_col not in frame:
                        continue
                    _plot_2d_contours(
                        ax,
                        frame[x_col].to_numpy(dtype=float),
                        frame[y_col].to_numpy(dtype=float),
                        ranges[x_col],
                        ranges[y_col],
                        color=item["color"],
                        linewidth=item["linewidth"],
                        linestyle=item["linestyle"],
                        fill_alpha=item["fill_alpha"],
                        alpha=item["contour_alpha"],
                    )
                    if item["points"]:
                        _plot_2d_points(
                            ax,
                            frame[x_col].to_numpy(dtype=float),
                            frame[y_col].to_numpy(dtype=float),
                            ranges[x_col],
                            ranges[y_col],
                            color=item["color"],
                        )
                ax.set_ylim(*ranges[y_col])
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
            ax.tick_params(labelsize=5)
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D(
            [0],
            [0],
            color=item["color"],
            linestyle=item["linestyle"],
            linewidth=max(1.4, float(item["linewidth"])),
            label=str(item["label"]),
        )
        for item in frames
    ]
    legend_ax = axes[0, min(1, n_columns - 1)]
    legend_ax.legend(handles=legend_items, frameon=False, fontsize=7, loc="center")
    fig.suptitle(title, y=1.002)
    fig.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _write_multi_overlay_corner_metadata(out, filename, columns, frames)
    return path


def _write_multi_overlay_corner_metadata(
    out: Path,
    filename: str,
    columns: list[str],
    frames: list[dict[str, Any]],
) -> None:
    rows = []
    for column in columns:
        row: dict[str, Any] = {"parameter": column}
        for item in frames:
            key = str(item["key"])
            frame = item["frame"]
            if column not in frame:
                row[f"{key}_finite_rows"] = 0
                row[f"{key}_available"] = False
                continue
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            row[f"{key}_finite_rows"] = int(finite.sum())
            row[f"{key}_available"] = bool(finite.any())
        rows.append(row)
    if not rows:
        return
    suffix = ".png"
    stem = filename[: -len(suffix)] if filename.endswith(suffix) else filename
    pd.DataFrame(rows).to_csv(out / f"{stem}_columns.csv", index=False)


def _corner_columns_for_config(
    frame: pd.DataFrame,
    config: dict[str, Any] | None,
) -> list[str]:
    configured = _configured_free_parameters(config)
    configured_order = [column for column in configured if column in frame]
    if len(configured_order) >= 2:
        return configured_order
    ordered = [
        column
        for column in _CORNER_PARAMETER_ORDER
        if column in frame and (not configured or column in configured)
    ]
    for column in configured:
        if column in frame and column not in ordered:
            ordered.append(column)
    if ordered:
        return ordered
    return _corner_columns(frame)


def _configured_free_parameters(config: dict[str, Any] | None) -> list[str]:
    if config is None:
        return []
    parameters = (config.get("fit", {}) or {}).get("free_parameters", {}) or {}
    return [str(name) for name in parameters]


def _finite_sample_partial(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    max_rows: int,
) -> pd.DataFrame:
    available = [column for column in columns if column in frame]
    if not available:
        return pd.DataFrame()
    work = frame[available].replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if len(work) > max_rows:
        work = work.sample(max_rows, random_state=0)
    return work


def _corner_ranges_multi(
    frames: list[pd.DataFrame],
    columns: list[str],
) -> dict[str, tuple[float, float]]:
    ranges = {}
    for column in columns:
        values = []
        for frame in frames:
            if column in frame:
                arr = frame[column].to_numpy(dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    values.append(arr)
        if not values:
            ranges[column] = (0.0, 1.0)
            continue
        merged = np.concatenate(values)
        lo, hi = np.nanquantile(merged, [0.005, 0.995])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.nanmin(merged)), float(np.nanmax(merged))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        pad = 0.05 * (hi - lo)
        ranges[column] = (float(lo - pad), float(hi + pad))
    return ranges


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
        posterior_z = (
            posterior_samples.groupby("object_id")["z_obs"].median().to_numpy()
        )
    series = [(posterior_z[np.isfinite(posterior_z)], "posterior median", "#2a9fd6")]
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
        ax.hist(
            value, bins=bins, histtype=histtype, color=color, alpha=alpha, label=label
        )
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
    ax.hist(
        values, bins=bins, histtype="step", color="#ef476f", lw=1.5, label="amortized"
    )
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


def _plot_1d_hist(
    ax,
    values,
    value_range,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    linewidth: float = 1.0,
    fill_alpha: float = 0.0,
) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    hist_kwargs = {
        "bins": 36,
        "range": value_range,
        "density": True,
        "color": color,
        "label": label,
    }
    if fill_alpha > 0.0:
        ax.hist(
            values,
            histtype="stepfilled",
            alpha=fill_alpha,
            **hist_kwargs,
        )
    ax.hist(
        values,
        histtype="step",
        lw=linewidth,
        ls=linestyle,
        alpha=0.95,
        **hist_kwargs,
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
    linestyle: str = "-",
    fill_alpha: float = 0.0,
    alpha: float = 1.0,
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
    if fill_alpha > 0.0:
        fill_levels = [levels[0], float(np.nanmax(hist))]
        if fill_levels[0] < fill_levels[1]:
            ax.contourf(
                x_centers,
                y_centers,
                hist.T,
                levels=fill_levels,
                colors=[color],
                alpha=fill_alpha,
            )
    ax.contour(
        x_centers,
        y_centers,
        hist.T,
        levels=levels,
        colors=color,
        linewidths=linewidth,
        linestyles=linestyle,
        alpha=alpha,
    )


def _plot_2d_points(
    ax,
    x_values,
    y_values,
    x_range,
    y_range,
    *,
    color: str,
) -> None:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    finite = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & (x_values >= x_range[0])
        & (x_values <= x_range[1])
        & (y_values >= y_range[0])
        & (y_values <= y_range[1])
    )
    if finite.sum() == 0:
        return
    x = x_values[finite]
    y = y_values[finite]
    if x.size > 900:
        rng = np.random.default_rng(0)
        choice = rng.choice(x.size, size=900, replace=False)
        x = x[choice]
        y = y[choice]
    ax.scatter(
        x,
        y,
        s=3.0,
        marker=".",
        color=color,
        alpha=0.18,
        linewidths=0,
        zorder=5,
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
