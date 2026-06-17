"""Same-parameter Diffsky truth-to-photometry closure diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.calibration import (
    alpha_metadata,
    delta_mag_from_alpha,
    global_sed_scale_config,
)
from euclid_dsps.filters import load_filters
from euclid_dsps.io import ensure_dir, iter_catalog_batches, write_json
from euclid_dsps.model import dynamic_model_args, load_context
from euclid_dsps.parameter_vectors import (
    free_parameter_bounds_from_config,
    model_mags_from_theta_matrix_jax,
)
from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES
from euclid_dsps.photometry import abmag_to_fnu_cgs

TRUTH_PARAMETER_MAP: dict[str, tuple[str, ...]] = {
    "z_obs": ("redshift_true",),
    "log10_stellar_mass": ("logsm_true",),
    "diffstar_lgmcrit": ("diffstar_lgmcrit",),
    "diffstar_lgy_at_mcrit": ("diffstar_lgy_at_mcrit",),
    "diffstar_indx_lo": ("diffstar_indx_lo",),
    "diffstar_indx_hi": ("diffstar_indx_hi",),
    "diffstar_lg_qt": ("diffstar_lg_qt",),
    "diffstar_qlglgdt": ("diffstar_qlglgdt",),
    "diffstar_lg_drop": ("diffstar_lg_drop",),
    "diffstar_lg_rejuv": ("diffstar_lg_rejuv",),
    "diffmah_logm0": ("diffmah_logm0",),
    "diffmah_logtc": ("diffmah_logtc",),
    "diffmah_early_index": ("diffmah_early_index",),
    "diffmah_late_index": ("diffmah_late_index",),
    "diffmah_t_peak": ("diffmah_t_peak",),
    "log10_stellar_metallicity": (
        "log10_stellar_metallicity_true",
        "stellar_metallicity_true",
        "metallicity_true",
    ),
    "dust_av": ("dust_av", "dust_av_true"),
    "dust_delta": ("dust_delta", "dust_delta_true"),
}

REQUIRED_TRUTH_PARAMETERS = tuple(
    name
    for name in DIFFSKY_BASIC_PARAMETER_NAMES
    if name != "log10_stellar_metallicity"
)


def build_trueparam_theta(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    allow_partial_truth: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build a Diffsky-basic theta matrix from prepared truth columns."""
    fixed = dict((config.get("model", {}) or {}).get("fixed_parameters", {}) or {})
    default_metallicity = float(fixed.get("log10_stellar_metallicity", -0.7))
    columns = []
    metadata = []
    missing = []
    for name in DIFFSKY_BASIC_PARAMETER_NAMES:
        column = _first_existing(frame, TRUTH_PARAMETER_MAP[name])
        if column is None:
            if name == "log10_stellar_metallicity":
                values = np.full(len(frame), default_metallicity, dtype=np.float32)
                metadata.append(
                    {
                        "parameter": name,
                        "source_column": "",
                        "source_kind": "nuisance_fixed",
                        "value": default_metallicity,
                    }
                )
            elif allow_partial_truth:
                value = float(fixed.get(name, np.nan))
                values = np.full(len(frame), value, dtype=np.float32)
                metadata.append(
                    {
                        "parameter": name,
                        "source_column": "",
                        "source_kind": "nuisance_fixed",
                        "value": value,
                    }
                )
            else:
                missing.append(name)
                continue
        else:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype=np.float32
            )
            source_kind = (
                "truth"
                if name in {"z_obs", "log10_stellar_mass"}
                else "generated_truth"
            )
            metadata.append(
                {
                    "parameter": name,
                    "source_column": column,
                    "source_kind": source_kind,
                    "value": np.nan,
                }
            )
        columns.append(values)
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            "Missing Diffsky true-parameter columns for forward closure: "
            f"{joined}. Set diffsky_forward_closure.allow_partial_truth=true "
            "only for explicit diagnostic runs."
        )
    theta = np.stack(columns, axis=1).astype(np.float32)
    finite = np.isfinite(theta).all(axis=1)
    if not finite.all():
        dropped = int((~finite).sum())
        raise ValueError(f"Forward closure theta contains {dropped} non-finite rows")
    return theta, pd.DataFrame(metadata)


def build_popcosmos_proxy_theta(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build a compact PopCosmos theta matrix from available Diffsky truth proxies."""
    names = tuple(
        str(name) for name in config.get("fit", {}).get("free_parameters", {})
    )
    if not names:
        raise ValueError("popcosmos proxy closure requires fit.free_parameters")
    lower, upper = free_parameter_bounds_from_config(config, names)
    rows = []
    metadata = []
    logssfr_proxy = _logssfr_proxy(frame)
    for index, name in enumerate(names):
        values, source_column, source_kind = _popcosmos_proxy_values(
            name,
            frame,
            config,
            logssfr_proxy=logssfr_proxy,
            low=float(lower[index]),
            high=float(upper[index]),
        )
        rows.append(values.astype(np.float32))
        metadata.append(
            {
                "parameter": name,
                "source_column": source_column,
                "source_kind": source_kind,
                "value": np.nan
                if source_kind != "fixed_or_initial"
                else float(values[0]),
                "lower": float(lower[index]),
                "upper": float(upper[index]),
                "clipped_fraction": float(
                    np.mean((values <= lower[index]) | (values >= upper[index]))
                ),
            }
        )
    theta = np.stack(rows, axis=1).astype(np.float32)
    finite = np.isfinite(theta).all(axis=1)
    if not finite.all():
        dropped = int((~finite).sum())
        raise ValueError(f"PopCosmos proxy theta contains {dropped} non-finite rows")
    return theta, pd.DataFrame(metadata)


def forward_closure_residuals(
    *,
    object_id: np.ndarray,
    observed_mag: np.ndarray,
    model_mag: np.ndarray,
    band_names: tuple[str, ...],
    model_mag_raw: np.ndarray | None = None,
    log_alpha_sed: float = 0.0,
    alpha_sed: float = 1.0,
    truth_context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return long photometry rows, by-band residual metrics, and summary."""
    object_id = np.asarray(object_id)
    observed_mag = np.asarray(observed_mag, dtype=float)
    model_mag = np.asarray(model_mag, dtype=float)
    model_mag_raw = (
        np.asarray(model_mag_raw, dtype=float)
        if model_mag_raw is not None
        else np.asarray(model_mag, dtype=float)
    )
    if observed_mag.shape != model_mag.shape:
        raise ValueError(
            f"observed_mag and model_mag shape mismatch: {observed_mag.shape} vs {model_mag.shape}"
        )
    if model_mag_raw.shape != model_mag.shape:
        raise ValueError(
            f"model_mag_raw and model_mag shape mismatch: {model_mag_raw.shape} vs {model_mag.shape}"
        )
    rows = []
    for obj_index, oid in enumerate(object_id):
        for band_index, band in enumerate(band_names):
            obs = observed_mag[obj_index, band_index]
            mod = model_mag[obj_index, band_index]
            raw = model_mag_raw[obj_index, band_index]
            model_flux_raw = float(abmag_to_fnu_cgs(raw))
            model_flux_scaled = float(abmag_to_fnu_cgs(mod))
            rows.append(
                {
                    "object_id": oid,
                    "band": band,
                    "observed_mag": float(obs),
                    "model_mag": float(mod),
                    "model_mag_raw": float(raw),
                    "model_mag_scaled": float(mod),
                    "residual_mag": float(mod - obs),
                    "residual_mag_raw": float(raw - obs),
                    "observed_flux_fnu_cgs": float(abmag_to_fnu_cgs(obs)),
                    "model_flux_fnu_cgs": model_flux_scaled,
                    "model_flux_raw_fnu_cgs": model_flux_raw,
                    "model_flux_scaled_fnu_cgs": model_flux_scaled,
                    "log_alpha_sed": float(log_alpha_sed),
                    "alpha_sed": float(alpha_sed),
                    "delta_mag_global": float(delta_mag_from_alpha(alpha_sed)),
                }
            )
    photometry = pd.DataFrame(rows)
    by_band = (
        photometry.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["residual_mag"])
        .groupby("band", sort=False)["residual_mag"]
        .agg(
            n="count",
            median_residual_mag="median",
            mean_residual_mag="mean",
            rms_residual_mag=lambda x: float(np.sqrt(np.mean(np.square(x)))),
        )
        .reset_index()
    )
    summary: dict[str, Any] = {
        "n_objects": int(len(object_id)),
        "n_bands": int(len(band_names)),
        "log_alpha_sed": float(log_alpha_sed),
        "alpha_sed": float(alpha_sed),
        "delta_mag_global": float(delta_mag_from_alpha(alpha_sed)),
        "median_abs_residual_mag": _nan_stat(
            np.nanmedian, np.abs(photometry["residual_mag"].to_numpy(dtype=float))
        ),
        "rms_residual_mag": _nan_stat(
            lambda x: np.sqrt(np.nanmean(np.square(x))),
            photometry["residual_mag"].to_numpy(dtype=float),
        ),
    }
    if truth_context is not None and not truth_context.empty:
        for column, bins in (
            ("redshift_true", [0.0, 0.3, 0.6, 0.9, 1.2, 2.0, 6.0]),
            ("logsm_true", [6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0]),
        ):
            if column in truth_context:
                summary[f"{column}_bins"] = _binned_residual_summary(
                    photometry,
                    truth_context[["object_id", column]],
                    column,
                    np.asarray(bins, dtype=float),
                )
    return photometry, by_band, summary


def run_diffsky_forward_closure(
    config: dict[str, Any],
    *,
    dataset_path: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    batch_size: int = 64,
) -> Path:
    """Run true-parameter Diffsky forward closure and write artifacts."""
    if str((config.get("model", {}) or {}).get("sfh_model")) != "diffsky_basic":
        raise ValueError(
            "diffsky-forward-closure requires model.sfh_model='diffsky_basic'."
        )
    out = ensure_dir(out_dir)
    dataset_path = Path(dataset_path)
    chunks = list(
        iter_catalog_batches(dataset_path, batch_size=int(batch_size), limit=limit)
    )
    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if frame.empty:
        raise ValueError("No rows selected for forward closure")
    allow_partial = bool(
        (config.get("diffsky_forward_closure", {}) or {}).get("allow_partial_truth")
    )
    theta, parameter_sources = build_trueparam_theta(
        frame,
        config,
        allow_partial_truth=allow_partial,
    )
    band_names, observed_mag = _observed_magnitudes(frame, config)
    object_id = (
        frame["object_id"].to_numpy()
        if "object_id" in frame
        else np.arange(len(frame), dtype=np.int64)
    )
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    model_args = dynamic_model_args(context)
    model_mags = []
    for start in range(0, len(theta), int(batch_size)):
        chunk = jnp.asarray(theta[start : start + int(batch_size)], dtype=jnp.float32)
        mags = model_mags_from_theta_matrix_jax(
            context,
            model_args,
            chunk,
            DIFFSKY_BASIC_PARAMETER_NAMES,
        )
        model_mags.append(np.asarray(jax.device_get(mags), dtype=float))
    model_mag = np.concatenate(model_mags, axis=0)
    scale_cfg = global_sed_scale_config(
        {"calibration": config.get("calibration", {}) or {}}
    )
    alpha_fit = _closure_alpha_fit(scale_cfg, observed_mag, model_mag)
    model_mag_scaled = model_mag + float(alpha_fit["delta_mag_global"])
    truth_context = frame[
        [col for col in ("object_id", "redshift_true", "logsm_true") if col in frame]
    ].copy()
    _, before_by_band, before_summary = forward_closure_residuals(
        object_id=object_id,
        observed_mag=observed_mag,
        model_mag=model_mag,
        band_names=band_names,
        log_alpha_sed=0.0,
        alpha_sed=1.0,
        truth_context=truth_context,
    )
    photometry, by_band, after_summary = forward_closure_residuals(
        object_id=object_id,
        observed_mag=observed_mag,
        model_mag=model_mag_scaled,
        band_names=band_names,
        model_mag_raw=model_mag,
        log_alpha_sed=float(alpha_fit["log_alpha_sed"]),
        alpha_sed=float(alpha_fit["alpha_sed"]),
        truth_context=truth_context,
    )
    parameter_sources.to_csv(out / "forward_closure_parameter_sources.csv", index=False)
    photometry.to_parquet(out / "forward_closure_photometry.parquet", index=False)
    before_by_band.to_csv(out / "residuals_by_band_before_alpha.csv", index=False)
    by_band.to_csv(out / "residuals_by_band_after_alpha.csv", index=False)
    by_band.to_csv(out / "forward_closure_residuals_by_band.csv", index=False)
    write_json(out / "alpha_sed_fit.json", alpha_fit)
    summary = {
        "before_alpha": before_summary,
        "after_alpha": after_summary,
        "global_sed_scale": alpha_fit,
        "closure_gate": _closure_gate(alpha_fit, after_summary),
    }
    summary.update(after_summary)
    summary.update(
        {
            "dataset_path": str(dataset_path),
            "sfh_model": "diffsky_basic",
            "allow_partial_truth": allow_partial,
            "nuisance_fixed": parameter_sources[
                parameter_sources["source_kind"] == "nuisance_fixed"
            ].to_dict(orient="records"),
        }
    )
    write_json(out / "forward_closure_summary.json", summary)
    write_json(out / "closure_gate.json", summary["closure_gate"])
    report = out / "forward_closure_report.md"
    _write_forward_closure_report(report, summary, by_band, parameter_sources)
    return report


def run_popcosmos_proxy_truth_closure(
    config: dict[str, Any],
    *,
    dataset_path: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    batch_size: int = 64,
) -> Path:
    """Run proxy-truth closure for the actual amortized PopCosmos decoder."""
    if str((config.get("model", {}) or {}).get("sfh_model")) != "popcosmos_bins":
        raise ValueError(
            "diffsky-popcosmos-proxy-closure requires model.sfh_model='popcosmos_bins'."
        )
    out = ensure_dir(out_dir)
    dataset_path = Path(dataset_path)
    frame = pd.read_parquet(dataset_path)
    if limit is not None:
        frame = frame.head(int(limit))
    if frame.empty:
        raise ValueError("No rows selected for PopCosmos proxy closure")
    theta, parameter_sources = build_popcosmos_proxy_theta(frame, config)
    parameter_names = tuple(str(name) for name in config["fit"]["free_parameters"])
    band_names, observed_mag = _observed_magnitudes(frame, config)
    object_id = (
        frame["object_id"].to_numpy()
        if "object_id" in frame
        else np.arange(len(frame), dtype=np.int64)
    )
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    model_args = dynamic_model_args(context)
    model_mags = []
    for start in range(0, len(theta), int(batch_size)):
        chunk = jnp.asarray(theta[start : start + int(batch_size)], dtype=jnp.float32)
        mags = model_mags_from_theta_matrix_jax(
            context,
            model_args,
            chunk,
            parameter_names,
        )
        model_mags.append(np.asarray(jax.device_get(mags), dtype=float))
    model_mag = np.concatenate(model_mags, axis=0)
    scale_cfg = global_sed_scale_config(
        {"calibration": config.get("calibration", {}) or {}}
    )
    alpha_fit = _closure_alpha_fit(scale_cfg, observed_mag, model_mag)
    model_mag_scaled = model_mag + float(alpha_fit["delta_mag_global"])
    truth_context = frame[
        [
            col
            for col in (
                "object_id",
                "redshift_true",
                "logsm_true",
                "dust_av",
                "dust_delta",
            )
            if col in frame
        ]
    ].copy()
    _, before_by_band, before_summary = forward_closure_residuals(
        object_id=object_id,
        observed_mag=observed_mag,
        model_mag=model_mag,
        band_names=band_names,
        log_alpha_sed=0.0,
        alpha_sed=1.0,
        truth_context=truth_context,
    )
    photometry, by_band, after_summary = forward_closure_residuals(
        object_id=object_id,
        observed_mag=observed_mag,
        model_mag=model_mag_scaled,
        band_names=band_names,
        model_mag_raw=model_mag,
        log_alpha_sed=float(alpha_fit["log_alpha_sed"]),
        alpha_sed=float(alpha_fit["alpha_sed"]),
        truth_context=truth_context,
    )
    parameter_sources.to_csv(
        out / "popcosmos_proxy_closure_parameter_sources.csv",
        index=False,
    )
    photometry.to_parquet(
        out / "popcosmos_proxy_closure_photometry.parquet", index=False
    )
    before_by_band.to_csv(
        out / "popcosmos_proxy_residuals_by_band_before_alpha.csv",
        index=False,
    )
    by_band.to_csv(
        out / "popcosmos_proxy_residuals_by_band_after_alpha.csv", index=False
    )
    by_band.to_csv(out / "popcosmos_proxy_closure_residuals_by_band.csv", index=False)
    write_json(out / "alpha_sed_fit.json", alpha_fit)
    summary = {
        "before_alpha": before_summary,
        "after_alpha": after_summary,
        "global_sed_scale": alpha_fit,
        "closure_gate": _closure_gate(alpha_fit, after_summary),
    }
    summary.update(after_summary)
    summary.update(
        {
            "dataset_path": str(dataset_path),
            "sfh_model": "popcosmos_bins",
            "parameter_names": list(parameter_names),
            "nuisance_fixed": parameter_sources[
                parameter_sources["source_kind"] == "fixed_or_initial"
            ].to_dict(orient="records"),
            "proxy_parameters": parameter_sources[
                parameter_sources["source_kind"].str.contains("proxy", na=False)
            ].to_dict(orient="records"),
        }
    )
    write_json(out / "popcosmos_proxy_closure_summary.json", summary)
    write_json(out / "closure_gate.json", summary["closure_gate"])
    _write_closure_plots(
        photometry,
        by_band,
        truth_context,
        out,
        prefix="popcosmos_proxy",
    )
    report = out / "popcosmos_proxy_closure_report.md"
    _write_forward_closure_report(report, summary, by_band, parameter_sources)
    return report


def _closure_alpha_fit(
    scale_cfg,
    observed_mag: np.ndarray,
    model_mag_raw: np.ndarray,
) -> dict[str, Any]:
    """Resolve the closure global SED scale from config and raw residuals."""
    if not scale_cfg.enabled or scale_cfg.mode == "disabled":
        log_alpha = 0.0
        fit_method = "disabled"
    elif scale_cfg.mode == "fixed":
        log_alpha = float(scale_cfg.initial_log_alpha)
        fit_method = "fixed_config_initial_log_alpha"
    elif scale_cfg.mode == "fit_global":
        residual = np.asarray(model_mag_raw, dtype=float) - np.asarray(
            observed_mag, dtype=float
        )
        finite = residual[np.isfinite(residual)]
        delta_mag = -float(np.median(finite)) if finite.size else 0.0
        log_alpha = -delta_mag * np.log(10.0) / 2.5
        fit_method = "median_residual_mag"
    else:
        raise ValueError(
            "diffsky-forward-closure supports calibration.global_sed_scale.mode "
            "'disabled', 'fixed', or 'fit_global'."
        )
    meta = alpha_metadata(log_alpha, scale_cfg.prior_sigma_log_alpha)
    return {
        **meta,
        "enabled": bool(scale_cfg.enabled),
        "mode": scale_cfg.mode,
        "fit_method": fit_method,
        "prior_sigma_log_alpha": float(scale_cfg.prior_sigma_log_alpha),
        "warning": (
            "Large global SED scale; check units/mass normalization/SSP scale."
            if bool(meta["large_scale_warning"])
            else ""
        ),
    }


def _logssfr_proxy(frame: pd.DataFrame) -> np.ndarray | None:
    if "logssfr_true" in frame:
        values = pd.to_numeric(frame["logssfr_true"], errors="coerce").to_numpy(float)
        return values
    if {"logsfr_true", "logsm_true"} <= set(frame.columns):
        logsfr = pd.to_numeric(frame["logsfr_true"], errors="coerce").to_numpy(float)
        logm = pd.to_numeric(frame["logsm_true"], errors="coerce").to_numpy(float)
        return logsfr - logm
    return None


def _popcosmos_proxy_values(
    name: str,
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    logssfr_proxy: np.ndarray | None,
    low: float,
    high: float,
) -> tuple[np.ndarray, str, str]:
    n_rows = len(frame)
    if name == "z_obs" and "redshift_true" in frame:
        return (
            _clip_values(frame["redshift_true"].to_numpy(float), low, high),
            "redshift_true",
            "truth",
        )
    if name == "log10_stellar_mass" and "logsm_true" in frame:
        return (
            _clip_values(frame["logsm_true"].to_numpy(float), low, high),
            "logsm_true",
            "truth",
        )
    if name.startswith("dlog10_sfr_") and logssfr_proxy is not None:
        reference = float(
            (config.get("model", {}) or {}).get("truth_basic_ssfr_reference", -10.0)
        )
        slope = (np.asarray(logssfr_proxy, dtype=float) - reference) / 6.0
        return (
            _clip_values(slope, low, high),
            "logssfr_true/logsfr_true-logsm_true",
            "truth_proxy",
        )
    if name == "tau2" and "dust_av" in frame:
        return (
            _clip_values(frame["dust_av"].to_numpy(float) / 1.086, low, high),
            "dust_av/1.086",
            "truth_proxy",
        )
    if name == "dust_index_n" and "dust_delta" in frame:
        return (
            _clip_values(frame["dust_delta"].to_numpy(float), low, high),
            "dust_delta",
            "truth_proxy",
        )
    value = _free_parameter_initial_or_fixed(config, name)
    return (
        np.full(n_rows, np.clip(value, low, high), dtype=np.float32),
        "",
        "fixed_or_initial",
    )


def _free_parameter_initial_or_fixed(config: dict[str, Any], name: str) -> float:
    free = (config.get("fit", {}) or {}).get("free_parameters", {}) or {}
    if name in free and "initial" in free[name]:
        raw = free[name]["initial"]
        if isinstance(raw, (int, float)) and np.isfinite(raw):
            return float(raw)
    fixed = (config.get("model", {}) or {}).get("fixed_parameters", {}) or {}
    raw = fixed.get(name, 0.0)
    return float(raw) if isinstance(raw, (int, float)) and np.isfinite(raw) else 0.0


def _clip_values(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), low, high).astype(np.float32)


def _closure_gate(
    alpha_fit: dict[str, Any],
    after_summary: dict[str, Any],
) -> dict[str, Any]:
    rms = after_summary.get("rms_residual_mag")
    finite_rms = rms is not None and np.isfinite(float(rms))
    return {
        "status": "INSPECT_RESIDUALS" if finite_rms else "NOT_READY",
        "finite_after_alpha_rms": bool(finite_rms),
        "large_global_sed_scale_warning": bool(
            alpha_fit.get("large_scale_warning", False)
        ),
        "warning": alpha_fit.get("warning", ""),
    }


def _observed_magnitudes(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[tuple[str, ...], np.ndarray]:
    names = []
    columns = []
    for band in config["bands"]:
        name = str(band["name"])
        column = str(band["column"])
        if str(band.get("units")) == "abmag":
            mag_col = column
        elif column.startswith("flux_"):
            mag_col = "mag_" + column.removeprefix("flux_")
        else:
            mag_col = column
        if mag_col not in frame:
            raise ValueError(f"Observed magnitude column missing: {mag_col}")
        names.append(name)
        columns.append(mag_col)
    observed = (
        frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    )
    return tuple(names), observed


def _first_existing(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame:
            return column
    return None


def _nan_stat(fn, values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(fn(values))


def _binned_residual_summary(
    photometry: pd.DataFrame,
    truth: pd.DataFrame,
    column: str,
    bins: np.ndarray,
) -> list[dict[str, Any]]:
    joined = photometry.merge(truth, on="object_id", how="inner")
    if joined.empty:
        return []
    values = pd.to_numeric(joined[column], errors="coerce").to_numpy(dtype=float)
    residual = joined["residual_mag"].to_numpy(dtype=float)
    rows = []
    for idx in range(len(bins) - 1):
        lo, hi = float(bins[idx]), float(bins[idx + 1])
        mask = (values >= lo) & (values < hi) & np.isfinite(residual)
        if not mask.any():
            continue
        rows.append(
            {
                "bin": f"{lo:g}-{hi:g}",
                "min": lo,
                "max": hi,
                "n": int(mask.sum()),
                "median_residual_mag": float(np.median(residual[mask])),
                "rms_residual_mag": float(np.sqrt(np.mean(residual[mask] ** 2))),
            }
        )
    return rows


def _write_closure_plots(
    photometry: pd.DataFrame,
    by_band: pd.DataFrame,
    truth_context: pd.DataFrame,
    out: Path,
    *,
    prefix: str,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths: list[Path] = []
    if not by_band.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        ax.bar(
            by_band["band"].astype(str), by_band["median_residual_mag"].astype(float)
        )
        ax.axhline(0.0, color="black", lw=1.0)
        ax.set_ylabel("median model - observed mag")
        ax.set_title("Closure residual by band")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = out / f"{prefix}_residual_by_band.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    if not photometry.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        grouped = [
            group["residual_mag"].to_numpy(dtype=float)
            for _band, group in photometry.groupby("band", sort=False)
        ]
        labels = [str(band) for band, _group in photometry.groupby("band", sort=False)]
        ax.boxplot(grouped, labels=labels, showfliers=False)
        ax.axhline(0.0, color="black", lw=1.0)
        ax.set_ylabel("model - observed mag")
        ax.set_title("Closure residual distribution")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = out / f"{prefix}_residual_boxplot.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    joined = photometry.merge(truth_context, on="object_id", how="left")
    for column in ("redshift_true", "logsm_true", "dust_av", "dust_delta"):
        if column not in joined:
            continue
        x = pd.to_numeric(joined[column], errors="coerce").to_numpy(float)
        y = pd.to_numeric(joined["residual_mag"], errors="coerce").to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            continue
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        ax.scatter(x[finite], y[finite], s=4, alpha=0.25)
        ax.axhline(0.0, color="black", lw=1.0)
        ax.set_xlabel(column)
        ax.set_ylabel("model - observed mag")
        ax.set_title(f"Closure residual vs {column}")
        fig.tight_layout()
        path = out / f"{prefix}_residual_vs_{column}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_forward_closure_report(
    path: Path,
    summary: dict[str, Any],
    by_band: pd.DataFrame,
    parameter_sources: pd.DataFrame,
) -> None:
    sfh_model = str(summary.get("sfh_model") or "unknown")
    title = (
        "Diffsky True-Parameter Forward Closure"
        if sfh_model == "diffsky_basic"
        else "PopCosmos Proxy-Truth Forward Closure"
    )
    lines = [
        f"# {title}",
        "",
        f"This report tests `{sfh_model}` parameters through DSPS photometry.",
        "It is a simulator closure diagnostic, not an optimizer result.",
        "",
        "## Summary",
        "",
    ]
    for key in (
        "dataset_path",
        "sfh_model",
        "n_objects",
        "n_bands",
        "median_abs_residual_mag",
        "rms_residual_mag",
        "allow_partial_truth",
    ):
        lines.append(f"- `{key}`: {summary.get(key)}")
    alpha = dict(summary.get("global_sed_scale", {}) or {})
    lines.extend(
        [
            "",
            "## Global SED Scale",
            "",
            f"- `mode`: {alpha.get('mode')}",
            f"- `alpha_sed`: {alpha.get('alpha_sed')}",
            f"- `log_alpha_sed`: {alpha.get('log_alpha_sed')}",
            f"- `delta_mag_global`: {alpha.get('delta_mag_global')}",
            f"- `alpha_prior_penalty`: {alpha.get('alpha_prior_penalty')}",
            f"- `warning`: {alpha.get('warning') or 'none'}",
            "",
            "A large global scale is a normalization or mass-scale diagnostic; "
            "it does not correct color-dependent residuals.",
        ]
    )
    lines.extend(["", "## Residuals By Band", "", _markdown_table(by_band), ""])
    lines.extend(
        [
            "## Parameter Sources",
            "",
            _markdown_table(parameter_sources),
            "",
            "## Interpretation",
            "",
            "If truth parameters do not reproduce HLTDS magnitudes here, "
            "later amortized posterior results must not be interpreted as "
            "physical recoveries.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for col in frame.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}" if np.isfinite(value) else "")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
