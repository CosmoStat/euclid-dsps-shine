"""Same-parameter Diffsky truth-to-photometry closure diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.filters import load_filters
from euclid_dsps.io import ensure_dir, write_json
from euclid_dsps.model import dynamic_model_args, load_context
from euclid_dsps.parameter_vectors import model_mags_from_theta_matrix_jax
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


def forward_closure_residuals(
    *,
    object_id: np.ndarray,
    observed_mag: np.ndarray,
    model_mag: np.ndarray,
    band_names: tuple[str, ...],
    truth_context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return long photometry rows, by-band residual metrics, and summary."""
    object_id = np.asarray(object_id)
    observed_mag = np.asarray(observed_mag, dtype=float)
    model_mag = np.asarray(model_mag, dtype=float)
    if observed_mag.shape != model_mag.shape:
        raise ValueError(
            f"observed_mag and model_mag shape mismatch: {observed_mag.shape} vs {model_mag.shape}"
        )
    rows = []
    for obj_index, oid in enumerate(object_id):
        for band_index, band in enumerate(band_names):
            obs = observed_mag[obj_index, band_index]
            mod = model_mag[obj_index, band_index]
            rows.append(
                {
                    "object_id": oid,
                    "band": band,
                    "observed_mag": float(obs),
                    "model_mag": float(mod),
                    "residual_mag": float(mod - obs),
                    "observed_flux_fnu_cgs": float(abmag_to_fnu_cgs(obs)),
                    "model_flux_fnu_cgs": float(abmag_to_fnu_cgs(mod)),
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
    frame = pd.read_parquet(dataset_path)
    if limit is not None:
        frame = frame.head(int(limit))
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
    truth_context = frame[
        [col for col in ("object_id", "redshift_true", "logsm_true") if col in frame]
    ].copy()
    photometry, by_band, summary = forward_closure_residuals(
        object_id=object_id,
        observed_mag=observed_mag,
        model_mag=model_mag,
        band_names=band_names,
        truth_context=truth_context,
    )
    parameter_sources.to_csv(out / "forward_closure_parameter_sources.csv", index=False)
    photometry.to_parquet(out / "forward_closure_photometry.parquet", index=False)
    by_band.to_csv(out / "forward_closure_residuals_by_band.csv", index=False)
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
    report = out / "forward_closure_report.md"
    _write_forward_closure_report(report, summary, by_band, parameter_sources)
    return report


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
    observed = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
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


def _write_forward_closure_report(
    path: Path,
    summary: dict[str, Any],
    by_band: pd.DataFrame,
    parameter_sources: pd.DataFrame,
) -> None:
    lines = [
        "# Diffsky True-Parameter Forward Closure",
        "",
        "This report tests `theta_true_diffsky -> DSPS/Diffstar -> photometry`.",
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
