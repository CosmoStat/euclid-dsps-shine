"""Selection gates for synthetic Diffsky DSPS closure catalogs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SELECTION: dict[str, Any] = {
    "min_logsm": None,
    "max_logsm": None,
    "require_metallicity_unclipped": False,
    "max_metallicity_clipped_fraction": None,
    "snr_threshold": 5.0,
    "min_true_snr_bands": 0,
    "min_observed_snr_bands": 0,
    "photometric_oversample_factor": 1.0,
}


def normalize_selection(selection: dict[str, Any] | None) -> dict[str, Any]:
    """Return a complete, typed selection configuration."""
    out = dict(DEFAULT_SELECTION)
    out.update(dict(selection or {}))
    for key in ("min_logsm", "max_logsm", "max_metallicity_clipped_fraction"):
        if out.get(key) is not None:
            out[key] = float(out[key])
    out["require_metallicity_unclipped"] = bool(out["require_metallicity_unclipped"])
    out["snr_threshold"] = float(out["snr_threshold"])
    out["min_true_snr_bands"] = int(out["min_true_snr_bands"])
    out["min_observed_snr_bands"] = int(out["min_observed_snr_bands"])
    out["photometric_oversample_factor"] = max(
        1.0, float(out["photometric_oversample_factor"])
    )
    return out


def photometric_selection_enabled(selection: dict[str, Any] | None) -> bool:
    """Return True when S/N cuts must be applied after DSPS photometry."""
    cfg = normalize_selection(selection)
    return bool(cfg["min_true_snr_bands"] > 0 or cfg["min_observed_snr_bands"] > 0)


def apply_proposal_selection(
    frame: pd.DataFrame,
    selection: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply compact truth-space proposal cuts and return diagnostics."""
    cfg = normalize_selection(selection)
    selected = pd.Series(True, index=frame.index)
    summary: dict[str, Any] = {
        "input_size": int(len(frame)),
        "cuts": {},
    }
    if "galaxy_weight" in frame:
        weights = pd.to_numeric(frame["galaxy_weight"], errors="coerce").to_numpy(float)
        mask = np.isfinite(weights) & (weights > 0.0)
        selected &= mask
        summary["cuts"]["positive_finite_weight"] = _cut_summary(mask)
    if cfg["min_logsm"] is not None:
        values = pd.to_numeric(frame["logsm_true"], errors="coerce").to_numpy(float)
        mask = np.isfinite(values) & (values >= float(cfg["min_logsm"]))
        selected &= mask
        summary["cuts"]["min_logsm"] = {
            **_cut_summary(mask),
            "threshold": float(cfg["min_logsm"]),
        }
    if cfg["max_logsm"] is not None:
        values = pd.to_numeric(frame["logsm_true"], errors="coerce").to_numpy(float)
        mask = np.isfinite(values) & (values <= float(cfg["max_logsm"]))
        selected &= mask
        summary["cuts"]["max_logsm"] = {
            **_cut_summary(mask),
            "threshold": float(cfg["max_logsm"]),
        }
    if bool(cfg["require_metallicity_unclipped"]):
        if "metallicity_clipped" not in frame:
            raise ValueError(
                "selection.require_metallicity_unclipped requires "
                "metallicity_clipped in proposals"
            )
        clipped = frame["metallicity_clipped"].astype(bool).to_numpy()
        mask = ~clipped
        selected &= mask
        summary["cuts"]["require_metallicity_unclipped"] = _cut_summary(mask)
    selected_frame = frame.loc[selected.to_numpy()].reset_index(drop=True)
    summary["selected_size"] = int(len(selected_frame))
    summary["selected_fraction"] = (
        float(len(selected_frame) / len(frame)) if len(frame) else 0.0
    )
    return selected_frame, summary


def append_snr_selection_columns(
    frame: pd.DataFrame,
    bands: list[str],
    selection: dict[str, Any] | None,
) -> pd.DataFrame:
    """Add compact S/N-count diagnostics used by the photometric selection."""
    cfg = normalize_selection(selection)
    out = frame.copy()
    threshold = float(cfg["snr_threshold"])
    true_counts = _snr_counts(out, bands, threshold=threshold, observed=False)
    observed_counts = _snr_counts(out, bands, threshold=threshold, observed=True)
    out["snr_selection_threshold"] = threshold
    out["n_bands_true_snr_ge_threshold"] = true_counts
    out["n_bands_observed_snr_ge_threshold"] = observed_counts
    return out


def apply_photometric_selection(
    frame: pd.DataFrame,
    bands: list[str],
    selection: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply S/N gates to DSPS-photometered candidate rows."""
    cfg = normalize_selection(selection)
    with_counts = append_snr_selection_columns(frame, bands, cfg)
    selected = pd.Series(True, index=with_counts.index)
    summary: dict[str, Any] = {
        "input_size": int(len(with_counts)),
        "snr_threshold": float(cfg["snr_threshold"]),
        "min_true_snr_bands": int(cfg["min_true_snr_bands"]),
        "min_observed_snr_bands": int(cfg["min_observed_snr_bands"]),
        "cuts": {},
    }
    if int(cfg["min_true_snr_bands"]) > 0:
        counts = with_counts["n_bands_true_snr_ge_threshold"].to_numpy(int)
        mask = counts >= int(cfg["min_true_snr_bands"])
        selected &= mask
        summary["cuts"]["min_true_snr_bands"] = {
            **_cut_summary(mask),
            "threshold": int(cfg["min_true_snr_bands"]),
        }
    if int(cfg["min_observed_snr_bands"]) > 0:
        counts = with_counts["n_bands_observed_snr_ge_threshold"].to_numpy(int)
        mask = counts >= int(cfg["min_observed_snr_bands"])
        selected &= mask
        summary["cuts"]["min_observed_snr_bands"] = {
            **_cut_summary(mask),
            "threshold": int(cfg["min_observed_snr_bands"]),
        }
    selected_frame = with_counts.loc[selected.to_numpy()].reset_index(drop=True)
    summary["selected_size"] = int(len(selected_frame))
    summary["selected_fraction"] = (
        float(len(selected_frame) / len(with_counts)) if len(with_counts) else 0.0
    )
    if len(with_counts):
        summary["true_snr_band_count_quantiles"] = _quantiles(
            with_counts["n_bands_true_snr_ge_threshold"].to_numpy(float)
        )
        summary["observed_snr_band_count_quantiles"] = _quantiles(
            with_counts["n_bands_observed_snr_ge_threshold"].to_numpy(float)
        )
    return selected_frame, summary


def _snr_counts(
    frame: pd.DataFrame,
    bands: list[str],
    *,
    threshold: float,
    observed: bool,
) -> np.ndarray:
    counts = np.zeros(len(frame), dtype=np.int16)
    prefix = "flux" if observed else "flux_true"
    for band in bands:
        flux_col = f"{prefix}_{band}"
        err_col = f"fluxerr_{band}"
        if flux_col not in frame or err_col not in frame:
            raise ValueError(
                f"Photometric selection requires {flux_col} and {err_col}"
            )
        flux = pd.to_numeric(frame[flux_col], errors="coerce").to_numpy(float)
        err = pd.to_numeric(frame[err_col], errors="coerce").to_numpy(float)
        if observed:
            snr = flux / err
        else:
            snr = flux / err
        counts += (np.isfinite(snr) & (snr >= float(threshold))).astype(np.int16)
    return counts


def _cut_summary(mask: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(mask, dtype=bool)
    return {
        "kept": int(np.count_nonzero(valid)),
        "rejected": int(valid.size - np.count_nonzero(valid)),
        "kept_fraction": float(np.count_nonzero(valid) / valid.size)
        if valid.size
        else 0.0,
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {}
    return {
        "min": float(np.min(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "q95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }
