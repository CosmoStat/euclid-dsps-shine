"""Canonical theta construction from synthetic FENIKS truth columns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES

TRUTH_PARAMETER_MAP: dict[str, tuple[str, ...]] = {
    "z_obs": ("redshift_true",),
    "log10_stellar_mass": ("logsm_true",),
    "diffstar_lgmcrit": ("diffstar_lgmcrit_true", "diffstar_lgmcrit"),
    "diffstar_lgy_at_mcrit": (
        "diffstar_lgy_at_mcrit_true",
        "diffstar_lgy_at_mcrit",
    ),
    "diffstar_indx_lo": ("diffstar_indx_lo_true", "diffstar_indx_lo"),
    "diffstar_indx_hi": ("diffstar_indx_hi_true", "diffstar_indx_hi"),
    "diffstar_lg_qt": ("diffstar_lg_qt_true", "diffstar_lg_qt"),
    "diffstar_qlglgdt": ("diffstar_qlglgdt_true", "diffstar_qlglgdt"),
    "diffstar_lg_drop": ("diffstar_lg_drop_true", "diffstar_lg_drop"),
    "diffstar_lg_rejuv": ("diffstar_lg_rejuv_true", "diffstar_lg_rejuv"),
    "diffmah_logm0": ("diffmah_logm0_true", "diffmah_logm0"),
    "diffmah_logtc": ("diffmah_logtc_true", "diffmah_logtc"),
    "diffmah_early_index": ("diffmah_early_index_true", "diffmah_early_index"),
    "diffmah_late_index": ("diffmah_late_index_true", "diffmah_late_index"),
    "diffmah_t_peak": ("diffmah_t_peak_true", "diffmah_t_peak"),
    "log10_stellar_metallicity": ("log10_stellar_metallicity_true",),
    "dust_av": ("dust_av_true", "dust_av"),
    "dust_delta": ("dust_delta_true", "dust_delta"),
}

REQUIRED_TRUTH_PARAMETERS = tuple(DIFFSKY_BASIC_PARAMETER_NAMES)


def build_trueparam_theta(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    allow_partial_truth: bool = False,
    truth_schema: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build the canonical Diffsky-basic theta matrix from truth columns."""
    fixed = dict((config.get("model", {}) or {}).get("fixed_parameters", {}) or {})
    default_metallicity = float(fixed.get("log10_stellar_metallicity", -0.7))
    strict_full_truth = str(truth_schema or _truth_schema(config)) == (
        "diffsky_dsps_closure_full"
    )
    columns = []
    metadata = []
    missing = []
    for name in DIFFSKY_BASIC_PARAMETER_NAMES:
        column = _first_existing(frame, TRUTH_PARAMETER_MAP[name])
        if column is None:
            if name == "log10_stellar_metallicity" and not strict_full_truth:
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
                "closure_ground_truth"
                if strict_full_truth
                else (
                    "truth"
                    if name in {"z_obs", "log10_stellar_mass"}
                    else "generated_truth"
                )
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
            "Missing Diffsky true-parameter columns for FENIKS theta: "
            f"{joined}. The diffsky_dsps_closure_full schema requires all "
            "18 truths and never uses fixed-metallicity fallback."
        )
    theta = np.stack(columns, axis=1).astype(np.float32)
    finite = np.isfinite(theta).all(axis=1)
    if not finite.all():
        dropped = int((~finite).sum())
        raise ValueError(f"FENIKS theta contains {dropped} non-finite rows")
    return theta, pd.DataFrame(metadata)


def _truth_schema(config: dict[str, Any]) -> str | None:
    truth = config.get("truth", {}) or {}
    return truth.get("schema")


def _first_existing(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame:
            return column
    return None
