"""Truth-parameter schemas for supervised Diffsky prior learning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.diffsky_data.schema import (
    HLTDS_BURST_COLUMNS,
    HLTDS_DIFFMAH_COLUMNS,
    HLTDS_DIFFSTAR_COLUMNS,
    HLTDS_DUST_COLUMNS,
)
from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES

SUPPORTED_TRUTH_SCHEMAS = (
    "diffsky_truth_basic",
    "diffsky_truth_extended",
    "diffsky_dsps_closure_full",
)
MISSING_POLICIES = ("reduce", "fail")


@dataclass(frozen=True)
class ParameterSpec:
    """One supervised-prior parameter and its source truth column."""

    name: str
    column: str
    semantic: str
    lower: float | None = None
    upper: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TruthSchema:
    """Resolved schema after applying available-column and missing policies."""

    name: str
    parameters: tuple[ParameterSpec, ...]
    missing_columns: tuple[str, ...]
    reduced: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": [param.to_dict() for param in self.parameters],
            "missing_columns": list(self.missing_columns),
            "reduced": bool(self.reduced),
        }


DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "z_obs": (0.0, 6.0),
    "log10_stellar_mass": (6.0, 13.5),
    "log10_ssfr_at_obs": (-14.5, -7.0),
    "log10_sfr_at_obs": (-5.0, 4.0),
    "dust_av": (0.0, 5.0),
    "dust_delta": (-2.5, 1.0),
    "diffstar_lgmcrit": (9.0, 14.5),
    "diffstar_lgy_at_mcrit": (-13.0, -8.0),
    "diffstar_indx_lo": (-2.0, 6.0),
    "diffstar_indx_hi": (-6.0, 3.0),
    "diffstar_lg_qt": (-2.0, 2.5),
    "diffstar_qlglgdt": (-4.0, 4.0),
    "diffstar_lg_drop": (-4.0, 1.0),
    "diffstar_lg_rejuv": (-4.0, 2.0),
    "diffmah_logm0": (7.0, 16.0),
    "diffmah_logmp0": (7.0, 16.0),
    "diffmah_logtc": (-2.5, 1.5),
    "diffmah_early_index": (-1.0, 6.0),
    "diffmah_late_index": (-1.0, 3.0),
    "diffmah_t_peak": (0.0, 15.0),
    "log10_stellar_metallicity": (-2.5, 0.5),
}


DIFFSKY_DSPS_CLOSURE_FULL_COLUMNS: dict[str, tuple[str, ...]] = {
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


def build_truth_schema(
    available_columns: Sequence[str],
    *,
    schema_name: str,
    missing_policy: str = "reduce",
) -> TruthSchema:
    """Resolve a supervised-prior truth schema against available columns."""
    if schema_name not in SUPPORTED_TRUTH_SCHEMAS:
        supported = ", ".join(SUPPORTED_TRUTH_SCHEMAS)
        raise ValueError(f"Unsupported truth schema {schema_name!r}; use {supported}")
    if missing_policy not in MISSING_POLICIES:
        raise ValueError("missing_policy must be 'reduce' or 'fail'")
    available = set(str(column) for column in available_columns)
    params: list[ParameterSpec] = []
    missing: list[str] = []

    def add_first(
        name: str,
        candidates: tuple[str, ...],
        *,
        semantic: str,
        required: bool,
    ) -> None:
        for column in candidates:
            if column in available:
                params.append(ParameterSpec(name=name, column=column, semantic=semantic))
                return
        missing.extend(candidates)
        if required:
            joined = ", ".join(candidates)
            raise ValueError(f"Missing required truth column for {name}: {joined}")

    if schema_name == "diffsky_dsps_closure_full":
        for name in DIFFSKY_BASIC_PARAMETER_NAMES:
            add_first(
                name,
                DIFFSKY_DSPS_CLOSURE_FULL_COLUMNS[name],
                semantic="closure_ground_truth",
                required=True,
            )
        return TruthSchema(
            name=schema_name,
            parameters=tuple(params),
            missing_columns=(),
            reduced=False,
        )

    add_first("z_obs", ("redshift_true",), semantic="truth", required=True)
    add_first("log10_stellar_mass", ("logsm_true",), semantic="truth", required=True)
    if "logssfr_true" in available:
        params.append(
            ParameterSpec(
                name="log10_ssfr_at_obs",
                column="logssfr_true",
                semantic="truth",
            )
        )
    elif "logsfr_true" in available:
        params.append(
            ParameterSpec(
                name="log10_sfr_at_obs",
                column="logsfr_true",
                semantic="derived_truth",
            )
        )
    else:
        raise ValueError(
            "Missing required SFR truth column: provide logssfr_true or logsfr_true"
        )

    add_first(
        "dust_av",
        ("dust_av", "dust_av_true"),
        semantic="generated_truth",
        required=False,
    )
    add_first(
        "dust_delta",
        ("dust_delta", "dust_delta_true"),
        semantic="generated_truth",
        required=False,
    )

    if schema_name == "diffsky_truth_extended":
        expected = [
            *(f"diffstar_{name}" for name in HLTDS_DIFFSTAR_COLUMNS),
            *(f"diffmah_{name}" for name in HLTDS_DIFFMAH_COLUMNS),
            *(f"dust_{name}" for name in HLTDS_DUST_COLUMNS),
            *(f"burst_{name}" for name in HLTDS_BURST_COLUMNS),
        ]
        existing_names = {param.name for param in params}
        for column in expected:
            if column in available:
                if column not in existing_names:
                    params.append(
                        ParameterSpec(
                            name=column,
                            column=column,
                            semantic="generated_truth",
                        )
                    )
                    existing_names.add(column)
            else:
                missing.append(column)

    if missing and missing_policy == "fail":
        joined = ", ".join(sorted(set(missing)))
        raise ValueError(f"Missing required truth columns for {schema_name}: {joined}")
    if not params:
        raise ValueError(f"No usable truth columns found for {schema_name}")
    return TruthSchema(
        name=schema_name,
        parameters=tuple(params),
        missing_columns=tuple(sorted(set(missing))),
        reduced=bool(missing),
    )


def with_parameter_bounds(
    frame: pd.DataFrame,
    schema: TruthSchema,
    configured_bounds: dict[str, Any] | None = None,
) -> TruthSchema:
    """Attach bounded-transform limits to every resolved parameter."""
    configured_bounds = dict(configured_bounds or {})
    params = []
    for param in schema.parameters:
        lower, upper = _bounds_for_parameter(frame, param, configured_bounds)
        params.append(
            ParameterSpec(
                name=param.name,
                column=param.column,
                semantic=param.semantic,
                lower=float(lower),
                upper=float(upper),
            )
        )
    return TruthSchema(
        name=schema.name,
        parameters=tuple(params),
        missing_columns=schema.missing_columns,
        reduced=schema.reduced,
    )


def _bounds_for_parameter(
    frame: pd.DataFrame,
    param: ParameterSpec,
    configured_bounds: dict[str, Any],
) -> tuple[float, float]:
    raw = configured_bounds.get(param.name, configured_bounds.get(param.column))
    if raw is not None:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"prior_learning.bounds.{param.name} must be [lower, upper]")
        lower, upper = float(raw[0]), float(raw[1])
    elif param.name in DEFAULT_BOUNDS:
        lower, upper = DEFAULT_BOUNDS[param.name]
    else:
        lower, upper = _quantile_bounds(frame[param.column])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(
            f"Invalid bounds for {param.name} from {param.column}: [{lower}, {upper}]"
        )
    return lower, upper


def _quantile_bounds(values: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        raise ValueError("Cannot infer bounds from an all-nonfinite truth column")
    lo = float(finite.quantile(0.001))
    hi = float(finite.quantile(0.999))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = float(finite.min())
        hi = float(finite.max())
    if lo == hi:
        width = max(abs(lo) * 0.05, 1.0)
        return lo - width, hi + width
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad
