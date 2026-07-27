"""Truth-column detection for Diffsky/OpenCosmo files."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .schema import (
    HLTDS_BURST_COLUMNS,
    HLTDS_DIFFMAH_COLUMNS,
    HLTDS_DIFFSTAR_COLUMNS,
    HLTDS_DUST_COLUMNS,
    HLTDS_TRUTH_COLUMNS,
)

COLUMN_SEMANTIC_CATEGORIES = (
    "truth",
    "generated_truth",
    "derived_truth",
    "diagnostic",
    "proxy",
    "unavailable",
)

DERIVED_TRUTH_COLUMNS = ("logsfr_true",)
DIAGNOSTIC_PREFIXES = ("mag_", "flux_", "fluxerr_", "mask_")
DIAGNOSTIC_COLUMNS = (
    "object_id",
    "global_object_id",
    "core_tag",
    "source_file",
    "source_row",
)
PROXY_COLUMNS = ("redshift_proxy", "photoz_proxy", "mass_proxy", "sfr_proxy")


@dataclass(frozen=True)
class TruthColumnReport:
    redshift: str | None
    stellar_mass: str | None
    ssfr: str | None
    sfr: str | None
    halo_mass: str | None
    central: str | None
    size: str | None
    diffmah_columns: tuple[str, ...]
    diffstar_columns: tuple[str, ...]
    dust_columns: tuple[str, ...]
    metallicity_columns: tuple[str, ...]
    photometry_columns: tuple[str, ...]
    sed_columns: tuple[str, ...]


@dataclass(frozen=True)
class ColumnSemantics:
    truth: tuple[str, ...]
    generated_truth: tuple[str, ...]
    derived_truth: tuple[str, ...]
    diagnostic: tuple[str, ...]
    proxy: tuple[str, ...]
    unavailable: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "truth": list(self.truth),
            "generated_truth": list(self.generated_truth),
            "derived_truth": list(self.derived_truth),
            "diagnostic": list(self.diagnostic),
            "proxy": list(self.proxy),
            "unavailable": list(self.unavailable),
        }


def get_expected_diffmah_columns() -> tuple[str, ...]:
    try:
        from diffmah import DEFAULT_MAH_PARAMS  # type: ignore

        return tuple(getattr(DEFAULT_MAH_PARAMS, "_fields", ()))
    except Exception:
        return ("early_index", "late_index", "logm0", "logmp0", "logtc", "t_peak")


def get_expected_diffstar_columns() -> tuple[str, ...]:
    try:
        from diffstar.defaults import DEFAULT_DIFFSTAR_PARAMS  # type: ignore

        fields = []
        for part in (
            DEFAULT_DIFFSTAR_PARAMS.ms_params,
            DEFAULT_DIFFSTAR_PARAMS.q_params,
        ):
            fields.extend(getattr(part, "_fields", ()))
        return tuple(fields)
    except Exception:
        return (
            "lgmcrit",
            "lgy_at_mcrit",
            "indx_lo",
            "indx_hi",
            "lg_qt",
            "qlglgdt",
            "lg_drop",
            "lg_rejuv",
        )


def detect_truth_columns(available_columns: Sequence[str]) -> TruthColumnReport:
    columns = tuple(str(column) for column in available_columns)
    lower = {column.lower(): column for column in columns}

    def first(*names: str) -> str | None:
        for name in names:
            if name.lower() in lower:
                return lower[name.lower()]
        for column in columns:
            low = column.lower().split("/")[-1]
            if low in {name.lower() for name in names}:
                return column
        return None

    diffmah_expected = set(get_expected_diffmah_columns())
    diffstar_expected = set(get_expected_diffstar_columns())

    def suffix(column: str) -> str:
        return column.split("/")[-1]

    return TruthColumnReport(
        redshift=first("redshift_true", "redshift", "redshiftHubble"),
        stellar_mass=first("logsm_obs", "stellar_mass", "um_source_galaxy_obs_sm"),
        ssfr=first("logssfr_obs", "log_ssfr"),
        sfr=first("sfr", "um_source_galaxy_obs_sfr"),
        halo_mass=first("logmp_obs", "diffmah_logmp_fit", "target_halo_mass"),
        central=first("central"),
        size=first("r50_disk", "diskHalfLightRadius"),
        diffmah_columns=tuple(
            column
            for column in columns
            if suffix(column) in diffmah_expected or "diffmah" in suffix(column).lower()
        ),
        diffstar_columns=tuple(
            column
            for column in columns
            if suffix(column) in diffstar_expected
            or "diffstar" in suffix(column).lower()
        ),
        dust_columns=tuple(
            column
            for column in columns
            if suffix(column).lower()
            in {"av", "delta", "dust_av", "dust_delta", "dust_eb"}
        ),
        metallicity_columns=tuple(
            column
            for column in columns
            if any(key in suffix(column).lower() for key in ("metal", "zmet", "lgmet"))
        ),
        photometry_columns=tuple(
            column for column in columns if _looks_like_photometry(suffix(column))
        ),
        sed_columns=tuple(
            column
            for column in columns
            if any(key in suffix(column).lower() for key in ("sed", "wave", "ssp_flux"))
        ),
    )


def classify_diffsky_columns(available_columns: Sequence[str]) -> ColumnSemantics:
    """Classify prepared Diffsky columns by scientific semantics.

    The classes describe how a column may be interpreted, not whether it is
    currently used by any optimizer. In particular, exported Diffstar/Diffmah
    latent parameters are generated-truth diagnostics of the HLTDS simulator
    and should not be silently treated as direct observational truth.
    """
    columns = tuple(str(column) for column in available_columns)
    column_set = set(columns)
    expected_truth = set(HLTDS_TRUTH_COLUMNS) | set(DERIVED_TRUTH_COLUMNS)
    expected_generated = {
        *(f"diffmah_{name}" for name in HLTDS_DIFFMAH_COLUMNS),
        *(f"diffstar_{name}" for name in HLTDS_DIFFSTAR_COLUMNS),
        *(f"dust_{name}" for name in HLTDS_DUST_COLUMNS),
        *(f"burst_{name}" for name in HLTDS_BURST_COLUMNS),
    }

    truth = []
    generated_truth = []
    derived_truth = []
    diagnostic = []
    proxy = []
    assigned: set[str] = set()
    for column in columns:
        if column in DERIVED_TRUTH_COLUMNS:
            derived_truth.append(column)
        elif column in HLTDS_TRUTH_COLUMNS:
            truth.append(column)
        elif column.startswith(("diffmah_", "diffstar_", "dust_", "burst_")):
            generated_truth.append(column)
        elif column in PROXY_COLUMNS or column.endswith("_proxy"):
            proxy.append(column)
        elif column in DIAGNOSTIC_COLUMNS or column.startswith(DIAGNOSTIC_PREFIXES):
            diagnostic.append(column)
        else:
            diagnostic.append(column)
        assigned.add(column)

    unavailable = sorted((expected_truth | expected_generated) - column_set)
    if assigned != column_set:  # pragma: no cover - defensive invariant
        missing = sorted(column_set - assigned)
        diagnostic.extend(missing)
    return ColumnSemantics(
        truth=tuple(truth),
        generated_truth=tuple(generated_truth),
        derived_truth=tuple(derived_truth),
        diagnostic=tuple(diagnostic),
        proxy=tuple(proxy),
        unavailable=tuple(unavailable),
    )


def _looks_like_photometry(name: str) -> bool:
    lower = name.lower()
    if any(part in lower for part in ("_bulge", "_disk", "_knots", "nodust", "rest")):
        return False
    return (
        lower.startswith("lsst_")
        or lower.startswith("roman_f")
        or lower.startswith("lsst_obs_")
        or lower.startswith("roman_obs_")
    )
