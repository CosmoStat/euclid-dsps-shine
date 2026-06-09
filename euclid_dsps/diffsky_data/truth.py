"""Truth-column detection for Diffsky/OpenCosmo files."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


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
        for part in (DEFAULT_DIFFSTAR_PARAMS.ms_params, DEFAULT_DIFFSTAR_PARAMS.q_params):
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
        diffmah_columns=tuple(column for column in columns if suffix(column) in diffmah_expected or "diffmah" in suffix(column).lower()),
        diffstar_columns=tuple(column for column in columns if suffix(column) in diffstar_expected or "diffstar" in suffix(column).lower()),
        dust_columns=tuple(column for column in columns if suffix(column).lower() in {"av", "delta", "dust_av", "dust_delta", "dust_eb"}),
        metallicity_columns=tuple(column for column in columns if any(key in suffix(column).lower() for key in ("metal", "zmet", "lgmet"))),
        photometry_columns=tuple(column for column in columns if _looks_like_photometry(suffix(column))),
        sed_columns=tuple(column for column in columns if any(key in suffix(column).lower() for key in ("sed", "wave", "ssp_flux"))),
    )


def _looks_like_photometry(name: str) -> bool:
    lower = name.lower()
    if any(part in lower for part in ("_bulge", "_disk", "_knots", "nodust", "rest")):
        return False
    return lower.startswith("lsst_") or lower.startswith("roman_f") or lower.startswith("lsst_obs_") or lower.startswith("roman_obs_")
