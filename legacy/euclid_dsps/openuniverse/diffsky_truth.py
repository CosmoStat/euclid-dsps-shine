"""OpenUniverse basic truths and optional Diffsky/Diffstar truth interface."""

from __future__ import annotations

import importlib.util
import json
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .schema import OU_TRUTH_COLUMNS


@dataclass(frozen=True)
class TruthSchema:
    available_columns: tuple[str, ...]
    redshift_column: str | None
    stellar_mass_column: str | None
    sfr_columns: tuple[str, ...]
    sfh_columns: tuple[str, ...]
    dust_columns: tuple[str, ...]
    metallicity_columns: tuple[str, ...]
    halo_columns: tuple[str, ...]


def infer_truth_schema(df: pd.DataFrame) -> TruthSchema:
    """Infer available OpenUniverse/Diffsky truth-like columns conservatively."""
    columns = tuple(str(column) for column in df.columns)
    lower = {column: column.lower() for column in columns}
    redshift_column = _first_existing(
        columns,
        ("redshift", "redshiftHubble", "z", "z_true", "z_obs"),
    )
    stellar_mass_column = _first_existing(
        columns,
        ("stellar_mass", OU_TRUTH_COLUMNS["stellar_mass"], "log_stellar_mass"),
    )
    return TruthSchema(
        available_columns=columns,
        redshift_column=redshift_column,
        stellar_mass_column=stellar_mass_column,
        sfr_columns=_matching_columns(lower, ("sfr", "ssfr")),
        sfh_columns=_matching_columns(lower, ("sfh", "diffstar", "mah")),
        dust_columns=_matching_columns(lower, ("dust", "atten", "av", "mw_av")),
        metallicity_columns=_matching_columns(lower, ("metal", "metalli", "lgmet")),
        halo_columns=_matching_columns(
            lower,
            ("halo", "mah", "shear", "convergence", "mvir", "rvir"),
        ),
    )


def extract_basic_truth_table(df: pd.DataFrame) -> pd.DataFrame:
    """Extract directly available public OpenUniverse truth columns."""
    schema = infer_truth_schema(df)
    out = pd.DataFrame()
    if "galaxy_id" in df:
        out["galaxy_id"] = df["galaxy_id"].to_numpy()
    if schema.redshift_column is not None:
        out["redshift_truth"] = df[schema.redshift_column].to_numpy()
    if "redshiftHubble" in df:
        out["redshift_hubble_truth"] = df["redshiftHubble"].to_numpy()
    if schema.stellar_mass_column is not None:
        out["stellar_mass_truth"] = df[schema.stellar_mass_column].to_numpy()
    out.attrs["truth_levels"] = {
        "redshift_truth": "truth" if "redshift_truth" in out else "unavailable",
        "redshift_hubble_truth": (
            "truth" if "redshift_hubble_truth" in out else "unavailable"
        ),
        "stellar_mass_truth": (
            "truth" if "stellar_mass_truth" in out else "unavailable"
        ),
        "sfh": "unavailable",
        "dust": "unavailable",
        "metallicity": "unavailable",
        "halo": "unavailable",
    }
    return out


def extract_extended_diffsky_truth(
    input_root: str | Path,
    hpix_ids: Sequence[int],
    output_path: str | Path,
    *,
    require_diffsky: bool = False,
) -> pd.DataFrame:
    """Placeholder interface for future optional Diffsky/Diffstar truth export."""
    package_name = _available_diffsky_package()
    if package_name is None:
        message = (
            "Extended Diffsky truth extraction requires optional dependency "
            "`diffsky` or `lsstdesc-diffsky`. Need custom Diffsky "
            "generation/export to obtain full latent truths when public "
            "OpenUniverse files do not contain them."
        )
        if require_diffsky:
            raise ImportError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        frame = pd.DataFrame(
            {
                "status": ["unavailable"],
                "reason": [message],
                "input_root": [str(input_root)],
                "hpix_ids": [json.dumps([int(hpix) for hpix in hpix_ids])],
            }
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
        return frame

    raise NotImplementedError(
        f"{package_name} is installed, but extended Diffsky truth extraction is "
        "not wired to the local OpenUniverse generation files yet. Need custom "
        "Diffsky generation/export to obtain full latent truths."
    )


def truth_schema_to_json(schema: TruthSchema) -> dict:
    """Return a JSON-compatible truth-schema payload."""
    return asdict(schema)


def _first_existing(
    columns: tuple[str, ...], candidates: tuple[str, ...]
) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _matching_columns(
    lower_columns: dict[str, str],
    tokens: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        column
        for column, lower in lower_columns.items()
        if any(token in lower for token in tokens)
    )


def _available_diffsky_package() -> str | None:
    if importlib.util.find_spec("diffsky") is not None:
        return "diffsky"
    if importlib.util.find_spec("lsstdesc_diffsky") is not None:
        return "lsstdesc_diffsky"
    return None
