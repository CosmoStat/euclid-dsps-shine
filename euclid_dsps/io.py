"""Catalog, observation, and artifact I/O."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BandObservation:
    name: str
    column: str
    flux_fnu_cgs: float
    mag_ab: float
    sigma_mag: float
    error_column: str | None = None
    flux_error_fnu_cgs: float | None = None


@dataclass(frozen=True)
class GalaxyObservation:
    row_index: int
    row: dict[str, Any]
    bands: list[BandObservation]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(to_jsonable(payload), stream, indent=2, sort_keys=True)


def configured_output_formats(config: dict[str, Any]) -> list[str]:
    """Return configured tabular output formats."""
    raw = (config.get("output", {}) or {}).get("format", "both")
    if isinstance(raw, str):
        if raw == "both":
            return ["parquet", "csv"]
        return [raw]
    formats = [str(item) for item in raw]
    return formats or ["parquet", "csv"]


def write_dataframe_outputs(
    frame: pd.DataFrame,
    out_dir: str | Path,
    stem: str,
    config: dict[str, Any],
    index: bool = False,
) -> list[str]:
    """Write a dataframe in configured formats and return filenames."""
    out = ensure_dir(out_dir)
    written: list[str] = []
    formats = configured_output_formats(config)
    if "parquet" in formats:
        path = out / f"{stem}.parquet"
        frame.to_parquet(path, index=index)
        written.append(path.name)
    if "csv" in formats:
        path = out / f"{stem}.csv"
        frame.to_csv(path, index=index)
        written.append(path.name)
    return written


def to_jsonable(value: Any) -> Any:
    if dataclass_is_instance(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if (
        pd.isna(value)
        if not isinstance(value, (list, tuple, dict, np.ndarray))
        else False
    ):
        return None
    return value


def dataclass_is_instance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__")


def read_catalog(
    path: str | Path, columns: list[str] | None = None, nrows: int | None = None
) -> pd.DataFrame:
    """Read a parquet catalog into memory, optionally truncating rows."""
    actual_columns = list(columns) if columns else None
    if actual_columns and "log10_metallicity_true" in actual_columns:
        actual_columns.remove("log10_metallicity_true")
        if "metallicity_true" not in actual_columns:
            actual_columns.append("metallicity_true")
    df = pd.read_parquet(path, columns=actual_columns)
    if "metallicity_true" in df.columns and "log10_metallicity_true" not in df.columns:
        # dataset has 12 + log(O/H), convert to absolute log10(Z)
        # DSPS internal solar log10(Z) = np.log10(0.012) = -1.92
        # log10(Z) = 12 + log(O/H) - 8.69 + (-1.92) = metallicity_true - 10.61
        df["log10_metallicity_true"] = df["metallicity_true"] - 10.61
    if nrows is not None:
        return df.head(nrows)
    return df


def truth_column_from_spec(spec: Any) -> str | None:
    """Return the catalog column named by a truth-column config entry."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        column = spec.get("column")
        return str(column) if column else None
    return None


def truth_value_from_spec(row: dict[str, Any], spec: Any) -> float | None:
    """Read and optionally transform a truth value from a catalog row."""
    column = truth_column_from_spec(spec)
    if not column or column not in row or pd.isna(row[column]):
        return None
    value = float(row[column])
    if not np.isfinite(value):
        return None

    if isinstance(spec, dict):
        transform = spec.get("transform")
        if transform == "log10":
            if value <= 0:
                return None
            value = float(np.log10(value))
        elif transform == "log_stellar_mass_h2_to_msun":
            h = float(spec.get("h"))
            if not np.isfinite(h) or h <= 0:
                raise ValueError(
                    "truth log_stellar_mass_h2_to_msun transform needs h > 0"
                )
            value = float(value + 2.0 * np.log10(h))
        elif transform not in {None, "linear"}:
            raise ValueError(f"Unsupported truth transform: {transform}")
        value = value * float(spec.get("scale", 1.0))
        value = value + float(spec.get("offset", 0.0))
    return value


def iter_catalog_batches(
    path: str | Path,
    columns: list[str] | None = None,
    batch_size: int = 10_000,
    limit: int | None = None,
    row_indices: set[int] | None = None,
) -> Iterable[pd.DataFrame]:
    """Yield catalog batches without loading the full parquet into memory."""
    import pyarrow.parquet as pq

    yielded = 0
    seen = 0
    max_row_index = max(row_indices) if row_indices else None
    parquet = pq.ParquetFile(path)

    actual_columns = list(columns) if columns else None
    if actual_columns and "log10_metallicity_true" in actual_columns:
        actual_columns.remove("log10_metallicity_true")
        if "metallicity_true" not in actual_columns:
            actual_columns.append("metallicity_true")

    for batch in parquet.iter_batches(batch_size=batch_size, columns=actual_columns):
        df = batch.to_pandas()
        if (
            "metallicity_true" in df.columns
            and "log10_metallicity_true" not in df.columns
        ):
            df["log10_metallicity_true"] = df["metallicity_true"] - 10.61
        raw_len = len(df)
        df.index = range(seen, seen + raw_len)
        seen += raw_len
        if row_indices is not None:
            df = df.loc[df.index.isin(row_indices)]
        if limit is not None:
            remaining = limit - yielded
            if remaining <= 0:
                break
            df = df.head(remaining)
        if len(df):
            yielded += len(df)
            yield df
        if max_row_index is not None and seen > max_row_index:
            break


def load_row_indices(path: str | Path) -> list[int]:
    """Load row indices from a one-column text or CSV file."""
    rows = pd.read_csv(path, comment="#", header=None)
    if rows.empty:
        return []
    return sorted(
        {int(value) for value in rows.iloc[:, 0].dropna().astype(int).tolist()}
    )


def flux_fnu_cgs_to_abmag(flux: float) -> float:
    """Convert F_nu in erg/s/cm^2/Hz to AB magnitude."""
    if not np.isfinite(flux) or flux <= 0:
        return float("nan")
    return float(-2.5 * np.log10(flux) - 48.6)


def abmag_to_flux_fnu_cgs(mag: float) -> float:
    """Convert AB magnitude to F_nu in erg/s/cm^2/Hz."""
    return float(10 ** (-0.4 * (mag + 48.6)))


def microjy_to_flux_fnu_cgs(flux_microjy: float) -> float:
    """Convert microJansky to F_nu in erg/s/cm^2/Hz."""
    return float(flux_microjy * 1.0e-29)


def microjy_to_abmag(flux_microjy: float) -> float:
    """Convert microJansky to AB magnitude."""
    if not np.isfinite(flux_microjy) or flux_microjy <= 0:
        return float("nan")
    return float(-2.5 * np.log10(flux_microjy) + 23.9)


def flux_error_to_sigma_mag(
    flux_fnu_cgs: float,
    flux_error_fnu_cgs: float,
    floor: float | None = None,
    ceiling: float | None = None,
) -> float:
    """Convert a flux-density uncertainty into a local AB-mag uncertainty."""
    if (
        not np.isfinite(flux_fnu_cgs)
        or not np.isfinite(flux_error_fnu_cgs)
        or flux_fnu_cgs <= 0.0
        or flux_error_fnu_cgs <= 0.0
    ):
        return float("nan")
    sigma = float((2.5 / np.log(10.0)) * abs(flux_error_fnu_cgs / flux_fnu_cgs))
    if floor is not None and np.isfinite(floor):
        sigma = max(sigma, float(floor))
    if ceiling is not None and np.isfinite(ceiling):
        sigma = min(sigma, float(ceiling))
    return sigma


def build_observation(
    row_index: int, row: pd.Series, band_configs: list[dict[str, Any]]
) -> GalaxyObservation:
    """Build one photometric observation from a catalog row.

    When a band declares ``error_column``, the catalog flux-density error is
    converted to a local AB-magnitude uncertainty and used by the likelihood.
    The configured ``sigma_mag`` remains the fallback for bands without usable
    per-object errors.
    """
    bands = []
    for band in band_configs:
        column = band["column"]
        value = float(row[column])
        units = band.get("units", "fnu_cgs")
        if units == "fnu_cgs":
            mag_ab = flux_fnu_cgs_to_abmag(value)
            flux_fnu_cgs = value
        elif units == "abmag":
            mag_ab = value
            flux_fnu_cgs = abmag_to_flux_fnu_cgs(value)
        elif units in {"microjy", "ujy"}:
            mag_ab = microjy_to_abmag(value)
            flux_fnu_cgs = microjy_to_flux_fnu_cgs(value)
        else:
            raise ValueError(
                f"Unsupported photometry units for {band['name']}: {units}"
            )
        error_column = band.get("error_column")
        flux_error = None
        sigma_mag = float(band.get("sigma_mag", 0.05))
        if error_column and error_column in row and pd.notna(row[error_column]):
            raw_error = float(row[error_column])
            error_units = band.get("error_units", units)
            if error_units == "abmag":
                if np.isfinite(raw_error) and raw_error > 0:
                    sigma_mag = raw_error
            else:
                if error_units == "fnu_cgs":
                    flux_error = raw_error
                elif error_units in {"microjy", "ujy"}:
                    flux_error = microjy_to_flux_fnu_cgs(raw_error)
                else:
                    raise ValueError(
                        f"Unsupported photometry error units for {band['name']}: {error_units}"
                    )
                converted = flux_error_to_sigma_mag(
                    flux_fnu_cgs,
                    flux_error,
                    floor=band.get("sigma_mag_floor"),
                    ceiling=band.get("sigma_mag_ceiling"),
                )
                if np.isfinite(converted):
                    sigma_mag = converted
        bands.append(
            BandObservation(
                name=band["name"],
                column=column,
                flux_fnu_cgs=flux_fnu_cgs,
                mag_ab=mag_ab,
                sigma_mag=sigma_mag,
                error_column=str(error_column) if error_column else None,
                flux_error_fnu_cgs=flux_error,
            )
        )
    return GalaxyObservation(row_index=row_index, row=row.to_dict(), bands=bands)


def required_catalog_columns(config: dict[str, Any]) -> list[str]:
    columns = {band["column"] for band in config["bands"]}
    for band in config["bands"]:
        if band.get("error_column"):
            columns.add(str(band["error_column"]))
    for col in config.get("extra_columns", []):
        columns.add(col)
    for col in (config.get("model", {}).get("parameter_columns") or {}).values():
        columns.add(col)
    redshift = config.get("redshift", {})
    for key in ("column", "truth_column"):
        col = truth_column_from_spec(redshift.get(key))
        if col:
            columns.add(col)
    truth = config.get("truth", {})
    truth_redshift = truth_column_from_spec(truth.get("redshift_column"))
    if truth_redshift:
        columns.add(truth_redshift)
    for spec in (truth.get("parameter_columns") or {}).values():
        col = truth_column_from_spec(spec)
        if col:
            columns.add(col)
    sort_col = config.get("selection", {}).get("sort_by_flux")
    if sort_col:
        columns.add(sort_col)
    return sorted(columns)
