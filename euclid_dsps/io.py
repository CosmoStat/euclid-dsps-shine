"""Catalog, observation, and artifact I/O."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BandObservation:
    name: str
    column: str
    flux_fnu_cgs: float
    mag_ab: float
    sigma_mag: float


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
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value


def dataclass_is_instance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__")


def read_catalog(path: str | Path, columns: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Read a parquet catalog into memory, optionally truncating rows."""
    df = pd.read_parquet(path, columns=columns)
    if nrows is not None:
        return df.head(nrows)
    return df


def iter_catalog_batches(
    path: str | Path,
    columns: list[str] | None = None,
    batch_size: int = 10_000,
    limit: int | None = None,
) -> Iterable[pd.DataFrame]:
    """Yield catalog batches without loading the full parquet into memory."""
    import pyarrow.parquet as pq

    emitted = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        df = batch.to_pandas()
        if limit is not None:
            remaining = limit - emitted
            if remaining <= 0:
                break
            df = df.head(remaining)
        df.index = range(emitted, emitted + len(df))
        emitted += len(df)
        if len(df):
            yield df


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


def build_observation(row_index: int, row: pd.Series, band_configs: list[dict[str, Any]]) -> GalaxyObservation:
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
            raise ValueError(f"Unsupported photometry units for {band['name']}: {units}")
        bands.append(
            BandObservation(
                name=band["name"],
                column=column,
                flux_fnu_cgs=flux_fnu_cgs,
                mag_ab=mag_ab,
                sigma_mag=float(band.get("sigma_mag", 0.05)),
            )
        )
    return GalaxyObservation(row_index=row_index, row=row.to_dict(), bands=bands)


def required_catalog_columns(config: dict[str, Any]) -> list[str]:
    columns = {band["column"] for band in config["bands"]}
    for col in config.get("extra_columns", []):
        columns.add(col)
    for col in (config.get("model", {}).get("parameter_columns") or {}).values():
        columns.add(col)
    redshift = config.get("redshift", {})
    for key in ("column", "truth_column"):
        col = redshift.get(key)
        if col:
            columns.add(col)
    truth = config.get("truth", {})
    truth_redshift = truth.get("redshift_column")
    if truth_redshift:
        columns.add(truth_redshift)
    for col in (truth.get("parameter_columns") or {}).values():
        columns.add(col)
    sort_col = config.get("selection", {}).get("sort_by_flux")
    if sort_col:
        columns.add(sort_col)
    return sorted(columns)
