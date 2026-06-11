"""Build normalized photometry + truth parquet files from Diffsky HDF5 shards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml

from .photometry import detect_photometry_columns, standardize_magnitude_photometry
from .schema import (
    HLTDS_BURST_COLUMNS,
    HLTDS_DIFFMAH_COLUMNS,
    HLTDS_DIFFSTAR_COLUMNS,
    HLTDS_DUST_COLUMNS,
    HLTDS_TRUTH_COLUMNS,
)
from .units import describe_photometry_unit


@dataclass(frozen=True)
class DatasetBuildReport:
    output_path: str
    manifest_path: str
    schema_path: str
    truth_report_path: str
    n_objects: int
    band_names: tuple[str, ...]
    readiness: str


def build_diffsky_photometric_dataset(
    *,
    raw_root: Path,
    inventory_path: Path | None,
    output_path: Path,
    max_objects: int | None = None,
    selected_photometry: str = "auto",
    require_truths: tuple[str, ...] = ("redshift", "stellar_mass"),
    add_synthetic_errors: bool = True,
    snr: float = 50.0,
    seed: int = 42,
) -> DatasetBuildReport:
    del inventory_path, selected_photometry, require_truths, seed
    shards = sorted(Path(raw_root).glob("*.diffsky_gals.hdf5"))
    if not shards:
        raise FileNotFoundError(f"No *.diffsky_gals.hdf5 files found under {raw_root}")
    frames: list[pd.DataFrame] = []
    remaining = None if max_objects is None else max(int(max_objects), 0)
    for shard in shards:
        if remaining is not None and remaining <= 0:
            break
        frame = _read_hltds_shard(
            shard,
            limit=remaining,
            snr=snr,
            add_synthetic_errors=add_synthetic_errors,
        )
        if remaining is not None:
            remaining -= len(frame)
        frames.append(frame)
    dataset = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    band_names = tuple(column.removeprefix("flux_") for column in dataset.columns if column.startswith("flux_"))
    schema = _schema_for_dataset(dataset, band_names)
    schema_path = output_path.with_suffix(".schema.json")
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    manifest = {
        "source_root": str(raw_root),
        "source_files": [str(path) for path in shards],
        "n_objects": int(len(dataset)),
        "band_names": list(band_names),
        "photometry_unit": describe_photometry_unit("magnitude", "mag(AB)"),
        "prepared_flux_unit": "fnu_cgs",
        "error_model": (
            {"type": "synthetic_fractional_snr", "snr": float(snr)}
            if add_synthetic_errors
            else {
                "type": "none",
                "note": "No native HLTDS photometric errors were found; fit configs should use explicit model-tolerance magnitudes.",
            }
        ),
        "truth_columns": [column for column in dataset.columns if column.endswith("_true")],
        "generated_truth_columns": [
            column
            for column in dataset.columns
            if column.startswith(("diffmah_", "diffstar_", "dust_", "burst_"))
        ],
        "warnings": [
            "Native HLTDS photometry is apparent AB magnitude; prepared fluxes are fnu_cgs converted from AB.",
            "No metallicity columns were found in inspected HLTDS diffsky_gals shards.",
        ],
    }
    manifest_path = output_path.with_suffix(".manifest.yaml")
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    truth_report_path = output_path.with_suffix(".truth_report.md")
    _write_truth_report(dataset, manifest, truth_report_path)
    readiness = "READY_EXTENDED" if any(column.startswith("diffstar_") for column in dataset.columns) else "READY_BASIC"
    return DatasetBuildReport(
        output_path=str(output_path),
        manifest_path=str(manifest_path),
        schema_path=str(schema_path),
        truth_report_path=str(truth_report_path),
        n_objects=int(len(dataset)),
        band_names=band_names,
        readiness=readiness,
    )


def _read_hltds_shard(
    path: Path,
    limit: int | None,
    snr: float,
    add_synthetic_errors: bool,
) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"{path} is not an HLTDS diffsky_gals file: missing /data")
        group = handle["data"]
        n_rows = int(group["redshift_true"].shape[0])
        n = n_rows if limit is None else min(n_rows, max(int(limit), 0))
        columns = list(group.keys())
        phot_report = detect_photometry_columns(columns)
        data = {column: group[column][:n] for column in phot_report.native_columns}
        frame = pd.DataFrame()
        frame["object_id"] = group["core_tag"][:n] if "core_tag" in group else np.arange(n)
        frame["source_file"] = path.name
        frame["source_row"] = np.arange(n, dtype=int)
        for output, source in HLTDS_TRUTH_COLUMNS.items():
            if source in group:
                frame[output] = group[source][:n]
        if "logsm_obs" in group and "logssfr_obs" in group:
            frame["logsfr_true"] = group["logsm_obs"][:n] + group["logssfr_obs"][:n]
        for column in HLTDS_DIFFMAH_COLUMNS:
            if column in group:
                frame[f"diffmah_{column}"] = group[column][:n]
        for column in HLTDS_DIFFSTAR_COLUMNS:
            if column in group:
                frame[f"diffstar_{column}"] = group[column][:n]
        for column in HLTDS_DUST_COLUMNS:
            if column in group:
                frame[f"dust_{column}"] = group[column][:n]
        for column in HLTDS_BURST_COLUMNS:
            if column in group:
                frame[f"burst_{column}"] = group[column][:n]
        phot = standardize_magnitude_photometry(
            data,
            phot_report,
            snr=snr,
            add_synthetic_errors=add_synthetic_errors,
        )
        return pd.concat([frame, phot], axis=1)


def _schema_for_dataset(frame: pd.DataFrame, band_names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "latent_schema": "diffsky",
        "object_id_column": "object_id",
        "band_names": list(band_names),
        "truth": {
            "redshift": "redshift_true" if "redshift_true" in frame else None,
            "stellar_mass": "logsm_true" if "logsm_true" in frame else None,
            "ssfr": "logssfr_true" if "logssfr_true" in frame else None,
            "sfr": "logsfr_true" if "logsfr_true" in frame else None,
            "halo_mass": "logmp_true" if "logmp_true" in frame else None,
        },
        "generated_truth": [
            column
            for column in frame.columns
            if column.startswith(("diffmah_", "diffstar_", "dust_", "burst_"))
        ],
        "parameters": _parameter_bounds(frame),
    }


def _parameter_bounds(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [
        column
        for column in frame.columns
        if column.endswith("_true") or column.startswith(("diffmah_", "diffstar_", "dust_", "burst_"))
    ]
    rows = []
    for column in cols:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        rows.append(
            {
                "name": column,
                "column": column,
                "lower": float(values.quantile(0.001)),
                "upper": float(values.quantile(0.999)),
                "median": float(values.median()),
            }
        )
    return rows


def _write_truth_report(frame: pd.DataFrame, manifest: dict[str, Any], path: Path) -> None:
    lines = [
        "# Diffsky Truth Report",
        "",
        f"- objects: {len(frame)}",
        f"- bands: {', '.join(manifest['band_names'])}",
        f"- readiness: {'READY_EXTENDED' if manifest['generated_truth_columns'] else 'READY_BASIC'}",
        "",
        "## Truth Columns",
        "",
    ]
    for column in manifest["truth_columns"]:
        lines.append(f"- `{column}`: truth from HLTDS `diffsky_gals`")
    lines.extend(["", "## Generated Truth Columns", ""])
    for column in manifest["generated_truth_columns"]:
        lines.append(f"- `{column}`: generated_truth from HLTDS `diffsky_gals`")
    lines.extend(["", "## Warnings", ""])
    for warning in manifest["warnings"]:
        lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
