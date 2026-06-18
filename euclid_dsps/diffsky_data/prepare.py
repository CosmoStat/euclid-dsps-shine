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

from ..photometric_uncertainty import (
    flux_error_model_payload,
    normalize_flux_error_model,
)
from .photometry import detect_photometry_columns, standardize_magnitude_photometry
from .schema import (
    HLTDS_BURST_COLUMNS,
    HLTDS_DIFFMAH_COLUMNS,
    HLTDS_DIFFSTAR_COLUMNS,
    HLTDS_DUST_COLUMNS,
    HLTDS_TRUTH_COLUMNS,
)
from .truth import classify_diffsky_columns
from .units import describe_photometry_unit


@dataclass(frozen=True)
class DatasetBuildReport:
    output_path: str
    manifest_path: str
    schema_path: str
    truth_report_path: str
    integrity_report_path: str
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
    error_model: dict[str, Any] | None = None,
    seed: int = 42,
) -> DatasetBuildReport:
    del inventory_path, selected_photometry, require_truths, seed
    shards = sorted(Path(raw_root).glob("*.diffsky_gals.hdf5"))
    if not shards:
        raise FileNotFoundError(f"No *.diffsky_gals.hdf5 files found under {raw_root}")
    frames: list[pd.DataFrame] = []
    used_shards: list[Path] = []
    remaining = None if max_objects is None else max(int(max_objects), 0)
    for shard in shards:
        if remaining is not None and remaining <= 0:
            break
        frame = _read_hltds_shard(
            shard,
            limit=remaining,
            snr=snr,
            add_synthetic_errors=add_synthetic_errors,
            error_model=error_model,
        )
        if remaining is not None:
            remaining -= len(frame)
        frames.append(frame)
        used_shards.append(shard)
    dataset = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    object_id_report = _assign_global_object_ids(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    band_names = tuple(column.removeprefix("flux_") for column in dataset.columns if column.startswith("flux_"))
    semantics = classify_diffsky_columns(dataset.columns)
    error_model_payload = _error_model_payload(
        add_synthetic_errors=add_synthetic_errors,
        snr=snr,
        model=error_model,
    )
    readiness = _readiness(dataset)
    schema = _schema_for_dataset(dataset, band_names, semantics, error_model_payload)
    schema_path = output_path.with_suffix(".schema.json")
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    manifest = {
        "source_root": str(raw_root),
        "source_files": [str(path) for path in used_shards],
        "n_shards": int(len(used_shards)),
        "n_objects": int(len(dataset)),
        "object_id": object_id_report,
        "band_names": list(band_names),
        "photometry_unit": describe_photometry_unit("magnitude", "mag(AB)"),
        "prepared_flux_unit": "fnu_cgs",
        "error_model": error_model_payload,
        "column_semantics": semantics.as_dict(),
        "truth_columns": list(semantics.truth),
        "derived_truth_columns": list(semantics.derived_truth),
        "generated_truth_columns": list(semantics.generated_truth),
        "missing_fields": list(semantics.unavailable),
        "warnings": [
            "Native HLTDS photometry is apparent AB magnitude; prepared fluxes are fnu_cgs converted from AB.",
            "No metallicity columns were found in inspected HLTDS diffsky_gals shards.",
        ],
        "readiness": readiness,
    }
    manifest_path = output_path.with_suffix(".manifest.yaml")
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    truth_report_path = output_path.with_suffix(".truth_report.md")
    _write_truth_report(dataset, manifest, truth_report_path)
    integrity_report_path = output_path.with_name("diffsky_dataset_integrity_report.md")
    _write_integrity_report(dataset, manifest, integrity_report_path)
    return DatasetBuildReport(
        output_path=str(output_path),
        manifest_path=str(manifest_path),
        schema_path=str(schema_path),
        truth_report_path=str(truth_report_path),
        integrity_report_path=str(integrity_report_path),
        n_objects=int(len(dataset)),
        band_names=band_names,
        readiness=readiness,
    )


def _read_hltds_shard(
    path: Path,
    limit: int | None,
    snr: float,
    add_synthetic_errors: bool,
    error_model: dict[str, Any] | None,
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
        if "core_tag" in group:
            frame["core_tag"] = group["core_tag"][:n]
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
            error_model=error_model,
        )
        return pd.concat([frame, phot], axis=1)


def _assign_global_object_ids(frame: pd.DataFrame) -> dict[str, Any]:
    """Populate a unique ``object_id`` while preserving source identifiers."""
    report: dict[str, Any] = {
        "object_id_column": "object_id",
        "source_id_column": "core_tag" if "core_tag" in frame else None,
        "global_object_id_column": None,
        "source_file_column": "source_file" if "source_file" in frame else None,
        "source_row_column": "source_row" if "source_row" in frame else None,
        "core_tag_unique_global": None,
        "object_id_unique": True,
        "strategy": "empty_dataset",
    }
    if frame.empty:
        if "object_id" not in frame:
            frame.insert(0, "object_id", pd.Series(dtype="int64"))
        return report
    if "object_id" in frame:
        frame.drop(columns=["object_id"], inplace=True)
    if "core_tag" in frame:
        core_tag = frame["core_tag"]
        core_tag_unique = bool(core_tag.notna().all() and core_tag.is_unique)
        report["core_tag_unique_global"] = core_tag_unique
        if core_tag_unique:
            frame.insert(0, "object_id", core_tag.to_numpy())
            report["strategy"] = "core_tag"
        else:
            frame.insert(0, "global_object_id", np.arange(len(frame), dtype=np.int64))
            frame.insert(0, "object_id", frame["global_object_id"].to_numpy())
            report["global_object_id_column"] = "global_object_id"
            report["strategy"] = "global_object_id_from_source_order"
    else:
        frame.insert(0, "global_object_id", np.arange(len(frame), dtype=np.int64))
        frame.insert(0, "object_id", frame["global_object_id"].to_numpy())
        report["global_object_id_column"] = "global_object_id"
        report["strategy"] = "global_object_id_no_core_tag"
    report["object_id_unique"] = bool(frame["object_id"].is_unique)
    return report


def _error_model_payload(
    *,
    add_synthetic_errors: bool,
    snr: float,
    model: dict[str, Any] | None,
) -> dict[str, Any]:
    if add_synthetic_errors:
        normalized = (
            {"type": "fractional_snr", "snr": float(snr)}
            if model is None
            else normalize_flux_error_model(model)
        )
        payload = flux_error_model_payload(normalized)
        payload["native_error"] = False
        payload["synthetic"] = True
        return payload
    return {
        "type": "none",
        "native_error": False,
        "description": (
            "No fluxerr_* columns were written. HLTDS native photometric errors "
            "were not available in the inspected diffsky_gals shards."
        ),
        "model_tolerance_mag": None,
    }


def _readiness(frame: pd.DataFrame) -> str:
    flux_cols = [column for column in frame.columns if column.startswith("flux_")]
    has_basic = bool(flux_cols) and "redshift_true" in frame and "logsm_true" in frame
    has_extended = (
        has_basic
        and any(column.startswith("diffstar_") for column in frame)
        and any(column.startswith("diffmah_") for column in frame)
    )
    if has_extended:
        return "READY_EXTENDED"
    return "READY_BASIC" if has_basic else "NOT_READY"


def _schema_for_dataset(
    frame: pd.DataFrame,
    band_names: tuple[str, ...],
    semantics,
    error_model: dict[str, Any],
) -> dict[str, Any]:
    return {
        "latent_schema": "diffsky",
        "object_id_column": "object_id",
        "source_identity": {
            "core_tag": "core_tag" if "core_tag" in frame else None,
            "global_object_id": "global_object_id" if "global_object_id" in frame else None,
            "source_file": "source_file" if "source_file" in frame else None,
            "source_row": "source_row" if "source_row" in frame else None,
        },
        "band_names": list(band_names),
        "photometry": {
            "prepared_flux_unit": "fnu_cgs",
            "native_unit": describe_photometry_unit("magnitude", "mag(AB)"),
            "error_model": error_model,
        },
        "column_semantics": semantics.as_dict(),
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
        f"- readiness: {manifest['readiness']}",
        f"- error_model: `{manifest['error_model']['type']}`",
        "",
        "## Truth Columns",
        "",
    ]
    for column in manifest["truth_columns"]:
        lines.append(f"- `{column}`: truth from HLTDS `diffsky_gals`")
    lines.extend(["", "## Derived Truth Columns", ""])
    for column in manifest["derived_truth_columns"]:
        lines.append(f"- `{column}`: derived_truth from prepared HLTDS columns")
    lines.extend(["", "## Generated Truth Columns", ""])
    for column in manifest["generated_truth_columns"]:
        lines.append(f"- `{column}`: generated_truth from HLTDS `diffsky_gals`")
    lines.extend(["", "## Missing Fields", ""])
    for column in manifest["missing_fields"]:
        lines.append(f"- `{column}`")
    lines.extend(["", "## Warnings", ""])
    for warning in manifest["warnings"]:
        lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_integrity_report(frame: pd.DataFrame, manifest: dict[str, Any], path: Path) -> None:
    object_id = manifest["object_id"]
    semantics = manifest["column_semantics"]
    lines = [
        "# Diffsky Dataset Integrity Report",
        "",
        f"- objects: {manifest['n_objects']}",
        f"- shards: {manifest['n_shards']}",
        f"- object_id_unique: {object_id['object_id_unique']}",
        f"- object_id_strategy: `{object_id['strategy']}`",
        f"- core_tag_unique_global: {object_id['core_tag_unique_global']}",
        f"- photometry_unit: `{manifest['prepared_flux_unit']}`",
        f"- native_photometry_unit: `{manifest['photometry_unit']}`",
        f"- error_model: `{manifest['error_model']['type']}`",
        f"- readiness: `{manifest['readiness']}`",
        "",
        "## Bands",
        "",
        ", ".join(manifest["band_names"]) if manifest["band_names"] else "_None._",
        "",
        "## Available Truths",
        "",
        _markdown_list(semantics["truth"]),
        "",
        "## Available Derived Truths",
        "",
        _markdown_list(semantics["derived_truth"]),
        "",
        "## Available Generated Truths",
        "",
        _markdown_list(semantics["generated_truth"]),
        "",
        "## Missing Fields",
        "",
        _markdown_list(semantics["unavailable"]),
        "",
        "## Diagnostic Columns",
        "",
        _markdown_list(semantics["diagnostic"]),
        "",
        "## Error Semantics",
        "",
        (
            "No `fluxerr_*` column in this dataset should be interpreted as a "
            "native observational error unless `error_model.type` is "
            "`native_error`."
        ),
    ]
    if frame.empty:
        lines.extend(["", "## Empty Dataset", "", "No rows were written."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_list(values: list[str]) -> str:
    if not values:
        return "_None._"
    return "\n".join(f"- `{value}`" for value in values)
