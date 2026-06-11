"""Validation checks for Diffsky datasets used in prior learning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .truth import classify_diffsky_columns


def validate_for_prior_learning(dataset_path: str | Path, manifest_path: str | Path | None = None) -> dict:
    frame = pd.read_parquet(dataset_path)
    manifest = {}
    if manifest_path is not None and Path(manifest_path).exists():
        manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
    flux_cols = [column for column in frame.columns if column.startswith("flux_")]
    err_cols = [column for column in frame.columns if column.startswith("fluxerr_")]
    mask_cols = [column for column in frame.columns if column.startswith("mask_")]
    semantics = classify_diffsky_columns(frame.columns)
    has_basic = bool(flux_cols) and "redshift_true" in frame and "logsm_true" in frame
    has_extended = has_basic and any(column.startswith("diffstar_") for column in frame) and any(column.startswith("diffmah_") for column in frame)
    readiness = "READY_EXTENDED" if has_extended else ("READY_BASIC" if has_basic else "NOT_READY")
    error_model = manifest.get("error_model") or _infer_error_model_from_columns(err_cols)
    report = {
        "readiness": readiness,
        "n_objects": int(len(frame)),
        "n_bands": len(flux_cols),
        "n_fluxerr_columns": len(err_cols),
        "object_id_unique": bool(frame["object_id"].is_unique) if "object_id" in frame else False,
        "core_tag_unique_global": bool(frame["core_tag"].is_unique) if "core_tag" in frame else None,
        "has_redshift_true": "redshift_true" in frame,
        "has_logsm_true": "logsm_true" in frame,
        "has_diffstar": any(column.startswith("diffstar_") for column in frame),
        "has_diffmah": any(column.startswith("diffmah_") for column in frame),
        "mask_fraction_valid": float(frame[mask_cols].to_numpy(dtype=bool).mean()) if mask_cols else None,
        "band_names": manifest.get("band_names", [column.removeprefix("flux_") for column in flux_cols]),
        "error_model": error_model,
        "column_semantics": manifest.get("column_semantics", semantics.as_dict()),
        "missing_fields": manifest.get("missing_fields", list(semantics.unavailable)),
    }
    return report


def _infer_error_model_from_columns(err_cols: list[str]) -> dict:
    if err_cols:
        return {
            "type": "unknown_non_native",
            "native_error": False,
            "description": "fluxerr_* columns are present but no manifest declared their provenance.",
        }
    return {
        "type": "none",
        "native_error": False,
        "description": "No fluxerr_* columns are present.",
    }


def write_validation_report(report: dict, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Diffsky Prior-Learning Validation",
        "",
        f"- readiness: `{report['readiness']}`",
        f"- objects: {report['n_objects']}",
        f"- bands: {report['n_bands']}",
        f"- fluxerr columns: {report['n_fluxerr_columns']}",
        f"- object_id_unique: {report['object_id_unique']}",
        f"- core_tag_unique_global: {report['core_tag_unique_global']}",
        f"- redshift truth: {report['has_redshift_true']}",
        f"- stellar mass truth: {report['has_logsm_true']}",
        f"- Diffstar generated truth: {report['has_diffstar']}",
        f"- Diffmah generated truth: {report['has_diffmah']}",
        f"- mask_fraction_valid: {report['mask_fraction_valid']}",
        f"- error_model: `{report['error_model']['type']}`",
        "",
        "## Bands",
        "",
        ", ".join(report["band_names"]),
        "",
        "## Truth Semantics",
        "",
        f"- truth: {', '.join(report['column_semantics'].get('truth', [])) or 'none'}",
        f"- derived_truth: {', '.join(report['column_semantics'].get('derived_truth', [])) or 'none'}",
        f"- generated_truth: {', '.join(report['column_semantics'].get('generated_truth', [])) or 'none'}",
        "",
        "## Missing Fields",
        "",
        ", ".join(report["missing_fields"]) if report["missing_fields"] else "none",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
