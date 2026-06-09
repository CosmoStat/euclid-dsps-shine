"""Validation checks for Diffsky datasets used in prior learning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def validate_for_prior_learning(dataset_path: str | Path, manifest_path: str | Path | None = None) -> dict:
    frame = pd.read_parquet(dataset_path)
    manifest = {}
    if manifest_path is not None and Path(manifest_path).exists():
        manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
    flux_cols = [column for column in frame.columns if column.startswith("flux_")]
    mask_cols = [column for column in frame.columns if column.startswith("mask_")]
    has_basic = bool(flux_cols) and "redshift_true" in frame and "logsm_true" in frame
    has_extended = has_basic and any(column.startswith("diffstar_") for column in frame) and any(column.startswith("diffmah_") for column in frame)
    readiness = "READY_EXTENDED" if has_extended else ("READY_BASIC" if has_basic else "NOT_READY")
    report = {
        "readiness": readiness,
        "n_objects": int(len(frame)),
        "n_bands": len(flux_cols),
        "object_id_unique": bool(frame["object_id"].is_unique) if "object_id" in frame else False,
        "has_redshift_true": "redshift_true" in frame,
        "has_logsm_true": "logsm_true" in frame,
        "has_diffstar": any(column.startswith("diffstar_") for column in frame),
        "has_diffmah": any(column.startswith("diffmah_") for column in frame),
        "mask_fraction_valid": float(frame[mask_cols].to_numpy(dtype=bool).mean()) if mask_cols else None,
        "band_names": manifest.get("band_names", [column.removeprefix("flux_") for column in flux_cols]),
    }
    return report


def write_validation_report(report: dict, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Diffsky Prior-Learning Validation",
        "",
        f"- readiness: `{report['readiness']}`",
        f"- objects: {report['n_objects']}",
        f"- bands: {report['n_bands']}",
        f"- object_id_unique: {report['object_id_unique']}",
        f"- redshift truth: {report['has_redshift_true']}",
        f"- stellar mass truth: {report['has_logsm_true']}",
        f"- Diffstar generated truth: {report['has_diffstar']}",
        f"- Diffmah generated truth: {report['has_diffmah']}",
        f"- mask_fraction_valid: {report['mask_fraction_valid']}",
        "",
        "## Bands",
        "",
        ", ".join(report["band_names"]),
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
