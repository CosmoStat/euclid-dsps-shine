"""Merge externally exported Diffsky/Diffstar truth tables into OU subsets."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from euclid_dsps.io import ensure_dir

from .inventory import classify_openuniverse_column
from .truth import GENERATED_TRUTH, PROXY, TRUTH, validate_truth_level

ExternalTruthLevel = Literal["truth", "generated_truth", "proxy"]


def merge_external_truth_table(
    *,
    input_path: str | Path,
    truth_path: str | Path,
    output_path: str | Path,
    schema_path: str | Path | None = None,
    id_column: str = "galaxy_id",
    truth_columns: Sequence[str] | None = None,
    prefix: str = "generated_truth_",
    truth_level: ExternalTruthLevel = GENERATED_TRUTH,
) -> dict[str, Any]:
    """Merge a real external truth table by object id.

    This function does not infer or reconstruct Diffsky latents. It only marks
    columns as ``generated_truth``/``truth``/``proxy`` when the caller provides
    an external table that actually contains those columns.
    """
    level = validate_truth_level(str(truth_level))
    if level not in {TRUTH, GENERATED_TRUTH, PROXY}:
        raise ValueError(
            "External merge truth_level must be one of "
            f"{(TRUTH, GENERATED_TRUTH, PROXY)}"
        )
    base = _read_table(input_path)
    external = _read_table(truth_path)
    if id_column not in base:
        raise ValueError(f"Input table is missing id column {id_column!r}")
    if id_column not in external:
        raise ValueError(f"Truth table is missing id column {id_column!r}")
    if external[id_column].duplicated().any():
        raise ValueError(f"Truth table id column {id_column!r} is not unique")

    columns = _selected_truth_columns(external, id_column, truth_columns)
    renamed = external[[id_column, *columns]].rename(
        columns={column: _prefixed_column(column, prefix) for column in columns}
    )
    merged = base.merge(renamed, on=id_column, how="left", validate="many_to_one")
    out_path = Path(output_path)
    ensure_dir(out_path.parent)
    merged.to_parquet(out_path, index=False)

    exported_columns = [_prefixed_column(column, prefix) for column in columns]
    payload = {
        "input_path": str(input_path),
        "truth_path": str(truth_path),
        "output_path": str(out_path),
        "id_column": id_column,
        "truth_level": level,
        "prefix": str(prefix),
        "n_input_rows": int(len(base)),
        "n_truth_rows": int(len(external)),
        "n_output_rows": int(len(merged)),
        "source_columns": list(columns),
        "exported_columns": exported_columns,
        "column_levels": {column: level for column in exported_columns},
        "column_categories": {
            column: classify_openuniverse_column(column) for column in exported_columns
        },
        "matched_fraction": float(merged[exported_columns].notna().any(axis=1).mean())
        if exported_columns
        else 0.0,
        "policy": (
            "Columns are labeled according to the caller-provided truth_level. "
            "This merge does not prove that a proxy column is a physical "
            "ground truth."
        ),
    }
    if schema_path is not None:
        schema_out = Path(schema_path)
        ensure_dir(schema_out.parent)
        schema_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return payload


def _read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(table_path)
    if suffix == ".csv":
        return pd.read_csv(table_path)
    raise ValueError(f"Unsupported table format for {path}; use parquet or csv")


def _selected_truth_columns(
    external: pd.DataFrame,
    id_column: str,
    truth_columns: Sequence[str] | None,
) -> tuple[str, ...]:
    if truth_columns:
        columns = tuple(str(column) for column in truth_columns)
    else:
        columns = tuple(str(column) for column in external.columns if column != id_column)
    missing = [column for column in columns if column not in external]
    if missing:
        raise ValueError("Truth table is missing columns: " + ", ".join(missing))
    if not columns:
        raise ValueError("No external truth columns were selected for merge")
    return columns


def _prefixed_column(column: str, prefix: str) -> str:
    prefix_text = str(prefix)
    column_text = str(column)
    if not prefix_text:
        return column_text
    if column_text.startswith(prefix_text):
        return column_text
    return f"{prefix_text}{column_text}"
