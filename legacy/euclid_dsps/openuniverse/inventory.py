"""Truth-field inventory reports for OpenUniverse SkyCatalog files."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from euclid_dsps.io import ensure_dir

from .access import resolve_openuniverse_paths
from .diffsky_truth import TruthSchema, infer_truth_schema, truth_schema_to_json
from .schema import OU_FLUX_COLUMNS, OU_TRUTH_COLUMNS
from .sed import inventory_openuniverse_sed, sed_inventory_to_dict
from .truth import GENERATED_TRUTH, PROXY, TRUTH, UNAVAILABLE


def inventory_openuniverse_truth_fields(
    *,
    output_dir: str | Path,
    processed_path: str | Path | None = None,
    input_root: str | Path | None = None,
    hpix_ids: Sequence[int] | None = None,
    include_sed: bool = False,
    sed_sample_limit: int = 5,
) -> dict[str, Any]:
    """Write JSON/Markdown inventory reports for OpenUniverse truth fields."""
    out = ensure_dir(output_dir)
    parquet_files = _resolve_parquet_files(
        processed_path=processed_path,
        input_root=input_root,
        hpix_ids=hpix_ids,
    )
    sed_files = _resolve_sed_files(
        input_root=input_root,
        hpix_ids=hpix_ids,
        include_sed=include_sed,
    )

    parquet_inventories = [_parquet_inventory(role, path) for role, path in parquet_files]
    combined_columns = tuple(
        dict.fromkeys(
            column
            for inventory in parquet_inventories
            for column in inventory.get("columns", [])
        )
    )
    schema = infer_truth_schema(pd.DataFrame(columns=combined_columns))
    sed_inventories = [
        sed_inventory_to_dict(
            inventory_openuniverse_sed(path, sample_limit=int(sed_sample_limit))
        )
        for path in sed_files
    ]
    summary = _physical_truth_summary(schema, sed_inventories)
    payload = {
        "dataset": "openuniverse",
        "processed_path": None if processed_path is None else str(processed_path),
        "input_root": None if input_root is None else str(input_root),
        "hpix_ids": [] if hpix_ids is None else [int(hpix) for hpix in hpix_ids],
        "parquet_files": parquet_inventories,
        "sed_files": sed_inventories,
        "truth_schema": truth_schema_to_json(schema),
        "physical_truth_summary": summary,
        "column_catalog": _column_catalog(combined_columns),
        "notes": [
            "Only directly present columns are labeled truth.",
            "Diffsky/Diffstar latent SFH, dust, metallicity, and halo parameters "
            "remain unavailable unless exported by a separate generation product.",
            "The low-resolution SED HDF5 is a generated SED product, not an "
            "inverse-inferred physical-parameter truth table.",
        ],
    }
    (out / "openuniverse_truth_inventory.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "truth_schema.json").write_text(
        json.dumps(payload["truth_schema"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "openuniverse_truth_inventory.md").write_text(
        render_truth_inventory_markdown(payload),
        encoding="utf-8",
    )
    return payload


def render_truth_inventory_markdown(payload: dict[str, Any]) -> str:
    """Render a compact Markdown inventory report."""
    lines = [
        "# OpenUniverse Truth Inventory",
        "",
        f"- processed_path: `{payload.get('processed_path')}`",
        f"- input_root: `{payload.get('input_root')}`",
        f"- hpix_ids: `{payload.get('hpix_ids')}`",
        "",
        "## Physical Truth Summary",
        "",
        "| quantity | level | status | source columns | notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload["physical_truth_summary"]:
        columns = ", ".join(f"`{column}`" for column in row["source_columns"])
        lines.append(
            "| {quantity} | {level} | {status} | {columns} | {notes} |".format(
                quantity=row["quantity"],
                level=row["truth_level"],
                status=row["status"],
                columns=columns or "-",
                notes=row["notes"],
            )
        )

    lines.extend(["", "## Parquet Files", ""])
    for inventory in payload["parquet_files"]:
        lines.extend(
            [
                f"### {inventory['role']}",
                "",
                f"- path: `{inventory['path']}`",
                f"- exists: `{inventory['exists']}`",
                f"- rows: `{inventory.get('num_rows')}`",
                f"- columns: `{inventory.get('num_columns')}`",
                "",
            ]
        )

    lines.extend(["## SED Files", ""])
    if not payload["sed_files"]:
        lines.append("No SED file was inventoried.")
    for inventory in payload["sed_files"]:
        lines.extend(
            [
                f"- path: `{inventory['path']}`",
                f"- exists: `{inventory['exists']}`",
                f"- wavelength dataset: `{inventory.get('wavelength_dataset')}`",
                f"- wavelength size: `{inventory.get('wavelength_size')}`",
                f"- galaxy prefix groups: `{inventory.get('galaxy_prefix_count')}`",
                f"- SED datasets: `{inventory.get('sed_dataset_count')}`",
                "",
            ]
        )
        if inventory.get("sample_datasets"):
            lines.extend(
                [
                    "| galaxy_id | path | shape | dtype | finite_fraction |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for sample in inventory["sample_datasets"]:
                lines.append(
                    "| {galaxy_id} | `{path}` | `{shape}` | `{dtype}` | `{finite}` |".format(
                        galaxy_id=sample["galaxy_id"],
                        path=sample["path"],
                        shape=sample["shape"],
                        dtype=sample["dtype"],
                        finite=sample.get("finite_fraction"),
                    )
                )
            lines.append("")

    lines.extend(
        [
            "## Caveats",
            "",
            "- Do not call SFH, Diffstar, dust, metallicity, or halo latents truth "
            "unless a real Diffsky export provides them.",
            "- OpenUniverse photometric fluxes remain in native photon-rate units.",
            "",
        ]
    )
    return "\n".join(lines)


def write_basic_truth_artifacts(
    *,
    input_path: str | Path,
    output_path: str | Path,
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract direct OpenUniverse truth columns and write companion schema JSON."""
    from .diffsky_truth import extract_basic_truth_table

    frame = pd.read_parquet(input_path)
    truth = extract_basic_truth_table(frame)
    out_path = Path(output_path)
    ensure_dir(out_path.parent)
    truth.to_parquet(out_path, index=False)
    schema = infer_truth_schema(frame)
    payload = {
        "input_path": str(input_path),
        "output_path": str(out_path),
        "number_of_rows": int(len(truth)),
        "columns": list(truth.columns),
        "truth_schema": truth_schema_to_json(schema),
        "truth_levels": dict(truth.attrs.get("truth_levels", {})),
    }
    if schema_path is not None:
        schema_out = Path(schema_path)
        ensure_dir(schema_out.parent)
        schema_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return payload


def _resolve_parquet_files(
    *,
    processed_path: str | Path | None,
    input_root: str | Path | None,
    hpix_ids: Sequence[int] | None,
) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    if processed_path is not None:
        files.append(("processed", str(processed_path)))
    if input_root is not None and hpix_ids:
        for hpix, paths in zip(
            (int(hpix) for hpix in hpix_ids),
            resolve_openuniverse_paths(hpix_ids, input_root),
            strict=True,
        ):
            files.append((f"main_hpix_{hpix}", paths["main"]))
            files.append((f"flux_hpix_{hpix}", paths["flux"]))
    return files


def _resolve_sed_files(
    *,
    input_root: str | Path | None,
    hpix_ids: Sequence[int] | None,
    include_sed: bool,
) -> list[str]:
    if not include_sed or input_root is None or not hpix_ids:
        return []
    return [
        paths["sed"]
        for paths in resolve_openuniverse_paths(
            [int(hpix) for hpix in hpix_ids],
            input_root,
        )
    ]


def _parquet_inventory(role: str, path: str) -> dict[str, Any]:
    parquet_path = Path(path)
    if "://" not in path and not parquet_path.exists():
        return {
            "role": role,
            "path": path,
            "exists": False,
            "columns": [],
            "column_details": [],
            "num_rows": None,
            "num_columns": None,
        }
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    return {
        "role": role,
        "path": path,
        "exists": True,
        "num_rows": int(parquet.metadata.num_rows),
        "num_columns": int(len(schema)),
        "columns": [str(field.name) for field in schema],
        "column_details": [
            {
                "name": str(field.name),
                "dtype": str(field.type),
                "nullable": bool(field.nullable),
                "category": classify_openuniverse_column(str(field.name)),
            }
            for field in schema
        ],
    }


def _physical_truth_summary(
    schema: TruthSchema,
    sed_inventories: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    sed_available = any(inventory.get("exists") for inventory in sed_inventories)
    rows = [
        _summary_row(
            "redshift",
            TRUTH if schema.redshift_column else UNAVAILABLE,
            (schema.redshift_column,) if schema.redshift_column else (),
            "Direct public OpenUniverse column when present.",
        ),
        _summary_row(
            "redshift_hubble",
            TRUTH if "redshiftHubble" in schema.available_columns else UNAVAILABLE,
            ("redshiftHubble",) if "redshiftHubble" in schema.available_columns else (),
            "Direct public OpenUniverse Hubble-flow redshift when present.",
        ),
        _summary_row(
            "stellar_mass",
            TRUTH if schema.stellar_mass_column else UNAVAILABLE,
            (schema.stellar_mass_column,) if schema.stellar_mass_column else (),
            "Direct public column `um_source_galaxy_obs_sm` or normalized alias.",
        ),
        _summary_row(
            "photometric_truth_flux",
            TRUTH if _all_flux_columns_present(schema.available_columns) else UNAVAILABLE,
            tuple(column for column in OU_FLUX_COLUMNS.values() if column in schema.available_columns)
            or tuple(column for column in schema.available_columns if column.startswith("flux_truth_")),
            "OpenUniverse public flux table or normalized truth-flux columns.",
        ),
        _summary_row(
            "low_resolution_sed",
            GENERATED_TRUTH if sed_available else UNAVAILABLE,
            tuple(
                inventory["path"]
                for inventory in sed_inventories
                if inventory.get("exists")
            ),
            "Generated low-resolution SED product in `galaxy_sed_<hpix>.hdf5`.",
        ),
        _summary_row(
            "sfr_columns",
            PROXY if schema.sfr_columns else UNAVAILABLE,
            schema.sfr_columns,
            "Only column-pattern inventory; not promoted to truth without datamodel validation.",
        ),
        _summary_row(
            "sfh_or_diffstar_latents",
            GENERATED_TRUTH if schema.sfh_columns else UNAVAILABLE,
            schema.sfh_columns,
            "Requires real Diffsky/Diffstar export to be complete generated truth.",
        ),
        _summary_row(
            "dust",
            PROXY if schema.dust_columns else UNAVAILABLE,
            schema.dust_columns,
            "MW or attenuation-like columns are not internal Diffsky dust latents by default.",
        ),
        _summary_row(
            "metallicity",
            GENERATED_TRUTH if schema.metallicity_columns else UNAVAILABLE,
            schema.metallicity_columns,
            "Unavailable unless metallicity columns are explicitly present.",
        ),
        _summary_row(
            "halo_or_mah",
            PROXY if schema.halo_columns else UNAVAILABLE,
            schema.halo_columns,
            "Lensing/shear columns are not halo mass or MAH latents.",
        ),
    ]
    return rows


def _summary_row(
    quantity: str,
    truth_level: str,
    source_columns: Sequence[str | None],
    notes: str,
) -> dict[str, Any]:
    columns = tuple(str(column) for column in source_columns if column)
    return {
        "quantity": quantity,
        "truth_level": truth_level,
        "status": "available" if truth_level != UNAVAILABLE else "unavailable",
        "source_columns": list(columns),
        "notes": notes,
    }


def _all_flux_columns_present(columns: Sequence[str]) -> bool:
    available = set(columns)
    raw_fluxes = set(OU_FLUX_COLUMNS.values())
    normalized_fluxes = {f"flux_truth_{band}" for band in OU_FLUX_COLUMNS}
    return raw_fluxes <= available or normalized_fluxes <= available


def _column_catalog(columns: Sequence[str]) -> list[dict[str, str]]:
    return [
        {"name": str(column), "category": classify_openuniverse_column(str(column))}
        for column in columns
    ]


def classify_openuniverse_column(column: str) -> str:
    """Classify an OpenUniverse column name for inventory reports."""
    lower = column.lower()
    if column == "galaxy_id" or lower.endswith("_id"):
        return "identifier"
    if column in {"ra", "dec"}:
        return "position"
    if "redshift" in lower or lower in {"z", "z_true"}:
        return "redshift"
    if "stellar_mass" in lower or column == OU_TRUTH_COLUMNS["stellar_mass"]:
        return "stellar_mass"
    if lower.startswith(("flux_truth_", "flux_")) or "_flux_" in lower:
        return "photometric_flux"
    if lower.startswith("fluxerr_"):
        return "photometric_error"
    if lower.startswith("mask_"):
        return "photometric_mask"
    if "sfr" in lower:
        return "sfr_or_sfh_proxy"
    if any(token in lower for token in ("sfh", "diffstar", "diffsky")):
        return "diffsky_or_sfh"
    if any(token in lower for token in ("dust", "atten", "av", "rv")):
        return "dust_or_extinction"
    if any(token in lower for token in ("metal", "lgmet")):
        return "metallicity"
    if any(token in lower for token in ("halo", "mah", "mvir", "rvir")):
        return "halo_or_mah"
    if any(token in lower for token in ("shear", "convergence", "kappa")):
        return "lensing"
    if any(token in lower for token in ("bulge", "disk", "knot", "size", "sersic")):
        return "morphology"
    if "velocity" in lower:
        return "velocity"
    return "other"
