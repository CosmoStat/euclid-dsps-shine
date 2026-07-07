#!/usr/bin/env python3
"""Validate compressed spectral assets without loading dense source grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
from build_compressed_agn_component_grid import (
    validate_compressed_grid as validate_agn_grid,
)
from build_compressed_gas_grid import validate_compressed_grid as validate_gas_grid
from build_compressed_ssp_grid import validate_compressed_grid as validate_ssp_grid
from fsps_grid_common import FspsGridError, fail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Compressed HDF5 asset to validate.")
    parser.add_argument(
        "--kind",
        choices=[
            "auto",
            "compressed_agn_component",
            "compressed_gas_grid",
            "compressed_stellar_ssp",
        ],
        default="auto",
        help="Compressed asset kind. auto reads HDF5 asset_kind.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        kind = args.kind
        if kind == "auto":
            kind = infer_kind(Path(args.path))
        if kind == "compressed_agn_component":
            summary = validate_agn_grid(args.path)
        elif kind == "compressed_gas_grid":
            summary = validate_gas_grid(args.path)
        elif kind == "compressed_stellar_ssp":
            summary = validate_ssp_grid(args.path)
        else:  # pragma: no cover - argparse and infer_kind constrain this
            raise FspsGridError(f"Unsupported compressed asset kind: {kind}")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except FspsGridError as exc:
        return fail(str(exc))


def infer_kind(path: Path) -> str:
    if not path.exists():
        raise FspsGridError(f"Compressed asset not found: {path}")
    with h5py.File(path, "r") as handle:
        asset_kind = str(handle.attrs.get("asset_kind", ""))
    if asset_kind == "popcosmos_chabrier_compressed_agn_component_grid":
        return "compressed_agn_component"
    if asset_kind == "popcosmos_chabrier_compressed_gas_grid":
        return "compressed_gas_grid"
    if asset_kind == "popcosmos_chabrier_compressed_stellar_ssp":
        return "compressed_stellar_ssp"
    raise FspsGridError(
        f"Could not infer compressed asset kind from {path}: asset_kind={asset_kind!r}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
