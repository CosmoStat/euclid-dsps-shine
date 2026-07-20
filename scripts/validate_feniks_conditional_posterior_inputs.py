#!/usr/bin/env python3
"""Validate the fixed catalog and prior contract for the posterior matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from euclid_dsps.prior_learning.spline15d_schema import SPLINE15D_PARAMETER_NAMES


def validate_inputs(catalog_dir: Path, prior_checkpoint: Path) -> None:
    catalog_dir = Path(catalog_dir)
    prior_checkpoint = Path(prior_checkpoint)
    contract_path = catalog_dir / "amortized_catalog_contract.json"
    sidecar_path = prior_checkpoint.with_suffix(prior_checkpoint.suffix + ".json")
    for path in (contract_path, prior_checkpoint, sidecar_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing required input: {path}")

    contract = _read_json(contract_path)
    _require(contract.get("version") == 1, "catalog contract version must be 1")
    _require(
        contract.get("truth_kind") == "exact_spline15d",
        "catalog truth_kind must be exact_spline15d",
    )
    _require(contract.get("join_key") == "object_id", "join_key must be object_id")
    _require(
        tuple(contract.get("parameter_names", ())) == SPLINE15D_PARAMETER_NAMES,
        "catalog parameter order does not match spline15d",
    )

    for split in ("train", "test"):
        path = catalog_dir / f"{split}.parquet"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing required input: {path}")
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        missing = [
            name
            for name in ("object_id", *SPLINE15D_PARAMETER_NAMES)
            if name not in columns
        ]
        _require(not missing, f"{split} parquet missing columns: {missing}")
        rows = int(parquet.metadata.num_rows)
        _require(rows > 0, f"{split} parquet is empty")
        recorded = ((contract.get("splits") or {}).get(split) or {}).get("rows")
        _require(recorded == rows, f"{split} row count mismatch: {recorded} != {rows}")

    sidecar = _read_json(sidecar_path)
    _require(sidecar.get("version") == 1, "prior checkpoint version must be 1")
    _require(
        tuple(sidecar.get("parameter_names", ())) == SPLINE15D_PARAMETER_NAMES,
        "prior parameter order does not match spline15d",
    )
    architecture = sidecar.get("architecture") or {}
    _require(
        int(architecture.get("latent_dim", -1)) == len(SPLINE15D_PARAMETER_NAMES),
        "prior latent dimension does not match spline15d",
    )
    _require(
        (sidecar.get("flow_integrity") or {}).get("status") == "PASS",
        "prior flow integrity status is not PASS",
    )
    print(
        "[contract] valid: "
        f"train={pq.ParquetFile(catalog_dir / 'train.parquet').metadata.num_rows} "
        f"test={pq.ParquetFile(catalog_dir / 'test.parquet').metadata.num_rows} "
        f"prior={prior_checkpoint}"
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--prior-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    validate_inputs(args.catalog_dir, args.prior_checkpoint)


if __name__ == "__main__":
    main()
