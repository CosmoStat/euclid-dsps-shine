#!/usr/bin/env python3
"""Validate COSMOS2020 assets, nested subsets, config, and optional run output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from euclid_dsps.config import load_config
from euclid_dsps.cosmos2020 import COSMOS_BANDS
from euclid_dsps.filters import load_filters
from euclid_dsps.observation_arrays import photometry_arrays_from_dataframe

SPECTRAL_ASSET_SHA256 = {
    "Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5": (
        "706a9e8442b26887b47c4b5bc3e59d1e680d5a245493b3118e58f1f3bb959b4c"
    ),
    "Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5": (
        "18a5b3047b1a4da67ec216e76223efcf3f300f6d8f0e8fd8af5cc54044ca2206"
    ),
    "Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5": (
        "1a9476123b89ae0aa148bddb1cf5d72c33919f6daa54a91cbaa9257a254a10a4"
    ),
    "Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5": (
        "a436c5a3692a3858962b49e194237894dd143c8776fd8f713388583317802a95"
    ),
    "Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5": (
        "4fefc42c3faf73d8b8055b1a68b5d71302a947b33d03c4b192ed8bf3d16c42fb"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/popcosmos_a24_rws_joint.yaml"),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--expected-full", type=int)
    return parser.parse_args()


def validate_spectral_assets(expected: dict[str, str] = SPECTRAL_ASSET_SHA256) -> None:
    for raw_path, expected_sha256 in expected.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing DSPS spectral asset: {path}. Transfer the generated "
                "assets documented in docs/popcosmos_a24_reproduction.md."
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {expected_sha256}, "
                f"found {digest}"
            )


def main() -> None:
    args = parse_args()
    validate_spectral_assets()
    manifest = json.loads(
        (args.data_dir / "preparation_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["n_bands"] != 26:
        raise ValueError("Preparation manifest does not declare 26 bands")
    full = args.data_dir / "farmer_a24_full.parquet"
    n_full = pq.ParquetFile(full).metadata.num_rows
    if args.expected_full is not None and n_full != args.expected_full:
        raise ValueError(f"Expected {args.expected_full} rows, found {n_full}")
    previous: set[int] = set()
    for size in (512, 5_000, 20_000, 40_000):
        path = args.data_dir / f"farmer_a24_n{min(size, n_full)}.parquet"
        if not path.exists():
            continue
        ids = set(pq.read_table(path, columns=["object_id"])["object_id"].to_pylist())
        if previous and not previous.issubset(ids):
            raise ValueError(f"Subset nesting failed at {path}")
        previous = ids

    config = load_config(args.config)
    config["catalog_path"] = str(full)
    for band in config["bands"]:
        band["filter"]["path"] = str(
            args.asset_dir / "filters" / f"{band['name']}.dat"
        )
    expected_names = tuple(band.name for band in COSMOS_BANDS)
    names = tuple(band["name"] for band in config["bands"])
    if names != expected_names:
        raise ValueError("Config band order does not match the public A24 order")
    filters = load_filters(config["bands"])
    if any(len(curve.wave) < 2 for curve in filters.values()):
        raise ValueError("At least one filter curve is empty")
    frame = pq.read_table(full).slice(0, min(n_full, 8)).to_pandas()
    arrays = photometry_arrays_from_dataframe(
        frame, config["bands"], object_id_column="object_id"
    )
    if arrays.flux.shape != (len(frame), 26):
        raise ValueError(f"Unexpected photometry shape: {arrays.flux.shape}")
    if not np.all(np.isfinite(arrays.flux_err[arrays.mask])):
        raise ValueError("Usable photometric errors are not finite")

    if args.run_dir is not None:
        checkpoint = args.run_dir / "train/checkpoints/best.eqx"
        gate = args.run_dir / "train/training_collapse_gate.json"
        done = args.run_dir / "DONE"
        for path in (checkpoint, gate, done):
            if not path.exists():
                raise FileNotFoundError(path)
        status = json.loads(gate.read_text(encoding="utf-8"))["status"]
        if status == "FAIL":
            raise RuntimeError(f"Training collapse gate failed in {args.run_dir}")
    print(
        f"[cosmos2020-contract] valid rows={n_full} bands=26 "
        f"run={'checked' if args.run_dir else 'not-requested'}"
    )


if __name__ == "__main__":
    main()
