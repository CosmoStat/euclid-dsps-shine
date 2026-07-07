#!/usr/bin/env python3
"""Estimate dense/compressed spectral payloads and batch VRAM pressure safely.

This script is metadata-first: by default it reads HDF5 dataset shapes and does
not instantiate `euclid_dsps.model.load_context`, so it does not copy multi-GiB
grids to GPU. Use it to choose safe batch sizes before running real GPU fits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from euclid_dsps.config import load_config

DEFAULT_OUT = "outputs/ssp_compression/baseline_vram_profile.json"
FLOAT32_BYTES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/popcosmos_binned.yaml",
        help="Config whose spectral assets should be profiled.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50, 100, 200, 500],
        help="Batch sizes for approximate intermediate-memory estimates.",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSON path.")
    parser.add_argument(
        "--compressed-gas-grid",
        default=None,
        help="Temporarily profile this compressed gas asset.",
    )
    parser.add_argument(
        "--compressed-agn-component-grid",
        default=None,
        help="Temporarily profile this compressed AGN component asset.",
    )
    parser.add_argument(
        "--query-jax-devices",
        action="store_true",
        help="Import JAX and report visible devices. Does not load model context.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config = apply_compressed_overrides(config, args)
    model = config.get("model", {}) or {}
    paths = spectral_paths(config)
    assets = {name: inspect_dataset_shapes(path) for name, path in paths.items() if path}
    axes = infer_primary_axes(assets)
    profile = {
        "config": str(args.config),
        "paths": {name: str(path) for name, path in paths.items() if path},
        "assets": assets,
        "static_resident_estimate": estimate_static_resident_bytes(assets, model),
        "batch_estimates": [
            estimate_batch_bytes(batch_size, axes) for batch_size in args.batch_sizes
        ],
    }
    if args.query_jax_devices:
        profile["jax_devices"] = query_jax_devices()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


def apply_compressed_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    model = config.get("model", {}) or {}
    if args.compressed_gas_grid:
        model["nebular_model"] = "compressed_gas_grid"
        model["compressed_gas_grid_path"] = str(args.compressed_gas_grid)
    if args.compressed_agn_component_grid:
        model["agn_model"] = "compressed_fsps_component_grid"
        model["compressed_agn_component_grid_path"] = str(
            args.compressed_agn_component_grid
        )
    return config


def spectral_paths(config: dict[str, Any]) -> dict[str, Path | None]:
    model = config.get("model", {}) or {}
    return {
        "base_ssp": path_or_none(config.get("ssp_path")),
        "stellar_only_ssp": path_or_none(model.get("stellar_only_ssp_path")),
        "gas_grid": path_or_none(model.get("gas_grid_path")),
        "compressed_gas_grid": path_or_none(model.get("compressed_gas_grid_path")),
        "agn_template": path_or_none(model.get("agn_template_path")),
        "agn_component": path_or_none(model.get("agn_component_grid_path")),
        "compressed_agn_component": path_or_none(
            model.get("compressed_agn_component_grid_path")
        ),
    }


def path_or_none(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def inspect_dataset_shapes(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path), "datasets": {}, "file_bytes": 0}
    datasets: dict[str, Any] = {}
    attrs: dict[str, Any]
    with h5py.File(path, "r") as handle:
        attrs = {str(key): jsonable(value) for key, value in handle.attrs.items()}

        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            datasets[name] = {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "logical_bytes": int(obj.size * obj.dtype.itemsize),
                "float32_resident_bytes": int(obj.size * FLOAT32_BYTES),
                "compression": obj.compression or "none",
                "chunks": None if obj.chunks is None else list(obj.chunks),
            }

        handle.visititems(visit)
    return {
        "exists": True,
        "path": str(path),
        "file_bytes": int(path.stat().st_size),
        "asset_kind": attrs.get("asset_kind", ""),
        "datasets": datasets,
    }


def estimate_static_resident_bytes(
    assets: dict[str, dict[str, Any]], model: dict[str, Any]
) -> dict[str, Any]:
    components: dict[str, int] = {}
    components["base_ssp"] = dataset_float32_bytes(assets, "base_ssp", "ssp_flux")
    if model.get("nebular_model") == "gas_grid":
        components["gas_grid"] = dataset_float32_bytes(assets, "gas_grid", "ssp_flux")
    elif model.get("nebular_model") == "compressed_gas_grid":
        components["compressed_gas_basis"] = dataset_float32_bytes(
            assets, "compressed_gas_grid", "gas_basis"
        )
        components["compressed_gas_coeff"] = dataset_float32_bytes(
            assets, "compressed_gas_grid", "gas_coeff"
        )
        components["compressed_gas_scale"] = dataset_float32_bytes(
            assets, "compressed_gas_grid", "gas_scale"
        )
    if model.get("agn_model") == "template_grid":
        components["agn_template"] = dataset_float32_bytes(
            assets, "agn_template", "template_lnu_per_lbol"
        )
    elif model.get("agn_model") == "fsps_component_grid":
        components["agn_component"] = dataset_float32_bytes(
            assets, "agn_component", "agn_lnu_per_mformed"
        )
    elif model.get("agn_model") == "compressed_fsps_component_grid":
        components["compressed_agn_basis"] = dataset_float32_bytes(
            assets, "compressed_agn_component", "agn_basis"
        )
        components["compressed_agn_coeff"] = dataset_float32_bytes(
            assets, "compressed_agn_component", "agn_coeff"
        )
        components["compressed_agn_scale"] = dataset_float32_bytes(
            assets, "compressed_agn_component", "agn_scale"
        )
    total = int(sum(components.values()))
    return {
        "components": components,
        "total_bytes": total,
        "total_mib": total / 1024**2,
        "note": (
            "Metadata estimate for resident float32 spectral arrays. It excludes "
            "JAX allocator overhead, compiled executables, optimizer state, and "
            "per-batch intermediates."
        ),
    }


def dataset_float32_bytes(
    assets: dict[str, dict[str, Any]], asset_name: str, dataset_name: str
) -> int:
    dataset = (
        assets.get(asset_name, {})
        .get("datasets", {})
        .get(dataset_name, {})
    )
    return int(dataset.get("float32_resident_bytes", 0))


def infer_primary_axes(assets: dict[str, dict[str, Any]]) -> dict[str, int]:
    for asset in assets.values():
        datasets = asset.get("datasets", {})
        if "ssp_flux" in datasets:
            shape = datasets["ssp_flux"]["shape"]
            if len(shape) == 3:
                return {"n_age": int(shape[1]), "n_wave": int(shape[2])}
            if len(shape) == 5:
                return {"n_age": int(shape[3]), "n_wave": int(shape[4])}
        if "agn_lnu_per_mformed" in datasets:
            shape = datasets["agn_lnu_per_mformed"]["shape"]
            return {"n_age": int(shape[3]), "n_wave": int(shape[4])}
        if "agn_basis" in datasets:
            shape = datasets["agn_basis"]["shape"]
            return {"n_age": 107, "n_wave": int(shape[1])}
        if "gas_basis" in datasets:
            shape = datasets["gas_basis"]["shape"]
            return {"n_age": 107, "n_wave": int(shape[1])}
    return {"n_age": 0, "n_wave": 0}


def estimate_batch_bytes(batch_size: int, axes: dict[str, int]) -> dict[str, Any]:
    n_age = int(axes.get("n_age", 0))
    n_wave = int(axes.get("n_wave", 0))
    sed_by_age = batch_size * n_age * n_wave * FLOAT32_BYTES
    spectra = batch_size * n_wave * FLOAT32_BYTES
    total = int(sed_by_age + 6 * spectra)
    return {
        "batch_size": int(batch_size),
        "approx_sed_by_age_bytes": int(sed_by_age),
        "approx_wave_spectra_bytes": int(6 * spectra),
        "approx_total_bytes": total,
        "approx_total_mib": total / 1024**2,
        "note": "Approximate major intermediates only; use GPU measurement for final sizing.",
    }


def query_jax_devices() -> list[dict[str, str]]:
    import jax

    return [
        {
            "platform": str(getattr(device, "platform", "")),
            "device_kind": str(getattr(device, "device_kind", "")),
            "repr": str(device),
        }
        for device in jax.devices()
    ]


def jsonable(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
