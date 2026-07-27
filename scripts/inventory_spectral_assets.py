#!/usr/bin/env python3
"""Inventory PopCosmos spectral HDF5 assets without loading dense tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

DEFAULT_ASSETS = (
    "Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5",
    "Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5",
    "Data/popcosmos_chabrier_gas_ssp_grid.h5",
    "Data/popcosmos_chabrier_agn_component_ssp_grid.h5",
)
DEFAULT_OUT = "outputs/ssp_compression/baseline_asset_inventory.json"
MAIN_DATASET_CANDIDATES = (
    "ssp_flux",
    "agn_lnu_per_mformed",
    "gas_coeff",
    "agn_coeff",
    "gas_basis",
    "agn_basis",
    "template_lnu_per_lbol",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "assets",
        nargs="*",
        default=list(DEFAULT_ASSETS),
        help="HDF5 assets to inspect. Defaults to active PopCosmos-like assets.",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSON path.")
    parser.add_argument(
        "--plot",
        default="",
        help="Optional PNG path for a file-size/resident-payload summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets = [inspect_hdf5_asset(Path(path)) for path in args.assets]
    result = {
        "assets": assets,
        "totals": {
            "file_bytes": int(sum(asset["file_bytes"] for asset in assets)),
            "logical_dataset_bytes": int(
                sum(asset["logical_dataset_bytes"] for asset in assets)
            ),
            "main_dataset_bytes": int(
                sum(asset.get("main_dataset_bytes", 0) for asset in assets)
            ),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.plot:
        write_size_plot(assets, Path(args.plot))
    print(f"wrote {out}")
    return 0


def inspect_hdf5_asset(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    if not expanded.exists():
        return {
            "path": str(expanded),
            "exists": False,
            "file_bytes": 0,
            "logical_dataset_bytes": 0,
            "main_dataset": "",
            "main_dataset_bytes": 0,
            "datasets": {},
            "attrs": {},
        }
    datasets: dict[str, Any] = {}
    attrs: dict[str, Any]
    with h5py.File(expanded, "r") as handle:
        attrs = {
            str(key): jsonable_hdf5_value(value) for key, value in handle.attrs.items()
        }

        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            logical_bytes = int(obj.size * obj.dtype.itemsize)
            datasets[name] = {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "logical_bytes": logical_bytes,
                "compression": obj.compression or "none",
                "compression_opts": jsonable_hdf5_value(obj.compression_opts),
                "chunks": None if obj.chunks is None else list(obj.chunks),
                "shuffle": bool(obj.shuffle),
            }

        handle.visititems(visit)
    main_name = main_dataset_name(datasets)
    return {
        "path": str(expanded),
        "exists": True,
        "file_bytes": int(expanded.stat().st_size),
        "logical_dataset_bytes": int(
            sum(item["logical_bytes"] for item in datasets.values())
        ),
        "main_dataset": main_name,
        "main_dataset_bytes": int(datasets.get(main_name, {}).get("logical_bytes", 0)),
        "asset_kind": attrs.get("asset_kind", ""),
        "imf_name": attrs.get("imf_name", ""),
        "imf_type": attrs.get("imf_type", ""),
        "z_sun": attrs.get("z_sun", ""),
        "attrs": attrs,
        "datasets": datasets,
    }


def main_dataset_name(datasets: dict[str, Any]) -> str:
    for name in MAIN_DATASET_CANDIDATES:
        if name in datasets:
            return name
    return next(iter(datasets), "")


def jsonable_hdf5_value(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [jsonable_hdf5_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def write_size_plot(assets: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plot") from exc
    labels = [Path(asset["path"]).name for asset in assets]
    file_mib = [asset["file_bytes"] / 1024**2 for asset in assets]
    main_mib = [asset["main_dataset_bytes"] / 1024**2 for asset in assets]
    x = np.arange(len(assets))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.bar(x - width / 2, file_mib, width, label="HDF5 file")
    ax.bar(x + width / 2, main_mib, width, label="main tensor payload")
    ax.set_yscale("log")
    ax.set_ylabel("MiB")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_title("Spectral asset storage and dense resident payload")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
