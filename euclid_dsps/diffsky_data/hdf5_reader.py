"""HDF5 inspection and column loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

INTERESTING_KEYWORDS = (
    "redshift",
    "logsm",
    "logssfr",
    "sfr",
    "logmp",
    "central",
    "r50",
    "diffmah",
    "diffstar",
    "lgmcrit",
    "lgy_at_mcrit",
    "indx",
    "lg_qt",
    "qlglgdt",
    "dust",
    "av",
    "delta",
    "metal",
    "burst",
    "flux",
    "mag",
    "lsst",
    "roman",
    "sed",
    "wave",
)


def inspect_hdf5_file(path: str | Path, sample_size: int = 1024) -> dict[str, Any]:
    file_path = Path(path)
    report: dict[str, Any] = {
        "path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "attrs": {},
        "groups": [],
        "datasets": [],
    }
    with h5py.File(file_path, "r") as handle:
        report["attrs"] = _attrs_to_dict(handle.attrs)

        def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Group):
                report["groups"].append(name)
                return
            if isinstance(obj, h5py.Dataset):
                item = {
                    "name": name,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "attrs": _attrs_to_dict(obj.attrs),
                    "interesting": _is_interesting(name),
                }
                item.update(_sample_stats(obj, sample_size=sample_size))
                report["datasets"].append(item)

        handle.visititems(visit)
    return report


def load_hdf5_columns(
    path: str | Path, columns: list[str], limit: int | None = None
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        for column in columns:
            dataset = handle[column]
            if limit is None:
                out[column] = dataset[:]
            else:
                out[column] = dataset[: max(int(limit), 0)]
    return out


def _sample_stats(dataset: h5py.Dataset, sample_size: int) -> dict[str, Any]:
    try:
        if dataset.shape == ():
            sample = np.asarray([dataset[()]])
        else:
            first_dim = min(int(dataset.shape[0]), int(sample_size))
            slices = (slice(0, first_dim),) + tuple(
                slice(0, min(int(dim), 4)) for dim in dataset.shape[1:]
            )
            sample = np.asarray(dataset[slices])
    except Exception as exc:
        return {"sample_error": str(exc)}
    if np.issubdtype(sample.dtype, np.number):
        flat = sample.reshape(-1)
        finite = np.isfinite(flat)
        stats: dict[str, Any] = {
            "finite_fraction": float(finite.mean()) if finite.size else None
        }
        if finite.any():
            vals = flat[finite]
            stats.update(
                {
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "mean": float(np.mean(vals)),
                }
            )
        return stats
    return {"sample": [str(value) for value in sample.reshape(-1)[:5].tolist()]}


def _attrs_to_dict(attrs: h5py.AttributeManager) -> dict[str, Any]:
    out = {}
    for key, value in attrs.items():
        if isinstance(value, bytes):
            out[key] = value.decode("utf-8", errors="replace")
        elif hasattr(value, "tolist"):
            raw = value.tolist()
            if isinstance(raw, list):
                raw = [
                    (
                        item.decode("utf-8", errors="replace")
                        if isinstance(item, bytes)
                        else item
                    )
                    for item in raw
                ]
            out[key] = raw
        else:
            out[key] = value
    return out


def _is_interesting(name: str) -> bool:
    lower = name.lower()
    return any(keyword in lower for keyword in INTERESTING_KEYWORDS)
