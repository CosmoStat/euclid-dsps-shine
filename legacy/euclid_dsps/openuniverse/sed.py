"""Bounded readers for OpenUniverse low-resolution SED HDF5 files."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

OU_SED_COMPONENT_NAMES = ("disk", "bulge", "knot")


@dataclass(frozen=True)
class SedSample:
    """Small metadata sample for one SED dataset."""

    galaxy_id: int
    path: str
    shape: tuple[int, ...]
    dtype: str
    finite_fraction: float | None = None
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class SedInventory:
    """Inventory for one OpenUniverse SED HDF5 file."""

    path: str
    exists: bool
    file_size_bytes: int | None
    top_level_groups: tuple[str, ...]
    meta_datasets: tuple[dict[str, Any], ...]
    wavelength_dataset: str | None
    wavelength_size: int | None
    wavelength_min: float | None
    wavelength_max: float | None
    galaxy_prefix_count: int
    sed_dataset_count: int | None
    sample_datasets: tuple[SedSample, ...]


@dataclass(frozen=True)
class SedComponents:
    """SED arrays read for selected galaxies.

    ``sed`` has shape ``[N, C, L]`` where ``C`` is the number of stored
    components and ``L`` is the low-resolution wavelength axis length.
    """

    galaxy_id: np.ndarray
    wavelength: np.ndarray | None
    sed: np.ndarray
    component_names: tuple[str, ...]
    sed_paths: tuple[str, ...]


def inventory_openuniverse_sed(
    path: str | Path,
    *,
    sample_limit: int = 5,
    read_sample_values: bool = True,
) -> SedInventory:
    """Inventory an OpenUniverse SED HDF5 file without scanning all payloads."""
    sed_path = Path(path)
    if not sed_path.exists():
        return SedInventory(
            path=str(sed_path),
            exists=False,
            file_size_bytes=None,
            top_level_groups=(),
            meta_datasets=(),
            wavelength_dataset=None,
            wavelength_size=None,
            wavelength_min=None,
            wavelength_max=None,
            galaxy_prefix_count=0,
            sed_dataset_count=None,
            sample_datasets=(),
        )

    with h5py.File(sed_path, "r") as handle:
        top_level_groups = tuple(str(key) for key in handle.keys())
        meta_datasets = tuple(_inventory_meta_datasets(handle))
        wavelength_path, wavelength = _read_wavelength_axis(handle)
        galaxy_group = handle.get("galaxy")
        galaxy_prefix_count = len(galaxy_group) if isinstance(galaxy_group, h5py.Group) else 0
        sed_dataset_count = (
            _count_direct_child_datasets(galaxy_group)
            if isinstance(galaxy_group, h5py.Group)
            else None
        )
        samples = tuple(
            _sample_sed_datasets(
                galaxy_group,
                sample_limit=max(int(sample_limit), 0),
                read_values=bool(read_sample_values),
            )
            if isinstance(galaxy_group, h5py.Group)
            else ()
        )

    if wavelength is None or wavelength.size == 0:
        wavelength_size = None
        wavelength_min = None
        wavelength_max = None
    else:
        wavelength_size = int(wavelength.size)
        wavelength_min = float(np.nanmin(wavelength))
        wavelength_max = float(np.nanmax(wavelength))

    return SedInventory(
        path=str(sed_path),
        exists=True,
        file_size_bytes=int(sed_path.stat().st_size),
        top_level_groups=top_level_groups,
        meta_datasets=meta_datasets,
        wavelength_dataset=wavelength_path,
        wavelength_size=wavelength_size,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        galaxy_prefix_count=int(galaxy_prefix_count),
        sed_dataset_count=sed_dataset_count,
        sample_datasets=samples,
    )


def read_sed_components(
    path: str | Path,
    galaxy_ids: Sequence[int],
    *,
    missing: str = "raise",
) -> SedComponents:
    """Read low-resolution SED components for selected galaxy ids."""
    if missing not in {"raise", "skip"}:
        raise ValueError("missing must be 'raise' or 'skip'")

    selected_ids = tuple(int(galaxy_id) for galaxy_id in galaxy_ids)
    arrays: list[np.ndarray] = []
    found_ids: list[int] = []
    paths: list[str] = []
    wavelength: np.ndarray | None
    with h5py.File(path, "r") as handle:
        _, wavelength = _read_wavelength_axis(handle)
        for galaxy_id in selected_ids:
            dataset_path = find_sed_dataset_path(handle, galaxy_id)
            if dataset_path is None:
                if missing == "raise":
                    raise KeyError(
                        f"SED dataset for galaxy_id={galaxy_id} was not found in {path}"
                    )
                continue
            values = np.asarray(handle[dataset_path][()], dtype=np.float32)
            if values.ndim != 2:
                raise ValueError(
                    f"Expected SED dataset {dataset_path} to be rank 2 [C,L], "
                    f"got shape {values.shape}"
                )
            arrays.append(values)
            found_ids.append(galaxy_id)
            paths.append(dataset_path)

    if arrays:
        sed = np.stack(arrays, axis=0).astype(np.float32)
        component_names = component_names_for_count(int(sed.shape[1]))
    else:
        wave_len = int(0 if wavelength is None else wavelength.size)
        sed = np.empty((0, 0, wave_len), dtype=np.float32)
        component_names = ()

    return SedComponents(
        galaxy_id=np.asarray(found_ids, dtype=np.int64),
        wavelength=None if wavelength is None else np.asarray(wavelength, dtype=np.float32),
        sed=sed,
        component_names=component_names,
        sed_paths=tuple(paths),
    )


def find_sed_dataset_path(handle: h5py.File, galaxy_id: int) -> str | None:
    """Find the HDF5 dataset path for an OpenUniverse ``galaxy_id``."""
    root = handle.get("galaxy")
    if not isinstance(root, h5py.Group):
        return None
    galaxy_text = str(int(galaxy_id))
    candidate_prefixes = []
    if len(galaxy_text) > 5:
        candidate_prefixes.append(galaxy_text[:-5])
    if len(galaxy_text) > 9:
        candidate_prefixes.append(galaxy_text[:9])
    candidate_prefixes.append(galaxy_text)
    for prefix in dict.fromkeys(candidate_prefixes):
        candidate = f"galaxy/{prefix}/{galaxy_text}"
        if candidate in handle:
            return candidate

    # Fallback: the real files use only tens of prefix groups per HEALPix.
    for prefix in root.keys():
        group = root[prefix]
        if isinstance(group, h5py.Group) and galaxy_text in group:
            return f"galaxy/{prefix}/{galaxy_text}"
    return None


def component_names_for_count(count: int) -> tuple[str, ...]:
    """Return component labels for an SED dataset with ``count`` rows."""
    count = int(count)
    if count == len(OU_SED_COMPONENT_NAMES):
        return OU_SED_COMPONENT_NAMES
    return tuple(f"component_{index}" for index in range(count))


def sed_inventory_to_dict(inventory: SedInventory) -> dict[str, Any]:
    """Convert a SED inventory to JSON-compatible primitives."""
    payload = asdict(inventory)
    payload["sample_datasets"] = [asdict(sample) for sample in inventory.sample_datasets]
    return payload


def _inventory_meta_datasets(handle: h5py.File) -> list[dict[str, Any]]:
    meta = handle.get("meta")
    if not isinstance(meta, h5py.Group):
        return []

    rows: list[dict[str, Any]] = []

    def visit(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            rows.append(
                {
                    "path": f"meta/{name}",
                    "shape": tuple(int(value) for value in obj.shape),
                    "dtype": str(obj.dtype),
                }
            )

    meta.visititems(visit)
    return rows


def _read_wavelength_axis(handle: h5py.File) -> tuple[str | None, np.ndarray | None]:
    for candidate in (
        "meta/wave_list",
        "meta/wavelength",
        "meta/wavelengths",
        "meta/wave",
    ):
        if candidate in handle and isinstance(handle[candidate], h5py.Dataset):
            return candidate, np.asarray(handle[candidate][()], dtype=np.float32)
    return None, None


def _count_direct_child_datasets(galaxy_group: h5py.Group) -> int:
    count = 0
    for prefix in galaxy_group.keys():
        group = galaxy_group[prefix]
        if isinstance(group, h5py.Group):
            count += len(group)
    return int(count)


def _sample_sed_datasets(
    galaxy_group: h5py.Group,
    *,
    sample_limit: int,
    read_values: bool,
) -> list[SedSample]:
    samples: list[SedSample] = []
    if sample_limit <= 0:
        return samples
    for prefix in galaxy_group.keys():
        group = galaxy_group[prefix]
        if not isinstance(group, h5py.Group):
            continue
        for galaxy_id in group.keys():
            obj = group[galaxy_id]
            if not isinstance(obj, h5py.Dataset):
                continue
            finite_fraction = None
            min_value = None
            max_value = None
            if read_values:
                values = np.asarray(obj[()], dtype=float)
                finite = np.isfinite(values)
                finite_fraction = float(np.mean(finite)) if values.size else None
                if finite.any():
                    min_value = float(np.nanmin(values))
                    max_value = float(np.nanmax(values))
            samples.append(
                SedSample(
                    galaxy_id=int(galaxy_id),
                    path=str(obj.name).lstrip("/"),
                    shape=tuple(int(value) for value in obj.shape),
                    dtype=str(obj.dtype),
                    finite_fraction=finite_fraction,
                    min_value=min_value,
                    max_value=max_value,
                )
            )
            if len(samples) >= sample_limit:
                return samples
    return samples
