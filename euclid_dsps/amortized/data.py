"""FS2 photometry batches for amortized inference."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from euclid_dsps.io import (
    iter_catalog_batches,
    required_catalog_columns,
    truth_column_from_spec,
)
from euclid_dsps.observation_arrays import (
    PhotometryArrays,
    photometry_arrays_from_dataframe,
    validate_fs2_band_contract,
)

from .config import amortized_config
from .features import FeatureStats, compute_feature_stats, make_encoder_features


@dataclass(frozen=True)
class PhotometryBatch:
    object_id: jnp.ndarray
    flux: jnp.ndarray
    flux_err: jnp.ndarray
    mask: jnp.ndarray
    features: jnp.ndarray
    row_index: np.ndarray | None = None
    truth_theta: jnp.ndarray | None = None


def compute_feature_stats_from_config(
    config: dict[str, Any],
    *,
    limit: int | None = None,
    batch_size: int = 10_000,
    row_indices: np.ndarray | None = None,
) -> FeatureStats:
    """Compute feature stats from configured catalog photometry."""
    band_names = tuple(str(band["name"]) for band in config["bands"])
    features_cfg = amortized_config(config)["features"]
    flux_chunks = []
    err_chunks = []
    mask_chunks = []
    for arrays in iter_photometry_arrays_from_config(
        config,
        batch_size=batch_size,
        limit=limit,
        row_indices=row_indices,
    ):
        flux_chunks.append(arrays.flux)
        err_chunks.append(arrays.flux_err)
        mask_chunks.append(arrays.mask)
    if not flux_chunks:
        raise ValueError("No FS2 rows available to compute feature stats")
    return compute_feature_stats(
        np.concatenate(flux_chunks, axis=0),
        np.concatenate(err_chunks, axis=0),
        np.concatenate(mask_chunks, axis=0),
        band_names=band_names,
        flux_transform=str(features_cfg.get("flux_transform", "asinh")),
        append_mask=bool(features_cfg.get("append_mask", False)),
        error_epsilon=float(features_cfg.get("error_epsilon", 1.0e-6)),
    )


def compute_fs2_feature_stats_from_config(
    config: dict[str, Any],
    *,
    limit: int | None = None,
    batch_size: int = 10_000,
    row_indices: np.ndarray | None = None,
) -> FeatureStats:
    """Compute feature stats from configured FS2 photometry."""
    validate_fs2_band_contract(config["bands"])
    return compute_feature_stats_from_config(
        config,
        limit=limit,
        batch_size=batch_size,
        row_indices=row_indices,
    )


def iter_photometry_arrays_from_config(
    config: dict[str, Any],
    *,
    batch_size: int,
    limit: int | None,
    row_indices: np.ndarray | None = None,
) -> Iterator[PhotometryArrays]:
    """Yield raw NumPy photometry arrays from a generic configured catalog."""
    columns = required_catalog_columns(config)
    id_column = _object_id_column_from_config(config)
    if id_column and id_column not in columns:
        columns.append(id_column)
    row_index_set = (
        None if row_indices is None else set(np.asarray(row_indices, dtype=int))
    )
    for frame in iter_catalog_batches(
        config["catalog_path"],
        columns=columns,
        batch_size=batch_size,
        limit=limit,
        row_indices=row_index_set,
    ):
        arrays = photometry_arrays_from_dataframe(
            frame,
            config["bands"],
            object_id_column=id_column,
        )
        truth = _truth_arrays_from_frame(frame, config)
        yield replace(arrays, truth=truth or None)


def iter_fs2_photometry_arrays_from_config(
    config: dict[str, Any],
    *,
    batch_size: int,
    limit: int | None,
    row_indices: np.ndarray | None = None,
) -> Iterator:
    """Yield raw NumPy FS2 photometry arrays from a config."""
    validate_fs2_band_contract(config["bands"])
    yield from iter_photometry_arrays_from_config(
        config,
        batch_size=batch_size,
        limit=limit,
        row_indices=row_indices,
    )


def load_fs2_photometry_arrays_from_config(
    config: dict[str, Any],
    *,
    batch_size: int,
    limit: int | None = None,
    row_indices: np.ndarray | None = None,
) -> PhotometryArrays:
    """Load selected FS2 photometry into one compact array block."""
    validate_fs2_band_contract(config["bands"])
    return load_photometry_arrays_from_config(
        config,
        batch_size=batch_size,
        limit=limit,
        row_indices=row_indices,
    )


def load_photometry_arrays_from_config(
    config: dict[str, Any],
    *,
    batch_size: int,
    limit: int | None = None,
    row_indices: np.ndarray | None = None,
) -> PhotometryArrays:
    """Load selected generic photometry into one compact array block."""
    chunks = list(
        iter_photometry_arrays_from_config(
            config,
            batch_size=batch_size,
            limit=limit,
            row_indices=row_indices,
        )
    )
    if not chunks:
        raise ValueError("No FS2 photometry rows were selected")
    if len(chunks) == 1:
        return chunks[0]
    return PhotometryArrays(
        object_id=np.concatenate([chunk.object_id for chunk in chunks], axis=0),
        row_index=_concat_row_indices(chunks),
        flux=np.concatenate([chunk.flux for chunk in chunks], axis=0),
        flux_err=np.concatenate([chunk.flux_err for chunk in chunks], axis=0),
        mask=np.concatenate([chunk.mask for chunk in chunks], axis=0),
        band_names=chunks[0].band_names,
        truth=_concatenate_truth(chunks),
    )


def iter_fs2_photometry_batches_from_config(
    config: dict[str, Any],
    batch_size: int,
    limit: int | None,
    feature_stats: FeatureStats | None,
    row_indices: np.ndarray | None = None,
) -> Iterator[PhotometryBatch]:
    """Yield JAX-ready FS2 photometry batches."""
    validate_fs2_band_contract(config["bands"])
    yield from iter_photometry_batches_from_config(
        config,
        batch_size=batch_size,
        limit=limit,
        feature_stats=feature_stats,
        row_indices=row_indices,
    )


def iter_photometry_batches_from_config(
    config: dict[str, Any],
    batch_size: int,
    limit: int | None,
    feature_stats: FeatureStats | None,
    row_indices: np.ndarray | None = None,
) -> Iterator[PhotometryBatch]:
    """Yield JAX-ready generic photometry batches."""
    stats = feature_stats
    if stats is None:
        stats = compute_feature_stats_from_config(
            config,
            limit=limit,
            batch_size=batch_size,
            row_indices=row_indices,
        )
    for arrays in iter_photometry_arrays_from_config(
        config,
        batch_size=batch_size,
        limit=limit,
        row_indices=row_indices,
    ):
        yield from iter_photometry_batches_from_arrays(
            arrays,
            batch_size=int(batch_size),
            feature_stats=stats,
        )


def iter_photometry_batches_from_arrays(
    arrays: PhotometryArrays,
    *,
    batch_size: int,
    feature_stats: FeatureStats,
    order: np.ndarray | None = None,
    truth_names: tuple[str, ...] | None = None,
) -> Iterator[PhotometryBatch]:
    """Yield JAX-ready batches from an in-memory photometry array block."""
    n_rows = int(arrays.flux.shape[0])
    if order is None:
        order = np.arange(n_rows)
    else:
        order = np.asarray(order, dtype=int)
    for start in range(0, len(order), int(batch_size)):
        idx = order[start : start + int(batch_size)]
        features = make_encoder_features(
            jnp.asarray(arrays.flux[idx], dtype=jnp.float32),
            jnp.asarray(arrays.flux_err[idx], dtype=jnp.float32),
            feature_stats,
            jnp.asarray(arrays.mask[idx]),
        )
        yield PhotometryBatch(
            object_id=np.asarray(arrays.object_id[idx], dtype=np.int64),
            flux=jnp.asarray(arrays.flux[idx], dtype=jnp.float32),
            flux_err=jnp.asarray(arrays.flux_err[idx], dtype=jnp.float32),
            mask=jnp.asarray(arrays.mask[idx]),
            features=features,
            row_index=np.asarray(_row_index_array(arrays)[idx], dtype=np.int64),
            truth_theta=_truth_theta_batch(arrays, idx, truth_names),
        )


def _truth_arrays_from_frame(frame, config: dict[str, Any]) -> dict[str, np.ndarray]:
    truth_cfg = config.get("truth", {}) or {}
    result: dict[str, np.ndarray] = {}
    for name, spec in (truth_cfg.get("parameter_columns") or {}).items():
        column = truth_column_from_spec(spec)
        if column and column in frame:
            result[str(name)] = _transform_truth_array(
                frame[column].to_numpy(dtype=float), spec
            )
    return result


def _transform_truth_array(values: np.ndarray, spec: Any) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not isinstance(spec, dict):
        return values.astype(np.float32)
    transform = spec.get("transform")
    if transform == "log10":
        positive = np.isfinite(values) & (values > 0.0)
        transformed = np.full(values.shape, np.nan, dtype=float)
        transformed[positive] = np.log10(values[positive])
        values = transformed
    elif transform == "log_stellar_mass_h2_to_msun":
        h = float(spec.get("h", np.nan))
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError("truth log_stellar_mass_h2_to_msun transform needs h > 0")
        values = values + 2.0 * np.log10(h)
    elif transform not in {None, "linear"}:
        raise ValueError(f"Unsupported truth transform: {transform}")
    values = values * float(spec.get("scale", 1.0))
    values = values + float(spec.get("offset", 0.0))
    return values.astype(np.float32)


def _concatenate_truth(chunks: list[PhotometryArrays]) -> dict[str, np.ndarray] | None:
    if not chunks or any(chunk.truth is None for chunk in chunks):
        return None
    names = tuple(chunks[0].truth or {})
    if any(tuple(chunk.truth or {}) != names for chunk in chunks):
        raise ValueError("Photometry truth columns differ across catalog chunks")
    return {
        name: np.concatenate([chunk.truth[name] for chunk in chunks], axis=0)
        for name in names
    }


def _truth_theta_batch(
    arrays: PhotometryArrays,
    indices: np.ndarray,
    names: tuple[str, ...] | None,
) -> jnp.ndarray | None:
    if names is None or arrays.truth is None:
        return None
    missing = [name for name in names if name not in arrays.truth]
    if missing:
        raise ValueError("Missing NPE truth columns: " + ", ".join(missing))
    truth = np.stack([arrays.truth[name][indices] for name in names], axis=-1)
    if not np.isfinite(truth).all():
        bad = np.argwhere(~np.isfinite(truth))[0]
        raise ValueError(
            "Non-finite NPE truth value: "
            f"batch_row={int(bad[0])} parameter={names[int(bad[1])]}"
        )
    return jnp.asarray(truth, dtype=jnp.float32)


def _object_id_column_from_config(config: dict[str, Any]) -> str | None:
    dataset = config.get("dataset", {}) or {}
    for key in ("id_column", "object_id_column"):
        value = dataset.get(key)
        if value:
            return str(value)
    value = config.get("object_id_column")
    return str(value) if value else None


def _row_index_array(arrays: PhotometryArrays) -> np.ndarray:
    if arrays.row_index is not None:
        return np.asarray(arrays.row_index, dtype=np.int64)
    return np.arange(int(arrays.flux.shape[0]), dtype=np.int64)


def _concat_row_indices(chunks: list[PhotometryArrays]) -> np.ndarray | None:
    if not chunks:
        return None
    return np.concatenate([_row_index_array(chunk) for chunk in chunks], axis=0)
