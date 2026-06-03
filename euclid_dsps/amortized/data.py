"""FS2 photometry batches for amortized inference."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from euclid_dsps.io import iter_catalog_batches, required_catalog_columns
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


def compute_fs2_feature_stats_from_config(
    config: dict[str, Any],
    *,
    limit: int | None = None,
    batch_size: int = 10_000,
    row_indices: np.ndarray | None = None,
) -> FeatureStats:
    """Compute feature stats from configured FS2 photometry."""
    band_names = validate_fs2_band_contract(config["bands"])
    features_cfg = amortized_config(config)["features"]
    flux_chunks = []
    err_chunks = []
    mask_chunks = []
    for arrays in iter_fs2_photometry_arrays_from_config(
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
    )


def iter_fs2_photometry_arrays_from_config(
    config: dict[str, Any],
    *,
    batch_size: int,
    limit: int | None,
    row_indices: np.ndarray | None = None,
) -> Iterator:
    """Yield raw NumPy FS2 photometry arrays from a config."""
    validate_fs2_band_contract(config["bands"])
    columns = required_catalog_columns(config)
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
        yield photometry_arrays_from_dataframe(frame, config["bands"])


def load_fs2_photometry_arrays_from_config(
    config: dict[str, Any],
    *,
    batch_size: int,
    limit: int | None = None,
    row_indices: np.ndarray | None = None,
) -> PhotometryArrays:
    """Load selected FS2 photometry into one compact array block."""
    chunks = list(
        iter_fs2_photometry_arrays_from_config(
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
        flux=np.concatenate([chunk.flux for chunk in chunks], axis=0),
        flux_err=np.concatenate([chunk.flux_err for chunk in chunks], axis=0),
        mask=np.concatenate([chunk.mask for chunk in chunks], axis=0),
        band_names=chunks[0].band_names,
    )


def iter_fs2_photometry_batches_from_config(
    config: dict[str, Any],
    batch_size: int,
    limit: int | None,
    feature_stats: FeatureStats | None,
    row_indices: np.ndarray | None = None,
) -> Iterator[PhotometryBatch]:
    """Yield JAX-ready FS2 photometry batches."""
    stats = feature_stats
    if stats is None:
        stats = compute_fs2_feature_stats_from_config(
            config,
            limit=limit,
            batch_size=batch_size,
            row_indices=row_indices,
        )
    for arrays in iter_fs2_photometry_arrays_from_config(
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
        )
        yield PhotometryBatch(
            object_id=jnp.asarray(arrays.object_id[idx], dtype=jnp.int32),
            flux=jnp.asarray(arrays.flux[idx], dtype=jnp.float32),
            flux_err=jnp.asarray(arrays.flux_err[idx], dtype=jnp.float32),
            mask=jnp.asarray(arrays.mask[idx]),
            features=features,
        )
