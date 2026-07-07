"""Encoder feature construction for FS2 amortized inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class FeatureStats:
    flux_scale: np.ndarray
    err_scale: np.ndarray
    band_names: tuple[str, ...]
    flux_transform: str = "asinh"


def compute_feature_stats(
    flux: np.ndarray,
    flux_err: np.ndarray,
    mask: np.ndarray | None = None,
    band_names: tuple[str, ...] | None = None,
    flux_transform: str = "asinh",
) -> FeatureStats:
    """Compute robust per-band feature scales."""
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    if flux.shape != flux_err.shape:
        raise ValueError(
            f"flux and flux_err shapes differ: {flux.shape} vs {flux_err.shape}"
        )
    if flux.ndim != 2:
        raise ValueError(f"flux arrays must be rank 2 [N,B], got {flux.shape}")
    if mask is None:
        valid = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0.0)
    else:
        valid = np.asarray(mask, dtype=bool) & np.isfinite(flux) & np.isfinite(flux_err)
        valid &= flux_err > 0.0
    flux_scale = _per_band_scale(np.abs(flux), valid)
    err_scale = _per_band_scale(flux_err, valid)
    if band_names is None:
        band_names = tuple(f"band_{index}" for index in range(flux.shape[1]))
    if len(band_names) != flux.shape[1]:
        raise ValueError(
            f"band_names length {len(band_names)} does not match bands {flux.shape[1]}"
        )
    return FeatureStats(
        flux_scale=flux_scale.astype(np.float32),
        err_scale=err_scale.astype(np.float32),
        band_names=tuple(band_names),
        flux_transform=str(flux_transform),
    )


def make_encoder_features(
    flux: jnp.ndarray,
    flux_err: jnp.ndarray,
    stats: FeatureStats,
) -> jnp.ndarray:
    """Return encoder input ``[flux_10, err_10]`` with stable normalization."""
    flux = jnp.asarray(flux, dtype=jnp.float32)
    flux_err = jnp.asarray(flux_err, dtype=jnp.float32)
    if flux.shape != flux_err.shape:
        raise ValueError(
            f"flux and flux_err shapes differ: {flux.shape} vs {flux_err.shape}"
        )
    flux_scale = jnp.asarray(stats.flux_scale, dtype=jnp.float32)
    err_scale = jnp.asarray(stats.err_scale, dtype=jnp.float32)
    if flux.shape[-1] != flux_scale.shape[0]:
        raise ValueError(f"Expected {flux_scale.shape[0]} bands, got {flux.shape[-1]}")
    scale_floor = jnp.asarray(1.0e-30, dtype=jnp.float32)
    relative_eps = jnp.asarray(1.0e-6, dtype=jnp.float32)
    flux_scale_safe = jnp.maximum(flux_scale, scale_floor)
    err_scale_safe = jnp.maximum(err_scale, scale_floor)
    flux_ratio = flux / flux_scale_safe
    flux_transform = str(getattr(stats, "flux_transform", "asinh"))
    if flux_transform == "asinh":
        flux_features = jnp.arcsinh(flux_ratio)
    elif flux_transform == "signed_log1p":
        flux_features = jnp.sign(flux_ratio) * jnp.log1p(jnp.abs(flux_ratio))
    elif flux_transform == "linear":
        flux_features = flux_ratio
    else:
        raise ValueError(f"Unsupported flux feature transform: {flux_transform}")
    err_positive = jnp.maximum(flux_err, relative_eps * err_scale_safe)
    err_features = jnp.log(err_positive / err_scale_safe + relative_eps)
    return jnp.concatenate([flux_features, err_features], axis=-1)


def feature_stats_to_json(stats: FeatureStats) -> dict:
    """Convert stats to a JSON-compatible mapping."""
    payload = asdict(stats)
    payload["flux_scale"] = np.asarray(stats.flux_scale, dtype=float).tolist()
    payload["err_scale"] = np.asarray(stats.err_scale, dtype=float).tolist()
    payload["band_names"] = list(stats.band_names)
    payload["flux_transform"] = str(stats.flux_transform)
    return payload


def feature_stats_from_json(payload: dict) -> FeatureStats:
    """Load stats from a JSON-compatible mapping."""
    return FeatureStats(
        flux_scale=np.asarray(payload["flux_scale"], dtype=np.float32),
        err_scale=np.asarray(payload["err_scale"], dtype=np.float32),
        band_names=tuple(str(name) for name in payload["band_names"]),
        flux_transform=str(payload.get("flux_transform", "linear")),
    )


def write_feature_stats(path: str | Path, stats: FeatureStats) -> None:
    """Write feature stats as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(feature_stats_to_json(stats), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_feature_stats(path: str | Path) -> FeatureStats:
    """Read feature stats from JSON."""
    return feature_stats_from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def _per_band_scale(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    scales = []
    for band in range(values.shape[1]):
        column = values[:, band]
        selected = column[valid[:, band] & np.isfinite(column) & (column > 0.0)]
        if selected.size:
            scale = float(np.nanmedian(selected))
        else:
            scale = 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        scales.append(scale)
    return np.asarray(scales, dtype=float)
