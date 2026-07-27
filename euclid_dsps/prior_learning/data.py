"""Data loading for supervised truth-prior learning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.amortized.latent import LatentSpec, theta_to_x, x_to_theta
from euclid_dsps.io import load_row_indices

from .schema import (
    ParameterSpec,
    TruthSchema,
    build_truth_schema,
    with_parameter_bounds,
)


@dataclass(frozen=True)
class TruthDataset:
    """Truth parameter matrix and bounded-transform metadata."""

    schema: TruthSchema
    latent_spec: LatentSpec
    theta: np.ndarray
    x: np.ndarray
    object_id: np.ndarray
    source_rows: np.ndarray
    dropped_rows: int
    dataset_path: str

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(param.name for param in self.schema.parameters)

    def theta_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.theta, columns=self.parameter_names)
        frame.insert(0, "source_row", self.source_rows)
        frame.insert(0, "object_id", self.object_id)
        return frame

    def x_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            self.x,
            columns=[f"x_{name}" for name in self.parameter_names],
        )
        frame.insert(0, "source_row", self.source_rows)
        frame.insert(0, "object_id", self.object_id)
        return frame


def load_truth_dataset(
    dataset_path: str | Path,
    *,
    schema_name: str,
    missing_policy: str = "reduce",
    bounds: dict[str, Any] | None = None,
    limit: int | None = None,
    row_indices_file: str | Path | None = None,
) -> TruthDataset:
    """Load a prepared Diffsky parquet as truth theta and unconstrained x."""
    dataset_path = Path(dataset_path)
    frame = pd.read_parquet(dataset_path)
    if row_indices_file:
        row_indices = load_row_indices(row_indices_file)
        if row_indices:
            if min(row_indices) < 0 or max(row_indices) >= len(frame):
                raise ValueError(
                    "row_indices_file contains row_index outside truth dataset "
                    f"bounds: min={min(row_indices)} max={max(row_indices)} "
                    f"rows={len(frame)}"
                )
        frame = frame.iloc[row_indices].copy()
    if limit is not None:
        frame = frame.head(max(int(limit), 0))
    source_row_base = frame.index.to_numpy(dtype=np.int64)
    schema = build_truth_schema(
        frame.columns,
        schema_name=schema_name,
        missing_policy=missing_policy,
    )
    schema = with_parameter_bounds(frame, schema, configured_bounds=bounds)
    columns = [param.column for param in schema.parameters]
    raw = frame[columns].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(raw.to_numpy(dtype=float)).all(axis=1)
    theta = raw.loc[finite_mask].to_numpy(dtype=np.float32)
    if theta.size == 0:
        raise ValueError(f"No finite truth rows found in {dataset_path}")
    latent_spec = latent_spec_from_parameter_specs(schema.parameters)
    x = np.asarray(theta_to_x(jnp.asarray(theta), latent_spec), dtype=np.float32)
    object_id = (
        frame.loc[finite_mask, "object_id"].to_numpy()
        if "object_id" in frame
        else np.nonzero(finite_mask)[0].astype(np.int64)
    )
    source_rows = source_row_base[finite_mask]
    return TruthDataset(
        schema=schema,
        latent_spec=latent_spec,
        theta=theta,
        x=x,
        object_id=object_id,
        source_rows=source_rows,
        dropped_rows=int((~finite_mask).sum()),
        dataset_path=str(dataset_path),
    )


def load_truth_dataset_with_schema(
    dataset_path: str | Path,
    *,
    schema: TruthSchema,
    limit: int | None = None,
    latent_spec: LatentSpec | None = None,
) -> TruthDataset:
    """Load a truth dataset using an already resolved schema and bounds."""
    dataset_path = Path(dataset_path)
    frame = pd.read_parquet(dataset_path)
    if limit is not None:
        frame = frame.head(max(int(limit), 0))
    missing = [param.column for param in schema.parameters if param.column not in frame]
    if missing:
        raise ValueError(f"{dataset_path} is missing schema columns: {missing}")
    source_row_base = frame.index.to_numpy(dtype=np.int64)
    columns = [param.column for param in schema.parameters]
    raw = frame[columns].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(raw.to_numpy(dtype=float)).all(axis=1)
    theta = raw.loc[finite_mask].to_numpy(dtype=np.float32)
    if theta.size == 0:
        raise ValueError(f"No finite truth rows found in {dataset_path}")
    latent_spec = latent_spec or latent_spec_from_parameter_specs(schema.parameters)
    x = np.asarray(theta_to_x(jnp.asarray(theta), latent_spec), dtype=np.float32)
    object_id = (
        frame.loc[finite_mask, "object_id"].to_numpy()
        if "object_id" in frame
        else np.nonzero(finite_mask)[0].astype(np.int64)
    )
    source_rows = source_row_base[finite_mask]
    return TruthDataset(
        schema=schema,
        latent_spec=latent_spec,
        theta=theta,
        x=x,
        object_id=object_id,
        source_rows=source_rows,
        dropped_rows=int((~finite_mask).sum()),
        dataset_path=str(dataset_path),
    )


def latent_spec_from_parameter_specs(
    parameters: tuple[ParameterSpec, ...],
) -> LatentSpec:
    """Build the amortized bounded-transform spec from truth parameter specs."""
    missing = [
        param.name for param in parameters if param.lower is None or param.upper is None
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Parameter bounds are missing for: {joined}")
    return LatentSpec(
        names=tuple(param.name for param in parameters),
        lower=jnp.asarray(
            [float(param.lower) for param in parameters], dtype=jnp.float32
        ),
        upper=jnp.asarray(
            [float(param.upper) for param in parameters], dtype=jnp.float32
        ),
    )


def truth_standardized_latent_spec(
    truth: TruthDataset,
    *,
    min_raw_scale: float = 0.1,
    max_raw_scale: float = 10.0,
) -> tuple[LatentSpec, dict[str, Any]]:
    """Return a latent spec that standardizes raw bounded logits from truth rows."""
    raw_x = np.asarray(truth.x, dtype=np.float32)
    center = np.nanmean(raw_x, axis=0).astype(np.float32)
    scale = np.nanstd(raw_x, axis=0).astype(np.float32)
    finite = np.isfinite(scale) & (scale > 0.0)
    scale = np.where(finite, scale, 1.0).astype(np.float32)
    min_raw_scale = float(min_raw_scale)
    max_raw_scale = float(max_raw_scale)
    if min_raw_scale <= 0.0 or max_raw_scale < min_raw_scale:
        raise ValueError(
            "truth standardized latent scales require 0 < min_raw_scale <= max_raw_scale"
        )
    clipped_scale = np.clip(scale, min_raw_scale, max_raw_scale).astype(np.float32)
    spec = latent_spec_from_parameter_specs(truth.schema.parameters)
    spec = LatentSpec(
        names=spec.names,
        lower=spec.lower,
        upper=spec.upper,
        raw_center=jnp.asarray(center, dtype=jnp.float32),
        raw_scale=jnp.asarray(clipped_scale, dtype=jnp.float32),
        normalization="truth_standardized_logit",
    )
    payload = {
        "normalization": "truth_standardized_logit",
        "min_raw_scale": min_raw_scale,
        "max_raw_scale": max_raw_scale,
        "raw_center": center.astype(float).tolist(),
        "raw_scale_unclipped": scale.astype(float).tolist(),
        "raw_scale": clipped_scale.astype(float).tolist(),
        "n_raw_scale_clipped_low": int(np.sum(scale < min_raw_scale)),
        "n_raw_scale_clipped_high": int(np.sum(scale > max_raw_scale)),
    }
    return spec, payload


def truth_dataset_with_latent_spec(
    truth: TruthDataset,
    latent_spec: LatentSpec,
) -> TruthDataset:
    """Return the same truth rows represented in ``latent_spec`` coordinates."""
    x = np.asarray(theta_to_x(jnp.asarray(truth.theta), latent_spec), dtype=np.float32)
    return replace(truth, latent_spec=latent_spec, x=x)


def prior_samples_frame(
    x: np.ndarray,
    latent_spec: LatentSpec,
    *,
    log_prob: np.ndarray | None = None,
) -> pd.DataFrame:
    """Convert prior x samples to a theta sample dataframe."""
    theta = np.asarray(x_to_theta(jnp.asarray(x, dtype=jnp.float32), latent_spec))
    frame = pd.DataFrame(theta, columns=latent_spec.names)
    frame.insert(0, "sample_id", np.arange(len(frame), dtype=np.int64))
    if log_prob is not None:
        frame["log_prob"] = np.asarray(log_prob, dtype=float)
    return frame
