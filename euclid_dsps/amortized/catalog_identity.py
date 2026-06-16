"""Catalog row identity and truth-snapshot helpers for amortized runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.io import iter_catalog_batches, truth_column_from_spec, write_json


def object_id_column_from_config(config: dict[str, Any]) -> str | None:
    """Return the configured object-id column, if any."""
    dataset = config.get("dataset", {}) or {}
    for key in ("id_column", "object_id_column"):
        value = dataset.get(key)
        if value:
            return str(value)
    value = config.get("object_id_column")
    return str(value) if value else None


def configured_redshift_column(config: dict[str, Any]) -> str | None:
    """Return the configured truth redshift column, if available."""
    data_cfg = ((config.get("amortized", {}) or {}).get("data", {}) or {})
    explicit = data_cfg.get("stratify_column")
    if explicit:
        return str(explicit)
    truth = config.get("truth", {}) or {}
    column = truth_column_from_spec(truth.get("redshift_column"))
    if column:
        return column
    redshift = config.get("redshift", {}) or {}
    return truth_column_from_spec(redshift.get("truth_column"))


def truth_columns_from_config(config: dict[str, Any]) -> list[str]:
    """Return configured truth/proxy columns used by diagnostics."""
    columns = ["redshift_true", "logsm_true", "logsfr_true", "logssfr_true"]
    truth = config.get("truth", {}) or {}
    redshift_column = truth_column_from_spec(truth.get("redshift_column"))
    if redshift_column:
        columns.append(redshift_column)
    for spec in dict(truth.get("parameter_columns") or {}).values():
        column = truth_column_from_spec(spec)
        if column:
            columns.append(column)
    id_column = object_id_column_from_config(config)
    if id_column:
        columns.append(id_column)
    return sorted(set(columns))


def available_columns(path: str | Path) -> set[str]:
    """Return available parquet column names."""
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema.names)


def read_redshift_column(path: str | Path, column: str | None) -> np.ndarray | None:
    """Read a redshift column as a NumPy array if present."""
    if column is None:
        return None
    try:
        frame = pd.read_parquet(path, columns=[column])
    except Exception:
        return None
    if column not in frame:
        return None
    return frame[column].to_numpy(dtype=float)


def select_catalog_row_indices(
    config: dict[str, Any],
    *,
    limit: int | None,
    selection_mode: str,
    stratified_strategy: str,
    seed: int,
    redshift_bins: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return selected catalog row indices for non-sequential inference."""
    mode = str(selection_mode or "sequential")
    path = Path(config["catalog_path"])
    n_rows = _catalog_num_rows(path)
    total = n_rows if limit is None else min(max(int(limit), 0), n_rows)
    if mode == "sequential":
        return None, {
            "selection_mode": "sequential",
            "limit": limit,
            "selected_rows": int(total),
            "catalog_rows": int(n_rows),
            "row_indices_path": None,
        }
    if total <= 0:
        return np.asarray([], dtype=np.int64), {
            "selection_mode": mode,
            "limit": limit,
            "selected_rows": 0,
            "catalog_rows": int(n_rows),
        }
    rng = np.random.default_rng(int(seed))
    all_indices = np.arange(n_rows, dtype=np.int64)
    redshift_column = configured_redshift_column(config)
    redshift = read_redshift_column(path, redshift_column)
    if mode == "random" or redshift is None:
        selected = rng.choice(all_indices, size=total, replace=False).astype(np.int64)
    elif mode == "stratified_redshift":
        bins = _redshift_bins(config, redshift_bins)
        selected = _select_stratified_indices(
            redshift,
            total=total,
            bins=bins,
            strategy=str(stratified_strategy or "balanced"),
            rng=rng,
        )
    else:
        raise ValueError(
            "inference selection_mode must be sequential, random, or "
            "stratified_redshift"
        )
    selected = np.asarray(selected, dtype=np.int64)
    summary: dict[str, Any] = {
        "selection_mode": mode,
        "stratified_strategy": str(stratified_strategy or "balanced"),
        "selection_seed": int(seed),
        "limit": limit,
        "selected_rows": int(selected.size),
        "catalog_rows": int(n_rows),
        "redshift_column": redshift_column,
        "row_index_min": int(selected.min()) if selected.size else None,
        "row_index_max": int(selected.max()) if selected.size else None,
    }
    if redshift is not None and selected.size:
        bins = _redshift_bins(config, redshift_bins)
        summary["redshift_histogram"] = redshift_histogram(redshift[selected], bins)
    return selected, summary


def catalog_fingerprint(
    config: dict[str, Any],
    *,
    redshift_bins: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Return a compact catalog fingerprint for reproducibility checks."""
    import pyarrow.parquet as pq

    path = Path(config["catalog_path"])
    parquet = pq.ParquetFile(path)
    schema_names = list(parquet.schema.names)
    payload: dict[str, Any] = {
        "catalog_path": str(path),
        "exists": bool(path.exists()),
        "file_size_bytes": int(path.stat().st_size) if path.exists() else None,
        "row_count": int(parquet.metadata.num_rows),
        "row_groups": int(parquet.num_row_groups),
        "schema_hash": _schema_hash(schema_names),
        "object_id_column": object_id_column_from_config(config),
        "redshift_column": configured_redshift_column(config),
    }
    columns = available_columns(path)
    id_column = payload["object_id_column"]
    if id_column and id_column in columns:
        ids = pd.read_parquet(path, columns=[id_column])[id_column]
        payload["object_id_dtype"] = str(ids.dtype)
        payload["object_id_unique"] = bool(ids.is_unique)
        payload["object_id_n_unique"] = int(ids.nunique(dropna=False))
        payload["object_id_n_duplicates"] = int(len(ids) - ids.nunique(dropna=False))
        if len(ids):
            payload["object_id_first"] = _jsonable_scalar(ids.iloc[0])
            payload["object_id_last"] = _jsonable_scalar(ids.iloc[-1])
    z_column = payload["redshift_column"]
    if z_column and z_column in columns:
        redshift = pd.read_parquet(path, columns=[z_column])[z_column].to_numpy(
            dtype=float
        )
        finite = redshift[np.isfinite(redshift)]
        payload["redshift_finite"] = int(finite.size)
        if finite.size:
            payload["redshift_min"] = float(np.min(finite))
            payload["redshift_max"] = float(np.max(finite))
            payload["redshift_quantiles"] = {
                str(q): float(np.quantile(finite, q))
                for q in (0.01, 0.05, 0.16, 0.5, 0.84, 0.95, 0.99)
            }
            if redshift_bins is not None:
                payload["redshift_histogram"] = redshift_histogram(
                    finite,
                    redshift_bins,
                )
    return payload


def write_catalog_fingerprint(
    out_dir: str | Path,
    config: dict[str, Any],
    *,
    redshift_bins: list[float] | tuple[float, ...] | np.ndarray | None = None,
    filename: str = "catalog_fingerprint.json",
) -> dict[str, Any]:
    """Write and return a catalog fingerprint JSON file."""
    out = Path(out_dir)
    payload = catalog_fingerprint(config, redshift_bins=redshift_bins)
    write_json(out / filename, payload)
    return payload


def redshift_histogram(
    redshift: np.ndarray,
    bins: list[float] | tuple[float, ...] | np.ndarray,
) -> list[dict[str, float | int]]:
    """Return JSON-friendly redshift histogram rows."""
    values = np.asarray(redshift, dtype=float)
    values = values[np.isfinite(values)]
    edges = np.asarray(bins, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        return []
    counts, edges = np.histogram(values, bins=edges)
    return [
        {
            "z_bin_lower": float(edges[index]),
            "z_bin_upper": float(edges[index + 1]),
            "n_objects": int(count),
        }
        for index, count in enumerate(counts)
    ]


def _catalog_num_rows(path: str | Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _redshift_bins(
    config: dict[str, Any],
    redshift_bins: list[float] | tuple[float, ...] | np.ndarray | None,
) -> np.ndarray:
    if redshift_bins is None:
        amortized = config.get("amortized", {}) or {}
        inference = dict(amortized.get("inference", {}) or {})
        data = dict(amortized.get("data", {}) or {})
        redshift_bins = inference.get("redshift_bins", data.get("redshift_bins"))
    if redshift_bins is None:
        redshift_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    bins = np.asarray(redshift_bins, dtype=float)
    bins = np.unique(bins[np.isfinite(bins)])
    if bins.ndim != 1 or bins.size < 2:
        raise ValueError("redshift_bins must contain at least two finite values")
    return bins


def _select_stratified_indices(
    redshift: np.ndarray,
    *,
    total: int,
    bins: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    all_indices = np.arange(len(redshift), dtype=np.int64)
    if total >= len(all_indices):
        selected = all_indices.copy()
        rng.shuffle(selected)
        return selected
    finite = np.isfinite(redshift)
    if not finite.any():
        return rng.choice(all_indices, size=total, replace=False).astype(np.int64)
    if strategy == "proportional":
        finite_indices = all_indices[finite]
        if total <= len(finite_indices):
            return rng.choice(finite_indices, size=total, replace=False).astype(
                np.int64
            )
        selected = list(finite_indices)
        remaining = np.setdiff1d(all_indices, finite_indices, assume_unique=False)
        selected.extend(
            rng.choice(remaining, size=total - len(selected), replace=False).tolist()
        )
        selected = np.asarray(selected, dtype=np.int64)
        rng.shuffle(selected)
        return selected
    if strategy != "balanced":
        raise ValueError("stratified_strategy must be balanced or proportional")
    groups = _indices_by_redshift_bin(redshift, bins)
    nonempty = [group for group in groups if len(group)]
    if not nonempty:
        return rng.choice(all_indices, size=total, replace=False).astype(np.int64)
    per_bin = int(np.ceil(total / float(len(nonempty))))
    selected: list[int] = []
    for group in nonempty:
        take = min(per_bin, len(group), total - len(selected))
        if take <= 0:
            break
        selected.extend(rng.choice(group, size=take, replace=False).tolist())
    if len(selected) < total:
        remaining = np.setdiff1d(all_indices, np.asarray(selected), assume_unique=False)
        selected.extend(
            rng.choice(remaining, size=total - len(selected), replace=False).tolist()
        )
    selected = np.asarray(selected[:total], dtype=np.int64)
    rng.shuffle(selected)
    return selected


def _indices_by_redshift_bin(redshift: np.ndarray, bins: np.ndarray) -> list[np.ndarray]:
    redshift = np.asarray(redshift, dtype=float)
    base_indices = np.arange(len(redshift), dtype=np.int64)
    groups = []
    for index in range(len(bins) - 1):
        lo = bins[index]
        hi = bins[index + 1]
        if index == len(bins) - 2:
            mask = (redshift >= lo) & (redshift <= hi)
        else:
            mask = (redshift >= lo) & (redshift < hi)
        groups.append(base_indices[mask & np.isfinite(redshift)])
    missing = base_indices[~np.isfinite(redshift)]
    if len(missing):
        groups.append(missing)
    return groups


def write_truth_snapshot(
    out_dir: str | Path,
    config: dict[str, Any],
    *,
    row_indices: np.ndarray | None,
    limit: int | None,
    batch_size: int = 10_000,
    filename: str = "inference_truth.parquet",
) -> pd.DataFrame:
    """Write truth rows for exactly the inference selection."""
    out = Path(out_dir)
    path = Path(config["catalog_path"])
    columns = [column for column in truth_columns_from_config(config) if column]
    available = available_columns(path)
    columns = [column for column in columns if column in available]
    id_column = object_id_column_from_config(config)
    if id_column and id_column in available and id_column not in columns:
        columns.append(id_column)
    frames = []
    for batch in iter_catalog_batches(
        path,
        columns=columns,
        batch_size=batch_size,
        limit=limit,
        row_indices=(
            None
            if row_indices is None
            else set(np.asarray(row_indices, dtype=np.int64).tolist())
        ),
    ):
        frame = batch.reset_index(names="row_index")
        if id_column and id_column in frame:
            frame["object_id"] = frame[id_column]
        elif "object_id" not in frame:
            frame["object_id"] = frame["row_index"]
        frames.append(frame)
    truth = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    truth.to_parquet(out / filename, index=False)
    if not truth.empty:
        z_column = configured_redshift_column(config)
        if z_column and z_column in truth:
            hist = redshift_histogram(
                truth[z_column].to_numpy(dtype=float),
                ((config.get("amortized", {}) or {}).get("data", {}) or {}).get(
                    "redshift_bins",
                    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
                ),
            )
            pd.DataFrame(hist).to_csv(out / "inference_redshift_histogram.csv", index=False)
    return truth


def _schema_hash(schema_names: list[str]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for name in schema_names:
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _jsonable_scalar(value) -> int | float | str | None:
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return value
    return str(value)
