#!/usr/bin/env python3
"""Build deterministic observed-r<25 manifests without reading FENIKS truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from euclid_dsps.photometry import abmag_to_fnu_cgs

OBSERVED_COLUMNS = ("flux_lsst_r", "fluxerr_lsst_r", "mask_lsst_r")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_observed_r(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(OBSERVED_COLUMNS[:2]) - schema_names
    if missing:
        raise ValueError(f"{path} is missing observed columns: {sorted(missing)}")
    columns = [name for name in OBSERVED_COLUMNS if name in schema_names]
    table = pq.read_table(path, columns=columns)
    flux = np.asarray(table["flux_lsst_r"], dtype=float)
    error = np.asarray(table["fluxerr_lsst_r"], dtype=float)
    mask = (
        np.asarray(table["mask_lsst_r"], dtype=bool)
        if "mask_lsst_r" in columns
        else np.ones(len(flux), dtype=bool)
    )
    return flux, error, mask


def _selected_rows(
    path: Path,
    *,
    flux_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    flux, error, mask = _read_observed_r(path)
    selected = mask & np.isfinite(flux) & np.isfinite(error) & (error > 0.0)
    selected &= flux > float(flux_limit)
    rows = np.flatnonzero(selected).astype(np.int64)
    standardized_margin = (flux[rows] - float(flux_limit)) / error[rows]
    return rows, standardized_margin


def _stratified_probe(
    rows: np.ndarray,
    score: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    if count <= 0 or count > len(rows):
        raise ValueError(f"exact probe count must be in [1, {len(rows)}]")
    edges = np.quantile(score, [0.0, 0.25, 0.5, 0.75, 1.0])
    chosen: list[np.ndarray] = []
    counts: dict[str, int] = {}
    remaining = int(count)
    for index in range(4):
        lower = edges[index]
        upper = edges[index + 1]
        in_bin = (score >= lower) & (
            (score <= upper) if index == 3 else (score < upper)
        )
        candidates = rows[in_bin]
        target = count // 4 + (1 if index < count % 4 else 0)
        take = min(target, len(candidates))
        selected = (
            rng.choice(candidates, size=take, replace=False)
            if take
            else np.empty(0, dtype=np.int64)
        )
        chosen.append(np.asarray(selected, dtype=np.int64))
        counts[f"observed_r_margin_quartile_{index + 1}"] = int(take)
        remaining -= int(take)
    selected_rows = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    if remaining:
        available = np.setdiff1d(rows, selected_rows, assume_unique=False)
        selected_rows = np.concatenate(
            (selected_rows, rng.choice(available, size=remaining, replace=False))
        )
    return np.sort(selected_rows), counts


def _write_indices(path: Path, values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=np.int64)
    np.save(path, array, allow_pickle=False)
    return {
        "path": str(path),
        "count": int(len(array)),
        "sha256": _sha256(path),
        "minimum": int(np.min(array)),
        "maximum": int(np.max(array)),
    }


def _write_exact_cohort(
    path: Path,
    *,
    catalog: Path,
    rows: np.ndarray,
    selected_rows: np.ndarray,
    selected_scores: np.ndarray,
) -> dict[str, object]:
    object_ids = np.asarray(
        pq.read_table(catalog, columns=["object_id"])["object_id"].to_pylist(),
        dtype=object,
    )
    score_lookup = dict(
        zip(selected_rows.tolist(), selected_scores.tolist(), strict=True)
    )
    scores = np.asarray([score_lookup[int(row)] for row in rows], dtype=float)
    quartiles = np.quantile(selected_scores, [0.25, 0.5, 0.75])
    labels = [
        f"observed_r_margin_q{int(np.searchsorted(quartiles, score)) + 1}"
        for score in scores
    ]
    frame = pd.DataFrame(
        {
            "order": np.arange(len(rows), dtype=np.int64),
            "example_key": labels,
            "row_index": rows,
            "object_id": object_ids[rows].astype(str),
            "observed_r_margin_sigma": scores,
        }
    )
    frame.to_csv(path, index=False)
    return {"path": str(path), "count": int(len(frame)), "sha256": _sha256(path)}


def build(
    *,
    train_catalog: Path,
    test_catalog: Path,
    out: Path,
    validation_fraction: float,
    n_exact: int,
    seed: int,
    max_selected_train: int | None = None,
) -> dict[str, object]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    flux_limit = float(np.asarray(abmag_to_fnu_cgs(25.0)))
    train_rows, _train_score = _selected_rows(train_catalog, flux_limit=flux_limit)
    test_rows, test_score = _selected_rows(test_catalog, flux_limit=flux_limit)
    selected_train_count = len(train_rows)
    rng = np.random.default_rng(int(seed))
    if max_selected_train is not None:
        if max_selected_train <= 1:
            raise ValueError("max_selected_train must exceed one")
        train_rows = rng.choice(
            train_rows,
            size=min(int(max_selected_train), len(train_rows)),
            replace=False,
        )
    if len(train_rows) < 2:
        raise ValueError("observed r<25 cut retained fewer than two training rows")
    shuffled = rng.permutation(train_rows)
    n_validation = max(1, int(round(len(shuffled) * validation_fraction)))
    n_validation = min(n_validation, len(shuffled) - 1)
    validation_rows = np.sort(shuffled[:n_validation])
    fit_rows = np.sort(shuffled[n_validation:])
    exact_rows, strata = _stratified_probe(
        test_rows,
        test_score,
        count=int(n_exact),
        rng=rng,
    )
    out.mkdir(parents=True, exist_ok=False)
    exact_cohort = _write_exact_cohort(
        out / "exact_cohort.csv",
        catalog=test_catalog,
        rows=exact_rows,
        selected_rows=test_rows,
        selected_scores=test_score,
    )
    payload = {
        "status": "complete",
        "contract": (
            "observed photometry only: flux_lsst_r > f_nu(25 AB); no true flux, "
            "true magnitude or latent truth column is read"
        ),
        "observed_columns_read": list(OBSERVED_COLUMNS),
        "selection": {
            "band": "lsst_r",
            "max_mag_ab": 25.0,
            "flux_min_fnu_cgs": flux_limit,
            "comparison": "flux_lsst_r > flux_min_fnu_cgs",
        },
        "seed": int(seed),
        "catalogs": {
            "train": {
                "path": str(train_catalog),
                "size_bytes": train_catalog.stat().st_size,
                "selected_rows_before_optional_limit": int(selected_train_count),
            },
            "test": {
                "path": str(test_catalog),
                "size_bytes": test_catalog.stat().st_size,
                "selected_rows": int(len(test_rows)),
            },
        },
        "manifests": {
            "train": _write_indices(out / "train_indices.npy", fit_rows),
            "validation": _write_indices(
                out / "validation_indices.npy", validation_rows
            ),
            "selected_test": _write_indices(
                out / "selected_test_indices.npy", test_rows
            ),
            "exact_probe": _write_indices(out / "exact_probe_indices.npy", exact_rows),
            "exact_cohort": exact_cohort,
        },
        "exact_probe_strata": strata,
    }
    (out / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-catalog", type=Path, required=True)
    parser.add_argument("--test-catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--n-exact", type=int, default=32)
    parser.add_argument("--seed", type=int, default=260821)
    parser.add_argument("--max-selected-train", type=int)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
