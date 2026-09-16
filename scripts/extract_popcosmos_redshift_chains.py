#!/usr/bin/env python3
"""Extract matched redshift draws from the public Pop-COSMOS HDF5 chains."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

ZENODO_RECORD = "13820043"
ZENODO_VERSION = "1.1.0"
ZENODO_URL = "https://zenodo.org/records/13820043"
DEFAULT_Z_INDEX = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-objects", type=int, default=1_395)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--z-index", type=int, default=DEFAULT_Z_INDEX)
    parser.add_argument("--read-chunk-size", type=int, default=128)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dataset_candidates(handle: h5py.File) -> dict[str, h5py.Dataset]:
    datasets: dict[str, h5py.Dataset] = {}

    def collect(name: str, value: Any) -> None:
        if isinstance(value, h5py.Dataset):
            datasets[name] = value

    handle.visititems(collect)
    return datasets


def _resolve_dataset(
    datasets: dict[str, h5py.Dataset],
    *,
    preferred: tuple[str, ...],
    basenames: tuple[str, ...],
) -> tuple[str, h5py.Dataset]:
    normalized = {
        name.lstrip("/").lower(): (name, value) for name, value in datasets.items()
    }
    for name in preferred:
        match = normalized.get(name.lower())
        if match is not None:
            return match
    matches = [
        (name, value)
        for name, value in datasets.items()
        if name.rsplit("/", 1)[-1].lower() in basenames
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Could not resolve one HDF5 dataset from basenames={basenames}; "
            f"matches={[name for name, _value in matches]}"
        )
    return matches[0]


def _integer_ids(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if flat.dtype.kind in "SUO":
        flat = np.asarray([int(value) for value in flat], dtype=np.int64)
    else:
        numeric = np.asarray(flat, dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ValueError("Pop-COSMOS index_cosmos values are not finite integers")
        flat = numeric.astype(np.int64)
    if len(np.unique(flat)) != len(flat):
        raise ValueError("Pop-COSMOS index_cosmos contains duplicate IDs")
    return flat


def evenly_spaced_sample_indices(total: int, selected: int) -> np.ndarray:
    """Return deterministic indices spanning the complete flattened chain."""
    if total < 2 or selected < 2 or selected > total:
        raise ValueError(f"Invalid sample selection selected={selected}, total={total}")
    indices = np.rint(np.linspace(0, total - 1, selected)).astype(np.int64)
    if len(np.unique(indices)) != selected:
        raise RuntimeError("Evenly spaced chain sample selection produced duplicates")
    return indices


def _read_redshift_rows(
    chains: h5py.Dataset,
    chain_rows: np.ndarray,
    *,
    z_index: int,
    read_chunk_size: int,
) -> np.ndarray:
    order = np.argsort(chain_rows, kind="stable")
    sorted_rows = chain_rows[order]
    sorted_values = np.empty((len(chain_rows), chains.shape[1]), dtype=chains.dtype)
    for start in range(0, len(sorted_rows), read_chunk_size):
        stop = min(start + read_chunk_size, len(sorted_rows))
        row_chunk = sorted_rows[start:stop]
        sorted_values[start:stop] = chains[row_chunk, :, z_index]
        print(
            f"[popcosmos-chains] read matched rows {stop}/{len(sorted_rows)}",
            flush=True,
        )
    restored = np.empty_like(sorted_values)
    restored[order] = sorted_values
    return restored


def _posterior_frame(
    cohort: pd.DataFrame,
    values: np.ndarray,
    original_sample_ids: np.ndarray,
) -> pd.DataFrame:
    n_objects, n_samples = values.shape
    return pd.DataFrame(
        {
            "object_id": np.repeat(cohort["object_id"].to_numpy(np.int64), n_samples),
            "row_index": np.repeat(cohort["row_index"].to_numpy(np.int64), n_samples),
            "sample_id": np.tile(np.arange(n_samples, dtype=np.int64), n_objects),
            "chain_sample_id": np.tile(original_sample_ids, n_objects),
            "z_obs": values.reshape(-1).astype(np.float32),
        }
    )


def extract_redshift_chains(
    chains_path: Path,
    cohort_path: Path,
    out: Path,
    *,
    expected_objects: int,
    samples: int,
    z_index: int,
    read_chunk_size: int,
) -> dict[str, Any]:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out}")
    cohort = pd.read_parquet(cohort_path)
    required = {"object_id", "row_index", "redshift_true"}
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"Matched cohort is missing columns: {missing}")
    cohort = cohort.loc[:, ["object_id", "row_index", "redshift_true"]].copy()
    if len(cohort) != expected_objects:
        raise ValueError(
            f"Expected {expected_objects} cohort rows, found {len(cohort)}"
        )
    if cohort["object_id"].duplicated().any() or cohort["row_index"].duplicated().any():
        raise ValueError("Matched cohort identities must be unique")
    if not np.all(np.isfinite(cohort["redshift_true"].to_numpy(float))):
        raise ValueError("Matched cohort contains non-finite spectroscopic redshifts")

    with h5py.File(chains_path, "r") as handle:
        datasets = _dataset_candidates(handle)
        chains_name, chains = _resolve_dataset(
            datasets,
            preferred=("chains", "samples", "posterior/chains"),
            basenames=("chains",),
        )
        index_name, index_dataset = _resolve_dataset(
            datasets,
            preferred=("metadata/index_cosmos", "index_cosmos"),
            basenames=("index_cosmos",),
        )
        if chains.ndim != 3:
            raise ValueError(
                f"Expected a 3D chains dataset, found shape={chains.shape}"
            )
        if chains.shape[0] != index_dataset.size:
            raise ValueError("Chain object axis and index_cosmos length differ")
        if not 0 <= z_index < chains.shape[2]:
            raise ValueError(f"z_index={z_index} outside latent axis {chains.shape[2]}")

        chain_ids = _integer_ids(index_dataset[...])
        lookup = {int(object_id): index for index, object_id in enumerate(chain_ids)}
        cohort_ids = cohort["object_id"].to_numpy(np.int64)
        missing_ids = [int(value) for value in cohort_ids if int(value) not in lookup]
        if missing_ids:
            raise ValueError(
                f"Pop-COSMOS chains are missing {len(missing_ids)} cohort IDs: "
                f"{missing_ids[:5]}"
            )
        chain_rows = np.asarray(
            [lookup[int(value)] for value in cohort_ids], dtype=np.int64
        )
        all_values = _read_redshift_rows(
            chains,
            chain_rows,
            z_index=z_index,
            read_chunk_size=read_chunk_size,
        )
        chain_shape = tuple(int(value) for value in chains.shape)
        chain_dtype = str(chains.dtype)

    if not np.all(np.isfinite(all_values)):
        bad = int(np.size(all_values) - np.isfinite(all_values).sum())
        raise ValueError(
            f"Extracted Pop-COSMOS redshift chains contain {bad} non-finite draws"
        )
    selected_ids = evenly_spaced_sample_indices(all_values.shape[1], samples)
    selected_values = all_values[:, selected_ids]

    out.mkdir(parents=True, exist_ok=False)
    truth_path = out / "inference_truth.parquet"
    primary_path = out / "posterior_samples.parquet"
    all_path = out / "posterior_samples_all.parquet"
    cohort.to_parquet(truth_path, index=False)
    _posterior_frame(cohort, selected_values, selected_ids).to_parquet(
        primary_path, index=False
    )
    _posterior_frame(
        cohort, all_values, np.arange(all_values.shape[1], dtype=np.int64)
    ).to_parquet(all_path, index=False)

    manifest = {
        "status": "complete",
        "scope": "redshift_only_same_1395_public_specz_objects",
        "zenodo": {
            "record": ZENODO_RECORD,
            "version": ZENODO_VERSION,
            "url": ZENODO_URL,
        },
        "chains": {
            "path": str(chains_path),
            "sha256": _sha256(chains_path),
            "dataset": chains_name,
            "index_dataset": index_name,
            "shape": list(chain_shape),
            "dtype": chain_dtype,
            "redshift_parameter_index": int(z_index),
        },
        "cohort": {
            "path": str(cohort_path),
            "sha256": _sha256(cohort_path),
            "objects": int(len(cohort)),
            "object_ids_sha256": hashlib.sha256(
                cohort_ids.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        },
        "primary": {
            "samples_per_object": int(samples),
            "selection": "round(linspace(0, n_chain_samples - 1, samples))",
            "chain_sample_ids": selected_ids.tolist(),
            "path": str(primary_path),
            "sha256": _sha256(primary_path),
        },
        "sensitivity": {
            "samples_per_object": int(all_values.shape[1]),
            "path": str(all_path),
            "sha256": _sha256(all_path),
        },
        "truth": {"path": str(truth_path), "sha256": _sha256(truth_path)},
        "git_commit": _git_sha(),
    }
    (out / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out / "DONE").touch()
    return manifest


def main() -> None:
    args = parse_args()
    summary = extract_redshift_chains(
        args.chains,
        args.cohort,
        args.out,
        expected_objects=args.expected_objects,
        samples=args.samples,
        z_index=args.z_index,
        read_chunk_size=args.read_chunk_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[popcosmos-chains] complete -> {args.out}")


if __name__ == "__main__":
    main()
