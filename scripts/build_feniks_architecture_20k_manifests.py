#!/usr/bin/env python3
"""Build immutable truth-free row manifests for the FENIKS architecture battle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _rows(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    return int(pq.ParquetFile(path).metadata.num_rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_indices(path: Path, values: np.ndarray) -> dict[str, object]:
    np.save(path, np.asarray(values, dtype=np.int64), allow_pickle=False)
    return {
        "path": str(path),
        "count": int(len(values)),
        "sha256": _sha256(path),
        "minimum": int(np.min(values)),
        "maximum": int(np.max(values)),
    }


def build(
    *,
    train_catalog: Path,
    test_catalog: Path,
    out: Path,
    n_train: int,
    n_validation: int,
    n_probe: int,
    seed: int,
) -> dict[str, object]:
    train_rows = _rows(train_catalog)
    test_rows = _rows(test_catalog)
    requested = int(n_train) + int(n_validation)
    if min(n_train, n_validation, n_probe) <= 0:
        raise ValueError("manifest sizes must be positive")
    if requested > train_rows:
        raise ValueError(
            f"requested {requested} train/validation rows but catalog has {train_rows}"
        )
    if int(n_probe) > test_rows:
        raise ValueError(
            f"requested {n_probe} probe rows but test catalog has {test_rows}"
        )
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(int(seed))
    selected = rng.permutation(train_rows)[:requested]
    train_indices = np.sort(selected[: int(n_train)])
    validation_indices = np.sort(selected[int(n_train) :])
    probe_indices = np.sort(rng.permutation(test_rows)[: int(n_probe)])
    payload = {
        "status": "complete",
        "contract": (
            "FENIKS row identities only; no truth column is read; train/validation "
            "select checkpoints and the separate test probe is used for ordinary IW"
        ),
        "seed": int(seed),
        "train_catalog": {
            "path": str(train_catalog),
            "rows": train_rows,
            "size_bytes": train_catalog.stat().st_size,
        },
        "test_catalog": {
            "path": str(test_catalog),
            "rows": test_rows,
            "size_bytes": test_catalog.stat().st_size,
        },
        "manifests": {
            "train": _write_indices(out / "train_indices.npy", train_indices),
            "validation": _write_indices(
                out / "validation_indices.npy", validation_indices
            ),
            "blind_iw_probe": _write_indices(
                out / "blind_iw_probe_indices.npy", probe_indices
            ),
        },
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
    parser.add_argument("--n-train", type=int, default=18_000)
    parser.add_argument("--n-validation", type=int, default=2_000)
    parser.add_argument("--n-probe", type=int, default=256)
    parser.add_argument("--seed", type=int, default=260820)
    args = parser.parse_args()
    payload = build(**vars(args))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
