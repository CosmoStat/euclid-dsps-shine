#!/usr/bin/env python3
"""Build truth-free observed-r<25 manifests for FENIKS RWS recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from euclid_dsps.photometry import abmag_to_fnu_cgs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_rows(path: Path, *, band: str, max_mag_ab: float) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    column = f"flux_{band}"
    schema = pq.ParquetFile(path).schema_arrow
    if column not in schema.names:
        raise ValueError(f"catalog is missing observed selection column {column}")
    flux = pq.read_table(path, columns=[column]).column(column).to_numpy()
    threshold = float(abmag_to_fnu_cgs(float(max_mag_ab)))
    selected = np.flatnonzero(np.isfinite(flux) & (flux > threshold))
    if not len(selected):
        raise ValueError(f"observed {band}<{max_mag_ab:g} selection is empty")
    return np.asarray(selected, dtype=np.int64)


def _write(path: Path, values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"manifest {path.name} must be a non-empty vector")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"manifest {path.name} contains duplicate rows")
    np.save(path, np.sort(values), allow_pickle=False)
    return {
        "path": str(path),
        "count": int(len(values)),
        "minimum": int(np.min(values)),
        "maximum": int(np.max(values)),
        "sha256": _sha256(path),
    }


def build(
    *,
    train_catalog: Path,
    test_catalog: Path,
    out: Path,
    validation_objects: int,
    pilot_objects: int,
    confirmation_objects: int,
    seed: int,
    band: str = "lsst_r",
    max_mag_ab: float = 25.0,
) -> dict[str, object]:
    sizes = (validation_objects, pilot_objects, confirmation_objects)
    if any(int(value) <= 0 for value in sizes):
        raise ValueError("all recovery manifest sizes must be positive")
    selected_train = _selected_rows(
        train_catalog, band=band, max_mag_ab=max_mag_ab
    )
    selected_test = _selected_rows(test_catalog, band=band, max_mag_ab=max_mag_ab)
    if len(selected_train) <= int(validation_objects):
        raise ValueError("training selection is too small for held-out validation")
    requested_test = int(pilot_objects) + int(confirmation_objects)
    if len(selected_test) < requested_test:
        raise ValueError(
            f"test selection has {len(selected_test)} rows, need {requested_test}"
        )
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(int(seed))
    train_order = rng.permutation(selected_train)
    test_order = rng.permutation(selected_test)
    validation = train_order[: int(validation_objects)]
    training = train_order[int(validation_objects) :]
    pilot = test_order[: int(pilot_objects)]
    confirmation = test_order[
        int(pilot_objects) : int(pilot_objects) + int(confirmation_objects)
    ]
    payload = {
        "status": "complete",
        "contract": (
            "Observed photometry only: row identities are selected with "
            f"flux_{band} > f_nu({max_mag_ab:g} AB). No truth column is read. "
            "Training/validation use the train parquet; pilot/confirmation are "
            "disjoint rows from the independent test parquet."
        ),
        "selection": {
            "band": band,
            "max_mag_ab": float(max_mag_ab),
            "flux_min_fnu_cgs": float(abmag_to_fnu_cgs(float(max_mag_ab))),
            "train_selected": int(len(selected_train)),
            "test_selected": int(len(selected_test)),
        },
        "seed": int(seed),
        "truth_columns_requested": [],
        "truth_used_for_training_or_checkpoint_selection": False,
        "catalogs": {
            "train": {"path": str(train_catalog), "sha256": _sha256(train_catalog)},
            "test": {"path": str(test_catalog), "sha256": _sha256(test_catalog)},
        },
        "manifests": {
            "train": _write(out / "train_indices.npy", training),
            "validation": _write(out / "validation_indices.npy", validation),
            "pilot": _write(out / "pilot_indices.npy", pilot),
            "confirmation": _write(
                out / "confirmation_indices.npy", confirmation
            ),
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-catalog", type=Path, required=True)
    parser.add_argument("--test-catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-objects", type=int, default=614)
    parser.add_argument("--pilot-objects", type=int, default=512)
    parser.add_argument("--confirmation-objects", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--band", default="lsst_r")
    parser.add_argument("--max-mag-ab", type=float, default=25.0)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
