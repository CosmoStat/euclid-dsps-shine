#!/usr/bin/env python3
"""Build immutable truth-free manifests for the FENIKS SC-DRWS workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from euclid_dsps.photometry import abmag_to_fnu_cgs

SCOPE_STATEMENT = (
    "We infer the parent distribution within the predefined FENIKS "
    "refinement and catalogue-support domain, while explicitly correcting "
    "the additional observed r<27.5 selection."
)
RETENTION_MAGNITUDES = (25.0, 26.0, 27.0, 27.5, 28.0)


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


def _retention_grid(path: Path, *, band: str) -> dict[str, dict[str, float | int]]:
    column = f"flux_{band}"
    schema = pq.ParquetFile(path).schema_arrow
    if column not in schema.names:
        raise ValueError(f"catalog is missing observed selection column {column}")
    flux = pq.read_table(path, columns=[column]).column(column).to_numpy()
    finite = np.isfinite(flux)
    total = int(len(flux))
    if total <= 0:
        raise ValueError(f"catalog is empty: {path}")
    return {
        f"{magnitude:g}": {
            "max_mag_ab": float(magnitude),
            "retained_rows": int(
                np.sum(finite & (flux > float(abmag_to_fnu_cgs(magnitude))))
            ),
            "retained_fraction_of_c0_catalog": float(
                np.mean(finite & (flux > float(abmag_to_fnu_cgs(magnitude))))
            ),
        }
        for magnitude in RETENTION_MAGNITUDES
    }


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
    final_validation_objects: int = 512,
    band: str = "lsst_r",
    max_mag_ab: float = 27.5,
    minimum_retained_fraction: float = 0.90,
) -> dict[str, object]:
    sizes = (
        validation_objects,
        pilot_objects,
        confirmation_objects,
        final_validation_objects,
    )
    if any(int(value) <= 0 for value in sizes):
        raise ValueError("all recovery manifest sizes must be positive")
    train_retention = _retention_grid(train_catalog, band=band)
    test_retention = _retention_grid(test_catalog, band=band)
    configured_key = f"{float(max_mag_ab):g}"
    if configured_key not in train_retention:
        raise ValueError(
            f"configured max_mag_ab={max_mag_ab:g} is not in the required "
            f"retention grid {RETENTION_MAGNITUDES}"
        )
    retained_fraction = float(
        train_retention[configured_key]["retained_fraction_of_c0_catalog"]
    )
    if retained_fraction < float(minimum_retained_fraction):
        recommendation = next(
            (
                item["max_mag_ab"]
                for item in train_retention.values()
                if float(item["max_mag_ab"]) > float(max_mag_ab)
                and float(item["retained_fraction_of_c0_catalog"])
                >= float(minimum_retained_fraction)
            ),
            None,
        )
        suffix = (
            f"; recommend max_mag_ab={recommendation:g}"
            if recommendation is not None
            else "; no configured retention-grid value reaches the requirement"
        )
        raise ValueError(
            f"observed {band}<{max_mag_ab:g} retains {retained_fraction:.3%}, "
            f"below required {float(minimum_retained_fraction):.1%}{suffix}"
        )
    selected_train = _selected_rows(
        train_catalog, band=band, max_mag_ab=max_mag_ab
    )
    selected_test = _selected_rows(test_catalog, band=band, max_mag_ab=max_mag_ab)
    requested_train = (
        int(validation_objects) + int(pilot_objects) + int(confirmation_objects)
    )
    if len(selected_train) < requested_train:
        raise ValueError(
            f"training selection has {len(selected_train)} rows, need "
            f"{requested_train} for disjoint validation/pilot/confirmation cohorts"
        )
    requested_test = (
        int(pilot_objects)
        + int(confirmation_objects)
        + int(final_validation_objects)
    )
    if len(selected_test) < requested_test:
        raise ValueError(
            f"test selection has {len(selected_test)} rows, need {requested_test}"
        )
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(int(seed))
    train_order = rng.permutation(selected_train)
    test_order = rng.permutation(selected_test)
    validation = train_order[: int(validation_objects)]
    pilot_train_start = int(validation_objects)
    pilot_train_stop = pilot_train_start + int(pilot_objects)
    confirmation_train_stop = pilot_train_stop + int(confirmation_objects)
    pilot_train = train_order[pilot_train_start:pilot_train_stop]
    confirmation_train = train_order[pilot_train_stop:confirmation_train_stop]
    training = train_order[int(validation_objects) :]
    pilot = test_order[: int(pilot_objects)]
    confirmation = test_order[
        int(pilot_objects) : int(pilot_objects) + int(confirmation_objects)
    ]
    final_validation = test_order[
        int(pilot_objects) + int(confirmation_objects) : requested_test
    ]
    payload = {
        "status": "complete",
        "contract": (
            "Observed photometry only: row identities are selected with "
            f"flux_{band} > f_nu({max_mag_ab:g} AB). No truth column is read. "
            "Training/validation use the train parquet; pilot/confirmation are "
            "disjoint rows from the independent test parquet."
        ),
        "c0_scope_statement": SCOPE_STATEMENT,
        "target_population": "p_eta(theta | C0)",
        "explicit_selection": f"A = 1[m_{band}_observed < {max_mag_ab:g}]",
        "upstream_true_space_selection": "conditioned_as_C0_not_inverted",
        "selection": {
            "band": band,
            "max_mag_ab": float(max_mag_ab),
            "flux_min_fnu_cgs": float(abmag_to_fnu_cgs(float(max_mag_ab))),
            "train_selected": int(len(selected_train)),
            "test_selected": int(len(selected_test)),
            "minimum_retained_fraction": float(minimum_retained_fraction),
            "configured_train_retained_fraction": retained_fraction,
            "retention_grid": {
                "train": train_retention,
                "test": test_retention,
            },
        },
        "seed": int(seed),
        "truth_columns_requested": [],
        "truth_used_for_training_or_checkpoint_selection": False,
        "catalogs": {
            "train": {"path": str(train_catalog), "sha256": _sha256(train_catalog)},
            "test": {"path": str(test_catalog), "sha256": _sha256(test_catalog)},
        },
        "manifests": {
            "full_train": _write(out / "full_train_indices.npy", selected_train),
            "full_test": _write(out / "full_test_indices.npy", selected_test),
            "train": _write(out / "train_indices.npy", training),
            "pilot_train": _write(
                out / "pilot_train_indices.npy", pilot_train
            ),
            "confirmation_train": _write(
                out / "confirmation_train_indices.npy", confirmation_train
            ),
            "validation": _write(out / "validation_indices.npy", validation),
            "pilot": _write(out / "pilot_indices.npy", pilot),
            "confirmation": _write(
                out / "confirmation_indices.npy", confirmation
            ),
            "final_validation": _write(
                out / "final_validation_indices.npy", final_validation
            ),
        },
        "final_full_dataset_contract": {
            "manifest": str(out / "full_train_indices.npy"),
            "expected_rows": int(len(selected_train)),
            "uses_every_observed_selected_training_row": True,
        },
        "training_promotion_contract": {
            "pilot_rows": int(len(pilot_train)),
            "confirmation_rows": int(len(confirmation_train)),
            "pilot_and_confirmation_are_disjoint": bool(
                set(pilot_train).isdisjoint(set(confirmation_train))
            ),
            "final_uses_every_selected_row": True,
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
    parser.add_argument("--final-validation-objects", type=int, default=512)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--band", default="lsst_r")
    parser.add_argument("--max-mag-ab", type=float, default=27.5)
    parser.add_argument("--minimum-retained-fraction", type=float, default=0.90)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
