#!/usr/bin/env python3
"""Collect dense exact-benchmark draws for joint MIRA/TARP diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = {
    "encoder": "encoder_samples.parquet",
    "q_is": "importance_resampled_samples.parquet",
    "defensive_is": "defensive_importance_resampled_samples.parquet",
    "nuts": "nuts/samples.parquet",
}


def _galaxy_dir(root: Path, item) -> Path:
    return (
        root
        / "galaxies"
        / (f"{int(item.order):02d}_{item.example_key}_row{int(item.row_index)}")
    )


def collect(*, root: Path, samples_per_object: int) -> dict[str, object]:
    cohort = pd.read_parquet(root / "cohort.parquet")
    truth_rows = []
    posterior_rows = {name: [] for name in METHODS}
    for item in cohort.itertuples(index=False):
        galaxy = _galaxy_dir(root, item)
        truth = pd.read_parquet(galaxy / "truth.parquet").iloc[[0]].copy()
        truth.insert(0, "row_index", int(item.row_index))
        truth.insert(0, "object_id", str(item.object_id))
        truth_rows.append(truth)
        for name, relative in METHODS.items():
            frame = pd.read_parquet(galaxy / relative)
            if len(frame) < samples_per_object:
                raise ValueError(
                    f"{name} row {item.row_index} has {len(frame)} draws; "
                    f"need {samples_per_object}"
                )
            positions = np.linspace(
                0,
                len(frame) - 1,
                samples_per_object,
                dtype=np.int64,
            )
            selected = frame.iloc[positions].copy()
            selected.insert(0, "sample_id", np.arange(samples_per_object))
            selected.insert(0, "row_index", int(item.row_index))
            selected.insert(0, "object_id", str(item.object_id))
            posterior_rows[name].append(selected)
    out = root / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    truth = pd.concat(truth_rows, ignore_index=True)
    truth.to_parquet(out / "inference_truth.parquet", index=False)
    artifacts = {"truth": str(out / "inference_truth.parquet")}
    for name, pieces in posterior_rows.items():
        method_dir = out / name
        method_dir.mkdir(parents=True, exist_ok=True)
        path = method_dir / "posterior_samples.parquet"
        pd.concat(pieces, ignore_index=True).to_parquet(path, index=False)
        artifacts[name] = str(path)
    payload = {
        "status": "complete",
        "objects": int(len(cohort)),
        "samples_per_object": int(samples_per_object),
        "methods": list(METHODS),
        "artifacts": artifacts,
        "truth_role": "closure diagnostics only; never training or checkpoint selection",
    }
    (out / "collection_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples-per-object", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(collect(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
