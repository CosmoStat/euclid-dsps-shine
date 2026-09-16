#!/usr/bin/env python3
"""Collect dense exact-benchmark draws for joint MIRA/TARP diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_METHODS = {
    "encoder": "encoder_samples.parquet",
    "q_is": "importance_resampled_samples.parquet",
    "defensive_is": "defensive_importance_resampled_samples.parquet",
    "nuts": "nuts/samples.parquet",
}
OPTIONAL_METHODS = {
    "adaptive_smc": "adaptive_smc_resampled_samples.parquet",
}


def _galaxy_dir(root: Path, item) -> Path:
    return (
        root
        / "galaxies"
        / (f"{int(item.order):02d}_{item.example_key}_row{int(item.row_index)}")
    )


def collect(
    *,
    root: Path,
    samples_per_object: int,
    domain: str | None = None,
) -> dict[str, object]:
    cohort = pd.read_parquet(root / "cohort.parquet")
    if domain is not None:
        if "domain" not in cohort:
            raise ValueError("cohort has no domain column")
        cohort = cohort.loc[cohort["domain"].astype(str).eq(domain)].copy()
        if cohort.empty:
            raise ValueError(f"cohort has no rows for domain={domain}")
    methods = dict(REQUIRED_METHODS)
    for name, relative in OPTIONAL_METHODS.items():
        present = [
            (_galaxy_dir(root, item) / relative).is_file()
            for item in cohort.itertuples(index=False)
        ]
        if any(present) and not all(present):
            raise ValueError(f"optional method {name} is incomplete across the cohort")
        if all(present):
            methods[name] = relative
    truth_rows = []
    posterior_rows = {name: [] for name in methods}
    parameter_names: tuple[str, ...] | None = None
    for item in cohort.itertuples(index=False):
        galaxy = _galaxy_dir(root, item)
        manifest = json.loads((galaxy / "prepare_manifest.json").read_text())
        current_names = tuple(manifest["latent_spec"]["names"])
        if parameter_names is None:
            parameter_names = current_names
        elif current_names != parameter_names:
            raise ValueError("inconsistent latent parameter names across cohort")
        truth = pd.read_parquet(galaxy / "truth.parquet").iloc[[0]].copy()
        truth.insert(0, "row_index", int(item.row_index))
        truth.insert(0, "object_id", str(item.object_id))
        truth_rows.append(truth)
        for name, relative in methods.items():
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
    out = root / ("calibration" if domain is None else f"calibration_{domain}")
    out.mkdir(parents=True, exist_ok=True)
    truth = pd.concat(truth_rows, ignore_index=True)
    truth.to_parquet(out / "inference_truth.parquet", index=False)
    artifacts = {"truth": str(out / "inference_truth.parquet")}
    posterior_frames = {}
    for name, pieces in posterior_rows.items():
        method_dir = out / name
        method_dir.mkdir(parents=True, exist_ok=True)
        path = method_dir / "posterior_samples.parquet"
        posterior_frames[name] = pd.concat(pieces, ignore_index=True)
        posterior_frames[name].to_parquet(path, index=False)
        artifacts[name] = str(path)
    coverage_rows = []
    truth_by_row = truth.set_index("row_index")
    for method, frame in posterior_frames.items():
        for parameter in parameter_names or ():
            covered_68 = []
            covered_95 = []
            for row_index, draws in frame.groupby("row_index", sort=False):
                value = float(truth_by_row.loc[int(row_index), parameter])
                quantiles = np.quantile(
                    draws[parameter].to_numpy(dtype=float),
                    [0.025, 0.16, 0.84, 0.975],
                )
                covered_68.append(quantiles[1] <= value <= quantiles[2])
                covered_95.append(quantiles[0] <= value <= quantiles[3])
            coverage_rows.append(
                {
                    "method": method,
                    "parameter": parameter,
                    "objects": len(covered_68),
                    "coverage_68": float(np.mean(covered_68)),
                    "coverage_95": float(np.mean(covered_95)),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out / "central_coverage.csv", index=False)
    coverage.to_parquet(out / "central_coverage.parquet", index=False)
    artifacts["central_coverage"] = str(out / "central_coverage.parquet")
    payload = {
        "status": "complete",
        "objects": int(len(cohort)),
        "domain": domain or "all",
        "samples_per_object": int(samples_per_object),
        "methods": list(methods),
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
    parser.add_argument("--domain")
    args = parser.parse_args()
    print(json.dumps(collect(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
