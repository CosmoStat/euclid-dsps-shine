#!/usr/bin/env python3
"""Build a fixed worst-support/control panel for proposal diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.amortized.data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from euclid_dsps.amortized.features import read_feature_stats
from euclid_dsps.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance-diagnostics", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worst-objects", type=int, default=32)
    parser.add_argument("--control-objects", type=int, default=32)
    parser.add_argument("--healthy-pareto-k-max", type=float, default=0.5)
    parser.add_argument("--n-shards", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=260819)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.importance_diagnostics,
        args.config,
        args.dataset,
        args.feature_stats,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic panel: {args.out}")
    if args.worst_objects <= 0 or args.control_objects != args.worst_objects:
        raise ValueError("require an equal positive number of worst/control objects")
    if args.n_shards <= 0 or 2 * args.worst_objects < args.n_shards:
        raise ValueError("invalid n_shards for panel size")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")

    diagnostics = pd.read_parquet(args.importance_diagnostics)
    required = {"row_index", "pareto_k", "raw_ess_fraction"}
    missing = sorted(required - set(diagnostics.columns))
    if missing:
        raise ValueError(f"importance diagnostics missing columns: {missing}")
    diagnostics = diagnostics.drop_duplicates("row_index", keep=False).copy()
    diagnostics = diagnostics.loc[
        np.isfinite(diagnostics["pareto_k"])
        & np.isfinite(diagnostics["raw_ess_fraction"])
    ].reset_index(drop=True)
    if len(diagnostics) < 2 * args.worst_objects:
        raise ValueError("importance probe is too small for the requested panel")

    rows = diagnostics["row_index"].to_numpy(dtype=np.int64)
    features = _load_features(
        args.config, args.dataset, args.feature_stats, row_indices=rows
    )
    if features.shape[0] != len(diagnostics):
        raise RuntimeError("feature and importance cohorts differ")
    if not np.all(np.isfinite(features)):
        raise ValueError("frozen encoder features contain non-finite values")
    feature_by_row = {int(row): features[index] for index, row in enumerate(rows)}

    worst = diagnostics.nlargest(args.worst_objects, "pareto_k").copy()
    healthy = diagnostics.loc[
        (diagnostics["pareto_k"] <= float(args.healthy_pareto_k_max))
        & ~diagnostics["row_index"].isin(worst["row_index"])
    ].copy()
    if len(healthy) < args.control_objects:
        raise ValueError(
            "not enough healthy controls below Pareto-k threshold: "
            f"need={args.control_objects} found={len(healthy)}"
        )

    controls = []
    available = set(int(value) for value in healthy["row_index"])
    healthy_lookup = healthy.set_index("row_index", drop=False)
    for worst_row in worst["row_index"].to_numpy(dtype=np.int64):
        candidate_rows = np.asarray(sorted(available), dtype=np.int64)
        candidate_features = np.stack(
            [feature_by_row[int(value)] for value in candidate_rows]
        )
        distance = np.linalg.norm(
            candidate_features - feature_by_row[int(worst_row)][None, :], axis=1
        )
        selected = int(candidate_rows[int(np.argmin(distance))])
        record = healthy_lookup.loc[selected].to_dict()
        record["matched_to_row_index"] = int(worst_row)
        record["feature_distance_to_match"] = float(np.min(distance))
        controls.append(record)
        available.remove(selected)
    controls = pd.DataFrame(controls)

    worst["panel_role"] = "worst_support"
    worst["matched_to_row_index"] = worst["row_index"].astype(np.int64)
    worst["feature_distance_to_match"] = 0.0
    controls["panel_role"] = "healthy_control"
    panel = pd.concat([worst, controls], ignore_index=True)
    panel = _assign_split(
        panel,
        validation_fraction=float(args.validation_fraction),
        seed=int(args.seed),
    )
    rng = np.random.default_rng(int(args.seed) + 1)
    panel = panel.iloc[rng.permutation(len(panel))].reset_index(drop=True)

    args.out.mkdir(parents=True)
    indices = panel["row_index"].to_numpy(dtype=np.int64)
    np.save(args.out / "smc_calibration_indices.npy", indices)
    np.save(
        args.out / "expressivity_train_indices.npy",
        panel.loc[panel["expressivity_split"] == "train", "row_index"].to_numpy(
            dtype=np.int64
        ),
    )
    np.save(
        args.out / "expressivity_validation_indices.npy",
        panel.loc[panel["expressivity_split"] == "validation", "row_index"].to_numpy(
            dtype=np.int64
        ),
    )
    for shard, shard_indices in enumerate(np.array_split(indices, args.n_shards)):
        np.save(
            args.out / f"smc_calibration_indices_shard_{shard:03d}.npy",
            np.asarray(shard_indices, dtype=np.int64),
        )
    panel.to_csv(args.out / "diagnostic_panel.csv", index=False)
    panel.to_parquet(args.out / "diagnostic_panel.parquet", index=False)
    summary = {
        "status": "complete",
        "cohort_role": (
            "proposal-support diagnosis and architecture tuning only; not an "
            "untouched production confirmation cohort"
        ),
        "n_objects": int(len(panel)),
        "n_worst_support": int((panel["panel_role"] == "worst_support").sum()),
        "n_healthy_control": int((panel["panel_role"] == "healthy_control").sum()),
        "n_train": int((panel["expressivity_split"] == "train").sum()),
        "n_validation": int((panel["expressivity_split"] == "validation").sum()),
        "n_shards": int(args.n_shards),
        "healthy_pareto_k_max": float(args.healthy_pareto_k_max),
        "matching": "greedy nearest neighbor in frozen standardized encoder features",
        "seed": int(args.seed),
        "inputs": {
            "importance_diagnostics": _receipt(args.importance_diagnostics),
            "config": _receipt(args.config),
            "dataset": _receipt(args.dataset),
            "feature_stats": _receipt(args.feature_stats),
        },
        "artifacts": {
            "panel_csv": _receipt(args.out / "diagnostic_panel.csv"),
            "panel_parquet": _receipt(args.out / "diagnostic_panel.parquet"),
            "smc_indices": _receipt(args.out / "smc_calibration_indices.npy"),
            "train_indices": _receipt(args.out / "expressivity_train_indices.npy"),
            "validation_indices": _receipt(
                args.out / "expressivity_validation_indices.npy"
            ),
        },
    }
    (args.out / "diagnostic_panel_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out / "DONE").touch()
    print(
        "[proposal-diagnostic-panel] "
        f"objects={len(panel)} worst={summary['n_worst_support']} "
        f"controls={summary['n_healthy_control']} -> {args.out}",
        flush=True,
    )


def _load_features(config_path, dataset, feature_stats_path, *, row_indices):
    config = load_config(config_path)
    config["catalog_path"] = str(dataset)
    arrays = load_photometry_arrays_from_config(
        config, batch_size=10_000, row_indices=row_indices
    )
    if arrays.row_index is None:
        raise ValueError("selected catalog does not expose stable row indices")
    position = {int(value): index for index, value in enumerate(arrays.row_index)}
    order = np.asarray([position[int(value)] for value in row_indices], dtype=int)
    batch = next(
        iter_photometry_batches_from_arrays(
            arrays,
            batch_size=len(row_indices),
            feature_stats=read_feature_stats(feature_stats_path),
            order=order,
        )
    )
    if not np.array_equal(np.asarray(batch.row_index), row_indices):
        raise RuntimeError("feature rows do not preserve requested order")
    return np.asarray(batch.features, dtype=np.float32)


def _assign_split(panel, *, validation_fraction, seed):
    result = panel.copy()
    result["expressivity_split"] = ""
    rng = np.random.default_rng(int(seed))
    for _role, group in result.groupby("panel_role", sort=False):
        positions = group.index.to_numpy(dtype=np.int64)
        order = rng.permutation(positions)
        n_validation = min(
            len(order) - 1, max(1, int(round(validation_fraction * len(order))))
        )
        result.loc[order[:n_validation], "expressivity_split"] = "validation"
        result.loc[order[n_validation:], "expressivity_split"] = "train"
    return result


def _receipt(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }


if __name__ == "__main__":
    main()
