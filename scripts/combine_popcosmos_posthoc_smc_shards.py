#!/usr/bin/env python3
"""Combine one likelihood/seed SMC run distributed over object shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-objects", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_shards <= 0 or args.expected_objects <= 0:
        raise ValueError("Expected shard and object counts must be positive")
    if (args.root / "DONE").exists():
        raise FileExistsError(f"Combined SMC run already exists: {args.root}")
    shards = [args.root / f"shard_{index:03d}" for index in range(args.expected_shards)]
    summaries = []
    object_frames = []
    stage_frames = []
    band_frames = []
    row_chunks = []
    for shard in shards:
        for path in (
            shard / "DONE",
            shard / "smc_summary.json",
            shard / "smc_object_diagnostics.parquet",
            shard / "smc_stage_diagnostics.parquet",
            shard / "posterior_predictive_band_objects.parquet",
            shard / "row_indices.npy",
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        summaries.append(json.loads((shard / "smc_summary.json").read_text()))
        object_frames.append(pd.read_parquet(shard / "smc_object_diagnostics.parquet"))
        stage_frames.append(pd.read_parquet(shard / "smc_stage_diagnostics.parquet"))
        band_frames.append(
            pd.read_parquet(shard / "posterior_predictive_band_objects.parquet")
        )
        row_chunks.append(
            np.asarray(np.load(shard / "row_indices.npy"), dtype=np.int64)
        )

    _validate_common_contract(summaries)
    row_indices = np.concatenate(row_chunks)
    if len(row_indices) != args.expected_objects:
        raise RuntimeError(
            f"Expected {args.expected_objects} objects, found {len(row_indices)}"
        )
    if len(np.unique(row_indices)) != len(row_indices):
        raise RuntimeError("SMC object shards overlap")
    objects = pd.concat(object_frames, ignore_index=True)
    stages = pd.concat(stage_frames, ignore_index=True)
    bands = pd.concat(band_frames, ignore_index=True)
    objects = objects.set_index("row_index").loc[row_indices].reset_index()
    stages = stages.sort_values(["row_index", "stage"]).reset_index(drop=True)
    bands = bands.sort_values(["row_index", "band"]).reset_index(drop=True)
    if len(objects) != args.expected_objects:
        raise RuntimeError("Combined object diagnostics have the wrong size")

    metrics, support_gate = _metrics_and_gate(objects, stages)
    first = summaries[0]
    summary = {
        **first,
        "n_objects": int(len(objects)),
        "wall_seconds": float(max(item["wall_seconds"] for item in summaries)),
        "aggregate_gpu_seconds": float(sum(item["wall_seconds"] for item in summaries)),
        "metrics": metrics,
        "support_gate": support_gate,
        "sharding": {
            "n_shards": int(args.expected_shards),
            "shards": [str(path) for path in shards],
            "particle_storage": "shard-local weighted_particles directories",
        },
        "inputs": {
            **first["inputs"],
            "row_indices": {
                "path": str(args.root / "row_indices.npy"),
                "source_shards": [str(path / "row_indices.npy") for path in shards],
            },
        },
    }
    objects.to_parquet(args.root / "smc_object_diagnostics.parquet", index=False)
    stages.to_parquet(args.root / "smc_stage_diagnostics.parquet", index=False)
    bands.to_parquet(
        args.root / "posterior_predictive_band_objects.parquet", index=False
    )
    _summarize_bands(bands).to_csv(
        args.root / "posterior_predictive_by_band.csv", index=False
    )
    np.save(args.root / "row_indices.npy", row_indices)
    (args.root / "smc_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.root / "support_gate.json").write_text(
        json.dumps(support_gate, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.root / "DONE").touch()
    print(
        f"[posthoc-smc-combine] objects={len(objects)} "
        f"support={support_gate['status']} -> {args.root}",
        flush=True,
    )


def _validate_common_contract(summaries: list[dict]) -> None:
    fields = (
        "algorithm",
        "density_space",
        "particles_per_object",
        "seed",
        "likelihood",
        "target_contract",
        "proposal_contract",
        "selection_contract",
        "target_ess_fraction",
        "mala_steps",
        "mala_step_size",
        "bands",
        "git_commit",
    )
    first = summaries[0]
    for summary in summaries[1:]:
        for field in fields:
            if summary[field] != first[field]:
                raise RuntimeError(f"SMC shard contract mismatch for {field}")


def _metrics_and_gate(objects: pd.DataFrame, stages: pd.DataFrame):
    checks = {
        "all_objects_reached_beta_one": bool(
            np.allclose(stages.groupby("row_index")["beta_to"].max(), 1.0)
        ),
        "median_final_ess_fraction_ge_0p2": bool(
            objects["final_ess_fraction"].median() >= 0.2
        ),
        "median_unique_ancestor_fraction_ge_0p05": bool(
            objects["unique_ancestor_fraction"].median() >= 0.05
        ),
        "median_max_final_weight_le_0p1": bool(
            objects["max_final_weight"].median() <= 0.1
        ),
        "median_mala_acceptance_between_0p15_0p8": bool(
            0.15 <= objects["mean_mala_acceptance"].median() <= 0.8
        ),
        "fraction_objects_final_ess_ge_0p1_ge_0p9": bool(
            np.mean(objects["final_ess_fraction"] >= 0.1) >= 0.9
        ),
        "fraction_objects_max_weight_le_0p2_ge_0p9": bool(
            np.mean(objects["max_final_weight"] <= 0.2) >= 0.9
        ),
        "fraction_objects_mala_acceptance_0p05_0p95_ge_0p9": bool(
            np.mean(objects["mean_mala_acceptance"].between(0.05, 0.95)) >= 0.9
        ),
    }
    metrics = {
        "mean_log_evidence": float(objects["log_evidence"].mean()),
        "median_log_evidence": float(objects["log_evidence"].median()),
        "median_final_ess_fraction": float(objects["final_ess_fraction"].median()),
        "median_unique_ancestor_fraction": float(
            objects["unique_ancestor_fraction"].median()
        ),
        "median_max_final_weight": float(objects["max_final_weight"].median()),
        "median_mala_acceptance": float(objects["mean_mala_acceptance"].median()),
        "p10_final_ess_fraction": float(objects["final_ess_fraction"].quantile(0.1)),
        "p10_unique_ancestor_fraction": float(
            objects["unique_ancestor_fraction"].quantile(0.1)
        ),
        "p90_max_final_weight": float(objects["max_final_weight"].quantile(0.9)),
        "p10_mala_acceptance": float(objects["mean_mala_acceptance"].quantile(0.1)),
        "p90_mala_acceptance": float(objects["mean_mala_acceptance"].quantile(0.9)),
        "median_chi2_per_valid_band": float(
            objects["weighted_chi2_per_valid_band"].median()
        ),
        "median_reduced_chi2": float(objects["weighted_reduced_chi2"].median()),
        "median_fraction_abs_gt_5": float(
            objects["weighted_fraction_abs_gt_5"].median()
        ),
    }
    return metrics, {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def _summarize_bands(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("band", sort=False)
        .agg(
            n_objects=("row_index", "size"),
            median_weighted_abs_chi=("weighted_abs_chi", "median"),
            median_weighted_frac_abs_gt_5=("weighted_frac_abs_gt_5", "median"),
        )
        .reset_index()
    )


if __name__ == "__main__":
    main()
