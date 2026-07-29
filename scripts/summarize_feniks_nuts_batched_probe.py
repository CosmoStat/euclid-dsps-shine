#!/usr/bin/env python3
"""Compare scalar and vmapped NUTS throughput on the first pilot galaxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    cohort = pd.read_parquet(args.root / "cohort.parquet")
    row = cohort.iloc[0]
    galaxy = (
        args.root
        / "galaxies"
        / f"{int(row['order']):02d}_{row['example_key']}_row{int(row['row_index'])}"
    )
    manifests = [
        json.loads(
            (galaxy / "nuts" / f"chain_{index:02d}" / "chain_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        for index in range(4)
    ]
    contract_keys = (
        "warmup_steps",
        "target_accept",
        "max_num_doublings",
        "sample_chunks",
        "stored_samples",
    )
    scalar_contract = {name: manifests[0][name] for name in contract_keys}
    compatible = all(
        {name: manifest[name] for name in contract_keys} == scalar_contract
        for manifest in manifests[1:]
    )
    batched_execution = all(
        manifest.get("execution") == "vmap_batched_chains"
        for manifest in manifests[1:]
    )
    scalar_elapsed = float(manifests[0]["total_elapsed_s"])
    batched_elapsed = float(
        max(manifest["total_elapsed_s"] for manifest in manifests[1:])
    )
    scalar_draws = int(manifests[0]["stored_samples"])
    batched_draws = sum(
        int(manifest["stored_samples"]) for manifest in manifests[1:]
    )
    scalar_info = _read_info(galaxy / "nuts" / "chain_00")
    batched_info = pd.concat(
        [
            _read_info(galaxy / "nuts" / f"chain_{index:02d}")
            for index in range(1, 4)
        ],
        ignore_index=True,
    )
    finite_samples = all(
        np.isfinite(
            pd.concat(
                [
                    pd.read_parquet(path)
                    for path in sorted(
                        (
                            galaxy / "nuts" / f"chain_{index:02d}" / "chunks"
                        ).glob("part_*.parquet")
                    )
                    if not path.name.endswith("_info.parquet")
                ],
                ignore_index=True,
            )
            .filter(regex=r"^x_")
            .to_numpy(dtype=float)
        ).all()
        for index in range(4)
    )
    summary = {
        "status": (
            "passed"
            if compatible and batched_execution and finite_samples
            else "failed"
        ),
        "galaxy": galaxy.name,
        "contract": scalar_contract,
        "compatible_contracts": compatible,
        "batched_execution": batched_execution,
        "finite_samples": finite_samples,
        "scalar_total_elapsed_s": scalar_elapsed,
        "batched_three_chain_total_elapsed_s": batched_elapsed,
        "scalar_draws_per_second": scalar_draws / scalar_elapsed,
        "batched_draws_per_second": batched_draws / batched_elapsed,
        "throughput_speedup": (
            (batched_draws / batched_elapsed)
            / (scalar_draws / scalar_elapsed)
        ),
        "scalar_mean_acceptance": _mean_column(
            scalar_info,
            "acceptance_rate",
        ),
        "batched_mean_acceptance": _mean_column(
            batched_info,
            "acceptance_rate",
        ),
        "scalar_divergences": _sum_column(scalar_info, "is_divergent"),
        "batched_divergences": _sum_column(batched_info, "is_divergent"),
        "scalar_mean_integration_steps": _mean_column(
            scalar_info,
            "num_integration_steps",
        ),
        "batched_mean_integration_steps": _mean_column(
            batched_info,
            "num_integration_steps",
        ),
    }
    output = galaxy / "nuts" / "batched_probe_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if summary["status"] != "passed":
        raise SystemExit(2)


def _read_info(chain: Path) -> pd.DataFrame:
    paths = sorted((chain / "chunks").glob("part_*_info.parquet"))
    if not paths:
        raise FileNotFoundError(f"No NUTS info chunks in {chain}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _mean_column(frame: pd.DataFrame, name: str) -> float | None:
    if name not in frame:
        return None
    values = frame[name].to_numpy(dtype=float)
    return float(np.mean(values)) if np.isfinite(values).all() else None


def _sum_column(frame: pd.DataFrame, name: str) -> int | None:
    if name not in frame:
        return None
    return int(frame[name].astype(bool).sum())


if __name__ == "__main__":
    main()
