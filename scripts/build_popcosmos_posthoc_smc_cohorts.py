#!/usr/bin/env python3
"""Build disjoint, optionally progressive Pop-COSMOS SMC cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-indices", type=Path, required=True)
    parser.add_argument("--evaluation-indices", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smc-objects", type=int, required=True)
    parser.add_argument("--probe-objects", type=int, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument("--seed", type=int, default=260817)
    parser.add_argument("--parent-smc-root", type=Path)
    return parser.parse_args()


def build_cohorts(
    *,
    calibration_path: Path,
    evaluation_path: Path,
    out: Path,
    smc_objects: int,
    probe_objects: int,
    n_shards: int,
    seed: int,
    parent_root: Path | None = None,
) -> dict[str, object]:
    if min(smc_objects, probe_objects, n_shards) <= 0:
        raise ValueError("SMC objects, probe objects and shards must be positive")
    if n_shards > smc_objects:
        raise ValueError("Number of shards cannot exceed SMC objects")

    calibration = _load_unique_indices(calibration_path, "calibration")
    evaluation = _load_unique_indices(evaluation_path, "evaluation")
    if calibration.size < smc_objects + probe_objects:
        raise ValueError(
            "Not enough calibration rows for disjoint SMC and probe cohorts"
        )

    selected = np.random.default_rng(seed).permutation(calibration)[
        : smc_objects + probe_objects
    ]
    smc = selected[:smc_objects]
    probe = selected[smc_objects:]
    if np.intersect1d(smc, probe).size:
        raise ValueError("SMC and proposal-probe cohorts overlap")
    if np.intersect1d(selected, evaluation).size:
        raise ValueError("SMC or proposal-probe cohort overlaps evaluation")

    progressive_parent = None
    if parent_root is not None:
        progressive_parent = _validate_progressive_parent(
            parent_root=parent_root,
            smc=smc,
            probe=probe,
        )

    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "smc_calibration_indices.npy", smc)
    np.save(out / "proposal_probe_indices.npy", probe)
    for index, shard in enumerate(np.array_split(smc, n_shards)):
        np.save(out / f"smc_calibration_indices_shard_{index:03d}.npy", shard)

    payload: dict[str, object] = {
        "status": "complete",
        "selection": "seeded random disjoint subsets of frozen validation rows",
        "seed": seed,
        "smc_objects": smc_objects,
        "proposal_probe_objects": probe_objects,
        "n_shards": n_shards,
        "spectroscopic_evaluation_overlap": 0,
        "source_calibration_indices": _receipt(calibration_path),
        "source_evaluation_indices": _receipt(evaluation_path),
    }
    if progressive_parent is not None:
        payload["progressive_parent"] = progressive_parent
    (out / "cohort_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _validate_progressive_parent(
    *, parent_root: Path, smc: np.ndarray, probe: np.ndarray
) -> dict[str, object]:
    parent_smc_path = parent_root / "cohorts/smc_calibration_indices.npy"
    parent_probe_path = parent_root / "cohorts/proposal_probe_indices.npy"
    parent_smc = _load_unique_indices(parent_smc_path, "parent SMC")
    parent_probe = _load_unique_indices(parent_probe_path, "parent probe")
    if np.intersect1d(parent_smc, parent_probe).size:
        raise ValueError("Progressive parent SMC and probe cohorts overlap")
    parent_combined = np.concatenate((parent_smc, parent_probe))
    if smc.size != parent_combined.size:
        raise ValueError(
            "Progressive SMC size must equal parent SMC plus parent probe sizes: "
            f"new={smc.size} parent_total={parent_combined.size}"
        )
    if not np.array_equal(smc, parent_combined):
        raise ValueError(
            "New SMC cohort must exactly concatenate parent SMC and parent probe"
        )
    if np.intersect1d(probe, parent_combined).size:
        raise ValueError("New proposal probe overlaps a previously used parent cohort")
    return {
        "root": str(parent_root),
        "contract": "new SMC = parent SMC followed by parent probe",
        "smc_objects": int(parent_smc.size),
        "proposal_probe_objects": int(parent_probe.size),
        "smc_indices": _receipt(parent_smc_path),
        "proposal_probe_indices": _receipt(parent_probe_path),
    }


def _load_unique_indices(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.asarray(np.load(path), dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{label} indices must be a non-empty vector")
    if np.unique(values).size != values.size:
        raise ValueError(f"{label} indices contain duplicates")
    return values


def _receipt(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    build_cohorts(
        calibration_path=args.calibration_indices,
        evaluation_path=args.evaluation_indices,
        out=args.out,
        smc_objects=args.smc_objects,
        probe_objects=args.probe_objects,
        n_shards=args.n_shards,
        seed=args.seed,
        parent_root=args.parent_smc_root,
    )


if __name__ == "__main__":
    main()
