#!/usr/bin/env python3
"""Build a deterministic COSMOS evaluation cohort outside the 40k train pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=260731)
    return parser.parse_args()


def build_indices(
    full_ids: np.ndarray,
    excluded_ids: np.ndarray,
    *,
    size: int,
    seed: int,
) -> np.ndarray:
    full_ids = np.asarray(full_ids)
    excluded_ids = np.asarray(excluded_ids)
    if pd.Series(full_ids).duplicated().any():
        raise ValueError("Full COSMOS catalog contains duplicate object_id values")
    if pd.Series(excluded_ids).duplicated().any():
        raise ValueError("Training pool contains duplicate object_id values")
    eligible = np.flatnonzero(~np.isin(full_ids, excluded_ids)).astype(np.int64)
    if size <= 0 or size > len(eligible):
        raise ValueError(
            f"Evaluation size must be in [1, {len(eligible)}], got {size}"
        )
    rng = np.random.default_rng(int(seed))
    return rng.choice(eligible, size=int(size), replace=False).astype(np.int64)


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _ids_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    full_ids = pd.read_parquet(args.full, columns=["object_id"])[
        "object_id"
    ].to_numpy()
    excluded_ids = pd.read_parquet(args.exclude, columns=["object_id"])[
        "object_id"
    ].to_numpy()
    selected = build_indices(
        full_ids,
        excluded_ids,
        size=args.size,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        existing = np.asarray(np.load(args.out), dtype=np.int64)
        if not np.array_equal(existing, selected):
            raise RuntimeError(
                f"Existing evaluation cohort differs from contract: {args.out}"
            )
    else:
        np.save(args.out, selected)
    selected_ids = full_ids[selected]
    if np.isin(selected_ids, excluded_ids).any():
        raise RuntimeError("Evaluation cohort overlaps the 40k training pool")
    manifest = {
        "status": "complete",
        "full_catalog": str(args.full),
        "excluded_training_pool": str(args.exclude),
        "selection": "uniform_without_replacement_no_redshift",
        "seed": int(args.seed),
        "n_full": int(len(full_ids)),
        "n_excluded": int(len(excluded_ids)),
        "n_eligible": int((~np.isin(full_ids, excluded_ids)).sum()),
        "n_selected": int(len(selected)),
        "row_indices_sha256": _array_sha256(selected),
        "object_ids_sha256": _ids_sha256(selected_ids),
        "redshift_used_for_selection": False,
    }
    manifest_path = args.out.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        "[cosmos-native15d-eval] "
        f"selected={len(selected)} excluded={len(excluded_ids)} -> {args.out}"
    )


if __name__ == "__main__":
    main()
