#!/usr/bin/env python3
"""Repartition an existing resampled FENIKS 50k into group-disjoint splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir, write_json

SPLITS = ("train", "validation", "test")
DEFAULT_SIZES = {"train": 40_000, "validation": 5_000, "test": 5_000}
OBJECT_ID_STARTS = {"train": 0, "validation": 1_000_000_000, "test": 2_000_000_000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260716)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def effective_proposal_keys(values: pd.Series) -> pd.Series:
    """Canonicalize keys that reused ``source_seed + shard_index``."""
    parts = values.astype(str).str.split(":", expand=True)
    if parts.shape[1] != 4:
        raise ValueError("source_proposal_id must be '<split>:<seed>:<shard>:<row>'")
    source_seed = pd.to_numeric(parts[1], errors="raise").astype(np.int64)
    shard = pd.to_numeric(parts[2], errors="raise").astype(np.int64)
    row = pd.to_numeric(parts[3], errors="raise").astype(np.int64)
    return (source_seed + shard).astype(str) + ":" + row.astype(str)


def grouped_split_assignment(
    keys: pd.Series,
    *,
    target_sizes: dict[str, int],
    seed: int,
) -> dict[str, str]:
    counts = keys.value_counts(sort=False)
    rng = np.random.default_rng(int(seed))
    ordered = counts.index.to_numpy(copy=True)
    rng.shuffle(ordered)
    current = {split: 0 for split in SPLITS}
    assignment: dict[str, str] = {}
    for key in ordered:
        eligible = [split for split in SPLITS if current[split] < target_sizes[split]]
        candidates = eligible or list(SPLITS)
        split = max(
            candidates,
            key=lambda name: (target_sizes[name] - current[name]) / target_sizes[name],
        )
        assignment[str(key)] = split
        current[split] += int(counts.loc[key])
    return assignment


def main() -> None:
    args = parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}")
    ensure_dir(out)
    parts = []
    for split in SPLITS:
        path = args.source_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        if "source_proposal_id" not in frame:
            raise ValueError(f"Missing source_proposal_id in {path}")
        frame = frame.copy()
        frame["original_split"] = split
        frame["original_object_id"] = frame["object_id"].to_numpy()
        parts.append(frame)
    pooled = pd.concat(parts, ignore_index=True)
    if args.limit is not None:
        pooled = pooled.head(max(int(args.limit), 0)).copy()
    if pooled.empty:
        raise ValueError("Cannot repartition an empty dataset")
    pooled["effective_proposal_key"] = effective_proposal_keys(
        pooled["source_proposal_id"]
    )
    total = len(pooled)
    if total == sum(DEFAULT_SIZES.values()):
        target_sizes = dict(DEFAULT_SIZES)
    else:
        fractions = {"train": 0.8, "validation": 0.1, "test": 0.1}
        target_sizes = {split: int(round(total * fractions[split])) for split in SPLITS}
        target_sizes["train"] += total - sum(target_sizes.values())
    assignment = grouped_split_assignment(
        pooled["effective_proposal_key"],
        target_sizes=target_sizes,
        seed=args.seed,
    )
    pooled["split"] = pooled["effective_proposal_key"].map(assignment)
    if pooled["split"].isna().any():
        raise RuntimeError("Some effective proposal groups were not assigned")

    rng = np.random.default_rng(int(args.seed) + 1)
    audit: dict[str, object] = {
        "version": 1,
        "source_dataset_dir": str(args.source_dir),
        "output_dataset_dir": str(out),
        "seed": int(args.seed),
        "rows": int(total),
        "target_sizes": target_sizes,
        "splits": {},
    }
    key_sets = {}
    for split in SPLITS:
        selected = pooled.loc[pooled["split"] == split].copy()
        selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
        selected["object_id"] = np.arange(
            OBJECT_ID_STARTS[split],
            OBJECT_ID_STARTS[split] + len(selected),
            dtype=np.int64,
        )
        selected.to_parquet(out / f"{split}.parquet", index=False)
        keys = set(selected["effective_proposal_key"].astype(str))
        key_sets[split] = keys
        multiplicities = selected["effective_proposal_key"].value_counts().to_numpy()
        audit["splits"][split] = {
            "rows": int(len(selected)),
            "unique_effective_proposals": int(len(keys)),
            "duplicate_rows": int(len(selected) - len(keys)),
            "multiplicity_max": int(np.max(multiplicities)),
        }
    overlaps = {}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        count = len(key_sets[left].intersection(key_sets[right]))
        overlaps[f"{left}__{right}"] = int(count)
        if count:
            raise RuntimeError(f"Effective proposal leakage between {left} and {right}")
    audit["effective_proposal_overlap"] = overlaps
    write_json(out / "grouped_split_audit.json", audit)
    print(f"[grouped-split] wrote {total} existing rows to {out}", flush=True)


if __name__ == "__main__":
    main()
