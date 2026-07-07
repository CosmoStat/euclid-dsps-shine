"""Shared deterministic row splitting for prior-learning datasets."""

from __future__ import annotations

import numpy as np


def train_validation_split(
    n_rows: int,
    *,
    validation_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Return shuffled train/validation row indices with at least one train row."""
    order = np.arange(int(n_rows), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(order)
    validation_fraction = min(max(float(validation_fraction), 0.0), 0.9)
    if validation_fraction <= 0.0 or n_rows < 2:
        return {"train": order, "validation": np.asarray([], dtype=np.int64)}
    n_val = int(round(validation_fraction * n_rows))
    n_val = min(max(n_val, 1), n_rows - 1)
    return {"train": order[n_val:], "validation": order[:n_val]}
