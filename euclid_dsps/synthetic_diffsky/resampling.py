"""Weighted proposal resampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import SplitGenerationConfig


@dataclass(frozen=True)
class ResamplingResult:
    frame: pd.DataFrame
    ess: float
    duplicate_fraction: float
    pool_size: int
    weight_sum: float
    weight_sum_sq: float


def effective_sample_size(weights: np.ndarray) -> float:
    """Return ESS = (sum w)^2 / sum(w^2)."""
    w = np.asarray(weights, dtype=float)
    finite = np.isfinite(w) & (w > 0.0)
    if not finite.any():
        return 0.0
    wf = w[finite]
    return float((wf.sum() ** 2) / np.sum(wf**2))


def resample_weighted_proposals(
    proposals: pd.DataFrame,
    split: SplitGenerationConfig,
) -> ResamplingResult:
    """Draw an unweighted final catalog from weighted proposals."""
    if "galaxy_weight" not in proposals:
        raise ValueError("Proposal catalog must contain galaxy_weight")
    if int(split.n_final) <= 0:
        return ResamplingResult(
            frame=proposals.head(0).copy(),
            ess=0.0,
            duplicate_fraction=0.0,
            pool_size=int(len(proposals)),
            weight_sum=0.0,
            weight_sum_sq=0.0,
        )
    weights = pd.to_numeric(proposals["galaxy_weight"], errors="coerce").to_numpy(float)
    valid = np.isfinite(weights) & (weights > 0.0)
    if valid.sum() == 0:
        raise ValueError("No positive finite galaxy_weight values in proposals")
    pool = proposals.loc[valid].reset_index(drop=True)
    weights = weights[valid]
    p = weights / weights.sum()
    rng = np.random.default_rng(int(split.resample_seed))
    indices = rng.choice(len(pool), size=int(split.n_final), replace=True, p=p)
    sampled = pool.iloc[indices].copy().reset_index(drop=True)
    sampled.insert(0, "split", split.name)
    sampled.insert(
        0,
        "object_id",
        np.arange(
            int(split.object_id_start),
            int(split.object_id_start) + int(split.n_final),
            dtype=np.int64,
        ),
    )
    duplicate_fraction = 1.0 - (
        float(pd.Series(sampled["source_proposal_id"]).nunique()) / float(len(sampled))
    )
    return ResamplingResult(
        frame=sampled,
        ess=effective_sample_size(weights),
        duplicate_fraction=float(duplicate_fraction),
        pool_size=int(len(pool)),
        weight_sum=float(weights.sum()),
        weight_sum_sq=float(np.sum(weights**2)),
    )


def resampling_summary(result: ResamplingResult) -> dict[str, Any]:
    """Return manifest-friendly resampling diagnostics."""
    return {
        "pool_size": int(result.pool_size),
        "ess": float(result.ess),
        "weight_sum": float(result.weight_sum),
        "weight_sum_sq": float(result.weight_sum_sq),
        "duplicate_fraction": float(result.duplicate_fraction),
        "final_size": int(len(result.frame)),
    }
