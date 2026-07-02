"""Metallicity truth and MDF diagnostics for DSPS closure catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from euclid_dsps.model import lognormal_mdf_lgmet_weights_jax


@dataclass(frozen=True)
class MetallicityTransformResult:
    log10_z_over_zsun: np.ndarray
    lgmet_abs_used: np.ndarray
    clipped_mask: np.ndarray
    clip_low_mask: np.ndarray
    clip_high_mask: np.ndarray
    clip_low_count: int
    clip_high_count: int


def absolute_lgmet_to_logzsol(
    lgmet_abs_median: np.ndarray,
    *,
    z_sun: float,
    ssp_lgmet: np.ndarray,
    policy: str,
) -> MetallicityTransformResult:
    """Convert absolute log10(Z) medians to log10(Z/Zsun) with explicit clipping."""
    values = np.asarray(lgmet_abs_median, dtype=float)
    grid = np.asarray(ssp_lgmet, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("SSP metallicity grid must be a non-empty 1D array")
    if not np.isfinite(z_sun) or z_sun <= 0.0:
        raise ValueError("z_sun must be positive")
    logzsun = np.log10(float(z_sun))
    logzsol = values - logzsun
    lo = float(np.nanmin(grid))
    hi = float(np.nanmax(grid))
    below = logzsol < lo
    above = logzsol > hi
    clipped = below | above
    policy = str(policy)
    if clipped.any() and policy == "fail":
        raise ValueError(
            "FENIKS metallicity medians exceed the SSP grid. Configure "
            "synthetic_diffsky.metallicity_grid_policy='clip_with_warning' "
            "only after confirming the clipped fraction is scientifically acceptable."
        )
    if clipped.any() and policy not in {"clip", "clip_with_warning"}:
        raise ValueError(
            "synthetic_diffsky.metallicity_grid_policy must be 'fail', "
            "'clip', or 'clip_with_warning'"
        )
    used_logzsol = np.clip(logzsol, lo, hi) if clipped.any() else logzsol.copy()
    used_abs = used_logzsol + logzsun
    return MetallicityTransformResult(
        log10_z_over_zsun=used_logzsol,
        lgmet_abs_used=used_abs,
        clipped_mask=clipped,
        clip_low_mask=below,
        clip_high_mask=above,
        clip_low_count=int(np.count_nonzero(below)),
        clip_high_count=int(np.count_nonzero(above)),
    )


def lognormal_mdf_weights(
    ssp_lgmet: np.ndarray,
    lgmet_abs_median: float,
    scatter_dex: float,
) -> np.ndarray:
    """Return finite, normalized DSPS MDF weights for tests and diagnostics."""
    weights = np.asarray(
        lognormal_mdf_lgmet_weights_jax(
            jnp.asarray(ssp_lgmet, dtype=jnp.float32),
            jnp.asarray(float(lgmet_abs_median), dtype=jnp.float32),
            jnp.asarray(float(scatter_dex), dtype=jnp.float32),
        ),
        dtype=float,
    )
    if not np.isfinite(weights).all():
        raise ValueError("MDF weights contain non-finite values")
    if np.any(weights < 0.0):
        raise ValueError("MDF weights must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("MDF weights sum to zero")
    return weights / total


def metallicity_summary_payload(result: MetallicityTransformResult) -> dict[str, Any]:
    """Return manifest-friendly clipping counts."""
    n = int(result.clipped_mask.size)
    n_clip = int(np.count_nonzero(result.clipped_mask))
    return {
        "n": n,
        "clip_low_count": int(result.clip_low_count),
        "clip_high_count": int(result.clip_high_count),
        "clipped_count": n_clip,
        "clipped_fraction": float(n_clip / n) if n else 0.0,
    }
