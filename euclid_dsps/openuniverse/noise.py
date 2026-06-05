"""Noise models for OpenUniverse truth photon fluxes."""

from __future__ import annotations

import numpy as np


def add_fractional_snr_noise(
    flux_truth: np.ndarray,
    snr: float,
    seed: int,
    min_sigma_fraction: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise with an approximate per-object fractional SNR."""
    snr = float(snr)
    if not np.isfinite(snr) or snr <= 0.0:
        raise ValueError("snr must be positive and finite")
    flux = np.asarray(flux_truth, dtype=float)
    scale = _finite_positive_reference_scale(np.abs(flux))
    sigma = np.maximum(np.abs(flux) / snr, float(min_sigma_fraction) * scale)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, scale / snr)
    rng = np.random.default_rng(int(seed))
    noisy = flux + rng.normal(loc=0.0, scale=sigma, size=flux.shape)
    return noisy.astype(np.float32), sigma.astype(np.float32)


def add_band_snr_noise(
    flux_truth: np.ndarray,
    band_snr: dict[str, float],
    band_names: tuple[str, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise using one target SNR per band."""
    flux = np.asarray(flux_truth, dtype=float)
    if flux.ndim < 1:
        raise ValueError("flux_truth must have at least one dimension")
    if flux.shape[-1] != len(band_names):
        raise ValueError(
            f"flux_truth last dimension {flux.shape[-1]} does not match "
            f"{len(band_names)} band_names"
        )
    missing = [band for band in band_names if band not in band_snr]
    if missing:
        raise ValueError("Missing SNR entries for bands: " + ", ".join(missing))
    sigma = np.empty_like(flux, dtype=float)
    for band_index, band in enumerate(band_names):
        snr = float(band_snr[band])
        if not np.isfinite(snr) or snr <= 0.0:
            raise ValueError(f"band_snr[{band!r}] must be positive and finite")
        column = np.abs(flux[..., band_index])
        scale = _finite_positive_reference_scale(column)
        sigma[..., band_index] = np.maximum(column / snr, 1.0e-4 * scale)
    rng = np.random.default_rng(int(seed))
    noisy = flux + rng.normal(loc=0.0, scale=sigma, size=flux.shape)
    return noisy.astype(np.float32), sigma.astype(np.float32)


def add_depth_like_noise(*args, **kwargs):
    """Placeholder for survey-depth and exposure-time based OpenUniverse noise."""
    raise NotImplementedError(
        "OpenUniverse depth-like noise is not implemented yet. Use "
        "`fractional_snr` or `band_snr` for the current validation subset."
    )


def _finite_positive_reference_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return 1.0
    scale = float(np.nanmedian(finite))
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0
