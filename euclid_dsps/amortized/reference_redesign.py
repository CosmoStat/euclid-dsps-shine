"""Reference geometry and paired-replay helpers; no target truths or q inputs."""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from scipy.special import logsumexp, softmax

from .native_reference import kernel_bandwidths


def local_basis(anchors: np.ndarray, components: int, cfg: dict, seed: int) -> dict:
    """Local overlapping anchor groups plus ONE independent broad component.

    All components remain normalized 15D Gaussian mixtures. Physical kernel
    widths follow local reference geometry; SFH noise stays stochastic at its
    declared reference width. No forced broad floor is added to local gates.
    """
    x = np.asarray(anchors, dtype=float)
    if x.ndim != 2 or x.shape[1] != 15 or not np.isfinite(x).all():
        raise ValueError("Finite 15D anchors required")
    if not 3 <= components <= len(x):
        raise ValueError("Invalid local component count")
    if not 2 <= cfg["gate_neighbors"] < components or not 1 < cfg[
        "kernel_neighbors"
    ] < len(x):
        raise ValueError("Invalid neighbor counts")
    if (
        not all(np.isfinite(v) for v in cfg.values())
        or not 0 < cfg["minimum_bandwidth"] <= cfg["maximum_bandwidth"]
        or cfg["bandwidth_factor"] <= 0
        or cfg["sfh_bandwidth"] <= 0
    ):
        raise ValueError("Invalid kernel widths")
    centers, _ = kmeans2(x[:, :5], components - 1, minit="++", seed=seed, iter=30)
    distance, neighbors = cKDTree(centers).query(x[:, :5], k=cfg["gate_neighbors"])
    # Relative distances avoid remote centers flattening local memberships.
    scale = np.maximum(distance[:, -1], 1e-6)
    gates = np.zeros((len(x), components - 1))
    gates[np.arange(len(x))[:, None], neighbors] = softmax(
        -2 * (distance / scale[:, None]) ** 2, axis=1
    )
    if np.any(gates.sum(0) == 0):
        raise ValueError("Empty local component")
    a = np.c_[gates / gates.sum(0), np.full(len(x), 1 / len(x))]
    _, nearest = cKDTree(x[:, :5]).query(x[:, :5], k=cfg["kernel_neighbors"] + 1)
    local_rms = np.sqrt(np.mean((x[nearest[:, 1:], :5] - x[:, None, :5]) ** 2, axis=1))
    h = np.full_like(x, cfg["sfh_bandwidth"])
    h[:, :5] = np.clip(
        cfg["bandwidth_factor"] * local_rms,
        cfg["minimum_bandwidth"],
        cfg["maximum_bandwidth"],
    )
    return dict(
        anchors=x,
        conditional=a,
        centers=centers,
        bandwidth=h,
        broad_component=np.asarray(components - 1),
    )


def mass_moment(basis: dict, spec: dict, weights: np.ndarray) -> dict:
    """Analytic log10 E[Mstar] for affine-logmass Gaussian kernels.

    Asinh(logmass) plus any Gaussian variance instead has an infinite ideal
    linear-mass moment. Return null, never non-JSON infinity, for that control.
    """
    k = spec["names"].index("log10_stellar_mass")
    if not spec.get("affine", [False] * 15)[k]:
        return dict(
            finite_linear_mass_moment=False,
            log10_mean_mass=None,
            reason="Gaussian in asinh(logmass): divergent ideal linear-mass moment",
        )
    u = np.asarray(weights, dtype=float)
    if (
        u.shape != (basis["conditional"].shape[1],)
        or not np.isfinite(u).all()
        or np.any(u < 0)
        or not np.isclose(u.sum(), 1)
    ):
        raise ValueError("Normalized parent weights required")
    w = basis["conditional"] @ u
    mu = spec["location"][k] + spec["width"][k] * (
        spec["center"][k] + spec["scale"][k] * basis["anchors"][:, k]
    )
    sd = spec["width"][k] * spec["scale"][k] * kernel_bandwidths(basis)[:, k]
    positive = w > 0
    log_mean = logsumexp(
        np.log(w[positive])
        + np.log(10) * mu[positive]
        + 0.5 * (np.log(10) * sd[positive]) ** 2
    )
    return dict(
        finite_linear_mass_moment=bool(np.isfinite(log_mean)),
        log10_mean_mass=float(log_mean / np.log(10)),
        reason="Analytic Gaussian log-mass moment",
    )


def replay_normals(
    block_rows: int, row_indices: np.ndarray, bands: int, seed: int
) -> np.ndarray:
    """Exact original observe() RNG stream: one length-N draw for each band."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.normal(size=block_rows)[row_indices] for _ in range(bands)], axis=1
    )


def apply_saved_noise(
    flux: np.ndarray, bands: list[dict], noise: dict, normals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    from euclid_dsps.photometric_uncertainty import flux_error_from_model

    errors = np.stack(
        [
            flux_error_from_model(
                flux[:, j], band.get("error_model") or noise, band_name=band["name"]
            )
            for j, band in enumerate(bands)
        ],
        axis=1,
    )
    if (
        flux.shape != normals.shape
        or not np.isfinite(errors).all()
        or np.any(errors <= 0)
    ):
        raise ValueError("Invalid paired noise inputs")
    return flux + normals * errors, errors


def qualification(metrics: dict, baseline: dict, contracts: dict) -> tuple[dict, dict]:
    """Development-validation engineering gates, not proof of blind recovery."""
    cdf_limit = max(
        contracts["maximum_cdf_error"],
        contracts["sampling_multiplier"] * baseline["cdf_max"],
    )
    sw_limit = max(
        contracts["maximum_physical_sw"],
        contracts["sampling_multiplier"] * baseline["physical_sw"],
    )
    return dict(
        finite_mass=bool(metrics["finite_linear_mass_moment"]),
        validation_cdf=bool(metrics["validation_cdf_max"] <= cdf_limit),
        validation_physical_sw=bool(metrics["physical_sw"] <= sw_limit),
        validation_physical_marginals=bool(
            metrics["physical_marginal_max"]
            <= contracts["maximum_physical_marginal_w1"]
        ),
    ), dict(cdf_limit=cdf_limit, physical_sw_limit=sw_limit)
