"""Diagnostics for conditional full-flow mixture utilization and separation."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from .avi_experiments import expert_responsibilities, with_encoder
from .posterior import sample_posterior


def gate_diagnostics(probabilities: np.ndarray, threshold: float = 0.05) -> dict:
    """Summarize conditional gate utilization without using latent truth."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all():
        raise ValueError("finite [objects, experts] gate probabilities required")
    if np.any(p < 0) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("gate probabilities must be normalized")
    entropy = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    winner = np.argmax(p, axis=1)
    return {
        "objects": int(len(p)),
        "experts": int(p.shape[1]),
        "mean_probability": p.mean(axis=0).tolist(),
        "winning_fraction": (
            np.bincount(winner, minlength=p.shape[1]).astype(float) / len(p)
        ).tolist(),
        "mean_normalized_entropy": float(np.mean(entropy) / np.log(p.shape[1])),
        "median_effective_experts": float(np.median(np.exp(entropy))),
        "median_max_probability": float(np.median(np.max(p, axis=1))),
        "median_active_experts": float(np.median(np.sum(p >= threshold, axis=1))),
        "active_probability_threshold": float(threshold),
    }


def responsibility_diagnostics(responsibilities: np.ndarray) -> dict:
    """Summarize exact component responsibilities over joint mixture draws."""
    r = np.asarray(responsibilities, dtype=np.float64)
    if r.ndim != 3 or not np.isfinite(r).all():
        raise ValueError("finite [experts, draws, objects] responsibilities required")
    if np.any(r < 0) or not np.allclose(r.sum(axis=0), 1.0, atol=1e-6):
        raise ValueError("responsibilities must be normalized over experts")
    entropy = -np.sum(np.where(r > 0, r * np.log(r), 0.0), axis=0)
    return {
        "mean_responsibility": r.mean(axis=(1, 2)).tolist(),
        "median_max_responsibility": float(np.median(np.max(r, axis=0))),
        "median_effective_responsibilities": float(np.median(np.exp(entropy))),
    }


def pairwise_mean_separation(means: np.ndarray) -> list[dict]:
    """RMS separation of expert means in normalized latent coordinates."""
    value = np.asarray(means, dtype=np.float64)
    if value.ndim != 3 or value.shape[-1] < 6 or not np.isfinite(value).all():
        raise ValueError("finite [experts, objects, dimensions] means required")
    rows = []
    for first, second in itertools.combinations(range(value.shape[0]), 2):
        delta = value[first] - value[second]
        physical = np.sqrt(np.mean(delta[:, :5] ** 2, axis=1))
        nuisance = np.sqrt(np.mean(delta[:, 5:] ** 2, axis=1))
        rows.append(
            {
                "expert_a": first,
                "expert_b": second,
                "physical_5d_median_rms": float(np.median(physical)),
                "physical_5d_q90_rms": float(np.quantile(physical, 0.9)),
                "sfh_10d_median_rms": float(np.median(nuisance)),
                "sfh_10d_q90_rms": float(np.quantile(nuisance, 0.9)),
            }
        )
    return rows


def evaluate_expert_geometry(model, candidate, features, key, draws: int = 32):
    """Evaluate gates, exact responsibilities and expert mean separation."""
    if draws < 2:
        raise ValueError("at least two draws per expert required")
    probabilities = jax.nn.softmax(candidate.logits(features), axis=-1)
    keys = jax.random.split(key, candidate.n_components)
    samples = jnp.stack(
        [
            sample_posterior(with_encoder(model, expert), sample_key, features, draws).x
            for expert, sample_key in zip(candidate.experts, keys, strict=True)
        ],
        axis=0,
    )
    # Pool equal counts from each expert. This diagnoses overlap independently
    # of how often the learned gate currently chooses an expert.
    pooled = samples.reshape(candidate.n_components * draws, *samples.shape[2:])
    responsibilities = expert_responsibilities(model, candidate, features, pooled)
    means = jnp.mean(samples, axis=1)
    return probabilities, responsibilities, means
