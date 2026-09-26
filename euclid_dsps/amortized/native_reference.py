"""Normalized continuous 15D mixtures over independent native proposal anchors.

g_j(x) = sum_l A_lj Normal(x; anchor_l, h^2 I), sum_l A_lj = 1.
The gates defining A depend on the five physical coordinates. Joint anchors
retain reference physical/SFH dependence; every coordinate is still stochastic.
This is a reference-family assumption, not an estimate of the target SFH law.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.special import logsumexp, softmax

from .forward_population import simplex


def make_basis(anchors, components, bandwidth, gate_width, broad_fraction, seed):
    x = np.asarray(anchors, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 15 or not np.isfinite(x).all():
        raise ValueError("Finite joint 15D anchors required")
    if not 1 < components <= len(x) or bandwidth <= 0 or gate_width <= 0:
        raise ValueError("Invalid component count or kernel widths")
    if not 0 < broad_fraction < 1:
        raise ValueError("Require a nonzero common broad anchor component")
    centers, _ = kmeans2(x[:, :5], components, minit="++", seed=seed, iter=30)
    distance = np.sum((x[:, None, :5] - centers[None]) ** 2, axis=-1)
    gates = (1 - broad_fraction) * softmax(-distance / (2 * gate_width**2), axis=1)
    gates += broad_fraction / components
    conditional = gates / gates.sum(axis=0, keepdims=True)
    return dict(
        anchors=x,
        conditional=conditional,
        centers=centers,
        bandwidth=np.asarray(bandwidth),
    )


def sample_basis(basis, labels, seed):
    """Draw theta in normalized coordinates, never from an inference network."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    a = basis["conditional"]
    if labels.ndim != 1 or np.any(labels < 0) or np.any(labels >= a.shape[1]):
        raise ValueError("Invalid component labels")
    anchors = np.empty(len(labels), dtype=int)
    for j in np.unique(labels):
        idx = np.flatnonzero(labels == j)
        anchors[idx] = rng.choice(len(a), len(idx), p=a[:, j])
    return basis["anchors"][anchors] + float(basis["bandwidth"]) * rng.normal(
        size=(len(labels), 15)
    )


def log_prob(basis, x, weights, chunk=256):
    """Exact normalized latent density; used for contracts, not per-object IS."""
    w = basis["conditional"] @ simplex(weights)
    h = float(basis["bandwidth"])
    result = []
    for rows in np.array_split(x, max(1, int(np.ceil(len(x) / chunk)))):
        delta = (rows[:, None, :] - basis["anchors"][None]) / h
        terms = -0.5 * np.sum(delta**2, axis=-1) - 15 * np.log(h * np.sqrt(2 * np.pi))
        result.extend(logsumexp(terms + np.log(w), axis=1))
    return np.asarray(result)


def supervised_weights(labels, parent, sampling):
    """Joint-label IS: u_j/r_j for selected simulation pairs (x, theta, j).

    Normalizing over selected pairs yields the learned-parent selected joint.
    No 1/beta(theta) belongs in an individual posterior. Targets remain simulator
    truth; weights never depend on q. r is the PARENT bank sampling probability.
    """
    u, r = simplex(parent), simplex(sampling)
    if len(u) != len(r) or np.any(r <= 0):
        raise ValueError("Reference must cover every parent component")
    weights = (u / r)[np.asarray(labels, dtype=int)]
    if not len(weights) or weights.sum() <= 0:
        raise ValueError("Empty selected training support")
    return weights / weights.mean()


def choose_penalty(log_likelihoods, strengths):
    """Largest penalty within one paired standard error of best held-out fit."""
    values = np.asarray(log_likelihoods, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Finite held-out per-object likelihoods required")
    best = int(np.argmax(values.mean(axis=1)))
    delta = values[best] - values
    se = delta.std(axis=1, ddof=1) / np.sqrt(values.shape[1])
    admissible = delta.mean(axis=1) <= se + 1e-12
    selected = max(np.flatnonzero(admissible), key=lambda i: strengths[i])
    return int(selected), admissible, delta.mean(axis=1), se
