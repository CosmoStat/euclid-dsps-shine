"""Invertible coordinates for coherent 15D densities.

Mass (already log10) and SFH contrasts have real-line support via asinh. Physical
coordinates use declared open bounds, with optional positive log coordinates.
No clipping, row removal, or dequantization is permitted. The caller determines
the fitting sample and records its provenance; old v1 specifications are unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.prior_learning.spline15d_schema import SPLINE15D_PARAMETER_NAMES

BOUNDED = ("z_obs", "log10_stellar_metallicity", "dust_av", "dust_delta")


def validate_theta(theta, spec):
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] != 15 or not np.isfinite(theta).all():
        raise ValueError("Expected finite (N,15) physical coordinates")
    for k, bounded in enumerate(spec["bounded"]):
        if bounded and np.any(
            (theta[:, k] <= spec["lower"][k]) | (theta[:, k] >= spec["upper"][k])
        ):
            raise ValueError(
                f"Outside declared OPEN support: {spec['names'][k]}; "
                f"range=[{theta[:, k].min():.12g}, {theta[:, k].max():.12g}], "
                f"required=({spec['lower'][k]}, {spec['upper'][k]})"
            )
    for k, positive in enumerate(spec.get("positive", [False] * theta.shape[1])):
        if positive and np.any(theta[:, k] <= 0):
            raise ValueError(
                f"Outside declared POSITIVE support: {spec['names'][k]}; "
                f"minimum={theta[:, k].min():.12g}. "
                "Exact zeros require a separate atom model, not clipping."
            )


def fit_coordinates(train, names, lower, upper, *, positive=()):
    """Fit scales on the supplied training/reference sample, never extend bounds."""
    if tuple(names) != tuple(SPLINE15D_PARAMETER_NAMES):
        raise ValueError("Expected canonical physical-5 + SFH-10 order")
    if not set(positive) <= set(names):
        raise ValueError("Unknown positive coordinate")
    positive_mask = np.array([n in positive for n in names])
    bounded = np.array([n in BOUNDED for n in names]) & ~positive_mask
    train = np.asarray(train, dtype=np.float64)
    spec = dict(
        version="coherent_coordinates_v1",
        names=list(names),
        bounded=bounded.tolist(),
        # Unbounded entries use finite placeholders for safe JAX branch evaluation.
        lower=np.where(bounded, lower, -1.0).tolist(),
        upper=np.where(bounded, upper, 1.0).tolist(),
        location=np.median(train, axis=0).tolist(),
        width=np.maximum(
            np.subtract(*np.percentile(train, [75, 25], axis=0)), 1e-3
        ).tolist(),
        center=[0.0] * 15,
        scale=[1.0] * 15,
        fitted_on="parent_train_truth_only",
        role="representation_oracle_only",
        clipping=False,
        dequantization=False,
    )
    if positive_mask.any():
        spec.update(version="coherent_coordinates_v2", positive=positive_mask.tolist())
    validate_theta(train, spec)
    raw = np.asarray(to_x(train, spec))
    spec["center"] = np.median(raw, axis=0).tolist()
    spec["scale"] = np.maximum(
        np.subtract(*np.percentile(raw, [75, 25], axis=0)) / 1.349, 0.05
    ).tolist()
    return spec


def to_x(theta, spec):
    t = jnp.asarray(theta, dtype=jnp.float64)
    bounded = jnp.asarray(spec["bounded"])
    lo, hi = jnp.asarray(spec["lower"]), jnp.asarray(spec["upper"])
    # Evaluate inactive logit branches at an interior constant, never at invalid t.
    bt = jnp.where(bounded, t, (lo + hi) / 2)
    logit = jnp.log(bt - lo) - jnp.log(hi - bt)
    raw = jnp.where(
        bounded,
        logit,
        jnp.arcsinh((t - jnp.asarray(spec["location"])) / jnp.asarray(spec["width"])),
    )
    positive = jnp.asarray(spec.get("positive", [False] * len(spec["names"])))
    # log: (0, infinity) -> R. No upper cap, epsilon floor or discarded rows.
    raw = jnp.where(positive, jnp.log(jnp.where(positive, t, 1.0)), raw)
    return (raw - jnp.asarray(spec["center"])) / jnp.asarray(spec["scale"])


def to_theta(x, spec):
    raw = jnp.asarray(x, dtype=jnp.float64) * jnp.asarray(spec["scale"]) + jnp.asarray(
        spec["center"]
    )
    lo, hi = jnp.asarray(spec["lower"]), jnp.asarray(spec["upper"])
    theta = jnp.where(
        jnp.asarray(spec["bounded"]),
        lo + (hi - lo) * jax.nn.sigmoid(raw),
        jnp.asarray(spec["location"]) + jnp.asarray(spec["width"]) * jnp.sinh(raw),
    )
    positive = jnp.asarray(spec.get("positive", [False] * len(spec["names"])))
    return jnp.where(positive, jnp.exp(jnp.where(positive, raw, 0.0)), theta)


def log_abs_det_dtheta_dx(x, spec):
    """log p_theta = log p_x - log|d theta/d x|, summed over all 15 dims."""
    raw = jnp.asarray(x, dtype=jnp.float64) * jnp.asarray(spec["scale"]) + jnp.asarray(
        spec["center"]
    )
    bounded_ld = (
        jnp.log(jnp.asarray(spec["upper"]) - jnp.asarray(spec["lower"]))
        - jax.nn.softplus(raw)
        - jax.nn.softplus(-raw)
    )
    unbounded_ld = (
        jnp.log(jnp.asarray(spec["width"])) + jnp.logaddexp(raw, -raw) - jnp.log(2.0)
    )
    positive = jnp.asarray(spec.get("positive", [False] * len(spec["names"])))
    # For t=exp(raw), log|dt/dx| = raw + log(scale).
    return jnp.sum(
        jnp.log(jnp.asarray(spec["scale"]))
        + jnp.where(
            positive,
            raw,
            jnp.where(jnp.asarray(spec["bounded"]), bounded_ld, unbounded_ld),
        ),
        axis=-1,
    )
