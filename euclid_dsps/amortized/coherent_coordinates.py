"""Coordinates for a labelled coherent-parent density oracle, not a fitted prior.

Mass (already log10) and SFH contrasts have real-line support via asinh. Four
physical coordinates retain the declared open bounds via logit. No clipping,
row removal, or dequantization is permitted. Train truth sets affine scales only.
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
            raise ValueError(f"Outside declared OPEN support: {spec['names'][k]}")


def fit_coordinates(train, names, lower, upper):
    """Fit only on parent train; held-out rows must never extend these bounds."""
    if tuple(names) != tuple(SPLINE15D_PARAMETER_NAMES):
        raise ValueError("Expected canonical physical-5 + SFH-10 order")
    bounded = np.array([n in BOUNDED for n in names])
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
    return (raw - jnp.asarray(spec["center"])) / jnp.asarray(spec["scale"])


def to_theta(x, spec):
    raw = jnp.asarray(x, dtype=jnp.float64) * jnp.asarray(spec["scale"]) + jnp.asarray(
        spec["center"]
    )
    lo, hi = jnp.asarray(spec["lower"]), jnp.asarray(spec["upper"])
    return jnp.where(
        jnp.asarray(spec["bounded"]),
        lo + (hi - lo) * jax.nn.sigmoid(raw),
        jnp.asarray(spec["location"]) + jnp.asarray(spec["width"]) * jnp.sinh(raw),
    )


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
    return jnp.sum(
        jnp.log(jnp.asarray(spec["scale"]))
        + jnp.where(jnp.asarray(spec["bounded"]), bounded_ld, unbounded_ld),
        axis=-1,
    )
