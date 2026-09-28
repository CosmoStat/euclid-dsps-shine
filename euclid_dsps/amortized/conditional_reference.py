"""Nested, normalized SFH refinements of a fixed joint 15D reference.

For old g_j = sum_l A_lj K_l, define g_js = sum_l A_lj h_ljs K_l / Z_js,
where sum_s h_ljs = 1 and Z_js = sum_l A_lj h_ljs. Thus
g_j = sum_s Z_js g_js EXACTLY. Kernels, support and all 15 coordinates survive.
Only independent reference anchors determine the gates, never target truth.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp

from .forward_population import simplex
from .native_reference import kernel_bandwidths


def split_sfh_basis(basis: dict, probability_floor: float = 0.05) -> tuple:
    x = np.asarray(basis["anchors"], float)
    a = np.asarray(basis["conditional"], float)
    if (
        x.ndim != 2
        or x.shape[1] != 15
        or a.ndim != 2
        or len(a) != len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(a).all()
        or np.any(a < 0)
        or not np.allclose(a.sum(axis=0), 1, atol=1e-12)
        or not 0 < probability_floor < 0.5
    ):
        raise ValueError(
            "Normalized joint anchors and a gate floor in (0, .5) required"
        )
    kernel_bandwidths(basis)
    reference = a.mean(axis=1)
    center = reference @ x[:, 5:]
    sfh = x[:, 5:] - center
    eigenvalues, eigenvectors = np.linalg.eigh((sfh.T * reference) @ sfh)
    if eigenvalues[-1] <= 1e-12:
        raise ValueError("No SFH reference variability to split")
    axis = eigenvectors[:, -1]
    axis *= 1 if axis[np.argmax(abs(axis))] >= 0 else -1
    score = sfh @ axis / np.sqrt(eigenvalues[-1])
    location = score @ a
    scale = np.sqrt(np.maximum((score**2) @ a - location**2, 1e-4))
    high = probability_floor + (1 - 2 * probability_floor) * expit(
        (score[:, None] - location) / scale
    )
    gates = np.stack([1 - high, high], axis=-1)
    unnormalized = a[:, :, None] * gates
    z = unnormalized.sum(axis=0)
    conditional = (unnormalized / z).reshape(len(a), -1)
    recovered = (conditional.reshape(len(a), -1, 2) * z).sum(axis=2)
    if not np.allclose(recovered, a, rtol=1e-12, atol=1e-15):
        raise ValueError("Nested-family normalization failure")
    result = dict(basis, conditional=conditional)
    result["centers"] = np.repeat(basis["centers"], 2, axis=0)
    info = dict(
        z=z.tolist(),
        probability_floor=probability_floor,
        sfh_axis=axis.tolist(),
        sfh_center=center.tolist(),
        variance_fraction=float(eigenvalues[-1] / eigenvalues.sum()),
        component_location=location.tolist(),
        component_scale=scale.tolist(),
        maximum_nesting_error=float(abs(recovered - a).max()),
        target_truth_used=False,
        kernels_changed=False,
        dimensions=15,
    )
    return result, gates, info


def normalized_split(z):
    """Remove summation roundoff only; reject genuinely unnormalized masses."""
    z = np.asarray(z, float)
    if (
        z.ndim != 2
        or z.shape[1] != 2
        or not np.isfinite(z).all()
        or np.any(z <= 0)
        or not np.allclose(z.sum(axis=1), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Two normalized positive subcomponent masses required")
    first = z[:, 0] / z.sum(axis=1)
    return np.column_stack([first, 1 - first])


def lift_parent(weights, z):
    z = normalized_split(z)
    u = simplex(weights)
    if z.shape != (len(u), 2):
        raise ValueError("Two normalized positive subcomponent masses required")
    return (u[:, None] * z).ravel()


def tied_selected_ratios(log_classifier, frequencies, alpha, z):
    """Selected g_j likelihood: mix subcomponents with Z_js alpha_js/alpha_j.

    Summing classifier probabilities alone is generally WRONG when the saved
    calibration frequencies differ from the theoretical sampling proportions.
    Returned logC need not sum to one: common per-object factors cancel in fits.
    """
    z = normalized_split(z)
    c, alpha = simplex(frequencies), np.asarray(alpha, float)
    logc = np.asarray(log_classifier, float)
    if (
        z.ndim != 2
        or z.shape[1] != 2
        or len(alpha) != z.size
        or not np.isfinite(alpha).all()
        or len(c) != z.size
        or np.any(alpha <= 0)
        or np.any(alpha > 1)
        or np.any(z <= 0)
        or not np.allclose(z.sum(axis=1), 1)
        or np.any(c <= 0)
        or not np.isfinite(logc).all()
        or logc.ndim != 2
        or logc.shape[1] != z.size
    ):
        raise ValueError("Inconsistent selected subcomponent ratios")
    selected_mass = z * alpha.reshape(z.shape)
    efficiencies = alpha.reshape(z.shape)
    low, high = efficiencies.min(axis=1), efficiencies.max(axis=1)
    high_weight = z[np.arange(len(z)), efficiencies.argmax(axis=1)]
    # Convex interpolation returns exactly 1 when both sub-efficiencies are 1.
    # A raw sum can produce 1+epsilon and fail strict probability validation.
    grouped_alpha = low + high_weight * (high - low)
    mix = selected_mass / selected_mass.sum(axis=1, keepdims=True)
    logd = logsumexp(
        (logc - np.log(c)).reshape(len(logc), *z.shape) + np.log(mix), axis=2
    )
    grouped_c = simplex(grouped_alpha)  # original parent reference is uniform in j
    return logd + np.log(grouped_c), grouped_c, grouped_alpha


def reserved_row_weights(labels, selected, role, parent, evaluation_role=4):
    """Stratified parent importance weights, including REJECTED row counts."""
    u = simplex(parent)
    labels, selected, role = np.asarray(labels), np.asarray(selected), np.asarray(role)
    take = role == evaluation_role
    counts = np.bincount(labels[take], minlength=len(u))
    if np.any(counts == 0):
        raise ValueError("Missing parent components in evaluation role")
    mask = take & selected
    weights = u[labels[mask]] / counts[labels[mask]]
    return mask, simplex(weights)
