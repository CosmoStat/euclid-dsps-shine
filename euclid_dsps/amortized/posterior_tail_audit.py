"""Full-support diagnostics, never clipping or replacing a posterior by medians."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def tail_table(draws, truth, names, *, far_iqr=100.0):
    draws, truth = np.asarray(draws), np.asarray(truth)
    if (
        draws.ndim != 3
        or truth.shape != (draws.shape[0], draws.shape[2])
        or draws.shape[2] != len(names)
        or not np.isfinite(draws).all()
        or not np.isfinite(truth).all()
    ):
        raise ValueError(
            "Finite joint draws [object, draw, dimension] and truths required"
        )
    center = np.median(truth, axis=0)
    width = np.maximum(np.subtract(*np.quantile(truth, [0.75, 0.25], axis=0)), 1e-8)
    rows = []
    for j, name in enumerate(names):
        values = draws[:, :, j]
        distance = abs(values - center[j]) / width[j]
        rows.append(
            dict(
                parameter=name,
                truth_iqr=float(width[j]),
                minimum=float(values.min()),
                maximum=float(values.max()),
                q0001=float(np.quantile(values, 0.0001)),
                q9999=float(np.quantile(values, 0.9999)),
                maximum_distance_iqr=float(distance.max()),
                far_draw_fraction=float(np.mean(distance > far_iqr)),
                far_object_fraction=float(np.mean(np.any(distance > far_iqr, axis=1))),
                w1_over_truth_iqr=float(
                    wasserstein_distance(values.ravel(), truth[:, j]) / width[j]
                ),
            )
        )
    return pd.DataFrame(rows)


def extreme_positions(draws, truth, count):
    """Identify joint draws for replay; the returned medians are only a scale."""
    width = np.maximum(np.subtract(*np.quantile(truth, [0.75, 0.25], axis=0)), 1e-8)
    score = np.max(abs(draws - np.median(truth, axis=0)) / width, axis=2)
    order = np.argsort(score.ravel())[-min(count, score.size) :][::-1]
    return np.column_stack(np.unravel_index(order, score.shape))


def replay_coordinates(replayed_x, saved_theta, spec, *, tolerance, saved_x=None):
    """Compare what the artifact actually retained, without clipping any draw.

    A bounded sigmoid can discard latent information in float64 theta. For old
    theta-only artifacts, an otherwise failed coordinate must reproduce theta
    within eight representable neighbours AND lie inside their monotone inverse
    interval. This is a representation check, not proof of exact latent replay.
    Unbounded coordinates and directly saved latent draws get no exemption.
    """
    from .coherent_coordinates import to_theta, to_x

    x, theta = np.asarray(replayed_x), np.asarray(saved_theta)
    if x.shape != theta.shape or x.ndim != 2 or x.shape[1] != len(spec["names"]):
        raise ValueError("Expected matching (N,D) replay and saved coordinates")
    if theta.dtype != np.float64 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Float64 physical artifacts and positive tolerance required")
    if saved_x is not None and np.asarray(saved_x).shape != x.shape:
        raise ValueError("Stored latent shape does not match physical draws")
    decoded = np.asarray(to_x(theta, spec))
    replay_theta = np.asarray(to_theta(x, spec))
    roundtrip = np.asarray(to_x(replay_theta, spec))
    lower, upper = theta.copy(), theta.copy()
    for _ in range(8):
        lower = np.nextafter(lower, -np.inf)
        upper = np.nextafter(upper, np.inf)
    bounded = np.asarray(spec["bounded"], dtype=bool)
    # Bounds may be reached by the ULP envelope, never by the actual saved draw.
    lower[:, bounded] = np.maximum(
        lower[:, bounded], np.asarray(spec["lower"])[bounded]
    )
    upper[:, bounded] = np.minimum(
        upper[:, bounded], np.asarray(spec["upper"])[bounded]
    )
    inverse_lower = np.asarray(to_x(lower, spec))
    inverse_upper = np.asarray(to_x(upper, spec))
    finite = (
        np.isfinite(x)
        & np.isfinite(theta)
        & np.isfinite(decoded)
        & np.isfinite(replay_theta)
        & np.isfinite(roundtrip)
    )
    raw_error = abs(x - decoded)
    encoded_agreement = raw_error <= tolerance
    compatible = (
        bounded[None, :]
        & (replay_theta >= lower)
        & (replay_theta <= upper)
        & (x >= inverse_lower)
        & (x <= inverse_upper)
    )
    passed = finite & (encoded_agreement | compatible)
    latent_error = None
    if saved_x is not None:
        latent_error = abs(x - np.asarray(saved_x))
        passed &= np.isfinite(saved_x) & (latent_error <= tolerance)

    def number(value):
        return float(value) if np.isfinite(value) else None

    rows = []
    for i, j in np.ndindex(x.shape):
        rows.append(
            dict(
                extreme_position=i,
                parameter=spec["names"][j],
                bounded=bool(bounded[j]),
                saved_theta=number(theta[i, j]),
                replay_theta=number(replay_theta[i, j]),
                replay_x=number(x[i, j]),
                decoded_saved_x=number(decoded[i, j]),
                replay_latent_error=number(raw_error[i, j]),
                replay_own_roundtrip_error=number(abs(x[i, j] - roundtrip[i, j])),
                physical_ulp_envelope_pass=bool(compatible[i, j]),
                stored_latent_error=(
                    None if latent_error is None else number(latent_error[i, j])
                ),
                passed=bool(passed[i, j]),
            )
        )
    return pd.DataFrame(rows), dict(
        passed=bool(passed.all()),
        maximum_replay_latent_error=number(np.max(raw_error)),
        maximum_stored_latent_error=(
            None if latent_error is None else number(np.max(latent_error))
        ),
        stored_latent_available=saved_x is not None,
        bounded_quantization_compatible_coordinates=int(
            np.sum(finite & ~encoded_agreement & compatible)
        ),
        maximum_physical_rounding_ulps=8,
        failed_coordinates=int(np.sum(~passed)),
        nonfinite_coordinates=int(np.sum(~finite)),
        latent_tolerance=tolerance,
        contract="float64_theta_replay_v2",
    )
