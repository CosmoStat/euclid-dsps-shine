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
