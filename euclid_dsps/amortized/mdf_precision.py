"""Small, replayable MDF arithmetic probes; no catalogue labels or fitting."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.model import lognormal_mdf_lgmet_weights_jax


def triweight_reference(center, scatter, edges):
    """DSPS's compact polynomial, not a replacement Gaussian CDF.

    Evaluate values and analytic derivatives in NumPy extended precision.
    Edges are exported from DSPS separately so grid construction is auditable.
    """
    edges = np.asarray(edges, dtype=np.longdouble)
    z = (np.longdouble(center) - edges) / np.longdouble(scatter)
    t = np.clip(z, -3, 3)
    cdf = (
        -5 * t**7 / 69984
        + 7 * t**5 / 2592
        - 35 * t**3 / 864
        + 35 * t / 96
        + np.longdouble("0.5")
    )
    pdf = np.where(
        abs(z) < 3, 35 / np.longdouble(96) * (1 - (t / 3) ** 2) ** 3 / scatter, 0
    )
    raw = np.maximum(cdf[:-1] - cdf[1:], 0)
    derivative = pdf[:-1] - pdf[1:]
    derivative = np.where(raw > 0, derivative, 0)
    total = raw.sum()
    if total == 0:
        weights = np.zeros(len(edges) - 1, dtype=np.longdouble)
        if center < edges[0]:
            weights[0] = 1
        elif center > edges[-1]:
            weights[-1] = 1
        return weights.astype(float), np.zeros_like(weights, dtype=float)
    return (raw / total).astype(float), (
        (derivative * total - raw * derivative.sum()) / total**2
    ).astype(float)


def probe(grid, centers, scatter, budget):
    """Hold the stored grid fixed; isolate physical log10(Z), not latent x."""
    from dsps.constants import LGMET_HI, LGMET_LO
    from dsps.utils import _get_bin_edges

    grid = np.asarray(grid)
    rows, cases = [], []
    for label, dtype in (("float32_legacy", jnp.float32), ("float64_v1", jnp.float64)):
        grid_jax = jnp.asarray(grid, dtype=dtype)
        edges = np.asarray(_get_bin_edges(grid_jax, LGMET_LO, LGMET_HI))
        fn = jax.jit(
            lambda m, grid_jax=grid_jax, dtype=dtype: lognormal_mdf_lgmet_weights_jax(
                grid_jax, m, scatter, numerical_dtype=dtype
            )
        )
        grad = jax.jit(jax.jacfwd(fn))
        for i, center in enumerate(centers):
            budget.charge(0)
            value = np.asarray(fn(jnp.asarray(center, dtype=dtype)))
            ad = np.asarray(grad(jnp.asarray(center, dtype=dtype)))
            actual_scatter = float(np.asarray(scatter, dtype=np.dtype(dtype)))
            ref, ref_ad = triweight_reference(
                float(np.asarray(center, dtype=np.dtype(dtype))), actual_scatter, edges
            )
            cases.append(
                dict(
                    variant=label,
                    point_index=i,
                    center=float(center),
                    output_dtype=str(value.dtype),
                    edges=edges.tolist(),
                    weights=value.tolist(),
                    autodiff=ad.tolist(),
                    reference_weights=ref.tolist(),
                    reference_derivative=ref_ad.tolist(),
                    max_abs_weight_error=float(np.max(abs(value - ref))),
                    max_abs_derivative_error=float(np.max(abs(ad - ref_ad))),
                    zero_weights=int(np.sum(value == 0)),
                    weight_sum=float(value.sum()),
                    derivative_sum=float(ad.sum()),
                    effective_scatter=actual_scatter,
                )
            )
            for h in (0.02 / 2**k for k in range(20)):
                plus = np.asarray(fn(jnp.asarray(center + h, dtype=dtype)), dtype=float)
                minus = np.asarray(
                    fn(jnp.asarray(center - h, dtype=dtype)), dtype=float
                )
                for m in range(len(grid)):
                    rows.append(
                        dict(
                            variant=label,
                            point_index=i,
                            metallicity_bin=m,
                            step=h,
                            ad=float(ad[m]),
                            fd=float((plus[m] - minus[m]) / (2 * h)),
                            reference_ad=float(ref_ad[m]),
                            plus=float(plus[m]),
                            minus=float(minus[m]),
                        )
                    )
    candidate_cases = [c for c in cases if c["variant"] == "float64_v1"]
    reference_pass = all(
        c["max_abs_weight_error"] <= 1e-12 and c["max_abs_derivative_error"] <= 1e-10
        for c in candidate_cases
    ) and bool(candidate_cases)
    return dict(
        candidate_reference_checks="PASS" if reference_pass else "NOT_PASSED",
        reference_absolute_tolerances=dict(weights=1e-12, derivative=1e-10),
        cases=cases,
        grid=grid.tolist(),
        scatter=float(scatter),
        coordinate="physical log10(Z); not latent x",
        reference="DSPS triweight polynomial, independent NumPy analytic derivative; same exported bin edges",
        numpy_reference_mantissa_bits=int(np.finfo(np.longdouble).nmant),
        interpretation="weight probes alone do not qualify the full decoder",
        scientific_promotion=False,
        truth_used=False,
    ), rows
