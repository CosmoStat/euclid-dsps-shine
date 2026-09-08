"""Single-point redshift branch/precision experiment; never a training target."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.model import diagnostic_spline_redshift_branches
from euclid_dsps.parameter_vectors import theta_vector_to_model_param_dict

from .latent import x_to_theta
from .posterior_target import safe_decoder_inputs
from .redshift_decomposition import floating_trace_dtypes
from .target_resolution import STEPS
from .target_resolution import analyze as analyze_stencils


def analyze(snapshot):
    branches, rows = [], []
    for branch in snapshot["branches"]:
        report, table = analyze_stencils(branch["snapshot"])
        branches.append(
            dict(
                name=branch["name"],
                checks=report["checks"],
                all_checks_passed=report["unresolved_checks_resolved"],
                floating_dtypes=branch["floating_dtypes"],
                max_center_delta_sigma=branch["max_center_delta_sigma"],
            )
        )
        rows.extend(dict(r, branch=branch["name"]) for r in table)
    complete = len(branches) == snapshot["expected_branches"]
    derivatives = {
        b["name"]: np.array(b["snapshot"]["cases"][0]["flux_jvp"])
        / np.array(b["snapshot"]["cases"][0]["sigma"])
        for b in snapshot["branches"]
    }
    chain = {}
    for label in ("mixed", "zpath64"):
        names = [label + "_" + k for k in ("stellar", "igm", "projection", "full")]
        if all(n in derivatives for n in names):
            chain[label] = (
                sum(derivatives[n] for n in names[:3]) - derivatives[names[3]]
            ).tolist()
    return dict(
        status="REDSHIFT_PRECISION_COMPLETE"
        if complete
        else "REDSHIFT_PRECISION_RUNNING",
        branches=branches,
        chain_minus_full_ad_sigma_per_x=chain,
        completed_branches=len(branches),
        expected_branches=snapshot["expected_branches"],
        point_index=4,
        physical_z=snapshot["physical_z"],
        dz_dx=snapshot["dz_dx"],
        source_ad_delta=snapshot["source_ad_delta"],
        next_stage="REVIEW_BRANCH_PRECISION_EVIDENCE"
        if complete
        else "AUDIT_IN_PROGRESS",
        precision_scope="z-dependent arithmetic only; fixed MDF/SSP and dust transmission retain stored/native precision; no production replacement",
        scientific_promotion=False,
        truth_used=False,
        npe_training_started=False,
        local_optimization_started=False,
        population_training_started=False,
    ), rows


def collect(
    context, spec, point, observation, target, source, budget, *, bands, progress
):
    point = jnp.asarray(point).reshape(1, 1, -1)
    safe, valid = safe_decoder_inputs(point, spec)
    if not bool(np.all(valid)):
        raise ValueError("point4 outside physical domain")
    params = theta_vector_to_model_param_dict(
        x_to_theta(safe, spec).reshape(-1), spec.names, context.model_config
    )
    zi = tuple(spec.names).index("z_obs")
    zero = jnp.asarray(0.0, dtype=jnp.float64)

    def moved(u):
        return point.at[..., zi].set((point[..., zi] + u).astype(point.dtype))

    def mapped_z(u):
        sx, _ = safe_decoder_inputs(moved(u), spec)
        return x_to_theta(sx, spec).reshape(-1)[zi]

    z0 = jnp.asarray(params["z_obs"], dtype=jnp.float64)
    scale = float(jax.grad(mapped_z)(zero))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("invalid redshift transform derivative")
    if float(z0) - abs(scale) * max(STEPS) <= float(spec.lower[zi]) or float(z0) + abs(
        scale
    ) * max(STEPS) >= float(spec.upper[zi]):
        raise ValueError("physical redshift stencil leaves support")

    def canonical(u):
        result = target(moved(u), observation)
        return jnp.where(
            jnp.all(result.physical_valid), result.model_flux.reshape(-1), jnp.nan
        )

    sigma = np.asarray(observation.flux_err, dtype=float).reshape(-1)
    observed = np.asarray(observation.flux, dtype=float).reshape(-1)
    if (
        not np.all(observation.mask)
        or not np.all(np.isfinite(sigma) & (sigma > 0))
        or not np.isfinite(observed).all()
    ):
        raise ValueError("complete finite photometry required")
    budget.charge(3)  # fixed SSP/dust kernels and two branch centers
    physical = diagnostic_spline_redshift_branches(context, params)
    functions = dict(canonical_x=canonical)
    functions.update(
        {name: (lambda u, fn=fn: fn(z0 + scale * u)) for name, fn in physical.items()}
    )
    snapshot = dict(
        schema_version=1,
        point_index=4,
        point_x=np.asarray(point).reshape(-1).tolist(),
        physical_z=float(z0),
        dz_dx=scale,
        branches=[],
        expected_branches=len(functions),
        source_ad_delta=None,
        truth_used=False,
    )
    reference = None
    for name, function in functions.items():
        traced = floating_trace_dtypes(jax.make_jaxpr(function)(zero))
        if name.startswith("zpath64") and traced != ["float64"]:
            raise ValueError(
                f"{name}: reduced precision in z-dependent trace: {traced}"
            )
        fn = jax.jit(function)
        budget.charge(1)
        center_jax = fn(zero)
        center = np.asarray(center_jax, dtype=float)
        budget.charge(1, gradient=True)
        derivative = np.asarray(
            jax.jvp(fn, (zero,), (jnp.ones_like(zero),))[1], dtype=float
        )
        if not np.isfinite(center).all() or not np.isfinite(derivative).all():
            raise ValueError(f"nonfinite center/JVP: {name}")
        if reference is None:
            reference = center
            old = next(
                c
                for c in source["checks"]
                if c["point_index"] == 4
                and c["coordinate"] == "z_obs"
                and c["component"] == "lsst_z"
            )
            delta = float(
                derivative[bands.index("lsst_z")] / sigma[bands.index("lsst_z")]
                - old["ad"]
            )
            snapshot["source_ad_delta"] = delta
            if abs(delta) > 1e-5 + 1e-5 * abs(old["ad"]):
                raise ValueError("point4 canonical source AD not reproduced")
        if name == "mixed_full" and np.max(abs(center - reference) / sigma) > 0.01:
            raise ValueError(
                "branch center differs from canonical flux: check calibration/recomposition"
            )
        samples = []
        for h in STEPS:
            budget.charge(2)
            plus, minus = (
                np.asarray(fn(jnp.asarray(v, dtype=jnp.float64)), dtype=float)
                for v in (h, -h)
            )
            if not np.isfinite(plus).all() or not np.isfinite(minus).all():
                raise ValueError(f"nonfinite stencil: {name}")
            if name == "canonical_x":
                a = float(np.asarray(moved(h) - point).reshape(-1)[zi])
                b = float(np.asarray(point - moved(-h)).reshape(-1)[zi])
            else:
                a = float((z0 + scale * h - z0) / scale)
                b = float((z0 - (z0 - scale * h)) / scale)
            samples.append(
                dict(
                    step=h,
                    actual_plus_step=a,
                    actual_minus_step=b,
                    plus=plus.tolist(),
                    minus=minus.tolist(),
                )
            )
        cases = []
        for j, component in enumerate((*bands, "centered_loglike")):
            density = component == "centered_loglike"
            ad = (
                -np.sum((center - observed) * derivative / sigma**2)
                if density
                else derivative[j] / sigma[j]
            )
            cases.append(
                dict(
                    point_index=4,
                    coordinate="z_obs",
                    component=component,
                    band_index=None if density else j,
                    ad=float(ad),
                    source_ad_delta=0.0,
                    atol=0.1 if density else 0.01,
                    rtol=0.05 if density else 0.01,
                    anchor=center.tolist(),
                    observed=observed.tolist(),
                    sigma=sigma.tolist(),
                    flux_dtype=str(center_jax.dtype),
                    flux_jvp=derivative.tolist(),
                    samples=samples,
                )
            )
        snapshot["branches"].append(
            dict(
                name=name,
                floating_dtypes=traced,
                max_center_delta_sigma=float(np.max(abs(center - reference) / sigma)),
                snapshot=dict(cases=cases, expected_checks=len(cases)),
            )
        )
        progress(snapshot)
        jax.clear_caches()
    return snapshot
