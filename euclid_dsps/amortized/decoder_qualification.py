"""Bounded full-target checks. Numerical completion never promotes a posterior."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from .redshift_decomposition import floating_trace_dtypes

STEPS = tuple(0.02 / 2**i for i in range(10))


def resolved_check(fd, resolution, ad, *, atol, rtol):
    """Finest three-step FD plateau, selected without consulting AD."""
    fd, resolution = np.asarray(fd), np.asarray(resolution)
    selected = None
    for end in range(2, len(fd)):
        window = fd[end - 2 : end + 1]
        tol = atol + rtol * abs(fd[end])
        if (
            np.isfinite(window).all()
            and np.ptp(window) <= tol
            and np.all(resolution[end - 2 : end + 1] <= tol)
        ):
            selected = end
    status = "INCONCLUSIVE"
    if not np.isfinite(ad) or not np.isfinite(fd).all():
        status = "FAIL"
    elif selected is not None:
        status = (
            "PASS"
            if abs(ad - fd[selected]) <= atol + rtol * abs(fd[selected])
            else "FAIL"
        )
    return dict(
        status=status,
        selected_step_index=selected,
        ad=float(ad),
        fd=None if selected is None else float(fd[selected]),
        atol=atol,
        rtol=rtol,
    )


def qualify(
    legacy,
    candidate,
    points,
    observations,
    budget,
    *,
    names,
    bands,
    progress,
    labels=("legacy", "merged"),
):
    """Check all latent-x coordinates with full SED and canonical likelihood.

    Caller enforces Gaussian likelihood, complete masks, zero floors and fixed
    calibration. Centered loglike removes only the observation-only constant.
    Derivatives still traverse the entire decoder. No finite draws are filtered.
    """
    if not len(points) or not len(names) or not len(bands):
        raise ValueError("nonempty qualification required")
    if len(labels) != 2 or labels[0] == labels[1]:
        raise ValueError("two distinct variant labels required")
    rows, cases = [], []
    components = [*bands, "centered_loglike", "logprior", "canonical_loglike"]
    centers = {}
    for label, target in zip(labels, (legacy, candidate), strict=True):

        def measure(x, observation, target=target):
            values = target(x, observation)
            flux = values.model_flux.reshape(-1).astype(jnp.float64)
            sigma = observation.flux_err.reshape(-1).astype(jnp.float64)
            residual = (flux - observation.flux.reshape(-1)) / sigma
            packed = jnp.concatenate(
                (
                    flux / sigma,
                    jnp.array(
                        [
                            -0.5 * jnp.sum(residual**2),
                            jnp.sum(values.logprior),
                            jnp.sum(values.loglike),
                        ]
                    ),
                )
            )
            return jnp.where(jnp.all(values.physical_valid), packed, jnp.nan)

        measured = jax.jit(measure)
        tangent_fn = jax.jit(
            lambda x, obs, d: jax.jvp(lambda xx: measure(xx, obs), (x,), (d,))[1]
        )
        reverse_fn = jax.jit(
            jax.grad(lambda x, obs, target=target: jnp.sum(target(x, obs).logtarget))
        )
        for index, (point, obs) in enumerate(zip(points, observations, strict=True)):
            started = time.monotonic()
            progress(label, index, "start", rows, cases)
            point = jnp.asarray(point).reshape(1, 1, -1)
            if not np.all(np.asarray(obs.mask)) or np.any(
                np.asarray(obs.flux_err) <= 0
            ):
                raise ValueError(
                    "qualification requires complete positive-error context"
                )
            budget.charge(1)
            center = np.asarray(measured(point, obs))
            if not np.isfinite(center).all():
                raise ValueError(f"nonfinite full target: {label} point {index}")
            centers[label, index] = center
            budget.charge(1, gradient=True)
            reverse = np.asarray(reverse_fn(point, obs)).reshape(-1)
            # Measure steady single-object cost after compilation separately.
            steady_start = time.monotonic()
            for _ in range(3):
                budget.charge(1)
                measured(point, obs).block_until_ready()
            steady_seconds = (time.monotonic() - steady_start) / 3
            traces = floating_trace_dtypes(jax.make_jaxpr(measure)(point, obs))
            checks, chains = [], []
            for coordinate, name in enumerate(names):
                tangent = jnp.zeros_like(point).at[..., coordinate].set(1)
                budget.charge(1, gradient=True)
                ad = np.asarray(tangent_fn(point, obs, tangent))
                fds, resolutions = [], []
                for h in STEPS:
                    budget.charge(2)
                    plus = np.asarray(measured(point + h * tangent, obs))
                    minus = np.asarray(measured(point - h * tangent, obs))
                    fd = (plus - minus) / (2 * h)
                    # Conservative float32 output screen: the full path remains mixed.
                    ulp = (
                        4
                        * (
                            abs(np.spacing(plus.astype(np.float32))).astype(float)
                            + abs(np.spacing(minus.astype(np.float32))).astype(float)
                        )
                        / (2 * h)
                    )
                    fds.append(fd)
                    resolutions.append(ulp)
                    for j, component in enumerate(components):
                        rows.append(
                            dict(
                                variant=label,
                                point_index=index,
                                coordinate=name,
                                component=component,
                                step=h,
                                ad=float(ad[j]),
                                fd=float(fd[j]),
                                resolution=float(ulp[j]),
                                plus=float(plus[j]),
                                minus=float(minus[j]),
                            )
                        )
                for j, component in enumerate(components):
                    check = resolved_check(
                        np.asarray(fds)[:, j],
                        np.asarray(resolutions)[:, j],
                        ad[j],
                        atol=0.01 if j < len(bands) else 0.1,
                        rtol=0.01 if j < len(bands) else 0.05,
                    )
                    checks.append(dict(coordinate=name, component=component, **check))
                chains.append(
                    dict(
                        coordinate=name,
                        reverse_target_delta=float(
                            reverse[coordinate] - (ad[-1] + ad[-2])
                        ),
                        reverse_passed=bool(
                            np.isfinite(reverse[coordinate])
                            and abs(reverse[coordinate] - (ad[-1] + ad[-2]))
                            <= 0.01 + 0.001 * abs(ad[-1] + ad[-2])
                        ),
                        delta=float(ad[-1] - ad[-3]),
                        passed=bool(
                            np.isfinite(ad[-1])
                            and np.isfinite(ad[-3])
                            and abs(ad[-1] - ad[-3]) <= 0.1 + 0.05 * abs(ad[-3])
                        ),
                    )
                )
                progress(label, index, name, rows, cases)
            # Canonical scalar FD is retained but may be unresolved after its float32
            # normalization sum. Require centered FD and agreement of canonical AD.
            required = [c for c in checks if c["component"] != "canonical_loglike"]
            passed = all(c["status"] == "PASS" for c in required) and all(
                c["passed"] and c["reverse_passed"] for c in chains
            )
            delta = center[: len(bands)] - centers[labels[0], index][: len(bands)]
            memory = jax.local_devices()[0].memory_stats() or {}
            cases.append(
                dict(
                    variant=label,
                    point_index=index,
                    numerical_checks="PASS" if passed else "NOT_PASSED",
                    checks=checks,
                    canonical_centered_gradient=chains,
                    floating_trace_dtypes=traces,
                    x_dtype=str(point.dtype),
                    steady_forward_seconds=steady_seconds,
                    total_seconds_including_compilation=time.monotonic() - started,
                    device_memory={
                        k: int(memory[k])
                        for k in ("bytes_in_use", "peak_bytes_in_use")
                        if k in memory
                    },
                    forward_delta_sigma=delta.tolist(),
                    max_abs_forward_delta_sigma=float(np.max(abs(delta))),
                    loglike_delta=float(center[-1] - centers[labels[0], index][-1]),
                    prior_delta=float(center[-2] - centers[labels[0], index][-2]),
                )
            )
            progress(label, index, "point_complete", rows, cases)
        # Free the old compilation before compiling the candidate full target.
        jax.clear_caches()
    numerical = all(
        c["numerical_checks"] == "PASS" for c in cases if c["variant"] == labels[1]
    )
    return dict(
        status="FULL_DECODER_QUALIFICATION_COMPLETE",
        variant_labels=list(labels),
        cases=cases,
        candidate_numerical_checks="PASS" if numerical else "NOT_PASSED",
        next_stage="SIMULATOR_COMPATIBILITY_REVIEW_REQUIRED"
        if numerical
        else "INVESTIGATE_FULL_TARGET",
        coordinates="latent_x, not physical theta; fixed observations and errors",
        resolution_contract="conservative float32 output ULP screen, not an internal rounding-error bound",
        catalogue_simulator_compatibility="NOT_VERIFIED",
        old_flux_bank_reuse_authorized=False,
        local_optimization_started=False,
        npe_training_started=False,
        population_training_started=False,
        scientific_promotion=False,
        truth_used=False,
    ), rows
