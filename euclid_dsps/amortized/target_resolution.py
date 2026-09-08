"""Bounded mixed-precision derivative audit, independent of likelihood offsets."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .decoder_qualification import resolved_check

STEPS = tuple(0.02 / 2 ** (i / 2) for i in range(25))


def gaussian_response(flux, anchor, observed, sigma):
    """Log L(flux)-log L(anchor), without subtracting two quadratic sums."""
    flux, anchor, observed, sigma = (
        np.asarray(v, dtype=np.float64) for v in (flux, anchor, observed, sigma)
    )
    delta = (flux - anchor) / sigma
    residual = (anchor - observed) / sigma
    return -delta * (residual + 0.5 * delta)


def output_ulp(flux, dtype):
    return 4 * abs(np.spacing(np.asarray(flux, dtype=np.dtype(dtype)))).astype(float)


def response_screen(flux, anchor, observed, sigma, dtype):
    """Propagate output quantization only, NOT unknown upstream roundoff."""
    flux, anchor, observed, sigma = map(np.asarray, (flux, anchor, observed, sigma))
    delta = (flux - anchor) / sigma
    residual = (anchor - observed) / sigma
    anchor_error = output_ulp(anchor, dtype) / sigma
    delta_error = output_ulp(flux, dtype) / sigma + anchor_error
    arithmetic = 8 * np.finfo(np.float64).eps * abs(delta * (residual + 0.5 * delta))
    return abs(residual + delta) * delta_error + abs(delta) * anchor_error + arithmetic


def analyze(snapshot):
    """CPU replay. Stencil choice sees FD and its resolution, never AD."""
    reports, rows = [], []
    for case in snapshot["cases"]:
        component = case["component"]
        density = component == "centered_loglike"
        anchor, observed, sigma = (
            np.asarray(case[k], dtype=float) for k in ("anchor", "observed", "sigma")
        )
        fds, screens, products = [], [], []
        for s in case["samples"]:
            plus, minus = np.asarray(s["plus"]), np.asarray(s["minus"])
            a, b = s["actual_plus_step"], s["actual_minus_step"]
            if a <= 0 or b <= 0:
                raise ValueError("unrepresentable finite-difference step")
            if density:
                dp = gaussian_response(plus, anchor, observed, sigma).sum()
                dm = gaussian_response(minus, anchor, observed, sigma).sum()
                ep = response_screen(
                    plus, anchor, observed, sigma, case["flux_dtype"]
                ).sum()
                em = response_screen(
                    minus, anchor, observed, sigma, case["flux_dtype"]
                ).sum()
            else:
                j = case["band_index"]
                dp, dm = (
                    (plus[j] - anchor[j]) / sigma[j],
                    (minus[j] - anchor[j]) / sigma[j],
                )
                ep = (
                    output_ulp(plus, case["flux_dtype"])[j]
                    + output_ulp(anchor, case["flux_dtype"])[j]
                ) / sigma[j]
                em = (
                    output_ulp(minus, case["flux_dtype"])[j]
                    + output_ulp(anchor, case["flux_dtype"])[j]
                ) / sigma[j]
            fd = (b * dp / a - a * dm / b) / (a + b)
            screen = (b * ep / a + a * em / b) / (a + b)
            fds.append(float(fd))
            screens.append(float(screen))
            products.append(a * b)
        fds, screens, products = map(np.asarray, (fds, screens, products))
        # Richardson comparison estimates truncation from nested stencils only.
        rich = np.full_like(fds, np.nan)
        rich_screen = np.full_like(fds, np.inf)
        for i in range(2, len(fds)):
            ratio = products[i - 2] / products[i]
            rich[i] = (ratio * fds[i] - fds[i - 2]) / (ratio - 1)
            rich_screen[i] = (ratio * screens[i] + screens[i - 2]) / (ratio - 1)
        guard = np.maximum.reduce((screens, rich_screen, abs(rich - fds)))
        # A plateau must exist in both second- and fourth-order estimates.
        selected = None
        for end in range(4, len(fds)):
            w = slice(end - 2, end + 1)
            tol = case["atol"] + case["rtol"] * abs(fds[end])
            if (
                np.isfinite(fds[w]).all()
                and np.isfinite(rich[w]).all()
                and np.ptp(fds[w]) <= tol
                and np.ptp(rich[w]) <= tol
                and np.all(guard[w] <= tol)
            ):
                selected = end
        if selected is None:
            status = (
                "FAIL"
                if not np.isfinite(fds).all() or not np.isfinite(case["ad"])
                else "INCONCLUSIVE"
            )
            check = dict(status=status, ad=case["ad"], fd=None)
        else:
            check = resolved_check(
                fds[selected - 2 : selected + 1],
                guard[selected - 2 : selected + 1],
                case["ad"],
                atol=case["atol"],
                rtol=case["rtol"],
            )
        check.update(
            point_index=case["point_index"],
            coordinate=case["coordinate"],
            component=component,
            selected_step_index=selected,
            atol=case["atol"],
            rtol=case["rtol"],
            source_ad_delta=case["source_ad_delta"],
        )
        reports.append(check)
        for i, s in enumerate(case["samples"]):
            rows.append(
                dict(
                    point_index=case["point_index"],
                    coordinate=case["coordinate"],
                    component=component,
                    step=s["step"],
                    fd=fds[i],
                    ad=case["ad"],
                    richardson_fd=rich[i],
                    truncation_proxy=abs(rich[i] - fds[i]),
                    output_resolution=screens[i],
                    guard=guard[i],
                    actual_plus_step=s["actual_plus_step"],
                    actual_minus_step=s["actual_minus_step"],
                )
            )
    complete = len(reports) == snapshot["expected_checks"]
    passed = complete and bool(reports) and all(c["status"] == "PASS" for c in reports)
    return dict(
        status="TARGET_RESOLUTION_COMPLETE" if complete else "TARGET_RESOLUTION_RUNNING",
        completed_checks=len(reports),
        expected_checks=snapshot["expected_checks"],
        checks=reports,
        unresolved_checks_resolved=passed,
        next_stage="AUDIT_IN_PROGRESS"
        if not complete
        else "FULL_REQUALIFICATION_AND_SIMULATOR_REVIEW_REQUIRED"
        if passed
        else "INVESTIGATE_REMAINING_STENCILS",
        interpretation="output roundoff screen plus empirical stencil convergence; not a bound on upstream rounding or a proof of all derivatives",
        scientific_promotion=False,
        truth_used=False,
        npe_training_started=False,
        population_training_started=False,
    ), rows


def collect(
    target, points, observations, source_report, budget, *, names, bands, progress
):
    """Revisit every unresolved required source check, at unchanged inputs."""
    source_label = source_report["variant_labels"][-1]
    pending = [
        dict(c, point_index=case["point_index"])
        for case in source_report["cases"]
        if case["variant"] == source_label
        for c in case["checks"]
        if c["status"] != "PASS" and c["component"] != "canonical_loglike"
    ]
    if not 1 <= len(pending) <= 8:
        raise ValueError("bounded residual audit requires 1..8 unresolved checks")
    snapshot = dict(
        schema_version=1,
        cases=[],
        steps=list(STEPS),
        source_variant=source_label,
        expected_checks=len(pending),
        truth_used=False,
    )

    def flux(x, obs):
        value = target(x, obs)
        return jnp.where(
            jnp.all(value.physical_valid), value.model_flux.reshape(-1), jnp.nan
        )

    forward = jax.jit(flux)
    tangent = jax.jit(
        lambda x, obs, d: jax.jvp(lambda xx: flux(xx, obs), (x,), (d,))[1]
    )
    for c in pending:
        if c["component"] != "centered_loglike" and c["component"] not in bands:
            raise ValueError(
                "residual audit supports flux and centered Gaussian likelihood only"
            )
        point = jnp.asarray(points[c["point_index"]]).reshape(1, 1, -1)
        obs = observations[c["point_index"]]
        coordinate = names.index(c["coordinate"])
        direction = jnp.zeros_like(point).at[..., coordinate].set(1)
        budget.charge(1)
        anchor_jax = forward(point, obs)
        anchor = np.asarray(anchor_jax, dtype=float)
        budget.charge(1, gradient=True)
        deriv = np.asarray(tangent(point, obs, direction), dtype=float)
        observed, sigma = (
            np.asarray(obs.flux).reshape(-1),
            np.asarray(obs.flux_err).reshape(-1),
        )
        if (
            not np.all(np.asarray(obs.mask))
            or not np.all(sigma > 0)
            or not np.isfinite(anchor).all()
        ):
            raise ValueError("invalid photometry or full-target flux")
        band_index = (
            None
            if c["component"] == "centered_loglike"
            else bands.index(c["component"])
        )
        ad = (
            -np.sum((anchor - observed) / sigma * deriv / sigma)
            if band_index is None
            else deriv[band_index] / sigma[band_index]
        )
        delta = float(ad - c["ad"])
        if not np.isfinite(ad) or abs(delta) > 1e-5 + 1e-5 * abs(c["ad"]):
            raise ValueError("source AD not reproduced; do not silently mix targets")
        case = dict(
            point_index=c["point_index"],
            coordinate=c["coordinate"],
            component=c["component"],
            ad=float(ad),
            source_ad_delta=delta,
            atol=c["atol"],
            rtol=c["rtol"],
            band_index=band_index,
            flux_dtype=str(anchor_jax.dtype),
            x_dtype=str(point.dtype),
            point_x=np.asarray(point).reshape(-1).tolist(),
            flux_jvp=deriv.tolist(),
            anchor=anchor.tolist(),
            observed=observed.tolist(),
            sigma=sigma.tolist(),
            samples=[],
        )
        for h in STEPS:
            xp, xm = point + h * direction, point - h * direction
            a = float(np.asarray(xp - point).reshape(-1)[coordinate])
            b = float(np.asarray(point - xm).reshape(-1)[coordinate])
            budget.charge(2)
            plus, minus = (
                np.asarray(forward(xp, obs), dtype=float),
                np.asarray(forward(xm, obs), dtype=float),
            )
            if not np.isfinite(plus).all() or not np.isfinite(minus).all():
                raise ValueError("nonfinite stencil flux")
            case["samples"].append(
                dict(
                    step=h,
                    actual_plus_step=a,
                    actual_minus_step=b,
                    plus=plus.tolist(),
                    minus=minus.tolist(),
                )
            )
        snapshot["cases"].append(case)
        progress(snapshot)
    return snapshot
