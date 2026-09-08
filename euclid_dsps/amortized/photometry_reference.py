"""Fixed-spectrum integration experiment; no training or posterior promotion."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps import model as sed
from euclid_dsps.parameter_vectors import theta_vector_to_model_param_dict
from euclid_dsps.photometry import abmag_to_fnu_cgs_jax
from euclid_dsps.photometry_quadrature import (
    filter_integral_numpy,
    interpolation_knot_audit,
    linear_product_integral_numpy,
)

from .latent import x_to_theta
from .posterior_target import safe_decoder_inputs
from .redshift_decomposition import floating_trace_dtypes

STEPS = tuple(0.02 / 2**i for i in range(20))


def export_spectra(root, context, spec, points, observation, budget, *, band_names):
    """Save small reproducible assets, not the full SSP bank or checkpoint."""
    if (
        context.model_config.get("sfh_model") != "spline15d"
        or context.model_config.get("agn_model", "none") != "none"
    ):
        raise ValueError("snapshot export requires spline15d without AGN")
    zi = tuple(spec.names).index("z_obs")

    def forward(theta):
        params = theta_vector_to_model_param_dict(
            theta, spec.names, context.model_config
        )
        result = sed.run_spline15d_model_jax(context, params)
        return result.post_igm_sed, abmag_to_fnu_cgs_jax(result.model_mags)

    compiled = jax.jit(forward)
    spectra, eager_spectra, fluxes, theta_rows, redshifts, scales = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for point in points:
        safe, valid = safe_decoder_inputs(point, spec)
        if not bool(np.all(valid)):
            raise ValueError("generated point outside support")
        theta = x_to_theta(safe, spec).reshape(-1)
        budget.charge(2)
        spectrum, flux = compiled(theta)
        eager, _ = forward(theta)

        def zmap(delta, point=point):
            sx, _ = safe_decoder_inputs(point.at[zi].add(delta), spec)
            return x_to_theta(sx, spec)[zi]

        dzdx = float(jax.grad(zmap)(jnp.array(0.0, dtype=point.dtype)))
        if (
            not np.isfinite(dzdx)
            or dzdx == 0
            or float(theta[zi]) <= STEPS[0] * abs(dzdx)
        ):
            raise ValueError("invalid physical redshift stencil")
        spectra.append(np.asarray(spectrum))
        eager_spectra.append(np.asarray(eager))
        fluxes.append(np.asarray(flux))
        theta_rows.append(np.asarray(theta))
        redshifts.append(float(theta[zi]))
        scales.append(abs(dzdx))
    arrays = dict(
        wave=np.asarray(sed._context_ssp_wave(context)),
        spectra_jit=np.stack(spectra),
        spectra_eager=np.stack(eager_spectra),
        model_flux_jit=np.stack(fluxes),
        x=np.asarray(points),
        theta=np.stack(theta_rows),
        z=np.asarray(redshifts),
        step_scale=np.asarray(scales),
        obs_flux=np.asarray(observation.flux).reshape(-1),
        flux_err=np.asarray(observation.flux_err).reshape(-1),
        band_names=np.asarray(band_names),
        coordinate_names=np.asarray(spec.names),
    )
    for i, (wave, trans) in enumerate(context.jax_filters):
        arrays[f"filter_wave_{i:02d}"] = np.asarray(wave)
        arrays[f"transmission_{i:02d}"] = np.asarray(trans)
    path = Path(root) / "FIXED_SPECTRA.npz"
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **arrays)
    return arrays


def reference_plateau(fd, ad, *, atol=1e-3, rtol=1e-3):
    """Choose a plateau from reference FD alone, then compare AD, per band."""
    values = np.asarray(fd, dtype=np.float64)
    reports = []
    for band in range(values.shape[1]):
        selected = None
        for end in range(2, len(values)):
            window = values[end - 2 : end + 1, band]
            if np.isfinite(window).all() and np.ptp(window) <= atol + rtol * np.max(
                np.abs(window)
            ):
                selected = end
        value = None if selected is None else float(values[selected, band])
        passed = (
            selected is not None
            and np.isfinite(ad[band])
            and abs(ad[band] - value) <= atol + rtol * abs(value)
        )
        reports.append(
            dict(
                band_index=band,
                reference_step_index=selected,
                reference_fd=value,
                candidate_ad=float(ad[band]),
                status="PASS"
                if passed
                else "INCONCLUSIVE"
                if selected is None
                else "FAIL",
            )
        )
    return reports


def analyze_snapshot(arrays, budget, *, progress):
    """Can be replayed on CPU from FIXED_SPECTRA.npz, without DSPS SSP assets."""
    from dsps.photometry.photometry_kernels import AB0

    if not jax.config.x64_enabled:
        raise ValueError("JAX x64 required before tracing reference experiment")
    wave = np.asarray(arrays["wave"], dtype=np.float64)
    names = [str(n) for n in arrays["band_names"]]
    filters = [
        (
            np.asarray(arrays[f"filter_wave_{i:02d}"], dtype=np.float64),
            np.asarray(arrays[f"transmission_{i:02d}"], dtype=np.float64),
        )
        for i in range(len(names))
    ]
    sigma = np.asarray(arrays["flux_err"], dtype=np.float64)
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("invalid photometric errors")
    denominators = np.asarray(
        [AB0 * filter_integral_numpy(fw, ft) for fw, ft in filters]
    )
    if not np.isfinite(denominators).all() or np.any(denominators <= 0):
        raise ValueError("non-positive filter normalization")
    records, reports = [], []
    for index, (spectrum, z, scale) in enumerate(
        zip(arrays["spectra_jit"], arrays["z"], arrays["step_scale"], strict=True)
    ):
        spectrum = np.asarray(spectrum, dtype=np.float64)
        z = jnp.asarray(z, dtype=jnp.float64)
        physical_steps = np.asarray(STEPS) * scale

        def independent(zz, spectrum=spectrum):
            # Dimming is deliberately shared; quadrature is independently analytic.
            factor = float(
                sed.fixed_spectrum_dimming_factor_jax(
                    jnp.asarray(zz, dtype=jnp.float64)
                )
            )
            return (
                factor
                * np.asarray(
                    [
                        linear_product_integral_numpy(wave, spectrum, fw, ft, float(zz))
                        for fw, ft in filters
                    ]
                )
                / denominators
            )

        progress(index, "analytic_reference_start", records, reports)
        budget.charge(1)
        reference = independent(z)
        reference_sides = []
        for h in physical_steps:
            budget.charge(2)
            reference_sides.append((independent(z + h), independent(z - h)))
        reference_fd = np.asarray(
            [
                (p - m) / (2 * h * sigma)
                for (p, m), h in zip(reference_sides, physical_steps, strict=True)
            ]
        )
        centers, derivatives, comparisons, traces = {}, {}, {}, {}
        for label, method, order in (
            ("legacy64", "legacy", 8),
            ("merged4", "merged", 4),
            ("merged8", "merged", 8),
            ("merged8_legacy_ab", "merged_legacy_ab", 8),
        ):
            progress(index, label + "_start", records, reports)

            def project(zz, method=method, order=order, spectrum=spectrum):
                return jnp.stack(
                    [
                        sed.fixed_spectrum_projection_jax(
                            jnp.asarray(wave),
                            jnp.asarray(spectrum),
                            jnp.asarray(fw),
                            jnp.asarray(ft),
                            zz,
                            method=method,
                            order=order,
                        )
                        for fw, ft in filters
                    ]
                )

            traces[label] = floating_trace_dtypes(jax.make_jaxpr(project)(z))
            if traces[label] != ["float64"]:
                raise ValueError(
                    f"hidden reduced precision in {label}: {traces[label]}"
                )
            project = jax.jit(project)
            budget.charge(1)
            center = np.asarray(project(z), dtype=np.float64)
            budget.charge(1, gradient=True)
            ad = np.asarray(jax.jvp(project, (z,), (jnp.ones_like(z),))[1]) / sigma
            centers[label], derivatives[label] = center, ad
            max_scaled_error = float(
                np.max(
                    abs(center - reference) / (1e-3 * sigma + 1e-10 * abs(reference))
                )
            )
            for step_index, (h, (rp, rm)) in enumerate(
                zip(physical_steps, reference_sides, strict=True)
            ):
                budget.charge(2)
                plus, minus = np.asarray(project(z + h)), np.asarray(project(z - h))
                fd = (plus - minus) / (2 * h * sigma)
                for f, ref in ((plus, rp), (minus, rm)):
                    if not np.isfinite(f).all() or not np.isfinite(ref).all():
                        max_scaled_error = float("inf")
                    max_scaled_error = max(
                        max_scaled_error,
                        float(np.max(abs(f - ref) / (1e-3 * sigma + 1e-10 * abs(ref)))),
                    )
                for band, name in enumerate(names):
                    records.append(
                        dict(
                            point_index=index,
                            method=label,
                            band=name,
                            step_index=step_index,
                            physical_z_step=float(h),
                            ad_sigma_per_z=float(ad[band]),
                            fd_sigma_per_z=float(fd[band]),
                            analytic_fd_sigma_per_z=float(
                                reference_fd[step_index, band]
                            ),
                            flux=center[band],
                            reference_flux=reference[band],
                            center_delta_sigma=(center[band] - reference[band])
                            / sigma[band],
                            plus_delta_sigma=(plus[band] - rp[band]) / sigma[band],
                            minus_delta_sigma=(minus[band] - rm[band]) / sigma[band],
                        )
                    )
            comparisons[label] = dict(
                max_scaled_reference_error=max_scaled_error,
                reference_agreement=bool(
                    np.isfinite(max_scaled_error) and max_scaled_error <= 1
                ),
            )
            progress(index, label, records, reports)
        # Spectrum construction differences are measured, not attributed to quadrature.
        eager = np.asarray(arrays["spectra_eager"][index], dtype=np.float64)
        budget.charge(1)
        eager_flux = np.asarray(
            [
                sed.fixed_spectrum_projection_jax(
                    jnp.asarray(wave),
                    jnp.asarray(eager),
                    jnp.asarray(fw),
                    jnp.asarray(ft),
                    z,
                )
                for fw, ft in filters
            ]
        )
        normalization_delta = (
            centers["merged8_legacy_ab"] - centers["merged8"]
        ) / sigma
        order_delta = (centers["merged4"] - centers["merged8"]) / sigma
        gradient_checks = reference_plateau(reference_fd, derivatives["merged8"])
        passed = (
            comparisons["merged8"]["reference_agreement"]
            and comparisons["merged4"]["reference_agreement"]
            and all(r["status"] == "PASS" for r in gradient_checks)
        )
        reports.append(
            dict(
                point_index=index,
                physical_z=float(z),
                floating_trace_dtypes=traces,
                integration_comparisons=comparisons,
                candidate_gradient_checks=gradient_checks,
                max_abs_legacy_reference_delta_sigma=float(
                    np.max(abs(centers["legacy64"] - reference) / sigma)
                ),
                max_abs_normalization_delta_sigma=float(
                    np.max(abs(normalization_delta))
                ),
                max_abs_order4_order8_delta_sigma=float(np.max(abs(order_delta))),
                max_abs_eager_jit_projection_delta_sigma=float(
                    np.max(abs(eager_flux - centers["legacy64"]) / sigma)
                ),
                max_abs_fixed_projection_full_model_delta_sigma=float(
                    np.max(
                        abs(centers["legacy64"] - arrays["model_flux_jit"][index])
                        / sigma
                    )
                ),
                knots={
                    name: interpolation_knot_audit(wave, fw, float(z), physical_steps)
                    for name, (fw, _) in zip(names, filters, strict=True)
                },
                numerical_reference_checks="PASS" if passed else "NOT_PASSED",
            )
        )
        progress(index, "point_complete", records, reports)
        jax.clear_caches()
    all_pass = all(r["numerical_reference_checks"] == "PASS" for r in reports)
    return dict(
        status="PHOTOMETRY_REFERENCE_COMPLETE",
        points=reports,
        numerical_reference_checks="PASS" if all_pass else "NOT_PASSED",
        next_stage="FULL_DECODER_QUALIFICATION_REQUIRED"
        if all_pass
        else "INVESTIGATE_INTEGRATION",
        reference_contract="analytic integral of supplied piecewise-linear SED and filter, with same DSPS cosmology/AB constant; not independent astrophysical truth",
        tolerances=dict(
            flux_absolute_in_sigma=1e-3,
            flux_relative=1e-10,
            gradient_absolute_sigma_per_z=1e-3,
            gradient_relative=1e-3,
        ),
        production_decoder_changed=False,
        local_optimization_started=False,
        population_training_started=False,
        npe_training_started=False,
        truth_used=False,
        scientific_promotion=False,
    ), records


def write_progress(root, index, stage, rows, points, budget):
    root = Path(root)
    pd.DataFrame(rows).to_csv(root / "photometry_reference.csv", index=False)
    for name, payload in (
        ("PHOTOMETRY_REFERENCE_PARTIAL.json", dict(points=points)),
        (
            "PROGRESS.json",
            dict(
                stage="photometry_reference",
                branch=stage,
                point_index=index,
                budget=budget.snapshot(),
            ),
        ),
    ):
        path = root / name
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(finite_report(payload), indent=2, allow_nan=False) + "\n"
        )
        temporary.replace(path)


def finite_report(value):
    if isinstance(value, dict):
        return {k: finite_report(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_report(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
