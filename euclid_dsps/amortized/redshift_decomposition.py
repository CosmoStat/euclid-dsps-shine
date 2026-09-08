"""Bounded redshift branch measurements, never an alternative training decoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps import model as sed
from euclid_dsps.parameter_vectors import theta_vector_to_model_param_dict
from euclid_dsps.photometry import abmag_to_fnu_cgs_jax
from euclid_dsps.prior_learning.spline15d import (
    SFH_CONTRAST_NAMES,
    reconstruct_relative_sfh_jax,
)

from .latent import x_to_theta
from .posterior_target import safe_decoder_inputs

STEPS = tuple(0.02 / 2**i for i in range(10))


def floating_trace_dtypes(closed):
    """Inspect nested traced arithmetic, including constants and explicit casts."""
    found = set()

    def visit(value):
        if hasattr(value, "jaxpr"):
            visit(value.jaxpr)
            for constant in getattr(value, "consts", ()):
                visit(constant)
        elif hasattr(value, "eqns"):
            for var in (*value.constvars, *value.invars, *value.outvars):
                visit(getattr(var, "aval", None))
            for equation in value.eqns:
                for var in (*equation.invars, *equation.outvars):
                    visit(getattr(var, "aval", None))
                visit(equation.params)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        else:
            dtype = getattr(value, "dtype", None)
            if dtype is not None:
                try:
                    dtype = np.dtype(dtype)
                except TypeError:
                    return
                if np.issubdtype(dtype, np.floating):
                    found.add(str(dtype))

    visit(closed)
    return sorted(found)


def measure_curve(fn, z, scale, sigma, budget, *, branch, labels, observed=None):
    """Differentiate physical z; report steps comparable to the earlier x audit."""
    compiled = jax.jit(fn)
    budget.charge(1)
    center = np.asarray(compiled(z), dtype=np.float64).reshape(-1)
    budget.charge(1, gradient=True)
    ad = np.asarray(jax.jvp(compiled, (z,), (jnp.ones_like(z),))[1], dtype=np.float64)
    ad = ad.reshape(-1) / sigma
    if center.shape != sigma.shape or len(labels) != center.size:
        raise ValueError("branch output/normalization shape mismatch")
    residual = None if observed is None else (center - observed) / sigma
    records = []
    for step in STEPS:
        h = step * scale
        zp, zm = z + h, z - h
        if float(zm) <= 0:
            raise ValueError("redshift stencil leaves positive physical domain")
        budget.charge(2)
        plus = np.asarray(compiled(zp), dtype=np.float64).reshape(-1)
        minus = np.asarray(compiled(zm), dtype=np.float64).reshape(-1)
        fd = (plus - minus) / (2 * h * sigma)
        for i, label in enumerate(labels):
            record = dict(
                branch=branch,
                component=label,
                equivalent_x_step=step,
                physical_z_step=h,
                z_plus=float(zp),
                z_minus=float(zm),
                actual_z_span=float(zp) - float(zm),
                center=center[i],
                plus=plus[i],
                minus=minus[i],
                normalization=sigma[i],
                ad=ad[i],
                fd=fd[i],
                finite=bool(np.isfinite([center[i], ad[i], fd[i]]).all()),
            )
            if residual is not None:
                rp, rm = (
                    (plus[i] - observed[i]) / sigma[i],
                    (minus[i] - observed[i]) / sigma[i],
                )
                record.update(
                    loglike_ad_contribution=-residual[i] * ad[i],
                    loglike_centered_fd_contribution=-0.5
                    * (rp - rm)
                    * (rp + rm)
                    / (2 * h),
                )
            records.append(record)
    return center, ad, records


def make_age_mass_weights(params, context, *, dtype=None):
    """Same SFH/mass equations; dtype=None reproduces production casts.

    Float64 here is restricted to weights. SSP/dust spectra and photometry are
    not part of this sub-calculation; casting stored assets recovers no lost bits.
    """
    from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
    from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table

    numerical = jnp.float32 if dtype is None else dtype
    cosmology = (
        DEFAULT_COSMOLOGY
        if dtype is None
        else tuple(jnp.asarray(v, dtype=dtype) for v in DEFAULT_COSMOLOGY)
    )
    contrasts = jnp.stack(
        [jnp.asarray(params[n], dtype=numerical) for n in SFH_CONTRAST_NAMES]
    )
    ages = sed._context_ssp_lg_age_gyr(context)
    met = sed.log10_stellar_metallicity_to_absolute_jax(
        params["log10_stellar_metallicity"], context.z_sun
    )
    surviving = sed._diffsky_basic_surviving_mstar_by_age_jax(
        context, context.model_config, met
    )
    if dtype is not None:
        ages = ages.astype(dtype)
        if surviving is not None:
            surviving = surviving.astype(dtype)
    logmass = jnp.asarray(params["log10_stellar_mass"], dtype=numerical)

    def weights(z):
        time = jnp.ravel(age_at_z(jnp.asarray(z, dtype=numerical), *cosmology))[0]
        grid = jnp.linspace(
            0.05, jnp.maximum(time, 0.06), context.n_sfh_bins, dtype=dtype
        )
        raw = reconstruct_relative_sfh_jax(grid, contrasts)
        sfr, mass, _ = sed.normalize_sfh_to_stellar_mass_jax(
            grid,
            raw,
            ages,
            time,
            logmass,
            surviving,
            numerical_dtype=numerical,
        )
        return calc_age_weights_from_sfh_table(grid, sfr, ages, time) * mass

    return weights


def build_branches(context, params):
    """Split z into independent stellar, IGM, and projection inputs."""
    config = context.model_config
    if (
        config.get("sfh_model") != "spline15d"
        or config.get("agn_model", "none") != "none"
    ):
        raise ValueError("redshift decomposition supports spline15d without AGN only")
    z0 = jnp.asarray(params["z_obs"], dtype=jnp.float32)
    native = sed.run_spline15d_model_jax(context, params)
    wave = sed._context_ssp_wave(context)
    rest = native.pre_igm_sed
    post = native.post_igm_sed
    wave64, post64 = wave.astype(jnp.float64), post.astype(jnp.float64)
    filters64 = tuple(
        (fw.astype(jnp.float64), ft.astype(jnp.float64))
        for fw, ft in context.jax_filters
    )

    def project(spectrum, z):
        return abmag_to_fnu_cgs_jax(sed.predict_mags_jax(context, wave, spectrum, z))

    def attenuate(spectrum, z):
        return sed.apply_igm_transmission_jax(wave, spectrum, z, config)

    def stellar(z):
        return sed.run_spline15d_model_jax(context, {**params, "z_obs": z}).pre_igm_sed

    def projection64(z):
        from dsps import calc_obs_mag
        from dsps.cosmology import DEFAULT_COSMOLOGY

        args = tuple(jnp.asarray(v, dtype=jnp.float64) for v in DEFAULT_COSMOLOGY)
        mags = jnp.stack(
            [calc_obs_mag(wave64, post64, fw, ft, z, *args) for fw, ft in filters64]
        )
        return abmag_to_fnu_cgs_jax(mags)

    return dict(
        full_native=lambda z: abmag_to_fnu_cgs_jax(
            sed.run_spline15d_model_jax(context, {**params, "z_obs": z}).model_mags
        ),
        recomposed_native=lambda z: project(attenuate(stellar(z), z), z),
        stellar_native=lambda z: project(attenuate(stellar(z), z0), z0),
        igm_native=lambda z: project(attenuate(rest, z), z0),
        projection_native=lambda z: project(post, z),
        projection_float64=projection64,
    )


def decompose_redshift(
    context,
    spec,
    points,
    observation,
    budget,
    *,
    band_names,
    progress,
    canonical_flux=None,
):
    """Three fixed generated points; Gaussian residuals for a fixed observed context."""
    if not jax.config.x64_enabled:
        raise ValueError(
            "enable JAX x64 before loading assets; native casts stay unchanged"
        )
    if not context.jax_filters:
        raise ValueError("loaded JAX filter arrays required")
    sigma = np.asarray(observation.flux_err, dtype=np.float64).reshape(-1)
    observed = np.asarray(observation.flux, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(sigma) & (sigma > 0)) or not np.all(observation.mask):
        raise ValueError("complete finite positive-error photometry required")
    results, rows = [], []
    zi = tuple(spec.names).index("z_obs")
    for index, point in enumerate(points):
        safe, valid = safe_decoder_inputs(point, spec)
        if not bool(np.all(valid)):
            raise ValueError("generated point outside configured physical support")
        physical = x_to_theta(safe, spec).reshape(-1)
        params = theta_vector_to_model_param_dict(
            physical, spec.names, context.model_config
        )
        z = jnp.asarray(params["z_obs"], dtype=jnp.float32)

        def mapped(delta, point=point):
            moved = point.at[..., zi].add(delta)
            sx, _ = safe_decoder_inputs(moved, spec)
            return x_to_theta(sx, spec).reshape(-1)[zi]

        dzdx = float(jax.grad(mapped)(jnp.array(0.0, dtype=jnp.float32)))
        if not np.isfinite(dzdx) or dzdx == 0:
            raise ValueError("zero/nonfinite redshift transform derivative")
        scale = abs(dzdx)
        if float(z) - STEPS[0] * scale <= float(spec.lower[zi]) or float(z) + STEPS[
            0
        ] * scale >= float(spec.upper[zi]):
            raise ValueError(
                "redshift stencil reaches configured support boundary; no silent point replacement"
            )
        budget.charge(1)
        branches = build_branches(context, params)
        centers, derivatives, trace = {}, {}, {}
        for name, fn in branches.items():
            progress(index, name + "_start", rows, results)
            value = z.astype(jnp.float64) if name.endswith("float64") else z
            trace[name] = floating_trace_dtypes(jax.make_jaxpr(fn)(value))
            if name.endswith("float64") and trace[name] != ["float64"]:
                raise ValueError(
                    f"double projection still contains reduced precision: {trace[name]}"
                )
            center, ad, records = measure_curve(
                fn,
                value,
                scale,
                sigma,
                budget,
                branch=name,
                labels=band_names,
                observed=observed,
            )
            centers[name], derivatives[name] = center, ad
            for record in records:
                record["point_index"] = index
            rows.extend(records)
            progress(index, name, rows, results)

        canonical_delta = None
        if canonical_flux is not None:
            budget.charge(1)
            reference = np.asarray(canonical_flux(point), dtype=np.float64).reshape(-1)
            canonical_delta = (centers["full_native"] - reference) / sigma
            if (
                not np.isfinite(canonical_delta).all()
                or np.max(np.abs(canonical_delta)) > 0.01
            ):
                raise ValueError(
                    "branch decoder disagrees with canonical decoder by more than 0.01 photometric error"
                )

        ages = sed._context_ssp_lg_age_gyr(context)
        weight_centers = {}
        normalization = np.full(ages.size, float(10.0 ** params["log10_stellar_mass"]))
        for name, fn, value in (
            (
                "age_mass_native",
                make_age_mass_weights(params, context),
                z,
            ),
            (
                "age_mass_float64",
                make_age_mass_weights(params, context, dtype=jnp.float64),
                z.astype(jnp.float64),
            ),
        ):
            trace[name] = floating_trace_dtypes(jax.make_jaxpr(fn)(value))
            if name.endswith("float64") and trace[name] != ["float64"]:
                raise ValueError(
                    f"double age calculation still contains reduced precision: {trace[name]}"
                )
            center, _, records = measure_curve(
                fn,
                value,
                scale,
                normalization,
                budget,
                branch=name,
                labels=[f"ssp_age_{i}" for i in range(ages.size)],
            )
            weight_centers[name] = center.tolist()
            for record in records:
                record["point_index"] = index
            rows.extend(records)
            progress(index, name, rows, results)
        chain = sum(
            derivatives[n]
            for n in ("stellar_native", "igm_native", "projection_native")
        )
        results.append(
            dict(
                point_index=index,
                point_x=np.asarray(point).tolist(),
                theta=np.asarray(physical).tolist(),
                physical_z=float(z),
                dz_dx=dzdx,
                x_stencil_physical_z=[
                    dict(step=h, plus=float(mapped(h)), minus=float(mapped(-h)))
                    for h in STEPS
                ],
                floating_trace_dtypes=trace,
                center_delta_sigma={
                    n: ((v - centers["full_native"]) / sigma).tolist()
                    for n, v in centers.items()
                },
                branch_ad_sigma_per_z={n: v.tolist() for n, v in derivatives.items()},
                chain_minus_full_ad_sigma=(chain - derivatives["full_native"]).tolist(),
                age_mass_centers=weight_centers,
                canonical_center_delta_sigma=None
                if canonical_delta is None
                else canonical_delta.tolist(),
            )
        )
        progress(index, "point_complete", rows, results)
        jax.clear_caches()
    return dict(
        status="REDSHIFT_DECOMPOSITION_COMPLETE",
        points=results,
        coordinate_names=list(spec.names),
        band_names=list(band_names),
        truth_used=False,
        scientific_promotion=False,
        local_optimization_started=False,
        population_training_started=False,
        interpretation="forensic only; no FD value or branch is a certified gradient reference",
        precision_scope="float64 observer projection and age/mass weights only; stored assets unchanged; full production decoder remains mixed precision",
        age_normalization="formed mass in each SSP age bin divided by fixed central target surviving stellar mass; not photometric sigma",
    ), rows
