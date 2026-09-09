"""Fixed-noise parameter derivatives of local VI, not a posterior quality gate."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .local_vi_diagnostic import Budget, LocalParameters, log_prob, objective_components

STEPS = (0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125)
ATOL = 0.01
RTOL = 0.01
IDENTITY_ATOL = 5e-4
IDENTITY_RTOL = 5e-4


def _components(parameters, encoder, context, observation, target, key, draws, noise):
    parts, x, sampled_logq = objective_components(
        encoder, parameters, context, observation, target, key, draws, noise=noise
    )
    inverse_logq = log_prob(encoder, parameters, context, x)
    return {
        **parts,
        "inverse_logq": jnp.mean(inverse_logq),
        "density_gap": jnp.mean(inverse_logq - sampled_logq),
    }


_evaluate = eqx.filter_jit(_components)


@eqx.filter_jit
def _differentiate(
    parameters, encoder, context, observation, target, key, draws, noise
):
    dynamic, static = eqx.partition(parameters, eqx.is_inexact_array)

    def components(p):
        return _components(
            eqx.combine(p, static),
            encoder,
            context,
            observation,
            target,
            key,
            draws,
            noise,
        )

    return jax.jacrev(components)(dynamic)


def _aggregate(statuses):
    if "FAIL" in statuses:
        return "FAIL"
    return "PASS" if statuses and all(s == "PASS" for s in statuses) else "INCONCLUSIVE"


def _direction(parameters, block, rng):
    dynamic = eqx.filter(parameters, eqx.is_inexact_array)
    zero = jax.tree.map(jnp.zeros_like, dynamic)
    selected = getattr(dynamic, block)
    leaves, structure = jax.tree.flatten(selected)
    if not leaves:
        raise ValueError(f"no differentiable parameters in block {block}")
    raw = [rng.normal(size=leaf.shape) for leaf in leaves]
    norm = np.sqrt(sum(float(np.sum(x * x)) for x in raw))
    values = [
        jnp.asarray(x / norm, dtype=p.dtype) for x, p in zip(raw, leaves, strict=True)
    ]
    return eqx.tree_at(
        lambda p: getattr(p, block), zero, jax.tree.unflatten(structure, values)
    )


def _shift(parameters, direction, step):
    return jax.tree.map(
        lambda p, d: p + jnp.asarray(step, dtype=p.dtype) * d,
        parameters,
        direction,
    )


@jax.jit
def _displacement(parameters, shifted, direction):
    triples = zip(
        jax.tree.leaves(parameters),
        jax.tree.leaves(shifted),
        jax.tree.leaves(direction),
        strict=True,
    )
    dot = square = norm = collapsed = active = jnp.asarray(0.0, dtype=jnp.float64)
    for p, q, d in triples:
        delta = q.astype(jnp.float64) - p.astype(jnp.float64)
        d = d.astype(jnp.float64)
        dot += jnp.sum(delta * d)
        square += jnp.sum(delta * delta)
        norm += jnp.sum(d * d)
        active += jnp.count_nonzero(d)
        collapsed += jnp.count_nonzero((d != 0) & (delta == 0))
    projected = dot / norm
    orthogonal = jnp.sqrt(jnp.maximum(0.0, square - dot * dot / norm))
    return (
        projected,
        orthogonal / jnp.maximum(jnp.sqrt(square), 1e-300),
        collapsed / active,
    )


@jax.jit
def _project_gradient(gradient, direction):
    return sum(
        jnp.sum(g.astype(jnp.float64) * d.astype(jnp.float64))
        for g, d in zip(
            jax.tree.leaves(gradient), jax.tree.leaves(direction), strict=True
        )
    )


def _identity(a, b):
    return bool(
        np.isfinite(a)
        and np.isfinite(b)
        and abs(a - b) <= IDENTITY_ATOL + IDENTITY_RTOL * abs(b)
    )


def _classify(samples, automatic, center):
    windows = []
    for end in range(2, len(samples)):
        window = samples[end - 2 : end + 1]
        fd = np.asarray([x["fd"] for x in window])
        tolerance = ATOL + RTOL * abs(fd[-1])
        if (
            np.all(np.isfinite(fd))
            and all(x["representable"] for x in window)
            and all(x["resolution_screen"] <= tolerance for x in window)
            and np.ptp(fd) <= tolerance
        ):
            windows.append(end)
    selected = windows[-1] if windows else None
    status = "INCONCLUSIVE"
    if not np.isfinite(automatic) or not np.isfinite(center):
        status = "FAIL"
    elif selected is not None:
        fd = samples[selected]["fd"]
        status = "PASS" if abs(automatic - fd) <= ATOL + RTOL * abs(fd) else "FAIL"
    return dict(
        status=status,
        ad=automatic,
        center=center,
        fd=samples[selected]["fd"] if selected is not None else None,
        selected_step_index=selected,
        stable_window_end_indices=windows,
    )


def audit_objective(
    encoder,
    parameters: LocalParameters,
    context,
    observation,
    target: Callable,
    budget: Budget,
    *,
    seed: int = 260912,
    draws: int = 8,
) -> dict:
    """Audit six fixed directions in native and parameter-promoted arithmetic.

    Target and context stay frozen, and noise is generated once in native dtype.
    Parameter64 is explanatory only: casts inside the encoder/target remain as
    implemented. Only the native variant contributes to the returned status.
    Decoder budget units include every draw/object and each reverse-mode seed.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("the objective audit requires JAX_ENABLE_X64=true")
    if not isinstance(parameters, LocalParameters) or draws < 1:
        raise ValueError("objective audit requires LocalParameters and positive draws")
    count = int(draws) * int(np.prod(parameters.mean.shape[:-1]))
    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(
        key, (int(draws),) + parameters.mean.shape, dtype=parameters.mean.dtype
    )
    native = eqx.filter(parameters, eqx.is_inexact_array)
    promoted = jax.tree.map(lambda x: x.astype(jnp.float64), native)
    _, static = eqx.partition(parameters, eqx.is_inexact_array)
    directions = [
        (
            block,
            i,
            _direction(parameters, block, np.random.default_rng(seed + 10 * j + i)),
        )
        for j, block in enumerate(("mean", "log_std", "layers"))
        for i in range(2)
    ]
    variants, rows = {}, []
    for variant, dynamic in (("native", native), ("parameter64", promoted)):

        def evaluate(p):
            return _evaluate(
                eqx.combine(p, static),
                encoder,
                context,
                observation,
                target,
                key,
                draws,
                noise,
            )

        budget.charge(count)
        center = jax.device_get(evaluate(dynamic))
        # Reverse-mode derivatives match make_step; forward JVP alone would not
        # exercise a target's custom VJP or the optimizer's adjoint arithmetic.
        budget.charge(count * len(center), gradient=True)
        jacobian = _differentiate(
            eqx.combine(dynamic, static),
            encoder,
            context,
            observation,
            target,
            key,
            draws,
            noise,
        )
        checks = []
        for block, index, native_direction in directions:
            direction = jax.tree.map(
                lambda d, p: d.astype(p.dtype), native_direction, dynamic
            )
            derivative = {
                name: float(_project_gradient(jacobian[name], direction))
                for name in center
            }
            samples = {name: [] for name in center}
            for step in STEPS:
                plus_parameters = _shift(dynamic, direction, step)
                minus_parameters = _shift(dynamic, direction, -step)
                sp, ep, cp = map(
                    float, _displacement(dynamic, plus_parameters, direction)
                )
                sm, em, cm = map(
                    float, _displacement(dynamic, minus_parameters, direction)
                )
                sm = -sm
                representable = bool(
                    sp > 0
                    and sm > 0
                    and max(ep, em, abs(sp / step - 1), abs(sm / step - 1)) <= 0.01
                )
                budget.charge(count)
                plus = jax.device_get(evaluate(plus_parameters))
                budget.charge(count)
                minus = jax.device_get(evaluate(minus_parameters))
                wp = sm / (sp * (sp + sm)) if sp > 0 and sm > 0 else np.nan
                wm = -sp / (sm * (sp + sm)) if sp > 0 and sm > 0 else np.nan
                for name in center:
                    fp, fm, f0 = map(float, (plus[name], minus[name], center[name]))
                    resolution = 4 * (
                        abs(wp) * abs(float(np.spacing(plus[name])))
                        + abs(wm) * abs(float(np.spacing(minus[name])))
                        + abs(wp + wm) * abs(float(np.spacing(center[name])))
                    )
                    row = dict(
                        variant=variant,
                        block=block,
                        direction=index,
                        component=name,
                        step=step,
                        actual_plus_step=sp,
                        actual_minus_step=sm,
                        plus_orthogonal_fraction=ep,
                        minus_orthogonal_fraction=em,
                        plus_collapsed_fraction=cp,
                        minus_collapsed_fraction=cm,
                        representable=representable,
                        ad=float(derivative[name]),
                        plus=fp,
                        minus=fm,
                        fd=wp * (fp - f0) + wm * (fm - f0),
                        resolution_screen=resolution,
                    )
                    samples[name].append(row)
                    rows.append(row)
            identities = dict(
                sample_inverse_value=_identity(float(center["density_gap"]), 0.0),
                sample_inverse_total_derivative=_identity(
                    float(derivative["inverse_logq"]), float(derivative["logq"])
                ),
                decomposition_value=_identity(
                    float(center["total"]),
                    sum(
                        float(center[k])
                        for k in ("logq", "negative_logprior", "negative_loglike")
                    ),
                ),
                decomposition_derivative=_identity(
                    float(derivative["total"]),
                    sum(
                        float(derivative[k])
                        for k in ("logq", "negative_logprior", "negative_loglike")
                    ),
                ),
            )
            component_checks = {
                name: _classify(
                    samples[name], float(derivative[name]), float(center[name])
                )
                for name in center
            }
            statuses = [c["status"] for c in component_checks.values()]
            statuses.append("PASS" if all(identities.values()) else "FAIL")
            checks.append(
                dict(
                    block=block,
                    direction=index,
                    status=_aggregate(statuses),
                    components=component_checks,
                    identities=identities,
                )
            )
        variants[variant] = dict(
            status=_aggregate([c["status"] for c in checks]),
            parameter_dtypes=sorted({str(p.dtype) for p in jax.tree.leaves(dynamic)}),
            output_dtypes={
                name: str(np.asarray(value).dtype) for name, value in center.items()
            },
            checks=checks,
        )
    return dict(
        status=variants["native"]["status"],
        variants=variants,
        rows=rows,
        steps=list(STEPS),
        atol=ATOL,
        rtol=RTOL,
        identity_atol=IDENTITY_ATOL,
        identity_rtol=IDENTITY_RTOL,
        draws=int(draws),
        decoder_draws_per_call=count,
        seed=int(seed),
        noise_sha256=hashlib.sha256(np.asarray(noise).tobytes()).hexdigest(),
        noise_dtype=str(noise.dtype),
        base_scale_clip_state=dict(
            lower=float(encoder.base.log_std_min),
            upper=float(encoder.base.log_std_max),
            at_or_below_lower=int(
                np.count_nonzero(
                    np.asarray(parameters.log_std) <= encoder.base.log_std_min
                )
            ),
            at_or_above_upper=int(
                np.count_nonzero(
                    np.asarray(parameters.log_std) >= encoder.base.log_std_max
                )
            ),
        ),
        selection="finest resolved three-step FD plateau, selected without AD",
        direction_contract="two seeded unit Euclidean directions per parameter block",
        resolution_contract="output ULP and actual parameter displacement screens, not a bound on internal roundoff",
        precision_contract="native status is mandatory; parameter64 copies do not remove internal casts or qualify production arithmetic",
        truth_used=False,
        scientific_promotion=False,
    )
