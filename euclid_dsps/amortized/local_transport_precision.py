"""Diagnostic-only float64 conditional transport; frozen target is unchanged."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .local_vi_diagnostic import sample
from .local_vi_objective_audit import STEPS, _direction, _shift, audit_objective
from .posterior import _ConditionalCoupling


class DiagnosticTransport64(eqx.Module):
    base: object
    layers: tuple
    permutations: tuple
    inverse_permutations: tuple

    def __init__(self, encoder):
        if not jax.config.x64_enabled:
            raise ValueError("transport precision diagnostic requires x64")
        if not all(isinstance(layer, _ConditionalCoupling) for layer in encoder.layers):
            raise ValueError(
                "diagnostic transport supports conditional coupling layers only"
            )
        self.base = encoder.base
        self.layers = promote(encoder.layers)
        self.permutations = encoder.permutations
        self.inverse_permutations = encoder.inverse_permutations

    def forward(self, value, context):
        value = jnp.asarray(value, jnp.float64)
        logdet = jnp.zeros(value.shape[:-1], jnp.float64)
        for layer, permutation in zip(self.layers, self.permutations, strict=True):
            value, delta = layer._transform(
                value, context, inverse=False, preserve_dtype=True
            )
            value = jnp.take(value, permutation, axis=-1)
            logdet += delta
        return value, logdet

    def inverse(self, value, context):
        value = jnp.asarray(value, jnp.float64)
        logdet = jnp.zeros(value.shape[:-1], jnp.float64)
        for layer, permutation in zip(
            reversed(self.layers), reversed(self.inverse_permutations), strict=True
        ):
            value = jnp.take(value, permutation, axis=-1)
            value, delta = layer._transform(
                value, context, inverse=True, preserve_dtype=True
            )
            logdet += delta
        return value, logdet


def promote(tree):
    return jax.tree.map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, tree
    )


@eqx.filter_jit
def _values(encoder, parameters, context, observation, target, key, draws, noise):
    x, logq = sample(encoder, parameters, context, key, draws, noise=noise)
    result = target(x, observation)
    return x, logq, result.model_flux, result.loglike


def compare_transport(
    encoder, parameters, context, observation, target, budget, *, seed, draws
):
    """Same native noise and directions; no optimizer or automatic qualification."""
    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(
        key, (draws,) + parameters.mean.shape, dtype=parameters.mean.dtype
    )
    diagnostic = DiagnosticTransport64(encoder)
    promoted = promote(parameters)
    audit_directions = [
        (
            block,
            i,
            _direction(parameters, block, np.random.default_rng(seed + 10 * j + i)),
        )
        for j, block in enumerate(("mean", "log_std", "layers"))
        for i in range(2)
    ]
    report = audit_objective(
        diagnostic,
        promoted,
        context,
        observation,
        target,
        budget,
        seed=seed,
        draws=draws,
        noise=noise,
        directions=audit_directions,
    )
    traces, centers, arrays = [], {}, {}
    directions = [
        _direction(parameters, "log_std", np.random.default_rng(seed + 10 + i))
        for i in range(2)
    ]
    count = draws * parameters.mean.shape[0]
    for variant, enc, p in (
        ("native", encoder, parameters),
        ("transport64", diagnostic, promoted),
    ):

        def values(q, enc=enc):
            budget.charge(count)
            return tuple(
                np.asarray(x)
                for x in jax.device_get(
                    _values(enc, q, context, observation, target, key, draws, noise)
                )
            )

        center = values(p)
        centers[variant] = center
        for field, value in zip(("x", "logq", "flux", "loglike"), center, strict=True):
            arrays[f"{variant}_center_{field}"] = value
        dynamic, static = eqx.partition(p, eqx.is_inexact_array)
        for index, direction in enumerate(directions):
            direction = jax.tree.map(lambda d, a: d.astype(a.dtype), direction, dynamic)
            for step_index, step in enumerate(STEPS):
                plus = values(eqx.combine(_shift(dynamic, direction, step), static))
                minus = values(eqx.combine(_shift(dynamic, direction, -step), static))
                for side, values_at_step in (("plus", plus), ("minus", minus)):
                    for field, value in zip(
                        ("x", "logq", "flux", "loglike"), values_at_step, strict=True
                    ):
                        arrays[
                            f"{variant}_direction_{index}_step_{step_index}_{side}_{field}"
                        ] = value
                for draw in range(draws):
                    dx = plus[0][draw].astype(float) - minus[0][draw].astype(float)
                    df = plus[2][draw].astype(float) - minus[2][draw].astype(float)
                    traces.append(
                        dict(
                            variant=variant,
                            direction=index,
                            step=step,
                            draw=draw,
                            x_dtype=str(center[0].dtype),
                            flux_dtype=str(center[2].dtype),
                            latent_delta_max=float(np.max(np.abs(dx))),
                            latent_unchanged_fraction=float(np.mean(dx == 0)),
                            flux_delta_max=float(np.max(np.abs(df))),
                            flux_unchanged_fraction=float(np.mean(df == 0)),
                            negative_loglike_fd=float(
                                np.mean(minus[3][draw] - plus[3][draw]) / (2 * step)
                            ),
                        )
                    )
    return dict(
        transport64_audit=report,
        traces=traces,
        arrays=arrays,
        center_max_abs_delta={
            name: float(
                np.max(
                    np.abs(
                        centers["transport64"][i].astype(float)
                        - centers["native"][i].astype(float)
                    )
                )
            )
            for i, name in enumerate(("x", "logq", "flux", "loglike"))
        },
        dtypes={
            v: dict(
                zip(
                    ("x", "logq", "flux", "loglike"),
                    (str(x.dtype) for x in c),
                    strict=True,
                )
            )
            for v, c in centers.items()
        },
        contract="Float64 conditional transport only; identical frozen context/noise/target. Internal target precision is not changed. No optimization, migration or promotion.",
        scientific_promotion=False,
    )
