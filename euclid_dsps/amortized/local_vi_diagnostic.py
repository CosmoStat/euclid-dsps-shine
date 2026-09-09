"""Direct-draw, fixed-context local VI for diagnostic use, never a teacher gate."""

from __future__ import annotations

import time
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .encoder import _diag_normal_log_prob
from .npe_validation import (
    summarize_model_generated_rank_calibration,
    summarize_projected_rank_calibration,
)
from .posterior import ConditionalFlowEncoder, posterior_encoder_state


class LocalParameters(eqx.Module):
    mean: jax.Array
    log_std: jax.Array
    layers: tuple


class MixtureParameters(eqx.Module):
    local: LocalParameters
    anchor: LocalParameters


def broaden(encoder, parameters, factor):
    """Scale the effective base std, rejecting saturation instead of clipping."""
    effective = jnp.clip(
        parameters.log_std, encoder.base.log_std_min, encoder.base.log_std_max
    ) + np.log(factor)
    if (
        factor < 1
        or not np.isfinite(factor)
        or np.any(np.asarray(effective) > encoder.base.log_std_max)
    ):
        raise ValueError("requested broadening exceeds the base scale contract")
    return eqx.tree_at(lambda p: p.log_std, parameters, effective)


def initialize(model, features):
    encoder = model.encoder
    if not isinstance(encoder, ConditionalFlowEncoder) or (
        encoder.output_space != "latent_x" or encoder.base_components != 1
    ):
        raise ValueError("local diagnostic requires a one-component latent_x flow")
    if encoder.context_encoder_type != "residual_photometry":
        raise ValueError("fixed context requires a residual photometry encoder")
    state = posterior_encoder_state(model, features)
    return (
        LocalParameters(state.mean, state.log_std, encoder.layers),
        state.flow_context,
    )


def sample(encoder, parameters, context, key, draws):
    if isinstance(parameters, MixtureParameters):
        left, right, choose = jax.random.split(key, 3)
        a, _ = sample(encoder, parameters.local, context, left, draws)
        b, _ = sample(encoder, parameters.anchor, context, right, draws)
        select = jax.random.bernoulli(choose, 0.5, a.shape[:-1] + (1,))
        x = jnp.where(select, a, b)
        return x, log_prob(encoder, parameters, context, x)
    log_std = jnp.clip(
        parameters.log_std, encoder.base.log_std_min, encoder.base.log_std_max
    )
    eps = jax.random.normal(
        key, (int(draws),) + parameters.mean.shape, dtype=parameters.mean.dtype
    )
    base = parameters.mean + jnp.exp(log_std) * eps
    local_encoder = eqx.tree_at(lambda e: e.layers, encoder, parameters.layers)
    x, logdet = local_encoder.forward(base, context)
    return x, _diag_normal_log_prob(base, parameters.mean, log_std) - logdet


def log_prob(encoder, parameters, context, x):
    if isinstance(parameters, MixtureParameters):
        return jnp.logaddexp(
            log_prob(encoder, parameters.local, context, x),
            log_prob(encoder, parameters.anchor, context, x),
        ) - np.log(2.0)
    local_encoder = eqx.tree_at(lambda e: e.layers, encoder, parameters.layers)
    base, inverse_logdet = local_encoder.inverse(x, context)
    log_std = jnp.clip(
        parameters.log_std, encoder.base.log_std_min, encoder.base.log_std_max
    )
    return _diag_normal_log_prob(base, parameters.mean, log_std) + inverse_logdet


def perturb(parameters, key, scale=0.05):
    """Second start perturbs base location in its own standard-deviation units."""
    mean = parameters.mean + float(scale) * jnp.exp(
        parameters.log_std
    ) * jax.random.normal(key, parameters.mean.shape, dtype=parameters.mean.dtype)
    return eqx.tree_at(lambda p: p.mean, parameters, mean)


def make_step(encoder, target: Callable, *, draws=4, learning_rate=1e-3, clip=5.0):
    """Only LocalParameters are differentiated; frozen target retains x gradients."""
    optimizer = optax.chain(optax.clip_by_global_norm(clip), optax.adam(learning_rate))

    def loss(parameters, context, observation, key):
        x, logq = sample(encoder, parameters, context, key, draws)
        values = target(x, observation)
        finite = jnp.all(jnp.isfinite(logq) & jnp.isfinite(values.logtarget))
        value = jnp.mean(logq - values.logtarget)
        return jnp.where(finite, value, jnp.inf), (
            jnp.mean(logq),
            jnp.mean(values.logprior),
            jnp.mean(values.loglike),
        )

    @eqx.filter_jit
    def step(parameters, state, context, observation, key):
        (value, parts), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            parameters, context, observation, key
        )
        updates, new_state = optimizer.update(grads, state, parameters)
        new_parameters = eqx.apply_updates(parameters, updates)
        leaves = jax.tree_util.tree_leaves(
            eqx.filter(new_parameters, eqx.is_inexact_array)
        )
        finite = jnp.isfinite(value) & jnp.isfinite(optax.global_norm(grads))
        finite &= jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
        metrics = dict(
            negative_elbo=value,
            logq=parts[0],
            logprior=parts[1],
            loglike=parts[2],
            raw_grad_norm=optax.global_norm(grads),
            update_norm=optax.global_norm(updates),
            finite=finite,
        )
        # The caller must reject nonfinite updates; never certify skipped draws.
        return new_parameters, new_state, metrics

    return optimizer, step


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds: float, evaluations: int):
        self.started = time.monotonic()
        self.seconds = float(seconds)
        self.maximum_evaluations = int(evaluations)
        self.forward = 0
        self.gradient = 0

    def charge(self, count: int, *, gradient=False):
        if time.monotonic() - self.started >= self.seconds:
            raise BudgetExceeded("wall-clock budget exhausted")
        if self.forward + self.gradient + count > self.maximum_evaluations:
            raise BudgetExceeded("decoder evaluation budget exhausted")
        if gradient:
            self.gradient += count
        else:
            self.forward += count

    def snapshot(self):
        return dict(
            elapsed_seconds=time.monotonic() - self.started,
            maximum_seconds=self.seconds,
            maximum_evaluations=self.maximum_evaluations,
            forward_evaluations=self.forward,
            gradient_evaluations=self.gradient,
            contract="gradient evaluations include backward work; not forward-equivalent cost",
        )


def analytic_controls():
    """Positive and negative Gaussian controls, with a data-dependent statistic."""
    rng = np.random.default_rng(260908)
    n, k, d = 1024, 128, 2
    theta = rng.normal(size=(n, d))
    y = theta + 0.1 * rng.normal(size=(n, d))
    epsilon = rng.normal(size=(k, n, d))
    result = {}
    for name, q in (
        ("ignores_data", epsilon),
        ("exact_posterior", y[None] / 1.01 + np.sqrt(0.01 / 1.01) * epsilon),
    ):
        kwargs = dict(seed=1, maximum_ks=0.06, maximum_coverage_ece=0.06)
        marginal = summarize_model_generated_rank_calibration(
            q, theta, parameter_names=("x0", "x1"), **kwargs
        )
        joint = summarize_projected_rank_calibration(
            q, theta, scale=np.ones(d), **kwargs
        )
        generated_loglike = -0.5 * np.sum(((theta - y) / 0.1) ** 2, axis=-1)
        proposed_loglike = -0.5 * np.sum(((q - y[None]) / 0.1) ** 2, axis=-1)
        likelihood = summarize_model_generated_rank_calibration(
            proposed_loglike[..., None],
            generated_loglike[..., None],
            parameter_names=("loglike",),
            **kwargs,
        )
        result[name] = {
            key: {"status": value["status"], "ks": value["maximum_coordinate_pit_ks"]}
            for key, value in (
                ("marginal", marginal),
                ("projection", joint),
                ("loglike", likelihood),
            )
        }
    if result["ignores_data"]["loglike"]["status"] != "FAIL" or any(
        value["status"] != "PASS" for value in result["exact_posterior"].values()
    ):
        raise ValueError("analytical calibration controls failed")
    return result


def assert_close(name, actual, expected, *, atol=5e-4, rtol=5e-4):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (
        actual.shape != expected.shape
        or not np.all(np.isfinite(actual))
        or not np.all(np.isfinite(expected))
    ):
        raise ValueError(f"{name}: nonfinite or shape mismatch")
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(
            f"{name}: mismatch, max absolute delta={np.max(np.abs(actual - expected))}"
        )
    return float(np.max(np.abs(actual - expected)))


def gradient_audit(objective, point, direction, budget, *, atol=0.1, rtol=0.05):
    """Check a scalar-component dict using AD-independent FD plateau selection.

    ULP estimates are a resolution screen, not rigorous propagated error bounds.
    They reject poorly resolved comparisons rather than enlarge tolerances.
    """
    steps = (0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625)
    budget.charge(1)
    center = jax.device_get(objective(point))
    names = tuple(center)
    budget.charge(len(names), gradient=True)
    jacobian = jax.jacrev(objective)(point)
    automatic = {name: float(jnp.sum(jacobian[name] * direction)) for name in names}
    values = []
    for h in steps:
        budget.charge(2)
        plus = jax.device_get(objective(point + h * direction))
        minus = jax.device_get(objective(point - h * direction))
        row = {}
        for name in names:
            fp, fm = np.asarray(plus[name]), np.asarray(minus[name])
            ulp_sum = abs(float(np.spacing(fp))) + abs(float(np.spacing(fm)))
            # The total may be promoted to float64 after summing float32 terms.
            if name == "logtarget" and {"loglike", "logprior"} <= set(names):
                ulp_sum = max(
                    ulp_sum,
                    sum(
                        abs(float(np.spacing(np.asarray(side[part]))))
                        for side in (plus, minus)
                        for part in ("loglike", "logprior")
                    ),
                )
            row[name] = dict(
                plus=float(fp),
                minus=float(fm),
                central_difference=(float(fp) - float(fm)) / (2 * h),
                resolution_screen=4 * ulp_sum / (2 * h),
            )
        values.append(row)
    components = {}
    for name in names:
        samples = [
            dict(step=h, **row[name]) for h, row in zip(steps, values, strict=True)
        ]
        stable_windows = []
        for end in range(2, len(samples)):
            window = samples[end - 2 : end + 1]
            derivatives = np.array([x["central_difference"] for x in window])
            tolerance = atol + rtol * abs(derivatives[-1])
            resolved = all(
                np.isfinite(x["resolution_screen"])
                and x["resolution_screen"] <= tolerance
                for x in window
            )
            if (
                np.all(np.isfinite(derivatives))
                and resolved
                and np.ptp(derivatives) <= tolerance
            ):
                stable_windows.append(end)
        selected = stable_windows[-1] if stable_windows else None
        ad = automatic[name]
        status = "INCONCLUSIVE"
        if not np.isfinite(ad) or not np.isfinite(float(center[name])):
            status = "FAIL"
        elif selected is not None:
            fd = samples[selected]["central_difference"]
            status = "PASS" if abs(ad - fd) <= atol + rtol * abs(fd) else "FAIL"
        components[name] = dict(
            status=status,
            center=float(center[name]),
            dtype=str(np.asarray(center[name]).dtype),
            autodiff=ad,
            samples=samples,
            selected_step=steps[selected] if selected is not None else None,
            stable_window_end_indices=stable_windows,
        )
    statuses = [value["status"] for value in components.values()]
    return dict(
        status=(
            "FAIL"
            if "FAIL" in statuses
            else ("PASS" if all(s == "PASS" for s in statuses) else "INCONCLUSIVE")
        ),
        components=components,
        atol=atol,
        rtol=rtol,
        point=np.asarray(point).tolist(),
        direction=np.asarray(direction).tolist(),
        selection="finest resolved three-step FD plateau, chosen without consulting AD",
        resolution_interpretation="ULP screen only; not a bound on internal decoder roundoff",
        truth_used=False,
        scientific_promotion=False,
    )
