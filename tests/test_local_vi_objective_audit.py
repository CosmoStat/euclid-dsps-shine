from __future__ import annotations

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.amortized.local_vi_diagnostic import (
    Budget,
    BudgetExceeded,
    LocalParameters,
    make_step,
    objective_components,
    sample,
)
from euclid_dsps.amortized.local_vi_objective_audit import STEPS, audit_objective
from euclid_dsps.amortized.posterior_target import PosteriorTargetValues


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


class Base(eqx.Module):
    log_std_min: float = -5.0
    log_std_max: float = 3.0


class Layer(eqx.Module):
    shift: jax.Array
    log_scale: jax.Array


class AffineEncoder(eqx.Module):
    base: Base
    layers: tuple

    def forward(self, x, context):
        logdet = jnp.zeros(x.shape[:-1], dtype=x.dtype)
        for layer in self.layers:
            x = x * jnp.exp(layer.log_scale) + layer.shift
            logdet = logdet + jnp.sum(layer.log_scale)
        return x, logdet

    def inverse(self, x, context):
        logdet = jnp.zeros(x.shape[:-1], dtype=x.dtype)
        for layer in reversed(self.layers):
            x = (x - layer.shift) * jnp.exp(-layer.log_scale)
            logdet = logdet - jnp.sum(layer.log_scale)
        return x, logdet


def fixture(dtype=jnp.float64):
    encoder = AffineEncoder(
        Base(), (Layer(jnp.array([0.2, -0.1], dtype), jnp.array([0.1, -0.2], dtype)),)
    )
    parameters = LocalParameters(
        jnp.array([[0.4, -0.3]], dtype), jnp.array([[-0.1, 0.2]], dtype), encoder.layers
    )
    return (
        encoder,
        parameters,
        jnp.zeros((1, 1), dtype),
        jnp.array([[1.0, -1.0]], dtype),
    )


def gaussian_target(x, observation):
    prior = -0.5 * jnp.sum(x**2, axis=-1)
    likelihood = -0.5 * jnp.sum(((x - observation) / 0.7) ** 2, axis=-1)
    return PosteriorTargetValues(
        prior + likelihood, likelihood, prior, jnp.isfinite(prior), x, x
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_complete_objective_audit_has_fixed_noise_and_actual_displacements(dtype):
    encoder, parameters, context, observation = fixture(dtype)
    budget = Budget(120, 4000)
    report = audit_objective(
        encoder, parameters, context, observation, gaussian_target, budget, seed=17
    )
    assert report["status"] == "PASS"
    assert report["variants"]["parameter64"]["status"] == "PASS"
    assert report["noise_dtype"] == np.dtype(dtype).name
    assert report["variants"]["native"]["parameter_dtypes"] == [np.dtype(dtype).name]
    assert report["variants"]["parameter64"]["parameter_dtypes"] == ["float64"]
    assert len(report["rows"]) == 2 * 6 * len(STEPS) * 6
    assert all(row["actual_plus_step"] > 0 for row in report["rows"])
    assert all(row["representable"] for row in report["rows"])
    assert all(
        all(check["identities"].values())
        for variant in report["variants"].values()
        for check in variant["checks"]
    )
    assert budget.forward == 2 * (1 + 6 * 2 * len(STEPS)) * 8
    assert budget.gradient == 2 * 6 * 8
    assert report["scientific_promotion"] is False
    json.dumps(report, allow_nan=False)


def test_fixed_noise_matches_optimizer_objective_exactly():
    encoder, parameters, context, observation = fixture(jnp.float32)
    key, draws = jax.random.PRNGKey(49), 8
    noise = jax.random.normal(key, (draws,) + parameters.mean.shape, dtype=jnp.float32)
    parts, x, logq = objective_components(
        encoder,
        parameters,
        context,
        observation,
        gaussian_target,
        key,
        draws,
        noise=noise,
    )
    direct_x, direct_logq = sample(encoder, parameters, context, key, draws)
    np.testing.assert_array_equal(x, direct_x)
    np.testing.assert_array_equal(logq, direct_logq)
    expected = jnp.mean(direct_logq - gaussian_target(direct_x, observation).logtarget)
    np.testing.assert_array_equal(parts["total"], expected)
    optimizer, step = make_step(encoder, gaussian_target, draws=draws)
    _, _, metrics = step(
        parameters,
        optimizer.init(eqx.filter(parameters, eqx.is_inexact_array)),
        context,
        observation,
        key,
    )
    np.testing.assert_allclose(metrics["negative_elbo"], expected, rtol=1e-6)


def test_real_conditional_flow_objective_audit():
    from test_local_vi_diagnostic import model_fixture

    from euclid_dsps.amortized.local_vi_diagnostic import initialize

    model = model_fixture()
    parameters, context = initialize(model, jnp.ones((1, 6)))
    report = audit_objective(
        model.encoder,
        parameters,
        context,
        jnp.array([[1.0, -1.0]]),
        gaussian_target,
        Budget(120, 4000),
        seed=17,
    )
    assert report["status"] == "PASS"
    assert report["variants"]["parameter64"]["status"] == "PASS"


def test_wrong_target_gradient_fails_even_when_total_decomposition_is_correct():
    encoder, parameters, context, observation = fixture()

    def stopped_target(x, observation):
        return gaussian_target(jax.lax.stop_gradient(x), observation)

    report = audit_objective(
        encoder, parameters, context, observation, stopped_target, Budget(120, 4000)
    )
    assert report["status"] == "FAIL"
    checks = report["variants"]["native"]["checks"]
    assert any(c["components"]["negative_loglike"]["status"] == "FAIL" for c in checks)
    assert all(c["identities"]["decomposition_derivative"] for c in checks)


def test_wrong_inverse_gradient_fails_total_derivative_identity():
    class BrokenInverse(AffineEncoder):
        def inverse(self, x, context):
            base, logdet = super().inverse(x, context)
            return jax.lax.stop_gradient(base), logdet

    encoder, parameters, context, observation = fixture()
    encoder = BrokenInverse(encoder.base, encoder.layers)
    report = audit_objective(
        encoder, parameters, context, observation, gaussian_target, Budget(120, 4000)
    )
    assert report["status"] == "FAIL"
    assert any(
        not check["identities"]["sample_inverse_total_derivative"]
        for check in report["variants"]["native"]["checks"]
    )


def test_audit_exercises_custom_reverse_mode_rules():
    encoder, parameters, context, observation = fixture()

    @jax.custom_vjp
    def broken_identity(x):
        return x

    broken_identity.defvjp(lambda x: (x, None), lambda _, g: (2 * g,))

    def target(x, observation):
        return gaussian_target(broken_identity(x), observation)

    report = audit_objective(
        encoder, parameters, context, observation, target, Budget(120, 4000)
    )
    assert report["status"] == "FAIL"
    assert any(
        c["components"]["total"]["status"] == "FAIL"
        for c in report["variants"]["native"]["checks"]
    )


def test_budget_refuses_before_first_decoder_call():
    encoder, parameters, context, observation = fixture()

    def forbidden(*args):
        pytest.fail("decoder called after budget exhausted")

    budget = Budget(120, 7)
    with pytest.raises(BudgetExceeded):
        audit_objective(
            encoder, parameters, context, observation, forbidden, budget, draws=8
        )
    assert budget.forward == budget.gradient == 0


def test_unresolved_native_arithmetic_is_not_promoted_by_parameter64_pass():
    encoder, parameters, context, observation = fixture(jnp.float32)

    def unresolved_target(x, observation):
        prior = jnp.asarray(1e8, dtype=x.dtype) + jnp.sum(x, axis=-1)
        likelihood = jnp.zeros_like(prior)
        return PosteriorTargetValues(
            prior, likelihood, prior, jnp.isfinite(prior), x, x
        )

    report = audit_objective(
        encoder, parameters, context, observation, unresolved_target, Budget(120, 4000)
    )
    assert report["status"] == "INCONCLUSIVE"
    assert report["variants"]["parameter64"]["status"] == "PASS"
    assert any(
        c["components"]["negative_logprior"]["selected_step_index"] is None
        for c in report["variants"]["native"]["checks"]
    )
