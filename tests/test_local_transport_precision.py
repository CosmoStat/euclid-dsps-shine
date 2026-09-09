import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.amortized.local_transport_precision import (
    DiagnosticTransport64,
    compare_transport,
    promote,
)
from euclid_dsps.amortized.local_vi_diagnostic import (
    Budget,
    LocalParameters,
    log_prob,
    sample,
)
from euclid_dsps.amortized.posterior import ConditionalFlowEncoder
from euclid_dsps.amortized.posterior_target import PosteriorTargetValues


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def fixture(family):
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(8),
        input_dim=6,
        latent_dim=2,
        hidden_sizes=(8,),
        activation="gelu",
        log_std_min=-4,
        log_std_max=2.5,
        initial_log_std=0,
        n_layers=2,
        hidden_size=8,
        output_space="latent_x",
        family=family,
    )
    parameters = LocalParameters(
        jnp.zeros((1, 2), jnp.float32), jnp.zeros((1, 2), jnp.float32), encoder.layers
    )
    return encoder, parameters, jnp.zeros((1, 4), jnp.float32)


@pytest.mark.parametrize("family", ["realnvp", "rq_spline"])
def test_transport64_really_preserves_precision_and_inverse(family):
    encoder, p, context = fixture(family)
    before = jax.tree.leaves(eqx.filter(encoder, eqx.is_array))
    diagnostic, promoted = DiagnosticTransport64(encoder), promote(p)
    noise = jax.random.normal(jax.random.PRNGKey(7), (8, 1, 2), jnp.float32)
    native = sample(encoder, p, context, jax.random.PRNGKey(7), 8, noise=noise)
    x, lq = sample(diagnostic, promoted, context, jax.random.PRNGKey(7), 8, noise=noise)
    assert x.dtype == lq.dtype == jnp.float64
    np.testing.assert_allclose(x, native[0], atol=2e-6)
    np.testing.assert_allclose(
        lq, log_prob(diagnostic, promoted, context, x), atol=1e-10
    )
    # A sub-float32 perturbation must survive the diagnostic path.
    u = jnp.ones((1, 2), jnp.float64)
    nx, _ = encoder.forward(u, context)
    ny, _ = encoder.forward(u + 1e-10, context)
    np.testing.assert_array_equal(nx, ny)
    dx, _ = diagnostic.forward(u, context)
    dy, _ = diagnostic.forward(u + 1e-10, context)
    assert float(jnp.max(jnp.abs(dx - dy))) > 1e-12
    for old, new in zip(
        before, jax.tree.leaves(eqx.filter(encoder, eqx.is_array)), strict=True
    ):
        np.testing.assert_array_equal(old, new)


def test_comparison_shared_noise_trace_and_decoder_budget():
    encoder, p, context = fixture("realnvp")

    def target(x, obs):
        prior = -jnp.sum(x * x, axis=-1) / 2
        like = -jnp.sum((x - obs) ** 2, axis=-1) / 2
        return PosteriorTargetValues(
            prior + like, like, prior, jnp.isfinite(prior), x, x
        )

    budget = Budget(120, 4000)
    report = compare_transport(
        encoder, p, context, jnp.ones((1, 2)), target, budget, seed=17, draws=4
    )
    assert report["transport64_audit"]["status"] == "PASS"
    assert report["transport64_audit"]["noise_dtype"] == "float32"
    assert report["dtypes"]["transport64"]["x"] == "float64"
    assert len(report["traces"]) == 2 * 2 * 8 * 4
    assert len(report["arrays"]) == 2 * (1 + 2 * 8 * 2) * 4
    assert report["arrays"]["transport64_center_x"].shape == (4, 1, 2)
    assert budget.forward == (194 + 66) * 4
    assert budget.gradient == 12 * 4
    assert report["scientific_promotion"] is False
