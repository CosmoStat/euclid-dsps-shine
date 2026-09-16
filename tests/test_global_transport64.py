import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.amortized.local_transport_precision import DiagnosticTransport64
from euclid_dsps.amortized.posterior import ConditionalFlowEncoder


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def test_global_transport_matches_diagnostic():
    e = ConditionalFlowEncoder(
        jax.random.PRNGKey(8),
        input_dim=4,
        latent_dim=2,
        hidden_sizes=(8,),
        activation="gelu",
        log_std_min=-4.0,
        log_std_max=2.5,
        initial_log_std=0.25,
        family="realnvp",
        n_layers=2,
        hidden_size=8,
        init_scale=0.1,
        output_space="latent_x",
        transport_float64=True,
    )
    ref = DiagnosticTransport64(e)
    x = jnp.asarray([[0.2, -0.7]], dtype=jnp.float64)
    c = jnp.zeros((1, e.context_dim))
    y, ld = e.forward(x, c)
    yr, ldr = ref.forward(x, c)
    np.testing.assert_array_equal(y, yr)
    np.testing.assert_array_equal(ld, ldr)
    back, ild = e.inverse(y, c)
    np.testing.assert_allclose(back, x, atol=1e-12)
    np.testing.assert_allclose(ld + ild, 0.0, atol=1e-12)
    assert y.dtype == jnp.float64
