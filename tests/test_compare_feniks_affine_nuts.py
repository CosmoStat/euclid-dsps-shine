import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.compare_feniks_affine_nuts import affine_from_bank


def test_affine_bank_is_regularized_even_with_one_nonzero_weight():
    points = np.random.default_rng(12).normal(size=(100, 5))
    weights = np.zeros(100)
    weights[3] = 1
    mean, factor, ess = affine_from_bank(points, weights)
    np.testing.assert_allclose(mean, points[3])
    assert ess == 1
    assert np.min(np.linalg.eigvalsh(factor @ factor.T)) > 0
    recovered = np.linalg.solve(factor, (points - mean).T).T @ factor.T + mean
    np.testing.assert_allclose(recovered, points, atol=1e-12)


def test_affine_density_and_gradient_chain_rule():
    points = np.random.default_rng(2).normal(size=(100, 5))
    mean, factor, _ = affine_from_bank(points, np.ones(100))
    mean, factor = jnp.asarray(mean), jnp.asarray(factor)
    q = jnp.ones(5)

    def transformed(q):
        x = mean + factor @ q
        return -0.5 * jnp.sum(x * x) + jnp.log(jnp.diag(factor)).sum()

    value, gradient = jax.value_and_grad(transformed)(q)
    x = mean + factor @ q
    np.testing.assert_allclose(
        value, -0.5 * jnp.sum(x * x) + jnp.linalg.slogdet(factor)[1], rtol=1e-6
    )
    np.testing.assert_allclose(gradient, factor.T @ -x, rtol=1e-6)


def test_invalid_bank_rejected():
    with pytest.raises(ValueError, match="invalid weights"):
        affine_from_bank(np.ones((10, 5)), np.zeros(10))
