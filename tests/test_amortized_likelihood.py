from __future__ import annotations

import jax
import jax.numpy as jnp

from euclid_dsps.amortized.likelihood import photometric_loglike


def test_loglike_max_when_model_matches_observation() -> None:
    obs = jnp.ones((2, 10))
    err = jnp.ones((2, 10)) * 0.1
    mask = jnp.ones((2, 10), dtype=bool)

    matched = photometric_loglike(obs, obs[None, :, :], err, mask)
    shifted = photometric_loglike(obs, obs[None, :, :] + 0.5, err, mask)

    assert jnp.all(matched > shifted)


def test_student_t_penalizes_outlier_less_than_gaussian() -> None:
    obs = jnp.zeros((1, 10))
    model = jnp.zeros((1, 1, 10)).at[:, :, 0].set(10.0)
    err = jnp.ones((1, 10))
    mask = jnp.ones((1, 10), dtype=bool)

    student = photometric_loglike(obs, model, err, mask, likelihood_type="student_t")
    gaussian = photometric_loglike(obs, model, err, mask, likelihood_type="gaussian")

    assert student[0, 0] > gaussian[0, 0]


def test_mask_and_gradients_are_finite() -> None:
    obs = jnp.ones((1, 2))
    err = jnp.ones((1, 2)) * 0.1
    mask = jnp.asarray([[True, False]])

    def objective(model):
        return jnp.sum(photometric_loglike(obs, model, err, mask))

    model = jnp.asarray([[[1.1, 1000.0]]])
    grad = jax.grad(objective)(model)

    assert jnp.all(jnp.isfinite(grad))
    assert grad[0, 0, 1] == 0.0


def test_nonfinite_model_in_observed_band_rejects_particle() -> None:
    obs = jnp.asarray([[1.0, 2.0]], dtype=jnp.float32)
    err = jnp.asarray([[0.1, 0.2]], dtype=jnp.float32)
    mask = jnp.asarray([[True, True]])
    model = jnp.asarray([[[1.0, jnp.nan]]], dtype=jnp.float32)

    loglike = photometric_loglike(
        obs,
        model,
        err,
        mask,
        likelihood_type="gaussian",
        error_floor_frac=0.0,
    )

    assert loglike.shape == (1, 1)
    assert jnp.isneginf(loglike[0, 0])


def test_nonfinite_model_in_masked_band_is_ignored() -> None:
    obs = jnp.asarray([[1.0, 2.0]], dtype=jnp.float32)
    err = jnp.asarray([[0.1, 0.2]], dtype=jnp.float32)
    mask = jnp.asarray([[True, False]])
    model = jnp.asarray([[[1.0, jnp.nan]]], dtype=jnp.float32)

    loglike = photometric_loglike(
        obs,
        model,
        err,
        mask,
        likelihood_type="gaussian",
        error_floor_frac=0.0,
    )

    assert jnp.isfinite(loglike[0, 0])
