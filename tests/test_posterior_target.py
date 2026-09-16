from __future__ import annotations

import inspect
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import euclid_dsps.amortized.map_adam as map_module
import euclid_dsps.amortized.train as train_module
import scripts.run_feniks_exact_posterior_benchmark as exact_module
from euclid_dsps.amortized.flows import StandardNormalPrior
from euclid_dsps.amortized.posterior_target import (
    PosteriorObservation,
    physical_bounds_diagnostics,
    posterior_log_target,
    posterior_log_target_from_model_flux,
)
from euclid_dsps.amortized.train import JitLatentSpec, LossBatch
from euclid_dsps.calibration import GlobalSedScaleState


def _latent_spec() -> JitLatentSpec:
    return JitLatentSpec(
        names=("a", "b"),
        lower=jnp.asarray([-1.0, -1.0]),
        upper=jnp.asarray([1.0, 1.0]),
        raw_center=jnp.zeros(2),
        raw_scale=jnp.ones(2),
        normalization="spline15d_mixed",
        transform_family=jnp.zeros(2, dtype=jnp.int32),
        transform_location=jnp.zeros(2),
        transform_lambda=jnp.ones(2),
    )


def _model():
    return SimpleNamespace(
        prior=StandardNormalPrior(latent_dim=2),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
        band_calibration=None,
    )


def _model_flux(x, *_args, **_kwargs):
    return jnp.stack((x[..., 0] + 0.2, x[..., 1] - 0.1), axis=-1)


def _observation() -> PosteriorObservation:
    return PosteriorObservation(
        flux=jnp.asarray([[0.25, -0.15]]),
        flux_err=jnp.asarray([[0.1, 0.2]]),
        mask=jnp.asarray([[True, True]]),
    )


def test_all_posterior_paths_are_wired_to_the_canonical_target() -> None:
    assert "posterior_log_target(" in inspect.getsource(
        train_module._importance_weighted_wake_outputs
    )
    assert "posterior_log_target(" in inspect.getsource(
        train_module._smc_tempered_terms
    )
    assert "posterior_log_target(" in inspect.getsource(
        exact_module._target_components_fn
    )
    assert "posterior_log_target(" in inspect.getsource(
        map_module._optimize_map_start_chunk_jit
    )


def test_wake_is_nuts_mclmc_and_map_target_values_match_numerically(
    monkeypatch,
) -> None:
    model = _model()
    spec = _latent_spec()
    observation = _observation()
    x = jnp.asarray([[[0.1, -0.2]]])
    likelihood = {
        "type": "student_t",
        "student_t_dof": 2.0,
        "error_floor_frac": 0.0,
        "error_jitter": 0.0,
    }
    wake_or_is = posterior_log_target(
        model,
        x,
        observation,
        spec,
        None,
        None,
        spec.names,
        likelihood,
        {},
        model_flux_fn=_model_flux,
    )
    map_target = posterior_log_target_from_model_flux(
        model,
        x,
        observation,
        spec,
        wake_or_is.model_flux,
        likelihood,
    )
    batch = LossBatch(
        flux=observation.flux,
        flux_err=observation.flux_err,
        mask=observation.mask,
        features=jnp.zeros((1, 4)),
        truth_theta=jnp.zeros((1, 0)),
    )
    runtime = SimpleNamespace(
        model=model,
        batch=batch,
        latent_spec=spec,
        context=None,
        model_args=None,
        likelihood=likelihood,
        config={"calibration": {}},
    )
    monkeypatch.setattr(exact_module, "model_flux_from_x", _model_flux)
    exact = exact_module._target_components_fn(runtime)(x[0, 0])
    expected = float(wake_or_is.logtarget[0, 0])
    assert np.isclose(float(map_target.logtarget[0, 0]), expected, atol=1.0e-7)
    assert np.isclose(float(exact.logtarget), expected, atol=1.0e-7)
    assert np.isclose(float(exact.loglike), float(wake_or_is.loglike[0, 0]))
    assert np.isclose(float(exact.logprior), float(wake_or_is.logprior[0, 0]))


def test_canonical_support_is_identical_and_reports_outside_bounds(
    monkeypatch,
) -> None:
    model = _model()
    spec = _latent_spec()
    observation = _observation()
    invalid_x = jnp.asarray([[[2.0, 0.0]]])
    likelihood = {"type": "gaussian", "error_floor_frac": 0.0}
    wake_or_is = posterior_log_target(
        model,
        invalid_x,
        observation,
        spec,
        None,
        None,
        spec.names,
        likelihood,
        {},
        model_flux_fn=_model_flux,
    )
    batch = LossBatch(
        flux=observation.flux,
        flux_err=observation.flux_err,
        mask=observation.mask,
        features=jnp.zeros((1, 4)),
        truth_theta=jnp.zeros((1, 0)),
    )
    runtime = SimpleNamespace(
        model=model,
        batch=batch,
        latent_spec=spec,
        context=None,
        model_args=None,
        likelihood=likelihood,
        config={"calibration": {}},
    )
    monkeypatch.setattr(exact_module, "model_flux_from_x", _model_flux)
    exact = exact_module._target_components_fn(runtime)(invalid_x[0, 0])
    assert not bool(wake_or_is.physical_valid[0, 0])
    assert np.isneginf(float(wake_or_is.logtarget[0, 0]))
    assert np.isneginf(float(exact.logtarget))
    diagnostics = physical_bounds_diagnostics(
        jnp.asarray([[0.0, 0.0], [2.0, 0.0]]),
        spec,
    )
    assert diagnostics["fraction_of_samples_outside_fit_bounds"] == 0.5
    assert diagnostics["fraction_outside_fit_bounds_by_parameter"]["a"] == 0.5
    assert diagnostics["fraction_outside_fit_bounds_by_parameter"]["b"] == 0.0
