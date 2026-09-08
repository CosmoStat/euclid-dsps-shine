import copy
import hashlib
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_model import _synthetic_context

from euclid_dsps import model as sed
from euclid_dsps.amortized.decoder_qualification import resolved_check


def test_resolved_check_does_not_pick_step_from_ad():
    a = resolved_check([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 1.0, atol=0.01, rtol=0.01)
    b = resolved_check([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 2.0, atol=0.01, rtol=0.01)
    assert a["selected_step_index"] == b["selected_step_index"] == 2
    assert a["status"] == "PASS" and b["status"] == "FAIL"
    assert (
        resolved_check([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1.0, atol=0.01, rtol=0.01)[
            "status"
        ]
        == "INCONCLUSIVE"
    )
    assert (
        resolved_check([1.0, np.nan, 1.0], [0.0, 0.0, 0.0], 1.0, atol=0.01, rtol=0.01)[
            "status"
        ]
        == "FAIL"
    )


@pytest.mark.parametrize("broken", [False, True])
def test_full_qualification_analytic_gradient_control(monkeypatch, broken):
    from euclid_dsps.amortized import decoder_qualification as dq
    from euclid_dsps.amortized.local_vi_diagnostic import Budget
    from euclid_dsps.amortized.posterior_target import (
        PosteriorObservation,
        PosteriorTargetValues,
    )

    monkeypatch.setattr(dq, "STEPS", (0.02, 0.01, 0.005))

    def target(x, obs, bad=False):
        flux = x + (jax.lax.stop_gradient(x) if bad else x)
        lp = -0.5 * jnp.sum(x**2, axis=-1)
        ll = -0.5 * jnp.sum(((flux - obs.flux) / obs.flux_err) ** 2, axis=-1)
        return PosteriorTargetValues(
            lp + ll, ll, lp, jnp.ones(lp.shape, bool), flux, flux
        )

    obs = PosteriorObservation(
        jnp.zeros((1, 1)), jnp.ones((1, 1)), jnp.ones((1, 1), bool)
    )
    report, _ = dq.qualify(
        target,
        lambda x, o: target(x, o, broken),
        jnp.array([[0.5]]),
        [obs],
        Budget(60, 100),
        names=["x"],
        bands=["flux"],
        progress=lambda *args: None,
    )
    assert report["candidate_numerical_checks"] == ("NOT_PASSED" if broken else "PASS")
    assert report["npe_training_started"] is False


def test_versioned_projection_and_full_sed_smoke():
    from euclid_dsps.prior_learning.spline15d import SPLINE15D_PARAMETER_NAMES

    context = _synthetic_context(
        dict(
            sfh_model="spline15d",
            agn_model="none",
            dust_model="prospector_fsps",
            igm_model="fsps_madau95",
            stellar_metallicity_model="lognormal_mdf_fixed_scatter",
            stellar_metallicity_scatter_dex=0.2,
        )
    )
    theta = jnp.array([0.5, 9.3, -0.2, 0.3, -0.2] + [0.02] * 10, dtype=jnp.float32)
    names = tuple(SPLINE15D_PARAMETER_NAMES)

    def forward(ctx, t):
        return sed.run_spline15d_model_jax(
            ctx, dict(zip(names, t, strict=True))
        ).model_mags

    before = forward(context, theta)
    old = copy.copy(context)
    old.model_config = dict(
        context.model_config, photometry_integrator="legacy_trapezoid_v1"
    )
    np.testing.assert_array_equal(forward(old, theta), before)
    candidate = copy.copy(context)
    candidate.model_config = dict(
        context.model_config, photometry_integrator="merged_gauss4_v1"
    )
    f = jax.jit(lambda t: forward(candidate, t))
    result = f(theta)
    assert np.isfinite(result).all() and result.dtype == jnp.float64
    # Full SFH/SED/IGM/projection chain, not a frozen spectrum.
    assert np.isfinite(jax.jacfwd(f)(theta)).all()
    np.testing.assert_array_equal(forward(context, theta), before)
    previous_x64 = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="JAX_ENABLE_X64"):
            sed.predict_mags_jax(candidate, jnp.array([1.0, 2.0]), jnp.ones(2), 0.5)
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
    with pytest.raises(ValueError, match="unsupported photometry_integrator"):
        sed._normalized_model_config(dict(photometry_integrator="typo"))


def test_sleep_cache_numerics_cannot_be_mixed(tmp_path):
    from euclid_dsps.amortized.train import (
        _array_tree_sha256,
        _prepare_sleep_noiseless_cache,
    )
    from euclid_dsps.config import load_config

    config = load_config(
        "configs/experiments/feniks_sc_drws_r29_frozen_parent_topology_sleep_npe.yaml"
    )
    config["amortized"]["prior"]["train_jointly"] = False
    config["amortized"]["training"]["jax_batch_size"] = 1
    path = tmp_path / "bank.npz"
    config["amortized"]["objective"]["sleep"]["noiseless_flux_cache"] = dict(
        enabled=True, path=str(path), candidates=4
    )
    model = SimpleNamespace(prior=jnp.ones(2))
    np.savez(
        path,
        x=np.zeros((4, 1)),
        model_flux=np.ones((4, 1)),
        physical_valid=np.ones(4, bool),
    )
    recorded = dict(
        schema_version=1,
        generator="direct_frozen_parent",
        density_space="latent_x",
        photometry_space="raw_noiseless_decoder_flux_before_calibration",
        parameter_names=["x"],
        prior_fingerprint_sha256=_array_tree_sha256(model.prior),
        calibration_fingerprint_sha256=hashlib.sha256(b"{}").hexdigest(),
        candidates=4,
        noise_cached=False,
        noise_resampled_each_optimization_step=True,
        mask_context_from_observed_training_rows=True,
        selection_applied_after_fresh_noise=True,
        catalogue_truth_used=False,
    )
    path.with_suffix(".npz.json").write_text(json.dumps(recorded))
    kwargs = dict(
        model=model,
        config=config,
        sleep_runtime_config=dict(enabled=True),
        latent_spec=None,
        context=None,
        model_args=None,
        parameter_names=("x",),
        calibration_config={},
        seed=1,
    )
    assert _prepare_sleep_noiseless_cache(**kwargs)["receipt"]["created"] is False
    config["model"]["photometry_integrator"] = "merged_gauss4_v1"
    with pytest.raises(ValueError, match="photometry_numerics"):
        _prepare_sleep_noiseless_cache(**kwargs)
    recorded["photometry_numerics"] = sed.photometry_numerics(config["model"])
    path.with_suffix(".npz.json").write_text(json.dumps(recorded))
    assert _prepare_sleep_noiseless_cache(**kwargs)["receipt"]["created"] is False
    config["model"]["photometry_integrator"] = "legacy_trapezoid_v1"
    with pytest.raises(ValueError, match="photometry_numerics"):
        _prepare_sleep_noiseless_cache(**kwargs)
