import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_model import _synthetic_context

from euclid_dsps import model as sed
from euclid_dsps.amortized.redshift_decomposition import floating_trace_dtypes
from euclid_dsps.prior_learning.spline15d import SPLINE15D_PARAMETER_NAMES


@pytest.mark.parametrize("igm", ["none", "madau95_approx", "fsps_madau95"])
def test_zpath64_synthetic_sed_and_legacy_unchanged(igm):
    context = _synthetic_context(
        dict(
            sfh_model="spline15d",
            agn_model="none",
            dust_model="prospector_fsps",
            igm_model=igm,
            stellar_metallicity_model="lognormal_mdf_fixed_scatter",
            stellar_metallicity_scatter_dex=0.2,
            photometry_integrator="merged_gauss4_v1",
            mdf_weight_precision="float64_v1",
        )
    )
    params = dict(
        zip(
            SPLINE15D_PARAMETER_NAMES,
            jnp.array([0.5, 9.3, -0.2, 0.3, -0.2] + [0.02] * 10, dtype=jnp.float32),
            strict=True,
        )
    )
    before = sed.run_spline15d_model_jax(context, params).model_mags
    cfg = copy.deepcopy(context.model_config)
    branches = sed.diagnostic_spline_redshift_branches(context, params)
    z = jnp.asarray(params["z_obs"], dtype=jnp.float64)
    gradients = {}
    for name, fn in branches.items():
        if name.startswith("zpath64"):
            assert floating_trace_dtypes(jax.make_jaxpr(fn)(z)) == ["float64"]
        compiled = jax.jit(fn)
        center = compiled(z)
        ad = jax.jvp(compiled, (z,), (jnp.ones_like(z),))[1]
        assert np.isfinite(center).all() and np.isfinite(ad).all()
        gradients[name] = ad
        if name.startswith("zpath64"):
            h = 1e-5
            fd = (compiled(z + h) - compiled(z - h)) / (2 * h)
            np.testing.assert_allclose(
                ad, fd, rtol=3e-3, atol=float(abs(center).max()) * 1e-4
            )
    np.testing.assert_allclose(
        sum(gradients["zpath64_" + k] for k in ("stellar", "igm", "projection")),
        gradients["zpath64_full"],
        rtol=1e-9,
        atol=1e-30,
    )
    np.testing.assert_array_equal(
        sed.run_spline15d_model_jax(context, params).model_mags, before
    )
    assert context.model_config == cfg


def test_igm_default_is_explicit_legacy():
    wave = jnp.linspace(500, 3000, 20)
    for mode in ("none", "madau95_approx", "fsps_madau95"):
        cfg = dict(igm_model=mode)
        a = sed.apply_igm_transmission_jax(wave, jnp.ones(20), 1.2, cfg)
        b = sed.apply_igm_transmission_jax(
            wave, jnp.ones(20), 1.2, cfg, numerical_dtype=jnp.float32
        )
        np.testing.assert_array_equal(a, b)


def test_replay_branch_does_not_promote_and_rejects_wrong_ad():
    from euclid_dsps.amortized.redshift_precision import analyze
    from euclid_dsps.amortized.target_resolution import STEPS

    case = dict(
        component="lsst_z",
        anchor=[1.0],
        observed=[1.0],
        sigma=[1.0],
        flux_jvp=[2.0],
        band_index=0,
        flux_dtype="float64",
        ad=9.0,
        point_index=4,
        coordinate="z_obs",
        source_ad_delta=0.0,
        atol=0.01,
        rtol=0.01,
        samples=[
            dict(
                step=h,
                actual_plus_step=h,
                actual_minus_step=h,
                plus=[1 + 2 * h],
                minus=[1 - 2 * h],
            )
            for h in STEPS
        ],
    )
    snapshot = dict(
        branches=[
            dict(
                name="zpath64_full",
                floating_dtypes=["float64"],
                max_center_delta_sigma=0.0,
                snapshot=dict(cases=[case], expected_checks=1),
            )
        ],
        expected_branches=2,
        physical_z=1.0,
        dz_dx=0.5,
        source_ad_delta=0.0,
    )
    report, _ = analyze(snapshot)
    assert report["status"] == "REDSHIFT_PRECISION_RUNNING"
    assert report["branches"][0]["checks"][0]["status"] == "FAIL"
    assert report["scientific_promotion"] is False


def test_collect_analytic_target_and_source_ad_guard(monkeypatch):
    from types import SimpleNamespace

    from euclid_dsps.amortized import redshift_precision as rp
    from euclid_dsps.amortized.latent import LatentSpec, x_to_theta
    from euclid_dsps.amortized.local_vi_diagnostic import Budget
    from euclid_dsps.amortized.posterior_target import PosteriorObservation

    spec = LatentSpec(("z_obs",), jnp.array([0.01]), jnp.array([4.0]))
    point = jnp.array([0.4], dtype=jnp.float32)
    obs = PosteriorObservation(
        jnp.array([[3.0]]), jnp.ones((1, 1)), jnp.ones((1, 1), bool)
    )

    def target(x, obs):
        return SimpleNamespace(
            model_flux=2 * x_to_theta(x, spec) + 1,
            physical_valid=jnp.ones(x.shape[:-1], bool),
        )

    monkeypatch.setattr(
        rp,
        "diagnostic_spline_redshift_branches",
        lambda ctx, p: {
            "zpath64_full": lambda z: jnp.array([2 * z + 1]),
        },
    )
    ad = float(jax.grad(lambda x: (2 * x_to_theta(x, spec) + 1).sum())(point)[0])
    source = dict(
        checks=[dict(point_index=4, coordinate="z_obs", component="lsst_z", ad=ad)]
    )
    snapshots = []
    data = rp.collect(
        SimpleNamespace(model_config={}),
        spec,
        point,
        obs,
        target,
        source,
        Budget(120, 200),
        bands=("lsst_z",),
        progress=snapshots.append,
    )
    report, rows = rp.analyze(data)
    assert report["status"] == "REDSHIFT_PRECISION_COMPLETE"
    assert report["branches"][1]["all_checks_passed"]
    assert len(rows) == 2 * 2 * 25
    assert (
        data["branches"][0]["snapshot"]["cases"][0]["samples"][0]["actual_plus_step"]
        != 0.02
    )
    source["checks"][0]["ad"] += 1.0
    with pytest.raises(ValueError, match="source AD not reproduced"):
        rp.collect(
            SimpleNamespace(model_config={}),
            spec,
            point,
            obs,
            target,
            source,
            Budget(120, 200),
            bands=("lsst_z",),
            progress=lambda s: None,
        )
