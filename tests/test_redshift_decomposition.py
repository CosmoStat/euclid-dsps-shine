"""Asset-free tests with real DSPS kernels, not catalogue validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_model import _synthetic_context

from euclid_dsps.amortized import redshift_decomposition as rd
from euclid_dsps.amortized.latent import LatentSpec, theta_to_x
from euclid_dsps.amortized.local_vi_diagnostic import Budget
from euclid_dsps.amortized.posterior_target import PosteriorObservation
from euclid_dsps.model import normalize_sfh_to_stellar_mass_jax
from euclid_dsps.prior_learning.spline15d import SPLINE15D_PARAMETER_NAMES


def test_dtype_trace_detects_hidden_float32():
    def broken(z):
        return jnp.asarray(z, dtype=jnp.float32).astype(jnp.float64)

    graph = jax.make_jaxpr(jax.jit(broken))(jnp.array(1.0, dtype=jnp.float64))
    assert rd.floating_trace_dtypes(graph) == ["float32", "float64"]
    graph = jax.make_jaxpr(lambda z: jnp.sin(z))(jnp.array(1.0, dtype=jnp.float64))
    assert rd.floating_trace_dtypes(graph) == ["float64"]


def test_measure_curve_analytic_chain_and_budget():
    budget = Budget(60, 100)
    center, ad, rows = rd.measure_curve(
        lambda z: jnp.array([z * z, 3 * z]),
        jnp.array(2.0, dtype=jnp.float64),
        0.5,
        np.array([2.0, 3.0]),
        budget,
        branch="analytic",
        labels=["a", "b"],
        observed=np.array([3.0, 5.0]),
    )
    np.testing.assert_allclose(center, [4, 6])
    np.testing.assert_allclose(ad, [2, 1])
    np.testing.assert_allclose(
        [r["fd"] for r in rows], np.tile(ad, len(rd.STEPS)), atol=1e-10
    )
    assert rows[0]["physical_z_step"] == 0.01
    assert rows[0]["loglike_ad_contribution"] == -1
    assert budget.forward == 21 and budget.gradient == 1


def test_mass_normalization_default_unchanged():
    args = (
        jnp.linspace(0.05, 10.0, 80),
        jnp.linspace(0.5, 2.0, 80),
        jnp.linspace(-3.0, 1.0, 24),
        jnp.array(10.0),
        jnp.array(9.3),
    )
    old = normalize_sfh_to_stellar_mass_jax(*args)
    explicit = normalize_sfh_to_stellar_mass_jax(*args, numerical_dtype=jnp.float32)
    for a, b in zip(old, explicit, strict=True):
        np.testing.assert_array_equal(a, b)


def test_real_dsps_synthetic_branch_smoke(monkeypatch):
    # Synthetic spectra only; the age, SFH, IGM and filter kernels are real.
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
    names = tuple(SPLINE15D_PARAMETER_NAMES)
    theta = jnp.array([0.5, 9.3, -0.2, 0.3, -0.2] + [0.02] * 10, dtype=jnp.float32)
    lower = jnp.array([0.01, 7.0, -2.0, 0.01, -2.0] + [-4.0] * 10)
    upper = jnp.array([3.0, 12.0, 1.0, 3.0, 2.0] + [4.0] * 10)
    spec = LatentSpec(names, lower, upper)
    point = theta_to_x(theta, spec)
    params = {n: v for n, v in zip(names, theta, strict=True)}
    branches = rd.build_branches(context, params)
    z = theta[0]
    full = np.asarray(branches["full_native"](z))
    for name in (
        "recomposed_native",
        "stellar_native",
        "igm_native",
        "projection_native",
    ):
        np.testing.assert_allclose(branches[name](z), full, rtol=3e-6, atol=0)
    grads = {
        name: jax.jvp(fn, (z,), (jnp.ones_like(z),))[1]
        for name, fn in branches.items()
        if name != "projection_float64"
    }
    chain = sum(grads[n] for n in ("stellar_native", "igm_native", "projection_native"))
    np.testing.assert_allclose(chain, grads["full_native"], rtol=2e-4, atol=1e-35)
    assert rd.floating_trace_dtypes(
        jax.make_jaxpr(branches["projection_float64"])(z.astype(jnp.float64))
    ) == ["float64"]
    monkeypatch.setattr(rd, "STEPS", (0.001, 0.0005))
    report, rows = rd.decompose_redshift(
        context,
        spec,
        point[None],
        PosteriorObservation(
            jnp.asarray(full)[None],
            jnp.asarray(full * 0.1)[None],
            jnp.ones((1, 1), bool),
        ),
        Budget(300, 1000),
        band_names=["wide"],
        progress=lambda *args: None,
    )
    assert report["status"] == "REDSHIFT_DECOMPOSITION_COMPLETE"
    assert report["points"][0]["floating_trace_dtypes"]["age_mass_float64"] == [
        "float64"
    ]
    assert {r["branch"] for r in rows} == {
        *branches,
        "age_mass_native",
        "age_mass_float64",
    }
    assert all(r["finite"] for r in rows)
    assert report["local_optimization_started"] is False
    assert report["truth_used"] is False


def test_mutually_exclusive_modes_before_io(tmp_path):
    from scripts.run_feniks_sc_drws_local_vi_diagnostic import prepare

    with pytest.raises(ValueError, match="only one"):
        prepare(
            tmp_path / "new",
            tmp_path / "absent",
            gradient_isolation=True,
            redshift_decomposition=True,
        )


def test_summary_rejects_changed_artifact(tmp_path):
    import hashlib
    import json

    from scripts.summarize_feniks_sc_drws_redshift_decomposition import summarize

    artifact = tmp_path / "redshift_decomposition.csv"
    artifact.write_text("original")
    receipt = dict(
        status="REDSHIFT_DECOMPOSITION_COMPLETE",
        artifacts={
            artifact.name: dict(
                sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()
            )
        },
    )
    (tmp_path / "FINAL.json").write_text(json.dumps(receipt))
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="changed artifact"):
        summarize(tmp_path)


def test_float64_age_trace_with_tabulated_survival():
    context = _synthetic_context(
        dict(
            stellar_metallicity_model="lognormal_mdf_fixed_scatter",
            stellar_metallicity_scatter_dex=0.2,
        )
    )
    context.ssp_surviving_mstar_jax = jnp.full((3, 24), 0.7, dtype=jnp.float32)
    theta = jnp.array([0.5, 9.3, -0.2, 0.3, -0.2] + [0.02] * 10, dtype=jnp.float32)
    params = dict(zip(SPLINE15D_PARAMETER_NAMES, theta, strict=True))
    fn = rd.make_age_mass_weights(params, context, dtype=jnp.float64)
    z = jnp.array(0.5, dtype=jnp.float64)
    assert rd.floating_trace_dtypes(jax.make_jaxpr(fn)(z)) == ["float64"]
    weights = jax.jit(fn)(z)
    assert np.isfinite(weights).all()
    assert float(weights.sum() * 0.7) == pytest.approx(10 ** float(theta[1]), rel=1e-6)
