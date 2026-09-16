import copy

import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.amortized.local_vi_diagnostic import Budget
from euclid_dsps.amortized.posterior_target import (
    PosteriorObservation,
    PosteriorTargetValues,
)
from euclid_dsps.amortized.target_resolution import analyze, collect, gaussian_response


def snapshot(offset=0.0):
    def target(x, obs):
        flux = x * 0.013 + 30
        ll = -0.5 * jnp.sum(((flux - obs.flux) / obs.flux_err) ** 2, axis=-1) + offset
        lp = jnp.zeros_like(ll)
        return PosteriorTargetValues(
            ll, ll, lp, jnp.ones_like(ll, dtype=bool), flux, flux
        )

    obs = PosteriorObservation(
        jnp.array([[-63.0]]), jnp.ones((1, 1)), jnp.ones((1, 1), bool)
    )
    point = jnp.array([[0.4]], dtype=jnp.float64)
    ad = -(0.4 * 0.013 + 93) * 0.013
    report = dict(
        variant_labels=["source"],
        cases=[
            dict(
                variant="source",
                point_index=0,
                checks=[
                    dict(
                        coordinate="x",
                        component="centered_loglike",
                        status="INCONCLUSIVE",
                        ad=ad,
                        atol=0.1,
                        rtol=0.05,
                    )
                ],
            )
        ],
    )
    return collect(
        target,
        point,
        [obs],
        report,
        Budget(60, 100),
        names=("x",),
        bands=("band",),
        progress=lambda s: None,
    )


def test_gaussian_difference_identity():
    rng = np.random.default_rng(17)
    anchor, observed, sigma = rng.normal(size=4), rng.normal(size=4), np.ones(4)
    plus = anchor + 0.001 * rng.normal(size=4)
    expected = -0.5 * (
        ((plus - observed) / sigma) ** 2 - ((anchor - observed) / sigma) ** 2
    )
    np.testing.assert_allclose(
        gaussian_response(plus, anchor, observed, sigma), expected, atol=1e-15
    )


def test_likelihood_offset_invariance_and_large_anchor():
    before, _ = analyze(snapshot())
    shifted, _ = analyze(snapshot(1e15))
    assert before == shifted
    assert before["checks"][0]["status"] == "PASS"
    assert before["scientific_promotion"] is False


def test_step_selection_does_not_consult_ad_and_rejects_bad_gradient():
    data = snapshot()
    good, _ = analyze(data)
    bad = copy.deepcopy(data)
    bad["cases"][0]["ad"] += 20
    wrong, _ = analyze(bad)
    assert (
        good["checks"][0]["selected_step_index"]
        == wrong["checks"][0]["selected_step_index"]
    )
    assert wrong["checks"][0]["status"] == "FAIL"


def test_no_plateau_is_not_converted_to_success():
    data = snapshot()
    c = data["cases"][0]
    for i, s in enumerate(c["samples"]):
        s["plus"][0] += (-1) ** i * s["step"] * 10
    result, _ = analyze(data)
    assert result["checks"][0]["status"] == "INCONCLUSIVE"


def test_unrepresentable_step_fails_closed():
    data = snapshot()
    data["cases"][0]["samples"][0]["actual_plus_step"] = 0
    with pytest.raises(ValueError, match="unrepresentable"):
        analyze(data)


def test_partial_snapshot_does_not_claim_completion():
    data = snapshot()
    data["expected_checks"] = 2
    result, _ = analyze(data)
    assert result["status"] == "TARGET_RESOLUTION_RUNNING"
    assert result["next_stage"] == "AUDIT_IN_PROGRESS"
    assert result["unresolved_checks_resolved"] is False


def test_unequal_stencil_is_exact_for_quadratic_flux():
    data = snapshot()
    c = data["cases"][0]
    c.update(
        component="band",
        band_index=0,
        anchor=[1.0],
        observed=[0.0],
        sigma=[1.0],
        ad=2.0,
    )
    for s in c["samples"]:
        a, b = 1.2 * s["step"], 0.8 * s["step"]
        s.update(
            actual_plus_step=a,
            actual_minus_step=b,
            plus=[(1 + a) ** 2],
            minus=[(1 - b) ** 2],
        )
    result, rows = analyze(data)
    assert result["checks"][0]["status"] == "PASS"
    np.testing.assert_allclose([r["fd"] for r in rows], 2.0, atol=1e-9)


def test_float32_actual_step_and_flux_path():
    def target(x, obs):
        flux = (x * 0.1).astype(jnp.float32)
        ll = -0.5 * jnp.sum((flux - obs.flux) ** 2, axis=-1)
        return PosteriorTargetValues(
            ll, ll, ll * 0, jnp.ones_like(ll, dtype=bool), flux, flux
        )

    point = jnp.array([[12.345]], dtype=jnp.float32)
    obs = PosteriorObservation(
        jnp.zeros((1, 1)), jnp.ones((1, 1)), jnp.ones((1, 1), bool)
    )
    source = dict(
        variant_labels=["source"],
        cases=[
            dict(
                variant="source",
                point_index=0,
                checks=[
                    dict(
                        coordinate="z_obs",
                        component="band",
                        status="INCONCLUSIVE",
                        ad=float(jnp.float32(0.1)),
                        atol=0.01,
                        rtol=0.01,
                    )
                ],
            )
        ],
    )
    data = collect(
        target,
        point,
        [obs],
        source,
        Budget(60, 100),
        names=("z_obs",),
        bands=("band",),
        progress=lambda s: None,
    )
    assert data["cases"][0]["flux_dtype"] == "float32"
    assert (
        data["cases"][0]["samples"][-1]["actual_plus_step"]
        != data["cases"][0]["samples"][-1]["step"]
    )
    result, _ = analyze(data)
    assert result["checks"][0]["status"] == "PASS"
