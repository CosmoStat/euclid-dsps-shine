import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from euclid_dsps.amortized.gradient_isolation import isolate_gradient
from euclid_dsps.amortized.likelihood import photometric_loglike
from euclid_dsps.amortized.local_vi_diagnostic import Budget
from euclid_dsps.amortized.posterior_target import (
    PosteriorObservation,
    PosteriorTargetValues,
)


@pytest.mark.parametrize("broken", [False, True])
def test_separates_likelihood_algebra_from_flux_derivative(broken):
    observation = PosteriorObservation(
        jnp.array([[1e-28, 1.2e-28]]), jnp.full((1, 2), 1e-29), jnp.ones((1, 2), bool)
    )

    def target(x, obs):
        y = jax.lax.stop_gradient(x) if broken else x
        flux = 1e-28 * (1 + 0.1 * y)
        ll = photometric_loglike(
            obs.flux,
            flux,
            obs.flux_err,
            obs.mask,
            likelihood_type="gaussian",
            error_floor_frac=0,
        )
        lp = -0.5 * jnp.sum(x * x, axis=-1)
        return PosteriorTargetValues(
            lp + ll, ll, lp, jnp.ones(lp.shape, bool), flux, flux
        )

    summary, rows = isolate_gradient(
        target,
        observation,
        jnp.array([[[0.2, 0.3]]]),
        jnp.array([[[1.0, 0.0]]]),
        Budget(120, 100),
        band_names=("b0", "b1"),
        coordinate_names=("x0", "x1"),
    )
    assert summary["local_optimization_started"] is False
    assert summary["likelihood_only"]["autodiff"] == pytest.approx(
        summary["likelihood_only"]["analytic"], abs=1e-4
    )
    frame = pd.DataFrame(rows)
    selected = frame[(frame.direction == "x0") & (frame.band == "b0")]
    assert np.allclose(selected.flux_fd_sigma, 1, atol=0.01)
    if broken:
        assert np.allclose(selected.flux_ad_sigma, 0)
    else:
        assert np.allclose(selected.flux_ad_sigma, 1, atol=0.001)
        assert summary["maximum_abs_chain_gradient_delta"] < 0.001
        sums = frame.groupby(["direction", "step"])[
            "loglike_centered_fd_contribution"
        ].sum()
        assert sums.loc[("x0", 0.02)] == pytest.approx(
            summary["canonical_loglike_gradient"][0], abs=0.005
        )
