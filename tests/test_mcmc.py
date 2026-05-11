from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from euclid_dsps.mcmc import _prior_distribution, _prior_location


def test_prior_location_can_use_row_resolved_base_value() -> None:
    value = _prior_location(
        "z_obs",
        {"initial": "from_base", "bounds": [0.0, 6.0]},
        {"loc": "from_base"},
        {"z_obs": 0.72},
    )

    assert value == 0.72


def test_scaled_beta_prior_uses_fit_bounds() -> None:
    prior = _prior_distribution(
        "dust_av",
        {"initial": 0.2, "bounds": [0.0, 3.0]},
        {"type": "scaled_beta", "alpha": 1.2, "beta": 3.0},
        {"dust_av": 0.2},
    )

    assert np.isfinite(float(prior.log_prob(jnp.asarray(1.5))))
    assert not np.isfinite(float(prior.log_prob(jnp.asarray(0.0))))
    assert not np.isfinite(float(prior.log_prob(jnp.asarray(3.0))))
