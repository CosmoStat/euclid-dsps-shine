import equinox as eqx
import jax.numpy as jnp
import pytest
from test_local_wake_diagnostic import assert_tree_equal, fixture

from euclid_dsps.amortized.local_wake_backtracking import make_guarded_wake_step


@pytest.mark.parametrize("trials,accepted", [(12, True), (1, False)])
def test_overshoot_backtracks_or_rolls_back_adam(trials, accepted):
    encoder, p, context = fixture()
    opt, step = make_guarded_wake_step(encoder, learning_rate=10.0, trials=trials)
    state = opt.init(eqx.filter(p, eqx.is_inexact_array))
    x = jnp.ones((32, 1, 2)) * 0.1
    q, new_state, m = step(p, state, context, x, jnp.zeros((32, 1)))
    assert bool(m["update_applied"]) == accepted
    assert m["proposal_loss"] > m["wake_loss"]
    if accepted:
        assert 0 < m["accepted_scale"] < 1
        assert m["wake_loss_after"] < m["wake_loss"]
        assert (
            m["wake_loss_after"]
            <= m["wake_loss"] + 1e-4 * m["accepted_scale"] * m["directional_derivative"]
        )
    else:
        assert_tree_equal(p, q)
        assert_tree_equal(state, new_state)


def test_ineligible_batch_preserves_parameters_and_moments():
    encoder, p, context = fixture(layers=2)
    opt, step = make_guarded_wake_step(encoder)
    state = opt.init(eqx.filter(p, eqx.is_inexact_array))
    weights = jnp.full((32, 1), -100.0).at[0].set(0.0)
    q, new_state, m = step(p, state, context, jnp.ones((32, 1, 2)), weights)
    assert not m["update_applied"]
    assert m["line_search_evaluations"] == 0
    assert_tree_equal(p, q)
    assert_tree_equal(state, new_state)
