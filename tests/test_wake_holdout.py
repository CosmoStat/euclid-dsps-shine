import equinox as eqx
import jax.numpy as jnp
import pytest
from test_local_wake_diagnostic import assert_tree_equal, fixture

from euclid_dsps.amortized.local_wake_backtracking import make_guarded_wake_step
from euclid_dsps.amortized.local_wake_holdout import compare_batch


def test_unchanged_parameters_have_zero_delta():
    encoder, p, context = fixture()
    result = compare_batch(
        encoder, p, p, context, jnp.ones((32, 1, 2)), jnp.zeros((32, 1))
    )
    assert result["loss_delta"] == 0
    assert result["ess"] == pytest.approx(32)
    assert result["informative"]


def test_dominated_and_nonfinite_batches_are_not_informative():
    encoder, p, context = fixture()
    x = jnp.ones((32, 1, 2))
    weights = jnp.full((32, 1), -100.0).at[0].set(0.0)
    result = compare_batch(encoder, p, p, context, x, weights)
    assert result["finite"]
    assert not result["informative"]
    result = compare_batch(encoder, p, p, context, x, weights.at[0].set(jnp.nan))
    assert not result["finite"]
    assert not result["informative"]


def test_training_descent_can_disagree_with_fresh_batch_without_mutation():
    encoder, p, context = fixture()
    optimizer, step = make_guarded_wake_step(encoder, learning_rate=0.1)
    state = optimizer.init(eqx.filter(p, eqx.is_inexact_array))
    x = jnp.ones((32, 1, 2))
    weights = jnp.zeros((32, 1))
    after, new_state, metrics = step(p, state, context, x, weights)
    assert metrics["update_applied"]
    training = compare_batch(encoder, p, after, context, x, weights)
    validation = compare_batch(encoder, p, after, context, -x, weights)
    assert training["loss_delta"] < 0
    assert validation["loss_delta"] > 0
    repeated, repeated_state, _ = step(p, state, context, x, weights)
    assert_tree_equal(after, repeated)
    assert_tree_equal(new_state, repeated_state)


@pytest.mark.parametrize("enabled", [False, True])
def test_holdout_preparation_is_opt_in(tmp_path, monkeypatch, enabled):
    from scripts import feniks_wake_descent as preparation
    from scripts.feniks_objective_night import read, write

    def reference(pilot, root):
        root.mkdir()
        write(
            root / "RUN_MANIFEST.json",
            dict(
                wake_forensic_reference={"path": str(pilot)},
                forensic_parameter_atol=1e-9,
            ),
        )

    monkeypatch.setattr(preparation, "prepare_reference", reference)
    root = tmp_path / "new"
    preparation.prepare(tmp_path / "original", root, holdout=enabled)
    manifest = read(root / "RUN_MANIFEST.json")
    assert ("wake_holdout" in manifest) == enabled
    assert manifest["adaptation_contract"] == "wake_armijo_v1"
    if enabled:
        assert manifest["wake_holdout"]["used_for_acceptance"] is False
        assert manifest["wake_holdout"]["replicates"] == 2
        assert manifest["wake_holdout"]["draws"] == 256
