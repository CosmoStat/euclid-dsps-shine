import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_local_wake_diagnostic import assert_tree_equal, fixture
from test_objective_night import night  # noqa: F401

from scripts.feniks_wake_forensics import inspect_update, interpolate


def test_prepare_pins_complete_pilot_without_night_gate(night, tmp_path):  # noqa: F811
    from scripts.feniks_wake_forensics import prepare, read

    pilot, _ = night
    root = tmp_path / "forensics"
    prepare(pilot, root)
    manifest = read(root / "RUN_MANIFEST.json")
    assert "night_extension" not in manifest
    assert "experiment_seed_offset" not in manifest
    assert manifest["objective_recipe"]["decoder_draw_budgets"] == [1024, 4096]
    assert manifest["wake_forensic_reference"]["hashes"]


def test_prepare_rejects_changed_checkpoint(night, tmp_path):  # noqa: F811
    from scripts.feniks_wake_forensics import prepare

    pilot, _ = night
    (pilot / "cases/observed_000/wake_0/draws_04096/parameters.eqx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint contract"):
        prepare(pilot, tmp_path / "forensics")
    assert not (tmp_path / "forensics").exists()


def test_replay_preserves_original_path_and_detects_mismatch(tmp_path, monkeypatch):
    import pandas as pd

    from euclid_dsps.amortized.local_vi_diagnostic import Budget
    from euclid_dsps.amortized.local_wake_diagnostic import make_wake_step
    from scripts import feniks_wake_forensics as forensic
    from scripts import run_feniks_sc_drws_local_vi_diagnostic as runner

    encoder, original, context = fixture()
    x = jnp.ones((32, 1, 2))
    weights = jnp.zeros((32, 1))
    keys = []

    def batch(*args, **kwargs):
        keys.append(np.asarray(args[7]).tolist())
        return x, weights, {}

    monkeypatch.setattr(forensic, "wake_batch", batch)
    evaluations = []

    def evaluate(
        folder,
        encoder,
        q,
        context,
        observation,
        target,
        spec,
        budget,
        seed,
        draws,
        generated,
    ):
        evaluations.append((np.asarray(q.mean), seed, draws))
        return {
            "raw_ess": {"fraction_median": 1.0},
            "maximum_raw_weight": {"median": 0.01},
            "negative_elbo": 0.0,
            "residual_rms": 1.0,
        }

    monkeypatch.setattr(runner, "evaluate_distribution", evaluate)
    opt, step = make_wake_step(encoder)
    state = opt.init(eqx.filter(original, eqx.is_inexact_array))
    q = original
    decisions = []
    for _ in range(2):
        q, state, metrics = step(q, state, context, x, weights)
        decisions.append(bool(metrics["update_applied"]))
    pilot = tmp_path / "pilot"
    saved = pilot / "cases/observed_000/wake_0/draws_04096"
    saved.mkdir(parents=True)
    eqx.tree_serialise_leaves(saved / "parameters.eqx", q)
    pd.DataFrame({"update_applied": decisions}).to_csv(
        saved.parent / "optimization.csv", index=False
    )
    manifest = {
        "wake_forensic_reference": {"path": str(pilot), "hashes": {}},
        "forensic_parameter_atol": 1e-9,
        "objective_recipe": {
            "learning_rate": 1e-4,
            "minimum_ess": 16.0,
            "maximum_weight": 0.2,
            "wake_draws": 256,
            "decoder_draw_budgets": [512],
        },
    }
    prepared = [("observed_000", None, None, original, context, [original])]
    result = forensic.run(
        tmp_path / "run", manifest, encoder, prepared, None, None, Budget(60, 10000)
    )
    assert result["status"] == "WAKE_FORENSICS_COMPLETE"
    assert len(evaluations) == 4
    assert len({(e[1], e[2]) for e in evaluations}) == 1
    np.testing.assert_array_equal(evaluations[0][0], original.mean)
    assert len(keys) == 2
    # A changed reference must not be labelled as a reproduced trajectory.
    eqx.tree_serialise_leaves(saved / "parameters.eqx", original)
    result = forensic.run(
        tmp_path / "mismatch",
        manifest,
        encoder,
        prepared,
        None,
        None,
        Budget(60, 10000),
    )
    assert result["status"] == "WAKE_REPLAY_MISMATCH"


def test_fixed_batch_directional_derivative_and_empty_layers():
    with jax.enable_x64():
        encoder, before, context = fixture()
        before = jax.tree.map(
            lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, before
        )
        after = eqx.tree_at(lambda p: p.mean, before, before.mean + 0.01)
        x = jnp.ones((32, 1, 2), dtype=jnp.float64)
        weights = jnp.zeros((32, 1), dtype=jnp.float64)
        report = inspect_update(encoder, before, after, context, x, weights)
        assert report["after_loss"] < report["before_loss"]
        for row in report["directional_stencils"]:
            np.testing.assert_allclose(row["ad"], row["fd"], atol=1e-10)
        assert report["parameter_deltas"]["layers"]["max_abs"] == 0


def test_interpolation_preserves_masks_and_endpoints():
    encoder, before, context = fixture(layers=2)
    after = eqx.tree_at(lambda p: p.mean, before, before.mean + 1)
    assert_tree_equal(before, interpolate(before, after, 0))
    assert_tree_equal(after, interpolate(before, after, 1))
    scaled = interpolate(before, after, 0.1)
    np.testing.assert_allclose(scaled.mean, before.mean + 0.1)
    assert_tree_equal(scaled.layers, before.layers)
