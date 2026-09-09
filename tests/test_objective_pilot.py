import json
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_local_vi_diagnostic import model_fixture

from euclid_dsps.amortized.features import FeatureStats, make_encoder_features
from euclid_dsps.amortized.latent import LatentSpec
from euclid_dsps.amortized.local_vi_diagnostic import Budget, initialize
from euclid_dsps.amortized.population_vem import sha256_file
from euclid_dsps.amortized.posterior_target import (
    PosteriorObservation,
    PosteriorTargetValues,
)
from euclid_dsps.amortized.train import _array_tree_sha256
from scripts import feniks_objective_pilot as pilot
from scripts.run_feniks_sc_drws_balanced_npe import write


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize(
    "extra",
    [
        {"controlled": True},
        {"long_optimization": True},
        {"support_probe_root": "old"},
        {"long_replay_root": "old"},
        {"objects": 2},
        {"steps": 32},
        {"draws": 64},
        {"arm": "B"},
    ],
)
def test_pilot_rejects_changed_protocol_before_preparation(tmp_path, extra):
    from scripts.feniks_qualified_local_vi import prepare

    with pytest.raises(ValueError, match="fixed exclusive C recipe"):
        prepare(
            tmp_path / "out",
            tmp_path / "source",
            tmp_path / "night",
            objective_pilot_root=tmp_path / "long",
            **extra,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("audit_status", ["PASS", "INCONCLUSIVE", "FAIL"])
def test_pilot_gate_and_real_updates(tmp_path, monkeypatch, audit_status):
    import euclid_dsps.amortized.local_vi_objective_audit as audit
    import euclid_dsps.amortized.local_wake_diagnostic as wake

    root, source = tmp_path / "pilot", tmp_path / "source"
    root.mkdir()
    source.mkdir()
    for p in (source, root):
        np.savez(p / "SIMULATED_INPUTS.npz", x=np.zeros((2, 2)))
    model = model_fixture()
    frozen = _array_tree_sha256(model)
    stats = FeatureStats(
        np.full(2, 100.0), np.ones(2), ("lsst_u", "lsst_r"), append_mask=True
    )
    observation = PosteriorObservation(
        jnp.full((1, 2), 100.0), jnp.ones((1, 2)), jnp.ones((1, 2), bool)
    )
    spec = LatentSpec(
        ("x0", "x1"),
        jnp.full(2, -10.0),
        jnp.full(2, 10.0),
        arithmetic_precision="float64_v1",
    )
    p, _ = initialize(
        model,
        make_encoder_features(
            observation.flux, observation.flux_err, stats, observation.mask
        ),
    )
    cases = [(group, 0, observation, None) for group in ("observed", "simulated")]
    hashes = {}
    for group, *_ in cases:
        for start in (0, 1):
            name = f"cases/{group}_000/start_{start}/parameters.eqx"
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(path, p)
            hashes[name] = sha256_file(path)
    recipe = dict(
        audit_draws=4,
        reverse_draws=4,
        wake_draws=4,
        decoder_draw_budgets=[4, 8],
        intermediate_evaluation_draws=32,
        final_evaluation_draws=32,
        learning_rate=1e-4,
        minimum_ess=1,
        maximum_weight=1,
    )
    manifest = dict(
        method="qualified_objective_pilot_v1",
        objective_pilot=dict(path=str(source), hashes=hashes),
        objective_recipe=recipe,
    )
    write(root / "RUN_MANIFEST.json", manifest)
    audited = []

    def fake_audit(*a, **k):
        audited.append(k["seed"])
        return dict(status=audit_status, rows=[])

    monkeypatch.setattr(audit, "audit_objective", fake_audit)
    old_reverse, old_wake = pilot.make_step, wake.make_wake_step

    def guarded(factory):
        def construct(*a, **k):
            assert audit_status == "PASS" and len(audited) == 4
            return factory(*a, **k)

        return construct

    monkeypatch.setattr(pilot, "make_step", guarded(old_reverse))
    monkeypatch.setattr(wake, "make_wake_step", guarded(old_wake))

    @eqx.filter_jit
    def target(x, obs):
        lp = model.prior.log_prob(x)
        flux = 100 + 0.2 * x
        ll = -0.5 * jnp.sum((flux - obs.flux) ** 2, axis=-1)
        return PosteriorTargetValues(lp + ll, ll, lp, jnp.ones(lp.shape, bool), x, flux)

    result = pilot.run_pilot(
        root, manifest, model, stats, spec, cases, target, Budget(300, 100000)
    )
    assert _array_tree_sha256(model) == frozen
    assert result["scientific_promotion"] is False
    assert len(audited) == 4 and len(set(audited)) == 4
    if audit_status == "PASS":
        assert result["status"] == "OBJECTIVE_PILOT_COMPLETE"
        assert result["cases_complete"] == 2
        receipt = json.loads((root / "cases/observed_000/COMPLETE.json").read_text())
        assert len(receipt["outcomes"]) == 8
        assert all(o["applied_updates"] == o["attempts"] for o in receipt["outcomes"])
    else:
        assert result["status"] == "OBJECTIVE_AUDIT_NOT_PASSED"
        assert result["optimization_started"] is False
        assert not list(root.glob("cases/*/*/optimization.csv"))
    output = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_feniks_sc_drws_objective_pilot.py",
            str(root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "No selection or promotion" in output.stdout
    detailed = root / "cases/observed_000/audit_start_0/stencils.csv"
    detailed.write_text("changed evidence")
    from scripts.summarize_feniks_sc_drws_objective_pilot import summarize

    with pytest.raises(ValueError, match="objective audit artifact changed"):
        summarize(root)
