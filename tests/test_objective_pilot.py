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


@pytest.mark.parametrize(
    "audit_status,precision",
    [
        ("PASS", False),
        ("INCONCLUSIVE", False),
        ("FAIL", False),
        ("PASS", True),
        ("PASS", "transport64"),
        ("FAIL", "transport64"),
        ("PASS", "guarded"),
        ("FAIL", "guarded"),
    ],
)
def test_pilot_gate_and_real_updates(tmp_path, monkeypatch, audit_status, precision):
    guarded_wake = precision == "guarded"
    if guarded_wake:
        precision = "transport64"
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
    if guarded_wake:
        manifest["wake_backtracking"] = dict(armijo=1e-4, trials=12)
        manifest["adaptation_contract"] = "wake_armijo_v1"
    if precision:
        import euclid_dsps.amortized.local_transport_precision as transport

        reference = tmp_path / "reference"
        reference_hashes = {}
        for number, (group, *_) in enumerate(cases):
            for start in (0, 1):
                name = f"cases/{group}_000/audit_start_{start}/AUDIT.json"
                write(
                    reference / name,
                    dict(
                        status="PASS",
                        seed=50000000 + number * 100000 + start * 10000,
                        noise_sha256="pinned",
                    ),
                )
                reference_hashes[name] = sha256_file(reference / name)
        manifest["method"] = "qualified_transport_precision_audit_v1"
        manifest["transport_precision_reference"] = dict(
            path=str(reference), hashes=reference_hashes
        )
        monkeypatch.setattr(
            transport,
            "compare_transport",
            lambda *a, **k: dict(
                transport64_audit=dict(status="PASS", rows=[]),
                traces=[],
                arrays={},
                dtypes={},
                center_max_abs_delta={},
            ),
        )
        if precision == "transport64":
            from scripts import feniks_transport_precision

            manifest["method"] = "qualified_objective_transport64_pilot_v1"
            manifest["transport_contract"] = "conditional_transport_float64_v1"
            manifest["objective_execution_recipe"] = {
                **recipe,
                "decoder_draw_budgets": [4, 12],
            }
            manifest["experiment_seed_offset"] = 100000000
            monkeypatch.setattr(
                feniks_transport_precision,
                "pin_reference",
                lambda *a, **k: manifest["transport_precision_reference"],
            )
    write(root / "RUN_MANIFEST.json", manifest)
    audited = []

    def fake_audit(*a, **k):
        if precision == "transport64":
            assert isinstance(a[0], transport.DiagnosticTransport64)
            assert a[1].mean.dtype == jnp.float64
            assert "noise" in k and "directions" in k
        audited.append(k["seed"])
        return dict(status=audit_status, rows=[], seed=k["seed"], noise_sha256="pinned")

    monkeypatch.setattr(audit, "audit_objective", fake_audit)
    old_reverse, old_wake = pilot.make_step, wake.make_wake_step

    def guarded(factory):
        def construct(*a, **k):
            assert (
                precision is not True and audit_status == "PASS" and len(audited) == 4
            )
            if precision == "transport64":
                assert isinstance(a[0], transport.DiagnosticTransport64)
            return factory(*a, **k)

        return construct

    monkeypatch.setattr(pilot, "make_step", guarded(old_reverse))
    monkeypatch.setattr(wake, "make_wake_step", guarded(old_wake))
    if guarded_wake:
        import euclid_dsps.amortized.local_wake_backtracking as descent

        monkeypatch.setattr(
            descent, "make_guarded_wake_step", guarded(descent.make_guarded_wake_step)
        )
    if precision == "transport64":
        from scripts import run_feniks_sc_drws_local_vi_diagnostic as runner

        old_evaluate, old_batch = runner.evaluate_distribution, wake.wake_batch

        def evaluate(*a, **k):
            assert isinstance(a[1], transport.DiagnosticTransport64)
            assert a[8] >= 160000000
            return old_evaluate(*a, **k)

        def batch(*a, **k):
            assert isinstance(a[0], transport.DiagnosticTransport64)
            return old_batch(*a, **k)

        monkeypatch.setattr(runner, "evaluate_distribution", evaluate)
        monkeypatch.setattr(wake, "wake_batch", batch)

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
    if precision is True:
        assert result["status"] == "TRANSPORT_PRECISION_DIAGNOSTIC_COMPLETE"
        assert result["optimization_started"] is False
        assert result["cases_complete"] == 0
        assert not list(root.glob("cases/*/*/optimization.csv"))
    elif audit_status == "PASS":
        assert result["status"] == "OBJECTIVE_PILOT_COMPLETE"
        assert result["cases_complete"] == 2
        receipt = json.loads((root / "cases/observed_000/COMPLETE.json").read_text())
        assert len(receipt["outcomes"]) == 8
        assert all(o["applied_updates"] == o["attempts"] for o in receipt["outcomes"])
        if precision == "transport64":
            paths = list(root.glob("cases/*/*/draws_*/TRANSPORT_CONTRACT.json"))
            assert len(paths) == 16
            for path in paths:
                contract = json.loads(path.read_text())
                assert contract["checkpoint_sha256"] == sha256_file(
                    path.parent / "parameters.eqx"
                )
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
    assert "no selection or promotion" in output.stdout.lower()
    if precision == "transport64" and audit_status == "PASS":
        path = paths[0]
        original_contract = path.read_text()
        contract = json.loads(original_contract)
        contract["version"] = "historical_native"
        path.write_text(json.dumps(contract))
        from scripts.summarize_feniks_sc_drws_objective_pilot import summarize

        with pytest.raises(ValueError, match="transport checkpoint contract mismatch"):
            summarize(root)
        path.write_text(original_contract)
    detailed = root / "cases/observed_000/audit_start_0/stencils.csv"
    detailed.write_text("changed evidence")
    from scripts.summarize_feniks_sc_drws_objective_pilot import summarize

    with pytest.raises(ValueError, match="objective audit artifact changed"):
        summarize(root)
