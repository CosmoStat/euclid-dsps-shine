"""Small numerical invariants, not a scientific pilot campaign."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def small_settings():
    return dict(
        experts=2,
        architecture=dict(
            hidden_sizes=[8],
            flow_layers=2,
            flow_hidden_size=8,
            flow_bins=4,
            flow_tail_bound=8,
            flow_init_scale=0.03,
            residual_trunk_width=8,
            residual_blocks=1,
            residual_representation_width=8,
            residual_context_dim=8,
        ),
    )


def names():
    from euclid_dsps.amortized.forward_population import PHYSICAL

    return (*PHYSICAL, *(f"sfh_dlog_sfr_{i:02d}" for i in range(1, 11)))


def test_structured_density_is_product_and_keeps_15d():
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.structured_population import (
        StructuredPopulation,
        factor_log_prob,
        factor_template,
        transport_audit,
    )

    physical = factor_template(small_settings(), 5, 1, 4)
    conditional = factor_template(small_settings(), 10, 5, 5)
    prior = StructuredPopulation(physical, conditional, names())
    x = prior.sample(jax.random.PRNGKey(7), 32)
    assert x.shape == (32, 15)
    assert np.all(np.std(x, axis=0) > 0)
    actual = prior.log_prob(x)
    expected = factor_log_prob(
        physical, jnp.zeros((32, 1)), x[:, :5]
    ) + factor_log_prob(conditional, x[:, :5], x[:, 5:])
    np.testing.assert_allclose(actual, expected)
    assert np.isfinite(actual).all()
    assert transport_audit(physical, jnp.zeros((4, 1)), jax.random.PRNGKey(9))["passed"]
    assert transport_audit(conditional, x[:4, :5], jax.random.PRNGKey(10))["passed"]


def test_nuisance_training_cannot_change_physical_factor(tmp_path):
    import equinox as eqx
    import jax.numpy as jnp

    from euclid_dsps.amortized.structured_population import (
        factor_log_prob,
        factor_template,
    )
    from scripts.feniks_forward_population import supervised_fit

    physical = factor_template(small_settings(), 5, 1, 1)
    before = eqx.filter(physical, eqx.is_array)
    conditional = factor_template(small_settings(), 10, 5, 2)
    rng = np.random.default_rng(11)
    x = rng.normal(size=(32, 15))
    out = tmp_path / "conditional"
    out.mkdir()
    fitted = supervised_fit(
        conditional,
        lambda n, c, t: -factor_log_prob(n, c, t),
        x[:, :5],
        x[:, 5:],
        (x[:, :5], x[:, 5:]),
        dict(
            batch_size=16,
            seed=1,
            epochs=1,
            learning_rate=1e-3,
            fixed_validation=True,
            validation_limit=32,
        ),
        out,
    )
    assert np.isfinite(
        factor_log_prob(fitted, jnp.asarray(x[:, :5]), jnp.asarray(x[:, 5:]))
    ).all()
    assert eqx.tree_equal(before, eqx.filter(physical, eqx.is_array))


def test_known_parent_and_likelihood_degeneracy():
    from euclid_dsps.amortized.forward_population import PhysicalBasis
    from scripts.feniks_clean_parent import known_parent_weights, likelihood_geometry

    basis = PhysicalBasis.create(names(), components=4)
    eligible = np.array([True, True, False, True])
    u, removed = known_parent_weights(basis, eligible, 1)
    assert np.isclose(u.sum(), 1) and u[2] == 0 and removed > 0
    flat = np.log(np.full((10, 4), 0.25))
    g = likelihood_geometry(flat, np.full(4, 0.25), np.full(4, 0.25))
    assert g["near_null_directions"] == 3


def test_matched_closure_uses_fresh_simulations_without_q(tmp_path, monkeypatch):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from scipy.special import logsumexp

    import euclid_dsps.amortized.forward_population_runtime as adapter
    import scripts.feniks_clean_parent as run
    import scripts.feniks_forward_population as forward
    from euclid_dsps.amortized.forward_population import PhysicalBasis
    from euclid_dsps.amortized.latent import LatentSpec, x_to_theta
    from scripts.feniks_avi_experiments import sha, write

    parent = tmp_path / "parent"
    parent.mkdir()
    out = tmp_path / "run"
    (out / "closure").mkdir(parents=True)
    basis = PhysicalBasis.create(names(), components=4, width=0.6)
    spec = LatentSpec(names(), jnp.full(15, -10.0), jnp.full(15, 10.0))
    config = dict(seed=9, simulation_batch=32, weak_parent_mass_cap=0.05, classifier={})
    write(parent / "MANIFEST.json", dict(settings=config))
    classifier = eqx.nn.Linear(15, 4, key=jax.random.PRNGKey(4))
    cp = parent / "classifier.eqx"
    eqx.tree_serialise_leaves(cp, classifier)
    weights = parent / "weights.csv"
    pd.DataFrame(
        dict(component=range(4), c=[0.25] * 4, alpha=[0.75] * 4, eligible=[True] * 4)
    ).to_csv(weights, index=False)
    settings = dict(
        seed=20260923,
        metric_seed=5,
        metric_draws=128,
        closure=dict(
            parent_simulations=4096, shard_size=1024, minimum_selected=100, bootstraps=2
        ),
    )
    write(
        out / "MANIFEST.json",
        dict(
            parent=str(parent),
            settings=settings,
            inputs={
                k: dict(path=str(p), sha256=sha(p))
                for k, p in (("classifier", cp), ("component_weights", weights))
            },
        ),
    )
    monkeypatch.setattr(
        run, "_runtime", lambda *a: (None, SimpleNamespace(latent_spec=spec), {}, {})
    )
    monkeypatch.setattr(forward, "basis_for", lambda *a: basis)
    monkeypatch.setattr(forward, "classifier_template", lambda *a: classifier)

    def classify(_, features):
        logp = basis.component_log_prob(features)
        return logp - logsumexp(logp, axis=1, keepdims=True)

    monkeypatch.setattr(forward, "classify", classify)

    def sim(x, key):
        return dict(
            x=x,
            theta=x_to_theta(x, spec),
            features=x,
            selected=jax.random.uniform(key, (len(x),)) < 0.75,
            valid=jnp.ones(len(x), dtype=bool),
        )

    monkeypatch.setattr(adapter, "simulator", lambda *a: sim)
    run.closure(out)
    final = json.loads((out / "closure/FINAL.json").read_text())
    assert final["status"] == "CLEAN_MATCHED_CLOSURE_COMPLETE"
    assert final["sum_u"] == pytest.approx(1)
    assert final["sum_v"] == pytest.approx(1)
    assert final["population_uses_q"] is False
    table = pd.read_csv(out / "closure/weights.csv")
    assert np.abs(table.fitted_parent - table.true_parent).sum() < 0.15
    assert abs(final["alpha_true_fresh"] - 0.75) < 0.05
    assert len(pd.read_csv(out / "closure/bootstrap.csv")) == 2


def test_contract_audit_separates_numerical_and_scientific_failures(
    tmp_path, monkeypatch
):
    import jax.numpy as jnp

    import scripts.feniks_clean_parent as run
    import scripts.feniks_failure_modes as failure_modes
    from euclid_dsps.amortized.latent import LatentSpec, x_to_theta
    from scripts.feniks_avi_experiments import sha, write

    root = tmp_path / "run"
    (root / "contracts").mkdir(parents=True)
    (root / "decoder_observation").mkdir()
    parent = tmp_path / "parent"
    parent.mkdir()
    write(parent / "MANIFEST.json", dict(settings={"cut": 29.0}))
    ns = names()
    spec = LatentSpec(ns, jnp.full(15, -10.0), jnp.full(15, 10.0))
    x = np.random.default_rng(44).normal(size=(64, 15))
    truth = pd.DataFrame(np.asarray(x_to_theta(jnp.asarray(x), spec)), columns=ns)
    truth.insert(0, "parent_row_index", np.arange(len(truth)))
    truth["population_weight"] = np.linspace(1, 2, len(truth))
    truth_path = root / "truth.parquet"
    truth.to_parquet(truth_path)
    selection = pd.DataFrame(
        dict(
            parent_row_index=np.arange(len(truth)),
            selected=np.arange(len(truth)) % 2 == 0,
        )
    )
    selection_path = root / "selection.parquet"
    selection.to_parquet(selection_path)
    audit = root / "audit_indices.npy"
    np.save(audit, np.arange(16))
    settings = dict(
        transform_tolerance=1e-4,
        contracts=dict(
            decoder_objects=16,
            decoder_p95_abs_residual_sigma=0.25,
            error_p95_relative_residual=0.02,
            noise_mean_abs=0.08,
            noise_std_abs_from_one=0.08,
        ),
    )
    write(
        root / "MANIFEST.json",
        dict(
            parent=str(parent),
            settings=settings,
            inputs={
                key: dict(path=str(path), sha256=sha(path))
                for key, path in (
                    ("truth_parent", truth_path),
                    ("selection_identities", selection_path),
                    ("audit_indices", audit),
                )
            },
        ),
    )
    monkeypatch.setattr(
        run,
        "_runtime",
        lambda *a: (None, SimpleNamespace(latent_spec=spec), {}, {}),
    )

    def fake_decoder(root):
        pd.DataFrame(
            dict(
                band=["lsst_r"],
                decoder_p95_abs_sigma=[0.5],
                error_true_p95_abs_relative=[0.0],
                noise_mean=[0.0],
                noise_std=[1.0],
            )
        ).to_csv(root / "decoder_observation/band_summary.csv", index=False)
        write(
            root / "decoder_observation/FINAL.json",
            dict(
                status="DECODER_OBSERVATION_AUDIT_COMPLETE",
                selection_identity_mismatches=0,
            ),
        )

    monkeypatch.setattr(failure_modes, "decoder_observation", fake_decoder)
    run.audit_contracts(root)
    final = json.loads((root / "contracts/FINAL.json").read_text())
    assert final["numerical_integrity_pass"] is True
    assert final["decisions"]["decoder"] is False
    assert final["all_scientific_gates_pass"] is False
    assert final["status"] == "CLEAN_CONTRACT_AUDIT_COMPLETE"


def test_launcher_gates_closure_on_representation_numerics():
    script = Path("scripts/submit_feniks_clean_parent.sh").read_text()
    assert '--array="0-3%$CONCURRENCY"' in script
    assert '--dependency="afterok:$CONTRACT_JOB"' in script
    assert '--dependency="afterok:$FIT_JOB"' in script
    assert '--dependency="afterok:$CLOSURE_JOB"' in script
    assert 'record FIT_JOB "$FIT_JOB"' in script


def test_representation_fit_and_report_end_to_end(tmp_path, monkeypatch):
    import jax.numpy as jnp

    import scripts.feniks_clean_parent as run
    from euclid_dsps.amortized.latent import LatentSpec, x_to_theta
    from euclid_dsps.amortized.structured_population import factor_template
    from scripts.feniks_avi_experiments import sha, write

    parent = tmp_path / "parent"
    parent.mkdir()
    settings = dict(
        seed=21,
        metric_seed=22,
        metric_draws=64,
        replicas=2,
        exact_repeats=1,
        bootstrap_reference=2,
        transform_tolerance=0.0001,
        flow=dict(epochs=1, batch_size=16, learning_rate=0.001),
    )
    write(parent / "MANIFEST.json", dict(settings=dict(posterior=small_settings())))
    root = tmp_path / "run"
    (root / "report").mkdir(parents=True)
    (root / "closure").mkdir()
    (root / "contracts").mkdir()
    ns = names()
    spec = LatentSpec(ns, jnp.full(15, -10.0), jnp.full(15, 10.0))
    x = np.random.default_rng(15).normal(size=(64, 15))
    theta = np.asarray(x_to_theta(jnp.asarray(x), spec))
    truth = pd.DataFrame(theta, columns=ns)
    truth["population_weight"] = np.linspace(0.5, 2.0, len(x))
    truth_path = root / "truth.parquet"
    truth.to_parquet(truth_path)
    split = root / "split.npz"
    np.savez(
        split, train=np.arange(32), validation=np.arange(32, 48), test=np.arange(48, 64)
    )
    write(
        root / "MANIFEST.json",
        dict(
            parent=str(parent),
            settings=settings,
            inputs={
                k: dict(path=str(p), sha256=sha(p))
                for k, p in (("truth_parent", truth_path), ("split", split))
            },
        ),
    )
    write(
        root / "contracts/FINAL.json",
        dict(status="CLEAN_CONTRACT_AUDIT_COMPLETE"),
    )
    monkeypatch.setattr(
        run,
        "_runtime",
        lambda *a: (
            None,
            SimpleNamespace(latent_spec=spec),
            {},
            {"amortized": {"encoder": {}}},
        ),
    )
    monkeypatch.setattr(
        "euclid_dsps.amortized.forward_population_runtime.posterior_template",
        lambda *a: factor_template(small_settings(), 15, 1, 21),
    )
    # Fit one replica per arm; replicate the small artifacts only for report IO.
    import shutil

    for task, arm in ((0, "joint"), (2, "structured")):
        d = root / arm / "replica_0"
        d.mkdir(parents=True)
        run.fit(root, task)
        receipt = json.loads((d / "FINAL.json").read_text())
        assert receipt["dimensions"] == 15
        assert receipt["training_identities"] == 32
        assert receipt["parameter_count"] > 0
        shutil.copytree(d, root / arm / "replica_1")
    write(root / "closure/FINAL.json", dict(status="CLEAN_MATCHED_CLOSURE_COMPLETE"))
    np.savez(
        root / "closure/draws.npz",
        true_parent=theta,
        learned_parent=theta,
        true_selected=theta,
        learned_selected=theta,
    )
    run.report(root)
    assert (root / "report/representation_latent_x_corner.png").is_file()
    assert (root / "report/representation_physical_theta_corner.png").is_file()
    assert (root / "report/matched_parent_selected.png").is_file()
    assert not json.loads((root / "report/FINAL.json").read_text())["production_ready"]
