import json

import numpy as np
import pytest

from scripts.feniks_geometry_nuts import GROUPS, load, weights, write


def test_weights_stable_and_shift_invariant():
    a = weights(np.array([10000.0, 10001.0, -np.inf]), np.zeros(3))
    b = weights(np.array([0.0, 1.0, -np.inf]), np.zeros(3))
    np.testing.assert_allclose(a, b)
    assert a[2] == 0
    assert a.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("value", [[np.nan, 1], [np.inf, 1], [-np.inf, -np.inf]])
def test_invalid_weights_are_not_replaced_by_a_floor(value):
    with pytest.raises(ValueError):
        weights(value, np.zeros(2))


def test_identical_target_and_proposal_have_uniform_weights():
    x = np.linspace(-10, 10, 100)
    np.testing.assert_allclose(weights(x, x), np.full(100, 0.01))


def test_manifest_rejects_changed_input(tmp_path, monkeypatch):
    from euclid_dsps.amortized.population_vem import sha256_file

    p = tmp_path / "input"
    p.write_text("original")
    write(
        tmp_path / "MANIFEST.json",
        dict(code_commit="frozen", inputs={str(p): sha256_file(p)}),
    )
    monkeypatch.setattr(
        "scripts.feniks_geometry_nuts.subprocess.check_output", lambda *a, **k: "frozen"
    )
    load(tmp_path)
    p.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        load(tmp_path)


def test_sampler_and_plot_artifacts_on_gaussian(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp
    import pandas as pd

    from euclid_dsps.amortized.exact_posterior import (
        NUTSSettings,
        run_batched_nuts_chains,
    )
    from scripts.feniks_geometry_nuts import summarize_chains

    jax.config.update("jax_enable_x64", True)
    out = tmp_path / "nuts"
    out.mkdir()
    settings = NUTSSettings(warmup_steps=20, sample_chunks=(16,), max_num_doublings=3)
    run_batched_nuts_chains(
        lambda x: -0.5 * jnp.sum(x * x),
        jnp.zeros((2, 15)),
        seeds=(1, 2),
        settings=settings,
        out_dirs=(out / "chain_0", out / "chain_1"),
    )
    x = np.random.default_rng(1).normal(size=(64, 15))
    np.savez(tmp_path / "bank_0.npz", x=x, weight=np.full(64, 1 / 64))
    monkeypatch.setattr("scripts.feniks_geometry_nuts.x_to_theta", lambda x, spec: x)
    receipt = summarize_chains(
        out,
        SimpleNamespace(names=tuple(f"p{i}" for i in range(15))),
        tmp_path,
        dict(chains=2, chunks=[16], max_num_doublings=3),
    )
    assert receipt["scientific_promotion"] is False
    assert receipt["diagnostics_pass"] is False
    assert len(pd.read_csv(out / "diagnostics.csv")) == 15
    for name in ("traces.png", "marginals.png", "corner_first5.png"):
        assert (out / name).stat().st_size > 1000


def test_geometry_reuses_same_cold_starts_and_preserves_banks(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import jax.numpy as jnp

    import scripts.feniks_geometry_nuts as experiment
    from euclid_dsps.amortized.flows import StandardNormalPrior
    from euclid_dsps.amortized.posterior_target import PosteriorObservation

    m = dict(cases=["observed_000"], seed=1, replicas=2, draws_per_replica=32, chains=8)
    write(tmp_path / "MANIFEST.json", m)
    monkeypatch.setattr(experiment, "load", lambda root: m)
    prior = StandardNormalPrior(latent_dim=15)
    r = SimpleNamespace(
        latent_spec=SimpleNamespace(names=tuple(f"p{i}" for i in range(15))),
        model=SimpleNamespace(prior=prior),
        model_args=(jnp.ones(2),),
    )
    obs = PosteriorObservation(
        jnp.ones((1, 10)), jnp.ones((1, 10)), jnp.ones((1, 10), bool)
    )

    def target(x):
        p = prior.log_prob(x)[:, None]
        return SimpleNamespace(
            loglike=p, logprior=p, logtarget=2 * p, model_flux=jnp.ones((len(x), 1, 10))
        )

    monkeypatch.setattr(
        experiment,
        "runtime",
        lambda *args: (r, obs, None, None, None, target, target, prior),
    )

    def draw(enc, params, ctx, key, n):
        x = prior.sample(key, n)[:, None]
        return x, prior.log_prob(x)

    monkeypatch.setattr(experiment, "sample", draw)
    monkeypatch.setattr(
        experiment, "log_prob", lambda enc, params, ctx, x: prior.log_prob(x)
    )
    experiment.geometry(tmp_path)
    case = tmp_path / "observed_000"
    np.testing.assert_array_equal(
        np.load(case / "starts_A.npy"), np.load(case / "starts_C.npy")
    )
    assert not np.array_equal(
        np.load(case / "starts_A.npy"), np.load(case / "starts_B.npy")
    )
    receipt = json.loads((tmp_path / "GEOMETRY_COMPLETE.json").read_text())
    assert "observed_000/bank_0.npz" in receipt["artifacts"]
    assert "observed_000/target_gradient_checks.csv" in receipt["artifacts"]


def test_three_groups_and_strict_json(tmp_path):
    assert GROUPS == ("A", "B", "C")
    with pytest.raises(ValueError):
        write(tmp_path / "bad.json", {"ess": np.nan})
    write(tmp_path / "good.json", {"scientific_promotion": False})
    assert (
        json.loads((tmp_path / "good.json").read_text())["scientific_promotion"]
        is False
    )
