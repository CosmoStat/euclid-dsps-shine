import json
from pathlib import Path

import numpy as np
import pytest

from scripts.feniks_geometry_nuts import (
    GROUPS,
    NUTS_PROFILES,
    _profile_tasks,
    load,
    weights,
    write,
)


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
    settings = NUTSSettings(warmup_steps=20, sample_chunks=(8, 8), max_num_doublings=3)
    import euclid_dsps.amortized.exact_posterior as sampler

    runners = []
    original = sampler._make_batched_nuts_runner

    def factory(*args):
        runner = original(*args)
        runners.append(runner)
        return runner

    monkeypatch.setattr(sampler, "_make_batched_nuts_runner", factory)

    def density(x):
        assert x.dtype == jnp.float64
        return -0.5 * jnp.sum(x * x)

    run_batched_nuts_chains(
        density,
        jnp.zeros((2, 15)),
        seeds=(1, 2),
        settings=settings,
        out_dirs=(out / "chain_0", out / "chain_1"),
        target_dtype="float64",
    )
    assert len(runners) == 1
    assert runners[0]._cache_size() == 1
    with pytest.raises(ValueError, match="Incompatible"):
        run_batched_nuts_chains(
            density,
            jnp.zeros((2, 15)),
            seeds=(1, 2),
            settings=settings,
            out_dirs=(out / "chain_0", out / "chain_1"),
            target_dtype="float32",
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


def test_target_wrapper_preserves_sub_float32_displacements():
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.exact_posterior import _float64_logdensity

    jax.config.update("jax_enable_x64", True)
    def target(x):
        return jnp.sum(x)
    full = jax.jit(_float64_logdensity(target, target_dtype="float64"))
    old = jax.jit(_float64_logdensity(target))
    x = jnp.array([1.0], dtype=jnp.float64)
    assert float(full(x + 1e-10) - full(x)) == pytest.approx(1e-10, rel=1e-5)
    assert float(old(x + 1e-10) - old(x)) == 0


def test_geometry_import_is_immutable_and_hash_checked(tmp_path, monkeypatch):
    from euclid_dsps.amortized.population_vem import sha256_file
    from scripts.feniks_geometry_nuts import prepare_nuts

    source = tmp_path / "geometry"
    source.mkdir()
    (source / "starts.npy").write_bytes(b"frozen starts")
    old_settings = dict(
        chains=8,
        warmup=1000,
        chunks=[512] * 8,
        max_num_doublings=10,
    )
    write(
        source / "MANIFEST.json",
        dict(code_commit="old", inputs={}, **old_settings),
    )
    write(
        source / "GEOMETRY_COMPLETE.json",
        dict(
            status="GEOMETRY_COMPLETE",
            manifest_sha256=sha256_file(source / "MANIFEST.json"),
            artifacts={"starts.npy": sha256_file(source / "starts.npy")},
        ),
    )
    before = (source / "MANIFEST.json").read_bytes()
    monkeypatch.setattr(
        "scripts.feniks_geometry_nuts.subprocess.check_output", lambda *a, **k: "new"
    )
    dest = tmp_path / "nuts"
    prepare_nuts(source, dest)
    assert (source / "MANIFEST.json").read_bytes() == before
    m = json.loads((dest / "MANIFEST.json").read_text())
    assert m["nuts_target_dtype"] == "float64" and m["code_commit"] == "new"
    assert m["nuts_execution_profile"]["name"] == "float64_depth4_parallel_v1"
    assert m["chains"] == 8
    assert m["warmup"] == 1000
    assert m["chunks"] == [512] * 8
    assert m["max_num_doublings"] == 4
    assert m["target_accept"] == 0.9
    assert m["geometry_proposed_nuts_settings"] == old_settings
    assert (dest / "starts.npy").read_bytes() == b"frozen starts"
    (source / "starts.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        prepare_nuts(source, tmp_path / "bad")


def test_recovery_launcher_uses_all_targets_and_bounded_depth():
    launcher = Path("scripts/submit_feniks_geometry_nuts.sh").read_text()
    wrapper = Path("scripts/feniks_geometry_nuts.slurm").read_text()
    assert "nuts-probe-new" in launcher
    assert "nuts-long-new" in launcher
    assert '--array="0-${ARRAY_LAST}%${NUTS_ARRAY_CONCURRENCY}"' in launcher
    assert '--time="$NUTS_TIME"' in launcher
    assert "warmup_progress=opaque" in wrapper


def test_dense_followup_profiles_are_b_only_and_bounded():
    probe = NUTS_PROFILES["float64_dense_depth56_probe_v1"]
    long = NUTS_PROFILES["float64_dense_depth6_long_v1"]
    probe_tasks = _profile_tasks(probe)
    long_tasks = _profile_tasks(long)
    assert len(probe_tasks) == 8
    assert {row["group"] for row in probe_tasks} == {"B"}
    assert {row["max_num_doublings"] for row in probe_tasks} == {5, 6}
    assert len({row["variant"] for row in probe_tasks}) == 2
    assert probe["mass_matrix"] == "dense"
    assert len(long_tasks) == 4
    assert {row["max_num_doublings"] for row in long_tasks} == {6}
    assert long["warmup"] == 1500
    assert sum(long["chunks"]) == 4096
    observed = NUTS_PROFILES["float64_dense_depth6_observed8_v1"]
    assert len(_profile_tasks(observed, tuple(f"observed_{i:03d}" for i in range(8)))) == 8


def test_simulation_truth_is_display_only(tmp_path, monkeypatch):
    import jax.numpy as jnp

    from scripts.feniks_geometry_nuts import simulation_truth_theta

    truth = np.arange(30, dtype=float).reshape(2, 15)
    np.savez(tmp_path / "SIMULATED_INPUTS.npz", generated_x=truth)
    monkeypatch.setattr(
        "scripts.feniks_geometry_nuts.x_to_theta", lambda x, spec: jnp.asarray(x)
    )
    spec = type("Spec", (), {"names": tuple(f"p{i}" for i in range(15))})()
    np.testing.assert_array_equal(
        simulation_truth_theta({"reference": str(tmp_path)}, "simulated_001", spec),
        truth[1],
    )
    assert simulation_truth_theta(
        {"reference": str(tmp_path)}, "observed_001", spec
    ) is None


def test_observed_launcher_chains_geometry_before_eight_nuts_tasks():
    launcher = Path("scripts/submit_feniks_observed_nuts.sh").read_text()
    assert "prepare-observed" in launcher
    assert '--dependency="afterok:${GEOMETRY_JOB}"' in launcher
    assert '--array="0-7%${NUTS_ARRAY_CONCURRENCY}"' in launcher
    assert "--kill-on-invalid-dep=yes" in launcher


def test_observed_cohort_summary_uses_no_truth(tmp_path, monkeypatch):
    import jax.numpy as jnp

    from scripts.feniks_geometry_nuts import write_observed_cohort_summary

    reference = tmp_path / "reference"
    reference.mkdir()
    np.save(reference / "observed_rows.npy", np.array([10, 20]))
    for index in range(2):
        case = tmp_path / f"observed_{index:03d}"
        case.mkdir()
        x = np.tile(np.arange(15, dtype=float), (16, 1)) + index
        np.savez(case / "bank_0.npz", x=x)
        np.savez(
            case / "observation.npz",
            flux=np.ones((1, 3)) * (index + 1),
            flux_err=np.ones((1, 3)),
            mask=np.ones((1, 3), dtype=bool),
        )
    monkeypatch.setattr(
        "scripts.feniks_geometry_nuts.x_to_theta", lambda x, spec: jnp.asarray(x)
    )
    spec = type("Spec", (), {"names": tuple(f"p{i}" for i in range(15))})()
    path = write_observed_cohort_summary(
        tmp_path,
        {
            "reference": str(reference),
            "cases": ["observed_000", "observed_001"],
        },
        spec,
    )
    frame = __import__("pandas").read_csv(path)
    assert len(frame) == 2
    assert not frame["truth_used"].any()
    assert set(frame["source_row"]) == {10, 20}
