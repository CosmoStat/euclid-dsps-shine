"""Small implementation invariants, not an additional scientific pilot campaign."""

import inspect
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from euclid_dsps.amortized.forward_population import (
    parent_from_selected,
    selected_from_parent,
)
from euclid_dsps.amortized.native_reference import (
    choose_penalty,
    log_prob,
    make_basis,
    sample_basis,
    supervised_weights,
)
from scripts import feniks_coherent_inference as pipeline
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import finish


def test_reference_is_normalized_joint_and_all_dimensions_vary():
    from scipy.stats import multivariate_normal

    x = np.random.default_rng(2).normal(size=(120, 15))
    x[:, 7] = x[:, 1] + 0.01 * x[:, 7]
    basis = make_basis(x, 3, 0.15, 1, 0.02, 3)
    np.testing.assert_allclose(basis["conditional"].sum(axis=0), 1)
    u = np.array([0.2, 0.3, 0.5])
    at = x[:4]
    expected = sum(
        w * multivariate_normal.pdf(at, mean=a, cov=0.15**2 * np.eye(15))
        for w, a in zip(basis["conditional"] @ u, x, strict=True)
    )
    np.testing.assert_allclose(np.exp(log_prob(basis, at, u)), expected, rtol=1e-10)
    draws = sample_basis(basis, np.tile(np.arange(3), 3000), 10)
    assert draws.shape == (9000, 15) and np.all(draws.std(axis=0) > 0.1)
    assert np.corrcoef(draws[:, 1], draws[:, 7])[0, 1] > 0.9


def test_parent_correction_and_supervised_selected_measure():
    u = np.array([0.5, 0.5])
    alpha = np.array([0.2, 0.8])
    v = selected_from_parent(u, alpha)
    np.testing.assert_allclose(v, [0.2, 0.8])
    np.testing.assert_allclose(parent_from_selected(v, alpha), u)
    np.testing.assert_allclose(parent_from_selected([0.5, 0.5], alpha), [0.8, 0.2])
    np.testing.assert_allclose(parent_from_selected(v, [0.4, 0.4]), v)
    # Balanced reference parent; selected reference counts proportional to alpha.
    labels = np.repeat([0, 1], [200, 800])
    learned = np.array([0.8, 0.2])
    w = supervised_weights(labels, learned, [0.5, 0.5])
    masses = np.bincount(labels, weights=w) / w.sum()
    np.testing.assert_allclose(masses, selected_from_parent(learned, alpha))
    assert w.mean() == pytest.approx(1)


def test_known_mixture_solver_recovers_selected_and_parent():
    from euclid_dsps.amortized.forward_population import fit_selected_weights

    # Exact component-observation family with negligible class overlap.
    labels = np.repeat([0, 1], [200, 800])
    probabilities = np.full((len(labels), 2), 1e-12)
    probabilities[np.arange(len(labels)), labels] = 1 - 1e-12
    v, diagnostics = fit_selected_weights(np.log(probabilities), [0.5, 0.5])
    np.testing.assert_allclose(v, [0.2, 0.8], atol=1e-5)
    np.testing.assert_allclose(
        parent_from_selected(v, [0.2, 0.8]), [0.5, 0.5], atol=1e-5
    )
    assert diagnostics["kkt_gap"] <= 2e-6


def test_penalty_uses_observations_not_parent_truth():
    values = np.array(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0]]
    )
    selected, accepted, _, _ = choose_penalty(values, [0.003, 0.01, 0.03])
    assert selected == 1 and accepted.tolist() == [True, True, False]
    src = inspect.getsource(pipeline.population)
    assert "factor_" not in src and "posterior(" not in src and "NAMES" not in src
    assert "columns=observation_columns(bands)" in inspect.getsource(pipeline.observed)
    assert "sample_" not in inspect.getsource(pipeline.posterior)


def fixture(tmp_path, monkeypatch):
    from test_feniks_clean_parent import small_settings

    from euclid_dsps.synthetic_diffsky.coherent_parent import catalogue_checks, observe

    source = tmp_path / "source"
    (source / "report").mkdir(parents=True)
    noise = dict(
        type="m5_depth",
        m5={"lsst_r": 27.5},
        gamma={"lsst_r": 0.039},
        sigma_sys_mag=0.005,
    )
    decoder = dict(bands=[dict(name="lsst_r")], model=dict(n_sfh_bins=80))
    selection = dict(band="lsst_r", max_mag_ab=29.0)
    write(source / "decoder.json", decoder)
    write(source / "noise.json", noise)
    rng = np.random.default_rng(17)
    checks, paths = {}, []
    for i, split in enumerate(("train", "validation", "test")):
        frame = pd.DataFrame(rng.uniform(-0.6, 0.6, (400, 15)), columns=pipeline.NAMES)
        frame["object_id"] = np.arange(400) + i * 1000
        frame["population_weight"] = frame["galaxy_weight"] = 1.0
        frame["split"] = split
        frame["effective_source_seed"] = i
        frame["effective_proposal_key"] = [f"{i}:{j}" for j in range(400)]
        frame = observe(
            frame,
            np.full((400, 1), 9e-32),
            decoder["bands"],
            noise,
            seed=i,
            selection=selection,
        )
        checks[split] = catalogue_checks(frame, decoder["bands"], selection)
        for kind, data in (
            ("parent", frame),
            ("selected_r29", frame[frame.selected_r29]),
        ):
            p = source / "dataset" / kind / f"{split}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(p, index=False)
            paths.append(p)
    proposals = source / "proposals.parquet"
    pool = pd.DataFrame(rng.uniform(-0.65, 0.65, (1200, 15)), columns=pipeline.NAMES)
    pool["effective_proposal_key"] = [f"0:{i}" for i in range(len(pool))]
    pool["galaxy_weight"] = np.exp(rng.normal(size=len(pool)))
    pool.to_parquet(proposals, index=False)
    write(
        source / "MANIFEST.json",
        dict(
            frozen_files={n: sha(source / n) for n in ("decoder.json", "noise.json")},
            settings=dict(
                parent_rows={s: 400 for s in ("train", "validation", "test")},
                selection=selection,
            ),
            assets={},
            sources=dict(
                train=dict(paths=[str(proposals)], source_split="train", source_seed=0)
            ),
        ),
    )
    write(
        source / "dataset/CONTRACT.json",
        dict(
            status="COHERENT_PARENT_DATASET_COMPLETE",
            ready_for_population_benchmark=True,
        ),
    )
    write(source / "report/checks.json", dict(splits=checks))
    finish(
        source / "report",
        paths + [source / "dataset/CONTRACT.json", source / "report/checks.json"],
        sha(source / "MANIFEST.json"),
    )
    monkeypatch.setattr(
        "euclid_dsps.synthetic_diffsky.coherent_parent.eligible_proposals",
        lambda p, *_: pd.read_parquet(p),
    )
    mod = types.ModuleType("euclid_dsps.prior_learning.spline15d")
    mod.project_diffsky_frame_to_spline15d = lambda frame, **_: (
        frame[["object_id", *pipeline.NAMES]].copy(),
        None,
    )
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(
        pipeline,
        "photometer",
        lambda *args: lambda theta: np.full((len(theta), 1), 9e-32),
    )
    cfg = yaml.safe_load(
        Path("configs/experiments/feniks_coherent_inference.yaml").read_text()
    )
    cfg["reference"].update(
        anchors=128,
        components=2,
        bounds={n: [-1.0, 1.0] for n in cfg["reference"]["bounds"]},
    )
    cfg["bank"].update(shards=2, rows_per_shard=1024, checkpoint_rows=512)
    cfg["posterior"].update(
        small_settings(), epochs=2, batch_size=128, validation_limit=128
    )
    cfg["classifier"].update(
        width=8, depth=1, epochs=2, batch_size=128, validation_limit=128
    )
    cfg["stopping"].update(block_epochs=1, minimum_epochs=10, patience_blocks=1)
    cfg["population"].update(min_selected=1, penalties=[0.01, 0.1])
    cfg["evaluation"].update(objects=16, draws_per_object=16, population_draws=128)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    root = tmp_path / "run"
    pipeline.prepare(source, root, config)
    return root, source, proposals


def test_reference_uniform_exclusion_and_source_integrity(tmp_path, monkeypatch):
    root, source, proposal = fixture(tmp_path, monkeypatch)
    record = read(root / "MANIFEST.json")["native_source"]
    one, _ = pipeline.uniform_anchors(record, {"0:0"}, 128, 5)
    frame = pd.read_parquet(proposal)
    frame.galaxy_weight *= 1e20
    frame.to_parquet(proposal, index=False)
    two, _ = pipeline.uniform_anchors(record, {"0:0"}, 128, 5)
    assert one.effective_proposal_key.tolist() == two.effective_proposal_key.tolist()
    assert "0:0" not in set(one.effective_proposal_key)
    pipeline.reference(root)
    identities = pd.read_csv(root / "reference/identities.csv").effective_proposal_key
    assert not set(identities) & {f"0:{i}" for i in range(400)}
    assert (
        read(root / "reference/coordinates.json")["fitted_on"]
        == "independent_unweighted_native_reference"
    )
    (source / "noise.json").write_text("{}")
    with pytest.raises(ValueError, match="source changed"):
        pipeline.settings(root)


def test_end_to_end_optimizers_report_and_idempotence(tmp_path, monkeypatch):
    root, source, _ = fixture(tmp_path, monkeypatch)
    before = {str(p): sha(p) for p in source.rglob("*") if p.is_file()}
    pipeline.reference(root)
    pipeline.report(root)
    assert (root / "report/BLOCKED.json").exists() and not (
        root / "report/FINAL.json"
    ).exists()
    for task in (0, 1):
        pipeline.bank(root, task)
    saved = sha(root / "banks/shard_000/block_0000000/bank.npz")
    (root / "banks/shard_000/FINAL.json").unlink()
    pipeline.bank(root, 0)
    assert sha(root / "banks/shard_000/block_0000000/bank.npz") == saved
    pipeline.posterior(root, oracle=True)
    pipeline.population(root)
    pipeline.posterior(root)
    pipeline.report(root)
    parent = read(root / "population/parent.json")
    assert sum(parent["u"]) == pytest.approx(1) and sum(parent["v"]) == pytest.approx(1)
    assert not parent["target_truth_used"] and not parent["population_uses_q"]
    assert not read(root / "report/FINAL.json")["ready_for_production"]
    assert read(root / "posterior/STOP.json")["reason"] == "maximum_epoch"
    for stage in ("oracle", "posterior"):
        with np.load(root / stage / "draws.npz") as f:
            assert f["draws"].shape == (16, 16, 15)
    calibration = pd.read_csv(root / "report/calibration_comparison.csv")
    assert len(calibration) == 30 and np.isfinite(calibration.coverage_68).all()
    after = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    pipeline.posterior(root, oracle=True)
    pipeline.population(root)
    pipeline.posterior(root)
    pipeline.report(root)
    assert after == {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    assert before == {str(p): sha(p) for p in source.rglob("*") if p.is_file()}


def test_launcher_resume_dag_and_no_mem(tmp_path, monkeypatch):
    root, _, _ = fixture(tmp_path, monkeypatch)
    (root / "CODE_DIR").write_text(str(Path.cwd()))
    bindir = tmp_path / "bin"
    bindir.mkdir()
    counter = tmp_path / "counter"
    counter.write_text("700")
    calls = tmp_path / "calls"
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        f'#!/bin/bash\nn=$(<{counter}); n=$((n+1)); echo "$n" > {counter}\nprintf "%s\\n" "$*" >> {calls}\necho "$n"\n'
    )
    sbatch.chmod(0o755)
    sq = bindir / "squeue"
    sq.write_text("#!/bin/bash\nexit 0\n")
    sq.chmod(0o755)
    env = dict(
        os.environ, PATH=f"{bindir}:{Path(sys.executable).parent}:{os.environ['PATH']}"
    )
    result = subprocess.run(
        ["bash", "scripts/submit_feniks_coherent_inference.sh", "--resume", str(root)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 6 and not any("--mem" in x for x in lines)
    assert (
        "--dependency=afterok:701" in lines[1]
        and "--dependency=afterok:701" in lines[2]
    )
    assert "--array=0,1%4" in lines[2]
    assert "--dependency=afterok:703" in lines[3]
    assert "--dependency=afterok:704" in lines[4]
    assert "--dependency=afterany:701:702:703:704:705" in lines[5]
    sq.write_text("#!/bin/bash\necho 701\n")
    again = subprocess.run(
        ["bash", "scripts/submit_feniks_coherent_inference.sh", "--resume", str(root)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert again.returncode and "still active" in again.stderr
    assert len(calls.read_text().splitlines()) == 6
    sq.write_text("#!/bin/bash\nexit 0\n")
    _, _, digest = pipeline.settings(root)
    for stage in ("reference", "oracle", "population", "banks/shard_000"):
        out = root / stage
        out.mkdir(parents=True, exist_ok=True)
        finish(out, [], digest)
    resumed = subprocess.run(
        ["bash", "scripts/submit_feniks_coherent_inference.sh", "--resume", str(root)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert resumed.returncode == 0, resumed.stderr
    lines = calls.read_text().splitlines()[6:]
    assert len(lines) == 3 and "--array=1%4" in lines[0]
    assert "--dependency=afterok:707" in lines[1]
    assert "--dependency=afterany:707:708" in lines[2]


def test_classifier_learns_and_resumes_optimizer(tmp_path, monkeypatch):
    import jax
    import jax.numpy as jnp

    from scripts import feniks_forward_population as shared

    rng = np.random.default_rng(7)
    x = rng.normal(size=(512, 2)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    cfg = dict(
        width=16,
        depth=1,
        seed=3,
        epochs=20,
        batch_size=64,
        learning_rate=0.01,
        fixed_validation=True,
        validation_limit=128,
    )
    stop = dict(
        block_epochs=5, minimum_epochs=100, patience_blocks=2, minimum_gain=0.001
    )
    model = shared.classifier_template(2, 2, cfg)

    def loss(net, f, t):
        return -jax.nn.log_softmax(jax.vmap(net)(f))[jnp.arange(len(t)), t]

    original = shared.supervised_fit

    def interrupt(*args):
        original(*args)
        raise RuntimeError("simulated interruption after checkpoint")

    monkeypatch.setattr(shared, "supervised_fit", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pipeline.bounded_fit(
            model, loss, (x[:384], y[:384]), (x[384:], y[384:]), cfg, stop, tmp_path
        )
    assert read(tmp_path / "RESUME.json")["epoch"] == 5
    monkeypatch.setattr(shared, "supervised_fit", original)
    fitted = pipeline.bounded_fit(
        model, loss, (x[:384], y[:384]), (x[384:], y[384:]), cfg, stop, tmp_path
    )
    logc = shared.classify(fitted, x[384:])
    assert -(logc[np.arange(128), y[384:]]).mean() < 0.2
    assert pd.read_json(tmp_path / "training.jsonl", lines=True).epoch.tolist() == list(
        range(1, 21)
    )


@pytest.mark.parametrize(
    "kind,variable", [("refinement", "REFINE"), ("inference", "INFERENCE")]
)
def test_download_guard_and_allowlist(tmp_path, kind, variable):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    ssh = bindir / "ssh"
    rsync = bindir / "rsync"
    calls = tmp_path / "calls"
    source = (
        "/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111/"
        f"avi_coherent_{kind}_20260926_010000"
    )
    ssh.write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$*" >> {calls}\nprintf "%s\\n" "$TEST_SOURCE"\n'
    )
    rsync.write_text(f'#!/bin/bash\nprintf "RSYNC %s\\n" "$*" >> {calls}\n')
    ssh.chmod(0o755)
    rsync.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bindir}:{os.environ['PATH']}",
        FENIKS_LOCAL_RESULTS=str(tmp_path / "results"),
        TEST_SOURCE=source,
    )
    command = ["bash", "scripts/rsync_feniks_coherent_results.sh", kind]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    text = calls.read_text()
    assert f'"${variable}"' in text
    assert (
        "--max-size=25m" in text and "--include=*.csv" in text and "--exclude=*" in text
    )
    assert "--include=*.npz" not in text and "--include=*.eqx" not in text
    for bad in ("", "/", "/tmp/avi_coherent_refinement_20260926_010000"):
        result = subprocess.run(
            command, env=dict(env, TEST_SOURCE=bad), capture_output=True, text=True
        )
        assert result.returncode and "Refusing unexpected" in result.stderr
    assert calls.read_text().count("RSYNC ") == 1
