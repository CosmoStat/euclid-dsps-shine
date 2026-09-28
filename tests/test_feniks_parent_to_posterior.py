"""Small end-to-end invariants; DSPS mocked, actual flow optimizer and draws."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import feniks_parent_to_posterior as ptp
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import complete, finish


def source_fixture(tmp_path):
    from euclid_dsps.amortized.coherent_coordinates import fit_coordinates, to_x
    from euclid_dsps.amortized.features import (
        compute_feature_stats,
        feature_stats_to_json,
    )
    from euclid_dsps.amortized.native_reference import make_basis
    from euclid_dsps.synthetic_diffsky.coherent_parent import observe

    source, dataset = tmp_path / "parent", tmp_path / "dataset"
    source.mkdir()
    dataset.mkdir()
    decoder = dict(bands=[dict(name="lsst_r")])
    noise = dict(
        type="m5_depth",
        m5={"lsst_r": 27.5},
        gamma={"lsst_r": 0.039},
        sigma_sys_mag=0.005,
    )
    selection = dict(band="lsst_r", max_mag_ab=29.0)
    write(dataset / "decoder.json", decoder)
    write(dataset / "noise.json", noise)
    write(
        dataset / "MANIFEST.json", dict(assets={}, settings=dict(selection=selection))
    )
    rng = np.random.default_rng(10)
    theta = rng.uniform(-0.6, 0.6, (512, 15))
    theta[:, 3] = np.exp(theta[:, 3])
    frame = observe(
        pd.DataFrame(theta, columns=ptp.NAMES),
        np.full((512, 1), 9e-32),
        decoder["bands"],
        noise,
        seed=23,
        selection=selection,
    )
    (dataset / "dataset/selected_r29").mkdir(parents=True)
    frame[frame.selected_r29].to_parquet(
        dataset / "dataset/selected_r29/test.parquet", index=False
    )
    cfg = dict(seed=19, reference=dict(components=2))
    (source / "experiment.yaml").write_text(yaml.safe_dump(cfg))
    write(
        source / "MANIFEST.json",
        dict(
            source=str(dataset),
            selection=selection,
            source_files={},
            settings=cfg,
            frozen_files={"experiment.yaml": sha(source / "experiment.yaml")},
            population_uses_q=False,
            population_uses_target_truth=False,
        ),
    )
    digest = sha(source / "MANIFEST.json")
    ref = source / "reference"
    ref.mkdir()
    spec = fit_coordinates(
        theta, ptp.NAMES, np.full(15, -1.0), np.ones(15), positive=["dust_av"]
    )
    write(ref / "coordinates.json", spec)
    stats = compute_feature_stats(
        frame[["flux_lsst_r"]].to_numpy(),
        frame[["fluxerr_lsst_r"]].to_numpy(),
        frame[["mask_lsst_r"]].to_numpy(),
        ("lsst_r",),
        append_mask=True,
    )
    write(ref / "feature_stats.json", feature_stats_to_json(stats))
    np.savez(
        ref / "basis.npz",
        **make_basis(np.asarray(to_x(theta[:128], spec)), 2, 0.1, 1, 0.02, 4),
    )
    finish(ref, list(ref.iterdir()), digest)
    pop = source / "population"
    pop.mkdir()
    u, alpha = np.array([0.8, 0.2]), np.array([0.3, 0.8])
    v = u * alpha / (u @ alpha)
    write(pop / "parent.json", dict(u=u.tolist(), v=v.tolist(), alpha=alpha.tolist()))
    finish(pop, [pop / "parent.json"], digest)
    (source / "qualification").mkdir()
    finish(source / "qualification", [], digest, passed=False)
    (source / "report").mkdir()
    write(
        source / "report/DECISION.json",
        dict(
            gates=dict(
                physical_marginals=False,
                parent_physical_sw=False,
                classifier_ratio_moment=False,
            ),
            strict_qualification_passed=False,
            ready_for_production=False,
            ready_for_fresh_posterior_benchmark=False,
        ),
    )
    finish(source / "report", [source / "report/DECISION.json"], digest)
    return source, dataset


def configuration(tmp_path):
    from test_feniks_clean_parent import small_settings

    cfg = yaml.safe_load(
        Path("configs/experiments/feniks_parent_to_posterior.yaml").read_text()
    )
    cfg["bank"].update(
        shards=2,
        rows_per_shard=512,
        checkpoint_rows=256,
        minimum_training_rows=1,
        minimum_validation_rows=1,
        role_probabilities=[0.6, 0.2, 0.2],
    )
    cfg["posterior"].update(
        small_settings(), epochs=2, batch_size=128, validation_limit=64
    )
    cfg["stopping"].update(block_epochs=1, minimum_epochs=2, patience_blocks=1)
    cfg["evaluation"].update(objects=16, draws_per_object=16)
    path = tmp_path / "posterior.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_requires_explicit_exploration_and_preserves_parent_failures(tmp_path):
    source, _ = source_fixture(tmp_path)
    config = configuration(tmp_path)
    root = tmp_path / "new"
    with pytest.raises(ValueError, match="--exploratory"):
        ptp.initialize(source, root, config)
    assert not root.exists()
    ptp.initialize(source, root, config, True)
    assert sha(root / "frozen_parent.json") == sha(source / "population/parent.json")
    assert sha(root / "PARENT_DECISION.json") == sha(source / "report/DECISION.json")
    assert not read(root / "MANIFEST.json")["production_promotion"]
    write(root / "frozen_parent.json", dict(u=[0.5, 0.5]))
    with pytest.raises(ValueError, match="configuration changed"):
        ptp.settings(root)


def test_fresh_direct_bank_resume_and_role_independence(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path)
    root = tmp_path / "new"
    ptp.initialize(source, root, configuration(tmp_path), True)
    calls = []

    def interrupted(theta):
        calls.append(theta.copy())
        if len(calls) == 2:
            raise RuntimeError("injected simulator interruption")
        return np.full((len(theta), 1), 9e-32)

    monkeypatch.setattr(ptp, "photometer", lambda *_: interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        ptp.bank(root, 0)
    block = root / "banks/shard_000/block_0000000/bank.npz"
    saved = sha(block)
    monkeypatch.setattr(
        ptp, "photometer", lambda *_: lambda t: np.full((len(t), 1), 9e-32)
    )
    ptp.bank(root, 0)
    ptp.bank(root, 1)
    assert sha(block) == saved
    _, cfg, digest = ptp.settings(root)
    data = [ptp.selected_rows(root, cfg, digest, i) for i in range(3)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert not np.intersect1d(data[a]["row_id"], data[b]["row_id"]).size
    for d in data:
        assert d["theta"].shape[1] == 15 and np.std(d["theta"][:, 5:], axis=0).min() > 0
    labels, seeds = [], []
    for path in (root / "banks").glob("shard_*/block_*/bank.npz"):
        with np.load(path) as f:
            labels.extend(f["component"])
        seeds.extend(read(path.parent / "FINAL.json")["seeds"])
    assert len(set(seeds)) == len(seeds)
    assert abs(np.mean(np.array(labels) == 0) - 0.8) < 0.05  # u, not selected v=.6
    monkeypatch.setattr(
        ptp, "photometer", lambda *_: pytest.fail("completed shard regenerated")
    )
    ptp.bank(root, 0)


def test_actual_small_flow_training_two_cohorts_report_and_immutability(
    tmp_path, monkeypatch
):
    from scripts import feniks_coherent_inference as ci

    source, dataset = source_fixture(tmp_path)
    before = {
        p: sha(p)
        for folder in (source, dataset)
        for p in folder.rglob("*")
        if p.is_file()
    }
    root = tmp_path / "new"
    ptp.initialize(source, root, configuration(tmp_path), True)
    monkeypatch.setattr(
        ptp, "photometer", lambda *_: lambda t: np.full((len(t), 1), 9e-32)
    )
    for task in range(2):
        ptp.bank(root, task)
    for name in ("population", "posterior", "load_bank", "supervised_weights"):
        monkeypatch.setattr(
            ci,
            name,
            lambda *a, **kw: pytest.fail(
                "old weighted/posterior/population route called"
            ),
        )
    original = ci.bounded_fit

    def checked_fit(model, loss, train, validation, *args):
        _, cfg, digest = ptp.settings(root)
        for role, actual in enumerate((train, validation)):
            expected = ptp.selected_rows(root, cfg, digest, role)
            np.testing.assert_array_equal(actual[1], expected["x"])
            assert actual[1].shape[1] == 15  # no weight column or q-generated target
        return original(model, loss, train, validation, *args)

    monkeypatch.setattr(ci, "bounded_fit", checked_fit)
    ptp.train(root)
    measure = read(root / "posterior/training_measure.json")
    assert measure["weights"] == "unit" and not measure["target_catalogue_truth_used"]
    assert measure["effective_training_rows"] == measure["training_rows"]
    assert read(root / "posterior/transport.json")["passed"]
    ptp.evaluate(root)
    ptp.report(root)
    for label in ("in_model", "coherent_target"):
        with np.load(root / "evaluation" / label / "draws.npz") as f:
            assert f["draws"].shape == (16, 16, 15)
            assert len(np.unique(f["evaluation_positions"])) == 16
        assert (root / "report" / f"{label}_physical_pit_truth.png").is_file()
    result = read(root / "report/DECISION.json")
    assert not result["ready_for_production"] and not result["parent_updated"]
    assert not result["inherited_parent_decision"]["strict_qualification_passed"]
    assert set(result["calibration_flags"]["in_model"]) == {
        "core_z_mass_metal",
        "dust",
        "sfh",
    }
    assert len(pd.read_csv(root / "report/calibration.csv")) == 30
    bank_measure = read(root / "report/bank_measure.json")
    assert bank_measure["parent_draws"] == 1024
    assert len(bank_measure["selected_by_role"]) == 3
    assert (root / "report/coherent_target_individual_00_physical_corner.png").exists()
    watcher = subprocess.run(
        ["bash", "scripts/watch_feniks_parent_to_posterior.sh", str(root), "--once"],
        capture_output=True,
        text=True,
        env=dict(
            os.environ, PATH=f"{Path(sys.executable).parent}:" + os.environ["PATH"]
        ),
    )
    assert watcher.returncode == 0, watcher.stderr
    assert (
        "2/2 complete" in watcher.stdout and "no production approval" in watcher.stdout
    )
    assert before == {p: sha(p) for p in before}
    posterior_sha = sha(root / "posterior/best.eqx")
    monkeypatch.setattr(
        ci, "bounded_fit", lambda *a, **kw: pytest.fail("completed training repeated")
    )
    ptp.train(root)
    ptp.evaluate(root)
    ptp.report(root)
    assert sha(root / "posterior/best.eqx") == posterior_sha
    assert complete(root / "report", sha(root / "MANIFEST.json"))


@pytest.mark.parametrize("kind", ["parent_to_posterior", "overnight_recovery"])
def test_download_uses_guarded_parent_to_posterior_root_and_excludes_large_arrays(
    tmp_path,
    kind,
):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls"
    for name, command in (
        ("ssh", 'printf "%s\\n" "$TEST_SOURCE"'),
        ("rsync", f'printf "%s\\n" "$*" >> {calls}'),
    ):
        script = bindir / name
        script.write_text(f"#!/bin/bash\n{command}\n")
        script.chmod(0o755)
    source = (
        "/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111/"
        f"avi_{kind}_20260927_220000"
    )
    env = dict(
        os.environ,
        PATH=f"{bindir}:" + os.environ["PATH"],
        TEST_SOURCE=source,
        FENIKS_LOCAL_RESULTS=str(tmp_path / "download"),
    )
    command = [
        "bash",
        "scripts/rsync_feniks_coherent_results.sh",
        kind,
    ]
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = calls.read_text()
    assert (
        "--max-size=25m" in args
        and "--exclude=banks/" in args
        and "--exclude=*" in args
    )
    assert "--include=*.eqx" not in args and "--include=*.npz" not in args
    for bad in ("", "/", "/tmp/avi_parent_to_posterior_20260927_220000"):
        result = subprocess.run(
            command, env=dict(env, TEST_SOURCE=bad), text=True, capture_output=True
        )
        assert result.returncode != 0 and "Refusing unexpected" in result.stderr
    assert calls.read_text() == args


def test_report_blocks_and_invalid_sizes_do_not_silently_shrink(tmp_path):
    source, _ = source_fixture(tmp_path)
    root = tmp_path / "new"
    config = configuration(tmp_path)
    ptp.initialize(source, root, config, True)
    ptp.report(root)
    assert read(root / "report/BLOCKED.json")["missing"] == ["posterior", "evaluation"]
    assert not (root / "report/FINAL.json").exists()
    cfg = yaml.safe_load(config.read_text())
    cfg["posterior"]["initial_checkpoint"] = "old.eqx"
    with pytest.raises(ValueError, match="from scratch"):
        ptp.validate_config(cfg)


def test_submit_freezes_code_no_mem_no_scancel_and_resume_guards(tmp_path):
    source, dataset = source_fixture(tmp_path)
    before = {
        p: sha(p)
        for folder in (source, dataset)
        for p in folder.rglob("*")
        if p.is_file()
    }
    root = tmp_path / "avi_parent_to_posterior_20260927_210000"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls, counter = tmp_path / "calls", tmp_path / "counter"
    counter.write_text("900")
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        f'#!/bin/bash\nn=$(<{counter}); n=$((n+1)); echo "$n" > {counter}\nprintf "%s\\n" "$*" >> {calls}\necho "$n"\n'
    )
    sbatch.chmod(0o755)
    squeue = bindir / "squeue"
    squeue.write_text('#!/bin/bash\necho "${LIVE:-}"\nexit "${QUEUE_STATUS:-0}"\n')
    squeue.chmod(0o755)
    scancel = bindir / "scancel"
    scancel.write_text("#!/bin/bash\nexit 99\n")
    scancel.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bindir}:{Path(sys.executable).parent}:" + os.environ["PATH"],
        SCRATCH=str(tmp_path / "scratch"),
    )
    command = ["bash", "scripts/submit_feniks_parent_to_posterior.sh"]
    result = subprocess.run(
        command
        + ["--exploratory", str(source), str(root), str(configuration(tmp_path))],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert (
        sha(Path(str(root) + ".code.tar")) == (root / "CODE_SHA256").read_text().strip()
    )
    assert (tmp_path / "avi_parent_to_posterior_latest.env").is_file()
    lines = calls.read_text().splitlines()
    assert len(lines) == 4
    assert "--array=0,1%4" in lines[0] and "--gres=gpu:1" in lines[0]
    assert "--dependency=afterok:901" in lines[1]
    assert "--dependency=afterok:902" in lines[2]
    assert "--dependency=afterany:901:902:903" in lines[3]
    assert "--partition=cpu_p1" in lines[3] and "--gres" not in lines[3]
    assert "--mem" not in "\n".join(lines)
    for patch in ({"LIVE": "901_1"}, {"QUEUE_STATUS": "1"}):
        result = subprocess.run(
            command + ["--resume", str(root)],
            env=dict(env, **patch),
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0
        assert len(calls.read_text().splitlines()) == 4
    # Missing-only restart: a completed shard is neither simulated nor submitted again.
    shard = root / "banks/shard_000"
    shard.mkdir(parents=True)
    finish(shard, [], sha(root / "MANIFEST.json"))
    result = subprocess.run(
        command + ["--resume", str(root)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert "--array=1%4" in calls.read_text().splitlines()[4]
    assert before == {p: sha(p) for p in before}
