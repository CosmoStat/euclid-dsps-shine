"""Small invariant and orchestration checks, not a scientific recovery result."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import feniks_parent_recovery as recovery
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import finish


def test_known_parent_weighting_must_use_rejected_parent_counts():
    labels = np.repeat([0, 1], [20, 100])
    weights = recovery.row_weights(labels, [0.5, 0.5], [100, 200])
    np.testing.assert_allclose(np.bincount(labels, weights=weights), [2 / 7, 5 / 7])
    with pytest.raises(ValueError):
        recovery.row_weights(labels, [0.5, 0.5], [0, 200])


def test_exact_label_ratios_recover_parent_and_penalty_is_explicit():
    labels = np.repeat([0, 1], [200, 800])
    logc = np.where(labels[:, None] == np.arange(2), 0.0, -700.0)
    parent = dict(
        alpha=[0.2, 0.8],
        classifier_reference_frequencies=[0.2, 0.8],
        selected_penalty=0.003,
    )
    controls, diagnostics, effective = recovery.fit_controls(
        logc,
        labels,
        [1000, 1000],
        [0.7, 0.3],
        parent,
        [True, True],
        {"weak_parent_mass": 0.05},
    )
    np.testing.assert_allclose(controls["label_unpenalized"], [0.7, 0.3])
    np.testing.assert_allclose(
        controls["classifier"], controls["label_penalized"], atol=2e-5
    )
    np.testing.assert_allclose(controls["classifier"], [0.7, 0.3], atol=0.003)
    assert effective > 400
    assert diagnostics["classifier"]["kkt_gap"] <= 2e-6


def test_dense_common_support_metrics_zero_for_same_density_and_use_all_15d():
    theta = np.random.default_rng(4).normal(size=(200, 15))
    known = np.ones(200) / 200
    shifted = np.exp(theta[:, 3])
    one, joint = recovery.common_distances(
        theta, known, {"same": known, "shifted": shifted}, directions=8, seed=5
    )
    assert len(one) == 30 and set(one.group) == {"physical", "sfh"}
    assert one.loc[one.model.eq("same"), "w1_over_truth_iqr"].max() == 0
    assert joint.loc[joint.model.eq("same"), "sliced_wasserstein"].max() == 0
    assert joint.loc[joint.model.eq("shifted"), "sliced_wasserstein"].min() > 0


def test_ratio_uncertainty_does_not_refit_and_is_conditionally_labelled():
    c = np.array([0.2, 0.8])
    result = recovery.ratio_uncertainty(
        np.tile(np.log(c), (200, 1)),
        c,
        np.repeat([0, 1], [40, 160]),
        blocks=8,
        resamples=32,
        seed=4,
    )
    assert result["median_relative_error"] < 1e-12
    assert result["centered_sampling_noise_median_q95"] < 1e-12
    assert result["conditional_on_frozen_calibration"]
    assert not result["full_ratio_validation"]


def make_source(tmp_path):
    import equinox as eqx

    from scripts.feniks_forward_population import classifier_template

    source, dataset = tmp_path / "source", tmp_path / "dataset"
    source.mkdir()
    dataset.mkdir()
    cfg = dict(
        seed=19,
        reference=dict(components=3),
        bank=dict(shards=2, rows_per_shard=1026, checkpoint_rows=1026),
        classifier=dict(width=8, depth=1),
        population=dict(weak_parent_mass=0.05),
        parent_contracts=dict(
            maximum_physical_sw=0.05, maximum_physical_marginal_w1=0.1
        ),
    )
    (source / "experiment.yaml").write_text(yaml.safe_dump(cfg))
    write(
        source / "MANIFEST.json",
        dict(
            source=str(dataset),
            settings=cfg,
            source_files={},
            frozen_files={"experiment.yaml": sha(source / "experiment.yaml")},
        ),
    )
    digest = sha(source / "MANIFEST.json")
    rng = np.random.default_rng(42)
    for task in range(2):
        shard = source / "banks" / f"shard_{task:03d}"
        block = shard / "block_0000000"
        block.mkdir(parents=True)
        labels = np.arange(1026) % 3
        theta = rng.normal(size=(1026, 15)) + labels[:, None] * 0.2
        selected = rng.random(1026) < np.array([0.2, 0.5, 0.8])[labels]
        np.savez(
            block / "bank.npz",
            component=labels,
            role=rng.choice([3, 4], 1026),
            selected=selected,
            theta=theta,
            features=theta[:, :3],
            flux=np.exp(theta[:, :1]) * 1e-30,
        )
        finish(block, [block / "bank.npz"], digest)
        finish(shard, [block / "FINAL.json"], digest)
    population = source / "population"
    population.mkdir()
    alpha, u = np.array([0.2, 0.5, 0.8]), np.array([0.2, 0.5, 0.3])
    v = u * alpha / (u @ alpha)
    write(
        population / "parent.json",
        dict(
            alpha=alpha.tolist(),
            u=u.tolist(),
            v=v.tolist(),
            selected_penalty=0.003,
            alpha_parent=float(u @ alpha),
            classifier_reference_frequencies=(alpha / alpha.sum()).tolist(),
            logit_offsets=[0, 0, 0],
        ),
    )
    write(population / "STOP.json", dict(reason="fixture_checkpoint"))
    pd.DataFrame(dict(component=np.arange(3), eligible=[True] * 3)).to_csv(
        population / "weights.csv", index=False
    )
    model = classifier_template(3, 3, dict(cfg["classifier"], seed=20))
    eqx.tree_serialise_leaves(population / "best.eqx", model)
    finish(population, list(population.iterdir()), digest)
    (source / "qualification").mkdir()
    write(source / "qualification/diagnostic_weights.json", dict(u=[0.7, 0.3, 0]))
    finish(
        source / "qualification",
        [source / "qualification/diagnostic_weights.json"],
        digest,
    )
    (source / "report").mkdir()
    finish(source / "report", [], digest, ready_for_production=False)
    write(dataset / "decoder.json", dict(bands=[dict(name="lsst_r")]))
    (dataset / "dataset/selected_r29").mkdir(parents=True)
    pd.DataFrame(dict(flux_lsst_r=np.exp(rng.normal(size=300)) * 1e-30)).to_parquet(
        dataset / "dataset/selected_r29/validation.parquet"
    )
    return source, dataset


def configuration(tmp_path):
    cfg = yaml.safe_load(
        Path("configs/experiments/feniks_parent_recovery.yaml").read_text()
    )
    cfg.update(
        minimum_effective_rows=1,
        evaluation_directions=4,
        ratio_blocks=4,
        ratio_resamples=8,
    )
    path = tmp_path / "diagnostic.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_end_to_end_frozen_checkpoint_disjoint_roles_and_resume(tmp_path, monkeypatch):
    from scripts import feniks_coherent_inference as ci
    from scripts import feniks_forward_population as forward

    source, dataset = make_source(tmp_path)
    before = {
        p: sha(p)
        for folder in (source, dataset)
        for p in folder.rglob("*")
        if p.is_file()
    }
    root = tmp_path / "diagnostic"
    recovery.initialize(source, root, configuration(tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("No training, q or DSPS allowed")

    monkeypatch.setattr(ci, "photometer", forbidden)
    monkeypatch.setattr(ci, "posterior", forbidden)
    monkeypatch.setattr(ci, "bounded_fit", forbidden)
    monkeypatch.setattr(forward, "supervised_fit", forbidden)
    real_read = pd.read_parquet

    def flux_only(path, *args, **kwargs):
        assert kwargs["columns"] == ["flux_lsst_r"]
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", flux_only)
    recovery.report(root)
    assert read(root / "report/BLOCKED.json")["missing"] == ["cache", *recovery.CASES]
    recovery.cache(root)
    with (
        np.load(root / "cache/fit.npz") as fit,
        np.load(root / "cache/evaluation.npz") as evaluation,
    ):
        assert not np.intersect1d(fit["row_id"], evaluation["row_id"]).size
        assert evaluation["theta"].shape[1] == 15
    with monkeypatch.context() as patch:
        patch.setattr(
            recovery,
            "tails",
            lambda *a: (_ for _ in ()).throw(RuntimeError("after fit")),
        )
        with pytest.raises(RuntimeError, match="after fit"):
            recovery.run_case(root, 0)
    assert (root / "learned_parent/fit/FINAL.json").exists()
    with monkeypatch.context() as patch:
        patch.setattr(recovery, "fit_controls", forbidden)
        recovery.run_case(root, 0)
    recovery.run_case(root, 1)
    recovery.report(root)
    decision = read(root / "report/DECISION.json")
    assert (
        not decision["ready_for_production"]
        and not decision["production_prior_modified"]
    )
    assert len(decision["cases"]) == 2
    assert not (root / "report/BLOCKED.json").exists()
    assert before == {p: sha(p) for p in before}
    saved = {p: sha(p) for p in root.rglob("*") if p.is_file()}
    with monkeypatch.context() as patch:
        patch.setattr(forward, "classify", forbidden)
        recovery.cache(root)
        recovery.run_case(root, 0)
        recovery.run_case(root, 1)
        recovery.report(root)
    assert saved == {p: sha(p) for p in saved}
    watched = subprocess.run(
        ["bash", "scripts/watch_feniks_parent_recovery.sh", str(root), "--once"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "DONE" in watched.stdout and "artifacts" not in watched.stdout
    write(source / "population/STOP.json", dict(changed=True))
    with pytest.raises(ValueError, match="source changed"):
        recovery.schedule(root)


def test_configuration_rejects_role_leakage_and_existing_root(tmp_path):
    source, _ = make_source(tmp_path)
    path = configuration(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["evaluation_role"] = 4
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="disjoint"):
        recovery.initialize(source, tmp_path / "new", path)
    with pytest.raises(ValueError, match="new diagnostic"):
        recovery.initialize(source, source, path)


def test_resume_launcher_only_missing_jobs_and_active_query_fail_closed(tmp_path):
    root = tmp_path / "audit"
    (root / "logs").mkdir(parents=True)
    (root / "INPUT.env").write_text(f'export FENIKS_RECOVERY_CODE="{Path.cwd()}"\n')
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python").write_text("""#!/bin/bash
printf 'CACHE_MINUTES=25\nCASE_MINUTES=25\nREPORT_MINUTES=5\nCPU_THREADS=8\nCASE_CONCURRENCY=2\n'
printf 'NEED_CACHE=%s\nMISSING_CASES=%s\nNEED_REPORT=1\n' "${CACHE:-1}" "${CASES:-0,1}"
""")
    calls = tmp_path / "calls"
    counter = tmp_path / "counter"
    counter.write_text("700")
    (bindir / "sbatch").write_text(
        f'#!/bin/bash\nn=$(<{counter}); n=$((n+1)); echo "$n" > {counter}\nprintf "%s\\n" "$*" >> {calls}\necho "$n"\n'
    )
    sq = bindir / "squeue"
    sq.write_text(
        '#!/bin/bash\nprintf "%s\\n" "${LIVE:-}"\nexit "${QUEUE_STATUS:-0}"\n'
    )
    for executable in bindir.iterdir():
        executable.chmod(0o755)
    env = dict(os.environ, PATH=f"{bindir}:" + os.environ["PATH"])
    command = [
        "bash",
        "scripts/submit_feniks_parent_recovery.sh",
        "--resume",
        str(root),
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 3 and not any("--mem" in x or "--gres" in x for x in lines)
    assert "--dependency=afterok:701" in lines[1] and "--array=0,1%2" in lines[1]
    assert "--dependency=afterany:701:702" in lines[2]
    for patch in ({"LIVE": "702_1"}, {"QUEUE_STATUS": "1"}):
        result = subprocess.run(
            command, env=dict(env, **patch), capture_output=True, text=True
        )
        assert result.returncode != 0
        assert len(calls.read_text().splitlines()) == 3
    result = subprocess.run(
        command, env=dict(env, CACHE="0", CASES="1"), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 5 and "--array=1%2" in lines[3]
    assert "--dependency=afterany:704" in lines[4]


def test_new_submission_freezes_code_and_keeps_source_untouched(tmp_path):
    source, dataset = make_source(tmp_path)
    before = {
        p: sha(p)
        for folder in (source, dataset)
        for p in folder.rglob("*")
        if p.is_file()
    }
    root = tmp_path / "avi_parent_recovery_20260927_200000"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls"
    counter = tmp_path / "counter"
    counter.write_text("800")
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        f'#!/bin/bash\nn=$(<{counter}); n=$((n+1)); echo "$n" > {counter}\nprintf "%s\\n" "$*" >> {calls}\necho "$n"\n'
    )
    sbatch.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bindir}:{Path(sys.executable).parent}:" + os.environ["PATH"],
        SCRATCH=str(tmp_path / "scratch"),
    )
    result = subprocess.run(
        [
            "bash",
            "scripts/submit_feniks_parent_recovery.sh",
            str(source),
            str(root),
            str(configuration(tmp_path)),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (
        sha(Path(str(root) + ".code.tar")) == (root / "CODE_SHA256").read_text().strip()
    )
    code = (
        tmp_path
        / "scratch/feniks_sc_drws_runtime/code"
        / ("parent-recovery-" + (root / "CODE_SHA256").read_text().strip())
    )
    assert (code / "scripts/feniks_parent_recovery.py").is_file()
    assert (tmp_path / "avi_parent_recovery_latest.env").exists()
    assert len(calls.read_text().splitlines()) == 3
    assert before == {p: sha(p) for p in before}
