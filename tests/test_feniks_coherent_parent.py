import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from euclid_dsps.synthetic_diffsky.coherent_parent import (
    TRUTH_COLUMNS,
    catalogue_checks,
    eligible_proposals,
    observe,
    partition_source_shards,
    sample_parent,
)
from scripts import feniks_coherent_parent as workflow


def raw_shard(path, split="train", seed=10, weights=(1.0, 3.0), clipped=None):
    n = len(weights)
    frame = pd.DataFrame({c: np.ones(n) for c in TRUTH_COLUMNS})
    index = int(path.stem.removeprefix("shard_"))
    frame["source_proposal_id"] = [f"{split}:{seed}:{index}:{i}" for i in range(n)]
    frame["source_split"] = split
    frame["source_seed"] = seed
    frame["source_shard"] = index
    frame["galaxy_weight"] = weights
    frame["metallicity_clipped"] = False if clipped is None else clipped
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def test_streaming_sampling_uses_mass_across_shards_then_unit_weights(tmp_path):
    a, b = tmp_path / "shard_00000.parquet", tmp_path / "shard_00001.parquet"
    raw_shard(a, weights=[1.0, 3.0])
    raw_shard(b, weights=[12.0])

    def draw():
        return sample_parent(
            [a, b], "train", 10, 30000, 17, sha=workflow.sha, progress=lambda _: None
        )

    frame, summary = draw()
    pd.testing.assert_frame_equal(frame, draw()[0])
    proportions = frame.source_proposal_id.value_counts(normalize=True)
    for label, expected in {
        "train:10:0:0": 1 / 16,
        "train:10:0:1": 3 / 16,
        "train:10:1:0": 12 / 16,
    }.items():
        assert proportions[label] == pytest.approx(expected, abs=0.01)
    assert frame.population_weight.eq(1).all()
    assert frame.galaxy_weight.eq(1).all()
    assert set(frame.source_proposal_weight) == {1.0, 3.0, 12.0}
    assert summary["eligible_proposals"] == 3
    assert summary["no_photometric_preselection"]


def test_only_explicit_support_cut_and_bad_source_rejected(tmp_path):
    path = tmp_path / "shard_00000.parquet"
    raw_shard(path, weights=[1.0, 2.0], clipped=[False, True])
    assert len(eligible_proposals(path, "train", 10)) == 1
    with pytest.raises(ValueError, match="identities"):
        eligible_proposals(path, "test", 10)
    raw_shard(path, weights=[1.0, -1.0])
    with pytest.raises(ValueError, match="weights"):
        eligible_proposals(path, "train", 10)


def test_source_change_between_passes_is_not_silently_used(tmp_path):
    path = tmp_path / "shard_00000.parquet"
    raw_shard(path)
    hashes = iter(["before", "after"])
    with pytest.raises(ValueError, match="changed"):
        sample_parent(
            [path],
            "train",
            10,
            3,
            2,
            sha=lambda _: next(hashes),
            progress=lambda _: None,
        )


def test_overlapping_seed_ranges_are_not_treated_as_independent_splits():
    sources = {
        s: dict(
            source_seed=10 + i, paths=[f"{s}/shard_{j:05d}.parquet" for j in range(n)]
        )
        for i, (s, n) in enumerate(zip(workflow.SPLITS, (256, 56, 59), strict=True))
    }
    sizes = dict(train=100000, validation=20000, test=20000)
    result, audit = partition_source_shards(sources, sizes, 13)
    assert partition_source_shards(sources, sizes, 13) == (result, audit)
    assert audit["canonical_unique_shards"] == 256
    assert audit["ignored_overlapping_shards"] == 115
    assert audit["shard_counts"] == dict(train=183, validation=37, test=36)
    sets = [set(result[s]["effective_seeds"]) for s in workflow.SPLITS]
    assert all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1 :])
    assert len(set.union(*sets)) == 256
    assert all(r["source_split"] == "train" for r in result.values())
    sources["test"]["source_seed"] = 1000
    with pytest.raises(ValueError, match="does not cover"):
        partition_source_shards(sources, sizes, 13)


def test_different_original_ids_can_refer_to_the_same_effective_proposal(tmp_path):
    a = tmp_path / "train/shard_00001.parquet"
    b = tmp_path / "validation/shard_00000.parquet"
    raw_shard(a, "train", 10)
    raw_shard(b, "validation", 11)
    left = eligible_proposals(a, "train", 10)
    right = eligible_proposals(b, "validation", 11)
    assert set(left.source_proposal_id).isdisjoint(right.source_proposal_id)
    assert set(left.effective_proposal_key) == set(right.effective_proposal_key)


def observation_fixture(n=20000):
    bands = [dict(name="lsst_r"), dict(name="lsst_g")]
    noise = dict(
        type="m5_depth",
        m5={"lsst_r": 27.5, "lsst_g": 27.4},
        gamma={"lsst_r": 0.039, "lsst_g": 0.038},
        sigma_sys_mag=0.005,
    )
    truth = pd.DataFrame(
        dict(
            object_id=np.arange(n),
            galaxy_weight=np.ones(n),
            population_weight=np.ones(n),
        )
    )
    flux = np.full((n, 2), 9e-32)
    selection = dict(band="lsst_r", max_mag_ab=29.0)
    return truth, flux, bands, noise, selection


def test_gaussian_noise_selection_and_rejected_rows_remain_in_parent():
    frame, flux, bands, noise, selection = observation_fixture()
    data = observe(frame, flux, bands, noise, seed=41, selection=selection)
    checks = catalogue_checks(data, bands, selection)
    assert checks["rows"] == len(frame)
    assert 0 < checks["selected"] < len(frame)
    assert abs(checks["empirical_alpha"] - checks["analytic_alpha"]) < 0.02
    assert all(
        abs(x["mean"]) < 0.03 and abs(x["std"] - 1) < 0.03 for x in checks["noise"]
    )
    assert (data.flux_lsst_r < 0).any()
    data.loc[0, "selected_r29"] = not data.loc[0, "selected_r29"]
    with pytest.raises(ValueError, match="selection"):
        catalogue_checks(data, bands, selection)


def test_double_weighting_and_nonfinite_flux_fail_closed():
    frame, flux, bands, noise, selection = observation_fixture(10)
    flux[0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid forward"):
        observe(frame, flux, bands, noise, seed=1, selection=selection)
    flux[0, 0] = 9e-32
    data = observe(frame, flux, bands, noise, seed=1, selection=selection)
    data.population_weight *= 2
    with pytest.raises(ValueError, match="second time"):
        catalogue_checks(data, bands, selection)


def prepared(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _, _, bands, noise, _ = observation_fixture(1)
    manifest = dict(splits={}, synthetic_diffsky=dict(flux_error_model=noise))
    counts = dict(train=6, validation=2, test=2)
    for i, split in enumerate(workflow.SPLITS):
        for shard in range(counts[split]):
            raw_shard(
                source / "proposals" / split / f"shard_{shard:05d}.parquet",
                split,
                10 + i,
            )
        manifest["splits"][split] = dict(source_seed=10 + i, n_shards=counts[split])
    (source / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    asset = tmp_path / "ssp.dat"
    asset.write_text("fixed test asset")
    decoder = tmp_path / "decoder.yaml"
    decoder.write_text(
        yaml.safe_dump(
            dict(
                ssp_path=str(asset),
                bands=bands,
                model=dict(
                    sfh_model="spline15d",
                    photometry_integrator="merged_gauss4_v1",
                    mdf_weight_precision="float64_v1",
                    spline_precision="float64_v1",
                    n_sfh_bins=80,
                ),
            )
        )
    )
    cfg = yaml.safe_load(
        Path("configs/experiments/feniks_coherent_parent.yaml").read_text()
    )
    cfg.update(
        parent_rows={s: 128 for s in workflow.SPLITS},
        expected_proposal_shards=counts,
        rows_per_task=128,
        checkpoint_rows=64,
    )
    cfg["contracts"].update(maximum_noise_mean_abs=0.4, maximum_noise_std_error=0.4)
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(workflow, "runtime_asset_paths", lambda _: [asset])
    monkeypatch.setattr(
        workflow, "load_config", lambda p: yaml.safe_load(Path(p).read_text())
    )
    root = tmp_path / "output"
    workflow.prepare(source, decoder, root, config)
    return root


def test_full_pipeline_partial_resume_integrity_and_selection(tmp_path, monkeypatch):
    from euclid_dsps.prior_learning import spline15d
    from euclid_dsps.prior_learning.spline15d_schema import SPLINE15D_PARAMETER_NAMES

    root = prepared(tmp_path, monkeypatch)
    calls = []

    def project(frame, **kwargs):
        calls.append(len(frame))
        projected = pd.DataFrame(
            {name: np.ones(len(frame)) for name in SPLINE15D_PARAMETER_NAMES}
        )
        projected.insert(0, "object_id", frame.object_id.to_numpy())
        return projected, None

    monkeypatch.setattr(spline15d, "project_diffsky_frame_to_spline15d", project)
    monkeypatch.setattr(
        workflow,
        "photometer",
        lambda cfg, batch: lambda theta: np.full((len(theta), 2), 9e-32),
    )
    workflow.report(root)
    assert workflow.read(root / "report/SUMMARY.json")["status"] == "PARTIAL"
    for i in range(3):
        workflow.sample(root, i)
        workflow.photometry(root, i)
    assert calls == [64] * 6
    calls.clear()
    # A lost outer receipt must not redo already completed photometry blocks.
    (root / "photometry/task_000/FINAL.json").unlink()
    workflow.photometry(root, 0)
    assert not calls
    workflow.report(root)
    contract = workflow.read(root / "dataset/CONTRACT.json")
    assert contract["ready_for_population_benchmark"]
    assert not contract["ready_for_production"]
    assert workflow.complete(root / "report", workflow.sha(root / "MANIFEST.json"))
    for split in workflow.SPLITS:
        parent = pd.read_parquet(root / "dataset/parent" / f"{split}.parquet")
        selected = pd.read_parquet(root / "dataset/selected_r29" / f"{split}.parquet")
        assert set(SPLINE15D_PARAMETER_NAMES).issubset(parent)
        assert len(parent) == 128 and 0 < len(selected) < 128
        assert (
            selected.object_id.tolist()
            == parent.loc[parent.selected_r29, "object_id"].tolist()
        )
    path = root / "photometry/task_001/block_000000/parent.parquet"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        workflow.photometry(root, 1)


def test_prepared_configuration_cannot_drift(tmp_path, monkeypatch):
    root = prepared(tmp_path, monkeypatch)
    (root / "noise.json").write_text("{}")
    with pytest.raises(ValueError, match="configuration changed"):
        workflow.sample(root, 0)


def test_launcher_has_no_memory_flags_and_valid_shell_syntax():
    files = [
        Path("scripts") / name
        for name in (
            "submit_feniks_coherent_parent.sh",
            "watch_feniks_coherent_parent.sh",
            "feniks_coherent_parent.slurm",
        )
    ]
    for file in files:
        subprocess.run(["bash", "-n", str(file)], check=True)
        assert "--mem" not in file.read_text()
    text = files[0].read_text()
    assert "afterany:" in text
    assert "complete(root/" in text


def test_resume_submits_only_missing_photometry_and_report(tmp_path, monkeypatch):
    root = prepared(tmp_path, monkeypatch)
    for i in range(3):
        workflow.sample(root, i)
    digest = workflow.sha(root / "MANIFEST.json")
    for i in (0, 2):
        out = root / "photometry" / f"task_{i:03d}"
        out.mkdir()
        workflow.finish(out, [], digest)
    (root / "CODE_DIR").write_text(str(Path.cwd()))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "sbatch.jsonl"
    script = bin_dir / "sbatch"
    script.write_text(
        '#!/usr/bin/env python3\nimport os,sys,json\nfrom pathlib import Path\np=Path(os.environ["SBATCH_LOG"])\nwith p.open("a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nprint(123)\n'
    )
    script.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{Path.cwd() / '.venv/bin'}:{os.environ['PATH']}",
        SBATCH_LOG=str(log),
        EUCLID_DSPS_REQUIRE_GPU="0",
    )
    subprocess.run(
        ["bash", "scripts/submit_feniks_coherent_parent.sh", "--resume", str(root)],
        env=env,
        check=True,
    )
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(commands) == 2
    assert any(v.startswith("--array=1%") for v in commands[0])
    assert any(v.startswith("--dependency=afterany:") for v in commands[1])
