from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from euclid_dsps.amortized.mira import FENIKS_SPLINE15D_PARAMETERS
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_avi_overnight import (
    REFRESH_ARMS,
    _corner,
    _marginals,
    population_weights,
    prepare_inference,
    prepare_training,
)


def _upstream_fixture(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    prior = tmp_path / "prior"
    source.mkdir()
    prior.mkdir()
    selected = tmp_path / "catalogs/amortized/test.parquet"
    parent = tmp_path / "catalogs/test.parquet"
    selected.parent.mkdir(parents=True)
    rng = np.random.default_rng(4)
    pd.DataFrame(
        rng.normal(size=(6, 15)), columns=FENIKS_SPLINE15D_PARAMETERS
    ).to_parquet(selected, index=False)
    pd.DataFrame(
        rng.normal(size=(10, 15)), columns=FENIKS_SPLINE15D_PARAMETERS
    ).to_parquet(parent, index=False)
    np.save(source / "train.npy", np.array([0, 2, 4]))
    np.save(source / "validation.npy", np.array([1, 3, 5]))
    np.savez(source / "teachers.npz", fixture=np.ones(1))
    (source / "teachers.json").write_text("{}\n")
    (source / "source_config.yaml").write_text("fixture: true\n")
    source_manifest = {
        "source": {"checkpoint": "source.eqx", "feature_stats": "features.json"},
        "seed": 3,
        "train_rows": 3,
        "validation_rows": 3,
        "validation_catalog": str(selected),
        "global_batch": 256,
        "local_microbatch": 32,
        "accumulation": 2,
        "gpus": 4,
        "particles": 128,
        "decoder_draw_block": 8,
        "validation_particles": 512,
        "learning_rate": 2e-5,
        "warmup_fraction": 0.05,
        "cycle": ["sleep", "sleep", "wake"],
    }
    write(source / "MANIFEST.json", source_manifest)
    b = source / "arms/B_experts"
    b.mkdir(parents=True)
    (b / "encoder.eqx").write_bytes(b"B4")
    write(
        b / "FINAL.json",
        {
            "status": "TRAINING_COMPLETE",
            "manifest_sha256": sha(source / "MANIFEST.json"),
        },
    )
    write(prior / "MANIFEST.json", {"suite": "prior-fixture"})
    for name in ("P_latest_prior", "P_scratch_prior"):
        dest = prior / "arms" / name
        dest.mkdir(parents=True)
        (dest / "prior.eqx").write_bytes(name.encode())
        write(
            dest / "FINAL.json",
            {
                "status": "PRIOR_TRAINING_COMPLETE",
                "manifest_sha256": sha(prior / "MANIFEST.json"),
            },
        )
    monkeypatch.setattr(
        "scripts.feniks_avi_experiments.check_inputs", lambda _root: source_manifest
    )
    return source, prior, selected, parent


def test_prepare_training_and_inference_freeze_exact_dependencies(
    tmp_path, monkeypatch
):
    source, prior, selected, parent = _upstream_fixture(tmp_path, monkeypatch)
    training = tmp_path / "training"
    prepare_training(source, prior, training)
    manifest = read(training / "MANIFEST.json")
    assert [item["name"] for item in manifest["arms"]] == [
        arm.name for arm in REFRESH_ARMS
    ]
    assert manifest["epochs"] == 12
    assert manifest["bootstrap_epochs"] == 0
    assert manifest["truth_used"] is False
    assert manifest["selection_in_object_weights"] is False
    assert manifest["initial_prior_checkpoint_by_arm"]["Q_source_refresh"] is None
    for arm in REFRESH_ARMS:
        dest = training / "arms" / arm.name
        dest.mkdir(parents=True)
        (dest / "encoder.eqx").write_bytes(arm.name.encode())
        write(
            dest / "FINAL.json",
            {
                "status": "TRAINING_COMPLETE",
                "manifest_sha256": sha(training / "MANIFEST.json"),
            },
        )

    inference = tmp_path / "inference"
    prepare_inference(
        training,
        source,
        prior,
        inference,
        parent_catalog=None,
        particles=4096,
    )
    prepared = read(inference / "MANIFEST.json")
    assert [item["name"] for item in prepared["variants"]] == [
        "B_source",
        "Q_source_refresh",
        "B_latest",
        "Q_latest_refresh",
        "B_scratch_prior",
    ]
    assert prepared["posterior_weight_contract"].startswith("ordinary full_15d")
    assert prepared["truth_used_for_training_or_weights"] is False
    assert Path(prepared["parent_catalog"]) == parent
    truth = pd.read_parquet(inference / "inference_truth.parquet")
    assert truth.row_index.tolist() == [1, 3, 5]
    assert len(pd.read_parquet(inference / "selected_population_truth.parquet")) == 6
    assert len(pd.read_parquet(inference / "parent_population_truth.parquet")) == 10
    assert json.loads((inference / "MANIFEST.json").read_text())["particles"] == 4096
    assert selected.is_file()


def test_optional_component_loader_round_trips_exact_tree(tmp_path):
    import equinox as eqx
    import jax

    from scripts.feniks_avi_experiments import load_optional_component

    template = eqx.nn.Linear(3, 2, key=jax.random.PRNGKey(2))
    path = tmp_path / "prior.eqx"
    eqx.tree_serialise_leaves(path, template)
    loaded = load_optional_component(path, template)
    expected = jax.tree_util.tree_leaves(eqx.filter(template, eqx.is_array))
    actual = jax.tree_util.tree_leaves(eqx.filter(loaded, eqx.is_array))
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)
    assert load_optional_component(None, template) is template


def test_prepare_inference_rejects_stale_training_receipt(tmp_path, monkeypatch):
    source, prior, _selected, _parent = _upstream_fixture(tmp_path, monkeypatch)
    training = tmp_path / "training"
    prepare_training(source, prior, training)
    for arm in REFRESH_ARMS:
        dest = training / "arms" / arm.name
        dest.mkdir(parents=True)
        (dest / "encoder.eqx").write_bytes(b"encoder")
        write(
            dest / "FINAL.json",
            {"status": "TRAINING_COMPLETE", "manifest_sha256": "stale"},
        )
    with pytest.raises(ValueError, match="stale final receipt"):
        prepare_inference(
            training,
            source,
            prior,
            tmp_path / "inference",
            parent_catalog=None,
            particles=4096,
        )


def test_population_weights_keep_equal_objects_and_inverse_selection():
    bank = pd.DataFrame(
        {
            "row_index": [10, 10, 20, 20],
            "weight": [0.9, 0.1, 0.2, 0.8],
            "log_beta": np.log([0.5, 0.25, 0.25, 1.0]),
        }
    )
    selected, parent, diagnostics = population_weights(bank)
    np.testing.assert_allclose(selected, [0.45, 0.05, 0.10, 0.40])
    np.testing.assert_allclose(parent, np.array([1.8, 0.4, 0.8, 0.8]) / 3.8)
    assert selected[:2].sum() == pytest.approx(0.5)
    assert selected[2:].sum() == pytest.approx(0.5)
    assert diagnostics["selected_ess"] == pytest.approx(1 / np.sum(selected**2))
    assert diagnostics["parent_projection_ess"] == pytest.approx(1 / np.sum(parent**2))

    bank.loc[0, "log_beta"] = -np.inf
    with pytest.raises(ValueError, match="infinite inverse-selection"):
        population_weights(bank)


def test_population_and_individual_plots_are_nonblank(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(9)
    series = {
        "source": rng.normal(size=(512, 3)),
        "latest": rng.normal(0.2, 1.1, size=(512, 3)),
    }
    truth = rng.normal(size=(256, 3))
    paths = [
        _corner(tmp_path / "corner.png", series, truth, ["a", "b", "c"]),
        _marginals(tmp_path / "marginals.png", series, truth, ["a", "b", "c"]),
    ]
    for path in paths:
        pixels = np.asarray(Image.open(path))
        assert pixels.std() > 10


def test_overnight_launch_contract_uses_bounded_four_gpu_arrays():
    slurm = Path("scripts/feniks_avi_overnight.slurm").read_text()
    submit = Path("scripts/submit_feniks_avi_overnight.sh").read_text()
    assert "#SBATCH --gres=gpu:4" in slurm
    assert "--mem" not in slurm
    assert "--array=0-1%2" in submit
    assert '--array="0-4%$INFERENCE_CONCURRENCY"' in submit
    assert '--dependency="afterok:$TRAIN"' in submit
    assert '--dependency="afterok:$PREPARE"' in submit
    assert '--dependency="afterok:$INFERENCE"' in submit
    assert 'INFERENCE_CONCURRENCY="${5:-5}"' in submit
