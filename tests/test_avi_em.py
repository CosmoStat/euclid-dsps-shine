from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.amortized.mira import FENIKS_SPLINE15D_PARAMETERS
from scripts.feniks_avi_em import (
    finalize_cycle,
    prepare,
    prepare_mstep,
    prepare_qstep,
    report,
)
from scripts.feniks_avi_experiments import read, sha, write


def _training_fixture(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    training = tmp_path / "training"
    prior = tmp_path / "prior"
    selection = tmp_path / "selection_validation"
    for path in (source, training, prior, selection / "selection"):
        path.mkdir(parents=True)
    selected_catalog = tmp_path / "selected.parquet"
    frame = pd.DataFrame(
        np.arange(90, dtype=np.float64).reshape(6, 15),
        columns=FENIKS_SPLINE15D_PARAMETERS,
    )
    frame.to_parquet(selected_catalog, index=False)
    np.save(source / "train.npy", np.array([0, 1, 2, 3]))
    np.save(source / "validation.npy", np.array([4, 5]))
    np.savez(source / "teachers.npz", fixture=np.ones(1))
    (source / "teachers.json").write_text("{}\n")
    (source / "source_config.yaml").write_text("fixture: true\n")
    source_manifest = {
        "source": {"checkpoint": "source.eqx", "feature_stats": "features.json"},
        "seed": 7,
        "train_rows": 4,
        "validation_rows": 2,
        "validation_catalog": str(selected_catalog),
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
    monkeypatch.setattr(
        "scripts.feniks_avi_experiments.check_inputs", lambda _root: source_manifest
    )

    write(training / "MANIFEST.json", {"hashes": {}})
    q = training / "arms/Q_latest_refresh"
    q.mkdir(parents=True)
    (q / "encoder.eqx").write_bytes(b"latest encoder")
    write(
        q / "FINAL.json",
        {
            "status": "TRAINING_COMPLETE",
            "manifest_sha256": sha(training / "MANIFEST.json"),
            "encoder_sha256": sha(q / "encoder.eqx"),
        },
    )

    write(prior / "MANIFEST.json", {"hashes": {}})
    p = prior / "arms/P_latest_prior"
    p.mkdir(parents=True)
    (p / "prior.eqx").write_bytes(b"latest prior")
    write(
        p / "FINAL.json",
        {
            "status": "PRIOR_TRAINING_COMPLETE",
            "manifest_sha256": sha(prior / "MANIFEST.json"),
            "prior_sha256": sha(p / "prior.eqx"),
        },
    )

    truth = frame.copy()
    truth.insert(0, "population_weight", np.ones(len(truth)))
    truth.to_parquet(selection / "selection/true_parent.parquet", index=False)
    truth.iloc[:-1].to_parquet(
        selection / "selection/true_selected.parquet", index=False
    )
    write(
        selection / "selection/FINAL.json",
        {
            "status": "EXACT_OBSERVED_SELECTION_COMPLETE",
            "parent_objects": 6,
            "selected_objects": 5,
            "weighted_alpha": 5 / 6,
        },
    )
    return source, training, prior, selection


def test_em_cycle_manifests_chain_latest_components(tmp_path, monkeypatch):
    source, training, prior, selection = _training_fixture(tmp_path, monkeypatch)
    root = tmp_path / "em"
    inference = tmp_path / "inference"
    prepare(
        source,
        training,
        prior,
        selection,
        root,
        inference,
        cycles=2,
        q_epochs=3,
    )
    manifest = read(root / "MANIFEST.json")
    assert manifest["em_contract"]["selection_in_object_weights"] is False
    assert manifest["em_contract"]["selection_in_prior_loss"] is True
    assert manifest["truth_used_for_training_or_checkpoint_selection"] is False

    mstep = prepare_mstep(root, 1)
    m_manifest = read(mstep / "MANIFEST.json")
    p_name = "P_em_01"
    assert m_manifest["prior_sweeps"] == 1
    assert m_manifest["upstream_b_encoder"] == manifest["initial_encoder"]
    assert (
        m_manifest["initial_prior_checkpoint_by_arm"][p_name]
        == manifest["initial_prior"]
    )
    p = mstep / "arms" / p_name
    p.mkdir(parents=True)
    (p / "prior.eqx").write_bytes(b"cycle one prior")
    pd.DataFrame(
        [
            {
                "log_alpha": -0.1,
                "posterior_ess_median": 12.0,
                "posterior_max_weight_q90": 0.2,
            }
        ]
    ).to_csv(p / "prior_training.csv", index=False)
    write(
        p / "FINAL.json",
        {
            "status": "PRIOR_TRAINING_COMPLETE",
            "manifest_sha256": sha(mstep / "MANIFEST.json"),
            "prior_sha256": sha(p / "prior.eqx"),
        },
    )

    qstep = prepare_qstep(root, 1)
    q_manifest = read(qstep / "MANIFEST.json")
    q_name = "Q_em_01"
    assert q_manifest["epochs"] == 3
    assert q_manifest["initial_prior_checkpoint_by_arm"][q_name] == str(
        (p / "prior.eqx").resolve()
    )
    assert (
        q_manifest["initial_encoder_checkpoint_by_arm"][q_name]
        == manifest["initial_encoder"]
    )
    q = qstep / "arms" / q_name
    q.mkdir(parents=True)
    (q / "encoder.eqx").write_bytes(b"cycle one encoder")
    write(
        q / "FINAL.json",
        {
            "status": "TRAINING_COMPLETE",
            "manifest_sha256": sha(qstep / "MANIFEST.json"),
            "encoder_sha256": sha(q / "encoder.eqx"),
        },
    )
    finalize_cycle(root, 1)
    next_mstep = prepare_mstep(root, 2)
    next_manifest = read(next_mstep / "MANIFEST.json")
    assert next_manifest["upstream_b_encoder"] == str((q / "encoder.eqx").resolve())
    assert next_manifest["initial_prior_checkpoint_by_arm"]["P_em_02"] == str(
        (p / "prior.eqx").resolve()
    )


def test_em_report_measures_selected_and_parent_fixed_points(tmp_path):
    training = tmp_path / "training"
    selection = tmp_path / "selection"
    inference = tmp_path / "inference"
    (selection / "selection").mkdir(parents=True)
    training.mkdir()
    inference.mkdir()
    write(training / "MANIFEST.json", {"hashes": {}, "selection_root": str(selection)})
    write(selection / "selection/FINAL.json", {"weighted_alpha": 0.8})
    rng = np.random.default_rng(8)
    for name, count in (("true_parent", 80), ("true_selected", 64)):
        frame = pd.DataFrame(
            rng.normal(size=(count, 15)), columns=FENIKS_SPLINE15D_PARAMETERS
        )
        frame.insert(0, "population_weight", rng.uniform(0.5, 1.5, count))
        frame.to_parquet(inference / f"{name}.parquet", index=False)
    variants = [{"name": "cycle_00"}, {"name": "cycle_01"}]
    write(
        inference / "MANIFEST.json",
        {"training": str(training), "variants": variants, "hashes": {}},
    )
    for cycle, variant in enumerate(variants):
        arm = inference / "arms" / variant["name"]
        (arm / "bank_0").mkdir(parents=True)
        draws = pd.DataFrame(
            rng.normal(0.1 * cycle, 1.0, size=(64, 15)),
            columns=FENIKS_SPLINE15D_PARAMETERS,
        )
        draws.insert(0, "log_beta", np.log(rng.uniform(0.5, 1.0, len(draws))))
        draws.insert(0, "weight", np.tile(np.full(16, 1 / 16), 4))
        draws.insert(0, "row_index", np.repeat(np.arange(4), 16))
        draws.to_parquet(arm / "bank_0/part_00000.parquet", index=False)
        theta = rng.normal(0.1 * cycle, 1.0, size=(128, 15))
        np.savez(
            arm / "prior_population.npz",
            theta=theta,
            selected_theta=theta[:96],
            beta=rng.uniform(0.5, 1.0, len(theta)),
        )
        pd.DataFrame(
            {
                "ess_fraction": rng.uniform(0.02, 0.2, 8),
                "max_weight": rng.uniform(0.1, 0.5, 8),
            }
        ).to_csv(arm / "metrics.csv", index=False)
        write(
            arm / "FINAL.json",
            {
                "status": "INFERENCE_COMPLETE",
                "manifest_sha256": sha(inference / "MANIFEST.json"),
            },
        )
    report(inference)
    final = read(inference / "report/FINAL.json")
    assert final["status"] == "AVI_EM_REPORT_COMPLETE"
    assert final["cycles"] == 1
    comparisons = set(
        pd.read_csv(inference / "report/fixed_point_distances.csv").comparison
    )
    assert "aggregate_selected_vs_prior_selected" in comparisons
    assert "aggregate_parent_vs_prior_parent" in comparisons
    assert (inference / "report/em_fixed_point.png").stat().st_size > 10_000


def test_em_launch_is_sequential_and_uses_four_h100s():
    slurm = Path("scripts/feniks_avi_em.slurm").read_text()
    submit = Path("scripts/submit_feniks_avi_em.sh").read_text()
    assert "#SBATCH --gres=gpu:4" in slurm
    assert "--mem" not in slurm
    assert 'AVI_EM_MODE=cycle AVI_EM_CYCLE="$CYCLE"' in submit
    assert '--dependency="afterok:$DEPENDENCY"' in submit
    assert '--dependency="afterok:$PREPARE_JOB"' in submit
    assert '--dependency="afterok:$INFERENCE_JOB"' in submit
    assert "prepare-mstep" in slurm and "prepare-qstep" in slurm
