import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_exact_weighted_rows_visit_every_identity_equally():
    from scripts.feniks_weighted_truth_flow_followup import (
        _exact_repeated_indices,
        _task,
        _weighted_loss_scale,
    )

    settings = {"replicas": 2, "arms": ["resampled", "exact_weighted"]}
    assert _task(settings, 0) == ("resampled", 0)
    assert _task(settings, 3) == ("exact_weighted", 1)
    indices = np.array([0, 2, 4])
    repeated = _exact_repeated_indices(indices, 7)
    values, counts = np.unique(repeated, return_counts=True)
    np.testing.assert_array_equal(values, indices)
    np.testing.assert_array_equal(counts, np.full(3, 7))
    weights = np.array([1.0, 99.0, 3.0, 99.0, 6.0])
    scale = _weighted_loss_scale(weights, indices)
    losses = np.array([2.0, 5.0, 11.0])
    expected = np.average(losses, weights=weights[indices])
    assert np.isclose(np.mean(scale * losses), expected)


def test_followup_report_tracks_common_target_and_decides(tmp_path):
    from euclid_dsps.amortized.forward_population import PHYSICAL
    from scripts.feniks_avi_experiments import sha, write
    from scripts.feniks_weighted_truth_flow_followup import _metric_rows, report

    names = (*PHYSICAL, *(f"sfh_dlog_sfr_{i:02d}" for i in range(1, 11)))
    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "followup.yaml"
    config.write_text("test: true\n")
    rng = np.random.default_rng(8)
    target_x = rng.normal(size=(512, 15))
    target_theta = target_x + np.arange(15)
    inputs = {"config": {"path": str(config), "sha256": sha(config)}}
    settings = {
        "replicas": 2,
        "arms": ["resampled", "exact_weighted"],
        "milestones": [50, 100, 150, 200],
        "metric_seed": 17,
        "metric_draws": 512,
        "decision": {
            "target_physical_sliced_wasserstein": 0.05,
            "material_improvement": 0.03,
            "continuing_improvement": 0.01,
        },
    }
    write(
        tmp_path / "MANIFEST.json",
        {
            "source_capacity": str(source),
            "settings": settings,
            "inputs": inputs,
        },
    )
    for replica in range(2):
        directory = source / f"replica_{replica}"
        directory.mkdir()
        baseline = target_x + rng.normal(scale=0.2, size=target_x.shape)
        np.savez(
            directory / "density_draws.npz",
            target_x=target_x,
            target_theta=target_theta,
            flow_x=baseline,
            flow_theta=baseline + np.arange(15),
            names=np.asarray(names),
        )
    for arm, scale in (("resampled", 0.04), ("exact_weighted", 0.02)):
        for replica in range(2):
            directory = tmp_path / arm / f"replica_{replica}"
            directory.mkdir(parents=True)
            marginal_rows, joint_rows = [], []
            for epoch in settings["milestones"]:
                predicted = target_theta + rng.normal(
                    scale=scale + (200 - epoch) / 1000,
                    size=target_theta.shape,
                )
                marginal, joint = _metric_rows(
                    predicted,
                    target_theta,
                    names,
                    arm=arm,
                    replica=replica,
                    epoch=epoch,
                    seed=settings["metric_seed"],
                )
                marginal.insert(3, "space", "physical_theta")
                joint.insert(3, "space", "physical_theta")
                marginal_rows.append(marginal)
                joint_rows.append(joint)
            pd.concat(marginal_rows).to_csv(
                directory / "trajectory_marginal.csv", index=False
            )
            pd.concat(joint_rows).to_csv(
                directory / "trajectory_joint.csv", index=False
            )
            final_flow = target_theta + rng.normal(scale=scale, size=target_theta.shape)
            np.savez_compressed(
                directory / "final_draws.npz",
                target_theta=target_theta.astype(np.float32),
                flow_theta=final_flow.astype(np.float32),
            )
            write(
                directory / "FINAL.json",
                {"status": "WEIGHTED_TRUTH_FLOW_FOLLOWUP_COMPLETE"},
            )
    report(tmp_path)
    final = json.loads((tmp_path / "report/FINAL.json").read_text())
    assert final["status"] == "WEIGHTED_TRUTH_FLOW_FOLLOWUP_REPORT_COMPLETE"
    assert final["best_arm"] == "exact_weighted"
    assert final["conclusion"] == "target_recovered_with_additional_optimization"
    assert (tmp_path / "report/followup_trajectory.png").is_file()
    assert (tmp_path / "report/followup_final_theta_15d.png").is_file()


def test_followup_launcher_is_four_task_fail_closed_array():
    submit = Path("scripts/submit_feniks_weighted_truth_flow_followup.sh").read_text()
    worker = Path("scripts/feniks_weighted_truth_flow_followup.slurm").read_text()
    watcher = Path("scripts/watch_feniks_weighted_truth_flow_followup.sh").read_text()
    assert '--array="0-3%$CONCURRENCY"' in submit
    assert '--dependency="afterok:$FIT_JOB"' in submit
    assert "4 * TASK_HOURS + REPORT_HOURS" in submit
    assert "#SBATCH --gres=gpu:1" in worker
    assert "exact_weighted" in watcher
    assert "PHYS SW" in watcher
