import json
from pathlib import Path

import numpy as np


def test_weighted_resampling_and_split_are_deterministic():
    from scripts.feniks_weighted_truth_flow_capacity import (
        _effective_size,
        _split_indices,
        _weighted_resample,
    )

    settings = {
        "split_seed": 13,
        "training_fraction": 0.6,
        "validation_fraction": 0.2,
    }
    first = _split_indices(100, settings)
    second = _split_indices(100, settings)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]).isdisjoint(first[2])
    weights = np.ones(100)
    weights[9] = 1000
    draws = _weighted_resample(np.random.default_rng(3), np.arange(100), weights, 5000)
    assert np.mean(draws == 9) > 0.85
    assert _effective_size([1, 1, 1, 1]) == 4


def test_weighted_capacity_report_compares_two_flows_in_both_spaces(tmp_path):
    from euclid_dsps.amortized.forward_population import PHYSICAL
    from scripts.feniks_avi_experiments import sha, write
    from scripts.feniks_weighted_truth_flow_capacity import report
    from scripts.report_feniks_forward_population import population_metrics

    names = (*PHYSICAL, *(f"sfh_dlog_sfr_{i:02d}" for i in range(1, 11)))
    config = tmp_path / "capacity.yaml"
    config.write_text("replicas: 2\n")
    inputs = {"config": {"path": str(config), "sha256": sha(config)}}
    write(
        tmp_path / "MANIFEST.json",
        {"settings": {"replicas": 2, "seed": 9}, "inputs": inputs},
    )
    (tmp_path / "report").mkdir()
    rng = np.random.default_rng(4)
    target_x = rng.normal(size=(256, 15))
    target_theta = target_x + np.arange(15)
    for replica in range(2):
        directory = tmp_path / f"replica_{replica}"
        directory.mkdir()
        flow_x = target_x + rng.normal(scale=0.05, size=target_x.shape)
        flow_theta = flow_x + np.arange(15)
        np.savez(
            directory / "density_draws.npz",
            target_x=target_x,
            target_theta=target_theta,
            flow_x=flow_x,
            flow_theta=flow_theta,
            names=np.asarray(names),
        )
        for space, predicted, target in (
            ("latent_x", flow_x, target_x),
            ("physical_theta", flow_theta, target_theta),
        ):
            marginal, joint = population_metrics(predicted, target, names, seed=replica)
            marginal.insert(0, "space", space)
            marginal.insert(0, "replica", replica)
            joint.insert(0, "space", space)
            joint.insert(0, "replica", replica)
            marginal.to_csv(directory / f"{space}_marginal.csv", index=False)
            joint.to_csv(directory / f"{space}_joint.csv", index=False)
        write(
            directory / "FINAL.json",
            {"status": "WEIGHTED_TRUTH_FLOW_CAPACITY_COMPLETE"},
        )
        with (directory / "training.jsonl").open("w") as stream:
            for epoch in range(1, 4):
                stream.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train_nll": 10 - epoch - replica,
                            "validation_nll": 11 - epoch - replica,
                        }
                    )
                    + "\n"
                )
    report(tmp_path)
    final = json.loads((tmp_path / "report/FINAL.json").read_text())
    assert final["status"] == "WEIGHTED_TRUTH_FLOW_CAPACITY_REPORT_COMPLETE"
    assert final["target_weighting"] == "population_weight"
    assert not final["population_basis_used"]
    expected = {
        "weighted_truth_theta_15d.png",
        "weighted_truth_latent_x_15d.png",
        "weighted_truth_theta_physical_corner.png",
        "weighted_truth_latent_x_physical_corner.png",
        "weighted_truth_flow_capacity_summary.png",
        "training_curves.png",
        "marginal_metrics.csv",
        "joint_metrics.csv",
    }
    assert expected.issubset({path.name for path in (tmp_path / "report").iterdir()})


def test_weighted_capacity_launcher_is_two_replica_fail_closed_array():
    submit = Path("scripts/submit_feniks_weighted_truth_flow_capacity.sh").read_text()
    worker = Path("scripts/feniks_weighted_truth_flow_capacity.slurm").read_text()
    watcher = Path("scripts/watch_feniks_weighted_truth_flow_capacity.sh").read_text()
    assert '--array="0-1%$CONCURRENCY"' in submit
    assert '--dependency="afterok:$FIT_JOB"' in submit
    assert "2 * REPLICA_HOURS + REPORT_HOURS" in submit
    assert "#SBATCH --gres=gpu:1" in worker
    assert "Flow {replica + 1" in watcher
    assert "max physical sliced-Wasserstein" in watcher
