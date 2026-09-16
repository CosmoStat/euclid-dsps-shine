from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feniks_sbeb_benchmark import (
    PHYSICAL,
    _factor_cells,
    _posterior_calibration_artifacts,
    _selection_config,
    _stable_bucket,
    _trajectory_tracks,
)


def test_sbeb_factor_and_trajectory_design_is_complete():
    cells = _factor_cells()
    assert len(cells) == 24
    assert len({cell["name"] for cell in cells}) == 24
    assert {cell["cut"] for cell in cells} == {25, 27, 29}
    assert {cell["initialization"] for cell in cells} == {"warm", "scratch"}
    assert {cell["e_step_mode"] for cell in cells} == {"raw_q", "ordinary_iw"}
    assert {cell["selection_objective"] for cell in cells} == {
        "naive",
        "corrected",
    }

    tracks = _trajectory_tracks()
    assert len(tracks) == 8
    assert len({track["name"] for track in tracks}) == 8
    assert sum(track["e_step_mode"] == "raw_q" for track in tracks) == 2
    assert all(
        track["cut"] == 29 for track in tracks if track["e_step_mode"] == "raw_q"
    )


def test_object_identity_split_is_deterministic_and_order_invariant():
    import pandas as pd

    identities = pd.Series(["a", "b", "c", "d", "e"])
    first = _stable_bucket(identities)
    second = _stable_bucket(identities.sample(frac=1, random_state=4))
    reordered = dict(
        zip(identities.sample(frac=1, random_state=4), second, strict=True)
    )
    np.testing.assert_array_equal(first, [reordered[value] for value in identities])
    assert np.all((first >= 0) & (first < 10_000))


def test_selection_cut_updates_sleep_and_parent_normalization():
    source = {
        "truth": {"parameter_columns": {"bad": "truth"}},
        "extra_columns": ["truth"],
        "amortized": {
            "data": {},
            "objective": {
                "selection_correction": {
                    "enabled": False,
                    "band": "lsst_r",
                    "max_mag_ab": 29.0,
                },
                "sleep": {
                    "selection": {
                        "enabled": True,
                        "band": "lsst_r",
                        "max_mag_ab": 29.0,
                    }
                },
            },
        },
    }
    result = _selection_config(source, 25)
    objective = result["amortized"]["objective"]
    assert objective["selection_correction"]["enabled"] is True
    assert objective["selection_correction"]["max_mag_ab"] == 25.0
    assert objective["sleep"]["selection"]["max_mag_ab"] == 25.0
    assert result["truth"] == {"parameter_columns": {}}
    assert result["extra_columns"] == []
    assert source["amortized"]["objective"]["selection_correction"]["enabled"] is False


def test_sbeb_launch_is_fail_closed_and_uses_four_h100s_per_task():
    submit = Path("scripts/submit_feniks_sbeb_benchmark.sh").read_text()
    slurm = Path("scripts/feniks_sbeb_benchmark.slurm").read_text()
    watcher = Path("scripts/watch_feniks_sbeb_benchmark.sh").read_text()
    assert "#SBATCH --gres=gpu:4" in slurm
    assert "--mem" not in slurm
    assert 'CONCURRENCY="${7:-8}"' in submit
    assert "afterok:$dependency" in submit
    assert '--array="0-23%$CONCURRENCY"' not in submit
    assert "submit_array factor 0-23" in submit
    assert "--scratch-epochs 180" in submit
    assert "--q-epochs 24" in submit
    assert "for CYCLE in 1 2 3 4" in submit
    assert "submit_array endpoint-factorial-infer 0-31" in submit
    assert "submit_array endpoint-factorial-report 0-7" in submit
    assert '"$EM_REPORT_JOB:$ENDPOINT_REPORT_JOB"' in submit
    assert "nuts-infer" in submit and "factor-report" in submit
    assert "EM TRAJECTORIES" in watcher


def test_posterior_calibration_writes_dense_blind_artifacts(tmp_path):
    rng = np.random.default_rng(260916)
    objects = 12
    samples = 32
    truth_values = rng.normal(size=(objects, len(PHYSICAL)))
    truth = pd.DataFrame(truth_values, columns=PHYSICAL)
    truth.insert(0, "row_index", np.arange(objects))
    draws = np.repeat(truth_values, samples, axis=0) + rng.normal(
        scale=0.3, size=(objects * samples, len(PHYSICAL))
    )
    posterior = pd.DataFrame(draws, columns=PHYSICAL)
    posterior.insert(0, "sample_id", np.tile(np.arange(samples), objects))
    posterior.insert(0, "row_index", np.repeat(np.arange(objects), samples))

    metrics = _posterior_calibration_artifacts(
        posterior, truth, tmp_path / "calibration", title="fixture"
    )

    assert np.isfinite(list(metrics.values())).all()
    assert (tmp_path / "calibration/posterior_calibration_5d.csv").is_file()
    assert (tmp_path / "calibration/posterior_pit_5d.png").stat().st_size > 1_000
    assert (
        tmp_path / "calibration/truth_vs_posterior_5d.png"
    ).stat().st_size > 1_000
