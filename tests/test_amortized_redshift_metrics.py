from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from euclid_dsps.amortized.redshift_metrics import (
    redshift_metrics_by_truth_bin,
    redshift_metrics_from_samples,
    summarize_redshift_metrics,
    write_redshift_metrics_for_run,
)


def test_redshift_metrics_on_toy_posterior_are_correct() -> None:
    samples = pd.DataFrame(
        {
            "object_id": [1, 1, 1, 2, 2, 2],
            "z_obs": [0.9, 1.0, 1.1, 0.4, 0.5, 0.6],
        }
    )
    truth = pd.DataFrame(
        {
            "object_id": [1, 2],
            "redshift_true": [1.0, 0.5],
        }
    )

    object_metrics, summary = redshift_metrics_from_samples(samples, truth)

    assert np.allclose(object_metrics["delta_z"], [0.0, 0.0])
    assert summary["median_bias"] == 0.0
    assert summary["rmse"] == 0.0
    assert summary["outlier_fraction_0p15"] == 0.0
    assert summary["coverage_68"] == 1.0
    assert summary["coverage_95"] == 1.0


def test_redshift_summary_outlier_fraction() -> None:
    frame = pd.DataFrame(
        {
            "delta_z": [0.0, 0.2],
            "pit": [0.5, 1.0],
            "covered_68": [True, False],
            "covered_95": [True, True],
            "posterior_width_68": [0.1, 0.3],
        }
    )

    summary = summarize_redshift_metrics(frame)

    assert summary["n_objects"] == 2
    assert summary["outlier_fraction_0p15"] == 0.5
    assert summary["coverage_68"] == 0.5
    assert summary["coverage_95"] == 1.0


def test_redshift_metrics_by_truth_bin() -> None:
    frame = pd.DataFrame(
        {
            "z_true": [0.5, 1.0],
            "delta_z": [0.01, -0.02],
            "pit": [0.5, 0.4],
            "covered_68": [True, False],
            "covered_95": [True, True],
            "posterior_width_68": [0.1, 0.2],
        }
    )

    binned = redshift_metrics_by_truth_bin(frame)

    assert int(binned["n_objects"].sum()) == 2
    assert {"z_bin_lower", "z_bin_upper", "median_bias", "coverage_68"} <= set(
        binned.columns
    )


def test_write_redshift_metrics_includes_alpha_and_likelihood_metadata(
    tmp_path,
) -> None:
    dataset = tmp_path / "truth.parquet"
    pd.DataFrame(
        {
            "object_id": [1],
            "redshift_true": [0.5],
            "logsm_true": [10.0],
        }
    ).to_parquet(dataset, index=False)
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame(
        {
            "object_id": [1, 1],
            "z_obs": [0.45, 0.55],
            "log10_stellar_mass": [9.8, 9.9],
            "log10_stellar_mass_alpha_corrected": [10.0, 10.1],
        }
    ).to_parquet(run / "posterior_samples.parquet", index=False)
    (run / "inference_summary.json").write_text(
        json.dumps(
            {
                "global_sed_scale": {
                    "alpha_sed": 2.0,
                    "log_alpha_sed": 0.693147,
                    "delta_mag_global": -0.752575,
                    "alpha_prior_penalty": 24.0,
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "normalized_config.json").write_text(
        json.dumps(
            {
                "amortized": {
                    "likelihood": {
                        "type": "student_t",
                        "student_t_dof": 2.0,
                        "error_floor_frac": 0.02,
                        "error_jitter": 0.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    write_redshift_metrics_for_run(
        dataset_path=dataset,
        run_dir=run,
        label="toy",
    )

    photoz = pd.read_csv(run / "photoz_metrics.csv")
    posterior = pd.read_csv(run / "posterior_vs_truth_metrics.csv")
    assert photoz.loc[0, "alpha_sed"] == 2.0
    assert photoz.loc[0, "likelihood_type"] == "student_t"
    assert set(posterior["metric_name"]) >= {
        "mass_bias_raw",
        "mass_bias_alpha_corrected",
    }


def test_write_redshift_metrics_prefers_inference_truth_row_index(tmp_path) -> None:
    dataset = tmp_path / "truth.parquet"
    pd.DataFrame(
        {
            "object_id": [999],
            "redshift_true": [9.0],
        }
    ).to_parquet(dataset, index=False)
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame(
        {
            "row_index": [7, 7],
            "object_id": [1, 1],
            "z_obs": [0.45, 0.55],
        }
    ).to_parquet(run / "posterior_samples.parquet", index=False)
    pd.DataFrame(
        {
            "row_index": [7],
            "object_id": [123456789],
            "redshift_true": [0.5],
        }
    ).to_parquet(run / "inference_truth.parquet", index=False)

    write_redshift_metrics_for_run(
        dataset_path=dataset,
        run_dir=run,
        label="row-index",
    )

    photoz = pd.read_csv(run / "photoz_metrics.csv")
    objects = pd.read_csv(run / "photoz_object_metrics.csv")
    assert int(photoz.loc[0, "n_objects"]) == 1
    assert objects.loc[0, "row_index"] == 7
    assert objects.loc[0, "object_id"] == 123456789
    assert objects.loc[0, "z_true"] == 0.5


def test_write_redshift_metrics_rejects_duplicate_object_id_without_row_index(
    tmp_path,
) -> None:
    dataset = tmp_path / "truth.parquet"
    pd.DataFrame(
        {
            "object_id": [1, 1],
            "redshift_true": [0.5, 0.6],
        }
    ).to_parquet(dataset, index=False)
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame({"object_id": [1, 1], "z_obs": [0.45, 0.55]}).to_parquet(
        run / "posterior_samples.parquet",
        index=False,
    )

    with pytest.raises(ValueError, match="truth identity column is not unique"):
        write_redshift_metrics_for_run(
            dataset_path=dataset,
            run_dir=run,
            label="duplicate",
        )


def test_write_redshift_metrics_reads_posterior_sample_shards(tmp_path) -> None:
    dataset = tmp_path / "truth.parquet"
    pd.DataFrame(
        {
            "object_id": [1, 2],
            "redshift_true": [0.5, 1.0],
            "logsm_true": [10.0, 11.0],
        }
    ).to_parquet(dataset, index=False)
    run = tmp_path / "run"
    shard_dir = run / "posterior_samples"
    shard_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "object_id": [1, 1],
            "z_obs": [0.45, 0.55],
            "log10_stellar_mass": [9.9, 10.1],
        }
    ).to_parquet(shard_dir / "batch_000001.parquet", index=False)
    pd.DataFrame(
        {
            "object_id": [2, 2],
            "z_obs": [0.95, 1.05],
            "log10_stellar_mass": [10.8, 11.2],
        }
    ).to_parquet(shard_dir / "batch_000002.parquet", index=False)

    write_redshift_metrics_for_run(
        dataset_path=dataset,
        run_dir=run,
        label="sharded",
    )

    photoz = pd.read_csv(run / "photoz_metrics.csv")
    posterior = pd.read_csv(run / "posterior_vs_truth_metrics.csv")
    binned = pd.read_csv(run / "photoz_metrics_by_redshift_bin.csv")
    assert int(photoz.loc[0, "n_objects"]) == 2
    assert int(binned["n_objects"].sum()) == 2
    assert "mass_bias_raw" in set(posterior["metric_name"])


def test_write_redshift_metrics_uses_manifest_to_ignore_stale_shards(tmp_path) -> None:
    dataset = tmp_path / "truth.parquet"
    pd.DataFrame(
        {
            "object_id": [1, 2, 99],
            "redshift_true": [0.5, 1.0, 3.0],
        }
    ).to_parquet(dataset, index=False)
    run = tmp_path / "run"
    shard_dir = run / "posterior_samples"
    shard_dir.mkdir(parents=True)
    shard_1 = shard_dir / "batch_000001.parquet"
    shard_2 = shard_dir / "batch_000002.parquet"
    stale = shard_dir / "batch_999999.parquet"
    pd.DataFrame({"object_id": [1, 1], "z_obs": [0.45, 0.55]}).to_parquet(
        shard_1,
        index=False,
    )
    pd.DataFrame({"object_id": [2, 2], "z_obs": [0.95, 1.05]}).to_parquet(
        shard_2,
        index=False,
    )
    pd.DataFrame({"object_id": [99, 99], "z_obs": [2.8, 3.2]}).to_parquet(
        stale,
        index=False,
    )
    (run / "posterior_shards_manifest.json").write_text(
        json.dumps(
            {
                "shards_written": [
                    {"batch": 1, "samples_path": str(shard_1)},
                    {"batch": 2, "samples_path": str(shard_2)},
                ],
                "shards_skipped": [],
            }
        ),
        encoding="utf-8",
    )

    write_redshift_metrics_for_run(
        dataset_path=dataset,
        run_dir=run,
        label="manifest",
    )

    photoz = pd.read_csv(run / "photoz_metrics.csv")
    objects = pd.read_csv(run / "photoz_object_metrics.csv")
    assert int(photoz.loc[0, "n_objects"]) == 2
    assert set(objects["object_id"]) == {1, 2}
