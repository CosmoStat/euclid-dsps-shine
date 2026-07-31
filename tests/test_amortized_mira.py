from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pytest

from euclid_dsps.amortized.mira import (
    FENIKS_SPLINE15D_PARAMETERS,
    evaluate_feniks_mira,
    mira_region_contributions,
    resolve_posterior_input,
)


def _numpy_mira(
    truth: np.ndarray,
    posterior: np.ndarray,
    centers: np.ndarray,
    reference_indices: np.ndarray,
) -> np.ndarray:
    n_regions = centers.shape[0]
    n_models, n_objects, n_samples, _ = posterior.shape
    result = np.empty((n_regions, n_models, n_objects), dtype=float)
    for region_index in range(n_regions):
        for model_index in range(n_models):
            for object_index in range(n_objects):
                center = centers[region_index, object_index]
                distances = np.sum(
                    (posterior[model_index, object_index] - center[None, :]) ** 2,
                    axis=1,
                )
                radius = distances[reference_indices[region_index, object_index]]
                count = int(np.sum(distances < radius))
                truth_distance = np.sum((truth[object_index] - center) ** 2)
                n_candidate = n_samples - 1
                if truth_distance <= radius:
                    probability = (count + 1) / (n_candidate + 2)
                else:
                    probability = (n_candidate - count + 1) / (n_candidate + 2)
                result[region_index, model_index, object_index] = probability / (
                    n_samples / (n_samples + 1)
                )
    return result


def test_mira_region_contributions_match_direct_numpy() -> None:
    truth = np.asarray([[0.1, 0.4], [0.8, 0.2], [0.3, 0.9]], dtype=np.float32)
    posterior = np.asarray(
        [
            [
                [[0.0, 0.3], [0.2, 0.5], [0.4, 0.2], [0.1, 0.8]],
                [[0.7, 0.1], [0.9, 0.3], [0.6, 0.4], [0.8, 0.0]],
                [[0.2, 0.7], [0.5, 0.8], [0.4, 1.0], [0.1, 0.9]],
            ],
            [
                [[0.2, 0.2], [0.3, 0.6], [0.5, 0.1], [0.0, 0.7]],
                [[0.6, 0.2], [1.0, 0.4], [0.5, 0.5], [0.7, 0.1]],
                [[0.1, 0.6], [0.6, 0.9], [0.3, 0.8], [0.2, 1.0]],
            ],
        ],
        dtype=np.float32,
    )
    centers = np.asarray(
        [
            [[0.5, 0.5], [0.1, 0.4], [0.8, 0.2]],
            [[0.0, 0.9], [0.9, 0.9], [0.4, 0.4]],
        ],
        dtype=np.float32,
    )
    reference_indices = np.asarray([[0, 2, 1], [3, 0, 2]], dtype=np.int32)

    actual = np.asarray(
        jax.device_get(
            mira_region_contributions(
                truth,
                posterior,
                centers,
                reference_indices,
            )
        )
    )
    expected = _numpy_mira(truth, posterior, centers, reference_indices)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_feniks_mira_parquet_workflow_writes_auditable_outputs(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(17)
    n_objects = 128
    n_samples = 32
    n_dimensions = len(FENIKS_SPLINE15D_PARAMETERS)
    object_ids = np.arange(20_000, 20_000 + n_objects)
    means = rng.normal(size=(n_objects, n_dimensions))
    truth_values = means + 0.35 * rng.normal(size=means.shape)
    posterior_values = means[:, None, :] + 0.35 * rng.normal(
        size=(n_objects, n_samples, n_dimensions)
    )
    truth = pd.DataFrame(
        truth_values,
        columns=FENIKS_SPLINE15D_PARAMETERS,
    )
    truth.insert(0, "row_index", np.arange(n_objects))
    truth.insert(0, "object_id", object_ids)
    inference = tmp_path / "inference"
    inference.mkdir()
    truth_path = inference / "inference_truth.parquet"
    truth.to_parquet(truth_path, index=False)

    samples = pd.DataFrame(
        posterior_values.reshape(n_objects * n_samples, n_dimensions),
        columns=FENIKS_SPLINE15D_PARAMETERS,
    )
    samples.insert(0, "sample_id", np.tile(np.arange(n_samples), n_objects))
    samples.insert(0, "row_index", np.repeat(np.arange(n_objects), n_samples))
    samples.insert(0, "object_id", np.repeat(object_ids, n_samples))
    samples = samples.sample(frac=1.0, random_state=9).reset_index(drop=True)
    shards = inference / "posterior_samples"
    shards.mkdir()
    samples.loc[samples["row_index"].lt(n_objects // 2)].to_parquet(
        shards / "batch_000001.parquet", index=False
    )
    samples.loc[samples["row_index"].ge(n_objects // 2)].to_parquet(
        shards / "batch_000002.parquet", index=False
    )

    out = tmp_path / "mira"
    summary = evaluate_feniks_mira(
        truth_path=truth_path,
        posterior_specs=[("encoder", inference)],
        out_dir=out,
        num_regions=30,
        num_bootstrap=50,
        samples_per_object=n_samples,
        seed=123,
    )

    assert summary["status"] == "complete"
    assert summary["num_objects"] == n_objects
    assert summary["num_posterior_samples"] == n_samples
    assert 0.5 < summary["full_15d"][0]["score"] < 0.8
    expected = {
        "DONE",
        "mira_manifest.json",
        "mira_summary.json",
        "mira_scores.csv",
        "mira_scores.parquet",
        "mira_region_scores.parquet",
        "mira_object_contributions.parquet",
        "mira_bootstrap_scores.parquet",
        "mira_pairwise_differences.csv",
        "mira_normalization.csv",
        "mira_normalization_diagnostics.csv",
        "mira_scores.png",
    }
    assert expected <= {path.name for path in out.iterdir()}
    scores = pd.read_parquet(out / "mira_scores.parquet")
    assert len(scores) == 18
    assert set(scores["group"].head(3)) == {
        "full_15d",
        "physical_5d",
        "sfh_contrasts_10d",
    }
    contributions = pd.read_parquet(out / "mira_object_contributions.parquet")
    assert len(contributions) == 18 * n_objects
    assert contributions.groupby(["model", "group"]).size().eq(n_objects).all()
    manifest = json.loads((out / "mira_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["bootstrap_unit"] == "held_out_object"
    assert manifest["shared_random_regions_across_models"] is True
    assert manifest["companion_truths"]["encoder"]["status"] == "primary_reference"
    assert summary["companion_truths_checked"] == 1


def test_resolver_prefers_monolithic_samples_without_double_counting(
    tmp_path: Path,
) -> None:
    inference = tmp_path / "inference"
    shard_dir = inference / "posterior_samples"
    shard_dir.mkdir(parents=True)
    frame = pd.DataFrame({"object_id": [1], "sample_id": [0]})
    monolithic = inference / "posterior_samples.parquet"
    frame.to_parquet(monolithic, index=False)
    frame.to_parquet(shard_dir / "batch_000001.parquet", index=False)

    resolved = resolve_posterior_input("model", inference)

    assert resolved.files == (monolithic,)


def test_workflow_rejects_duplicate_object_sample_rows(tmp_path: Path) -> None:
    n_dimensions = len(FENIKS_SPLINE15D_PARAMETERS)
    truth = pd.DataFrame(
        np.ones((2, n_dimensions)),
        columns=FENIKS_SPLINE15D_PARAMETERS,
    )
    truth["z_obs"] = [0.1, 0.2]
    truth.insert(0, "object_id", [1, 2])
    truth_path = tmp_path / "truth.parquet"
    truth.to_parquet(truth_path, index=False)
    posterior = pd.DataFrame(
        np.ones((4, n_dimensions)),
        columns=FENIKS_SPLINE15D_PARAMETERS,
    )
    posterior.insert(0, "sample_id", [0, 0, 0, 1])
    posterior.insert(0, "object_id", [1, 1, 2, 2])
    posterior_path = tmp_path / "posterior.parquet"
    posterior.to_parquet(posterior_path, index=False)

    with pytest.raises(ValueError, match="duplicate object_id/sample_id"):
        evaluate_feniks_mira(
            truth_path=truth_path,
            posterior_specs=[("bad", posterior_path)],
            out_dir=tmp_path / "out",
            num_regions=2,
            num_bootstrap=0,
            samples_per_object=2,
        )


def test_workflow_rejects_mismatched_companion_truth(tmp_path: Path) -> None:
    n_dimensions = len(FENIKS_SPLINE15D_PARAMETERS)
    truth = pd.DataFrame(
        np.vstack([np.arange(n_dimensions), np.arange(n_dimensions) + 1.0]),
        columns=FENIKS_SPLINE15D_PARAMETERS,
    )
    truth.insert(0, "row_index", [0, 1])
    truth.insert(0, "object_id", [100, 101])
    seed2 = tmp_path / "seed2"
    seed3 = tmp_path / "seed3"
    seed2.mkdir()
    seed3.mkdir()
    truth_path = seed2 / "inference_truth.parquet"
    truth.to_parquet(truth_path, index=False)
    mismatched = truth.copy()
    mismatched.loc[0, "dust_av"] += 0.25
    mismatched.to_parquet(seed3 / "inference_truth.parquet", index=False)

    posterior = truth.loc[truth.index.repeat(2)].reset_index(drop=True)
    posterior["sample_id"] = np.tile([0, 1], len(truth))
    posterior = posterior.sort_values(["object_id", "sample_id"])
    posterior.to_parquet(seed3 / "posterior_samples.parquet", index=False)

    with pytest.raises(ValueError, match="differs from the reference truth"):
        evaluate_feniks_mira(
            truth_path=truth_path,
            posterior_specs=[("seed3", seed3)],
            out_dir=tmp_path / "out",
            num_regions=2,
            num_bootstrap=0,
            samples_per_object=2,
        )


def test_h100_wrapper_uses_h100_module_and_existing_samples() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1] / "scripts" / "feniks_mira_h100.slurm"
    ).read_text(encoding="utf-8")

    assert "module purge" in wrapper
    assert "module load arch/h100" in wrapper
    assert "posterior_samples/*.parquet" in wrapper
    assert "scripts/evaluate_feniks_mira.py" in wrapper
    assert "feniks_selfsup_paper_v1" in wrapper
    assert "rws_k8_t2_seed2" in wrapper
    assert "rws_k8_t2_seed3" in wrapper
    assert wrapper.count("--posterior") == 2
    assert wrapper.index("module load arch/h100") < wrapper.index(
        'conda activate "$CONDA_ENV"'
    )
