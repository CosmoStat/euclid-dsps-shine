from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_synthetic_chains(path: Path) -> None:
    object_ids = np.asarray([30, 10, 40, 20], dtype=np.int64)
    values = np.empty((4, 10, 16), dtype=np.float32)
    for row, object_id in enumerate(object_ids):
        for sample in range(10):
            values[row, sample] = object_id + sample / 100.0 + np.arange(16)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("chains", data=values)
        metadata = handle.create_group("metadata")
        metadata.create_dataset("index_cosmos", data=object_ids)


def test_extract_popcosmos_redshift_chains_preserves_cohort_order(tmp_path) -> None:
    extractor = _load_script(
        "extract_popcosmos_redshift_chains",
        "extract_popcosmos_redshift_chains.py",
    )
    chains = tmp_path / "chains.h5"
    _write_synthetic_chains(chains)
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "object_id": [10, 20, 30],
            "row_index": [100, 200, 300],
            "redshift_true": [0.1, 0.2, 0.3],
        }
    ).to_parquet(cohort, index=False)
    out = tmp_path / "out"

    summary = extractor.extract_redshift_chains(
        chains,
        cohort,
        out,
        expected_objects=3,
        samples=4,
        z_index=15,
        read_chunk_size=2,
    )

    selected = pd.read_parquet(out / "posterior_samples.parquet")
    assert selected["object_id"].drop_duplicates().tolist() == [10, 20, 30]
    assert selected["sample_id"].drop_duplicates().tolist() == [0, 1, 2, 3]
    assert selected["chain_sample_id"].drop_duplicates().tolist() == [0, 3, 6, 9]
    first = selected.loc[selected["object_id"] == 10, "z_obs"].to_numpy()
    np.testing.assert_allclose(first, [25.0, 25.03, 25.06, 25.09])
    all_draws = pd.read_parquet(out / "posterior_samples_all.parquet")
    assert len(all_draws) == 3 * 10
    assert summary["chains"]["shape"] == [4, 10, 16]
    assert summary["cohort"]["objects"] == 3
    assert (out / "DONE").exists()


def test_extract_popcosmos_redshift_chains_rejects_missing_ids(tmp_path) -> None:
    extractor = _load_script(
        "extract_popcosmos_redshift_chains_missing",
        "extract_popcosmos_redshift_chains.py",
    )
    chains = tmp_path / "chains.h5"
    _write_synthetic_chains(chains)
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {"object_id": [10, 99], "row_index": [0, 1], "redshift_true": [0.1, 0.2]}
    ).to_parquet(cohort, index=False)
    with pytest.raises(ValueError, match="missing 1 cohort IDs"):
        extractor.extract_redshift_chains(
            chains,
            cohort,
            tmp_path / "out",
            expected_objects=2,
            samples=4,
            z_index=15,
            read_chunk_size=2,
        )


def test_redshift_pit_and_coverage_contract() -> None:
    evaluator = _load_script(
        "evaluate_redshift_pit_coverage",
        "evaluate_redshift_pit_coverage.py",
    )
    truth = np.asarray([0.5, 1.0, 1.5])
    samples = np.asarray(
        [
            [0.3, 0.4, 0.6, 0.7],
            [0.8, 0.9, 1.1, 1.2],
            [1.3, 1.4, 1.6, 1.7],
        ]
    )
    pit = evaluator.finite_rank_pit(samples, truth)
    np.testing.assert_allclose(pit, np.full(3, 0.5))
    levels = np.asarray(evaluator.DEFAULT_LEVELS)
    arrays = evaluator.calibration_arrays(samples, truth, levels)
    metrics = evaluator.scalar_metrics(arrays, levels)
    assert metrics["coverage_68"] == 1.0
    assert metrics["coverage_95"] == 1.0
    assert metrics["pit_mean"] == 0.5


def test_popcosmos_chain_submission_separates_network_and_gpu() -> None:
    prepost = (ROOT / "scripts/popcosmos_chains_prepost.slurm").read_text()
    evaluation = (
        ROOT / "scripts/popcosmos_chains_redshift_calibration_h100.slurm"
    ).read_text()
    submit = (
        ROOT / "scripts/submit_popcosmos_chains_redshift_calibration.sh"
    ).read_text()
    assert "#SBATCH --partition=prepost" in prepost
    assert "chains.h5.zip?download=1" in prepost
    assert "657a17446fc8b13d15e60bd54552b015" in prepost
    assert "curl" in prepost and "--continue-at -" in prepost
    assert "#SBATCH --constraint=h100" in evaluation
    assert "rws26=$RWS26_INFERENCE" in evaluation
    assert "popcosmos=$PRIMARY" in evaluation
    assert "evaluate_redshift_pit_coverage.py" in evaluation
    assert 'RUN_ALL_DRAW_SENSITIVITY="${RUN_ALL_DRAW_SENSITIVITY:-1}"' in evaluation
    assert '--dependency="afterok:$prepost_job"' in submit
