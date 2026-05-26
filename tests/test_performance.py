from __future__ import annotations

import json

from euclid_dsps.performance import write_performance_outputs


def test_performance_summary_reports_seconds_and_gpu_hours(tmp_path) -> None:
    rows = [
        {
            "stage": "read_chunk",
            "elapsed_seconds": 0.5,
            "total_seconds": 0.5,
            "chunk_index": 0,
            "n_rows": 10,
        },
        {
            "stage": "fit_chunk",
            "elapsed_seconds": 1.5,
            "total_seconds": 2.0,
            "chunk_index": 0,
            "n_rows": 10,
        },
    ]

    write_performance_outputs(rows, tmp_path, "batch_fit")

    summary = json.loads((tmp_path / "batch_fit_performance_summary.json").read_text())
    assert summary["n_galaxies_processed"] == 10
    assert summary["seconds_per_galaxy"] == 0.2
    assert summary["galaxies_per_second"] == 5.0
    assert "gpu_hours_per_galaxy" in summary
    assert (tmp_path / "batch_fit_performance_by_batch.csv").exists()
