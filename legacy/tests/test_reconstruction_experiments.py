from __future__ import annotations

import numpy as np
import pandas as pd

from euclid_dsps.reconstruction_experiments import (
    build_reconstruction_rowsets,
    compare_reconstruction_runs,
)


def test_build_reconstruction_rowsets_and_compare(tmp_path) -> None:
    train_run = tmp_path / "train"
    infer_run = tmp_path / "infer"
    train_run.mkdir()
    infer_run.mkdir()
    np.save(train_run / "train_indices.npy", np.asarray([0, 1, 2]))
    np.save(train_run / "validation_indices.npy", np.asarray([3, 4]))
    np.save(infer_run / "inference_indices.npy", np.asarray([0, 1, 2, 3, 4]))
    residual = pd.DataFrame(
        {
            "object_id": [10, 10, 11, 11, 12, 12],
            "row_index": [0, 0, 1, 1, 2, 2],
            "band": ["a", "b", "a", "b", "a", "b"],
            "obs_flux_fnu_cgs": 1.0,
            "obs_err_fnu_cgs": 0.1,
            "residual_sigma_median": [1.0, -2.0, 5.0, 4.0, 0.5, 0.25],
            "valid": True,
        }
    )
    residual["abs_residual_sigma_median"] = residual[
        "residual_sigma_median"
    ].abs()
    residual.to_parquet(infer_run / "posterior_predictive_residual_summary.parquet")

    outputs = build_reconstruction_rowsets(
        train_run=train_run,
        infer_run=infer_run,
        out_dir=tmp_path / "rowsets",
        worst_sizes=(2,),
    )

    assert (tmp_path / "rowsets" / "worst_2.txt").read_text(
        encoding="utf-8"
    ).splitlines() == ["1", "0"]
    assert outputs["manifest"].endswith("rowsets_manifest.json")

    map_run = tmp_path / "map"
    map_run.mkdir()
    pd.DataFrame(
        {
            "object_id": [10, 10, 11, 11],
            "row_index": [0, 0, 1, 1],
            "band": ["a", "b", "a", "b"],
            "observed_flux_fnu_cgs": [1.0, 1.0, 1.0, 1.0],
            "observed_flux_error_fnu_cgs": [0.1, 0.1, 0.1, 0.1],
            "model_flux_fnu_cgs": [0.9, 1.1, 0.5, 0.6],
            "likelihood_sigma": [0.1, 0.1, 0.1, 0.1],
            "chi_likelihood": [1.0, -1.0, 5.0, 4.0],
            "band_used_in_likelihood": True,
        }
    ).to_parquet(map_run / "batch_fit_photometry_comparison.parquet")

    comparison = compare_reconstruction_runs(
        out_dir=tmp_path / "compare",
        runs=[("nn", infer_run), ("map", map_run)],
        rowset_path=tmp_path / "rowsets" / "worst_2.txt",
    )

    assert (tmp_path / "compare" / "reconstruction_method_summary.csv").exists()
    assert comparison["report"].endswith("reconstruction_comparison.md")
