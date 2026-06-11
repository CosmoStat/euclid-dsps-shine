from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from euclid_dsps.diffsky_data.validation import validate_for_prior_learning


def test_validation_returns_ready_extended(tmp_path: Path) -> None:
    path = tmp_path / "dataset.parquet"
    pd.DataFrame(
        {
            "object_id": [1, 2],
            "redshift_true": [0.1, 0.2],
            "logsm_true": [9.0, 10.0],
            "flux_lsst_u": [1.0, 2.0],
            "mask_lsst_u": [True, True],
            "diffstar_lgmcrit": [11.0, 11.1],
            "diffmah_early_index": [1.0, 1.1],
        }
    ).to_parquet(path)

    report = validate_for_prior_learning(path)

    assert report["readiness"] == "READY_EXTENDED"
    assert report["n_bands"] == 1


def test_validation_preserves_no_native_error_semantics(tmp_path: Path) -> None:
    path = tmp_path / "dataset.parquet"
    manifest_path = tmp_path / "dataset.manifest.yaml"
    pd.DataFrame(
        {
            "object_id": [1, 2],
            "redshift_true": [0.1, 0.2],
            "logsm_true": [9.0, 10.0],
            "flux_lsst_u": [1.0, 2.0],
            "mask_lsst_u": [True, True],
        }
    ).to_parquet(path)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "band_names": ["lsst_u"],
                "error_model": {"type": "none", "native_error": False},
            }
        ),
        encoding="utf-8",
    )

    report = validate_for_prior_learning(path, manifest_path)

    assert report["error_model"]["type"] == "none"
    assert report["error_model"]["native_error"] is False
    assert report["n_fluxerr_columns"] == 0
