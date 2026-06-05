from __future__ import annotations

import pandas as pd
import yaml

from euclid_dsps.cli import main as cli_main
from euclid_dsps.openuniverse.prepare import prepare_openuniverse_lsst_roman_subset
from euclid_dsps.openuniverse.schema import (
    OU_FLUX_COLUMNS,
    OU_LSST_ROMAN_14_BANDS,
    normalized_flux_column,
    normalized_flux_truth_column,
    normalized_fluxerr_column,
    normalized_mask_column,
)


def test_prepare_openuniverse_subset_writes_14_band_parquet_and_manifest(
    tmp_path,
) -> None:
    _write_mock_hpix(tmp_path, 9812, n_rows=5)
    out = tmp_path / "processed" / "ou_lsst_roman_14.parquet"

    manifest = prepare_openuniverse_lsst_roman_subset(
        hpix_ids=[9812],
        input_root=tmp_path,
        output_path=out,
        limit=4,
        min_flux_valid_bands=8,
        noise_model={"type": "fractional_snr", "snr": 50},
        seed=11,
    )

    frame = pd.read_parquet(out)
    assert len(frame) == 4
    assert manifest["number_of_rows"] == 4
    assert (tmp_path / "processed" / "ou_lsst_roman_14.manifest.yaml").exists()
    assert {
        "galaxy_id",
        "ra",
        "dec",
        "redshift",
        "redshiftHubble",
        "stellar_mass",
    } <= set(frame)
    for band in OU_LSST_ROMAN_14_BANDS:
        assert normalized_flux_truth_column(band) in frame
        assert normalized_flux_column(band) in frame
        assert normalized_fluxerr_column(band) in frame
        assert normalized_mask_column(band) in frame
        assert frame[normalized_fluxerr_column(band)].gt(0.0).all()


def test_openuniverse_prepare_cli_on_mock_hpix(tmp_path) -> None:
    _write_mock_hpix(tmp_path, 9813, n_rows=3)
    out = tmp_path / "cli" / "subset.parquet"

    cli_main(
        [
            "--config",
            "configs/openuniverse_lsst_roman_14.yaml",
            "openuniverse-prepare",
            "--input-root",
            str(tmp_path),
            "--hpix",
            "9813",
            "--limit",
            "2",
            "--out",
            str(out),
        ]
    )

    frame = pd.read_parquet(out)
    manifest = yaml.safe_load(out.with_suffix(".manifest.yaml").read_text())
    assert len(frame) == 2
    assert manifest["hpix_ids"] == [9813]
    assert manifest["flux_unit"] == "photon_per_sec_cm2"


def _write_mock_hpix(root, hpix: int, *, n_rows: int) -> None:
    galaxy_id = list(range(1000, 1000 + n_rows))
    main = pd.DataFrame(
        {
            "galaxy_id": galaxy_id,
            "ra": [10.0 + index for index in range(n_rows)],
            "dec": [-1.0 - index for index in range(n_rows)],
            "redshift": [0.1 + 0.01 * index for index in range(n_rows)],
            "redshiftHubble": [0.11 + 0.01 * index for index in range(n_rows)],
            "peculiarVelocity": [0.0] * n_rows,
            "um_source_galaxy_obs_sm": [1.0e10 + index for index in range(n_rows)],
        }
    )
    flux = pd.DataFrame({"galaxy_id": galaxy_id})
    for band_index, column in enumerate(OU_FLUX_COLUMNS.values()):
        flux[column] = [100.0 + 10.0 * band_index + index for index in range(n_rows)]
    main.to_parquet(root / f"galaxy_{hpix}.parquet", index=False)
    flux.to_parquet(root / f"galaxy_flux_{hpix}.parquet", index=False)
