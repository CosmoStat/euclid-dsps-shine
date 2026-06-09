from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from euclid_dsps.openuniverse.cli import main as ou_cli_main
from euclid_dsps.openuniverse.fit_ready import (
    compute_lensing_magnification,
    make_openuniverse_fit_ready_table,
)
from euclid_dsps.openuniverse.photometry import photon_rate_to_fnu_cgs
from euclid_dsps.openuniverse.schema import OU_LSST_ROMAN_14_BANDS


def test_compute_lensing_magnification() -> None:
    mu = compute_lensing_magnification(
        convergence=np.asarray([0.0, 0.1]),
        shear1=np.asarray([0.0, 0.0]),
        shear2=np.asarray([0.0, 0.0]),
    )

    assert mu.tolist() == pytest.approx([1.0, 1.0 / 0.9**2])


def test_make_fit_ready_table_preserves_photon_fluxes_and_writes_fnu(tmp_path) -> None:
    input_path = tmp_path / "prepared.parquet"
    main_path = tmp_path / "main.parquet"
    filter_root = tmp_path / "filters"
    output_path = tmp_path / "fit_ready.parquet"
    filter_root.mkdir()
    _write_filters(filter_root)
    _write_prepared(input_path)
    _write_main(main_path)

    manifest = make_openuniverse_fit_ready_table(
        input_path=input_path,
        main_path=main_path,
        output_path=output_path,
        filter_root=filter_root,
        band_names=("lsst_u",),
    )

    frame = pd.read_parquet(output_path)
    ab0 = manifest["ab0_photon_rate_by_band"]["lsst_u"]
    mu = frame["mu_lensing"].to_numpy()[0]
    assert mu == pytest.approx(1.0 / 0.9**2)
    assert frame["flux_lensed_photon_lsst_u"].iloc[0] == 100.0
    assert frame["flux_unlensed_photon_lsst_u"].iloc[0] == pytest.approx(100.0 / mu)
    assert frame["flux_lsst_u"].iloc[0] == pytest.approx(
        photon_rate_to_fnu_cgs(100.0 / mu, ab0)
    )
    assert frame["fluxerr_lsst_u"].iloc[0] == pytest.approx(
        photon_rate_to_fnu_cgs(2.0 / mu, ab0)
    )
    assert frame["photometry_unit"].iloc[0] == "fnu_cgs"
    assert output_path.with_suffix(".manifest.yaml").exists()
    assert yaml.safe_load(output_path.with_suffix(".manifest.yaml").read_text())[
        "lensing_mode"
    ] == "unlensed"
    assert manifest["filter_response_mode"] == "dsps_clipped"


def test_make_fit_ready_clips_filter_response_by_default(tmp_path) -> None:
    input_path = tmp_path / "prepared.parquet"
    main_path = tmp_path / "main.parquet"
    filter_root = tmp_path / "filters"
    clipped_path = tmp_path / "fit_ready_clipped.parquet"
    native_path = tmp_path / "fit_ready_native.parquet"
    filter_root.mkdir()
    _write_filters(filter_root, transmission_scale=2.0)
    _write_prepared(input_path)
    _write_main(main_path)

    clipped_manifest = make_openuniverse_fit_ready_table(
        input_path=input_path,
        main_path=main_path,
        output_path=clipped_path,
        filter_root=filter_root,
        band_names=("lsst_u",),
    )
    native_manifest = make_openuniverse_fit_ready_table(
        input_path=input_path,
        main_path=main_path,
        output_path=native_path,
        filter_root=filter_root,
        band_names=("lsst_u",),
        filter_response_mode="native",
    )

    assert native_manifest["ab0_photon_rate_by_band"]["lsst_u"] == pytest.approx(
        2.0 * clipped_manifest["ab0_photon_rate_by_band"]["lsst_u"]
    )
    clipped = pd.read_parquet(clipped_path)
    native = pd.read_parquet(native_path)
    assert clipped["flux_lsst_u"].iloc[0] == pytest.approx(
        2.0 * native["flux_lsst_u"].iloc[0]
    )


def test_make_fit_ready_cli(tmp_path) -> None:
    input_path = tmp_path / "prepared.parquet"
    main_path = tmp_path / "main.parquet"
    filter_root = tmp_path / "filters"
    output_path = tmp_path / "fit_ready.parquet"
    filter_root.mkdir()
    _write_filters(filter_root)
    _write_prepared(input_path)
    _write_main(main_path)

    ou_cli_main(
        [
            "make-fit-ready",
            "--input",
            str(input_path),
            "--main",
            str(main_path),
            "--out",
            str(output_path),
            "--filter-root",
            str(filter_root),
        ]
    )

    assert output_path.exists()
    frame = pd.read_parquet(output_path)
    assert frame["photometry_unit"].tolist() == ["fnu_cgs", "fnu_cgs"]


def _write_filters(root, transmission_scale: float = 1.0) -> None:
    wave = np.linspace(1000.0, 2000.0, 32)
    transmission = np.ones_like(wave) * float(transmission_scale)
    for band in OU_LSST_ROMAN_14_BANDS:
        if band.startswith("lsst_"):
            suffix = band.split("_", 1)[1]
            path = root / f"LSST_LSST.{suffix}.dat"
        else:
            suffix = {
                "roman_R062": "F062",
                "roman_Z087": "F087",
                "roman_Y106": "F106",
                "roman_J129": "F129",
                "roman_H158": "F158",
                "roman_F184": "F184",
                "roman_K213": "F213",
                "roman_W146": "F146",
            }[band]
            path = root / f"Roman_WFI.{suffix}.dat"
        np.savetxt(path, np.column_stack([wave, transmission]))


def _write_prepared(path) -> None:
    frame = pd.DataFrame({"galaxy_id": [1, 2], "redshift": [0.1, 0.2]})
    for band in OU_LSST_ROMAN_14_BANDS:
        frame[f"flux_truth_{band}"] = [100.0, 200.0]
        frame[f"flux_{band}"] = [100.0, 200.0]
        frame[f"fluxerr_{band}"] = [2.0, 4.0]
        frame[f"mask_{band}"] = [True, True]
    frame.to_parquet(path, index=False)


def _write_main(path) -> None:
    pd.DataFrame(
        {
            "galaxy_id": [1, 2],
            "convergence": [0.1, 0.0],
            "shear1": [0.0, 0.0],
            "shear2": [0.0, 0.0],
        }
    ).to_parquet(path, index=False)
