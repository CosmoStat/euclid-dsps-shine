from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from euclid_dsps.openuniverse.cli import main as ou_cli_main
from euclid_dsps.openuniverse.filter_curves import (
    OpenUniverseFilterCurve,
    load_openuniverse_filter_curves,
    parse_filter_path_overrides,
)
from euclid_dsps.openuniverse.flux_closure import (
    run_sed_flux_closure,
    write_sed_flux_closure_outputs,
)
from euclid_dsps.openuniverse.photometry import (
    PLANCK_ERG_S,
    photon_rate_from_fnu_sed,
)


def test_photon_rate_from_fnu_sed_matches_flat_fnu_analytic_integral() -> None:
    wave = np.linspace(1000.0, 2000.0, 128)
    fnu = np.full_like(wave, 2.0e-29)
    curve = OpenUniverseFilterCurve(
        band_name="lsst_u",
        wave_angstrom=wave,
        transmission=np.ones_like(wave),
        source="test",
    )

    rate = photon_rate_from_fnu_sed(wave, fnu, curve, fnu_unit="fnu_cgs")
    expected = 2.0e-29 / PLANCK_ERG_S * np.log(2.0)

    assert rate == pytest.approx(expected, rel=1.0e-4)


def test_filter_loader_requires_roman_exact_unless_approx_enabled(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No filter curve"):
        load_openuniverse_filter_curves(["roman_W146"], filter_root=tmp_path)

    filters = load_openuniverse_filter_curves(
        ["roman_W146"],
        filter_root=tmp_path,
        allow_approx_filters=True,
    )
    assert filters["roman_W146"].approximate


def test_filter_loader_finds_default_roman_filename(tmp_path) -> None:
    _write_filter(tmp_path / "Roman_WFI.F146.dat")

    filters = load_openuniverse_filter_curves(["roman_W146"], filter_root=tmp_path)

    assert not filters["roman_W146"].approximate
    assert filters["roman_W146"].source.endswith("Roman_WFI.F146.dat")


def test_parse_filter_path_overrides(tmp_path) -> None:
    path = tmp_path / "filter.dat"
    parsed = parse_filter_path_overrides([f"lsst_u={path}"])

    assert parsed == {"lsst_u": path}


def test_sed_flux_closure_writes_metrics_and_calibration(tmp_path) -> None:
    sed_path = tmp_path / "galaxy_sed_1.hdf5"
    catalog_path = tmp_path / "catalog.parquet"
    filter_path = tmp_path / "u.dat"
    out_dir = tmp_path / "closure"
    _write_filter(filter_path)
    expected = _write_sed_and_catalog(sed_path, catalog_path, filter_path)

    result = run_sed_flux_closure(
        catalog_path=catalog_path,
        sed_path=sed_path,
        band_names=("lsst_u",),
        filter_paths={"lsst_u": filter_path},
        limit=2,
    )
    paths = write_sed_flux_closure_outputs(result, out_dir)

    assert result.summary["n_objects"] == 2
    assert result.calibration["calibration_factor"].iloc[0] == pytest.approx(0.5)
    assert result.metrics["median_relative_error_calibrated"].iloc[0] == pytest.approx(
        0.0,
        abs=1.0e-6,
    )
    assert result.rows["catalog_flux_photon"].tolist() == pytest.approx(expected)
    assert (out_dir / "sed_flux_closure_rows.parquet").exists()
    assert json.loads((out_dir / "sed_flux_closure_summary.json").read_text())[
        "bands"
    ] == ["lsst_u"]
    assert set(paths) == {"rows", "metrics", "calibration", "summary"}


def test_sed_flux_closure_cli_on_mock_data(tmp_path) -> None:
    sed_path = tmp_path / "galaxy_sed_1.hdf5"
    catalog_path = tmp_path / "catalog.parquet"
    filter_path = tmp_path / "u.dat"
    out_dir = tmp_path / "closure"
    _write_filter(filter_path)
    _write_sed_and_catalog(sed_path, catalog_path, filter_path)

    ou_cli_main(
        [
            "sed-flux-closure",
            "--catalog",
            str(catalog_path),
            "--sed",
            str(sed_path),
            "--bands",
            "lsst_u",
            "--filter",
            f"lsst_u={filter_path}",
            "--out",
            str(out_dir),
        ]
    )

    assert (out_dir / "sed_flux_closure_metrics.csv").exists()


def _write_filter(path) -> None:
    wave = np.linspace(1000.0, 2000.0, 32)
    transmission = np.ones_like(wave)
    np.savetxt(path, np.column_stack([wave, transmission]))


def _write_sed_and_catalog(sed_path, catalog_path, filter_path) -> list[float]:
    wave = np.linspace(1000.0, 2000.0, 32)
    fnu = np.full_like(wave, 2.0e-29)
    curve = load_openuniverse_filter_curves(
        ["lsst_u"],
        filter_paths={"lsst_u": filter_path},
    )["lsst_u"]
    rate = photon_rate_from_fnu_sed(wave, fnu, curve, fnu_unit="native")
    with h5py.File(sed_path, "w") as handle:
        handle.create_dataset("meta/wave_list", data=wave.astype(np.float32))
        group = handle.create_group("galaxy/1")
        group.create_dataset(
            "100001",
            data=np.stack([fnu, np.zeros_like(fnu), np.zeros_like(fnu)]).astype(
                np.float32
            ),
        )
        group.create_dataset(
            "100002",
            data=np.stack([2.0 * fnu, np.zeros_like(fnu), np.zeros_like(fnu)]).astype(
                np.float32
            ),
        )
    expected = [0.5 * rate, rate]
    pd.DataFrame(
        {
            "galaxy_id": [100001, 100002],
            "redshift": [0.0, 0.0],
            "flux_truth_lsst_u": expected,
        }
    ).to_parquet(catalog_path, index=False)
    return expected
