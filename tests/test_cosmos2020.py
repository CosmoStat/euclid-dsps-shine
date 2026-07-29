from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from euclid_dsps.cosmos2020 import (
    COSMOS_BANDS,
    ESO_FARMER_V21_URL,
    R_LIMIT_UJY,
    deterministic_nested_order,
    farmer_adql,
    farmer_columns,
    prepare_farmer_catalog,
    read_farmer_table,
    write_nested_subsets,
)
from scripts.download_cosmos2020_assets import _download_direct, parse_args
from scripts.validate_cosmos2020_reproduction import validate_spectral_assets


def _farmer_fixture(n_rows: int = 4) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ID": np.arange(10, 10 + n_rows),
            "ALPHA_J2000": np.linspace(149.0, 150.0, n_rows),
            "DELTA_J2000": np.linspace(1.5, 2.5, n_rows),
            "FLAG_COMBINED": [0, 1, 0, 0],
            "EBV_MW": [0.0, 0.0, 0.1, 0.0],
            "lp_type": [0, 0, 1, 0],
            "lp_zBEST": [0.2, 0.3, 1.0, 2.0],
        }
    )
    for band in COSMOS_BANDS:
        frame[band.flux_column] = np.full(n_rows, 1.0)
        frame[band.error_column] = np.full(n_rows, 0.1)
        frame[band.valid_column] = np.ones(n_rows, dtype=int)
    frame.loc[3, "HSC_r_FLUX"] = R_LIMIT_UJY / 2.0
    return frame


def test_farmer_contract_has_public_a24_order_and_columns() -> None:
    assert len(COSMOS_BANDS) == 26
    assert COSMOS_BANDS[0].name == "u_megaprime_sagem"
    assert COSMOS_BANDS[-1].name == "irac2_cosmos"
    assert len(farmer_columns()) == 7 + 3 * 26
    assert "COSMOS2020_FARMER_V1" in farmer_adql(32)
    assert "TOP 32" in farmer_adql(32)


def test_downloader_exposes_async_tap_resume(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "download_cosmos2020_assets.py",
            "--out",
            "assets",
            "--tap-job-url",
            "https://archive.example/async/123",
            "--timeout",
            "14400",
        ],
    )
    args = parse_args()
    assert args.tap_job_url.endswith("/123")
    assert args.timeout == 14400
    assert args.farmer_url == ESO_FARMER_V21_URL


class _FakeResponse:
    def __init__(self, payload: bytes, status: int) -> None:
        self.payload = payload
        self.status = status
        self.offset = 0

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self, size: int) -> bytes:
        block = self.payload[self.offset : self.offset + size]
        self.offset += len(block)
        return block


def test_direct_farmer_download_resumes_partial_file(
    tmp_path, monkeypatch
) -> None:
    target = tmp_path / "farmer.fits"
    partial = tmp_path / "farmer.fits.part"
    partial.write_bytes(b"abc")

    def fake_urlopen(request, timeout):
        assert request.headers["Range"] == "bytes=3-"
        assert timeout == 300
        return _FakeResponse(b"def", 206)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    _download_direct(target, "https://archive.example/farmer", expected_size=6)
    assert target.read_bytes() == b"abcdef"
    assert not partial.exists()


def test_prepare_farmer_applies_selection_and_extinction() -> None:
    fixture = _farmer_fixture()
    fixture.loc[0, "HSC_g_VALID"] = 0
    selected, manifest = prepare_farmer_catalog(fixture)
    assert selected["object_id"].tolist() == [10]
    assert manifest["selected_rows"] == 1
    assert manifest["selection_modelled_in_rws"] is False
    assert selected.loc[0, "flux_hsc_g"] == 1.0
    assert np.isnan(selected.loc[0, "fluxerr_hsc_g"])


def test_read_farmer_fits_only_materializes_required_columns(tmp_path) -> None:
    from astropy.table import Table

    fixture = _farmer_fixture()
    fixture["UNUSED_LARGE_COLUMN"] = "unused"
    path = tmp_path / "farmer.fits"
    Table.from_pandas(fixture).write(path)
    loaded = read_farmer_table(path)
    assert set(loaded.columns) == set(farmer_columns())


def test_nested_subsets_are_deterministic_and_nested(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "row_index": np.arange(10),
            "object_id": np.arange(100, 110),
            "flux_hsc_r": np.ones(10),
        }
    )
    first = deterministic_nested_order(frame["object_id"], 7)
    second = deterministic_nested_order(frame["object_id"], 7)
    np.testing.assert_array_equal(first, second)
    payload = write_nested_subsets(frame, tmp_path, sizes=(3, 6), seed=7)
    small = pd.read_parquet(payload["paths"]["3"])
    large = pd.read_parquet(payload["paths"]["6"])
    assert set(small["object_id"]).issubset(set(large["object_id"]))
    assert small["object_id"].tolist() == large["object_id"].iloc[:3].tolist()


def test_spectral_asset_validation_checks_content(tmp_path) -> None:
    asset = tmp_path / "asset.h5"
    asset.write_bytes(b"known spectral asset")
    expected = {
        str(asset): (
            "6e06de7b8d1822462b6eb14534f51b6b73a78358f2957075760ab3e6e93ba526"
        )
    }
    validate_spectral_assets(expected)
    asset.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_spectral_assets(expected)
