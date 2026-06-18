from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from euclid_dsps.diffsky_data.prepare import build_diffsky_photometric_dataset


def _write_hltds_shard(path: Path, core_tag: list[int]) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data")
        n = len(core_tag)
        group.create_dataset("core_tag", data=np.asarray(core_tag))
        group.create_dataset("redshift_true", data=np.linspace(0.1, 0.2, n))
        group.create_dataset("logsm_obs", data=np.linspace(9.0, 10.0, n))
        group.create_dataset("logssfr_obs", data=np.linspace(-10.0, -9.5, n))
        group.create_dataset("logmp_obs", data=np.linspace(12.0, 12.5, n))
        group.create_dataset("central", data=np.asarray([1, 0][:n]))
        group.create_dataset("lgmcrit", data=np.linspace(11.0, 11.1, n))
        group.create_dataset("early_index", data=np.linspace(1.0, 1.1, n))
        group.create_dataset("av", data=np.linspace(0.2, 0.3, n))
        group.create_dataset("lsst_u", data=np.linspace(25.0, 26.0, n))
        group.create_dataset("lsst_g", data=np.linspace(24.0, 25.0, n))


def test_prepare_dataset_creates_truth_photometry_and_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "lc_cores-001.diffsky_gals.hdf5"
    _write_hltds_shard(path, [10, 11])

    out = tmp_path / "processed.parquet"
    report = build_diffsky_photometric_dataset(
        raw_root=raw,
        inventory_path=None,
        output_path=out,
        snr=20.0,
    )
    frame = pd.read_parquet(out)

    assert report.readiness == "READY_EXTENDED"
    assert frame["logsfr_true"].tolist() == [-1.0, 0.5]
    assert frame["core_tag"].tolist() == [10, 11]
    assert frame["object_id"].tolist() == [10, 11]
    assert {"flux_lsst_u", "fluxerr_lsst_g", "mask_lsst_g"} <= set(frame.columns)
    assert out.with_suffix(".manifest.yaml").exists()
    assert out.with_suffix(".schema.json").exists()
    assert (tmp_path / "diffsky_dataset_integrity_report.md").exists()
    manifest = yaml.safe_load(out.with_suffix(".manifest.yaml").read_text())
    schema = json.loads(out.with_suffix(".schema.json").read_text())
    assert manifest["error_model"]["type"] == "fractional_snr"
    assert manifest["error_model"]["synthetic"] is True
    assert "diffstar_lgmcrit" in manifest["generated_truth_columns"]
    assert "dust_av" in schema["column_semantics"]["generated_truth"]
    assert "logsfr_true" in schema["column_semantics"]["derived_truth"]


def test_prepare_dataset_duplicate_core_tag_gets_global_object_id(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_hltds_shard(raw / "lc_cores-001.diffsky_gals.hdf5", [10])
    _write_hltds_shard(raw / "lc_cores-002.diffsky_gals.hdf5", [10])

    out = tmp_path / "processed.parquet"
    build_diffsky_photometric_dataset(
        raw_root=raw,
        inventory_path=None,
        output_path=out,
        snr=20.0,
    )

    frame = pd.read_parquet(out)
    manifest = yaml.safe_load(out.with_suffix(".manifest.yaml").read_text())
    assert frame["core_tag"].tolist() == [10, 10]
    assert frame["object_id"].is_unique
    assert "global_object_id" in frame
    assert manifest["object_id"]["strategy"] == "global_object_id_from_source_order"
    assert manifest["object_id"]["core_tag_unique_global"] is False


def test_prepare_dataset_noerr_does_not_claim_native_errors(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_hltds_shard(raw / "lc_cores-001.diffsky_gals.hdf5", [10, 11])

    out = tmp_path / "processed.parquet"
    build_diffsky_photometric_dataset(
        raw_root=raw,
        inventory_path=None,
        output_path=out,
        add_synthetic_errors=False,
    )

    frame = pd.read_parquet(out)
    manifest = yaml.safe_load(out.with_suffix(".manifest.yaml").read_text())
    assert not any(column.startswith("fluxerr_") for column in frame.columns)
    assert manifest["error_model"]["type"] == "none"
    assert manifest["error_model"]["native_error"] is False
