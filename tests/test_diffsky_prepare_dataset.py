from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from euclid_dsps.diffsky_data.prepare import build_diffsky_photometric_dataset


def test_prepare_dataset_creates_truth_photometry_and_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "lc_cores-001.diffsky_gals.hdf5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data")
        group.create_dataset("core_tag", data=np.asarray([10, 11]))
        group.create_dataset("redshift_true", data=np.asarray([0.1, 0.2]))
        group.create_dataset("logsm_obs", data=np.asarray([9.0, 10.0]))
        group.create_dataset("logssfr_obs", data=np.asarray([-10.0, -9.5]))
        group.create_dataset("logmp_obs", data=np.asarray([12.0, 12.5]))
        group.create_dataset("central", data=np.asarray([1, 0]))
        group.create_dataset("lgmcrit", data=np.asarray([11.0, 11.1]))
        group.create_dataset("early_index", data=np.asarray([1.0, 1.1]))
        group.create_dataset("av", data=np.asarray([0.2, 0.3]))
        group.create_dataset("lsst_u", data=np.asarray([25.0, 26.0]))
        group.create_dataset("lsst_g", data=np.asarray([24.0, 25.0]))

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
    assert {"flux_lsst_u", "fluxerr_lsst_g", "mask_lsst_g"} <= set(frame.columns)
    assert out.with_suffix(".manifest.yaml").exists()
    assert out.with_suffix(".schema.json").exists()
