from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_fsps_agn_grid
import generate_fsps_gas_grid


class FakeStellarPopulation:
    zlegend = np.asarray([0.001, 0.01], dtype=float)
    isoc_library = "MIST"
    spec_library = "C3K"
    fsps_version = "fake-fsps"

    def __init__(self, **kwargs):
        self.params = dict(kwargs)
        self.zcontinuous = int(kwargs.get("zcontinuous", 0))

    def get_spectrum(self, zmet=None, tage=0.0, peraa=False):
        wave = np.asarray([1000.0, 2000.0, 3000.0], dtype=float)
        if tage == 0.0:
            met_factor = 1.0 if self.zcontinuous > 0 else float(zmet or 1.0)
            gas_factor = 1.0 + 0.1 * float(self.params.get("gas_logz", 0.0))
            gas_factor += 0.01 * float(self.params.get("gas_logu", 0.0))
            age_factor = np.asarray([[1.0], [0.5]], dtype=float)
            wave_factor = np.asarray([[1.0, 1.5, 2.0]], dtype=float)
            return wave, 1.0e-10 * met_factor * gas_factor * age_factor * wave_factor

        stellar = np.asarray([1.0, 2.0, 3.0], dtype=float) * 1.0e-10
        fagn = float(self.params.get("fagn", 0.0))
        agn_tau = float(self.params.get("agn_tau", 10.0))
        agn = fagn * agn_tau * np.asarray([1.0, 1.2, 1.4], dtype=float) * 1.0e-12
        return wave, stellar + agn


def test_gas_generator_help_does_not_require_fsps() -> None:
    env = dict(os.environ)
    env.pop("SPS_HOME", None)

    result = subprocess.run(
        [sys.executable, "scripts/generate_fsps_gas_grid.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "python-fsps" in result.stdout


def test_gas_generator_missing_sps_home_is_actionable(tmp_path) -> None:
    env = dict(os.environ)
    env.pop("SPS_HOME", None)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_fsps_gas_grid.py",
            "--output",
            str(tmp_path / "gas.h5"),
            "--skip-model-validation",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "SPS_HOME is not set" in result.stderr


def test_gas_generator_writes_dsps_shape_with_fake_fsps(
    tmp_path, monkeypatch
) -> None:
    _install_fake_fsps(monkeypatch, tmp_path)
    reference = _write_reference_ssp(tmp_path)
    output = tmp_path / "gas_grid.h5"

    status = generate_fsps_gas_grid.main(
        [
            "--output",
            str(output),
            "--reference-ssp",
            str(reference),
            "--gas-lgmet-grid",
            "-1.0",
            "0.0",
            "--gas-lgu-grid",
            "-3.0",
            "-2.0",
            "--skip-model-validation",
        ]
    )

    assert status == 0
    with h5py.File(output, "r") as handle:
        assert set(
            [
                "ssp_wave",
                "ssp_lg_age_gyr",
                "ssp_lgmet",
                "gas_lgmet_grid",
                "gas_lgu_grid",
                "ssp_flux",
            ]
        ).issubset(handle.keys())
        assert handle["ssp_flux"].shape == (2, 2, 2, 2, 3)
        assert handle.attrs["units_ssp_flux"] == "Lsun/Hz/Msun formed"
        assert "independently" in handle.attrs["scientific_caveat"]


def test_agn_generator_writes_template_with_fake_fsps(
    tmp_path, monkeypatch
) -> None:
    _install_fake_fsps(monkeypatch, tmp_path)
    output = tmp_path / "agn_grid.h5"

    status = generate_fsps_agn_grid.main(
        [
            "--output",
            str(output),
            "--agn-tau-grid",
            "5.0",
            "10.0",
            "--no-progress",
            "--skip-model-validation",
        ]
    )

    assert status == 0
    with h5py.File(output, "r") as handle:
        assert handle["wave"].shape == (3,)
        assert handle["agn_tau_grid"][:].tolist() == [5.0, 10.0]
        assert handle["template_lnu_per_lbol"].shape == (2, 3)
        assert np.all(handle["template_lnu_per_lbol"][:] > 0.0)
        assert handle.attrs["normalization_status"] == "approximate"


def test_agn_generator_default_tau_grid_matches_popcosmos_config() -> None:
    args = generate_fsps_agn_grid.parse_args([])

    assert args.agn_tau_grid == [
        5.0,
        10.0,
        20.0,
        30.0,
        40.0,
        60.0,
        80.0,
        100.0,
        150.0,
    ]


def _install_fake_fsps(monkeypatch, tmp_path: Path) -> None:
    fake_module = SimpleNamespace(
        __version__="fake-python-fsps",
        StellarPopulation=FakeStellarPopulation,
    )
    monkeypatch.setitem(sys.modules, "fsps", fake_module)
    monkeypatch.setenv("SPS_HOME", str(tmp_path))


def _write_reference_ssp(tmp_path: Path) -> Path:
    reference = tmp_path / "reference_ssp.h5"
    with h5py.File(reference, "w") as handle:
        handle["ssp_wave"] = np.asarray([1000.0, 2000.0, 3000.0], dtype=np.float32)
        handle["ssp_lg_age_gyr"] = np.asarray([-3.0, -2.0], dtype=np.float32)
        handle["ssp_lgmet"] = np.log10(FakeStellarPopulation.zlegend).astype(np.float32)
    return reference
