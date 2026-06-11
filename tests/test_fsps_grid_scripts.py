from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_compressed_agn_component_grid  # noqa: E402
import build_compressed_gas_grid  # noqa: E402
import build_compressed_ssp_grid  # noqa: E402
import generate_fsps_agn_component_grid  # noqa: E402
import generate_fsps_agn_grid  # noqa: E402
import generate_fsps_gas_grid  # noqa: E402
import generate_fsps_ssp_grid  # noqa: E402


class FakeStellarPopulation:
    zlegend = np.asarray([0.001, 0.01], dtype=float)
    ssp_ages = np.asarray([6.0, 7.0], dtype=float)
    emline_wavelengths = np.asarray([5007.0, 6563.0], dtype=float)
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
            fagn = float(self.params.get("fagn", 0.0))
            agn_tau = float(self.params.get("agn_tau", 10.0))
            agn = (
                fagn
                * agn_tau
                * np.asarray([[1.0, 1.2, 1.4], [0.5, 0.6, 0.7]], dtype=float)
                * 1.0e-12
            )
            self.emline_luminosity = (
                1.0e-4
                * met_factor
                * gas_factor
                * np.asarray([[1.0, 2.0], [0.5, 1.0]], dtype=float)
            )
            self.stellar_mass = np.asarray([0.9, 0.6], dtype=float)
            return (
                wave,
                1.0e-10 * met_factor * gas_factor * age_factor * wave_factor + agn,
            )

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


def test_ssp_generator_help_does_not_require_fsps() -> None:
    env = dict(os.environ)
    env.pop("SPS_HOME", None)

    result = subprocess.run(
        [sys.executable, "scripts/generate_fsps_ssp_grid.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "python-fsps" in result.stdout


def test_compressed_agn_builder_help_does_not_load_dense_grid() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_compressed_agn_component_grid.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--k" in result.stdout
    assert "--normalization" in result.stdout


def test_compressed_gas_builder_help_does_not_load_dense_grid() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_compressed_gas_grid.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--k" in result.stdout
    assert "--normalization" in result.stdout


def test_dense_vs_compressed_benchmark_help_does_not_load_assets() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_dense_vs_compressed_spectral_assets.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--dense-config" in result.stdout
    assert "--compressed-gas-grid" in result.stdout
    assert "--dense-agn-mode" in result.stdout


def test_photometry_engines_benchmark_help_does_not_load_assets() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_photometry_engines.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "dsps_dense_lazy" in result.stdout
    assert "dsps_dense_resident" in result.stdout
    assert "fsps_prospector" in result.stdout
    assert "--compressed-gas-grid" in result.stdout


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


def test_ssp_generator_writes_dsps_shape_with_fake_fsps(
    tmp_path, monkeypatch
) -> None:
    _install_fake_fsps(monkeypatch, tmp_path)
    line_reference = _write_line_name_reference(tmp_path)
    output = tmp_path / "ssp_grid.h5"

    status = generate_fsps_ssp_grid.main(
        [
            "--output",
            str(output),
            "--reference-line-names",
            str(line_reference),
            "--no-progress",
        ]
    )

    assert status == 0
    with h5py.File(output, "r") as handle:
        assert set(
            [
                "ssp_wave",
                "ssp_lg_age_gyr",
                "ssp_lgmet",
                "ssp_flux",
                "ssp_surviving_mstar",
                "ssp_emline_wave",
                "ssp_emline_name",
                "ssp_emline_luminosity",
            ]
        ).issubset(handle.keys())
        assert handle["ssp_flux"].shape == (2, 2, 3)
        assert handle["ssp_surviving_mstar"].shape == (2, 2)
        np.testing.assert_allclose(handle["ssp_surviving_mstar"][0], [0.9, 0.6])
        assert handle["ssp_emline_luminosity"].shape == (2, 2, 2)
        assert handle["ssp_emline_name"][:].tolist() == [b"OIII_5007", b"Halpha"]
        assert handle.attrs["units_ssp_flux"] == "Lsun/Hz/Msun formed"
        assert handle.attrs["imf_type"] == 1
        assert handle.attrs["imf_name"] == "chabrier"
        assert handle.attrs["z_sun"] == 0.0142


def test_ssp_generator_writes_pure_stellar_no_nebular_grid(
    tmp_path, monkeypatch
) -> None:
    _install_fake_fsps(monkeypatch, tmp_path)
    output = tmp_path / "ssp_no_ne.h5"

    status = generate_fsps_ssp_grid.main(
        [
            "--stellar-only",
            "--output",
            str(output),
            "--no-progress",
        ]
    )

    assert status == 0
    with h5py.File(output, "r") as handle:
        assert handle["ssp_flux"].shape == (2, 2, 3)
        assert handle["ssp_surviving_mstar"].shape == (2, 2)
        np.testing.assert_allclose(handle["ssp_surviving_mstar"][0], [0.9, 0.6])
        assert "ssp_emline_luminosity" not in handle
        assert "ssp_emline_wave" not in handle
        assert handle.attrs["asset_kind"] == "popcosmos_chabrier_stellar_only_ssp"
        assert handle.attrs["add_neb_emission"] == 0
        assert handle.attrs["add_neb_continuum"] == 0
        assert handle.attrs["imf_type"] == 1
        assert handle.attrs["imf_name"] == "chabrier"
        assert handle.attrs["z_sun"] == 0.0142
        assert "disabled" in handle.attrs["nebular_reference"]


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
        assert handle.attrs["imf_type"] == 1
        assert handle.attrs["imf_name"] == "chabrier"
        assert handle.attrs["z_sun"] == 0.0142
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
        assert handle.attrs["imf_type"] == 1
        assert handle.attrs["imf_name"] == "chabrier"
        assert handle.attrs["z_sun"] == 0.0142
        assert handle.attrs["normalization_status"] == "approximate"


def test_agn_generator_writes_signed_audit_grid_with_fake_fsps(
    tmp_path, monkeypatch
) -> None:
    _install_fake_fsps(monkeypatch, tmp_path)
    output = tmp_path / "agn_audit_grid.h5"

    status = generate_fsps_agn_grid.main(
        [
            "--output",
            str(output),
            "--agn-tau-grid",
            "5.0",
            "10.0",
            "--fagn-grid",
            "0.001",
            "0.01",
            "--tage-grid",
            "1.0",
            "3.0",
            "--stellar-logzsol-grid",
            "-1.0",
            "0.0",
            "--signed-delta",
            "--no-progress",
            "--skip-model-validation",
        ]
    )

    assert status == 0
    with h5py.File(output, "r") as handle:
        assert handle["wave"].shape == (3,)
        assert handle["fagn_grid"][:].tolist() == pytest.approx([0.001, 0.01])
        assert handle["agn_tau_grid"][:].tolist() == [5.0, 10.0]
        assert handle["tage_gyr_grid"][:].tolist() == [1.0, 3.0]
        assert handle["stellar_logzsol_grid"][:].tolist() == [-1.0, 0.0]
        assert handle["template_lnu_per_lbol"].shape == (2, 2, 2, 2, 3)
        assert handle.attrs["asset_kind"] == (
            "popcosmos_chabrier_agn_fspsdiff_audit_grid"
        )
        assert handle.attrs["normalization_status"] == "audit"
        assert bool(handle.attrs["signed_delta"]) is True


def test_agn_component_generator_writes_fsps_native_grid_with_fake_fsps(
    tmp_path, monkeypatch
) -> None:
    _install_fake_fsps(monkeypatch, tmp_path)
    reference = _write_reference_ssp(tmp_path)
    output = tmp_path / "agn_component_grid.h5"

    status = generate_fsps_agn_component_grid.main(
        [
            "--output",
            str(output),
            "--reference-ssp",
            str(reference),
            "--fagn-grid",
            "0.001",
            "0.01",
            "--agn-tau-grid",
            "5.0",
            "10.0",
            "--no-progress",
        ]
    )

    assert status == 0
    with h5py.File(output, "r") as handle:
        assert handle["ssp_wave"].shape == (3,)
        assert handle["ssp_lg_age_gyr"].shape == (2,)
        assert handle["ssp_lgmet"].shape == (2,)
        assert handle["fagn_grid"][:].tolist() == pytest.approx([0.001, 0.01])
        assert handle["agn_tau_grid"][:].tolist() == [5.0, 10.0]
        assert handle["agn_lnu_per_mformed"].shape == (2, 2, 2, 2, 3)
        assert np.nanmax(handle["agn_lnu_per_mformed"][:]) > 0.0
        assert handle.attrs["asset_kind"] == (
            "popcosmos_chabrier_agn_component_ssp_grid"
        )
        assert handle.attrs["normalization_status"] == "fsps_native_component"
        assert handle.attrs["units_agn_lnu_per_mformed"] == "Lsun/Hz/Msun formed"


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


def test_compressed_agn_builder_round_trips_low_rank_grid(tmp_path) -> None:
    dense_path = tmp_path / "dense_agn.h5"
    compressed_path = tmp_path / "compressed_agn.h5"
    wave = np.asarray([1000.0, 2000.0, 3000.0, 4000.0, 5000.0], dtype=np.float32)
    age = np.asarray([-3.0, -2.0, -1.0], dtype=np.float32)
    lgmet = np.asarray([-2.0, -1.0], dtype=np.float32)
    fagn = np.asarray([1.0e-4, 1.0e-2], dtype=np.float32)
    tau = np.asarray([5.0, 20.0], dtype=np.float32)
    basis0 = 1.0 + wave / wave.max()
    basis1 = 0.4 + 0.2 * wave / wave.max()
    dense = np.zeros((len(fagn), len(tau), len(lgmet), len(age), len(wave)), dtype=np.float32)
    for i, fagn_value in enumerate(fagn):
        for j, tau_value in enumerate(tau):
            for m, met_value in enumerate(lgmet):
                for a, age_value in enumerate(age):
                    dense[i, j, m, a] = (
                        fagn_value * tau_value * (1.0 + m) * basis0
                        + 0.01 * (1.0 + a) * (1.0 + abs(age_value)) * basis1
                    )
    with h5py.File(dense_path, "w") as handle:
        handle["ssp_wave"] = wave
        handle["ssp_lg_age_gyr"] = age
        handle["ssp_lgmet"] = lgmet
        handle["fagn_grid"] = fagn
        handle["agn_tau_grid"] = tau
        handle["agn_lnu_per_mformed"] = dense
        handle.attrs["asset_kind"] = "popcosmos_chabrier_agn_component_ssp_grid"
        handle.attrs["imf_type"] = 1
        handle.attrs["imf_name"] = "chabrier"
        handle.attrs["z_sun"] = 0.0142
        handle.attrs["units_agn_lnu_per_mformed"] = "Lsun/Hz/Msun formed"

    args = SimpleNamespace(
        input=str(dense_path),
        output=str(compressed_path),
        k=2,
        oversample=2,
        seed=0,
        normalization="none",
        compression="none",
        gzip_level=4,
        overwrite=True,
        no_progress=True,
    )
    output = build_compressed_agn_component_grid.build_compressed_grid(args)
    summary = build_compressed_agn_component_grid.validate_compressed_grid(output)

    assert summary["k_basis"] == 2
    with h5py.File(output, "r") as handle:
        coeff = np.asarray(handle["agn_coeff"])
        basis = np.asarray(handle["agn_basis"])
        scale = np.asarray(handle["agn_scale"])
    reconstructed = (coeff @ basis) * scale[..., None]
    np.testing.assert_allclose(reconstructed, dense, rtol=2.0e-5, atol=1.0e-7)


def test_compressed_agn_builder_can_factor_linear_fagn_axis(tmp_path) -> None:
    dense_path = tmp_path / "dense_agn_linear.h5"
    compressed_path = tmp_path / "compressed_agn_linear.h5"
    wave = np.asarray([1000.0, 2000.0, 3000.0, 4000.0], dtype=np.float32)
    age = np.asarray([-3.0, -2.0], dtype=np.float32)
    lgmet = np.asarray([-2.0], dtype=np.float32)
    fagn = np.asarray([0.1, 1.0], dtype=np.float32)
    tau = np.asarray([5.0, 20.0], dtype=np.float32)
    shape = (len(fagn), len(tau), len(lgmet), len(age), len(wave))
    dense = np.zeros(shape, dtype=np.float32)
    template = np.asarray([1.0, 2.0, 1.5, 0.5], dtype=np.float32)
    for i, fagn_value in enumerate(fagn):
        for j, tau_value in enumerate(tau):
            for a, _age_value in enumerate(age):
                dense[i, j, 0, a] = fagn_value * (tau_value + a) * template
    with h5py.File(dense_path, "w") as handle:
        handle["ssp_wave"] = wave
        handle["ssp_lg_age_gyr"] = age
        handle["ssp_lgmet"] = lgmet
        handle["fagn_grid"] = fagn
        handle["agn_tau_grid"] = tau
        handle["agn_lnu_per_mformed"] = dense
        handle.attrs["asset_kind"] = "popcosmos_chabrier_agn_component_ssp_grid"
        handle.attrs["imf_type"] = 1
        handle.attrs["imf_name"] = "chabrier"
        handle.attrs["z_sun"] = 0.0142

    args = SimpleNamespace(
        input=str(dense_path),
        output=str(compressed_path),
        k=1,
        oversample=1,
        seed=0,
        normalization="none",
        compression="none",
        gzip_level=4,
        overwrite=True,
        no_progress=True,
        factor_fagn=True,
        basis_dtype="float32",
        coeff_dtype="float16",
    )
    output = build_compressed_agn_component_grid.build_compressed_grid(args)
    summary = build_compressed_agn_component_grid.validate_compressed_grid(output)

    assert summary["fagn_handling"] == "linear_runtime_multiplier"
    with h5py.File(output, "r") as handle:
        assert handle["agn_coeff"].shape == (len(tau), len(lgmet), len(age), 1)
        assert handle["agn_coeff"].dtype == np.dtype("float16")


def test_compressed_gas_builder_round_trips_low_rank_grid(tmp_path) -> None:
    dense_path = tmp_path / "dense_gas.h5"
    compressed_path = tmp_path / "compressed_gas.h5"
    wave = np.asarray([1000.0, 2000.0, 3000.0, 4000.0, 5000.0], dtype=np.float32)
    age = np.asarray([-3.0, -2.0, -1.0], dtype=np.float32)
    lgmet = np.asarray([-2.0, -1.0], dtype=np.float32)
    gas_lgmet = np.asarray([-1.0, 0.0], dtype=np.float32)
    gas_lgu = np.asarray([-3.0, -2.0], dtype=np.float32)
    basis0 = 1.0 + wave / wave.max()
    basis1 = 0.4 + 0.2 * wave / wave.max()
    dense = np.zeros(
        (len(gas_lgmet), len(gas_lgu), len(lgmet), len(age), len(wave)),
        dtype=np.float32,
    )
    for i, _gas_z in enumerate(gas_lgmet):
        for j, _gas_u in enumerate(gas_lgu):
            for m, _met_value in enumerate(lgmet):
                for a, age_value in enumerate(age):
                    dense[i, j, m, a] = (
                        (1.0 + i + 0.2 * j) * (1.0 + m) * basis0
                        + 0.01 * (1.0 + a) * (1.0 + abs(age_value)) * basis1
                    )
    with h5py.File(dense_path, "w") as handle:
        handle["ssp_wave"] = wave
        handle["ssp_lg_age_gyr"] = age
        handle["ssp_lgmet"] = lgmet
        handle["gas_lgmet_grid"] = gas_lgmet
        handle["gas_lgu_grid"] = gas_lgu
        handle["ssp_flux"] = dense
        handle.attrs["asset_kind"] = "popcosmos_chabrier_gas_ssp_grid"
        handle.attrs["imf_type"] = 1
        handle.attrs["imf_name"] = "chabrier"
        handle.attrs["z_sun"] = 0.0142
        handle.attrs["units_ssp_flux"] = "Lsun/Hz/Msun formed"
        handle.attrs["units_ssp_wave"] = "Angstrom"
        handle.attrs["units_ssp_lg_age_gyr"] = "log10(age/Gyr)"
        handle.attrs["units_ssp_lgmet"] = (
            "log10(absolute stellar metallicity mass fraction)"
        )
        handle.attrs["units_gas_lgmet_grid"] = "log10(Zgas/Zsun)"
        handle.attrs["units_gas_lgu_grid"] = "log10 ionization parameter U"

    args = SimpleNamespace(
        input=str(dense_path),
        output=str(compressed_path),
        k=2,
        oversample=2,
        seed=0,
        normalization="none",
        compression="none",
        gzip_level=4,
        overwrite=True,
        no_progress=True,
    )
    output = build_compressed_gas_grid.build_compressed_grid(args)
    summary = build_compressed_gas_grid.validate_compressed_grid(output)

    assert summary["k_basis"] == 2
    with h5py.File(output, "r") as handle:
        coeff = np.asarray(handle["gas_coeff"])
        basis = np.asarray(handle["gas_basis"])
        scale = np.asarray(handle["gas_scale"])
    reconstructed = (coeff @ basis) * scale[..., None]
    np.testing.assert_allclose(reconstructed, dense, rtol=2.0e-5, atol=1.0e-7)


def test_compressed_ssp_builder_round_trips_low_rank_grid(tmp_path) -> None:
    dense_path = tmp_path / "dense_ssp.h5"
    compressed_path = tmp_path / "compressed_ssp.h5"
    wave = np.asarray([1000.0, 2000.0, 3000.0, 99_514_021.2543], dtype=np.float64)
    age = np.asarray([-3.0, -2.0], dtype=np.float32)
    lgmet = np.asarray([-2.0, -1.0], dtype=np.float32)
    basis0 = 1.0 + wave / wave.max()
    basis1 = 0.4 + 0.2 * wave / wave.max()
    dense = np.zeros((len(lgmet), len(age), len(wave)), dtype=np.float32)
    for m, _met_value in enumerate(lgmet):
        for a, age_value in enumerate(age):
            dense[m, a] = (1.0 + m) * basis0 + 0.01 * (1.0 + abs(age_value)) * basis1
    with h5py.File(dense_path, "w") as handle:
        handle["ssp_wave"] = wave
        handle["ssp_lg_age_gyr"] = age
        handle["ssp_lgmet"] = lgmet
        handle["ssp_flux"] = dense
        handle.attrs["asset_kind"] = "popcosmos_chabrier_ssp_grid"
        handle.attrs["imf_type"] = 1
        handle.attrs["imf_name"] = "chabrier"
        handle.attrs["z_sun"] = 0.0142

    args = SimpleNamespace(
        input=str(dense_path),
        output=str(compressed_path),
        k=2,
        oversample=2,
        seed=0,
        normalization="none",
        basis_dtype="float32",
        coeff_dtype="float32",
        compression="none",
        gzip_level=4,
        overwrite=True,
    )
    output = build_compressed_ssp_grid.build_compressed_grid(args)
    summary = build_compressed_ssp_grid.validate_compressed_grid(output)

    assert summary["k_basis"] == 2
    with h5py.File(output, "r") as handle:
        assert handle["ssp_wave"].dtype == wave.dtype
        coeff = np.asarray(handle["ssp_coeff"])
        basis = np.asarray(handle["ssp_basis"])
        scale = np.asarray(handle["ssp_scale"])
        np.testing.assert_array_equal(np.asarray(handle["ssp_wave"]), wave)
    reconstructed = (coeff @ basis) * scale[..., None]
    np.testing.assert_allclose(reconstructed, dense, rtol=2.0e-5, atol=1.0e-7)


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


def _write_line_name_reference(tmp_path: Path) -> Path:
    reference = tmp_path / "line_names.h5"
    with h5py.File(reference, "w") as handle:
        handle["ssp_emline_name"] = np.asarray([b"OIII_5007", b"Halpha"], dtype="S16")
    return reference
