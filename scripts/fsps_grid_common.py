#!/usr/bin/env python3
"""Shared helpers for FSPS-generated DSPS asset grids."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_REFERENCE_SSP = (
    "Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5"
)
DEFAULT_STELLAR_ONLY_SSP = "Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5"
C_ANGSTROM_PER_S = 2.99792458e18
POPCOSMOS_Z_SUN = 0.0142


class FspsGridError(RuntimeError):
    """User-facing failure while creating or validating FSPS grids."""


def progress_bar(
    total: int, enabled: bool, desc: str, unit: str = "step"
) -> Any | None:
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print(
            "tqdm is not installed; continuing without progress bar.", file=sys.stderr
        )
        return None
    return tqdm(total=total, desc=desc, unit=unit)


@dataclass(frozen=True)
class MetallicityPlan:
    mode: str
    ssp_lgmet: np.ndarray
    zmet_indices: tuple[int, ...] = ()
    logzsol_values: tuple[float, ...] = ()
    fsps_z_sun: float | None = None


def require_fsps() -> Any:
    """Import python-fsps only after CLI parsing has completed."""
    sps_home = os.environ.get("SPS_HOME")
    if not sps_home:
        raise FspsGridError(
            "SPS_HOME is not set. Install FSPS, install python-fsps in this "
            "environment, and export SPS_HOME=/path/to/fsps before generating "
            "FSPS grids. Use --validate-only to inspect an existing HDF5 asset "
            "without python-fsps."
        )
    try:
        import fsps  # type: ignore[import-not-found]
    except ImportError as exc:
        raise FspsGridError(
            "python-fsps is not installed in this environment. Install it after "
            "FSPS is built and SPS_HOME is set, then rerun this command."
        ) from exc
    except Exception as exc:  # pragma: no cover - depends on local FSPS install
        raise FspsGridError(
            "python-fsps was found but could not initialize. Check that SPS_HOME "
            f"points to a valid FSPS checkout with built data files: {sps_home}"
        ) from exc
    return fsps


def parse_grid(values: list[float], name: str, min_size: int = 1) -> np.ndarray:
    grid = np.asarray(values, dtype=np.float32)
    if grid.ndim != 1 or grid.size < min_size:
        raise FspsGridError(f"{name} must contain at least {min_size} values")
    if not np.all(np.isfinite(grid)):
        raise FspsGridError(f"{name} contains non-finite values")
    if grid.size > 1 and not np.all(np.diff(grid) > 0.0):
        raise FspsGridError(f"{name} must be strictly increasing")
    return grid


def ensure_output_path(path: str | Path, overwrite: bool) -> Path:
    output = Path(path).expanduser()
    if output.exists() and not overwrite:
        raise FspsGridError(f"{output} exists; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def read_ssp_axes(path: str | Path) -> dict[str, np.ndarray]:
    reference = Path(path).expanduser()
    if not reference.exists():
        raise FspsGridError(f"Reference SSP file not found: {reference}")
    required = ("ssp_wave", "ssp_lg_age_gyr", "ssp_lgmet")
    with h5py.File(reference, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise FspsGridError(
                f"Reference SSP {reference} is missing datasets: {', '.join(missing)}"
            )
        return {key: np.asarray(handle[key], dtype=np.float32) for key in required}


def discover_fsps_axes(sp: Any) -> dict[str, np.ndarray]:
    wave, spectra = sp.get_spectrum(tage=0.0, peraa=False)
    spectra = np.asarray(spectra)
    if spectra.ndim != 2:
        raise FspsGridError(
            "FSPS get_spectrum(tage=0.0) did not return an age-by-wavelength grid"
        )
    ssp_ages = getattr(sp, "ssp_ages", None)
    if ssp_ages is None:
        raise FspsGridError(
            "Could not discover the FSPS SSP age grid. Pass --reference-ssp to "
            "reuse axes from an existing DSPS SSP asset."
        )
    zlegend = getattr(sp, "zlegend", None)
    if zlegend is None:
        raise FspsGridError(
            "Could not discover the FSPS stellar metallicity grid. Pass "
            "--reference-ssp or --stellar-lgmet-grid."
        )
    return {
        "ssp_wave": np.asarray(wave, dtype=np.float32),
        "ssp_lg_age_gyr": np.asarray(ssp_ages, dtype=np.float32) - 9.0,
        "ssp_lgmet": np.log10(np.asarray(zlegend, dtype=np.float32)),
    }


def axes_from_reference_or_fsps(
    reference_ssp: str | Path | None, sp: Any
) -> dict[str, np.ndarray]:
    if reference_ssp:
        reference = Path(reference_ssp).expanduser()
        if reference.exists():
            return read_ssp_axes(reference)
    return discover_fsps_axes(sp)


def assert_wave_matches(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, rtol=0, atol=1.0e-4
    ):
        raise FspsGridError(
            f"{label} wavelength grid does not match the target DSPS SSP axes. "
            "Use an FSPS build with matching libraries or pass a compatible "
            "--reference-ssp and matching base SSP config."
        )


def build_metallicity_plan(
    sp: Any,
    ssp_lgmet: np.ndarray,
    mode: str,
    fsps_z_sun: float | None,
) -> MetallicityPlan:
    mode = str(mode)
    if mode not in {"auto", "discrete", "continuous"}:
        raise FspsGridError("metallicity mode must be auto, discrete, or continuous")
    ssp_lgmet = np.asarray(ssp_lgmet, dtype=np.float32)
    zlegend = getattr(sp, "zlegend", None)
    zlegend_values = None if zlegend is None else np.asarray(zlegend, dtype=np.float64)

    if mode in {"auto", "discrete"} and zlegend_values is not None:
        indices: list[int] = []
        for lgmet in ssp_lgmet:
            target = float(10.0 ** float(lgmet))
            match_index = int(np.argmin(np.abs(zlegend_values - target)))
            if np.isclose(zlegend_values[match_index], target, rtol=2.0e-3, atol=0.0):
                indices.append(match_index + 1)
            elif mode == "discrete":
                raise FspsGridError(
                    "Reference stellar metallicity grid does not match FSPS zlegend; "
                    "use --metallicity-mode continuous with --fsps-z-sun."
                )
            else:
                indices = []
                break
        if indices:
            return MetallicityPlan(
                mode="discrete",
                ssp_lgmet=ssp_lgmet,
                zmet_indices=tuple(indices),
            )

    if mode == "discrete":
        raise FspsGridError(
            "FSPS zlegend is unavailable or incompatible with the requested "
            "stellar metallicity grid."
        )
    if fsps_z_sun is None or not np.isfinite(fsps_z_sun) or fsps_z_sun <= 0.0:
        raise FspsGridError(
            "Continuous stellar metallicity generation requires --fsps-z-sun "
            "because python-fsps logzsol is relative to the compiled FSPS solar Z."
        )
    logzsol = tuple(float(lgmet - np.log10(fsps_z_sun)) for lgmet in ssp_lgmet)
    return MetallicityPlan(
        mode="continuous",
        ssp_lgmet=ssp_lgmet,
        logzsol_values=logzsol,
        fsps_z_sun=float(fsps_z_sun),
    )


def fsps_metadata(fsps_module: Any, sp: Any, command: list[str]) -> dict[str, Any]:
    isochrones = _decode_metadata_value(getattr(sp, "isoc_library", "unknown"))
    spectral_library = _decode_metadata_value(getattr(sp, "spec_library", "unknown"))
    return {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "generated_by": " ".join(command),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "python_fsps_version": str(getattr(fsps_module, "__version__", "unknown")),
        "fsps_version": str(getattr(sp, "fsps_version", "unknown")),
        "sps_home": os.environ.get("SPS_HOME", ""),
        "isoc_library": isochrones,
        "spec_library": spectral_library,
        "isochrones": isochrones,
        "spectral_library": spectral_library,
    }


def _decode_metadata_value(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value)
    aliases = {
        "mist": "MIST",
        "c3k_a": "C3K_A",
    }
    return aliases.get(text.lower(), text)


def write_attrs(handle: h5py.File, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        if isinstance(value, (list, tuple, dict)):
            handle.attrs[key] = json.dumps(value)
        elif isinstance(value, np.ndarray):
            handle.attrs[key] = json.dumps(np.asarray(value).tolist())
        elif value is None:
            handle.attrs[key] = ""
        else:
            handle.attrs[key] = value


def validate_gas_grid_hdf5(
    path: str | Path, reference_ssp: str | Path | None = None
) -> tuple[int, int, int, int, int]:
    grid_path = Path(path).expanduser()
    required = (
        "ssp_wave",
        "ssp_lg_age_gyr",
        "ssp_lgmet",
        "gas_lgmet_grid",
        "gas_lgu_grid",
        "ssp_flux",
    )
    with h5py.File(grid_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise FspsGridError(
                f"Gas SSP grid {grid_path} is missing datasets: {', '.join(missing)}"
            )
        axes = {key: np.asarray(handle[key]) for key in required[:-1]}
        flux_shape = tuple(handle["ssp_flux"].shape)
        if len(flux_shape) != 5:
            raise FspsGridError(
                "Gas SSP grid ssp_flux must have shape "
                "(n_gas_lgmet, n_gas_lgu, n_stellar_lgmet, n_age, n_wave)"
            )
        expected = (
            len(axes["gas_lgmet_grid"]),
            len(axes["gas_lgu_grid"]),
            len(axes["ssp_lgmet"]),
            len(axes["ssp_lg_age_gyr"]),
            len(axes["ssp_wave"]),
        )
        if flux_shape != expected:
            raise FspsGridError(
                f"Gas SSP grid ssp_flux shape {flux_shape} does not match axes {expected}"
            )
        _validate_monotonic_axes(axes)
    if reference_ssp:
        reference_path = Path(reference_ssp).expanduser()
        if reference_path.exists():
            reference_axes = read_ssp_axes(reference_path)
            for key in ("ssp_wave", "ssp_lg_age_gyr", "ssp_lgmet"):
                if axes[key].shape != reference_axes[key].shape or not np.allclose(
                    axes[key], reference_axes[key], rtol=0, atol=1.0e-4
                ):
                    raise FspsGridError(
                        f"Gas grid {key} does not match reference SSP {reference_path}"
                    )
    return flux_shape  # type: ignore[return-value]


def validate_ssp_grid_hdf5(
    path: str | Path, require_popcosmos_chabrier: bool = True
) -> tuple[int, int, int]:
    grid_path = Path(path).expanduser()
    required = ("ssp_wave", "ssp_lg_age_gyr", "ssp_lgmet", "ssp_flux")
    with h5py.File(grid_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise FspsGridError(
                f"SSP grid {grid_path} is missing datasets: {', '.join(missing)}"
            )
        axes = {key: np.asarray(handle[key]) for key in required[:-1]}
        flux_shape = tuple(handle["ssp_flux"].shape)
        expected = (
            len(axes["ssp_lgmet"]),
            len(axes["ssp_lg_age_gyr"]),
            len(axes["ssp_wave"]),
        )
        if len(flux_shape) != 3 or flux_shape != expected:
            raise FspsGridError(
                f"SSP grid ssp_flux shape {flux_shape} does not match axes {expected}"
            )
        _validate_monotonic_axes(axes)

        if "ssp_emline_luminosity" in handle:
            if "ssp_emline_wave" not in handle:
                raise FspsGridError(
                    "SSP grid has ssp_emline_luminosity but no ssp_emline_wave"
                )
            line_shape = tuple(handle["ssp_emline_luminosity"].shape)
            n_line = len(handle["ssp_emline_wave"])
            expected_line_shape = (
                len(axes["ssp_lgmet"]),
                len(axes["ssp_lg_age_gyr"]),
                n_line,
            )
            if line_shape != expected_line_shape:
                raise FspsGridError(
                    "SSP grid ssp_emline_luminosity shape "
                    f"{line_shape} does not match axes {expected_line_shape}"
                )
            if "ssp_emline_name" in handle and len(handle["ssp_emline_name"]) != n_line:
                raise FspsGridError(
                    "SSP grid ssp_emline_name length does not match ssp_emline_wave"
                )

        if "ssp_surviving_mstar" in handle:
            surviving_shape = tuple(handle["ssp_surviving_mstar"].shape)
            expected_surviving_shape = (
                len(axes["ssp_lgmet"]),
                len(axes["ssp_lg_age_gyr"]),
            )
            if surviving_shape != expected_surviving_shape:
                raise FspsGridError(
                    "SSP grid ssp_surviving_mstar shape "
                    f"{surviving_shape} does not match axes {expected_surviving_shape}"
                )

        if require_popcosmos_chabrier:
            _validate_popcosmos_chabrier_attrs(handle, grid_path)
    return flux_shape  # type: ignore[return-value]


def validate_agn_grid_hdf5(path: str | Path) -> tuple[int, ...]:
    grid_path = Path(path).expanduser()
    required = ("wave", "agn_tau_grid", "template_lnu_per_lbol")
    with h5py.File(grid_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise FspsGridError(
                f"AGN template grid {grid_path} is missing datasets: {', '.join(missing)}"
            )
        wave = np.asarray(handle["wave"])
        tau = np.asarray(handle["agn_tau_grid"])
        template_shape = tuple(handle["template_lnu_per_lbol"].shape)
        if len(template_shape) == 2 and template_shape == (len(tau), len(wave)):
            axes = {"wave": wave, "agn_tau_grid": tau}
        elif len(template_shape) == 5:
            fagn_key = (
                "fagn_grid" if "fagn_grid" in handle else "fagn_normalization_grid"
            )
            required_axes = (fagn_key, "tage_gyr_grid", "stellar_logzsol_grid")
            missing_axes = [key for key in required_axes if key not in handle]
            if missing_axes:
                raise FspsGridError(
                    "5D AGN template grid is missing axis datasets: "
                    f"{', '.join(missing_axes)}"
                )
            fagn = np.asarray(handle[fagn_key])
            tage = np.asarray(handle["tage_gyr_grid"])
            logzsol = np.asarray(handle["stellar_logzsol_grid"])
            expected = (len(fagn), len(tau), len(tage), len(logzsol), len(wave))
            if template_shape != expected:
                raise FspsGridError(
                    "5D AGN template_lnu_per_lbol must have shape "
                    "(n_fagn, n_agn_tau, n_tage_gyr, n_stellar_logzsol, n_wave)"
                )
            axes = {
                "wave": wave,
                fagn_key: fagn,
                "agn_tau_grid": tau,
                "tage_gyr_grid": tage,
                "stellar_logzsol_grid": logzsol,
            }
        else:
            raise FspsGridError(
                "AGN template_lnu_per_lbol must have shape (n_agn_tau, n_wave) "
                "or (n_fagn, n_agn_tau, n_tage_gyr, n_stellar_logzsol, n_wave)"
            )
        _validate_monotonic_axes(axes)
    return template_shape  # type: ignore[return-value]


def validate_gas_grid_with_model(
    grid_path: str | Path,
    ssp_path: str | Path,
    skip_run: bool = False,
) -> None:
    from euclid_dsps.filters import FilterCurve
    from euclid_dsps.model import load_context, run_dsps_model_jax

    wave = np.linspace(4500.0, 8500.0, 64)
    filters = {
        "synthetic_wide": FilterCurve(
            name="synthetic_wide",
            wave=wave,
            transmission=np.ones_like(wave),
            source="synthetic_validation",
        )
    }
    context = load_context(
        str(ssp_path),
        filters,
        model_config={
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "dust_model": "charlot_fall",
            "emission_line_corrections": "none",
            "igm_model": "none",
            "nebular_model": "gas_grid",
            "gas_grid_path": str(grid_path),
            "agn_model": "none",
            "z_sun": POPCOSMOS_Z_SUN,
        },
    )
    if not skip_run:
        result = run_dsps_model_jax(
            context, synthetic_popcosmos_params(include_agn=False)
        )
        mags = np.asarray(result.model_mags)
        if not np.all(np.isfinite(mags)):
            raise FspsGridError(
                "Synthetic gas-grid model validation produced non-finite magnitudes"
            )


def validate_agn_grid_with_model(
    grid_path: str | Path,
    ssp_path: str | Path,
    skip_run: bool = False,
    progress: Any | None = None,
) -> None:
    from euclid_dsps.filters import FilterCurve
    from euclid_dsps.model import load_context, run_dsps_model_jax

    wave = np.linspace(4500.0, 8500.0, 64)
    filters = {
        "synthetic_wide": FilterCurve(
            name="synthetic_wide",
            wave=wave,
            transmission=np.ones_like(wave),
            source="synthetic_validation",
        )
    }
    if progress is not None:
        progress.set_postfix(stage="load_context")
    context = load_context(
        str(ssp_path),
        filters,
        model_config={
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "dust_model": "charlot_fall",
            "igm_model": "none",
            "nebular_model": "fixed_ssp",
            "agn_model": "template_grid",
            "agn_template_path": str(grid_path),
            "z_sun": POPCOSMOS_Z_SUN,
        },
    )
    if progress is not None:
        progress.update(1)
    if not skip_run:
        if progress is not None:
            progress.set_postfix(stage="forward")
        result = run_dsps_model_jax(
            context, synthetic_popcosmos_params(include_agn=True)
        )
        mags = np.asarray(result.model_mags)
        if not np.all(np.isfinite(mags)):
            raise FspsGridError(
                "Synthetic AGN-grid model validation produced non-finite magnitudes"
            )
        if progress is not None:
            progress.update(1)


def synthetic_popcosmos_params(include_agn: bool) -> dict[str, float]:
    params = {
        "z_obs": 0.5,
        "log10_stellar_mass": 10.0,
        "log10_stellar_metallicity": 0.0,
        "tau2": 0.2,
        "dust_index_n": -0.7,
        "tau1_over_tau2": 1.0,
        "log10_gas_metallicity": 0.0,
        "log10_gas_ionization": -2.0,
        "ln_fagn": -8.0,
        "ln_tauagn": np.log(10.0),
    }
    params.update({f"dlog10_sfr_{index}": 0.0 for index in range(1, 7)})
    if not include_agn:
        params.pop("ln_fagn")
        params.pop("ln_tauagn")
    return params


def lbol_from_lnu(wave: np.ndarray, lnu: np.ndarray) -> float:
    wave_safe = np.maximum(np.asarray(wave, dtype=np.float64), 1.0)
    lnu_safe = np.maximum(np.asarray(lnu, dtype=np.float64), 0.0)
    return float(np.trapezoid(lnu_safe * C_ANGSTROM_PER_S / wave_safe**2, wave_safe))


def fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 2


def _validate_monotonic_axes(axes: dict[str, np.ndarray]) -> None:
    for key, values in axes.items():
        values = np.asarray(values)
        if values.ndim != 1 or len(values) == 0:
            raise FspsGridError(f"{key} must be a non-empty 1D dataset")
        if not np.all(np.isfinite(values)):
            raise FspsGridError(f"{key} contains non-finite values")
        if len(values) > 1 and not np.all(np.diff(values) > 0.0):
            raise FspsGridError(f"{key} must be strictly increasing")


def _validate_popcosmos_chabrier_attrs(handle: h5py.File, path: Path) -> None:
    if "kroupa" in path.name.lower():
        raise FspsGridError(
            f"PopCosmos-like SSP path must not contain 'kroupa': {path}"
        )
    imf_type = handle.attrs.get("imf_type")
    imf_name = _decode_metadata_value(handle.attrs.get("imf_name", "")).lower()
    imf_errors = []
    if imf_type is None:
        imf_errors.append("missing imf_type")
    elif not np.isclose(float(imf_type), 1.0, rtol=0.0, atol=0.0):
        imf_errors.append(f"imf_type={imf_type!r}")
    if not imf_name:
        imf_errors.append("missing imf_name")
    elif imf_name != "chabrier":
        imf_errors.append(f"imf_name={imf_name!r}")
    if imf_errors:
        raise FspsGridError(
            "PopCosmos-like SSP metadata must consistently declare "
            "imf_type=1 and imf_name='chabrier'; found " + ", ".join(imf_errors)
        )
    z_sun = handle.attrs.get("z_sun")
    if z_sun is None or not np.isclose(
        float(z_sun), POPCOSMOS_Z_SUN, rtol=0, atol=5.0e-6
    ):
        raise FspsGridError(
            f"PopCosmos-like SSP metadata z_sun must be {POPCOSMOS_Z_SUN}"
        )
    required_units = (
        "units_ssp_wave",
        "units_ssp_lg_age_gyr",
        "units_ssp_lgmet",
        "units_ssp_flux",
    )
    missing_units = [key for key in required_units if key not in handle.attrs]
    if missing_units:
        raise FspsGridError(
            "PopCosmos-like SSP metadata is missing unit attrs: "
            + ", ".join(missing_units)
        )
