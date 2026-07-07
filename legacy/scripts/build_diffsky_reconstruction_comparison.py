#!/usr/bin/env python
"""Build a local comparison dashboard for the Diffsky reconstruction runs.

The dashboard is intentionally file-based: it creates symlinks to the original
run directories, normalized CSV/Parquet tables, static plots, a README, and a
small HTML gallery under outputs/comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from euclid_dsps.photometric_uncertainty import (
    DEFAULT_LSST_COADD_M5,
    DEFAULT_LSST_GAMMA,
    DEFAULT_ROMAN_ETA,
    DEFAULT_ROMAN_WFI_ONE_HOUR_POINT_M5,
)
from euclid_dsps.photometry import abmag_to_fnu_cgs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/comparison/diffsky_reconstruction_debug"

REFERENCE_TRAIN = ROOT / "outputs/runs/diffsky_autoencoder_nokl_m5sys_z035_rand20k_e30_b128"
REFERENCE_INFER = (
    ROOT / "outputs/runs/diffsky_autoencoder_nokl_m5sys_z035_rand20k_e30_b128_infer"
)

SYNC_RUNS_LATEST = (
    ROOT / "outputs/logs/jean-zay/recon_infer_compare_20260624_1141/outputs/runs"
)
SYNC_RUNS_FULL = ROOT / "outputs/logs/jean-zay/recon_20260624_1121/outputs/runs"

ROWSET_DIR_CANDIDATES = [
    ROOT / "outputs/rowsets/diffsky_autoencoder_nokl_m5sys_z035_rand20k",
    ROOT
    / "outputs/logs/jean-zay/recon_infer_compare_20260624_1141/outputs/rowsets/diffsky_autoencoder_nokl_m5sys_z035_rand20k",
    ROOT
    / "outputs/logs/jean-zay/recon_20260624_1121/outputs/rowsets/diffsky_autoencoder_nokl_m5sys_z035_rand20k",
]

PARAMETER_ORDER = [
    "z_obs",
    "log10_stellar_mass",
    "log10_sfr_at_obs",
    "dlog10_sfr_1",
    "dlog10_sfr_6",
    "log10_stellar_metallicity",
    "tau2",
    "dust_index_n",
    "tau1_over_tau2",
]

MCLMC_CORE_CORNER_PARAMETERS = [
    "z_obs",
    "log10_stellar_mass",
    "dlog10_sfr_1",
    "dlog10_sfr_6",
    "log10_stellar_metallicity",
    "tau2",
    "dust_index_n",
    "tau1_over_tau2",
]

MCLMC_TRUTH_COMPARABLE_PARAMETERS = [
    "z_obs",
    "log10_stellar_mass",
]

MCLMC_PROJECTED_TRUTH_PARAMETERS = [
    "z_obs",
    "log10_stellar_mass",
    "dlog10_sfr_1",
    "dlog10_sfr_2",
    "dlog10_sfr_3",
    "dlog10_sfr_4",
    "dlog10_sfr_5",
    "dlog10_sfr_6",
    "tau2",
    "dust_index_n",
]

MCLMC_PROJECTED_TRUTH_DISTRIBUTION_PARAMETERS = [
    "z_obs",
    "log10_stellar_mass",
    "log10_sfr_at_obs",
    "log10_ssfr_at_obs",
    "dlog10_sfr_1",
    "dlog10_sfr_2",
    "dlog10_sfr_3",
    "dlog10_sfr_4",
    "dlog10_sfr_5",
    "dlog10_sfr_6",
    "tau2",
    "dust_index_n",
]

DIFFSKY_SFH_TRUTH_COLUMNS = [
    "redshift_true",
    "diffstar_lgmcrit",
    "diffstar_lgy_at_mcrit",
    "diffstar_indx_lo",
    "diffstar_indx_hi",
    "diffstar_lg_qt",
    "diffstar_qlglgdt",
    "diffstar_lg_drop",
    "diffstar_lg_rejuv",
    "diffmah_logm0",
    "diffmah_logtc",
    "diffmah_early_index",
    "diffmah_late_index",
    "diffmah_t_peak",
]

PARAMETER_LABELS = {
    "z_obs": "z_obs",
    "log10_stellar_mass": "log10 Mstar",
    "log10_sfr_at_obs": "log10 SFR",
    "log10_ssfr_at_obs": "log10 sSFR",
    "dlog10_sfr_1": "dlog10 SFR 1",
    "dlog10_sfr_2": "dlog10 SFR 2",
    "dlog10_sfr_3": "dlog10 SFR 3",
    "dlog10_sfr_4": "dlog10 SFR 4",
    "dlog10_sfr_5": "dlog10 SFR 5",
    "dlog10_sfr_6": "dlog10 SFR 6",
    "log10_stellar_metallicity": "log10 Zstar",
    "tau2": "tau2",
    "dust_index_n": "dust index n",
    "tau1_over_tau2": "tau1/tau2",
}

RESIDUAL_LABEL = r"$(F_{\rm in} - F_{\rm out}) / \sigma_{\rm eff}$"

TRUTH_COLUMN_BY_PARAMETER = {
    "z_obs": "redshift_truth",
    "log10_stellar_mass": "truth_log10_stellar_mass",
    "log10_sfr_at_obs": "truth_log10_sfr_at_obs",
}

METHOD_DISPLAY_LABELS = {
    "reference_nn": "NN baseline",
    "map_500_iter200": "DSPS MAP 500/200",
    "map_1000_iter400": "DSPS MAP",
    "mclmc": "DSPS MCLMC",
}


def _m5_depth_defaults_for_band(band: str) -> tuple[float, float]:
    """Return the default synthetic depth parameters used for a dashboard band."""
    name = str(band)
    if name in DEFAULT_LSST_COADD_M5:
        return float(DEFAULT_LSST_COADD_M5[name]), float(DEFAULT_LSST_GAMMA[name])
    if name in DEFAULT_ROMAN_WFI_ONE_HOUR_POINT_M5:
        eta = float(DEFAULT_ROMAN_ETA.get(name, 1.0))
        return float(DEFAULT_ROMAN_WFI_ONE_HOUR_POINT_M5[name]), 0.04 * eta
    raise KeyError(f"No default m5_depth parameters for band {band!r}")


def _m5_depth_example_values(band: str) -> dict[str, float]:
    m5, gamma = _m5_depth_defaults_for_band(band)
    f5 = float(abmag_to_fnu_cgs(m5))
    return {
        "m5": m5,
        "gamma": gamma,
        "f5": f5,
        "zero_flux_sigma": float(math.sqrt(gamma) * f5),
    }

NN_INFERENCE_PLOTS = [
    "posterior_predictive_normalized_residual_hist.png",
    "posterior_predictive_normalized_residual_hist_by_band.png",
    "posterior_predictive_residuals_by_band.png",
    "posterior_predictive_chi2_hist.png",
    "top_posterior_predictive_chi2.png",
    "corner_truth_prior_posterior_map.png",
    "posterior_corner.png",
    "learned_prior_corner.png",
    "posterior_vs_learned_prior_corner.png",
    "photoz_truth_vs_pred.png",
    "photoz_delta_vs_ztrue.png",
]


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    family: str
    path: Path
    source_kind: str
    description: str


@dataclass(frozen=True)
class TrainingSpec:
    key: str
    label: str
    path: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--copy-runs",
        action="store_true",
        help="Copy run directories instead of creating symlinks. Defaults to symlinks.",
    )
    args = parser.parse_args()

    out = args.out.resolve()
    rowset_dir = _first_complete_rowset_dir(ROWSET_DIR_CANDIDATES)
    if rowset_dir is None:
        raise SystemExit("No rowset directory found. Expected outputs/rowsets/... or synced copy.")

    run_specs = _run_specs()
    training_specs = _training_specs()
    _validate_inputs(run_specs, training_specs, rowset_dir)

    _prepare_layout(out)
    _link_reference_and_runs(out, run_specs, training_specs, rowset_dir, copy_runs=args.copy_runs)

    manifest = {
        "canonical_reference_run": str(REFERENCE_INFER),
        "canonical_reference_train_run": str(REFERENCE_TRAIN),
        "rowsets_derived_from": str(REFERENCE_INFER),
        "rowset_dir": str(rowset_dir),
        "diagnostic_rowsets": ["worst_500", "worst_100"],
        "purpose": (
            "Use worst-case slices from the canonical no-KL NN inference to test "
            "whether deterministic MAP and MCLMC can recover photometry where the "
            "amortized neural network fails."
        ),
        "runs": [
            {
                "key": spec.key,
                "label": spec.label,
                "family": spec.family,
                "path": str(spec.path),
                "source_kind": spec.source_kind,
                "description": spec.description,
            }
            for spec in run_specs
        ],
        "trainings": [
            {"key": spec.key, "label": spec.label, "path": str(spec.path)}
            for spec in training_specs
            if spec.path.exists()
        ],
    }
    _write_json(out / "manifest.json", manifest)

    tables_dir = _ensure(out / "tables")
    plots_dir = _ensure(out / "plots")

    reference_full = _load_run_residuals(
        _spec_by_key(run_specs, "reference_full_20k"),
        method="reference_nn_full20k",
    )
    _write_comparison_tables(
        reference_full,
        tables_dir / "reference_full",
        "reference_full",
    )
    _plot_reference_full(reference_full, plots_dir / "reference_full")

    worst500 = _comparison_frame(
        run_specs,
        rowset_dir / "worst_500.txt",
        [
            "reference_full_20k",
            "nn_input_noise_sigma1_worst500",
            "nn_kl001_worst500",
            "nn_kl005_worst500",
            "nn_supervised_prior_worst500",
            "map_worst500_iter200",
            "map_worst1000_iter400",
        ],
    )
    worst500_tables = _write_comparison_tables(worst500, tables_dir / "worst500", "worst500")
    _plot_comparison_suite(
        worst500,
        plots_dir / "worst500_nn_variants",
        "Worst 500 from canonical NN reference",
    )

    worst100 = _comparison_frame(
        run_specs,
        rowset_dir / "worst_100.txt",
        [
            "reference_full_20k",
            "nn_input_noise_sigma1_worst500",
            "nn_kl001_worst500",
            "nn_kl005_worst500",
            "nn_supervised_prior_worst500",
            "map_worst500_iter200",
            "map_worst1000_iter400",
            "mclmc_worst100_b32_w64_s256",
        ],
    )
    worst100_tables = _write_comparison_tables(worst100, tables_dir / "worst100", "worst100")
    _write_huge_error_bar_diagnostics(
        worst100,
        tables_dir / "worst100",
        plots_dir / "worst100_dsps_recovery",
        baseline_method="reference_nn",
        recovery_method="map_1000_iter400",
        mcmc_method="mclmc",
    )
    _plot_comparison_suite(
        worst100,
        plots_dir / "worst100_dsps_recovery",
        "Worst 100: DSPS recovery diagnostic",
    )
    _plot_selected_signed_residual_histograms(
        worst100,
        plots_dir / "worst100_dsps_recovery/baseline_map_mclmc_signed_residual_histograms.png",
        "Worst 100 signed photometric residuals",
        ["reference_nn", "map_1000_iter400", "mclmc"],
    )
    _plot_worst_slice_location_in_full_reference(
        reference_full,
        worst100,
        plots_dir / "worst100_dsps_recovery/worst100_location_in_full_nn_and_map.png",
        "Worst 100 location in the full NN baseline distribution",
        baseline_method="reference_nn",
        map_method="map_1000_iter400",
    )
    _plot_sed_comparison_grid(
        worst100,
        plots_dir / "worst100_dsps_recovery/sed_examples_baseline_map_mclmc_grid.png",
        "Worst 100 photometric SED examples",
        methods=["reference_nn", "map_1000_iter400", "mclmc"],
        wavelength_map=_global_wavelength_map(),
        baseline_method="reference_nn",
        recovery_method="map_1000_iter400",
        n_per_group=3,
    )

    _plot_map_suite(_spec_by_key(run_specs, "map_worst500_iter200"), plots_dir / "map/worst500_iter200")
    _plot_map_suite(
        _spec_by_key(run_specs, "map_worst1000_iter400"),
        plots_dir / "map/worst1000_iter400",
    )
    _plot_mclmc_suite(
        _spec_by_key(run_specs, "mclmc_worst100_b32_w64_s256"),
        plots_dir / "mclmc/worst100_b32_w64_s256",
        worst100,
    )
    _plot_training_suite(training_specs, plots_dir / "nn_step_by_step/training")
    _link_nn_inference_plots(run_specs, plots_dir / "nn_step_by_step/inference")

    _write_readme(
        out,
        manifest,
        {
            "reference_full": tables_dir / "reference_full/reference_full_method_summary.csv",
            "worst500": worst500_tables["method"],
            "worst100": worst100_tables["method"],
        },
    )
    _write_gallery(out)
    print(f"[comparison] dashboard -> {out}")


def _run_specs() -> list[RunSpec]:
    latest = SYNC_RUNS_LATEST
    full = SYNC_RUNS_FULL
    return [
        RunSpec(
            "reference_full_20k",
            "reference_nn",
            "nn",
            REFERENCE_INFER,
            "nn_residual_summary",
            "Canonical full 20k no-KL reference inference.",
        ),
        RunSpec(
            "nn_input_noise_sigma1_worst500",
            "input_noise_sigma1",
            "nn",
            latest / "diffsky_autoencoder_input_noise_sigma1_infer_worst_500",
            "nn_residual_summary",
            "Input-noise NN inference on the reference worst_500 rowset.",
        ),
        RunSpec(
            "nn_kl001_worst500",
            "kl001",
            "nn",
            latest / "diffsky_autoencoder_kl001_infer_worst_500",
            "nn_residual_summary",
            "KL=0.01 NN inference on the reference worst_500 rowset.",
        ),
        RunSpec(
            "nn_kl005_worst500",
            "kl005",
            "nn",
            latest / "diffsky_autoencoder_kl005_infer_worst_500",
            "nn_residual_summary",
            "KL=0.05 NN inference on the reference worst_500 rowset.",
        ),
        RunSpec(
            "nn_supervised_prior_worst500",
            "supervised_prior_nn",
            "nn",
            latest / "diffsky_supervised_prior_basic_nn_infer_worst_500",
            "nn_residual_summary",
            "Amortized NN trained with the supervised basic prior, inferred on worst_500.",
        ),
        RunSpec(
            "map_worst500_iter200",
            "map_500_iter200",
            "map",
            full / "diffsky_reconstruction_map_worst_500",
            "map_photometry_comparison",
            "Standalone DSPS MAP/Adam reconstruction on worst_500.",
        ),
        RunSpec(
            "map_worst1000_iter400",
            "map_1000_iter400",
            "map",
            full / "diffsky_reconstruction_map_worst_1000_b512_iter400",
            "map_photometry_comparison",
            "Standalone DSPS MAP/Adam reconstruction on worst_1000, filtered where needed.",
        ),
        RunSpec(
            "mclmc_worst100_b32_w64_s256",
            "mclmc",
            "mclmc",
            full / "diffsky_reconstruction_mclmc_worst_100_b32_w64_s256",
            "mclmc_posterior_predictive_summary",
            "Batched BlackJAX MCLMC posterior predictive reconstruction on worst_100.",
        ),
    ]


def _training_specs() -> list[TrainingSpec]:
    full = SYNC_RUNS_FULL
    return [
        TrainingSpec("reference_nokl", "reference no-KL", REFERENCE_TRAIN),
        TrainingSpec("input_noise_sigma1", "input noise sigma=1", full / "diffsky_autoencoder_input_noise_sigma1"),
        TrainingSpec("kl001", "KL=0.01", full / "diffsky_autoencoder_kl001"),
        TrainingSpec("kl005", "KL=0.05", full / "diffsky_autoencoder_kl005"),
        TrainingSpec("supervised_prior_nn", "supervised-prior NN", full / "diffsky_supervised_prior_basic_nn"),
    ]


def _prepare_layout(out: Path) -> None:
    for rel in [
        "reference",
        "rowsets",
        "runs",
        "training",
        "tables",
        "plots/reference_full",
        "plots/worst500_nn_variants",
        "plots/worst100_dsps_recovery",
        "plots/map",
        "plots/mclmc",
        "plots/nn_step_by_step/training",
        "plots/nn_step_by_step/inference",
    ]:
        _ensure(out / rel)


def _link_reference_and_runs(
    out: Path,
    run_specs: list[RunSpec],
    training_specs: list[TrainingSpec],
    rowset_dir: Path,
    *,
    copy_runs: bool,
) -> None:
    _link_or_copy(REFERENCE_INFER, out / "reference/nn_full_20k_infer", copy_runs=copy_runs)
    _link_or_copy(REFERENCE_TRAIN, out / "reference/nn_full_20k_train", copy_runs=copy_runs)
    _link_or_copy(rowset_dir, out / "rowsets/canonical_reference_rowsets", copy_runs=copy_runs)
    for spec in run_specs:
        if spec.key == "reference_full_20k":
            continue
        if spec.path.exists():
            _link_or_copy(spec.path, out / f"runs/{spec.key}", copy_runs=copy_runs)
    for spec in training_specs:
        if spec.path.exists():
            _link_or_copy(spec.path, out / f"training/{spec.key}", copy_runs=copy_runs)


def _validate_inputs(
    run_specs: list[RunSpec],
    training_specs: list[TrainingSpec],
    rowset_dir: Path,
) -> None:
    missing = []
    required_files = [rowset_dir / "worst_500.txt", rowset_dir / "worst_100.txt"]
    for path in required_files:
        if not path.exists():
            missing.append(str(path))
    for spec in run_specs:
        if not spec.path.exists():
            missing.append(str(spec.path))
    if missing:
        raise SystemExit("Missing required comparison inputs:\n" + "\n".join(missing))
    # Training dirs are useful but not strictly required for the reconstruction dashboard.
    missing_train = [spec.path for spec in training_specs if not spec.path.exists()]
    if missing_train:
        print("[comparison][warning] missing training dirs:")
        for path in missing_train:
            print(f"  - {path}")


def _comparison_frame(run_specs: list[RunSpec], rowset_path: Path, keys: list[str]) -> pd.DataFrame:
    rowset = set(_read_rowset(rowset_path))
    frames = []
    for key in keys:
        spec = _spec_by_key(run_specs, key)
        frame = _load_run_residuals(spec, method=spec.label)
        frame = frame[frame["row_index"].isin(rowset)].copy()
        frames.append(frame)
    if not frames:
        raise ValueError(f"No frames built for {rowset_path}")
    return pd.concat(frames, ignore_index=True)


def _load_run_residuals(spec: RunSpec, *, method: str) -> pd.DataFrame:
    if spec.family == "map":
        frame = _read_table_stem(spec.path, "batch_fit_photometry_comparison")
        out = _normalize_map_comparison(frame)
    elif spec.family == "mclmc":
        frame = _read_table_stem(spec.path, "batch_posterior_predictive_flux_residual_summary")
        out = _normalize_residual_summary(frame)
    else:
        frame = _read_table_stem(spec.path, "posterior_predictive_residual_summary")
        out = _normalize_residual_summary(frame)
    out["method"] = method
    out["family"] = spec.family
    out["source_key"] = spec.key
    out["run_path"] = str(spec.path)
    return out


def _normalize_residual_summary(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    _require_columns(out, ["row_index", "band", "residual_sigma_median"])
    if "object_id" not in out:
        out["object_id"] = out["row_index"]
    if "obs_flux_fnu_cgs" not in out and "observed_flux_fnu_cgs" in out:
        out["obs_flux_fnu_cgs"] = out["observed_flux_fnu_cgs"]
    if "obs_err_fnu_cgs" not in out and "observed_flux_error_fnu_cgs" in out:
        out["obs_err_fnu_cgs"] = out["observed_flux_error_fnu_cgs"]
    if "model_flux_median" not in out and "model_flux_fnu_cgs" in out:
        out["model_flux_median"] = out["model_flux_fnu_cgs"]
    if "valid" not in out:
        out["valid"] = np.isfinite(pd.to_numeric(out["residual_sigma_median"], errors="coerce"))
    flux_residual = _numeric_column(out, "flux_residual_obs_minus_model_median")
    sigma_eff = _numeric_column(out, "sigma_eff_median")
    explicit_residual = flux_residual / sigma_eff
    stored_residual = pd.to_numeric(out["residual_sigma_median"], errors="coerce")
    out["residual_sigma_median"] = explicit_residual.where(
        np.isfinite(explicit_residual),
        stored_residual,
    )
    if "abs_residual_sigma_median" not in out:
        out["abs_residual_sigma_median"] = out["residual_sigma_median"].abs()
    else:
        stored_abs = pd.to_numeric(out["abs_residual_sigma_median"], errors="coerce")
        out["abs_residual_sigma_median"] = out["residual_sigma_median"].abs().where(
            np.isfinite(out["residual_sigma_median"]),
            stored_abs,
        )
    out["residual_definition"] = "(flux_in - flux_out) / sigma_eff"
    return out


def _normalize_map_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ["row_index", "band", "observed_flux_fnu_cgs", "model_flux_fnu_cgs"])
    obs_flux = pd.to_numeric(frame["observed_flux_fnu_cgs"], errors="coerce")
    model_flux = pd.to_numeric(frame["model_flux_fnu_cgs"], errors="coerce")
    obs_err = pd.to_numeric(frame.get("observed_flux_error_fnu_cgs"), errors="coerce")
    sigma_eff = pd.to_numeric(
        frame.get("likelihood_sigma", frame.get("observed_flux_error_fnu_cgs")),
        errors="coerce",
    )
    residual = obs_flux - model_flux
    chi_likelihood = pd.to_numeric(frame.get("chi_likelihood"), errors="coerce")
    explicit_residual = residual / sigma_eff
    raw_residual = residual / obs_err
    out = pd.DataFrame(
        {
            "object_id": frame["object_id"] if "object_id" in frame else frame["row_index"],
            "row_index": frame["row_index"],
            "band": frame["band"],
            "obs_flux_fnu_cgs": obs_flux,
            "obs_err_fnu_cgs": obs_err,
            "model_flux_median": model_flux,
            "sigma_eff_median": sigma_eff,
            "flux_residual_obs_minus_model_median": residual,
            "residual_sigma_median": explicit_residual.where(np.isfinite(explicit_residual), chi_likelihood),
            "raw_residual_sigma_median": raw_residual,
            "valid": (
                frame["band_used_in_likelihood"].astype(bool)
                if "band_used_in_likelihood" in frame
                else np.isfinite(chi_likelihood)
            ),
        }
    )
    out["abs_residual_sigma_median"] = out["residual_sigma_median"].abs()
    out["residual_definition"] = "(flux_in - flux_out) / sigma_eff"
    if "effective_wavelength_angstrom" in frame:
        out["effective_wavelength_angstrom"] = pd.to_numeric(
            frame["effective_wavelength_angstrom"], errors="coerce"
        )
    for column in [
        "z_obs",
        "redshift_truth",
        "delta_z_obs_minus_truth",
        "truth_log10_stellar_mass",
        "truth_log10_sfr_at_obs",
        "map_chi2",
    ]:
        if column in frame:
            out[column] = frame[column]
    return out


def _write_comparison_tables(frame: pd.DataFrame, out_dir: Path, prefix: str) -> dict[str, Path]:
    _ensure(out_dir)
    residual_path = out_dir / f"{prefix}_residual_summary"
    _write_table(frame, residual_path)
    method = _method_summary(frame)
    band = _band_summary(frame)
    obj = _object_summary(frame)
    paired_band, paired_object, paired_summary = _paired_against_baseline(frame)
    method_path = out_dir / f"{prefix}_method_summary"
    band_path = out_dir / f"{prefix}_band_summary"
    obj_path = out_dir / f"{prefix}_object_summary"
    _write_table(method, method_path)
    _write_table(band, band_path)
    _write_table(obj, obj_path)
    paths = {
        "residual": residual_path.with_suffix(".csv"),
        "method": method_path.with_suffix(".csv"),
        "band": band_path.with_suffix(".csv"),
        "object": obj_path.with_suffix(".csv"),
    }
    if not paired_band.empty:
        paired_band_path = out_dir / f"{prefix}_paired_band_improvement"
        _write_table(paired_band, paired_band_path)
        paths["paired_band"] = paired_band_path.with_suffix(".csv")
    if not paired_object.empty:
        paired_object_path = out_dir / f"{prefix}_paired_object_improvement"
        _write_table(paired_object, paired_object_path)
        paths["paired_object"] = paired_object_path.with_suffix(".csv")
    if not paired_summary.empty:
        paired_summary_path = out_dir / f"{prefix}_paired_method_improvement_summary"
        _write_table(paired_summary, paired_summary_path)
        paths["paired_summary"] = paired_summary_path.with_suffix(".csv")
    return paths


def _method_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in frame.groupby("method", sort=False):
        valid = _valid(group)
        residual = valid["residual_sigma_median"].to_numpy(float)
        abs_residual = np.abs(residual)
        rows.append(
            {
                "method": method,
                "family": str(valid["family"].iloc[0]) if "family" in valid and len(valid) else "",
                "n_band_rows": int(len(group)),
                "n_valid_band_rows": int(len(valid)),
                "n_objects": int(valid["row_index"].nunique()) if len(valid) else 0,
                "median_residual_sigma": _nanmedian(residual),
                "median_abs_residual_sigma": _nanmedian(abs_residual),
                "mean_abs_residual_sigma": _nanmean(abs_residual),
                "p90_abs_residual_sigma": _nanpercentile(abs_residual, 90),
                "p95_abs_residual_sigma": _nanpercentile(abs_residual, 95),
                "frac_abs_gt3": _frac(abs_residual > 3.0),
                "frac_abs_gt5": _frac(abs_residual > 5.0),
            }
        )
    return pd.DataFrame(rows)


def _band_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, band), group in frame.groupby(["method", "band"], sort=False):
        valid = _valid(group)
        residual = valid["residual_sigma_median"].to_numpy(float)
        abs_residual = np.abs(residual)
        rows.append(
            {
                "method": method,
                "band": band,
                "family": str(valid["family"].iloc[0]) if "family" in valid and len(valid) else "",
                "n_valid_band_rows": int(len(valid)),
                "median_residual_sigma": _nanmedian(residual),
                "median_abs_residual_sigma": _nanmedian(abs_residual),
                "p90_abs_residual_sigma": _nanpercentile(abs_residual, 90),
                "p95_abs_residual_sigma": _nanpercentile(abs_residual, 95),
                "frac_abs_gt3": _frac(abs_residual > 3.0),
                "frac_abs_gt5": _frac(abs_residual > 5.0),
            }
        )
    return pd.DataFrame(rows)


def _object_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, row_index), group in frame.groupby(["method", "row_index"], sort=False):
        valid = _valid(group)
        residual = valid["residual_sigma_median"].to_numpy(float)
        abs_residual = np.abs(residual)
        rows.append(
            {
                "method": method,
                "row_index": row_index,
                "object_id": valid["object_id"].iloc[0] if len(valid) and "object_id" in valid else row_index,
                "family": str(valid["family"].iloc[0]) if "family" in valid and len(valid) else "",
                "n_valid_bands": int(len(valid)),
                "median_residual_sigma": _nanmedian(residual),
                "median_abs_residual_sigma": _nanmedian(abs_residual),
                "max_abs_residual_sigma": _nanmax(abs_residual),
                "frac_abs_gt3": _frac(abs_residual > 3.0),
                "frac_abs_gt5": _frac(abs_residual > 5.0),
            }
        )
    return pd.DataFrame(rows)


def _paired_against_baseline(
    frame: pd.DataFrame,
    *,
    baseline_method: str = "reference_nn",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = _valid(frame).copy()
    if valid.empty or baseline_method not in set(valid["method"].astype(str)):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    valid["method"] = valid["method"].astype(str)
    band_base = valid[valid["method"] == baseline_method][
        ["row_index", "band", "residual_sigma_median", "abs_residual_sigma_median"]
    ].rename(
        columns={
            "residual_sigma_median": "baseline_residual_sigma",
            "abs_residual_sigma_median": "baseline_abs_residual_sigma",
        }
    )
    band_other = valid[valid["method"] != baseline_method][
        ["method", "row_index", "band", "residual_sigma_median", "abs_residual_sigma_median"]
    ].rename(
        columns={
            "residual_sigma_median": "method_residual_sigma",
            "abs_residual_sigma_median": "method_abs_residual_sigma",
        }
    )
    paired_band = band_other.merge(band_base, on=["row_index", "band"], how="inner")
    paired_band["band_improvement"] = (
        paired_band["baseline_abs_residual_sigma"] - paired_band["method_abs_residual_sigma"]
    )
    paired_band["band_improved"] = paired_band["band_improvement"] > 0
    paired_band["band_error_ratio"] = paired_band["method_abs_residual_sigma"] / paired_band[
        "baseline_abs_residual_sigma"
    ].replace(0, np.nan)

    obj = _object_summary(valid)
    obj["method"] = obj["method"].astype(str)
    obj_base = obj[obj["method"] == baseline_method][
        ["row_index", "median_abs_residual_sigma", "max_abs_residual_sigma", "frac_abs_gt3", "frac_abs_gt5"]
    ].rename(
        columns={
            "median_abs_residual_sigma": "baseline_object_median_abs_residual_sigma",
            "max_abs_residual_sigma": "baseline_object_max_abs_residual_sigma",
            "frac_abs_gt3": "baseline_object_frac_abs_gt3",
            "frac_abs_gt5": "baseline_object_frac_abs_gt5",
        }
    )
    obj_other = obj[obj["method"] != baseline_method][
        ["method", "row_index", "median_abs_residual_sigma", "max_abs_residual_sigma", "frac_abs_gt3", "frac_abs_gt5"]
    ].rename(
        columns={
            "median_abs_residual_sigma": "method_object_median_abs_residual_sigma",
            "max_abs_residual_sigma": "method_object_max_abs_residual_sigma",
            "frac_abs_gt3": "method_object_frac_abs_gt3",
            "frac_abs_gt5": "method_object_frac_abs_gt5",
        }
    )
    paired_object = obj_other.merge(obj_base, on="row_index", how="inner")
    paired_object["object_improvement"] = (
        paired_object["baseline_object_median_abs_residual_sigma"]
        - paired_object["method_object_median_abs_residual_sigma"]
    )
    paired_object["object_improved"] = paired_object["object_improvement"] > 0
    paired_object["object_error_ratio"] = paired_object[
        "method_object_median_abs_residual_sigma"
    ] / paired_object["baseline_object_median_abs_residual_sigma"].replace(0, np.nan)

    rows = []
    for method, band_group in paired_band.groupby("method", sort=False):
        obj_group = paired_object[paired_object["method"] == method]
        rows.append(
            {
                "method": method,
                "n_band_pairs": int(len(band_group)),
                "n_object_pairs": int(len(obj_group)),
                "baseline_band_median_abs_residual_sigma": _nanmedian(
                    band_group["baseline_abs_residual_sigma"].to_numpy(float)
                ),
                "method_band_median_abs_residual_sigma": _nanmedian(
                    band_group["method_abs_residual_sigma"].to_numpy(float)
                ),
                "median_band_improvement": _nanmedian(
                    band_group["band_improvement"].to_numpy(float)
                ),
                "mean_band_improvement": _nanmean(
                    band_group["band_improvement"].to_numpy(float)
                ),
                "frac_band_improved": _frac(
                    band_group["band_improvement"].to_numpy(float) > 0
                ),
                "baseline_object_median_abs_residual_sigma": _nanmedian(
                    obj_group["baseline_object_median_abs_residual_sigma"].to_numpy(float)
                ),
                "method_object_median_abs_residual_sigma": _nanmedian(
                    obj_group["method_object_median_abs_residual_sigma"].to_numpy(float)
                ),
                "median_object_improvement": _nanmedian(
                    obj_group["object_improvement"].to_numpy(float)
                ),
                "mean_object_improvement": _nanmean(
                    obj_group["object_improvement"].to_numpy(float)
                ),
                "frac_object_improved": _frac(
                    obj_group["object_improvement"].to_numpy(float) > 0
                ),
            }
        )
    paired_summary = pd.DataFrame(rows)
    return paired_band, paired_object, paired_summary


def _plot_reference_full(frame: pd.DataFrame, out_dir: Path) -> None:
    _ensure(out_dir)
    valid = _valid(frame)
    _plot_residual_hist(valid, out_dir / "reference_full_residual_hist.png", "Canonical reference full 20k")
    _plot_boxplot_by_band(
        valid,
        out_dir / "reference_full_residual_boxplot_by_band.png",
        "Canonical reference full 20k residuals by band",
    )
    _plot_hist_by_band(
        valid,
        out_dir / "reference_full_residual_hist_by_band.png",
        "Canonical reference full 20k residual histograms by band",
    )
    obj = _object_summary(frame)
    worst = obj.sort_values("median_abs_residual_sigma", ascending=False).head(30)
    _plot_top_objects(worst, out_dir / "reference_full_top_objects.png", "Worst reference NN objects")
    _plot_cdf_by_method(frame, out_dir / "reference_full_abs_residual_cdf.png", "Reference full 20k")


def _plot_comparison_suite(frame: pd.DataFrame, out_dir: Path, title: str) -> None:
    _ensure(out_dir)
    method = _method_summary(frame)
    band = _band_summary(frame)
    paired_band, paired_object, paired_summary = _paired_against_baseline(frame)
    _plot_method_summary(method, out_dir / "method_residual_summary.png", title)
    _plot_outlier_fraction(method, out_dir / "method_outlier_fraction.png", title)
    _plot_band_heatmap(band, out_dir / "band_median_abs_residual_heatmap.png", title)
    _plot_cdf_by_method(frame, out_dir / "method_abs_residual_cdf.png", title)
    _plot_method_hist_grid(frame, out_dir / "method_residual_histograms.png", title)
    _plot_same_object_error_distribution(
        frame,
        out_dir / "same_object_error_distribution_by_method.png",
        title,
    )
    _plot_same_band_error_distribution(
        frame,
        out_dir / "same_band_error_distribution_by_method.png",
        title,
    )
    _plot_paired_improvement_summary(
        paired_summary,
        out_dir / "paired_baseline_improvement_summary.png",
        title,
    )
    _plot_paired_object_scatter(
        paired_object,
        out_dir / "paired_object_baseline_vs_methods.png",
        title,
    )
    _plot_paired_band_scatter(
        paired_band,
        out_dir / "paired_band_baseline_vs_methods.png",
        title,
    )
    _plot_paired_improvement_histograms(
        paired_object,
        "object_improvement",
        out_dir / "paired_object_improvement_histograms.png",
        f"{title}: object-level improvement",
    )
    _plot_paired_improvement_histograms(
        paired_band,
        "band_improvement",
        out_dir / "paired_band_improvement_histograms.png",
        f"{title}: band-level improvement",
    )
    _plot_boxplot_by_method_and_band(
        frame,
        out_dir / "residual_boxplot_by_band_by_method.png",
        f"{title} residuals by band and method",
    )
    for method_name, group in frame.groupby("method", sort=False):
        safe = _safe_name(str(method_name))
        _plot_hist_by_band(
            group,
            out_dir / f"residual_hist_by_band_{safe}.png",
            f"{title}: {method_name}",
        )


def _plot_map_suite(spec: RunSpec, out_dir: Path) -> None:
    _ensure(out_dir)
    residuals = _load_run_residuals(spec, method=spec.label)
    wavelength_map = _wavelength_map_from_residuals(residuals)
    _plot_residual_hist(residuals, out_dir / "map_residual_hist.png", f"{spec.label} residuals")
    _plot_boxplot_by_band(residuals, out_dir / "map_residual_boxplot_by_band.png", f"{spec.label} residuals by band")
    _plot_hist_by_band(residuals, out_dir / "map_residual_hist_by_band.png", f"{spec.label} residual histograms")
    _plot_observed_vs_model_by_band(
        residuals,
        out_dir / "map_observed_vs_model_flux_by_band.png",
        f"{spec.label}: observed vs model flux",
    )
    results = _read_table_stem(spec.path, "batch_fit_results")
    param = _extract_map_parameters(results)
    truth = _truth_frame_for_parameters(results)
    prior_bounds = _prior_bounds_from_config(spec.path)
    if not param.empty:
        _plot_parameter_distributions(
            param,
            out_dir / "map_parameter_distributions.png",
            f"{spec.label} MAP parameters",
            truth=truth,
            prior_bounds=prior_bounds,
            sample_label="MAP fit",
        )
        _plot_corner_like(
            param,
            out_dir / "map_parameter_corner.png",
            f"{spec.label} MAP parameter corner",
            truth=truth,
            prior_bounds=prior_bounds,
            sample_label="MAP fit",
            truth_label="catalog truth",
        )
        if "chi2" in results:
            param_chi = param.copy()
            param_chi["chi2"] = pd.to_numeric(results["chi2"], errors="coerce")
            _plot_parameter_vs_score(param_chi, "chi2", out_dir / "map_chi2_vs_parameters.png", f"{spec.label} chi2 vs parameters")
    obj = _object_summary(residuals)
    _plot_top_objects(
        obj.sort_values("median_abs_residual_sigma", ascending=False).head(30),
        out_dir / "map_worst_objects.png",
        f"{spec.label} worst objects",
    )
    _plot_photometric_sed_triplet(
        residuals,
        out_dir / "map_photometric_sed_best_median_worst.png",
        f"{spec.label}: photometric SED points",
        wavelength_map=wavelength_map,
    )


def _plot_mclmc_suite(spec: RunSpec, out_dir: Path, comparison_frame: pd.DataFrame) -> None:
    _ensure(out_dir)
    residuals = _load_run_residuals(spec, method=spec.label)
    wavelength_map = _global_wavelength_map()
    _plot_residual_hist(residuals, out_dir / "mclmc_residual_hist.png", f"{spec.label} residuals")
    _plot_boxplot_by_band(
        residuals,
        out_dir / "mclmc_residual_boxplot_by_band.png",
        f"{spec.label} residuals by band",
    )
    _plot_hist_by_band(residuals, out_dir / "mclmc_residual_hist_by_band.png", f"{spec.label} residual histograms")
    _plot_observed_vs_model_by_band(
        residuals,
        out_dir / "mclmc_observed_vs_posterior_predictive_by_band.png",
        f"{spec.label}: observed vs posterior predictive median",
    )

    summary_path = spec.path / "batch_posterior_summary.csv"
    samples_path = spec.path / "batch_posterior_samples.csv"
    diag_path = spec.path / "batch_mcmc_diagnostics.csv"
    medians = pd.DataFrame()
    truth = pd.DataFrame()
    prior_bounds = _prior_bounds_from_config(spec.path)
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        medians = _posterior_summary_to_wide(summary, value="median")
        truth = _truth_frame_for_parameters(summary).drop_duplicates("row_index")
        projected_truth, truth_metadata = _mclmc_projected_truth_frame(summary, spec.path)
        if not projected_truth.empty:
            truth = projected_truth
            _write_table(projected_truth, out_dir / "mclmc_projected_truth_parameters")
        if not truth_metadata.empty:
            _write_table(truth_metadata, out_dir / "mclmc_projected_truth_metadata")
        if not medians.empty:
            median_truth_cols = _mclmc_truth_corner_columns(medians, truth)
            _plot_parameter_distributions(
                medians,
                out_dir / "mclmc_posterior_median_distributions.png",
                "MCLMC posterior median parameter distributions",
                columns=median_truth_cols,
                truth=truth,
                prior_bounds=prior_bounds,
                sample_label="posterior median per galaxy",
                truth_label="catalog/projected truth",
            )
        truth_distribution_cols = _finite_numeric_columns(
            truth,
            MCLMC_PROJECTED_TRUTH_DISTRIBUTION_PARAMETERS,
        )
        if truth_distribution_cols:
            _plot_parameter_distributions(
                truth,
                out_dir / "mclmc_projected_truth_distributions.png",
                "MCLMC catalog/projected truth distributions",
                columns=truth_distribution_cols,
                prior_bounds=prior_bounds,
                sample_label="catalog/projected truth",
                sample_color="#F58518",
            )
    if samples_path.exists():
        samples = pd.read_csv(samples_path)
        cols = _mclmc_truth_corner_columns(samples, truth)
        if len(cols) >= 2:
            _plot_corner_density(
                samples,
                out_dir / "mclmc_corner_pooled_samples.png",
                "MCLMC pooled posterior samples",
                columns=cols,
                truth=truth,
                prior_bounds=prior_bounds,
                point_frame=medians,
                sample_label="posterior sample density",
                truth_label="catalog/projected truth",
                point_label="posterior median per galaxy",
            )
            if not medians.empty:
                _plot_corner_density(
                    medians,
                    out_dir / "mclmc_corner_posterior_medians.png",
                    "MCLMC posterior medians vs catalog/projected truth",
                    columns=cols,
                    truth=truth,
                    prior_bounds=prior_bounds,
                    sample_label="posterior median density",
                    truth_label="catalog/projected truth",
                )
                truth_cols = _mclmc_truth_corner_columns(medians, truth)
                if len(truth_cols) >= 2:
                    _plot_corner_density(
                        medians,
                        out_dir / "mclmc_corner_truth_comparable.png",
                        "MCLMC projected-truth posterior medians",
                        columns=truth_cols,
                        truth=truth,
                        prior_bounds=prior_bounds,
                        sample_label="posterior median density",
                        truth_label="catalog/projected truth",
                    )
            _plot_mclmc_individual_corners(
                samples,
                residuals,
                out_dir / "individual_corners",
                truth=truth,
                prior_bounds=prior_bounds,
            )
    if diag_path.exists():
        diagnostics = pd.read_csv(diag_path)
        _plot_mclmc_diagnostics(diagnostics, out_dir / "mclmc_diagnostics_summary.png")

    mclmc_only = comparison_frame[comparison_frame["method"] == spec.label]
    if not mclmc_only.empty:
        obj = _object_summary(mclmc_only)
        _plot_top_objects(
            obj.sort_values("median_abs_residual_sigma", ascending=False).head(30),
            out_dir / "mclmc_worst_objects.png",
            "MCLMC worst objects",
        )
    _plot_photometric_sed_triplet(
        residuals,
        out_dir / "mclmc_photometric_sed_best_median_worst.png",
        "MCLMC: photometric SED points",
        wavelength_map=wavelength_map,
    )


def _plot_training_suite(training_specs: list[TrainingSpec], out_dir: Path) -> None:
    _ensure(out_dir)
    frames = []
    logs = []
    redshift = []
    for spec in training_specs:
        epoch_path = spec.path / "training_epoch_summary.csv"
        log_path = spec.path / "training_log.csv"
        z_path = spec.path / "validation_redshift_bin_metrics.csv"
        if epoch_path.exists():
            frame = pd.read_csv(epoch_path)
            frame["run"] = spec.key
            frame["label"] = spec.label
            frames.append(frame)
        if log_path.exists():
            frame = pd.read_csv(log_path)
            frame["run"] = spec.key
            frame["label"] = spec.label
            logs.append(frame)
        if z_path.exists():
            frame = pd.read_csv(z_path)
            frame["run"] = spec.key
            frame["label"] = spec.label
            redshift.append(frame)
    if frames:
        epoch = pd.concat(frames, ignore_index=True)
        epoch.to_csv(out_dir / "training_epoch_summary_all_runs.csv", index=False)
        _plot_training_metric_grid(epoch, out_dir / "training_loss_nll_kl_curves.png")
        _plot_training_metric(epoch, "negative_loglike", out_dir / "training_negative_loglike.png")
        _plot_training_metric(epoch, "loss", out_dir / "training_loss.png")
        _plot_training_metric(epoch, "kl_mc_mean", out_dir / "training_kl_mc_mean.png")
        _plot_training_metric(epoch, "residual_rms", out_dir / "training_residual_rms.png")
    if logs:
        log = pd.concat(logs, ignore_index=True)
        _plot_batch_gradient_summary(log, out_dir / "training_gradient_norms_by_epoch.png")
        _plot_batch_update_finiteness(log, out_dir / "training_update_health.png")
    if redshift:
        z = pd.concat(redshift, ignore_index=True)
        _plot_validation_redshift(z, out_dir / "validation_redshift_bin_chi2.png")


def _link_nn_inference_plots(run_specs: list[RunSpec], out_dir: Path) -> None:
    _ensure(out_dir)
    for spec in run_specs:
        if spec.family != "nn":
            continue
        method_dir = _ensure(out_dir / spec.key)
        for filename in NN_INFERENCE_PLOTS:
            src = spec.path / filename
            if src.exists():
                _link_or_copy(src, method_dir / filename, copy_runs=True)


def _plot_method_summary(method: pd.DataFrame, path: Path, title: str) -> None:
    if method.empty:
        return
    x = np.arange(len(method))
    labels = method["method"].astype(str).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 1.1), 4.8), constrained_layout=True)
    axes[0].bar(x, method["median_abs_residual_sigma"], color="#4C78A8")
    axes[0].set_ylabel(f"median |{RESIDUAL_LABEL}|")
    axes[0].set_title("Median residual")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].bar(x, method["p95_abs_residual_sigma"], color="#F58518")
    axes[1].set_ylabel(f"p95 |{RESIDUAL_LABEL}|")
    axes[1].set_title("Tail residual")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    fig.suptitle(title)
    _savefig(fig, path)


def _plot_outlier_fraction(method: pd.DataFrame, path: Path, title: str) -> None:
    if method.empty:
        return
    labels = method["method"].astype(str).tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 4.8), constrained_layout=True)
    ax.bar(x - 0.18, method["frac_abs_gt3"], width=0.36, label=">3 sigma", color="#54A24B")
    ax.bar(x + 0.18, method["frac_abs_gt5"], width=0.36, label=">5 sigma", color="#E45756")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction of valid band residuals")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(frameon=False)
    ax.set_title(f"{title}: outlier fractions")
    _savefig(fig, path)


def _plot_band_heatmap(band: pd.DataFrame, path: Path, title: str) -> None:
    if band.empty:
        return
    pivot = band.pivot(index="method", columns="band", values="median_abs_residual_sigma")
    method_order = band["method"].drop_duplicates().astype(str).tolist()
    band_order = band["band"].drop_duplicates().astype(str).tolist()
    pivot = pivot.reindex(index=method_order, columns=band_order)
    fig, ax = plt.subplots(
        figsize=(max(11, len(band_order) * 0.65), max(4.5, len(method_order) * 0.55)),
        constrained_layout=True,
    )
    values = pivot.to_numpy(dtype=float)
    vmax = np.nanpercentile(values, 90) if np.isfinite(values).any() else 1.0
    im = ax.imshow(values, aspect="auto", cmap="magma_r", vmin=0, vmax=max(vmax, 1.0))
    ax.set_xticks(np.arange(len(band_order)))
    ax.set_xticklabels(band_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels(method_order)
    ax.set_title(f"{title}: median absolute normalized residual by band")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"median |{RESIDUAL_LABEL}|")
    _savefig(fig, path)


def _plot_cdf_by_method(frame: pd.DataFrame, path: Path, title: str) -> None:
    valid = _valid(frame)
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for method, group in valid.groupby("method", sort=False):
        values = np.sort(group["abs_residual_sigma_median"].dropna().to_numpy(float))
        if values.size == 0:
            continue
        y = np.arange(1, values.size + 1) / values.size
        ax.plot(values, y, label=str(method), linewidth=1.8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel(f"|{RESIDUAL_LABEL}|")
    ax.set_ylabel("CDF")
    ax.set_title(f"{title}: residual CDF")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    _savefig(fig, path)


def _plot_method_hist_grid(frame: pd.DataFrame, path: Path, title: str) -> None:
    valid = _valid(frame)
    methods = valid["method"].drop_duplicates().astype(str).tolist()
    if not methods:
        return
    ncols = 2
    nrows = math.ceil(len(methods) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(3.0, nrows * 2.8)), constrained_layout=True)
    axes_flat = np.ravel(np.asarray(axes))
    for ax, method in zip(axes_flat, methods):
        values = valid.loc[valid["method"].astype(str) == method, "residual_sigma_median"].dropna().to_numpy(float)
        clipped = _clip_for_display(values)
        ax.hist(clipped, bins=60, color="#4C78A8", alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(-3, color="#E45756", linestyle="--", linewidth=0.9)
        ax.axvline(3, color="#E45756", linestyle="--", linewidth=0.9)
        ax.set_title(method)
        ax.set_xlabel(RESIDUAL_LABEL)
    for ax in axes_flat[len(methods) :]:
        ax.axis("off")
    fig.suptitle(f"{title}: residual histograms")
    _savefig(fig, path)


def _plot_selected_signed_residual_histograms(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    methods: list[str],
) -> None:
    valid = _valid(frame).copy()
    valid["method"] = valid["method"].astype(str)
    selected = [method for method in methods if method in set(valid["method"])]
    if not selected:
        return
    all_values = pd.to_numeric(
        valid.loc[valid["method"].isin(selected), "residual_sigma_median"],
        errors="coerce",
    ).dropna().to_numpy(float)
    if all_values.size == 0:
        return
    lo, hi = np.nanpercentile(all_values, [0.5, 99.5])
    limit = max(abs(lo), abs(hi), 10.0)
    limit = min(math.ceil(limit / 5.0) * 5.0, 60.0)
    bins = np.linspace(-limit, limit, 121)

    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(10.5, max(3.2, 2.7 * len(selected))),
        sharex=True,
        constrained_layout=True,
    )
    axes_flat = np.ravel(np.asarray(axes))
    colors = {
        "reference_nn": "#4C78A8",
        "map_500_iter200": "#F58518",
        "map_1000_iter400": "#F58518",
        "mclmc": "#54A24B",
    }
    for ax, method in zip(axes_flat, selected):
        values = pd.to_numeric(
            valid.loc[valid["method"] == method, "residual_sigma_median"],
            errors="coerce",
        ).dropna().to_numpy(float)
        if values.size == 0:
            ax.axis("off")
            continue
        clipped = np.clip(values, -limit, limit)
        ax.hist(clipped, bins=bins, color=colors.get(method, "#4C78A8"), alpha=0.82)
        ax.axvline(-3, color="#E45756", linestyle="--", linewidth=1.0, label="-3/+3")
        ax.axvline(3, color="#E45756", linestyle="--", linewidth=1.0)
        ax.axvline(0, color="black", linewidth=0.9, label="0")
        median_abs = float(np.nanmedian(np.abs(values)))
        frac_gt3 = float(np.nanmean(np.abs(values) > 3))
        frac_gt5 = float(np.nanmean(np.abs(values) > 5))
        ax.set_title(
            f"{method}: median |res|={median_abs:.2f}, "
            f"frac |res|>3={frac_gt3:.2f}, frac |res|>5={frac_gt5:.2f}"
        )
        ax.set_ylabel("band rows")
        ax.grid(axis="y", alpha=0.18)
    axes_flat[-1].set_xlabel(RESIDUAL_LABEL)
    fig.suptitle(f"{title}  (values clipped to +/-{limit:g} for display)")
    _savefig(fig, path)


def _plot_worst_slice_location_in_full_reference(
    reference_full: pd.DataFrame,
    slice_frame: pd.DataFrame,
    path: Path,
    title: str,
    *,
    baseline_method: str,
    map_method: str,
) -> None:
    full_valid = _valid(reference_full).copy()
    slice_valid = _valid(slice_frame).copy()
    if full_valid.empty or slice_valid.empty:
        return
    slice_valid["method"] = slice_valid["method"].astype(str)
    baseline = slice_valid[slice_valid["method"] == baseline_method]
    mapped = slice_valid[slice_valid["method"] == map_method]
    if baseline.empty or mapped.empty:
        return

    full_obj = _object_summary(full_valid)
    baseline_obj = _object_summary(baseline)
    map_obj = _object_summary(mapped)
    full_values = _finite_array(full_obj.get("median_abs_residual_sigma"))
    baseline_values = _finite_array(baseline_obj.get("median_abs_residual_sigma"))
    map_values = _finite_array(map_obj.get("median_abs_residual_sigma"))
    if not (full_values.size and baseline_values.size and map_values.size):
        return

    paired = baseline_obj[
        ["row_index", "median_abs_residual_sigma"]
    ].rename(columns={"median_abs_residual_sigma": "baseline_abs"})
    paired = paired.merge(
        map_obj[["row_index", "median_abs_residual_sigma"]].rename(
            columns={"median_abs_residual_sigma": "map_abs"}
        ),
        on="row_index",
        how="inner",
    )
    _, object_diag = _huge_error_bar_diagnostic_frames(
        slice_frame,
        baseline_method=baseline_method,
        recovery_method=map_method,
        mcmc_method=None,
    )
    if not object_diag.empty:
        paired = paired.merge(
            object_diag[["row_index", "max_err_over_abs_flux"]],
            on="row_index",
            how="left",
        )
    else:
        paired["max_err_over_abs_flux"] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    _plot_location_strip_panel(
        axes[0],
        full_values,
        baseline_values,
        map_values,
    )
    _plot_location_paired_panel(
        axes[1],
        paired,
        full_values,
    )
    fig.suptitle(title)
    _savefig(fig, path)


def _plot_location_strip_panel(
    ax,
    full_values: np.ndarray,
    baseline_values: np.ndarray,
    mapped_values: np.ndarray,
) -> None:
    rng = np.random.default_rng(42)
    sample = full_values
    if sample.size > 4000:
        sample = rng.choice(sample, size=4000, replace=False)
    y_full = np.full(sample.size, 2.0) + rng.normal(0, 0.035, sample.size)
    y_base = np.full(baseline_values.size, 1.0) + rng.normal(0, 0.035, baseline_values.size)
    y_map = np.full(mapped_values.size, 0.0) + rng.normal(0, 0.035, mapped_values.size)
    ax.scatter(sample, y_full, s=7, color="#9D9D9D", alpha=0.16, rasterized=True)
    ax.scatter(baseline_values, y_base, s=20, color="#E45756", alpha=0.65, label="worst100 in NN")
    ax.scatter(mapped_values, y_map, s=20, color="#F58518", alpha=0.75, label="same rows after MAP")
    full_med = float(np.nanmedian(full_values))
    base_med = float(np.nanmedian(baseline_values))
    map_med = float(np.nanmedian(mapped_values))
    ax.scatter([full_med], [2.0], marker="D", s=70, color="#555555", label="median")
    ax.scatter([base_med], [1.0], marker="D", s=70, color="#E45756")
    ax.scatter([map_med], [0.0], marker="D", s=70, color="#F58518")
    for value, text, color, y in [
        (full_med, f"full median {full_med:.2f}", "#555555", 2.18),
        (base_med, f"worst100 NN median {base_med:.2f}", "#E45756", 1.18),
        (map_med, f"MAP median {map_med:.2f}", "#F58518", 0.18),
    ]:
        ax.annotate(text, xy=(value, y - 0.13), xytext=(value, y), color=color, fontsize=8)
    ax.axvline(3, color="#E45756", linestyle=":", linewidth=0.9)
    ax.axvline(5, color="#B279A2", linestyle=":", linewidth=0.9)
    ax.set_xscale("symlog", linthresh=1)
    all_values = np.concatenate([full_values, baseline_values, mapped_values])
    all_values = all_values[np.isfinite(all_values)]
    if all_values.size:
        ax.set_xlim(left=0.0, right=max(float(np.nanmax(all_values)) * 1.15, 5.0))
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["same 100 after MAP", "same 100 in NN", "full 20k NN"])
    ax.set_xlabel(f"object median |{RESIDUAL_LABEL}|")
    ax.set_title("Where the same objects move")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)


def _plot_location_paired_panel(
    ax,
    paired: pd.DataFrame,
    full_values: np.ndarray,
) -> None:
    if paired.empty:
        ax.axis("off")
        return
    work = paired.copy()
    for column in ["baseline_abs", "map_abs", "max_err_over_abs_flux"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    finite = work[np.isfinite(work["baseline_abs"]) & np.isfinite(work["map_abs"])]
    if finite.empty:
        ax.axis("off")
        return
    huge = finite["max_err_over_abs_flux"] > 100.0
    ax.scatter(
        finite.loc[~huge, "baseline_abs"],
        finite.loc[~huge, "map_abs"],
        s=26,
        color="#4C78A8",
        alpha=0.7,
        label="ordinary error bars",
    )
    ax.scatter(
        finite.loc[huge, "baseline_abs"],
        finite.loc[huge, "map_abs"],
        s=54,
        facecolor="none",
        edgecolor="#E45756",
        linewidth=1.3,
        label="max err/|flux| > 100",
    )
    paired_values = np.concatenate(
        [finite["baseline_abs"].to_numpy(float), finite["map_abs"].to_numpy(float)]
    )
    values = np.concatenate([paired_values, full_values])
    values = values[np.isfinite(values)]
    positive_paired = paired_values[np.isfinite(paired_values) & (paired_values > 0)]
    lo = max(float(np.nanmin(positive_paired)) * 0.65, 1.0e-3) if positive_paired.size else 0.02
    hi = max(
        float(np.nanpercentile(values, 99.5)) * 1.4,
        float(np.nanmax(paired_values[np.isfinite(paired_values)])) * 1.25,
        5.0,
    )
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.9, label="no change")
    ax.axhline(3, color="#E45756", linestyle=":", linewidth=0.9)
    ax.axvline(3, color="#E45756", linestyle=":", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"NN baseline object median |{RESIDUAL_LABEL}|")
    ax.set_ylabel(f"MAP object median |{RESIDUAL_LABEL}|")
    ax.set_title("Same-object comparison\nbelow diagonal = lower normalized error")
    ax.grid(alpha=0.22)
    for _, row in finite.loc[huge].sort_values("max_err_over_abs_flux", ascending=False).head(5).iterrows():
        ax.annotate(
            str(int(row["row_index"])),
            (row["baseline_abs"], row["map_abs"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="#E45756",
        )
    ax.legend(frameon=False, fontsize=8)


def _write_huge_error_bar_diagnostics(
    frame: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    *,
    baseline_method: str,
    recovery_method: str,
    mcmc_method: str | None,
) -> None:
    band_diag, object_diag = _huge_error_bar_diagnostic_frames(
        frame,
        baseline_method=baseline_method,
        recovery_method=recovery_method,
        mcmc_method=mcmc_method,
    )
    if band_diag.empty or object_diag.empty:
        return
    _write_table(
        band_diag,
        tables_dir / "worst100_huge_error_bar_band_diagnostics",
    )
    _write_table(
        object_diag,
        tables_dir / "worst100_huge_error_bar_object_diagnostics",
    )
    _plot_huge_error_bar_diagnostics(
        object_diag,
        plots_dir / "huge_error_bar_diagnostics.png",
    )
    _write_huge_error_bar_report(
        band_diag,
        object_diag,
        tables_dir / "worst100_huge_error_bar_explanation.md",
    )


def _huge_error_bar_diagnostic_frames(
    frame: pd.DataFrame,
    *,
    baseline_method: str,
    recovery_method: str,
    mcmc_method: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = _valid(frame).copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()
    valid["method"] = valid["method"].astype(str)
    recovery = valid[valid["method"] == recovery_method].copy()
    baseline = valid[valid["method"] == baseline_method].copy()
    if recovery.empty or baseline.empty:
        return pd.DataFrame(), pd.DataFrame()

    for col in [
        "obs_flux_fnu_cgs",
        "obs_err_fnu_cgs",
        "sigma_eff_median",
        "model_flux_median",
        "residual_sigma_median",
        "abs_residual_sigma_median",
        "redshift_truth",
        "truth_log10_stellar_mass",
        "truth_log10_sfr_at_obs",
    ]:
        if col in recovery:
            recovery[col] = pd.to_numeric(recovery[col], errors="coerce")
        if col in baseline:
            baseline[col] = pd.to_numeric(baseline[col], errors="coerce")

    out = recovery[
        [
            col
            for col in [
                "row_index",
                "object_id",
                "band",
                "obs_flux_fnu_cgs",
                "obs_err_fnu_cgs",
                "sigma_eff_median",
                "model_flux_median",
                "residual_sigma_median",
                "abs_residual_sigma_median",
                "redshift_truth",
                "truth_log10_stellar_mass",
                "truth_log10_sfr_at_obs",
            ]
            if col in recovery
        ]
    ].copy()
    out = out.rename(
        columns={
            "model_flux_median": "map_model_flux_median",
            "residual_sigma_median": "map_residual_sigma",
            "abs_residual_sigma_median": "map_abs_residual_sigma",
        }
    )
    base_cols = baseline[
        ["row_index", "band", "model_flux_median", "abs_residual_sigma_median"]
    ].rename(
        columns={
            "model_flux_median": "nn_model_flux_median",
            "abs_residual_sigma_median": "nn_abs_residual_sigma",
        }
    )
    out = out.merge(base_cols, on=["row_index", "band"], how="left")
    if mcmc_method and mcmc_method in set(valid["method"]):
        mcmc = valid[valid["method"] == mcmc_method][
            ["row_index", "band", "model_flux_median", "abs_residual_sigma_median"]
        ].rename(
            columns={
                "model_flux_median": "mclmc_model_flux_median",
                "abs_residual_sigma_median": "mclmc_abs_residual_sigma",
            }
        )
        out = out.merge(mcmc, on=["row_index", "band"], how="left")

    obs_flux = pd.to_numeric(out["obs_flux_fnu_cgs"], errors="coerce")
    obs_err = pd.to_numeric(out["obs_err_fnu_cgs"], errors="coerce")
    sigma_eff = pd.to_numeric(out["sigma_eff_median"], errors="coerce")
    map_flux = pd.to_numeric(out["map_model_flux_median"], errors="coerce")
    abs_flux = obs_flux.abs().replace(0.0, np.nan)
    out["obs_abs_flux_fnu_cgs"] = obs_flux.abs()
    out["obs_mag_ab"] = _abmag_from_fnu_cgs(obs_flux)
    out["obs_err_over_abs_flux"] = obs_err / abs_flux
    out["sigma_eff_over_abs_flux"] = sigma_eff / abs_flux
    out["map_abs_flux_residual_fnu_cgs"] = (obs_flux - map_flux).abs()
    out["map_abs_flux_residual_over_abs_flux"] = out["map_abs_flux_residual_fnu_cgs"] / abs_flux
    out["likelihood_floor_term_fnu_cgs"] = 0.02 * obs_flux.abs()
    out["sigma_eff_recomputed_fnu_cgs"] = np.sqrt(
        obs_err**2 + out["likelihood_floor_term_fnu_cgs"] ** 2
    )
    out["sigma_eff_minus_recomputed_fnu_cgs"] = sigma_eff - out["sigma_eff_recomputed_fnu_cgs"]
    out["map_normalized_gain_vs_nn"] = out["nn_abs_residual_sigma"] - out["map_abs_residual_sigma"]
    out["map_student_t_neg2loglike_contribution_nu2"] = 3.0 * np.log1p(
        pd.to_numeric(out["map_residual_sigma"], errors="coerce") ** 2 / 2.0
    )
    out["error_bar_warning"] = np.select(
        [
            out["obs_err_over_abs_flux"] > 100.0,
            out["obs_err_over_abs_flux"] > 10.0,
        ],
        [
            "huge error bar relative to flux",
            "large error bar relative to flux",
        ],
        default="ordinary",
    )
    out = out.sort_values(
        ["obs_err_over_abs_flux", "map_normalized_gain_vs_nn"],
        ascending=[False, False],
    ).reset_index(drop=True)

    object_rows = []
    for row_index, group in out.groupby("row_index", sort=False):
        ratio = pd.to_numeric(group["obs_err_over_abs_flux"], errors="coerce")
        worst_idx = ratio.idxmax()
        worst = group.loc[worst_idx]
        object_rows.append(
            {
                "row_index": int(row_index),
                "object_id": worst.get("object_id", row_index),
                "worst_error_band": worst["band"],
                "max_err_over_abs_flux": float(np.nanmax(ratio.to_numpy(float))),
                "median_err_over_abs_flux": _nanmedian(ratio.to_numpy(float)),
                "max_map_abs_flux_residual_over_abs_flux": _nanmax(
                    pd.to_numeric(group["map_abs_flux_residual_over_abs_flux"], errors="coerce").to_numpy(float)
                ),
                "nn_object_median_abs_residual_sigma": _nanmedian(
                    pd.to_numeric(group["nn_abs_residual_sigma"], errors="coerce").to_numpy(float)
                ),
                "map_object_median_abs_residual_sigma": _nanmedian(
                    pd.to_numeric(group["map_abs_residual_sigma"], errors="coerce").to_numpy(float)
                ),
                "mclmc_object_median_abs_residual_sigma": _nanmedian(
                    pd.to_numeric(group.get("mclmc_abs_residual_sigma"), errors="coerce").to_numpy(float)
                    if "mclmc_abs_residual_sigma" in group
                    else np.asarray([], dtype=float)
                ),
                "map_object_normalized_gain_vs_nn": _nanmedian(
                    pd.to_numeric(group["map_normalized_gain_vs_nn"], errors="coerce").to_numpy(float)
                ),
                "redshift_truth": _nanmedian(
                    pd.to_numeric(group.get("redshift_truth"), errors="coerce").to_numpy(float)
                    if "redshift_truth" in group
                    else np.asarray([], dtype=float)
                ),
                "truth_log10_stellar_mass": _nanmedian(
                    pd.to_numeric(group.get("truth_log10_stellar_mass"), errors="coerce").to_numpy(float)
                    if "truth_log10_stellar_mass" in group
                    else np.asarray([], dtype=float)
                ),
                "truth_log10_sfr_at_obs": _nanmedian(
                    pd.to_numeric(group.get("truth_log10_sfr_at_obs"), errors="coerce").to_numpy(float)
                    if "truth_log10_sfr_at_obs" in group
                    else np.asarray([], dtype=float)
                ),
            }
        )
    obj = pd.DataFrame(object_rows).sort_values(
        ["max_err_over_abs_flux", "map_object_normalized_gain_vs_nn"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return out, obj


def _plot_huge_error_bar_diagnostics(object_diag: pd.DataFrame, path: Path) -> None:
    if object_diag.empty:
        return
    work = object_diag.head(30).copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    top = work.head(12).iloc[::-1]
    axes[0].barh(
        top["row_index"].astype(str),
        pd.to_numeric(top["max_err_over_abs_flux"], errors="coerce"),
        color="#E45756",
        alpha=0.82,
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("max obs_err / |obs_flux| across bands")
    axes[0].set_ylabel("row_index")
    axes[0].set_title("Largest catalog error bars relative to flux")
    axes[0].grid(axis="x", alpha=0.22)

    x = pd.to_numeric(object_diag["max_err_over_abs_flux"], errors="coerce")
    y = pd.to_numeric(object_diag["map_object_normalized_gain_vs_nn"], errors="coerce")
    c = pd.to_numeric(object_diag["map_object_median_abs_residual_sigma"], errors="coerce")
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    scatter = axes[1].scatter(
        x[m],
        y[m],
        c=c[m],
        cmap="viridis_r",
        s=38,
        alpha=0.85,
    )
    axes[1].axvline(10, color="#F58518", linestyle=":", linewidth=1.0)
    axes[1].axvline(100, color="#E45756", linestyle="--", linewidth=1.0)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("max obs_err / |obs_flux| across bands")
    axes[1].set_ylabel("object normalized gain: NN |res| - MAP |res|")
    axes[1].set_title("Large normalized gains can be low-SNR artifacts")
    axes[1].grid(alpha=0.22)
    cbar = fig.colorbar(scatter, ax=axes[1])
    cbar.set_label(f"MAP object median |{RESIDUAL_LABEL}|")
    for _, row in object_diag.head(5).iterrows():
        axes[1].annotate(
            str(int(row["row_index"])),
            (row["max_err_over_abs_flux"], row["map_object_normalized_gain_vs_nn"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    fig.suptitle("Huge-error-bar diagnostics on worst100")
    _savefig(fig, path)


def _write_huge_error_bar_report(
    band_diag: pd.DataFrame,
    object_diag: pd.DataFrame,
    path: Path,
) -> None:
    top_band = band_diag.head(15)[
        [
            "row_index",
            "band",
            "obs_flux_fnu_cgs",
            "obs_err_fnu_cgs",
            "obs_err_over_abs_flux",
            "map_model_flux_median",
            "map_abs_flux_residual_over_abs_flux",
            "nn_abs_residual_sigma",
            "map_abs_residual_sigma",
            "map_normalized_gain_vs_nn",
        ]
    ]
    top_obj = object_diag.head(12)[
        [
            "row_index",
            "worst_error_band",
            "max_err_over_abs_flux",
            "median_err_over_abs_flux",
            "nn_object_median_abs_residual_sigma",
            "map_object_median_abs_residual_sigma",
            "map_object_normalized_gain_vs_nn",
            "truth_log10_stellar_mass",
            "truth_log10_sfr_at_obs",
        ]
    ]
    example = band_diag.iloc[0]
    example_depth = _m5_depth_example_values(str(example["band"]))
    obs_flux = float(example["obs_flux_fnu_cgs"])
    map_flux = float(example["map_model_flux_median"])
    abs_flux_residual = float(
        example.get("map_abs_flux_residual_fnu_cgs", abs(obs_flux - map_flux))
    )
    sigma_eff = float(example["sigma_eff_median"])
    floor_term = float(example.get("likelihood_floor_term_fnu_cgs", 0.02 * abs(obs_flux)))
    normalized_calc = abs_flux_residual / sigma_eff if sigma_eff > 0 else float("nan")
    fractional_calc = (
        abs_flux_residual / abs(obs_flux) if abs(obs_flux) > 0 else float("nan")
    )
    text = f"""# Worst100 Huge Error-Bar Diagnostics

These diagnostics explain why some objects look like very large MAP gains in
normalized residual space while their photometric SED panels are still a
disaster in flux space.

## Formula Chain, Step By Step

The processed Diffsky parquet has no native photometric errors. During
preparation, `fluxerr_*` is synthetic. The current `m5_depth` model is the
`photo_err`/PhotErr-style formula used in the debugging notes: the Rubin/LSST
`m5,gamma` point-source form rewritten in flux space:

`f5 = fnu(m5)`

`sigma_rand^2 = (0.04 - gamma) * abs(F_obs) * f5 + gamma * f5^2`

`sys_frac = 10^(sigma_sys_mag / 2.5) - 1`

`fluxerr^2 = sigma_rand^2 + (sys_frac * abs(F_obs))^2`

with `sigma_sys_mag = 0.005` in this dataset. The terms mean:

- `gamma * f5^2`: finite background/depth floor.
- `(0.04 - gamma) * abs(F_obs) * f5`: source/Poisson-like term.
- `(sys_frac * abs(F_obs))^2`: small PhotErr-style systematic floor.

For very faint objects, `abs(F_obs)` is close to zero. The source term and
systematic term disappear, so:

`fluxerr ~= sqrt(gamma) * f5`

The standalone MAP/MCMC fit then uses flux-space likelihood sigma:

`sigma_eff = sqrt(fluxerr^2 + (0.02 * abs(F_obs))^2 + jitter^2)`

with `jitter = 0` in this run. The amortized JAX likelihood uses the same
catalog `fluxerr`, but its 2% floor is taken from `abs(F_model)` instead of
`abs(F_obs)`. This dashboard compares normalized residuals:

`residual_sigma = (F_obs - F_model) / sigma_eff`

Therefore a model can be millions or billions of times larger than a
near-zero `F_obs`, but still have a small normalized residual if the absolute
model flux remains below the survey-depth noise scale.

## Top Example

Top band-row by `obs_err / abs(obs_flux)`:

- row_index: `{int(example["row_index"])}`
- band: `{example["band"]}`
- observed flux: `{obs_flux:.6e}` fnu cgs
- `m5`: `{example_depth["m5"]:.3g}`
- `gamma`: `{example_depth["gamma"]:.3g}`
- `f5 = fnu(m5)`: `{example_depth["f5"]:.6e}` fnu cgs
- `sqrt(gamma) * f5`: `{example_depth["zero_flux_sigma"]:.6e}` fnu cgs
- flux error: `{example["obs_err_fnu_cgs"]:.6e}` fnu cgs
- `obs_err / abs(obs_flux)`: `{example["obs_err_over_abs_flux"]:.6g}`
- likelihood 2% floor term: `{floor_term:.6e}` fnu cgs
- `sigma_eff`: `{sigma_eff:.6e}` fnu cgs
- MAP model flux: `{map_flux:.6e}` fnu cgs
- `abs(F_obs - F_map)`: `{abs_flux_residual:.6e}` fnu cgs
- `abs(F_obs - F_map) / sigma_eff`: `{normalized_calc:.6g}`
- `abs(F_obs - F_map) / abs(F_obs)`: `{fractional_calc:.6g}`
- NN abs residual sigma: `{example["nn_abs_residual_sigma"]:.6g}`
- MAP abs residual sigma: `{example["map_abs_residual_sigma"]:.6g}`

This is a normalized-residual success but not a meaningful fractional-flux
recovery. The MAP flux is far from the catalog flux in relative terms, but the
absolute difference is tiny compared with the synthetic survey-depth floor.

## Interpretation

Rows like this should be annotated or masked when making the argument that
`DSPS can improve the fit relative to the neural network`. They mostly show
that the likelihood is forgiving for near-zero fluxes with finite synthetic
depth errors. For a science-facing comparison, report both the normalized
residual and a flux-scale quantity such as `obs_err / abs(obs_flux)` or
`abs(F_obs - F_model) / abs(F_obs)`.

## Top Objects

{_markdown_table(top_obj, max_rows=12)}

## Top Band Rows

{_markdown_table(top_band, max_rows=15)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _plot_sed_comparison_grid(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    *,
    methods: list[str],
    wavelength_map: dict[str, float],
    baseline_method: str,
    recovery_method: str | None,
    n_per_group: int,
) -> None:
    valid = _valid(frame).copy()
    if valid.empty:
        return
    valid["method"] = valid["method"].astype(str)
    methods = [method for method in methods if method in set(valid["method"])]
    if not methods:
        return
    row_choices = _select_sed_example_rows(
        valid,
        baseline_method=baseline_method,
        recovery_method=recovery_method,
        n_per_group=n_per_group,
    )
    if not row_choices:
        return

    nrows = len(row_choices)
    ncols = len(methods)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(10.5, 4.1 * ncols), max(8.5, 2.45 * nrows)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = {
        "reference_nn": "#4C78A8",
        "map_500_iter200": "#F58518",
        "map_1000_iter400": "#F58518",
        "mclmc": "#54A24B",
    }
    for row_number, choice in enumerate(row_choices):
        y_limits = _sed_flux_limits_for_row(valid, choice["row_index"], methods)
        for col_number, method in enumerate(methods):
            ax = axes[row_number, col_number]
            group = valid[
                (valid["method"] == method)
                & (pd.to_numeric(valid["row_index"], errors="coerce") == choice["row_index"])
            ].copy()
            if group.empty:
                ax.axis("off")
                continue
            _plot_single_sed_panel(
                ax,
                group,
                method=method,
                color=colors.get(method, "#4C78A8"),
                wavelength_map=wavelength_map,
                y_limits=y_limits,
                show_ylabel=col_number == 0,
                show_xlabel=row_number == nrows - 1,
            )
            if row_number == 0:
                ax.set_title(METHOD_DISPLAY_LABELS.get(method, method), fontsize=10)
            metric = _object_metric_for_method(valid, method, choice["row_index"])
            ax.text(
                0.02,
                0.96,
                f"med |res|={metric:.2f}" if np.isfinite(metric) else "med |res|=nan",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7.5,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
        axes[row_number, 0].set_ylabel(
            f"{choice['group']}\nrank {choice['rank']}/100\nrow {int(choice['row_index'])}\n"
            r"$f_\nu$ [cgs]",
            fontsize=8,
        )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_items = [
        Line2D([0], [0], color="black", marker="o", linewidth=1.0, markersize=4, label="catalog target / truth"),
        Line2D([0], [0], color="#4C78A8", marker="s", linewidth=1.3, markersize=3.5, label="NN median"),
        Line2D([0], [0], color="#F58518", marker="s", linewidth=1.3, markersize=3.5, label="MAP model"),
        Line2D([0], [0], color="#54A24B", marker="s", linewidth=1.3, markersize=3.5, label="MCLMC median"),
        Patch(facecolor="#54A24B", alpha=0.16, label="16-84% where available"),
    ]
    fig.legend(handles=legend_items, loc="outside lower center", ncol=5, fontsize=8)
    selection_note = (
        f"rows selected by {METHOD_DISPLAY_LABELS.get(recovery_method, recovery_method)} recovery outcome"
        if recovery_method
        else "rows selected by NN-baseline rank"
    )
    fig.suptitle(
        title
        + f" ({selection_note} inside worst100; catalog flux is the reconstruction target)"
    )
    _savefig(fig, path)


def _plot_single_sed_panel(
    ax,
    group: pd.DataFrame,
    *,
    method: str,
    color: str,
    wavelength_map: dict[str, float],
    y_limits: tuple[float, float] | None,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    work = group.copy()
    work["wave"] = _effective_wavelengths(work, wavelength_map)
    work = work[np.isfinite(work["wave"])].sort_values("wave")
    if work.empty:
        ax.axis("off")
        return
    wave = pd.to_numeric(work["wave"], errors="coerce").to_numpy(float)
    obs = _numeric_column(work, "obs_flux_fnu_cgs").to_numpy(float)
    err = _numeric_column(work, "obs_err_fnu_cgs").to_numpy(float)
    model = _numeric_column(work, "model_flux_median").to_numpy(float)

    ax.plot(wave, obs, color="black", linewidth=0.9, alpha=0.55)
    ax.errorbar(
        wave,
        obs,
        yerr=err,
        fmt="o",
        color="black",
        ecolor="black",
        alpha=0.8,
        markersize=3.2,
        elinewidth=0.8,
        capsize=1.8,
    )
    ax.plot(wave, model, marker="s", color=color, linewidth=1.35, markersize=3.2)
    if {"model_flux_q16", "model_flux_q84"}.issubset(work.columns):
        q16 = _numeric_column(work, "model_flux_q16").to_numpy(float)
        q84 = _numeric_column(work, "model_flux_q84").to_numpy(float)
        finite = np.isfinite(wave) & np.isfinite(q16) & np.isfinite(q84)
        if np.any(finite):
            ax.fill_between(wave[finite], q16[finite], q84[finite], color=color, alpha=0.16, linewidth=0)

    residual = _numeric_column(work, "residual_sigma_median").to_numpy(float)
    finite_residual = residual[np.isfinite(residual)]
    if finite_residual.size:
        worst_band_index = int(np.nanargmax(np.abs(residual)))
        ax.scatter(
            [wave[worst_band_index]],
            [obs[worst_band_index]],
            s=34,
            facecolor="none",
            edgecolor="#E45756",
            linewidth=1.1,
            zorder=5,
        )
    ax.set_xscale("log")
    finite_flux = np.concatenate(
        [
            obs[np.isfinite(obs)],
            model[np.isfinite(model)],
            (obs + err)[np.isfinite(obs + err)],
            (obs - err)[np.isfinite(obs - err)],
        ]
    )
    if finite_flux.size and np.nanmin(finite_flux) > 0:
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=_flux_linthresh(finite_flux))
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if show_xlabel:
        ax.set_xlabel("effective wavelength [Angstrom]", fontsize=8)
    else:
        ax.set_xticklabels([])
    if not show_ylabel:
        ax.set_yticklabels([])
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=7)


def _select_sed_example_rows(
    frame: pd.DataFrame,
    *,
    baseline_method: str,
    recovery_method: str | None,
    n_per_group: int,
) -> list[dict[str, float | int | str]]:
    baseline = frame[frame["method"].astype(str) == baseline_method]
    obj = _object_summary(baseline).sort_values(
        "median_abs_residual_sigma",
        ascending=False,
    ).reset_index(drop=True)
    if obj.empty:
        return []
    rank_by_row = {int(row["row_index"]): int(idx + 1) for idx, row in obj.iterrows()}
    if recovery_method and recovery_method in set(frame["method"].astype(str)):
        choices = _select_sed_recovery_rows(
            frame,
            recovery_method=recovery_method,
            rank_by_row=rank_by_row,
            n_per_group=n_per_group,
        )
        if choices:
            return choices
    n = len(obj)
    groups = [
        ("worst NN", list(range(min(n_per_group, n)))),
        (
            "middle NN",
            _centered_indices(n, center=n // 2, width=n_per_group),
        ),
        ("least bad NN", list(range(max(0, n - n_per_group), n))),
    ]
    choices: list[dict[str, float | int | str]] = []
    seen: set[int] = set()
    for label, indices in groups:
        for idx in indices:
            if idx < 0 or idx >= n:
                continue
            row = obj.iloc[idx]
            row_index = int(row["row_index"])
            if row_index in seen:
                continue
            seen.add(row_index)
            choices.append(
                {
                    "group": label,
                    "rank": rank_by_row.get(row_index, int(idx + 1)),
                    "row_index": row_index,
                    "baseline_error": float(row["median_abs_residual_sigma"]),
                }
            )
    return choices


def _select_sed_recovery_rows(
    frame: pd.DataFrame,
    *,
    recovery_method: str,
    rank_by_row: dict[int, int],
    n_per_group: int,
) -> list[dict[str, float | int | str]]:
    _, paired_object, _ = _paired_against_baseline(frame)
    if paired_object.empty:
        return []
    paired = paired_object[paired_object["method"].astype(str) == recovery_method].copy()
    if paired.empty:
        return []
    paired["row_index"] = pd.to_numeric(paired["row_index"], errors="coerce")
    paired = paired[np.isfinite(paired["row_index"])]
    if paired.empty:
        return []
    method_rows = frame[frame["method"].astype(str) == recovery_method].copy()
    obs_flux = pd.to_numeric(method_rows["obs_flux_fnu_cgs"], errors="coerce")
    obs_err = pd.to_numeric(method_rows["obs_err_fnu_cgs"], errors="coerce")
    method_rows["obs_err_over_abs_flux"] = obs_err / obs_flux.abs().replace(0.0, np.nan)
    err_summary = method_rows.groupby("row_index", sort=False)["obs_err_over_abs_flux"].agg(
        max_err_over_abs_flux="max",
        median_err_over_abs_flux="median",
    ).reset_index()
    paired = paired.merge(err_summary, on="row_index", how="left")
    gain_sorted = paired.sort_values("object_improvement", ascending=False).reset_index(drop=True)
    huge_sorted = paired.sort_values("max_err_over_abs_flux", ascending=False).reset_index(drop=True)
    credible = paired[paired["max_err_over_abs_flux"] <= 10.0]
    if credible.empty:
        credible = paired[paired["max_err_over_abs_flux"] <= 100.0]
    credible_sorted = credible.sort_values("object_improvement", ascending=False).reset_index(drop=True)
    hard_sorted = paired.sort_values(
        "method_object_median_abs_residual_sigma",
        ascending=False,
    ).reset_index(drop=True)
    groups = [
        ("huge-error gain", huge_sorted.head(n_per_group)),
        (
            "credible MAP gain",
            credible_sorted.head(n_per_group)
            if not credible_sorted.empty
            else gain_sorted.iloc[
                _centered_indices(len(gain_sorted), center=len(gain_sorted) // 2, width=n_per_group)
            ],
        ),
        ("still hard after MAP", hard_sorted.head(n_per_group)),
    ]
    choices: list[dict[str, float | int | str]] = []
    seen: set[int] = set()
    for label, group in groups:
        for _, row in group.iterrows():
            row_index = int(row["row_index"])
            if row_index in seen:
                continue
            seen.add(row_index)
            choices.append(
                {
                    "group": label,
                    "rank": rank_by_row.get(row_index, -1),
                    "row_index": row_index,
                    "baseline_error": float(row["baseline_object_median_abs_residual_sigma"]),
                    "recovery_error": float(row["method_object_median_abs_residual_sigma"]),
                }
            )
    return choices


def _centered_indices(n: int, *, center: int, width: int) -> list[int]:
    if n <= 0 or width <= 0:
        return []
    start = max(0, center - width // 2)
    end = min(n, start + width)
    start = max(0, end - width)
    return list(range(start, end))


def _sed_flux_limits_for_row(
    frame: pd.DataFrame,
    row_index: int | float,
    methods: list[str],
) -> tuple[float, float] | None:
    subset = frame[
        frame["method"].astype(str).isin(methods)
        & (pd.to_numeric(frame["row_index"], errors="coerce") == float(row_index))
    ]
    if subset.empty:
        return None
    values = []
    for column in [
        "obs_flux_fnu_cgs",
        "model_flux_median",
        "model_flux_q16",
        "model_flux_q84",
    ]:
        if column in subset:
            values.append(pd.to_numeric(subset[column], errors="coerce").to_numpy(float))
    if not values:
        return None
    arr = np.concatenate(values)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    positive = arr[arr > 0]
    if positive.size == arr.size:
        lo = max(float(np.nanmin(positive)) * 0.65, np.nextafter(0, 1))
        hi = float(np.nanmax(positive)) * 1.55
        return lo, hi
    span = float(np.nanmax(np.abs(arr)))
    if not np.isfinite(span) or span <= 0:
        return None
    return -1.15 * span, 1.15 * span


def _object_metric_for_method(frame: pd.DataFrame, method: str, row_index: int | float) -> float:
    subset = frame[
        (frame["method"].astype(str) == method)
        & (pd.to_numeric(frame["row_index"], errors="coerce") == float(row_index))
    ]
    if subset.empty:
        return float("nan")
    values = _finite_array(subset.get("abs_residual_sigma_median"))
    if values.size == 0:
        return float("nan")
    return float(np.nanmedian(values))


def _finite_array(values) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _distribution_percentile(value: float, reference: np.ndarray) -> float:
    finite = reference[np.isfinite(reference)]
    if finite.size == 0 or not np.isfinite(value):
        return float("nan")
    return float(100.0 * np.mean(finite <= value))


def _flux_linthresh(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    finite = finite[finite > 0]
    if finite.size == 0:
        return 1.0
    return float(max(np.nanpercentile(finite, 5), np.nanmax(finite) * 1e-3))


def _abmag_from_fnu_cgs(flux_fnu_cgs: pd.Series | np.ndarray) -> pd.Series:
    index = flux_fnu_cgs.index if isinstance(flux_fnu_cgs, pd.Series) else None
    flux = pd.Series(pd.to_numeric(flux_fnu_cgs, errors="coerce"), index=index)
    out = pd.Series(np.nan, index=flux.index, dtype=float)
    positive = flux > 0
    out.loc[positive] = -2.5 * np.log10(flux.loc[positive].to_numpy(float) / 3.631e-20)
    return out


def _plot_same_object_error_distribution(frame: pd.DataFrame, path: Path, title: str) -> None:
    obj = _object_summary(frame)
    if obj.empty:
        return
    methods = obj["method"].drop_duplicates().astype(str).tolist()
    data = [
        obj.loc[obj["method"].astype(str) == method, "median_abs_residual_sigma"].dropna().to_numpy(float)
        for method in methods
    ]
    data = [d for d in data if d.size]
    labels = [method for method in methods if len(obj.loc[obj["method"].astype(str) == method])]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.05), 5.2), constrained_layout=True)
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(3, color="#E45756", linestyle="--", linewidth=0.9, label="3 sigma")
    ax.axhline(5, color="#B279A2", linestyle=":", linewidth=0.9, label="5 sigma")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel(f"object median |{RESIDUAL_LABEL}|")
    ax.set_title(f"{title}: same-object error distributions")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    _savefig(fig, path)


def _plot_same_band_error_distribution(frame: pd.DataFrame, path: Path, title: str) -> None:
    valid = _valid(frame)
    if valid.empty:
        return
    methods = valid["method"].drop_duplicates().astype(str).tolist()
    data = [
        valid.loc[valid["method"].astype(str) == method, "abs_residual_sigma_median"].dropna().to_numpy(float)
        for method in methods
    ]
    data = [d for d in data if d.size]
    labels = [method for method in methods if len(valid.loc[valid["method"].astype(str) == method])]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.05), 5.2), constrained_layout=True)
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(3, color="#E45756", linestyle="--", linewidth=0.9, label="3 sigma")
    ax.axhline(5, color="#B279A2", linestyle=":", linewidth=0.9, label="5 sigma")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel(f"band |{RESIDUAL_LABEL}|")
    ax.set_title(f"{title}: same-band-row error distributions")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    _savefig(fig, path)


def _plot_paired_improvement_summary(summary: pd.DataFrame, path: Path, title: str) -> None:
    if summary.empty:
        return
    labels = summary["method"].astype(str).tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(max(11, len(labels) * 1.15), 4.8), constrained_layout=True)
    axes[0].bar(x - 0.18, summary["median_object_improvement"], width=0.36, label="object median")
    axes[0].bar(x + 0.18, summary["median_band_improvement"], width=0.36, label="band rows")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel(f"baseline |error| - method |error|")
    axes[0].set_title("Median improvement")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].legend(frameon=False)

    axes[1].bar(x - 0.18, summary["frac_object_improved"], width=0.36, label="objects")
    axes[1].bar(x + 0.18, summary["frac_band_improved"], width=0.36, label="band rows")
    axes[1].axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("fraction improved vs baseline")
    axes[1].set_title("Paired improvement fraction")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].legend(frameon=False)
    fig.suptitle(f"{title}: paired comparison to reference_nn")
    _savefig(fig, path)


def _plot_paired_object_scatter(paired: pd.DataFrame, path: Path, title: str) -> None:
    if paired.empty:
        return
    methods = paired["method"].drop_duplicates().astype(str).tolist()
    _plot_paired_scatter_grid(
        paired,
        methods,
        path,
        title=f"{title}: object-level baseline vs method",
        x_col="baseline_object_median_abs_residual_sigma",
        y_col="method_object_median_abs_residual_sigma",
        improvement_col="object_improvement",
        xlabel=f"baseline object median |{RESIDUAL_LABEL}|",
        ylabel=f"method object median |{RESIDUAL_LABEL}|",
    )


def _plot_paired_band_scatter(paired: pd.DataFrame, path: Path, title: str) -> None:
    if paired.empty:
        return
    methods = paired["method"].drop_duplicates().astype(str).tolist()
    _plot_paired_scatter_grid(
        paired,
        methods,
        path,
        title=f"{title}: band-row baseline vs method",
        x_col="baseline_abs_residual_sigma",
        y_col="method_abs_residual_sigma",
        improvement_col="band_improvement",
        xlabel=f"baseline band |{RESIDUAL_LABEL}|",
        ylabel=f"method band |{RESIDUAL_LABEL}|",
    )


def _plot_paired_scatter_grid(
    paired: pd.DataFrame,
    methods: list[str],
    path: Path,
    *,
    title: str,
    x_col: str,
    y_col: str,
    improvement_col: str,
    xlabel: str,
    ylabel: str,
) -> None:
    ncols = 2
    nrows = math.ceil(len(methods) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(3.5, nrows * 3.4)), constrained_layout=True)
    axes_flat = np.ravel(np.asarray(axes))
    for ax, method in zip(axes_flat, methods):
        group = paired[paired["method"].astype(str) == method].copy()
        if len(group) > 5000:
            group = group.sample(5000, random_state=42)
        x = pd.to_numeric(group[x_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(group[y_col], errors="coerce").to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size == 0:
            ax.axis("off")
            continue
        ax.scatter(x, y, s=10, alpha=0.35, color="#4C78A8")
        lim = np.nanpercentile(np.concatenate([x, y]), 98)
        lim = max(lim, 5.0)
        ax.plot([0, lim], [0, lim], color="black", linewidth=0.8)
        ax.axhline(3, color="#E45756", linestyle="--", linewidth=0.8)
        ax.axvline(3, color="#E45756", linestyle="--", linewidth=0.8)
        ax.set_xlim(left=0, right=lim)
        ax.set_ylim(bottom=0, top=lim)
        frac = _frac(pd.to_numeric(group[improvement_col], errors="coerce").to_numpy(float) > 0)
        ax.set_title(f"{method}  improved={frac:.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    for ax in axes_flat[len(methods) :]:
        ax.axis("off")
    fig.suptitle(title)
    _savefig(fig, path)


def _plot_paired_improvement_histograms(
    paired: pd.DataFrame,
    improvement_col: str,
    path: Path,
    title: str,
) -> None:
    if paired.empty or improvement_col not in paired:
        return
    methods = paired["method"].drop_duplicates().astype(str).tolist()
    ncols = 2
    nrows = math.ceil(len(methods) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(3.0, nrows * 2.8)), constrained_layout=True)
    axes_flat = np.ravel(np.asarray(axes))
    for ax, method in zip(axes_flat, methods):
        values = pd.to_numeric(
            paired.loc[paired["method"].astype(str) == method, improvement_col],
            errors="coerce",
        ).dropna().to_numpy(float)
        if values.size == 0:
            ax.axis("off")
            continue
        clipped = np.clip(values, -50, 50)
        ax.hist(clipped, bins=60, color="#4C78A8", alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_title(f"{method} median={np.nanmedian(values):.2f}")
        ax.set_xlabel("positive = lower absolute photometric error than baseline")
    for ax in axes_flat[len(methods) :]:
        ax.axis("off")
    fig.suptitle(title)
    _savefig(fig, path)


def _plot_residual_hist(frame: pd.DataFrame, path: Path, title: str) -> None:
    valid = _valid(frame)
    values = valid["residual_sigma_median"].dropna().to_numpy(float)
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    clipped = _clip_for_display(values)
    ax.hist(clipped, bins=80, color="#4C78A8", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(-3, color="#E45756", linestyle="--", linewidth=0.9)
    ax.axvline(3, color="#E45756", linestyle="--", linewidth=0.9)
    ax.set_xlabel(RESIDUAL_LABEL)
    ax.set_ylabel("band rows")
    ax.set_title(title)
    _savefig(fig, path)


def _plot_boxplot_by_band(frame: pd.DataFrame, path: Path, title: str) -> None:
    valid = _valid(frame)
    bands = valid["band"].drop_duplicates().astype(str).tolist()
    data = [
        _clip_for_display(valid.loc[valid["band"].astype(str) == band, "residual_sigma_median"].dropna().to_numpy(float))
        for band in bands
    ]
    data = [d for d in data if d.size]
    labels = [band for band in bands if len(valid.loc[valid["band"].astype(str) == band])]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 5.2), constrained_layout=True)
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(-3, color="#E45756", linestyle="--", linewidth=0.9)
    ax.axhline(3, color="#E45756", linestyle="--", linewidth=0.9)
    ax.set_ylabel(f"{RESIDUAL_LABEL}, clipped for display")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    _savefig(fig, path)


def _plot_boxplot_by_method_and_band(frame: pd.DataFrame, path: Path, title: str) -> None:
    valid = _valid(frame)
    methods = valid["method"].drop_duplicates().astype(str).tolist()
    if not methods:
        return
    nrows = len(methods)
    fig, axes = plt.subplots(nrows, 1, figsize=(max(11, len(valid["band"].unique()) * 0.55), max(3.0, nrows * 2.2)), sharex=True, constrained_layout=True)
    axes_flat = np.ravel(np.asarray(axes))
    for ax, method in zip(axes_flat, methods):
        group = valid[valid["method"].astype(str) == method]
        bands = group["band"].drop_duplicates().astype(str).tolist()
        data = [
            _clip_for_display(group.loc[group["band"].astype(str) == band, "residual_sigma_median"].dropna().to_numpy(float))
            for band in bands
        ]
        ax.boxplot(data, tick_labels=bands, showfliers=False)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.axhline(-3, color="#E45756", linestyle="--", linewidth=0.8)
        ax.axhline(3, color="#E45756", linestyle="--", linewidth=0.8)
        ax.set_ylabel(method)
    fig.supylabel(f"{RESIDUAL_LABEL}, clipped for display")
    axes_flat[-1].tick_params(axis="x", rotation=45)
    fig.suptitle(title)
    _savefig(fig, path)


def _plot_hist_by_band(frame: pd.DataFrame, path: Path, title: str) -> None:
    valid = _valid(frame)
    bands = valid["band"].drop_duplicates().astype(str).tolist()
    if not bands:
        return
    ncols = 4
    nrows = math.ceil(len(bands) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(3.0, nrows * 2.6)), constrained_layout=True)
    axes_flat = np.ravel(np.asarray(axes))
    for ax, band in zip(axes_flat, bands):
        values = valid.loc[valid["band"].astype(str) == band, "residual_sigma_median"].dropna().to_numpy(float)
        ax.hist(_clip_for_display(values), bins=40, color="#4C78A8", alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.axvline(-3, color="#E45756", linestyle="--", linewidth=0.8)
        ax.axvline(3, color="#E45756", linestyle="--", linewidth=0.8)
        ax.set_title(band)
        ax.set_xlabel(RESIDUAL_LABEL)
    for ax in axes_flat[len(bands) :]:
        ax.axis("off")
    fig.suptitle(title)
    _savefig(fig, path)


def _plot_observed_vs_model_by_band(frame: pd.DataFrame, path: Path, title: str) -> None:
    valid = _valid(frame).copy()
    if "obs_flux_fnu_cgs" not in valid or "model_flux_median" not in valid:
        return
    valid["obs_flux_fnu_cgs"] = pd.to_numeric(valid["obs_flux_fnu_cgs"], errors="coerce")
    valid["model_flux_median"] = pd.to_numeric(valid["model_flux_median"], errors="coerce")
    valid = valid[np.isfinite(valid["obs_flux_fnu_cgs"]) & np.isfinite(valid["model_flux_median"])]
    bands = valid["band"].drop_duplicates().astype(str).tolist()
    if not bands:
        return
    ncols = 4
    nrows = math.ceil(len(bands) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(3.0, nrows * 2.8)), constrained_layout=True)
    axes_flat = np.ravel(np.asarray(axes))
    for ax, band in zip(axes_flat, bands):
        group = valid[valid["band"].astype(str) == band]
        if len(group) > 1000:
            group = group.sample(1000, random_state=42)
        x = group["obs_flux_fnu_cgs"].to_numpy(float)
        y = group["model_flux_median"].to_numpy(float)
        ax.scatter(x, y, s=8, alpha=0.45)
        finite = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
        finite = finite[finite > 0]
        if finite.size:
            lo, hi = np.nanpercentile(finite, [1, 99])
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_title(band)
    for ax in axes_flat[len(bands) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.supxlabel(r"observed input flux $F_{\rm in}$ [fnu cgs]")
    fig.supylabel(r"DSPS fitted/output flux $F_{\rm out}$ [fnu cgs]")
    _savefig(fig, path)


def _plot_top_objects(frame: pd.DataFrame, path: Path, title: str) -> None:
    if frame.empty:
        return
    work = frame.head(30).copy()
    labels = work["row_index"].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.barh(np.arange(len(work)), work["median_abs_residual_sigma"], color="#4C78A8")
    ax.set_yticks(np.arange(len(work)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"object median |{RESIDUAL_LABEL}|")
    ax.set_ylabel("row_index")
    ax.set_title(title)
    _savefig(fig, path)


def _plot_parameter_distributions(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    truth: pd.DataFrame | None = None,
    prior_bounds: dict[str, tuple[float, float]] | None = None,
    sample_label: str = "samples",
    truth_label: str = "truth",
    sample_color: str = "#4C78A8",
) -> None:
    cols = [c for c in columns if c in frame] if columns is not None else [c for c in PARAMETER_ORDER if c in frame]
    if not cols:
        cols = list(frame.select_dtypes(include="number").columns[:8])
    if not cols:
        return
    ncols = 4
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(3, nrows * 2.6)), constrained_layout=True)
    axes_flat = np.ravel(np.asarray(axes))
    for ax, col in zip(axes_flat, cols):
        values = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy(float)
        if prior_bounds and col in prior_bounds:
            lo, hi = prior_bounds[col]
            ax.axvspan(lo, hi, color="#BAB0AC", alpha=0.18, label="flat prior support")
            ax.axvline(lo, color="#666666", linestyle=":", linewidth=1.0)
            ax.axvline(hi, color="#666666", linestyle=":", linewidth=1.0)
        ax.hist(values, bins=35, color=sample_color, alpha=0.75, label=sample_label)
        if truth is not None and col in truth:
            truth_values = pd.to_numeric(truth[col], errors="coerce").dropna().to_numpy(float)
            if truth_values.size:
                ax.hist(
                    truth_values,
                    bins=35,
                    histtype="step",
                    color="#F58518",
                    linewidth=1.6,
                    label=truth_label,
                )
        ax.set_title(_parameter_label(col))
        ax.set_xlabel(_parameter_label(col))
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    for ax in axes_flat[len(cols) :]:
        ax.axis("off")
    fig.suptitle(title)
    _savefig(fig, path)


def _plot_corner_like(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    *,
    truth: pd.DataFrame | None = None,
    prior_bounds: dict[str, tuple[float, float]] | None = None,
    sample_label: str = "samples",
    truth_label: str = "truth",
) -> None:
    cols = [c for c in PARAMETER_ORDER if c in frame]
    if len(cols) < 2:
        cols = list(frame.select_dtypes(include="number").columns[:8])
    if len(cols) < 2:
        return
    work = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if work.empty:
        return
    if len(work) > 3000:
        work = work.sample(3000, random_state=42)
    truth_work = pd.DataFrame()
    if truth is not None:
        truth_cols = [col for col in cols if col in truth]
        if truth_cols:
            truth_work = truth[truth_cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")
            if len(truth_work) > 3000:
                truth_work = truth_work.sample(3000, random_state=43)
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(max(9, n * 1.8), max(9, n * 1.8)), constrained_layout=True)
    for i, ycol in enumerate(cols):
        for j, xcol in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                if prior_bounds and xcol in prior_bounds:
                    lo, hi = prior_bounds[xcol]
                    ax.axvspan(lo, hi, color="#BAB0AC", alpha=0.16)
                    ax.axvline(lo, color="#666666", linestyle=":", linewidth=0.8)
                    ax.axvline(hi, color="#666666", linestyle=":", linewidth=0.8)
                ax.hist(work[xcol].to_numpy(float), bins=30, color="#4C78A8", alpha=0.75)
                if xcol in truth_work:
                    ax.hist(
                        truth_work[xcol].dropna().to_numpy(float),
                        bins=30,
                        histtype="step",
                        color="#F58518",
                        linewidth=1.3,
                    )
            elif i > j:
                ax.scatter(work[xcol], work[ycol], s=4, alpha=0.28, color="#4C78A8")
                if xcol in truth_work and ycol in truth_work:
                    ax.scatter(
                        truth_work[xcol],
                        truth_work[ycol],
                        s=10,
                        alpha=0.55,
                        color="#F58518",
                    )
                if prior_bounds:
                    if xcol in prior_bounds:
                        lo, hi = prior_bounds[xcol]
                        ax.axvline(lo, color="#666666", linestyle=":", linewidth=0.6)
                        ax.axvline(hi, color="#666666", linestyle=":", linewidth=0.6)
                    if ycol in prior_bounds:
                        lo, hi = prior_bounds[ycol]
                        ax.axhline(lo, color="#666666", linestyle=":", linewidth=0.6)
                        ax.axhline(hi, color="#666666", linestyle=":", linewidth=0.6)
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(xcol, fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(ycol, fontsize=8)
            elif j != 0:
                ax.set_yticklabels([])
    fig.suptitle(title)
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C78A8", markersize=6, label=sample_label),
    ]
    if not truth_work.empty:
        legend_items.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#F58518", markersize=6, label=truth_label)
        )
    if prior_bounds:
        legend_items.append(Line2D([0], [0], color="#666666", linestyle=":", label="flat prior bounds"))
    fig.legend(handles=legend_items, loc="outside lower center", ncol=len(legend_items), fontsize=9)
    _savefig(fig, path)


def _available_parameters(frame: pd.DataFrame, preferred: list[str] | tuple[str, ...]) -> list[str]:
    return [name for name in preferred if name in frame]


def _truth_available_parameters(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    preferred: list[str] | tuple[str, ...],
) -> list[str]:
    cols = []
    for name in preferred:
        if name not in frame or name not in truth:
            continue
        values = pd.to_numeric(truth[name], errors="coerce").to_numpy(float)
        if np.isfinite(values).any():
            cols.append(name)
    return cols


def _finite_numeric_columns(
    frame: pd.DataFrame,
    preferred: list[str] | tuple[str, ...],
) -> list[str]:
    cols = []
    for name in preferred:
        if name not in frame:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        if np.isfinite(values).any():
            cols.append(name)
    return cols


def _mclmc_truth_corner_columns(frame: pd.DataFrame, truth: pd.DataFrame) -> list[str]:
    cols = _truth_available_parameters(frame, truth, MCLMC_PROJECTED_TRUTH_PARAMETERS)
    if len(cols) >= 2:
        return cols
    return _available_parameters(frame, MCLMC_CORE_CORNER_PARAMETERS)


def _parameter_label(name: str) -> str:
    return PARAMETER_LABELS.get(name, name)


def _plot_corner_density(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    *,
    columns: list[str] | tuple[str, ...],
    truth: pd.DataFrame | None = None,
    prior_bounds: dict[str, tuple[float, float]] | None = None,
    point_frame: pd.DataFrame | None = None,
    sample_label: str = "posterior density",
    truth_label: str = "truth density",
    point_label: str = "points",
    use_prior_ranges: bool = True,
) -> None:
    cols = [col for col in columns if col in frame]
    if len(cols) < 2:
        return
    work = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if work.empty:
        return
    truth_work = pd.DataFrame()
    if truth is not None:
        truth_cols = [col for col in cols if col in truth]
        if truth_cols:
            truth_work = (
                truth[truth_cols]
                .apply(pd.to_numeric, errors="coerce")
                .dropna(how="all")
            )
    truth_as_points = not truth_work.empty and len(truth_work) < 20
    point_work = pd.DataFrame()
    if point_frame is not None:
        point_cols = [col for col in cols if col in point_frame]
        if point_cols:
            point_work = point_frame[point_cols].apply(pd.to_numeric, errors="coerce").dropna()
    ranges = _corner_axis_ranges(
        cols,
        [work, truth_work, point_work],
        prior_bounds,
        use_prior_ranges=use_prior_ranges,
        extrema_frames=[truth_work] if truth_as_points else None,
    )
    draw_prior_bounds = prior_bounds if use_prior_ranges else None
    n = len(cols)
    size = max(7.5, 2.0 * n)
    fig, axes = plt.subplots(n, n, figsize=(size, size), constrained_layout=True)
    axes = np.asarray(axes)
    for i, ycol in enumerate(cols):
        for j, xcol in enumerate(cols):
            ax = axes[i, j]
            x_range = ranges[xcol]
            if i == j:
                _draw_prior_1d(ax, xcol, x_range, draw_prior_bounds)
                _plot_corner_hist(ax, work[xcol], x_range, "#4C78A8", alpha=0.72)
                if xcol in truth_work:
                    truth_values = pd.to_numeric(truth_work[xcol], errors="coerce").dropna().to_numpy(float)
                    if truth_as_points:
                        for value in truth_values:
                            ax.axvline(value, color="#F58518", linestyle="--", linewidth=1.4, alpha=0.95)
                    else:
                        _plot_corner_hist(
                            ax,
                            pd.Series(truth_values),
                            x_range,
                            "#F58518",
                            alpha=1.0,
                            histtype="step",
                            linewidth=1.5,
                        )
                        truth_values = truth_values[
                            (truth_values >= x_range[0]) & (truth_values <= x_range[1])
                        ]
                        if truth_values.size:
                            for value in truth_values:
                                ax.axvline(
                                    float(value),
                                    color="#F58518",
                                    linewidth=0.45,
                                    alpha=0.12,
                                    zorder=4,
                                )
                            ax.axvline(
                                float(np.nanmedian(truth_values)),
                                color="#F58518",
                                linestyle="--",
                                linewidth=1.2,
                                alpha=0.95,
                            )
                            ax.plot(
                                truth_values,
                                np.zeros_like(truth_values),
                                linestyle="None",
                                marker="|",
                                markersize=5,
                                markeredgewidth=0.8,
                                color="#F58518",
                                alpha=0.85,
                                zorder=5,
                            )
                if xcol in point_work:
                    for value in point_work[xcol].dropna().to_numpy(float):
                        ax.axvline(value, color="#222222", alpha=0.18, linewidth=0.8)
                ax.set_xlim(*x_range)
                ax.set_yticks([])
            elif i > j:
                y_range = ranges[ycol]
                _plot_density_2d(
                    ax,
                    work[xcol],
                    work[ycol],
                    x_range,
                    y_range,
                    color="#4C78A8",
                    filled=True,
                    linewidth=1.0,
                )
                if xcol in truth_work and ycol in truth_work:
                    if truth_as_points:
                        truth_pair = truth_work[[xcol, ycol]].dropna()
                        for x_value, y_value in truth_pair.to_numpy(float):
                            ax.axvline(x_value, color="#F58518", linestyle="--", linewidth=0.8, alpha=0.35)
                            ax.axhline(y_value, color="#F58518", linestyle="--", linewidth=0.8, alpha=0.35)
                        ax.scatter(
                            truth_pair[xcol],
                            truth_pair[ycol],
                            marker="*",
                            s=46,
                            color="#F58518",
                            edgecolor="white",
                            linewidths=0.35,
                            zorder=5,
                        )
                    else:
                        _plot_density_2d(
                            ax,
                            truth_work[xcol],
                            truth_work[ycol],
                            x_range,
                            y_range,
                            color="#F58518",
                            filled=False,
                            linewidth=1.4,
                            linestyle="--",
                        )
                        truth_pair = truth_work[[xcol, ycol]].dropna()
                        truth_pair = truth_pair[
                            truth_pair[xcol].between(x_range[0], x_range[1])
                            & truth_pair[ycol].between(y_range[0], y_range[1])
                        ]
                        if not truth_pair.empty:
                            ax.scatter(
                                truth_pair[xcol],
                                truth_pair[ycol],
                                marker="*",
                                s=18,
                                facecolor="#F58518",
                                edgecolor="#7A3E00",
                                linewidths=0.25,
                                alpha=0.85,
                                zorder=6,
                            )
                if xcol in point_work and ycol in point_work:
                    ax.scatter(
                        point_work[xcol],
                        point_work[ycol],
                        marker="x",
                        s=14,
                        color="#222222",
                        alpha=0.55,
                        linewidths=0.8,
                    )
                _draw_prior_2d(ax, xcol, ycol, draw_prior_bounds)
                ax.set_xlim(*x_range)
                ax.set_ylim(*y_range)
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(_parameter_label(xcol), fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(_parameter_label(ycol), fontsize=8)
            elif j != 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color="#4C78A8", linewidth=2.0, label=sample_label),
    ]
    if not truth_work.empty:
        if truth_as_points:
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    color="#F58518",
                    linestyle="--",
                    marker="*",
                    markersize=8,
                    linewidth=1.5,
                    label=truth_label,
                )
            )
        else:
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    color="#F58518",
                    linestyle="--",
                    marker="*",
                    markerfacecolor="#F58518",
                    markeredgecolor="#7A3E00",
                    markersize=7,
                    linewidth=2.0,
                    label=truth_label,
                )
            )
    if not point_work.empty:
        legend_items.append(
            Line2D([0], [0], marker="x", color="#222222", linestyle="None", label=point_label)
        )
    if draw_prior_bounds:
        legend_items.append(
            Line2D([0], [0], color="#666666", linestyle=":", linewidth=1.0, label="flat prior bounds")
        )
    fig.suptitle(title)
    fig.legend(handles=legend_items, loc="outside lower center", ncol=min(len(legend_items), 4), fontsize=8)
    _savefig(fig, path)


def _plot_corner_hist(
    ax,
    values: pd.Series,
    value_range: tuple[float, float],
    color: str,
    *,
    alpha: float,
    histtype: str = "bar",
    linewidth: float = 1.0,
) -> None:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if arr.size == 0:
        return
    arr = arr[(arr >= value_range[0]) & (arr <= value_range[1])]
    if arr.size == 0:
        return
    bins = np.linspace(value_range[0], value_range[1], 36)
    kwargs = {"bins": bins, "density": True, "color": color, "alpha": alpha}
    if histtype != "bar":
        kwargs.update({"histtype": histtype, "linewidth": linewidth})
    ax.hist(arr, **kwargs)


def _plot_density_2d(
    ax,
    x_values: pd.Series,
    y_values: pd.Series,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    color: str,
    filled: bool,
    linewidth: float,
    linestyle: str = "-",
) -> None:
    x = pd.to_numeric(x_values, errors="coerce").to_numpy(float)
    y = pd.to_numeric(y_values, errors="coerce").to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return
    if x.size < 30 or np.nanstd(x) <= 0.0 or np.nanstd(y) <= 0.0:
        ax.scatter(x, y, s=8 if x.size < 300 else 3, alpha=0.45, color=color)
        return
    hist, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=42,
        range=[x_range, y_range],
    )
    hist = _smooth_hist2d(hist)
    if not np.isfinite(hist).any() or float(np.nanmax(hist)) <= 0.0:
        ax.scatter(x, y, s=3, alpha=0.35, color=color)
        return
    levels = _density_contour_levels(hist, masses=(0.90, 0.50))
    if not levels:
        ax.scatter(x, y, s=3, alpha=0.35, color=color)
        return
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    grid_x, grid_y = np.meshgrid(xcenters, ycenters, indexing="ij")
    max_level = float(np.nanmax(hist))
    if filled and len(levels) >= 1:
        fill_levels = [levels[0], max_level * 1.000001]
        ax.contourf(
            grid_x,
            grid_y,
            hist,
            levels=fill_levels,
            colors=[color],
            alpha=0.13,
        )
    ax.contour(
        grid_x,
        grid_y,
        hist,
        levels=levels,
        colors=[color],
        linewidths=linewidth,
        linestyles=linestyle,
    )


def _density_contour_levels(hist: np.ndarray, *, masses: tuple[float, ...]) -> list[float]:
    flat = np.asarray(hist, dtype=float).ravel()
    flat = flat[np.isfinite(flat) & (flat > 0.0)]
    if flat.size == 0:
        return []
    flat = np.sort(flat)[::-1]
    cdf = np.cumsum(flat)
    cdf = cdf / cdf[-1]
    levels = []
    for mass in masses:
        idx = int(np.searchsorted(cdf, float(mass), side="left"))
        idx = min(max(idx, 0), flat.size - 1)
        levels.append(float(flat[idx]))
    max_value = float(flat[0])
    unique = sorted({level for level in levels if 0.0 < level < max_value})
    return unique


def _smooth_hist2d(hist: np.ndarray) -> np.ndarray:
    padded = np.pad(hist, 1, mode="edge")
    return (
        4.0 * padded[1:-1, 1:-1]
        + 2.0
        * (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )
        + padded[:-2, :-2]
        + padded[:-2, 2:]
        + padded[2:, :-2]
        + padded[2:, 2:]
    ) / 16.0


def _corner_axis_ranges(
    cols: list[str],
    frames: list[pd.DataFrame],
    prior_bounds: dict[str, tuple[float, float]] | None,
    *,
    use_prior_ranges: bool = True,
    extrema_frames: list[pd.DataFrame] | None = None,
) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for col in cols:
        if use_prior_ranges and prior_bounds and col in prior_bounds:
            lo, hi = prior_bounds[col]
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ranges[col] = (float(lo), float(hi))
                continue
        values = []
        for frame in frames:
            if col not in frame:
                continue
            arr = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy(float)
            if arr.size:
                values.append(arr)
        if not values:
            ranges[col] = (0.0, 1.0)
            continue
        joined = np.concatenate(values)
        joined = joined[np.isfinite(joined)]
        if joined.size == 0:
            ranges[col] = (0.0, 1.0)
            continue
        lo, hi = np.nanpercentile(joined, [0.5, 99.5])
        extrema = []
        for frame in extrema_frames or []:
            if col not in frame:
                continue
            arr = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy(float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                extrema.append(arr)
        if extrema:
            exact = np.concatenate(extrema)
            lo = min(float(lo), float(np.nanmin(exact)))
            hi = max(float(hi), float(np.nanmax(exact)))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            center = float(np.nanmedian(joined))
            width = max(abs(center) * 0.05, 1.0e-3)
            lo, hi = center - width, center + width
        pad = 0.06 * (hi - lo)
        ranges[col] = (float(lo - pad), float(hi + pad))
    return ranges


def _draw_prior_1d(
    ax,
    col: str,
    value_range: tuple[float, float],
    prior_bounds: dict[str, tuple[float, float]] | None,
) -> None:
    if not prior_bounds or col not in prior_bounds:
        return
    lo, hi = prior_bounds[col]
    ax.axvspan(max(lo, value_range[0]), min(hi, value_range[1]), color="#BAB0AC", alpha=0.13)
    ax.axvline(lo, color="#666666", linestyle=":", linewidth=0.8)
    ax.axvline(hi, color="#666666", linestyle=":", linewidth=0.8)


def _draw_prior_2d(
    ax,
    xcol: str,
    ycol: str,
    prior_bounds: dict[str, tuple[float, float]] | None,
) -> None:
    if not prior_bounds:
        return
    if xcol in prior_bounds:
        lo, hi = prior_bounds[xcol]
        ax.axvline(lo, color="#666666", linestyle=":", linewidth=0.6)
        ax.axvline(hi, color="#666666", linestyle=":", linewidth=0.6)
    if ycol in prior_bounds:
        lo, hi = prior_bounds[ycol]
        ax.axhline(lo, color="#666666", linestyle=":", linewidth=0.6)
        ax.axhline(hi, color="#666666", linestyle=":", linewidth=0.6)


def _plot_parameter_vs_score(frame: pd.DataFrame, score: str, path: Path, title: str) -> None:
    cols = [c for c in PARAMETER_ORDER if c in frame]
    if not cols or score not in frame:
        return
    ncols = 4
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(3, nrows * 2.8)), constrained_layout=True)
    axes_flat = np.ravel(np.asarray(axes))
    y = pd.to_numeric(frame[score], errors="coerce")
    for ax, col in zip(axes_flat, cols):
        x = pd.to_numeric(frame[col], errors="coerce")
        ax.scatter(x, y, s=12, alpha=0.55)
        ax.set_xlabel(col)
        ax.set_ylabel(score)
        ax.set_yscale("symlog", linthresh=1)
    for ax in axes_flat[len(cols) :]:
        ax.axis("off")
    fig.suptitle(title)
    _savefig(fig, path)


def _plot_mclmc_individual_corners(
    samples: pd.DataFrame,
    residuals: pd.DataFrame,
    out_dir: Path,
    *,
    truth: pd.DataFrame,
    prior_bounds: dict[str, tuple[float, float]],
) -> None:
    _ensure(out_dir)
    obj = _object_summary(residuals).sort_values("median_abs_residual_sigma")
    if obj.empty:
        return
    metric = pd.to_numeric(obj["median_abs_residual_sigma"], errors="coerce")
    median_metric = float(np.nanmedian(metric.to_numpy(float)))
    average_idx = (metric - median_metric).abs().idxmin()
    candidates = [
        ("good", obj.iloc[0]),
        ("average", obj.loc[average_idx]),
        ("bad", obj.iloc[-1]),
    ]
    legacy_aliases = {
        "good": "best",
        "average": "median",
        "bad": "worst",
    }
    cols = _mclmc_truth_corner_columns(samples, truth)
    if len(cols) < 2:
        return
    seen: set[int] = set()
    for label, row in candidates:
        row_index = int(row["row_index"])
        if row_index in seen:
            continue
        seen.add(row_index)
        subset = samples[samples["row_index"] == row_index]
        if subset.empty:
            continue
        truth_subset = truth[truth["row_index"] == row_index] if "row_index" in truth else pd.DataFrame()
        score = float(row["median_abs_residual_sigma"])
        row_path = out_dir / f"mclmc_corner_{label}_row_{int(row_index)}.png"
        _plot_corner_density(
            subset,
            row_path,
            f"MCLMC {label} row {row_index}: median abs residual {score:.3g} sigma",
            columns=cols,
            truth=truth_subset,
            prior_bounds=prior_bounds,
            sample_label=f"posterior density row {row_index}",
            truth_label="catalog/projected truth",
            use_prior_ranges=False,
        )
        if row_path.exists():
            shutil.copy2(row_path, out_dir / f"mclmc_corner_{label}.png")
            legacy_label = legacy_aliases.get(label)
            if legacy_label is not None:
                shutil.copy2(row_path, out_dir / f"mclmc_corner_{legacy_label}_row_{row_index}.png")


def _plot_mclmc_diagnostics(diagnostics: pd.DataFrame, path: Path) -> None:
    metrics = [
        "fraction_nonans",
        "warmup_fraction_nonans",
        "mean_abs_energy_change",
        "p95_abs_energy_change",
        "aggregate_galaxy_samples_per_second",
    ]
    metrics = [m for m in metrics if m in diagnostics]
    if not metrics:
        return
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, max(3, len(metrics) * 2.0)), constrained_layout=True)
    axes_flat = np.ravel(np.asarray(axes))
    for ax, metric in zip(axes_flat, metrics):
        values = pd.to_numeric(diagnostics[metric], errors="coerce").dropna().to_numpy(float)
        ax.hist(values, bins=30, color="#4C78A8", alpha=0.85)
        ax.set_title(metric)
    fig.suptitle("MCLMC diagnostics")
    _savefig(fig, path)


def _plot_photometric_sed_triplet(
    residuals: pd.DataFrame,
    path: Path,
    title: str,
    *,
    wavelength_map: dict[str, float],
) -> None:
    obj = _object_summary(residuals).sort_values("median_abs_residual_sigma")
    if obj.empty:
        return
    choices = [
        ("best", obj.iloc[0]["row_index"]),
        ("median", obj.iloc[len(obj) // 2]["row_index"]),
        ("worst", obj.iloc[-1]["row_index"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False, constrained_layout=True)
    for ax, (label, row_index) in zip(axes, choices):
        group = residuals[residuals["row_index"] == row_index].copy()
        if group.empty:
            ax.axis("off")
            continue
        group["wave"] = _effective_wavelengths(group, wavelength_map)
        group = group[np.isfinite(group["wave"])].sort_values("wave")
        obs = _numeric_column(group, "obs_flux_fnu_cgs")
        err = _numeric_column(group, "obs_err_fnu_cgs")
        model = _numeric_column(group, "model_flux_median")
        ax.errorbar(
            group["wave"],
            obs,
            yerr=err,
            fmt="o",
            color="#4C78A8",
            markersize=4,
            capsize=2,
            label=r"input photometry $F_{\rm in}$",
        )
        ax.plot(
            group["wave"],
            model,
            marker="s",
            color="#F58518",
            linewidth=1.5,
            markersize=3.5,
            label=r"DSPS fitted $F_{\rm out}$",
        )
        if {"model_flux_q16", "model_flux_q84"}.issubset(group.columns):
            q16 = _numeric_column(group, "model_flux_q16")
            q84 = _numeric_column(group, "model_flux_q84")
            ax.fill_between(group["wave"], q16, q84, color="#F58518", alpha=0.18, label="posterior 16-84%" if label == "best" else None)
        ax.set_xscale("log")
        if np.all(obs > 0) and np.all(model > 0):
            ax.set_yscale("log")
        ax.set_title(f"{label} row {int(row_index)}")
        ax.set_xlabel("effective wavelength [Angstrom]")
        ax.grid(alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel(r"flux density $f_\nu$ [cgs]")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title + " (photometric points; full spectra not present in synced outputs)")
    _savefig(fig, path)


def _plot_training_metric_grid(epoch: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("loss", "loss"),
        ("negative_loglike", "negative loglike"),
        ("kl_mc_mean", "MC KL"),
        ("residual_rms", "residual RMS"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes_flat = np.ravel(axes)
    for ax, (metric, label) in zip(axes_flat, metrics):
        if metric not in epoch:
            ax.axis("off")
            continue
        for run, group in epoch.groupby("label", sort=False):
            for split, style in [("train", "-"), ("validation", "--")]:
                sub = group[group["split"].astype(str) == split]
                if sub.empty:
                    continue
                ax.plot(sub["epoch"], sub[metric], linestyle=style, label=f"{run} {split}", linewidth=1.5)
        ax.set_title(label)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle("NN training curves")
    _savefig(fig, path)


def _plot_training_metric(epoch: pd.DataFrame, metric: str, path: Path) -> None:
    if metric not in epoch:
        return
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    for run, group in epoch.groupby("label", sort=False):
        for split, style in [("train", "-"), ("validation", "--")]:
            sub = group[group["split"].astype(str) == split]
            if sub.empty:
                continue
            ax.plot(sub["epoch"], sub[metric], linestyle=style, label=f"{run} {split}", linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    _savefig(fig, path)


def _plot_batch_gradient_summary(log: pd.DataFrame, path: Path) -> None:
    metrics = [m for m in ["encoder_grad_norm", "prior_grad_norm", "band_alpha_grad_norm", "joint_grad_norm"] if m in log]
    if not metrics:
        return
    grouped = log.groupby(["label", "epoch"], sort=False)[metrics].median().reset_index()
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, max(3, len(metrics) * 2.4)), sharex=True, constrained_layout=True)
    axes_flat = np.ravel(np.asarray(axes))
    for ax, metric in zip(axes_flat, metrics):
        for label, group in grouped.groupby("label", sort=False):
            ax.plot(group["epoch"], group[metric], label=label, linewidth=1.6)
        ax.set_ylabel(metric)
        ax.set_yscale("symlog", linthresh=1)
        ax.grid(alpha=0.25)
    axes_flat[-1].set_xlabel("epoch")
    axes_flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Median batch gradient norms")
    _savefig(fig, path)


def _plot_batch_update_finiteness(log: pd.DataFrame, path: Path) -> None:
    metrics = [m for m in ["loss_finite", "grads_finite", "update_applied"] if m in log]
    if not metrics:
        return
    grouped = log.groupby(["label", "epoch"], sort=False)[metrics].mean().reset_index()
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, max(3, len(metrics) * 2.2)), sharex=True, constrained_layout=True)
    axes_flat = np.ravel(np.asarray(axes))
    for ax, metric in zip(axes_flat, metrics):
        for label, group in grouped.groupby("label", sort=False):
            ax.plot(group["epoch"], group[metric], label=label, linewidth=1.6)
        ax.set_ylabel(metric)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
    axes_flat[-1].set_xlabel("epoch")
    axes_flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Training update health")
    _savefig(fig, path)


def _plot_validation_redshift(frame: pd.DataFrame, path: Path) -> None:
    metric = "posterior_predictive_chi2" if "posterior_predictive_chi2" in frame else "negative_loglike"
    latest = frame.sort_values("epoch").groupby(["label", "z_bin"], sort=False).tail(1)
    if latest.empty:
        return
    pivot = latest.pivot(index="label", columns="z_bin", values=metric)
    fig, ax = plt.subplots(figsize=(max(9, pivot.shape[1] * 0.7), max(4, pivot.shape[0] * 0.6)), constrained_layout=True)
    values = pivot.to_numpy(float)
    vmax = np.nanpercentile(values, 90) if np.isfinite(values).any() else 1.0
    im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.astype(str))
    ax.set_title(f"Validation redshift-bin {metric} at last epoch")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    _savefig(fig, path)


def _extract_map_parameters(results: pd.DataFrame) -> pd.DataFrame:
    data = {}
    for name in PARAMETER_ORDER:
        col = f"fit_{name}"
        if col in results:
            data[name] = pd.to_numeric(results[col], errors="coerce")
    return pd.DataFrame(data)


def _truth_frame_for_parameters(frame: pd.DataFrame) -> pd.DataFrame:
    data = {}
    if "row_index" in frame:
        data["row_index"] = frame["row_index"]
    for parameter, truth_column in TRUTH_COLUMN_BY_PARAMETER.items():
        if truth_column in frame:
            data[parameter] = pd.to_numeric(frame[truth_column], errors="coerce")
    return pd.DataFrame(data)


def _mclmc_projected_truth_frame(
    summary: pd.DataFrame,
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the richest defensible truth/proxy table for PopCosmos MCLMC plots."""
    if "row_index" not in summary:
        return pd.DataFrame(), pd.DataFrame()
    row_indices = (
        pd.to_numeric(summary["row_index"], errors="coerce")
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .to_list()
    )
    if not row_indices:
        return pd.DataFrame(), pd.DataFrame()

    config = _read_normalized_config(run_dir)
    catalog = _catalog_rows_for_indices(config, row_indices)
    if catalog.empty:
        fallback = _truth_frame_for_parameters(summary).drop_duplicates("row_index")
        metadata = _truth_metadata_from_columns(fallback)
        return fallback, metadata

    truth = pd.DataFrame({"row_index": catalog["row_index"].to_numpy(int)})
    metadata_rows: list[dict[str, object]] = []

    def assign(
        parameter: str,
        values,
        *,
        source_kind: str,
        source_columns: str,
        formula: str,
        notes: str = "",
    ) -> None:
        arr = pd.to_numeric(pd.Series(values, index=truth.index), errors="coerce")
        truth[parameter] = arr.to_numpy(float)
        metadata_rows.append(
            {
                "parameter": parameter,
                "source_kind": source_kind,
                "source_columns": source_columns,
                "formula": formula,
                "finite_fraction": float(np.isfinite(arr.to_numpy(float)).mean())
                if len(arr)
                else float("nan"),
                "notes": notes,
            }
        )

    def missing(parameter: str, notes: str) -> None:
        truth[parameter] = np.nan
        metadata_rows.append(
            {
                "parameter": parameter,
                "source_kind": "missing",
                "source_columns": "",
                "formula": "",
                "finite_fraction": 0.0,
                "notes": notes,
            }
        )

    if "redshift_true" in catalog:
        assign(
            "z_obs",
            catalog["redshift_true"],
            source_kind="direct_truth",
            source_columns="redshift_true",
            formula="z_obs = redshift_true",
        )
    if "logsm_true" in catalog:
        assign(
            "log10_stellar_mass",
            catalog["logsm_true"],
            source_kind="direct_truth",
            source_columns="logsm_true",
            formula="log10_stellar_mass = logsm_true",
        )
    if "logsfr_true" in catalog:
        assign(
            "log10_sfr_at_obs",
            catalog["logsfr_true"],
            source_kind="direct_truth",
            source_columns="logsfr_true",
            formula="log10_sfr_at_obs = logsfr_true",
        )
    if "logssfr_true" in catalog:
        assign(
            "log10_ssfr_at_obs",
            catalog["logssfr_true"],
            source_kind="direct_truth",
            source_columns="logssfr_true",
            formula="log10_ssfr_at_obs = logssfr_true",
        )
    elif {"logsfr_true", "logsm_true"} <= set(catalog.columns):
        assign(
            "log10_ssfr_at_obs",
            catalog["logsfr_true"] - catalog["logsm_true"],
            source_kind="derived_truth",
            source_columns="logsfr_true,logsm_true",
            formula="log10_ssfr_at_obs = logsfr_true - logsm_true",
        )

    sfh_projection, sfh_kind, sfh_note = _project_diffsky_sfh_truth_to_popcosmos(
        catalog,
        n_sfh_bins=int((config.get("model", {}) or {}).get("n_sfh_bins", 80)),
    )
    if not sfh_projection.empty:
        for index in range(1, 7):
            name = f"dlog10_sfr_{index}"
            assign(
                name,
                sfh_projection[name],
                source_kind=sfh_kind,
                source_columns=",".join(DIFFSKY_SFH_TRUTH_COLUMNS),
                formula=(
                    "Diffstar/Diffmah generated SFH -> "
                    "project_sfh_to_popcosmos_dlogsfr_jax"
                ),
                notes=sfh_note,
            )
    else:
        for index in range(1, 7):
            missing(
                f"dlog10_sfr_{index}",
                "No Diffstar/Diffmah projection was available for this run.",
            )

    if "dust_av" in catalog:
        assign(
            "tau2",
            catalog["dust_av"] / 1.086,
            source_kind="projected_generated_truth",
            source_columns="dust_av",
            formula="tau2 = dust_av / 1.086",
            notes="Matches diffsky_basic_dust_params_jax.",
        )
    else:
        missing("tau2", "No dust_av or equivalent catalog column is available.")
    if "dust_delta" in catalog:
        assign(
            "dust_index_n",
            catalog["dust_delta"],
            source_kind="projected_generated_truth",
            source_columns="dust_delta",
            formula="dust_index_n = dust_delta",
            notes="Matches diffsky_basic_dust_params_jax.",
        )
    else:
        missing("dust_index_n", "No dust_delta or equivalent catalog column is available.")

    if "log10_stellar_metallicity_true" in catalog:
        assign(
            "log10_stellar_metallicity",
            catalog["log10_stellar_metallicity_true"],
            source_kind="direct_truth",
            source_columns="log10_stellar_metallicity_true",
            formula="log10_stellar_metallicity = log10_stellar_metallicity_true",
        )
    else:
        missing(
            "log10_stellar_metallicity",
            "The active Diffsky parquet has no object-level Zstar truth column.",
        )
    missing(
        "tau1_over_tau2",
        "The active Diffsky parquet has no object-level birth-cloud dust ratio.",
    )

    metadata = pd.DataFrame(metadata_rows)
    return truth, metadata


def _read_normalized_config(run_dir: Path) -> dict:
    path = run_dir / "normalized_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _catalog_rows_for_indices(config: dict, row_indices: list[int]) -> pd.DataFrame:
    catalog_path = config.get("catalog_path")
    if not catalog_path:
        return pd.DataFrame()
    path = Path(str(catalog_path))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return pd.DataFrame()
    try:
        catalog = pd.read_parquet(path)
    except Exception as exc:
        print(f"[comparison][warning] could not read catalog truth table {path}: {exc}")
        return pd.DataFrame()
    valid = [idx for idx in row_indices if 0 <= idx < len(catalog)]
    if not valid:
        return pd.DataFrame()
    rows = catalog.iloc[valid].copy()
    rows.insert(0, "row_index", valid)
    return rows.reset_index(drop=True)


def _truth_metadata_from_columns(truth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for parameter in truth.columns:
        if parameter == "row_index":
            continue
        values = pd.to_numeric(truth[parameter], errors="coerce").to_numpy(float)
        rows.append(
            {
                "parameter": parameter,
                "source_kind": "summary_truth",
                "source_columns": TRUTH_COLUMN_BY_PARAMETER.get(parameter, ""),
                "formula": "",
                "finite_fraction": float(np.isfinite(values).mean())
                if values.size
                else float("nan"),
                "notes": "Fallback from run summary because catalog_path was unavailable.",
            }
        )
    return pd.DataFrame(rows)


def _project_diffsky_sfh_truth_to_popcosmos(
    catalog: pd.DataFrame,
    *,
    n_sfh_bins: int,
) -> tuple[pd.DataFrame, str, str]:
    if not set(DIFFSKY_SFH_TRUTH_COLUMNS).issubset(catalog.columns):
        return _constant_logssfr_sfh_proxy(catalog)
    try:
        import jax.numpy as jnp
        from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z

        from euclid_dsps.model import (
            build_diffsky_basic_sfh_table_jax,
            project_sfh_to_popcosmos_dlogsfr_jax,
        )
    except Exception as exc:
        proxy, kind, note = _constant_logssfr_sfh_proxy(catalog)
        if not proxy.empty:
            return (
                proxy,
                kind,
                f"{note}; exact Diffstar/Diffmah projection unavailable: {exc}",
            )
        return pd.DataFrame(), "missing", f"Diffstar/Diffmah projection unavailable: {exc}"

    rows = []
    try:
        for _, row in catalog.iterrows():
            params = {
                "z_obs": float(row["redshift_true"]),
                "diffstar_lgmcrit": float(row["diffstar_lgmcrit"]),
                "diffstar_lgy_at_mcrit": float(row["diffstar_lgy_at_mcrit"]),
                "diffstar_indx_lo": float(row["diffstar_indx_lo"]),
                "diffstar_indx_hi": float(row["diffstar_indx_hi"]),
                "diffstar_lg_qt": float(row["diffstar_lg_qt"]),
                "diffstar_qlglgdt": float(row["diffstar_qlglgdt"]),
                "diffstar_lg_drop": float(row["diffstar_lg_drop"]),
                "diffstar_lg_rejuv": float(row["diffstar_lg_rejuv"]),
                "diffmah_logm0": float(row["diffmah_logm0"]),
                "diffmah_logtc": float(row["diffmah_logtc"]),
                "diffmah_early_index": float(row["diffmah_early_index"]),
                "diffmah_late_index": float(row["diffmah_late_index"]),
                "diffmah_t_peak": float(row["diffmah_t_peak"]),
            }
            z_obs = jnp.asarray(params["z_obs"], dtype=jnp.float32)
            t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]
            gal_t_table = jnp.linspace(0.05, jnp.maximum(t_obs, 0.06), int(n_sfh_bins))
            sfh = build_diffsky_basic_sfh_table_jax(gal_t_table, t_obs, params)
            dlogs = np.asarray(
                project_sfh_to_popcosmos_dlogsfr_jax(gal_t_table, sfh, t_obs),
                dtype=float,
            )
            rows.append({f"dlog10_sfr_{idx + 1}": dlogs[idx] for idx in range(6)})
    except Exception as exc:
        proxy, kind, note = _constant_logssfr_sfh_proxy(catalog)
        if not proxy.empty:
            return (
                proxy,
                kind,
                f"{note}; exact Diffstar/Diffmah projection failed: {exc}",
            )
        return pd.DataFrame(), "missing", f"Diffstar/Diffmah projection failed: {exc}"
    return (
        pd.DataFrame(rows, index=catalog.index),
        "projected_generated_truth",
        "Projected from Diffstar/Diffmah generated SFH into PopCosmos lookback bins.",
    )


def _constant_logssfr_sfh_proxy(catalog: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    if "logssfr_true" in catalog:
        logssfr = pd.to_numeric(catalog["logssfr_true"], errors="coerce")
    elif {"logsfr_true", "logsm_true"} <= set(catalog.columns):
        logssfr = pd.to_numeric(catalog["logsfr_true"], errors="coerce") - pd.to_numeric(
            catalog["logsm_true"], errors="coerce"
        )
    else:
        return pd.DataFrame(), "missing", "No logssfr_true or logsfr_true-logsm_true proxy."
    slope = ((logssfr.astype(float) - (-10.0)) / 6.0).clip(-3.0, 3.0)
    data = {f"dlog10_sfr_{idx}": slope.to_numpy(float) for idx in range(1, 7)}
    return (
        pd.DataFrame(data, index=catalog.index),
        "truth_proxy",
        "Fallback constant-slope proxy from logssfr_true; less accurate than SFH projection.",
    )


def _prior_bounds_from_config(run_dir: Path) -> dict[str, tuple[float, float]]:
    path = run_dir / "normalized_config.json"
    if not path.exists():
        return {}
    try:
        config = json.loads(path.read_text())
    except Exception:
        return {}
    free = config.get("fit", {}).get("free_parameters", {})
    if not free:
        free = config.get("sample", {}).get("free_parameters", {})
    bounds: dict[str, tuple[float, float]] = {}
    if isinstance(free, dict):
        for name, spec in free.items():
            if isinstance(spec, dict) and "bounds" in spec:
                raw = spec["bounds"]
                if isinstance(raw, (list, tuple)) and len(raw) == 2:
                    try:
                        bounds[str(name)] = (float(raw[0]), float(raw[1]))
                    except (TypeError, ValueError):
                        pass
    return bounds


def _wavelength_map_from_residuals(frame: pd.DataFrame) -> dict[str, float]:
    if "effective_wavelength_angstrom" not in frame:
        return {}
    work = frame[["band", "effective_wavelength_angstrom"]].dropna().drop_duplicates()
    return {
        str(row["band"]): float(row["effective_wavelength_angstrom"])
        for _, row in work.iterrows()
    }


def _effective_wavelengths(frame: pd.DataFrame, wavelength_map: dict[str, float]) -> pd.Series:
    wave = pd.Series(np.nan, index=frame.index, dtype=float)
    if "effective_wavelength_angstrom" in frame:
        wave = pd.to_numeric(frame["effective_wavelength_angstrom"], errors="coerce")
    if "band" in frame:
        fallback = frame["band"].astype(str).map(wavelength_map)
        wave = wave.where(np.isfinite(wave), fallback)
    return pd.to_numeric(wave, errors="coerce")


def _global_wavelength_map() -> dict[str, float]:
    for path in [
        SYNC_RUNS_FULL / "diffsky_reconstruction_map_worst_500",
        SYNC_RUNS_FULL / "diffsky_reconstruction_map_worst_1000_b512_iter400",
    ]:
        try:
            frame = _read_table_stem(path, "batch_fit_photometry_comparison")
        except FileNotFoundError:
            continue
        mapping = _wavelength_map_from_residuals(
            _normalize_map_comparison(frame)
        )
        if mapping:
            return mapping
    return {}


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _posterior_summary_to_wide(summary: pd.DataFrame, *, value: str) -> pd.DataFrame:
    if not {"row_index", "parameter", value}.issubset(summary.columns):
        return pd.DataFrame()
    wide = summary.pivot(index="row_index", columns="parameter", values=value).reset_index()
    wide.columns = [str(c) for c in wide.columns]
    return wide


def _write_readme(out: Path, manifest: dict, table_paths: dict[str, Path]) -> None:
    worst500 = _markdown_table(pd.read_csv(table_paths["worst500"]))
    worst100 = _markdown_table(pd.read_csv(table_paths["worst100"]))
    reference = _markdown_table(pd.read_csv(table_paths["reference_full"]))
    text = f"""# Diffsky Reconstruction Debug Comparison

This directory is the local comparison workspace for the Diffsky reconstruction
debug experiments.

## Reference Contract

Canonical reference inference:

`{manifest["canonical_reference_run"]}`

The `worst_500` and `worst_100` rowsets are diagnostic slices derived from this
canonical reference. MAP and MCLMC are not replacement references; they test
whether DSPS can recover photometry on the NN failure cases.

## Residual Definition

All boxplots and histograms in this dashboard use the same normalized flux
residual:

`(flux_in - flux_out) / sigma_eff`

where `flux_in` is the observed/input photometry, `flux_out` is the model output
or posterior-predictive median, and `sigma_eff` is the effective likelihood
uncertainty used by the run. Red dashed guide lines mark `-3` and `+3`.

`reference_full_residual_hist_by_band.png` is the grid of these residual
histograms for the canonical full-20k reference, one panel per photometric band.

## What The Worst-Slice Comparisons Mean

The `worst_500` and `worst_100` objects are selected from the canonical
full-20k NN reference residuals. The other methods are then evaluated on exactly
these same row indices. For example, `plots/worst100_dsps_recovery/` answers:
given the 100 worst objects from the reference NN, what happens if we rerun
input-noise NN, KL NN, supervised-prior NN, MAP, and MCLMC?

The paired baseline plots are the most direct photometric test:

- `same_object_error_distribution_by_method.png`: distribution of per-object
  median absolute photometric errors for the same row indices.
- `paired_object_baseline_vs_methods.png`: each point is one object; points
  below the diagonal have lower error than the baseline NN.
- `paired_band_baseline_vs_methods.png`: same comparison at `(object, band)`
  level.
- `paired_*_improvement_histograms.png`: positive values mean the tested method
  has lower absolute photometric error than the baseline on the same object or
  band row.

## Input Noise Experiment

`input_noise_sigma1` is an encoder-input denoising experiment. Gaussian noise is
injected into the encoder input features during training, with
`sigma_scale=1.0`; the photometric likelihood targets are not noised. The test
is whether this makes the amortized network more robust on the same worst-case
objects selected from the canonical reference.

## Source Layout

- `reference/`: symlinks to the canonical full 20k NN training and inference runs.
- `rowsets/`: symlink to the canonical rowsets used for the diagnostic slices.
- `runs/`: symlinks to NN variant, MAP, and MCLMC comparison runs.
- `training/`: symlinks to NN training runs when available.
- `tables/`: normalized residual summaries.
- `plots/`: generated visual diagnostics.

## Key Plots

Reference full 20k:

- [reference_full_residual_hist.png](plots/reference_full/reference_full_residual_hist.png)
- [reference_full_residual_boxplot_by_band.png](plots/reference_full/reference_full_residual_boxplot_by_band.png)
- [reference_full_residual_hist_by_band.png](plots/reference_full/reference_full_residual_hist_by_band.png)
- [reference_full_top_objects.png](plots/reference_full/reference_full_top_objects.png)

Worst 500 NN variants:

- [method_residual_summary.png](plots/worst500_nn_variants/method_residual_summary.png)
- [same_object_error_distribution_by_method.png](plots/worst500_nn_variants/same_object_error_distribution_by_method.png)
- [paired_object_baseline_vs_methods.png](plots/worst500_nn_variants/paired_object_baseline_vs_methods.png)
- [paired_band_baseline_vs_methods.png](plots/worst500_nn_variants/paired_band_baseline_vs_methods.png)
- [paired_baseline_improvement_summary.png](plots/worst500_nn_variants/paired_baseline_improvement_summary.png)
- [band_median_abs_residual_heatmap.png](plots/worst500_nn_variants/band_median_abs_residual_heatmap.png)
- [residual_boxplot_by_band_by_method.png](plots/worst500_nn_variants/residual_boxplot_by_band_by_method.png)
- [method_abs_residual_cdf.png](plots/worst500_nn_variants/method_abs_residual_cdf.png)

Worst 100 DSPS recovery:

- [method_residual_summary.png](plots/worst100_dsps_recovery/method_residual_summary.png)
- [baseline_map_mclmc_signed_residual_histograms.png](plots/worst100_dsps_recovery/baseline_map_mclmc_signed_residual_histograms.png)
- [worst100_location_in_full_nn_and_map.png](plots/worst100_dsps_recovery/worst100_location_in_full_nn_and_map.png)
- [sed_examples_baseline_map_mclmc_grid.png](plots/worst100_dsps_recovery/sed_examples_baseline_map_mclmc_grid.png)
- [huge_error_bar_diagnostics.png](plots/worst100_dsps_recovery/huge_error_bar_diagnostics.png)
- [huge error-bar explanation](tables/worst100/worst100_huge_error_bar_explanation.md)
- [same_object_error_distribution_by_method.png](plots/worst100_dsps_recovery/same_object_error_distribution_by_method.png)
- [paired_object_baseline_vs_methods.png](plots/worst100_dsps_recovery/paired_object_baseline_vs_methods.png)
- [paired_band_baseline_vs_methods.png](plots/worst100_dsps_recovery/paired_band_baseline_vs_methods.png)
- [paired_baseline_improvement_summary.png](plots/worst100_dsps_recovery/paired_baseline_improvement_summary.png)
- [band_median_abs_residual_heatmap.png](plots/worst100_dsps_recovery/band_median_abs_residual_heatmap.png)
- [residual_boxplot_by_band_by_method.png](plots/worst100_dsps_recovery/residual_boxplot_by_band_by_method.png)
- [method_abs_residual_cdf.png](plots/worst100_dsps_recovery/method_abs_residual_cdf.png)

MAP:

- [map worst500 residuals](plots/map/worst500_iter200/map_residual_boxplot_by_band.png)
- [map worst500 corner](plots/map/worst500_iter200/map_parameter_corner.png)
- [map worst500 photometric SED points](plots/map/worst500_iter200/map_photometric_sed_best_median_worst.png)
- [map worst1000 residuals](plots/map/worst1000_iter400/map_residual_boxplot_by_band.png)
- [map worst1000 corner](plots/map/worst1000_iter400/map_parameter_corner.png)
- [map worst1000 photometric SED points](plots/map/worst1000_iter400/map_photometric_sed_best_median_worst.png)

MCLMC:

- [mclmc residuals by band](plots/mclmc/worst100_b32_w64_s256/mclmc_residual_boxplot_by_band.png)
- [mclmc pooled samples density corner](plots/mclmc/worst100_b32_w64_s256/mclmc_corner_pooled_samples.png)
- [mclmc projected-truth distributions](plots/mclmc/worst100_b32_w64_s256/mclmc_projected_truth_distributions.png)
- [mclmc posterior medians density corner](plots/mclmc/worst100_b32_w64_s256/mclmc_corner_posterior_medians.png)
- [mclmc projected-truth corner](plots/mclmc/worst100_b32_w64_s256/mclmc_corner_truth_comparable.png)
- [mclmc projected-truth parameter table](plots/mclmc/worst100_b32_w64_s256/mclmc_projected_truth_parameters.csv)
- [mclmc projected-truth metadata table](plots/mclmc/worst100_b32_w64_s256/mclmc_projected_truth_metadata.csv)
- [mclmc good object corner](plots/mclmc/worst100_b32_w64_s256/individual_corners/mclmc_corner_good.png)
- [mclmc average object corner](plots/mclmc/worst100_b32_w64_s256/individual_corners/mclmc_corner_average.png)
- [mclmc bad object corner](plots/mclmc/worst100_b32_w64_s256/individual_corners/mclmc_corner_bad.png)
- [mclmc diagnostics](plots/mclmc/worst100_b32_w64_s256/mclmc_diagnostics_summary.png)
- [mclmc photometric SED points](plots/mclmc/worst100_b32_w64_s256/mclmc_photometric_sed_best_median_worst.png)

For MCLMC:

- `pooled posterior samples` means all posterior samples from all galaxies are
  stacked together. The corner now uses 50%/90% density contours. It is a
  population diagnostic, not the posterior of one galaxy.
- `posterior medians` means one point per galaxy, using the median posterior
  value of each parameter. It is better for comparing fitted population
  locations against catalog/projected truth.
- `projected-truth distributions` shows the direct/projected truth table alone,
  including direct SFR and sSFR truth, so missing overlays in a posterior plot
  can be separated from missing projected truth.
- `projected-truth` tables use direct catalog truth for redshift, stellar mass,
  SFR, and sSFR; Diffstar/Diffmah generated-truth SFHs projected into the
  PopCosmos `dlog10_sfr_*` bins; and the existing Diffsky dust convention
  `tau2=dust_av/1.086`, `dust_index_n=dust_delta`. The corner overlays the
  subset of these quantities that also exists as MCLMC posterior coordinates.
  The MCLMC corners use every posterior coordinate with finite truth:
  `z_obs`, `log10_stellar_mass`, all six `dlog10_sfr_*` coordinates, `tau2`,
  and `dust_index_n`.
  Orange stars, rugs, and dashed median lines are the catalog/projected truth
  overlays; blue contours/histograms are the MCLMC posterior samples or
  posterior medians.
  `log10 Zstar` and `tau1/tau2` remain marked as missing truth in the metadata
  table because the active parquet has no object-level columns for those
  quantities.

NN step-by-step:

- [training curves](plots/nn_step_by_step/training/training_loss_nll_kl_curves.png)
- [gradient norms](plots/nn_step_by_step/training/training_gradient_norms_by_epoch.png)
- [validation redshift bins](plots/nn_step_by_step/training/validation_redshift_bin_chi2.png)
- [inference plot gallery](plots/nn_step_by_step/inference/)

Open [index.html](index.html) for a visual gallery.

## Reference Full 20k Summary

{reference}

## Worst 500 Summary

{worst500}

## Worst 100 Summary

{worst100}

## MCLMC Caveat

The current MCLMC diagnostic run uses one chain. It is useful as a photometric
recoverability baseline, but posterior-science claims still need multi-chain
diagnostics such as ESS/R-hat.
"""
    (out / "README.md").write_text(text)


def _write_gallery(out: Path) -> None:
    images = sorted((out / "plots").rglob("*.png"))
    core = [
        out / "plots/reference_full/reference_full_residual_boxplot_by_band.png",
        out / "plots/worst500_nn_variants/method_residual_summary.png",
        out / "plots/worst500_nn_variants/paired_baseline_improvement_summary.png",
        out / "plots/worst500_nn_variants/paired_object_baseline_vs_methods.png",
        out / "plots/worst500_nn_variants/band_median_abs_residual_heatmap.png",
        out / "plots/worst100_dsps_recovery/method_residual_summary.png",
        out / "plots/worst100_dsps_recovery/baseline_map_mclmc_signed_residual_histograms.png",
        out / "plots/worst100_dsps_recovery/worst100_location_in_full_nn_and_map.png",
        out / "plots/worst100_dsps_recovery/sed_examples_baseline_map_mclmc_grid.png",
        out / "plots/worst100_dsps_recovery/huge_error_bar_diagnostics.png",
        out / "plots/worst100_dsps_recovery/paired_baseline_improvement_summary.png",
        out / "plots/worst100_dsps_recovery/paired_object_baseline_vs_methods.png",
        out / "plots/worst100_dsps_recovery/band_median_abs_residual_heatmap.png",
        out / "plots/map/worst500_iter200/map_photometric_sed_best_median_worst.png",
        out / "plots/mclmc/worst100_b32_w64_s256/mclmc_photometric_sed_best_median_worst.png",
        out / "plots/nn_step_by_step/training/training_loss_nll_kl_curves.png",
    ]
    core = [path for path in core if path.exists()]
    grouped: dict[str, list[Path]] = {}
    for image in images:
        rel_parts = image.relative_to(out / "plots").parts
        group = rel_parts[0] if rel_parts else "plots"
        grouped.setdefault(group, []).append(image)

    def card(image: Path) -> str:
        rel = image.relative_to(out)
        return f'<section class="card"><h3>{rel}</h3><a href="{rel}"><img src="{rel}" alt="{rel}"></a></section>'

    core_cards = "\n".join(card(image) for image in core)
    details = []
    for group, group_images in sorted(grouped.items()):
        content = "\n".join(card(image) for image in group_images)
        details.append(f"<details><summary>{group} ({len(group_images)} plots)</summary>{content}</details>")
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Diffsky Reconstruction Comparison</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; color: #202020; }
    section.card { margin: 0 0 28px; }
    h1 { margin-bottom: 4px; }
    h2 { margin-top: 32px; }
    h3 { font-size: 15px; font-weight: 600; margin: 0 0 8px; }
    img { max-width: 100%; border: 1px solid #ddd; }
    a { color: #1f5f9f; }
    details { margin: 16px 0 28px; }
    summary { cursor: pointer; font-weight: 700; font-size: 18px; }
    .note { max-width: 980px; line-height: 1.45; }
  </style>
</head>
<body>
  <h1>Diffsky Reconstruction Comparison</h1>
  <p class="note">Residual plots use (flux_in - flux_out) / sigma_eff. The
  core plots below are the recommended starting point; full plot groups are
  collapsed afterwards to keep the page navigable.</p>
  <h2>Core Plots</h2>
"""
    html += core_cards
    html += "\n<h2>All Plot Groups</h2>\n"
    html += "\n".join(details)
    html += "\n</body>\n</html>\n"
    (out / "index.html").write_text(html)


def _read_table_stem(run: Path, stem: str) -> pd.DataFrame:
    for suffix in [".parquet", ".csv"]:
        path = run / f"{stem}{suffix}"
        if path.exists():
            return pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
    chunk_dir = run / "_chunks"
    if chunk_dir.exists():
        chunks = sorted(chunk_dir.glob(f"{stem}_chunk_*.parquet"))
        if chunks:
            return pd.concat([pd.read_parquet(path) for path in chunks], ignore_index=True)
    raise FileNotFoundError(f"Could not find {stem}.parquet/.csv under {run}")


def _read_rowset(path: Path) -> list[int]:
    values = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.append(int(line.split(",")[0]))
    return values


def _valid(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["valid"].astype(bool)].copy() if "valid" in frame else frame.copy()
    valid["residual_sigma_median"] = pd.to_numeric(valid["residual_sigma_median"], errors="coerce")
    valid["abs_residual_sigma_median"] = valid["residual_sigma_median"].abs()
    return valid[np.isfinite(valid["residual_sigma_median"])]


def _clip_for_display(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    lo, hi = np.nanpercentile(values, [1, 99])
    bound = max(abs(lo), abs(hi), 5.0)
    return np.clip(values, -bound, bound)


def _markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    work = frame.head(max_rows).copy()
    for col in work.select_dtypes(include="number").columns:
        work[col] = work[col].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
    columns = [str(col) for col in work.columns]
    rows = [columns, ["---"] * len(columns)]
    for _, row in work.iterrows():
        rows.append([_markdown_cell(row[col]) for col in work.columns])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _write_table(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    try:
        frame.to_parquet(stem.with_suffix(".parquet"), index=False)
    except Exception as exc:
        print(f"[comparison][warning] could not write parquet {stem}: {exc}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _link_or_copy(src: Path, dst: Path, *, copy_runs: bool) -> None:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists() and copy_runs:
        shutil.rmtree(dst)
    elif dst.exists():
        return
    if copy_runs:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src, target_is_directory=src.is_dir())


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _first_complete_rowset_dir(paths: Iterable[Path]) -> Path | None:
    required = ["worst_500.txt", "worst_100.txt"]
    for path in paths:
        if path.exists() and all((path / name).exists() for name in required):
            return path
    return _first_existing(paths)


def _spec_by_key(specs: list[RunSpec], key: str) -> RunSpec:
    for spec in specs:
        if spec.key == key:
            return spec
    raise KeyError(key)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _nanmedian(values: np.ndarray) -> float:
    return float(np.nanmedian(values)) if values.size else float("nan")


def _nanmean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if values.size else float("nan")


def _nanpercentile(values: np.ndarray, q: float) -> float:
    return float(np.nanpercentile(values, q)) if values.size else float("nan")


def _nanmax(values: np.ndarray) -> float:
    return float(np.nanmax(values)) if values.size else float("nan")


def _frac(mask: np.ndarray) -> float:
    mask = np.asarray(mask)
    return float(mask.mean()) if mask.size else float("nan")


if __name__ == "__main__":
    main()
