"""Lightweight collapse gates for amortized Diffsky runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.io import write_json


def write_inference_collapse_gate(run_dir: str | Path) -> dict[str, Any]:
    """Write a truth-aware closure gate for an inference or MAP output directory."""
    out = Path(run_dir)
    checks: list[dict[str, Any]] = []
    _add_photoz_checks(out, checks)
    _add_extended_truth_checks(out, checks)
    _add_prior_population_checks(out, checks)
    _add_real_photometry_checks(out, checks)
    status = _status_from_checks(checks)
    payload = {
        "status": status,
        "n_checks": len(checks),
        "n_fail": sum(check["status"] == "FAIL" for check in checks),
        "n_warn": sum(check["status"] == "WARN" for check in checks),
        "checks": checks,
        "interpretation": (
            "Closure diagnostic only: this gate uses truth columns when available. "
            "On real data, use it as a runtime and self-consistency gate, not as "
            "a science validation."
        ),
    }
    write_json(out / "collapse_gate.json", payload)
    return payload


def write_training_collapse_gate(run_dir: str | Path) -> dict[str, Any]:
    """Write a lightweight gate from training logs alone."""
    out = Path(run_dir)
    checks: list[dict[str, Any]] = []
    log_path = out / "training_log.csv"
    progress_path = out / "training_progress.json"
    if log_path.exists():
        try:
            log = pd.read_csv(log_path)
        except Exception as exc:  # pragma: no cover - corrupt external artifact
            checks.append(_check("training_log_readable", "WARN", message=str(exc)))
        else:
            _add_training_log_checks(log, checks)
            _add_trainable_calibration_checks(out, log, checks)
    else:
        checks.append(
            _check("training_log_exists", "WARN", message="missing training_log.csv")
        )
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - corrupt external artifact
            checks.append(
                _check("training_progress_readable", "WARN", message=str(exc))
            )
        else:
            skipped = int(progress.get("updates_skipped", 0) or 0)
            checks.append(
                _check(
                    "updates_skipped_zero",
                    "PASS" if skipped == 0 else "FAIL",
                    value=skipped,
                    threshold=0,
                )
            )
    status = _status_from_checks(checks)
    payload = {
        "status": status,
        "n_checks": len(checks),
        "n_fail": sum(check["status"] == "FAIL" for check in checks),
        "n_warn": sum(check["status"] == "WARN" for check in checks),
        "checks": checks,
    }
    write_json(out / "training_collapse_gate.json", payload)
    return payload


def _add_photoz_checks(out: Path, checks: list[dict[str, Any]]) -> None:
    photoz_path = out / "photoz_metrics.csv"
    if not photoz_path.exists():
        photoz_path = out / "map_closure_photoz_metrics.csv"
    if not photoz_path.exists():
        checks.append(
            _check("photoz_metrics_exists", "WARN", message="missing photo-z metrics")
        )
        return
    frame = pd.read_csv(photoz_path)
    if frame.empty:
        checks.append(_check("photoz_metrics_nonempty", "FAIL"))
        return
    row = frame.iloc[0].to_dict()
    rmse = _first_number(row, "rmse", "rmse_dz", "sigma_dz")
    outlier = _first_number(row, "outlier_fraction_0p15", "outlier_fraction")
    bias = _first_number(row, "median_bias", "bias_median", "median_dz")
    coverage68 = _first_number(row, "coverage_68", "coverage_68_fraction")
    if rmse is not None:
        checks.append(
            _threshold("photoz_rmse", rmse, warn=0.12, fail=0.20, lower_is_better=True)
        )
    if outlier is not None:
        checks.append(
            _threshold(
                "photoz_outlier_fraction_0p15",
                outlier,
                warn=0.20,
                fail=0.35,
                lower_is_better=True,
            )
        )
    if bias is not None:
        checks.append(
            _threshold(
                "photoz_abs_median_bias",
                abs(bias),
                warn=0.03,
                fail=0.07,
                lower_is_better=True,
            )
        )
    if coverage68 is not None:
        checks.append(
            _threshold(
                "photoz_coverage_68",
                coverage68,
                warn=0.45,
                fail=0.25,
                lower_is_better=False,
            )
        )


def _add_extended_truth_checks(out: Path, checks: list[dict[str, Any]]) -> None:
    posterior_path = out / "posterior_vs_truth_extended.csv"
    map_path = out / "map_vs_truth_extended.csv"
    path = posterior_path if posterior_path.exists() else map_path
    if not path.exists():
        checks.append(
            _check(
                "extended_truth_metrics_exists",
                "WARN",
                message="missing posterior/map extended truth metrics",
            )
        )
        return
    frame = pd.read_csv(path)
    if frame.empty:
        checks.append(_check("extended_truth_metrics_nonempty", "WARN"))
        return
    for param, warn, fail in (
        ("z_obs", 0.05, 0.10),
        ("log10_stellar_mass", 0.35, 0.70),
        ("tau2_proxy", 0.50, 1.00),
        ("dust_index_n_proxy", 0.35, 0.70),
    ):
        row = (
            frame.loc[frame["parameter"] == param]
            if "parameter" in frame
            else pd.DataFrame()
        )
        if row.empty or "median_bias" not in row:
            continue
        value = float(abs(row.iloc[0]["median_bias"]))
        if not np.isfinite(value):
            continue
        checks.append(
            _threshold(
                f"{param}_abs_median_bias",
                value,
                warn=warn,
                fail=fail,
                lower_is_better=True,
            )
        )


def _add_prior_population_checks(out: Path, checks: list[dict[str, Any]]) -> None:
    path = out / "prior_vs_truth_population.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty or "parameter" not in frame:
        return
    for row in frame.itertuples(index=False):
        param = str(row.parameter)
        for col in ("quantile_l1_iqr", "median_delta_iqr"):
            if not hasattr(row, col):
                continue
            value = float(abs(getattr(row, col)))
            if not np.isfinite(value):
                continue
            checks.append(
                _threshold(
                    f"prior_{param}_{col}",
                    value,
                    warn=0.25,
                    fail=0.50,
                    lower_is_better=True,
                )
            )
    expected_parameters = 15
    checks.append(
        _check(
            "prior_native_marginals_complete",
            "PASS" if frame.parameter.nunique() >= expected_parameters else "FAIL",
            value=int(frame.parameter.nunique()),
            threshold=expected_parameters,
        )
    )
    correlation_path = out / "prior_vs_truth_correlations.csv"
    if correlation_path.exists():
        correlations = pd.read_csv(correlation_path)
        delta = pd.to_numeric(correlations.get("abs_delta"), errors="coerce")
        finite = delta[np.isfinite(delta)]
        if not finite.empty:
            checks.append(
                _threshold(
                    "prior_15d_spearman_mean_abs_delta",
                    float(finite.mean()),
                    warn=0.15,
                    fail=0.30,
                    lower_is_better=True,
                )
            )
            checks.append(
                _threshold(
                    "prior_15d_spearman_q90_abs_delta",
                    float(finite.quantile(0.90)),
                    warn=0.30,
                    fail=0.55,
                    lower_is_better=True,
                )
            )
    else:
        checks.append(_check("prior_15d_correlations_exist", "FAIL"))


def _add_training_log_checks(log: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    for column in (
        "loss",
        "negative_loglike",
        "residual_rms",
        "posterior_predictive_chi2",
    ):
        if column not in log:
            continue
        values = pd.to_numeric(log[column], errors="coerce").to_numpy(float)
        finite_fraction = float(np.mean(np.isfinite(values))) if values.size else 0.0
        checks.append(
            _threshold(
                f"{column}_finite_fraction",
                finite_fraction,
                warn=0.99,
                fail=0.95,
                lower_is_better=False,
            )
        )
    if "encoder_grad_norm" in log:
        gradient_rows = log
        if "split" in gradient_rows:
            gradient_rows = gradient_rows.loc[
                gradient_rows["split"].astype(str).str.lower() == "train"
            ]
        if "update_applied" in gradient_rows:
            applied = pd.to_numeric(
                gradient_rows["update_applied"], errors="coerce"
            ).fillna(0.0)
            gradient_rows = gradient_rows.loc[applied > 0.0]
        values = pd.to_numeric(
            gradient_rows["encoder_grad_norm"], errors="coerce"
        ).to_numpy(float)
        latest = values[np.isfinite(values)]
        if latest.size:
            checks.append(
                _threshold(
                    "latest_encoder_grad_norm",
                    float(latest[-1]),
                    warn=100.0,
                    fail=1000.0,
                    lower_is_better=True,
                )
            )
    wake_rows = log
    if "split" in wake_rows:
        wake_rows = wake_rows.loc[
            wake_rows["split"].astype(str).str.lower() == "train"
        ]
    if "wake_active" in wake_rows:
        active = pd.to_numeric(
            wake_rows["wake_active"], errors="coerce"
        ).fillna(0.0)
        wake_rows = wake_rows.loc[active > 0.5]
    elif "update_phase" in wake_rows:
        wake_rows = wake_rows.loc[
            wake_rows["update_phase"].astype(str).isin(
                {"encoder_wake", "joint_wake"}
            )
        ]
    else:
        wake_rows = wake_rows.iloc[0:0]
    if not wake_rows.empty:
        if "wake_ess_fraction_mean" in wake_rows:
            ess = pd.to_numeric(
                wake_rows["wake_ess_fraction_mean"], errors="coerce"
            )
            finite_ess = ess[np.isfinite(ess)]
        else:
            finite_ess = pd.Series(dtype=float)
        if not finite_ess.empty:
            checks.append(
                _threshold(
                    "wake_ess_fraction_mean",
                    float(finite_ess.mean()),
                    warn=0.25,
                    fail=0.15,
                    lower_is_better=False,
                )
            )
        if "wake_weight_max_mean" in wake_rows:
            max_weight = pd.to_numeric(
                wake_rows["wake_weight_max_mean"], errors="coerce"
            )
            finite_weight = max_weight[np.isfinite(max_weight)]
        else:
            finite_weight = pd.Series(dtype=float)
        if not finite_weight.empty:
            checks.append(
                _threshold(
                    "wake_weight_max_mean",
                    float(finite_weight.mean()),
                    warn=0.80,
                    fail=0.95,
                    lower_is_better=True,
                )
            )
        if "wake_physical_valid_fraction" in wake_rows:
            physical = pd.to_numeric(
                wake_rows["wake_physical_valid_fraction"], errors="coerce"
            )
            finite_physical = physical[np.isfinite(physical)]
        else:
            finite_physical = pd.Series(dtype=float)
        if not finite_physical.empty:
            checks.append(
                _threshold(
                    "wake_physical_valid_fraction",
                    float(finite_physical.mean()),
                    warn=0.99,
                    fail=0.95,
                    lower_is_better=False,
                )
            )


def _add_real_photometry_checks(
    out: Path, checks: list[dict[str, Any]]
) -> None:
    summary_path = out / "posterior_diagnostics_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        chi2 = _first_number(summary, "median_posterior_predictive_chi2")
        n_bands = _first_number(summary, "median_valid_bands")
        if chi2 is not None and n_bands is not None and n_bands > 0:
            checks.append(
                _threshold(
                    "median_posterior_predictive_reduced_chi2",
                    chi2 / n_bands,
                    warn=5.0,
                    fail=25.0,
                    lower_is_better=True,
                )
            )
    tails_path = out / "posterior_predictive_normalized_residual_tails.csv"
    if tails_path.exists():
        tails = pd.read_csv(tails_path)
        all_bands = tails.loc[tails.get("band") == "__all__"]
        if not all_bands.empty:
            frac = _first_number(
                all_bands.iloc[0].to_dict(), "frac_abs_gt_5"
            )
            if frac is not None:
                checks.append(
                    _threshold(
                        "posterior_predictive_frac_abs_gt_5",
                        frac,
                        warn=0.20,
                        fail=0.50,
                        lower_is_better=True,
                    )
                )
    bounds_path = out / "parameter_bound_diagnostics.csv"
    if bounds_path.exists():
        bounds = pd.read_csv(bounds_path)
        if "frac_within_5pct_boundary" not in bounds:
            return
        values = pd.to_numeric(
            bounds["frac_within_5pct_boundary"], errors="coerce"
        )
        finite = values[np.isfinite(values)]
        if not finite.empty:
            checks.append(
                _threshold(
                    "max_parameter_frac_within_5pct_boundary",
                    float(finite.max()),
                    warn=0.50,
                    fail=0.80,
                    lower_is_better=True,
                )
            )


def _add_trainable_calibration_checks(
    out: Path,
    log: pd.DataFrame,
    checks: list[dict[str, Any]],
) -> None:
    summary_path = out / "training_summary.json"
    if not summary_path.exists() or "band_alpha_grad_norm" not in log:
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    calibration = summary.get("per_band_flux_calibration") or {}
    if not bool(calibration.get("trainable", False)):
        return
    wake = log
    if "split" in wake:
        wake = wake.loc[wake["split"].astype(str).str.lower() == "train"]
    if "wake_active" in wake:
        active = pd.to_numeric(wake["wake_active"], errors="coerce").fillna(0.0)
        wake = wake.loc[active > 0.5]
    values = pd.to_numeric(wake["band_alpha_grad_norm"], errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        checks.append(
            _check(
                "trainable_band_calibration_has_wake_gradient",
                "FAIL",
                message="no finite wake calibration gradients",
            )
        )
        return
    maximum = float(finite.max())
    checks.append(
        _check(
            "trainable_band_calibration_has_wake_gradient",
            "PASS" if maximum > 1.0e-12 else "FAIL",
            value=maximum,
            threshold=1.0e-12,
        )
    )


def _first_number(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in row:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def _threshold(
    name: str,
    value: float,
    *,
    warn: float,
    fail: float,
    lower_is_better: bool,
) -> dict[str, Any]:
    if lower_is_better:
        status = "FAIL" if value > fail else "WARN" if value > warn else "PASS"
    else:
        status = "FAIL" if value < fail else "WARN" if value < warn else "PASS"
    return _check(name, status, value=value, warn=warn, fail=fail)


def _check(
    name: str,
    status: str,
    *,
    value: float | int | None = None,
    warn: float | None = None,
    fail: float | None = None,
    threshold: float | int | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "value": value,
        "warn": warn,
        "fail": fail,
        "threshold": threshold,
        "message": message,
    }


def _status_from_checks(checks: list[dict[str, Any]]) -> str:
    if any(check["status"] == "FAIL" for check in checks):
        return "FAIL"
    if any(check["status"] == "WARN" for check in checks):
        return "WARN"
    return "PASS"
