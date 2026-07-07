"""Lightweight collapse gates for amortized Diffsky runs."""

from __future__ import annotations

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
    else:
        checks.append(
            _check("training_log_exists", "WARN", message="missing training_log.csv")
        )
    if progress_path.exists():
        try:
            import json

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
    for param, warn, fail in (
        ("z_obs", 0.10, 0.20),
        ("log10_stellar_mass", 0.50, 1.00),
    ):
        row = frame.loc[frame["parameter"] == param]
        if row.empty:
            continue
        for col in ("quantile_l1", "median_delta"):
            if col in row:
                value = float(abs(row.iloc[0][col]))
                if not np.isfinite(value):
                    continue
                checks.append(
                    _threshold(
                        f"prior_{param}_{col}",
                        value,
                        warn=warn,
                        fail=fail,
                        lower_is_better=True,
                    )
                )


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
        values = pd.to_numeric(log["encoder_grad_norm"], errors="coerce").to_numpy(
            float
        )
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
