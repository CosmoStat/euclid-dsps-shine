#!/usr/bin/env python3
"""Audit and normalize the dequantized FENIKS spline 15D prior target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize_scalar
from scipy.stats import chi2, ks_2samp, kurtosis, skew

from euclid_dsps.prior_learning.marginal_normalization import (
    forward_marginal,
    inverse_marginal,
)

PHYSICAL_NAMES = (
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
)
SFH_NAMES = tuple(f"sfh_dlog_sfr_{index:02d}" for index in range(1, 11))
NAMES = PHYSICAL_NAMES + SFH_NAMES
LABELS = {
    "z_obs": "Redshift z",
    "log10_stellar_mass": "log10 stellar mass",
    "log10_stellar_metallicity": "log10 stellar metallicity",
    "dust_av": "Dust A_V",
    "dust_delta": "Dust slope delta",
    **{name: f"SFH contrast q{index}" for index, name in enumerate(SFH_NAMES, 1)},
}
SCORE_PROBABILITIES = np.linspace(0.005, 0.995, 199)
NORMAL_QUANTILES = np.array(
    [NormalDist().inv_cdf(float(value)) for value in SCORE_PROBABILITIES]
)
QUANTILE_KNOTS = 257
SELECTION_TOLERANCE = 0.02


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("outputs/analysis/feniks_spline_15d_prior_20260710")
    parser.add_argument(
        "--exact-train", type=Path, default=root / "feniks_spline_15d_train.parquet"
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=root / "feniks_spline_15d_train_dequantized.parquet",
    )
    parser.add_argument(
        "--exact-test", type=Path, default=root / "feniks_spline_15d_test.parquet"
    )
    parser.add_argument(
        "--test", type=Path, default=root / "feniks_spline_15d_test_dequantized.parquet"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/analysis/feniks_spline_15d_normalization_20260715"),
    )
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser.parse_args()


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if tuple(frame.columns) != NAMES:
        raise ValueError(f"Unexpected columns in {path}: {tuple(frame.columns)}")
    if not np.isfinite(frame.to_numpy(float)).all():
        raise ValueError(f"Non-finite value in {path}")
    return frame


def _row_hash(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame, index=False).to_numpy(np.uint64)


def _group_split(
    exact: pd.DataFrame, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    hashes = _row_hash(exact)
    groups = np.unique(hashes)
    rng = np.random.default_rng(seed)
    validation_groups = rng.choice(
        groups,
        size=max(1, int(round(fraction * len(groups)))),
        replace=False,
    )
    validation = np.isin(hashes, validation_groups)
    return ~validation, validation


def _gaussian_score(values: np.ndarray) -> float:
    quantiles = np.quantile(np.asarray(values, dtype=float), SCORE_PROBABILITIES)
    return float(np.sqrt(np.mean((quantiles - NORMAL_QUANTILES) ** 2)))


def _fit_affine(values: np.ndarray) -> dict[str, Any]:
    center = float(np.mean(values))
    scale = float(np.std(values))
    if scale <= 1.0e-12:
        raise ValueError("Cannot normalize a constant coordinate")
    return {"family": "affine", "center": center, "scale": scale}


def _fit_asinh(values: np.ndarray) -> dict[str, Any]:
    data_scale = max(float(np.std(values)), 1.0e-12)

    def objective(log10_relative_lambda: float) -> float:
        lam = data_scale * 10.0**log10_relative_lambda
        raw = lam * np.arcsinh(values / lam)
        spec = _fit_affine(raw)
        return _gaussian_score(forward_marginal(raw, spec))

    result = minimize_scalar(
        objective,
        bounds=(-4.0, 4.0),
        method="bounded",
        options={"xatol": 1.0e-5},
    )
    lam = data_scale * 10.0 ** float(result.x)
    transformed = lam * np.arcsinh(values / lam)
    affine = _fit_affine(transformed)
    return {
        "family": "asinh",
        "lambda": lam,
        "center": affine["center"],
        "scale": affine["scale"],
    }


def _fit_shifted_asinh(values: np.ndarray, seed: int) -> dict[str, Any]:
    median = float(np.median(values))
    data_scale = max(float(np.std(values)), 1.0e-12)
    physical_quantiles = np.quantile(values, SCORE_PROBABILITIES)
    rng = np.random.default_rng(seed)
    moment_sample = (
        values if len(values) <= 8192 else rng.choice(values, size=8192, replace=False)
    )

    def objective(parameters: np.ndarray) -> float:
        shift = median + float(parameters[0]) * data_scale
        lam = data_scale * 10.0 ** float(parameters[1])
        sample_raw = np.arcsinh((moment_sample - shift) / lam)
        sample_scale = float(np.std(sample_raw))
        if not np.isfinite(sample_scale) or sample_scale <= 1.0e-12:
            return 1.0e6
        quantile_raw = np.arcsinh((physical_quantiles - shift) / lam)
        normalized = (quantile_raw - np.mean(sample_raw)) / sample_scale
        return float(np.sqrt(np.mean((normalized - NORMAL_QUANTILES) ** 2)))

    result = differential_evolution(
        objective,
        bounds=((-100.0, 100.0), (-4.0, 4.0)),
        seed=seed,
        maxiter=40,
        popsize=8,
        tol=1.0e-7,
        polish=True,
        workers=1,
    )
    shift = median + float(result.x[0]) * data_scale
    lam = data_scale * 10.0 ** float(result.x[1])
    transformed = lam * np.arcsinh((values - shift) / lam)
    affine = _fit_affine(transformed)
    support_distance = min(
        abs(shift - float(np.min(values))),
        abs(shift - float(np.max(values))),
    )
    return {
        "family": "shifted_asinh",
        "shift": shift,
        "lambda": lam,
        "center": affine["center"],
        "scale": affine["scale"],
        "shift_support_distance_std": support_distance / data_scale,
        "tail_fragile": bool(
            float(result.x[1]) <= -3.0 and support_distance / data_scale < 0.5
        ),
    }


def _fit_quantile_spline(values: np.ndarray) -> dict[str, Any]:
    probabilities = np.linspace(
        0.5 / len(values), 1.0 - 0.5 / len(values), QUANTILE_KNOTS
    )
    theta = np.quantile(values, probabilities)
    theta[0], theta[-1] = np.min(values), np.max(values)
    normal = np.array([NormalDist().inv_cdf(float(value)) for value in probabilities])
    unique, inverse, counts = np.unique(theta, return_inverse=True, return_counts=True)
    targets = np.zeros(len(unique), dtype=float)
    np.add.at(targets, inverse, normal)
    targets /= counts
    if len(unique) < 2:
        raise ValueError("Quantile spline requires at least two unique values")
    return {
        "family": "quantile_spline",
        "theta_knots": unique.tolist(),
        "normal_knots": targets.tolist(),
        "n_knots": int(len(unique)),
    }


def _fit_transforms(
    values: np.ndarray,
    validation: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    transforms: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    complexity = {"affine": 0, "asinh": 1, "quantile_spline": 2}
    for column, name in enumerate(NAMES):
        train_values = values[:, column]
        validation_values = validation[:, column]
        candidates = {
            "affine": _fit_affine(train_values),
            "asinh": _fit_asinh(train_values),
            "quantile_spline": _fit_quantile_spline(train_values),
        }
        scores: dict[str, float] = {}
        for family, spec in candidates.items():
            train_x = forward_marginal(train_values, spec)
            validation_x = forward_marginal(validation_values, spec)
            scores[family] = _gaussian_score(validation_x)
            rows.append(
                {
                    "parameter": name,
                    "family": family,
                    "fit_gaussian_qrmse": _gaussian_score(train_x),
                    "validation_gaussian_qrmse": scores[family],
                }
            )
        best = min(scores.values())
        eligible = [
            family
            for family, score in scores.items()
            if score <= best + SELECTION_TOLERANCE
        ]
        selected = min(eligible, key=complexity.__getitem__)
        transforms[name] = candidates[selected]
        for row in rows[-3:]:
            row["selected"] = row["family"] == selected
    return transforms, pd.DataFrame(rows)


def _forward_matrix(
    values: np.ndarray, transforms: dict[str, dict[str, Any]]
) -> np.ndarray:
    return np.column_stack(
        [
            forward_marginal(values[:, index], transforms[name])
            for index, name in enumerate(NAMES)
        ]
    )


def _inverse_matrix(
    values: np.ndarray, transforms: dict[str, dict[str, Any]]
) -> np.ndarray:
    return np.column_stack(
        [
            inverse_marginal(values[:, index], transforms[name])
            for index, name in enumerate(NAMES)
        ]
    )


def _fit_whitening(values: np.ndarray) -> dict[str, Any]:
    center = np.mean(values, axis=0)
    covariance = np.cov(values, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(eigenvalues)) * 1.0e-8, 1.0e-10)
    clipped = np.maximum(eigenvalues, floor)
    matrix = eigenvectors @ np.diag(1.0 / np.sqrt(clipped)) @ eigenvectors.T
    inverse = eigenvectors @ np.diag(np.sqrt(clipped)) @ eigenvectors.T
    return {
        "center": center.tolist(),
        "matrix": matrix.tolist(),
        "inverse_matrix": inverse.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "eigenvalue_floor": floor,
    }


def _whiten(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    return (values - np.asarray(spec["center"])) @ np.asarray(spec["matrix"])


def _unwhiten(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    return values @ np.asarray(spec["inverse_matrix"]) + np.asarray(spec["center"])


def _affine_matrix(values: np.ndarray, fit: np.ndarray) -> np.ndarray:
    center = np.mean(fit, axis=0)
    scale = np.std(fit, axis=0)
    return (values - center) / scale


def _effective_rank(covariance: np.ndarray) -> float:
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    probabilities = eigenvalues / np.sum(eigenvalues)
    probabilities = probabilities[probabilities > 0.0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _stage_metrics(values: np.ndarray, projections: np.ndarray) -> dict[str, float]:
    covariance = np.cov(values, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    marginal_scores = np.array(
        [_gaussian_score(values[:, index]) for index in range(values.shape[1])]
    )
    projection_scores = np.array(
        [_gaussian_score(values @ vector) for vector in projections]
    )
    radius_squared = np.sum(values**2, axis=1)
    chi_quantiles = chi2.ppf(SCORE_PROBABILITIES, values.shape[1])
    radius_quantiles = np.quantile(radius_squared, SCORE_PROBABILITIES)
    return {
        "marginal_qrmse_mean": float(np.mean(marginal_scores)),
        "marginal_qrmse_p95": float(np.quantile(marginal_scores, 0.95)),
        "marginal_qrmse_max": float(np.max(marginal_scores)),
        "mean_abs_skew": float(np.mean(np.abs(skew(values, axis=0, bias=False)))),
        "mean_abs_excess_kurtosis": float(
            np.mean(np.abs(kurtosis(values, axis=0, fisher=True, bias=False)))
        ),
        "covariance_condition": float(
            np.max(eigenvalues) / max(np.min(eigenvalues), 1.0e-15)
        ),
        "covariance_frobenius_from_identity": float(
            np.linalg.norm(covariance - np.eye(values.shape[1]), ord="fro")
        ),
        "covariance_effective_rank": _effective_rank(covariance),
        "projection_qrmse_mean": float(np.mean(projection_scores)),
        "projection_qrmse_p95": float(np.quantile(projection_scores, 0.95)),
        "projection_qrmse_max": float(np.max(projection_scores)),
        "chi2_radius_relative_qrmse": float(
            np.sqrt(np.mean(((radius_quantiles - chi_quantiles) / chi_quantiles) ** 2))
        ),
        "fraction_abs_gt_5": float(np.mean(np.abs(values) > 5.0)),
        "maximum_abs": float(np.max(np.abs(values))),
    }


def _duplicate_audit(
    exact_train: pd.DataFrame,
    train: pd.DataFrame,
    exact_test: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    exact_train_hash = _row_hash(exact_train)
    train_hash = _row_hash(train)
    exact_test_hash = _row_hash(exact_test)
    test_hash = _row_hash(test)
    rows = []
    for split, hashes in (
        ("exact_train", exact_train_hash),
        ("dequantized_train", train_hash),
        ("exact_test", exact_test_hash),
        ("dequantized_test", test_hash),
    ):
        _, counts = np.unique(hashes, return_counts=True)
        rows.append(
            {
                "split": split,
                "rows": len(hashes),
                "unique_truths": len(counts),
                "duplicate_excess_rows": int(len(hashes) - len(counts)),
                "fraction_rows_in_repeated_groups": float(
                    np.sum(counts[counts > 1]) / len(hashes)
                ),
                "maximum_multiplicity": int(np.max(counts)),
                "truths_repeated": int(np.sum(counts > 1)),
            }
        )
    for split, left, right in (
        ("exact_test_in_train", exact_test_hash, exact_train_hash),
        ("dequantized_test_in_train", test_hash, train_hash),
    ):
        overlap = np.isin(left, np.unique(right))
        rows.append(
            {
                "split": split,
                "rows": len(left),
                "unique_truths": int(np.sum(~overlap)),
                "duplicate_excess_rows": int(np.sum(overlap)),
                "fraction_rows_in_repeated_groups": float(np.mean(overlap)),
                "maximum_multiplicity": np.nan,
                "truths_repeated": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _feature_metrics(
    exact_train: np.ndarray,
    train: np.ndarray,
    fit: np.ndarray,
    validation: np.ndarray,
    full_test: np.ndarray,
    novel_test: np.ndarray,
    transforms: dict[str, dict[str, Any]],
    asinh_transforms: dict[str, dict[str, Any]],
    shifted_asinh_transforms: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(NAMES):
        spec = transforms[name]
        asinh_spec = asinh_transforms[name]
        shifted_asinh_spec = shifted_asinh_transforms[name]
        train_support = fit[:, index]
        full_test_values = full_test[:, index]
        test_values = novel_test[:, index]
        fit_x = forward_marginal(fit[:, index], spec)
        full_test_x = forward_marginal(full_test_values, spec)
        test_x = forward_marginal(test_values, spec)
        inverse = inverse_marginal(test_x, spec)
        scale = max(float(np.ptp(test_values)), 1.0)
        rows.append(
            {
                "parameter": name,
                "selected_family": spec["family"],
                "exact_zero_fraction": float(np.mean(exact_train[:, index] == 0.0)),
                "dequantized_zero_fraction": float(np.mean(train[:, index] == 0.0)),
                "near_zero_fraction_abs_lt_1e-4": float(
                    np.mean(np.abs(train[:, index]) < 1.0e-4)
                ),
                "unique_fraction": float(len(np.unique(train[:, index])) / len(train)),
                "raw_skew": float(skew(train[:, index], bias=False)),
                "raw_excess_kurtosis": float(
                    kurtosis(train[:, index], fisher=True, bias=False)
                ),
                "affine_validation_qrmse": _gaussian_score(
                    _affine_matrix(validation, fit)[:, index]
                ),
                "asinh_validation_qrmse": _gaussian_score(
                    forward_marginal(validation[:, index], asinh_spec)
                ),
                "shifted_asinh_validation_qrmse": _gaussian_score(
                    forward_marginal(validation[:, index], shifted_asinh_spec)
                ),
                "selected_validation_qrmse": _gaussian_score(
                    forward_marginal(validation[:, index], spec)
                ),
                "asinh_full_test_qrmse": _gaussian_score(
                    forward_marginal(full_test_values, asinh_spec)
                ),
                "shifted_asinh_full_test_qrmse": _gaussian_score(
                    forward_marginal(full_test_values, shifted_asinh_spec)
                ),
                "selected_full_test_qrmse": _gaussian_score(full_test_x),
                "asinh_novel_test_qrmse": _gaussian_score(
                    forward_marginal(test_values, asinh_spec)
                ),
                "shifted_asinh_novel_test_qrmse": _gaussian_score(
                    forward_marginal(test_values, shifted_asinh_spec)
                ),
                "shifted_asinh_tail_fragile": bool(shifted_asinh_spec["tail_fragile"]),
                "selected_novel_test_qrmse": _gaussian_score(test_x),
                "full_test_ks_vs_fit": float(ks_2samp(fit_x, full_test_x).statistic),
                "novel_test_ks_vs_fit": float(ks_2samp(fit_x, test_x).statistic),
                "novel_test_outside_fit_support_fraction": float(
                    np.mean(
                        (test_values < np.min(train_support))
                        | (test_values > np.max(train_support))
                    )
                ),
                "novel_test_fraction_abs_normalized_gt_5": float(
                    np.mean(np.abs(test_x) > 5.0)
                ),
                "inverse_max_relative_range_error": float(
                    np.max(np.abs(inverse - test_values)) / scale
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_marginals(
    names: tuple[str, ...],
    exact: np.ndarray,
    dequantized: np.ndarray,
    asinh_normalized: np.ndarray,
    selected_normalized: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(names), 4, figsize=(19, 2.35 * len(names)), constrained_layout=True
    )
    normal_grid = np.linspace(-4.5, 4.5, 400)
    normal_density = np.exp(-0.5 * normal_grid**2) / np.sqrt(2.0 * np.pi)
    for row, name in enumerate(names):
        index = NAMES.index(name)
        low, high = np.quantile(dequantized[:, index], [0.001, 0.999])
        for axis, values, title, color in (
            (axes[row, 0], exact[:, index], "Exact scientific truth", "#2f6f9f"),
            (axes[row, 1], dequantized[:, index], "Continuous flow target", "#bb6b3d"),
        ):
            axis.hist(
                values,
                bins=70,
                range=(low, high),
                density=True,
                color=color,
                alpha=0.82,
            )
            axis.axvline(0.0, color="0.15", linewidth=0.7, alpha=0.6)
            axis.set_title(title, fontsize=9)
        axes[row, 2].hist(
            asinh_normalized[:, index],
            bins=70,
            range=(-4.5, 4.5),
            density=True,
            color="#7563a8",
            alpha=0.82,
        )
        axes[row, 2].plot(normal_grid, normal_density, color="0.15", linewidth=1.0)
        axes[row, 2].set_title("After optimized asinh", fontsize=9)
        axes[row, 3].hist(
            selected_normalized[:, index],
            bins=70,
            range=(-4.5, 4.5),
            density=True,
            color="#4f8357",
            alpha=0.82,
        )
        axes[row, 3].plot(normal_grid, normal_density, color="0.15", linewidth=1.0)
        axes[row, 3].set_title("After selected hybrid transform", fontsize=9)
        axes[row, 0].set_ylabel(LABELS[name], fontsize=9)
        for axis in axes[row]:
            axis.tick_params(labelsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_correlations(stages: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(
        1, len(stages), figsize=(6 * len(stages), 6), constrained_layout=True
    )
    image = None
    for axis, (name, values) in zip(axes, stages.items(), strict=True):
        image = axis.imshow(
            np.corrcoef(values, rowvar=False), vmin=-1.0, vmax=1.0, cmap="coolwarm"
        )
        axis.set_title(name)
        axis.set_xticks(range(len(NAMES)), NAMES, rotation=90, fontsize=6)
        axis.set_yticks(range(len(NAMES)), NAMES, fontsize=6)
    fig.colorbar(image, ax=axes, shrink=0.72, label="Pearson correlation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_simple_marginals(
    names: tuple[str, ...],
    affine: np.ndarray,
    asinh: np.ndarray,
    shifted_asinh: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(names), 3, figsize=(15, 2.35 * len(names)), constrained_layout=True
    )
    normal_grid = np.linspace(-4.5, 4.5, 400)
    normal_density = np.exp(-0.5 * normal_grid**2) / np.sqrt(2.0 * np.pi)
    columns = (
        (affine, "Base: affine standardization", "#bb6b3d"),
        (asinh, "Optimized asinh", "#7563a8"),
        (shifted_asinh, "Optimized shifted asinh", "#287f75"),
    )
    for row, name in enumerate(names):
        index = NAMES.index(name)
        for column, (values, title, color) in enumerate(columns):
            axis = axes[row, column]
            axis.hist(
                values[:, index],
                bins=70,
                range=(-4.5, 4.5),
                density=True,
                color=color,
                alpha=0.82,
            )
            axis.plot(normal_grid, normal_density, color="0.15", linewidth=1.0)
            axis.set_title(title, fontsize=9)
            axis.tick_params(labelsize=8)
        axes[row, 0].set_ylabel(LABELS[name], fontsize=9)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_simple_scores(feature_metrics: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(feature_metrics))
    fig, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    for offset, column, label, color in (
        (-0.27, "affine_validation_qrmse", "Base affine", "#bb6b3d"),
        (0.0, "asinh_validation_qrmse", "Optimized asinh", "#7563a8"),
        (
            0.27,
            "shifted_asinh_validation_qrmse",
            "Optimized shifted asinh",
            "#287f75",
        ),
    ):
        axis.bar(
            x + offset, feature_metrics[column], width=0.27, label=label, color=color
        )
    axis.axhline(
        0.12,
        color="0.2",
        linestyle="--",
        linewidth=1.0,
        label="QRMSE 0.12 reference",
    )
    axis.set_xticks(x, feature_metrics["parameter"], rotation=70, ha="right")
    axis.set_ylabel("Validation Gaussian quantile RMSE")
    axis.set_title("Simple normalization comparison")
    axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_scores(feature_metrics: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(feature_metrics))
    fig, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    axis.bar(
        x - 0.27,
        feature_metrics["affine_validation_qrmse"],
        width=0.27,
        label="Affine",
        color="#bb6b3d",
    )
    axis.bar(
        x,
        feature_metrics["asinh_validation_qrmse"],
        width=0.27,
        label="Optimized asinh",
        color="#7563a8",
    )
    axis.bar(
        x + 0.27,
        feature_metrics["selected_validation_qrmse"],
        width=0.27,
        label="Selected hybrid",
        color="#4f8357",
    )
    axis.axhline(
        0.12, color="0.2", linestyle="--", linewidth=1.0, label="QRMSE 0.12 reference"
    )
    axis.set_xticks(x, feature_metrics["parameter"], rotation=70, ha="right")
    axis.set_ylabel("Validation Gaussian quantile RMSE")
    axis.set_title("Marginal complexity before and after normalization")
    axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_duplicates(
    audit: pd.DataFrame, exact_train: pd.DataFrame, path: Path
) -> None:
    _, counts = np.unique(_row_hash(exact_train), return_counts=True)
    multiplicity, number = np.unique(counts, return_counts=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(multiplicity, number, color="#2f6f9f")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Multiplicity of one exact 15D truth")
    axes[0].set_ylabel("Number of unique truths")
    axes[0].set_title("Train resampling multiplicities")
    row = audit.set_index("split").loc["exact_test_in_train"]
    axes[1].bar(
        ["Novel test truth", "Already in train"],
        [row["unique_truths"], row["duplicate_excess_rows"]],
        color=["#4f8357", "#bb6b3d"],
    )
    axes[1].set_ylabel("Test rows")
    axes[1].set_title("Truth-level train/test overlap")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.5g}"
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join((header, separator, *rows))


def _write_reports(
    out: Path,
    feature: pd.DataFrame,
    stages: pd.DataFrame,
    candidates: pd.DataFrame,
    duplicates: pd.DataFrame,
    inverse_error: float,
) -> None:
    selected_counts = feature["selected_family"].value_counts().to_dict()
    shifted_fragile = feature.loc[
        feature["shifted_asinh_tail_fragile"], "parameter"
    ].tolist()
    full_test = stages[(stages["split"] == "full_test")].set_index("representation")
    novel = stages[(stages["split"] == "novel_test")].set_index("representation")
    full_marginal = full_test.loc["selected_marginal"]
    full_asinh = full_test.loc["all_asinh"]
    full_shifted_asinh = full_test.loc["all_shifted_asinh"]
    marginal = novel.loc["selected_marginal"]
    novel_asinh = novel.loc["all_asinh"]
    whitened = novel.loc["selected_marginal_plus_whitening"]
    overlap = float(
        duplicates.set_index("split").loc[
            "exact_test_in_train", "fraction_rows_in_repeated_groups"
        ]
    )
    verdict = (
        "Les 15 coordonnées déquantifiées sont utilisables par un flow continu après "
        "normalisation `asinh`, mais plusieurs marginales restent multimodales ou très "
        "asymétriques. L'hybride quantile est nettement plus proche d'une base gaussienne. "
        "Le blanchiment fixe n'est pas recommandé: il réduit les corrélations linéaires, "
        "mais crée des marginales et des queues plus difficiles."
    )
    report = f"""# FENIKS spline 15D: distribution and normalization audit

## Verdict

{verdict}

- Selected marginal families: `{selected_counts}`.
- Full-test maximum marginal Gaussian QRMSE: `{full_asinh["marginal_qrmse_max"]:.4f}` with all-`asinh` versus `{full_marginal["marginal_qrmse_max"]:.4f}` with the selected hybrid.
- Full-test shifted-`asinh` maximum marginal Gaussian QRMSE: `{full_shifted_asinh["marginal_qrmse_max"]:.4f}`; its mean is `{full_shifted_asinh["marginal_qrmse_mean"]:.4f}`.
- Novel-test maximum marginal Gaussian QRMSE: `{novel_asinh["marginal_qrmse_max"]:.4f}` with all-`asinh` versus `{marginal["marginal_qrmse_max"]:.4f}` with the selected hybrid.
- Full-test covariance condition: `{full_asinh["covariance_condition"]:.1f}` with all-`asinh` versus `{full_marginal["covariance_condition"]:.1f}` with the selected hybrid.
- Full-test tail fraction `|x| > 5`: `{full_asinh["fraction_abs_gt_5"]:.3%}` with all-`asinh` versus `{full_marginal["fraction_abs_gt_5"]:.3%}` with the selected hybrid; maxima are `{full_asinh["maximum_abs"]:.2f}` and `{full_marginal["maximum_abs"]:.2f}`.
- Shifted-`asinh` tail-fragile coordinates: `{shifted_fragile}`.
- Novel-test random-projection QRMSE p95: `{marginal["projection_qrmse_p95"]:.4f}` marginal-only versus `{whitened["projection_qrmse_p95"]:.4f}` whitened.
- Novel-test tail fraction `|x| > 5`: `{marginal["fraction_abs_gt_5"]:.3%}` marginal-only versus `{whitened["fraction_abs_gt_5"]:.3%}` whitened.
- Full joint round-trip maximum absolute error: `{inverse_error:.3e}`.

## Critical provenance finding

`{overlap:.1%}` of test rows have an exact 15D scientific truth already present in
train. The novel-test metrics are reported as a conservative leakage audit, but this
subset is distribution-shifted because removing repeated high-weight proposals changes
the target abundance. Full-test metrics remain the relevant IID distribution
diagnostic. The duplicated rows are consistent with Monte Carlo resampling and should
be treated as multiplicities/weights, not as new physical Dirac components. A future
production split should be performed on unique proposal/truth identifiers before
resampling.

## Before and after

### Simple transforms only

![Simple physical normalization comparison](simple_normalizations_physical.png)

![Simple SFH normalization comparison](simple_normalizations_sfh.png)

![Simple normalization scores](simple_normalizations_scores.png)

![Simple normalization correlations](simple_normalizations_correlations.png)

### Quantile-hybrid ablation

![Physical marginals](marginals_physical_before_after.png)

![SFH marginals](marginals_sfh_before_after.png)

![Gaussian scores](marginal_gaussian_scores.png)

![Correlation matrices](correlations_before_after.png)

![Duplicate audit](duplicate_truth_audit.png)

## Joint metrics

{_markdown_table(stages)}

## Per-coordinate metrics

{_markdown_table(feature)}

## Candidate marginal transforms

{_markdown_table(candidates)}

## Duplicate audit

{_markdown_table(duplicates)}

## Recommended data contract

1. Keep the exact parquet as the scientific audit truth.
2. Train the first continuous-NF benchmark from
   `feniks_spline_15d_train_normalized_marginal.parquet`.
3. Use `feniks_spline_15d_train_normalized_asinh.parquet` as the simpler matched
   normalization ablation.
4. Use `feniks_spline_15d_train_normalized_shifted_asinh.parquet` for the
   two-shape-parameter simple-transform benchmark.
5. Invert each marginal transform to recover the five physical parameters and ten
   SFH contrasts. Keep the fixed-whitening file for ablation only.
6. Reconstruct the eleven spline ordinates by cumulative sum of the ten contrasts;
   stellar mass fixes the common SFH amplitude.
7. Rebuild train/validation/test from unique proposal identifiers before the final
   production prior benchmark. Until then, report both IID full-test metrics and the
   conservative, distribution-shifted novel-truth audit.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    tables = {
        "Joint metrics": stages,
        "Per-coordinate metrics": feature,
        "Candidate transforms": candidates,
        "Duplicate audit": duplicates,
    }
    table_html = "".join(
        f"<h2>{title}</h2>{frame.to_html(index=False, float_format=lambda x: f'{x:.5g}')}"
        for title, frame in tables.items()
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>FENIKS spline 15D normalization</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;color:#20252b;max-width:1500px;margin:auto;padding:28px}}h1,h2{{letter-spacing:0}}.verdict{{border-left:5px solid #4f8357;padding:10px 16px;background:#f2f6f2}}.warning{{border-left:5px solid #bb6b3d;padding:10px 16px;background:#fff5ee}}img{{max-width:100%;display:block;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:12px;display:block;overflow:auto}}th,td{{border:1px solid #d5d9dc;padding:5px 7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#eef0f2;padding:2px 4px}}</style></head><body>
<h1>FENIKS spline 15D: distributions before/after normalization</h1><div class="verdict"><strong>Verdict.</strong> {verdict}</div>
<div class="warning"><strong>Provenance.</strong> {overlap:.1%} of test truths already occur in train. Metrics labelled novel test exclude this overlap.</div>
<h2>Simple transform comparison</h2><img src="simple_normalizations_physical.png"><img src="simple_normalizations_sfh.png"><img src="simple_normalizations_scores.png"><img src="simple_normalizations_correlations.png">
<h2>Marginals</h2><img src="marginals_physical_before_after.png"><img src="marginals_sfh_before_after.png">
<h2>Normalization scores</h2><img src="marginal_gaussian_scores.png"><h2>Joint geometry</h2><img src="correlations_before_after.png">
<h2>Duplicate truths</h2><img src="duplicate_truth_audit.png">{table_html}
<h2>Operational files</h2><p><code>normalization_spec.json</code> contains all-<code>asinh</code>, shifted-<code>asinh</code>, selected hybrid transforms, and optional whitening matrices. The corresponding normalized train/test parquets are exported for a matched NF comparison.</p>
</body></html>"""
    (out / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    exact_train = _read(args.exact_train)
    train = _read(args.train)
    exact_test = _read(args.exact_test)
    test = _read(args.test)
    if len(exact_train) != len(train) or len(exact_test) != len(test):
        raise ValueError("Exact and dequantized tables are not row-aligned")

    fit_mask, validation_mask = _group_split(
        exact_train, args.validation_fraction, args.seed
    )
    train_hash = _row_hash(exact_train)
    test_hash = _row_hash(exact_test)
    novel_test_mask = ~np.isin(test_hash, np.unique(train_hash))
    fit = train.to_numpy(float)[fit_mask]
    validation = train.to_numpy(float)[validation_mask]
    test_values = test.to_numpy(float)
    novel_test = test_values[novel_test_mask]

    transforms, candidates = _fit_transforms(fit, validation)
    asinh_transforms = {
        name: _fit_asinh(fit[:, index]) for index, name in enumerate(NAMES)
    }
    shifted_asinh_transforms = {
        name: _fit_shifted_asinh(fit[:, index], args.seed + index)
        for index, name in enumerate(NAMES)
    }
    marginal_fit = _forward_matrix(fit, transforms)
    whitening = _fit_whitening(marginal_fit)
    rng = np.random.default_rng(args.seed)
    projections = rng.normal(size=(64, len(NAMES)))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    split_values = {
        "fit": fit,
        "validation": validation,
        "full_test": test_values,
        "novel_test": novel_test,
    }
    stage_rows = []
    stage_arrays: dict[tuple[str, str], np.ndarray] = {}
    for split, values in split_values.items():
        marginal = _forward_matrix(values, transforms)
        asinh = _forward_matrix(values, asinh_transforms)
        shifted_asinh = _forward_matrix(values, shifted_asinh_transforms)
        representations = {
            "affine": _affine_matrix(values, fit),
            "all_asinh": asinh,
            "all_shifted_asinh": shifted_asinh,
            "selected_marginal": marginal,
            "selected_marginal_plus_whitening": _whiten(marginal, whitening),
        }
        for representation, array in representations.items():
            stage_arrays[(split, representation)] = array
            stage_rows.append(
                {
                    "split": split,
                    "representation": representation,
                    "rows": len(array),
                    **_stage_metrics(array, projections),
                }
            )
    stages = pd.DataFrame(stage_rows)

    full_train_marginal = _forward_matrix(train.to_numpy(float), transforms)
    full_test_marginal = _forward_matrix(test_values, transforms)
    full_train_asinh = _forward_matrix(train.to_numpy(float), asinh_transforms)
    full_test_asinh = _forward_matrix(test_values, asinh_transforms)
    full_train_shifted_asinh = _forward_matrix(
        train.to_numpy(float), shifted_asinh_transforms
    )
    full_test_shifted_asinh = _forward_matrix(test_values, shifted_asinh_transforms)
    asinh_roundtrip = _inverse_matrix(full_test_asinh, asinh_transforms)
    asinh_inverse_error = float(np.max(np.abs(asinh_roundtrip - test_values)))
    shifted_asinh_roundtrip = _inverse_matrix(
        full_test_shifted_asinh, shifted_asinh_transforms
    )
    shifted_asinh_inverse_error = float(
        np.max(np.abs(shifted_asinh_roundtrip - test_values))
    )
    full_train_whitened = _whiten(full_train_marginal, whitening)
    full_test_whitened = _whiten(full_test_marginal, whitening)
    roundtrip = _inverse_matrix(_unwhiten(full_test_whitened, whitening), transforms)
    inverse_error = float(np.max(np.abs(roundtrip - test_values)))
    feature = _feature_metrics(
        exact_train.to_numpy(float),
        train.to_numpy(float),
        fit,
        validation,
        test_values,
        novel_test,
        transforms,
        asinh_transforms,
        shifted_asinh_transforms,
    )
    duplicates = _duplicate_audit(exact_train, train, exact_test, test)

    pd.DataFrame(full_train_marginal, columns=NAMES).to_parquet(
        args.out / "feniks_spline_15d_train_normalized_marginal.parquet", index=False
    )
    pd.DataFrame(full_test_marginal, columns=NAMES).to_parquet(
        args.out / "feniks_spline_15d_test_normalized_marginal.parquet", index=False
    )
    pd.DataFrame(full_train_asinh, columns=NAMES).to_parquet(
        args.out / "feniks_spline_15d_train_normalized_asinh.parquet", index=False
    )
    pd.DataFrame(full_test_asinh, columns=NAMES).to_parquet(
        args.out / "feniks_spline_15d_test_normalized_asinh.parquet", index=False
    )
    pd.DataFrame(full_train_shifted_asinh, columns=NAMES).to_parquet(
        args.out / "feniks_spline_15d_train_normalized_shifted_asinh.parquet",
        index=False,
    )
    pd.DataFrame(full_test_shifted_asinh, columns=NAMES).to_parquet(
        args.out / "feniks_spline_15d_test_normalized_shifted_asinh.parquet",
        index=False,
    )
    pd.DataFrame(
        full_train_whitened, columns=[f"flow_x_{index:02d}" for index in range(1, 16)]
    ).to_parquet(
        args.out / "feniks_spline_15d_train_normalized_whitened.parquet", index=False
    )
    pd.DataFrame(
        full_test_whitened, columns=[f"flow_x_{index:02d}" for index in range(1, 16)]
    ).to_parquet(
        args.out / "feniks_spline_15d_test_normalized_whitened.parquet", index=False
    )
    feature.to_csv(args.out / "feature_metrics.csv", index=False)
    stages.to_csv(args.out / "joint_stage_metrics.csv", index=False)
    candidates.to_csv(args.out / "candidate_transform_metrics.csv", index=False)
    duplicates.to_csv(args.out / "duplicate_truth_audit.csv", index=False)
    pd.DataFrame(
        {
            "row_index": np.arange(len(train)),
            "normalization_split": np.where(fit_mask, "fit", "validation"),
        }
    ).to_csv(args.out / "train_normalization_split.csv", index=False)
    pd.DataFrame(
        {"row_index": np.arange(len(test)), "truth_novel_vs_train": novel_test_mask}
    ).to_csv(args.out / "test_truth_overlap_mask.csv", index=False)

    spec = {
        "version": 1,
        "dimension": len(NAMES),
        "names": list(NAMES),
        "fit_source": str(args.train),
        "fit_rows": int(np.sum(fit_mask)),
        "validation_rows": int(np.sum(validation_mask)),
        "split_rule": "grouped by exact 15D scientific truth; seeded unique-group split",
        "seed": args.seed,
        "marginal_transforms": transforms,
        "all_asinh_transforms": asinh_transforms,
        "all_shifted_asinh_transforms": shifted_asinh_transforms,
        "whitening": whitening,
        "recommended_flow_input": "marginal_transforms_only",
        "whitening_status": "ablation_only; creates heavy mixed-coordinate tails",
        "forward_order": ["marginal_transforms", "whitening"],
        "inverse_order": ["inverse_whitening", "inverse_marginal_transforms"],
        "maximum_test_roundtrip_absolute_error": inverse_error,
        "maximum_asinh_test_roundtrip_absolute_error": asinh_inverse_error,
        "maximum_shifted_asinh_test_roundtrip_absolute_error": (
            shifted_asinh_inverse_error
        ),
    }
    (args.out / "normalization_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )

    _plot_marginals(
        PHYSICAL_NAMES,
        exact_train.to_numpy(float),
        train.to_numpy(float),
        full_train_asinh,
        full_train_marginal,
        args.out / "marginals_physical_before_after.png",
    )
    _plot_marginals(
        SFH_NAMES,
        exact_train.to_numpy(float),
        train.to_numpy(float),
        full_train_asinh,
        full_train_marginal,
        args.out / "marginals_sfh_before_after.png",
    )
    _plot_correlations(
        {
            "Affine standardized": stage_arrays[("fit", "affine")],
            "All optimized asinh": stage_arrays[("fit", "all_asinh")],
            "Selected marginals": stage_arrays[("fit", "selected_marginal")],
            "Marginals + whitening": stage_arrays[
                ("fit", "selected_marginal_plus_whitening")
            ],
        },
        args.out / "correlations_before_after.png",
    )
    _plot_scores(feature, args.out / "marginal_gaussian_scores.png")
    full_train_affine = _affine_matrix(train.to_numpy(float), fit)
    _plot_simple_marginals(
        PHYSICAL_NAMES,
        full_train_affine,
        full_train_asinh,
        full_train_shifted_asinh,
        args.out / "simple_normalizations_physical.png",
    )
    _plot_simple_marginals(
        SFH_NAMES,
        full_train_affine,
        full_train_asinh,
        full_train_shifted_asinh,
        args.out / "simple_normalizations_sfh.png",
    )
    _plot_simple_scores(feature, args.out / "simple_normalizations_scores.png")
    _plot_correlations(
        {
            "Base affine": stage_arrays[("fit", "affine")],
            "Optimized asinh": stage_arrays[("fit", "all_asinh")],
            "Optimized shifted asinh": stage_arrays[("fit", "all_shifted_asinh")],
        },
        args.out / "simple_normalizations_correlations.png",
    )
    _plot_duplicates(duplicates, exact_train, args.out / "duplicate_truth_audit.png")
    _write_reports(args.out, feature, stages, candidates, duplicates, inverse_error)
    print(f"Wrote {args.out / 'report.html'}")
    print(duplicates.to_string(index=False))
    print(stages[stages["split"].eq("novel_test")].to_string(index=False))


if __name__ == "__main__":
    main()
