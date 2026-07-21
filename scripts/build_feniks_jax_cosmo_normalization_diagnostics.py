#!/usr/bin/env python3
"""Build mixed-marginal normalization diagnostics for a spline-15D dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from euclid_dsps.prior_learning.spline15d import (
    SPLINE15D_PARAMETER_NAMES,
    fit_log_transforms,
    fit_shifted_asinh_transforms,
    forward_asinh_matrix,
    gaussian_quantile_rmse,
    inverse_asinh_matrix,
)

POSITIVE_LOG_PARAMETERS = ("z_obs", "dust_av")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "outputs/analysis/feniks_jax_cosmo_spline_15d_prior_20260716"
        ),
    )
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def read_matrix(path: Path) -> np.ndarray:
    frame = pd.read_parquet(path, columns=list(SPLINE15D_PARAMETER_NAMES))
    matrix = frame.to_numpy(dtype=np.float64)
    if len(matrix) == 0 or not np.isfinite(matrix).all():
        raise ValueError(f"Spline-15D matrix is empty or non-finite: {path}")
    return matrix


def fit_transforms(train: np.ndarray) -> dict[str, dict[str, float | str]]:
    transforms = fit_shifted_asinh_transforms(train)
    indices = [SPLINE15D_PARAMETER_NAMES.index(name) for name in POSITIVE_LOG_PARAMETERS]
    transforms.update(fit_log_transforms(train[:, indices], POSITIVE_LOG_PARAMETERS))
    return transforms


def plot_before_after(
    train: np.ndarray,
    test: np.ndarray,
    train_normalized: np.ndarray,
    test_normalized: np.ndarray,
    transforms: dict[str, dict[str, float | str]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(len(SPLINE15D_PARAMETER_NAMES), 2, figsize=(13, 42))
    gaussian_x = np.linspace(-5.0, 5.0, 600)
    gaussian_y = np.exp(-0.5 * gaussian_x**2) / np.sqrt(2.0 * np.pi)
    for index, name in enumerate(SPLINE15D_PARAMETER_NAMES):
        physical_axis, normalized_axis = axes[index]
        physical_axis.hist(
            train[:, index], bins=70, density=True, alpha=0.65, label="train"
        )
        physical_axis.hist(
            test[:, index], bins=70, density=True, histtype="step", label="test"
        )
        physical_axis.set_title(f"{name} - physical JAX-COSMO projection", fontsize=9)
        normalized_axis.hist(
            train_normalized[:, index],
            bins=70,
            density=True,
            alpha=0.55,
            label="train",
        )
        normalized_axis.hist(
            test_normalized[:, index],
            bins=70,
            density=True,
            histtype="step",
            label="test",
        )
        normalized_axis.plot(
            gaussian_x, gaussian_y, color="black", lw=1.2, label="N(0,1)"
        )
        transform = transforms[name]
        family = str(transform["family"])
        details = ""
        if family == "shifted_asinh":
            details = (
                f" | location={float(transform['location']):.4g}"
                f" | lambda={float(transform['lambda']):.4g}"
            )
        normalized_axis.set_title(
            f"after {family}{details} | "
            f"test QRMSE={gaussian_quantile_rmse(test_normalized[:, index]):.3f}",
            fontsize=9,
        )
        physical_axis.legend(fontsize=7)
        normalized_axis.legend(fontsize=7)
        physical_axis.grid(alpha=0.18)
        normalized_axis.grid(alpha=0.18)
    fig.suptitle(
        "JAX-COSMO spline-15D distributions before and after normalization",
        fontsize=16,
        y=0.999,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    out = args.out or dataset_dir
    out.mkdir(parents=True, exist_ok=True)
    train = read_matrix(dataset_dir / "feniks_spline_15d_train.parquet")
    test = read_matrix(dataset_dir / "feniks_spline_15d_test.parquet")
    transforms = fit_transforms(train)
    train_normalized = forward_asinh_matrix(train, transforms)
    test_normalized = forward_asinh_matrix(test, transforms)
    test_roundtrip = inverse_asinh_matrix(test_normalized, transforms)
    roundtrip_max_abs = float(np.max(np.abs(test_roundtrip - test)))
    if roundtrip_max_abs > 1.0e-10:
        raise ValueError(f"Normalization roundtrip failed: {roundtrip_max_abs:.4g}")

    pd.DataFrame.from_dict(transforms, orient="index").rename_axis(
        "parameter"
    ).to_csv(out / "normalization_parameters.csv")
    payload = {
        "version": 3,
        "family": "mixed_log_shifted_asinh",
        "fit_split": "train",
        "fit_rows": len(train),
        "test_rows": len(test),
        "source_dataset": str(dataset_dir),
        "spline_contract": (
            "JAX-COSMO InterpolatedUnivariateSpline k=3 not-a-knot in "
            "log cosmic time and log SFR"
        ),
        "parameter_names": list(SPLINE15D_PARAMETER_NAMES),
        "positive_log_parameters": list(POSITIVE_LOG_PARAMETERS),
        "whitening": None,
        "transforms": transforms,
        "test_roundtrip_max_abs": roundtrip_max_abs,
        "train_normalized_mean_abs_max": float(
            np.max(np.abs(np.mean(train_normalized, axis=0)))
        ),
        "train_normalized_std_max_abs_error": float(
            np.max(np.abs(np.std(train_normalized, axis=0) - 1.0))
        ),
    }
    (out / "normalization.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    plot_before_after(
        train,
        test,
        train_normalized,
        test_normalized,
        transforms,
        out / "normalization_before_after.png",
    )
    print(f"Wrote JAX-COSMO normalization diagnostics to {out}")
    print(f"Test roundtrip max abs error: {roundtrip_max_abs:.3e}")


if __name__ == "__main__":
    main()
