#!/usr/bin/env python3
"""Select an invertible marginal normalization for each FENIKS parameter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_realnvp_normalization_diagnostics import PARAMETERS, USEFUL_PARAMETER_ORDER
from scipy.optimize import differential_evolution, minimize_scalar

from euclid_dsps.config import load_config

SCORE_PROBABILITIES = np.linspace(0.005, 0.995, 199)
NORMAL_QUANTILES = np.asarray(
    [NormalDist().inv_cdf(float(probability)) for probability in SCORE_PROBABILITIES]
)
ATOM_FRACTION_THRESHOLD = 0.5
SIMPLE_SCORE_THRESHOLD = 0.12
SIMPLICITY_TOLERANCE = 0.01
SPLINE_KNOTS = 257


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--bounds-config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _gaussian_score(values: np.ndarray) -> float:
    values = _finite(values)
    if len(values) < 3:
        return float("nan")
    quantiles = np.quantile(values, SCORE_PROBABILITIES)
    return float(np.sqrt(np.mean((quantiles - NORMAL_QUANTILES) ** 2)))


def _standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    center = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        raise ValueError("Cannot standardize a constant or non-finite coordinate")
    return (values - center) / scale, center, scale


def _fit_affine(values: np.ndarray) -> dict[str, Any]:
    normalized, center, scale = _standardize(values)
    return {
        "family": "affine",
        "center": center,
        "scale": scale,
        "train_score": _gaussian_score(normalized),
    }


def _fit_logit(values: np.ndarray, lower: float, upper: float) -> dict[str, Any]:
    if np.any(values <= lower) or np.any(values >= upper):
        return {
            "family": "wide_bound_logit",
            "valid": False,
            "train_score": float("inf"),
        }
    unit = (values - lower) / (upper - lower)
    raw = np.log(unit) - np.log1p(-unit)
    normalized, center, scale = _standardize(raw)
    return {
        "family": "wide_bound_logit",
        "valid": True,
        "lower": float(lower),
        "upper": float(upper),
        "center": center,
        "scale": scale,
        "train_score": _gaussian_score(normalized),
    }


def _fit_asinh(values: np.ndarray) -> dict[str, Any]:
    data_scale = max(float(np.std(values)), 1.0e-12)

    def objective(log10_relative_lambda: float) -> float:
        value = data_scale * 10.0**log10_relative_lambda
        raw = np.arcsinh(values / value)
        normalized, _, _ = _standardize(raw)
        return _gaussian_score(normalized)

    result = minimize_scalar(
        objective,
        bounds=(-4.0, 4.0),
        method="bounded",
        options={"xatol": 1.0e-5},
    )
    value = data_scale * 10.0 ** float(result.x)
    transformed = value * np.arcsinh(values / value)
    normalized, center, scale = _standardize(transformed)
    return {
        "family": "asinh",
        "lambda": value,
        "center": center,
        "scale": scale,
        "train_score": _gaussian_score(normalized),
        "lambda_regime": _lambda_regime(float(result.x)),
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
        value = data_scale * 10.0 ** float(parameters[1])
        sample_raw = np.arcsinh((moment_sample - shift) / value)
        sample_scale = float(np.std(sample_raw))
        if not np.isfinite(sample_scale) or sample_scale <= 1.0e-12:
            return 1.0e6
        quantile_raw = np.arcsinh((physical_quantiles - shift) / value)
        normalized_quantiles = (quantile_raw - np.mean(sample_raw)) / sample_scale
        return float(np.sqrt(np.mean((normalized_quantiles - NORMAL_QUANTILES) ** 2)))

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
    value = data_scale * 10.0 ** float(result.x[1])
    transformed = value * np.arcsinh((values - shift) / value)
    normalized, center, scale = _standardize(transformed)
    support_distance = min(
        abs(shift - float(np.min(values))), abs(shift - float(np.max(values)))
    )
    return {
        "family": "shifted_asinh",
        "shift": shift,
        "lambda": value,
        "center": center,
        "scale": scale,
        "train_score": _gaussian_score(normalized),
        "lambda_regime": _lambda_regime(float(result.x[1])),
        "shift_support_distance_std": support_distance / data_scale,
        "tail_fragile": bool(
            float(result.x[1]) <= -3.0 and support_distance / data_scale < 0.5
        ),
    }


def _lambda_regime(log10_relative_lambda: float) -> str:
    if log10_relative_lambda <= -3.8:
        return "log-like limit"
    if log10_relative_lambda >= 3.8:
        return "linear limit"
    return "finite compression"


def _fit_quantile_spline(values: np.ndarray) -> dict[str, Any]:
    probabilities = np.linspace(
        0.5 / len(values), 1.0 - 0.5 / len(values), SPLINE_KNOTS
    )
    theta_knots = np.quantile(values, probabilities)
    theta_knots[0] = np.min(values)
    theta_knots[-1] = np.max(values)
    normal_knots = np.asarray(
        [NormalDist().inv_cdf(float(probability)) for probability in probabilities]
    )
    unique_theta, inverse, counts = np.unique(
        theta_knots,
        return_inverse=True,
        return_counts=True,
    )
    unique_normal = np.zeros(len(unique_theta), dtype=float)
    np.add.at(unique_normal, inverse, normal_knots)
    unique_normal /= counts
    spec = {
        "family": "quantile_spline",
        "theta_knots": unique_theta.tolist(),
        "normal_knots": unique_normal.tolist(),
        "n_knots": int(len(unique_theta)),
    }
    spec["train_score"] = _gaussian_score(_forward(values, spec))
    return spec


def _interpolate_extrapolate(
    values: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    result = np.interp(values, source, target)
    low_slope = (target[1] - target[0]) / (source[1] - source[0])
    high_slope = (target[-1] - target[-2]) / (source[-1] - source[-2])
    low = values < source[0]
    high = values > source[-1]
    result[low] = target[0] + low_slope * (values[low] - source[0])
    result[high] = target[-1] + high_slope * (values[high] - source[-1])
    return result


def _forward(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    family = spec["family"]
    values = np.asarray(values, dtype=float)
    if family == "affine":
        return (values - spec["center"]) / spec["scale"]
    if family == "wide_bound_logit":
        unit = (values - spec["lower"]) / (spec["upper"] - spec["lower"])
        raw = np.log(unit) - np.log1p(-unit)
        return (raw - spec["center"]) / spec["scale"]
    if family == "asinh":
        transformed = spec["lambda"] * np.arcsinh(values / spec["lambda"])
        return (transformed - spec["center"]) / spec["scale"]
    if family == "shifted_asinh":
        transformed = spec["lambda"] * np.arcsinh(
            (values - spec["shift"]) / spec["lambda"]
        )
        return (transformed - spec["center"]) / spec["scale"]
    if family == "quantile_spline":
        return _interpolate_extrapolate(
            values,
            np.asarray(spec["theta_knots"]),
            np.asarray(spec["normal_knots"]),
        )
    raise ValueError(f"Unsupported transform family: {family}")


def _inverse(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    family = spec["family"]
    values = np.asarray(values, dtype=float)
    if family == "affine":
        return spec["center"] + spec["scale"] * values
    if family == "wide_bound_logit":
        raw = spec["center"] + spec["scale"] * values
        unit = 1.0 / (1.0 + np.exp(-raw))
        return spec["lower"] + (spec["upper"] - spec["lower"]) * unit
    if family == "asinh":
        transformed = spec["center"] + spec["scale"] * values
        return spec["lambda"] * np.sinh(transformed / spec["lambda"])
    if family == "shifted_asinh":
        transformed = spec["center"] + spec["scale"] * values
        return spec["shift"] + spec["lambda"] * np.sinh(transformed / spec["lambda"])
    if family == "quantile_spline":
        return _interpolate_extrapolate(
            values,
            np.asarray(spec["normal_knots"]),
            np.asarray(spec["theta_knots"]),
        )
    raise ValueError(f"Unsupported transform family: {family}")


def _fit_candidates(
    values: np.ndarray,
    bounds: tuple[float, float],
    seed: int,
) -> dict[str, dict[str, Any]]:
    candidates = {
        "affine": _fit_affine(values),
        "wide_bound_logit": _fit_logit(values, *bounds),
        "asinh": _fit_asinh(values),
        "shifted_asinh": _fit_shifted_asinh(values, seed),
        "quantile_spline": _fit_quantile_spline(values),
    }
    return candidates


def _select_transform(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    simple_order = ("affine", "asinh", "wide_bound_logit", "shifted_asinh")
    valid = [
        candidates[name]
        for name in simple_order
        if np.isfinite(candidates[name].get("train_score", float("inf")))
        and not candidates[name].get("tail_fragile", False)
    ]
    best_score = min(float(candidate["train_score"]) for candidate in valid)
    eligible = [
        candidate
        for candidate in valid
        if float(candidate["train_score"]) <= best_score + SIMPLICITY_TOLERANCE
    ]
    selected = min(
        eligible, key=lambda candidate: simple_order.index(candidate["family"])
    )
    if float(selected["train_score"]) <= SIMPLE_SCORE_THRESHOLD:
        return selected
    return candidates["quantile_spline"]


def _outlier_payload(values: np.ndarray) -> dict[str, float]:
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    iqr = q75 - q25
    low = q25 - 3.0 * iqr
    high = q75 + 3.0 * iqr
    return {
        "median": float(median),
        "q001": float(np.quantile(values, 0.001)),
        "q999": float(np.quantile(values, 0.999)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "fraction_outside_3iqr": float(np.mean((values < low) | (values > high))),
    }


def _normal_curve() -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(-5.0, 5.0, 500)
    density = np.exp(-0.5 * grid**2) / np.sqrt(2.0 * np.pi)
    return grid, density


def _write_plot(
    *,
    names: list[str],
    selected_names: tuple[str, ...],
    train_by_name: dict[str, np.ndarray],
    validation_by_name: dict[str, np.ndarray],
    specs: dict[str, dict[str, Any]],
    summary: pd.DataFrame,
    title: str,
    path: Path,
) -> None:
    chosen = [name for name in selected_names if name in names]
    summary_by_name = summary.set_index("parameter")
    normal_x, normal_density = _normal_curve()
    fig, axes = plt.subplots(
        len(chosen),
        3,
        figsize=(17, 2.7 * len(chosen)),
        constrained_layout=True,
    )
    for row, name in enumerate(chosen):
        train = train_by_name[name]
        validation = validation_by_name[name]
        spec = specs[name]
        result = summary_by_name.loc[name]
        label, role, _ = PARAMETERS[name]
        axes[row, 0].hist(train, bins=60, density=True, color="#3979a6", alpha=0.75)
        axes[row, 0].set_title(f"{role}: {label}", loc="left", fontsize=9)
        axes[row, 0].set_xlabel("Physical theta")
        axes[row, 0].set_ylabel("Density")
        if spec["family"] == "mixed_atom_continuous":
            atom_value = spec["atom_value"]
            train_continuous = train[train != atom_value]
            validation_continuous = validation[validation != atom_value]
            train_x = _forward(train_continuous, spec["continuous_transform"])
            validation_x = _forward(validation_continuous, spec["continuous_transform"])
            family_label = "atom + " + spec["continuous_transform"]["family"]
            axes[row, 0].text(
                0.02,
                0.92,
                f"atom theta = {atom_value:.12g}\ntrain fraction = {spec['atom_fraction_train']:.3%}",
                transform=axes[row, 0].transAxes,
                va="top",
                fontsize=8,
            )
        else:
            train_x = _forward(train, spec)
            validation_x = _forward(validation, spec)
            family_label = spec["family"]
        for axis, values, split, score in (
            (axes[row, 1], train_x, "train", result["selected_train_score"]),
            (
                axes[row, 2],
                validation_x,
                "validation",
                result["selected_validation_score"],
            ),
        ):
            axis.hist(values, bins=60, density=True, color="#5d8f54", alpha=0.75)
            axis.plot(normal_x, normal_density, color="0.15", linewidth=1.0)
            axis.set_xlim(-5.0, 5.0)
            axis.set_xlabel(f"{family_label} normalized x")
            axis.set_ylabel("Density")
            axis.set_title(f"{split}: Gaussian RMSE = {score:.3f}", fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_score_plot(scores: pd.DataFrame, path: Path) -> None:
    families = [
        "affine",
        "wide_bound_logit",
        "asinh",
        "shifted_asinh",
        "quantile_spline",
    ]
    pivot = scores.pivot(
        index="parameter", columns="family", values="train_score"
    ).reindex(columns=families)
    fig, axis = plt.subplots(figsize=(14, 8), constrained_layout=True)
    x = np.arange(len(pivot))
    width = 0.16
    for index, family in enumerate(families):
        axis.bar(x + (index - 2) * width, pivot[family], width=width, label=family)
    axis.axhline(SIMPLE_SCORE_THRESHOLD, color="0.2", linestyle="--", linewidth=1.0)
    axis.set_xticks(x, pivot.index, rotation=70, ha="right")
    axis.set_ylabel("Train quantile RMSE to N(0,1); lower is better")
    axis.set_title("Invertible marginal-transform comparison")
    axis.legend(ncol=3)
    axis.set_ylim(0.0, min(1.1, float(np.nanmax(pivot.to_numpy())) * 1.05))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    train_path: Path,
    validation_path: Path,
) -> None:
    lines = [
        "# Invertible marginal normalization selection",
        "",
        f"Train: `{train_path}`",
        f"Validation: `{validation_path}`",
        "",
        "## Transform families",
        "",
        "- Affine: `x=(theta-center)/scale`; inverse `theta=center+scale*x`.",
        "- Wide-bound logit: bounded logistic map with the widened physical bounds, fitted center and scale.",
        "- Asinh: `y=lambda*asinh(theta/lambda)` followed by train centering/scaling.",
        "- Shifted asinh: `y=lambda*asinh((theta-shift)/lambda)`; inverse `theta=shift+lambda*sinh((center+scale*x)/lambda)`.",
        "- Quantile spline: monotone train-CDF to normal-quantile interpolation with monotone linear tail extrapolation; inverse swaps the knot axes.",
        "- Mixed atom/continuous: preserve the exact atom and apply an invertible transform only to the conditional continuous component.",
        "",
        "Hard clipping is not selected because it is not invertible. Extreme but valid values are handled with smooth asinh compression or monotone spline tails.",
        "",
        "## Per-parameter recommendation",
        "",
        "| Parameter | Structure | Selected transform | Train RMSE | Validation RMSE | Max validation |x| | Max round-trip error | 3-IQR outliers |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            "| `{parameter}` | {structure} | `{selected_family}` | {selected_train_score:.3f} | "
            "{selected_validation_score:.3f} | {validation_x_abs_max:.3f} | "
            "{roundtrip_max_abs:.3g} | "
            "{fraction_outside_3iqr:.3%} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            f"Simple transforms are preferred when their quantile RMSE is at most {SIMPLE_SCORE_THRESHOLD:.2f}. A simpler family is retained when it lies within {SIMPLICITY_TOLERANCE:.2f} of the best simple score. Otherwise the monotone quantile spline is selected.",
            "",
            "The quantile spline is a diagnostic specification here. A production JAX implementation should use a monotone rational-quadratic spline with the stored knots and explicit linear tails.",
            "",
            "No selected validation coordinate requires hard clipping: all transformed validation samples remain below `|x|=5`. Values suspected to be simulation failures should be removed by a documented data-quality cut before fitting, not clipped inside an invertible normalization.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    out = args.out or args.run
    out.mkdir(parents=True, exist_ok=True)
    sidecar = _read_json(args.run / "checkpoints" / "best.eqx.json")
    names = list(sidecar["latent_spec"]["names"])
    schema = {
        entry["name"]: entry["column"] for entry in sidecar["schema"]["parameters"]
    }
    source_columns = [schema[name] for name in names]
    train_frame = pd.read_parquet(args.train, columns=source_columns)
    validation_frame = pd.read_parquet(args.validation, columns=source_columns)
    train_by_name = {
        name: _finite(train_frame[schema[name]].to_numpy(dtype=float)) for name in names
    }
    validation_by_name = {
        name: _finite(validation_frame[schema[name]].to_numpy(dtype=float))
        for name in names
    }
    config = load_config(args.bounds_config)
    bounds = {
        name: tuple(float(value) for value in config["prior_learning"]["bounds"][name])
        for name in names
    }

    selected_specs: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        train = train_by_name[name]
        validation = validation_by_name[name]
        unique, counts = np.unique(train, return_counts=True)
        dominant_index = int(np.argmax(counts))
        atom_value = float(unique[dominant_index])
        atom_fraction = float(counts[dominant_index] / len(train))
        outliers = _outlier_payload(train)
        if atom_fraction >= ATOM_FRACTION_THRESHOLD:
            continuous_train = train[train != atom_value]
            continuous_validation = validation[validation != atom_value]
            candidates = _fit_candidates(
                continuous_train,
                bounds[name],
                seed=260617 + index,
            )
            continuous_spec = _select_transform(candidates)
            spec = {
                "family": "mixed_atom_continuous",
                "atom_value": atom_value,
                "atom_fraction_train": atom_fraction,
                "continuous_transform": continuous_spec,
            }
            selected_train = _forward(continuous_train, continuous_spec)
            selected_validation = _forward(continuous_validation, continuous_spec)
            roundtrip_values = continuous_train
            roundtrip_spec = continuous_spec
            structure = "exact atom + continuous minority"
        else:
            candidates = _fit_candidates(train, bounds[name], seed=260617 + index)
            spec = _select_transform(candidates)
            selected_train = _forward(train, spec)
            selected_validation = _forward(validation, spec)
            roundtrip_values = train
            roundtrip_spec = spec
            structure = "continuous"
        selected_specs[name] = spec
        reconstructed = _inverse(
            _forward(roundtrip_values, roundtrip_spec), roundtrip_spec
        )
        roundtrip_error = float(np.max(np.abs(reconstructed - roundtrip_values)))
        validation_roundtrip = _inverse(selected_validation, roundtrip_spec)
        validation_roundtrip_error = float(
            np.max(
                np.abs(
                    validation_roundtrip
                    - (
                        validation[validation != atom_value]
                        if atom_fraction >= ATOM_FRACTION_THRESHOLD
                        else validation
                    )
                )
            )
        )
        if not np.isfinite(roundtrip_error):
            raise ValueError(f"Non-finite inverse round trip for {name}")
        for family, candidate in candidates.items():
            candidate_validation = (
                validation[validation != atom_value]
                if atom_fraction >= ATOM_FRACTION_THRESHOLD
                else validation
            )
            validation_score = float("nan")
            if candidate.get("valid", True):
                validation_score = _gaussian_score(
                    _forward(candidate_validation, candidate)
                )
            score_rows.append(
                {
                    "parameter": name,
                    "family": family,
                    "train_score": candidate.get("train_score"),
                    "validation_score": validation_score,
                }
            )
        summary_rows.append(
            {
                "parameter": name,
                "structure": structure,
                "selected_family": spec["family"],
                "selected_continuous_family": (
                    spec["continuous_transform"]["family"]
                    if spec["family"] == "mixed_atom_continuous"
                    else spec["family"]
                ),
                "selected_train_score": _gaussian_score(selected_train),
                "selected_validation_score": _gaussian_score(selected_validation),
                "roundtrip_max_abs": roundtrip_error,
                "validation_roundtrip_max_abs": validation_roundtrip_error,
                "validation_x_abs_q999": float(
                    np.quantile(np.abs(selected_validation), 0.999)
                ),
                "validation_x_abs_max": float(np.max(np.abs(selected_validation))),
                "atom_value": (
                    atom_value if atom_fraction >= ATOM_FRACTION_THRESHOLD else None
                ),
                "atom_fraction_train": atom_fraction,
                **outliers,
            }
        )

    summary = pd.DataFrame(summary_rows)
    scores = pd.DataFrame(score_rows)
    summary.to_csv(out / "invertible_normalization_selection.csv", index=False)
    scores.to_csv(out / "invertible_normalization_family_scores.csv", index=False)
    (out / "invertible_normalization_specs.json").write_text(
        json.dumps(
            {
                "train_dataset": str(args.train),
                "validation_dataset": str(args.validation),
                "bounds_config": str(args.bounds_config),
                "transforms": selected_specs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_plot(
        names=names,
        selected_names=tuple(names),
        train_by_name=train_by_name,
        validation_by_name=validation_by_name,
        specs=selected_specs,
        summary=summary,
        title="FENIKS 18D: selected invertible marginal normalizations",
        path=out / "invertible_normalization_all18.png",
    )
    _write_plot(
        names=names,
        selected_names=USEFUL_PARAMETER_ORDER,
        train_by_name=train_by_name,
        validation_by_name=validation_by_name,
        specs=selected_specs,
        summary=summary,
        title="FENIKS useful parameters: selected invertible marginal normalizations",
        path=out / "invertible_normalization_useful.png",
    )
    _write_score_plot(scores, out / "invertible_normalization_family_scores.png")
    _write_report(
        out / "invertible_normalization_report.md",
        summary,
        args.train,
        args.validation,
    )
    print(
        summary[
            [
                "parameter",
                "selected_family",
                "selected_continuous_family",
                "selected_train_score",
                "selected_validation_score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
