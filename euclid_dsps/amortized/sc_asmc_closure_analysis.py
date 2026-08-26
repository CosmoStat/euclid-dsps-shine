"""Truth-aware figures and metrics for a frozen SC-ASMC-EM closure."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

METHOD_LABELS = {
    "q0": "q0 sleep",
    "smc_em1": "SMC EM1",
    "q1": "q1 distilled",
    "smc_em2": "SMC EM2",
}
METHOD_COLORS = {
    "q0": "#6b7280",
    "smc_em1": "#0072b2",
    "q1": "#d55e00",
    "smc_em2": "#009e73",
}


def write_closure_analysis(
    out_dir: str | Path,
    *,
    draws: dict[str, np.ndarray],
    truth_selected: np.ndarray,
    truth_selected_catalog: np.ndarray,
    row_indices: np.ndarray,
    object_ids: np.ndarray,
    truth_c0: np.ndarray,
    prior_artifacts: dict[str, Path],
    parameters: Sequence[str],
    observed_covariates: dict[str, np.ndarray],
) -> dict[str, Path]:
    """Write distribution-safe metrics and plots for all closure populations."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = tuple(parameters)
    _validate_draw_contract(draws, truth_selected, names)

    mixture_path = out / "selected_catalog_posterior_mixtures.npz"
    _atomic_npz(
        mixture_path,
        {
            "parameter_names": np.asarray(names),
            "row_index": np.asarray(row_indices, dtype=np.int64),
            "object_id": np.asarray(object_ids).astype(str),
            "truth_selected": np.asarray(truth_selected, dtype=np.float32),
            **{
                f"{method}_theta": np.asarray(values, dtype=np.float32)
                for method, values in draws.items()
            },
        },
    )

    calibration_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    pit_arrays: dict[str, np.ndarray] = {}
    for method, values in draws.items():
        pit = np.mean(values <= truth_selected[:, None, :], axis=1)
        pit_arrays[method] = pit
        mean = np.mean(values, axis=1)
        std = np.std(values, axis=1)
        for index, name in enumerate(names):
            residual = mean[:, index] - truth_selected[:, index]
            safe_std = np.maximum(std[:, index], 1.0e-12)
            bias_rows.append(
                {
                    "method": method,
                    "parameter": name,
                    "mean_bias": float(np.mean(residual)),
                    "median_bias": float(np.median(residual)),
                    "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                    "mean_pull": float(np.mean(residual / safe_std)),
                    "mean_posterior_std": float(np.mean(std[:, index])),
                    "objects": int(len(values)),
                }
            )
            for level in (0.50, 0.68, 0.90, 0.95):
                tail = (1.0 - level) / 2.0
                lower, upper = np.quantile(
                    values[:, :, index], (tail, 1.0 - tail), axis=1
                )
                covered = (truth_selected[:, index] >= lower) & (
                    truth_selected[:, index] <= upper
                )
                calibration_rows.append(
                    {
                        "method": method,
                        "parameter": name,
                        "nominal_coverage": level,
                        "empirical_coverage": float(np.mean(covered)),
                        "coverage_error": float(np.mean(covered) - level),
                        "objects": int(len(values)),
                    }
                )
    calibration_path = out / "coverage_all_methods.csv"
    bias_path = out / "posterior_bias_all_methods.csv"
    pit_path = out / "pit_all_parameters.npz"
    _atomic_csv(calibration_path, calibration_rows)
    _atomic_csv(bias_path, bias_rows)
    _atomic_npz(
        pit_path,
        {"parameter_names": np.asarray(names), **pit_arrays},
    )

    redshift_path = out / "photoz_metrics.csv"
    redshift_rows = _photoz_rows(draws, truth_selected[:, 0], row_indices, object_ids)
    _atomic_csv(redshift_path, redshift_rows)
    redshift_summary_path = out / "photoz_summary.csv"
    _atomic_csv(redshift_summary_path, _photoz_summary(redshift_rows))
    conditional_path = out / "conditional_calibration.csv"
    _atomic_csv(
        conditional_path,
        _conditional_calibration_rows(
            draws, truth_selected, observed_covariates, names
        ),
    )

    priors = _load_prior_populations(prior_artifacts)
    population_path = out / "population_recovery_all_priors.csv"
    _atomic_csv(
        population_path,
        _population_rows(truth_c0, truth_selected_catalog, priors, names),
    )
    joint_path = out / "joint_distribution_metrics.csv"
    _atomic_csv(
        joint_path,
        _joint_distribution_rows(
            draws,
            truth_selected,
            priors,
            truth_c0,
            truth_selected_catalog,
        ),
    )

    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_paths = {
        "mixture_marginals": _plot_mixture_marginals(
            plots / "selected_posterior_mixture_marginals.png",
            draws,
            truth_selected,
            names,
        ),
        "mixture_corner": _plot_corner(
            plots / "selected_posterior_mixture_corner.png",
            {
                method: values.reshape(-1, len(names))
                for method, values in draws.items()
            },
            truth_selected,
            names,
            max_dimensions=5,
        ),
        "population_marginals": _plot_population_marginals(
            plots / "population_parent_and_selected_marginals.png",
            priors,
            truth_c0,
            truth_selected_catalog,
            names,
        ),
        "population_corner": _plot_corner(
            plots / "population_parent_corner.png",
            {name: values["theta"] for name, values in priors.items()},
            truth_c0,
            names,
            max_dimensions=5,
        ),
        "population_selected_corner": _plot_corner(
            plots / "population_selected_corner.png",
            {
                f"beta*{name}": values["selected_theta"]
                for name, values in priors.items()
            },
            truth_selected_catalog,
            names,
            max_dimensions=5,
        ),
        "coverage": _plot_coverage(
            plots / "coverage_15d_all_methods.png", calibration_rows, names
        ),
        "bias": _plot_bias(
            plots / "posterior_bias_and_width_15d.png", bias_rows, names
        ),
        "pit": _plot_pit_grid(plots / "pit_15d_all_methods.png", pit_arrays, names),
        "photoz": _plot_photoz(
            plots / "photoz_inferred_vs_truth.png", draws, truth_selected[:, 0]
        ),
        "photoz_pit": _plot_photoz_pit(plots / "photoz_pit.png", pit_arrays),
    }
    individual_dir = plots / "individual"
    individual_dir.mkdir(exist_ok=True)
    individual_paths = []
    for index in _representative_indices(truth_selected[:, 0], count=8):
        path = individual_dir / f"individual_corner_row_{int(row_indices[index])}.png"
        _plot_individual_corner(path, draws, truth_selected[index], names, index)
        individual_paths.append(path)
        marginal_path = (
            individual_dir
            / f"individual_marginals_15d_row_{int(row_indices[index])}.png"
        )
        _plot_individual_marginals(
            marginal_path, draws, truth_selected[index], names, index
        )
        individual_paths.append(marginal_path)
    individual_manifest = out / "individual_plot_manifest.json"
    _atomic_json(
        individual_manifest,
        {
            "status": "complete",
            "truth_used": True,
            "objects": len(individual_paths) // 2,
            "plots": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in individual_paths
            ],
        },
    )

    return {
        "posterior_mixtures": mixture_path,
        "coverage_all_methods": calibration_path,
        "posterior_bias_all_methods": bias_path,
        "pit_all_parameters": pit_path,
        "photoz_metrics": redshift_path,
        "photoz_summary": redshift_summary_path,
        "conditional_calibration": conditional_path,
        "population_recovery_all_priors": population_path,
        "joint_distribution_metrics": joint_path,
        **plot_paths,
        "individual_plot_manifest": individual_manifest,
    }


def _validate_draw_contract(
    draws: dict[str, np.ndarray], truth: np.ndarray, parameters: Sequence[str]
) -> None:
    if set(draws) != set(METHOD_LABELS):
        raise ValueError("closure requires q0, SMC EM1, q1 and SMC EM2 draws")
    expected = (len(truth), len(parameters))
    for name, values in draws.items():
        if values.ndim != 3 or (values.shape[0], values.shape[2]) != expected:
            raise ValueError(f"invalid dense draw shape for {name}: {values.shape}")


def _photoz_rows(draws, truth, rows, object_ids):
    result = []
    for method, values in draws.items():
        z = values[:, :, 0]
        median = np.median(z, axis=1)
        residual = (median - truth) / (1.0 + truth)
        pit = np.mean(z <= truth[:, None], axis=1)
        q16, q84 = np.quantile(z, (0.16, 0.84), axis=1)
        q025, q975 = np.quantile(z, (0.025, 0.975), axis=1)
        crps = np.mean(np.abs(z - truth[:, None]), axis=1) - 0.5 * np.mean(
            np.abs(z[:, :, None] - z[:, None, :]), axis=(1, 2)
        )
        for index in range(len(truth)):
            result.append(
                {
                    "method": method,
                    "row_index": int(rows[index]),
                    "object_id": str(object_ids[index]),
                    "z_true": float(truth[index]),
                    "z_median": float(median[index]),
                    "delta_z_over_1pz": float(residual[index]),
                    "pit": float(pit[index]),
                    "covered_68": bool(q16[index] <= truth[index] <= q84[index]),
                    "covered_95": bool(q025[index] <= truth[index] <= q975[index]),
                    "width_68": float(q84[index] - q16[index]),
                    "crps": float(crps[index]),
                }
            )
    return result


def _photoz_summary(rows):
    result = []
    for method in METHOD_LABELS:
        selected = [row for row in rows if row["method"] == method]
        residual = np.asarray([row["delta_z_over_1pz"] for row in selected])
        median = float(np.median(residual))
        nmad = float(1.4826 * np.median(np.abs(residual - median)))
        result.append(
            {
                "method": method,
                "objects": len(selected),
                "bias_delta_z_over_1pz": float(np.mean(residual)),
                "median_delta_z_over_1pz": median,
                "nmad": nmad,
                "outlier_fraction_abs_delta_gt_0p15": float(
                    np.mean(np.abs(residual) > 0.15)
                ),
                "coverage_68": float(np.mean([row["covered_68"] for row in selected])),
                "coverage_95": float(np.mean([row["covered_95"] for row in selected])),
                "pit_mean": float(np.mean([row["pit"] for row in selected])),
                "mean_crps": float(np.mean([row["crps"] for row in selected])),
            }
        )
    return result


def _conditional_calibration_rows(draws, truth, covariates, names):
    rows = []
    sources = {"z_true": truth[:, 0], **covariates}
    for covariate, values in sources.items():
        values = np.asarray(values, dtype=np.float64)
        edges = np.unique(
            np.quantile(values[np.isfinite(values)], np.linspace(0, 1, 5))
        )
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            selected = np.isfinite(values) & (values >= lower) & (values <= upper)
            if not np.any(selected):
                continue
            for method, posterior in draws.items():
                for index, parameter in enumerate(names):
                    samples = posterior[selected, :, index]
                    target = truth[selected, index]
                    q16, q84 = np.quantile(samples, (0.16, 0.84), axis=1)
                    rows.append(
                        {
                            "covariate": covariate,
                            "bin_lower": float(lower),
                            "bin_upper": float(upper),
                            "method": method,
                            "parameter": parameter,
                            "objects": int(np.sum(selected)),
                            "coverage_68": float(
                                np.mean((target >= q16) & (target <= q84))
                            ),
                            "pit_mean": float(np.mean(samples <= target[:, None])),
                            "mean_bias": float(
                                np.mean(np.mean(samples, axis=1) - target)
                            ),
                        }
                    )
    return rows


def _load_prior_populations(paths):
    result = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            result[name] = {
                "theta": np.asarray(data["theta"]),
                "selected_theta": np.asarray(data["selected_theta"]),
            }
    return result


def _population_rows(truth_c0, truth_selected, priors, names):
    rows = []
    probabilities = np.linspace(0.0, 1.0, 1001)
    for prior_name, populations in priors.items():
        for population, inferred, truth in (
            ("parent", populations["theta"], truth_c0),
            ("selected", populations["selected_theta"], truth_selected),
        ):
            inferred_q = np.quantile(inferred, probabilities, axis=0)
            truth_q = np.quantile(truth, probabilities, axis=0)
            for index, name in enumerate(names):
                rows.append(
                    {
                        "prior": prior_name,
                        "population": population,
                        "parameter": name,
                        "wasserstein_1d": float(
                            np.mean(np.abs(inferred_q[:, index] - truth_q[:, index]))
                        ),
                        "mean_difference": float(
                            np.mean(inferred[:, index]) - np.mean(truth[:, index])
                        ),
                        "std_ratio": float(
                            np.std(inferred[:, index]) / np.std(truth[:, index])
                        ),
                    }
                )
    return rows


def _joint_distribution_rows(
    draws, truth_posterior, priors, truth_c0, truth_selected_catalog
):
    rows = []
    for method, values in draws.items():
        rows.append(
            _joint_metric_row(
                f"posterior_mixture_{method}",
                values.reshape(-1, values.shape[-1]),
                truth_posterior,
            )
        )
    for prior, values in priors.items():
        rows.append(_joint_metric_row(f"{prior}_parent", values["theta"], truth_c0))
        rows.append(
            _joint_metric_row(
                f"beta_{prior}_selected",
                values["selected_theta"],
                truth_selected_catalog,
            )
        )
    return rows


def _joint_metric_row(label, inferred, truth):
    rng = np.random.default_rng(260826)
    count = min(512, len(inferred), len(truth))
    inferred_index = rng.choice(len(inferred), count, replace=False)
    truth_index = rng.choice(len(truth), count, replace=False)
    x = np.asarray(inferred[inferred_index], dtype=np.float64)
    y = np.asarray(truth[truth_index], dtype=np.float64)
    scale = np.std(np.concatenate((x, y)), axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    x = (x - np.mean(y, axis=0)) / scale
    y = (y - np.mean(y, axis=0)) / scale
    xy = _pairwise_squared_distance(x, y)
    xx = _pairwise_squared_distance(x, x)
    yy = _pairwise_squared_distance(y, y)
    bandwidth = max(float(np.median(xy)), 1.0e-6)
    mmd2 = float(
        np.mean(np.exp(-xx / (2.0 * bandwidth)))
        + np.mean(np.exp(-yy / (2.0 * bandwidth)))
        - 2.0 * np.mean(np.exp(-xy / (2.0 * bandwidth)))
    )
    energy = float(
        2.0 * np.mean(np.sqrt(xy)) - np.mean(np.sqrt(xx)) - np.mean(np.sqrt(yy))
    )
    correlation_rmse = float(
        np.sqrt(
            np.mean(
                np.square(np.corrcoef(x, rowvar=False) - np.corrcoef(y, rowvar=False))
            )
        )
    )
    return {
        "distribution": label,
        "subsample_size": count,
        "standardized_energy_distance": energy,
        "rbf_mmd_squared": max(mmd2, 0.0),
        "correlation_rmse": correlation_rmse,
    }


def _pairwise_squared_distance(first, second):
    result = (
        np.sum(np.square(first), axis=1)[:, None]
        + np.sum(np.square(second), axis=1)[None, :]
        - 2.0 * first @ second.T
    )
    return np.maximum(result, 0.0)


def _plot_mixture_marginals(path, draws, truth, names):
    plt = _plt()
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for index, axis in enumerate(axes.flat):
        axis.hist(
            truth[:, index],
            bins=40,
            density=True,
            histtype="step",
            color="black",
            label="truth selected",
        )
        for method, values in draws.items():
            axis.hist(
                values[:, :, index].ravel(),
                bins=40,
                density=True,
                histtype="step",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(names[index], fontsize=8)
    axes.flat[0].legend(fontsize=6)
    return _save(fig, path)


def _plot_population_marginals(path, priors, truth_c0, truth_selected, names):
    plt = _plt()
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    colors = {"p0": "#6b7280", "p1": "#0072b2", "p2": "#009e73"}
    for index, axis in enumerate(axes.flat):
        axis.hist(
            truth_c0[:, index],
            bins=40,
            density=True,
            histtype="step",
            color="black",
            label="truth C0",
        )
        axis.hist(
            truth_selected[:, index],
            bins=40,
            density=True,
            histtype="step",
            color="#cc79a7",
            label="truth selected",
        )
        for name, values in priors.items():
            axis.hist(
                values["theta"][:, index],
                bins=40,
                density=True,
                histtype="step",
                color=colors[name],
                label=f"{name} parent",
            )
            axis.hist(
                values["selected_theta"][:, index],
                bins=40,
                density=True,
                histtype="step",
                linestyle="--",
                color=colors[name],
                label=f"beta*{name}",
            )
        axis.set_title(names[index], fontsize=8)
    axes.flat[0].legend(fontsize=6)
    return _save(fig, path)


def _plot_corner(path, series, truth, names, max_dimensions):
    plt = _plt()
    count = min(int(max_dimensions), len(names))
    fig, axes = plt.subplots(count, count, figsize=(14, 14))
    rng = np.random.default_rng(42)
    for row in range(count):
        for col in range(count):
            axis = axes[row, col]
            if row < col:
                axis.axis("off")
            elif row == col:
                axis.hist(
                    truth[:, col], bins=35, density=True, histtype="step", color="black"
                )
                for label, values in series.items():
                    axis.hist(
                        values[:, col],
                        bins=35,
                        density=True,
                        histtype="step",
                        label=label,
                    )
            else:
                for label, values in series.items():
                    take = rng.choice(len(values), min(800, len(values)), replace=False)
                    axis.scatter(
                        values[take, col],
                        values[take, row],
                        s=2,
                        alpha=0.12,
                        label=label,
                    )
                take = rng.choice(len(truth), min(800, len(truth)), replace=False)
                axis.scatter(
                    truth[take, col], truth[take, row], s=2, alpha=0.15, color="black"
                )
            if row == count - 1:
                axis.set_xlabel(names[col], fontsize=7)
            if col == 0 and row:
                axis.set_ylabel(names[row], fontsize=7)
    axes[0, 0].legend(fontsize=6)
    return _save(fig, path)


def _plot_coverage(path, rows, names):
    plt = _plt()
    methods = list(METHOD_LABELS)
    levels = (0.50, 0.68, 0.90, 0.95)
    fig, axes = plt.subplots(len(methods), 1, figsize=(16, 10), sharex=True)
    for axis, method in zip(axes, methods, strict=True):
        matrix = np.asarray(
            [
                [
                    next(
                        item["empirical_coverage"]
                        for item in rows
                        if item["method"] == method
                        and item["parameter"] == name
                        and item["nominal_coverage"] == level
                    )
                    - level
                    for name in names
                ]
                for level in levels
            ]
        )
        image = axis.imshow(
            matrix, aspect="auto", vmin=-0.25, vmax=0.25, cmap="coolwarm"
        )
        axis.set_yticks(range(len(levels)), labels=levels)
        axis.set_ylabel(METHOD_LABELS[method], fontsize=8)
    axes[-1].set_xticks(
        range(len(names)), labels=names, rotation=75, ha="right", fontsize=7
    )
    fig.colorbar(image, ax=axes, label="empirical - nominal")
    return _save(fig, path)


def _plot_pit_grid(path, pits, names):
    plt = _plt()
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    bins = np.linspace(0, 1, 16)
    for index, axis in enumerate(axes.flat):
        for method, values in pits.items():
            axis.hist(
                values[:, index],
                bins=bins,
                density=True,
                histtype="step",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.axhline(1.0, color="black", lw=0.8)
        axis.set_title(names[index], fontsize=8)
    axes.flat[0].legend(fontsize=6)
    return _save(fig, path)


def _plot_bias(path, rows, names):
    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    x = np.arange(len(names))
    width = 0.18
    for offset, method in enumerate(METHOD_LABELS):
        selected = {row["parameter"]: row for row in rows if row["method"] == method}
        position = x + (offset - 1.5) * width
        axes[0].bar(
            position,
            [selected[name]["mean_bias"] for name in names],
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        axes[1].bar(
            position,
            [selected[name]["mean_posterior_std"] for name in names],
            width=width,
            color=METHOD_COLORS[method],
        )
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylabel("mean posterior mean - truth")
    axes[1].set_ylabel("mean posterior std")
    axes[1].set_xticks(x, labels=names, rotation=75, ha="right", fontsize=7)
    axes[0].legend(ncol=4, fontsize=7)
    return _save(fig, path)


def _plot_photoz(path, draws, truth):
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    for axis, (method, values) in zip(axes.flat, draws.items(), strict=True):
        estimate = np.median(values[:, :, 0], axis=1)
        axis.hexbin(truth, estimate, gridsize=45, mincnt=1, bins="log")
        limits = (
            float(min(np.min(truth), np.min(estimate))),
            float(max(np.max(truth), np.max(estimate))),
        )
        axis.plot(limits, limits, color="black", lw=1)
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlabel("z true")
        axis.set_ylabel("posterior median z")
    return _save(fig, path)


def _plot_photoz_pit(path, pits):
    plt = _plt()
    fig, axis = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    for method, values in pits.items():
        axis.hist(
            values[:, 0],
            bins=bins,
            density=True,
            histtype="step",
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.axhline(1.0, color="black", lw=1)
    axis.set_xlabel("PIT(z)")
    axis.set_ylabel("density")
    axis.legend()
    return _save(fig, path)


def _plot_individual_corner(path, draws, truth, names, object_index):
    series = {method: values[object_index] for method, values in draws.items()}
    return _plot_corner(path, series, truth[None, :], names, max_dimensions=5)


def _plot_individual_marginals(path, draws, truth, names, object_index):
    plt = _plt()
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for index, axis in enumerate(axes.flat):
        for method, values in draws.items():
            axis.hist(
                values[object_index, :, index],
                bins=30,
                density=True,
                histtype="step",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.axvline(truth[index], color="black", lw=1)
        axis.set_title(names[index], fontsize=8)
    axes.flat[0].legend(fontsize=6)
    return _save(fig, path)


def _representative_indices(redshift, count):
    order = np.argsort(redshift)
    return order[np.linspace(0, len(order) - 1, min(count, len(order)), dtype=int)]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path):
    fig = fig
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    _plt().close(fig)
    return Path(path)


def _atomic_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty closure table")
    temporary = Path(path).with_name(f".{Path(path).name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_npz(path, arrays):
    temporary = Path(path).with_name(f".{Path(path).name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _atomic_json(path, payload):
    temporary = Path(path).with_name(f".{Path(path).name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
