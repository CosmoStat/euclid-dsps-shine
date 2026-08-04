"""TARP coverage diagnostics for held-out amortized posterior samples.

This module mirrors the ``tarp`` 0.1.1 DRP implementation while keeping the
FENIKS input contract used by :mod:`euclid_dsps.amortized.mira`.  The posterior
files are resolved and ordered by the same helpers, the same first ``N``
sample IDs are selected, and truth min-max normalization uses the same
``1e-8`` range epsilon.  JAX evaluates the distance comparisons on the H100;
the ECP histogram and object bootstrap follow the upstream NumPy algorithm.
"""

from __future__ import annotations

import itertools
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.amortized.mira import (
    FENIKS_SPLINE15D_PARAMETERS,
    _derived_seed,
    _file_record,
    _read_dense_posterior,
    _read_truth,
    _validate_companion_truths,
    feniks_mira_groups,
    resolve_posterior_input,
    resolve_truth_path,
)
from euclid_dsps.io import ensure_dir, write_json

TARP_PAPER_URL = "https://arxiv.org/abs/2302.03026"
TARP_UPSTREAM_URL = "https://github.com/Ciela-Institute/tarp"
TARP_UPSTREAM_VERSION = "0.1.1"
TARP_UPSTREAM_COMMIT = "b40f118d25dbc29cf00f3342be633c536d0464ab"


@jax.jit
def tarp_coverage_values(
    truth: jnp.ndarray,
    posterior: jnp.ndarray,
    references: jnp.ndarray,
) -> jnp.ndarray:
    """Return TARP coverage values with shape ``[models, objects]``.

    ``truth`` is ``[objects, dimensions]``, ``posterior`` is
    ``[models, objects, samples, dimensions]``, and ``references`` is
    ``[objects, dimensions]``.  A coverage value is the fraction of posterior
    samples closer to the random reference point than the truth value, using
    the strict comparison from upstream TARP.  Squared Euclidean distances
    are equivalent to Euclidean distances and avoid an unnecessary square
    root.
    """
    sample_delta = posterior - references[None, :, None, :]
    sample_distance_squared = jnp.sum(sample_delta * sample_delta, axis=-1)
    truth_delta = truth - references
    truth_distance_squared = jnp.sum(truth_delta * truth_delta, axis=-1)
    return jnp.mean(
        sample_distance_squared < truth_distance_squared[None, :, None],
        axis=2,
    )


def evaluate_feniks_tarp(
    *,
    truth_path: str | Path,
    posterior_specs: Sequence[tuple[str, str | Path]],
    out_dir: str | Path,
    num_alpha_bins: int | None = None,
    num_bootstrap: int = 1000,
    samples_per_object: int | None = 128,
    seed: int = 260730,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate TARP and write ECP curves, bands, and provenance artifacts."""
    if num_alpha_bins is not None and num_alpha_bins < 2:
        raise ValueError("num_alpha_bins must be at least 2")
    if num_bootstrap < 0:
        raise ValueError("num_bootstrap must be non-negative")
    if samples_per_object is not None and samples_per_object < 2:
        raise ValueError("samples_per_object must be at least 2")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if not posterior_specs:
        raise ValueError("At least one posterior specification is required")

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out}")
    ensure_dir(out)
    started = time.perf_counter()
    parameters = FENIKS_SPLINE15D_PARAMETERS
    truth_file = resolve_truth_path(truth_path)
    posterior_inputs = [
        resolve_posterior_input(name, source) for name, source in posterior_specs
    ]
    model_names = [item.name for item in posterior_inputs]
    if len(model_names) != len(set(model_names)):
        raise ValueError(f"Posterior model names must be unique: {model_names}")

    initial_manifest = {
        "status": "running",
        "truth_path": str(truth_file),
        "posterior_sources": {
            item.name: [str(path) for path in item.files] for item in posterior_inputs
        },
        "parameters": list(parameters),
        "num_alpha_bins_requested": num_alpha_bins,
        "num_bootstrap": int(num_bootstrap),
        "samples_per_object_requested": samples_per_object,
        "seed": int(seed),
        "limit": limit,
        "paper": TARP_PAPER_URL,
        "upstream_package": f"tarp=={TARP_UPSTREAM_VERSION}",
        "upstream_reference_commit": TARP_UPSTREAM_COMMIT,
        "upstream_url": TARP_UPSTREAM_URL,
    }
    write_json(out / "tarp_manifest.json", initial_manifest)

    truth = _read_truth(truth_file, parameters, limit=limit)
    companion_truths = _validate_companion_truths(
        truth,
        truth_file,
        posterior_inputs,
        parameters,
        limit=limit,
    )
    dense_models = [
        _read_dense_posterior(
            item,
            truth,
            parameters,
            samples_per_object=samples_per_object,
            require_exact_object_set=limit is None,
        )
        for item in posterior_inputs
    ]
    sample_ids = dense_models[0].sample_ids
    if any(model.sample_ids != sample_ids for model in dense_models[1:]):
        raise ValueError("All posterior models must select identical sample IDs")
    sample_counts = {model.values.shape[1] for model in dense_models}
    if len(sample_counts) != 1:
        raise ValueError(
            "All posterior models must use the same number of samples; "
            f"received {sorted(sample_counts)}"
        )
    posterior = np.stack([model.values for model in dense_models], axis=0)
    truth_values = truth.loc[:, parameters].to_numpy(dtype=np.float32)
    lower = truth_values.min(axis=0)
    upper = truth_values.max(axis=0)
    truth_range = upper - lower
    constant = [
        parameters[index]
        for index, value in enumerate(truth_range)
        if not np.isfinite(value) or value <= 0
    ]
    if constant:
        raise ValueError(
            f"Truth normalization has constant/non-finite dimensions: {constant}"
        )
    scale = truth_range + np.float32(1.0e-8)
    truth_normalized = (truth_values - lower) / scale
    posterior_normalized = (posterior - lower[None, None, None, :]) / scale[
        None, None, None, :
    ]
    _write_tarp_normalization_diagnostics(
        out,
        parameters,
        lower,
        upper,
        scale,
        posterior_normalized,
        model_names,
    )

    group_definitions = feniks_mira_groups(parameters)
    resolved_bins = num_alpha_bins or max(2, len(truth) // 10)
    curve_rows: list[dict[str, Any]] = []
    coverage_value_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    object_ids = truth["object_id"].to_numpy()
    row_indices = (
        truth["row_index"].to_numpy()
        if "row_index" in truth
        else np.arange(len(truth), dtype=np.int64)
    )

    for group_index, (group_name, indices) in enumerate(group_definitions.items()):
        reference_seed = _derived_seed(seed, group_index, 41)
        bootstrap_seed = _derived_seed(seed, group_index, 53)
        reference_rng = np.random.RandomState(reference_seed)
        references = reference_rng.uniform(
            0.0,
            1.0,
            size=(len(truth), len(indices)),
        ).astype(np.float32)
        coverage = np.asarray(
            jax.device_get(
                tarp_coverage_values(
                    jnp.asarray(truth_normalized[:, indices]),
                    jnp.asarray(posterior_normalized[..., indices]),
                    jnp.asarray(references),
                )
            ),
            dtype=np.float64,
        )
        bootstrap_ecp, alpha = _tarp_bootstrap_ecp(
            coverage,
            num_alpha_bins=resolved_bins,
            num_bootstrap=num_bootstrap,
            seed=bootstrap_seed,
        )
        group_atc: list[float] = []
        group_bootstrap_atc: list[np.ndarray] = []
        for model_index, model_name in enumerate(model_names):
            ecp, model_alpha = _tarp_ecp(coverage[model_index], resolved_bins)
            if not np.array_equal(model_alpha, alpha):
                raise RuntimeError("TARP alpha grids differ across model outputs")
            model_bootstrap = bootstrap_ecp[:, model_index]
            bootstrap_mean, bootstrap_std, bootstrap_q025, bootstrap_q975 = (
                _bootstrap_curve_summary(model_bootstrap, len(alpha))
            )
            scalars = _tarp_scalar_summary(ecp, alpha)
            bootstrap_atc = (
                np.asarray([_tarp_atc(values, alpha) for values in model_bootstrap])
                if num_bootstrap
                else np.empty(0, dtype=np.float64)
            )
            group_atc.append(scalars["atc"])
            group_bootstrap_atc.append(bootstrap_atc)
            summary_rows.append(
                {
                    "model": model_name,
                    "group": group_name,
                    "dimensions": len(indices),
                    "parameters": ",".join(parameters[index] for index in indices),
                    "num_objects": len(truth),
                    "num_posterior_samples": posterior.shape[2],
                    "num_alpha_bins": resolved_bins,
                    **scalars,
                    **_scalar_bootstrap_summary(bootstrap_atc, "bootstrap_atc_"),
                    "reference_seed": int(reference_seed),
                    "bootstrap_seed": int(bootstrap_seed),
                }
            )
            for alpha_index, alpha_value in enumerate(alpha):
                curve_rows.append(
                    {
                        "model": model_name,
                        "group": group_name,
                        "num_objects": len(truth),
                        "num_posterior_samples": posterior.shape[2],
                        "alpha": float(alpha_value),
                        "ecp": float(ecp[alpha_index]),
                        "ideal_ecp": float(alpha_value),
                        "ecp_minus_ideal": float(ecp[alpha_index] - alpha_value),
                        "bootstrap_mean": float(bootstrap_mean[alpha_index]),
                        "bootstrap_std": float(bootstrap_std[alpha_index]),
                        "bootstrap_q025": float(bootstrap_q025[alpha_index]),
                        "bootstrap_q975": float(bootstrap_q975[alpha_index]),
                        "reference_seed": int(reference_seed),
                        "bootstrap_seed": int(bootstrap_seed),
                    }
                )
            coverage_value_frames.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "group": group_name,
                        "object_id": object_ids,
                        "row_index": row_indices,
                        "coverage_value": coverage[model_index],
                    }
                )
            )

        for first, second in itertools.combinations(range(len(model_names)), 2):
            delta = group_bootstrap_atc[first] - group_bootstrap_atc[second]
            pairwise_rows.append(
                {
                    "group": group_name,
                    "model_a": model_names[first],
                    "model_b": model_names[second],
                    "atc_a_minus_b": float(group_atc[first] - group_atc[second]),
                    **_scalar_bootstrap_summary(delta, "delta_"),
                }
            )

    coverage_frame = pd.DataFrame(curve_rows)
    summary_frame = pd.DataFrame(summary_rows)
    values_frame = pd.concat(coverage_value_frames, ignore_index=True)
    pairwise_frame = pd.DataFrame(
        pairwise_rows,
        columns=[
            "group",
            "model_a",
            "model_b",
            "atc_a_minus_b",
            "delta_mean",
            "delta_std",
            "delta_q025",
            "delta_q975",
        ],
    )
    coverage_frame.to_csv(out / "tarp_coverage.csv", index=False)
    coverage_frame.to_parquet(out / "tarp_coverage.parquet", index=False)
    summary_frame.to_csv(out / "tarp_summary.csv", index=False)
    values_frame.to_parquet(out / "tarp_coverage_values.parquet", index=False)
    pairwise_frame.to_csv(out / "tarp_pairwise_differences.csv", index=False)
    _write_tarp_plot(coverage_frame, out / "tarp_coverage.png")

    elapsed = time.perf_counter() - started
    summary = {
        "status": "complete",
        "companion_truths_checked": len(companion_truths),
        "elapsed_seconds": float(elapsed),
        "models": model_names,
        "num_objects": len(truth),
        "num_posterior_samples": posterior.shape[2],
        "num_alpha_bins": resolved_bins,
        "num_bootstrap": int(num_bootstrap),
        "selected_sample_ids": [_json_scalar(value) for value in sample_ids],
        "score_groups": list(group_definitions),
        "full_15d": summary_frame.loc[summary_frame["group"].eq("full_15d")].to_dict(
            orient="records"
        ),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "outputs": {
            "coverage": "tarp_coverage.parquet",
            "coverage_csv": "tarp_coverage.csv",
            "coverage_values": "tarp_coverage_values.parquet",
            "summary_csv": "tarp_summary.csv",
            "pairwise_differences": "tarp_pairwise_differences.csv",
            "normalization": "tarp_normalization.csv",
            "normalization_diagnostics": "tarp_normalization_diagnostics.csv",
            "plot": "tarp_coverage.png",
        },
    }
    manifest = {
        **initial_manifest,
        **summary,
        "truth_file": _file_record(truth_file),
        "companion_truths": companion_truths,
        "posterior_files": {
            item.name: [_file_record(path) for path in item.files]
            for item in posterior_inputs
        },
        "models": model_names,
        "parameters": list(parameters),
        "num_objects": len(truth),
        "num_posterior_samples": posterior.shape[2],
        "num_alpha_bins": resolved_bins,
        "num_bootstrap": int(num_bootstrap),
        "selected_sample_ids": [_json_scalar(value) for value in sample_ids],
        "sample_selection": "first sorted sample IDs per object",
        "paper": TARP_PAPER_URL,
        "upstream_package": f"tarp=={TARP_UPSTREAM_VERSION}",
        "upstream_reference_commit": TARP_UPSTREAM_COMMIT,
        "upstream_url": TARP_UPSTREAM_URL,
        "reference_distribution": "uniform_unit_hypercube",
        "shared_random_references_across_models": True,
        "distance": "euclidean_squared_equivalent",
        "normalization": {
            "type": "truth_minmax",
            "range_epsilon": 1.0e-8,
            "posterior_clipped": False,
        },
        "bootstrap_unit": "held_out_object",
        "git_sha": _git_sha(),
    }
    write_json(out / "tarp_summary.json", summary)
    write_json(out / "tarp_manifest.json", manifest)
    (out / "DONE").touch()
    return summary


def _tarp_ecp(
    coverage_values: np.ndarray,
    num_alpha_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    histogram, alpha = np.histogram(
        coverage_values,
        density=True,
        bins=num_alpha_bins,
        range=(0.0, 1.0),
    )
    delta_alpha = alpha[1] - alpha[0]
    ecp = np.concatenate(([0.0], np.cumsum(histogram) * delta_alpha))
    return ecp, alpha


def _tarp_bootstrap_ecp(
    coverage_values: np.ndarray,
    *,
    num_alpha_bins: int,
    num_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_models, n_objects = coverage_values.shape
    _, alpha = _tarp_ecp(coverage_values[0], num_alpha_bins)
    if num_bootstrap == 0:
        return np.empty((0, n_models, len(alpha)), dtype=np.float64), alpha
    rng = np.random.RandomState(seed)
    result = np.empty(
        (num_bootstrap, n_models, len(alpha)),
        dtype=np.float64,
    )
    for bootstrap_index in range(num_bootstrap):
        indices = rng.randint(0, n_objects, size=n_objects)
        for model_index in range(n_models):
            result[bootstrap_index, model_index], _ = _tarp_ecp(
                coverage_values[model_index, indices],
                num_alpha_bins,
            )
    return result, alpha


def _tarp_scalar_summary(ecp: np.ndarray, alpha: np.ndarray) -> dict[str, float]:
    differences = ecp - alpha
    try:
        from scipy.stats import kstest

        ks_pvalue = float(kstest(ecp, alpha).pvalue)
    except (ImportError, ValueError):
        ks_pvalue = float("nan")
    return {
        "atc": _tarp_atc(ecp, alpha),
        "ks_pvalue": ks_pvalue,
        "coverage_rmse": float(np.sqrt(np.mean(differences**2))),
        "coverage_max_abs_error": float(np.max(np.abs(differences))),
    }


def _tarp_atc(ecp: np.ndarray, alpha: np.ndarray) -> float:
    midindex = len(alpha) // 2
    delta_alpha = alpha[1] - alpha[0]
    return float(np.sum((ecp[midindex:] - alpha[midindex:]) * delta_alpha))


def _bootstrap_curve_summary(
    values: np.ndarray,
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if values.size == 0:
        empty = np.full(length, np.nan, dtype=np.float64)
        return empty, empty, empty, empty
    return (
        values.mean(axis=0),
        values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(length),
        np.quantile(values, 0.025, axis=0),
        np.quantile(values, 0.975, axis=0),
    )


def _scalar_bootstrap_summary(
    values: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}mean": float("nan"),
            f"{prefix}std": float("nan"),
            f"{prefix}q025": float("nan"),
            f"{prefix}q975": float("nan"),
        }
    return {
        f"{prefix}mean": float(values.mean()),
        f"{prefix}std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}q025": float(np.quantile(values, 0.025)),
        f"{prefix}q975": float(np.quantile(values, 0.975)),
    }


def _write_tarp_normalization_diagnostics(
    out: Path,
    parameters: Sequence[str],
    lower: np.ndarray,
    upper: np.ndarray,
    divisor: np.ndarray,
    posterior_normalized: np.ndarray,
    model_names: Sequence[str],
) -> None:
    pd.DataFrame(
        {
            "parameter": parameters,
            "truth_min": lower,
            "truth_max": upper,
            "truth_range": upper - lower,
            "normalization_divisor": divisor,
        }
    ).to_csv(out / "tarp_normalization.csv", index=False)
    rows: list[dict[str, Any]] = []
    for model_index, model_name in enumerate(model_names):
        values = posterior_normalized[model_index]
        for parameter_index, parameter in enumerate(parameters):
            coordinate = values[..., parameter_index]
            rows.append(
                {
                    "model": model_name,
                    "parameter": parameter,
                    "fraction_below_truth_min": float(np.mean(coordinate < 0.0)),
                    "fraction_above_truth_max": float(np.mean(coordinate > 1.0)),
                    "normalized_min": float(np.min(coordinate)),
                    "normalized_max": float(np.max(coordinate)),
                }
            )
    pd.DataFrame(rows).to_csv(
        out / "tarp_normalization_diagnostics.csv",
        index=False,
    )


def _write_tarp_plot(coverage: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = coverage["group"].drop_duplicates().tolist()
    models = coverage["model"].drop_duplicates().tolist()
    num_objects = (
        int(coverage["num_objects"].iloc[0]) if "num_objects" in coverage else None
    )
    num_samples = (
        int(coverage["num_posterior_samples"].iloc[0])
        if "num_posterior_samples" in coverage
        else None
    )
    n_columns = 3
    n_rows = int(np.ceil(len(groups) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(15, 3.5 * n_rows),
        constrained_layout=False,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.94,
        bottom=0.06,
        hspace=0.38,
        wspace=0.28,
    )
    flat_axes = np.asarray(axes).reshape(-1)
    for group_index, (axis, group) in enumerate(zip(flat_axes, groups, strict=False)):
        for model in models:
            subset = coverage.loc[
                coverage["group"].eq(group) & coverage["model"].eq(model)
            ].sort_values("alpha")
            axis.plot(subset["alpha"], subset["ecp"], label=model)
            axis.fill_between(
                subset["alpha"].to_numpy(),
                subset["bootstrap_q025"].to_numpy(),
                subset["bootstrap_q975"].to_numpy(),
                alpha=0.10,
            )
        axis.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            "k--",
            linewidth=1.1,
            label="ideal ECP = alpha",
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Credibility level alpha")
        axis.set_ylabel("Expected coverage")
        axis.set_title(group.replace("marginal_", ""))
        axis.grid(color="#e6e6e6", linewidth=0.8)
        if group_index == 0:
            axis.legend(frameon=False, fontsize=8)
    for axis in flat_axes[len(groups) :]:
        axis.set_visible(False)
    if num_objects is not None and num_samples is not None:
        figure.text(
            0.5,
            0.96,
            f"Fiducials L = {num_objects:,} | posterior samples/object N = {num_samples:,}",
            ha="center",
            va="top",
        )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
