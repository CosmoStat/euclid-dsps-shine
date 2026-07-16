#!/usr/bin/env python3
"""Select a JAX spline node count from held-out FENIKS closure diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z  # noqa: E402
from dsps.sed.stellar_age_weights import (  # noqa: E402
    calc_age_weights_from_sfh_table,
)
from matplotlib import pyplot as plt  # noqa: E402

from euclid_dsps import model as dsps_model  # noqa: E402
from euclid_dsps.config import load_config  # noqa: E402
from euclid_dsps.filters import load_filters  # noqa: E402
from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES  # noqa: E402
from euclid_dsps.prior_learning.spline15d import (  # noqa: E402
    cubic_spline_interpolate_jax,
)
from euclid_dsps.synthetic_diffsky.photometry import (  # noqa: E402
    GROUND_TRUTH_COLUMNS,
    theta_from_truth_frame,
)

ATOM_NAMES = (
    "diffstar_lg_qt",
    "diffstar_qlglgdt",
    "diffstar_lg_drop",
    "diffstar_lg_rejuv",
)
K_VALUES = (6, 8, 10, 12, 16, 20)
GRID_STRATEGIES = ("uniform_log_time", "recent_lookback", "hybrid")
WINDOW_LABELS = ("recent_0_0p1_gyr", "intermediate_0p1_1_gyr", "old_gt_1_gyr")
CORE_METRICS = (
    "sfh_mass_normalized_l1",
    "log_sfh_rmse_dex",
    "max_mass_fraction_abs_error",
    "max_log_mean_sfr_abs_error_dex",
    "age_weight_l1",
    "sed_relative_l1",
    "max_abs_delta_mag",
    "noise_rms",
    "noise_max_abs",
)
GATE_DEFINITIONS = (
    ("log_sfh_rmse_dex", "p95", 0.05),
    ("log_sfh_rmse_dex", "p99", 0.10),
    ("max_mass_fraction_abs_error", "p95", 0.02),
    ("max_mass_fraction_abs_error", "p99", 0.05),
    ("max_log_mean_sfr_abs_error_dex", "p95", 0.05),
    ("max_log_mean_sfr_abs_error_dex", "p99", 0.15),
    ("age_weight_l1", "p95", 0.05),
    ("age_weight_l1", "p99", 0.10),
    ("sed_relative_l1", "p95", 0.01),
    ("sed_relative_l1", "p99", 0.03),
    ("max_abs_delta_mag", "p95", 0.01),
    ("max_abs_delta_mag", "p99", 0.03),
    ("noise_rms", "p95", 0.10),
    ("noise_rms", "p99", 0.30),
)
COLORS = {
    "main_sequence": "#276A9C",
    "quenched": "#B55432",
    "high_ssfr_tail": "#28735A",
    "threshold": "#273136",
    "uniform_log_time": "#276A9C",
    "recent_lookback": "#B55432",
    "hybrid": "#28735A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("Data/diffsky/synthetic/feniks_260617_dsps_closure_18band"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/analysis/feniks_spline_node_scan_20260710"),
    )
    parser.add_argument("--seed", type=int, default=260710)
    parser.add_argument("--high-ssfr-quantile", type=float, default=0.90)
    parser.add_argument("--min-group-size", type=int, default=15)
    return parser.parse_args()


def _mode(values: pd.Series) -> float:
    counts = values.value_counts(dropna=False)
    if counts.empty:
        raise ValueError(f"Cannot identify a mode for {values.name}")
    return float(counts.index[0])


def _atom_values(train: pd.DataFrame) -> dict[str, float]:
    return {name: _mode(train[GROUND_TRUTH_COLUMNS[name]]) for name in ATOM_NAMES}


def _atom_mask(frame: pd.DataFrame, atom_values: dict[str, float]) -> np.ndarray:
    bits = np.column_stack(
        [
            frame[GROUND_TRUTH_COLUMNS[name]].to_numpy(float) == atom_values[name]
            for name in ATOM_NAMES
        ]
    )
    coherent = np.all(bits, axis=1) | np.all(~bits, axis=1)
    if not np.all(coherent):
        raise ValueError(f"Found {int(np.sum(~coherent))} incoherent atom rows")
    return np.all(bits, axis=1)


def _balanced_test_sample(
    test: pd.DataFrame,
    atom_values: dict[str, float],
    *,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    atom = _atom_mask(test, atom_values)
    main_indices = np.flatnonzero(atom)
    quenched_indices = np.flatnonzero(~atom)
    rng = np.random.default_rng(seed)
    selected_main = rng.choice(main_indices, len(quenched_indices), replace=False)
    selected = np.concatenate((selected_main, quenched_indices))
    rng.shuffle(selected)
    sample = test.iloc[selected].copy().reset_index(names="test_row")
    sample_atom = _atom_mask(sample, atom_values)
    sample["state"] = np.where(sample_atom, "main_sequence", "quenched")
    prevalence = float(np.mean(atom))
    sample["population_weight"] = np.where(
        sample_atom,
        prevalence / max(int(np.sum(sample_atom)), 1),
        (1.0 - prevalence) / max(int(np.sum(~sample_atom)), 1),
    )
    return sample, prevalence


def _quantile_labels(
    values: np.ndarray, reference: np.ndarray, prefix: str
) -> tuple[np.ndarray, list[float]]:
    edges = np.quantile(reference[np.isfinite(reference)], [0.0, 0.25, 0.5, 0.75, 1.0])
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = np.asarray(
        [f"{prefix}_q{index + 1}" for index in np.digitize(values, edges[1:-1])],
        dtype=object,
    )
    return labels, [float(value) for value in edges[1:-1]]


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[finite]
    weights = weights[finite]
    if not len(values):
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    return float(np.interp(quantile, cdf, values))


def _build_forward_functions(context: dsps_model.DspsContext):
    names = tuple(DIFFSKY_BASIC_PARAMETER_NAMES)
    ssp_lg_age_gyr = dsps_model._context_ssp_lg_age_gyr(context)

    def params_from_theta(theta: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return {name: theta[index] for index, name in enumerate(names)}

    def render_from_age_weights(
        theta: jnp.ndarray, age_weights: jnp.ndarray, formed_mass: jnp.ndarray
    ):
        params = params_from_theta(theta)
        model_config = dsps_model._normalized_model_config(context.model_config)
        z_obs = jnp.asarray(params["z_obs"], dtype=jnp.float32)
        lgmet_abs = dsps_model.log10_stellar_metallicity_to_absolute_jax(
            params["log10_stellar_metallicity"], context.z_sun
        )
        ssp_flux_z = dsps_model.diffsky_basic_ssp_flux_by_age_jax(
            context, model_config, lgmet_abs
        )
        weights = age_weights / jnp.maximum(jnp.sum(age_weights), 1.0e-30)
        sed_by_age = jnp.clip(ssp_flux_z, 0.0, jnp.inf) * weights[:, None] * formed_mass
        tau2, dust_index_n, tau1_over_tau2 = dsps_model.diffsky_basic_dust_params_jax(
            params
        )
        wave = dsps_model._context_ssp_wave(context)
        dusted_by_age = dsps_model.apply_popcosmos_dust_by_age_jax(
            wave,
            ssp_lg_age_gyr,
            sed_by_age,
            tau2,
            dust_index_n,
            tau1_over_tau2,
            model_config,
        )
        dusted_sed = jnp.nan_to_num(
            dusted_by_age.sum(axis=0), nan=0.0, posinf=1.0e30, neginf=0.0
        )
        _, post_igm_sed = dsps_model.combine_agn_and_igm_jax(
            wave,
            dusted_sed,
            jnp.zeros_like(dusted_sed),
            z_obs,
            model_config,
        )
        mags = dsps_model.predict_mags_jax(context, wave, post_igm_sed, z_obs)
        return mags, post_igm_sed

    def sfh_single(theta: jnp.ndarray, raw_sfr: jnp.ndarray):
        params = params_from_theta(theta)
        z_obs = jnp.asarray(params["z_obs"], dtype=jnp.float32)
        t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]
        t_table = jnp.linspace(0.05, jnp.maximum(t_obs, 0.06), context.n_sfh_bins)
        model_config = dsps_model._normalized_model_config(context.model_config)
        lgmet_abs = dsps_model.log10_stellar_metallicity_to_absolute_jax(
            params["log10_stellar_metallicity"], context.z_sun
        )
        surviving = dsps_model._diffsky_basic_surviving_mstar_by_age_jax(
            context, model_config, lgmet_abs
        )
        scaled_sfr, formed_mass, surviving_mass = (
            dsps_model.normalize_sfh_to_stellar_mass_jax(
                t_table,
                raw_sfr,
                ssp_lg_age_gyr,
                t_obs,
                params["log10_stellar_mass"],
                surviving,
            )
        )
        age_weights = calc_age_weights_from_sfh_table(
            t_table, scaled_sfr, ssp_lg_age_gyr, t_obs
        )
        mags, sed = render_from_age_weights(theta, age_weights, formed_mass)
        return (
            t_table,
            scaled_sfr,
            age_weights,
            formed_mass,
            surviving_mass,
            mags,
            sed,
        )

    def native_sfh_single(theta: jnp.ndarray):
        params = params_from_theta(theta)
        z_obs = jnp.asarray(params["z_obs"], dtype=jnp.float32)
        t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]
        t_table = jnp.linspace(0.05, jnp.maximum(t_obs, 0.06), context.n_sfh_bins)
        return dsps_model.build_diffsky_basic_sfh_table_jax(t_table, t_obs, params)

    return (
        jax.jit(jax.vmap(native_sfh_single, in_axes=0)),
        jax.jit(jax.vmap(sfh_single, in_axes=(0, 0))),
    )


def _build_spline_approximator(n_nodes: int, grid_strategy: str):
    if grid_strategy not in GRID_STRATEGIES:
        raise ValueError(f"Unsupported spline grid strategy: {grid_strategy}")

    recent_fractions = jnp.concatenate(
        (jnp.zeros(1), jnp.geomspace(1.0e-3, 1.0, n_nodes - 1))
    )
    n_log_nodes = max(3, int(np.ceil(0.65 * n_nodes)))
    n_recent_nodes = n_nodes - n_log_nodes
    hybrid_recent_fractions = (
        jnp.geomspace(1.0e-3, 0.5, n_recent_nodes + 2)[1:-1]
        if n_recent_nodes
        else jnp.empty(0)
    )

    def single(time: jnp.ndarray, sfr: jnp.ndarray) -> jnp.ndarray:
        t_obs = time[-1]
        span = jnp.maximum(t_obs - time[0], 1.0e-6)
        log_time = jnp.log10(jnp.maximum(time, 1.0e-6))
        if grid_strategy == "uniform_log_time":
            knot_log_time = jnp.linspace(log_time[0], log_time[-1], n_nodes)
            knot_time = 10**knot_log_time
        elif grid_strategy == "recent_lookback":
            knot_time = jnp.flip(t_obs - recent_fractions * span)
            knot_log_time = jnp.log10(jnp.maximum(knot_time, 1.0e-6))
        else:
            log_grid = jnp.geomspace(time[0], t_obs, n_log_nodes)
            recent_grid = t_obs - hybrid_recent_fractions * span
            knot_time = jnp.sort(jnp.concatenate((log_grid, recent_grid)))
            knot_log_time = jnp.log10(jnp.maximum(knot_time, 1.0e-6))
        log_sfr = jnp.log10(jnp.maximum(sfr, 1.0e-30))
        knot_log_sfr = jnp.interp(knot_log_time, log_time, log_sfr)
        spline_log_sfr = cubic_spline_interpolate_jax(
            knot_log_time, knot_log_sfr, log_time
        )
        return 10**spline_log_sfr

    return jax.jit(jax.vmap(single, in_axes=(0, 0)))


def _time_window_summaries(
    time_gyr: np.ndarray, sfr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    dt = np.diff(time_gyr, axis=1)
    segment_sfr = 0.5 * (sfr[:, 1:] + sfr[:, :-1])
    segment_mass = segment_sfr * dt * 1.0e9
    midpoint = 0.5 * (time_gyr[:, 1:] + time_gyr[:, :-1])
    lookback = time_gyr[:, -1, None] - midpoint
    masks = (
        lookback < 0.1,
        (lookback >= 0.1) & (lookback < 1.0),
        lookback >= 1.0,
    )
    masses = np.column_stack(
        [np.sum(np.where(mask, segment_mass, 0.0), axis=1) for mask in masks]
    )
    durations = np.column_stack(
        [np.sum(np.where(mask, dt, 0.0), axis=1) for mask in masks]
    )
    fractions = masses / np.maximum(np.sum(masses, axis=1, keepdims=True), 1.0e-30)
    mean_sfr = masses / np.maximum(durations * 1.0e9, 1.0)
    return fractions, mean_sfr


def _abmag_to_fnu_cgs(mags: np.ndarray) -> np.ndarray:
    return 10 ** (-0.4 * (np.asarray(mags, dtype=float) + 48.6))


def _object_metrics(
    sample: pd.DataFrame,
    native: dict[str, np.ndarray],
    result: dict[str, np.ndarray],
    fluxerr: np.ndarray,
    *,
    n_nodes: int,
    grid_strategy: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dt = np.diff(native["time_gyr"], axis=1)
    sfh_delta = np.abs(result["sfh"] - native["sfh"])
    sfh_l1 = np.sum(
        0.5 * (sfh_delta[:, 1:] + sfh_delta[:, :-1]) * dt, axis=1
    ) / np.maximum(
        np.sum(
            0.5 * (native["sfh"][:, 1:] + native["sfh"][:, :-1]) * dt,
            axis=1,
        ),
        1.0e-30,
    )
    log_sfh_rmse = np.sqrt(
        np.mean(
            (
                np.log10(np.maximum(result["sfh"], 1.0e-12))
                - np.log10(np.maximum(native["sfh"], 1.0e-12))
            )
            ** 2,
            axis=1,
        )
    )
    native_fractions, native_mean_sfr = _time_window_summaries(
        native["time_gyr"], native["sfh"]
    )
    spline_fractions, spline_mean_sfr = _time_window_summaries(
        result["time_gyr"], result["sfh"]
    )
    mass_fraction_error = np.abs(spline_fractions - native_fractions)
    log_mean_sfr_error = np.abs(
        np.log10(np.maximum(spline_mean_sfr, 1.0e-12))
        - np.log10(np.maximum(native_mean_sfr, 1.0e-12))
    )
    age_l1 = np.sum(np.abs(result["age_weights"] - native["age_weights"]), axis=1)
    sed_l1 = np.sum(np.abs(result["sed"] - native["sed"]), axis=1) / np.maximum(
        np.sum(np.abs(native["sed"]), axis=1), 1.0e-30
    )
    delta_mag = result["mags"] - native["mags"]
    abs_delta_mag = np.abs(delta_mag)
    delta_flux_sigma = (
        _abmag_to_fnu_cgs(result["mags"]) - _abmag_to_fnu_cgs(native["mags"])
    ) / np.maximum(fluxerr, 1.0e-40)

    payload: dict[str, Any] = {
        "grid": np.full(len(sample), grid_strategy, dtype=object),
        "K": np.full(len(sample), n_nodes, dtype=int),
        "test_row": sample["test_row"].to_numpy(int),
        "state": sample["state"].to_numpy(str),
        "redshift_bin": sample["redshift_bin"].to_numpy(str),
        "mass_bin": sample["mass_bin"].to_numpy(str),
        "high_ssfr_tail": sample["high_ssfr_tail"].to_numpy(bool),
        "population_weight": sample["population_weight"].to_numpy(float),
        "sfh_mass_normalized_l1": sfh_l1,
        "log_sfh_rmse_dex": log_sfh_rmse,
        "max_mass_fraction_abs_error": np.max(mass_fraction_error, axis=1),
        "max_log_mean_sfr_abs_error_dex": np.max(log_mean_sfr_error, axis=1),
        "age_weight_l1": age_l1,
        "sed_relative_l1": sed_l1,
        "max_abs_delta_mag": np.max(abs_delta_mag, axis=1),
        "noise_rms": np.sqrt(np.mean(delta_flux_sigma**2, axis=1)),
        "noise_max_abs": np.max(np.abs(delta_flux_sigma), axis=1),
        "surviving_mass_log10_error": (
            np.log10(np.maximum(result["surviving_mass"], 1.0e-30))
            - sample[GROUND_TRUTH_COLUMNS["log10_stellar_mass"]].to_numpy(float)
        ),
    }
    for index, label in enumerate(WINDOW_LABELS):
        payload[f"mass_fraction_abs_error_{label}"] = mass_fraction_error[:, index]
        payload[f"log_mean_sfr_abs_error_dex_{label}"] = log_mean_sfr_error[:, index]
    return pd.DataFrame(payload), abs_delta_mag, np.abs(delta_flux_sigma)


def _group_masks(frame: pd.DataFrame) -> dict[str, tuple[str, np.ndarray]]:
    masks: dict[str, tuple[str, np.ndarray]] = {
        "population_weighted": (
            "population",
            np.ones(len(frame), dtype=bool),
        ),
        "balanced_all": ("population", np.ones(len(frame), dtype=bool)),
        "main_sequence": ("state", frame["state"].to_numpy() == "main_sequence"),
        "quenched": ("state", frame["state"].to_numpy() == "quenched"),
        "high_ssfr_tail": ("activity", frame["high_ssfr_tail"].to_numpy(bool)),
    }
    for column, group_type in (("redshift_bin", "redshift"), ("mass_bin", "mass")):
        for value in sorted(frame[column].unique()):
            masks[str(value)] = (group_type, frame[column].to_numpy() == value)
    return masks


def _summarize_metrics(object_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_names = [
        name
        for name in object_metrics.columns
        if name
        not in {
            "K",
            "grid",
            "test_row",
            "state",
            "redshift_bin",
            "mass_bin",
            "high_ssfr_tail",
            "population_weight",
        }
    ]
    for (grid_strategy, n_nodes), frame in object_metrics.groupby(
        ["grid", "K"], sort=True
    ):
        for group, (group_type, mask) in _group_masks(frame).items():
            selected = frame.loc[mask]
            if selected.empty:
                continue
            weights = (
                selected["population_weight"].to_numpy(float)
                if group == "population_weighted"
                else np.ones(len(selected), dtype=float)
            )
            for metric in metric_names:
                values = selected[metric].to_numpy(float)
                rows.append(
                    {
                        "grid": str(grid_strategy),
                        "K": int(n_nodes),
                        "group": group,
                        "group_type": group_type,
                        "metric": metric,
                        "n": len(values),
                        "mean": float(np.average(values, weights=weights)),
                        "median": _weighted_quantile(values, weights, 0.5),
                        "p95": _weighted_quantile(values, weights, 0.95),
                        "p99": _weighted_quantile(values, weights, 0.99),
                        "max": float(np.max(values)),
                    }
                )
    return pd.DataFrame(rows)


def _summarize_bands(
    sample: pd.DataFrame,
    band_names: list[str],
    all_abs_delta_mag: dict[tuple[str, int], np.ndarray],
    all_abs_noise: dict[tuple[str, int], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = {
        "balanced_all": np.ones(len(sample), dtype=bool),
        "main_sequence": sample["state"].to_numpy() == "main_sequence",
        "quenched": sample["state"].to_numpy() == "quenched",
        "high_ssfr_tail": sample["high_ssfr_tail"].to_numpy(bool),
    }
    for grid_strategy in GRID_STRATEGIES:
        for n_nodes in K_VALUES:
            for group, mask in groups.items():
                for band_index, band in enumerate(band_names):
                    for metric, matrix in (
                        (
                            "abs_delta_mag",
                            all_abs_delta_mag[(grid_strategy, n_nodes)],
                        ),
                        (
                            "abs_delta_flux_sigma",
                            all_abs_noise[(grid_strategy, n_nodes)],
                        ),
                    ):
                        values = matrix[mask, band_index]
                        rows.append(
                            {
                                "grid": grid_strategy,
                                "K": n_nodes,
                                "group": group,
                                "band": band,
                                "metric": metric,
                                "n": len(values),
                                "median": float(np.quantile(values, 0.5)),
                                "p95": float(np.quantile(values, 0.95)),
                                "p99": float(np.quantile(values, 0.99)),
                                "max": float(np.max(values)),
                            }
                        )
    return pd.DataFrame(rows)


def _selection_gates(
    summary: pd.DataFrame, *, min_group_size: int
) -> tuple[pd.DataFrame, tuple[str, int] | None, tuple[str, int]]:
    rows: list[dict[str, Any]] = []
    eligible_types = {"state", "activity", "redshift", "mass"}
    for grid_strategy in GRID_STRATEGIES:
        for n_nodes in K_VALUES:
            for metric, statistic, threshold in GATE_DEFINITIONS:
                candidates = summary[
                    (summary["grid"] == grid_strategy)
                    & (summary["K"] == n_nodes)
                    & (summary["metric"] == metric)
                    & (summary["group_type"].isin(eligible_types))
                    & (summary["n"] >= min_group_size)
                ]
                values = candidates[statistic].to_numpy(float)
                worst_index = int(np.nanargmax(values))
                worst = candidates.iloc[worst_index]
                worst_value = float(worst[statistic])
                rows.append(
                    {
                        "grid": grid_strategy,
                        "K": n_nodes,
                        "metric": metric,
                        "statistic": statistic,
                        "threshold": threshold,
                        "worst_value": worst_value,
                        "worst_group": str(worst["group"]),
                        "threshold_ratio": worst_value / threshold,
                        "passed": bool(worst_value <= threshold),
                    }
                )
    gates = pd.DataFrame(rows)
    decisions = (
        gates.groupby(["grid", "K"], as_index=False)
        .agg(
            passed=("passed", "all"),
            failed_gates=("passed", lambda values: int((~values).sum())),
            max_threshold_ratio=("threshold_ratio", "max"),
        )
        .sort_values(["K", "max_threshold_ratio", "grid"])
    )
    passing = decisions[decisions["passed"]]
    selected = (
        (str(passing.iloc[0]["grid"]), int(passing.iloc[0]["K"]))
        if len(passing)
        else None
    )
    best = decisions.sort_values(
        ["failed_gates", "max_threshold_ratio", "K", "grid"]
    ).iloc[0]
    best_scanned = (str(best["grid"]), int(best["K"]))
    return gates, selected, best_scanned


def _plot_gate_curves(gates: pd.DataFrame, path: Path) -> None:
    definitions = list(GATE_DEFINITIONS)
    fig, axes = plt.subplots(4, 4, figsize=(16, 13), constrained_layout=True)
    for axis, (metric, statistic, threshold) in zip(
        axes.flat, definitions, strict=False
    ):
        for grid_strategy in GRID_STRATEGIES:
            selected = gates[
                (gates["grid"] == grid_strategy)
                & (gates["metric"] == metric)
                & (gates["statistic"] == statistic)
            ]
            axis.plot(
                selected["K"],
                selected["worst_value"],
                marker="o",
                color=COLORS[grid_strategy],
                label=grid_strategy,
            )
        axis.axhline(threshold, color=COLORS["threshold"], linestyle="--")
        axis.set_title(f"{metric}\n{statistic}, worst group", fontsize=9)
        axis.set_xticks(K_VALUES)
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(definitions) :]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Spline node-count decision gates", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summary_value(
    summary: pd.DataFrame,
    grid_strategy: str,
    n_nodes: int,
    group: str,
    metric: str,
    statistic: str,
) -> float:
    row = summary[
        (summary["grid"] == grid_strategy)
        & (summary["K"] == n_nodes)
        & (summary["group"] == group)
        & (summary["metric"] == metric)
    ]
    return float(row.iloc[0][statistic])


def _plot_state_tail_curves(
    summary: pd.DataFrame, grid_strategy: str, path: Path
) -> None:
    panels = (
        ("max_abs_delta_mag", "Maximum |delta magnitude|"),
        ("noise_rms", "RMS delta flux / sigma"),
        ("age_weight_l1", "Age-weight L1"),
        ("log_sfh_rmse_dex", "log-SFH RMSE [dex]"),
    )
    groups = ("main_sequence", "quenched", "high_ssfr_tail")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, panels, strict=True):
        for group in groups:
            p95 = [
                _summary_value(summary, grid_strategy, k, group, metric, "p95")
                for k in K_VALUES
            ]
            p99 = [
                _summary_value(summary, grid_strategy, k, group, metric, "p99")
                for k in K_VALUES
            ]
            axis.plot(
                K_VALUES, p95, marker="o", color=COLORS[group], label=f"{group} p95"
            )
            axis.plot(
                K_VALUES,
                p99,
                marker="x",
                linestyle="--",
                color=COLORS[group],
                label=f"{group} p99",
            )
        axis.set_title(label)
        axis.set_xticks(K_VALUES)
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(f"State and high-sSFR tail robustness: {grid_strategy}", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_group_heatmaps(summary: pd.DataFrame, grid_strategy: str, path: Path) -> None:
    group_types = ("redshift", "mass")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for axis, group_type in zip(axes, group_types, strict=True):
        groups = sorted(
            summary.loc[
                (summary["grid"] == grid_strategy)
                & (summary["group_type"] == group_type),
                "group",
            ].unique()
        )
        matrix = np.asarray(
            [
                [
                    _summary_value(
                        summary,
                        grid_strategy,
                        n_nodes,
                        group,
                        "max_abs_delta_mag",
                        "p95",
                    )
                    for n_nodes in K_VALUES
                ]
                for group in groups
            ]
        )
        image = axis.imshow(matrix, aspect="auto", cmap="magma", origin="lower")
        axis.set_xticks(range(len(K_VALUES)), K_VALUES)
        axis.set_yticks(range(len(groups)), groups)
        axis.set_xlabel("Spline nodes K")
        axis.set_title(f"{grid_strategy}: p95 max |delta mag| by {group_type}")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.3g}",
                    ha="center",
                    va="center",
                    color="black"
                    if matrix[row, column] > 0.65 * np.nanmax(matrix)
                    else "white",
                    fontsize=8,
                )
        fig.colorbar(image, ax=axis, label="mag")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_band_heatmap(bands: pd.DataFrame, grid_strategy: str, path: Path) -> None:
    frame = bands[
        (bands["grid"] == grid_strategy)
        & (bands["group"] == "quenched")
        & (bands["metric"] == "abs_delta_mag")
    ]
    band_names = list(frame["band"].drop_duplicates())
    matrix = np.asarray(
        [
            [
                float(
                    frame[(frame["K"] == n_nodes) & (frame["band"] == band)][
                        "p95"
                    ].iloc[0]
                )
                for n_nodes in K_VALUES
            ]
            for band in band_names
        ]
    )
    fig, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", origin="lower")
    axis.set_xticks(range(len(K_VALUES)), K_VALUES)
    axis.set_yticks(range(len(band_names)), band_names)
    axis.set_xlabel("Spline nodes K")
    axis.set_title(f"{grid_strategy}: quenched-state p95 |delta magnitude| by band")
    fig.colorbar(image, ax=axis, label="mag")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _smallest_k_below(
    summary: pd.DataFrame,
    *,
    grid_strategy: str,
    group: str,
    metric: str,
    statistic: str,
    threshold: float,
) -> int | None:
    selected = summary[
        (summary["grid"] == grid_strategy)
        & (summary["group"] == group)
        & (summary["metric"] == metric)
    ].sort_values("K")
    passing = selected[selected[statistic] <= threshold]
    return int(passing.iloc[0]["K"]) if len(passing) else None


def _practical_thresholds(
    summary: pd.DataFrame, grid_strategy: str
) -> dict[str, dict[str, int | None]]:
    groups = (
        "population_weighted",
        "main_sequence",
        "quenched",
        "high_ssfr_tail",
    )
    return {
        group: {
            "p95_max_abs_delta_mag_lt_0p01": _smallest_k_below(
                summary,
                grid_strategy=grid_strategy,
                group=group,
                metric="max_abs_delta_mag",
                statistic="p95",
                threshold=0.01,
            ),
            "p95_noise_rms_lt_0p1": _smallest_k_below(
                summary,
                grid_strategy=grid_strategy,
                group=group,
                metric="noise_rms",
                statistic="p95",
                threshold=0.1,
            ),
        }
        for group in groups
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    def format_value(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.5g}"
        return str(value)

    headers = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    rows.extend(
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def _write_reports(
    out: Path,
    *,
    selected: tuple[str, int] | None,
    best_scanned: tuple[str, int],
    practical_thresholds: dict[str, dict[str, int | None]],
    sample: pd.DataFrame,
    prevalence: float,
    ssfr_threshold: float,
    gates: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    decision = (
        f"{selected[0]} with K={selected[1]} is the smallest configuration passing every gate."
        if selected is not None
        else (
            "No scanned configuration passes every predeclared p95/p99 gate. "
            f"The least-bad scanned configuration is {best_scanned[0]} with K={best_scanned[1]}."
        )
    )
    decision_table = gates.groupby(["grid", "K"], as_index=False).agg(
        passed=("passed", "all"),
        failed_gates=("passed", lambda values: int((~values).sum())),
        max_threshold_ratio=("threshold_ratio", "max"),
    )
    pass_table = _markdown_table(decision_table)
    best_grid = selected[0] if selected is not None else best_scanned[0]
    state_table = summary[
        (summary["grid"] == best_grid)
        & summary["group"].isin(["main_sequence", "quenched", "high_ssfr_tail"])
        & summary["metric"].isin(
            ["log_sfh_rmse_dex", "age_weight_l1", "max_abs_delta_mag", "noise_rms"]
        )
    ][["grid", "K", "group", "metric", "n", "p95", "p99", "max"]]
    state_markdown = _markdown_table(state_table)
    practical_table = _markdown_table(
        pd.DataFrame(
            [
                {"group": group, **thresholds}
                for group, thresholds in practical_thresholds.items()
            ]
        )
    )
    report = f"""# FENIKS JAX spline node-count selection

## Decision

**{decision}**

The scan uses `{len(sample)}` held-out test objects: equal counts of the native
main-sequence atom and continuous quenched branch. Population summaries restore
the test prevalence (`{prevalence:.3%}` main sequence).

{pass_table}

![Decision gates](spline_k_selection_gates.png)

## Practical thresholds on the best grid

{practical_table}

The single population-weighted `0.01 mag` p95 criterion can be reached with a
smaller K than the full worst-group decision. Failure of the quenched row means
that increasing one global fixed grid is inefficient; it motivates a
state-specific or adaptive-knot model rather than automatically adopting K=20
for every galaxy.

## Method

- K values: `{list(K_VALUES)}`.
- Knot grids: `{list(GRID_STRATEGIES)}`, all evaluated by the JAX-COSMO
  not-a-knot cubic kernel
  in log cosmic time and log SFR.
- SFH windows: recent `<0.1 Gyr`, intermediate `0.1-1 Gyr`, old `>1 Gyr`.
- Photometric noise metric: `sqrt(mean_band((f_spline-f_native)^2/sigma_f^2))`.
- High-sSFR tail threshold: `log sSFR >= {ssfr_threshold:.4f}`.
- Selection: worst p95/p99 over state, high-sSFR, redshift-quartile and
  mass-quartile groups with sufficient counts.

## State robustness

{state_markdown}

![State tails](spline_k_state_tail_curves.png)

![Redshift and mass](spline_k_redshift_mass_heatmaps.png)

![Band robustness](spline_k_quenched_band_heatmap.png)

## Limitation: burstiness

The current parquet has no `mc_sfh_type`, burst parameters, stored bursty SFH,
or full generator SSP weights. The high-sSFR tail is a robustness subset, **not**
a Diffsky bursty label. A true bursty-state p95/p99 requires regenerating the
dataset with `mc_sfh_type` and the realized SFH/age weights.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>FENIKS spline K scan</title>
<style>body{{font:15px/1.5 sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#1d2528}}img{{max-width:100%;border:1px solid #d5dbd8;margin:12px 0}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d5dbd8;padding:6px;text-align:right}}th{{background:#edf1ef}}code{{background:#eef1ef;padding:2px 4px}}</style></head><body>
<h1>FENIKS JAX spline node-count selection</h1><p><strong>{decision}</strong></p>
<p>Held-out sample: {len(sample)} objects; main-sequence prevalence restored to {prevalence:.3%} for population metrics.</p>
{decision_table.to_html(index=False)}
<img src="spline_k_selection_gates.png" alt="Decision gates">
<h2>Practical thresholds on the best grid</h2>{pd.DataFrame([{"group": group, **thresholds} for group, thresholds in practical_thresholds.items()]).to_html(index=False)}
<h2>State robustness</h2>{state_table.to_html(index=False, float_format=lambda value: f"{value:.5g}")}
<img src="spline_k_state_tail_curves.png" alt="State tail curves">
<img src="spline_k_redshift_mass_heatmaps.png" alt="Redshift and mass heatmaps">
<img src="spline_k_quenched_band_heatmap.png" alt="Quenched band heatmap">
<h2>Burstiness limitation</h2><p>The current parquet omits the true Diffsky bursty realization. The high-sSFR tail is not a bursty-state label.</p>
</body></html>"""
    (out / "report.html").write_text(html, encoding="utf-8")


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def main() -> None:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(args.dataset_dir / "train.parquet")
    test = pd.read_parquet(args.dataset_dir / "test.parquet")
    atom_values = _atom_values(train)
    sample, prevalence = _balanced_test_sample(test, atom_values, seed=args.seed)

    z_column = GROUND_TRUTH_COLUMNS["z_obs"]
    mass_column = GROUND_TRUTH_COLUMNS["log10_stellar_mass"]
    sample["redshift_bin"], redshift_edges = _quantile_labels(
        sample[z_column].to_numpy(float), test[z_column].to_numpy(float), "redshift"
    )
    sample["mass_bin"], mass_edges = _quantile_labels(
        sample[mass_column].to_numpy(float), test[mass_column].to_numpy(float), "mass"
    )
    ssfr_threshold = float(
        np.quantile(test["logssfr_true"].to_numpy(float), args.high_ssfr_quantile)
    )
    sample["high_ssfr_tail"] = sample["logssfr_true"].to_numpy(float) >= ssfr_threshold

    config = load_config(args.config)
    filters = load_filters(config["bands"])
    band_names = [str(item["name"]) for item in config["bands"]]
    context = dsps_model.load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"]["n_sfh_bins"]),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    theta = theta_from_truth_frame(sample).astype(np.float32)
    theta_jax = jnp.asarray(theta)
    native_sfh_fn, sfh_forward_fn = _build_forward_functions(context)
    raw_native = np.asarray(jax.device_get(native_sfh_fn(theta_jax)))
    native_values = [
        np.asarray(jax.device_get(value))
        for value in sfh_forward_fn(theta_jax, jnp.asarray(raw_native))
    ]
    native = dict(
        zip(
            (
                "time_gyr",
                "sfh",
                "age_weights",
                "formed_mass",
                "surviving_mass",
                "mags",
                "sed",
            ),
            native_values,
            strict=True,
        )
    )
    fluxerr = np.column_stack(
        [sample[f"fluxerr_{band}"].to_numpy(float) for band in band_names]
    )

    metric_frames: list[pd.DataFrame] = []
    all_abs_delta_mag: dict[tuple[str, int], np.ndarray] = {}
    all_abs_noise: dict[tuple[str, int], np.ndarray] = {}
    for grid_strategy in GRID_STRATEGIES:
        for n_nodes in K_VALUES:
            approximator = _build_spline_approximator(n_nodes, grid_strategy)
            raw_spline = np.asarray(
                jax.device_get(
                    approximator(
                        jnp.asarray(native["time_gyr"]), jnp.asarray(raw_native)
                    )
                )
            )
            values = [
                np.asarray(jax.device_get(value))
                for value in sfh_forward_fn(theta_jax, jnp.asarray(raw_spline))
            ]
            result = dict(zip(native.keys(), values, strict=True))
            metrics, abs_delta_mag, abs_noise = _object_metrics(
                sample,
                native,
                result,
                fluxerr,
                n_nodes=n_nodes,
                grid_strategy=grid_strategy,
            )
            metric_frames.append(metrics)
            all_abs_delta_mag[(grid_strategy, n_nodes)] = abs_delta_mag
            all_abs_noise[(grid_strategy, n_nodes)] = abs_noise

    object_metrics = pd.concat(metric_frames, ignore_index=True)
    summary = _summarize_metrics(object_metrics)
    band_summary = _summarize_bands(
        sample, band_names, all_abs_delta_mag, all_abs_noise
    )
    gates, selected, best_scanned = _selection_gates(
        summary, min_group_size=args.min_group_size
    )
    display_grid = selected[0] if selected is not None else best_scanned[0]
    practical_thresholds = _practical_thresholds(summary, display_grid)

    object_metrics.to_csv(out / "spline_k_object_metrics.csv", index=False)
    summary.to_csv(out / "spline_k_summary_metrics.csv", index=False)
    band_summary.to_csv(out / "spline_k_band_metrics.csv", index=False)
    gates.to_csv(out / "spline_k_selection_gates.csv", index=False)
    sample[
        [
            "test_row",
            "state",
            "redshift_bin",
            "mass_bin",
            "high_ssfr_tail",
            "population_weight",
        ]
    ].to_csv(out / "spline_k_sample.csv", index=False)

    _plot_gate_curves(gates, out / "spline_k_selection_gates.png")
    _plot_state_tail_curves(
        summary, display_grid, out / "spline_k_state_tail_curves.png"
    )
    _plot_group_heatmaps(
        summary, display_grid, out / "spline_k_redshift_mass_heatmaps.png"
    )
    _plot_band_heatmap(
        band_summary, display_grid, out / "spline_k_quenched_band_heatmap.png"
    )
    _write_reports(
        out,
        selected=selected,
        best_scanned=best_scanned,
        practical_thresholds=practical_thresholds,
        sample=sample,
        prevalence=prevalence,
        ssfr_threshold=ssfr_threshold,
        gates=gates,
        summary=summary,
    )

    pass_by_configuration = gates.groupby(["grid", "K"])["passed"].all()
    payload = _clean_json(
        {
            "metadata": {
                "dataset": str(args.dataset_dir / "test.parquet"),
                "config": str(args.config),
                "sample_size": len(sample),
                "main_sequence_count": int(np.sum(sample["state"] == "main_sequence")),
                "quenched_count": int(np.sum(sample["state"] == "quenched")),
                "main_sequence_prevalence": prevalence,
                "high_ssfr_quantile": args.high_ssfr_quantile,
                "high_ssfr_threshold": ssfr_threshold,
                "high_ssfr_count": int(np.sum(sample["high_ssfr_tail"])),
                "redshift_quartile_edges": redshift_edges,
                "mass_quartile_edges": mass_edges,
                "min_group_size": args.min_group_size,
                "node_grids": list(GRID_STRATEGIES),
                "interpolator": (
                    "JAX-COSMO InterpolatedUnivariateSpline k=3 not-a-knot "
                    "in log cosmic time and log SFR"
                ),
                "bursty_state_available": False,
                "bursty_limitation": (
                    "Current parquet omits mc_sfh_type and the realized bursty SFH. "
                    "The high-sSFR tail is not a bursty-state label."
                ),
            },
            "k_values": list(K_VALUES),
            "selected": (
                {"grid": selected[0], "K": selected[1]}
                if selected is not None
                else None
            ),
            "best_scanned": {"grid": best_scanned[0], "K": best_scanned[1]},
            "display_grid": display_grid,
            "practical_thresholds": practical_thresholds,
            "pass_by_configuration": {
                f"{grid}:{n_nodes}": bool(value)
                for (grid, n_nodes), value in pass_by_configuration.items()
            },
            "gates": gates.to_dict(orient="records"),
            "summary": summary[summary["metric"].isin(CORE_METRICS)].to_dict(
                orient="records"
            ),
            "bands": band_summary.to_dict(orient="records"),
        }
    )
    (out / "spline_k_scan_payload.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote spline node scan to {out}")
    print(f"Sample: {len(sample)} ({sample['state'].value_counts().to_dict()})")
    print(f"High-sSFR tail rows: {int(np.sum(sample['high_ssfr_tail']))}")
    print(f"Selected configuration: {selected if selected is not None else 'none'}")
    print(f"Best scanned configuration: {best_scanned}")
    print(f"Passing gates by configuration: {pass_by_configuration.to_dict()}")


if __name__ == "__main__":
    main()
