#!/usr/bin/env python3
"""Evaluate a 15D FENIKS prior with ten independent spline-SFH contrasts."""

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
from analyze_feniks_spline_node_scan import (  # noqa: E402
    CORE_METRICS,
    GATE_DEFINITIONS,
    _atom_mask,
    _atom_values,
    _balanced_test_sample,
    _build_forward_functions,
    _clean_json,
    _object_metrics,
    _quantile_labels,
    _summarize_metrics,
)
from build_feniks_forward_explorer import (  # noqa: E402
    pchip_interpolate_jax,
)
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z  # noqa: E402
from dsps.sed.stellar_age_weights import (  # noqa: E402
    calc_age_weights_from_sfh_table,
)
from matplotlib import pyplot as plt  # noqa: E402

from euclid_dsps import model as dsps_model  # noqa: E402
from euclid_dsps.config import load_config  # noqa: E402
from euclid_dsps.filters import load_filters  # noqa: E402
from euclid_dsps.synthetic_diffsky.photometry import (  # noqa: E402
    GROUND_TRUTH_COLUMNS,
    theta_from_truth_frame,
)

N_NODES = 11
N_SHAPE = N_NODES - 1
PLACEMENTS = (
    "uniform_log_time",
    "recent_lookback",
    "hybrid",
    "optimized_balanced",
)
PHYSICAL_PARAMETERS = (
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
)
LATENT_NAMES = PHYSICAL_PARAMETERS + tuple(
    f"sfh_dlog_sfr_{index:02d}" for index in range(1, N_SHAPE + 1)
)
COLORS = {
    "uniform_log_time": "#276A9C",
    "recent_lookback": "#B55432",
    "hybrid": "#28735A",
    "optimized_balanced": "#6A4C93",
    "main_sequence": "#276A9C",
    "quenched": "#B55432",
    "high_ssfr_tail": "#28735A",
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
        default=Path("outputs/analysis/feniks_spline_15d_prior_20260710"),
    )
    parser.add_argument("--seed", type=int, default=260710)
    parser.add_argument("--optimization-per-state", type=int, default=512)
    parser.add_argument("--validation-per-state", type=int, default=256)
    parser.add_argument("--optimization-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--min-log-time-gap", type=float, default=0.025)
    parser.add_argument("--dequantization-half-width-dex", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--high-ssfr-quantile", type=float, default=0.90)
    parser.add_argument("--min-group-size", type=int, default=15)
    return parser.parse_args()


def _balanced_train_splits(
    train: pd.DataFrame,
    atom_values: dict[str, float],
    *,
    optimization_per_state: int,
    validation_per_state: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_atom = _atom_mask(train, atom_values)
    rng = np.random.default_rng(seed)
    split_indices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for state, indices in (
        ("main_sequence", np.flatnonzero(state_atom)),
        ("quenched", np.flatnonzero(~state_atom)),
    ):
        required = optimization_per_state + validation_per_state
        if len(indices) < required:
            raise ValueError(f"Need {required} {state} rows, found {len(indices)}")
        shuffled = rng.permutation(indices)
        split_indices[state] = (
            shuffled[:optimization_per_state],
            shuffled[optimization_per_state:required],
        )

    def build(part: int) -> pd.DataFrame:
        indices = np.concatenate([value[part] for value in split_indices.values()])
        rng.shuffle(indices)
        frame = train.iloc[indices].copy().reset_index(names="train_row")
        atom = _atom_mask(frame, atom_values)
        frame["state"] = np.where(atom, "main_sequence", "quenched")
        return frame

    return build(0), build(1)


def _time_tables(z_obs: np.ndarray, n_sfh_bins: int) -> np.ndarray:
    def single(z_value: jnp.ndarray) -> jnp.ndarray:
        t_obs = jnp.ravel(age_at_z(z_value, *DEFAULT_COSMOLOGY))[0]
        return jnp.linspace(0.05, jnp.maximum(t_obs, 0.06), n_sfh_bins)

    function = jax.jit(jax.vmap(single))
    return np.asarray(jax.device_get(function(jnp.asarray(z_obs, dtype=jnp.float32))))


def _native_sfh_batch(
    frame: pd.DataFrame,
    native_sfh_fn: Any,
    *,
    batch_size: int,
) -> np.ndarray:
    theta = theta_from_truth_frame(frame).astype(np.float32)
    parts = []
    for start in range(0, len(frame), batch_size):
        value = native_sfh_fn(jnp.asarray(theta[start : start + batch_size]))
        parts.append(np.asarray(jax.device_get(value)))
    return np.concatenate(parts, axis=0)


def _fixed_knot_times(
    time: jnp.ndarray,
    placement: str,
    optimized_u: jnp.ndarray | None = None,
) -> jnp.ndarray:
    t_obs = time[-1]
    span = jnp.maximum(t_obs - time[0], 1.0e-6)
    if placement == "uniform_log_time":
        return jnp.geomspace(time[0], t_obs, N_NODES)
    if placement == "recent_lookback":
        fractions = jnp.concatenate(
            (jnp.zeros(1), jnp.geomspace(1.0e-3, 1.0, N_NODES - 1))
        )
        return jnp.flip(t_obs - fractions * span)
    if placement == "hybrid":
        n_log_nodes = int(np.ceil(0.65 * N_NODES))
        n_recent_nodes = N_NODES - n_log_nodes
        log_grid = jnp.geomspace(time[0], t_obs, n_log_nodes)
        recent_fractions = jnp.geomspace(1.0e-3, 0.5, n_recent_nodes + 2)[1:-1]
        return jnp.sort(jnp.concatenate((log_grid, t_obs - recent_fractions * span)))
    if placement == "optimized_balanced" and optimized_u is not None:
        log_time_min = jnp.log10(jnp.maximum(time[0], 1.0e-6))
        log_time_max = jnp.log10(jnp.maximum(t_obs, 1.0e-6))
        return 10 ** (log_time_min + optimized_u * (log_time_max - log_time_min))
    raise ValueError(f"Unsupported placement {placement!r}")


def _build_approximator(placement: str, optimized_u: np.ndarray | None = None) -> Any:
    optimized_u_jax = None if optimized_u is None else jnp.asarray(optimized_u)

    def single(
        time: jnp.ndarray, sfr: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        knot_time = _fixed_knot_times(time, placement, optimized_u_jax)
        log_time = jnp.log10(jnp.maximum(time, 1.0e-6))
        knot_log_time = jnp.log10(jnp.maximum(knot_time, 1.0e-6))
        log_sfr = jnp.log10(jnp.maximum(sfr, 1.0e-30))
        knot_log_sfr = jnp.interp(knot_log_time, log_time, log_sfr)
        spline_log_sfr = pchip_interpolate_jax(knot_log_time, knot_log_sfr, log_time)
        contrasts = jnp.diff(knot_log_sfr)
        return 10**spline_log_sfr, contrasts, knot_log_sfr, knot_time

    return jax.jit(jax.vmap(single, in_axes=(0, 0)))


def _build_contrast_reconstructor(
    placement: str, optimized_u: np.ndarray | None = None
) -> Any:
    optimized_u_jax = None if optimized_u is None else jnp.asarray(optimized_u)

    def single(time: jnp.ndarray, contrasts: jnp.ndarray) -> jnp.ndarray:
        knot_time = _fixed_knot_times(time, placement, optimized_u_jax)
        knot_log_time = jnp.log10(jnp.maximum(knot_time, 1.0e-6))
        log_time = jnp.log10(jnp.maximum(time, 1.0e-6))
        knot_log_sfr = jnp.concatenate((jnp.zeros(1), jnp.cumsum(contrasts)))
        return 10 ** pchip_interpolate_jax(knot_log_time, knot_log_sfr, log_time)

    return jax.jit(jax.vmap(single, in_axes=(0, 0)))


def _dequantize_exact_atoms(
    latent: pd.DataFrame,
    *,
    half_width_dex: float,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    result = latent.copy()
    rng = np.random.default_rng(seed)
    count = 0
    for name in LATENT_NAMES[len(PHYSICAL_PARAMETERS) :]:
        values = result[name].to_numpy(float, copy=True)
        atom = values == 0.0
        count += int(np.sum(atom))
        values[atom] = rng.uniform(-half_width_dex, half_width_dex, np.sum(atom))
        result[name] = values
    return result, count


def _dequantize_contrasts(
    contrasts: np.ndarray, *, half_width_dex: float, seed: int
) -> tuple[np.ndarray, int]:
    result = np.asarray(contrasts, dtype=float).copy()
    atom = result == 0.0
    rng = np.random.default_rng(seed)
    result[atom] = rng.uniform(-half_width_dex, half_width_dex, np.sum(atom))
    return result, int(np.sum(atom))


def _evaluate_dequantization_widths(
    sample: pd.DataFrame,
    native: dict[str, np.ndarray],
    theta_jax: jnp.ndarray,
    exact_contrasts: np.ndarray,
    sfh_forward_fn: Any,
    fluxerr: np.ndarray,
    *,
    placement: str,
    optimized_u: np.ndarray,
    widths: list[float],
    seed: int,
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame], dict[float, pd.DataFrame]]:
    reconstructor = _build_contrast_reconstructor(placement, optimized_u)
    rows: list[dict[str, Any]] = []
    metrics_by_width: dict[float, pd.DataFrame] = {}
    summaries_by_width: dict[float, pd.DataFrame] = {}
    for width in widths:
        dequantized, count = _dequantize_contrasts(
            exact_contrasts, half_width_dex=width, seed=seed
        )
        raw_sfh = np.asarray(
            jax.device_get(
                reconstructor(jnp.asarray(native["time_gyr"]), jnp.asarray(dequantized))
            )
        )
        values = [
            np.asarray(jax.device_get(value))
            for value in sfh_forward_fn(theta_jax, jnp.asarray(raw_sfh))
        ]
        result = dict(zip(native.keys(), values, strict=True))
        grid = f"{placement}_dequantized_{width:.5g}"
        metrics, _, _ = _object_metrics(
            sample,
            native,
            result,
            fluxerr,
            n_nodes=N_NODES,
            grid_strategy=grid,
        )
        summary = _summarize_metrics(metrics)
        population_mag = _summary_value(
            summary,
            grid,
            "population_weighted",
            "max_abs_delta_mag",
            "p95",
        )
        population_noise = _summary_value(
            summary,
            grid,
            "population_weighted",
            "noise_rms",
            "p95",
        )
        rows.append(
            {
                "half_width_dex": width,
                "values_dequantized": count,
                "population_p95_max_abs_delta_mag": population_mag,
                "population_p95_noise_rms": population_noise,
                "passes_population_targets": bool(
                    population_mag <= 0.01 and population_noise <= 0.10
                ),
            }
        )
        metrics_by_width[width] = metrics
        summaries_by_width[width] = summary
    return pd.DataFrame(rows), metrics_by_width, summaries_by_width


def _logits_to_u(logits: jnp.ndarray, min_gap: float) -> jnp.ndarray:
    n_intervals = N_NODES - 1
    remaining = 1.0 - n_intervals * min_gap
    gaps = min_gap + remaining * jax.nn.softmax(logits)
    return jnp.concatenate((jnp.zeros(1), jnp.cumsum(gaps)))


def _mass_fractions_jax(time: jnp.ndarray, sfr: jnp.ndarray) -> jnp.ndarray:
    dt = jnp.diff(time)
    segment_sfr = 0.5 * (sfr[1:] + sfr[:-1])
    midpoint = 0.5 * (time[1:] + time[:-1])
    lookback = time[-1] - midpoint
    mass = segment_sfr * dt
    windows = (
        lookback < 0.1,
        (lookback >= 0.1) & (lookback < 1.0),
        lookback >= 1.0,
    )
    values = jnp.asarray([jnp.sum(jnp.where(mask, mass, 0.0)) for mask in windows])
    return values / jnp.maximum(jnp.sum(values), 1.0e-30)


def _build_placement_objective(
    time: np.ndarray,
    raw_sfh: np.ndarray,
    state: np.ndarray,
    ssp_lg_age_gyr: np.ndarray,
    *,
    min_gap: float,
) -> Any:
    time_jax = jnp.asarray(time)
    raw_jax = jnp.asarray(raw_sfh)
    native_log = jnp.log10(jnp.maximum(raw_jax, 1.0e-12))
    main_indices = jnp.asarray(np.flatnonzero(state == "main_sequence"))
    quenched_indices = jnp.asarray(np.flatnonzero(state == "quenched"))
    age_grid = jnp.asarray(ssp_lg_age_gyr)

    def age_single(t: jnp.ndarray, sfr: jnp.ndarray) -> jnp.ndarray:
        return calc_age_weights_from_sfh_table(t, sfr, age_grid, t[-1])

    native_age = jax.vmap(age_single)(time_jax, raw_jax)
    native_mass_fraction = jax.vmap(_mass_fractions_jax)(time_jax, raw_jax)

    def reconstruct(logits: jnp.ndarray) -> jnp.ndarray:
        u = _logits_to_u(logits, min_gap)

        def single(t: jnp.ndarray, sfr: jnp.ndarray) -> jnp.ndarray:
            log_time = jnp.log10(jnp.maximum(t, 1.0e-6))
            knot_log_time = log_time[0] + u * (log_time[-1] - log_time[0])
            log_sfr = jnp.log10(jnp.maximum(sfr, 1.0e-30))
            knot_log_sfr = jnp.interp(knot_log_time, log_time, log_sfr)
            return 10 ** pchip_interpolate_jax(knot_log_time, knot_log_sfr, log_time)

        return jax.vmap(single)(time_jax, raw_jax)

    def cvar95(values: jnp.ndarray, count: int) -> jnp.ndarray:
        tail_size = max(1, int(np.ceil(0.05 * count)))
        return jnp.mean(jax.lax.top_k(values, tail_size)[0])

    def objective(logits: jnp.ndarray) -> jnp.ndarray:
        spline = reconstruct(logits)
        spline_log = jnp.log10(jnp.maximum(spline, 1.0e-12))
        log_rmse = jnp.sqrt(jnp.mean((spline_log - native_log) ** 2, axis=1))
        spline_age = jax.vmap(age_single)(time_jax, spline)
        age_l1 = jnp.sum(jnp.abs(spline_age - native_age), axis=1)
        spline_fraction = jax.vmap(_mass_fractions_jax)(time_jax, spline)
        fraction_error = jnp.max(
            jnp.abs(spline_fraction - native_mass_fraction), axis=1
        )
        score = log_rmse / 0.05 + age_l1 / 0.05 + fraction_error / 0.02
        state_terms = []
        for indices in (main_indices, quenched_indices):
            values = score[indices]
            state_terms.append(jnp.mean(values) + cvar95(values, len(indices)))
        return 0.5 * (state_terms[0] + state_terms[1])

    return jax.jit(jax.value_and_grad(objective)), jax.jit(objective)


def _optimize_shared_positions(
    optimization_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ssp_lg_age_gyr: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    min_gap: float,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    train_value_grad, _ = _build_placement_objective(
        *optimization_data, ssp_lg_age_gyr, min_gap=min_gap
    )
    _, validation_objective = _build_placement_objective(
        *validation_data, ssp_lg_age_gyr, min_gap=min_gap
    )
    rng = np.random.default_rng(seed)
    starts = (
        np.zeros(N_NODES - 1),
        np.linspace(-0.5, 0.5, N_NODES - 1),
        np.linspace(0.5, -0.5, N_NODES - 1),
        rng.normal(0.0, 0.25, N_NODES - 1),
    )
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, np.ndarray]] = []
    for restart, initial in enumerate(starts):
        logits = jnp.asarray(initial, dtype=jnp.float32)
        first_moment = jnp.zeros_like(logits)
        second_moment = jnp.zeros_like(logits)
        train_loss = float("nan")
        for step in range(1, steps + 1):
            loss, gradient = train_value_grad(logits)
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient**2
            corrected_first = first_moment / (1.0 - 0.9**step)
            corrected_second = second_moment / (1.0 - 0.999**step)
            cosine = 0.5 * (1.0 + np.cos(np.pi * (step - 1) / max(steps, 1)))
            rate = learning_rate * (0.1 + 0.9 * cosine)
            logits = logits - rate * corrected_first / (
                jnp.sqrt(corrected_second) + 1.0e-8
            )
            logits = logits - jnp.mean(logits)
            train_loss = float(loss)
        validation_loss = float(validation_objective(logits))
        u = np.asarray(jax.device_get(_logits_to_u(logits, min_gap)))
        rows.append(
            {
                "restart": restart,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "min_gap": float(np.min(np.diff(u))),
                "max_gap": float(np.max(np.diff(u))),
            }
        )
        candidates.append((validation_loss, u))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], pd.DataFrame(rows)


def _proxy_score(
    time: np.ndarray,
    native_sfh: np.ndarray,
    spline_sfh: np.ndarray,
    state: np.ndarray,
    ssp_lg_age_gyr: np.ndarray,
) -> tuple[float, dict[str, float]]:
    native_log = np.log10(np.maximum(native_sfh, 1.0e-12))
    spline_log = np.log10(np.maximum(spline_sfh, 1.0e-12))
    log_rmse = np.sqrt(np.mean((spline_log - native_log) ** 2, axis=1))

    age_grid = jnp.asarray(ssp_lg_age_gyr)

    def age_single(t: jnp.ndarray, sfr: jnp.ndarray) -> jnp.ndarray:
        return calc_age_weights_from_sfh_table(t, sfr, age_grid, t[-1])

    age_function = jax.jit(jax.vmap(age_single))
    native_age = np.asarray(
        jax.device_get(age_function(jnp.asarray(time), jnp.asarray(native_sfh)))
    )
    spline_age = np.asarray(
        jax.device_get(age_function(jnp.asarray(time), jnp.asarray(spline_sfh)))
    )
    age_l1 = np.sum(np.abs(spline_age - native_age), axis=1)

    native_fraction = np.asarray(
        jax.device_get(
            jax.jit(jax.vmap(_mass_fractions_jax))(
                jnp.asarray(time), jnp.asarray(native_sfh)
            )
        )
    )
    spline_fraction = np.asarray(
        jax.device_get(
            jax.jit(jax.vmap(_mass_fractions_jax))(
                jnp.asarray(time), jnp.asarray(spline_sfh)
            )
        )
    )
    fraction_error = np.max(np.abs(spline_fraction - native_fraction), axis=1)
    combined = log_rmse / 0.05 + age_l1 / 0.05 + fraction_error / 0.02
    state_scores = []
    details: dict[str, float] = {}
    for label in ("main_sequence", "quenched"):
        mask = state == label
        state_score = float(
            np.mean(combined[mask])
            + np.mean(
                np.sort(combined[mask])[-max(1, int(np.ceil(0.05 * np.sum(mask)))) :]
            )
        )
        state_scores.append(state_score)
        details[f"{label}_score"] = state_score
        details[f"{label}_p95_log_sfh_rmse"] = float(np.quantile(log_rmse[mask], 0.95))
        details[f"{label}_p95_age_weight_l1"] = float(np.quantile(age_l1[mask], 0.95))
    return float(np.mean(state_scores)), details


def _summarize_bands(
    sample: pd.DataFrame,
    band_names: list[str],
    all_abs_delta_mag: dict[str, np.ndarray],
    all_abs_noise: dict[str, np.ndarray],
) -> pd.DataFrame:
    groups = {
        "balanced_all": np.ones(len(sample), dtype=bool),
        "main_sequence": sample["state"].to_numpy() == "main_sequence",
        "quenched": sample["state"].to_numpy() == "quenched",
        "high_ssfr_tail": sample["high_ssfr_tail"].to_numpy(bool),
    }
    rows: list[dict[str, Any]] = []
    for placement in PLACEMENTS:
        for group, mask in groups.items():
            for band_index, band in enumerate(band_names):
                for metric, matrix in (
                    ("abs_delta_mag", all_abs_delta_mag[placement]),
                    ("abs_delta_flux_sigma", all_abs_noise[placement]),
                ):
                    values = matrix[mask, band_index]
                    rows.append(
                        {
                            "placement": placement,
                            "group": group,
                            "band": band,
                            "metric": metric,
                            "n": len(values),
                            "median": float(np.quantile(values, 0.50)),
                            "p95": float(np.quantile(values, 0.95)),
                            "p99": float(np.quantile(values, 0.99)),
                            "max": float(np.max(values)),
                        }
                    )
    return pd.DataFrame(rows)


def _selection_gates(
    summary: pd.DataFrame, *, min_group_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    eligible_types = {"state", "activity", "redshift", "mass"}
    for placement in PLACEMENTS:
        for metric, statistic, threshold in GATE_DEFINITIONS:
            candidates = summary[
                (summary["grid"] == placement)
                & (summary["metric"] == metric)
                & (summary["group_type"].isin(eligible_types))
                & (summary["n"] >= min_group_size)
            ]
            worst = candidates.iloc[int(np.nanargmax(candidates[statistic]))]
            worst_value = float(worst[statistic])
            rows.append(
                {
                    "placement": placement,
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
        gates.groupby("placement", as_index=False)
        .agg(
            passed=("passed", "all"),
            failed_gates=("passed", lambda values: int((~values).sum())),
            max_threshold_ratio=("threshold_ratio", "max"),
        )
        .sort_values(["failed_gates", "max_threshold_ratio", "placement"])
        .reset_index(drop=True)
    )
    return gates, decisions


def _latent_frame(
    frame: pd.DataFrame,
    contrasts: np.ndarray,
) -> pd.DataFrame:
    payload = {
        name: frame[GROUND_TRUTH_COLUMNS[name]].to_numpy(float)
        for name in PHYSICAL_PARAMETERS
    }
    payload.update(
        {
            name: contrasts[:, index]
            for index, name in enumerate(LATENT_NAMES[len(PHYSICAL_PARAMETERS) :])
        }
    )
    result = pd.DataFrame(payload)
    if tuple(result.columns) != LATENT_NAMES:
        raise AssertionError("15D latent column order drifted")
    return result


def _project_full_split(
    frame: pd.DataFrame,
    native_sfh_fn: Any,
    *,
    placement: str,
    optimized_u: np.ndarray,
    n_sfh_bins: int,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    approximator = _build_approximator(placement, optimized_u)
    contrast_parts: list[np.ndarray] = []
    knot_log_sfr_parts: list[np.ndarray] = []
    knot_time_parts: list[np.ndarray] = []
    z_column = GROUND_TRUTH_COLUMNS["z_obs"]
    for start in range(0, len(frame), batch_size):
        part = frame.iloc[start : start + batch_size]
        raw_sfh = _native_sfh_batch(part, native_sfh_fn, batch_size=batch_size)
        time = _time_tables(part[z_column].to_numpy(float), n_sfh_bins)
        _, contrasts, knot_log_sfr, knot_time = approximator(
            jnp.asarray(time), jnp.asarray(raw_sfh)
        )
        contrast_parts.append(np.asarray(jax.device_get(contrasts)))
        knot_log_sfr_parts.append(np.asarray(jax.device_get(knot_log_sfr)))
        knot_time_parts.append(np.asarray(jax.device_get(knot_time)))
    all_contrasts = np.concatenate(contrast_parts)
    all_knot_log_sfr = np.concatenate(knot_log_sfr_parts)
    all_knot_time = np.concatenate(knot_time_parts)
    return (
        _latent_frame(frame, all_contrasts),
        all_knot_log_sfr,
        all_knot_time,
    )


def _latent_diagnostics(
    latent: pd.DataFrame,
    state: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    rows: list[dict[str, Any]] = []
    shape_names = LATENT_NAMES[len(PHYSICAL_PARAMETERS) :]
    for group, mask in (
        ("all", np.ones(len(latent), dtype=bool)),
        ("main_sequence", state == "main_sequence"),
        ("quenched", state == "quenched"),
    ):
        for name in shape_names:
            values = latent.loc[mask, name].to_numpy(float)
            rounded = np.round(values, 6)
            mode_fraction = float(
                pd.Series(rounded).value_counts().iloc[0] / len(values)
            )
            rows.append(
                {
                    "group": group,
                    "coordinate": name,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "p01": float(np.quantile(values, 0.01)),
                    "p05": float(np.quantile(values, 0.05)),
                    "median": float(np.quantile(values, 0.50)),
                    "p95": float(np.quantile(values, 0.95)),
                    "p99": float(np.quantile(values, 0.99)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "near_zero_fraction": float(np.mean(np.abs(values) < 1.0e-6)),
                    "exact_zero_fraction": float(np.mean(values == 0.0)),
                    "rounded_mode_fraction": mode_fraction,
                    "finite_fraction": float(np.mean(np.isfinite(values))),
                }
            )
    diagnostics = pd.DataFrame(rows)
    matrix = latent.to_numpy(float)
    correlation = np.corrcoef(matrix, rowvar=False)
    shape_correlation = correlation[
        len(PHYSICAL_PARAMETERS) :, len(PHYSICAL_PARAMETERS) :
    ]
    eigenvalues = np.maximum(np.linalg.eigvalsh(shape_correlation), 0.0)
    probabilities = eigenvalues / np.maximum(np.sum(eigenvalues), 1.0e-30)
    positive = probabilities > 0.0
    effective_rank = float(
        np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    )
    off_diagonal = shape_correlation - np.eye(N_SHAPE)
    summary = {
        "max_rounded_mode_fraction": float(
            diagnostics.loc[
                diagnostics["group"] == "all", "rounded_mode_fraction"
            ].max()
        ),
        "max_near_zero_fraction": float(
            diagnostics.loc[diagnostics["group"] == "all", "near_zero_fraction"].max()
        ),
        "max_exact_zero_fraction": float(
            diagnostics.loc[diagnostics["group"] == "all", "exact_zero_fraction"].max()
        ),
        "worst_exact_zero_coordinate": str(
            diagnostics.loc[diagnostics["group"] == "all"]
            .sort_values("exact_zero_fraction", ascending=False)
            .iloc[0]["coordinate"]
        ),
        "min_coordinate_std": float(
            diagnostics.loc[diagnostics["group"] == "all", "std"].min()
        ),
        "max_abs_shape_correlation": float(np.max(np.abs(off_diagonal))),
        "shape_correlation_condition_number": float(np.linalg.cond(shape_correlation)),
        "shape_effective_rank": effective_rank,
    }
    return diagnostics, summary, correlation


def _summary_value(
    summary: pd.DataFrame,
    placement: str,
    group: str,
    metric: str,
    statistic: str,
) -> float:
    row = summary[
        (summary["grid"] == placement)
        & (summary["group"] == group)
        & (summary["metric"] == metric)
    ]
    return float(row.iloc[0][statistic])


def _plot_closure_comparison(summary: pd.DataFrame, path: Path) -> None:
    panels = (
        ("max_abs_delta_mag", "p95 maximum |delta magnitude|", 0.01),
        ("noise_rms", "p95 RMS delta flux / sigma", 0.10),
        ("age_weight_l1", "p95 age-weight L1", 0.05),
        ("log_sfh_rmse_dex", "p95 log-SFH RMSE [dex]", 0.05),
    )
    groups = ("population_weighted", "main_sequence", "quenched", "high_ssfr_tail")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = np.arange(len(PLACEMENTS))
    width = 0.19
    for axis, (metric, title, threshold) in zip(axes.flat, panels, strict=True):
        for group_index, group in enumerate(groups):
            values = [
                _summary_value(summary, placement, group, metric, "p95")
                for placement in PLACEMENTS
            ]
            axis.bar(
                x + (group_index - 1.5) * width,
                values,
                width,
                label=group,
                color=COLORS.get(group, "#777777"),
                alpha=0.88,
            )
        axis.axhline(threshold, color="#202628", linestyle="--", linewidth=1.2)
        axis.set_xticks(x, [name.replace("_", "\n") for name in PLACEMENTS])
        axis.set_yscale("log")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Fixed 10-parameter SFH shape: held-out closure", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_node_placements(
    validation_time: np.ndarray,
    validation_sfh: np.ndarray,
    optimized_u: np.ndarray,
    path: Path,
) -> None:
    median_t_obs = float(np.median(validation_time[:, -1]))
    representative_time = np.linspace(0.05, median_t_obs, validation_time.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for row, placement in enumerate(PLACEMENTS):
        knots = np.asarray(
            _fixed_knot_times(
                jnp.asarray(representative_time),
                placement,
                jnp.asarray(optimized_u),
            )
        )
        axes[0].scatter(
            knots,
            np.full_like(knots, row, dtype=float),
            s=44,
            color=COLORS[placement],
            label=placement,
        )
    axes[0].set_xscale("log")
    axes[0].set_yticks(range(len(PLACEMENTS)), PLACEMENTS)
    axes[0].set_xlabel("Cosmic time [Gyr]")
    axes[0].set_title(f"Node positions at median t_obs={median_t_obs:.2f} Gyr")
    axes[0].grid(axis="x", alpha=0.2)

    examples = [
        int(np.argmax(np.max(validation_sfh, axis=1))),
        int(np.argmin(np.max(validation_sfh, axis=1))),
    ]
    optimized = _build_approximator("optimized_balanced", optimized_u)
    reconstructed = np.asarray(
        jax.device_get(
            optimized(jnp.asarray(validation_time), jnp.asarray(validation_sfh))[0]
        )
    )
    for order, index in enumerate(examples):
        color = ("#276A9C", "#B55432")[order]
        axes[1].plot(
            validation_time[index],
            validation_sfh[index],
            color=color,
            linewidth=2.0,
            label=f"native example {order + 1}",
        )
        axes[1].plot(
            validation_time[index],
            reconstructed[index],
            color=color,
            linestyle="--",
            label=f"optimized spline {order + 1}",
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Cosmic time [Gyr]")
    axes[1].set_ylabel("SFR [Msun / yr]")
    axes[1].set_title("Train-validation examples")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_contrast_distributions(
    latent: pd.DataFrame,
    state: np.ndarray,
    path: Path,
) -> None:
    names = LATENT_NAMES[len(PHYSICAL_PARAMETERS) :]
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), constrained_layout=True)
    for axis, name in zip(axes.flat, names, strict=True):
        for group in ("main_sequence", "quenched"):
            values = latent.loc[state == group, name].to_numpy(float)
            low, high = np.quantile(values, [0.005, 0.995])
            clipped = values[(values >= low) & (values <= high)]
            axis.hist(
                clipped,
                bins=45,
                density=True,
                histtype="stepfilled",
                alpha=0.42,
                color=COLORS[group],
                label=group,
            )
        axis.axvline(0.0, color="#303638", linewidth=0.8)
        axis.set_title(name.replace("sfh_dlog_sfr_", "q"))
        axis.grid(alpha=0.15)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Ten train-set SFH contrasts by native state", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_latent_correlation(correlation: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    image = axis.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    labels = [
        "z",
        "mass",
        "metal",
        "A_V",
        "delta",
        *[f"q{index}" for index in range(1, N_SHAPE + 1)],
    ]
    axis.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Pearson correlation of the 15D training latent")
    fig.colorbar(image, ax=axis, label="correlation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    def format_value(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.5g}"
        return str(value)

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _write_report(
    out: Path,
    *,
    selected_placement: str,
    optimized_u: np.ndarray,
    proxy_scores: pd.DataFrame,
    decisions: pd.DataFrame,
    summary: pd.DataFrame,
    latent_summary: dict[str, Any],
    optimization_history: pd.DataFrame,
    dequantization_summary: dict[str, Any],
    sample_size: int,
    prevalence: float,
) -> None:
    selected_population = summary[
        (summary["grid"] == selected_placement)
        & (summary["group"] == "population_weighted")
        & summary["metric"].isin(
            ["max_abs_delta_mag", "noise_rms", "age_weight_l1", "log_sfh_rmse_dex"]
        )
    ][["metric", "median", "p95", "p99", "max"]]
    selected_states = summary[
        (summary["grid"] == selected_placement)
        & summary["group"].isin(["main_sequence", "quenched", "high_ssfr_tail"])
        & summary["metric"].isin(
            ["max_abs_delta_mag", "noise_rms", "age_weight_l1", "log_sfh_rmse_dex"]
        )
    ][["group", "metric", "n", "p95", "p99", "max"]]
    population_mag = _summary_value(
        summary, selected_placement, "population_weighted", "max_abs_delta_mag", "p95"
    )
    population_noise = _summary_value(
        summary, selected_placement, "population_weighted", "noise_rms", "p95"
    )
    quenched_mag = _summary_value(
        summary, selected_placement, "quenched", "max_abs_delta_mag", "p95"
    )
    no_atom = latent_summary["max_exact_zero_fraction"] < 0.01
    dequant_validation_note = (
        "At least one width passed both train-validation population targets."
        if dequantization_summary["validation_target_passed"]
        else (
            "No width passed the train-validation 0.1-sigma target; the smallest "
            "candidate was retained as the minimum-distortion fallback."
        )
    )
    closure_status = (
        "passes the population photometric and noise targets"
        if population_mag < 0.01 and population_noise < 0.10
        else "does not pass both population closure targets"
    )
    report = f"""# FENIKS 15D spline-prior scan

## Decision

The exact latent is
`[z_obs, log10_stellar_mass, log10_stellar_metallicity, dust_av, dust_delta, q01..q10]`.
The ten `q` coordinates are adjacent log-SFR contrasts across eleven shared
PCHIP nodes. Stellar mass supplies the discarded common amplitude.

**Placement selected without using the test closure: `{selected_placement}`.**
It {closure_status}. The held-out population p95 is `{population_mag:.5g}` mag
and `{population_noise:.5g}` in RMS flux-error units. The quenched p95 maximum
magnitude residual is `{quenched_mag:.5g}` mag.

The contrast audit {"finds no percent-level exact atom" if no_atom else "finds at least one percent-level exact atom"}.
The ten-dimensional shape correlation has effective rank
`{latent_summary["shape_effective_rank"]:.3f}` and maximum absolute off-diagonal
correlation `{latent_summary["max_abs_shape_correlation"]:.3f}`.

The worst exact atom is `{latent_summary["worst_exact_zero_coordinate"]}` with
fraction `{latent_summary["max_exact_zero_fraction"]:.3%}`. A reproducible
uniform dequantization of width `+/-{dequantization_summary["half_width_dex"]:.4g}`
dex removes these exact zeros while keeping the held-out population p95 at
`{dequantization_summary["population_p95_max_abs_delta_mag"]:.5g}` mag and
`{dequantization_summary["population_p95_noise_rms"]:.5g}` RMS flux-error units.
{dequant_validation_note}

Training-validation width scan used for selection:

{_markdown_table(pd.DataFrame(dequantization_summary["validation_scan"]))}

Held-out test width scan reported after selection:

{_markdown_table(pd.DataFrame(dequantization_summary["test_scan"]))}

## Leakage-safe placement selection

The optimized grid was fitted on a balanced training subset. The final placement
was selected by the SFH/age-weight proxy on a separate balanced training
validation subset. Test photometry was evaluated only afterwards.

{_markdown_table(proxy_scores)}

Normalized log-cosmic-time positions of the optimized eleven nodes:

`{np.array2string(optimized_u, precision=5, separator=", ")}`

{_markdown_table(optimization_history)}

![Node placement](spline_15d_node_placements.png)

## Held-out closure

The test sample contains `{sample_size}` balanced diagnostic galaxies. Population
statistics restore the native `{prevalence:.3%}` main-sequence prevalence.

{_markdown_table(decisions)}

{_markdown_table(selected_population)}

{_markdown_table(selected_states)}

![Closure comparison](spline_15d_closure_comparison.png)

## Prior learnability

- Dimension: exactly 15 independent coordinates.
- Maximum rounded marginal mode fraction: `{latent_summary["max_rounded_mode_fraction"]:.5g}`.
- Maximum exact-zero atom fraction: `{latent_summary["max_exact_zero_fraction"]:.5g}`.
- Maximum near-zero fraction: `{latent_summary["max_near_zero_fraction"]:.5g}`.
- Minimum contrast standard deviation: `{latent_summary["min_coordinate_std"]:.5g}` dex.
- Shape correlation condition number: `{latent_summary["shape_correlation_condition_number"]:.5g}`.
- Shape effective rank: `{latent_summary["shape_effective_rank"]:.5g}` / 10.

The exact latent should not be passed unchanged to a continuous normalizing
flow. `feniks_spline_15d_train_dequantized.parquet` is the recommended first
training table; the exact table remains the scientific truth/audit product.

![Contrast distributions](spline_15d_contrast_distributions.png)

![Latent correlation](spline_15d_latent_correlation.png)

## Scientific scope

This is sufficient for the current local DSPS closure contract: fixed SSP,
lognormal stellar MDF with fixed scatter, deterministic IGM from redshift, and
the configured two-parameter dust mapping. It does not preserve a full native
Diffsky age-metallicity distribution or a separately stored burst realization.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>FENIKS 15D spline prior</title>
<style>body{{font:15px/1.5 sans-serif;max-width:1250px;margin:30px auto;padding:0 20px;color:#1d2528}}img{{max-width:100%;border:1px solid #d5dbd8;margin:12px 0}}table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{border:1px solid #d5dbd8;padding:6px;text-align:right}}th{{background:#edf1ef}}code{{background:#eef1ef;padding:2px 4px}}</style></head><body>
<h1>FENIKS 15D spline-prior scan</h1>
<p><strong>Selected placement: {selected_placement}.</strong> It {closure_status}.</p>
<p>Population p95: {population_mag:.5g} mag and {population_noise:.5g} RMS flux-error units. Quenched p95: {quenched_mag:.5g} mag.</p>
<h2>Leakage-safe placement selection</h2>{proxy_scores.to_html(index=False)}
<p>Optimized normalized log-time nodes: <code>{np.array2string(optimized_u, precision=5, separator=", ")}</code></p>
{optimization_history.to_html(index=False)}<img src="spline_15d_node_placements.png" alt="Node placement">
<h2>Held-out closure</h2>{decisions.to_html(index=False)}{selected_population.to_html(index=False)}{selected_states.to_html(index=False)}
<img src="spline_15d_closure_comparison.png" alt="Closure comparison">
<h2>Prior learnability</h2><pre>{json.dumps(_clean_json(latent_summary), indent=2)}</pre>
<p>Dequantized closure: {dequantization_summary["population_p95_max_abs_delta_mag"]:.5g} mag and {dequantization_summary["population_p95_noise_rms"]:.5g} RMS flux-error units.</p>
<h3>Training-validation dequantization scan</h3>{pd.DataFrame(dequantization_summary["validation_scan"]).to_html(index=False)}
<h3>Held-out test dequantization scan</h3>{pd.DataFrame(dequantization_summary["test_scan"]).to_html(index=False)}
<img src="spline_15d_contrast_distributions.png" alt="Contrast distributions">
<img src="spline_15d_latent_correlation.png" alt="Latent correlation">
<h2>Scientific scope</h2><p>This result uses the current local DSPS closure contract, including the fixed-scatter lognormal MDF and deterministic IGM.</p>
</body></html>"""
    (out / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.dataset_dir / "train.parquet")
    test = pd.read_parquet(args.dataset_dir / "test.parquet")
    atom_values = _atom_values(train)
    optimization_sample, validation_sample = _balanced_train_splits(
        train,
        atom_values,
        optimization_per_state=args.optimization_per_state,
        validation_per_state=args.validation_per_state,
        seed=args.seed,
    )
    test_sample, prevalence = _balanced_test_sample(test, atom_values, seed=args.seed)

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
    n_sfh_bins = int(config["model"]["n_sfh_bins"])
    native_sfh_fn, sfh_forward_fn = _build_forward_functions(context)
    ssp_lg_age_gyr = np.asarray(dsps_model._context_ssp_lg_age_gyr(context))
    z_column = GROUND_TRUTH_COLUMNS["z_obs"]
    mass_column = GROUND_TRUTH_COLUMNS["log10_stellar_mass"]

    optimization_time = _time_tables(
        optimization_sample[z_column].to_numpy(float), n_sfh_bins
    )
    validation_time = _time_tables(
        validation_sample[z_column].to_numpy(float), n_sfh_bins
    )
    optimization_sfh = _native_sfh_batch(
        optimization_sample, native_sfh_fn, batch_size=args.batch_size
    )
    validation_sfh = _native_sfh_batch(
        validation_sample, native_sfh_fn, batch_size=args.batch_size
    )
    optimized_u, optimization_history = _optimize_shared_positions(
        (
            optimization_time,
            optimization_sfh,
            optimization_sample["state"].to_numpy(str),
        ),
        (
            validation_time,
            validation_sfh,
            validation_sample["state"].to_numpy(str),
        ),
        ssp_lg_age_gyr,
        steps=args.optimization_steps,
        learning_rate=args.learning_rate,
        min_gap=args.min_log_time_gap,
        seed=args.seed,
    )

    proxy_rows: list[dict[str, Any]] = []
    for placement in PLACEMENTS:
        approximator = _build_approximator(placement, optimized_u)
        spline = np.asarray(
            jax.device_get(
                approximator(jnp.asarray(validation_time), jnp.asarray(validation_sfh))[
                    0
                ]
            )
        )
        score, details = _proxy_score(
            validation_time,
            validation_sfh,
            spline,
            validation_sample["state"].to_numpy(str),
            ssp_lg_age_gyr,
        )
        proxy_rows.append({"placement": placement, "proxy_score": score, **details})
    proxy_scores = (
        pd.DataFrame(proxy_rows).sort_values("proxy_score").reset_index(drop=True)
    )
    selected_placement = str(proxy_scores.iloc[0]["placement"])

    test_sample["redshift_bin"], redshift_edges = _quantile_labels(
        test_sample[z_column].to_numpy(float),
        test[z_column].to_numpy(float),
        "redshift",
    )
    test_sample["mass_bin"], mass_edges = _quantile_labels(
        test_sample[mass_column].to_numpy(float),
        test[mass_column].to_numpy(float),
        "mass",
    )
    ssfr_threshold = float(
        np.quantile(test["logssfr_true"].to_numpy(float), args.high_ssfr_quantile)
    )
    test_sample["high_ssfr_tail"] = (
        test_sample["logssfr_true"].to_numpy(float) >= ssfr_threshold
    )

    theta = theta_from_truth_frame(test_sample).astype(np.float32)
    theta_jax = jnp.asarray(theta)
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
        [test_sample[f"fluxerr_{band}"].to_numpy(float) for band in band_names]
    )

    metric_frames: list[pd.DataFrame] = []
    all_abs_delta_mag: dict[str, np.ndarray] = {}
    all_abs_noise: dict[str, np.ndarray] = {}
    for placement in PLACEMENTS:
        approximator = _build_approximator(placement, optimized_u)
        raw_spline, _, _, _ = approximator(
            jnp.asarray(native["time_gyr"]), jnp.asarray(raw_native)
        )
        raw_spline = np.asarray(jax.device_get(raw_spline))
        values = [
            np.asarray(jax.device_get(value))
            for value in sfh_forward_fn(theta_jax, jnp.asarray(raw_spline))
        ]
        result = dict(zip(native.keys(), values, strict=True))
        metrics, abs_delta_mag, abs_noise = _object_metrics(
            test_sample,
            native,
            result,
            fluxerr,
            n_nodes=N_NODES,
            grid_strategy=placement,
        )
        metric_frames.append(metrics)
        all_abs_delta_mag[placement] = abs_delta_mag
        all_abs_noise[placement] = abs_noise

    object_metrics = pd.concat(metric_frames, ignore_index=True)
    summary = _summarize_metrics(object_metrics)
    bands = _summarize_bands(test_sample, band_names, all_abs_delta_mag, all_abs_noise)
    gates, decisions = _selection_gates(summary, min_group_size=args.min_group_size)

    latent_train, train_knot_log_sfr, train_knot_time = _project_full_split(
        train,
        native_sfh_fn,
        placement=selected_placement,
        optimized_u=optimized_u,
        n_sfh_bins=n_sfh_bins,
        batch_size=args.batch_size,
    )
    latent_test, test_knot_log_sfr, test_knot_time = _project_full_split(
        test,
        native_sfh_fn,
        placement=selected_placement,
        optimized_u=optimized_u,
        n_sfh_bins=n_sfh_bins,
        batch_size=args.batch_size,
    )
    train_state = np.where(_atom_mask(train, atom_values), "main_sequence", "quenched")
    latent_diagnostics, latent_summary, correlation = _latent_diagnostics(
        latent_train, train_state
    )

    candidate_widths = sorted(
        {
            0.0001,
            0.00025,
            0.0005,
            0.001,
            0.002,
            args.dequantization_half_width_dex,
        }
    )
    validation_eval = validation_sample.copy()
    validation_atom = validation_eval["state"].to_numpy() == "main_sequence"
    train_prevalence = float(np.mean(_atom_mask(train, atom_values)))
    validation_eval["test_row"] = validation_eval["train_row"].to_numpy(int)
    validation_eval["redshift_bin"] = "train_validation"
    validation_eval["mass_bin"] = "train_validation"
    validation_eval["high_ssfr_tail"] = False
    validation_eval["population_weight"] = np.where(
        validation_atom,
        train_prevalence / max(int(np.sum(validation_atom)), 1),
        (1.0 - train_prevalence) / max(int(np.sum(~validation_atom)), 1),
    )
    validation_theta = theta_from_truth_frame(validation_eval).astype(np.float32)
    validation_theta_jax = jnp.asarray(validation_theta)
    validation_native_values = [
        np.asarray(jax.device_get(value))
        for value in sfh_forward_fn(validation_theta_jax, jnp.asarray(validation_sfh))
    ]
    validation_native = dict(zip(native.keys(), validation_native_values, strict=True))
    validation_fluxerr = np.column_stack(
        [validation_eval[f"fluxerr_{band}"].to_numpy(float) for band in band_names]
    )
    selected_approximator = _build_approximator(selected_placement, optimized_u)
    validation_contrasts = np.asarray(
        jax.device_get(
            selected_approximator(
                jnp.asarray(validation_time), jnp.asarray(validation_sfh)
            )[1]
        )
    )
    validation_dequantization_scan, _, _ = _evaluate_dequantization_widths(
        validation_eval,
        validation_native,
        validation_theta_jax,
        validation_contrasts,
        sfh_forward_fn,
        validation_fluxerr,
        placement=selected_placement,
        optimized_u=optimized_u,
        widths=candidate_widths,
        seed=args.seed + 1,
    )
    passing_widths = validation_dequantization_scan.loc[
        validation_dequantization_scan["passes_population_targets"],
        "half_width_dex",
    ].to_numpy(float)
    validation_target_passed = bool(len(passing_widths))
    selected_width = (
        float(np.max(passing_widths))
        if validation_target_passed
        else float(np.min(candidate_widths))
    )
    latent_train_dequantized, train_dequantized_count = _dequantize_exact_atoms(
        latent_train,
        half_width_dex=selected_width,
        seed=args.seed,
    )
    latent_test_dequantized, test_dequantized_count = _dequantize_exact_atoms(
        latent_test,
        half_width_dex=selected_width,
        seed=args.seed + 1,
    )
    test_contrasts = np.asarray(
        jax.device_get(
            selected_approximator(
                jnp.asarray(native["time_gyr"]), jnp.asarray(raw_native)
            )[1]
        )
    )
    test_dequantization_scan, test_metrics_by_width, test_summaries_by_width = (
        _evaluate_dequantization_widths(
            test_sample,
            native,
            theta_jax,
            test_contrasts,
            sfh_forward_fn,
            fluxerr,
            placement=selected_placement,
            optimized_u=optimized_u,
            widths=candidate_widths,
            seed=args.seed + 2,
        )
    )
    dequantized_metrics = test_metrics_by_width[selected_width]
    dequantized_summary_frame = test_summaries_by_width[selected_width]
    dequantized_grid = f"{selected_placement}_dequantized_{selected_width:.5g}"
    dequantization_summary = {
        "half_width_dex": selected_width,
        "selection_source": "balanced train validation; test not used",
        "validation_target_passed": validation_target_passed,
        "selection_rule": (
            "largest width passing both population targets"
            if validation_target_passed
            else "smallest candidate as minimum-distortion fallback"
        ),
        "train_values_dequantized": train_dequantized_count,
        "test_values_dequantized": test_dequantized_count,
        "validation_scan": validation_dequantization_scan.to_dict(orient="records"),
        "test_scan": test_dequantization_scan.to_dict(orient="records"),
        "population_p95_max_abs_delta_mag": _summary_value(
            dequantized_summary_frame,
            dequantized_grid,
            "population_weighted",
            "max_abs_delta_mag",
            "p95",
        ),
        "population_p95_noise_rms": _summary_value(
            dequantized_summary_frame,
            dequantized_grid,
            "population_weighted",
            "noise_rms",
            "p95",
        ),
        "quenched_p95_max_abs_delta_mag": _summary_value(
            dequantized_summary_frame,
            dequantized_grid,
            "quenched",
            "max_abs_delta_mag",
            "p95",
        ),
        "quenched_p95_noise_rms": _summary_value(
            dequantized_summary_frame,
            dequantized_grid,
            "quenched",
            "noise_rms",
            "p95",
        ),
    }

    latent_train.to_parquet(out / "feniks_spline_15d_train.parquet", index=False)
    latent_test.to_parquet(out / "feniks_spline_15d_test.parquet", index=False)
    latent_train_dequantized.to_parquet(
        out / "feniks_spline_15d_train_dequantized.parquet", index=False
    )
    latent_test_dequantized.to_parquet(
        out / "feniks_spline_15d_test_dequantized.parquet", index=False
    )
    pd.DataFrame(
        train_knot_log_sfr,
        columns=[f"node_log_sfr_{index:02d}" for index in range(N_NODES)],
    ).to_parquet(out / "feniks_spline_nodes_train.parquet", index=False)
    pd.DataFrame(
        test_knot_log_sfr,
        columns=[f"node_log_sfr_{index:02d}" for index in range(N_NODES)],
    ).to_parquet(out / "feniks_spline_nodes_test.parquet", index=False)
    object_metrics.to_csv(out / "spline_15d_object_metrics.csv", index=False)
    summary.to_csv(out / "spline_15d_summary_metrics.csv", index=False)
    bands.to_csv(out / "spline_15d_band_metrics.csv", index=False)
    gates.to_csv(out / "spline_15d_selection_gates.csv", index=False)
    proxy_scores.to_csv(out / "spline_15d_validation_proxy.csv", index=False)
    optimization_history.to_csv(
        out / "spline_15d_optimization_history.csv", index=False
    )
    latent_diagnostics.to_csv(out / "spline_15d_latent_diagnostics.csv", index=False)
    dequantized_metrics.to_csv(
        out / "spline_15d_dequantized_object_metrics.csv", index=False
    )
    dequantized_summary_frame.to_csv(
        out / "spline_15d_dequantized_summary_metrics.csv", index=False
    )
    validation_dequantization_scan.to_csv(
        out / "spline_15d_dequantization_validation_scan.csv", index=False
    )
    test_dequantization_scan.to_csv(
        out / "spline_15d_dequantization_test_scan.csv", index=False
    )
    np.savetxt(
        out / "spline_15d_latent_correlation.csv",
        correlation,
        delimiter=",",
        header=",".join(LATENT_NAMES),
        comments="",
    )

    _plot_closure_comparison(summary, out / "spline_15d_closure_comparison.png")
    _plot_node_placements(
        validation_time,
        validation_sfh,
        optimized_u,
        out / "spline_15d_node_placements.png",
    )
    _plot_contrast_distributions(
        latent_train,
        train_state,
        out / "spline_15d_contrast_distributions.png",
    )
    _plot_latent_correlation(correlation, out / "spline_15d_latent_correlation.png")

    contract = _clean_json(
        {
            "dimension": len(LATENT_NAMES),
            "columns": list(LATENT_NAMES),
            "physical_parameters": list(PHYSICAL_PARAMETERS),
            "sfh_parameterization": {
                "interpolator": "JAX PCHIP in log cosmic time and log SFR",
                "node_count": N_NODES,
                "independent_shape_contrasts": N_SHAPE,
                "contrast_definition": "q_i = logSFR(t_{i+1}) - logSFR(t_i)",
                "common_amplitude": "discarded; reconstructed from log10_stellar_mass",
                "selected_placement": selected_placement,
                "optimized_normalized_log_time_nodes": optimized_u,
                "positions_are_per_galaxy_latents": False,
            },
            "selection": {
                "source": "balanced train validation proxy, before test closure",
                "proxy_scores": proxy_scores.to_dict(orient="records"),
            },
            "dataset": {
                "source_train": str(args.dataset_dir / "train.parquet"),
                "source_test": str(args.dataset_dir / "test.parquet"),
                "projected_train": str(out / "feniks_spline_15d_train.parquet"),
                "projected_test": str(out / "feniks_spline_15d_test.parquet"),
                "projected_train_dequantized": str(
                    out / "feniks_spline_15d_train_dequantized.parquet"
                ),
                "projected_test_dequantized": str(
                    out / "feniks_spline_15d_test_dequantized.parquet"
                ),
                "train_rows": len(train),
                "test_rows": len(test),
            },
            "held_out_test": {
                "diagnostic_rows": len(test_sample),
                "main_sequence_prevalence": prevalence,
                "redshift_edges": redshift_edges,
                "mass_edges": mass_edges,
                "high_ssfr_threshold": ssfr_threshold,
            },
            "latent_diagnostics": latent_summary,
            "dequantization": dequantization_summary,
            "fixed_forward_assumptions": {
                "stellar_mdf": config["model"].get("stellar_metallicity_model"),
                "stellar_mdf_scatter_dex": config["model"].get(
                    "stellar_metallicity_scatter_dex"
                ),
                "igm": config["model"].get("igm_model"),
                "filters": band_names,
            },
            "storage_audit": {
                "node_log_sfr_train_shape": list(train_knot_log_sfr.shape),
                "node_time_train_range_gyr": [
                    float(np.min(train_knot_time)),
                    float(np.max(train_knot_time)),
                ],
                "node_log_sfr_test_shape": list(test_knot_log_sfr.shape),
                "node_time_test_range_gyr": [
                    float(np.min(test_knot_time)),
                    float(np.max(test_knot_time)),
                ],
            },
        }
    )
    (out / "feniks_spline_15d_contract.json").write_text(
        json.dumps(contract, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    payload = _clean_json(
        {
            "contract": contract,
            "decisions": decisions.to_dict(orient="records"),
            "gates": gates.to_dict(orient="records"),
            "summary": summary[summary["metric"].isin(CORE_METRICS)].to_dict(
                orient="records"
            ),
            "latent_diagnostics": latent_diagnostics.to_dict(orient="records"),
            "dequantization_summary": dequantization_summary,
        }
    )
    (out / "spline_15d_scan_payload.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(
        out,
        selected_placement=selected_placement,
        optimized_u=optimized_u,
        proxy_scores=proxy_scores,
        decisions=decisions,
        summary=summary,
        latent_summary=latent_summary,
        optimization_history=optimization_history,
        dequantization_summary=dequantization_summary,
        sample_size=len(test_sample),
        prevalence=prevalence,
    )
    print(f"Wrote 15D spline-prior scan to {out}")
    print(f"Selected placement from train validation: {selected_placement}")
    print(f"Optimized normalized log-time nodes: {optimized_u.tolist()}")
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()
