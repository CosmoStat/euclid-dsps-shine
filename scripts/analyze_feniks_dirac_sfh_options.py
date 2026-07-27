#!/usr/bin/env python3
"""Analyze FENIKS Diffstar atoms and DSPS-sufficient SFH representations."""

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
from scipy.interpolate import PchipInterpolator

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
STATE_LABELS = {
    True: "main_sequence_atom",
    False: "quenched_continuous",
}
REPRESENTATION_ORDER = (
    "age_weights_107_exact",
    "popcosmos_7_bins",
    "lookback_16_bins",
    "logtime_pchip_12",
    "logtime_pchip_20",
)
REPRESENTATION_COLORS = {
    "native_80": "#111111",
    "age_weights_107_exact": "#0072B2",
    "popcosmos_7_bins": "#D55E00",
    "lookback_16_bins": "#009E73",
    "logtime_pchip_12": "#CC79A7",
    "logtime_pchip_20": "#E69F00",
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
        default=Path("outputs/analysis/feniks_dirac_sfh_options_20260710"),
    )
    parser.add_argument("--sample-per-state", type=int, default=96)
    parser.add_argument("--seed", type=int, default=260710)
    return parser.parse_args()


def _mode(values: pd.Series) -> float:
    counts = values.value_counts(dropna=False)
    if counts.empty:
        raise ValueError(f"Cannot identify a mode for {values.name}")
    return float(counts.index[0])


def _atom_values(train: pd.DataFrame) -> dict[str, float]:
    return {name: _mode(train[GROUND_TRUTH_COLUMNS[name]]) for name in ATOM_NAMES}


def _atom_bits(frame: pd.DataFrame, atom_values: dict[str, float]) -> np.ndarray:
    return np.column_stack(
        [
            frame[GROUND_TRUTH_COLUMNS[name]].to_numpy(float) == atom_values[name]
            for name in ATOM_NAMES
        ]
    )


def _shared_atom_mask(frame: pd.DataFrame, atom_values: dict[str, float]) -> np.ndarray:
    bits = _atom_bits(frame, atom_values)
    coherent = np.all(bits, axis=1) | np.all(~bits, axis=1)
    if not np.all(coherent):
        raise ValueError(
            f"Found {int(np.sum(~coherent))} rows with incoherent Diffstar atom states"
        )
    return np.all(bits, axis=1)


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
    cdf = np.cumsum(weights) - 0.5 * weights
    cdf = cdf / np.sum(weights)
    return float(np.interp(quantile, cdf, values))


def _state_prevalence(
    frames: dict[str, pd.DataFrame], atom_values: dict[str, float]
) -> pd.DataFrame:
    rows = []
    for split, frame in frames.items():
        mask = _shared_atom_mask(frame, atom_values)
        bits = _atom_bits(frame, atom_values)
        patterns, counts = np.unique(bits.astype(int), axis=0, return_counts=True)
        pattern_text = "; ".join(
            f"{''.join(map(str, pattern.tolist()))}:{int(count)}"
            for pattern, count in zip(patterns, counts, strict=True)
        )
        rows.append(
            {
                "split": split,
                "rows": len(frame),
                "atom_count": int(np.sum(mask)),
                "atom_fraction": float(np.mean(mask)),
                "continuous_count": int(np.sum(~mask)),
                "continuous_fraction": float(np.mean(~mask)),
                "joint_atom_patterns": pattern_text,
                "all_four_masks_identical": bool(
                    all(np.array_equal(bits[:, 0], bits[:, i]) for i in range(1, 4))
                ),
            }
        )
    return pd.DataFrame(rows)


def _atom_support_table(
    frames: dict[str, pd.DataFrame], atom_values: dict[str, float]
) -> pd.DataFrame:
    rows = []
    for split, frame in frames.items():
        shared = _shared_atom_mask(frame, atom_values)
        for name in ATOM_NAMES:
            values = frame[GROUND_TRUTH_COLUMNS[name]].to_numpy(float)
            non_atom = values[~shared]
            atom = atom_values[name]
            closest_index = int(np.argmin(np.abs(non_atom - atom)))
            closest = float(non_atom[closest_index])
            rows.append(
                {
                    "split": split,
                    "parameter": name,
                    "atom_value": atom,
                    "atom_count": int(np.sum(values == atom)),
                    "atom_fraction": float(np.mean(values == atom)),
                    "continuous_count": len(non_atom),
                    "continuous_unique_count": int(np.unique(non_atom).size),
                    "continuous_min": float(np.min(non_atom)),
                    "continuous_max": float(np.max(non_atom)),
                    "closest_continuous_value": closest,
                    "signed_gap_atom_minus_closest": atom - closest,
                    "absolute_gap": abs(atom - closest),
                }
            )
    return pd.DataFrame(rows)


def _state_contrast_table(
    frame: pd.DataFrame, atom_values: dict[str, float]
) -> pd.DataFrame:
    mask = _shared_atom_mask(frame, atom_values)
    z = frame["redshift_true"].to_numpy(float)
    mass = frame["logsm_true"].to_numpy(float)
    logssfr = frame["logssfr_true"].to_numpy(float)
    color = frame["mag_true_lsst_r"].to_numpy(float) - frame[
        "mag_true_euclid_nisp_j"
    ].to_numpy(float)
    design = np.column_stack([np.ones(len(frame)), z, z**2, mass, mass**2, z * mass])
    finite_design = np.isfinite(design).all(axis=1)

    def residual_from_atom_trend(values: np.ndarray) -> np.ndarray:
        fit_mask = mask & finite_design & np.isfinite(values)
        coefficients = np.linalg.lstsq(design[fit_mask], values[fit_mask], rcond=None)[
            0
        ]
        return values - design @ coefficients

    variables = {
        "redshift": z,
        "log10_stellar_mass": mass,
        "log10_sSFR": logssfr,
        "log10_sSFR_residual_at_fixed_z_mass": residual_from_atom_trend(logssfr),
        "log10_SFR": frame["logsfr_true"].to_numpy(float),
        "log10_stellar_metallicity": frame["log10_stellar_metallicity_true"].to_numpy(
            float
        ),
        "dust_Av": frame["dust_av_true"].to_numpy(float),
        "dust_delta": frame["dust_delta_true"].to_numpy(float),
        "halo_logm0": frame["diffmah_logm0_true"].to_numpy(float),
        "central_indicator": frame["central_true"].to_numpy(float),
        "color_lsst_r_minus_euclid_J": color,
        "color_residual_at_fixed_z_mass": residual_from_atom_trend(color),
    }
    rows = []
    for label, values in variables.items():
        values = np.asarray(values, dtype=float)
        atom = values[mask]
        continuous = values[~mask]
        pooled = np.sqrt(0.5 * (np.nanvar(atom) + np.nanvar(continuous)))
        mean_delta = float(np.nanmean(continuous) - np.nanmean(atom))
        rows.append(
            {
                "variable": label,
                "main_sequence_atom_median": float(np.nanmedian(atom)),
                "main_sequence_atom_q16": float(np.nanquantile(atom, 0.16)),
                "main_sequence_atom_q84": float(np.nanquantile(atom, 0.84)),
                "quenched_continuous_median": float(np.nanmedian(continuous)),
                "quenched_continuous_q16": float(np.nanquantile(continuous, 0.16)),
                "quenched_continuous_q84": float(np.nanquantile(continuous, 0.84)),
                "continuous_minus_atom_mean": mean_delta,
                "standardized_mean_difference": (
                    mean_delta / pooled if pooled > 0.0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _state_fraction_grid(
    frame: pd.DataFrame, atom_values: dict[str, float]
) -> pd.DataFrame:
    atom = _shared_atom_mask(frame, atom_values)
    z = frame["redshift_true"].to_numpy(float)
    mass = frame["logsm_true"].to_numpy(float)
    z_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5])
    mass_edges = np.array([7.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 12.5])
    rows = []
    for iz in range(len(z_edges) - 1):
        for im in range(len(mass_edges) - 1):
            selected = (
                (z >= z_edges[iz])
                & (z < z_edges[iz + 1])
                & (mass >= mass_edges[im])
                & (mass < mass_edges[im + 1])
            )
            count = int(np.sum(selected))
            rows.append(
                {
                    "z_low": z_edges[iz],
                    "z_high": z_edges[iz + 1],
                    "mass_low": mass_edges[im],
                    "mass_high": mass_edges[im + 1],
                    "count": count,
                    "quenched_continuous_fraction": (
                        float(np.mean(~atom[selected])) if count else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _balanced_sample(
    frame: pd.DataFrame,
    atom_values: dict[str, float],
    *,
    sample_per_state: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    atom = _shared_atom_mask(frame, atom_values)
    rng = np.random.default_rng(seed)
    atom_indices = np.flatnonzero(atom)
    continuous_indices = np.flatnonzero(~atom)
    n_atom = min(sample_per_state, len(atom_indices))
    n_continuous = min(sample_per_state, len(continuous_indices))
    chosen = np.concatenate(
        [
            rng.choice(atom_indices, n_atom, replace=False),
            rng.choice(continuous_indices, n_continuous, replace=False),
        ]
    )
    rng.shuffle(chosen)
    sample = frame.iloc[chosen].reset_index(drop=True)
    sample_atom = _shared_atom_mask(sample, atom_values)
    prevalence = float(np.mean(atom))
    weights = np.where(
        sample_atom,
        prevalence / max(int(np.sum(sample_atom)), 1),
        (1.0 - prevalence) / max(int(np.sum(~sample_atom)), 1),
    )
    return sample, sample_atom, weights


def _build_forward_functions(context: dsps_model.DspsContext):
    names = tuple(DIFFSKY_BASIC_PARAMETER_NAMES)

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
        weights = jnp.asarray(age_weights, dtype=jnp.float32)
        weights = weights / jnp.maximum(jnp.sum(weights), 1.0e-30)
        sed_by_age = (
            jnp.clip(ssp_flux_z, 0.0, jnp.inf)
            * weights[:, None]
            * jnp.asarray(formed_mass, dtype=jnp.float32)
        )
        tau2, dust_index_n, tau1_over_tau2 = dsps_model.diffsky_basic_dust_params_jax(
            params
        )
        wave = dsps_model._context_ssp_wave(context)
        dusted_by_age = dsps_model.apply_popcosmos_dust_by_age_jax(
            wave,
            dsps_model._context_ssp_lg_age_gyr(context),
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
        scaled_sfr, formed_mass, _ = dsps_model.normalize_sfh_to_stellar_mass_jax(
            t_table,
            raw_sfr,
            dsps_model._context_ssp_lg_age_gyr(context),
            t_obs,
            params["log10_stellar_mass"],
            surviving,
        )
        age_weights = calc_age_weights_from_sfh_table(
            t_table,
            scaled_sfr,
            dsps_model._context_ssp_lg_age_gyr(context),
            t_obs,
        )
        mags, sed = render_from_age_weights(theta, age_weights, formed_mass)
        return t_table, scaled_sfr, age_weights, formed_mass, mags, sed

    def native_sfh_single(theta: jnp.ndarray):
        params = params_from_theta(theta)
        z_obs = jnp.asarray(params["z_obs"], dtype=jnp.float32)
        t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]
        t_table = jnp.linspace(0.05, jnp.maximum(t_obs, 0.06), context.n_sfh_bins)
        return dsps_model.build_diffsky_basic_sfh_table_jax(t_table, t_obs, params)

    sfh_forward = jax.jit(jax.vmap(sfh_single, in_axes=(0, 0)))
    native_sfh = jax.jit(jax.vmap(native_sfh_single, in_axes=0))
    age_forward = jax.jit(jax.vmap(render_from_age_weights, in_axes=(0, 0, 0)))
    return native_sfh, sfh_forward, age_forward


def _pchip_approximation(
    t_table: np.ndarray, sfh: np.ndarray, n_knots: int
) -> np.ndarray:
    output = np.empty_like(sfh)
    for index, (time, sfr) in enumerate(zip(t_table, sfh, strict=True)):
        log_time = np.log10(np.maximum(time, 1.0e-5))
        log_sfr = np.log10(np.maximum(sfr, 1.0e-30))
        knots = np.linspace(log_time[0], log_time[-1], n_knots)
        values = np.interp(knots, log_time, log_sfr)
        output[index] = 10.0 ** PchipInterpolator(knots, values)(log_time)
    return np.clip(output, 1.0e-30, np.inf)


def _bin_average(time: np.ndarray, sfr: np.ndarray, low: float, high: float) -> float:
    lo = max(float(low), float(time[0]))
    hi = min(float(high), float(time[-1]))
    if hi <= lo:
        return float(np.interp(0.5 * (lo + hi), time, sfr))
    interior = time[(time > lo) & (time < hi)]
    nodes = np.concatenate([[lo], interior, [hi]])
    values = np.interp(nodes, time, sfr)
    return float(np.trapezoid(values, nodes) / max(hi - lo, 1.0e-8))


def _lookback_bin_approximation(
    t_table: np.ndarray, sfh: np.ndarray, n_bins: int
) -> np.ndarray:
    output = np.empty_like(sfh)
    fractions = np.concatenate([[0.0], np.geomspace(1.0e-3, 1.0, n_bins)])
    for index, (time, sfr) in enumerate(zip(t_table, sfh, strict=True)):
        t_obs = float(time[-1])
        lookback_edges = fractions * t_obs
        cosmic_low = t_obs - lookback_edges[1:]
        cosmic_high = t_obs - lookback_edges[:-1]
        averages = np.asarray(
            [
                _bin_average(time, sfr, low, high)
                for low, high in zip(cosmic_low, cosmic_high, strict=True)
            ]
        )
        lookback = np.clip(t_obs - time, 0.0, t_obs)
        bins = np.searchsorted(lookback_edges, lookback, side="right") - 1
        output[index] = averages[np.clip(bins, 0, n_bins - 1)]
    return np.clip(output, 1.0e-30, np.inf)


def _popcosmos_approximation(t_table: np.ndarray, sfh: np.ndarray) -> np.ndarray:
    output = np.empty_like(sfh)
    for index, (time, sfr) in enumerate(zip(t_table, sfh, strict=True)):
        t_obs = float(time[-1])
        bins = np.asarray(
            dsps_model.project_sfh_to_popcosmos_sfr_bins_jax(
                jnp.asarray(time), jnp.asarray(sfr), jnp.asarray(t_obs)
            )
        )
        edges = np.asarray(dsps_model.build_popcosmos_lookback_bin_edges_jax(t_obs))
        lookback = np.clip(t_obs - time, 0.0, edges[-1])
        indices = np.searchsorted(edges, lookback, side="right") - 1
        output[index] = bins[np.clip(indices, 0, 6)]
    return np.clip(output, 1.0e-30, np.inf)


def _representation_metrics(
    name: str,
    result: dict[str, np.ndarray],
    native: dict[str, np.ndarray],
    atom_mask: np.ndarray,
    population_weights: np.ndarray,
) -> list[dict[str, Any]]:
    dt = np.diff(native["t_table"], axis=1)
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
    age_l1 = np.sum(np.abs(result["age_weights"] - native["age_weights"]), axis=1)
    sed_l1 = np.sum(np.abs(result["sed"] - native["sed"]), axis=1) / np.maximum(
        np.sum(np.abs(native["sed"]), axis=1), 1.0e-30
    )
    mag_abs = np.abs(result["mags"] - native["mags"])
    max_mag = np.max(mag_abs, axis=1)
    metrics = {
        "sfh_mass_normalized_l1": sfh_l1,
        "log_sfh_rmse_dex": log_sfh_rmse,
        "age_weight_l1": age_l1,
        "sed_relative_l1": sed_l1,
        "max_abs_delta_mag": max_mag,
    }
    rows = []
    groups = {
        "population_weighted": np.ones(len(atom_mask), dtype=bool),
        "main_sequence_atom": atom_mask,
        "quenched_continuous": ~atom_mask,
    }
    for group, selected in groups.items():
        weights = (
            population_weights[selected]
            if group == "population_weighted"
            else np.ones(int(np.sum(selected)), dtype=float)
        )
        for metric, values in metrics.items():
            values = values[selected]
            rows.append(
                {
                    "representation": name,
                    "group": group,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.average(values, weights=weights)),
                    "median": _weighted_quantile(values, weights, 0.5),
                    "p95": _weighted_quantile(values, weights, 0.95),
                    "max": float(np.max(values)),
                }
            )
        for threshold in (0.001, 0.01, 0.05, 0.1):
            values = max_mag[selected] > threshold
            rows.append(
                {
                    "representation": name,
                    "group": group,
                    "metric": f"fraction_max_delta_mag_gt_{threshold:g}",
                    "n": len(values),
                    "mean": float(np.average(values.astype(float), weights=weights)),
                    "median": float("nan"),
                    "p95": float("nan"),
                    "max": float(np.max(values)),
                }
            )
    return rows


def _counterfactual_metrics(
    label: str,
    reference_mags: np.ndarray,
    counterfactual_mags: np.ndarray,
    selected: np.ndarray,
    band_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta = np.abs(counterfactual_mags[selected] - reference_mags[selected])
    per_object = np.max(delta, axis=1)
    summary = pd.DataFrame(
        [
            {
                "counterfactual": label,
                "n": len(per_object),
                "median_max_abs_delta_mag": float(np.median(per_object)),
                "p95_max_abs_delta_mag": float(np.quantile(per_object, 0.95)),
                "max_abs_delta_mag": float(np.max(per_object)),
                "fraction_gt_0p01_mag": float(np.mean(per_object > 0.01)),
                "fraction_gt_0p05_mag": float(np.mean(per_object > 0.05)),
                "fraction_gt_0p10_mag": float(np.mean(per_object > 0.10)),
            }
        ]
    )
    by_band = pd.DataFrame(
        {
            "counterfactual": label,
            "band": band_names,
            "median_abs_delta_mag": np.median(delta, axis=0),
            "p95_abs_delta_mag": np.quantile(delta, 0.95, axis=0),
            "max_abs_delta_mag": np.max(delta, axis=0),
        }
    )
    return summary, by_band


def _plot_dirac_support(
    train: pd.DataFrame,
    atom_values: dict[str, float],
    path: Path,
) -> None:
    atom = _shared_atom_mask(train, atom_values)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for axis, name in zip(axes.flat, ATOM_NAMES, strict=True):
        values = train[GROUND_TRUTH_COLUMNS[name]].to_numpy(float)
        continuous = values[~atom]
        axis.hist(continuous, bins=45, density=True, color="#009E73", alpha=0.8)
        axis.axvline(atom_values[name], color="#111111", linewidth=2.2)
        axis.text(
            0.03,
            0.94,
            f"shared atom = {atom_values[name]:.6g}\n"
            f"{np.mean(atom):.3%} of train rows",
            transform=axis.transAxes,
            va="top",
            fontsize=10,
        )
        axis.set_title(name)
        axis.set_xlabel("physical parameter value")
        axis.set_ylabel("continuous-branch density")
    fig.suptitle(
        "One shared main-sequence atom, not four independent discrete variables",
        fontsize=15,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_state_contrasts(
    frame: pd.DataFrame,
    atom_values: dict[str, float],
    path: Path,
) -> None:
    atom = _shared_atom_mask(frame, atom_values)
    variables = [
        ("redshift_true", "Redshift"),
        ("logsm_true", "log10 stellar mass"),
        ("logssfr_true", "log10 sSFR"),
        ("dust_av_true", "Dust A_V"),
        ("diffmah_logm0_true", "Halo logm0"),
    ]
    color = frame["mag_true_lsst_r"] - frame["mag_true_euclid_nisp_j"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, (column, label) in zip(axes.flat[:5], variables, strict=True):
        values = frame[column].to_numpy(float)
        bins = np.histogram_bin_edges(values[np.isfinite(values)], bins=38)
        axis.hist(
            values[atom],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color="#0072B2",
            label="main-sequence atom",
        )
        axis.hist(
            values[~atom],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color="#D55E00",
            label="quenched continuous",
        )
        axis.set_xlabel(label)
        axis.set_ylabel("density")
    bins = np.histogram_bin_edges(color[np.isfinite(color)], bins=38)
    axes.flat[5].hist(
        color[atom],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        color="#0072B2",
    )
    axes.flat[5].hist(
        color[~atom],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        color="#D55E00",
    )
    axes.flat[5].set_xlabel("LSST r - Euclid J true color")
    axes.flat[5].set_ylabel("density")
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "The rare continuous branch is a distinct galaxy population", fontsize=15
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_state_grid(grid: pd.DataFrame, path: Path) -> None:
    z_pairs = grid[["z_low", "z_high"]].drop_duplicates().to_numpy()
    mass_pairs = grid[["mass_low", "mass_high"]].drop_duplicates().to_numpy()
    values = np.full((len(mass_pairs), len(z_pairs)), np.nan)
    counts = np.zeros_like(values)
    for row in grid.itertuples(index=False):
        iz = int(np.flatnonzero(np.all(z_pairs == [row.z_low, row.z_high], axis=1))[0])
        im = int(
            np.flatnonzero(np.all(mass_pairs == [row.mass_low, row.mass_high], axis=1))[
                0
            ]
        )
        if row.count >= 20:
            values[im, iz] = row.quenched_continuous_fraction
        counts[im, iz] = row.count
    fig, axis = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    image = axis.imshow(values, origin="lower", aspect="auto", cmap="viridis")
    for im in range(values.shape[0]):
        for iz in range(values.shape[1]):
            text = "<20" if counts[im, iz] < 20 else f"{values[im, iz]:.1%}"
            axis.text(iz, im, text, ha="center", va="center", fontsize=8)
    axis.set_xticks(range(len(z_pairs)), [f"{a:g}-{b:g}" for a, b in z_pairs])
    axis.set_yticks(range(len(mass_pairs)), [f"{a:g}-{b:g}" for a, b in mass_pairs])
    axis.set_xlabel("redshift bin")
    axis.set_ylabel("log10 stellar-mass bin")
    axis.set_title("Quenched continuous-branch fraction (test split)")
    fig.colorbar(image, ax=axis, label="continuous-branch fraction")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_sfh_examples(
    sample: pd.DataFrame,
    atom_mask: np.ndarray,
    native: dict[str, np.ndarray],
    representations: dict[str, dict[str, np.ndarray]],
    ssp_lg_age_gyr: np.ndarray,
    path: Path,
) -> None:
    atom_indices = np.flatnonzero(atom_mask)
    continuous_indices = np.flatnonzero(~atom_mask)
    atom_ssfr = sample.loc[atom_indices, "logssfr_true"].to_numpy(float)
    continuous_ssfr = sample.loc[continuous_indices, "logssfr_true"].to_numpy(float)
    selected = [
        atom_indices[int(np.argmin(np.abs(atom_ssfr - np.median(atom_ssfr))))],
        atom_indices[int(np.argmax(atom_ssfr))],
        continuous_indices[
            int(np.argmin(np.abs(continuous_ssfr - np.median(continuous_ssfr))))
        ],
        continuous_indices[int(np.argmin(continuous_ssfr))],
    ]
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
    for row, index in enumerate(selected):
        label = STATE_LABELS[bool(atom_mask[index])]
        time = native["t_table"][index]
        axes[row, 0].plot(
            time,
            native["sfh"][index],
            color=REPRESENTATION_COLORS["native_80"],
            linewidth=2.3,
            label="native 80-point SFH",
        )
        for name in REPRESENTATION_ORDER[1:]:
            axes[row, 0].plot(
                time,
                representations[name]["sfh"][index],
                color=REPRESENTATION_COLORS[name],
                linewidth=1.25,
                alpha=0.9,
                label=name,
            )
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_ylim(bottom=1.0e-5)
        axes[row, 0].set_xlabel("cosmic time [Gyr]")
        axes[row, 0].set_ylabel("SFR [Msun/yr]")
        axes[row, 0].set_title(
            f"{label}; z={sample.loc[index, 'redshift_true']:.2f}; "
            f"log sSFR={sample.loc[index, 'logssfr_true']:.2f}"
        )
        age_gyr = 10.0 ** np.asarray(ssp_lg_age_gyr)
        axes[row, 1].plot(
            age_gyr,
            native["age_weights"][index],
            color=REPRESENTATION_COLORS["native_80"],
            linewidth=2.3,
            label="native",
        )
        for name in REPRESENTATION_ORDER[1:]:
            axes[row, 1].plot(
                age_gyr,
                representations[name]["age_weights"][index],
                color=REPRESENTATION_COLORS[name],
                linewidth=1.25,
                alpha=0.9,
                label=name,
            )
        axes[row, 1].set_xscale("log")
        axes[row, 1].set_xlabel("SSP stellar age [Gyr]")
        axes[row, 1].set_ylabel("formed-mass fraction")
        axes[row, 1].set_title("DSPS age weights")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("SFH approximations and the age weights actually consumed by DSPS")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_representation_accuracy(
    representations: dict[str, dict[str, np.ndarray]],
    native: dict[str, np.ndarray],
    atom_mask: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
    for name in REPRESENTATION_ORDER:
        delta = np.max(np.abs(representations[name]["mags"] - native["mags"]), axis=1)
        for selected, style, label_suffix in (
            (atom_mask, "-", "main sequence"),
            (~atom_mask, "--", "quenched"),
        ):
            values = np.sort(np.maximum(delta[selected], 1.0e-8))
            cdf = np.arange(1, len(values) + 1) / len(values)
            axes[0].plot(
                values,
                cdf,
                linestyle=style,
                color=REPRESENTATION_COLORS[name],
                linewidth=1.8,
                label=f"{name}, {label_suffix}",
            )
        age_l1 = np.sum(
            np.abs(representations[name]["age_weights"] - native["age_weights"]),
            axis=1,
        )
        axes[1].scatter(
            age_l1[atom_mask],
            delta[atom_mask],
            s=13,
            alpha=0.45,
            color=REPRESENTATION_COLORS[name],
            marker="o",
        )
        axes[1].scatter(
            age_l1[~atom_mask],
            delta[~atom_mask],
            s=18,
            alpha=0.55,
            color=REPRESENTATION_COLORS[name],
            marker="x",
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("maximum absolute magnitude error over 18 bands")
    axes[0].set_ylabel("empirical CDF")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].set_xscale("symlog", linthresh=1.0e-7)
    axes[1].set_yscale("symlog", linthresh=1.0e-7)
    axes[1].set_xlabel("L1 distance in 107 DSPS age weights")
    axes[1].set_ylabel("maximum absolute magnitude error")
    axes[1].grid(alpha=0.2)
    axes[1].text(
        0.03,
        0.96,
        "circles: main sequence\ncrosses: quenched",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
    )
    fig.suptitle("Photometric fidelity of SFH compression choices")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_counterfactuals(by_band: pd.DataFrame, path: Path) -> None:
    labels = list(dict.fromkeys(by_band["counterfactual"]))
    fig, axes = plt.subplots(
        len(labels), 1, figsize=(13, 4.3 * len(labels)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for axis, label in zip(axes, labels, strict=True):
        subset = by_band[by_band["counterfactual"] == label]
        x = np.arange(len(subset))
        axis.bar(
            x,
            subset["p95_abs_delta_mag"],
            color="#D55E00" if "atom_to" in label else "#0072B2",
            alpha=0.85,
            label="p95 |delta mag|",
        )
        axis.plot(
            x,
            subset["median_abs_delta_mag"],
            color="#111111",
            marker="o",
            linewidth=1.5,
            label="median |delta mag|",
        )
        axis.axhline(0.05, color="#009E73", linestyle="--", linewidth=1.2)
        axis.set_ylabel("absolute magnitude change")
        axis.set_title(label)
        axis.legend(frameon=False)
    axes[-1].set_xticks(np.arange(len(subset)), subset["band"], rotation=55, ha="right")
    fig.suptitle("Changing the shared quenching state changes the photometry")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_forward_drift(
    stored_mags: np.ndarray,
    current_mags: np.ndarray,
    band_names: list[str],
    path: Path,
) -> pd.DataFrame:
    delta = current_mags - stored_mags
    summary = pd.DataFrame(
        {
            "band": band_names,
            "median_current_minus_stored_mag": np.median(delta, axis=0),
            "p95_abs_delta_mag": np.quantile(np.abs(delta), 0.95, axis=0),
            "max_abs_delta_mag": np.max(np.abs(delta), axis=0),
        }
    )
    fig, axis = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    x = np.arange(len(band_names))
    axis.bar(x, summary["p95_abs_delta_mag"], color="#CC79A7", alpha=0.85)
    axis.plot(
        x,
        np.abs(summary["median_current_minus_stored_mag"]),
        color="#111111",
        marker="o",
        label="|median current - stored|",
    )
    axis.axhline(5.0e-4, color="#009E73", linestyle="--", label="closure tolerance")
    axis.set_yscale("log")
    axis.set_xticks(x, band_names, rotation=55, ha="right")
    axis.set_ylabel("magnitude difference")
    axis.set_title("Stored 18-band closure photometry vs current local forward")
    axis.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return summary


def _format_metric(
    metrics: pd.DataFrame,
    representation: str,
    metric: str,
    statistic: str = "p95",
    group: str = "population_weighted",
) -> float:
    row = metrics[
        (metrics["representation"] == representation)
        & (metrics["group"] == group)
        & (metrics["metric"] == metric)
    ]
    return float(row.iloc[0][statistic]) if len(row) else float("nan")


def _build_reports(
    out: Path,
    *,
    prevalence: pd.DataFrame,
    support: pd.DataFrame,
    contrasts: pd.DataFrame,
    representation_metrics: pd.DataFrame,
    counterfactual_summary: pd.DataFrame,
    forward_drift: pd.DataFrame,
    sample_size: int,
    n_ssp_ages: int,
    n_ssp_metallicities: int,
) -> None:
    train_prev = prevalence.loc[prevalence["split"] == "train"].iloc[0]
    test_prev = prevalence.loc[prevalence["split"] == "test"].iloc[0]
    clip = counterfactual_summary[
        counterfactual_summary["counterfactual"] == "atom_to_nearest_continuous_proxy"
    ].iloc[0]
    force = counterfactual_summary[
        counterfactual_summary["counterfactual"] == "continuous_to_atom"
    ].iloc[0]
    best_candidates = []
    for name in REPRESENTATION_ORDER[1:]:
        state_p95 = max(
            _format_metric(
                representation_metrics,
                name,
                "max_abs_delta_mag",
                group="main_sequence_atom",
            ),
            _format_metric(
                representation_metrics,
                name,
                "max_abs_delta_mag",
                group="quenched_continuous",
            ),
        )
        best_candidates.append((name, state_p95))
    best_name, best_worst_state_p95 = min(best_candidates, key=lambda item: item[1])
    direct_p95 = _format_metric(
        representation_metrics, "age_weights_107_exact", "max_abs_delta_mag"
    )
    drift_p95 = float(forward_drift["p95_abs_delta_mag"].max())
    support_train = support[support["split"] == "train"]
    support_lines = "\n".join(
        f"| `{row.parameter}` | {row.atom_value:.9g} | {row.atom_fraction:.3%} | "
        f"{row.continuous_min:.6g} | {row.continuous_max:.6g} | "
        f"{row.absolute_gap:.6g} |"
        for row in support_train.itertuples(index=False)
    )
    metric_lines = "\n".join(
        f"| `{name}` | "
        f"{_format_metric(representation_metrics, name, 'max_abs_delta_mag', 'median'):.4g} | "
        f"{_format_metric(representation_metrics, name, 'max_abs_delta_mag', 'p95'):.4g} | "
        f"{_format_metric(representation_metrics, name, 'max_abs_delta_mag', 'p95', 'quenched_continuous'):.4g} | "
        f"{_format_metric(representation_metrics, name, 'age_weight_l1', 'p95'):.4g} | "
        f"{_format_metric(representation_metrics, name, 'sfh_mass_normalized_l1', 'p95'):.4g} |"
        for name in REPRESENTATION_ORDER
    )
    contrast_lines = "\n".join(
        f"| {row.variable} | {row.main_sequence_atom_median:.4g} | "
        f"{row.quenched_continuous_median:.4g} | "
        f"{row.standardized_mean_difference:.3g} |"
        for row in contrasts.itertuples(index=False)
    )
    report = f"""# FENIKS Dirac and SFH representation decision study

## Executive conclusion

The four apparent Dirac peaks are **one shared binary Diffstar state**, not four
independent discrete coordinates. In the train split, {train_prev.atom_fraction:.3%}
of rows carry the exact main-sequence/no-quenching tuple and only
{train_prev.continuous_fraction:.3%} ({int(train_prev.continuous_count)} rows) carry
continuous quenching parameters. Diffstar's population model explicitly generates
main-sequence and quenched outcomes and selects them with one `mc_is_q` flag.

**Recommended design:** keep the native 18D Diffsky truth for auditability, but do
not fit one ordinary continuous density to all 18 coordinates. Use a hierarchical
mixed model with one quenching indicator and branch-specific continuous densities.
For the SED-closure benchmark, add an SED-native product containing DSPS age-mass
weights (or normalized age weights plus formed mass), metallicity, dust, and
redshift. This two-track design is more faithful than forcing a single latent
representation to serve both population generation and photometric sufficiency.

Option 2 (move the atom into the continuous support and clip it back later) is
rejected unless it is rewritten as an explicit censored/mixture likelihood. The
distance or clipping flag is already a discrete latent, and removing it makes the
mapping non-invertible. Option 3 is acceptable only for a deliberately
main-sequence-only benchmark; it is not a model of the full FENIKS population.

## 1. Data audit

| Parameter | Exact atom | Train fraction | Continuous min | Continuous max | Nearest-support gap |
| --- | ---: | ---: | ---: | ---: | ---: |
{support_lines}

All four atom masks are identical in train, validation, and test. Therefore the
joint state space contains only `1111` (the shared atom) and `0000` (the fully
continuous quenching branch), rather than 16 combinations of four Bernoulli
variables.

![Shared atom support](dirac_joint_support.png)

## 2. What the discrete state represents

The local Diffsky pipeline calls `mc_diffstar_params_galpop`, which returns
main-sequence parameters, quenched parameters, `frac_q`, and `mc_is_q`. Diffstar
sets the unquenched quenching coordinates from the same fixed unbounded tuple
`(5, 5, 5, 5)`; bounding sigmoids produce the four exact physical values in the
catalog. The continuous branch is the quenched outcome.

| Observable/property | Main-sequence atom median | Quenched continuous median | Standardized mean difference (Q - MS) |
| --- | ---: | ---: | ---: |
{contrast_lines}

![State contrasts](state_population_contrasts.png)

![State fraction](state_fraction_redshift_mass.png)

The continuous branch is rare but not random contamination. It occupies a
different mass/redshift/color/halo region. Its absolute sSFR need not be lower in
this observability-selected sample because the branch is concentrated at higher
redshift and halo mass; `mc_is_q` is a generative state, not an observed-sSFR
threshold. Removing it still changes scientifically meaningful conditional and
population tails, even if the raw row fraction is only
{test_prev.continuous_fraction:.3%} in the inference-ready test split.

## 3. Assessment of the four proposed options

### Option 1: hierarchical discrete/continuous model - recommended for the 18D prior

Use one state `q` and factorize the joint distribution, for example
`p(q) p(theta_nonq | q) p(theta_q | q=1, theta_nonq)`. An equivalent mixture is
`p(q) p(theta | q)`. A discrete normalizing flow is unnecessary for this binary
state: use a Bernoulli mass for `q` and ordinary flows only on continuous branch
coordinates. For amortized inference, predict `p(q | photometry)` with a Bernoulli
head and condition the continuous posterior on `q`.

The practical limitation is sample size: only {int(train_prev.continuous_count)}
quenched rows are present in the 40k train split. A free 18D quenched-branch flow is
over-parameterized. Prefer either more quenched proposals/stratified generation,
a lower-dimensional conditional quenching model, or partial parameter sharing
between branches.

### Option 2: clip/move the atom - reject as stated

Even the nearest-support joint proxy (the closest observed continuous quenching
tuple in standardized parameter distance) changes the p95 maximum 18-band magnitude by {clip.p95_max_abs_delta_mag:.3g}
mag for atom galaxies before restoration; {clip.fraction_gt_0p05_mag:.1%} exceed
0.05 mag. Restoring the atom requires retaining a clipping distance or flag, which
is precisely the missing discrete state. Thresholding flow samples back to the
atom additionally creates an arbitrary non-invertible map and miscalibrates the
atom probability.

### Option 3: remove/fix the continuous branch - only for a restricted benchmark

Dropping the continuous branch removes {test_prev.continuous_fraction:.3%} of the
test population and disproportionately removes high-mass/high-redshift objects
with distinct conditional SFHs and colors. If quenched objects are retained but
their four parameters are forced to the atom, the p95
maximum magnitude change is {force.p95_max_abs_delta_mag:.3g} mag, with
{force.fraction_gt_0p05_mag:.1%} above 0.05 mag. This is not an exact closure model.

![Counterfactual state changes](quenching_state_counterfactual_photometry.png)

### Option 4: store an SED-native SFH representation - recommended as a second product

The current local DSPS path evaluates Diffstar on 80 cosmic-time points, converts
that SFH to {n_ssp_ages} SSP-age weights, combines them with a
{n_ssp_metallicities}-point lognormal metallicity distribution, normalizes by
stellar mass/survival fraction, applies dust and IGM, and integrates 18 filters.

For this exact current forward, a sufficient per-galaxy representation is:

1. `z_obs`;
2. either `formed_mass_msun + age_weights[{n_ssp_ages}]`, or absolute formed-mass
   weights on the SSP age grid;
3. `log10_stellar_metallicity` and the MDF scatter (currently globally fixed to
   0.2 dex);
4. `dust_av` and `dust_delta` (with the remaining dust settings fixed globally).

The SSP grid, cosmology, filters, IGM prescription, dust constants, and code
version are global provenance, not learned per-galaxy truths. Diffstar and Diffmah
parameters are not needed once the age-mass weights have been stored.

For exact *Diffsky/FENIKS* rather than local DSPS-closure photometry, store the full
`phot_info.ssp_weights[metallicity, age]` (or enough factors to reconstruct it),
plus its mass normalization and attenuation variables. `age_weights =
sum(ssp_weights, axis=metallicity)` alone loses metallicity-age coupling unless a
separable MDF is explicitly assumed. The current 18D parquet also omits the
Diffsky burstiness realization, so it cannot reconstruct the original FENIKS SED
exactly from its 18 columns.

### Audited code path and primary references

- `euclid_dsps/synthetic_diffsky/backend.py:93-116` builds the FENIKS proposal
  realization, while `backend.py:232-266` copies the 18 compact coordinates but
  currently omits `mc_sfh_type`, `sfh_table`, and `ssp_weights`.
- `diffsky/experimental/mc_diffstarpop_wrappers.py:121-151` receives
  `sfh_params_ms`, `sfh_params_q`, `frac_q`, and `mc_is_q`, then selects one branch.
- The installed Diffstar source defines the common unquenched tuple at
  `diffstar/kernels/quenching_kernels.py:211-213`.
- `euclid_dsps/model.py:2253-2396` shows the active local sequence from Diffstar SFH
  through DSPS age weights, mass normalization, SSP combination, dust, IGM, and
  magnitudes; the per-object Diffstar/Diffmah evaluation starts at
  `model.py:2606`.
- Diffsky's `ssp_weight_kernels.py:390-522` shows the separate quenched/smooth/bursty
  selection and stores full SSP weights in `MCPhotInfo`.

The [official Diffstar population tutorial](https://diffstar.readthedocs.io/en/latest/demo_diffstar_sfh.html)
documents the two main-sequence/quenched outcomes and the `mc_is_q` selection. The
[official DSPS quickstart](https://dsps.readthedocs.io/en/stable/dsps_quickstart.html)
documents that galaxy SEDs are weighted sums over the SSP age-metallicity grid and
accept a tabulated SFH plus metallicity model.

## 4. SFH compression benchmark

The benchmark uses {sample_size} held-out objects, balanced across the two states,
with population metrics reweighted to the actual test prevalence.

| Representation | Median max abs mag | Population p95 max abs mag | Quenched p95 max abs mag | p95 age-weight L1 | p95 normalized SFH L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
{metric_lines}

Direct age weights reproduce the current local forward to p95
{direct_p95:.3g} mag. Among reduced SFH approximations, `{best_name}` minimizes
the worse of the two state-specific p95 errors in this tested set, at
{best_worst_state_p95:.3g} mag. A spline should only be promoted if its worst-state
and tail errors satisfy the scientific tolerance; a visually good SFH is not
sufficient.

![SFH examples](sfh_reconstruction_examples.png)

![Representation accuracy](sfh_representation_photometry_accuracy.png)

## 5. Current dataset/forward drift blocker

The stored parquet was validated at repository commit
`5a41c67e66d88bec61b4158b54562909cf340223` with JAX 0.10.2. In the currently
available local `shine` environment and current checkout, recomputation no longer
matches the stored 18-band magnitudes: the largest per-band p95 discrepancy in the
balanced test sample is {drift_p95:.3g} mag, far above the 5e-4 closure tolerance.
The SFH-compression comparisons above are therefore relative to one internally
consistent current-forward baseline; they must not be presented as a new exact
validation of the stored parquet.

Before producing the next science dataset, freeze and record all runtime package
builds, hash the compressed SSP asset as well as the dense SSP, store the resolved
model config, and rerun the 512-row closure gate in the target Jean-Zay environment.

![Forward drift](stored_vs_current_forward_drift.png)

## 6. Concrete implementation plan

1. Add `mc_sfh_type`/`mc_is_q` to proposal and final parquet rows. Do not infer the
   state later from floating-point equality when the generator already supplies it.
2. Keep the hybrid prior already prototyped, but reduce or regularize the quenched
   branch and benchmark calibration by state. Generate more quenched training rows
   if a flexible conditional flow is required.
3. Add an optional SED-native sidecar with `formed_mass_msun`, 107 age weights, and
   provenance. Keep the native 18D parameters beside it for audit/interpretation.
4. If storage/dimensionality matters, benchmark 12/20-knot log-time PCHIP and
   mass-conserving bins against age-weight, SED, color, and magnitude gates. Do not
   choose bin count from SFH plots alone.
5. For a future exact FENIKS-forward branch, save full 12x107 SSP weights or a
   validated nonnegative low-rank factorization that retains burstiness and
   metallicity-age coupling.
6. Resolve the stored/current forward drift, regenerate a versioned closure sample,
   then rerun hybrid-vs-SED-native prior learning on identical train/validation/test
   object IDs.

## Generated artifacts

- `state_prevalence.csv`
- `atom_support.csv`
- `state_contrasts.csv`
- `state_fraction_redshift_mass.csv`
- `sfh_representation_metrics.csv`
- `quenching_counterfactual_summary.csv`
- `quenching_counterfactual_by_band.csv`
- `stored_vs_current_forward_by_band.csv`
- `analysis_summary.json`
"""
    (out / "report.md").write_text(report, encoding="utf-8")

    html_report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FENIKS Dirac / SFH decision study</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; color: #202124; background: #fff; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 32px 28px 80px; }}
h1 {{ font-size: 30px; margin: 0 0 18px; }} h2 {{ margin-top: 34px; }}
h3 {{ margin-top: 24px; }} p, li {{ line-height: 1.55; }}
.summary {{ border-left: 5px solid #0072b2; padding: 12px 18px; background: #f4f8fb; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 14px 0 24px; }}
th, td {{ border: 1px solid #c8cdd2; padding: 7px 9px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
img {{ width: 100%; height: auto; margin: 10px 0 24px; border: 1px solid #d8dde2; }}
code {{ background: #f2f3f5; padding: 1px 4px; }} .warn {{ color: #8a2d18; font-weight: 700; }}
</style></head><body><main>
<h1>FENIKS Dirac and SFH representation decision study</h1>
<div class="summary"><strong>Recommendation.</strong> Use one explicit quenching
state plus branch-specific continuous models for the native 18D prior. Add a
separate DSPS age-weight product for exact SED closure. Reject atom clipping as a
standalone normalization.</div>
<h2>1. Shared discrete state</h2>
<p>The four peaks are one shared main-sequence/no-quenching state. Train prevalence:
<strong>{train_prev.atom_fraction:.3%}</strong>; continuous quenched rows:
<strong>{int(train_prev.continuous_count)}</strong>.</p>
{support_train.to_html(index=False, float_format=lambda value: f"{value:.6g}")}
<img src="dirac_joint_support.png" alt="Shared atom support">
<img src="state_population_contrasts.png" alt="State population contrasts">
<img src="state_fraction_redshift_mass.png" alt="State fraction by redshift and mass">
<h2>2. Decision on the options</h2>
<h3>Hierarchical mixed model</h3><p><strong>Recommended.</strong> Model one Bernoulli
state and use flows only on continuous coordinates. The quenched branch is
sample-limited, so it should be lower-dimensional, regularized, or trained on a
stratified larger sample.</p>
<h3>Move/clip the atom</h3><p class="warn">Rejected as stated.</p><p>The nearest
observed continuous proxy already gives p95 maximum photometric change
{clip.p95_max_abs_delta_mag:.3g} mag. A restoration distance is itself the missing
discrete variable.</p>
<h3>Remove the continuous branch</h3><p>Valid only for a named main-sequence-only
benchmark. It removes {test_prev.continuous_fraction:.3%} of this selected catalog
and erases a distinct high-mass/high-redshift population with different conditional
SFHs and colors.</p>
<h3>SED-native truth</h3><p><strong>Recommended as a second product.</strong> Store
formed mass plus {n_ssp_ages} DSPS age weights, redshift, metallicity, and dust.
For exact Diffsky rather than the current separable-MDF closure, retain the full
{n_ssp_metallicities}x{n_ssp_ages} SSP-weight grid or a validated factorization.</p>
<img src="quenching_state_counterfactual_photometry.png" alt="Quenching state counterfactuals">
<h2>3. SFH compression benchmark</h2>
{representation_metrics[representation_metrics['group'] == 'population_weighted'].to_html(index=False, float_format=lambda value: f"{value:.5g}")}
<img src="sfh_reconstruction_examples.png" alt="SFH reconstruction examples">
<img src="sfh_representation_photometry_accuracy.png" alt="Representation accuracy">
<h2>4. Closure drift</h2><p class="warn">The current local forward does not
reproduce the stored parquet photometry.</p><p>The maximum per-band p95 difference is
{drift_p95:.3g} mag. Resolve and freeze this runtime contract before regenerating
the production dataset.</p>
{forward_drift.to_html(index=False, float_format=lambda value: f"{value:.5g}")}
<img src="stored_vs_current_forward_drift.png" alt="Stored versus current forward drift">
<h2>5. Full written analysis</h2><p>The detailed formulas, caveats, sufficient-input
list, and six-step implementation plan are in <a href="report.md">report.md</a>.</p>
</main></body></html>"""
    (out / "report.html").write_text(html_report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    frames = {
        split: pd.read_parquet(args.dataset_dir / f"{split}.parquet")
        for split in ("train", "validation", "test")
    }
    atom_values = _atom_values(frames["train"])
    prevalence = _state_prevalence(frames, atom_values)
    support = _atom_support_table(frames, atom_values)
    contrasts = _state_contrast_table(frames["test"], atom_values)
    state_grid = _state_fraction_grid(frames["test"], atom_values)
    prevalence.to_csv(out / "state_prevalence.csv", index=False)
    support.to_csv(out / "atom_support.csv", index=False)
    contrasts.to_csv(out / "state_contrasts.csv", index=False)
    state_grid.to_csv(out / "state_fraction_redshift_mass.csv", index=False)

    _plot_dirac_support(frames["train"], atom_values, out / "dirac_joint_support.png")
    _plot_state_contrasts(
        frames["test"], atom_values, out / "state_population_contrasts.png"
    )
    _plot_state_grid(state_grid, out / "state_fraction_redshift_mass.png")

    sample, sample_atom, population_weights = _balanced_sample(
        frames["test"],
        atom_values,
        sample_per_state=args.sample_per_state,
        seed=args.seed,
    )
    theta = theta_from_truth_frame(sample).astype(np.float32)
    config = load_config(args.config)
    filters = load_filters(config["bands"])
    context = dsps_model.load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    native_sfh_fn, sfh_forward_fn, age_forward_fn = _build_forward_functions(context)
    raw_native = np.asarray(native_sfh_fn(jnp.asarray(theta)))
    native_result = sfh_forward_fn(jnp.asarray(theta), jnp.asarray(raw_native))
    t_table, scaled_sfh, age_weights, formed_mass, mags, sed = [
        np.asarray(jax.device_get(value)) for value in native_result
    ]
    native = {
        "t_table": t_table,
        "sfh": scaled_sfh,
        "age_weights": age_weights,
        "formed_mass": formed_mass,
        "mags": mags,
        "sed": sed,
    }
    age_mags, age_sed = age_forward_fn(
        jnp.asarray(theta), jnp.asarray(age_weights), jnp.asarray(formed_mass)
    )
    representations: dict[str, dict[str, np.ndarray]] = {
        "age_weights_107_exact": {
            **native,
            "mags": np.asarray(jax.device_get(age_mags)),
            "sed": np.asarray(jax.device_get(age_sed)),
        }
    }
    raw_approximations = {
        "popcosmos_7_bins": _popcosmos_approximation(t_table, raw_native),
        "lookback_16_bins": _lookback_bin_approximation(t_table, raw_native, 16),
        "logtime_pchip_12": _pchip_approximation(t_table, raw_native, 12),
        "logtime_pchip_20": _pchip_approximation(t_table, raw_native, 20),
    }
    for name, approximation in raw_approximations.items():
        result = sfh_forward_fn(jnp.asarray(theta), jnp.asarray(approximation))
        values = [np.asarray(jax.device_get(value)) for value in result]
        representations[name] = {
            "t_table": values[0],
            "sfh": values[1],
            "age_weights": values[2],
            "formed_mass": values[3],
            "mags": values[4],
            "sed": values[5],
        }

    metric_rows = []
    for name in REPRESENTATION_ORDER:
        metric_rows.extend(
            _representation_metrics(
                name,
                representations[name],
                native,
                sample_atom,
                population_weights,
            )
        )
    representation_metrics = pd.DataFrame(metric_rows)
    representation_metrics.to_csv(out / "sfh_representation_metrics.csv", index=False)

    _plot_sfh_examples(
        sample,
        sample_atom,
        native,
        representations,
        np.asarray(dsps_model._context_ssp_lg_age_gyr(context)),
        out / "sfh_reconstruction_examples.png",
    )
    _plot_representation_accuracy(
        representations,
        native,
        sample_atom,
        out / "sfh_representation_photometry_accuracy.png",
    )

    train_theta = theta_from_truth_frame(frames["train"]).astype(np.float64)
    train_atom = _shared_atom_mask(frames["train"], atom_values)
    q_indices = [DIFFSKY_BASIC_PARAMETER_NAMES.index(name) for name in ATOM_NAMES]
    atom_vector = np.asarray([atom_values[name] for name in ATOM_NAMES])
    continuous_q = train_theta[~train_atom][:, q_indices]
    scales = np.maximum(np.std(continuous_q, axis=0), 1.0e-8)
    nearest_index = int(
        np.argmin(np.sum(((continuous_q - atom_vector) / scales) ** 2, axis=1))
    )
    nearest_proxy = continuous_q[nearest_index]

    atom_to_proxy_theta = theta.copy()
    atom_to_proxy_theta[
        np.flatnonzero(sample_atom)[:, None], np.asarray(q_indices)[None, :]
    ] = nearest_proxy
    continuous_to_atom_theta = theta.copy()
    continuous_to_atom_theta[
        np.flatnonzero(~sample_atom)[:, None], np.asarray(q_indices)[None, :]
    ] = atom_vector
    counterfactual_mags = {}
    for label, changed_theta in (
        ("atom_to_nearest_continuous_proxy", atom_to_proxy_theta),
        ("continuous_to_atom", continuous_to_atom_theta),
    ):
        changed_raw = np.asarray(native_sfh_fn(jnp.asarray(changed_theta)))
        changed_result = sfh_forward_fn(
            jnp.asarray(changed_theta), jnp.asarray(changed_raw)
        )
        counterfactual_mags[label] = np.asarray(jax.device_get(changed_result[4]))
    band_names = [str(item["name"]) for item in config["bands"]]
    summaries = []
    by_bands = []
    for label, selected in (
        ("atom_to_nearest_continuous_proxy", sample_atom),
        ("continuous_to_atom", ~sample_atom),
    ):
        summary, by_band = _counterfactual_metrics(
            label,
            native["mags"],
            counterfactual_mags[label],
            selected,
            band_names,
        )
        summaries.append(summary)
        by_bands.append(by_band)
    counterfactual_summary = pd.concat(summaries, ignore_index=True)
    counterfactual_by_band = pd.concat(by_bands, ignore_index=True)
    counterfactual_summary.to_csv(
        out / "quenching_counterfactual_summary.csv", index=False
    )
    counterfactual_by_band.to_csv(
        out / "quenching_counterfactual_by_band.csv", index=False
    )
    pd.DataFrame(
        {
            "parameter": ATOM_NAMES,
            "atom_value": atom_vector,
            "nearest_joint_continuous_proxy": nearest_proxy,
            "signed_displacement": nearest_proxy - atom_vector,
        }
    ).to_csv(out / "nearest_joint_continuous_proxy.csv", index=False)
    _plot_counterfactuals(
        counterfactual_by_band,
        out / "quenching_state_counterfactual_photometry.png",
    )

    stored_mags = sample[[f"mag_true_{band}" for band in band_names]].to_numpy(float)
    forward_drift = _plot_forward_drift(
        stored_mags,
        native["mags"],
        band_names,
        out / "stored_vs_current_forward_drift.png",
    )
    forward_drift.to_csv(out / "stored_vs_current_forward_by_band.csv", index=False)

    summary_payload = {
        "dataset_dir": str(args.dataset_dir),
        "config": str(args.config),
        "sample_rows": len(sample),
        "sample_per_state_requested": args.sample_per_state,
        "atom_values": atom_values,
        "n_ssp_ages": int(len(context.ssp.ssp_lg_age_gyr)),
        "n_ssp_metallicities": int(len(context.ssp.ssp_lgmet)),
        "nearest_joint_continuous_proxy": {
            name: float(value)
            for name, value in zip(ATOM_NAMES, nearest_proxy, strict=True)
        },
        "state_prevalence": prevalence.to_dict(orient="records"),
        "counterfactuals": counterfactual_summary.to_dict(orient="records"),
        "forward_drift_max_band_p95_mag": float(
            forward_drift["p95_abs_delta_mag"].max()
        ),
        "runtime": {
            "jax_version": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
        },
    }
    (out / "analysis_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    _build_reports(
        out,
        prevalence=prevalence,
        support=support,
        contrasts=contrasts,
        representation_metrics=representation_metrics,
        counterfactual_summary=counterfactual_summary,
        forward_drift=forward_drift,
        sample_size=len(sample),
        n_ssp_ages=int(len(context.ssp.ssp_lg_age_gyr)),
        n_ssp_metallicities=int(len(context.ssp.ssp_lgmet)),
    )
    print(f"Wrote FENIKS Dirac/SFH analysis to {out}")


if __name__ == "__main__":
    main()
