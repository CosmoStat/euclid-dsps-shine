#!/usr/bin/env python3
"""Build a standalone interactive explorer of the FENIKS forward model."""

from __future__ import annotations

import argparse
import base64
import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table

from euclid_dsps import model as dsps_model
from euclid_dsps.config import load_config
from euclid_dsps.filters import load_filters
from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES
from euclid_dsps.photometric_uncertainty import (
    _gamma_for_band,
    _model_value_for_band,
    normalize_flux_error_model,
)
from euclid_dsps.photometry import abmag_to_fnu_cgs
from euclid_dsps.prior_learning.spline15d import cubic_spline_interpolate_jax
from euclid_dsps.synthetic_diffsky.photometry import (
    GROUND_TRUTH_COLUMNS,
    theta_from_truth_frame,
)

PARAMETER_META: dict[str, dict[str, str]] = {
    "z_obs": {
        "group": "Observation",
        "label": "Observed redshift",
        "unit": "dimensionless",
        "meaning": "Epoch, cosmic age, luminosity distance, and spectral redshift.",
        "code_role": "age_at_z, IGM, and calc_obs_mag",
    },
    "log10_stellar_mass": {
        "group": "Stellar population",
        "label": "Log surviving stellar mass",
        "unit": "log10(Mstar/Msun)",
        "meaning": "Mass still locked in stars at observation.",
        "code_role": "Exact target of the global SFH normalization",
    },
    "diffstar_lgmcrit": {
        "group": "Diffstar main sequence",
        "label": "Pivot halo mass",
        "unit": "log10(Mhalo/Msun)",
        "meaning": "Center of the transition between low- and high-mass slopes.",
        "code_role": "Center of beta(logMh), k fixed to 9",
    },
    "diffstar_lgy_at_mcrit": {
        "group": "Diffstar main sequence",
        "label": "Raw amplitude at Mcrit",
        "unit": "log10(1/yr)",
        "meaning": "Raw SFR per unit baryonic mass at the pivot.",
        "code_role": "Global factor later removed by Mstar normalization",
    },
    "diffstar_indx_lo": {
        "group": "Diffstar main sequence",
        "label": "Low-mass slope",
        "unit": "dimensionless",
        "meaning": "Change in log y below the pivot mass.",
        "code_role": "Low-mass asymptote of beta(logMh)",
    },
    "diffstar_indx_hi": {
        "group": "Diffstar main sequence",
        "label": "High-mass slope",
        "unit": "dimensionless",
        "meaning": "Change in log y above the pivot mass.",
        "code_role": "High-mass asymptote of beta(logMh)",
    },
    "diffstar_lg_qt": {
        "group": "Diffstar quenching",
        "label": "Time of minimum SFR",
        "unit": "log10(Gyr)",
        "meaning": "Cosmic time of maximum quenching depth.",
        "code_role": "tq = 10^lg_qt",
    },
    "diffstar_qlglgdt": {
        "group": "Diffstar quenching",
        "label": "Quenching width",
        "unit": "log10(time dex)",
        "meaning": "Controls the total transition duration in log time.",
        "code_role": "Delta log10(t) = 10^qlglgdt",
    },
    "diffstar_lg_drop": {
        "group": "Diffstar quenching",
        "label": "Quenching depth",
        "unit": "log10 facteur SFR",
        "meaning": "Minimum multiplicative factor applied to the main sequence.",
        "code_role": "Qmin = 10^lg_drop",
    },
    "diffstar_lg_rejuv": {
        "group": "Diffstar quenching",
        "label": "Post-rejuvenation level",
        "unit": "log10 facteur SFR",
        "meaning": "Asymptotic SFR level after quenching.",
        "code_role": "Qfinal = 10^lg_rejuv",
    },
    "diffmah_logm0": {
        "group": "Diffmah halo",
        "label": "Halo-mass normalization",
        "unit": "log10(Mhalo/Msun)",
        "meaning": "Anchor parameter of the Diffmah mass history.",
        "code_role": "In this wrapper, lgt0=log10(t_obs), not the age at z=0",
    },
    "diffmah_logtc": {
        "group": "Diffmah halo",
        "label": "Halo transition time",
        "unit": "log10(Gyr)",
        "meaning": "Transition from rapid to slow growth.",
        "code_role": "Center of alpha(log t), k fixed to 3.5",
    },
    "diffmah_early_index": {
        "group": "Diffmah halo",
        "label": "Early growth index",
        "unit": "dimensionless",
        "meaning": "Asymptotic early-time growth exponent.",
        "code_role": "Early-time asymptote of alpha(log t)",
    },
    "diffmah_late_index": {
        "group": "Diffmah halo",
        "label": "Late growth index",
        "unit": "dimensionless",
        "meaning": "Asymptotic late-time growth exponent.",
        "code_role": "Late-time asymptote of alpha(log t)",
    },
    "diffmah_t_peak": {
        "group": "Diffmah halo",
        "label": "Time of maximum mass",
        "unit": "Gyr",
        "meaning": "Cosmic time after which halo mass is frozen.",
        "code_role": "dMh/dt=0 and Mh=Mh(t_peak) after t_peak",
    },
    "log10_stellar_metallicity": {
        "group": "Stellar population",
        "label": "Median stellar metallicity",
        "unit": "log10(Zstar/Zsun)",
        "meaning": "Median of the population metallicity distribution.",
        "code_role": "MDF lognormal, scatter fixed to 0.2 dex, Zsun=0.0142",
    },
    "dust_av": {
        "group": "Dust",
        "label": "V-band attenuation",
        "unit": "mag",
        "meaning": "Amplitude of diffuse stellar-light attenuation.",
        "code_role": "tau2 = A_V / 1.086",
    },
    "dust_delta": {
        "group": "Dust",
        "label": "Dust-law slope",
        "unit": "dimensionless",
        "meaning": "Tilts the UV-optical curve and changes the 2175 A bump.",
        "code_role": "dust_index_n in the Prospector/FSPS law",
    },
}

RESULT_RUNS = {
    "failed_flow": Path("outputs/runs/feniks_spline15d_realnvp_v2_a_control"),
    "prior_initial": Path("outputs/runs/feniks_spline15d_v6_positive_support"),
    "prior_resume_155": Path(
        "outputs/runs/feniks_spline15d_v6_positive_support_resume155"
    ),
    "prior_resume_400": Path(
        "outputs/runs/feniks_spline15d_v6_positive_support_resume400_to800"
    ),
    "encoder": Path(
        "outputs/runs/feniks_spline15d_amortized_epoch645_4xh100_b2048_v4"
    ),
}
JAX_COSMO_SPLINE15D_ANALYSIS = Path(
    "outputs/analysis/feniks_jax_cosmo_spline_15d_prior_20260716"
)


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
        "--dataset",
        type=Path,
        default=Path(
            "Data/diffsky/synthetic/feniks_260617_dsps_closure_18band/train.parquet"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "Data/diffsky/synthetic/feniks_260617_dsps_closure_18band/manifest.yaml"
        ),
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).with_name("templates") / "feniks_forward_explorer.html",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/reports/feniks_forward_model_explorer.html"),
    )
    parser.add_argument(
        "--payload-out",
        type=Path,
        default=Path("outputs/reports/feniks_forward_model_explorer_payload.json"),
    )
    parser.add_argument(
        "--spline-nodes",
        type=int,
        default=20,
        help="Number of log-time/log-SFR JAX-COSMO cubic nodes used by the study.",
    )
    parser.add_argument(
        "--spline-grid",
        choices=("uniform_log_time", "recent_lookback", "hybrid"),
        default="hybrid",
        help="Deterministic spline-knot placement used for the per-galaxy study.",
    )
    parser.add_argument(
        "--spline-scan",
        type=Path,
        default=Path(
            "outputs/analysis/feniks_jax_cosmo_spline_node_scan_20260716/"
            "spline_k_scan_payload.json"
        ),
        help="Optional node-count scan payload embedded in the standalone report.",
    )
    parser.add_argument(
        "--spline-prior",
        type=Path,
        default=Path(
            "outputs/analysis/feniks_jax_cosmo_spline_15d_prior_20260716/"
            "spline_15d_scan_payload.json"
        ),
        help="Optional 15D spline-prior payload embedded in the standalone report.",
    )
    return parser.parse_args()


SPEED_OF_LIGHT_ANGSTROM_PER_S = 2.99792458e18


def _histogram(
    values: np.ndarray,
    bins: int = 56,
    value_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    counts, edges = np.histogram(finite, bins=bins, range=value_range)
    fractions = counts.astype(float) / max(len(finite), 1)
    quantiles = np.quantile(finite, [0.01, 0.16, 0.5, 0.84, 0.99])
    return {
        "edges": edges.tolist(),
        "fractions": fractions.tolist(),
        "q01": float(quantiles[0]),
        "q16": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q84": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "n": int(len(finite)),
    }


def _tw_cdf(y: np.ndarray) -> np.ndarray:
    value = -5 * y**7 / 69984 + 7 * y**5 / 2592 - 35 * y**3 / 864 + 35 * y / 96 + 0.5
    return np.where(y < -3, 0.0, np.where(y > 3, 1.0, value))


def _q_at_observation(frame: pd.DataFrame) -> np.ndarray:
    z = frame[GROUND_TRUTH_COLUMNS["z_obs"]].to_numpy(float)
    t_obs = np.asarray(age_at_z(z, *DEFAULT_COSMOLOGY), dtype=float).reshape(-1)
    lgt = np.log10(np.maximum(t_obs, 1.0e-8))
    lg_qt = frame[GROUND_TRUTH_COLUMNS["diffstar_lg_qt"]].to_numpy(float)
    q_width = 10 ** frame[GROUND_TRUTH_COLUMNS["diffstar_qlglgdt"]].to_numpy(float)
    q_drop = frame[GROUND_TRUTH_COLUMNS["diffstar_lg_drop"]].to_numpy(float)
    q_rejuv = frame[GROUND_TRUTH_COLUMNS["diffstar_lg_rejuv"]].to_numpy(float)
    y = (lgt - lg_qt) / np.maximum(q_width / 12.0, 1.0e-12)
    falling = q_drop * _tw_cdf(y + 3)
    rising = q_drop - (q_drop - q_rejuv) * _tw_cdf(y - 3)
    return 10 ** np.where(y < 0, falling, rising)


def _mode(values: pd.Series) -> float:
    counts = values.value_counts(dropna=False)
    return float(counts.index[0])


def _select_nearest(
    values: np.ndarray,
    target: float,
    *,
    allowed: np.ndarray,
    used: set[int],
) -> int:
    order = np.argsort(np.abs(np.asarray(values, dtype=float) - float(target)))
    for index in order:
        integer = int(index)
        if allowed[integer] and integer not in used:
            used.add(integer)
            return integer
    raise ValueError("No unused representative row satisfies the selection")


def _representative_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    q_obs = _q_at_observation(frame)
    atom_names = (
        "diffstar_lg_qt",
        "diffstar_qlglgdt",
        "diffstar_lg_drop",
        "diffstar_lg_rejuv",
    )
    atom = np.ones(len(frame), dtype=bool)
    for name in atom_names:
        column = GROUND_TRUTH_COLUMNS[name]
        atom &= frame[column].to_numpy(float) == _mode(frame[column])
    continuous = ~atom
    finite = np.ones(len(frame), dtype=bool)
    used: set[int] = set()
    definitions: list[tuple[str, str, int]] = []

    truth_matrix = theta_from_truth_frame(frame).astype(float)
    median = np.nanmedian(truth_matrix, axis=0)
    q16, q84 = np.nanquantile(truth_matrix, [0.16, 0.84], axis=0)
    scale = np.maximum(q84 - q16, 1.0e-6)
    distance = np.sqrt(np.mean(((truth_matrix - median) / scale) ** 2, axis=1))
    typical = _select_nearest(distance, 0.0, allowed=atom, used=used)
    definitions.append(
        ("typical", "Galaxy near the 18D medians, unquenched branch", typical)
    )

    z = frame[GROUND_TRUTH_COLUMNS["z_obs"]].to_numpy(float)
    nearby = _select_nearest(z, np.quantile(z, 0.03), allowed=finite, used=used)
    definitions.append(("nearby", "Low-redshift example", nearby))
    high_z = _select_nearest(z, np.quantile(z, 0.985), allowed=finite, used=used)
    definitions.append(("high_z", "Example in the high-redshift tail", high_z))

    logm = frame[GROUND_TRUTH_COLUMNS["log10_stellar_mass"]].to_numpy(float)
    massive = _select_nearest(logm, np.quantile(logm, 0.995), allowed=finite, used=used)
    definitions.append(("massive", "Massive stellar galaxy", massive))

    dust = frame[GROUND_TRUTH_COLUMNS["dust_av"]].to_numpy(float)
    dusty = _select_nearest(dust, np.quantile(dust, 0.997), allowed=finite, used=used)
    definitions.append(("dusty", "Strongly attenuated galaxy", dusty))

    q_rank = np.argsort(q_obs)
    quenched = next(int(index) for index in q_rank if continuous[int(index)])
    used.add(quenched)
    definitions.append(
        ("quenched", "Quenching le plus profond a l'epoque observee", quenched)
    )

    logssfr = pd.to_numeric(frame.get("logssfr_true"), errors="coerce").to_numpy()
    star_forming = _select_nearest(
        logssfr,
        np.nanquantile(logssfr, 0.995),
        allowed=np.isfinite(logssfr),
        used=used,
    )
    definitions.append(
        ("star_forming", "High-sSFR galaxy in the catalog", star_forming)
    )

    rows = frame.iloc[[item[2] for item in definitions]].copy()
    rows["explorer_key"] = [item[0] for item in definitions]
    rows["explorer_description"] = [item[1] for item in definitions]
    rows["source_row"] = [item[2] for item in definitions]
    return rows.reset_index(drop=True), q_obs


def _validate_jax_cosmo_cubic() -> float:
    """Return eager/JIT agreement for the interpolation kernel."""
    x = jnp.asarray([0.0, 0.2, 0.55, 0.8, 1.0], dtype=jnp.float32)
    y = jnp.asarray([-2.0, -0.7, 0.2, -0.1, 0.5], dtype=jnp.float32)
    x_new = jnp.linspace(0.0, 1.0, 97)
    eager = cubic_spline_interpolate_jax(x, y, x_new)
    compiled = jax.jit(cubic_spline_interpolate_jax)(x, y, x_new)
    return float(jnp.max(jnp.abs(eager - compiled)))


def _spline_knot_times_jax(
    time: jnp.ndarray, n_nodes: int, grid_strategy: str
) -> jnp.ndarray:
    t_obs = time[-1]
    span = jnp.maximum(t_obs - time[0], 1.0e-6)
    if grid_strategy == "uniform_log_time":
        return jnp.geomspace(time[0], t_obs, n_nodes)
    if grid_strategy == "recent_lookback":
        fractions = jnp.concatenate(
            (jnp.zeros(1), jnp.geomspace(1.0e-3, 1.0, n_nodes - 1))
        )
        return jnp.flip(t_obs - fractions * span)
    if grid_strategy == "hybrid":
        n_log_nodes = max(3, int(np.ceil(0.65 * n_nodes)))
        n_recent_nodes = n_nodes - n_log_nodes
        log_grid = jnp.geomspace(time[0], t_obs, n_log_nodes)
        if not n_recent_nodes:
            return log_grid
        recent_fractions = jnp.geomspace(1.0e-3, 0.5, n_recent_nodes + 2)[1:-1]
        recent_grid = t_obs - recent_fractions * span
        return jnp.sort(jnp.concatenate((log_grid, recent_grid)))
    raise ValueError(f"Unsupported spline grid strategy: {grid_strategy}")


def _forward_batch(
    context: dsps_model.DspsContext,
    theta: np.ndarray,
    *,
    spline_nodes: int,
    spline_grid: str,
) -> dict[str, np.ndarray]:
    try:
        from diffmah.diffmah_kernels import (
            DiffmahParams,
            _diffmah_kern,
            _power_law_index_vs_logt,
        )
        from diffstar.defaults import FB
        from diffstar.kernels.main_sequence_kernels import MSParams, _sfr_eff_plaw
        from diffstar.kernels.quenching_kernels import _quenching_kern
    except ImportError as exc:
        raise RuntimeError(
            "Run this script with `uv run --extra diffstar python "
            "scripts/build_feniks_forward_explorer.py`."
        ) from exc

    names = tuple(DIFFSKY_BASIC_PARAMETER_NAMES)
    ssp_lg_age_gyr = dsps_model._context_ssp_lg_age_gyr(context)

    def single(theta_row: jnp.ndarray):
        params = {name: theta_row[index] for index, name in enumerate(names)}
        z_obs = params["z_obs"]
        t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]
        t_table = jnp.linspace(0.05, jnp.maximum(t_obs, 0.06), context.n_sfh_bins)
        lgt0 = jnp.log10(jnp.maximum(t_obs, 1.0e-6))

        mah_params = DiffmahParams(
            params["diffmah_logm0"],
            params["diffmah_logtc"],
            params["diffmah_early_index"],
            params["diffmah_late_index"],
            params["diffmah_t_peak"],
        )
        dmhdt, logmh = _diffmah_kern(mah_params, t_table, lgt0)
        alpha = _power_law_index_vs_logt(
            jnp.log10(t_table),
            params["diffmah_logtc"],
            params["diffmah_early_index"],
            params["diffmah_late_index"],
        )

        ms_params = MSParams(
            params["diffstar_lgmcrit"],
            params["diffstar_lgy_at_mcrit"],
            params["diffstar_indx_lo"],
            params["diffstar_indx_hi"],
        )
        logy = _sfr_eff_plaw(logmh, *ms_params)
        sfr_ms = FB * 10**logmh * 10**logy
        q_width = 10 ** params["diffstar_qlglgdt"]
        q_function = _quenching_kern(
            jnp.log10(t_table),
            params["diffstar_lg_qt"],
            q_width,
            params["diffstar_lg_drop"],
            params["diffstar_lg_rejuv"],
        )
        raw_components = jnp.maximum(sfr_ms * q_function, 1.0e-14)
        raw_sfr = dsps_model.build_diffsky_basic_sfh_table_jax(t_table, t_obs, params)

        model_config = dsps_model._normalized_model_config(context.model_config)
        lgmet_abs = dsps_model.log10_stellar_metallicity_to_absolute_jax(
            params["log10_stellar_metallicity"], context.z_sun
        )
        surviving_by_age = dsps_model._diffsky_basic_surviving_mstar_by_age_jax(
            context, model_config, lgmet_abs
        )
        normalized_sfr, formed_mass, surviving_mass = (
            dsps_model.normalize_sfh_to_stellar_mass_jax(
                t_table,
                raw_sfr,
                ssp_lg_age_gyr,
                t_obs,
                params["log10_stellar_mass"],
                surviving_by_age,
            )
        )
        age_weights = calc_age_weights_from_sfh_table(
            t_table, normalized_sfr, ssp_lg_age_gyr, t_obs
        )

        log_time = jnp.log10(jnp.maximum(t_table, 1.0e-6))
        spline_knot_time = _spline_knot_times_jax(t_table, spline_nodes, spline_grid)
        spline_knot_log_time = jnp.log10(jnp.maximum(spline_knot_time, 1.0e-6))
        raw_log_sfr = jnp.log10(jnp.maximum(raw_sfr, 1.0e-30))
        spline_knot_log_sfr_raw = jnp.interp(
            spline_knot_log_time, log_time, raw_log_sfr
        )
        spline_raw_sfr = 10 ** cubic_spline_interpolate_jax(
            spline_knot_log_time, spline_knot_log_sfr_raw, log_time
        )
        spline_sfr, spline_formed_mass, spline_surviving_mass = (
            dsps_model.normalize_sfh_to_stellar_mass_jax(
                t_table,
                spline_raw_sfr,
                ssp_lg_age_gyr,
                t_obs,
                params["log10_stellar_mass"],
                surviving_by_age,
            )
        )
        spline_age_weights = calc_age_weights_from_sfh_table(
            t_table, spline_sfr, ssp_lg_age_gyr, t_obs
        )
        spline_scale = jnp.median(spline_sfr / jnp.maximum(spline_raw_sfr, 1.0e-30))
        spline_knot_log_sfr = spline_knot_log_sfr_raw + jnp.log10(spline_scale)

        ssp_flux_z = dsps_model.diffsky_basic_ssp_flux_by_age_jax(
            context, model_config, lgmet_abs
        )
        spline_sed_by_age = (
            jnp.clip(ssp_flux_z, 0.0, jnp.inf)
            * spline_age_weights[:, None]
            * spline_formed_mass
        )
        tau2, dust_index_n, tau1_over_tau2 = dsps_model.diffsky_basic_dust_params_jax(
            params
        )
        wave = dsps_model._context_ssp_wave(context)
        spline_dusted_by_age = dsps_model.apply_popcosmos_dust_by_age_jax(
            wave,
            ssp_lg_age_gyr,
            spline_sed_by_age,
            tau2,
            dust_index_n,
            tau1_over_tau2,
            model_config,
        )
        spline_dusted_sed = jnp.nan_to_num(
            spline_dusted_by_age.sum(axis=0),
            nan=0.0,
            posinf=1.0e30,
            neginf=0.0,
        )
        _, spline_post_igm_sed = dsps_model.combine_agn_and_igm_jax(
            wave,
            spline_dusted_sed,
            jnp.zeros_like(spline_dusted_sed),
            z_obs,
            model_config,
        )
        spline_model_mags = dsps_model.predict_mags_jax(
            context, wave, spline_post_igm_sed, z_obs
        )
        cumulative_mass = jnp.concatenate(
            [
                jnp.zeros(1, dtype=normalized_sfr.dtype),
                jnp.cumsum(
                    0.5
                    * (normalized_sfr[1:] + normalized_sfr[:-1])
                    * jnp.diff(t_table)
                    * 1.0e9
                ),
            ]
        )
        result = dsps_model.run_diffsky_basic_model_jax(context, params)
        relative_raw_error = jnp.max(
            jnp.abs(raw_components - raw_sfr) / jnp.maximum(raw_sfr, 1.0e-14)
        )
        return (
            t_table,
            logmh,
            dmhdt,
            alpha,
            logy,
            q_function,
            sfr_ms,
            raw_sfr,
            normalized_sfr,
            cumulative_mass,
            age_weights,
            formed_mass,
            surviving_mass,
            spline_knot_time,
            spline_knot_log_sfr,
            spline_sfr,
            spline_age_weights,
            spline_formed_mass,
            spline_surviving_mass,
            spline_model_mags,
            result.stellar_intrinsic_sed,
            result.stellar_dusted_sed,
            result.post_igm_sed,
            result.model_mags,
            relative_raw_error,
        )

    values = jax.jit(jax.vmap(single))(jnp.asarray(theta, dtype=jnp.float32))
    arrays = [np.asarray(jax.device_get(value)) for value in values]
    keys = (
        "time_gyr",
        "logmh",
        "dmhdt",
        "alpha",
        "logy",
        "q_function",
        "sfr_ms",
        "sfr_raw",
        "sfr_normalized",
        "cumulative_formed_mass",
        "age_weights",
        "formed_mass",
        "surviving_mass",
        "spline_knot_time_gyr",
        "spline_knot_log_sfr",
        "spline_sfr_normalized",
        "spline_age_weights",
        "spline_formed_mass",
        "spline_surviving_mass",
        "spline_model_mags",
        "sed_intrinsic",
        "sed_dusted",
        "sed_post_igm",
        "model_mags",
        "raw_component_relative_error",
    )
    return dict(zip(keys, arrays, strict=True))


def _finite_log10(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    positive = array[np.isfinite(array) & (array > 0.0)]
    floor = max(float(np.max(positive)) * 1.0e-12, 1.0e-40)
    return np.log10(np.maximum(array, floor))


def _downsample_indices(wave: np.ndarray, size: int = 720) -> np.ndarray:
    wave = np.asarray(wave, dtype=float)
    useful = np.flatnonzero((wave >= 300.0) & (wave <= 60_000.0))
    targets = np.geomspace(wave[useful[0]], wave[useful[-1]], size)
    indices = np.searchsorted(wave, targets)
    return np.unique(np.clip(indices, useful[0], useful[-1]))


def _downsample_filter_curve(
    curve: Any, size: int = 180
) -> tuple[np.ndarray, np.ndarray]:
    wave = np.asarray(curve.wave, dtype=float)
    transmission = np.asarray(curve.transmission, dtype=float)
    positive = np.flatnonzero(transmission > np.nanmax(transmission) * 1.0e-5)
    if not len(positive):
        return wave[:0], transmission[:0]
    low = max(int(positive[0]) - 1, 0)
    high = min(int(positive[-1]) + 1, len(wave) - 1)
    indices = np.unique(np.linspace(low, high, size).round().astype(int))
    scale = max(float(np.nanmax(transmission[indices])), 1.0e-30)
    return wave[indices], transmission[indices] / scale


def _lnu_to_llambda_per_angstrom(
    wave_angstrom: np.ndarray, lnu: np.ndarray
) -> np.ndarray:
    wave = np.asarray(wave_angstrom, dtype=float)
    return (
        np.asarray(lnu, dtype=float)
        * SPEED_OF_LIGHT_ANGSTROM_PER_S
        / np.maximum(wave, 1.0) ** 2
    )


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def _build_payload(
    frame: pd.DataFrame,
    examples: pd.DataFrame,
    q_obs: np.ndarray,
    forward: dict[str, np.ndarray],
    context: dsps_model.DspsContext,
    filters: dict[str, Any],
    config: dict[str, Any],
    manifest: dict[str, Any],
    args: argparse.Namespace,
    spline_jit_max_abs_error: float,
    spline_scan: dict[str, Any] | None,
    spline_prior: dict[str, Any] | None,
) -> dict[str, Any]:
    distributions: dict[str, Any] = {}
    for name in DIFFSKY_BASIC_PARAMETER_NAMES:
        column = GROUND_TRUTH_COLUMNS[name]
        distributions[name] = {
            **PARAMETER_META[name],
            "raw_column": column,
            "histogram": _histogram(frame[column].to_numpy(float)),
        }

    z_all = frame[GROUND_TRUTH_COLUMNS["z_obs"]].to_numpy(float)
    t_obs_all = np.asarray(age_at_z(z_all, *DEFAULT_COSMOLOGY)).reshape(-1)
    derived_distributions = {
        "t_obs_gyr": _histogram(t_obs_all),
        "q_at_obs": _histogram(q_obs),
        "logssfr_true": _histogram(
            pd.to_numeric(frame["logssfr_true"], errors="coerce").to_numpy()
        ),
        "logsfr_true": _histogram(
            pd.to_numeric(frame["logsfr_true"], errors="coerce").to_numpy()
        ),
        "tau_v": _histogram(
            frame[GROUND_TRUTH_COLUMNS["dust_av"]].to_numpy(float) / 1.086
        ),
    }

    bands = [str(item["name"]) for item in config["bands"]]
    error_cfg = normalize_flux_error_model(
        (config.get("synthetic_diffsky", {}) or {}).get("flux_error_model")
    )
    all_noise_pulls = []
    band_payload = []
    for band in bands:
        curve = filters[band]
        filter_wave, filter_transmission = _downsample_filter_curve(curve)
        flux_true = frame[f"flux_true_{band}"].to_numpy(float)
        flux = frame[f"flux_{band}"].to_numpy(float)
        fluxerr = frame[f"fluxerr_{band}"].to_numpy(float)
        noise_pull = (flux - flux_true) / np.maximum(fluxerr, 1.0e-40)
        all_noise_pulls.append(noise_pull)
        m5 = _model_value_for_band(error_cfg, ("m5", "m5_by_band", "depth_m5"), band)
        gamma = _gamma_for_band(error_cfg, band)
        band_payload.append(
            {
                "name": band,
                "effective_wavelength_angstrom": float(curve.effective_wavelength),
                "filter_wave_observed_angstrom": filter_wave.tolist(),
                "filter_transmission": filter_transmission.tolist(),
                "m5": float(m5),
                "gamma": float(gamma),
                "magnitude_distribution": _histogram(
                    frame[f"mag_true_{band}"].to_numpy(float)
                ),
                "fluxerr_njy_distribution": _histogram(fluxerr / 1.0e-32),
                "true_snr_distribution": _histogram(
                    flux_true / np.maximum(fluxerr, 1.0e-40)
                ),
                "noise_pull_distribution": _histogram(
                    noise_pull, value_range=(-5.0, 5.0)
                ),
            }
        )

    sed_indices = _downsample_indices(dsps_model._context_ssp_wave(context))
    wave = np.asarray(dsps_model._context_ssp_wave(context), dtype=float)
    age_gyr = 10 ** np.asarray(dsps_model._context_ssp_lg_age_gyr(context), dtype=float)
    example_payload = []
    theta = theta_from_truth_frame(examples).astype(float)
    for index, row in examples.iterrows():
        parameter_values = {
            name: float(theta[index, pindex])
            for pindex, name in enumerate(DIFFSKY_BASIC_PARAMETER_NAMES)
        }
        stored_mags = np.asarray(
            [float(row[f"mag_true_{band}"]) for band in bands], dtype=float
        )
        stored_flux_true = np.asarray(
            [float(row[f"flux_true_{band}"]) for band in bands], dtype=float
        )
        stored_flux = np.asarray(
            [float(row[f"flux_{band}"]) for band in bands], dtype=float
        )
        stored_fluxerr = np.asarray(
            [float(row[f"fluxerr_{band}"]) for band in bands], dtype=float
        )
        current_mags = forward["model_mags"][index]
        current_flux = np.asarray(abmag_to_fnu_cgs(current_mags), dtype=float)
        noise = stored_flux - stored_flux_true
        noise_pull = noise / np.maximum(stored_fluxerr, 1.0e-40)
        sigma_source = []
        sigma_background = []
        sigma_systematic = []
        for band_index, _band in enumerate(bands):
            band_info = band_payload[band_index]
            f5 = float(abmag_to_fnu_cgs(band_info["m5"]))
            flux_abs = abs(float(stored_flux_true[band_index]))
            gamma = float(band_info["gamma"])
            source_variance = max((0.04 - gamma) * flux_abs * f5, 0.0)
            background_variance = max(gamma * f5**2, 0.0)
            sys_frac = np.expm1(
                np.log(10.0) * float(error_cfg.get("sigma_sys_mag", 0.005)) / 2.5
            )
            systematic_variance = (sys_frac * flux_abs) ** 2
            sigma_source.append(np.sqrt(source_variance) / 1.0e-32)
            sigma_background.append(np.sqrt(background_variance) / 1.0e-32)
            sigma_systematic.append(np.sqrt(systematic_variance) / 1.0e-32)
        raw = forward["sfr_raw"][index]
        normalized = forward["sfr_normalized"][index]
        spline_sfr = forward["spline_sfr_normalized"][index]
        spline_age_weights = forward["spline_age_weights"][index]
        spline_mags = forward["spline_model_mags"][index]
        delta_spline_mag = spline_mags - current_mags
        time_gyr = forward["time_gyr"][index]
        sfh_l1 = float(
            np.trapezoid(np.abs(spline_sfr - normalized), time_gyr)
            / max(float(np.trapezoid(normalized, time_gyr)), 1.0e-30)
        )
        log_sfh_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        np.log10(np.maximum(spline_sfr, 1.0e-30))
                        - np.log10(np.maximum(normalized, 1.0e-30))
                    )
                    ** 2
                )
            )
        )
        age_weight_l1 = float(
            np.sum(np.abs(spline_age_weights - forward["age_weights"][index]))
        )
        spline_mass_log_error = float(
            np.log10(max(float(forward["spline_surviving_mass"][index]), 1.0e-30))
            - parameter_values["log10_stellar_mass"]
        )
        scale = float(np.median(normalized / np.maximum(raw, 1.0e-30)))
        q_values = forward["q_function"][index]
        example_payload.append(
            {
                "key": str(row["explorer_key"]),
                "description": str(row["explorer_description"]),
                "source_row": int(row["source_row"]),
                "object_id": str(row.get("object_id", row["source_row"])),
                "parameters": parameter_values,
                "derived": {
                    "t_obs_gyr": float(forward["time_gyr"][index, -1]),
                    "formed_mass_msun": float(forward["formed_mass"][index]),
                    "surviving_mass_msun": float(forward["surviving_mass"][index]),
                    "sfr_at_obs": float(normalized[-1]),
                    "q_at_obs": float(q_values[-1]),
                    "tau_v": float(parameter_values["dust_av"] / 1.086),
                    "normalization_scale": scale,
                    "max_abs_stored_current_mag": float(
                        np.max(np.abs(stored_mags - current_mags))
                    ),
                    "raw_component_relative_error": float(
                        forward["raw_component_relative_error"][index]
                    ),
                    "stored_logssfr": float(row["logssfr_true"]),
                    "max_abs_native_spline_mag": float(
                        np.max(np.abs(delta_spline_mag))
                    ),
                },
                "halo": {
                    "time_gyr": forward["time_gyr"][index].tolist(),
                    "logmh": forward["logmh"][index].tolist(),
                    "dmhdt": forward["dmhdt"][index].tolist(),
                    "alpha": forward["alpha"][index].tolist(),
                    "logy": forward["logy"][index].tolist(),
                },
                "sfh": {
                    "time_gyr": forward["time_gyr"][index].tolist(),
                    "sfr_ms": forward["sfr_ms"][index].tolist(),
                    "q_function": q_values.tolist(),
                    "sfr_raw": raw.tolist(),
                    "sfr_normalized": normalized.tolist(),
                    "cumulative_formed_mass": forward["cumulative_formed_mass"][
                        index
                    ].tolist(),
                },
                "stellar": {
                    "ssp_age_gyr": age_gyr.tolist(),
                    "age_weights": forward["age_weights"][index].tolist(),
                    "wave_angstrom": wave[sed_indices].tolist(),
                    "log_llambda_intrinsic": _finite_log10(
                        _lnu_to_llambda_per_angstrom(
                            wave[sed_indices],
                            forward["sed_intrinsic"][index, sed_indices],
                        )
                    ).tolist(),
                    "log_llambda_dusted": _finite_log10(
                        _lnu_to_llambda_per_angstrom(
                            wave[sed_indices],
                            forward["sed_dusted"][index, sed_indices],
                        )
                    ).tolist(),
                    "log_llambda_post_igm": _finite_log10(
                        _lnu_to_llambda_per_angstrom(
                            wave[sed_indices],
                            forward["sed_post_igm"][index, sed_indices],
                        )
                    ).tolist(),
                },
                "spline": {
                    "n_nodes": int(args.spline_nodes),
                    "coordinate": str(args.spline_grid),
                    "interpolator": "JAX-COSMO cubic not-a-knot in log10(SFR)",
                    "knot_time_gyr": forward["spline_knot_time_gyr"][index].tolist(),
                    "knot_log_sfr": forward["spline_knot_log_sfr"][index].tolist(),
                    "sfr_normalized": spline_sfr.tolist(),
                    "age_weights": spline_age_weights.tolist(),
                    "native_model_mag": current_mags.tolist(),
                    "model_mag": spline_mags.tolist(),
                    "delta_mag_spline_minus_native": delta_spline_mag.tolist(),
                    "metrics": {
                        "sfh_mass_normalized_l1": sfh_l1,
                        "log_sfh_rmse_dex": log_sfh_rmse,
                        "age_weight_l1": age_weight_l1,
                        "age_weight_sum_native": float(
                            np.sum(forward["age_weights"][index])
                        ),
                        "age_weight_sum_spline": float(np.sum(spline_age_weights)),
                        "formed_mass_relative_error": float(
                            forward["spline_formed_mass"][index]
                            / max(float(forward["formed_mass"][index]), 1.0e-30)
                            - 1.0
                        ),
                        "surviving_mass_log10_error": spline_mass_log_error,
                        "magnitude_rmse": float(np.sqrt(np.mean(delta_spline_mag**2))),
                        "magnitude_max_abs": float(np.max(np.abs(delta_spline_mag))),
                    },
                },
                "photometry": {
                    "stored_mag_true": stored_mags.tolist(),
                    "current_model_mag": current_mags.tolist(),
                    "delta_mag_current_minus_stored": (
                        current_mags - stored_mags
                    ).tolist(),
                    "stored_flux_true_njy": (stored_flux_true / 1.0e-32).tolist(),
                    "stored_flux_njy": (stored_flux / 1.0e-32).tolist(),
                    "stored_fluxerr_njy": (stored_fluxerr / 1.0e-32).tolist(),
                    "current_model_flux_njy": (current_flux / 1.0e-32).tolist(),
                    "noise_draw_njy": (noise / 1.0e-32).tolist(),
                    "noise_pull": noise_pull.tolist(),
                    "sigma_source_njy": sigma_source,
                    "sigma_background_njy": sigma_background,
                    "sigma_systematic_njy": sigma_systematic,
                },
            }
        )

    generation_versions = {
        "repo_git_sha": str(manifest.get("repo_git_sha", "unknown")),
        "diffsky": str(manifest.get("diffsky_version", "unknown")),
        "diffstar": str(manifest.get("diffstar_version", "unknown")),
        "diffmah": str(manifest.get("diffmah_version", "unknown")),
        "dsps": str(manifest.get("dsps_version", "unknown")),
        "jax": str(manifest.get("jax_version", "unknown")),
    }
    current_versions = {
        "repo_git_sha": _git_sha(),
        "diffsky": _version("diffsky"),
        "diffstar": _version("diffstar"),
        "diffmah": _version("diffmah"),
        "dsps": _version("dsps"),
        "jax": _version("jax"),
    }
    return {
        "metadata": {
            "built_at": datetime.now(UTC).isoformat(),
            "dataset": str(args.dataset),
            "config": str(args.config),
            "manifest": str(args.manifest),
            "rows": int(len(frame)),
            "n_parameters": len(DIFFSKY_BASIC_PARAMETER_NAMES),
            "n_bands": len(bands),
            "generation_versions": generation_versions,
            "current_versions": current_versions,
            "model": {
                "n_sfh_bins": int(config["model"]["n_sfh_bins"]),
                "ssp_model": str(config["model"]["ssp_model"]),
                "stellar_metallicity_model": str(
                    config["model"]["stellar_metallicity_model"]
                ),
                "stellar_metallicity_scatter_dex": float(
                    config["model"]["stellar_metallicity_scatter_dex"]
                ),
                "dust_model": str(config["model"]["dust_model"]),
                "igm_model": str(config["model"]["igm_model"]),
                "nebular_model": str(config["model"]["nebular_model"]),
                "agn_model": str(config["model"]["agn_model"]),
                "z_sun": float(config["model"]["z_sun"]),
                "sed_display_unit": "Lsun/Angstrom",
                "sed_wavelength_unit": "Angstrom rest-frame",
                "spline": {
                    "n_nodes": int(args.spline_nodes),
                    "node_coordinate": str(args.spline_grid),
                    "value_coordinate": "log10 SFR",
                    "interpolator": "JAX-COSMO InterpolatedUnivariateSpline k=3 not-a-knot",
                    "mass_normalization": "same surviving-stellar-mass constraint as native SFH",
                    "eager_jit_max_abs_error": float(spline_jit_max_abs_error),
                },
            },
        },
        "parameter_order": list(DIFFSKY_BASIC_PARAMETER_NAMES),
        "distributions": distributions,
        "derived_distributions": derived_distributions,
        "error_model": {
            "type": str(error_cfg["type"]),
            "sigma_sys_mag": float(error_cfg.get("sigma_sys_mag", 0.005)),
            "default_eta": float(error_cfg.get("default_eta", 1.0)),
            "all_band_noise_pull_distribution": _histogram(
                np.concatenate(all_noise_pulls), value_range=(-5.0, 5.0)
            ),
            "formula": (
                "sigma_f^2=(0.04-gamma)|f_true|f5+gamma f5^2+(sys_frac |f_true|)^2"
            ),
        },
        "bands": band_payload,
        "spline_scan": spline_scan,
        "spline_prior": spline_prior,
        "spline_summary": {
            "example_count": len(example_payload),
            "max_example_magnitude_error": float(
                max(
                    item["spline"]["metrics"]["magnitude_max_abs"]
                    for item in example_payload
                )
            ),
            "max_example_age_weight_l1": float(
                max(
                    item["spline"]["metrics"]["age_weight_l1"]
                    for item in example_payload
                )
            ),
            "max_example_log_sfh_rmse_dex": float(
                max(
                    item["spline"]["metrics"]["log_sfh_rmse_dex"]
                    for item in example_payload
                )
            ),
        },
        "examples": example_payload,
    }


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


def _load_spline_scan(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    figures = {}
    for filename in (
        "spline_k_selection_gates.png",
        "spline_k_state_tail_curves.png",
        "spline_k_redshift_mass_heatmaps.png",
        "spline_k_quenched_band_heatmap.png",
    ):
        figure_path = path.parent / filename
        if not figure_path.exists():
            raise FileNotFoundError(f"Missing spline scan figure: {figure_path}")
        encoded = base64.b64encode(figure_path.read_bytes()).decode("ascii")
        figures[filename] = f"data:image/png;base64,{encoded}"
    payload["figures"] = figures
    return payload


def _load_spline_prior(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    figures = {}
    for filename in (
        "spline_15d_node_placements.png",
        "spline_15d_closure_comparison.png",
        "spline_15d_contrast_distributions.png",
        "spline_15d_latent_correlation.png",
    ):
        figure_path = path.parent / filename
        if not figure_path.exists():
            raise FileNotFoundError(f"Missing 15D spline-prior figure: {figure_path}")
        encoded = base64.b64encode(figure_path.read_bytes()).decode("ascii")
        figures[filename] = f"data:image/png;base64,{encoded}"
    payload["figures"] = figures
    return payload


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _load_results() -> dict[str, Any]:
    required_images = {
        "failed_flow": RESULT_RUNS["failed_flow"] / "learned_prior_vs_truth.png",
        "normalization": JAX_COSMO_SPLINE15D_ANALYSIS
        / "normalization_before_after.png",
        "prior_recovery": RESULT_RUNS["prior_resume_400"]
        / "snapshots/epoch_645/truth_vs_prior.png",
        "encoder_overview": RESULT_RUNS["encoder"] / "training_history_overview.png",
    }
    missing = [str(path) for path in required_images.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing explorer result figures: {missing}")

    histories = []
    for key in ("prior_initial", "prior_resume_155", "prior_resume_400"):
        path = RESULT_RUNS[key] / "epoch_snapshot_history.csv"
        if path.exists():
            histories.append(pd.read_csv(path))
    prior_history = (
        pd.concat(histories, ignore_index=True)
        .sort_values("epoch")
        .drop_duplicates("epoch", keep="last")
    )
    prior_history = prior_history[prior_history["epoch"] <= 645]

    normalization = pd.read_csv(
        JAX_COSMO_SPLINE15D_ANALYSIS / "normalization_parameters.csv"
    ).replace({np.nan: None})
    normalization_contract = json.loads(
        (JAX_COSMO_SPLINE15D_ANALYSIS / "normalization.json").read_text(
            encoding="utf-8"
        )
    )
    encoder_log = pd.read_csv(RESULT_RUNS["encoder"] / "training_log.csv")
    encoder_epochs = (
        encoder_log.groupby(["split", "epoch"], as_index=False)
        .agg(
            loss=("loss", "mean"),
            negative_loglike=("negative_loglike", "mean"),
            kl_mc_mean=("kl_mc_mean", "mean"),
            kl_weight=("kl_weight", "mean"),
            posterior_median_log_std=("posterior_median_log_std", "mean"),
            encoder_grad_norm=("encoder_grad_norm", "mean"),
            prior_grad_norm=("prior_grad_norm", "mean"),
        )
        .sort_values(["split", "epoch"])
    )
    summary = json.loads(
        (RESULT_RUNS["encoder"] / "training_summary.json").read_text(encoding="utf-8")
    )
    return {
        "figures": {key: _image_data_url(path) for key, path in required_images.items()},
        "prior_history": prior_history.to_dict(orient="records"),
        "normalization_parameters": normalization.to_dict(orient="records"),
        "normalization_contract": normalization_contract,
        "encoder_history": encoder_epochs.to_dict(orient="records"),
        "encoder_summary": summary,
        "prior_checkpoint_epoch": 645,
    }


def main() -> None:
    args = parse_args()
    if args.spline_nodes < 3:
        raise ValueError("--spline-nodes must be at least 3")
    spline_jit_max_abs_error = _validate_jax_cosmo_cubic()
    config = load_config(args.config)
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8")) or {}
    frame = pd.read_parquet(args.dataset)
    examples, q_obs = _representative_rows(frame)
    theta = theta_from_truth_frame(examples).astype(np.float32)

    filters = load_filters(config["bands"])
    context = dsps_model.load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"]["n_sfh_bins"]),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    forward = _forward_batch(
        context,
        theta,
        spline_nodes=args.spline_nodes,
        spline_grid=args.spline_grid,
    )
    max_kernel_error = float(np.max(forward["raw_component_relative_error"]))
    if max_kernel_error > 5.0e-5:
        raise ValueError(
            "Diagnostic Diffstar decomposition does not reproduce the wrapper: "
            f"max relative error={max_kernel_error:.4g}"
        )

    spline_scan = _load_spline_scan(args.spline_scan)
    spline_prior = _load_spline_prior(args.spline_prior)
    results = _load_results()
    payload = _clean_json(
        _build_payload(
            frame,
            examples,
            q_obs,
            forward,
            context,
            filters,
            config,
            manifest,
            args,
            spline_jit_max_abs_error,
            spline_scan,
            spline_prior,
        )
    )
    payload["results"] = _clean_json(results)
    payload_text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    args.payload_out.parent.mkdir(parents=True, exist_ok=True)
    args.payload_out.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )

    template = args.template.read_text(encoding="utf-8")
    if "__FENIKS_PAYLOAD__" not in template:
        raise ValueError(f"Missing __FENIKS_PAYLOAD__ marker in {args.template}")
    required_navigation_contract = (
        ".stage-panel { display: none; }",
        ".stage-panel[hidden] { display: none !important; }",
        "panel.hidden = !active;",
    )
    missing_navigation = [
        marker for marker in required_navigation_contract if marker not in template
    ]
    if missing_navigation:
        raise ValueError(
            "Explorer navigation contract is incomplete: "
            + ", ".join(missing_navigation)
        )
    html = template.replace("__FENIKS_PAYLOAD__", payload_text.replace("</", "<\\/"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size:,} bytes)")
    print(f"Wrote {args.payload_out} ({args.payload_out.stat().st_size:,} bytes)")
    print(f"Examples: {', '.join(examples['explorer_key'])}")
    print(f"Max raw-kernel decomposition relative error: {max_kernel_error:.3e}")
    print(f"JAX-COSMO cubic eager/JIT max abs error: {spline_jit_max_abs_error:.3e}")


if __name__ == "__main__":
    main()
