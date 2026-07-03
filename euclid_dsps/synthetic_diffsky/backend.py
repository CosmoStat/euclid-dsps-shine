"""Proposal backends for synthetic Diffsky DSPS closure catalogs."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.filters import FilterCurve
from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES

from .config import SplitGenerationConfig, SyntheticDiffskyConfig
from .metallicity import absolute_lgmet_to_logzsol


def generate_proposal_shard(
    cfg: SyntheticDiffskyConfig,
    split: SplitGenerationConfig,
    shard_index: int,
    *,
    filters: dict[str, FilterCurve],
    ssp_lgmet: np.ndarray,
    z_sun: float,
    ssp_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate one weighted proposal shard with the configured backend."""
    if cfg.proposal_backend == "diffsky":
        return generate_diffsky_proposal_shard(
            cfg,
            split,
            shard_index,
            filters=filters,
            ssp_lgmet=ssp_lgmet,
            z_sun=z_sun,
            ssp_path=ssp_path,
        )
    if cfg.proposal_backend == "toy":
        return generate_toy_proposal_shard(
            cfg,
            split,
            shard_index,
            ssp_lgmet=ssp_lgmet,
            z_sun=z_sun,
        )
    raise ValueError(
        "synthetic_diffsky.proposal_backend must be 'diffsky' for science runs "
        "or explicit 'toy' for tests."
    )


def generate_diffsky_proposal_shard(
    cfg: SyntheticDiffskyConfig,
    split: SplitGenerationConfig,
    shard_index: int,
    *,
    filters: dict[str, FilterCurve],
    ssp_lgmet: np.ndarray,
    z_sun: float,
    ssp_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate one proposal shard with Diffsky's weighted analytic lightcone."""
    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jran
        from diffsky.experimental.lightcone_generators import weighted_lc_photdata
        from diffsky.experimental.mc_phot import mc_lc_phot
        from diffsky.param_utils.load_calib_params import get_calib_params
        from dsps import load_ssp_templates
        from dsps.data_loaders.defaults import TransmissionCurve
        from dsps.metallicity.umzr import mzr_model
    except ImportError as exc:  # pragma: no cover - depends on science env
        missing = getattr(exc, "name", None)
        missing_msg = f" Missing module: {missing}." if missing else ""
        raise ImportError(
            "Diffsky synthetic closure generation requires diffsky, diffstar, "
            "diffmah, diffhalos, dsps, and JAX in the active environment. "
            "Install the FENIKS-capable Diffsky stack before running the "
            "science backend."
            f"{missing_msg} Original import error: {type(exc).__name__}: {exc}"
        ) from exc

    key = _jax_random_key(jran, int(split.source_seed) + int(shard_index))
    lc_key, phot_key = jran.split(key)
    ssp_data = load_ssp_templates(fn=str(ssp_path))
    tcurves = _diffsky_transmission_curves(filters, TransmissionCurve)
    z_phot_table = jnp.linspace(
        float(cfg.z_min), float(cfg.z_max), int(cfg.z_phot_table_size)
    )
    feniks_params = get_calib_params(
        calibration_dir=cfg.calibration_dir,
        calibration_name=cfg.calibration_name,
    )
    lc_data = weighted_lc_photdata(
        lc_key,
        int(cfg.n_host_halos_per_shard),
        float(cfg.z_min),
        float(cfg.z_max),
        float(cfg.lgmp_min),
        float(cfg.lgmp_max),
        float(cfg.sky_area_degsq),
        ssp_data,
        tcurves,
        z_phot_table,
        logmp_cutoff=float(cfg.logmp_cutoff),
        **cfg.weighted_lc_photdata_kwargs,
    )
    phot_info, phot_randoms, merging_randoms = mc_lc_phot(
        phot_key,
        lc_data,
        int(cfg.mc_merge),
        param_collection=feniks_params,
        **cfg.mc_lc_phot_kwargs,
    )
    del phot_randoms, merging_randoms
    lgmet_abs = np.asarray(
        jax.device_get(
            mzr_model(
                _attr_array(phot_info, "logsm_obs"),
                _attr_array(lc_data, "t_obs"),
                *feniks_params.mzr_params,
            )
        ),
        dtype=float,
    )
    frame = _extract_common_truth_frame(
        lc_data=lc_data,
        phot_info=phot_info,
        lgmet_abs_median=lgmet_abs,
        ssp_lgmet=ssp_lgmet,
        z_sun=z_sun,
        metallicity_grid_policy=cfg.metallicity_grid_policy,
        metallicity_scatter_dex=cfg.stellar_metallicity_scatter_dex,
    )
    metadata = {
        "backend": "diffsky",
        "calibration_name": cfg.calibration_name,
        "n_host_halos": int(cfg.n_host_halos_per_shard),
        "n_proposals": int(len(frame)),
    }
    return frame, metadata


def generate_toy_proposal_shard(
    cfg: SyntheticDiffskyConfig,
    split: SplitGenerationConfig,
    shard_index: int,
    *,
    ssp_lgmet: np.ndarray,
    z_sun: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate a small deterministic correlated proposal for tests only."""
    rng = np.random.default_rng(int(split.source_seed) + int(shard_index))
    n = max(int(cfg.n_host_halos_per_shard), 1)
    z = rng.uniform(cfg.z_min, cfg.z_max, n)
    logm = rng.normal(10.2 + 0.8 * z / max(cfg.z_max, 1.0e-6), 0.45, n)
    logm = np.clip(logm, 7.5, 12.2)
    lgmet_abs = np.log10(z_sun) + np.clip(-0.6 + 0.25 * (logm - 10.0) - 0.3 * z, -2.0, 0.2)
    transform = absolute_lgmet_to_logzsol(
        lgmet_abs,
        z_sun=z_sun,
        ssp_lgmet=ssp_lgmet,
        policy=cfg.metallicity_grid_policy,
    )
    is_central = rng.random(n) < 0.78
    ssfr = np.clip(rng.normal(-10.2 - 0.7 * (logm - 10.0), 0.35, n), -13.5, -8.0)
    data: dict[str, Any] = {
        "redshift_true": z,
        "logsm_true": logm,
        "diffstar_lgmcrit_true": rng.normal(12.0, 0.25, n),
        "diffstar_lgy_at_mcrit_true": rng.normal(-10.0, 0.25, n),
        "diffstar_indx_lo_true": rng.normal(1.0, 0.4, n),
        "diffstar_indx_hi_true": rng.normal(-1.0, 0.35, n),
        "diffstar_lg_qt_true": rng.normal(1.0, 0.25, n),
        "diffstar_qlglgdt_true": rng.normal(-0.5, 0.25, n),
        "diffstar_lg_drop_true": rng.normal(-1.0, 0.25, n),
        "diffstar_lg_rejuv_true": rng.normal(-0.2, 0.25, n),
        "diffmah_logm0_true": np.clip(logm + rng.normal(1.8, 0.35, n), 8.0, 15.5),
        "diffmah_logtc_true": rng.normal(0.05, 0.25, n),
        "diffmah_early_index_true": rng.normal(2.5, 0.5, n),
        "diffmah_late_index_true": rng.normal(0.2, 0.2, n),
        "diffmah_t_peak_true": rng.uniform(2.0, 13.5, n),
        "log10_stellar_metallicity_true": transform.log10_z_over_zsun,
        "dust_av_true": np.clip(rng.normal(0.25 + 0.18 * (logm - 10.0), 0.18, n), 0.0, 2.5),
        "dust_delta_true": np.clip(rng.normal(-0.4, 0.25, n), -2.0, 0.5),
        "logssfr_true": ssfr,
        "logsfr_true": ssfr + logm,
        "logmp_true": np.clip(logm + rng.normal(1.5, 0.3, n), 8.0, 15.5),
        "logmp0_true": np.clip(logm + rng.normal(1.8, 0.3, n), 8.0, 15.5),
        "central_true": is_central.astype(bool),
        "cen_weight": np.exp(rng.normal(0.0, 0.25, n)),
        "sat_weight": np.where(is_central, 1.0, np.exp(rng.normal(-0.2, 0.35, n))),
        "lgmet_abs_median_true": lgmet_abs,
        "lgmet_abs_used_true": transform.lgmet_abs_used,
        "metallicity_scatter_dex": np.full(n, cfg.stellar_metallicity_scatter_dex),
        "metallicity_clipped": transform.clipped_mask,
        "metallicity_clip_low": transform.clip_low_mask,
        "metallicity_clip_high": transform.clip_high_mask,
    }
    frame = pd.DataFrame(data)
    frame["galaxy_weight"] = frame["cen_weight"] * frame["sat_weight"]
    return frame, {"backend": "toy", "n_proposals": int(n)}


def _extract_common_truth_frame(
    *,
    lc_data: Any,
    phot_info: Any,
    lgmet_abs_median: np.ndarray,
    ssp_lgmet: np.ndarray,
    z_sun: float,
    metallicity_grid_policy: str,
    metallicity_scatter_dex: float,
) -> pd.DataFrame:
    transform = absolute_lgmet_to_logzsol(
        lgmet_abs_median,
        z_sun=z_sun,
        ssp_lgmet=ssp_lgmet,
        policy=metallicity_grid_policy,
    )
    logsm = _np_attr(phot_info, "logsm_obs")
    logsfr = _first_np_attr(phot_info, ("logsfr_obs", "log_sfr_obs", "logsfr"))
    logssfr = _first_np_attr(phot_info, ("logssfr_obs", "log_ssfr_obs", "logssfr"))
    if logssfr is None and logsfr is not None:
        logssfr = logsfr - logsm
    if logsfr is None and logssfr is not None:
        logsfr = logssfr + logsm
    n = len(logsm)
    data: dict[str, Any] = {
        "redshift_true": _np_attr(lc_data, "z_obs"),
        "logsm_true": logsm,
        "diffstar_lgmcrit_true": _np_attr(phot_info, "lgmcrit"),
        "diffstar_lgy_at_mcrit_true": _np_attr(phot_info, "lgy_at_mcrit"),
        "diffstar_indx_lo_true": _np_attr(phot_info, "indx_lo"),
        "diffstar_indx_hi_true": _np_attr(phot_info, "indx_hi"),
        "diffstar_lg_qt_true": _np_attr(phot_info, "lg_qt"),
        "diffstar_qlglgdt_true": _np_attr(phot_info, "qlglgdt"),
        "diffstar_lg_drop_true": _np_attr(phot_info, "lg_drop"),
        "diffstar_lg_rejuv_true": _np_attr(phot_info, "lg_rejuv"),
        "diffmah_logm0_true": _np_attr(lc_data.mah_params, "logm0"),
        "diffmah_logtc_true": _np_attr(lc_data.mah_params, "logtc"),
        "diffmah_early_index_true": _np_attr(lc_data.mah_params, "early_index"),
        "diffmah_late_index_true": _np_attr(lc_data.mah_params, "late_index"),
        "diffmah_t_peak_true": _np_attr(lc_data.mah_params, "t_peak"),
        "log10_stellar_metallicity_true": transform.log10_z_over_zsun,
        "dust_av_true": _np_attr(phot_info, "av"),
        "dust_delta_true": _np_attr(phot_info, "delta"),
        "logssfr_true": _nan_if_missing(logssfr, n),
        "logsfr_true": _nan_if_missing(logsfr, n),
        "logmp_true": _np_attr(lc_data, "logmp_obs"),
        "logmp0_true": _np_attr(lc_data, "logmp0"),
        "central_true": _np_attr(lc_data, "is_central").astype(bool),
        "cen_weight": _np_attr(lc_data, "cen_weight"),
        "sat_weight": _np_attr(lc_data, "sat_weight"),
        "lgmet_abs_median_true": np.asarray(lgmet_abs_median, dtype=float),
        "lgmet_abs_used_true": transform.lgmet_abs_used,
        "metallicity_scatter_dex": np.full(n, metallicity_scatter_dex),
        "metallicity_clipped": transform.clipped_mask,
        "metallicity_clip_low": transform.clip_low_mask,
        "metallicity_clip_high": transform.clip_high_mask,
    }
    frame = pd.DataFrame(data)
    frame["galaxy_weight"] = frame["cen_weight"] * frame["sat_weight"]
    _validate_truth_columns(frame)
    return frame


def _diffsky_transmission_curves(
    filters: dict[str, FilterCurve],
    transmission_curve_type: Any,
) -> Any:
    names = [str(name) for name in filters]
    TCurves = namedtuple("DiffskyTransmissionCurves", names)
    return TCurves(
        *[
            transmission_curve_type(
                wave=np.asarray(filters[name].wave, dtype=float),
                transmission=np.asarray(filters[name].transmission, dtype=float),
            )
            for name in names
        ]
    )


def _jax_random_key(jran: Any, seed: int) -> Any:
    """Return a JAX random key compatible with old and new JAX releases."""
    key_factory = getattr(jran, "key", None)
    if key_factory is not None:
        return key_factory(int(seed))
    return jran.PRNGKey(int(seed))


def _attr_array(obj: Any, name: str) -> Any:
    if not hasattr(obj, name):
        raise AttributeError(f"Diffsky object is missing required field {name!r}")
    return getattr(obj, name)


def _np_attr(obj: Any, name: str) -> np.ndarray:
    return np.asarray(_attr_array(obj, name), dtype=float)


def _first_np_attr(obj: Any, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if hasattr(obj, name):
            return np.asarray(getattr(obj, name), dtype=float)
    return None


def _nan_if_missing(values: np.ndarray | None, n: int) -> np.ndarray:
    if values is None:
        return np.full(int(n), np.nan, dtype=float)
    return np.asarray(values, dtype=float)


def _validate_truth_columns(frame: pd.DataFrame) -> None:
    missing = [
        f"{name}_true"
        for name in DIFFSKY_BASIC_PARAMETER_NAMES
        if name not in {"z_obs", "log10_stellar_mass"}
        and f"{name}_true" not in frame.columns
    ]
    for column in ("redshift_true", "logsm_true"):
        if column not in frame.columns:
            missing.append(column)
    if missing:
        raise ValueError(f"Proposal frame missing required truth columns: {missing}")
