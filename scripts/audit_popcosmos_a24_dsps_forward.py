#!/usr/bin/env python3
"""Evaluate public A24 posterior medians through the configured DSPS model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

A24_PARAMETER_MAP = {
    "z_obs": "z_pc",
    "log10_stellar_mass": "log10M_pc",
    "log10_stellar_metallicity": "log10Z_pc",
    "dlog10_sfr_1": "log10sfr_ratio_1_pc",
    "dlog10_sfr_2": "log10sfr_ratio_2_pc",
    "dlog10_sfr_3": "log10sfr_ratio_3_pc",
    "dlog10_sfr_4": "log10sfr_ratio_4_pc",
    "dlog10_sfr_5": "log10sfr_ratio_5_pc",
    "dlog10_sfr_6": "log10sfr_ratio_6_pc",
    "tau2": "dust2_pc",
    "dust_index_n": "dust_index_pc",
    "tau1_over_tau2": "dust1_fraction_pc",
    "log10_gas_metallicity": "log10Zgas_pc",
    "log10_gas_ionization": "log10Ugas_pc",
    "ln_fagn": "lnfAGN_pc",
    "ln_tauagn": "lntauAGN_pc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--runtime", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def a24_parameter_matrix(
    matched: pd.DataFrame,
    parameter_names: tuple[str, ...],
) -> np.ndarray:
    columns = []
    for name in parameter_names:
        if name not in A24_PARAMETER_MAP:
            raise KeyError(f"No public A24 mapping for {name}")
        column = f"a24_{A24_PARAMETER_MAP[name]}_500"
        if column not in matched:
            raise KeyError(f"Matched A24 table is missing {column}")
        columns.append(pd.to_numeric(matched[column], errors="coerce").to_numpy())
    return np.column_stack(columns).astype(np.float32)


def forward_audit_tables(
    object_id: np.ndarray,
    observed_flux: np.ndarray,
    observed_error: np.ndarray,
    mask: np.ndarray,
    model_flux: np.ndarray,
    band_names: tuple[str, ...],
    *,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    valid = (
        mask
        & np.isfinite(observed_flux)
        & np.isfinite(observed_error)
        & (observed_error > 0.0)
        & np.isfinite(model_flux)
    )
    residual = np.zeros_like(model_flux, dtype=np.float64)
    np.divide(
        model_flux.astype(np.float64) - observed_flux.astype(np.float64),
        observed_error.astype(np.float64),
        out=residual,
        where=valid,
    )
    chi2 = np.sum(np.where(valid, residual**2, 0.0), axis=1)
    n_valid = np.maximum(valid.sum(axis=1), 1)
    reduced_chi2 = chi2 / n_valid
    object_table = pd.DataFrame(
        {
            "object_id": object_id,
            "label": label,
            "n_valid_bands": valid.sum(axis=1),
            "chi2": chi2,
            "reduced_chi2": reduced_chi2,
            "median_abs_residual_sigma": np.nanmedian(
                np.where(valid, np.abs(residual), np.nan), axis=1
            ),
            "frac_abs_gt_5": np.sum(valid & (np.abs(residual) > 5.0), axis=1)
            / n_valid,
        }
    )
    band_rows = []
    for index, band in enumerate(band_names):
        values = residual[valid[:, index], index]
        band_rows.append(
            {
                "label": label,
                "band": band,
                "n": int(len(values)),
                "median_residual_sigma": float(np.median(values)),
                "median_abs_residual_sigma": float(np.median(np.abs(values))),
                "frac_abs_gt_5": float(np.mean(np.abs(values) > 5.0)),
            }
        )
    summary = {
        "label": label,
        "n_objects": int(len(object_table)),
        "median_reduced_chi2": float(np.median(reduced_chi2)),
        "median_abs_residual_sigma": float(
            np.nanmedian(np.where(valid, np.abs(residual), np.nan))
        ),
        "frac_abs_gt_5": float(
            np.sum(valid & (np.abs(residual) > 5.0)) / np.sum(valid)
        ),
    }
    return object_table, pd.DataFrame(band_rows), summary


def main() -> None:
    args = parse_args()
    if args.runtime == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD"] = "1"
    else:
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ["EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD"] = "0"

    import jax
    import jax.numpy as jnp

    from euclid_dsps.config import load_config
    from euclid_dsps.filters import load_filters
    from euclid_dsps.model import dynamic_model_args, load_context
    from euclid_dsps.observation_arrays import photometry_arrays_from_dataframe
    from euclid_dsps.parameter_vectors import model_mags_from_theta_matrix_jax
    from euclid_dsps.photometry import abmag_to_fnu_cgs_jax

    config = load_config(args.config)
    matched = pd.read_parquet(args.matched)
    if args.limit is not None:
        matched = matched.iloc[: args.limit].copy()
    identity = "rws_object_id"
    if identity not in matched:
        raise KeyError(f"Matched table is missing {identity}")
    object_ids = pd.to_numeric(matched[identity], errors="raise").astype(np.int64)
    dataset = pd.read_parquet(args.dataset)
    dataset = dataset.set_index("object_id", drop=False).loc[object_ids].reset_index(
        drop=True
    )
    arrays = photometry_arrays_from_dataframe(
        dataset,
        config["bands"],
        object_id_column="object_id",
    )
    parameter_names = tuple(config["fit"]["free_parameters"])
    theta = a24_parameter_matrix(matched, parameter_names)
    finite = np.all(np.isfinite(theta), axis=1)
    if not np.all(finite):
        matched = matched.loc[finite].reset_index(drop=True)
        dataset = dataset.loc[finite].reset_index(drop=True)
        theta = theta[finite]
        arrays = photometry_arrays_from_dataframe(
            dataset,
            config["bands"],
            object_id_column="object_id",
        )

    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    model_args = dynamic_model_args(context)
    chunks = []
    for start in range(0, len(theta), args.batch_size):
        stop = min(start + args.batch_size, len(theta))
        mags = model_mags_from_theta_matrix_jax(
            context,
            model_args,
            jnp.asarray(theta[start:stop]),
            parameter_names,
        )
        chunks.append(np.asarray(jax.device_get(abmag_to_fnu_cgs_jax(mags))))
        print(f"[a24-dsps-forward] decoded {stop}/{len(theta)}", flush=True)
    raw_flux = np.concatenate(chunks, axis=0)

    evaluations = [("a24_median_dsps_raw", raw_flux)]
    if args.calibration is not None:
        calibration = pd.read_csv(args.calibration).set_index("band")
        alpha = np.asarray(
            [calibration.loc[name, "alpha_band"] for name in arrays.band_names]
        )
        evaluations.append(("a24_median_dsps_rws_calibrated", raw_flux * alpha))

    object_tables = []
    band_tables = []
    summaries = []
    for label, model_flux in evaluations:
        objects, bands, summary = forward_audit_tables(
            arrays.object_id,
            arrays.flux,
            arrays.flux_err,
            arrays.mask,
            model_flux,
            arrays.band_names,
            label=label,
        )
        object_tables.append(objects)
        band_tables.append(bands)
        summaries.append(summary)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(object_tables, ignore_index=True).to_parquet(
        args.out / "a24_dsps_forward_objects.parquet", index=False
    )
    pd.concat(band_tables, ignore_index=True).to_csv(
        args.out / "a24_dsps_forward_by_band.csv", index=False
    )
    payload = {
        "status": "complete",
        "dataset": str(args.dataset),
        "matched": str(args.matched),
        "n_input_matched": int(len(object_ids)),
        "n_finite_a24_vectors": int(len(theta)),
        "parameter_names": list(parameter_names),
        "summaries": summaries,
        "interpretation": (
            "Diagnostic only: a vector of marginal posterior medians is not "
            "a joint A24 posterior draw. Large residuals nevertheless reveal "
            "a forward-model or parameter-contract mismatch before RWS scaling."
        ),
    }
    (args.out / "a24_dsps_forward_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
