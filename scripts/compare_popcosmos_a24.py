#!/usr/bin/env python3
"""Compare RWS posterior summaries with the public T24/A24 Pop-COSMOS table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

PARAMETER_MAP = {
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
    "ln_fagn": "lnfAGN_pc",
    "ln_tauagn": "lntauAGN_pc",
    "log10_gas_metallicity": "log10Zgas_pc",
    "log10_gas_ionization": "log10Ugas_pc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--a24-summaries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--match-radius-arcsec", type=float, default=0.3)
    return parser.parse_args()


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    return np.column_stack(
        (np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec))
    )


def match_catalogs(
    ours: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    radius_arcsec: float,
) -> pd.DataFrame:
    """Return unique nearest sky matches within ``radius_arcsec``."""
    reference = reference.copy()
    ref_valid = np.isfinite(reference["RA"]) & np.isfinite(reference["DEC"])
    reference = reference.loc[ref_valid].reset_index(drop=True)
    tree = cKDTree(
        _unit_vectors(reference["RA"].to_numpy(), reference["DEC"].to_numpy())
    )
    distance, index = tree.query(
        _unit_vectors(ours["ra_deg"].to_numpy(), ours["dec_deg"].to_numpy()),
        k=1,
    )
    angle_arcsec = np.rad2deg(2.0 * np.arcsin(np.clip(distance / 2.0, 0, 1)))
    angle_arcsec *= 3600.0
    valid = np.isfinite(angle_arcsec) & (angle_arcsec <= radius_arcsec)
    left = ours.loc[valid].reset_index(drop=True).add_prefix("rws_")
    right = reference.iloc[index[valid]].reset_index(drop=True).add_prefix("a24_")
    matched = pd.concat((left, right), axis=1)
    matched["match_arcsec"] = angle_arcsec[valid]
    matched = matched.sort_values("match_arcsec")
    matched = matched.drop_duplicates("a24_INDEX", keep="first")
    return matched.reset_index(drop=True)


def _photoz_metrics(
    frame: pd.DataFrame,
    prefix: str,
    *,
    truth_column: str = "rws_redshift_true",
) -> dict[str, float | int]:
    zspec = pd.to_numeric(frame[truth_column], errors="coerce").to_numpy(float)
    median = frame[f"{prefix}_median"].to_numpy(float)
    q16 = frame[f"{prefix}_q16"].to_numpy(float)
    q84 = frame[f"{prefix}_q84"].to_numpy(float)
    valid = np.isfinite(zspec) & (zspec >= 0.0) & np.isfinite(median)
    if not valid.any():
        return {"n_spec": 0}
    dz = (median[valid] - zspec[valid]) / (1.0 + zspec[valid])
    center = float(np.median(dz))
    interval = np.isfinite(q16[valid]) & np.isfinite(q84[valid])
    covered = (
        (zspec[valid][interval] >= q16[valid][interval])
        & (zspec[valid][interval] <= q84[valid][interval])
    )
    return {
        "n_spec": int(valid.sum()),
        "bias_median": center,
        "nmad": float(1.48 * np.median(np.abs(dz - center))),
        "rmse": float(np.sqrt(np.mean(dz**2))),
        "outlier_fraction_0p15": float(np.mean(np.abs(dz) > 0.15)),
        "coverage_68": float(np.mean(covered)) if len(covered) else np.nan,
    }


def main() -> None:
    args = parse_args()
    posterior = pd.read_parquet(args.inference / "posterior_summary.parquet")
    available = set(pq.ParquetFile(args.dataset).schema.names)
    position_columns = ["object_id", "ra_deg", "dec_deg"]
    position_columns.extend(
        column
        for column in (
            "redshift_true",
            "redshift_spec",
            "specz_confidence_level",
            "specz_survey",
            "t24_specz_flag",
        )
        if column in available
    )
    positions = pd.read_parquet(args.dataset, columns=position_columns)
    ours = posterior.merge(positions, on="object_id", validate="many_to_one")
    reference = pd.read_csv(
        args.a24_summaries,
        sep=r"\s+",
        na_values=["-99", "None"],
        low_memory=False,
    )
    reference = reference.loc[
        (reference["MAGCUT_r"] == "Y") & (reference["XRAY"] == "N")
    ].copy()
    matched = match_catalogs(
        ours, reference, radius_arcsec=args.match_radius_arcsec
    )
    if matched.empty:
        raise RuntimeError("No RWS objects matched the public A24 summary table")

    parameter_rows = []
    for rws_name, a24_stem in PARAMETER_MAP.items():
        rws_column = f"rws_{rws_name}_median"
        a24_column = f"a24_{a24_stem}_500"
        if rws_column not in matched or a24_column not in matched:
            continue
        rws = pd.to_numeric(matched[rws_column], errors="coerce").to_numpy(float)
        a24 = pd.to_numeric(matched[a24_column], errors="coerce").to_numpy(float)
        valid = np.isfinite(rws) & np.isfinite(a24)
        if not valid.any():
            continue
        delta = rws[valid] - a24[valid]
        parameter_rows.append(
            {
                "parameter": rws_name,
                "n": int(valid.sum()),
                "median_delta_rws_minus_a24": float(np.median(delta)),
                "median_abs_delta": float(np.median(np.abs(delta))),
                "rmse_delta": float(np.sqrt(np.mean(delta**2))),
            }
        )

    matched["rws_z_median"] = matched["rws_z_obs_median"]
    matched["rws_z_q16"] = matched["rws_z_obs_q16"]
    matched["rws_z_q84"] = matched["rws_z_obs_q84"]
    matched["a24_z_median"] = pd.to_numeric(
        matched["a24_z_pc_500"], errors="coerce"
    )
    matched["a24_z_q16"] = pd.to_numeric(
        matched["a24_z_pc_160"], errors="coerce"
    )
    matched["a24_z_q84"] = pd.to_numeric(
        matched["a24_z_pc_840"], errors="coerce"
    )
    photoz = {
        "rws": _photoz_metrics(matched, "rws_z"),
        "a24": _photoz_metrics(matched, "a24_z"),
    }
    t24_flagged = (
        matched.loc[matched["rws_t24_specz_flag"].fillna(False).astype(bool)]
        if "rws_t24_specz_flag" in matched
        else matched.iloc[0:0]
    )
    photoz_t24_flagged = {
        method: _photoz_metrics(t24_flagged, f"{method}_z")
        for method in ("rws", "a24")
    }
    args.out.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(args.out / "matched_posteriors.parquet", index=False)
    pd.DataFrame(parameter_rows).to_csv(
        args.out / "parameter_median_comparison.csv", index=False
    )
    pd.DataFrame(
        [
            {"method": method, "sample": "public_specz_dr1p1", **metrics}
            for method, metrics in photoz.items()
        ]
        + [
            {"method": method, "sample": "t24_flagged_intersection", **metrics}
            for method, metrics in photoz_t24_flagged.items()
        ]
    ).to_csv(args.out / "photoz_metrics.csv", index=False)
    summary = {
        "status": "complete",
        "n_rws": int(len(ours)),
        "n_a24_r_cut_non_xray": int(len(reference)),
        "n_matched": int(len(matched)),
        "match_radius_arcsec": float(args.match_radius_arcsec),
        "max_match_arcsec": float(matched["match_arcsec"].max()),
        "photoz": photoz,
        "photoz_t24_flagged_intersection": photoz_t24_flagged,
        "spectroscopic_reference": {
            "column": "rws_redshift_true",
            "semantics": (
                "COSMOS DR1.1 public spectroscopy with Confidence_level >= 50, "
                "joined by Id_COS20_Farmer"
            ),
            "a24_z_SPEC_semantics": (
                "Y/N availability flag only; never interpreted as a redshift"
            ),
        },
        "parameter_comparison": parameter_rows,
        "interpretation": (
            "Matched same-object total-method comparison. DSPS and A24 "
            "Photulator forward models are not identical."
        ),
    }
    (args.out / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"[popcosmos-comparison] matched={len(matched)}/{len(ours)} "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()
