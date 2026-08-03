#!/usr/bin/env python3
"""Validate the native spline15D transfer contract on COSMOS2020 photometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from euclid_dsps.amortized.config import amortized_config
from euclid_dsps.config import load_config
from euclid_dsps.cosmos2020 import cosmos_band_names_for_subset
from euclid_dsps.filters import load_filters
from euclid_dsps.observation_arrays import photometry_arrays_from_dataframe

NATIVE_PARAMETERS = (
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
    "sfh_dlog_sfr_01",
    "sfh_dlog_sfr_02",
    "sfh_dlog_sfr_03",
    "sfh_dlog_sfr_04",
    "sfh_dlog_sfr_05",
    "sfh_dlog_sfr_06",
    "sfh_dlog_sfr_07",
    "sfh_dlog_sfr_08",
    "sfh_dlog_sfr_09",
    "sfh_dlog_sfr_10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/popcosmos_native15d_rws.yaml"),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    _require_file(args.data_dir / "PREPOST_COMPLETE.json")
    manifest_path = args.data_dir / "preparation_manifest.json"
    _require_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("n_bands", 0)) != 26:
        raise ValueError("Prepared COSMOS catalog does not declare 26 bands")
    spectroscopy = manifest.get("spectroscopic_redshifts") or {}
    if int(spectroscopy.get("matched_selected_rows", 0)) <= 0:
        raise ValueError("Prepared catalog has no reliable public spectroscopy")

    full = args.data_dir / "farmer_a24_full.parquet"
    _require_file(full)
    schema = set(pq.ParquetFile(full).schema.names)
    required = {
        "object_id",
        "redshift_true",
        "redshift_spec",
        "specz_confidence_level",
    }
    if not required.issubset(schema):
        raise ValueError(
            "Prepared catalog is missing columns: "
            + ", ".join(sorted(required - schema))
        )
    n_full = pq.ParquetFile(full).metadata.num_rows
    previous: set[int] = set()
    for requested in (512, 5_000, 20_000, 40_000):
        size = min(requested, n_full)
        path = args.data_dir / f"farmer_a24_n{size}.parquet"
        _require_file(path)
        ids = set(pq.read_table(path, columns=["object_id"])["object_id"].to_pylist())
        if previous and not previous.issubset(ids):
            raise ValueError(f"COSMOS subsets are not nested at {path}")
        previous = ids

    config = load_config(args.config)
    amortized = amortized_config(config)
    names = tuple(config["fit"]["free_parameters"])
    if names != NATIVE_PARAMETERS:
        raise ValueError(f"Native spline15D parameter mismatch: {names}")
    if config["model"].get("sfh_model") != "spline15d":
        raise ValueError("Native spline15D SFH model is not active")
    target = config.get("science_target", {})
    if target.get("reported_parameters") != ["z_obs"]:
        raise ValueError("Science target must report only z_obs")
    if target.get("truth_evaluated_parameters") != ["z_obs"]:
        raise ValueError("Truth evaluation must contain only z_obs")
    if len(target.get("nuisance_parameters", ())) != 14:
        raise ValueError("Expected fourteen native nuisance coordinates")
    data_config = amortized["data"]
    if data_config.get("selection_mode") != "random":
        raise ValueError("Training selection must be random and truth-independent")
    if data_config.get("use_redshift_for_split") is not False:
        raise ValueError("Public redshift truth must not influence training splits")
    if data_config.get("stratify_column") is not None:
        raise ValueError("Native training must not configure a redshift proxy")
    if "lp_zbest" in config.get("extra_columns", ()):
        raise ValueError("lp_zbest must not enter the native redshift-only workflow")
    if amortized["objective"]["sleep"].get("error_model") != "observed_catalog":
        raise ValueError("Sleep must reuse Farmer-reported uncertainty vectors")
    if Path(amortized["features"]["stats_catalog_path"]).resolve() != (
        args.data_dir / "farmer_a24_n40000.parquet"
    ).resolve():
        raise ValueError("Feature statistics must use only the fixed 40k train pool")
    likelihood = amortized["likelihood"]
    if likelihood.get("type") != "student_t" or float(
        likelihood.get("student_t_dof", np.nan)
    ) != 2.0:
        raise ValueError("Native transfer requires a Student-t2 likelihood")
    if amortized["objective"].get("mode") != "reweighted_wake_sleep":
        raise ValueError("Native transfer requires the RWS objective")
    if int(amortized["objective"].get("wake", {}).get("n_particles", 0)) != 8:
        raise ValueError("Native transfer requires the RWS k=8 wake objective")

    band_names = tuple(band["name"] for band in config["bands"])
    subset = (config.get("dataset", {}) or {}).get("band_subset", "cosmos26")
    expected_names = cosmos_band_names_for_subset(subset)
    if band_names != expected_names:
        raise ValueError(
            "Native config does not use the expected ordered COSMOS band subset: "
            f"subset={subset!r} expected={expected_names} got={band_names}"
        )
    if any(str(band.get("units", "")).lower() != "microjy" for band in config["bands"]):
        raise ValueError("All Farmer input bands must explicitly declare microjy")
    for band in config["bands"]:
        band["filter"]["path"] = str(
            args.asset_dir / "filters" / f"{band['name']}.dat"
        )
    filters = load_filters(config["bands"])
    if any(len(curve.wave) < 2 for curve in filters.values()):
        raise ValueError("At least one COSMOS passband is empty")

    for path in (
        Path(config["ssp_path"]),
        Path(config["model"]["compressed_ssp_path"]),
        Path(amortized["latent"]["normalization_checkpoint"]),
        Path(str(amortized["latent"]["normalization_checkpoint"]) + ".json"),
    ):
        _require_file(path)

    frame = pq.read_table(full).slice(0, min(n_full, 8)).to_pandas()
    arrays = photometry_arrays_from_dataframe(
        frame,
        config["bands"],
        object_id_column="object_id",
    )
    if arrays.flux.shape != (len(frame), len(expected_names)):
        raise ValueError(f"Unexpected Farmer photometry shape: {arrays.flux.shape}")
    for index, band in enumerate(config["bands"]):
        raw = frame[band["column"]].to_numpy(float)
        valid = arrays.mask[:, index] & np.isfinite(raw)
        if valid.any() and not np.allclose(
            arrays.flux[valid, index],
            raw[valid] * 1.0e-29,
            rtol=2.0e-6,
            atol=0.0,
        ):
            raise ValueError(f"microJy to fnu-cgs conversion failed for {band['name']}")

    if args.run_dir is not None:
        for path in (
            args.run_dir / "DONE",
            args.run_dir / "stage_contract.json",
            args.run_dir / "train/checkpoints/best.eqx",
            args.run_dir / "train/training_summary.json",
            args.run_dir / "inference/inference_summary.json",
            args.run_dir / "inference/posterior_summary.parquet",
            args.run_dir / "inference/redshift_metrics.json",
            args.run_dir / "inference/redshift_predictions.parquet",
        ):
            _require_file(path)
        redshift = json.loads(
            (args.run_dir / "inference/redshift_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        if redshift.get("science_target") != "z_obs_only":
            raise ValueError("Run does not declare the redshift-only science target")
        if int((redshift.get("metrics") or {}).get("n_spec", 0)) <= 0:
            raise ValueError("Held-out inference cohort contains no public spectroscopy")
        stage_contract = json.loads(
            (args.run_dir / "stage_contract.json").read_text(encoding="utf-8")
        )
        cohort = stage_contract.get("evaluation_cohort", {})
        if cohort.get("excluded_from_training_pool") is not True:
            raise ValueError("Evaluation cohort is not held out from the 40k pool")
        if cohort.get("redshift_used_for_selection") is not False:
            raise ValueError("Evaluation cohort selection used redshift truth")
        evaluation_indices = Path(str(cohort.get("row_indices", "")))
        _require_file(evaluation_indices)
        indices = np.asarray(np.load(evaluation_indices), dtype=np.int64)
        if len(indices) < int(cohort.get("selected_rows", 0)):
            raise ValueError("Evaluation index file is shorter than the stage cohort")
        full_ids = pq.read_table(full, columns=["object_id"])[
            "object_id"
        ].to_numpy()
        train_ids = pq.read_table(
            args.data_dir / "farmer_a24_n40000.parquet",
            columns=["object_id"],
        )["object_id"].to_numpy()
        if np.isin(full_ids[indices], train_ids).any():
            raise ValueError("Evaluation cohort overlaps the 40k training pool")
        selection = json.loads(
            (args.run_dir / "inference/inference_selection.json").read_text(
                encoding="utf-8"
            )
        )
        if selection.get("selection_mode") != "row_indices_file":
            raise ValueError("Inference did not use the fixed held-out row indices")
        if stage_contract.get("stage") == "n40k":
            for path in (
                args.run_dir.parent / "redshift_scaling/redshift_scaling_metrics.csv",
                args.run_dir.parent
                / "redshift_scaling/redshift_scaling_summary.json",
            ):
                _require_file(path)

    print(
        "[cosmos-native15d-contract] "
        f"rows={n_full} bands={len(expected_names)} subset={subset} "
        "latent=15 target=z_obs objective=RWS-k8 "
        f"run={'checked' if args.run_dir else 'not-requested'}"
    )


if __name__ == "__main__":
    main()
