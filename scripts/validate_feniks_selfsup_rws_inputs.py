#!/usr/bin/env python3
"""Fail-fast validation for the self-supervised RWS experiment array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from validate_feniks_mode_covering_inputs import validate_mode_covering_inputs

from euclid_dsps.config import load_config


def validate_selfsup_rws_inputs(
    catalog_dir: Path,
    reference_checkpoint: Path,
    config_paths: list[Path],
) -> None:
    validate_mode_covering_inputs(catalog_dir, reference_checkpoint, config_paths)
    sidecar = json.loads(
        reference_checkpoint.with_suffix(
            reference_checkpoint.suffix + ".json"
        ).read_text(encoding="utf-8")
    )
    normalization = dict(sidecar.get("normalization", {}) or {})
    if normalization.get("family") != "mixed_log_shifted_asinh":
        raise ValueError("RWS array requires the production mixed asinh normalization")

    for path in config_paths:
        config = load_config(path)
        amortized = dict(config.get("amortized", {}) or {})
        likelihood = dict(amortized.get("likelihood", {}) or {})
        objective = dict(amortized.get("objective", {}) or {})
        wake = dict(objective.get("wake", {}) or {})
        sleep = dict(objective.get("sleep", {}) or {})
        prior = dict(amortized.get("prior", {}) or {})
        if str(likelihood.get("type", "")).lower() != "gaussian":
            raise ValueError(f"{path}: closure likelihood must be gaussian")
        if float(likelihood.get("error_floor_frac", -1.0)) != 0.0:
            raise ValueError(f"{path}: closure error_floor_frac must be zero")
        if float(likelihood.get("error_jitter", -1.0)) != 0.0:
            raise ValueError(f"{path}: closure error_jitter must be zero")
        if str(objective.get("mode", "")).lower() != "reweighted_wake_sleep":
            raise ValueError(f"{path}: expected reweighted_wake_sleep objective")
        if not bool(sleep.get("enabled", False)):
            raise ValueError(f"{path}: model-generated sleep contract must be enabled")
        if int(wake.get("n_particles", 0)) < 4:
            raise ValueError(f"{path}: wake requires at least four particles")
        learned = str(prior.get("source", "")) == "joint_realnvp"
        if learned:
            if not bool(prior.get("train_jointly", False)):
                raise ValueError(f"{path}: learned prior is unexpectedly frozen")
            if str(prior.get("init", "")) != "identity":
                raise ValueError(f"{path}: learned prior must start at identity")
            if str(prior.get("update_schedule", "")) != "joint":
                raise ValueError(f"{path}: RWS prior must update on shared wake batches")
            if not bool(wake.get("train_prior", False)):
                raise ValueError(f"{path}: learned prior wake loss is disabled")
        elif bool(wake.get("train_prior", False)):
            raise ValueError(f"{path}: frozen reference cannot update its prior")

        noise_model = dict(
            ((config.get("synthetic_diffsky", {}) or {}).get("flux_error_model", {}))
            or {}
        )
        if str(noise_model.get("type", "")) != "m5_depth":
            raise ValueError(f"{path}: sleep simulator must reuse the m5 noise model")
        if float(noise_model.get("sigma_sys_mag", -1.0)) != 0.005:
            raise ValueError(f"{path}: unexpected synthetic systematic noise")

    noise = _catalog_noise_check(catalog_dir / "test.parquet", load_config(config_paths[0]))
    if not 0.95 <= noise["residual_std"] <= 1.05:
        raise ValueError(f"Catalog fluxerr is not generator-matched: {noise}")
    if not 14.0 <= noise["median_chi2"] <= 21.0:
        raise ValueError(f"Unexpected 18-band oracle chi2: {noise}")
    print(
        "[selfsup-rws-contract] valid: "
        f"configs={len(config_paths)} residual_std={noise['residual_std']:.6f} "
        f"median_chi2={noise['median_chi2']:.6f}"
    )


def _catalog_noise_check(path: Path, config: dict) -> dict[str, float]:
    bands = tuple(str(band["name"]) for band in config.get("bands", ()))
    if len(bands) != 18:
        raise ValueError(f"Expected 18 bands, got {len(bands)}")
    columns = [
        column
        for band in bands
        for column in (f"flux_true_{band}", f"flux_{band}", f"fluxerr_{band}")
    ]
    frame = pq.read_table(path, columns=columns).to_pandas()
    truth = np.column_stack([frame[f"flux_true_{band}"] for band in bands])
    flux = np.column_stack([frame[f"flux_{band}"] for band in bands])
    error = np.column_stack([frame[f"fluxerr_{band}"] for band in bands])
    valid = np.isfinite(truth) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    if not bool(np.all(valid)):
        raise ValueError("Closure noise check requires finite positive errors in all bands")
    residual = (flux - truth) / error
    chi2 = np.sum(residual**2, axis=1)
    return {
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "median_chi2": float(np.median(chi2)),
        "median_reduced_chi2": float(np.median(chi2 / len(bands))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    args = parser.parse_args()
    validate_selfsup_rws_inputs(
        args.catalog_dir,
        args.reference_checkpoint,
        args.config,
    )


if __name__ == "__main__":
    main()
