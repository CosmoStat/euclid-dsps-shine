#!/usr/bin/env python3
"""Fail-fast validation for the common-15D mode-covering array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from euclid_dsps.amortized.latent import latent_spec_from_config, latent_spec_hash
from euclid_dsps.amortized.train import (
    _spline15d_latent_spec_from_checkpoint,
    _validate_loaded_prior_spec,
)
from euclid_dsps.config import load_config


def validate_mode_covering_inputs(
    catalog_dir: Path,
    reference_checkpoint: Path,
    config_paths: list[Path],
) -> None:
    catalog_dir = Path(catalog_dir)
    reference_checkpoint = Path(reference_checkpoint)
    contract_path = catalog_dir / "amortized_catalog_contract.json"
    sidecar_path = reference_checkpoint.with_suffix(
        reference_checkpoint.suffix + ".json"
    )
    for path in (contract_path, reference_checkpoint, sidecar_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing required input: {path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    parameter_names = tuple(contract.get("parameter_names", ()))
    if contract.get("version") != 1 or len(parameter_names) != 15:
        raise ValueError("Catalog must provide the version-1 spline15d contract")
    if contract.get("truth_kind") != "exact_spline15d":
        raise ValueError("Catalog truth_kind must be exact_spline15d")
    if contract.get("join_key") != "object_id":
        raise ValueError("Catalog join_key must be object_id")
    for split in ("train", "test"):
        path = catalog_dir / f"{split}.parquet"
        parquet = pq.ParquetFile(path)
        missing = [
            name
            for name in ("object_id", *parameter_names)
            if name not in parquet.schema_arrow.names
        ]
        if missing:
            raise ValueError(f"{split} parquet missing columns: {missing}")
        recorded = ((contract.get("splits") or {}).get(split) or {}).get("rows")
        if int(parquet.metadata.num_rows) != int(recorded):
            raise ValueError(f"{split} parquet row count does not match contract")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if tuple(sidecar.get("parameter_names", ())) != parameter_names:
        raise ValueError("Reference checkpoint parameter order does not match catalog")
    if sidecar.get("version") != 1:
        raise ValueError("Reference checkpoint version must be 1")
    if (sidecar.get("flow_integrity") or {}).get("status") != "PASS":
        raise ValueError("Reference checkpoint flow integrity status is not PASS")
    architecture = dict(sidecar.get("architecture", {}) or {})
    if architecture.get("type") != "realnvp":
        raise ValueError(
            "The controlled frozen reference must be the serialized JAX-COSMO "
            "RealNVP checkpoint"
        )
    if int(architecture.get("latent_dim", -1)) != len(parameter_names):
        raise ValueError("Reference checkpoint latent dimension does not match catalog")

    hashes: dict[str, str] = {}
    for path in config_paths:
        config = load_config(path)
        cfg = dict(config.get("amortized", {}) or {})
        latent = dict(cfg.get("latent", {}) or {})
        objective = dict(cfg.get("objective", {}) or {})
        encoder = dict(cfg.get("encoder", {}) or {})
        prior = dict(cfg.get("prior", {}) or {})
        normalization_checkpoint = Path(str(latent.get("normalization_checkpoint", "")))
        if normalization_checkpoint != reference_checkpoint:
            raise ValueError(
                f"{path}: normalization checkpoint {normalization_checkpoint} "
                f"does not match {reference_checkpoint}"
            )
        objective_mode = str(objective.get("mode", "stochastic_elbo")).lower()
        if objective_mode in {
            "hybrid",
            "hybrid_elbo",
            "npe",
            "neural_posterior_estimation",
        }:
            raise ValueError(f"{path}: training objective unexpectedly uses truth")
        if any(
            float(objective.get(key, 0.0)) != 0.0
            for key in ("npe_weight", "prior_truth_weight")
        ):
            raise ValueError(f"{path}: supervised objective weight must be zero")
        if str(encoder.get("flow_output_space")) != "latent_x":
            raise ValueError(f"{path}: wake controls require independent latent_x q")
        if (
            str(prior.get("source", "")) == "spline15d_checkpoint"
            and Path(str(prior.get("checkpoint", ""))) != reference_checkpoint
        ):
            raise ValueError(
                f"{path}: frozen prior does not use the reference checkpoint"
            )
        configured_names = tuple(
            (config.get("fit", {}) or {}).get("free_parameters", {})
        )
        if configured_names != parameter_names:
            raise ValueError(f"{path}: parameter order does not match catalog")
        active_spec = latent_spec_from_config(config)
        normalized_spec = _spline15d_latent_spec_from_checkpoint(
            normalization_checkpoint,
            active_spec,
            sidecar=sidecar,
        )
        _validate_loaded_prior_spec(
            active_spec,
            normalized_spec,
            require_normalization=False,
        )
        hashes[str(path)] = latent_spec_hash(normalized_spec)

    unique_hashes = set(hashes.values())
    if len(unique_hashes) != 1:
        raise ValueError(
            f"Configs do not share one latent normalization hash: {hashes}"
        )
    print(
        "[mode-covering-contract] valid: "
        f"configs={len(config_paths)} normalization_hash={next(iter(unique_hashes))} "
        f"reference_family={architecture.get('type')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    args = parser.parse_args()
    validate_mode_covering_inputs(
        args.catalog_dir,
        args.reference_checkpoint,
        args.config,
    )


if __name__ == "__main__":
    main()
