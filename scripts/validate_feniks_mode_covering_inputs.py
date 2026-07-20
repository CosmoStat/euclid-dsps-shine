#!/usr/bin/env python3
"""Fail-fast validation for the common-15D mode-covering array."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

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
        hashes[str(path)] = _normalization_hash(config, sidecar, parameter_names)

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


def _normalization_hash(
    config: dict,
    sidecar: dict,
    parameter_names: tuple[str, ...],
) -> str:
    """Reproduce the runtime LatentSpec hash without importing JAX."""
    free = (config.get("fit", {}) or {}).get("free_parameters", {}) or {}
    lower = []
    upper = []
    for name in parameter_names:
        bounds = (free.get(name, {}) or {}).get("bounds")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Missing [low, high] bounds for {name}")
        low, high = float(bounds[0]), float(bounds[1])
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError(f"Invalid bounds for {name}: {bounds}")
        lower.append(low)
        upper.append(high)

    normalization = dict(sidecar.get("normalization", {}) or {})
    if normalization.get("whitening") is not None:
        raise ValueError("Common spline15d normalization must not use whitening")
    if float(normalization.get("normalized_atom_half_width", 0.0)) != 0.0:
        raise ValueError("Runtime normalized atom jitter is not supported")
    transforms = dict(normalization.get("transforms", {}) or {})
    family = []
    center = []
    scale = []
    location = []
    lam = []
    for name in parameter_names:
        transform = dict(transforms.get(name, {}) or {})
        transform_family = str(transform.get("family", ""))
        if transform_family == "log":
            family.append(1)
            location.append(0.0)
            lam.append(1.0)
        elif transform_family == "shifted_asinh":
            family.append(0)
            location.append(float(transform["location"]))
            lam.append(float(transform["lambda"]))
        else:
            raise ValueError(
                f"Unsupported spline15d transform for {name}: {transform_family}"
            )
        center.append(float(transform["center"]))
        scale.append(float(transform["scale"]))

    def as_float32(values: list[float] | list[int]) -> list[float]:
        return np.asarray(values, dtype=np.float32).astype(float).tolist()

    payload = {
        "names": list(parameter_names),
        "lower": as_float32(lower),
        "upper": as_float32(upper),
        "raw_center": as_float32(center),
        "raw_scale": as_float32(scale),
        "normalization": "spline15d_mixed",
        "transform_family": as_float32(family),
        "transform_location": as_float32(location),
        "transform_lambda": as_float32(lam),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
