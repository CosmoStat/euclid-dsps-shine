#!/usr/bin/env python3
"""Run a frozen-checkpoint FENIKS adaptive-SMC E-step pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

from euclid_dsps.amortized.adaptive_smc_trainer import (
    run_feniks_adaptive_smc_e_step_pilot,
)
from euclid_dsps.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--train-indices-file", type=Path, required=True)
    parser.add_argument("--validation-indices-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--primary-rw-scale", type=float, default=0.30)
    parser.add_argument("--fallback-rw-scale", type=float, default=0.15)
    parser.add_argument("--expected-devices", type=int, default=4)
    args = parser.parse_args()
    for path in (
        args.config,
        args.catalog,
        args.train_indices_file,
        args.validation_indices_file,
        args.checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"expected GPU backend, got {jax.default_backend()}")
    if len(jax.local_devices()) != args.expected_devices:
        raise RuntimeError(
            f"expected {args.expected_devices} devices, got {jax.local_devices()}"
        )
    config = load_config(args.config)
    config["catalog_path"] = str(args.catalog)
    receipt = run_feniks_adaptive_smc_e_step_pilot(
        config,
        args.out,
        train_indices_file=args.train_indices_file,
        validation_indices_file=args.validation_indices_file,
        checkpoint=args.checkpoint,
        primary_rw_scale=args.primary_rw_scale,
        fallback_rw_scale=args.fallback_rw_scale,
    )
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
