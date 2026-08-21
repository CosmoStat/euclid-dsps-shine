#!/usr/bin/env python3
"""Train the no-truth FENIKS q and parent prior with adaptive bridge SMC."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

from euclid_dsps.amortized.adaptive_smc_trainer import (
    train_feniks_adaptive_smc,
)
from euclid_dsps.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--train-indices-file", type=Path, required=True)
    parser.add_argument("--validation-indices-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--expected-devices", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.config,
        args.catalog,
        args.train_indices_file,
        args.validation_indices_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.resume_state is not None and not args.resume_state.is_file():
        raise FileNotFoundError(args.resume_state)
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"expected GPU backend, got {jax.default_backend()}")
    if args.expected_devices is not None and len(jax.local_devices()) != int(
        args.expected_devices
    ):
        raise RuntimeError(
            f"expected {args.expected_devices} local devices, "
            f"got {jax.local_devices()}"
        )
    config = load_config(args.config)
    config["catalog_path"] = str(args.catalog)
    receipt = train_feniks_adaptive_smc(
        config,
        args.out,
        train_indices_file=args.train_indices_file,
        validation_indices_file=args.validation_indices_file,
        smoke=args.smoke,
        resume_state=args.resume_state,
    )
    if receipt["status"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
