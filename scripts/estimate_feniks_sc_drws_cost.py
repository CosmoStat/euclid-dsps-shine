#!/usr/bin/env python3
"""Estimate SC-DRWS DSPS work and extrapolate from an optional smoke rate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def estimate(objects: int, hard_fraction: float, dsps_per_second: float | None):
    n = int(objects)
    hard = float(hard_fraction)
    phase_a_wake = 15 * n * 64
    phase_b_first = 30 * n * 128
    phase_b_expansion = 30 * n * hard * 384
    sleep = (45 + 90) * n * 8
    alpha = 30 * math.ceil(n / 1024) * 4096
    training = phase_a_wake + phase_b_first + phase_b_expansion + sleep + alpha
    result = {
        "objects": n,
        "assumed_hard_expansion_fraction": hard,
        "latent_object_dsps_evaluations": {
            "phase_a_wake_k64": phase_a_wake,
            "phase_b_first_pass_k128": phase_b_first,
            "phase_b_hard_additional_k384": phase_b_expansion,
            "selected_sleep_candidate_factor_8": sleep,
            "selection_alpha_mc": alpha,
            "training_total": training,
            "confirmation_raw_ema_k2048_on_2000": 2 * 2000 * 2048,
            "final_raw_ema_k2048_on_512": 2 * 512 * 2048,
        },
        "memory_shape_contract_per_h100": {
            "ordinary_k128_at_16_objects": 2048,
            "hard_k512_at_4_objects": 2048,
            "k2048_validation_at_8_objects": 16384,
            "note": "counts are concurrent latent-object DSPS evaluations, not bytes",
        },
    }
    if dsps_per_second and dsps_per_second > 0:
        seconds = training / dsps_per_second
        result["calibrated_runtime"] = {
            "aggregate_four_h100_dsps_per_second": dsps_per_second,
            "hours_per_seed_four_h100": seconds / 3600,
            "idealized_hours_per_seed_eight_h100": seconds / 7200,
            "idealized_hours_two_seeds_sixteen_h100": seconds / 7200,
            "warning": "8-H100 per-seed scaling is not authorized until the dedicated multi-node collective smoke passes",
        }
    else:
        result["calibrated_runtime"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=int, required=True)
    parser.add_argument("--hard-fraction", type=float, default=0.20)
    parser.add_argument("--smoke-training-log", type=Path)
    args = parser.parse_args()
    rate = None
    if args.smoke_training_log:
        frame = pd.read_csv(args.smoke_training_log)
        evaluations = pd.to_numeric(
            frame["estimated_dsps_evaluations"], errors="coerce"
        )
        elapsed = pd.to_numeric(frame["batch_elapsed_seconds"], errors="coerce")
        valid = evaluations.notna() & elapsed.notna() & (elapsed > 0)
        if valid.any():
            rate = float(evaluations[valid].sum() / elapsed[valid].sum())
    print(json.dumps(estimate(args.objects, args.hard_fraction, rate), indent=2))


if __name__ == "__main__":
    main()
