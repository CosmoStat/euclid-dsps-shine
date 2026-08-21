#!/usr/bin/env python3
"""Estimate batched latent-object DSPS evaluations for adaptive-SMC training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from euclid_dsps.config import load_config


def _smc_evaluations(
    *,
    particles: int,
    resamples: int,
    mutation_steps: int,
    final_steps: int,
) -> int:
    return particles * (1 + resamples * mutation_steps + final_steps)


def estimate(config_path: Path, manifest_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    objective = config["amortized"]["objective"]
    smc = objective["adaptive_smc"]
    fallback = smc["hard_fallback"]
    training = config["amortized"]["training"]["adaptive_smc"]
    sleep = objective["sleep"]
    selection = objective["selection_correction"]
    n_train = int(manifest["manifests"]["train"]["count"])
    n_validation = int(manifest["manifests"]["validation"]["count"])
    sweeps = int(training["observed_sweeps"])
    micro_batch = int(training["micro_batch_size"])
    bootstrap_epochs = int(training["bootstrap_sleep_epochs"])
    replay_every = int(training["sleep_replay_every_smc_updates"])
    primary_k = int(smc["n_particles"])
    fallback_k = int(fallback["n_particles"])
    primary_moves = int(smc["steps_after_resample"])
    fallback_moves = int(fallback["steps_after_resample"])
    primary_final = int(smc["final_steps_at_beta1"])
    fallback_final = int(fallback["final_steps_at_beta1"])
    easy = _smc_evaluations(
        particles=primary_k,
        resamples=0,
        mutation_steps=primary_moves,
        final_steps=primary_final,
    )
    typical = _smc_evaluations(
        particles=primary_k,
        resamples=1,
        mutation_steps=primary_moves,
        final_steps=primary_final,
    )
    two_resamples = _smc_evaluations(
        particles=primary_k,
        resamples=2,
        mutation_steps=primary_moves,
        final_steps=primary_final,
    )
    primary_worst = _smc_evaluations(
        particles=primary_k,
        resamples=int(smc["max_stages"]),
        mutation_steps=primary_moves,
        final_steps=primary_final,
    )
    fallback_typical = _smc_evaluations(
        particles=fallback_k,
        resamples=1,
        mutation_steps=fallback_moves,
        final_steps=fallback_final,
    )
    fallback_worst = _smc_evaluations(
        particles=fallback_k,
        resamples=int(fallback["max_stages"]),
        mutation_steps=fallback_moves,
        final_steps=fallback_final,
    )
    candidate_factor = int(sleep["selection"]["candidate_factor"])
    bootstrap = bootstrap_epochs * n_train * candidate_factor
    total_micro_batches = sweeps * math.ceil(n_train / micro_batch)
    replay_updates = total_micro_batches // max(replay_every, 1)
    replay = replay_updates * micro_batch * candidate_factor
    prior_updates = sweeps * math.ceil(
        n_train / int(training["prior_macro_objects"])
    )
    alpha = prior_updates * int(selection["n_prior_samples"])
    validation_objects = min(
        int(training["validation_objects"]),
        n_validation,
    )
    validation = sweeps * (
        validation_objects * typical
        + validation_objects * int(training["validation_q_is_particles"])
        + int(selection["n_prior_samples"])
    )

    def total(observed_per_object: float) -> int:
        observed = round(sweeps * n_train * observed_per_object)
        return int(observed + bootstrap + replay + alpha + validation)

    hard_conservative = 0.8 * typical + 0.2 * (
        primary_worst + fallback_typical
    )
    return {
        "status": "estimate",
        "unit": "latent-object DSPS forward evaluations; evaluations are vectorized in large GPU batches",
        "objects": {"train": n_train, "validation": n_validation},
        "smc_per_object_per_sweep": {
            "easy_zero_resamples": easy,
            "typical_one_resample": typical,
            "two_resamples": two_resamples,
            "hard_primary_exhausted_plus_typical_fallback": (
                primary_worst + fallback_typical
            ),
            "absolute_configured_primary_plus_fallback_upper_bound": (
                primary_worst + fallback_worst
            ),
        },
        "non_smc": {
            "bootstrap_sleep": bootstrap,
            "sleep_replay": replay,
            "selection_alpha": alpha,
            "validation": validation,
            "expected_prior_macro_updates": prior_updates,
        },
        "complete_training_scenarios": {
            "all_easy": total(easy),
            "all_typical": total(typical),
            "all_two_resamples": total(two_resamples),
            "twenty_percent_hard_conservative": total(hard_conservative),
        },
        "assumptions": {
            "observed_sweeps": sweeps,
            "primary_final_move_always_counted": True,
            "hard_conservative": (
                "20% of objects exhaust primary stages then require one-resample fallback"
            ),
            "not_included": (
                "cheap q/prior density evaluations, rejected sleep candidates beyond the fixed candidate pool, compilation"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(estimate(args.config, args.manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
