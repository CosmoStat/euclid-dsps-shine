#!/usr/bin/env python3
"""Fail-fast contract for the final four-candidate self-supervised array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_feniks_mode_covering_inputs import validate_mode_covering_inputs

from euclid_dsps.config import load_config


def validate_paper_inputs(
    catalog_dir: Path,
    reference_checkpoint: Path,
    config_paths: list[Path],
) -> None:
    if len(config_paths) != 4:
        raise ValueError("Final paper array requires exactly four configs")
    validate_mode_covering_inputs(catalog_dir, reference_checkpoint, config_paths)
    sidecar_path = reference_checkpoint.with_suffix(
        reference_checkpoint.suffix + ".json"
    )
    sidecar = json.loads(
        sidecar_path.read_text(encoding="utf-8")
    )
    if (
        (sidecar.get("normalization") or {}).get("family")
        != "mixed_log_shifted_asinh"
    ):
        raise ValueError("Paper array requires the immutable mixed-asinh normalization")

    seeds = []
    expected_modes = (
        "reweighted_wake_sleep",
        "reweighted_wake_sleep",
        "reweighted_wake_sleep",
        "stochastic_elbo",
    )
    expected_priors = (
        "joint_realnvp",
        "joint_realnvp",
        "spline15d_checkpoint",
        "joint_realnvp",
    )
    for path, expected_mode, expected_prior in zip(
        config_paths, expected_modes, expected_priors, strict=True
    ):
        config = load_config(path)
        cfg = dict(config.get("amortized", {}) or {})
        likelihood = dict(cfg.get("likelihood", {}) or {})
        objective = dict(cfg.get("objective", {}) or {})
        sleep = dict(objective.get("sleep", {}) or {})
        wake = dict(objective.get("wake", {}) or {})
        encoder = dict(cfg.get("encoder", {}) or {})
        prior = dict(cfg.get("prior", {}) or {})
        training = dict(cfg.get("training", {}) or {})
        if str(likelihood.get("type")) != "student_t":
            raise ValueError(f"{path}: likelihood must be Student-t")
        if float(likelihood.get("student_t_dof", -1.0)) != 2.0:
            raise ValueError(f"{path}: Student-t must use dof=2")
        if float(likelihood.get("error_floor_frac", -1.0)) != 0.0:
            raise ValueError(f"{path}: error floor must remain zero")
        if float(likelihood.get("error_jitter", -1.0)) != 0.0:
            raise ValueError(f"{path}: error jitter must remain zero")
        if str(objective.get("mode", "")).lower() != expected_mode:
            raise ValueError(f"{path}: unexpected objective mode")
        if str(prior.get("source", "")) != expected_prior:
            raise ValueError(f"{path}: unexpected prior source")
        if str(encoder.get("type")) != "conditional_flow":
            raise ValueError(f"{path}: encoder must be a conditional flow")
        if str(encoder.get("flow_family")) != "realnvp":
            raise ValueError(f"{path}: encoder flow must be RealNVP")
        if int(encoder.get("flow_layers", 0)) != 4:
            raise ValueError(f"{path}: expected four conditional-flow layers")
        if expected_mode == "reweighted_wake_sleep":
            if not bool(sleep.get("enabled", False)):
                raise ValueError(f"{path}: RWS sleep phase must be enabled")
            if str(sleep.get("noise_family")) != "match_likelihood":
                raise ValueError(f"{path}: sleep noise must match Student-t2")
            if str(wake.get("sampler")) != "importance":
                raise ValueError(f"{path}: final RWS control must use importance wake")
            if int(wake.get("n_particles", 0)) != 8:
                raise ValueError(f"{path}: final RWS control must use K=8")
        if expected_prior == "joint_realnvp":
            if not bool(prior.get("train_jointly", False)):
                raise ValueError(f"{path}: learned prior is unexpectedly frozen")
            if str(prior.get("init")) != "identity":
                raise ValueError(f"{path}: learned prior must start at identity")
        elif bool(prior.get("train_jointly", False)):
            raise ValueError(f"{path}: reference prior must remain frozen")
        if expected_mode == "reweighted_wake_sleep":
            expected_train_prior = expected_prior == "joint_realnvp"
            if bool(wake.get("train_prior", False)) != expected_train_prior:
                raise ValueError(f"{path}: wake prior-update contract is inconsistent")
        seeds.append(int(training.get("seed", -1)))

    if seeds[0] == seeds[1]:
        raise ValueError("RWS replication seeds must differ")
    print(
        "[selfsup-paper-contract] valid: "
        "two RWS seeds + frozen-prior RWS + learned-prior AVI"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    args = parser.parse_args()
    validate_paper_inputs(
        args.catalog_dir,
        args.reference_checkpoint,
        args.config,
    )


if __name__ == "__main__":
    main()
