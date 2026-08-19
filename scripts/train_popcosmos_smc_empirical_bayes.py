#!/usr/bin/env python3
"""Fit and validate one population-prior candidate from Pop-COSMOS SMC banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.amortized.smc_empirical_bayes import (
    direct_smc_validation_gate,
    evaluate_prior,
    fit_smc_weighted_prior,
    load_weighted_smc_banks,
    pooled_particles_and_weights,
    prior_ratio_diagnostics,
    split_object_positions,
)
from euclid_dsps.amortized.train import (
    _latent_spec_for_amortized_config,
    load_checkpoint,
)
from euclid_dsps.config import load_config
from euclid_dsps.io import write_json

eqx, _optax = require_amortized_dependencies()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--object-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--trust-strength", type=float, default=0.2)
    parser.add_argument("--trust-samples", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=260819)
    parser.add_argument("--source-logprior-atol", type=float, default=5.0e-3)
    parser.add_argument("--min-mean-logevidence-delta", type=float, default=0.0)
    parser.add_argument("--min-median-ratio-ess-fraction", type=float, default=0.5)
    parser.add_argument("--min-fraction-ratio-ess-ge-0p2", type=float, default=0.9)
    parser.add_argument(
        "--max-seed-mean-logevidence-delta-difference", type=float, default=0.25
    )
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.config, args.checkpoint, *args.bank):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected GPU backend, got {jax.default_backend()}")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    (args.out / "checkpoints").mkdir()

    config = load_config(args.config)
    latent_spec = _latent_spec_for_amortized_config(config)
    bank_summaries = [_validated_bank_summary(path) for path in args.bank]
    banks = load_weighted_smc_banks(args.bank, latent_spec.names)
    model = load_checkpoint(args.checkpoint, config)
    source_logprior = evaluate_prior(model.prior, banks.particles)
    discrepancy = np.abs(source_logprior - banks.stored_logprior)
    max_source_logprior_error = float(np.max(discrepancy))
    p99_source_logprior_error = float(np.quantile(discrepancy, 0.99))
    if not np.all(np.isfinite(discrepancy)):
        raise ValueError("Source checkpoint or stored SMC logprior is non-finite")
    if max_source_logprior_error > float(args.source_logprior_atol):
        raise ValueError(
            "Source checkpoint prior does not reproduce the SMC target prior: "
            f"max_abs_error={max_source_logprior_error:.6g} "
            f"atol={args.source_logprior_atol:.6g}"
        )

    train_positions, validation_positions = split_object_positions(
        banks.n_objects,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    np.save(args.out / "train_object_positions.npy", train_positions)
    np.save(args.out / "validation_object_positions.npy", validation_positions)
    np.save(args.out / "train_row_indices.npy", banks.row_indices[train_positions])
    np.save(
        args.out / "validation_row_indices.npy",
        banks.row_indices[validation_positions],
    )

    particles, weights = pooled_particles_and_weights(banks)
    print(
        "[smc-empirical-bayes] "
        f"objects={banks.n_objects} train={len(train_positions)} "
        f"validation={len(validation_positions)} banks={banks.n_banks} "
        f"particles_per_bank={banks.particles_per_object}",
        flush=True,
    )
    candidate_prior, history = fit_smc_weighted_prior(
        model.prior,
        particles,
        weights,
        train_positions,
        epochs=args.epochs,
        object_batch_size=args.object_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        trust_strength=args.trust_strength,
        trust_samples=args.trust_samples,
        seed=args.seed,
    )
    pd.DataFrame(history).to_csv(args.out / "mstep_history.csv", index=False)
    for row in history:
        print(
            "[smc-empirical-bayes] "
            f"epoch={int(row['epoch'])}/{args.epochs} "
            f"mean_train_loss={row['mean_train_loss']:.6g}",
            flush=True,
        )

    candidate_logprior = evaluate_prior(candidate_prior, banks.particles)
    diagnostics = prior_ratio_diagnostics(
        source_logprior,
        candidate_logprior,
        banks.weights,
        row_indices=banks.row_indices,
        bank_names=tuple(path.name for path in args.bank),
    )
    diagnostics["split"] = np.where(
        diagnostics["object_position"].isin(validation_positions),
        "validation",
        "train",
    )
    diagnostics.to_parquet(args.out / "prior_ratio_diagnostics.parquet", index=False)
    diagnostics.to_csv(args.out / "prior_ratio_diagnostics.csv", index=False)
    gate = direct_smc_validation_gate(
        diagnostics,
        validation_positions,
        min_mean_log_evidence_delta=args.min_mean_logevidence_delta,
        min_median_ratio_ess_fraction=args.min_median_ratio_ess_fraction,
        min_fraction_ratio_ess_ge_0p2=args.min_fraction_ratio_ess_ge_0p2,
        max_seed_mean_logevidence_delta_difference=(
            args.max_seed_mean_logevidence_delta_difference
        ),
    )
    write_json(args.out / "support_gate.json", gate)

    metadata = {
        "algorithm": "direct SMC-weighted single M-step",
        "population_contract": "selected COSMOS2020 Farmer catalog population",
        "e_step": "stopped normalized joint SMC particles and smc_weight",
        "ordinary_importance_sampling_used": False,
        "psis_used": False,
        "posterior_medians_used": False,
        "encoder_frozen_exactly": True,
        "likelihood_frozen_exactly": True,
        "calibration_frozen_exactly": True,
        "validation_gate": gate,
    }
    candidate_checkpoint = args.out / "checkpoints" / "candidate.eqx"
    _write_prior_checkpoint(
        candidate_checkpoint,
        model,
        candidate_prior,
        source_checkpoint=args.checkpoint,
        metadata=metadata,
    )
    selected = gate["status"] == "PASS"
    best_checkpoint = args.out / "checkpoints" / "best.eqx"
    if selected:
        _copy_checkpoint(candidate_checkpoint, best_checkpoint)
        next_action = "CONFIRM_UPDATED_PRIOR_ON_NEW_DISJOINT_SMC_COHORT"
    else:
        _copy_checkpoint(args.checkpoint, best_checkpoint)
        next_action = "STOP_PRIOR_UPDATE"

    train_diagnostic = diagnostics[diagnostics["split"] == "train"]
    summary = {
        "status": "complete",
        "selection_status": gate["status"],
        "selected_candidate": "updated_prior" if selected else "source_prior",
        "next_action": next_action,
        "algorithm": "direct SMC-weighted single M-step",
        "population_contract": (
            "selected-catalog empirical prior conditional on the COSMOS2020 "
            "Farmer sample; not an intrinsic pre-selection population prior"
        ),
        "distribution_contract": (
            "M-step consumes stopped normalized joint SMC particles; no logq, "
            "ordinary importance weights, PSIS, or marginal posterior medians"
        ),
        "spectroscopy_used": False,
        "n_objects": banks.n_objects,
        "n_train_objects": int(len(train_positions)),
        "n_validation_objects": int(len(validation_positions)),
        "n_banks": banks.n_banks,
        "particles_per_object_per_bank": banks.particles_per_object,
        "pooled_particles_per_object": int(particles.shape[1]),
        "mstep_epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "trust_strength": float(args.trust_strength),
        "source_logprior_reproduction": {
            "status": "PASS",
            "max_absolute_error": max_source_logprior_error,
            "p99_absolute_error": p99_source_logprior_error,
            "absolute_tolerance": float(args.source_logprior_atol),
        },
        "train_diagnostics": {
            "mean_log_evidence_delta": float(
                train_diagnostic["log_evidence_delta"].mean()
            ),
            "median_prior_ratio_ess_fraction": float(
                train_diagnostic["prior_ratio_ess_fraction"].median()
            ),
        },
        "validation_gate": gate,
        "encoder_frozen_exactly": True,
        "likelihood_frozen_exactly": True,
        "calibration_frozen_exactly": True,
        "ordinary_importance_sampling_used": False,
        "proposal_refresh_launched": False,
        "further_em_iterations_launched": False,
        "inputs": {
            "config": _receipt(args.config),
            "source_checkpoint": _receipt(args.checkpoint),
            "smc_banks": [
                {
                    "path": str(path),
                    "summary": _receipt(path / "smc_summary.json"),
                    "row_indices": _receipt(path / "row_indices.npy"),
                    "support_status": payload["support_gate"]["status"],
                }
                for path, payload in zip(args.bank, bank_summaries, strict=True)
            ],
        },
        "git_commit": _git_commit(),
    }
    write_json(args.out / "smc_empirical_bayes_summary.json", summary)
    write_json(args.out / "DONE", {"status": "complete"})
    print(
        "[smc-empirical-bayes] "
        f"selection={summary['selection_status']} "
        f"selected={summary['selected_candidate']} next={next_action} -> {args.out}",
        flush=True,
    )


def _validated_bank_summary(path: Path) -> dict:
    summary_path = path / "smc_summary.json"
    if not summary_path.is_file() or not (path / "DONE").is_file():
        raise FileNotFoundError(f"Incomplete combined SMC bank: {path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"SMC bank is not complete: {path}")
    if payload.get("support_gate", {}).get("status") != "PASS":
        raise ValueError(f"SMC bank support gate failed: {path}")
    return payload


def _write_prior_checkpoint(
    path: Path, model, prior, *, source_checkpoint: Path, metadata: dict
) -> None:
    updated_model = eqx.tree_at(lambda value: value.prior, model, prior)
    eqx.tree_serialise_leaves(path, updated_model)
    source_sidecar = Path(str(source_checkpoint) + ".json")
    sidecar = (
        json.loads(source_sidecar.read_text(encoding="utf-8"))
        if source_sidecar.is_file()
        else {}
    )
    sidecar["direct_smc_empirical_bayes"] = {
        **metadata,
        "source_checkpoint": _receipt(source_checkpoint),
    }
    write_json(Path(str(path) + ".json"), sidecar)


def _copy_checkpoint(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)
    sidecar = Path(str(source) + ".json")
    if sidecar.is_file():
        shutil.copy2(sidecar, Path(str(destination) + ".json"))


def _receipt(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


if __name__ == "__main__":
    main()
