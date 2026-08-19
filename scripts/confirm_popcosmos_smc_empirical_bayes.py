#!/usr/bin/env python3
"""Confirm an updated prior on fresh candidate-prior SMC posteriors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from euclid_dsps.amortized.smc_empirical_bayes import (
    direct_smc_validation_gate,
    evaluate_prior,
    load_weighted_smc_banks,
    prior_ratio_diagnostics,
    validate_smc_checkpoint_provenance,
)
from euclid_dsps.amortized.train import (
    _latent_spec_for_amortized_config,
    load_checkpoint,
)
from euclid_dsps.config import load_config
from euclid_dsps.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-logprior-atol", type=float, default=3.0e-2)
    parser.add_argument("--min-mean-logevidence-delta", type=float, default=0.0)
    parser.add_argument("--min-median-ratio-ess-fraction", type=float, default=0.5)
    parser.add_argument("--min-fraction-ratio-ess-ge-0p2", type=float, default=0.9)
    parser.add_argument(
        "--max-seed-mean-logevidence-delta-difference", type=float, default=0.25
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    config = load_config(args.config)
    latent_spec = _latent_spec_for_amortized_config(config)
    bank_summaries = [_validated_bank_summary(path) for path in args.bank]
    banks = load_weighted_smc_banks(args.bank, latent_spec.names)
    source_receipt = _receipt(args.source_checkpoint)
    candidate_receipt = _receipt(args.candidate_checkpoint)
    if source_receipt["sha256"] == candidate_receipt["sha256"]:
        raise ValueError("Source and candidate prior checkpoints are identical")
    validate_smc_checkpoint_provenance(
        bank_summaries,
        checkpoint_sha256=candidate_receipt["sha256"],
    )

    source_model = load_checkpoint(args.source_checkpoint, config)
    candidate_model = load_checkpoint(args.candidate_checkpoint, config)
    candidate_logprior = evaluate_prior(
        candidate_model.prior,
        banks.particles,
        smc_object_batch_size=4,
    )
    candidate_reproduction_error = np.abs(candidate_logprior - banks.stored_logprior)
    max_reproduction_error = float(np.max(candidate_reproduction_error))
    if not np.all(np.isfinite(candidate_reproduction_error)):
        raise ValueError("Candidate checkpoint or stored SMC logprior is non-finite")
    if max_reproduction_error > float(args.candidate_logprior_atol):
        raise ValueError(
            "Candidate checkpoint prior does not reproduce the SMC target prior: "
            f"max_abs_error={max_reproduction_error:.6g} "
            f"atol={args.candidate_logprior_atol:.6g}"
        )
    source_logprior = evaluate_prior(
        source_model.prior,
        banks.particles,
        smc_object_batch_size=4,
    )

    diagnostics = prior_ratio_diagnostics(
        candidate_logprior,
        source_logprior,
        banks.weights,
        row_indices=banks.row_indices,
        bank_names=tuple(path.name for path in args.bank),
    )
    diagnostics = diagnostics.rename(
        columns={"log_evidence_delta": "source_minus_candidate_log_evidence"}
    )
    diagnostics["candidate_minus_source_log_evidence"] = -diagnostics[
        "source_minus_candidate_log_evidence"
    ]
    gate_input = diagnostics.rename(
        columns={"candidate_minus_source_log_evidence": "log_evidence_delta"}
    )
    gate = direct_smc_validation_gate(
        gate_input,
        np.arange(banks.n_objects),
        min_mean_log_evidence_delta=args.min_mean_logevidence_delta,
        min_median_ratio_ess_fraction=args.min_median_ratio_ess_fraction,
        min_fraction_ratio_ess_ge_0p2=args.min_fraction_ratio_ess_ge_0p2,
        max_seed_mean_logevidence_delta_difference=(
            args.max_seed_mean_logevidence_delta_difference
        ),
    )
    diagnostics.to_parquet(args.out / "confirmation_diagnostics.parquet", index=False)
    diagnostics.to_csv(args.out / "confirmation_diagnostics.csv", index=False)
    write_json(args.out / "support_gate.json", gate)
    summary = {
        "status": "complete",
        "confirmation_status": gate["status"],
        "selected_candidate": (
            "updated_prior" if gate["status"] == "PASS" else "source_prior"
        ),
        "next_action": (
            "REFRESH_ENCODER_AND_TEST_ORDINARY_IS"
            if gate["status"] == "PASS"
            else "STOP_UPDATED_PRIOR_CONFIRMATION_FAILED"
        ),
        "algorithm": "reverse prior-ratio confirmation on candidate-prior SMC",
        "evidence_direction": "candidate_prior_minus_source_prior",
        "n_objects": banks.n_objects,
        "n_banks": banks.n_banks,
        "particles_per_object_per_bank": banks.particles_per_object,
        "spectroscopy_used": False,
        "population_contract": "selected COSMOS2020 Farmer catalog population",
        "candidate_logprior_reproduction": {
            "status": "PASS",
            "checkpoint_sha256_match": True,
            "max_absolute_error": max_reproduction_error,
            "p99_absolute_error": float(
                np.quantile(candidate_reproduction_error, 0.99)
            ),
            "absolute_tolerance": float(args.candidate_logprior_atol),
        },
        "validation_gate": gate,
        "inputs": {
            "config": _receipt(args.config),
            "source_checkpoint": source_receipt,
            "candidate_checkpoint": candidate_receipt,
            "smc_banks": [
                {
                    "path": str(path),
                    "summary": _receipt(path / "smc_summary.json"),
                }
                for path in args.bank
            ],
        },
    }
    write_json(args.out / "prior_confirmation_summary.json", summary)
    if gate["status"] == "PASS":
        write_json(args.out / "DONE", {"status": "complete"})
    else:
        write_json(args.out / "FAILED", {"status": "gate_failed"})
    print(
        "[smc-em-confirm] "
        f"confirmation={gate['status']} "
        f"mean_logz_delta={gate['mean_log_evidence_delta']:.6g} "
        f"ratio_ess={gate['median_prior_ratio_ess_fraction']:.4g} "
        f"-> {args.out}",
        flush=True,
    )
    if gate["status"] != "PASS":
        raise SystemExit(3)


def _validated_bank_summary(path: Path) -> dict:
    summary_path = path / "smc_summary.json"
    if not summary_path.is_file() or not (path / "DONE").is_file():
        raise FileNotFoundError(f"Incomplete combined SMC bank: {path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("support_gate", {}).get("status") != "PASS":
        raise ValueError(f"SMC support gate failed: {path}")
    return payload


def _receipt(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    main()
