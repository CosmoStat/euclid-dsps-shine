#!/usr/bin/env python3
"""Write no-truth parent/selected-population artifacts for frozen SC-DRWS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from euclid_dsps.amortized.adaptive_smc_trainer import (
    prepare_adaptive_training_runtime,
)
from euclid_dsps.amortized.sc_asmc_report import _prior_report_arrays
from euclid_dsps.amortized.sc_drws import C0_SCOPE_STATEMENT
from euclid_dsps.amortized.train import load_checkpoint
from euclid_dsps.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weighted_correlation(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalized = weights / weights.sum()
    mean = np.sum(values * normalized[:, None], axis=0)
    centered = values - mean
    covariance = (centered * normalized[:, None]).T @ centered
    scale = np.sqrt(np.maximum(np.diag(covariance), 1.0e-30))
    return covariance / np.outer(scale, scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-indices-file", type=Path, required=True)
    parser.add_argument("--validation-indices-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior-samples", type=int, default=8192)
    parser.add_argument("--selected-resamples", type=int, default=8192)
    parser.add_argument("--decoder-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=260827)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    runtime = prepare_adaptive_training_runtime(
        config,
        args.out / "runtime",
        train_indices_file=args.train_indices_file,
        validation_indices_file=args.validation_indices_file,
    )
    if runtime.train_arrays.truth or runtime.validation_arrays.truth:
        raise RuntimeError("SC-DRWS report loaded truth")
    model = load_checkpoint(args.checkpoint, config)
    arrays, summary = _prior_report_arrays(
        model,
        runtime,
        key=jax.random.PRNGKey(args.seed),
        n_samples=args.prior_samples,
        selected_resamples=args.selected_resamples,
        decoder_batch_size=args.decoder_batch_size,
    )
    np.savez_compressed(args.out / "parent_and_selected_prior.npz", **arrays)
    rows = []
    for population, values in (
        ("parent", arrays["theta"]),
        ("beta_weighted_selected", arrays["selected_theta"]),
    ):
        for index, name in enumerate(runtime.parameter_names):
            q05, q50, q95 = np.quantile(values[:, index], [0.05, 0.5, 0.95])
            rows.append(
                {
                    "population": population,
                    "parameter": name,
                    "q05": q05,
                    "q50": q50,
                    "q95": q95,
                }
            )
    pd.DataFrame(rows).to_csv(args.out / "population_marginals.csv", index=False)
    np.savez_compressed(
        args.out / "population_correlations.npz",
        parent=np.corrcoef(arrays["theta"], rowvar=False),
        beta_weighted_selected=_weighted_correlation(
            arrays["theta"], arrays["selected_weights"]
        ),
    )
    receipt = {
        "status": "PASS",
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "target_population": "p_eta(theta | C0)",
        "selected_population": "beta(theta) p_eta(theta | C0) / alpha_eta",
        "observed_selection": "A = 1[m_r_observed < 29.0]",
        "upstream_true_space_selection": "conditioned_as_C0_not_inverted",
        "truth_used": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "selection": summary,
        "artifacts": [
            "parent_and_selected_prior.npz",
            "population_marginals.csv",
            "population_correlations.npz",
        ],
    }
    (args.out / "report_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    (args.out / "SC_DRWS_REPORT.md").write_text(
        "# Frozen SC-DRWS no-truth report\n\n"
        f"> {C0_SCOPE_STATEMENT}\n\n"
        f"Selection alpha: `{summary['alpha']:.7g}`; relative MC error: "
        f"`{summary['alpha_mc_relative_error']:.3%}`.\n\n"
        "The selected distribution is derived by beta-weighting the single "
        "learned parent flow. Truth closure is not part of this report.\n"
    )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
