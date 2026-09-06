#!/usr/bin/env python3
"""Summarize a dense FENIKS q bank without catalogue truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from euclid_dsps.amortized.npe_validation import (
    posterior_support_gate,
    summarize_normalized_residuals,
    summarize_truth_free_joint_bank,
)
from euclid_dsps.amortized.posthoc_calibration import load_posterior_bank
from euclid_dsps.io import ensure_dir, write_json


def _read_residual_frames(source: Path) -> pd.DataFrame:
    monolithic = source / "posterior_predictive_residuals.parquet"
    if monolithic.is_file():
        return pd.read_parquet(monolithic)
    directory = source / "posterior_predictive_residuals"
    paths = tuple(sorted(directory.glob("batch_*.parquet")))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def summarize(*, inference: Path, out: Path) -> dict:
    bank = load_posterior_bank(inference)
    summary, diagnostics = summarize_truth_free_joint_bank(
        bank.frame,
        parameter_names=bank.parameter_names,
        identity_column=bank.identity_column,
    )
    gate = posterior_support_gate(summary)
    residuals = _read_residual_frames(inference)
    residual_summary = (
        summarize_normalized_residuals(residuals) if not residuals.empty else None
    )
    ensure_dir(out)
    diagnostics.to_parquet(out / "object_support_diagnostics.parquet", index=False)
    payload = {
        "status": "COMPLETE",
        "truth_used": False,
        "posterior_bank": str(inference.resolve()),
        "support": summary,
        "technical_gate": gate,
        "photometric_residuals": residual_summary,
        "q_direct_and_importance_are_reported_separately": True,
        "scientific_promotion": False,
        "limitations": [
            "importance support does not by itself validate posterior calibration",
            "catalogue truth was intentionally unavailable to this validator",
            "held-out-band and model-generated reference runs are separate required artifacts",
        ],
        "artifacts": {
            "object_support": str(
                (out / "object_support_diagnostics.parquet").resolve()
            )
        },
    }
    write_json(out / "TRUTH_FREE_POSTERIOR_VALIDATION.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
