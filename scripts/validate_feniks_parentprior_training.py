#!/usr/bin/env python3
"""Validate the sleep-q / wake-prior training contract and write a receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _finite_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else None


def validate(
    *,
    train: Path,
    manifest: Path,
    smoke: bool,
) -> dict[str, object]:
    summary = json.loads((train / "training_summary.json").read_text())
    contract = json.loads(manifest.read_text())
    history = pd.read_csv(train / "training_log.csv")
    fit = history.loc[history["split"] == "train"].copy()
    sleep = fit.loc[fit["update_phase"] == "encoder_sleep"]
    wake = fit.loc[fit["update_phase"] == "prior_wake"]
    expected_train = int(contract["manifests"]["train"]["count"])
    expected_validation = int(contract["manifests"]["validation"]["count"])
    checks = {
        "checkpoint_exists": (train / "checkpoints/best.eqx").is_file(),
        "manifest_train_rows_match": int(summary["train_rows"]) == expected_train,
        "manifest_validation_rows_match": int(summary["validation_rows"])
        == expected_validation,
        "checkpoint_uses_validation_sleep_nll": summary["best_checkpoint_metric"]
        == "validation_sleep_nll",
        "effective_normalization_is_spline15d_mixed": summary["effective_latent_spec"][
            "normalization"
        ]
        == "spline15d_mixed",
        "selection_correction_enabled": bool(
            summary["selection_correction"].get("enabled", False)
        ),
        "sleep_uses_observed_catalog_errors": summary["sleep"].get("error_model")
        == "observed_catalog",
        "wake_freezes_encoder": summary["wake"].get("train_encoder") is False,
        "wake_trains_prior": summary["wake"].get("train_prior") is True,
        "sleep_rows_exist": not sleep.empty,
        "sleep_prior_gradients_zero": bool(
            sleep.empty
            or np.allclose(
                pd.to_numeric(sleep["prior_raw_grad_norm"], errors="coerce"),
                0.0,
                atol=1.0e-10,
            )
        ),
        "wake_encoder_gradients_zero": bool(
            wake.empty
            or np.allclose(
                pd.to_numeric(wake["encoder_raw_grad_norm"], errors="coerce"),
                0.0,
                atol=1.0e-10,
            )
        ),
        "best_sleep_nll_finite": bool(np.isfinite(float(summary["best_loss"]))),
    }
    if smoke:
        checks["wake_rows_exist"] = not wake.empty
    else:
        checks.update(
            {
                "sleep_burnin_is_24_epochs": bool(
                    wake.empty or int(wake["epoch"].min()) == 25
                ),
                "wake_rows_exist": not wake.empty,
                "wake_selection_alpha_finite": bool(
                    not wake.empty
                    and "selection/alpha" in wake
                    and np.all(
                        np.isfinite(
                            pd.to_numeric(wake["selection/alpha"], errors="coerce")
                        )
                    )
                ),
            }
        )
    encoder_clip_fraction = _finite_mean(sleep, "encoder_grad_clipped_fraction")
    wake_ess_fraction = _finite_mean(wake, "wake_median_ess_fraction")
    wake_updates = int(
        np.sum(
            pd.to_numeric(
                wake.get("wake_prior_update_applied", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0.0)
            > 0.5
        )
    )
    entropy_first = (
        _finite_mean(
            sleep.loc[sleep["epoch"] == sleep["epoch"].min()],
            "posterior_full_entropy_mc",
        )
        if not sleep.empty
        else None
    )
    entropy_last = (
        _finite_mean(
            sleep.loc[sleep["epoch"] == sleep["epoch"].max()],
            "posterior_full_entropy_mc",
        )
        if not sleep.empty
        else None
    )
    warnings = []
    if encoder_clip_fraction is not None and encoder_clip_fraction > 0.5:
        warnings.append(
            "encoder gradients are clipped in more than half of sleep updates"
        )
    if not smoke and wake_updates == 0:
        warnings.append(
            "all defensive wake prior updates were support-gated; exact q validation "
            "is still useful but the parent prior was not updated"
        )
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "contract": (
            "q is optimized only by selected observed-covariate sleep NPE; p_eta is "
            "optimized only by support-gated defensive wake with +log(alpha_eta)"
        ),
        "checks": checks,
        "warnings": warnings,
        "diagnostics": {
            "sleep_updates": int(len(sleep)),
            "wake_batches": int(len(wake)),
            "wake_prior_updates_applied": wake_updates,
            "mean_wake_median_ess_fraction": wake_ess_fraction,
            "encoder_grad_clipped_fraction": encoder_clip_fraction,
            "posterior_full_entropy_mc_first": entropy_first,
            "posterior_full_entropy_mc_last": entropy_last,
            "posterior_full_entropy_mc_delta": (
                None
                if entropy_first is None or entropy_last is None
                else entropy_last - entropy_first
            ),
        },
        "next_action": (
            "RUN_EXACT_POSTERIOR_BENCHMARK"
            if all(checks.values())
            else "STOP_FIX_TRAINING_CONTRACT"
        ),
    }
    (train / "parentprior_training_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    payload = validate(**vars(args))
    print(json.dumps(payload, indent=2), flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
