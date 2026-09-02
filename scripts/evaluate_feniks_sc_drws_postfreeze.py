#!/usr/bin/env python3
"""Run held-out posterior and population closure after SC-DRWS is frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.amortized.catalog_identity import write_truth_snapshot
from euclid_dsps.amortized.mira import (
    FENIKS_SPLINE15D_PARAMETERS,
    evaluate_feniks_mira,
)
from euclid_dsps.amortized.tarp import evaluate_feniks_tarp
from euclid_dsps.config import load_config
from euclid_dsps.io import truth_column_from_spec
from euclid_dsps.photometry import abmag_to_fnu_cgs


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _truth_matrix(
    frame: pd.DataFrame,
    mappings: dict[str, Any],
    parameters: tuple[str, ...],
) -> np.ndarray:
    columns = [truth_column_from_spec(mappings[name]) for name in parameters]
    if any(column is None for column in columns):
        raise ValueError("truth closure mappings must name physical parquet columns")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"truth closure is missing columns: {missing}")
    return frame.loc[:, columns].to_numpy(dtype=np.float64)


def _population_rows(
    *,
    truth: np.ndarray,
    model: np.ndarray,
    parameters: tuple[str, ...],
    variant: str,
    population: str,
) -> list[dict[str, Any]]:
    probabilities = np.linspace(0.0, 1.0, 1001)
    truth_quantiles = np.quantile(truth, probabilities, axis=0)
    model_quantiles = np.quantile(model, probabilities, axis=0)
    truth_std = np.std(truth, axis=0)
    truth_iqr = truth_quantiles[750] - truth_quantiles[250]
    rows = []
    for index, name in enumerate(parameters):
        wasserstein = float(
            np.mean(
                np.abs(model_quantiles[:, index] - truth_quantiles[:, index])
            )
        )
        rows.append(
            {
                "variant": variant,
                "population": population,
                "parameter": name,
                "truth_objects": int(len(truth)),
                "model_samples": int(len(model)),
                "wasserstein_1d": wasserstein,
                "wasserstein_over_truth_iqr": float(
                    wasserstein / max(float(truth_iqr[index]), 1.0e-12)
                ),
                "mean_difference_over_truth_std": float(
                    (np.mean(model[:, index]) - np.mean(truth[:, index]))
                    / max(float(truth_std[index]), 1.0e-12)
                ),
                "std_ratio": float(
                    np.std(model[:, index])
                    / max(float(truth_std[index]), 1.0e-12)
                ),
                "q05_difference": float(
                    model_quantiles[50, index] - truth_quantiles[50, index]
                ),
                "q50_difference": float(
                    model_quantiles[500, index] - truth_quantiles[500, index]
                ),
                "q95_difference": float(
                    model_quantiles[950, index] - truth_quantiles[950, index]
                ),
            }
        )
    return rows


def _correlation_row(
    *,
    truth: np.ndarray,
    model: np.ndarray,
    variant: str,
    population: str,
) -> dict[str, Any]:
    truth_correlation = np.corrcoef(truth, rowvar=False)
    model_correlation = np.corrcoef(model, rowvar=False)
    upper = np.triu_indices(truth.shape[1], k=1)
    delta = np.abs(model_correlation[upper] - truth_correlation[upper])
    return {
        "variant": variant,
        "population": population,
        "mean_absolute_correlation_error": float(np.mean(delta)),
        "maximum_absolute_correlation_error": float(np.max(delta)),
    }


def run(
    *,
    training_config_path: Path,
    truth_config_path: Path,
    test_catalog: Path,
    manifest_path: Path,
    run_root: Path,
    out_dir: Path,
    samples_per_object: int,
    num_regions: int,
    num_bootstrap: int,
) -> dict[str, Any]:
    if out_dir.joinpath("POSTFREEZE_COMPLETE.json").exists():
        return _read_json(out_dir / "POSTFREEZE_COMPLETE.json")
    training_config = load_config(training_config_path)
    truth_config = load_config(truth_config_path)
    if (training_config.get("truth", {}) or {}).get("parameter_columns"):
        raise ValueError("training config passed to closure contains truth mappings")
    parameters = tuple(FENIKS_SPLINE15D_PARAMETERS)
    mappings = dict(
        (truth_config.get("truth", {}) or {}).get("parameter_columns") or {}
    )
    if set(mappings) != set(parameters):
        raise ValueError("truth config must map all and only spline15d parameters")
    training = _read_json(run_root / "train" / "training_receipt.json")
    if (
        training.get("truth_used_for_training_validation_or_checkpoint_selection")
        is not False
    ):
        raise ValueError("training receipt does not prove the no-truth contract")
    full_summary = _read_json(run_root / "full_summary.json")
    if full_summary.get("truth_used_for_training_or_checkpoint_selection") is not False:
        raise ValueError("full summary does not prove truth-free checkpoint selection")
    manifest = _read_json(manifest_path)
    catalog_record = manifest["catalogs"]["test"]
    if _sha256(test_catalog) != catalog_record["sha256"]:
        raise ValueError("test catalogue differs from the frozen manifest")

    out_dir.mkdir(parents=True, exist_ok=True)
    truth_config["catalog_path"] = str(test_catalog.resolve())
    population_truth_path = out_dir / "population_truth_C0.parquet"
    if population_truth_path.exists():
        population_truth = pd.read_parquet(population_truth_path)
    else:
        population_truth = write_truth_snapshot(
            out_dir,
            truth_config,
            row_indices=None,
            limit=None,
            filename=population_truth_path.name,
        )
    final_rows = np.load(
        manifest_path.parent / "final_validation_indices.npy", allow_pickle=False
    ).astype(np.int64)
    row_lookup = population_truth.set_index("row_index", drop=False)
    missing_rows = sorted(set(final_rows.tolist()) - set(row_lookup.index))
    if missing_rows:
        raise ValueError(f"final truth rows are absent from test catalogue: {missing_rows[:5]}")
    inference_truth = row_lookup.loc[final_rows].reset_index(drop=True)
    inference_truth_path = out_dir / "inference_truth.parquet"
    inference_truth.to_parquet(inference_truth_path, index=False)

    posterior_specs = (
        ("raw_q", run_root / "full_raw_exact_gaussian_k2048"),
        (
            "raw_iw",
            run_root / "full_raw_exact_gaussian_iw" / "resampled_samples",
        ),
        ("ema_q", run_root / "full_ema_exact_gaussian_k2048"),
        (
            "ema_iw",
            run_root / "full_ema_exact_gaussian_iw" / "resampled_samples",
        ),
    )
    mira_dir = out_dir / "mira"
    if not (mira_dir / "DONE").is_file():
        evaluate_feniks_mira(
            truth_path=inference_truth_path,
            posterior_specs=posterior_specs,
            out_dir=mira_dir,
            num_regions=int(num_regions),
            num_bootstrap=int(num_bootstrap),
            samples_per_object=int(samples_per_object),
            seed=260902,
            parameters=parameters,
            drop_nonfinite_truth=True,
        )
    tarp_dir = out_dir / "tarp"
    if not (tarp_dir / "DONE").is_file():
        evaluate_feniks_tarp(
            truth_path=inference_truth_path,
            posterior_specs=posterior_specs,
            out_dir=tarp_dir,
            num_bootstrap=int(num_bootstrap),
            samples_per_object=int(samples_per_object),
            seed=260903,
            parameters=parameters,
            drop_nonfinite_truth=True,
        )

    population_matrix = _truth_matrix(population_truth, mappings, parameters)
    finite = np.all(np.isfinite(population_matrix), axis=1)
    population_matrix = population_matrix[finite]
    population_rows = population_truth.loc[finite, "row_index"].to_numpy(
        dtype=np.int64
    )
    flux = pd.read_parquet(test_catalog, columns=["flux_lsst_r"])[
        "flux_lsst_r"
    ].to_numpy(dtype=np.float64)
    selected = flux[population_rows] > float(np.asarray(abmag_to_fnu_cgs(29.0)))
    selected_truth = population_matrix[selected]
    recovery_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    prior_receipts = {}
    for variant in ("raw", "ema"):
        prior_dir = out_dir / f"prior_{variant}"
        prior_receipt = _read_json(prior_dir / "report_receipt.json")
        if prior_receipt.get("truth_used") is not False:
            raise ValueError(f"{variant} prior report used truth")
        prior_receipts[variant] = prior_receipt
        with np.load(prior_dir / "parent_and_selected_prior.npz", allow_pickle=False) as arrays:
            distributions = {
                "parent_C0": (population_matrix, np.asarray(arrays["theta"])),
                "observed_selected": (
                    selected_truth,
                    np.asarray(arrays["selected_theta"]),
                ),
            }
        for population, (truth_values, model_values) in distributions.items():
            recovery_rows.extend(
                _population_rows(
                    truth=truth_values,
                    model=model_values,
                    parameters=parameters,
                    variant=variant,
                    population=population,
                )
            )
            correlation_rows.append(
                _correlation_row(
                    truth=truth_values,
                    model=model_values,
                    variant=variant,
                    population=population,
                )
            )
    recovery_path = out_dir / "population_recovery.csv"
    correlation_path = out_dir / "population_correlation_recovery.csv"
    pd.DataFrame(recovery_rows).to_csv(recovery_path, index=False)
    pd.DataFrame(correlation_rows).to_csv(correlation_path, index=False)

    receipt = {
        "status": "DIAGNOSTIC_COMPLETE",
        "phase": "postfreeze_truth_closure",
        "scientific_promotion": False,
        "training_frozen_before_truth": True,
        "truth_used": True,
        "truth_used_for_training_or_checkpoint_selection": False,
        "training_status": full_summary["status"],
        "posterior_models": [name for name, _ in posterior_specs],
        "final_validation_objects": int(len(inference_truth)),
        "population_truth_C0_objects": int(len(population_matrix)),
        "population_truth_selected_objects": int(len(selected_truth)),
        "prior_selection_alpha": {
            name: value["selection"]["alpha"]
            for name, value in prior_receipts.items()
        },
        "contracts": {
            "posterior": "dense joint draws only; no pointwise distribution replacement",
            "parent": "p_eta(theta | C0)",
            "selected": "beta(theta) p_eta(theta | C0) / alpha_eta",
            "truth_role": "post-freeze closure diagnostics only",
        },
        "artifacts": {
            "inference_truth": _file_record(inference_truth_path),
            "population_truth_C0": _file_record(population_truth_path),
            "population_recovery": _file_record(recovery_path),
            "population_correlation_recovery": _file_record(correlation_path),
            "mira_summary": _file_record(mira_dir / "mira_summary.json"),
            "tarp_summary": _file_record(tarp_dir / "tarp_summary.json"),
        },
    }
    completion = out_dir / "POSTFREEZE_COMPLETE.json"
    completion.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--truth-config", type=Path, required=True)
    parser.add_argument("--test-catalog", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-object", type=int, default=128)
    parser.add_argument("--num-regions", type=int, default=100)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    result = run(
        training_config_path=args.training_config,
        truth_config_path=args.truth_config,
        test_catalog=args.test_catalog,
        manifest_path=args.manifest,
        run_root=args.run_root,
        out_dir=args.out,
        samples_per_object=args.samples_per_object,
        num_regions=args.num_regions,
        num_bootstrap=args.num_bootstrap,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
