#!/usr/bin/env python3
"""Fit the production 15D FENIKS spline-SFH prior with RealNVP only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.io import ensure_dir, write_json
from euclid_dsps.prior_learning.diagnostics import (
    write_supervised_prior_diagnostics,
)
from euclid_dsps.prior_learning.flows import RealNVPPrior, assert_flow_integrity
from euclid_dsps.prior_learning.spline15d import (
    SPLINE15D_PARAMETER_NAMES,
    fit_asinh_transforms,
    forward_asinh_matrix,
    gaussian_quantile_rmse,
    inverse_asinh_matrix,
)
from euclid_dsps.prior_learning.spline15d_checkpoint import (
    load_spline15d_realnvp_checkpoint,
)
from euclid_dsps.prior_learning.spline15d_evaluation import (
    evaluate_generated_prior,
    evaluate_sample_pair,
    novel_truth_mask,
    plot_truth_prior_physical_normalized,
    select_temperature,
    selection_payload,
    temperature_scan_frame,
)
from euclid_dsps.prior_learning.train import (
    _prior_architecture_payload,
    fit_realnvp_to_x,
)

eqx, _optax = require_amortized_dependencies()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--prior-samples", type=int)
    parser.add_argument("--evaluation-every", type=int)
    parser.add_argument("--evaluation-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-parallel", choices=("single", "auto", "pmap"))
    return parser.parse_args()


def _read_split(path: Path, limit: int | None) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["object_id", *SPLINE15D_PARAMETER_NAMES])
    if limit is not None:
        frame = frame.head(max(int(limit), 0))
    matrix = frame.loc[:, SPLINE15D_PARAMETER_NAMES].to_numpy(dtype=np.float64)
    if len(frame) == 0 or not np.isfinite(matrix).all():
        raise ValueError(f"Split is empty or non-finite: {path}")
    return frame


def _read_exact_split(path: Path, limit: int | None) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if limit is not None:
        frame = frame.head(max(int(limit), 0))
    missing = set(SPLINE15D_PARAMETER_NAMES).difference(frame.columns)
    if missing:
        raise ValueError(f"Exact split is missing columns {sorted(missing)}: {path}")
    return frame


def _flow_nll(prior: RealNVPPrior, matrix: np.ndarray) -> float:
    values = prior.log_prob(jnp.asarray(matrix, dtype=jnp.float32))
    return -float(np.mean(np.asarray(jax.device_get(values))))


def _tempered_flow_nll(
    prior: RealNVPPrior,
    matrix: np.ndarray,
    temperature: float,
) -> float:
    values = prior.log_prob_with_temperature(
        jnp.asarray(matrix, dtype=jnp.float32),
        temperature=float(temperature),
    )
    return -float(np.mean(np.asarray(jax.device_get(values))))


def _jsonable_data_parallel(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: ([str(device) for device in value] if key == "devices" else value)
        for key, value in payload.items()
    }


def _progress(out: Path, total_epochs: int):
    def callback(record: dict[str, Any]) -> None:
        validation = record.get("validation_negative_mean_log_prob")
        validation_text = "nan" if validation is None else f"{validation:.6g}"
        print(
            f"[spline15d-prior] epoch={record['epoch']}/{total_epochs} "
            f"train_nll={record['train_negative_mean_log_prob']:.6g} "
            f"validation_nll={validation_text} "
            f"best={record['best_metric']:.6g}@{record['best_epoch']}",
            flush=True,
        )
        write_json(
            out / "training_progress.json",
            {"status": "training", "total_epochs": total_epochs, **record},
        )

    return callback


def _save_checkpoint(
    path: Path,
    prior: RealNVPPrior,
    *,
    flow_config: dict[str, Any],
    normalization: dict[str, Any],
    dataset_dir: Path,
    epoch: int,
    metric: float,
    selection: dict[str, Any] | None = None,
) -> None:
    if not isinstance(prior, RealNVPPrior):
        raise TypeError("The spline15d production path only accepts RealNVPPrior")
    integrity = assert_flow_integrity(
        prior,
        context=f"spline15d RealNVP checkpoint {path}",
        sample_count=64,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, prior)
    write_json(
        path.with_suffix(path.suffix + ".json"),
        {
            "version": 1,
            "epoch": epoch,
            "metric": metric,
            "architecture": _prior_architecture_payload(prior, flow_config),
            "parameter_names": SPLINE15D_PARAMETER_NAMES,
            "normalization": normalization,
            "source_dataset": str(dataset_dir),
            "flow_integrity": integrity,
            "selection": selection or {},
        },
    )


def _plot_normalization(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    train_x: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    transforms: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(len(SPLINE15D_PARAMETER_NAMES), 2, figsize=(13, 42))
    grid = np.linspace(-5.0, 5.0, 600)
    gaussian = np.exp(-0.5 * grid**2) / np.sqrt(2.0 * np.pi)
    for index, name in enumerate(SPLINE15D_PARAMETER_NAMES):
        left, right = axes[index]
        left.hist(train[:, index], bins=70, density=True, alpha=0.65, label="train")
        left.hist(test[:, index], bins=70, density=True, histtype="step", label="test")
        left.set_title(f"{name} - before", fontsize=9)
        right.hist(train_x[:, index], bins=70, density=True, alpha=0.55, label="train")
        right.hist(
            validation_x[:, index], bins=70, density=True, histtype="step", label="val"
        )
        right.hist(test_x[:, index], bins=70, density=True, histtype="step", label="test")
        right.plot(grid, gaussian, color="black", lw=1.2, label="N(0,1)")
        transform = transforms[name]
        right.set_title(
            f"after asinh | lambda={transform['lambda']:.4g} | "
            f"test QRMSE={gaussian_quantile_rmse(test_x[:, index]):.3f}",
            fontsize=9,
        )
        right.set_xlim(-6.0, 6.0)
        if index == 0:
            left.legend(fontsize=8)
            right.legend(fontsize=8, ncol=2)
    fig.suptitle("Spline-15D marginals before/after train-fitted asinh", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_history(training: pd.DataFrame, validation: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    train_epoch = training.groupby("epoch", as_index=False)[
        ["negative_mean_log_prob", "loss"]
    ].mean()
    axis.plot(
        train_epoch["epoch"],
        train_epoch["negative_mean_log_prob"],
        label="train NLL",
    )
    if not np.allclose(
        train_epoch["negative_mean_log_prob"], train_epoch["loss"]
    ):
        axis.plot(
            train_epoch["epoch"],
            train_epoch["loss"],
            linestyle="--",
            label="train objective",
        )
    if not validation.empty:
        axis.plot(
            validation["epoch"], validation["negative_mean_log_prob"],
            label="validation NLL",
        )
    axis.set_xlabel("epoch")
    axis.set_ylabel("negative mean log probability")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = dict(config.get("spline15d_prior", {}) or {})
    flow_cfg = dict(cfg.get("flow", {}) or {})
    if str(flow_cfg.get("type", "realnvp")).lower() != "realnvp":
        raise ValueError("This production command supports RealNVP only")
    training_cfg = dict(cfg.get("training", {}) or {})
    output_cfg = dict(cfg.get("output", {}) or {})
    evaluation_cfg = dict(cfg.get("evaluation", {}) or {})
    norm_cfg = dict(cfg.get("normalization", {}) or {})
    if str(norm_cfg.get("family", "asinh")).lower() != "asinh":
        raise ValueError("This production command supports asinh normalization only")
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "data_parallel": args.data_parallel,
        "seed": args.seed,
    }
    training_cfg.update({key: value for key, value in overrides.items() if value is not None})
    if args.n_layers is not None:
        flow_cfg["n_layers"] = args.n_layers
    if args.hidden_size is not None:
        flow_cfg["hidden_size"] = args.hidden_size
    if args.prior_samples is not None:
        output_cfg["prior_samples"] = args.prior_samples
    if args.evaluation_every is not None:
        evaluation_cfg["every_epochs"] = args.evaluation_every
    if args.evaluation_samples is not None:
        evaluation_cfg["sample_count"] = args.evaluation_samples

    dataset_dir = Path(args.dataset_dir or cfg["dataset_dir"])
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    ensure_dir(out)
    started = time.time()
    write_json(out / "resolved_config.json", {"spline15d_prior": {
        "dataset_dir": str(dataset_dir), "normalization": norm_cfg,
        "flow": flow_cfg, "training": training_cfg, "evaluation": evaluation_cfg,
        "output": output_cfg,
    }})
    contract_path = dataset_dir / "spline15d_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError(f"Missing projection contract: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if tuple(contract["parameter_names"]) != SPLINE15D_PARAMETER_NAMES:
        raise ValueError("Dataset parameter order does not match spline15d contract")

    frames = {
        split: _read_split(dataset_dir / f"{split}.parquet", args.limit)
        for split in ("train", "validation", "test")
    }
    exact_frames = {
        split: _read_exact_split(dataset_dir / f"{split}_exact.parquet", args.limit)
        for split in ("train", "validation", "test")
    }
    novel_masks = {
        split: novel_truth_mask(exact_frames["train"], exact_frames[split])
        for split in ("validation", "test")
    }
    overlap_audit = {
        split: {
            "rows": int(len(exact_frames[split])),
            "unique_exact_truth_rows": int(
                len(exact_frames[split].drop_duplicates(list(SPLINE15D_PARAMETER_NAMES)))
            ),
            "novel_vs_train_rows": (
                None if split == "train" else int(np.sum(novel_masks[split]))
            ),
            "exact_train_overlap_rows": (
                None
                if split == "train"
                else int(len(exact_frames[split]) - np.sum(novel_masks[split]))
            ),
        }
        for split in ("train", "validation", "test")
    }
    write_json(out / "dataset_overlap_audit.json", overlap_audit)
    matrices = {
        split: frame.loc[:, SPLINE15D_PARAMETER_NAMES].to_numpy(dtype=np.float64)
        for split, frame in frames.items()
    }
    transforms = fit_asinh_transforms(
        matrices["train"],
        log10_lambda_min=float(norm_cfg.get("log10_lambda_min", -8.0)),
        log10_lambda_max=float(norm_cfg.get("log10_lambda_max", 8.0)),
        grid_size=int(norm_cfg.get("grid_size", 257)),
    )
    normalized = {
        split: forward_asinh_matrix(matrix, transforms)
        for split, matrix in matrices.items()
    }
    roundtrip = inverse_asinh_matrix(normalized["test"], transforms)
    norm_payload = {
        "version": 1,
        "family": "asinh",
        "fit_split": "train",
        "fit_rows": len(matrices["train"]),
        "parameter_names": SPLINE15D_PARAMETER_NAMES,
        "formula": "x_norm=(lambda*asinh(x/lambda)-center)/scale",
        "inverse_formula": "x=lambda*sinh((center+scale*x_norm)/lambda)",
        "transforms": transforms,
        "test_roundtrip_max_abs": float(np.max(np.abs(roundtrip - matrices["test"]))),
        "gaussian_qrmse": {
            split: {
                name: gaussian_quantile_rmse(normalized[split][:, index])
                for index, name in enumerate(SPLINE15D_PARAMETER_NAMES)
            }
            for split in normalized
        },
    }
    write_json(out / "normalization.json", norm_payload)
    pd.DataFrame.from_dict(transforms, orient="index").rename_axis("parameter").to_csv(
        out / "normalization_parameters.csv"
    )
    _plot_normalization(
        matrices["train"], matrices["validation"], matrices["test"],
        normalized["train"], normalized["validation"], normalized["test"],
        transforms, out / "normalization_before_after.png",
    )

    seed = int(training_cfg.get("seed", 260715))
    epochs = int(training_cfg.get("epochs", 120))
    evaluation_every = max(int(evaluation_cfg.get("every_epochs", 5)), 1)
    evaluation_samples = max(int(evaluation_cfg.get("sample_count", 10000)), 256)
    thresholds = {
        "max_median_ks": float(evaluation_cfg.get("max_median_ks", 0.10)),
        "max_max_ks": float(evaluation_cfg.get("max_max_ks", 0.20)),
        "max_correlation_frobenius": float(
            evaluation_cfg.get("max_correlation_frobenius", 1.5)
        ),
        "min_base_std_mean": float(evaluation_cfg.get("min_base_std_mean", 0.8)),
        "max_base_std_mean": float(evaluation_cfg.get("max_base_std_mean", 1.2)),
        "max_normalized_tail_fraction": float(
            evaluation_cfg.get("max_normalized_tail_fraction", 0.002)
        ),
        "max_negative_fraction": float(
            evaluation_cfg.get("max_negative_fraction", 0.001)
        ),
    }
    validation_generative_rows: list[dict[str, Any]] = []

    def _select_checkpoint(
        epoch: int,
        prior: RealNVPPrior,
        validation_nll: float,
    ) -> dict[str, Any] | None:
        if epoch != 1 and epoch != epochs and epoch % evaluation_every:
            return None
        metrics, _prior_theta, _prior_x = evaluate_generated_prior(
            prior,
            truth_theta=matrices["validation"],
            truth_x=normalized["validation"],
            transforms=transforms,
            sample_count=evaluation_samples,
            seed=seed + 10_000,
        )
        payload = selection_payload(metrics, thresholds=thresholds)
        novel = novel_masks["validation"]
        payload["validation_nll"] = float(validation_nll)
        payload["validation_novel_nll"] = (
            _flow_nll(prior, normalized["validation"][novel])
            if np.any(novel)
            else float("nan")
        )
        validation_generative_rows.append({"epoch": int(epoch), **payload})
        return payload

    result = fit_realnvp_to_x(
        normalized["train"], normalized["validation"],
        latent_dim=len(SPLINE15D_PARAMETER_NAMES), flow_config=flow_cfg,
        training_config=training_cfg, seed=seed,
        progress_callback=_progress(out, epochs),
        selection_callback=_select_checkpoint,
    )
    if not isinstance(result.prior, RealNVPPrior):
        raise TypeError("Training returned a non-RealNVP model")
    result.training_log.to_csv(out / "training_log.csv", index=False)
    result.validation_log.to_csv(out / "validation_log.csv", index=False)
    pd.DataFrame(validation_generative_rows).to_csv(
        out / "validation_generative_history.csv", index=False
    )
    _plot_history(result.training_log, result.validation_log, out / "training_history.png")
    checkpoints = ensure_dir(out / "checkpoints")
    _save_checkpoint(
        checkpoints / "best.eqx", result.prior, flow_config=flow_cfg,
        normalization=norm_payload, dataset_dir=dataset_dir,
        epoch=result.best_epoch, metric=result.best_metric,
        selection={
            "eligible": result.best_selection_eligible,
            **result.best_selection_diagnostics,
        },
    )
    _save_checkpoint(
        checkpoints / "last.eqx", result.last_prior, flow_config=flow_cfg,
        normalization=norm_payload, dataset_dir=dataset_dir,
        epoch=epochs, metric=_flow_nll(result.last_prior, normalized["validation"]),
    )

    n_samples = int(output_cfg.get("prior_samples", 50000))
    loaded_prior, _checkpoint_metadata = load_spline15d_realnvp_checkpoint(
        checkpoints / "best.eqx"
    )
    prior_x = np.asarray(
        jax.device_get(loaded_prior.sample(jax.random.PRNGKey(seed + 1), n_samples)),
        dtype=np.float64,
    )
    prior_theta = inverse_asinh_matrix(prior_x, transforms)
    prior_x_frame = pd.DataFrame(prior_x, columns=SPLINE15D_PARAMETER_NAMES)
    prior_theta_frame = pd.DataFrame(prior_theta, columns=SPLINE15D_PARAMETER_NAMES)
    prior_x_frame.to_parquet(out / "learned_prior_samples_normalized.parquet", index=False)
    prior_theta_frame.to_parquet(out / "learned_prior_samples.parquet", index=False)
    truth_limit = int(output_cfg.get("truth_sample_limit", len(frames["test"])))
    truth_frame = frames["test"].loc[:, SPLINE15D_PARAMETER_NAMES].head(truth_limit)
    truth_frame.to_parquet(out / "heldout_test_truth.parquet", index=False)
    truth_theta = truth_frame.to_numpy(dtype=float)
    truth_x = normalized["test"][: len(truth_frame)]
    plot_truth_prior_physical_normalized(
        truth_theta=truth_theta,
        prior_theta=prior_theta,
        truth_x=truth_x,
        prior_x=prior_x,
        path=str(out / "learned_prior_vs_truth.png"),
        title_suffix="unit-temperature base",
    )
    nll = {split: _flow_nll(result.prior, value) for split, value in normalized.items()}
    test_unit_metrics, _test_unit_theta, _test_unit_x = evaluate_generated_prior(
        loaded_prior,
        truth_theta=matrices["test"],
        truth_x=normalized["test"],
        transforms=transforms,
        sample_count=n_samples,
        seed=seed + 1,
    )

    temperature_min = float(evaluation_cfg.get("temperature_min", 0.08))
    temperature_max = float(evaluation_cfg.get("temperature_max", 1.0))
    temperature_step = float(evaluation_cfg.get("temperature_step", 0.02))
    if temperature_step <= 0.0 or temperature_max < temperature_min:
        raise ValueError("Invalid validation temperature scan range")
    temperatures = np.arange(
        temperature_min,
        temperature_max + 0.5 * temperature_step,
        temperature_step,
    )
    if bool(evaluation_cfg.get("include_unit_temperature", True)):
        temperatures = np.unique(np.append(temperatures, 1.0))
    temperature_scan = temperature_scan_frame(
        loaded_prior,
        truth_theta=matrices["validation"],
        truth_x=normalized["validation"],
        transforms=transforms,
        temperatures=temperatures,
        sample_count=evaluation_samples,
        seed=seed + 20_000,
        thresholds=thresholds,
    )
    temperature_scan.to_csv(out / "validation_temperature_scan.csv", index=False)
    selected_temperature_row = select_temperature(temperature_scan)
    selected_temperature = float(selected_temperature_row["base_temperature"])
    temperature_calibration = {
        "fit_split": "validation",
        "selected_base_temperature": selected_temperature,
        "selected_eligible": bool(selected_temperature_row["eligible"]),
        "selected_metric": float(selected_temperature_row["metric"]),
        "thresholds": thresholds,
        "sample_count_per_temperature": evaluation_samples,
        "seed": seed + 20_000,
    }
    write_json(out / "temperature_calibration.json", temperature_calibration)
    best_sidecar_path = checkpoints / "best.eqx.json"
    best_sidecar = json.loads(best_sidecar_path.read_text(encoding="utf-8"))
    best_sidecar["recommended_base_temperature"] = selected_temperature
    best_sidecar["temperature_calibration"] = temperature_calibration
    write_json(best_sidecar_path, best_sidecar)
    calibrated_x = np.asarray(
        jax.device_get(
            loaded_prior.sample_with_temperature(
                jax.random.PRNGKey(seed + 2),
                n_samples,
                temperature=selected_temperature,
            )
        ),
        dtype=np.float64,
    )
    calibrated_theta = inverse_asinh_matrix(calibrated_x, transforms)
    calibrated_x_frame = pd.DataFrame(
        calibrated_x, columns=SPLINE15D_PARAMETER_NAMES
    )
    calibrated_theta_frame = pd.DataFrame(
        calibrated_theta, columns=SPLINE15D_PARAMETER_NAMES
    )
    calibrated_x_frame.to_parquet(
        out / "learned_prior_samples_temperature_calibrated_normalized.parquet",
        index=False,
    )
    calibrated_theta_frame.to_parquet(
        out / "learned_prior_samples_temperature_calibrated.parquet", index=False
    )
    plot_truth_prior_physical_normalized(
        truth_theta=truth_theta,
        prior_theta=calibrated_theta,
        truth_x=truth_x,
        prior_x=calibrated_x,
        path=str(out / "learned_prior_vs_truth_temperature_calibrated.png"),
        title_suffix=f"validation-calibrated base T={selected_temperature:.3g}",
    )
    test_calibrated_metrics, _calibrated_theta, _calibrated_x = (
        evaluate_generated_prior(
            loaded_prior,
            truth_theta=matrices["test"],
            truth_x=normalized["test"],
            transforms=transforms,
            sample_count=n_samples,
            seed=seed + 2,
            temperature=selected_temperature,
        )
    )

    baseline_rng = np.random.default_rng(seed + 3)
    baseline_x = baseline_rng.normal(size=(n_samples, len(SPLINE15D_PARAMETER_NAMES)))
    baseline_theta = inverse_asinh_matrix(baseline_x, transforms)
    baseline_metrics = evaluate_sample_pair(
        truth_theta=matrices["test"],
        truth_x=normalized["test"],
        prior_theta=baseline_theta,
        prior_x=baseline_x,
    )
    comparison_rows = []
    for model_name, temperature, metrics in (
        ("independent_normal", 1.0, baseline_metrics),
        ("realnvp_unit_temperature", 1.0, test_unit_metrics),
        ("realnvp_validation_calibrated", selected_temperature, test_calibrated_metrics),
    ):
        comparison_rows.append(
            {"model": model_name, "base_temperature": temperature, **metrics}
        )
    pd.DataFrame(comparison_rows).to_csv(out / "test_baseline_comparison.csv", index=False)

    novel_nll = {
        split: (
            _flow_nll(loaded_prior, normalized[split][novel_masks[split]])
            if np.any(novel_masks[split])
            else float("nan")
        )
        for split in ("validation", "test")
    }
    calibrated_nll = {
        split: _tempered_flow_nll(loaded_prior, value, selected_temperature)
        for split, value in normalized.items()
    }
    best_validation_row = result.validation_log.loc[
        result.validation_log["epoch"] == result.best_epoch
    ]
    best_validation_nll = float(
        best_validation_row.iloc[-1]["negative_mean_log_prob"]
    )
    diagnostics = write_supervised_prior_diagnostics(
        truth=truth_frame, prior=prior_theta_frame,
        parameter_names=SPLINE15D_PARAMETER_NAMES, out_dir=out / "diagnostics",
        summary={
            "model": "RealNVP", "latent_dim": 15,
            "normalization": "train-fitted marginal asinh", "nll": nll,
            "best_epoch": result.best_epoch,
            "best_validation_nll": best_validation_nll,
            "base_temperature": 1.0,
        },
        max_corner_rows=int(output_cfg.get("max_corner_rows", 4000)),
    )
    calibrated_diagnostics = write_supervised_prior_diagnostics(
        truth=truth_frame,
        prior=calibrated_theta_frame,
        parameter_names=SPLINE15D_PARAMETER_NAMES,
        out_dir=out / "diagnostics_temperature_calibrated",
        summary={
            "model": "RealNVP",
            "latent_dim": 15,
            "normalization": "train-fitted marginal asinh",
            "nll": calibrated_nll,
            "best_epoch": result.best_epoch,
            "base_temperature": selected_temperature,
            "temperature_fit_split": "validation",
        },
        max_corner_rows=int(output_cfg.get("max_corner_rows", 4000)),
    )
    summary = {
        "status": "complete", "model": "RealNVP", "latent_dim": 15,
        "dataset_dir": str(dataset_dir), "rows": {key: len(value) for key, value in frames.items()},
        "normalization_path": str(out / "normalization.json"), "nll": nll,
        "initial_train_nll": result.initial_train_nll,
        "best_epoch": result.best_epoch,
        "best_validation_nll": best_validation_nll,
        "best_selection_metric": result.best_metric,
        "best_selection_eligible": result.best_selection_eligible,
        "best_selection_diagnostics": result.best_selection_diagnostics,
        "validation_novel_nll": novel_nll["validation"],
        "test_novel_nll": novel_nll["test"],
        "test_unit_temperature_metrics": test_unit_metrics,
        "temperature_calibration": temperature_calibration,
        "test_temperature_calibrated_metrics": test_calibrated_metrics,
        "test_baseline_metrics": baseline_metrics,
        "calibrated_nll": calibrated_nll,
        "prior_samples": n_samples,
        "data_parallel": _jsonable_data_parallel(result.data_parallel),
        "diagnostics": diagnostics,
        "calibrated_diagnostics": calibrated_diagnostics,
        "dataset_overlap_audit": overlap_audit,
        "elapsed_seconds": time.time() - started,
    }
    write_json(out / "run_summary.json", summary)
    write_json(out / "training_progress.json", summary)
    print(f"[spline15d-prior] complete: {out}", flush=True)


if __name__ == "__main__":
    main()
