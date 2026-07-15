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
    dequantize_normalized_zero_atoms,
    fit_affine_whitening,
    fit_asinh_transforms,
    fit_shifted_asinh_transforms,
    forward_affine_whitening,
    forward_asinh_matrix,
    gaussian_quantile_rmse,
    inverse_affine_whitening,
    inverse_asinh_arguments_matrix,
    inverse_asinh_matrix,
    inverse_spline15d_flow_coordinates,
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
    parser.add_argument("--snapshot-samples", type=int)
    parser.add_argument("--atom-half-width", type=float)
    parser.add_argument("--whitening", choices=("true", "false"))
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
    *,
    whitening_enabled: bool,
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
        right.hist(
            test_x[:, index], bins=70, density=True, histtype="step", label="test"
        )
        right.plot(grid, gaussian, color="black", lw=1.2, label="N(0,1)")
        transform = transforms[name]
        family = str(transform.get("family", "asinh"))
        stage = (
            f"after {family} + whitening" if whitening_enabled else f"after {family}"
        )
        location_text = (
            f" | location={transform['location']:.4g}"
            if family == "shifted_asinh"
            else ""
        )
        right.set_title(
            f"{stage} | lambda={transform['lambda']:.4g}{location_text} | "
            f"test QRMSE={gaussian_quantile_rmse(test_x[:, index]):.3f}",
            fontsize=9,
        )
        right.set_xlim(-6.0, 6.0)
        if index == 0:
            left.legend(fontsize=8)
            right.legend(fontsize=8, ncol=2)
    family = str(next(iter(transforms.values())).get("family", "asinh"))
    suffix = " + Cholesky whitening" if whitening_enabled else ""
    fig.suptitle(
        f"Spline-15D marginals before/after train-fitted {family}{suffix}",
        fontsize=14,
    )
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
    if not np.allclose(train_epoch["negative_mean_log_prob"], train_epoch["loss"]):
        axis.plot(
            train_epoch["epoch"],
            train_epoch["loss"],
            linestyle="--",
            label="train objective",
        )
    if not validation.empty:
        axis.plot(
            validation["epoch"],
            validation["negative_mean_log_prob"],
            label="validation NLL",
        )
    axis.set_xlabel("epoch")
    axis.set_ylabel("negative mean log probability")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _marginal_snapshot_frame(
    truth: np.ndarray,
    prior: np.ndarray,
    *,
    space: str,
) -> pd.DataFrame:
    rows = []
    quantile_grid = np.linspace(0.001, 0.999, 999)
    for index, name in enumerate(SPLINE15D_PARAMETER_NAMES):
        truth_values = np.asarray(truth[:, index], dtype=np.float64)
        prior_values = np.asarray(prior[:, index], dtype=np.float64)
        truth_values = truth_values[np.isfinite(truth_values)]
        prior_values = prior_values[np.isfinite(prior_values)]
        if not len(truth_values) or not len(prior_values):
            rows.append({"space": space, "parameter": name, "finite": False})
            continue
        combined = np.sort(np.concatenate((truth_values, prior_values)))
        truth_cdf = np.searchsorted(
            np.sort(truth_values), combined, side="right"
        ) / len(truth_values)
        prior_cdf = np.searchsorted(
            np.sort(prior_values), combined, side="right"
        ) / len(prior_values)
        truth_quantiles = np.quantile(truth_values, quantile_grid)
        prior_quantiles = np.quantile(prior_values, quantile_grid)
        rows.append(
            {
                "space": space,
                "parameter": name,
                "finite": True,
                "ks": float(np.max(np.abs(truth_cdf - prior_cdf))),
                "quantile_wasserstein": float(
                    np.mean(np.abs(truth_quantiles - prior_quantiles))
                ),
                "truth_mean": float(np.mean(truth_values)),
                "prior_mean": float(np.mean(prior_values)),
                "truth_std": float(np.std(truth_values)),
                "prior_std": float(np.std(prior_values)),
                "truth_q001": float(np.quantile(truth_values, 0.001)),
                "prior_q001": float(np.quantile(prior_values, 0.001)),
                "truth_q999": float(np.quantile(truth_values, 0.999)),
                "prior_q999": float(np.quantile(prior_values, 0.999)),
            }
        )
    return pd.DataFrame(rows)


def _write_correlation_matrix(path: Path, values: np.ndarray) -> None:
    matrix = np.asarray(values, dtype=np.float64)
    correlation = np.corrcoef(matrix, rowvar=False)
    pd.DataFrame(
        correlation,
        index=SPLINE15D_PARAMETER_NAMES,
        columns=SPLINE15D_PARAMETER_NAMES,
    ).to_csv(path)


def _spearman_correlation(values: np.ndarray) -> np.ndarray:
    ranks = pd.DataFrame(np.asarray(values, dtype=np.float64)).rank(
        method="average", axis=0
    )
    return ranks.corr(method="pearson").to_numpy(dtype=np.float64)


def _correlation_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right), ord="fro"))


def _write_epoch_snapshot(
    snapshot_root: Path,
    *,
    epoch: int,
    prior: RealNVPPrior,
    validation_theta: np.ndarray,
    validation_x: np.ndarray,
    transforms: dict[str, dict[str, Any]],
    whitening: dict[str, Any] | None,
    atom_half_width: float | None,
    sample_count: int,
    seed: int,
    flow_config: dict[str, Any],
    normalization: dict[str, Any],
    dataset_dir: Path,
    save_samples: bool,
    save_checkpoint: bool,
) -> dict[str, Any]:
    snapshot = ensure_dir(snapshot_root / f"epoch_{int(epoch):03d}")
    validation_nll = _flow_nll(prior, validation_x)
    metrics, prior_theta, prior_x = evaluate_generated_prior(
        prior,
        truth_theta=validation_theta,
        truth_x=validation_x,
        transforms=transforms,
        sample_count=sample_count,
        seed=seed,
        whitening=whitening,
        atom_half_width=atom_half_width,
    )
    if save_checkpoint:
        _save_checkpoint(
            snapshot / "checkpoint.eqx",
            prior,
            flow_config=flow_config,
            normalization=normalization,
            dataset_dir=dataset_dir,
            epoch=epoch,
            metric=validation_nll,
            selection={"criterion": "validation_nll"},
        )
    if save_samples:
        pd.DataFrame(prior_x, columns=SPLINE15D_PARAMETER_NAMES).to_parquet(
            snapshot / "prior_samples_normalized.parquet", index=False
        )
        pd.DataFrame(prior_theta, columns=SPLINE15D_PARAMETER_NAMES).to_parquet(
            snapshot / "prior_samples_physical.parquet", index=False
        )
    plot_truth_prior_physical_normalized(
        truth_theta=validation_theta,
        prior_theta=prior_theta,
        truth_x=validation_x,
        prior_x=prior_x,
        path=str(snapshot / "truth_vs_prior.png"),
        title_suffix=f"validation snapshot epoch {int(epoch)}",
    )
    marginal = pd.concat(
        [
            _marginal_snapshot_frame(validation_theta, prior_theta, space="physical"),
            _marginal_snapshot_frame(validation_x, prior_x, space="normalized"),
        ],
        ignore_index=True,
    )
    marginal.to_csv(snapshot / "marginal_metrics.csv", index=False)
    for label, values in (
        ("physical_truth", validation_theta),
        ("physical_prior", prior_theta),
        ("normalized_truth", validation_x),
        ("normalized_prior", prior_x),
    ):
        _write_correlation_matrix(snapshot / f"correlation_{label}.csv", values)
    truth_marginal = (
        validation_x
        if whitening is None
        else inverse_affine_whitening(validation_x, whitening)
    )
    prior_marginal = (
        prior_x if whitening is None else inverse_affine_whitening(prior_x, whitening)
    )
    truth_arguments = inverse_asinh_arguments_matrix(truth_marginal, transforms)
    prior_arguments = inverse_asinh_arguments_matrix(prior_marginal, transforms)
    argument_rows = []
    for index, name in enumerate(SPLINE15D_PARAMETER_NAMES):
        argument_rows.append(
            {
                "parameter": name,
                "truth_abs_gt4_fraction": float(
                    np.mean(np.abs(truth_arguments[:, index]) > 4.0)
                ),
                "prior_abs_gt4_fraction": float(
                    np.mean(np.abs(prior_arguments[:, index]) > 4.0)
                ),
                "truth_abs_gt5_fraction": float(
                    np.mean(np.abs(truth_arguments[:, index]) > 5.0)
                ),
                "prior_abs_gt5_fraction": float(
                    np.mean(np.abs(prior_arguments[:, index]) > 5.0)
                ),
                "truth_max_abs": float(np.max(np.abs(truth_arguments[:, index]))),
                "prior_max_abs": float(np.max(np.abs(prior_arguments[:, index]))),
                "truth_exact_zero_fraction": float(
                    np.mean(validation_theta[:, index] == 0.0)
                ),
                "prior_exact_zero_fraction": float(
                    np.mean(prior_theta[:, index] == 0.0)
                ),
            }
        )
    pd.DataFrame(argument_rows).to_csv(
        snapshot / "sinh_argument_and_atom_metrics.csv", index=False
    )
    truth_central = np.all(np.abs(truth_arguments) <= 4.0, axis=1)
    prior_central = np.all(np.abs(prior_arguments) <= 4.0, axis=1)
    physical_marginal = marginal.loc[marginal["space"] == "physical"]
    tail_diagnostics = {
        "truth_sinh_argument_abs_gt5_fraction": float(
            np.mean(np.abs(truth_arguments) > 5.0)
        ),
        "prior_sinh_argument_abs_gt5_fraction": float(
            np.mean(np.abs(prior_arguments) > 5.0)
        ),
        "truth_sinh_argument_max_abs": float(np.max(np.abs(truth_arguments))),
        "prior_sinh_argument_max_abs": float(np.max(np.abs(prior_arguments))),
        "truth_central_abs_sinh_le4_fraction": float(np.mean(truth_central)),
        "prior_central_abs_sinh_le4_fraction": float(np.mean(prior_central)),
        "median_ks_physical": float(physical_marginal["ks"].median()),
        "max_ks_physical": float(physical_marginal["ks"].max()),
        "median_quantile_wasserstein_physical": float(
            physical_marginal["quantile_wasserstein"].median()
        ),
        "max_quantile_wasserstein_physical": float(
            physical_marginal["quantile_wasserstein"].max()
        ),
        "correlation_frobenius_physical_spearman": _correlation_frobenius(
            _spearman_correlation(validation_theta),
            _spearman_correlation(prior_theta),
        ),
        "correlation_frobenius_physical_central_abs_sinh_le4": (
            _correlation_frobenius(
                np.corrcoef(validation_theta[truth_central], rowvar=False),
                np.corrcoef(prior_theta[prior_central], rowvar=False),
            )
            if np.sum(truth_central) > 2 and np.sum(prior_central) > 2
            else float("nan")
        ),
    }
    payload = {
        "epoch": int(epoch),
        "selection_criterion": "validation_nll",
        "validation_nll": float(validation_nll),
        **metrics,
        **tail_diagnostics,
    }
    write_json(snapshot / "metrics.json", payload)
    return payload


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
    preprocessing_cfg = dict(cfg.get("preprocessing", {}) or {})
    if args.atom_half_width is not None:
        preprocessing_cfg["normalized_atom_half_width"] = args.atom_half_width
    if args.whitening is not None:
        preprocessing_cfg["whitening"] = args.whitening == "true"
    normalization_family = str(norm_cfg.get("family", "asinh")).lower()
    if normalization_family not in {"asinh", "shifted_asinh"}:
        raise ValueError(
            "This production command supports asinh or shifted_asinh normalization"
        )
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "data_parallel": args.data_parallel,
        "seed": args.seed,
    }
    training_cfg.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
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
    snapshot_cfg = dict(cfg.get("snapshots", {}) or {})
    if args.snapshot_samples is not None:
        snapshot_cfg["sample_count"] = args.snapshot_samples

    dataset_dir = Path(args.dataset_dir or cfg["dataset_dir"])
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    ensure_dir(out)
    started = time.time()
    write_json(
        out / "resolved_config.json",
        {
            "spline15d_prior": {
                "dataset_dir": str(dataset_dir),
                "normalization": norm_cfg,
                "preprocessing": preprocessing_cfg,
                "flow": flow_cfg,
                "training": training_cfg,
                "evaluation": evaluation_cfg,
                "snapshots": snapshot_cfg,
                "output": output_cfg,
            }
        },
    )
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
                len(
                    exact_frames[split].drop_duplicates(list(SPLINE15D_PARAMETER_NAMES))
                )
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
    if bool(preprocessing_cfg.get("require_zero_exact_overlap", False)):
        leaking_splits = [
            split
            for split in ("validation", "test")
            if int(overlap_audit[split]["exact_train_overlap_rows"] or 0) > 0
        ]
        if leaking_splits:
            raise ValueError(
                "Exact spline-15D truth leakage remains after grouped repartition: "
                + ", ".join(leaking_splits)
            )
    atom_half_width = float(preprocessing_cfg.get("normalized_atom_half_width", 0.0))
    whitening_enabled = bool(preprocessing_cfg.get("whitening", False))
    target_table = str(preprocessing_cfg.get("target_table", "auto")).lower()
    if target_table == "exact":
        target_frames = exact_frames
    elif target_table == "projected":
        target_frames = frames
    elif target_table == "auto":
        target_frames = (
            exact_frames if (atom_half_width > 0.0 or whitening_enabled) else frames
        )
    else:
        raise ValueError("preprocessing.target_table must be exact, projected, or auto")
    matrices = {
        split: frame.loc[:, SPLINE15D_PARAMETER_NAMES].to_numpy(dtype=np.float64)
        for split, frame in target_frames.items()
    }
    if normalization_family == "shifted_asinh":
        transforms = fit_shifted_asinh_transforms(
            matrices["train"],
            lower_quantile=float(norm_cfg.get("lower_quantile", 0.1586552539)),
            upper_quantile=float(norm_cfg.get("upper_quantile", 0.8413447461)),
            minimum_lambda=float(norm_cfg.get("minimum_lambda", 1.0e-6)),
        )
    else:
        transforms = fit_asinh_transforms(
            matrices["train"],
            log10_lambda_min=float(norm_cfg.get("log10_lambda_min", -8.0)),
            log10_lambda_max=float(norm_cfg.get("log10_lambda_max", 8.0)),
            grid_size=int(norm_cfg.get("grid_size", 257)),
        )
    asinh_normalized = {
        split: forward_asinh_matrix(matrix, transforms)
        for split, matrix in matrices.items()
    }
    atom_counts: dict[str, dict[str, int]] = {}
    if atom_half_width > 0.0:
        for split_index, split in enumerate(("train", "validation", "test")):
            asinh_normalized[split], atom_counts[split] = (
                dequantize_normalized_zero_atoms(
                    asinh_normalized[split],
                    matrices[split],
                    half_width=atom_half_width,
                    seed=int(preprocessing_cfg.get("atom_seed", 260716)) + split_index,
                )
            )
    whitening = None
    if whitening_enabled:
        whitening = fit_affine_whitening(
            asinh_normalized["train"],
            covariance_jitter=float(preprocessing_cfg.get("covariance_jitter", 1.0e-5)),
        )
        normalized = {
            split: forward_affine_whitening(value, whitening)
            for split, value in asinh_normalized.items()
        }
    else:
        normalized = asinh_normalized
    continuous_roundtrip = inverse_spline15d_flow_coordinates(
        normalized["test"],
        transforms=transforms,
        whitening=whitening,
    )
    scientific_roundtrip = inverse_spline15d_flow_coordinates(
        normalized["test"],
        transforms=transforms,
        whitening=whitening,
        atom_half_width=atom_half_width or None,
    )
    dequantized_physical_test = inverse_asinh_matrix(
        asinh_normalized["test"], transforms
    )
    norm_payload = {
        "version": 2 if normalization_family == "shifted_asinh" else 1,
        "family": normalization_family,
        "fit_split": "train",
        "fit_rows": len(matrices["train"]),
        "parameter_names": SPLINE15D_PARAMETER_NAMES,
        "formula": (
            f"{normalization_family} marginal transform followed by optional "
            "Cholesky whitening"
        ),
        "inverse_formula": "inverse whitening, inverse asinh, atom reclipping",
        "transforms": transforms,
        "whitening": whitening,
        "normalized_atom_half_width": atom_half_width,
        "target_table": target_table,
        "normalized_atom_counts": atom_counts,
        "test_roundtrip_max_abs": float(
            np.max(np.abs(continuous_roundtrip - dequantized_physical_test))
        ),
        "test_dequantization_physical_max_abs": float(
            np.max(np.abs(dequantized_physical_test - matrices["test"]))
        ),
        "test_scientific_reclip_max_abs": float(
            np.max(np.abs(scientific_roundtrip - matrices["test"]))
        ),
        "train_flow_mean_abs_max": float(
            np.max(np.abs(np.mean(normalized["train"], axis=0)))
        ),
        "train_flow_covariance_frobenius": float(
            np.linalg.norm(
                np.cov(normalized["train"], rowvar=False)
                - np.eye(len(SPLINE15D_PARAMETER_NAMES)),
                ord="fro",
            )
        ),
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
        matrices["train"],
        matrices["validation"],
        matrices["test"],
        normalized["train"],
        normalized["validation"],
        normalized["test"],
        transforms,
        out / "normalization_before_after.png",
        whitening_enabled=whitening_enabled,
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
        "max_scale_saturation_fraction": float(
            evaluation_cfg.get("max_scale_saturation_fraction", 0.20)
        ),
        "max_sliced_wasserstein": float(
            evaluation_cfg.get("max_sliced_wasserstein", 0.25)
        ),
    }
    validation_generative_rows: list[dict[str, Any]] = []
    epoch_zero_selection: dict[str, Any] = {}
    checkpoint_selection = str(
        evaluation_cfg.get("checkpoint_selection", "generative_gates")
    ).lower()
    if checkpoint_selection not in {"validation_nll", "generative_gates"}:
        raise ValueError(
            "evaluation.checkpoint_selection must be validation_nll or generative_gates"
        )

    def _select_checkpoint(
        epoch: int,
        prior: RealNVPPrior,
        validation_nll: float,
    ) -> dict[str, Any] | None:
        if epoch not in {0, epochs} and epoch % evaluation_every:
            return None
        metrics, _prior_theta, _prior_x = evaluate_generated_prior(
            prior,
            truth_theta=matrices["validation"],
            truth_x=normalized["validation"],
            transforms=transforms,
            sample_count=evaluation_samples,
            seed=seed + 10_000,
            whitening=whitening,
            atom_half_width=atom_half_width or None,
        )
        payload = selection_payload(metrics, thresholds=thresholds)
        if epoch == 0:
            payload["quality_gates_eligible"] = bool(payload["eligible"])
            payload["eligible"] = True
            payload["is_epoch_zero_baseline"] = True
            epoch_zero_selection.update(payload)
        else:
            median_tolerance = float(
                evaluation_cfg.get("baseline_median_ks_tolerance", 0.01)
            )
            sw_tolerance = float(
                evaluation_cfg.get("baseline_sliced_wasserstein_tolerance", 0.02)
            )
            score_improvement = float(
                evaluation_cfg.get("baseline_score_min_improvement", 0.005)
            )
            relative_checks = {
                "beats_epoch_zero_score": float(payload["metric"])
                <= float(epoch_zero_selection["metric"]) - score_improvement,
                "preserves_epoch_zero_median_ks": float(payload["median_ks_normalized"])
                <= float(epoch_zero_selection["median_ks_normalized"])
                + median_tolerance,
                "preserves_epoch_zero_sliced_wasserstein": float(
                    payload["sliced_wasserstein_normalized"]
                )
                <= float(epoch_zero_selection["sliced_wasserstein_normalized"])
                + sw_tolerance,
            }
            payload["eligible"] = bool(
                payload["eligible"] and all(relative_checks.values())
            )
            payload["is_epoch_zero_baseline"] = False
            payload.update(relative_checks)
        novel = novel_masks["validation"]
        payload["validation_nll"] = float(validation_nll)
        payload["validation_novel_nll"] = (
            _flow_nll(prior, normalized["validation"][novel])
            if np.any(novel)
            else float("nan")
        )
        validation_generative_rows.append({"epoch": int(epoch), **payload})
        return payload

    snapshot_enabled = bool(snapshot_cfg.get("enabled", False))
    snapshot_every = max(int(snapshot_cfg.get("every_epochs", 1)), 1)
    snapshot_samples = max(int(snapshot_cfg.get("sample_count", 10000)), 256)
    snapshot_rows: list[dict[str, Any]] = []
    snapshot_root = ensure_dir(out / "snapshots") if snapshot_enabled else None

    def _snapshot_callback(epoch: int, prior: RealNVPPrior) -> None:
        if not snapshot_enabled or snapshot_root is None:
            return
        if epoch not in {0, epochs} and epoch % snapshot_every:
            return
        payload = _write_epoch_snapshot(
            snapshot_root,
            epoch=epoch,
            prior=prior,
            validation_theta=matrices["validation"],
            validation_x=normalized["validation"],
            transforms=transforms,
            whitening=whitening,
            atom_half_width=atom_half_width or None,
            sample_count=snapshot_samples,
            seed=seed + 20_000,
            flow_config=flow_cfg,
            normalization=norm_payload,
            dataset_dir=dataset_dir,
            save_samples=bool(snapshot_cfg.get("save_samples", True)),
            save_checkpoint=bool(snapshot_cfg.get("save_checkpoints", True)),
        )
        snapshot_rows.append(payload)
        pd.DataFrame(snapshot_rows).to_csv(
            out / "epoch_snapshot_history.csv", index=False
        )

    result = fit_realnvp_to_x(
        normalized["train"],
        normalized["validation"],
        latent_dim=len(SPLINE15D_PARAMETER_NAMES),
        flow_config=flow_cfg,
        training_config=training_cfg,
        seed=seed,
        epoch_callback=_snapshot_callback,
        progress_callback=_progress(out, epochs),
        selection_callback=(
            None if checkpoint_selection == "validation_nll" else _select_checkpoint
        ),
    )
    if not isinstance(result.prior, RealNVPPrior):
        raise TypeError("Training returned a non-RealNVP model")
    result.training_log.to_csv(out / "training_log.csv", index=False)
    result.validation_log.to_csv(out / "validation_log.csv", index=False)
    pd.DataFrame(validation_generative_rows).to_csv(
        out / "validation_generative_history.csv", index=False
    )
    _plot_history(
        result.training_log, result.validation_log, out / "training_history.png"
    )
    checkpoints = ensure_dir(out / "checkpoints")
    _save_checkpoint(
        checkpoints / "best.eqx",
        result.prior,
        flow_config=flow_cfg,
        normalization=norm_payload,
        dataset_dir=dataset_dir,
        epoch=result.best_epoch,
        metric=result.best_metric,
        selection={
            "eligible": result.best_selection_eligible,
            **result.best_selection_diagnostics,
        },
    )
    last_epoch = (
        int(result.training_log["epoch"].max()) if not result.training_log.empty else 0
    )
    _save_checkpoint(
        checkpoints / "last.eqx",
        result.last_prior,
        flow_config=flow_cfg,
        normalization=norm_payload,
        dataset_dir=dataset_dir,
        epoch=last_epoch,
        metric=_flow_nll(result.last_prior, normalized["validation"]),
    )

    n_samples = int(output_cfg.get("prior_samples", 50000))
    loaded_prior, _checkpoint_metadata = load_spline15d_realnvp_checkpoint(
        checkpoints / "best.eqx"
    )
    prior_x = np.asarray(
        jax.device_get(loaded_prior.sample(jax.random.PRNGKey(seed + 1), n_samples)),
        dtype=np.float64,
    )
    prior_theta = inverse_spline15d_flow_coordinates(
        prior_x,
        transforms=transforms,
        whitening=whitening,
        atom_half_width=atom_half_width or None,
    )
    prior_x_frame = pd.DataFrame(prior_x, columns=SPLINE15D_PARAMETER_NAMES)
    prior_theta_frame = pd.DataFrame(prior_theta, columns=SPLINE15D_PARAMETER_NAMES)
    prior_x_frame.to_parquet(
        out / "learned_prior_samples_normalized.parquet", index=False
    )
    prior_theta_frame.to_parquet(out / "learned_prior_samples.parquet", index=False)
    truth_limit = int(output_cfg.get("truth_sample_limit", len(exact_frames["test"])))
    truth_frame = (
        exact_frames["test"].loc[:, SPLINE15D_PARAMETER_NAMES].head(truth_limit)
    )
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
        whitening=whitening,
        atom_half_width=atom_half_width or None,
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
        whitening=whitening,
        atom_half_width=atom_half_width or None,
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
    calibrated_theta = inverse_spline15d_flow_coordinates(
        calibrated_x,
        transforms=transforms,
        whitening=whitening,
        atom_half_width=atom_half_width or None,
    )
    calibrated_x_frame = pd.DataFrame(calibrated_x, columns=SPLINE15D_PARAMETER_NAMES)
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
            whitening=whitening,
            atom_half_width=atom_half_width or None,
        )
    )

    baseline_rng = np.random.default_rng(seed + 3)
    baseline_x = baseline_rng.normal(size=(n_samples, len(SPLINE15D_PARAMETER_NAMES)))
    baseline_theta = inverse_spline15d_flow_coordinates(
        baseline_x,
        transforms=transforms,
        whitening=whitening,
        atom_half_width=atom_half_width or None,
    )
    baseline_metrics = evaluate_sample_pair(
        truth_theta=matrices["test"],
        truth_x=normalized["test"],
        prior_theta=baseline_theta,
        prior_x=baseline_x,
    )
    comparison_rows = []
    for model_name, temperature, metrics in (
        ("affine_gaussian_epoch0_baseline", 1.0, baseline_metrics),
        ("realnvp_unit_temperature", 1.0, test_unit_metrics),
        (
            "realnvp_validation_calibrated",
            selected_temperature,
            test_calibrated_metrics,
        ),
    ):
        comparison_rows.append(
            {"model": model_name, "base_temperature": temperature, **metrics}
        )
    pd.DataFrame(comparison_rows).to_csv(
        out / "test_baseline_comparison.csv", index=False
    )

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
    best_validation_nll = float(best_validation_row.iloc[-1]["negative_mean_log_prob"])
    diagnostics = write_supervised_prior_diagnostics(
        truth=truth_frame,
        prior=prior_theta_frame,
        parameter_names=SPLINE15D_PARAMETER_NAMES,
        out_dir=out / "diagnostics",
        summary={
            "model": "RealNVP",
            "latent_dim": 15,
            "normalization": (
                f"train-fitted {normalization_family} plus Cholesky whitening"
            ),
            "nll": nll,
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
            "normalization": (
                f"train-fitted {normalization_family} plus Cholesky whitening"
            ),
            "nll": calibrated_nll,
            "best_epoch": result.best_epoch,
            "base_temperature": selected_temperature,
            "temperature_fit_split": "validation",
        },
        max_corner_rows=int(output_cfg.get("max_corner_rows", 4000)),
    )
    summary = {
        "status": "complete",
        "model": "RealNVP",
        "latent_dim": 15,
        "dataset_dir": str(dataset_dir),
        "rows": {key: len(value) for key, value in frames.items()},
        "normalization_path": str(out / "normalization.json"),
        "nll": nll,
        "initial_train_nll": result.initial_train_nll,
        "best_epoch": result.best_epoch,
        "best_validation_nll": best_validation_nll,
        "best_selection_metric": result.best_metric,
        "best_selection_eligible": result.best_selection_eligible,
        "best_selection_diagnostics": result.best_selection_diagnostics,
        "checkpoint_selection": checkpoint_selection,
        "epoch_snapshots": len(snapshot_rows),
        "epoch_zero_selection": epoch_zero_selection,
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
