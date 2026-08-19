#!/usr/bin/env python3
"""Evaluate exact defensive proposal mixtures from immutable proposal banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.amortized.data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from euclid_dsps.amortized.features import read_feature_stats
from euclid_dsps.amortized.latent import theta_to_x
from euclid_dsps.amortized.posterior import posterior_log_prob
from euclid_dsps.amortized.posthoc_calibration import (
    build_defensive_mixture_bank,
    importance_weight_bank,
    load_posterior_bank,
)
from euclid_dsps.amortized.train import (
    _latent_spec_for_amortized_config,
    load_checkpoint,
)
from euclid_dsps.config import load_config

eqx, _optax = require_amortized_dependencies()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-posterior", type=Path, required=True)
    parser.add_argument("--tail-posterior", type=Path, required=True)
    parser.add_argument("--tail-temperature", type=float, required=True)
    parser.add_argument("--tail-fractions", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--object-batch-size", type=int, default=64)
    parser.add_argument("--sample-chunk-size", type=int, default=128)
    parser.add_argument("--resample-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=260817)
    parser.add_argument("--self-logq-atol", type=float, default=0.05)
    parser.add_argument("--min-median-ess-fraction", type=float, default=0.05)
    parser.add_argument("--max-fraction-pareto-k-gt-0p7", type=float, default=0.2)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fractions = _parse_fractions(args.tail_fractions)
    if not math.isfinite(args.tail_temperature) or args.tail_temperature <= 1.0:
        raise ValueError("tail_temperature must be finite and greater than one")
    if args.object_batch_size <= 0 or args.sample_chunk_size <= 0:
        raise ValueError("evaluation batch sizes must be positive")
    if args.resample_count <= 0:
        raise ValueError("resample_count must be positive")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected GPU backend, got {jax.default_backend()}")
    for path in (args.config, args.dataset, args.checkpoint, args.feature_stats):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite defensive scan: {args.out}")
    args.out.mkdir(parents=True)

    config = load_config(args.config)
    config["catalog_path"] = str(args.dataset)
    latent_spec = _latent_spec_for_amortized_config(config)
    base_bank = load_posterior_bank(
        args.base_posterior, parameter_names=latent_spec.names
    )
    tail_bank = load_posterior_bank(
        args.tail_posterior, parameter_names=latent_spec.names
    )
    identities = _validate_bank_contract(base_bank, tail_bank)
    base_temperature_contract = _validate_inference_temperature(
        args.base_posterior,
        expected=1.0,
        allow_implicit_unit=True,
    )
    tail_temperature_contract = _validate_inference_temperature(
        args.tail_posterior, expected=args.tail_temperature
    )

    features = _load_features(
        config,
        row_indices=identities,
        feature_stats_path=args.feature_stats,
    )
    model = load_checkpoint(args.checkpoint, config)
    base_eval = _evaluate_bank_densities(
        base_bank.frame,
        parameter_names=base_bank.parameter_names,
        model=model,
        features=features,
        latent_spec=latent_spec,
        tail_temperature=args.tail_temperature,
        object_batch_size=args.object_batch_size,
        sample_chunk_size=args.sample_chunk_size,
    )
    tail_eval = _evaluate_bank_densities(
        tail_bank.frame,
        parameter_names=tail_bank.parameter_names,
        model=model,
        features=features,
        latent_spec=latent_spec,
        tail_temperature=args.tail_temperature,
        object_batch_size=args.object_batch_size,
        sample_chunk_size=args.sample_chunk_size,
    )
    density_validation = {
        "base_saved_vs_recomputed_q1": _density_difference(
            base_bank.frame["logq"].to_numpy(), base_eval["logq_base"]
        ),
        "tail_saved_vs_recomputed_qtail": _density_difference(
            tail_bank.frame["logq"].to_numpy(), tail_eval["logq_tail"]
        ),
        "base_saved_vs_recomputed_prior": _density_difference(
            base_bank.frame["logprior"].to_numpy(), base_eval["logprior"]
        ),
        "tail_saved_vs_recomputed_prior": _density_difference(
            tail_bank.frame["logprior"].to_numpy(), tail_eval["logprior"]
        ),
        "p99_absolute_tolerance": float(args.self_logq_atol),
    }
    for name, diagnostic in density_validation.items():
        if name == "p99_absolute_tolerance":
            continue
        if diagnostic["p99_absolute_difference"] > float(args.self_logq_atol):
            raise RuntimeError(
                f"{name} exceeds density validation tolerance: {diagnostic}"
            )
    (args.out / "cross_density_validation.json").write_text(
        json.dumps(density_validation, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    candidates = []
    for fraction in fractions:
        candidate_bank, allocation = build_defensive_mixture_bank(
            base_bank,
            tail_bank,
            base_logq_on_base=base_eval["logq_base"],
            tail_logq_on_base=base_eval["logq_tail"],
            base_logq_on_tail=tail_eval["logq_base"],
            tail_logq_on_tail=tail_eval["logq_tail"],
            base_target_logprior=base_eval["logprior"],
            tail_target_logprior=tail_eval["logprior"],
            requested_tail_fraction=fraction,
            seed=args.seed,
        )
        _weighted, _resampled, diagnostics = importance_weight_bank(
            candidate_bank,
            target_logprior=candidate_bank.frame["logprior"].to_numpy(),
            resample_count=args.resample_count,
            seed=args.seed,
        )
        candidate_dir = args.out / f"epsilon_{_slug(fraction)}"
        candidate_dir.mkdir()
        diagnostics.to_parquet(
            candidate_dir / "importance_diagnostics.parquet", index=False
        )
        diagnostics.to_csv(candidate_dir / "importance_diagnostics.csv", index=False)
        median_ess = float(diagnostics["raw_ess_fraction"].median())
        bad_k = float(np.nanmean(diagnostics["pareto_k"] > 0.7))
        support_pass = median_ess >= args.min_median_ess_fraction and bad_k <= (
            args.max_fraction_pareto_k_gt_0p7
        )
        support = {
            "status": "PASS" if support_pass else "FAIL",
            "median_raw_ess_fraction": median_ess,
            "min_median_raw_ess_fraction": args.min_median_ess_fraction,
            "fraction_pareto_k_gt_0p7": bad_k,
            "max_fraction_pareto_k_gt_0p7": args.max_fraction_pareto_k_gt_0p7,
        }
        summary = {
            "status": "complete",
            "proposal": "defensive deterministic-mixture MIS",
            "tail_temperature": float(args.tail_temperature),
            "allocation": allocation,
            "n_objects": int(len(diagnostics)),
            "n_joint_draws": int(len(candidate_bank.frame)),
            "median_raw_ess_fraction": median_ess,
            "median_psis_ess_fraction": float(
                diagnostics["psis_ess_fraction"].median()
            ),
            "fraction_pareto_k_gt_0p7": bad_k,
            "fraction_pareto_k_gt_1": float(np.nanmean(diagnostics["pareto_k"] > 1.0)),
            "support_gate": support,
            "cohort_role": "proposal tuning on one frozen disjoint probe",
            "spectroscopy_used": False,
            "ready_for_empirical_bayes": False,
        }
        (candidate_dir / "support_gate.json").write_text(
            json.dumps(support, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (candidate_dir / "importance_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (candidate_dir / "DONE").touch()
        candidates.append(summary)
        print(
            "[defensive-proposal] "
            f"temperature={args.tail_temperature:g} epsilon={fraction:g} "
            f"realized={allocation['realized_tail_fraction']:.6g} "
            f"ESS={median_ess:.5f} bad_k={bad_k:.5f} "
            f"support={support['status']}",
            flush=True,
        )

    task_summary = {
        "status": "complete",
        "tail_temperature": float(args.tail_temperature),
        "source_temperature_contracts": {
            "base": base_temperature_contract,
            "tail": tail_temperature_contract,
        },
        "tail_fractions": fractions,
        "n_objects": int(len(identities)),
        "proposal_samples_per_object": int(
            base_bank.frame.groupby(base_bank.identity_column).size().iloc[0]
        ),
        "density_validation": density_validation,
        "inputs": {
            "base_posterior": [_receipt(path) for path in base_bank.source_files],
            "tail_posterior": [_receipt(path) for path in tail_bank.source_files],
            "base_inference_summary": _receipt(
                args.base_posterior / "inference_summary.json"
            ),
            "base_inference_indices": _receipt(
                args.base_posterior / "inference_indices.npy"
            ),
            "tail_inference_summary": _receipt(
                args.tail_posterior / "inference_summary.json"
            ),
            "tail_inference_indices": _receipt(
                args.tail_posterior / "inference_indices.npy"
            ),
            "config": _receipt(args.config),
            "dataset": _receipt(args.dataset),
            "checkpoint": _receipt(args.checkpoint),
            "feature_stats": _receipt(args.feature_stats),
        },
        "candidates": candidates,
    }
    (args.out / "task_summary.json").write_text(
        json.dumps(task_summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.out / "DONE").touch()


def _parse_fractions(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(
        not math.isfinite(item) or not 0.0 < item < 1.0 for item in result
    ):
        raise ValueError("tail fractions must be a non-empty CSV within (0, 1)")
    if len(set(result)) != len(result):
        raise ValueError("tail fractions must be unique")
    return result


def _slug(value: float) -> str:
    return f"{value:.8g}".replace(".", "p")


def _validate_inference_temperature(
    root: Path,
    *,
    expected: float,
    allow_implicit_unit: bool = False,
) -> dict[str, object]:
    summary_path = root / "inference_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    if "posterior_base_temperature" not in summary:
        if allow_implicit_unit and float(expected) == 1.0:
            return {
                "expected": 1.0,
                "recorded": None,
                "source": "implicit unit temperature from legacy inference",
                "validated_by_recomputed_density": True,
            }
        raise ValueError(f"proposal temperature missing from {summary_path}")
    actual = float(summary["posterior_base_temperature"])
    if actual != float(expected):
        raise ValueError(f"proposal temperature mismatch: {actual} != {expected}")
    return {
        "expected": float(expected),
        "recorded": actual,
        "source": "inference_summary",
        "validated_by_recomputed_density": True,
    }


def _validate_bank_contract(base_bank, tail_bank) -> np.ndarray:
    if (
        base_bank.identity_column != "row_index"
        or tail_bank.identity_column != "row_index"
    ):
        raise ValueError("defensive Pop-COSMOS scan requires row_index identity")
    base_counts = base_bank.frame.groupby("row_index", sort=False).size()
    tail_counts = tail_bank.frame.groupby("row_index", sort=False).size()
    base_rows = base_counts.index.to_numpy(dtype=np.int64)
    tail_rows = tail_counts.index.to_numpy(dtype=np.int64)
    if not np.array_equal(base_rows, tail_rows):
        raise ValueError("base and tail proposal cohorts differ")
    if base_counts.nunique() != 1 or tail_counts.nunique() != 1:
        raise ValueError("proposal banks have unequal draws per object")
    if int(base_counts.iloc[0]) != int(tail_counts.iloc[0]):
        raise ValueError("base and tail proposal counts differ")
    base_objects = base_bank.frame.groupby("row_index", sort=False)["object_id"].first()
    tail_objects = tail_bank.frame.groupby("row_index", sort=False)["object_id"].first()
    if not np.array_equal(base_objects.to_numpy(), tail_objects.to_numpy()):
        raise ValueError("base and tail object identifiers differ")
    return base_rows


def _load_features(config, *, row_indices, feature_stats_path):
    stats = read_feature_stats(feature_stats_path)
    arrays = load_photometry_arrays_from_config(
        config, batch_size=10_000, row_indices=row_indices
    )
    if arrays.row_index is None:
        raise ValueError("selected catalog does not expose row indices")
    position = {int(value): index for index, value in enumerate(arrays.row_index)}
    order = np.asarray([position[int(value)] for value in row_indices], dtype=int)
    batch = next(
        iter_photometry_batches_from_arrays(
            arrays,
            batch_size=len(row_indices),
            feature_stats=stats,
            order=order,
        )
    )
    if not np.array_equal(np.asarray(batch.row_index), row_indices):
        raise RuntimeError("feature cohort/order does not match proposal banks")
    return batch.features


def _evaluate_bank_densities(
    frame,
    *,
    parameter_names,
    model,
    features,
    latent_spec,
    tail_temperature,
    object_batch_size,
    sample_chunk_size,
):
    counts = frame.groupby("row_index", sort=False).size()
    n_objects = len(counts)
    n_samples = int(counts.iloc[0])
    theta = frame.loc[:, parameter_names].to_numpy(dtype=np.float32)
    theta = theta.reshape(n_objects, n_samples, len(parameter_names))
    x = theta_to_x(jnp.asarray(theta), latent_spec)
    x = jnp.swapaxes(x, 0, 1)
    features = jnp.asarray(features)

    @eqx.filter_jit
    def evaluate(batch_features, batch_x):
        return (
            posterior_log_prob(model, batch_features, batch_x, base_temperature=1.0),
            posterior_log_prob(
                model,
                batch_features,
                batch_x,
                base_temperature=float(tail_temperature),
            ),
            model.prior.log_prob(batch_x),
        )

    result = {
        "logq_base": np.empty((n_objects, n_samples), dtype=np.float64),
        "logq_tail": np.empty((n_objects, n_samples), dtype=np.float64),
        "logprior": np.empty((n_objects, n_samples), dtype=np.float64),
    }
    for object_start in range(0, n_objects, int(object_batch_size)):
        object_stop = min(n_objects, object_start + int(object_batch_size))
        for sample_start in range(0, n_samples, int(sample_chunk_size)):
            sample_stop = min(n_samples, sample_start + int(sample_chunk_size))
            values = evaluate(
                features[object_start:object_stop],
                x[sample_start:sample_stop, object_start:object_stop],
            )
            for name, value in zip(result, values, strict=True):
                result[name][object_start:object_stop, sample_start:sample_stop] = (
                    np.asarray(jax.device_get(value)).T
                )
    return {name: values.reshape(-1) for name, values in result.items()}


def _density_difference(saved, evaluated) -> dict[str, float]:
    difference = np.abs(
        np.asarray(saved, dtype=np.float64) - np.asarray(evaluated, dtype=np.float64)
    )
    if not np.all(np.isfinite(difference)):
        raise ValueError("density validation contains non-finite differences")
    return {
        "median_absolute_difference": float(np.median(difference)),
        "p99_absolute_difference": float(np.quantile(difference, 0.99)),
        "max_absolute_difference": float(np.max(difference)),
    }


def _receipt(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


if __name__ == "__main__":
    main()
