#!/usr/bin/env python3
"""Train one FENIKS normalization/flow combination and write diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.config import load_config
from euclid_dsps.io import ensure_dir, write_json
from euclid_dsps.prior_learning.data import load_truth_dataset
from euclid_dsps.prior_learning.diagnostics import write_supervised_prior_diagnostics
from euclid_dsps.prior_learning.flows import assert_flow_integrity
from euclid_dsps.prior_learning.marginal_normalization import (
    ATOM_PARAMETER_NAMES,
    forward_matrix,
    inverse_matrix,
    load_marginal_transforms,
    non_atom_names,
    shared_atom_mask,
)
from euclid_dsps.prior_learning.train import (
    _prior_architecture_payload,
    fit_realnvp_to_x,
    prior_learning_config,
)

eqx, _optax = require_amortized_dependencies()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--prior-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--data-parallel",
        choices=("single", "auto", "pmap"),
    )
    return parser.parse_args()


def _read_physical_matrix(
    path: str | Path,
    *,
    columns: tuple[str, ...],
    limit: int | None,
) -> np.ndarray:
    frame = pd.read_parquet(path, columns=list(columns))
    if limit is not None:
        frame = frame.head(max(int(limit), 0))
    matrix = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(matrix).all(axis=1)
    matrix = matrix[finite]
    if len(matrix) == 0:
        raise ValueError(f"No finite rows in {path}")
    return matrix


def _flow_nll(prior, x: np.ndarray) -> float:
    return -float(
        np.mean(np.asarray(prior.log_prob(jnp.asarray(x, dtype=jnp.float32))))
    )


def _progress(label: str, total_epochs: int):
    def callback(record: dict[str, Any]) -> None:
        validation = record.get("validation_negative_mean_log_prob")
        validation_text = "nan" if validation is None else f"{float(validation):.6g}"
        print(
            f"[benchmark][{label}] epoch={record['epoch']}/{total_epochs} "
            f"train_nll={record['train_negative_mean_log_prob']:.6g} "
            f"validation_nll={validation_text} "
            f"best={record['best_metric']:.6g}@{record['best_epoch']}",
            flush=True,
        )

    return callback


def _fit_branch(
    label: str,
    train_x: np.ndarray,
    validation_x: np.ndarray,
    *,
    flow_config: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
):
    print(
        f"[benchmark] fitting {label}: rows={len(train_x)} dim={train_x.shape[1]}",
        flush=True,
    )
    result = fit_realnvp_to_x(
        train_x,
        validation_x,
        latent_dim=train_x.shape[1],
        flow_config=flow_config,
        training_config=training_config,
        seed=seed,
        progress_callback=_progress(label, int(training_config["epochs"])),
    )
    assert_flow_integrity(
        result.prior,
        context=f"normalization benchmark {label}",
        key=jax.random.PRNGKey(seed + 90_000),
    )
    return result


def _save_flow(
    path: Path, prior, flow_config: dict[str, Any], metadata: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, prior)
    write_json(
        path.with_suffix(path.suffix + ".json"),
        {
            **metadata,
            "architecture": _prior_architecture_payload(prior, flow_config),
            "flow_integrity": assert_flow_integrity(
                prior,
                context=f"normalization benchmark checkpoint {path}",
                sample_count=64,
            ),
        },
    )


def _sample_single(
    prior,
    *,
    count: int,
    seed: int,
    names: tuple[str, ...],
    transforms: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(prior.sample(jax.random.PRNGKey(seed), count), dtype=np.float64)
    return inverse_matrix(x, names, transforms), np.full(count, "continuous")


def _sample_hybrid(
    atom_prior,
    continuous_prior,
    *,
    atom_probability: float,
    count: int,
    seed: int,
    names: tuple[str, ...],
    transforms: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    atom_draw = rng.random(count) < atom_probability
    theta = np.empty((count, len(names)), dtype=np.float64)
    branch = np.where(atom_draw, "atom", "continuous")
    continuous_count = int(np.sum(~atom_draw))
    if continuous_count:
        continuous_x = np.asarray(
            continuous_prior.sample(jax.random.PRNGKey(seed + 1), continuous_count)
        )
        theta[~atom_draw] = inverse_matrix(continuous_x, names, transforms)
    atom_count = int(np.sum(atom_draw))
    if atom_count:
        atom_names = non_atom_names(names)
        atom_x = np.asarray(atom_prior.sample(jax.random.PRNGKey(seed + 2), atom_count))
        atom_theta = inverse_matrix(
            atom_x,
            atom_names,
            {name: transforms[name] for name in atom_names},
        )
        for index, name in enumerate(atom_names):
            theta[atom_draw, names.index(name)] = atom_theta[:, index]
        for name in ATOM_PARAMETER_NAMES:
            theta[atom_draw, names.index(name)] = transforms[name]["atom_value"]
    return theta, branch


def _hybrid_nll(
    atom_prior,
    continuous_prior,
    theta: np.ndarray,
    *,
    atom_probability: float,
    names: tuple[str, ...],
    transforms: dict[str, dict[str, Any]],
) -> float:
    mask = shared_atom_mask(theta, names, transforms)
    terms = np.empty(len(theta), dtype=np.float64)
    if np.any(mask):
        atom_names = non_atom_names(names)
        atom_theta = theta[mask][:, [names.index(name) for name in atom_names]]
        atom_x = forward_matrix(
            atom_theta,
            atom_names,
            {name: transforms[name] for name in atom_names},
        )
        terms[mask] = np.log(atom_probability) + np.asarray(atom_prior.log_prob(atom_x))
    if np.any(~mask):
        continuous_x = forward_matrix(theta[~mask], names, transforms)
        terms[~mask] = np.log1p(-atom_probability) + np.asarray(
            continuous_prior.log_prob(continuous_x)
        )
    return -float(np.mean(terms))


def _roundtrip_error(
    theta: np.ndarray,
    names: tuple[str, ...],
    transforms: dict[str, dict[str, Any]],
) -> float:
    reconstructed = inverse_matrix(
        forward_matrix(theta, names, transforms), names, transforms
    )
    return float(np.max(np.abs(reconstructed - theta)))


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    cfg = prior_learning_config(config)
    benchmark = dict(config.get("normalization_benchmark", {}) or {})
    version = str(benchmark.get("version", "")).strip().lower()
    if version not in {"hybrid", "dirac_preserved"}:
        raise ValueError(
            "normalization_benchmark.version must be hybrid or dirac_preserved"
        )
    specs_path = Path(benchmark["specs"])
    transforms = load_marginal_transforms(specs_path)
    out = ensure_dir(args.out)
    flow_config = dict(cfg["flow"])
    training_config = dict(cfg["training"])
    if args.epochs is not None:
        training_config["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        training_config["batch_size"] = int(args.batch_size)
    if args.data_parallel is not None:
        training_config["data_parallel"] = args.data_parallel
    if args.n_layers is not None:
        flow_config["n_layers"] = int(args.n_layers)
    if args.hidden_size is not None:
        flow_config["hidden_size"] = int(args.hidden_size)
    seed = int(args.seed if args.seed is not None else training_config.get("seed", 42))
    prior_samples = int(
        args.prior_samples
        if args.prior_samples is not None
        else cfg["output"].get("prior_samples", 8192)
    )

    truth_schema = load_truth_dataset(
        cfg["train_dataset"],
        schema_name=cfg["schema"],
        missing_policy=cfg["missing_policy"],
        bounds=cfg["bounds"],
        limit=1,
    ).schema
    names = tuple(parameter.name for parameter in truth_schema.parameters)
    columns = tuple(parameter.column for parameter in truth_schema.parameters)
    missing_specs = sorted(set(names) - set(transforms))
    if missing_specs:
        raise ValueError(f"Missing marginal transforms for {missing_specs}")
    train_theta = _read_physical_matrix(
        cfg["train_dataset"], columns=columns, limit=args.limit
    )
    validation_theta = _read_physical_matrix(
        cfg["validation_dataset"], columns=columns, limit=args.limit
    )
    test_theta = _read_physical_matrix(
        cfg["test_dataset"], columns=columns, limit=args.limit
    )
    start = time.time()
    write_json(
        out / "benchmark_config.json",
        {
            "config": str(args.config),
            "normalization_version": version,
            "normalization_specs": str(specs_path),
            "flow": flow_config,
            "training": training_config,
            "parameter_names": list(names),
            "train_dataset": cfg["train_dataset"],
            "validation_dataset": cfg["validation_dataset"],
            "test_dataset": cfg["test_dataset"],
            "limit_per_split": args.limit,
        },
    )

    branch_results = {}
    if version == "dirac_preserved":
        train_x = forward_matrix(train_theta, names, transforms)
        validation_x = forward_matrix(validation_theta, names, transforms)
        test_x = forward_matrix(test_theta, names, transforms)
        result = _fit_branch(
            "all",
            train_x,
            validation_x,
            flow_config=flow_config,
            training_config=training_config,
            seed=seed,
        )
        branch_results["all"] = result
        _save_flow(
            out / "checkpoints" / "flow.eqx",
            result.prior,
            flow_config,
            {"normalization_version": version, "parameter_names": list(names)},
        )
        sampled_theta, sampled_branch = _sample_single(
            result.prior,
            count=prior_samples,
            seed=seed + 1,
            names=names,
            transforms=transforms,
        )
        nll = {
            "train": _flow_nll(result.prior, train_x),
            "validation": _flow_nll(result.prior, validation_x),
            "test": _flow_nll(result.prior, test_x),
        }
        atom_probability = float(
            np.mean(shared_atom_mask(train_theta, names, transforms))
        )
        transformed_abs_max = {
            "train": float(np.max(np.abs(train_x))),
            "validation": float(np.max(np.abs(validation_x))),
            "test": float(np.max(np.abs(test_x))),
        }
    else:
        train_atom = shared_atom_mask(train_theta, names, transforms)
        validation_atom = shared_atom_mask(validation_theta, names, transforms)
        if not np.any(train_atom) or not np.any(~train_atom):
            raise ValueError("Hybrid training requires both atom and continuous rows")
        atom_probability = float(np.mean(train_atom))
        atom_names = non_atom_names(names)
        atom_indices = [names.index(name) for name in atom_names]
        atom_transforms = {name: transforms[name] for name in atom_names}
        atom_train_x = forward_matrix(
            train_theta[train_atom][:, atom_indices], atom_names, atom_transforms
        )
        atom_validation_x = forward_matrix(
            validation_theta[validation_atom][:, atom_indices],
            atom_names,
            atom_transforms,
        )
        continuous_train_x = forward_matrix(train_theta[~train_atom], names, transforms)
        continuous_validation_x = forward_matrix(
            validation_theta[~validation_atom], names, transforms
        )
        atom_result = _fit_branch(
            "atom",
            atom_train_x,
            atom_validation_x,
            flow_config=flow_config,
            training_config=training_config,
            seed=seed,
        )
        continuous_result = _fit_branch(
            "continuous",
            continuous_train_x,
            continuous_validation_x,
            flow_config=flow_config,
            training_config=training_config,
            seed=seed + 1,
        )
        branch_results.update(atom=atom_result, continuous=continuous_result)
        for label, result, branch_names in (
            ("atom", atom_result, atom_names),
            ("continuous", continuous_result, names),
        ):
            _save_flow(
                out / "checkpoints" / f"{label}.eqx",
                result.prior,
                flow_config,
                {
                    "normalization_version": version,
                    "branch": label,
                    "parameter_names": list(branch_names),
                    "atom_probability": atom_probability,
                },
            )
        sampled_theta, sampled_branch = _sample_hybrid(
            atom_result.prior,
            continuous_result.prior,
            atom_probability=atom_probability,
            count=prior_samples,
            seed=seed + 2,
            names=names,
            transforms=transforms,
        )
        nll = {
            split: _hybrid_nll(
                atom_result.prior,
                continuous_result.prior,
                theta,
                atom_probability=atom_probability,
                names=names,
                transforms=transforms,
            )
            for split, theta in (
                ("train", train_theta),
                ("validation", validation_theta),
                ("test", test_theta),
            )
        }
        transformed_abs_max = {
            "train_atom_14d": float(np.max(np.abs(atom_train_x))),
            "train_continuous_18d": float(np.max(np.abs(continuous_train_x))),
            "validation_atom_14d": float(np.max(np.abs(atom_validation_x))),
            "validation_continuous_18d": float(np.max(np.abs(continuous_validation_x))),
        }

    for label, result in branch_results.items():
        result.training_log.assign(branch=label).to_csv(
            out / f"training_log_{label}.csv", index=False
        )
        result.validation_log.assign(branch=label).to_csv(
            out / f"validation_log_{label}.csv", index=False
        )
    prior_frame = pd.DataFrame(sampled_theta, columns=names)
    prior_frame.insert(0, "branch", sampled_branch)
    prior_frame.insert(0, "sample_id", np.arange(len(prior_frame), dtype=np.int64))
    prior_frame.to_parquet(out / "learned_prior_samples.parquet", index=False)
    truth_frame = pd.DataFrame(test_theta, columns=names)
    sampled_atom = shared_atom_mask(sampled_theta, names, transforms)
    summary = {
        "normalization_version": version,
        "flow_type": flow_config["type"],
        "normalization_specs": str(specs_path),
        "train_rows": int(len(train_theta)),
        "validation_rows": int(len(validation_theta)),
        "test_rows": int(len(test_theta)),
        "prior_samples": int(prior_samples),
        "atom_probability_train": atom_probability,
        "atom_fraction_test": float(
            np.mean(shared_atom_mask(test_theta, names, transforms))
        ),
        "atom_values": {
            name: float(transforms[name]["atom_value"]) for name in ATOM_PARAMETER_NAMES
        },
        "atom_fraction_sampled": float(np.mean(sampled_atom)),
        "exact_atom_sample_rows": int(np.sum(sampled_atom)),
        "negative_mean_log_prob": nll,
        "nll_comparison_scope": (
            "Compare NLL only between flows with the same normalization version; "
            "hybrid and continuous-18D densities use different reference measures."
        ),
        "transformed_abs_max": transformed_abs_max,
        "test_transform_roundtrip_max_abs": _roundtrip_error(
            test_theta, names, transforms
        ),
        "elapsed_time_s": float(time.time() - start),
        "branch_best_epochs": {
            label: int(result.best_epoch) for label, result in branch_results.items()
        },
        "branch_best_metrics": {
            label: float(result.best_metric) for label, result in branch_results.items()
        },
    }
    write_supervised_prior_diagnostics(
        truth=truth_frame,
        prior=prior_frame,
        parameter_names=names,
        out_dir=out,
        summary=summary,
    )
    write_json(out / "benchmark_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
