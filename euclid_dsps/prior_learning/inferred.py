"""Post-hoc prior learning from MAP or MCLMC inferred theta samples."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.amortized.latent import latent_spec_from_config, theta_to_x
from euclid_dsps.io import ensure_dir, write_json

from .data import TruthDataset, prior_samples_frame
from .diagnostics import write_supervised_prior_diagnostics
from .schema import ParameterSpec, TruthSchema
from .splits import train_validation_split as _train_validation_split
from .train import (
    fit_realnvp_to_x,
    prior_learning_config,
    save_prior_checkpoint,
)


def train_inferred_prior(
    config: dict[str, Any],
    out_dir: str | Path,
    *,
    input_paths: tuple[str | Path, ...],
    limit: int | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    seed: int | None = None,
    validation_fraction: float | None = None,
    verbose: bool = True,
) -> None:
    """Train a RealNVP prior from inferred per-galaxy theta samples."""
    if not input_paths:
        raise ValueError("train_inferred_prior requires at least one input path")
    out = ensure_dir(out_dir)
    cfg = prior_learning_config(config)
    training_cfg = dict(cfg["training"])
    if batch_size is not None:
        training_cfg["batch_size"] = int(batch_size)
    if epochs is not None:
        training_cfg["epochs"] = int(epochs)
    if seed is not None:
        training_cfg["seed"] = int(seed)
    if validation_fraction is not None:
        training_cfg["validation_fraction"] = float(validation_fraction)
    dataset = load_inferred_theta_dataset(
        config,
        input_paths=input_paths,
        limit=limit,
        seed=int(training_cfg.get("seed", 42)),
    )
    _log(verbose, "[prior] inferred-prior training")
    _log(verbose, f"[prior] inputs: {', '.join(str(path) for path in input_paths)}")
    _log(
        verbose,
        "[prior] inferred matrix: "
        f"rows={dataset.theta.shape[0]} dim={dataset.theta.shape[1]} "
        f"dropped_nonfinite={dataset.dropped_rows}",
    )
    write_json(out / "normalized_config.json", config)
    dataset.theta_frame().to_parquet(out / "inferred_theta_samples.parquet", index=False)
    dataset.x_frame().to_parquet(out / "inferred_x_samples.parquet", index=False)
    split = _train_validation_split(
        len(dataset.x),
        validation_fraction=float(training_cfg.get("validation_fraction", 0.1)),
        seed=int(training_cfg.get("seed", 42)),
    )
    np.save(out / "train_indices.npy", split["train"])
    np.save(out / "validation_indices.npy", split["validation"])
    start = time.time()
    result = fit_realnvp_to_x(
        dataset.x[split["train"]],
        dataset.x[split["validation"]] if len(split["validation"]) else None,
        latent_dim=dataset.x.shape[1],
        flow_config=cfg["flow"],
        training_config=training_cfg,
        seed=int(training_cfg.get("seed", 42)),
    )
    result.training_log.to_csv(out / "prior_training_log.csv", index=False)
    result.validation_log.to_csv(out / "prior_validation_loglike.csv", index=False)
    ckpt_dir = ensure_dir(out / "checkpoints")
    save_prior_checkpoint(
        ckpt_dir / "best.eqx",
        result.prior,
        config=config,
        truth=dataset,
        flow_config=cfg["flow"],
        epoch=result.best_epoch,
        metric=result.best_metric,
    )
    save_prior_checkpoint(
        ckpt_dir / "last.eqx",
        result.last_prior,
        config=config,
        truth=dataset,
        flow_config=cfg["flow"],
        epoch=int(training_cfg.get("epochs", 20)),
        metric=float(result.training_log["loss"].iloc[-1]),
    )
    sample_count = int(cfg["output"].get("prior_samples", 8192))
    key = jax.random.PRNGKey(int(training_cfg.get("seed", 42)) + 1)
    x_prior = np.asarray(result.prior.sample(key, sample_count), dtype=np.float32)
    log_prob = np.asarray(result.prior.log_prob(jnp.asarray(x_prior)), dtype=float)
    prior_frame = prior_samples_frame(x_prior, dataset.latent_spec, log_prob=log_prob)
    prior_frame.to_parquet(out / "learned_prior_samples.parquet", index=False)
    summary = {
        "input_paths": [str(path) for path in input_paths],
        "schema": dataset.schema.to_dict(),
        "parameter_names": list(dataset.parameter_names),
        "n_inferred_rows": int(dataset.theta.shape[0]),
        "dropped_nonfinite_rows": int(dataset.dropped_rows),
        "train_rows": int(len(split["train"])),
        "validation_rows": int(len(split["validation"])),
        "epochs": int(training_cfg.get("epochs", 20)),
        "batch_size": int(training_cfg.get("batch_size", 256)),
        "initial_train_nll": float(result.initial_train_nll),
        "best_metric": float(result.best_metric),
        "best_epoch": int(result.best_epoch),
        "elapsed_time_s": float(time.time() - start),
        "checkpoint_best": "checkpoints/best.eqx",
        "checkpoint_last": "checkpoints/last.eqx",
        "objective": "negative_mean_log_prob_inferred_theta",
    }
    write_json(out / "inferred_prior_training_summary.json", summary)
    write_supervised_prior_diagnostics(
        truth=dataset.theta_frame(),
        prior=prior_frame,
        parameter_names=dataset.parameter_names,
        out_dir=out,
        summary=summary,
    )
    _log(verbose, f"[prior] inferred prior checkpoint: {ckpt_dir / 'best.eqx'}")


def load_inferred_theta_dataset(
    config: dict[str, Any],
    *,
    input_paths: tuple[str | Path, ...],
    limit: int | None = None,
    seed: int = 42,
) -> TruthDataset:
    """Load MAP/MCLMC theta rows and convert them to the configured latent space."""
    latent_spec = latent_spec_from_config(config)
    frames = []
    for path in input_paths:
        frame = _read_table(Path(path))
        frame["source_file"] = str(path)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        raise ValueError("No inferred theta rows found")
    if limit is not None and int(limit) > 0 and len(raw) > int(limit):
        raw = raw.sample(n=int(limit), random_state=int(seed)).reset_index(drop=True)
    names = tuple(str(name) for name in latent_spec.names)
    missing = [name for name in names if name not in raw]
    if missing:
        raise ValueError(
            "Inferred prior input is missing parameter columns: "
            + ", ".join(missing)
        )
    theta_frame = raw[list(names)].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(theta_frame.to_numpy(dtype=float)).all(axis=1)
    theta = theta_frame.loc[finite].to_numpy(dtype=np.float32)
    if theta.size == 0:
        raise ValueError("No finite inferred theta rows remain after filtering")
    x = np.asarray(theta_to_x(jnp.asarray(theta), latent_spec), dtype=np.float32)
    if "row_index" in raw:
        source_values = pd.to_numeric(raw.loc[finite, "row_index"], errors="coerce")
        source_rows_float = source_values.to_numpy(dtype=float)
        fallback = np.arange(int(finite.sum()), dtype=float)
        source_rows = np.where(np.isfinite(source_rows_float), source_rows_float, fallback)
        source_rows = source_rows.astype(np.int64)
    else:
        source_rows = np.arange(int(finite.sum()), dtype=np.int64)
    object_id = (
        raw.loc[finite, "object_id"].to_numpy()
        if "object_id" in raw
        else source_rows
    )
    lower = np.asarray(latent_spec.lower, dtype=float)
    upper = np.asarray(latent_spec.upper, dtype=float)
    schema = TruthSchema(
        name="inferred_map_mclmc",
        parameters=tuple(
            ParameterSpec(
                name=name,
                column=name,
                semantic="inferred_theta",
                lower=float(lower[index]),
                upper=float(upper[index]),
            )
            for index, name in enumerate(names)
        ),
        missing_columns=(),
        reduced=False,
    )
    return TruthDataset(
        schema=schema,
        latent_spec=latent_spec,
        theta=theta,
        x=x,
        object_id=np.asarray(object_id),
        source_rows=np.asarray(source_rows, dtype=np.int64),
        dropped_rows=int((~finite).sum()),
        dataset_path=";".join(str(path) for path in input_paths),
    )


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message)
