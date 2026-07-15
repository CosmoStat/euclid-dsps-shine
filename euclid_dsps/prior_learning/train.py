"""Training loop for supervised Diffsky truth priors."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.amortized.latent import LatentSpec, latent_spec_to_jsonable
from euclid_dsps.io import ensure_dir, write_json

from .data import (
    TruthDataset,
    load_truth_dataset,
    load_truth_dataset_with_schema,
    prior_samples_frame,
    truth_dataset_with_latent_spec,
    truth_standardized_latent_spec,
)
from .diagnostics import write_supervised_prior_diagnostics
from .flows import (
    RealNVPPrior,
    RQSplineCouplingPrior,
    assert_flow_integrity,
    flow_integrity_diagnostics,
)
from .schema import ParameterSpec, TruthSchema
from .splits import train_validation_split as _train_validation_split

eqx, optax = require_amortized_dependencies()

PriorFlow = RealNVPPrior | RQSplineCouplingPrior


@dataclass(frozen=True)
class PriorTrainingResult:
    prior: PriorFlow
    last_prior: PriorFlow
    training_log: pd.DataFrame
    validation_log: pd.DataFrame
    initial_train_nll: float
    best_metric: float
    best_epoch: int
    data_parallel: dict[str, Any]
    best_selection_eligible: bool
    best_selection_diagnostics: dict[str, Any]


def prior_learning_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return supervised-prior config with local defaults."""
    raw = dict(config.get("prior_learning", {}) or {})
    raw.setdefault("dataset", config.get("catalog_path"))
    raw.setdefault("train_dataset", None)
    raw.setdefault("validation_dataset", None)
    raw.setdefault("test_dataset", None)
    raw.setdefault("schema", "diffsky_truth_basic")
    raw.setdefault("missing_policy", "reduce")
    raw.setdefault("bounds", {})
    raw.setdefault("flow", {})
    raw.setdefault("latent", {})
    raw.setdefault("training", {})
    raw.setdefault("snapshots", {})
    raw.setdefault("output", {})
    raw["flow"].setdefault("type", "realnvp")
    raw["flow"].setdefault("n_layers", 8)
    raw["flow"].setdefault("hidden_size", 128)
    raw["flow"].setdefault("scale_clamp", 0.05)
    raw["flow"].setdefault("shift_clamp", 5.0)
    raw["flow"].setdefault("n_bins", 16)
    raw["flow"].setdefault("tail_bound", 12.0)
    raw["flow"].setdefault("min_bin_width", 1.0e-3)
    raw["flow"].setdefault("min_bin_height", 1.0e-3)
    raw["flow"].setdefault("min_derivative", 1.0e-3)
    raw["flow"].setdefault("permutation", "roll")
    raw["flow"].setdefault("init", "default")
    raw["flow"].setdefault("init_scale", 1.0)
    raw["latent"].setdefault("normalization", "identity")
    raw["latent"].setdefault("min_raw_scale", 0.1)
    raw["latent"].setdefault("max_raw_scale", 10.0)
    raw["training"].setdefault("epochs", 20)
    raw["training"].setdefault("batch_size", 256)
    raw["training"].setdefault("learning_rate", 1.0e-3)
    raw["training"].setdefault("weight_decay", 1.0e-5)
    raw["training"].setdefault("gradient_clip_norm", 1.0)
    raw["training"].setdefault("validation_fraction", 0.1)
    raw["training"].setdefault("seed", 42)
    raw["training"].setdefault("epoch_shuffle", True)
    raw["training"].setdefault("data_parallel", "single")
    raw["snapshots"].setdefault("enabled", False)
    raw["snapshots"].setdefault("every_epochs", 5)
    raw["snapshots"].setdefault("include_epoch_zero", True)
    raw["snapshots"].setdefault("prior_samples", 10_000)
    raw["snapshots"].setdefault("truth_sample_limit", 10_000)
    raw["snapshots"].setdefault("write_corner", True)
    raw["snapshots"].setdefault("max_corner_rows", 4000)
    raw["snapshots"].setdefault("checkpoint_every", 5)
    raw["output"].setdefault("prior_samples", 8192)
    raw["output"].setdefault("truth_sample_limit", 200_000)
    return raw


def _normalize_supervised_prior_truth(
    truth: TruthDataset,
    latent_config: dict[str, Any] | None,
) -> tuple[TruthDataset, dict[str, Any]]:
    cfg = dict(latent_config or {})
    mode = str(cfg.get("normalization", "identity")).strip().lower()
    if mode in {"identity", "none", "raw_logit", "raw_logit_identity"}:
        return truth, latent_spec_to_jsonable(truth.latent_spec)
    if mode not in {
        "truth_standardized",
        "truth_standardized_logit",
        "standardized_truth_logit",
    }:
        raise ValueError(
            "prior_learning.latent.normalization must be 'identity' or "
            "'truth_standardized_logit'"
        )
    latent_spec, payload = truth_standardized_latent_spec(
        truth,
        min_raw_scale=float(cfg.get("min_raw_scale", 0.1)),
        max_raw_scale=float(cfg.get("max_raw_scale", 10.0)),
    )
    normalized = truth_dataset_with_latent_spec(truth, latent_spec)
    payload.update(
        {
            "network_x_abs_q99": float(np.quantile(np.abs(normalized.x).reshape(-1), 0.99)),
            "network_x_abs_max": float(np.max(np.abs(normalized.x))),
        }
    )
    return normalized, payload


def train_supervised_prior(
    config: dict[str, Any],
    out_dir: str | Path,
    *,
    dataset_path: str | Path | None = None,
    schema_name: str | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    seed: int | None = None,
    validation_fraction: float | None = None,
    missing_policy: str | None = None,
    row_indices_file: str | Path | None = None,
    verbose: bool = True,
    progress: bool = True,
) -> None:
    """Train ``p_beta(theta_true)`` directly from prepared truth columns."""
    out = ensure_dir(out_dir)
    cfg = prior_learning_config(config)
    explicit_train_dataset = cfg.get("train_dataset")
    dataset = Path(dataset_path or explicit_train_dataset or cfg["dataset"])
    schema = str(schema_name or cfg["schema"])
    policy = str(missing_policy or cfg["missing_policy"])
    training_cfg = dict(cfg["training"])
    if batch_size is not None:
        training_cfg["batch_size"] = int(batch_size)
    if epochs is not None:
        training_cfg["epochs"] = int(epochs)
    if seed is not None:
        training_cfg["seed"] = int(seed)
    if validation_fraction is not None:
        training_cfg["validation_fraction"] = float(validation_fraction)

    _log(verbose, "[prior] supervised Diffsky truth-prior training")
    _log(verbose, f"[prior] dataset: {dataset}")
    _log(verbose, f"[prior] schema: {schema} missing_policy={policy}")
    _log(verbose, f"[prior] output directory: {out}")
    write_json(out / "normalized_config.json", config)
    truth = load_truth_dataset(
        dataset,
        schema_name=schema,
        missing_policy=policy,
        bounds=cfg.get("bounds", {}),
        limit=limit,
        row_indices_file=row_indices_file,
    )
    truth, latent_normalization = _normalize_supervised_prior_truth(
        truth,
        cfg.get("latent", {}),
    )
    validation_truth = (
        load_truth_dataset_with_schema(
            cfg["validation_dataset"],
            schema=truth.schema,
            limit=limit,
            latent_spec=truth.latent_spec,
        )
        if cfg.get("validation_dataset")
        else None
    )
    test_truth = (
        load_truth_dataset_with_schema(
            cfg["test_dataset"],
            schema=truth.schema,
            limit=limit,
            latent_spec=truth.latent_spec,
        )
        if cfg.get("test_dataset")
        else None
    )
    _log(
        verbose,
        "[prior] truth matrix: "
        f"rows={truth.theta.shape[0]} dim={truth.theta.shape[1]} "
        f"dropped_nonfinite={truth.dropped_rows}",
    )
    truth.theta_frame().head(int(cfg["output"].get("truth_sample_limit", 200_000))).to_parquet(
        out / "truth_theta_samples.parquet",
        index=False,
    )
    truth.x_frame().head(int(cfg["output"].get("truth_sample_limit", 200_000))).to_parquet(
        out / "truth_x_samples.parquet",
        index=False,
    )
    if validation_truth is None:
        split = _train_validation_split(
            len(truth.x),
            validation_fraction=float(training_cfg.get("validation_fraction", 0.1)),
            seed=int(training_cfg.get("seed", 42)),
        )
        x_train = truth.x[split["train"]]
        x_validation = truth.x[split["validation"]] if len(split["validation"]) else None
        validation_rows = int(len(split["validation"]))
    else:
        split = {
            "train": np.arange(len(truth.x), dtype=np.int64),
            "validation": np.arange(len(validation_truth.x), dtype=np.int64),
        }
        x_train = truth.x
        x_validation = validation_truth.x
        validation_rows = int(len(validation_truth.x))
    np.save(out / "train_indices.npy", split["train"])
    np.save(out / "validation_indices.npy", split["validation"])
    _log(
        verbose,
        "[prior] split: "
        f"train={len(split['train'])} validation={validation_rows}",
    )
    _log(
        verbose,
        "[prior] flow: "
        f"type={cfg['flow'].get('type')} "
        f"layers={cfg['flow'].get('n_layers')} "
        f"hidden={cfg['flow'].get('hidden_size')} "
        f"init={cfg['flow'].get('init')}",
    )
    _log(
        verbose,
        "[prior] training: "
        f"epochs={training_cfg.get('epochs')} "
        f"batch_size={training_cfg.get('batch_size')} "
        f"data_parallel={training_cfg.get('data_parallel')} "
        "first epoch includes JAX compilation",
    )
    write_json(
        out / "prior_training_progress.json",
        {
            "status": "starting",
            "completed_epochs": 0,
            "total_epochs": int(training_cfg.get("epochs", 20)),
            "flow": dict(cfg["flow"]),
            "data_parallel_requested": str(training_cfg.get("data_parallel", "single")),
        },
    )
    start = time.time()
    ckpt_dir = ensure_dir(out / "checkpoints")
    snapshot_cfg = dict(cfg.get("snapshots", {}) or {})
    progress_callback = _supervised_prior_progress_callback(
        out,
        verbose=verbose,
        progress=bool(progress),
        total_epochs=int(training_cfg.get("epochs", 20)),
        start_time=start,
    )
    epoch_callback = _supervised_prior_epoch_callback(
        out,
        config=config,
        truth=truth,
        flow_config=cfg["flow"],
        snapshot_config=snapshot_cfg,
        seed=int(training_cfg.get("seed", 42)),
        checkpoint_dir=ckpt_dir,
    )
    result = fit_realnvp_to_x(
        x_train,
        x_validation,
        latent_dim=truth.x.shape[1],
        flow_config=cfg["flow"],
        training_config=training_cfg,
        seed=int(training_cfg.get("seed", 42)),
        epoch_callback=epoch_callback,
        progress_callback=progress_callback,
    )
    result.training_log.to_csv(out / "prior_training_log.csv", index=False)
    result.validation_log.to_csv(out / "prior_validation_loglike.csv", index=False)
    integrity = assert_flow_integrity(
        result.prior,
        context=f"supervised prior training output {out}",
        key=jax.random.PRNGKey(int(training_cfg.get("seed", 42)) + 97),
    )
    save_prior_checkpoint(
        ckpt_dir / "best.eqx",
        result.prior,
        config=config,
        truth=truth,
        flow_config=cfg["flow"],
        epoch=result.best_epoch,
        metric=result.best_metric,
    )
    save_prior_checkpoint(
        ckpt_dir / "last.eqx",
        result.last_prior,
        config=config,
        truth=truth,
        flow_config=cfg["flow"],
        epoch=int(training_cfg.get("epochs", 20)),
        metric=float(result.training_log["loss"].iloc[-1]),
    )
    sample_count = int(cfg["output"].get("prior_samples", 8192))
    key = jax.random.PRNGKey(int(training_cfg.get("seed", 42)) + 1)
    x_prior = np.asarray(result.prior.sample(key, sample_count), dtype=np.float32)
    log_prob = np.asarray(result.prior.log_prob(jnp.asarray(x_prior)), dtype=float)
    prior_frame = prior_samples_frame(x_prior, truth.latent_spec, log_prob=log_prob)
    prior_frame.to_parquet(out / "learned_prior_samples.parquet", index=False)
    final_train_nll = float(_prior_nll_jit(result.prior, jnp.asarray(x_train)))
    final_validation_nll = (
        None
        if x_validation is None
        else float(_prior_nll_jit(result.prior, jnp.asarray(x_validation)))
    )
    final_test_nll = (
        None
        if test_truth is None
        else float(_prior_nll_jit(result.prior, jnp.asarray(test_truth.x)))
    )
    summary = {
        "dataset": str(dataset),
        "train_dataset": str(dataset),
        "validation_dataset": str(cfg.get("validation_dataset") or ""),
        "test_dataset": str(cfg.get("test_dataset") or ""),
        "row_indices_file": str(row_indices_file) if row_indices_file else None,
        "schema": truth.schema.to_dict(),
        "latent_normalization": latent_normalization,
        "parameter_names": list(truth.parameter_names),
        "n_truth_rows": int(truth.theta.shape[0]),
        "dropped_nonfinite_rows": int(truth.dropped_rows),
        "train_rows": int(len(split["train"])),
        "validation_rows": validation_rows,
        "test_rows": 0 if test_truth is None else int(len(test_truth.x)),
        "epochs": int(training_cfg.get("epochs", 20)),
        "batch_size": int(training_cfg.get("batch_size", 256)),
        "data_parallel": {
            "requested": str(training_cfg.get("data_parallel", "single")),
            "effective": str(result.data_parallel["effective"]),
            "enabled": bool(result.data_parallel["enabled"]),
            "local_device_count": int(result.data_parallel["n_devices"]),
            "global_batch_size": int(result.data_parallel["global_batch_size"]),
            "per_device_batch_size": int(result.data_parallel["per_device_batch_size"]),
        },
        "initial_train_nll": float(result.initial_train_nll),
        "final_train_nll": final_train_nll,
        "final_validation_nll": final_validation_nll,
        "final_test_nll": final_test_nll,
        "best_metric": float(result.best_metric),
        "best_epoch": int(result.best_epoch),
        "elapsed_time_s": float(time.time() - start),
        "checkpoint_best": "checkpoints/best.eqx",
        "checkpoint_last": "checkpoints/last.eqx",
        "objective": "negative_mean_log_prob_on_truth_x",
        "objective_distribution_direction": "empirical_truth_to_model_mle",
        "equivalent_kl_note": (
            "minimizes KL(p_truth || p_beta) up to the entropy of p_truth"
        ),
        "flow_integrity": integrity,
        "realnvp_integrity": integrity if isinstance(result.prior, RealNVPPrior) else None,
        "snapshots": {
            "enabled": bool(snapshot_cfg.get("enabled", False)),
            "every_epochs": int(snapshot_cfg.get("every_epochs", 5)),
            "checkpoint_every": int(snapshot_cfg.get("checkpoint_every", 5)),
        },
    }
    write_supervised_prior_diagnostics(
        truth=truth.theta_frame(),
        prior=prior_frame,
        parameter_names=truth.parameter_names,
        out_dir=out,
        summary=summary,
    )
    _log(verbose, "[prior] training complete")
    _log(verbose, f"[prior] best checkpoint: {ckpt_dir / 'best.eqx'}")
    _log(verbose, f"[prior] report: {out / 'supervised_prior_vs_truth_report.md'}")


def _supervised_prior_epoch_callback(
    out: Path,
    *,
    config: dict[str, Any],
    truth: TruthDataset,
    flow_config: dict[str, Any],
    snapshot_config: dict[str, Any],
    seed: int,
    checkpoint_dir: Path,
) -> Callable[[int, PriorFlow], None] | None:
    enabled = bool(snapshot_config.get("enabled", False))
    checkpoint_every = int(snapshot_config.get("checkpoint_every", 0) or 0)
    if not enabled and checkpoint_every <= 0:
        return None

    def callback(epoch: int, prior: PriorFlow) -> None:
        if _should_write_supervised_prior_snapshot(snapshot_config, epoch):
            _write_supervised_prior_snapshot(
                out,
                epoch=epoch,
                prior=prior,
                truth=truth,
                snapshot_config=snapshot_config,
                seed=seed,
            )
        if checkpoint_every > 0 and int(epoch) > 0 and int(epoch) % checkpoint_every == 0:
            save_prior_checkpoint(
                checkpoint_dir / f"epoch_{int(epoch):04d}.eqx",
                prior,
                config=config,
                truth=truth,
                flow_config=flow_config,
                epoch=int(epoch),
                metric=float(_prior_nll_jit(prior, jnp.asarray(truth.x))),
                strict_integrity=False,
            )

    return callback


def _should_write_supervised_prior_snapshot(
    snapshot_config: dict[str, Any],
    epoch: int,
) -> bool:
    if not bool(snapshot_config.get("enabled", False)):
        return False
    if int(epoch) == 0:
        return bool(snapshot_config.get("include_epoch_zero", True))
    every_epochs = int(snapshot_config.get("every_epochs", 5) or 0)
    return every_epochs > 0 and int(epoch) % every_epochs == 0


def _supervised_prior_progress_callback(
    out: Path,
    *,
    verbose: bool,
    progress: bool,
    total_epochs: int,
    start_time: float,
) -> Callable[[dict[str, Any]], None]:
    path = out / "prior_training_progress.json"
    rows: list[dict[str, Any]] = []

    def callback(record: dict[str, Any]) -> None:
        rows.append(dict(record))
        payload = {
            "total_epochs": int(total_epochs),
            "completed_epochs": int(record["epoch"]),
            "fraction_complete": float(int(record["epoch"]) / max(int(total_epochs), 1)),
            "elapsed_time_s": float(time.time() - start_time),
            "best_epoch": int(record["best_epoch"]),
            "best_metric": float(record["best_metric"]),
            "last": record,
            "epochs": rows,
        }
        write_json(path, payload)
        if verbose and progress:
            _log(verbose, _prior_epoch_progress_line(record, int(total_epochs)))

    return callback


def _prior_epoch_progress_line(record: dict[str, Any], total_epochs: int) -> str:
    val = record.get("validation_negative_mean_log_prob")
    val_text = "nan" if val is None else f"{float(val):.6g}"
    return (
        "[prior] "
        f"epoch={int(record['epoch'])}/{int(total_epochs)} "
        f"train_nll={float(record['train_negative_mean_log_prob']):.6g} "
        f"val_nll={val_text} "
        f"best={float(record['best_metric']):.6g}@{int(record['best_epoch'])} "
        f"grad={float(record['mean_grad_norm']):.3g} "
        f"updates={float(record['update_fraction']):.3f}"
    )


def _write_supervised_prior_snapshot(
    out: Path,
    *,
    epoch: int,
    prior: PriorFlow,
    truth: TruthDataset,
    snapshot_config: dict[str, Any],
    seed: int,
) -> None:
    snap = ensure_dir(out / "snapshots" / f"epoch_{int(epoch):04d}")
    key = jax.random.PRNGKey(int(seed) + 10_000 + int(epoch))
    n_prior = max(int(snapshot_config.get("prior_samples", 10_000)), 1)
    x_prior = np.asarray(jax.device_get(prior.sample(key, n_prior)), dtype=np.float32)
    log_prob = np.asarray(prior.log_prob(jnp.asarray(x_prior)), dtype=float)
    prior_frame = prior_samples_frame(x_prior, truth.latent_spec, log_prob=log_prob)
    prior_frame.to_parquet(snap / "prior_samples.parquet", index=False)

    truth_frame = _snapshot_truth_sample(
        truth,
        limit=max(int(snapshot_config.get("truth_sample_limit", 10_000)), 1),
        seed=int(seed) + int(epoch),
    )
    truth_frame.to_parquet(snap / "truth_samples.parquet", index=False)

    diagnostics = write_supervised_prior_diagnostics(
        truth=truth_frame,
        prior=prior_frame,
        parameter_names=truth.parameter_names,
        out_dir=snap,
        summary={
            "epoch": int(epoch),
            "objective": "negative_mean_log_prob_on_truth_x",
            "objective_distribution_direction": "empirical_truth_to_model_mle",
            "equivalent_kl_note": (
                "minimizes KL(p_truth || p_beta) up to the entropy of p_truth"
            ),
            "prior_samples": int(len(prior_frame)),
            "truth_samples": int(len(truth_frame)),
        },
        max_corner_rows=int(snapshot_config.get("max_corner_rows", 4000)),
    )
    write_json(
        snap / "snapshot_summary.json",
        {
            "epoch": int(epoch),
            "prior_samples": int(len(prior_frame)),
            "truth_samples": int(len(truth_frame)),
            "write_corner": bool(snapshot_config.get("write_corner", True)),
            "diagnostics": diagnostics,
        },
    )


def _snapshot_truth_sample(
    truth: TruthDataset,
    *,
    limit: int,
    seed: int,
) -> pd.DataFrame:
    frame = truth.theta_frame()
    if len(frame) <= int(limit):
        return frame
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(len(frame), size=int(limit), replace=False))
    return frame.iloc[indices].reset_index(drop=True)


def fit_realnvp_to_x(
    x_train: np.ndarray,
    x_validation: np.ndarray | None,
    *,
    latent_dim: int,
    flow_config: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
    epoch_callback: Callable[[int, PriorFlow], None] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    selection_callback: Callable[
        [int, PriorFlow, float], dict[str, Any] | None
    ]
    | None = None,
    initial_prior: PriorFlow | None = None,
    initial_epoch: int = 0,
) -> PriorTrainingResult:
    """Fit the configured exact flow prior to an unconstrained truth matrix."""
    x_train = np.asarray(x_train, dtype=np.float32)
    if x_train.ndim != 2 or x_train.shape[1] != int(latent_dim):
        raise ValueError("x_train must have shape [n, latent_dim]")
    x_validation = (
        None
        if x_validation is None or len(x_validation) == 0
        else np.asarray(x_validation, dtype=np.float32)
    )
    initial_epoch = int(initial_epoch)
    if initial_epoch < 0:
        raise ValueError("initial_epoch must be non-negative")
    key = jax.random.PRNGKey(int(seed))
    prior = (
        _build_prior_from_flow_config(
            key,
            latent_dim=int(latent_dim),
            flow_config=flow_config,
        )
        if initial_prior is None
        else initial_prior
    )
    if int(prior.latent_dim) != int(latent_dim):
        raise ValueError("initial_prior has the wrong latent dimension")
    prior = _cast_inexact_arrays(prior, jnp.float32)
    optimizer = _make_optimizer(training_config)
    opt_state = optimizer.init(eqx.filter(prior, eqx.is_inexact_array))
    batch_size = max(int(training_config.get("batch_size", 256)), 1)
    epochs = max(int(training_config.get("epochs", 20)), 1)
    if initial_epoch >= epochs:
        raise ValueError("initial_epoch must be smaller than the target epochs")
    epoch_shuffle = bool(training_config.get("epoch_shuffle", True))
    base_moment_weight = float(training_config.get("base_moment_weight", 0.0))
    training_jitter_std = float(training_config.get("training_jitter_std", 0.0))
    early_stopping_patience = int(training_config.get("early_stopping_patience", 0))
    if training_jitter_std < 0.0:
        raise ValueError("training_jitter_std must be non-negative")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    data_parallel = _resolve_data_parallel_training(
        training_config,
        batch_size=batch_size,
    )
    pmap_train_step = None
    prior_replicated = None
    opt_state_replicated = None
    if bool(data_parallel["enabled"]):
        pmap_train_step = _make_prior_pmap_train_step(
            optimizer,
            base_moment_weight=base_moment_weight,
        )
        prior_replicated = _replicate_tree(prior, data_parallel["devices"])
        opt_state_replicated = _replicate_tree(opt_state, data_parallel["devices"])
    rng = np.random.default_rng(int(seed) + 100 + initial_epoch)
    initial_train_nll = float(_prior_nll_jit(prior, jnp.asarray(x_train)))
    best_metric = float("inf")
    best_epoch = 0
    best_prior = prior
    best_selection_eligible = False
    best_selection_diagnostics: dict[str, Any] = {}
    rows: list[dict[str, float | int | str]] = []
    val_rows: list[dict[str, float | int | str]] = []
    if epoch_callback is not None:
        epoch_callback(initial_epoch, prior)
    evaluations_since_best = 0
    if selection_callback is not None:
        initial_validation_nll = (
            initial_train_nll
            if x_validation is None
            else float(_prior_nll_jit(prior, jnp.asarray(x_validation)))
        )
        initial_selection = selection_callback(
            initial_epoch, prior, initial_validation_nll
        )
        if initial_selection is not None:
            initial_diagnostics = dict(initial_selection)
            initial_metric = float(initial_diagnostics.pop("metric"))
            initial_eligible = bool(initial_diagnostics.pop("eligible", False))
            best_metric = initial_metric
            best_epoch = initial_epoch
            best_prior = prior
            best_selection_eligible = initial_eligible
            best_selection_diagnostics = dict(initial_diagnostics)
            if x_validation is not None:
                val_rows.append(
                    {
                        "epoch": initial_epoch,
                        "split": "validation",
                        "mean_log_prob": -initial_validation_nll,
                        "negative_mean_log_prob": initial_validation_nll,
                        "n_objects": int(len(x_validation)),
                        "selection_evaluated": True,
                        "selection_eligible": initial_eligible,
                        "selection_metric": initial_metric,
                        **initial_diagnostics,
                    }
                )
    elif initial_prior is not None:
        initial_validation_nll = (
            initial_train_nll
            if x_validation is None
            else float(_prior_nll_jit(prior, jnp.asarray(x_validation)))
        )
        best_metric = initial_validation_nll
        best_epoch = initial_epoch
        best_prior = prior
        best_selection_eligible = True
        if x_validation is not None:
            val_rows.append(
                {
                    "epoch": initial_epoch,
                    "split": "validation",
                    "mean_log_prob": -initial_validation_nll,
                    "negative_mean_log_prob": initial_validation_nll,
                    "n_objects": int(len(x_validation)),
                    "selection_evaluated": True,
                    "selection_eligible": True,
                    "selection_metric": initial_validation_nll,
                    "resumed_checkpoint": True,
                }
            )
    for epoch in range(initial_epoch + 1, epochs + 1):
        order = np.arange(len(x_train))
        if epoch_shuffle:
            rng.shuffle(order)
        order, padded_rows = _pad_epoch_order_for_data_parallel(
            order,
            global_batch_size=batch_size,
            enabled=bool(data_parallel["enabled"]),
            rng=rng,
        )
        epoch_losses = []
        epoch_nlls = []
        epoch_base_penalties = []
        epoch_grad_norms = []
        epoch_updates = []
        for batch_index, start in enumerate(range(0, len(order), batch_size)):
            batch_values = x_train[order[start : start + batch_size]]
            if training_jitter_std > 0.0:
                batch_values = batch_values + rng.normal(
                    scale=training_jitter_std,
                    size=batch_values.shape,
                ).astype(np.float32)
            batch = jnp.asarray(batch_values)
            if bool(data_parallel["enabled"]):
                if (
                    pmap_train_step is None
                    or prior_replicated is None
                    or opt_state_replicated is None
                ):
                    raise RuntimeError("pmap prior training state was not initialized")
                sharded_batch = _shard_x_batch(batch, int(data_parallel["n_devices"]))
                (
                    prior_replicated,
                    opt_state_replicated,
                    loss_value,
                    mean_log_prob_value,
                    base_penalty_value,
                    grad_norm_value,
                    loss_finite_value,
                    grads_finite_value,
                    update_applied_value,
                ) = pmap_train_step(
                    prior_replicated,
                    opt_state_replicated,
                    sharded_batch,
                )
                loss = _unreplicate_scalar(loss_value)
                mean_log_prob = _unreplicate_scalar(mean_log_prob_value)
                base_penalty = _unreplicate_scalar(base_penalty_value)
                grad_norm = _unreplicate_scalar(grad_norm_value)
                loss_finite = _unreplicate_bool(loss_finite_value)
                grads_finite = _unreplicate_bool(grads_finite_value)
                update_applied = _unreplicate_bool(update_applied_value)
            else:
                (loss_raw, aux_raw), grads = _loss_and_grads_jit(
                    prior,
                    batch,
                    jnp.asarray(base_moment_weight, dtype=batch.dtype),
                )
                mean_log_prob_raw, base_penalty_raw = aux_raw
                loss = float(loss_raw)
                mean_log_prob = float(mean_log_prob_raw)
                base_penalty = float(base_penalty_raw)
                loss_finite = bool(np.isfinite(loss))
                grads_finite = _tree_all_finite(grads)
                update_applied = bool(loss_finite and grads_finite)
                grad_norm = float(_tree_l2_norm(grads))
                if update_applied:
                    updates, opt_state = optimizer.update(
                        grads,
                        opt_state,
                        eqx.filter(prior, eqx.is_inexact_array),
                    )
                    prior = eqx.apply_updates(prior, updates)
            epoch_losses.append(float(loss))
            epoch_nlls.append(float(-mean_log_prob))
            epoch_base_penalties.append(float(base_penalty))
            epoch_grad_norms.append(float(grad_norm))
            epoch_updates.append(float(update_applied))
            rows.append(
                {
                    "epoch": int(epoch),
                    "batch": int(batch_index),
                    "split": "train",
                    "loss": float(loss),
                    "mean_log_prob": float(mean_log_prob),
                    "negative_mean_log_prob": float(-mean_log_prob),
                    "base_moment_penalty": float(base_penalty),
                    "base_moment_weight": float(base_moment_weight),
                    "n_objects": int(batch.shape[0]),
                    "data_parallel_mode": str(data_parallel["effective"]),
                    "data_parallel_devices": int(data_parallel["n_devices"]),
                    "data_parallel_per_device_batch_size": int(
                        data_parallel["per_device_batch_size"]
                    ),
                    "data_parallel_epoch_padded_rows": int(padded_rows),
                    "loss_finite": float(loss_finite),
                    "grads_finite": float(grads_finite),
                    "update_applied": float(update_applied),
                    "grad_norm": float(grad_norm),
                }
            )
        if bool(data_parallel["enabled"]):
            prior = _unreplicate_tree(prior_replicated)
        train_metric = float(np.nanmean(epoch_nlls))
        if x_validation is not None:
            val_log_prob = float(
                np.mean(np.asarray(prior.log_prob(jnp.asarray(x_validation))))
            )
            val_metric = -val_log_prob
            metric = val_metric
        else:
            metric = train_metric
        selection_diagnostics: dict[str, Any] = {}
        selection_eligible = True
        selection_evaluated = selection_callback is None
        if selection_callback is not None:
            selection = selection_callback(epoch, prior, float(metric))
            if selection is not None:
                selection_diagnostics = dict(selection)
                metric = float(selection_diagnostics.pop("metric"))
                selection_eligible = bool(
                    selection_diagnostics.pop("eligible", False)
                )
                selection_evaluated = True
            else:
                selection_eligible = False
        if x_validation is not None:
            val_rows.append(
                {
                    "epoch": int(epoch),
                    "split": "validation",
                    "mean_log_prob": float(val_log_prob),
                    "negative_mean_log_prob": float(val_metric),
                    "n_objects": int(len(x_validation)),
                    "selection_evaluated": bool(selection_evaluated),
                    "selection_eligible": bool(selection_eligible),
                    "selection_metric": (
                        float(metric) if selection_evaluated else np.nan
                    ),
                    **selection_diagnostics,
                }
            )
        candidate_rank = (0 if selection_eligible else 1, float(metric))
        best_rank = (0 if best_selection_eligible else 1, float(best_metric))
        improved = bool(
            selection_evaluated and np.isfinite(metric) and candidate_rank < best_rank
        )
        if improved:
            best_metric = float(metric)
            best_epoch = int(epoch)
            best_prior = prior
            best_selection_eligible = bool(selection_eligible)
            best_selection_diagnostics = dict(selection_diagnostics)
            evaluations_since_best = 0
        elif selection_evaluated:
            evaluations_since_best += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "epoch": int(epoch),
                    "train_negative_mean_log_prob": float(train_metric),
                    "validation_negative_mean_log_prob": (
                        None if x_validation is None else float(val_metric)
                    ),
                    "metric": float(metric),
                    "best_metric": float(best_metric),
                    "best_epoch": int(best_epoch),
                    "mean_grad_norm": float(np.nanmean(epoch_grad_norms)),
                    "mean_base_moment_penalty": float(
                        np.nanmean(epoch_base_penalties)
                    ),
                    "selection_evaluated": bool(selection_evaluated),
                    "selection_eligible": bool(selection_eligible),
                    **selection_diagnostics,
                    "update_fraction": float(np.nanmean(epoch_updates)),
                    "padded_rows": int(padded_rows),
                    "data_parallel_mode": str(data_parallel["effective"]),
                    "data_parallel_devices": int(data_parallel["n_devices"]),
                }
            )
        if epoch_callback is not None:
            epoch_callback(epoch, prior)
        if (
            early_stopping_patience > 0
            and evaluations_since_best >= early_stopping_patience
        ):
            break
    return PriorTrainingResult(
        prior=best_prior,
        last_prior=prior,
        training_log=pd.DataFrame(rows),
        validation_log=pd.DataFrame(val_rows),
        initial_train_nll=initial_train_nll,
        best_metric=best_metric,
        best_epoch=best_epoch,
        data_parallel=data_parallel,
        best_selection_eligible=best_selection_eligible,
        best_selection_diagnostics=best_selection_diagnostics,
    )


def _build_prior_from_flow_config(
    key,
    *,
    latent_dim: int,
    flow_config: dict[str, Any],
) -> PriorFlow:
    flow_type = _normalize_flow_type(flow_config.get("type", "realnvp"))
    if flow_type == "realnvp":
        return RealNVPPrior(
            key,
            latent_dim=int(latent_dim),
            n_layers=int(flow_config.get("n_layers", 8)),
            hidden_size=int(flow_config.get("hidden_size", 128)),
            scale_clamp=float(flow_config.get("scale_clamp", 0.05)),
            shift_clamp=float(flow_config.get("shift_clamp", 5.0)),
            permutation=str(flow_config.get("permutation", "none")),
            init=str(flow_config.get("init", "default")),
            init_scale=float(flow_config.get("init_scale", 1.0)),
        )
    if flow_type == "rq_spline_coupling":
        return RQSplineCouplingPrior(
            key,
            latent_dim=int(latent_dim),
            n_layers=int(flow_config.get("n_layers", 12)),
            hidden_size=int(flow_config.get("hidden_size", 256)),
            n_bins=int(flow_config.get("n_bins", 16)),
            tail_bound=float(flow_config.get("tail_bound", 12.0)),
            min_bin_width=float(flow_config.get("min_bin_width", 1.0e-3)),
            min_bin_height=float(flow_config.get("min_bin_height", 1.0e-3)),
            min_derivative=float(flow_config.get("min_derivative", 1.0e-3)),
            permutation=str(flow_config.get("permutation", "roll")),
            init=str(flow_config.get("init", "default")),
            init_scale=float(flow_config.get("init_scale", 1.0)),
        )
    raise ValueError(
        "prior_learning.flow.type must be one of: realnvp, rq_spline_coupling"
    )


def _prior_architecture_payload(
    prior: PriorFlow,
    flow_config: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(prior, RealNVPPrior):
        return {
            "type": "realnvp",
            "latent_dim": int(prior.latent_dim),
            "n_layers": int(flow_config.get("n_layers", len(prior.layers))),
            "hidden_size": int(flow_config.get("hidden_size", 128)),
            "scale_clamp": float(flow_config.get("scale_clamp", 0.05)),
            "shift_clamp": float(flow_config.get("shift_clamp", 5.0)),
            "permutation": str(getattr(prior, "permutation", "none")),
            "init": str(flow_config.get("init", "default")),
            "init_scale": float(flow_config.get("init_scale", 1.0)),
            "parameter_dtype": "float32",
        }
    if isinstance(prior, RQSplineCouplingPrior):
        return {
            "type": "rq_spline_coupling",
            "latent_dim": int(prior.latent_dim),
            "n_layers": int(flow_config.get("n_layers", len(prior.layers))),
            "hidden_size": int(flow_config.get("hidden_size", 256)),
            "n_bins": int(prior.n_bins),
            "tail_bound": float(prior.tail_bound),
            "min_bin_width": float(prior.min_bin_width),
            "min_bin_height": float(prior.min_bin_height),
            "min_derivative": float(prior.min_derivative),
            "permutation": str(prior.permutation),
            "init": str(flow_config.get("init", "default")),
            "init_scale": float(flow_config.get("init_scale", 1.0)),
            "parameter_dtype": "float32",
        }
    raise TypeError(f"Unsupported prior checkpoint type: {type(prior)!r}")


def _prior_template_from_architecture(arch: dict[str, Any]) -> PriorFlow:
    flow_type = _normalize_flow_type(arch.get("type", "realnvp"))
    if flow_type == "realnvp":
        return RealNVPPrior(
            jax.random.PRNGKey(0),
            latent_dim=int(arch["latent_dim"]),
            n_layers=int(arch["n_layers"]),
            hidden_size=int(arch["hidden_size"]),
            scale_clamp=float(arch["scale_clamp"]),
            shift_clamp=float(arch.get("shift_clamp", 5.0)),
            permutation=str(arch.get("permutation", "none")),
            init=str(arch.get("init", "default")),
            init_scale=float(arch.get("init_scale", 1.0)),
        )
    if flow_type == "rq_spline_coupling":
        return RQSplineCouplingPrior(
            jax.random.PRNGKey(0),
            latent_dim=int(arch["latent_dim"]),
            n_layers=int(arch["n_layers"]),
            hidden_size=int(arch["hidden_size"]),
            n_bins=int(arch.get("n_bins", 16)),
            tail_bound=float(arch.get("tail_bound", 12.0)),
            min_bin_width=float(arch.get("min_bin_width", 1.0e-3)),
            min_bin_height=float(arch.get("min_bin_height", 1.0e-3)),
            min_derivative=float(arch.get("min_derivative", 1.0e-3)),
            permutation=str(arch.get("permutation", "roll")),
            init=str(arch.get("init", "default")),
            init_scale=float(arch.get("init_scale", 1.0)),
        )
    raise ValueError(f"Unsupported prior checkpoint architecture type: {arch.get('type')}")


def _normalize_flow_type(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "real_nvp": "realnvp",
        "nvp": "realnvp",
        "spline": "rq_spline_coupling",
        "neural_spline": "rq_spline_coupling",
        "neural_spline_flow": "rq_spline_coupling",
        "rqspline": "rq_spline_coupling",
        "rq_spline": "rq_spline_coupling",
        "rq_spline_coupling": "rq_spline_coupling",
        "rational_quadratic_spline": "rq_spline_coupling",
    }
    return aliases.get(normalized, normalized)


def save_prior_checkpoint(
    path: str | Path,
    prior: PriorFlow,
    *,
    config: dict[str, Any],
    truth: TruthDataset,
    flow_config: dict[str, Any],
    epoch: int,
    metric: float,
    strict_integrity: bool = True,
) -> None:
    """Serialize a supervised exact flow prior and its truth schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = _cast_inexact_arrays(prior, jnp.float32)
    integrity = _checkpoint_integrity_payload(
        prior,
        path=path,
        strict=bool(strict_integrity),
    )
    eqx.tree_serialise_leaves(path, prior)
    sidecar = {
        "epoch": int(epoch),
        "metric": float(metric),
        "flow_integrity": integrity,
        "realnvp_integrity": integrity,
        "architecture": _prior_architecture_payload(prior, flow_config),
        "latent_spec": latent_spec_to_jsonable(truth.latent_spec),
        "schema": truth.schema.to_dict(),
        "source_dataset": truth.dataset_path,
        "prior_learning": prior_learning_config(config),
    }
    write_json(path.with_suffix(path.suffix + ".json"), sidecar)


def _checkpoint_integrity_payload(
    prior: PriorFlow,
    *,
    path: Path,
    strict: bool,
) -> dict[str, Any]:
    context = f"prior checkpoint save {path}"
    if strict:
        return assert_flow_integrity(
            prior,
            context=context,
            sample_count=64,
        )
    diagnostics = flow_integrity_diagnostics(prior, sample_count=64)
    if diagnostics["status"] == "FAIL":
        diagnostics = {
            **diagnostics,
            "strict": False,
            "warning": (
                "Intermediate checkpoint failed flow integrity diagnostics. "
                "It is kept for debugging and will be rejected by strict loaders."
            ),
        }
    return diagnostics


def load_prior_checkpoint(
    path: str | Path,
) -> tuple[PriorFlow, dict[str, Any], LatentSpec, TruthSchema]:
    """Load a supervised prior checkpoint plus schema metadata."""
    path = Path(path)
    sidecar_path = path.with_suffix(path.suffix + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    arch = sidecar["architecture"]
    template = _prior_template_from_architecture(arch)
    template = _cast_inexact_arrays(
        template,
        _jax_dtype_from_name(str(arch.get("parameter_dtype", "float32"))),
    )
    try:
        prior = eqx.tree_deserialise_leaves(path, template)
    except RuntimeError as exc:
        _raise_realnvp_mask_checkpoint_error(path, exc)
    assert_flow_integrity(
        prior,
        context=f"prior checkpoint load {path}",
        sample_count=64,
    )
    latent = sidecar["latent_spec"]
    latent_spec = LatentSpec(
        names=tuple(latent["names"]),
        lower=jnp.asarray(latent["lower"], dtype=jnp.float32),
        upper=jnp.asarray(latent["upper"], dtype=jnp.float32),
        raw_center=jnp.asarray(
            latent.get("raw_center", [0.0] * len(latent["names"])),
            dtype=jnp.float32,
        ),
        raw_scale=jnp.asarray(
            latent.get("raw_scale", [1.0] * len(latent["names"])),
            dtype=jnp.float32,
        ),
        normalization=str(latent.get("normalization", "identity")),
    )
    schema_payload = sidecar["schema"]
    schema = TruthSchema(
        name=str(schema_payload["name"]),
        parameters=tuple(
            ParameterSpec(
                name=str(param["name"]),
                column=str(param["column"]),
                semantic=str(param["semantic"]),
                lower=float(param["lower"]) if param.get("lower") is not None else None,
                upper=float(param["upper"]) if param.get("upper") is not None else None,
            )
            for param in schema_payload["parameters"]
        ),
        missing_columns=tuple(str(col) for col in schema_payload.get("missing_columns", [])),
        reduced=bool(schema_payload.get("reduced", False)),
    )
    return prior, sidecar, latent_spec, schema


def _raise_realnvp_mask_checkpoint_error(path: Path, exc: RuntimeError):
    message = str(exc)
    if "mask" in message and "changed dtype from bool" in message:
        raise RuntimeError(
            "This RealNVP prior checkpoint contains float coupling masks from "
            "an older buggy training path where masks were trainable. It is not "
            f"scientifically usable and must be retrained with the fixed code: {path}"
        ) from exc
    raise exc


def _cast_inexact_arrays(tree, dtype):
    return jax.tree_util.tree_map(
        lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf,
        tree,
    )


def _jax_dtype_from_name(name: str):
    normalized = str(name).strip().lower()
    if normalized in {"float32", "f32"}:
        return jnp.float32
    if normalized in {"float64", "f64"}:
        return jnp.float64
    if normalized in {"float16", "f16"}:
        return jnp.float16
    if normalized in {"bfloat16", "bf16"}:
        return jnp.bfloat16
    raise ValueError(f"Unsupported prior checkpoint parameter_dtype: {name}")


def _make_optimizer(training_config: dict[str, Any]):
    transforms = []
    clip = float(training_config.get("gradient_clip_norm", 1.0))
    if clip > 0.0:
        transforms.append(optax.clip_by_global_norm(clip))
    transforms.append(
        optax.adamw(
            learning_rate=float(training_config.get("learning_rate", 1.0e-3)),
            weight_decay=float(training_config.get("weight_decay", 1.0e-5)),
        )
    )
    return optax.chain(*transforms)


def _resolve_data_parallel_training(
    training_config: dict[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    requested = str(training_config.get("data_parallel", "single")).strip().lower()
    requested = {"none": "single", "false": "single", "true": "auto"}.get(
        requested,
        requested,
    )
    if requested not in {"single", "auto", "pmap"}:
        raise ValueError(
            "prior_learning.training.data_parallel must be one of: single, auto, pmap"
        )
    devices = tuple(jax.local_devices())
    n_devices = len(devices)
    if requested == "single":
        enabled = False
        effective = "single"
    elif requested == "auto":
        enabled = n_devices > 1
        effective = "pmap" if enabled else "single"
    else:
        if n_devices < 2:
            raise ValueError(
                "prior_learning.training.data_parallel='pmap' requires at least two "
                f"local JAX devices; visible devices={devices}"
            )
        enabled = True
        effective = "pmap"
    if enabled and int(batch_size) % int(n_devices) != 0:
        raise ValueError(
            "In pmap mode, prior_learning.training.batch_size must be divisible "
            f"by local device count: batch_size={int(batch_size)} devices={n_devices}"
        )
    return {
        "requested": requested,
        "effective": effective,
        "enabled": bool(enabled),
        "devices": devices,
        "n_devices": int(n_devices),
        "global_batch_size": int(batch_size),
        "per_device_batch_size": (
            int(batch_size) // int(n_devices) if enabled else int(batch_size)
        ),
    }


def _make_prior_pmap_train_step(optimizer, *, base_moment_weight: float = 0.0):
    pmap_array_axis = eqx.if_array(0)

    @eqx.filter_pmap(
        axis_name="devices",
        in_axes=(pmap_array_axis, pmap_array_axis, pmap_array_axis),
        out_axes=(pmap_array_axis, pmap_array_axis, 0, 0, 0, 0, 0, 0, 0),
    )
    def step(prior, opt_state, batch):
        (loss, aux), grads = eqx.filter_value_and_grad(
            _prior_objective,
            has_aux=True,
        )(prior, batch, jnp.asarray(base_moment_weight, dtype=batch.dtype))
        mean_log_prob, base_penalty = aux
        grads = jax.lax.pmean(grads, axis_name="devices")
        loss = jax.lax.pmean(loss, axis_name="devices")
        mean_log_prob = jax.lax.pmean(mean_log_prob, axis_name="devices")
        base_penalty = jax.lax.pmean(base_penalty, axis_name="devices")
        grad_norm = _tree_l2_norm_jax(grads)
        loss_finite = jnp.isfinite(loss)
        grads_finite = _tree_all_finite_jax(grads)
        update_applied = jax.lax.pmin(
            (loss_finite & grads_finite).astype(jnp.int32),
            axis_name="devices",
        ).astype(jnp.bool_)
        safe_grads = _zero_tree_when_false(grads, update_applied)
        updates, new_opt_state = optimizer.update(
            safe_grads,
            opt_state,
            eqx.filter(prior, eqx.is_inexact_array),
        )
        new_prior = eqx.apply_updates(prior, updates)
        prior = _select_tree_when_false(new_prior, prior, update_applied)
        opt_state = _select_tree_when_false(new_opt_state, opt_state, update_applied)
        return (
            prior,
            opt_state,
            loss,
            mean_log_prob,
            base_penalty,
            grad_norm,
            loss_finite,
            grads_finite,
            update_applied,
        )

    return step


def _pad_epoch_order_for_data_parallel(
    order: np.ndarray,
    *,
    global_batch_size: int,
    enabled: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    order = np.asarray(order, dtype=np.int64)
    if not enabled or len(order) == 0:
        return order, 0
    remainder = len(order) % int(global_batch_size)
    if remainder == 0:
        return order, 0
    pad_count = int(global_batch_size) - int(remainder)
    padding = rng.choice(order, size=pad_count, replace=True)
    return np.concatenate([order, np.asarray(padding, dtype=np.int64)]), pad_count


def _shard_x_batch(batch: jnp.ndarray, n_devices: int) -> jnp.ndarray:
    batch = jnp.asarray(batch)
    if batch.shape[0] % int(n_devices) != 0:
        raise ValueError(
            "pmap prior batch leading dimension must be divisible by device count: "
            f"shape={batch.shape} devices={n_devices}"
        )
    return batch.reshape((int(n_devices), batch.shape[0] // int(n_devices), *batch.shape[1:]))


def _replicate_tree(tree, devices: tuple[Any, ...]):
    n_devices = len(tuple(devices))

    def replicate_leaf(leaf):
        if eqx.is_array(leaf):
            value = jnp.asarray(leaf)
            return jnp.broadcast_to(value, (n_devices, *value.shape))
        return leaf

    return jax.tree_util.tree_map(replicate_leaf, tree)


def _unreplicate_tree(tree):
    def first_leaf(leaf):
        if eqx.is_array(leaf):
            return leaf[0]
        return leaf

    return jax.tree_util.tree_map(first_leaf, tree)


def _unreplicate_scalar(value) -> float:
    arr = np.asarray(jax.device_get(value))
    if arr.shape:
        arr = arr.reshape(-1)[0]
    return float(arr)


def _unreplicate_bool(value) -> bool:
    arr = np.asarray(jax.device_get(value))
    if arr.shape:
        arr = arr.reshape(-1)[0]
    return bool(arr)


def _prior_nll(prior: PriorFlow, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    log_prob = prior.log_prob(x)
    return -jnp.mean(log_prob), jnp.mean(log_prob)


def _prior_objective(
    prior: PriorFlow,
    x: jnp.ndarray,
    base_moment_weight: jnp.ndarray,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    nll, mean_log_prob = _prior_nll(prior, x)
    u, _logdet = prior.inverse(x)
    center = jnp.mean(u, axis=0)
    centered = u - center
    std = jnp.sqrt(jnp.mean(centered**2, axis=0) + 1.0e-6)
    covariance = centered.T @ centered / jnp.maximum(centered.shape[0], 1)
    off_diagonal = covariance - jnp.diag(jnp.diag(covariance))
    base_penalty = (
        jnp.sum(center**2)
        + jnp.sum((std - 1.0) ** 2)
        + jnp.sum(off_diagonal**2) / u.shape[-1]
    )
    loss = nll + base_moment_weight * base_penalty
    return loss, (mean_log_prob, base_penalty)


def _prior_nll_scalar(prior: PriorFlow, x: jnp.ndarray) -> jnp.ndarray:
    return _prior_nll(prior, x)[0]


_loss_and_grads_jit = eqx.filter_jit(
    eqx.filter_value_and_grad(_prior_objective, has_aux=True)
)
_prior_nll_jit = eqx.filter_jit(_prior_nll_scalar)


def _tree_all_finite_jax(tree) -> jnp.ndarray:
    leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_inexact_array(leaf)
    ]
    if not leaves:
        return jnp.asarray(True)
    flags = [jnp.all(jnp.isfinite(leaf)) for leaf in leaves]
    return jnp.all(jnp.asarray(flags))


def _tree_l2_norm_jax(tree) -> jnp.ndarray:
    leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_inexact_array(leaf)
    ]
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    total = sum(jnp.sum(jnp.asarray(leaf) ** 2) for leaf in leaves)
    return jnp.sqrt(total)


def _zero_tree_when_false(tree, predicate):
    def zero_leaf(leaf):
        if eqx.is_inexact_array(leaf):
            return jnp.where(predicate, leaf, jnp.zeros_like(leaf))
        return leaf

    return jax.tree_util.tree_map(zero_leaf, tree)


def _select_tree_when_false(true_tree, false_tree, predicate):
    def select_leaf(true_leaf, false_leaf):
        if eqx.is_array(true_leaf):
            return jnp.where(predicate, true_leaf, false_leaf)
        return true_leaf

    return jax.tree_util.tree_map(select_leaf, true_tree, false_tree)


def _tree_all_finite(tree) -> bool:
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(tree) if hasattr(leaf, "dtype")]
    if not leaves:
        return True
    return bool(all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves))


def _tree_l2_norm(tree) -> float:
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(tree) if hasattr(leaf, "dtype")]
    if not leaves:
        return 0.0
    total = sum(float(jnp.sum(jnp.asarray(leaf) ** 2)) for leaf in leaves)
    return float(np.sqrt(total))


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)
