"""Training loop for supervised Diffsky truth priors."""

from __future__ import annotations

import json
import time
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

from .data import TruthDataset, load_truth_dataset, prior_samples_frame
from .diagnostics import write_supervised_prior_diagnostics
from .flows import RealNVPPrior
from .schema import ParameterSpec, TruthSchema

eqx, optax = require_amortized_dependencies()


@dataclass(frozen=True)
class PriorTrainingResult:
    prior: RealNVPPrior
    last_prior: RealNVPPrior
    training_log: pd.DataFrame
    validation_log: pd.DataFrame
    initial_train_nll: float
    best_metric: float
    best_epoch: int


def prior_learning_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return supervised-prior config with local defaults."""
    raw = dict(config.get("prior_learning", {}) or {})
    raw.setdefault("dataset", config.get("catalog_path"))
    raw.setdefault("schema", "diffsky_truth_basic")
    raw.setdefault("missing_policy", "reduce")
    raw.setdefault("bounds", {})
    raw.setdefault("flow", {})
    raw.setdefault("training", {})
    raw.setdefault("output", {})
    raw["flow"].setdefault("n_layers", 8)
    raw["flow"].setdefault("hidden_size", 128)
    raw["flow"].setdefault("scale_clamp", 0.05)
    raw["training"].setdefault("epochs", 20)
    raw["training"].setdefault("batch_size", 256)
    raw["training"].setdefault("learning_rate", 1.0e-3)
    raw["training"].setdefault("weight_decay", 1.0e-5)
    raw["training"].setdefault("gradient_clip_norm", 1.0)
    raw["training"].setdefault("validation_fraction", 0.1)
    raw["training"].setdefault("seed", 42)
    raw["training"].setdefault("epoch_shuffle", True)
    raw["output"].setdefault("prior_samples", 8192)
    raw["output"].setdefault("truth_sample_limit", 200_000)
    return raw


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
    del progress
    out = ensure_dir(out_dir)
    cfg = prior_learning_config(config)
    dataset = Path(dataset_path or cfg["dataset"])
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
    split = _train_validation_split(
        len(truth.x),
        validation_fraction=float(training_cfg.get("validation_fraction", 0.1)),
        seed=int(training_cfg.get("seed", 42)),
    )
    np.save(out / "train_indices.npy", split["train"])
    np.save(out / "validation_indices.npy", split["validation"])
    _log(
        verbose,
        "[prior] split: "
        f"train={len(split['train'])} validation={len(split['validation'])}",
    )
    start = time.time()
    result = fit_realnvp_to_x(
        truth.x[split["train"]],
        truth.x[split["validation"]] if len(split["validation"]) else None,
        latent_dim=truth.x.shape[1],
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
    summary = {
        "dataset": str(dataset),
        "row_indices_file": str(row_indices_file) if row_indices_file else None,
        "schema": truth.schema.to_dict(),
        "parameter_names": list(truth.parameter_names),
        "n_truth_rows": int(truth.theta.shape[0]),
        "dropped_nonfinite_rows": int(truth.dropped_rows),
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
        "objective": "negative_mean_log_prob_theta_true",
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


def fit_realnvp_to_x(
    x_train: np.ndarray,
    x_validation: np.ndarray | None,
    *,
    latent_dim: int,
    flow_config: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
) -> PriorTrainingResult:
    """Fit a RealNVP prior to an unconstrained truth matrix."""
    x_train = np.asarray(x_train, dtype=np.float32)
    if x_train.ndim != 2 or x_train.shape[1] != int(latent_dim):
        raise ValueError("x_train must have shape [n, latent_dim]")
    x_validation = (
        None
        if x_validation is None or len(x_validation) == 0
        else np.asarray(x_validation, dtype=np.float32)
    )
    key = jax.random.PRNGKey(int(seed))
    prior = RealNVPPrior(
        key,
        latent_dim=int(latent_dim),
        n_layers=int(flow_config.get("n_layers", 8)),
        hidden_size=int(flow_config.get("hidden_size", 128)),
        scale_clamp=float(flow_config.get("scale_clamp", 0.05)),
    )
    prior = _cast_inexact_arrays(prior, jnp.float32)
    optimizer = _make_optimizer(training_config)
    opt_state = optimizer.init(eqx.filter(prior, eqx.is_inexact_array))
    batch_size = max(int(training_config.get("batch_size", 256)), 1)
    epochs = max(int(training_config.get("epochs", 20)), 1)
    epoch_shuffle = bool(training_config.get("epoch_shuffle", True))
    rng = np.random.default_rng(int(seed) + 100)
    initial_train_nll = float(_prior_nll_jit(prior, jnp.asarray(x_train)))
    best_metric = float("inf")
    best_epoch = 0
    best_prior = prior
    rows: list[dict[str, float | int | str]] = []
    val_rows: list[dict[str, float | int | str]] = []
    for epoch in range(1, epochs + 1):
        order = np.arange(len(x_train))
        if epoch_shuffle:
            rng.shuffle(order)
        epoch_losses = []
        for batch_index, start in enumerate(range(0, len(order), batch_size)):
            batch = jnp.asarray(x_train[order[start : start + batch_size]])
            (loss, mean_log_prob), grads = _loss_and_grads_jit(prior, batch)
            loss_finite = bool(np.isfinite(float(loss)))
            grads_finite = _tree_all_finite(grads)
            update_applied = bool(loss_finite and grads_finite)
            if update_applied:
                updates, opt_state = optimizer.update(
                    grads,
                    opt_state,
                    eqx.filter(prior, eqx.is_inexact_array),
                )
                prior = eqx.apply_updates(prior, updates)
            epoch_losses.append(float(loss))
            rows.append(
                {
                    "epoch": int(epoch),
                    "batch": int(batch_index),
                    "split": "train",
                    "loss": float(loss),
                    "mean_log_prob": float(mean_log_prob),
                    "n_objects": int(batch.shape[0]),
                    "loss_finite": float(loss_finite),
                    "grads_finite": float(grads_finite),
                    "update_applied": float(update_applied),
                    "grad_norm": float(_tree_l2_norm(grads)),
                }
            )
        train_metric = float(np.nanmean(epoch_losses))
        if x_validation is not None:
            val_log_prob = float(
                np.mean(np.asarray(prior.log_prob(jnp.asarray(x_validation))))
            )
            val_metric = -val_log_prob
            val_rows.append(
                {
                    "epoch": int(epoch),
                    "split": "validation",
                    "mean_log_prob": float(val_log_prob),
                    "negative_mean_log_prob": float(val_metric),
                    "n_objects": int(len(x_validation)),
                }
            )
            metric = val_metric
        else:
            metric = train_metric
        if np.isfinite(metric) and metric < best_metric:
            best_metric = float(metric)
            best_epoch = int(epoch)
            best_prior = prior
    return PriorTrainingResult(
        prior=best_prior,
        last_prior=prior,
        training_log=pd.DataFrame(rows),
        validation_log=pd.DataFrame(val_rows),
        initial_train_nll=initial_train_nll,
        best_metric=best_metric,
        best_epoch=best_epoch,
    )


def save_prior_checkpoint(
    path: str | Path,
    prior: RealNVPPrior,
    *,
    config: dict[str, Any],
    truth: TruthDataset,
    flow_config: dict[str, Any],
    epoch: int,
    metric: float,
) -> None:
    """Serialize a supervised RealNVP prior and its truth schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = _cast_inexact_arrays(prior, jnp.float32)
    eqx.tree_serialise_leaves(path, prior)
    sidecar = {
        "epoch": int(epoch),
        "metric": float(metric),
        "architecture": {
            "type": "realnvp",
            "latent_dim": int(prior.latent_dim),
            "n_layers": int(flow_config.get("n_layers", 8)),
            "hidden_size": int(flow_config.get("hidden_size", 128)),
            "scale_clamp": float(flow_config.get("scale_clamp", 0.05)),
            "parameter_dtype": "float32",
        },
        "latent_spec": latent_spec_to_jsonable(truth.latent_spec),
        "schema": truth.schema.to_dict(),
        "source_dataset": truth.dataset_path,
        "prior_learning": prior_learning_config(config),
    }
    write_json(path.with_suffix(path.suffix + ".json"), sidecar)


def load_prior_checkpoint(path: str | Path) -> tuple[RealNVPPrior, dict[str, Any], LatentSpec, TruthSchema]:
    """Load a supervised prior checkpoint plus schema metadata."""
    path = Path(path)
    sidecar_path = path.with_suffix(path.suffix + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    arch = sidecar["architecture"]
    template = RealNVPPrior(
        jax.random.PRNGKey(0),
        latent_dim=int(arch["latent_dim"]),
        n_layers=int(arch["n_layers"]),
        hidden_size=int(arch["hidden_size"]),
        scale_clamp=float(arch["scale_clamp"]),
    )
    template = _cast_inexact_arrays(
        template,
        _jax_dtype_from_name(str(arch.get("parameter_dtype", "float32"))),
    )
    prior = eqx.tree_deserialise_leaves(path, template)
    latent = sidecar["latent_spec"]
    latent_spec = LatentSpec(
        names=tuple(latent["names"]),
        lower=jnp.asarray(latent["lower"], dtype=jnp.float32),
        upper=jnp.asarray(latent["upper"], dtype=jnp.float32),
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


def _train_validation_split(
    n_rows: int,
    *,
    validation_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    order = np.arange(int(n_rows), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(order)
    validation_fraction = min(max(float(validation_fraction), 0.0), 0.9)
    if validation_fraction <= 0.0 or n_rows < 2:
        return {"train": order, "validation": np.asarray([], dtype=np.int64)}
    n_val = int(round(validation_fraction * n_rows))
    n_val = min(max(n_val, 1), n_rows - 1)
    return {"train": order[n_val:], "validation": order[:n_val]}


def _prior_nll(prior: RealNVPPrior, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    log_prob = prior.log_prob(x)
    return -jnp.mean(log_prob), jnp.mean(log_prob)


def _prior_nll_scalar(prior: RealNVPPrior, x: jnp.ndarray) -> jnp.ndarray:
    return _prior_nll(prior, x)[0]


_loss_and_grads_jit = eqx.filter_jit(eqx.filter_value_and_grad(_prior_nll, has_aux=True))
_prior_nll_jit = eqx.filter_jit(_prior_nll_scalar)


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
        print(message)
