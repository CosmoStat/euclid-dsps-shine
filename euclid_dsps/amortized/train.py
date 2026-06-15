"""Training loop for FS2 amortized inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.calibration import (
    alpha_from_log_alpha,
    alpha_metadata,
    apply_global_sed_scale_to_flux,
    global_sed_scale_config,
    global_sed_scale_prior_penalty,
    make_global_sed_scale_state,
)
from euclid_dsps.filters import load_filters
from euclid_dsps.io import ensure_dir, truth_column_from_spec, write_json
from euclid_dsps.model import dynamic_model_args, load_context

from .config import amortized_config, require_amortized_dependencies
from .data import (
    PhotometryBatch,
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from .decoder import model_flux_from_x
from .diagnostics import write_training_diagnostics
from .elbo import AmortizedModel, negative_elbo
from .encoder import GaussianEncoder
from .features import FeatureStats, compute_feature_stats, write_feature_stats
from .flows import RealNVPPrior, StandardNormalPrior
from .latent import (
    LatentSpec,
    latent_spec_from_config,
    latent_spec_to_jsonable,
    theta_to_x,
)
from .likelihood import photometric_loglike

eqx, optax = require_amortized_dependencies()

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is a core dependency today
    tqdm = None


@dataclass(frozen=True)
class TrainingSplit:
    """Catalog-row split used by FS2 amortized training."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_redshift: np.ndarray
    validation_redshift: np.ndarray
    redshift_column: str | None
    redshift_bins: np.ndarray
    selection_mode: str
    stratified_strategy: str
    validation_fraction: float


class LossBatch(NamedTuple):
    """JAX-friendly batch payload used by the compiled training step."""

    flux: jnp.ndarray
    flux_err: jnp.ndarray
    mask: jnp.ndarray
    features: jnp.ndarray


class JitLatentSpec(NamedTuple):
    """JAX-friendly latent transform spec used by compiled training."""

    names: tuple[str, ...]
    lower: jnp.ndarray
    upper: jnp.ndarray


class _StaticArg:
    """Hash static JIT payloads by identity instead of recursive contents."""

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return f"_StaticArg({type(self.value).__name__})"


def build_amortized_model(config: dict[str, Any], key) -> AmortizedModel:
    """Instantiate encoder and RealNVP prior from config."""
    cfg = amortized_config(config)
    k_encoder, k_prior = jax.random.split(key)
    encoder_cfg = cfg["encoder"]
    input_dim = int(encoder_cfg.get("input_dim", 20))
    latent_dim = int(encoder_cfg.get("latent_dim", 16))
    encoder = GaussianEncoder(
        k_encoder,
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_sizes=tuple(
            int(v) for v in encoder_cfg.get("hidden_sizes", [256, 256, 256])
        ),
        activation=str(encoder_cfg.get("activation", "gelu")),
        log_std_min=float(encoder_cfg.get("log_std_min", -6.0)),
        log_std_max=float(encoder_cfg.get("log_std_max", 2.0)),
        initial_log_std=float(encoder_cfg.get("initial_log_std", -1.0)),
    )
    encoder = _initialize_encoder_mean_if_possible(config, encoder)
    prior = build_prior_from_config(config, k_prior, latent_dim=latent_dim)
    sed_scale = make_global_sed_scale_state(config)
    return AmortizedModel(encoder=encoder, prior=prior, sed_scale=sed_scale)


def build_prior_from_config(config: dict[str, Any], key, *, latent_dim: int):
    """Build or load the configured amortized prior source."""
    cfg = amortized_config(config)
    prior_cfg = cfg["prior"]
    source = str(prior_cfg.get("source", "joint_realnvp"))
    if source == "standard_normal":
        return StandardNormalPrior(latent_dim=int(latent_dim))
    if source in {"joint_realnvp", "realnvp"}:
        return RealNVPPrior(
            key,
            latent_dim=int(latent_dim),
            n_layers=int(prior_cfg.get("n_layers", 8)),
            hidden_size=int(prior_cfg.get("hidden_size", 128)),
            scale_clamp=float(prior_cfg.get("scale_clamp", 0.05)),
        )
    if source == "supervised_checkpoint":
        checkpoint = prior_cfg.get("checkpoint")
        if not checkpoint:
            raise ValueError(
                "amortized.prior.source='supervised_checkpoint' requires "
                "amortized.prior.checkpoint"
            )
        from euclid_dsps.prior_learning.train import load_prior_checkpoint

        prior, _sidecar, prior_spec, _schema = load_prior_checkpoint(checkpoint)
        active_spec = latent_spec_from_config(config)
        _validate_loaded_prior_spec(active_spec, prior_spec)
        return prior
    raise ValueError(
        "amortized.prior.source must be one of "
        "'standard_normal', 'supervised_checkpoint', or 'joint_realnvp'"
    )


def _validate_loaded_prior_spec(active: LatentSpec, loaded: LatentSpec) -> None:
    if active.names != loaded.names:
        raise ValueError(
            "Supervised prior latent names do not match amortized config: "
            f"checkpoint={loaded.names}, config={active.names}"
        )
    if not np.allclose(
        np.asarray(active.lower),
        np.asarray(loaded.lower),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("Supervised prior lower bounds do not match amortized config")
    if not np.allclose(
        np.asarray(active.upper),
        np.asarray(loaded.upper),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("Supervised prior upper bounds do not match amortized config")


def _initialize_encoder_mean_if_possible(
    config: dict[str, Any],
    encoder: GaussianEncoder,
) -> GaussianEncoder:
    """Start the real FS2 encoder near a stable PopCosmos theta point."""
    try:
        spec = latent_spec_from_config(config)
    except (KeyError, ValueError):
        return encoder
    theta0 = _default_initial_theta(spec)
    x0 = theta_to_x(theta0, spec)
    zero_mean_weight = jnp.zeros_like(encoder.mean_head.weight)
    zero_log_std_weight = jnp.zeros_like(encoder.log_std_head.weight)
    zero_log_std_bias = jnp.zeros_like(encoder.log_std_head.bias)
    return eqx.tree_at(
        lambda enc: (
            enc.mean_head.weight,
            enc.mean_head.bias,
            enc.log_std_head.weight,
            enc.log_std_head.bias,
        ),
        encoder,
        (
            zero_mean_weight,
            x0,
            zero_log_std_weight,
            zero_log_std_bias,
        ),
    )


def _default_initial_theta(spec: LatentSpec) -> jnp.ndarray:
    defaults = {
        "z_obs": 0.8,
        "log10_stellar_mass": 10.0,
        "log10_stellar_metallicity": -0.7,
        "tau2": 0.4,
        "dust_index_n": -0.7,
        "tau1_over_tau2": 1.0,
        "log10_gas_metallicity": -0.3,
        "log10_gas_ionization": -2.5,
        "ln_fagn": -8.0,
        "ln_tauagn": float(np.log(20.0)),
    }
    defaults.update({f"dlog10_sfr_{index}": 0.0 for index in range(1, 7)})
    values = []
    for index, name in enumerate(spec.names):
        midpoint = 0.5 * (float(spec.lower[index]) + float(spec.upper[index]))
        value = float(defaults.get(name, midpoint))
        low = float(spec.lower[index])
        high = float(spec.upper[index])
        values.append(min(max(value, low + 1.0e-5), high - 1.0e-5))
    return jnp.asarray(values, dtype=jnp.float32)


def make_optimizer(config: dict[str, Any]):
    """Build the joint encoder/prior optimizer."""
    training = amortized_config(config)["training"]
    transforms = []
    clip = float(training.get("gradient_clip_norm", 1.0))
    if clip > 0.0:
        transforms.append(optax.clip_by_global_norm(clip))
    transforms.append(
        optax.adamw(
            learning_rate=float(training.get("learning_rate", 1.0e-4)),
            weight_decay=float(training.get("weight_decay", 1.0e-5)),
        )
    )
    return optax.chain(*transforms)


def train_amortized_fs2(
    config: dict[str, Any],
    out_dir: Path,
    limit: int | None,
    batch_size: int,
    epochs: int,
    n_samples: int,
    seed: int,
    verbose: bool = True,
    progress: bool = True,
    dataset_label: str = "FS2",
) -> None:
    """Train encoder and RealNVP prior jointly on configured photometry."""
    out = ensure_dir(out_dir)
    cfg = amortized_config(config)
    _log(verbose, f"[amortized] {dataset_label} joint encoder/RealNVP training")
    _log(verbose, f"[amortized] output directory: {out}")
    _log(
        verbose,
        "[amortized] run config: "
        f"limit={limit if limit is not None else 'all'} "
        f"batch_size={int(batch_size)} epochs={int(epochs)} "
        f"n_samples={int(n_samples)} seed={int(seed)}",
    )
    _log(
        verbose,
        "[amortized] JAX backend: " f"{jax.default_backend()} devices={jax.devices()}",
    )
    write_json(out / "normalized_config.json", config)
    split = build_training_split(config, limit=limit, seed=seed)
    write_training_split_artifacts(out, split)
    _log(
        verbose,
        "[amortized] data split: "
        f"mode={split.selection_mode} strategy={split.stratified_strategy} "
        f"train={len(split.train_indices)} validation={len(split.validation_indices)} "
        f"z_column={split.redshift_column}",
    )
    catalog_batch_size = int(
        cfg["data"].get("catalog_batch_size", max(int(batch_size), 10_000))
    )
    jax_batch_size = _effective_jax_batch_size(cfg["training"], int(batch_size))
    if jax_batch_size != int(batch_size):
        _log(
            verbose,
            "[amortized] capping JAX/DSPS batch size: "
            f"requested_batch_size={int(batch_size)} jax_batch_size={jax_batch_size}",
        )
    _log(verbose, "[amortized] loading selected train photometry arrays...")
    train_arrays = load_photometry_arrays_from_config(
        config,
        batch_size=catalog_batch_size,
        row_indices=split.train_indices,
    )
    validation_arrays = None
    if len(split.validation_indices):
        _log(verbose, "[amortized] loading selected validation photometry arrays...")
        validation_arrays = load_photometry_arrays_from_config(
            config,
            batch_size=catalog_batch_size,
            row_indices=split.validation_indices,
        )
    _log(verbose, "[amortized] computing feature stats from train flux/errors...")
    feature_stats = compute_feature_stats(
        train_arrays.flux,
        train_arrays.flux_err,
        train_arrays.mask,
        band_names=train_arrays.band_names,
        flux_transform=str(cfg["features"].get("flux_transform", "asinh")),
    )
    write_feature_stats(out / "feature_stats.json", feature_stats)
    _log(
        verbose,
        "[amortized] feature stats ready: "
        f"{len(feature_stats.band_names)} bands, "
        f"feature_dim={2 * len(feature_stats.band_names)} "
        f"flux_transform={feature_stats.flux_transform}",
    )

    _log(verbose, "[amortized] loading configured filters...")
    filters = load_filters(config["bands"])
    _log(verbose, f"[amortized] loading DSPS context from {config['ssp_path']}...")
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    model_args = dynamic_model_args(context)
    jit_context = _StaticArg(context)
    _log(
        verbose,
        "[amortized] DSPS context ready: "
        f"{len(filters)} filters, n_sfh_bins={context.n_sfh_bins}, "
        f"model={context.model_config}",
    )
    latent_spec = latent_spec_from_config(config)
    jit_latent_spec = JitLatentSpec(
        names=latent_spec.names,
        lower=latent_spec.lower,
        upper=latent_spec.upper,
    )
    _log(
        verbose,
        "[amortized] latent spec ready: "
        f"{len(latent_spec.names)} parameters, "
        f"first={latent_spec.names[0]}, last={latent_spec.names[-1]}",
    )
    key = jax.random.PRNGKey(int(seed))
    key, model_key = jax.random.split(key)
    model = build_amortized_model(config, model_key)
    train_prior = _train_prior_jointly(cfg["prior"])
    calibration_runtime_config = {"calibration": config.get("calibration", {}) or {}}
    sed_scale_cfg = global_sed_scale_config(calibration_runtime_config)
    train_alpha = bool(sed_scale_cfg.enabled and sed_scale_cfg.trainable)
    _log(
        verbose,
        "[amortized] model built: "
        f"encoder_hidden={cfg['encoder'].get('hidden_sizes')} "
        f"realnvp_layers={cfg['prior'].get('n_layers')} "
        f"realnvp_hidden={cfg['prior'].get('hidden_size')} "
        f"prior_source={cfg['prior'].get('source')} "
        f"train_prior={train_prior} "
        f"alpha_sed_enabled={sed_scale_cfg.enabled} "
        f"train_alpha={train_alpha}",
    )
    optimizer = make_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    ckpt_dir = ensure_dir(out / "checkpoints")
    rows = []
    configured_best_metric = str(
        cfg["training"].get("best_checkpoint_metric", "validation_negative_loglike")
    )
    best_checkpoint_metric = _effective_best_checkpoint_metric(
        configured_best_metric,
        has_validation=validation_arrays is not None,
    )
    best_checkpoint_min_epoch = _best_checkpoint_min_epoch(
        cfg["training"],
        has_validation=validation_arrays is not None,
    )
    best_metric_value = np.inf
    best_checkpoint_epoch: int | None = None
    start_time = time.time()
    checkpoint_every = int(cfg["output"].get("checkpoint_every", 1))
    diagnostics_every = int(cfg["output"].get("diagnostics_every", 1))
    save_training_curves = bool(cfg["output"].get("save_training_curves", True))
    validation_every = int(cfg["training"].get("validation_every", 1))
    epoch_shuffle = bool(cfg["data"].get("epoch_shuffle", True))
    kl_weight_max = float(cfg["training"].get("kl_weight_max", 1.0))
    expected_batches = _expected_batch_count(
        len(train_arrays.object_id), int(jax_batch_size)
    )
    val_expected_batches = (
        None
        if validation_arrays is None
        else _expected_batch_count(len(validation_arrays.object_id), int(jax_batch_size))
    )
    validation_bin_rows: list[dict[str, float | int | str]] = []
    train_rng = np.random.default_rng(int(seed) + 10_000)
    _log(
        verbose,
        "[amortized] training start: "
        "first batch includes JAX/DSPS compilation and can be noticeably slower.",
    )
    for epoch in range(1, int(epochs) + 1):
        kl_weight = _kl_weight(
            epoch,
            int(cfg["training"].get("kl_annealing_epochs", 5)),
            max_weight=kl_weight_max,
        )
        epoch_rows = []
        _log(
            verbose, f"[amortized] epoch {epoch}/{int(epochs)} start kl={kl_weight:.3f}"
        )
        train_order = np.arange(len(train_arrays.object_id))
        if epoch_shuffle:
            train_rng.shuffle(train_order)
        batch_iter = iter_photometry_batches_from_arrays(
            train_arrays,
            batch_size=int(jax_batch_size),
            feature_stats=feature_stats,
            order=train_order,
        )
        with _progress_bar(
            enabled=bool(progress),
            total=expected_batches,
            desc=f"epoch {epoch}/{int(epochs)}",
            unit="batch",
        ) as pbar:
            for batch_index, batch in enumerate(batch_iter):
                key, step_key = jax.random.split(key)
                (loss, metrics), grads = _loss_and_grads_jit(
                    model,
                    _loss_batch(batch),
                    jit_latent_spec,
                    jit_context,
                    model_args,
                    latent_spec.names,
                    step_key,
                    int(n_samples),
                    float(kl_weight),
                    cfg["likelihood"],
                    calibration_runtime_config,
                )
                if not train_prior:
                    grads = zero_prior_grads(grads)
                if not train_alpha:
                    grads = zero_sed_scale_grads(grads)
                record = _metrics_record(metrics)
                record.update(component_grad_norms(grads))
                loss_finite = bool(np.isfinite(float(np.asarray(jax.device_get(loss)))))
                grads_finite = tree_all_finite(grads)
                update_applied = bool(loss_finite and grads_finite)
                if update_applied:
                    updates, opt_state = optimizer.update(
                        grads,
                        opt_state,
                        eqx.filter(model, eqx.is_inexact_array),
                    )
                    model = eqx.apply_updates(model, updates)
                elif verbose:
                    _log(
                        verbose,
                        "[amortized] skipped non-finite update: "
                        f"epoch={epoch} batch={batch_index} "
                        f"loss_finite={loss_finite} grads_finite={grads_finite} "
                        f"finite_fraction={record.get('finite_fraction', float('nan')):.3g}",
                    )
                record.update(
                    {
                        "split": "train",
                        "epoch": int(epoch),
                        "batch": int(batch_index),
                        "kl_weight": float(kl_weight),
                        "n_objects": int(batch.flux.shape[0]),
                        "loss_finite": float(loss_finite),
                        "grads_finite": float(grads_finite),
                        "update_applied": float(update_applied),
                    }
                )
                epoch_rows.append(record)
                rows.append(record)
                if validation_arrays is None and loss_finite:
                    checkpoint_metric_value = _checkpoint_metric_from_rows(
                        [record],
                        best_checkpoint_metric,
                    )
                    if _should_update_best_checkpoint(
                        epoch=epoch,
                        value=checkpoint_metric_value,
                        best_value=best_metric_value,
                        min_epoch=best_checkpoint_min_epoch,
                    ):
                        best_metric_value = float(checkpoint_metric_value)
                        best_checkpoint_epoch = int(epoch)
                        save_checkpoint(
                            ckpt_dir / "best.eqx",
                            model,
                            config=config,
                            latent_spec=latent_spec,
                            feature_stats=feature_stats,
                            epoch=epoch,
                            metric=best_metric_value,
                            metric_name=best_checkpoint_metric,
                        )
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{record['loss']:.3g}",
                        "nll": f"{record['negative_loglike']:.3g}",
                        "kl": f"{record['kl_mc_mean']:.3g}",
                        "enc_g": f"{record['encoder_grad_norm']:.2g}",
                        "prior_g": f"{record['prior_grad_norm']:.2g}",
                        "upd": int(update_applied),
                    },
                    refresh=False,
                )
                if verbose and not progress:
                    _log(verbose, _batch_progress_line(record))
        if not epoch_rows:
            raise ValueError("No FS2 batches were produced for amortized training")
        validation_rows = []
        epoch_bin_rows = []
        if (
            validation_arrays is not None
            and validation_every > 0
            and epoch % validation_every == 0
        ):
            _log(
                verbose,
                f"[amortized] epoch {epoch}/{int(epochs)} validation start",
            )
            key, val_key = jax.random.split(key)
            validation_rows, epoch_bin_rows = evaluate_validation_epoch(
                model,
                validation_arrays,
                split,
                feature_stats,
                latent_spec,
                context,
                model_args,
                val_key,
                batch_size=int(jax_batch_size),
                n_samples=int(n_samples),
                kl_weight=float(kl_weight),
                likelihood_config=cfg["likelihood"],
                calibration_config=calibration_runtime_config,
                progress=bool(progress),
                total=val_expected_batches,
                desc=f"val {epoch}/{int(epochs)}",
                epoch=epoch,
            )
            rows.extend(validation_rows)
            validation_bin_rows.extend(epoch_bin_rows)
            val_loss = _finite_mean([row["loss"] for row in validation_rows])
            val_nll = _finite_mean(
                [row["negative_loglike"] for row in validation_rows]
            )
            checkpoint_metric_value = _checkpoint_metric_from_rows(
                validation_rows,
                best_checkpoint_metric,
            )
            if _should_update_best_checkpoint(
                epoch=epoch,
                value=checkpoint_metric_value,
                best_value=best_metric_value,
                min_epoch=best_checkpoint_min_epoch,
            ):
                best_metric_value = float(checkpoint_metric_value)
                best_checkpoint_epoch = int(epoch)
                save_checkpoint(
                    ckpt_dir / "best.eqx",
                    model,
                    config=config,
                    latent_spec=latent_spec,
                    feature_stats=feature_stats,
                    epoch=epoch,
                    metric=best_metric_value,
                    metric_name=best_checkpoint_metric,
                )
            _log(
                verbose,
                "[amortized] epoch "
                f"{epoch}/{int(epochs)} validation done: "
                f"mean_loss={val_loss:.6g} mean_nll={val_nll:.6g} "
                f"{best_checkpoint_metric}={checkpoint_metric_value:.6g} "
                f"binned_rows={len(epoch_bin_rows)}",
            )
        save_checkpoint(
            ckpt_dir / "last.eqx",
            model,
            config=config,
            latent_spec=latent_spec,
            feature_stats=feature_stats,
            epoch=epoch,
            metric=_finite_mean([row["loss"] for row in epoch_rows]),
            metric_name="train_loss",
        )
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:04d}.eqx",
                model,
                config=config,
                latent_spec=latent_spec,
                feature_stats=feature_stats,
                epoch=epoch,
                metric=_finite_mean([row["loss"] for row in epoch_rows]),
                metric_name="train_loss",
            )
        pd.DataFrame(rows).to_csv(out / "training_log.csv", index=False)
        if validation_bin_rows:
            pd.DataFrame(validation_bin_rows).to_csv(
                out / "validation_redshift_bin_metrics.csv",
                index=False,
            )
        write_training_progress(
            out,
            rows=rows,
            epoch=epoch,
            best_loss=best_metric_value,
            start_time=start_time,
            checkpoint_last="checkpoints/last.eqx",
            best_checkpoint_metric=best_checkpoint_metric,
            best_checkpoint_epoch=best_checkpoint_epoch,
        )
        if (
            save_training_curves
            and diagnostics_every > 0
            and epoch % diagnostics_every == 0
        ):
            write_training_diagnostics(out / "training_log.csv", out)
        epoch_loss = _finite_mean([row["loss"] for row in epoch_rows])
        epoch_nll = _finite_mean([row["negative_loglike"] for row in epoch_rows])
        epoch_kl = _finite_mean([row["kl_mc_mean"] for row in epoch_rows])
        epoch_updates = int(sum(row.get("update_applied", 0.0) for row in epoch_rows))
        _log(
            verbose,
            "[amortized] epoch "
            f"{epoch}/{int(epochs)} done: mean_loss={epoch_loss:.6g} "
            f"mean_nll={epoch_nll:.6g} mean_kl={epoch_kl:.6g} "
            f"updates={epoch_updates}/{len(epoch_rows)} "
            f"best_{best_checkpoint_metric}={best_metric_value:.6g} "
            f"last_checkpoint={ckpt_dir / 'last.eqx'}",
        )

    if not (ckpt_dir / "best.eqx").exists():
        fallback_metric = _finite_mean(
            [row["loss"] for row in rows if row.get("split") == "train"]
        )
        best_metric_value = fallback_metric
        best_checkpoint_epoch = int(epochs)
        save_checkpoint(
            ckpt_dir / "best.eqx",
            model,
            config=config,
            latent_spec=latent_spec,
            feature_stats=feature_stats,
            epoch=int(epochs),
            metric=fallback_metric,
            metric_name=f"{best_checkpoint_metric}_fallback_train_loss",
        )

    summary = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "jax_batch_size": int(jax_batch_size),
        "n_samples": int(n_samples),
        "limit": limit,
        "train_rows": int(len(split.train_indices)),
        "validation_rows": int(len(split.validation_indices)),
        "validation_fraction": float(split.validation_fraction),
        "selection_mode": split.selection_mode,
        "stratified_strategy": split.stratified_strategy,
        "redshift_column": split.redshift_column,
        "kl_annealing_epochs": int(cfg["training"].get("kl_annealing_epochs", 5)),
        "kl_weight_max": float(kl_weight_max),
        "best_loss": float(best_metric_value),
        "best_checkpoint_metric": best_checkpoint_metric,
        "best_checkpoint_metric_requested": configured_best_metric,
        "best_checkpoint_min_epoch": int(best_checkpoint_min_epoch),
        "best_checkpoint_epoch": (
            int(best_checkpoint_epoch)
            if best_checkpoint_epoch is not None
            else None
        ),
        "elapsed_time_s": float(time.time() - start_time),
        "checkpoint_best": "checkpoints/best.eqx",
        "checkpoint_last": "checkpoints/last.eqx",
        "checkpoint_every": checkpoint_every,
        "diagnostics_every": diagnostics_every,
        "updates_applied": int(
            sum(row.get("update_applied", 0.0) for row in _training_rows(rows))
        ),
        "updates_skipped": int(
            sum(1.0 - row.get("update_applied", 0.0) for row in _training_rows(rows))
        ),
        "joint_training": {
            "encoder": True,
            "realnvp_prior": bool(train_prior),
            "prior_source": str(cfg["prior"].get("source", "joint_realnvp")),
            "global_sed_scale": bool(train_alpha),
            "decoder": False,
            "kl_estimator": "monte_carlo_logq_minus_logp",
        },
        "global_sed_scale": alpha_metadata(
            float(np.asarray(jax.device_get(model.sed_scale.log_alpha_sed))),
            sed_scale_cfg.prior_sigma_log_alpha,
        )
        | {
            "enabled": bool(sed_scale_cfg.enabled),
            "trainable": bool(train_alpha),
            "mode": sed_scale_cfg.mode,
        },
    }
    write_json(out / "training_summary.json", summary)
    if save_training_curves:
        write_training_diagnostics(out / "training_log.csv", out)
    _log(verbose, "[amortized] training complete")
    _log(verbose, f"[amortized] summary: {out / 'training_summary.json'}")
    _log(verbose, f"[amortized] progress: {out / 'training_progress.json'}")
    _log(verbose, f"[amortized] best checkpoint: {out / 'checkpoints' / 'best.eqx'}")
    return


def save_checkpoint(
    path: str | Path,
    model: AmortizedModel,
    *,
    config: dict[str, Any],
    latent_spec: LatentSpec,
    feature_stats: FeatureStats,
    epoch: int,
    metric: float,
    metric_name: str | None = None,
) -> None:
    """Write an Equinox checkpoint plus a JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)
    sidecar = {
        "epoch": int(epoch),
        "metric": float(metric),
        "metric_name": metric_name,
        "latent_spec": latent_spec_to_jsonable(latent_spec),
        "feature_stats_path": "../feature_stats.json",
        "amortized": amortized_config(config),
        "architecture": architecture_summary(config),
    }
    write_json(path.with_suffix(path.suffix + ".json"), sidecar)


def load_checkpoint(
    path: str | Path,
    config: dict[str, Any],
) -> AmortizedModel:
    """Load an amortized model checkpoint using the active config architecture."""
    key = jax.random.PRNGKey(int(amortized_config(config)["training"].get("seed", 42)))
    template = build_amortized_model(config, key)
    return eqx.tree_deserialise_leaves(path, template)


def build_training_split(
    config: dict[str, Any],
    *,
    limit: int | None,
    seed: int,
) -> TrainingSplit:
    """Build a reproducible train/validation row split for FS2."""
    cfg = amortized_config(config)
    data_cfg = cfg["data"]
    validation_fraction = float(data_cfg.get("validation_fraction", 0.1))
    validation_fraction = min(max(validation_fraction, 0.0), 0.9)
    selection_mode = str(data_cfg.get("selection_mode", "stratified_redshift"))
    stratified_strategy = str(data_cfg.get("stratified_strategy", "balanced"))
    rng = np.random.default_rng(int(data_cfg.get("selection_seed", seed)))
    n_rows = _catalog_num_rows(config["catalog_path"])
    redshift_column = _configured_redshift_column(config, data_cfg)
    redshift = _read_redshift_column(config["catalog_path"], redshift_column)
    if redshift is not None:
        n_rows = len(redshift)
    total = n_rows if limit is None else min(int(limit), n_rows)
    redshift_bins = np.asarray(
        data_cfg.get(
            "redshift_bins",
            [0.0, 0.3, 0.6, 0.9, 1.2, 1.6, 2.0, 2.5, 3.5, 6.0],
        ),
        dtype=float,
    )
    if redshift_bins.ndim != 1 or redshift_bins.size < 2:
        redshift_bins = np.asarray([0.0, 6.0], dtype=float)
    redshift_bins = np.unique(redshift_bins)

    if selection_mode == "sequential":
        selected = np.arange(total, dtype=np.int64)
    elif selection_mode == "random" or redshift is None:
        selected = rng.choice(n_rows, size=total, replace=False).astype(np.int64)
    elif selection_mode == "stratified_redshift":
        selected = _select_stratified_indices(
            redshift,
            total=total,
            bins=redshift_bins,
            strategy=stratified_strategy,
            rng=rng,
        )
    else:
        raise ValueError(
            "amortized.data.selection_mode must be one of "
            "'sequential', 'random', or 'stratified_redshift'"
        )

    train_indices, validation_indices = _split_train_validation(
        selected,
        redshift,
        redshift_bins,
        validation_fraction,
        rng,
    )
    return TrainingSplit(
        train_indices=np.asarray(train_indices, dtype=np.int64),
        validation_indices=np.asarray(validation_indices, dtype=np.int64),
        train_redshift=_redshift_for_indices(redshift, train_indices),
        validation_redshift=_redshift_for_indices(redshift, validation_indices),
        redshift_column=redshift_column if redshift is not None else None,
        redshift_bins=redshift_bins,
        selection_mode=selection_mode if redshift is not None else "random",
        stratified_strategy=stratified_strategy,
        validation_fraction=validation_fraction,
    )


def write_training_split_artifacts(out: Path, split: TrainingSplit) -> None:
    """Write reproducibility artifacts for a training split."""
    np.save(out / "train_indices.npy", split.train_indices)
    np.save(out / "validation_indices.npy", split.validation_indices)
    if split.train_redshift.size:
        np.save(out / "train_redshift_proxy.npy", split.train_redshift)
    if split.validation_redshift.size:
        np.save(out / "validation_redshift_proxy.npy", split.validation_redshift)
    write_json(
        out / "training_split.json",
        {
            "selection_mode": split.selection_mode,
            "stratified_strategy": split.stratified_strategy,
            "validation_fraction": split.validation_fraction,
            "redshift_column": split.redshift_column,
            "redshift_bins": split.redshift_bins.tolist(),
            "train_rows": int(len(split.train_indices)),
            "validation_rows": int(len(split.validation_indices)),
            "train_redshift_finite": int(np.isfinite(split.train_redshift).sum()),
            "validation_redshift_finite": int(
                np.isfinite(split.validation_redshift).sum()
            ),
        },
    )


def evaluate_validation_epoch(
    model: AmortizedModel,
    arrays,
    split: TrainingSplit,
    feature_stats: FeatureStats,
    latent_spec: LatentSpec,
    context,
    model_args,
    key,
    *,
    batch_size: int,
    n_samples: int,
    kl_weight: float,
    likelihood_config: dict[str, Any],
    progress: bool,
    total: int | None,
    desc: str,
    epoch: int,
    calibration_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Evaluate validation ELBO and redshift-bin metrics without updates."""
    rows: list[dict[str, float | int | str]] = []
    object_rows: list[dict[str, float | int | str]] = []
    redshift_lookup = {
        int(object_id): float(z)
        for object_id, z in zip(
            np.asarray(arrays.object_id),
            split.validation_redshift,
            strict=False,
        )
    }
    jit_context = context if isinstance(context, _StaticArg) else _StaticArg(context)
    jit_latent_spec = (
        latent_spec
        if isinstance(latent_spec, JitLatentSpec)
        else JitLatentSpec(
            names=latent_spec.names,
            lower=latent_spec.lower,
            upper=latent_spec.upper,
        )
    )
    with _progress_bar(
        enabled=bool(progress),
        total=total,
        desc=desc,
        unit="batch",
    ) as pbar:
        for batch_index, batch in enumerate(
            iter_photometry_batches_from_arrays(
                arrays,
                batch_size=batch_size,
                feature_stats=feature_stats,
            )
        ):
            key, batch_key = jax.random.split(key)
            metrics, object_metrics = _evaluation_metrics_jit(
                model,
                _loss_batch(batch),
                jit_latent_spec,
                jit_context,
                model_args,
                jit_latent_spec.names,
                batch_key,
                int(n_samples),
                float(kl_weight),
                likelihood_config,
                calibration_config,
            )
            record = _metrics_record(metrics)
            record.update(
                {
                    "split": "validation",
                    "epoch": int(epoch),
                    "batch": int(batch_index),
                    "kl_weight": float(kl_weight),
                    "n_objects": int(batch.flux.shape[0]),
                    "loss_finite": float(np.isfinite(record["loss"])),
                    "grads_finite": 1.0,
                    "update_applied": 0.0,
                    "encoder_grad_norm": 0.0,
                    "prior_grad_norm": 0.0,
                    "alpha_grad_norm": 0.0,
                    "joint_grad_norm": 0.0,
                    "encoder_grad_nonzero": 0.0,
                    "prior_grad_nonzero": 0.0,
                    "alpha_grad_nonzero": 0.0,
                }
            )
            rows.append(record)
            object_rows.extend(
                _validation_object_rows(
                    batch,
                    object_metrics,
                    redshift_lookup,
                    split.redshift_bins,
                )
            )
            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": f"{record['loss']:.3g}",
                    "nll": f"{record['negative_loglike']:.3g}",
                    "kl": f"{record['kl_mc_mean']:.3g}",
                },
                refresh=False,
            )
    bin_rows = _redshift_bin_rows(
        object_rows,
        bins=split.redshift_bins,
        epoch=int(epoch),
    )
    return rows, bin_rows


def _evaluation_metrics(
    model,
    batch,
    latent_spec,
    context,
    model_args,
    parameter_names,
    key,
    n_samples,
    kl_weight,
    likelihood_config,
    calibration_config,
):
    x_samples, logq = model.encoder.sample_and_log_prob(
        key,
        batch.features,
        int(n_samples),
    )
    model_flux_raw = model_flux_from_x(
        x_samples,
        latent_spec,
        context,
        model_args,
        parameter_names,
    )
    scale_cfg = global_sed_scale_config(calibration_config)
    log_alpha_sed = model.sed_scale.log_alpha_sed
    model_flux = (
        apply_global_sed_scale_to_flux(model_flux_raw, log_alpha_sed)
        if scale_cfg.enabled
        else model_flux_raw
    )
    alpha_prior_penalty = (
        global_sed_scale_prior_penalty(
            log_alpha_sed,
            scale_cfg.prior_sigma_log_alpha,
        )
        if scale_cfg.enabled
        else jnp.asarray(0.0, dtype=model_flux.dtype)
    )
    loglike = photometric_loglike(
        obs_flux=batch.flux,
        model_flux=model_flux,
        obs_err=batch.flux_err,
        mask=batch.mask,
        likelihood_type=str(likelihood_config.get("type", "student_t")),
        student_t_dof=float(likelihood_config.get("student_t_dof", 2.0)),
        error_floor_frac=float(likelihood_config.get("error_floor_frac", 0.02)),
        error_jitter=float(likelihood_config.get("error_jitter", 0.0)),
    )
    logp = model.prior.log_prob(x_samples)
    kl = logq - logp
    objective = -loglike + float(kl_weight) * kl
    chi = (model_flux - batch.flux[None, :, :]) / batch.flux_err[None, :, :]
    chi = jnp.where(batch.mask[None, :, :], chi, 0.0)
    object_loss = jnp.mean(objective, axis=0)
    object_nll = jnp.mean(-loglike, axis=0)
    object_kl = jnp.mean(kl, axis=0)
    object_chi2 = jnp.mean(jnp.sum(chi**2, axis=-1), axis=0)
    metrics = {
        "loss": jnp.mean(objective) + alpha_prior_penalty,
        "negative_loglike": jnp.mean(-loglike),
        "loglike_mean": jnp.mean(loglike),
        "logprior_mean": jnp.mean(logp),
        "logq_mean": jnp.mean(logq),
        "kl_mc_mean": jnp.mean(kl),
        "model_flux_mean": jnp.mean(model_flux),
        "mean_model_flux_raw": jnp.mean(model_flux_raw),
        "mean_model_flux_scaled": jnp.mean(model_flux),
        "log_alpha_sed": log_alpha_sed,
        "alpha_sed": alpha_from_log_alpha(log_alpha_sed),
        "alpha_prior_penalty": alpha_prior_penalty,
        "residual_rms": jnp.sqrt(jnp.mean(chi**2)),
        "finite_fraction": jnp.mean(jnp.isfinite(objective)),
    }
    object_metrics = {
        "loss": object_loss,
        "negative_loglike": object_nll,
        "kl_mc_mean": object_kl,
        "posterior_predictive_chi2": object_chi2,
    }
    return metrics, object_metrics


def _validation_object_rows(
    batch: PhotometryBatch,
    object_metrics: dict[str, jnp.ndarray],
    redshift_lookup: dict[int, float],
    bins: np.ndarray,
) -> list[dict[str, float | int | str]]:
    object_ids = np.asarray(jax.device_get(batch.object_id), dtype=int)
    losses = np.asarray(jax.device_get(object_metrics["loss"]), dtype=float)
    nll = np.asarray(jax.device_get(object_metrics["negative_loglike"]), dtype=float)
    kl = np.asarray(jax.device_get(object_metrics["kl_mc_mean"]), dtype=float)
    chi2 = np.asarray(
        jax.device_get(object_metrics["posterior_predictive_chi2"]), dtype=float
    )
    rows = []
    for index, object_id in enumerate(object_ids):
        z = redshift_lookup.get(int(object_id), float("nan"))
        bin_index = _redshift_bin_index(z, bins)
        rows.append(
            {
                "object_id": int(object_id),
                "z_reference": float(z),
                "z_bin_index": int(bin_index),
                "loss": float(losses[index]),
                "negative_loglike": float(nll[index]),
                "kl_mc_mean": float(kl[index]),
                "posterior_predictive_chi2": float(chi2[index]),
            }
        )
    return rows


def _redshift_bin_rows(
    object_rows: list[dict[str, float | int | str]],
    *,
    bins: np.ndarray,
    epoch: int,
) -> list[dict[str, float | int | str]]:
    if not object_rows:
        return []
    frame = pd.DataFrame(object_rows)
    rows = []
    for bin_index, group in frame.groupby("z_bin_index", sort=True):
        bin_index = int(bin_index)
        if bin_index < 0:
            z_min = float("nan")
            z_max = float("nan")
            label = "missing"
        else:
            z_min = float(bins[bin_index])
            z_max = float(bins[bin_index + 1])
            label = f"{z_min:.3g}-{z_max:.3g}"
        rows.append(
            {
                "epoch": int(epoch),
                "split": "validation",
                "z_bin_index": bin_index,
                "z_bin": label,
                "z_min": z_min,
                "z_max": z_max,
                "n_objects": int(len(group)),
                "loss": float(np.nanmean(group["loss"])),
                "negative_loglike": float(np.nanmean(group["negative_loglike"])),
                "kl_mc_mean": float(np.nanmean(group["kl_mc_mean"])),
                "posterior_predictive_chi2": float(
                    np.nanmedian(group["posterior_predictive_chi2"])
                ),
                "z_reference_median": float(np.nanmedian(group["z_reference"])),
            }
        )
    return rows


def _catalog_num_rows(path: str | Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _configured_redshift_column(
    config: dict[str, Any],
    data_cfg: dict[str, Any],
) -> str | None:
    explicit = data_cfg.get("stratify_column")
    if explicit:
        return str(explicit)
    truth = config.get("truth", {}) or {}
    column = truth_column_from_spec(truth.get("redshift_column"))
    if column:
        return column
    redshift = config.get("redshift", {}) or {}
    return truth_column_from_spec(redshift.get("truth_column"))


def _read_redshift_column(
    catalog_path: str | Path,
    column: str | None,
) -> np.ndarray | None:
    if column is None:
        return None
    try:
        frame = pd.read_parquet(catalog_path, columns=[column])
    except Exception:
        return None
    if column not in frame:
        return None
    values = frame[column].to_numpy(dtype=float)
    return values


def _select_stratified_indices(
    redshift: np.ndarray,
    *,
    total: int,
    bins: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    all_indices = np.arange(len(redshift), dtype=np.int64)
    if total >= len(all_indices):
        selected = all_indices.copy()
        rng.shuffle(selected)
        return selected
    finite = np.isfinite(redshift)
    if not finite.any():
        return rng.choice(all_indices, size=total, replace=False).astype(np.int64)
    if strategy == "proportional":
        finite_indices = all_indices[finite]
        if total <= len(finite_indices):
            return rng.choice(finite_indices, size=total, replace=False).astype(
                np.int64
            )
        selected = list(finite_indices)
        remaining = np.setdiff1d(all_indices, finite_indices, assume_unique=False)
        needed = total - len(selected)
        selected.extend(rng.choice(remaining, size=needed, replace=False).tolist())
        selected = np.asarray(selected, dtype=np.int64)
        rng.shuffle(selected)
        return selected
    if strategy != "balanced":
        raise ValueError(
            "amortized.data.stratified_strategy must be balanced or proportional"
        )
    groups = _indices_by_redshift_bin(redshift, bins)
    nonempty = [group for group in groups if len(group)]
    if not nonempty:
        return rng.choice(all_indices, size=total, replace=False).astype(np.int64)
    per_bin = int(np.ceil(total / float(len(nonempty))))
    selected: list[int] = []
    for group in nonempty:
        take = min(per_bin, len(group), total - len(selected))
        if take <= 0:
            break
        selected.extend(rng.choice(group, size=take, replace=False).tolist())
    if len(selected) < total:
        remaining = np.setdiff1d(all_indices, np.asarray(selected), assume_unique=False)
        selected.extend(
            rng.choice(remaining, size=total - len(selected), replace=False).tolist()
        )
    selected = np.asarray(selected[:total], dtype=np.int64)
    rng.shuffle(selected)
    return selected


def _split_train_validation(
    selected: np.ndarray,
    redshift: np.ndarray | None,
    bins: np.ndarray,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(selected, dtype=np.int64)
    if validation_fraction <= 0.0 or len(selected) < 2:
        return selected, np.asarray([], dtype=np.int64)
    if redshift is None:
        shuffled = selected.copy()
        rng.shuffle(shuffled)
        n_val = int(round(validation_fraction * len(shuffled)))
        n_val = min(max(n_val, 1), len(shuffled) - 1)
        return shuffled[n_val:], shuffled[:n_val]
    val: list[int] = []
    train: list[int] = []
    for group in _indices_by_redshift_bin(
        redshift[selected], bins, base_indices=selected
    ):
        group = np.asarray(group, dtype=np.int64)
        if len(group) == 0:
            continue
        rng.shuffle(group)
        n_val = int(round(validation_fraction * len(group)))
        if len(group) > 1 and validation_fraction > 0.0:
            n_val = min(max(n_val, 1), len(group) - 1)
        val.extend(group[:n_val].tolist())
        train.extend(group[n_val:].tolist())
    if not train:
        shuffled = selected.copy()
        rng.shuffle(shuffled)
        return shuffled[1:], shuffled[:1]
    train = np.asarray(train, dtype=np.int64)
    val = np.asarray(val, dtype=np.int64)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _indices_by_redshift_bin(
    redshift: np.ndarray,
    bins: np.ndarray,
    *,
    base_indices: np.ndarray | None = None,
) -> list[np.ndarray]:
    if base_indices is None:
        base_indices = np.arange(len(redshift), dtype=np.int64)
    redshift = np.asarray(redshift, dtype=float)
    base_indices = np.asarray(base_indices, dtype=np.int64)
    groups = []
    for index in range(len(bins) - 1):
        lo = bins[index]
        hi = bins[index + 1]
        if index == len(bins) - 2:
            mask = (redshift >= lo) & (redshift <= hi)
        else:
            mask = (redshift >= lo) & (redshift < hi)
        mask &= np.isfinite(redshift)
        groups.append(base_indices[mask])
    missing = base_indices[~np.isfinite(redshift)]
    if len(missing):
        groups.append(missing)
    return groups


def _redshift_for_indices(
    redshift: np.ndarray | None,
    indices: np.ndarray,
) -> np.ndarray:
    if redshift is None or len(indices) == 0:
        return np.asarray([], dtype=float)
    return np.asarray(redshift[np.asarray(indices, dtype=int)], dtype=float)


def _redshift_bin_index(z: float, bins: np.ndarray) -> int:
    if not np.isfinite(z):
        return -1
    index = int(np.searchsorted(bins, z, side="right") - 1)
    if index < 0 or index >= len(bins) - 1:
        return -1
    return index


def _loss_with_metrics(
    model,
    batch,
    latent_spec,
    context,
    model_args,
    parameter_names,
    key,
    n_samples,
    kl_weight,
    likelihood_config,
    calibration_config,
):
    return negative_elbo(
        model,
        batch,
        latent_spec,
        context,
        model_args,
        parameter_names,
        key,
        n_samples,
        kl_weight,
        likelihood_config,
        calibration_config,
    )


def _loss_batch(batch: PhotometryBatch) -> LossBatch:
    return LossBatch(
        flux=batch.flux,
        flux_err=batch.flux_err,
        mask=batch.mask,
        features=batch.features,
    )


@eqx.filter_jit
def _loss_and_grads_jit(
    model,
    batch,
    latent_spec,
    context,
    model_args,
    parameter_names,
    key,
    n_samples,
    kl_weight,
    likelihood_config,
    calibration_config,
):
    actual_context = context.value if isinstance(context, _StaticArg) else context
    return eqx.filter_value_and_grad(_loss_with_metrics, has_aux=True)(
        model,
        batch,
        latent_spec,
        actual_context,
        model_args,
        parameter_names,
        key,
        n_samples,
        kl_weight,
        likelihood_config,
        calibration_config,
    )


@eqx.filter_jit
def _evaluation_metrics_jit(
    model,
    batch,
    latent_spec,
    context,
    model_args,
    parameter_names,
    key,
    n_samples,
    kl_weight,
    likelihood_config,
    calibration_config,
):
    actual_context = context.value if isinstance(context, _StaticArg) else context
    return _evaluation_metrics(
        model,
        batch,
        latent_spec,
        actual_context,
        model_args,
        parameter_names,
        key,
        n_samples,
        kl_weight,
        likelihood_config,
        calibration_config,
    )


def _kl_weight(epoch: int, annealing_epochs: int, *, max_weight: float = 1.0) -> float:
    max_weight = min(max(float(max_weight), 0.0), 1.0)
    if annealing_epochs <= 0:
        return max_weight
    return max_weight * min(1.0, max(0.0, float(epoch) / float(annealing_epochs)))


def _metrics_record(metrics: dict[str, jnp.ndarray]) -> dict[str, float]:
    return {
        name: float(np.asarray(jax.device_get(value)))
        for name, value in metrics.items()
    }


def component_grad_norms(grads: AmortizedModel) -> dict[str, float]:
    """Return L2 gradient norms for the jointly trained neural components."""
    encoder_norm = _tree_l2_norm(getattr(grads, "encoder", None))
    prior_norm = _tree_l2_norm(getattr(grads, "prior", None))
    alpha_norm = _tree_l2_norm(getattr(grads, "sed_scale", None))
    return {
        "encoder_grad_norm": encoder_norm,
        "prior_grad_norm": prior_norm,
        "alpha_grad_norm": alpha_norm,
        "joint_grad_norm": _tree_l2_norm(grads),
        "encoder_grad_nonzero": float(encoder_norm > 0.0),
        "prior_grad_nonzero": float(prior_norm > 0.0),
        "alpha_grad_nonzero": float(alpha_norm > 0.0),
    }


def zero_prior_grads(grads: AmortizedModel) -> AmortizedModel:
    """Return gradients with all trainable prior leaves set to zero."""
    prior_grads = getattr(grads, "prior", None)
    if prior_grads is None:
        return grads

    def zero_leaf(leaf):
        if eqx.is_inexact_array(leaf):
            return jnp.zeros_like(leaf)
        return leaf

    return eqx.tree_at(
        lambda tree: tree.prior,
        grads,
        jax.tree_util.tree_map(zero_leaf, prior_grads),
    )


def zero_sed_scale_grads(grads: AmortizedModel) -> AmortizedModel:
    """Return gradients with global SED-scale leaves set to zero."""
    sed_scale_grads = getattr(grads, "sed_scale", None)
    if sed_scale_grads is None:
        return grads

    def zero_leaf(leaf):
        if eqx.is_inexact_array(leaf):
            return jnp.zeros_like(leaf)
        return leaf

    return eqx.tree_at(
        lambda tree: tree.sed_scale,
        grads,
        jax.tree_util.tree_map(zero_leaf, sed_scale_grads),
    )


def _train_prior_jointly(prior_cfg: dict[str, Any]) -> bool:
    source = str(prior_cfg.get("source", "joint_realnvp"))
    if source == "standard_normal":
        return False
    return bool(prior_cfg.get("train_jointly", source == "joint_realnvp"))


def tree_all_finite(tree) -> bool:
    """Return whether every inexact array leaf in ``tree`` is finite."""
    leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_inexact_array(leaf)
    ]
    if not leaves:
        return True
    finite_flags = [jnp.all(jnp.isfinite(leaf)) for leaf in leaves]
    return bool(np.asarray(jax.device_get(jnp.all(jnp.asarray(finite_flags)))))


def architecture_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON architecture summary for checkpoint sidecars."""
    cfg = amortized_config(config)
    train_prior = _train_prior_jointly(cfg["prior"])
    scale_cfg = global_sed_scale_config(config)
    optimized = ["encoder"]
    if train_prior:
        optimized.append("realnvp_prior")
    if scale_cfg.enabled and scale_cfg.trainable:
        optimized.append("global_log_alpha_sed")
    return {
        "encoder": {
            "type": cfg["encoder"].get("type", "gaussian_mlp"),
            "input_dim": int(cfg["encoder"].get("input_dim", 20)),
            "latent_dim": int(cfg["encoder"].get("latent_dim", 16)),
            "input_contract": (
                f"flux_{int(cfg['features'].get('n_flux_bands', 10))}"
                f"_then_err_{int(cfg['features'].get('n_error_bands', 10))}"
            ),
            "flux_transform": cfg["features"].get("flux_transform", "asinh"),
            "error_transform": cfg["features"].get("error_transform", "log"),
            "hidden_sizes": list(cfg["encoder"].get("hidden_sizes", [])),
            "activation": cfg["encoder"].get("activation", "gelu"),
            "posterior_family": "diagonal_gaussian",
        },
        "prior": {
            "type": cfg["prior"].get("type", "realnvp"),
            "source": cfg["prior"].get("source", "joint_realnvp"),
            "train_jointly": _train_prior_jointly(cfg["prior"]),
            "checkpoint": cfg["prior"].get("checkpoint"),
            "latent_dim": int(cfg["encoder"].get("latent_dim", 16)),
            "n_layers": int(cfg["prior"].get("n_layers", 8)),
            "hidden_size": int(cfg["prior"].get("hidden_size", 128)),
            "scale_clamp": float(cfg["prior"].get("scale_clamp", 0.05)),
            "density": "exact_change_of_variables",
        },
        "global_sed_scale": {
            "enabled": scale_cfg.enabled,
            "mode": scale_cfg.mode,
            "trainable": scale_cfg.trainable,
            "parameterization": "log_alpha",
            "prior_sigma_log_alpha": scale_cfg.prior_sigma_log_alpha,
        },
        "decoder": {
            "type": "fixed_dsps",
            "trainable": False,
            "input": "theta_popcosmos_16",
            "output": f"flux_{int(cfg['features'].get('n_flux_bands', 10))}",
        },
        "objective": {
            "loss": "negative_elbo",
            "kl_estimator": "monte_carlo_logq_minus_logp",
            "likelihood": cfg["likelihood"].get("type", "student_t"),
            "jointly_optimized_components": optimized,
        },
    }


def _tree_l2_norm(tree) -> float:
    leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_inexact_array(leaf)
    ]
    if not leaves:
        return 0.0
    total = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
    return float(np.asarray(jax.device_get(jnp.sqrt(total))))


def _finite_mean(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _effective_best_checkpoint_metric(
    configured: str,
    *,
    has_validation: bool,
) -> str:
    metric = str(configured or "").strip() or "validation_negative_loglike"
    if has_validation:
        return metric
    if metric.startswith("validation_"):
        return "train_loss"
    return metric


def _best_checkpoint_min_epoch(
    training_cfg: dict[str, Any],
    *,
    has_validation: bool,
) -> int:
    configured = training_cfg.get("best_checkpoint_min_epoch")
    if configured is None:
        configured = training_cfg.get("kl_annealing_epochs", 1) if has_validation else 1
    return max(1, int(configured))


def _checkpoint_metric_from_rows(
    rows: list[dict[str, Any]],
    metric_name: str,
) -> float:
    column = _checkpoint_metric_column(metric_name)
    return _finite_mean([float(row.get(column, float("nan"))) for row in rows])


def _checkpoint_metric_column(metric_name: str) -> str:
    metric = str(metric_name)
    for prefix in ("validation_", "train_"):
        if metric.startswith(prefix):
            metric = metric[len(prefix) :]
            break
    aliases = {
        "nll": "negative_loglike",
        "negative_log_likelihood": "negative_loglike",
        "posterior_predictive_chi2": "posterior_predictive_chi2",
    }
    return aliases.get(metric, metric)


def _should_update_best_checkpoint(
    *,
    epoch: int,
    value: float,
    best_value: float,
    min_epoch: int,
) -> bool:
    return int(epoch) >= int(min_epoch) and np.isfinite(value) and value < best_value


def write_training_progress(
    out: Path,
    *,
    rows: list[dict[str, float]],
    epoch: int,
    best_loss: float,
    start_time: float,
    checkpoint_last: str,
    best_checkpoint_metric: str | None = None,
    best_checkpoint_epoch: int | None = None,
) -> None:
    if not rows:
        return
    first_loss = float(rows[0]["loss"])
    last_loss = float(rows[-1]["loss"])
    write_json(
        out / "training_progress.json",
        {
            "epoch": int(epoch),
            "steps": int(len(rows)),
            "first_loss": first_loss,
            "last_loss": last_loss,
            "best_loss": float(best_loss),
            "best_checkpoint_metric": best_checkpoint_metric,
            "best_checkpoint_epoch": (
                int(best_checkpoint_epoch)
                if best_checkpoint_epoch is not None
                else None
            ),
            "loss_improved_from_first": bool(
                np.isfinite(first_loss)
                and np.isfinite(best_loss)
                and best_loss < first_loss
            ),
            "elapsed_time_s": float(time.time() - start_time),
            "checkpoint_last": checkpoint_last,
            "latest_encoder_grad_norm": float(rows[-1].get("encoder_grad_norm", 0.0)),
            "latest_prior_grad_norm": float(rows[-1].get("prior_grad_norm", 0.0)),
            "updates_applied": int(
                sum(row.get("update_applied", 0.0) for row in _training_rows(rows))
            ),
            "updates_skipped": int(
                sum(
                    1.0 - row.get("update_applied", 0.0) for row in _training_rows(rows)
                )
            ),
        },
    )


def _expected_batch_count(limit: int | None, batch_size: int) -> int | None:
    if limit is None:
        return None
    if limit <= 0:
        return 0
    return int(np.ceil(float(limit) / float(max(batch_size, 1))))


def _effective_jax_batch_size(training_config: dict[str, Any], batch_size: int) -> int:
    requested = max(int(batch_size), 1)
    configured = training_config.get("jax_batch_size")
    if configured is None:
        return requested
    value = int(configured)
    if value <= 0:
        raise ValueError("amortized.training.jax_batch_size must be positive")
    return min(requested, value)


def _training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("split", "train") == "train"]


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _batch_progress_line(record: dict[str, float]) -> str:
    return (
        "[amortized] "
        f"epoch={int(record['epoch'])} batch={int(record['batch'])} "
        f"n={int(record['n_objects'])} loss={record['loss']:.6g} "
        f"nll={record['negative_loglike']:.6g} kl={record['kl_mc_mean']:.6g} "
        f"logq={record['logq_mean']:.6g} logprior={record['logprior_mean']:.6g} "
        f"enc_grad={record['encoder_grad_norm']:.3g} "
        f"prior_grad={record['prior_grad_norm']:.3g} "
        f"update={int(record.get('update_applied', 0.0))}"
    )


def _progress_bar(*, enabled: bool, total: int | None, desc: str, unit: str):
    if not enabled or tqdm is None:
        return _NullProgress()
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        mininterval=0.2,
        smoothing=0.05,
    )


class _NullProgress:
    def update(self, _: int = 1) -> None:
        return None

    def set_postfix(self, *_args, **_kwargs) -> None:
        return None

    def __enter__(self) -> _NullProgress:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None
