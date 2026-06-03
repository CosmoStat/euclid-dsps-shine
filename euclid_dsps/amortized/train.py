"""Training loop for FS2 amortized inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.filters import load_filters
from euclid_dsps.io import ensure_dir, write_json
from euclid_dsps.model import dynamic_model_args, load_context

from .config import amortized_config, require_amortized_dependencies
from .data import (
    compute_fs2_feature_stats_from_config,
    iter_fs2_photometry_batches_from_config,
)
from .diagnostics import write_training_diagnostics
from .elbo import AmortizedModel, negative_elbo
from .encoder import GaussianEncoder
from .features import FeatureStats, write_feature_stats
from .flows import RealNVPPrior
from .latent import (
    LatentSpec,
    latent_spec_from_config,
    latent_spec_to_jsonable,
    theta_to_x,
)

eqx, optax = require_amortized_dependencies()

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is a core dependency today
    tqdm = None


def build_amortized_model(config: dict[str, Any], key) -> AmortizedModel:
    """Instantiate encoder and RealNVP prior from config."""
    cfg = amortized_config(config)
    k_encoder, k_prior = jax.random.split(key)
    encoder_cfg = cfg["encoder"]
    prior_cfg = cfg["prior"]
    encoder = GaussianEncoder(
        k_encoder,
        input_dim=20,
        latent_dim=16,
        hidden_sizes=tuple(
            int(v) for v in encoder_cfg.get("hidden_sizes", [256, 256, 256])
        ),
        activation=str(encoder_cfg.get("activation", "gelu")),
        log_std_min=float(encoder_cfg.get("log_std_min", -6.0)),
        log_std_max=float(encoder_cfg.get("log_std_max", 2.0)),
        initial_log_std=float(encoder_cfg.get("initial_log_std", -1.0)),
    )
    encoder = _initialize_encoder_mean_if_possible(config, encoder)
    prior = RealNVPPrior(
        k_prior,
        latent_dim=16,
        n_layers=int(prior_cfg.get("n_layers", 8)),
        hidden_size=int(prior_cfg.get("hidden_size", 128)),
        scale_clamp=float(prior_cfg.get("scale_clamp", 0.05)),
    )
    return AmortizedModel(encoder=encoder, prior=prior)


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
) -> None:
    """Train encoder and RealNVP prior jointly on FS2 photometry."""
    out = ensure_dir(out_dir)
    cfg = amortized_config(config)
    _log(verbose, "[amortized] FS2 joint encoder/RealNVP training")
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
    _log(verbose, "[amortized] computing feature stats from FS2 flux/errors...")
    feature_stats = compute_fs2_feature_stats_from_config(
        config,
        limit=limit,
        batch_size=max(int(batch_size), 1),
    )
    write_feature_stats(out / "feature_stats.json", feature_stats)
    _log(
        verbose,
        "[amortized] feature stats ready: "
        f"{len(feature_stats.band_names)} bands, feature_dim=20 "
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
    _log(
        verbose,
        "[amortized] DSPS context ready: "
        f"{len(filters)} filters, n_sfh_bins={context.n_sfh_bins}, "
        f"model={context.model_config}",
    )
    latent_spec = latent_spec_from_config(config)
    _log(
        verbose,
        "[amortized] latent spec ready: "
        f"{len(latent_spec.names)} parameters, "
        f"first={latent_spec.names[0]}, last={latent_spec.names[-1]}",
    )
    key = jax.random.PRNGKey(int(seed))
    key, model_key = jax.random.split(key)
    model = build_amortized_model(config, model_key)
    _log(
        verbose,
        "[amortized] model built: "
        f"encoder_hidden={cfg['encoder'].get('hidden_sizes')} "
        f"realnvp_layers={cfg['prior'].get('n_layers')} "
        f"realnvp_hidden={cfg['prior'].get('hidden_size')}",
    )
    optimizer = make_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    ckpt_dir = ensure_dir(out / "checkpoints")
    rows = []
    best_loss = np.inf
    start_time = time.time()
    checkpoint_every = int(cfg["output"].get("checkpoint_every", 1))
    diagnostics_every = int(cfg["output"].get("diagnostics_every", 1))
    save_training_curves = bool(cfg["output"].get("save_training_curves", True))
    expected_batches = _expected_batch_count(limit, int(batch_size))
    _log(
        verbose,
        "[amortized] training start: "
        "first batch includes JAX/DSPS compilation and can be noticeably slower.",
    )
    for epoch in range(1, int(epochs) + 1):
        kl_weight = _kl_weight(
            epoch, int(cfg["training"].get("kl_annealing_epochs", 5))
        )
        epoch_rows = []
        _log(
            verbose, f"[amortized] epoch {epoch}/{int(epochs)} start kl={kl_weight:.3f}"
        )
        batch_iter = iter_fs2_photometry_batches_from_config(
            config,
            batch_size=int(batch_size),
            limit=limit,
            feature_stats=feature_stats,
        )
        with _progress_bar(
            enabled=bool(progress),
            total=expected_batches,
            desc=f"epoch {epoch}/{int(epochs)}",
            unit="batch",
        ) as pbar:
            for batch_index, batch in enumerate(batch_iter):
                key, step_key = jax.random.split(key)
                (loss, metrics), grads = eqx.filter_value_and_grad(
                    _loss_with_metrics,
                    has_aux=True,
                )(
                    model,
                    batch,
                    latent_spec,
                    context,
                    model_args,
                    latent_spec.names,
                    step_key,
                    int(n_samples),
                    float(kl_weight),
                    cfg["likelihood"],
                )
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
                if loss_finite and float(loss) < best_loss:
                    best_loss = float(loss)
                    save_checkpoint(
                        ckpt_dir / "best.eqx",
                        model,
                        config=config,
                        latent_spec=latent_spec,
                        feature_stats=feature_stats,
                        epoch=epoch,
                        metric=best_loss,
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
        save_checkpoint(
            ckpt_dir / "last.eqx",
            model,
            config=config,
            latent_spec=latent_spec,
            feature_stats=feature_stats,
            epoch=epoch,
            metric=_finite_mean([row["loss"] for row in epoch_rows]),
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
            )
        pd.DataFrame(rows).to_csv(out / "training_log.csv", index=False)
        write_training_progress(
            out,
            rows=rows,
            epoch=epoch,
            best_loss=best_loss,
            start_time=start_time,
            checkpoint_last="checkpoints/last.eqx",
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
            f"best_loss={best_loss:.6g} "
            f"last_checkpoint={ckpt_dir / 'last.eqx'}",
        )

    summary = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "n_samples": int(n_samples),
        "limit": limit,
        "best_loss": float(best_loss),
        "elapsed_time_s": float(time.time() - start_time),
        "checkpoint_best": "checkpoints/best.eqx",
        "checkpoint_last": "checkpoints/last.eqx",
        "checkpoint_every": checkpoint_every,
        "diagnostics_every": diagnostics_every,
        "updates_applied": int(sum(row.get("update_applied", 0.0) for row in rows)),
        "updates_skipped": int(
            sum(1.0 - row.get("update_applied", 0.0) for row in rows)
        ),
        "joint_training": {
            "encoder": True,
            "realnvp_prior": True,
            "decoder": False,
            "kl_estimator": "monte_carlo_logq_minus_logp",
        },
    }
    write_json(out / "training_summary.json", summary)
    if save_training_curves:
        write_training_diagnostics(out / "training_log.csv", out)
    _log(verbose, "[amortized] training complete")
    _log(verbose, f"[amortized] summary: {out / 'training_summary.json'}")
    _log(verbose, f"[amortized] progress: {out / 'training_progress.json'}")
    _log(verbose, f"[amortized] best checkpoint: {out / 'checkpoints' / 'best.eqx'}")


def save_checkpoint(
    path: str | Path,
    model: AmortizedModel,
    *,
    config: dict[str, Any],
    latent_spec: LatentSpec,
    feature_stats: FeatureStats,
    epoch: int,
    metric: float,
) -> None:
    """Write an Equinox checkpoint plus a JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)
    sidecar = {
        "epoch": int(epoch),
        "metric": float(metric),
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
    )


def _kl_weight(epoch: int, annealing_epochs: int) -> float:
    if annealing_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, float(epoch) / float(annealing_epochs)))


def _metrics_record(metrics: dict[str, jnp.ndarray]) -> dict[str, float]:
    return {
        name: float(np.asarray(jax.device_get(value)))
        for name, value in metrics.items()
    }


def component_grad_norms(grads: AmortizedModel) -> dict[str, float]:
    """Return L2 gradient norms for the jointly trained neural components."""
    encoder_norm = _tree_l2_norm(getattr(grads, "encoder", None))
    prior_norm = _tree_l2_norm(getattr(grads, "prior", None))
    return {
        "encoder_grad_norm": encoder_norm,
        "prior_grad_norm": prior_norm,
        "joint_grad_norm": _tree_l2_norm(grads),
        "encoder_grad_nonzero": float(encoder_norm > 0.0),
        "prior_grad_nonzero": float(prior_norm > 0.0),
    }


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
    return {
        "encoder": {
            "type": cfg["encoder"].get("type", "gaussian_mlp"),
            "input_dim": 20,
            "latent_dim": 16,
            "input_contract": "flux_10_then_err_10",
            "flux_transform": cfg["features"].get("flux_transform", "asinh"),
            "error_transform": cfg["features"].get("error_transform", "log"),
            "hidden_sizes": list(cfg["encoder"].get("hidden_sizes", [])),
            "activation": cfg["encoder"].get("activation", "gelu"),
            "posterior_family": "diagonal_gaussian",
        },
        "prior": {
            "type": cfg["prior"].get("type", "realnvp"),
            "latent_dim": 16,
            "n_layers": int(cfg["prior"].get("n_layers", 8)),
            "hidden_size": int(cfg["prior"].get("hidden_size", 128)),
            "scale_clamp": float(cfg["prior"].get("scale_clamp", 0.05)),
            "density": "exact_change_of_variables",
        },
        "decoder": {
            "type": "fixed_dsps",
            "trainable": False,
            "input": "theta_popcosmos_16",
            "output": "flux_10",
        },
        "objective": {
            "loss": "negative_elbo",
            "kl_estimator": "monte_carlo_logq_minus_logp",
            "likelihood": cfg["likelihood"].get("type", "student_t"),
            "jointly_optimized_components": ["encoder", "realnvp_prior"],
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


def write_training_progress(
    out: Path,
    *,
    rows: list[dict[str, float]],
    epoch: int,
    best_loss: float,
    start_time: float,
    checkpoint_last: str,
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
            "loss_improved_from_first": bool(
                np.isfinite(first_loss)
                and np.isfinite(best_loss)
                and best_loss < first_loss
            ),
            "elapsed_time_s": float(time.time() - start_time),
            "checkpoint_last": checkpoint_last,
            "latest_encoder_grad_norm": float(rows[-1].get("encoder_grad_norm", 0.0)),
            "latest_prior_grad_norm": float(rows[-1].get("prior_grad_norm", 0.0)),
            "updates_applied": int(sum(row.get("update_applied", 0.0) for row in rows)),
            "updates_skipped": int(
                sum(1.0 - row.get("update_applied", 0.0) for row in rows)
            ),
        },
    )


def _expected_batch_count(limit: int | None, batch_size: int) -> int | None:
    if limit is None:
        return None
    if limit <= 0:
        return 0
    return int(np.ceil(float(limit) / float(max(batch_size, 1))))


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
