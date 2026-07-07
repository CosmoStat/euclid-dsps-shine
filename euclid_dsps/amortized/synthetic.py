"""Asset-free synthetic smoke tests for amortized inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir, write_json

from .config import amortized_config, require_amortized_dependencies
from .data import PhotometryBatch
from .diagnostics import write_training_diagnostics
from .elbo import negative_elbo
from .features import compute_feature_stats, make_encoder_features, write_feature_stats
from .latent import LatentSpec
from .train import (
    build_amortized_model,
    component_grad_norms,
    make_optimizer,
    save_checkpoint,
    write_training_progress,
)

eqx, optax = require_amortized_dependencies()


def run_synthetic_smoke(
    config: dict[str, Any],
    out_dir: Path,
    *,
    n_objects: int = 128,
    epochs: int = 2,
    batch_size: int = 32,
    seed: int = 42,
    mock_decoder: bool = True,
) -> None:
    """Run a small joint encoder/RealNVP training smoke with a mock decoder."""
    if not mock_decoder:
        raise ValueError("Only --mock-decoder synthetic smoke is implemented")
    out = ensure_dir(out_dir)
    cfg = amortized_config(config)
    key = jax.random.PRNGKey(int(seed))
    key, data_key, decoder_key, model_key = jax.random.split(key, 4)
    n_bands = int(cfg["data"].get("expected_n_bands", 10))
    data = _make_synthetic_data(data_key, decoder_key, int(n_objects), n_bands)
    stats = compute_feature_stats(
        np.asarray(data["flux"]),
        np.asarray(data["flux_err"]),
        np.asarray(data["mask"]),
        band_names=tuple(f"mock_band_{index}" for index in range(n_bands)),
    )
    write_feature_stats(out / "feature_stats.json", stats)
    model = build_amortized_model(config, model_key)
    optimizer = make_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    latent_spec = LatentSpec(
        names=tuple(f"theta_{index}" for index in range(16)),
        lower=jnp.zeros(16, dtype=jnp.float32),
        upper=jnp.ones(16, dtype=jnp.float32),
    )
    rows = []
    best_loss = np.inf
    start = time.time()
    checkpoint_every = int(cfg["output"].get("checkpoint_every", 1))
    diagnostics_every = int(cfg["output"].get("diagnostics_every", 1))
    save_training_curves = bool(cfg["output"].get("save_training_curves", True))
    for epoch in range(1, int(epochs) + 1):
        order = np.arange(int(n_objects))
        np.random.default_rng(int(seed) + epoch).shuffle(order)
        kl_weight = min(
            1.0,
            epoch / max(float(cfg["training"].get("kl_annealing_epochs", 5)), 1.0),
        )
        epoch_rows = []
        for batch_index, start_index in enumerate(
            range(0, len(order), int(batch_size))
        ):
            selected = order[start_index : start_index + int(batch_size)]
            batch = _synthetic_batch(data, stats, selected)
            key, step_key = jax.random.split(key)
            (loss, metrics), grads = eqx.filter_value_and_grad(
                _synthetic_loss,
                has_aux=True,
            )(
                model,
                batch,
                latent_spec,
                step_key,
                int(cfg["training"].get("n_samples", 1)),
                float(kl_weight),
                cfg["likelihood"],
                data["decoder"],
            )
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                eqx.filter(model, eqx.is_inexact_array),
            )
            model = eqx.apply_updates(model, updates)
            record = {
                name: float(np.asarray(jax.device_get(value)))
                for name, value in metrics.items()
            }
            record.update(component_grad_norms(grads))
            record.update(
                {
                    "epoch": int(epoch),
                    "batch": int(batch_index),
                    "kl_weight": float(kl_weight),
                    "n_objects": int(len(selected)),
                }
            )
            rows.append(record)
            epoch_rows.append(record)
            if float(loss) < best_loss:
                best_loss = float(loss)
                save_checkpoint(
                    out / "checkpoints" / "best.eqx",
                    model,
                    config=config,
                    latent_spec=latent_spec,
                    feature_stats=stats,
                    epoch=epoch,
                    metric=best_loss,
                )
        epoch_loss = float(np.nanmean([row["loss"] for row in epoch_rows]))
        save_checkpoint(
            out / "checkpoints" / "last.eqx",
            model,
            config=config,
            latent_spec=latent_spec,
            feature_stats=stats,
            epoch=epoch,
            metric=epoch_loss,
        )
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(
                out / "checkpoints" / f"epoch_{epoch:04d}.eqx",
                model,
                config=config,
                latent_spec=latent_spec,
                feature_stats=stats,
                epoch=epoch,
                metric=epoch_loss,
            )
        pd.DataFrame(rows).to_csv(out / "training_log.csv", index=False)
        write_training_progress(
            out,
            rows=rows,
            epoch=epoch,
            best_loss=best_loss,
            start_time=start,
            checkpoint_last="checkpoints/last.eqx",
        )
        if (
            save_training_curves
            and diagnostics_every > 0
            and epoch % diagnostics_every == 0
        ):
            write_training_diagnostics(out / "training_log.csv", out)
    save_checkpoint(
        out / "checkpoints" / "last.eqx",
        model,
        config=config,
        latent_spec=latent_spec,
        feature_stats=stats,
        epoch=int(epochs),
        metric=float(rows[-1]["loss"]) if rows else float("nan"),
    )
    pd.DataFrame(rows).to_csv(out / "training_log.csv", index=False)
    first_loss = float(rows[0]["loss"]) if rows else float("nan")
    last_loss = float(rows[-1]["loss"]) if rows else float("nan")
    best_loss_value = float(best_loss)
    loss_decreased = bool(
        np.isfinite(first_loss)
        and np.isfinite(best_loss_value)
        and best_loss_value < first_loss
    )
    write_json(
        out / "training_summary.json",
        {
            "mode": "synthetic_mock_decoder",
            "n_objects": int(n_objects),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "first_loss": first_loss,
            "last_loss": last_loss,
            "loss_decreased": loss_decreased,
            "last_loss_decreased": bool(
                np.isfinite(first_loss) and last_loss < first_loss
            ),
            "best_loss": best_loss_value,
            "checkpoint_every": checkpoint_every,
            "diagnostics_every": diagnostics_every,
            "joint_training": {
                "encoder": True,
                "realnvp_prior": True,
                "decoder": False,
                "kl_estimator": "monte_carlo_logq_minus_logp",
            },
            "elapsed_time_s": float(time.time() - start),
        },
    )
    write_training_progress(
        out,
        rows=rows,
        epoch=int(epochs),
        best_loss=best_loss_value,
        start_time=start,
        checkpoint_last="checkpoints/last.eqx",
    )
    if save_training_curves:
        write_training_diagnostics(out / "training_log.csv", out)


def _make_synthetic_data(
    data_key,
    decoder_key,
    n_objects: int,
    n_bands: int = 10,
) -> dict[str, Any]:
    x_true = jax.random.normal(data_key, (n_objects, 16), dtype=jnp.float32)
    k_w, k_b, k_noise = jax.random.split(decoder_key, 3)
    weights = 0.08 * jax.random.normal(k_w, (16, int(n_bands)), dtype=jnp.float32)
    bias = -25.0 + 0.2 * jax.random.normal(k_b, (int(n_bands),), dtype=jnp.float32)
    clean_flux = jnp.exp(jnp.clip(x_true @ weights + bias, -30.0, 30.0))
    flux_err = 0.08 * jnp.maximum(clean_flux, jnp.median(clean_flux)) + 1.0e-13
    noisy_flux = clean_flux + flux_err * jax.random.normal(k_noise, clean_flux.shape)
    return {
        "x_true": x_true,
        "flux": noisy_flux,
        "flux_err": flux_err,
        "mask": jnp.ones_like(noisy_flux, dtype=bool),
        "decoder": {"weights": weights, "bias": bias},
    }


def _synthetic_batch(
    data: dict[str, Any], stats, indices: np.ndarray
) -> PhotometryBatch:
    flux = jnp.asarray(np.asarray(data["flux"])[indices], dtype=jnp.float32)
    flux_err = jnp.asarray(np.asarray(data["flux_err"])[indices], dtype=jnp.float32)
    mask = jnp.asarray(np.asarray(data["mask"])[indices])
    features = make_encoder_features(flux, flux_err, stats)
    return PhotometryBatch(
        object_id=jnp.asarray(indices, dtype=jnp.int32),
        flux=flux,
        flux_err=flux_err,
        mask=mask,
        features=features,
    )


def _synthetic_loss(
    model,
    batch,
    latent_spec,
    key,
    n_samples,
    kl_weight,
    likelihood_config,
    decoder_params,
):
    return negative_elbo(
        model,
        batch,
        latent_spec,
        None,
        None,
        latent_spec.names,
        key,
        n_samples,
        kl_weight,
        likelihood_config,
        use_mock_decoder=True,
        mock_decoder_params=decoder_params,
    )
