"""Frozen-prior encoder distillation from weighted joint latent-x particles."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .config import require_amortized_dependencies
from .posterior import posterior_log_prob

eqx, optax = require_amortized_dependencies()


@dataclass(frozen=True)
class ProposalRefreshResult:
    model: object
    history: tuple[dict[str, float | int], ...]
    train_indices: np.ndarray
    validation_indices: np.ndarray
    initial_validation_nll: float
    best_validation_nll: float
    best_epoch: int


def refresh_encoder_from_weighted_particles(
    model,
    *,
    features: jnp.ndarray,
    particles: jnp.ndarray,
    weights: jnp.ndarray,
    epochs: int = 20,
    object_batch_size: int = 32,
    learning_rate: float = 2.0e-5,
    weight_decay: float = 1.0e-6,
    validation_fraction: float = 0.1,
    seed: int = 260817,
) -> ProposalRefreshResult:
    """Minimize weighted ``-log q`` while leaving the population prior frozen."""
    features = jnp.asarray(features, dtype=jnp.float32)
    particles = jnp.asarray(particles, dtype=jnp.float32)
    weights = jnp.asarray(weights, dtype=jnp.float32)
    if particles.ndim != 3:
        raise ValueError("particles must have shape [K, N, D]")
    if weights.shape != particles.shape[:2]:
        raise ValueError("weights must have shape [K, N]")
    if features.ndim != 2 or features.shape[0] != particles.shape[1]:
        raise ValueError("features must have shape [N, F] matching particles")
    if int(epochs) <= 0 or int(object_batch_size) <= 0:
        raise ValueError("epochs and object_batch_size must be positive")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    if not bool(jnp.all(jnp.isfinite(particles))):
        raise ValueError("particles contain non-finite values")
    if not bool(jnp.all(jnp.isfinite(weights))) or bool(jnp.any(weights < 0.0)):
        raise ValueError("weights must be finite and non-negative")
    weight_sum = jnp.sum(weights, axis=0, keepdims=True)
    if not bool(jnp.all(weight_sum > 0.0)):
        raise ValueError("every object must have positive total particle weight")
    weights = weights / weight_sum

    n_objects = int(features.shape[0])
    if n_objects < 2:
        raise ValueError("proposal refresh requires at least two objects")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n_objects)
    n_validation = min(
        n_objects - 1,
        max(1, int(round(float(validation_fraction) * n_objects))),
    )
    validation_indices = np.sort(order[:n_validation])
    train_indices = np.sort(order[n_validation:])

    frozen_model = model
    encoder = model.encoder
    optimizer = optax.adamw(
        learning_rate=float(learning_rate), weight_decay=float(weight_decay)
    )
    opt_state = optimizer.init(eqx.filter(encoder, eqx.is_inexact_array))

    @eqx.filter_jit
    def loss_and_grad(
        candidate_encoder, batch_features, batch_particles, batch_weights
    ):
        def loss_fn(value):
            candidate_model = eqx.tree_at(
                lambda item: item.encoder, frozen_model, value
            )
            logq = posterior_log_prob(candidate_model, batch_features, batch_particles)
            return -jnp.mean(jnp.sum(batch_weights * logq, axis=0))

        return eqx.filter_value_and_grad(loss_fn)(candidate_encoder)

    @eqx.filter_jit
    def evaluate(candidate_encoder, batch_features, batch_particles, batch_weights):
        candidate_model = eqx.tree_at(
            lambda item: item.encoder, frozen_model, candidate_encoder
        )
        logq = posterior_log_prob(candidate_model, batch_features, batch_particles)
        return -jnp.mean(jnp.sum(batch_weights * logq, axis=0))

    initial_validation_nll = float(
        evaluate(
            encoder,
            features[validation_indices],
            particles[:, validation_indices],
            weights[:, validation_indices],
        )
    )
    best_validation_nll = initial_validation_nll
    best_encoder = encoder
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(epochs) + 1):
        shuffled = rng.permutation(train_indices)
        losses = []
        for start in range(0, len(shuffled), int(object_batch_size)):
            index = shuffled[start : start + int(object_batch_size)]
            loss, grads = loss_and_grad(
                encoder,
                features[index],
                particles[:, index],
                weights[:, index],
            )
            updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(encoder, eqx.is_inexact_array)
            )
            encoder = eqx.apply_updates(encoder, updates)
            losses.append(float(loss))
        validation_nll = float(
            evaluate(
                encoder,
                features[validation_indices],
                particles[:, validation_indices],
                weights[:, validation_indices],
            )
        )
        record = {
            "epoch": epoch,
            "train_weighted_nll": float(np.mean(losses)),
            "validation_weighted_nll": validation_nll,
        }
        history.append(record)
        print(
            "[proposal-refresh] "
            f"epoch={epoch}/{epochs} train_nll={record['train_weighted_nll']:.6g} "
            f"validation_nll={validation_nll:.6g}",
            flush=True,
        )
        if np.isfinite(validation_nll) and validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            best_encoder = encoder
            best_epoch = epoch
    best_model = eqx.tree_at(lambda item: item.encoder, model, best_encoder)
    return ProposalRefreshResult(
        model=best_model,
        history=tuple(history),
        train_indices=train_indices,
        validation_indices=validation_indices,
        initial_validation_nll=initial_validation_nll,
        best_validation_nll=best_validation_nll,
        best_epoch=best_epoch,
    )
