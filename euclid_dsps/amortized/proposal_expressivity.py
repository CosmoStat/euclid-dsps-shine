"""Controlled diagnostics for conditional posterior proposal capacity.

This module is intentionally separate from the production posterior API.  It
supports a diagnostic experiment comparing the existing single conditional
flow with a mixture whose experts each own a complete conditional flow.  The
population prior is never optimized here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from .config import require_amortized_dependencies
from .posterior import ConditionalFlowEncoder, posterior_log_prob, sample_posterior

eqx, optax = require_amortized_dependencies()


class IndependentFlowMixture(eqx.Module):
    """Conditional mixture with one complete flow per expert."""

    experts: tuple
    gate: object
    n_components: int = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)

    def __init__(
        self,
        key,
        source_encoder: ConditionalFlowEncoder,
        *,
        n_components: int = 2,
        mean_offset: float = 0.05,
    ) -> None:
        if not isinstance(source_encoder, ConditionalFlowEncoder):
            raise TypeError("independent experts require a ConditionalFlowEncoder")
        if source_encoder.output_space != "latent_x":
            raise ValueError("independent experts currently require latent_x output")
        if int(n_components) < 2:
            raise ValueError("n_components must be at least two")
        if float(mean_offset) <= 0.0:
            raise ValueError("mean_offset must be positive")
        if not hasattr(source_encoder.base, "mean_head"):
            raise TypeError("source encoder must have a Gaussian base")

        offsets = np.linspace(-float(mean_offset), float(mean_offset), n_components)
        direction = jnp.where(jnp.arange(source_encoder.latent_dim) % 2 == 0, 1.0, -1.0)
        self.experts = tuple(
            eqx.tree_at(
                lambda encoder: encoder.base.mean_head.bias,
                source_encoder,
                source_encoder.base.mean_head.bias + float(offset) * direction,
            )
            for offset in offsets
        )
        input_dim = int(source_encoder.base.trunk[0].weight.shape[1])
        gate_key, _ = jax.random.split(key)
        gate = eqx.nn.MLP(
            in_size=input_dim,
            out_size=int(n_components),
            width_size=max(32, min(256, 2 * source_encoder.latent_dim)),
            depth=2,
            activation=jax.nn.gelu,
            key=gate_key,
        )
        self.gate = eqx.tree_at(
            lambda model: (model.layers[-1].weight, model.layers[-1].bias),
            gate,
            (
                jnp.zeros_like(gate.layers[-1].weight),
                jnp.zeros_like(gate.layers[-1].bias),
            ),
        )
        self.n_components = int(n_components)
        self.input_dim = input_dim

    def logits(self, features) -> jnp.ndarray:
        features = jnp.asarray(features, dtype=jnp.float32)
        if features.ndim == 1:
            return self.gate(features)
        return jax.vmap(self.gate)(features)


class MixturePosteriorSample(NamedTuple):
    x: jnp.ndarray
    logq: jnp.ndarray
    logprior: jnp.ndarray
    component: jnp.ndarray


@dataclass(frozen=True)
class ExpressivityFitResult:
    candidate: object
    history: tuple[dict[str, float | int], ...]
    initial_train_nll: float
    final_train_nll: float
    initial_validation_nll: float
    best_validation_nll: float
    best_epoch: int


def independent_mixture_log_prob(model, mixture, features, x) -> jnp.ndarray:
    """Evaluate the exact conditional density of independent flow experts."""
    component_logq = []
    for expert in mixture.experts:
        expert_model = eqx.tree_at(lambda item: item.encoder, model, expert)
        component_logq.append(posterior_log_prob(expert_model, features, x))
    stacked = jnp.stack(component_logq, axis=0)
    log_weights = jax.nn.log_softmax(mixture.logits(features), axis=-1)
    log_weights = jnp.moveaxis(log_weights, -1, 0)
    while log_weights.ndim < stacked.ndim:
        log_weights = jnp.expand_dims(log_weights, axis=1)
    return jax.scipy.special.logsumexp(stacked + log_weights, axis=0)


def sample_independent_mixture(
    model,
    mixture,
    key,
    features,
    n_samples: int,
) -> MixturePosteriorSample:
    """Sample the independent-expert mixture and return its exact log density."""
    component_key, *expert_keys = jax.random.split(key, mixture.n_components + 1)
    logits = mixture.logits(features)
    component = jax.random.categorical(
        component_key,
        logits,
        axis=-1,
        shape=(int(n_samples),) + logits.shape[:-1],
    )
    expert_x = []
    for expert, expert_key in zip(mixture.experts, expert_keys, strict=True):
        expert_model = eqx.tree_at(lambda item: item.encoder, model, expert)
        expert_x.append(
            sample_posterior(expert_model, expert_key, features, int(n_samples)).x
        )
    stacked = jnp.stack(expert_x, axis=0)
    one_hot = jax.nn.one_hot(component, mixture.n_components, dtype=stacked.dtype)
    x = jnp.sum(stacked * jnp.moveaxis(one_hot, -1, 0)[..., None], axis=0)
    logq = independent_mixture_log_prob(model, mixture, features, x)
    return MixturePosteriorSample(x, logq, model.prior.log_prob(x), component)


def fit_proposal_candidate(
    model,
    candidate,
    *,
    features,
    particles,
    weights,
    train_indices,
    validation_indices,
    epochs: int,
    object_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    progress_label: str = "candidate",
) -> ExpressivityFitResult:
    """Fit one proposal candidate on an explicit, shared object split."""
    features = jnp.asarray(features, dtype=jnp.float32)
    particles = jnp.asarray(particles, dtype=jnp.float32)
    weights = jnp.asarray(weights, dtype=jnp.float32)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    _validate_fit_inputs(
        features,
        particles,
        weights,
        train_indices,
        validation_indices,
        epochs=epochs,
        object_batch_size=object_batch_size,
    )
    weights = weights / jnp.sum(weights, axis=0, keepdims=True)
    is_independent = isinstance(candidate, IndependentFlowMixture)

    def candidate_log_prob(value, batch_features, batch_particles):
        if is_independent:
            return independent_mixture_log_prob(
                model, value, batch_features, batch_particles
            )
        candidate_model = eqx.tree_at(lambda item: item.encoder, model, value)
        return posterior_log_prob(candidate_model, batch_features, batch_particles)

    @eqx.filter_jit
    def loss_and_grad(value, batch_features, batch_particles, batch_weights):
        def loss_fn(item):
            logq = candidate_log_prob(item, batch_features, batch_particles)
            return -jnp.mean(jnp.sum(batch_weights * logq, axis=0))

        return eqx.filter_value_and_grad(loss_fn)(value)

    @eqx.filter_jit
    def evaluate(value, index):
        logq = candidate_log_prob(value, features[index], particles[:, index])
        return -jnp.mean(jnp.sum(weights[:, index] * logq, axis=0))

    initial_train = float(evaluate(candidate, train_indices))
    initial_validation = float(evaluate(candidate, validation_indices))
    optimizer = optax.adamw(
        learning_rate=float(learning_rate), weight_decay=float(weight_decay)
    )
    opt_state = optimizer.init(eqx.filter(candidate, eqx.is_inexact_array))
    best = candidate
    best_validation = initial_validation
    best_epoch = 0
    history = []
    rng = np.random.default_rng(int(seed))
    for epoch in range(1, int(epochs) + 1):
        losses = []
        order = rng.permutation(train_indices)
        for start in range(0, len(train_indices), int(object_batch_size)):
            index = order[start : start + int(object_batch_size)]
            loss, grads = loss_and_grad(
                candidate, features[index], particles[:, index], weights[:, index]
            )
            updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(candidate, eqx.is_inexact_array)
            )
            candidate = eqx.apply_updates(candidate, updates)
            losses.append(float(loss))
        train_nll = float(evaluate(candidate, train_indices))
        validation_nll = float(evaluate(candidate, validation_indices))
        history.append(
            {
                "epoch": epoch,
                "minibatch_train_weighted_nll": float(np.mean(losses)),
                "train_weighted_nll": train_nll,
                "validation_weighted_nll": validation_nll,
            }
        )
        print(
            "[proposal-expressivity] "
            f"candidate={progress_label} epoch={epoch}/{epochs} "
            f"train_nll={train_nll:.6g} validation_nll={validation_nll:.6g}",
            flush=True,
        )
        if np.isfinite(validation_nll) and validation_nll < best_validation:
            best = candidate
            best_validation = validation_nll
            best_epoch = epoch
    return ExpressivityFitResult(
        candidate=best,
        history=tuple(history),
        initial_train_nll=initial_train,
        final_train_nll=float(evaluate(best, train_indices)),
        initial_validation_nll=initial_validation,
        best_validation_nll=best_validation,
        best_epoch=best_epoch,
    )


def joint_distribution_metrics(
    target_x,
    target_weight,
    proposal_x,
    *,
    seed: int,
    max_draws: int = 256,
    n_projections: int = 64,
) -> dict[str, float]:
    """Compare dense joint draws after target-centered scale normalization."""
    target_x = np.asarray(target_x, dtype=np.float64)
    proposal_x = np.asarray(proposal_x, dtype=np.float64)
    weight = np.asarray(target_weight, dtype=np.float64)
    if target_x.ndim != 2 or proposal_x.ndim != 2:
        raise ValueError("target_x and proposal_x must be two-dimensional")
    if target_x.shape[1] != proposal_x.shape[1] or weight.shape != target_x.shape[:1]:
        raise ValueError("incompatible target/proposal shapes")
    weight = np.maximum(weight, 0.0)
    weight /= weight.sum()
    mean = np.sum(weight[:, None] * target_x, axis=0)
    scale = np.sqrt(np.sum(weight[:, None] * (target_x - mean) ** 2, axis=0))
    scale = np.maximum(scale, 1.0e-3)
    target_z = (target_x - mean) / scale
    proposal_z = (proposal_x - mean) / scale
    rng = np.random.default_rng(int(seed))
    n = min(int(max_draws), len(target_z), len(proposal_z))
    target_index = rng.choice(len(target_z), size=n, replace=True, p=weight)
    proposal_index = rng.choice(len(proposal_z), size=n, replace=False)
    target_draw = target_z[target_index]
    proposal_draw = proposal_z[proposal_index]

    projection = rng.normal(size=(int(n_projections), target_x.shape[1]))
    projection /= np.linalg.norm(projection, axis=1, keepdims=True)
    sliced = np.mean(
        [
            wasserstein_distance(
                target_z @ vector, proposal_z @ vector, u_weights=weight
            )
            for vector in projection
        ]
    )
    marginal = np.mean(
        [
            wasserstein_distance(target_z[:, dim], proposal_z[:, dim], u_weights=weight)
            for dim in range(target_x.shape[1])
        ]
    )
    covariance_target = _weighted_covariance(target_z, weight)
    covariance_proposal = np.cov(proposal_z, rowvar=False)
    covariance_error = np.linalg.norm(covariance_proposal - covariance_target) / max(
        np.linalg.norm(covariance_target), 1.0e-12
    )
    cross = cdist(target_draw, proposal_draw)
    within_target = cdist(target_draw[: n // 2], target_draw[n // 2 :])
    nearest_cross = float(np.median(np.min(cross, axis=1)))
    nearest_reference = float(np.median(np.min(within_target, axis=1)))
    energy = float(
        2.0 * np.mean(cross)
        - np.mean(cdist(target_draw, target_draw))
        - np.mean(cdist(proposal_draw, proposal_draw))
    )
    return {
        "marginal_wasserstein": float(marginal),
        "sliced_wasserstein": float(sliced),
        "energy_distance": max(0.0, energy),
        "covariance_relative_frobenius": float(covariance_error),
        "nearest_cover_distance": nearest_cross,
        "nearest_cover_ratio": nearest_cross / max(nearest_reference, 1.0e-12),
    }


def count_parameters(candidate) -> int:
    """Count trainable scalar arrays in an Equinox candidate."""
    leaves = jax.tree.leaves(eqx.filter(candidate, eqx.is_inexact_array))
    return int(sum(np.prod(value.shape) for value in leaves if value is not None))


def _weighted_covariance(values, weights):
    mean = np.sum(weights[:, None] * values, axis=0)
    centered = values - mean
    return (centered * weights[:, None]).T @ centered


def _validate_fit_inputs(
    features,
    particles,
    weights,
    train_indices,
    validation_indices,
    *,
    epochs,
    object_batch_size,
):
    if particles.ndim != 3 or weights.shape != particles.shape[:2]:
        raise ValueError("particles/weights must have shapes [K,N,D] and [K,N]")
    if features.ndim != 2 or features.shape[0] != particles.shape[1]:
        raise ValueError("features must have shape [N,F]")
    if int(epochs) <= 0 or int(object_batch_size) <= 0:
        raise ValueError("epochs and object_batch_size must be positive")
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("train and validation splits must be non-empty")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("train and validation splits overlap")
    all_indices = np.concatenate([train_indices, validation_indices])
    if np.min(all_indices) < 0 or np.max(all_indices) >= particles.shape[1]:
        raise ValueError("split index outside the object axis")
    if not bool(jnp.all(jnp.isfinite(particles))):
        raise ValueError("particles contain non-finite values")
    if not bool(jnp.all(jnp.isfinite(weights))) or bool(jnp.any(weights < 0.0)):
        raise ValueError("weights must be finite and non-negative")
    if not bool(jnp.all(jnp.sum(weights, axis=0) > 0.0)):
        raise ValueError("every object must have positive total weight")
