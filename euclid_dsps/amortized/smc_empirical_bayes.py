"""Direct empirical-Bayes updates from weighted joint SMC particles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .config import require_amortized_dependencies

eqx, optax = require_amortized_dependencies()


@dataclass(frozen=True)
class WeightedSMCBanks:
    """Two or more SMC replicates on one ordered object cohort."""

    row_indices: np.ndarray
    particles: np.ndarray
    weights: np.ndarray
    stored_logprior: np.ndarray
    roots: tuple[Path, ...]

    @property
    def n_banks(self) -> int:
        return int(self.particles.shape[0])

    @property
    def n_objects(self) -> int:
        return int(self.particles.shape[1])

    @property
    def particles_per_object(self) -> int:
        return int(self.particles.shape[2])


def load_weighted_smc_banks(
    paths: list[Path] | tuple[Path, ...], parameter_names: tuple[str, ...]
) -> WeightedSMCBanks:
    """Load completed SMC replicates without collapsing their seed identity."""
    roots = tuple(Path(path) for path in paths)
    if len(roots) < 2:
        raise ValueError("Direct SMC empirical Bayes requires at least two banks")
    expected_rows: np.ndarray | None = None
    expected_samples: int | None = None
    particles: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    logpriors: list[np.ndarray] = []
    x_columns = [f"latent_x_{name}" for name in parameter_names]
    required = {"row_index", "sample_id", "smc_weight", "logprior", *x_columns}

    for root in roots:
        files = _particle_files(root)
        if not (root / "DONE").is_file() or not files:
            raise FileNotFoundError(f"Incomplete weighted SMC bank: {root}")
        frame = pd.concat(
            [pd.read_parquet(path, columns=sorted(required)) for path in files],
            ignore_index=True,
        )
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"SMC bank missing columns {missing}: {root}")
        if frame.duplicated(["row_index", "sample_id"]).any():
            raise ValueError(f"Duplicate SMC particle identities in {root}")
        frame = frame.sort_values(["row_index", "sample_id"]).reset_index(drop=True)
        counts = frame.groupby("row_index", sort=True).size()
        if counts.empty or counts.nunique() != 1:
            raise ValueError(f"Unequal particles per object in {root}")
        rows = counts.index.to_numpy(dtype=np.int64)
        n_samples = int(counts.iloc[0])
        if expected_rows is None:
            expected_rows = rows
            expected_samples = n_samples
        elif not np.array_equal(expected_rows, rows):
            raise ValueError(f"SMC bank cohort mismatch in {root}")
        elif expected_samples != n_samples:
            raise ValueError(f"SMC bank particle-count mismatch in {root}")

        n_objects = len(rows)
        x = (
            frame[x_columns]
            .to_numpy(np.float32)
            .reshape(n_objects, n_samples, len(x_columns))
        )
        weight = (
            frame["smc_weight"]
            .to_numpy(np.float64, copy=True)
            .reshape(n_objects, n_samples)
        )
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(weight)):
            raise ValueError(f"Non-finite SMC particles or weights in {root}")
        if np.any(weight < 0.0) or np.any(np.sum(weight, axis=1) <= 0.0):
            raise ValueError(f"Invalid SMC weights in {root}")
        weight /= np.sum(weight, axis=1, keepdims=True)
        particles.append(x)
        weights.append(weight)
        logpriors.append(
            frame["logprior"].to_numpy(np.float64).reshape(n_objects, n_samples)
        )

    assert expected_rows is not None
    return WeightedSMCBanks(
        row_indices=expected_rows,
        particles=np.stack(particles),
        weights=np.stack(weights),
        stored_logprior=np.stack(logpriors),
        roots=roots,
    )


def split_object_positions(
    n_objects: int, *, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic train and validation object positions."""
    if n_objects < 2:
        raise ValueError("At least two objects are required")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(n_objects))
    n_validation = min(
        n_objects - 1, max(1, int(round(float(validation_fraction) * n_objects)))
    )
    return np.sort(order[n_validation:]), np.sort(order[:n_validation])


def pooled_particles_and_weights(
    banks: WeightedSMCBanks,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool equal-weight SMC replicates while retaining per-object normalization."""
    particles = banks.particles.transpose(1, 0, 2, 3).reshape(
        banks.n_objects,
        banks.n_banks * banks.particles_per_object,
        banks.particles.shape[-1],
    )
    weights = banks.weights.transpose(1, 0, 2).reshape(
        banks.n_objects, banks.n_banks * banks.particles_per_object
    )
    weights /= float(banks.n_banks)
    return particles, weights


@eqx.filter_jit
def _loss_and_grad(prior, x, weights, trust_x, trust_strength):
    def loss_fn(candidate):
        weighted_nll = -jnp.mean(jnp.sum(weights * candidate.log_prob(x), axis=1))
        source_to_candidate_cross_entropy = -jnp.mean(candidate.log_prob(trust_x))
        return weighted_nll + trust_strength * source_to_candidate_cross_entropy

    return eqx.filter_value_and_grad(loss_fn)(prior)


def fit_smc_weighted_prior(
    source_prior,
    particles: np.ndarray,
    weights: np.ndarray,
    train_positions: np.ndarray,
    *,
    epochs: int,
    object_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    trust_strength: float,
    trust_samples: int,
    seed: int,
) -> tuple[Any, list[dict[str, float]]]:
    """Run one stopped-weight M-step while keeping the source prior immutable."""
    if int(epochs) <= 0 or int(object_batch_size) <= 0 or int(trust_samples) <= 0:
        raise ValueError(
            "epochs, object_batch_size, and trust_samples must be positive"
        )
    if particles.shape[:2] != weights.shape:
        raise ValueError("particles and weights have incompatible shapes")
    if not np.allclose(np.sum(weights, axis=1), 1.0, atol=1.0e-6):
        raise ValueError("SMC weights must be normalized per object")

    prior = source_prior
    optimizer = optax.adamw(
        learning_rate=float(learning_rate), weight_decay=float(weight_decay)
    )
    state = optimizer.init(eqx.filter(prior, eqx.is_inexact_array))
    key = jax.random.PRNGKey(int(seed))
    history: list[dict[str, float]] = []
    for epoch in range(1, int(epochs) + 1):
        rng = np.random.default_rng(int(seed) + epoch)
        order = rng.permutation(np.asarray(train_positions, dtype=np.int64))
        losses: list[float] = []
        for start in range(0, len(order), int(object_batch_size)):
            positions = order[start : start + int(object_batch_size)]
            key, trust_key = jax.random.split(key)
            trust_x = source_prior.sample(trust_key, (int(trust_samples),))
            loss, grads = _loss_and_grad(
                prior,
                jnp.asarray(particles[positions]),
                jnp.asarray(weights[positions]),
                jnp.asarray(trust_x),
                float(trust_strength),
            )
            loss_value = float(jax.device_get(loss))
            if not np.isfinite(loss_value):
                raise FloatingPointError("Non-finite direct-SMC M-step loss")
            updates, state = optimizer.update(
                grads, state, eqx.filter(prior, eqx.is_inexact_array)
            )
            prior = eqx.apply_updates(prior, updates)
            losses.append(loss_value)
        history.append(
            {
                "epoch": float(epoch),
                "mean_train_loss": float(np.mean(losses)),
                "min_train_loss": float(np.min(losses)),
                "max_train_loss": float(np.max(losses)),
            }
        )
    return prior, history


def evaluate_prior(prior, particles: np.ndarray, *, batch_size: int = 262_144):
    """Evaluate an exact prior density on a bank without device over-allocation."""
    flat = particles.reshape(-1, particles.shape[-1])
    evaluate = eqx.filter_jit(prior.log_prob)
    result = []
    for start in range(0, len(flat), int(batch_size)):
        value = evaluate(jnp.asarray(flat[start : start + int(batch_size)]))
        result.append(np.asarray(jax.device_get(value), dtype=np.float64))
    return np.concatenate(result).reshape(particles.shape[:-1])


def prior_ratio_diagnostics(
    source_logprior: np.ndarray,
    candidate_logprior: np.ndarray,
    weights: np.ndarray,
    *,
    row_indices: np.ndarray,
    bank_names: tuple[str, ...],
) -> pd.DataFrame:
    """Estimate held-out evidence changes and ratio ESS from source SMC draws."""
    if source_logprior.shape != candidate_logprior.shape or weights.shape != (
        source_logprior.shape
    ):
        raise ValueError("prior log densities and SMC weights must share one shape")
    if source_logprior.ndim != 3:
        raise ValueError("expected arrays shaped (bank, object, particle)")
    if len(bank_names) != source_logprior.shape[0]:
        raise ValueError("bank_names does not match the number of banks")
    rows: list[dict[str, Any]] = []
    for bank_index, bank_name in enumerate(bank_names):
        for object_index, row_index in enumerate(row_indices):
            log_ratio = (
                candidate_logprior[bank_index, object_index]
                - source_logprior[bank_index, object_index]
            )
            log_weight = np.log(weights[bank_index, object_index]) + log_ratio
            maximum = float(np.max(log_weight))
            shifted = np.exp(log_weight - maximum)
            total = float(np.sum(shifted))
            normalized = shifted / total
            rows.append(
                {
                    "bank": str(bank_name),
                    "bank_index": int(bank_index),
                    "object_position": int(object_index),
                    "row_index": int(row_index),
                    "log_evidence_delta": maximum + np.log(total),
                    "prior_ratio_ess_fraction": float(
                        1.0 / np.sum(normalized**2) / len(normalized)
                    ),
                    "max_reweighted_particle_weight": float(np.max(normalized)),
                    "median_log_prior_ratio": float(np.median(log_ratio)),
                    "p90_abs_log_prior_ratio": float(
                        np.quantile(np.abs(log_ratio), 0.9)
                    ),
                }
            )
    return pd.DataFrame(rows)


def direct_smc_validation_gate(
    diagnostics: pd.DataFrame,
    validation_positions: np.ndarray,
    *,
    min_mean_log_evidence_delta: float,
    min_median_ratio_ess_fraction: float,
    min_fraction_ratio_ess_ge_0p2: float,
    max_seed_mean_logevidence_delta_difference: float,
) -> dict[str, Any]:
    """Apply a fail-closed gate to held-out prior-ratio diagnostics."""
    validation = diagnostics[
        diagnostics["object_position"].isin(np.asarray(validation_positions))
    ].copy()
    if validation.empty:
        raise ValueError("No validation diagnostics selected")
    grouped = validation.groupby("bank", sort=False)["log_evidence_delta"].mean()
    finite = bool(
        np.all(np.isfinite(validation["log_evidence_delta"]))
        and np.all(np.isfinite(validation["prior_ratio_ess_fraction"]))
    )
    seed_means = {str(key): float(value) for key, value in grouped.items()}
    seed_spread = float(grouped.max() - grouped.min())
    median_ess = float(np.median(validation["prior_ratio_ess_fraction"]))
    fraction_ess = float(np.mean(validation["prior_ratio_ess_fraction"] >= 0.2))
    checks = {
        "all_validation_diagnostics_finite": finite,
        "every_seed_mean_logevidence_delta_positive": bool(
            np.all(grouped.to_numpy() > float(min_mean_log_evidence_delta))
        ),
        "seed_mean_logevidence_delta_stable": bool(
            seed_spread <= float(max_seed_mean_logevidence_delta_difference)
        ),
        "median_prior_ratio_ess_fraction_adequate": bool(
            median_ess >= float(min_median_ratio_ess_fraction)
        ),
        "fraction_objects_prior_ratio_ess_ge_0p2_adequate": bool(
            fraction_ess >= float(min_fraction_ratio_ess_ge_0p2)
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "n_validation_objects": int(validation["object_position"].nunique()),
        "n_seed_object_evaluations": int(len(validation)),
        "seed_mean_log_evidence_delta": seed_means,
        "seed_mean_log_evidence_delta_spread": seed_spread,
        "mean_log_evidence_delta": float(validation["log_evidence_delta"].mean()),
        "median_log_evidence_delta": float(validation["log_evidence_delta"].median()),
        "median_prior_ratio_ess_fraction": median_ess,
        "fraction_objects_prior_ratio_ess_ge_0p2": fraction_ess,
        "thresholds": {
            "min_mean_log_evidence_delta_per_seed": float(min_mean_log_evidence_delta),
            "min_median_prior_ratio_ess_fraction": float(min_median_ratio_ess_fraction),
            "min_fraction_objects_prior_ratio_ess_ge_0p2": float(
                min_fraction_ratio_ess_ge_0p2
            ),
            "max_seed_mean_logevidence_delta_difference": float(
                max_seed_mean_logevidence_delta_difference
            ),
        },
    }


def _particle_files(path: Path) -> list[Path]:
    direct = sorted((path / "weighted_particles").glob("batch_*.parquet"))
    if direct:
        return direct
    return sorted(path.glob("shard_*/weighted_particles/batch_*.parquet"))
