"""Gated population VI using frozen direct-proposal particles."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .config import require_amortized_dependencies

eqx, _optax = require_amortized_dependencies()


class PopulationMarginalTerms(NamedTuple):
    loss: jnp.ndarray
    mean_log_evidence: jnp.ndarray
    mean_log_alpha: jnp.ndarray
    mean_objective: jnp.ndarray
    object_ess_fraction: jnp.ndarray
    object_maximum_weight: jnp.ndarray
    object_finite: jnp.ndarray


def frozen_proposal_population_objective(
    candidate_prior: Any,
    particles: jnp.ndarray,
    loglike: jnp.ndarray,
    logproposal: jnp.ndarray,
    log_alpha: jnp.ndarray,
) -> PopulationMarginalTerms:
    """Estimate selected-catalogue marginal likelihood from a frozen proposal.

    Inputs use shape ``[K, N, ...]``.  ``logproposal`` must be the complete
    density of the direct proposal that generated every particle, including
    every component of a defensive mixture.  The particles and proposal
    denominator are stopped so a prior M-step cannot adapt its own samples.
    """
    particles = jax.lax.stop_gradient(jnp.asarray(particles))
    loglike = jax.lax.stop_gradient(jnp.asarray(loglike))
    logproposal = jax.lax.stop_gradient(jnp.asarray(logproposal))
    if particles.ndim != 3:
        raise ValueError("population particles must have shape [K, N, D]")
    if loglike.shape != particles.shape[:2] or logproposal.shape != loglike.shape:
        raise ValueError("loglike/logproposal must match particle [K, N] axes")
    if particles.shape[0] < 2:
        raise ValueError("population marginal objective requires K >= 2")
    candidate_logprior = candidate_prior.log_prob(particles)
    logweight = loglike + candidate_logprior - logproposal
    finite = jnp.isfinite(logweight)
    safe = jnp.where(finite, logweight, -jnp.inf)
    finite_count = jnp.sum(finite, axis=0)
    log_evidence = jax.scipy.special.logsumexp(safe, axis=0) - jnp.log(
        jnp.asarray(particles.shape[0], dtype=particles.dtype)
    )
    normalized = jax.nn.softmax(safe, axis=0)
    ess = 1.0 / jnp.sum(normalized**2, axis=0)
    object_finite = (finite_count > 0) & jnp.isfinite(log_evidence)
    alpha = jnp.broadcast_to(jnp.asarray(log_alpha), log_evidence.shape)
    object_objective = log_evidence - alpha
    valid_count = jnp.maximum(jnp.sum(object_finite), 1)
    mean_objective = (
        jnp.sum(jnp.where(object_finite, object_objective, 0.0)) / valid_count
    )
    return PopulationMarginalTerms(
        loss=-mean_objective,
        mean_log_evidence=jnp.sum(jnp.where(object_finite, log_evidence, 0.0))
        / valid_count,
        mean_log_alpha=jnp.sum(jnp.where(object_finite, alpha, 0.0)) / valid_count,
        mean_objective=mean_objective,
        object_ess_fraction=ess / float(particles.shape[0]),
        object_maximum_weight=jnp.max(normalized, axis=0),
        object_finite=object_finite,
    )


def require_population_vi_gate(
    validation_receipt: dict[str, Any],
    *,
    require_held_out_band: bool = True,
    require_simulated_calibration: bool = True,
) -> dict[str, Any]:
    """Return a durable gate payload or fail before any prior update."""
    technical = dict(validation_receipt.get("technical_gate", {}) or {})
    checks = {
        "support_gate": technical.get("status") == "PASS",
        "truth_free_validation": validation_receipt.get("truth_used") is False,
        "held_out_band_predictive": (
            not require_held_out_band
            or validation_receipt.get("held_out_band", {}).get("status") == "PASS"
        ),
        "model_generated_calibration": (
            not require_simulated_calibration
            or validation_receipt.get("model_generated_calibration", {}).get("status")
            == "PASS"
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checks": checks,
        "truth_used": False,
        "scientific_promotion": False,
        "contract": (
            "all representative objects remain in the population objective; "
            "no good-ESS object filtering"
        ),
    }
    if payload["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"population VI gate is blocked: {failed}")
    return payload


def population_prior_value_and_grad(
    prior: Any,
    particles: jnp.ndarray,
    loglike: jnp.ndarray,
    logproposal: jnp.ndarray,
    log_alpha_fn,
):
    """Differentiate one prior M-step with fixed particles and denominator."""

    def objective(candidate):
        return frozen_proposal_population_objective(
            candidate,
            particles,
            loglike,
            logproposal,
            log_alpha_fn(candidate),
        ).loss

    return eqx.filter_value_and_grad(objective)(prior)
