from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

HAS_EQUINOX = importlib.util.find_spec("equinox") is not None
pytestmark = pytest.mark.skipif(not HAS_EQUINOX, reason="equinox is not installed")

if HAS_EQUINOX:
    from euclid_dsps.amortized.flows import StandardNormalPrior
    from euclid_dsps.amortized.population_vi import (
        frozen_proposal_population_objective,
        require_population_vi_gate,
    )
    from scripts.train_feniks_sc_drws_population_marginal_vi import (
        _validate_bank_provenance,
    )


def test_gaussian_importance_evidence_matches_analytic_value() -> None:
    grid = jnp.linspace(-8.0, 8.0, 20001)
    particles = grid[:, None, None]
    logproposal = jnp.full((len(grid), 1), jnp.log(1.0 / 16.0))
    loglike = -0.5 * ((grid[:, None] - 1.0) / 0.7) ** 2 - jnp.log(
        0.7 * jnp.sqrt(2.0 * jnp.pi)
    )
    terms = frozen_proposal_population_objective(
        StandardNormalPrior(latent_dim=1),
        particles,
        loglike,
        logproposal,
        jnp.asarray(0.0),
    )
    analytic = -0.5 * (jnp.log(2.0 * jnp.pi * (1.0 + 0.7**2)) + 1.0**2 / (1.0 + 0.7**2))
    assert np.isclose(float(terms.mean_log_evidence), float(analytic), atol=3e-3)


def test_bimodal_likelihood_evidence_matches_low_dimensional_quadrature() -> None:
    grid = jnp.linspace(-10.0, 10.0, 40001)
    dx = float(grid[1] - grid[0])
    particles = grid[:, None, None]
    logproposal = jnp.full((len(grid), 1), jnp.log(1.0 / 20.0))
    component_scale = 0.35
    component_loglike = jnp.stack(
        [
            -0.5 * ((grid - center) / component_scale) ** 2
            - jnp.log(component_scale * jnp.sqrt(2.0 * jnp.pi))
            for center in (-2.5, 2.5)
        ],
        axis=0,
    )
    loglike = (jax.scipy.special.logsumexp(component_loglike, axis=0) - jnp.log(2.0))[
        :, None
    ]
    terms = frozen_proposal_population_objective(
        StandardNormalPrior(latent_dim=1),
        particles,
        loglike,
        logproposal,
        jnp.asarray(0.0),
    )
    prior_density = jnp.exp(StandardNormalPrior(latent_dim=1).log_prob(particles))
    quadrature = jnp.sum(jnp.exp(loglike[:, 0]) * prior_density[:, 0]) * dx

    assert np.isclose(
        float(terms.mean_log_evidence),
        float(jnp.log(quadrature)),
        atol=3e-3,
    )


def test_selection_normalizer_enters_with_negative_log_alpha() -> None:
    particles = jnp.asarray([[[-1.0]], [[0.0]], [[1.0]], [[2.0]]])
    loglike = jnp.zeros((4, 1))
    logproposal = StandardNormalPrior(latent_dim=1).log_prob(particles)
    no_selection = frozen_proposal_population_objective(
        StandardNormalPrior(latent_dim=1),
        particles,
        loglike,
        logproposal,
        jnp.log(1.0),
    )
    half_selected = frozen_proposal_population_objective(
        StandardNormalPrior(latent_dim=1),
        particles,
        loglike,
        logproposal,
        jnp.log(0.5),
    )
    assert np.isclose(
        float(half_selected.mean_objective - no_selection.mean_objective),
        -np.log(0.5),
        atol=1e-6,
    )


def test_population_gate_requires_all_truth_free_prerequisites() -> None:
    receipt = {
        "truth_used": False,
        "technical_gate": {"status": "PASS"},
        "held_out_band": {"status": "PASS"},
        "model_generated_calibration": {"status": "PASS"},
    }
    assert require_population_vi_gate(receipt)["status"] == "PASS"
    receipt["held_out_band"]["status"] = "FAIL"
    with pytest.raises(ValueError, match="held_out_band_predictive"):
        require_population_vi_gate(receipt)


def test_population_bank_provenance_is_bound_to_exact_posterior(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "winner.eqx"
    checkpoint.write_bytes(b"frozen-posterior")
    import hashlib

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "inference_summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "truth_used_for_inference_or_checkpoint_selection": False,
                "checkpoint": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    assert _validate_bank_provenance(bank, digest)["posterior_checkpoint_sha256"] == digest
    with pytest.raises(ValueError, match="different posterior"):
        _validate_bank_provenance(bank, "0" * 64)


def test_population_bank_provenance_rejects_truth_and_bare_table(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "winner.eqx"
    checkpoint.write_bytes(b"frozen-posterior")
    import hashlib

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    bank = tmp_path / "bank"
    bank.mkdir()
    summary = {
        "complete": True,
        "truth_used_for_inference_or_checkpoint_selection": True,
        "checkpoint": str(checkpoint),
    }
    (bank / "inference_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not truth-free"):
        _validate_bank_provenance(bank, digest)
    bare = tmp_path / "posterior_samples.parquet"
    bare.touch()
    with pytest.raises(ValueError, match="not a bare posterior table"):
        _validate_bank_provenance(bare, digest)
