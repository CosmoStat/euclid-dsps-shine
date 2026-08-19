from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HAS_DEPS = (
    importlib.util.find_spec("equinox") is not None
    and importlib.util.find_spec("optax") is not None
)
pytestmark = pytest.mark.skipif(
    not HAS_DEPS,
    reason="Equinox/Optax optional dependencies are not installed",
)

if HAS_DEPS:
    import jax

    from euclid_dsps.amortized.flows import RealNVPPrior
    from euclid_dsps.amortized.smc_empirical_bayes import (
        direct_smc_validation_gate,
        evaluate_prior,
        fit_smc_weighted_prior,
        load_weighted_smc_banks,
        pooled_particles_and_weights,
        prior_ratio_diagnostics,
        split_object_positions,
        validate_smc_checkpoint_provenance,
    )


def _write_bank(root: Path, *, offset: float = 0.0, rows=(11, 17)) -> None:
    target = root / "weighted_particles"
    target.mkdir(parents=True)
    records = []
    for row_index in rows:
        for sample_id in range(3):
            records.append(
                {
                    "row_index": row_index,
                    "sample_id": sample_id,
                    "smc_weight": [0.2, 0.3, 0.5][sample_id],
                    "logprior": -1.0 - sample_id,
                    "latent_x_a": offset + row_index / 100.0 + sample_id,
                    "latent_x_b": offset - row_index / 100.0 - sample_id,
                }
            )
    pd.DataFrame(records).to_parquet(target / "batch_000000.parquet", index=False)
    (root / "DONE").touch()


def test_load_and_pool_smc_banks_preserves_replicates(tmp_path: Path) -> None:
    first = tmp_path / "seed_1"
    second = tmp_path / "seed_2"
    _write_bank(first)
    _write_bank(second, offset=0.5)

    banks = load_weighted_smc_banks([first, second], ("a", "b"))
    particles, weights = pooled_particles_and_weights(banks)

    assert banks.particles.shape == (2, 2, 3, 2)
    assert particles.shape == (2, 6, 2)
    assert weights.shape == (2, 6)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.allclose(weights[0], [0.1, 0.15, 0.25] * 2)


def test_load_smc_banks_rejects_cohort_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "seed_1"
    second = tmp_path / "seed_2"
    _write_bank(first)
    _write_bank(second, rows=(11, 19))

    with pytest.raises(ValueError, match="cohort mismatch"):
        load_weighted_smc_banks([first, second], ("a", "b"))


def test_prior_ratio_gate_requires_both_seed_evidence_gains() -> None:
    source = np.zeros((2, 4, 8), dtype=np.float64)
    candidate = source + 0.1
    weights = np.full((2, 4, 8), 1.0 / 8.0)
    diagnostic = prior_ratio_diagnostics(
        source,
        candidate,
        weights,
        row_indices=np.arange(4),
        bank_names=("seed_a", "seed_b"),
    )
    gate = direct_smc_validation_gate(
        diagnostic,
        np.arange(4),
        min_mean_log_evidence_delta=0.0,
        min_median_ratio_ess_fraction=0.5,
        min_fraction_ratio_ess_ge_0p2=0.9,
        max_seed_mean_logevidence_delta_difference=0.25,
    )

    assert gate["status"] == "PASS"
    assert gate["median_prior_ratio_ess_fraction"] == pytest.approx(1.0)
    assert gate["mean_log_evidence_delta"] == pytest.approx(0.1)

    candidate[1] = source[1] - 0.1
    diagnostic = prior_ratio_diagnostics(
        source,
        candidate,
        weights,
        row_indices=np.arange(4),
        bank_names=("seed_a", "seed_b"),
    )
    gate = direct_smc_validation_gate(
        diagnostic,
        np.arange(4),
        min_mean_log_evidence_delta=0.0,
        min_median_ratio_ess_fraction=0.5,
        min_fraction_ratio_ess_ge_0p2=0.9,
        max_seed_mean_logevidence_delta_difference=0.25,
    )
    assert gate["status"] == "FAIL"
    assert not gate["checks"]["every_seed_mean_logevidence_delta_positive"]


def test_direct_smc_mstep_improves_weighted_prior_fit() -> None:
    prior = RealNVPPrior(
        jax.random.PRNGKey(1),
        latent_dim=2,
        n_layers=2,
        hidden_size=8,
        init="identity",
        init_scale=0.0,
    )
    particles = np.asarray(
        [
            [[1.8, 1.9], [2.0, 2.1], [2.2, 2.0]],
            [[1.7, 2.2], [2.1, 1.8], [1.9, 2.0]],
        ],
        dtype=np.float32,
    )
    weights = np.full((2, 3), 1.0 / 3.0, dtype=np.float32)
    before = evaluate_prior(prior, particles).mean()
    candidate, history = fit_smc_weighted_prior(
        prior,
        particles,
        weights,
        np.arange(2),
        epochs=10,
        object_batch_size=2,
        learning_rate=1.0e-2,
        weight_decay=0.0,
        trust_strength=0.0,
        trust_samples=4,
        seed=3,
    )
    after = evaluate_prior(candidate, particles).mean()

    assert len(history) == 10
    assert after > before


def test_object_split_is_reproducible_and_disjoint() -> None:
    train_a, validation_a = split_object_positions(16, validation_fraction=0.25, seed=7)
    train_b, validation_b = split_object_positions(16, validation_fraction=0.25, seed=7)

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(validation_a, validation_b)
    assert len(validation_a) == 4
    assert np.intersect1d(train_a, validation_a).size == 0


def test_smc_layout_prior_evaluation_matches_flat_evaluation() -> None:
    prior = RealNVPPrior(
        jax.random.PRNGKey(5),
        latent_dim=2,
        n_layers=2,
        hidden_size=8,
    )
    particles = np.random.default_rng(4).normal(size=(2, 4, 6, 2)).astype(np.float32)

    flat = evaluate_prior(prior, particles)
    smc_layout = evaluate_prior(prior, particles, smc_object_batch_size=2)

    assert flat.shape == smc_layout.shape == (2, 4, 6)
    assert np.allclose(flat, smc_layout, atol=1.0e-5)


def test_smc_checkpoint_provenance_requires_exact_hash() -> None:
    summaries = [
        {"inputs": {"checkpoint": {"sha256": "abc"}}},
        {"inputs": {"checkpoint": {"sha256": "abc"}}},
    ]

    assert validate_smc_checkpoint_provenance(summaries, checkpoint_sha256="abc") == (
        "abc",
        "abc",
    )
    with pytest.raises(ValueError, match="requested source checkpoint"):
        validate_smc_checkpoint_provenance(summaries, checkpoint_sha256="def")
