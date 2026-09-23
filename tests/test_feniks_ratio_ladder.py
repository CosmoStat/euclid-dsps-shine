from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.amortized.forward_population import (
    PHYSICAL,
    PhysicalBasis,
    fit_selected_weights,
    parent_from_selected,
)
from scripts.feniks_ratio_ladder import (
    _localize_failure,
    _stratified_split,
    conditional_normal_score,
    exact_selected_log_classifier,
)


def _names():
    return tuple(PHYSICAL) + tuple(f"sfh_{i}" for i in range(10))


def test_conditional_normal_score_recovers_standard_normal():
    rng = np.random.default_rng(81)
    z = rng.normal(size=300_000)
    threshold = rng.uniform(-1.5, 1.5, size=len(z))
    selected = z > threshold

    score = conditional_normal_score(z, threshold, selected)

    assert abs(score.mean()) < 0.01
    assert abs(score.std() - 1.0) < 0.01
    assert np.quantile(abs(score), 0.99) < 2.7


def test_exact_classifier_encodes_selected_component_density_ratios():
    rng = np.random.default_rng(93)
    basis = PhysicalBasis.create(_names(), components=8, seed=12)
    x, _ = basis.sample(rng, 32)
    alpha = np.linspace(0.15, 0.9, basis.components)
    frequencies = np.linspace(1, basis.components, basis.components, dtype=float)
    frequencies /= frequencies.sum()

    logc = exact_selected_log_classifier(x, basis, alpha, frequencies)
    expected = basis.component_log_prob(x) - np.log(alpha)
    recovered = logc - np.log(frequencies)

    # A classifier only identifies a common row normalization. Component
    # contrasts must equal the analytic selected-density contrasts exactly.
    np.testing.assert_allclose(
        recovered - recovered[:, :1], expected - expected[:, :1], atol=1e-11
    )
    np.testing.assert_allclose(np.exp(logc).sum(axis=1), 1.0, atol=1e-12)


def test_exact_ratio_recovers_parent_under_latent_dependent_selection():
    rng = np.random.default_rng(104)
    centers = np.array(
        [
            [-1.5, -1.0, 0.0, 0.0, 0.0],
            [1.5, -1.0, 0.0, 0.0, 0.0],
            [-1.5, 1.0, 0.0, 0.0, 0.0],
            [1.5, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    basis = PhysicalBasis(_names(), centers, np.full_like(centers, 0.35))
    u_true = np.array([0.12, 0.28, 0.25, 0.35])

    labels = np.arange(80_000) % basis.components
    rng.shuffle(labels)
    reference_x, _ = basis.sample(rng, len(labels), labels=labels)
    beta = 0.1 + 0.8 / (1 + np.exp(-(0.8 * reference_x[:, 0] - reference_x[:, 1])))
    selected = rng.random(len(labels)) < beta
    attempts = np.bincount(labels, minlength=basis.components)
    successes = np.bincount(labels[selected], minlength=basis.components)
    alpha = successes / attempts
    frequencies = successes / successes.sum()

    target_x, _ = basis.sample(rng, 80_000, weights=u_true)
    target_beta = 0.1 + 0.8 / (1 + np.exp(-(0.8 * target_x[:, 0] - target_x[:, 1])))
    observed = target_x[rng.random(len(target_x)) < target_beta]
    logc = exact_selected_log_classifier(observed, basis, alpha, frequencies)

    selected_weights, _ = fit_selected_weights(logc, frequencies)
    recovered = parent_from_selected(selected_weights, alpha)

    np.testing.assert_allclose(recovered, u_true, atol=0.015)


def test_stratified_split_is_disjoint_and_uses_only_selected_rows():
    labels = np.repeat(np.arange(5), 100)
    selected = np.tile(np.arange(100) % 3 != 0, 5)

    train, validation, test = _stratified_split(labels, selected, seed=17)

    assert all(len(part) > 0 for part in (train, validation, test))
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert selected[np.concatenate((train, validation, test))].all()
    assert set(np.concatenate((train, validation, test))) == set(
        np.flatnonzero(selected)
    )


def test_submission_chain_and_four_ratio_arms_are_explicit():
    repository = Path(__file__).parents[1]
    launcher = (repository / "scripts/submit_feniks_ratio_ladder.sh").read_text()
    watcher = (repository / "scripts/watch_feniks_ratio_ladder.sh").read_text()

    assert '--array="0-${REFERENCE_SHARDS}%${BANK_CONCURRENCY}"' in launcher
    assert '--dependency="afterok:$BANK_JOB"' in launcher
    assert '--array="0-3%$RATIO_CONCURRENCY"' in launcher
    assert '--dependency="afterok:$OBSERVATION_JOB:$RATIO_JOB"' in launcher
    for arm in (
        "exact_theta",
        "physical_a",
        "noiseless_photometry",
        "noisy_photometry",
    ):
        assert arm in watcher


def test_failure_localization_returns_first_broken_rung():
    summary = pd.DataFrame(
        {
            "arm": (
                "exact_theta",
                "physical_a",
                "noiseless_photometry",
                "noisy_photometry",
            ),
            "parent_physical_sliced_wasserstein": (0.03, 0.04, 0.10, 0.11),
            "selected_physical_sliced_wasserstein": (0.03, 0.04, 0.05, 0.06),
        }
    )
    contracts = {
        "exact_parent_physical_sw": 0.06,
        "exact_selected_physical_sw": 0.06,
        "maximum_stage_sw_increase": 0.04,
    }

    decisions, increments, conclusion = _localize_failure(summary, contracts)

    assert decisions["exact_ratio_and_selection"]
    assert decisions["physical_classifier"]
    assert not decisions["photometric_projection"]
    assert np.isclose(increments["photometric_projection_parent"], 0.06)
    assert conclusion == "decoder_or_photometric_information_loss"
