import numpy as np
import pytest
from scipy.special import logsumexp

from euclid_dsps.amortized.conditional_reference import (
    lift_parent,
    reserved_row_weights,
    split_sfh_basis,
    tied_selected_ratios,
)
from euclid_dsps.amortized.forward_population import parent_from_selected
from euclid_dsps.amortized.native_reference import log_prob, make_basis, sample_basis
from euclid_dsps.amortized.population_regularization import fit_selected_weights_kl


def test_nested_density_and_sampling_retain_full_joint_15d():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(200, 15))
    x[:, 5] += 2 * x[:, 0]
    old = make_basis(x, 4, 0.2, 1, 0.02, 2)
    new, gates, info = split_sfh_basis(old)
    z = np.array(info["z"])
    assert gates.shape == (200, 4, 2)
    np.testing.assert_allclose(new["conditional"].sum(axis=0), 1, atol=1e-13)
    np.testing.assert_allclose(z.sum(axis=1), 1, atol=1e-13)
    assert gates.min() >= 0.05 and gates.max() <= 0.95
    weights = np.array([0.1, 0.4, 0.3, 0.2])
    lifted = lift_parent(weights, z)
    np.testing.assert_allclose(
        log_prob(old, x[:20], weights), log_prob(new, x[:20], lifted), atol=1e-12
    )
    draws = sample_basis(new, rng.choice(8, 4000, p=lifted), 15)
    assert draws.shape == (4000, 15) and draws.std(axis=0).min() > 0.4
    assert np.corrcoef(draws[:, 0], draws[:, 5])[0, 1] > 0.6
    assert not info["target_truth_used"] and not info["kernels_changed"]


def test_tied_selected_likelihood_uses_selection_and_reference_frequencies():
    rng = np.random.default_rng(34)
    z = np.array([[0.2, 0.8], [0.7, 0.3]])
    alpha = np.array([0.1, 0.9, 0.2, 0.6])
    c = np.array([0.15, 0.3, 0.4, 0.15])
    logs = rng.normal(size=(51, 4))
    logs -= logsumexp(logs, axis=1, keepdims=True)
    grouped, gc, ga = tied_selected_ratios(logs, c, alpha, z)
    u = np.array([0.3, 0.7])
    lifted = lift_parent(u, z)
    v = lifted * alpha / (lifted @ alpha)
    vg = u * ga / (u @ ga)
    np.testing.assert_allclose(
        logsumexp(logs - np.log(c) + np.log(v), axis=1),
        logsumexp(grouped - np.log(gc) + np.log(vg), axis=1),
        atol=1e-12,
    )
    # Directly summing C is not the same likelihood under unequal c and alpha.
    wrong = np.log(np.exp(logs).reshape(-1, 2, 2).sum(axis=2))
    assert np.max(abs(grouped - wrong)) > 0.05


def test_known_split_mixture_recovery_is_parent_not_selected():
    alpha = np.array([0.1, 0.5, 0.8, 0.4])
    u = np.array([0.4, 0.2, 0.1, 0.3])
    v = u * alpha / (u @ alpha)
    labels = np.repeat(np.arange(4), np.rint(v * 17000).astype(int))
    logs = np.where(labels[:, None] == np.arange(4), 0, -700.0)
    selected, info = fit_selected_weights_kl(logs, np.full(4, 0.25), strength=1e-7)
    recovered = parent_from_selected(selected, alpha)
    np.testing.assert_allclose(recovered, u, atol=1e-4)
    assert info["kkt_gap"] <= 2e-6
    assert abs(recovered.sum() - 1) < 1e-12
    assert abs(selected.sum() - 1) < 1e-12
    np.testing.assert_allclose(
        parent_from_selected(selected, np.ones(4) * 0.4), selected
    )
    assert parent_from_selected([0.5, 0.5], [0.1, 0.9])[0] == pytest.approx(0.9)


def test_reserved_measure_accounts_for_nonuniform_parent_sampling():
    labels = np.repeat([0, 1], [100, 400])
    selected = np.r_[np.arange(100) < 20, np.arange(400) < 320]
    mask, weights = reserved_row_weights(labels, selected, np.full(500, 4), [0.5, 0.5])
    np.testing.assert_allclose(np.bincount(labels[mask], weights=weights), [0.2, 0.8])
    with pytest.raises(ValueError, match="Missing parent"):
        reserved_row_weights(labels, selected, np.zeros(500), [0.5, 0.5])


def test_invalid_or_constant_reference_is_not_silently_split():
    basis = make_basis(
        np.random.default_rng(5).normal(size=(64, 15)), 2, 0.2, 1, 0.02, 1
    )
    for floor in (0, 0.5, float("nan")):
        with pytest.raises(ValueError):
            split_sfh_basis(basis, floor)
    basis["anchors"][:, 5:] = 0
    with pytest.raises(ValueError, match="No SFH"):
        split_sfh_basis(basis)


def test_saturated_tied_efficiency_roundoff_is_a_valid_probability():
    z = np.array([[0.50872431, 0.49127569], [0.3, 0.7]])
    z[0] *= 1 + 1.6e-15
    assert z[0].sum() > 1
    logs = np.log(np.full((5, 4), 0.25))
    alpha = np.array([1.0, 1.0, 0.2, 0.8])
    tied, c, a = tied_selected_ratios(logs, [0.25] * 4, alpha, z)
    assert a[0] == 1.0 and np.all(a <= 1)
    u = parent_from_selected([0.5, 0.5], a)
    expanded = lift_parent(u, z)
    np.testing.assert_allclose(expanded.sum(), 1, atol=1e-15)
    v = expanded * alpha / (expanded @ alpha)
    np.testing.assert_allclose(
        logsumexp(tied - np.log(c) + np.log([0.5, 0.5]), axis=1),
        logsumexp(logs - np.log(0.25) + np.log(v), axis=1),
        atol=1e-12,
    )
    with pytest.raises(ValueError):
        tied_selected_ratios(logs, [0.25] * 4, [1.00001, 1, 0.2, 0.8], z)
    with pytest.raises(ValueError):
        tied_selected_ratios(logs, [0.25] * 4, alpha, z * 1.00001)
