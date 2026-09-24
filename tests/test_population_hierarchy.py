import numpy as np
from scipy.special import logsumexp

from euclid_dsps.amortized.forward_population import (
    parent_from_selected,
    selected_from_parent,
)
from euclid_dsps.amortized.population_hierarchy import (
    aggregate_log_probabilities,
    aggregate_selection_efficiency,
    aggregate_vector,
    build_joint_hierarchy,
    expand_group_parent,
)


def test_grouping_preserves_probability_and_density_ratio_identity():
    probability = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    frequency = np.array([0.2, 0.1, 0.3, 0.4])
    groups = np.array([0, 0, 1, 1])

    grouped_log = aggregate_log_probabilities(np.log(probability), groups)
    grouped_frequency = aggregate_vector(frequency, groups)

    assert np.allclose(np.exp(grouped_log).sum(axis=1), 1)
    for group in range(2):
        member = groups == group
        expected = probability[:, member].sum(axis=1) / frequency[member].sum()
        assert np.allclose(
            np.exp(grouped_log[:, group]) / grouped_frequency[group], expected
        )


def test_group_selection_correction_recovers_reference_parent():
    reference = np.array([0.1, 0.2, 0.3, 0.4])
    alpha = np.array([0.2, 0.4, 0.5, 0.8])
    groups = np.array([0, 0, 1, 1])
    grouped_reference, grouped_alpha = aggregate_selection_efficiency(
        reference, alpha, groups
    )
    selected = selected_from_parent(grouped_reference, grouped_alpha)

    assert np.allclose(parent_from_selected(selected, grouped_alpha), grouped_reference)
    assert np.allclose(
        expand_group_parent(grouped_reference, reference, groups), reference
    )


def test_joint_hierarchy_is_nested_and_preserves_tail_component():
    rng = np.random.default_rng(8)
    components = 12
    centers = rng.normal(size=(components, 5))
    scales = np.full_like(centers, 0.7)
    scales[0] = 2.5
    logits = rng.normal(size=(480, components))
    probability = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    confusion = probability.reshape(components, -1, components).mean(axis=1)

    hierarchy, embedding = build_joint_hierarchy(
        centers,
        scales,
        [confusion, confusion],
        [4, 8, 12],
        response_rank=4,
    )

    assert embedding.shape[0] == components
    assert all(groups[0] == 0 for groups in hierarchy.values())
    assert np.array_equal(hierarchy[12], np.arange(components))
    for fine, coarse in ((8, 4), (12, 8)):
        for fine_group in np.unique(hierarchy[fine]):
            members = hierarchy[fine] == fine_group
            assert len(np.unique(hierarchy[coarse][members])) == 1
