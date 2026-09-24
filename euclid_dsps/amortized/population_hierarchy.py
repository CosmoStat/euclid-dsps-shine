"""Identifiable coarsenings of a fixed selected-population component basis."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.special import logsumexp

from euclid_dsps.amortized.forward_population import simplex


def _groups(groups, components):
    groups = np.asarray(groups, dtype=int)
    if groups.shape != (components,) or np.any(groups < 0):
        raise ValueError("one nonnegative group per component required")
    unique = np.unique(groups)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("group labels must be contiguous from zero")
    return groups, len(unique)


def aggregate_vector(values, groups):
    """Sum component masses over a complete partition."""
    values = np.asarray(values, dtype=np.float64)
    groups, count = _groups(groups, len(values))
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("finite component vector required")
    return np.bincount(groups, weights=values, minlength=count)


def aggregate_log_probabilities(log_probability, groups):
    """Marginalize calibrated component probabilities over groups."""
    log_probability = np.asarray(log_probability, dtype=np.float64)
    if log_probability.ndim != 2 or not np.isfinite(log_probability).all():
        raise ValueError("finite matrix of log probabilities required")
    groups, count = _groups(groups, log_probability.shape[1])
    result = np.column_stack(
        [
            logsumexp(log_probability[:, groups == group], axis=1)
            for group in range(count)
        ]
    )
    # A partition of normalized component probabilities remains normalized.
    return result - logsumexp(result, axis=1, keepdims=True)


def aggregate_selection_efficiency(reference_parent, alpha, groups):
    """Return parent mass and selection efficiency for each grouped component."""
    reference_parent = simplex(reference_parent)
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.shape != reference_parent.shape or np.any(
        ~np.isfinite(alpha) | (alpha <= 0) | (alpha > 1)
    ):
        raise ValueError("valid component selection efficiencies required")
    parent = aggregate_vector(reference_parent, groups)
    selected_mass = aggregate_vector(reference_parent * alpha, groups)
    grouped_alpha = selected_mass / parent
    return simplex(parent), grouped_alpha


def expand_group_parent(group_parent, reference_parent, groups):
    """Expand group masses using the fixed reference conditional within groups."""
    reference_parent = simplex(reference_parent)
    groups, count = _groups(groups, len(reference_parent))
    group_parent = simplex(group_parent)
    if len(group_parent) != count:
        raise ValueError("group-parent length mismatch")
    reference_group = aggregate_vector(reference_parent, groups)
    expanded = group_parent[groups] * reference_parent / reference_group[groups]
    return simplex(expanded)


def component_confusion(log_probability, labels, components):
    """Mean calibrated classifier response for each simulated source component."""
    log_probability = np.asarray(log_probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if log_probability.shape != (len(labels), components):
        raise ValueError("classifier response shape mismatch")
    probability = np.exp(log_probability)
    result = np.empty((components, components), dtype=np.float64)
    for component in range(components):
        member = labels == component
        if not member.any():
            raise ValueError(f"component {component} absent from hierarchy reference")
        result[component] = probability[member].mean(axis=0)
    return result / result.sum(axis=1, keepdims=True)


def _normalized_block(values):
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    standardized = centered / np.where(scale > 1e-10, scale, 1.0)
    rms = np.sqrt(np.mean(np.sum(standardized**2, axis=1)))
    return standardized / max(float(rms), 1e-12)


def build_joint_hierarchy(
    centers,
    scales,
    confusions,
    resolutions,
    *,
    physical_weight=1.0,
    photometric_weight=1.0,
    response_rank=16,
    tail_component=0,
):
    """Build nested groups from joint physical and photometric similarity.

    Classifier responses come only from labeled reference simulations. Known
    target-parent weights are deliberately absent from this function.
    """
    centers = np.asarray(centers, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if centers.ndim != 2 or scales.shape != centers.shape:
        raise ValueError("component centers and scales must have matching matrices")
    components = len(centers)
    resolutions = tuple(sorted({int(value) for value in resolutions}))
    if (
        not resolutions
        or resolutions[0] < 2
        or resolutions[-1] > components
        or not 0 <= tail_component < components
    ):
        raise ValueError("invalid hierarchy resolutions or tail component")
    if (
        physical_weight < 0
        or photometric_weight < 0
        or not (physical_weight + photometric_weight > 0)
    ):
        raise ValueError("hierarchy block weights must be nonnegative and nonzero")

    confusion_blocks = []
    for confusion in confusions:
        confusion = np.asarray(confusion, dtype=np.float64)
        if confusion.shape != (components, components) or np.any(confusion < 0):
            raise ValueError("invalid component confusion matrix")
        confusion = confusion / confusion.sum(axis=1, keepdims=True)
        confusion_blocks.append(np.sqrt(np.maximum(confusion, 0)))
    if not confusion_blocks and photometric_weight:
        raise ValueError("photometric hierarchy requires classifier responses")

    physical = _normalized_block(np.column_stack([centers, np.log(scales)]))
    blocks = []
    if physical_weight:
        blocks.append(np.sqrt(physical_weight) * physical)
    if photometric_weight:
        response = np.concatenate(confusion_blocks, axis=1)
        response -= response.mean(axis=0, keepdims=True)
        u, singular, _ = np.linalg.svd(response, full_matrices=False)
        rank = min(int(response_rank), components - 1, len(singular))
        if rank <= 0:
            raise ValueError("response_rank must be positive")
        embedding = _normalized_block(u[:, :rank] * singular[:rank])
        blocks.append(np.sqrt(photometric_weight) * embedding)
    embedding = np.concatenate(blocks, axis=1)

    retained = np.flatnonzero(np.arange(components) != tail_component)
    tree = linkage(embedding[retained], method="ward", optimal_ordering=True)
    result = {}
    for resolution in resolutions:
        if resolution == components:
            groups = np.arange(components)
        else:
            cut = cut_tree(tree, n_clusters=resolution - 1).ravel()
            groups = np.empty(components, dtype=int)
            groups[tail_component] = 0
            groups[retained] = cut + 1
        groups, count = _groups(groups, components)
        if count != resolution:
            raise RuntimeError("hierarchy cut did not produce requested resolution")
        result[resolution] = groups
    return result, embedding
