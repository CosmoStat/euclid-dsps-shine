from __future__ import annotations

import numpy as np

from euclid_dsps.amortized.posterior_bank import POSTERIOR_METHOD_CODES
from euclid_dsps.amortized.sc_asmc_estep import (
    _micro_batch_candidates,
    _prefetch_iterator,
)


def test_method_codes_have_all_required_hierarchy_levels() -> None:
    assert tuple(POSTERIOR_METHOD_CODES) == (
        "IS",
        "primary SMC",
        "fallback SMC",
        "extended SMC",
        "unresolved",
    )


def test_array_hierarchy_helpers_compile() -> None:
    import jax.numpy as jnp

    from euclid_dsps.amortized.hierarchical_e_step import (
        _pad_particle_objects,
        _shard_particle_objects,
    )

    particles = jnp.arange(64 * 3 * 2).reshape(64, 3, 2)
    padded = _pad_particle_objects(particles, 8)
    sharded = _shard_particle_objects(padded, 4)
    assert padded.shape == (64, 8, 2)
    assert sharded.shape == (4, 64, 2, 2)
    assert isinstance(sharded, np.ndarray)
    assert sharded.flags.c_contiguous
    np.testing.assert_array_equal(np.asarray(padded[:, 3]), np.asarray(particles[:, 0]))
    np.testing.assert_array_equal(sharded[1, :, 1], np.asarray(particles[:, 0]))


def test_micro_batch_candidates_are_device_divisible_and_descending() -> None:
    values = _micro_batch_candidates(128, 4)

    assert values[0] == 128
    assert values[-1] == 4
    assert values == sorted(set(values), reverse=True)
    assert all(value % 4 == 0 for value in values)


def test_host_prefetch_preserves_batch_order() -> None:
    assert list(_prefetch_iterator(iter(range(9)), depth=2)) == list(range(9))
