from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.benchmark_feniks_production_local import compile_function, digest, measure


def test_file_digest(tmp_path):
    path = tmp_path / "asset"
    path.write_bytes(b"production-asset")
    assert digest(path) == hashlib.sha256(b"production-asset").hexdigest()


def test_compiled_measurement_checks_values_and_records_samples():
    arguments = (jnp.array([1.0, 2.0, 3.0]),)
    executable, value, metadata = compile_function(
        jax.value_and_grad(lambda x: jnp.sum(x * x)), arguments
    )
    np.testing.assert_allclose(value[0], 14.0)
    np.testing.assert_allclose(value[1], [2.0, 4.0, 6.0])
    timing = measure(executable, arguments, 4, metadata)
    assert len(timing["raw_ms"]) == 4
    assert timing["median_ms"] > 0
    assert timing["p95_ms"] >= timing["median_ms"]
    assert timing["compile_seconds"] >= 0
    assert timing["memory_bytes"]["argument_size_in_bytes"] > 0


def test_nonfinite_benchmark_fails():
    with pytest.raises(ValueError, match="Nonfinite"):
        compile_function(lambda x: x / 0, (jnp.ones(2),))


def test_candidate_reference_export_survives_production_move():
    from euclid_dsps.photometry_quadrature import merged_integral_jax
    from scripts.photometry_merge_candidate import (
        reference_integral,
        specialized_integral,
    )

    assert reference_integral is merged_integral_jax
    assert reference_integral is not specialized_integral
