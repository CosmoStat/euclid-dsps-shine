from __future__ import annotations

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.amortized.exact_posterior import (
    MCLMCSettings,
    NUTSSettings,
    _float64_logdensity,
    combine_chain_diagnostics,
    normalized_importance_weights,
    run_adjusted_mclmc_chain,
    run_nuts_chain,
    systematic_resample,
)


def _normal_logdensity(x):
    return -0.5 * jnp.sum(x**2)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_importance_weights_and_systematic_resampling_are_normalized() -> None:
    result = normalized_importance_weights(
        np.array([-3.0, -2.0, -1.0]),
        np.array([-2.0, -2.0, -2.0]),
    )

    assert np.isclose(np.sum(result["weight"]), 1.0)
    assert np.isclose(np.sum(result["psis_weight"]), 1.0)
    assert 1.0 <= result["raw_ess"] <= 3.0
    indices = systematic_resample(result["psis_weight"], 100, seed=4)
    assert indices.shape == (100,)
    assert np.all((indices >= 0) & (indices < 3))


def test_nuts_chain_is_resumable_and_writes_diagnostics(tmp_path: Path) -> None:
    pytest.importorskip("blackjax")
    settings = NUTSSettings(warmup_steps=40, sample_chunks=(12, 16))
    chain_dirs = []
    for chain in range(4):
        directory = tmp_path / f"chain_{chain:02d}"
        run_nuts_chain(
            _normal_logdensity,
            jnp.array([0.2, -0.1]),
            seed=chain,
            settings=settings,
            out_dir=directory,
        )
        chain_dirs.append(directory)

    chunk = chain_dirs[0] / "chunks" / "part_000000.parquet"
    digest = _sha256(chunk)
    resumed = run_nuts_chain(
        _normal_logdensity,
        jnp.array([0.2, -0.1]),
        seed=0,
        settings=settings,
        out_dir=chain_dirs[0],
    )
    assert _sha256(chunk) == digest
    assert resumed["stored_samples"] == 28

    diagnostics, summary = combine_chain_diagnostics(
        chain_dirs, parameter_names=("a", "b")
    )
    assert diagnostics["parameter"].tolist() == ["a", "b"]
    assert np.isfinite(summary["max_rhat"])
    assert summary["min_bulk_ess"] > 0


def test_adjusted_mclmc_uses_real_thinning_and_valid_step_size(
    tmp_path: Path,
) -> None:
    pytest.importorskip("blackjax")
    settings = MCLMCSettings(
        tune_steps=60,
        sample_chunks=(12,),
        thinning=4,
        initial_step_size=1.0e-3,
        frac_tune1=0.4,
        frac_tune2=0.4,
        frac_tune3=0.2,
    )
    manifest = run_adjusted_mclmc_chain(
        _normal_logdensity,
        jnp.array([0.2, -0.1]),
        seed=12,
        settings=settings,
        out_dir=tmp_path / "mclmc",
    )

    assert manifest["stored_samples"] == 12
    assert manifest["kernel_transitions"] == 48
    assert manifest["step_size"] > 0.0
    assert manifest["L"] > 0.0
    assert manifest["integrator_steps_after_warmup"] >= 48


def test_samplers_promote_a_float32_target_to_float64(tmp_path: Path) -> None:
    pytest.importorskip("blackjax")

    jax.config.update("jax_enable_x64", False)

    def float32_logdensity(x):
        observed_target_dtypes.append(x.dtype)
        value = x.astype(jnp.float32)
        return -0.5 * jnp.sum(value**2)

    observed_target_dtypes = []
    jax.config.update("jax_enable_x64", True)
    wrapped = _float64_logdensity(float32_logdensity)
    position = jnp.array([0.1, -0.2], dtype=jnp.float64)
    assert wrapped(position).dtype == jnp.float64
    assert jax.grad(wrapped)(position).dtype == jnp.float64
    assert observed_target_dtypes
    assert all(dtype == jnp.float32 for dtype in observed_target_dtypes)

    try:
        nuts = run_nuts_chain(
            float32_logdensity,
            jnp.array([0.1, -0.2], dtype=jnp.float32),
            seed=51,
            settings=NUTSSettings(warmup_steps=20, sample_chunks=(4,)),
            out_dir=tmp_path / "nuts_mixed_dtype",
        )
        mclmc = run_adjusted_mclmc_chain(
            float32_logdensity,
            jnp.array([0.1, -0.2], dtype=jnp.float32),
            seed=52,
            settings=MCLMCSettings(
                tune_steps=30,
                sample_chunks=(4,),
                initial_step_size=1.0e-3,
            ),
            out_dir=tmp_path / "mclmc_mixed_dtype",
        )
    finally:
        jax.config.update("jax_enable_x64", False)

    assert nuts["stored_samples"] == 4
    assert mclmc["stored_samples"] == 4
    assert nuts["sampling_dtype"] == "float64"
    assert nuts["target_dtype"] == "float32"
    assert mclmc["sampling_dtype"] == "float64"
    assert mclmc["target_dtype"] == "float32"
