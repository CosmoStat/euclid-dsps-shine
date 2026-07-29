from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from euclid_dsps.amortized.exact_posterior import (
    MCLMCSettings,
    NUTSSettings,
    _adjusted_mclmc_adaptation_adapter,
    _float64_logdensity,
    combine_chain_diagnostics,
    normalized_importance_weights,
    run_adjusted_mclmc_chain,
    run_batched_nuts_chains,
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


def test_batched_nuts_writes_independent_standard_chain_artifacts(
    tmp_path: Path,
) -> None:
    pytest.importorskip("blackjax")
    directories = tuple(tmp_path / f"chain_{index:02d}" for index in range(3))
    manifests = run_batched_nuts_chains(
        _normal_logdensity,
        jnp.asarray(
            [
                [0.2, -0.1],
                [-0.3, 0.4],
                [0.1, 0.3],
            ]
        ),
        seeds=(11, 12, 13),
        settings=NUTSSettings(
            warmup_steps=12,
            sample_chunks=(6,),
            max_num_doublings=3,
        ),
        out_dirs=directories,
    )

    assert len(manifests) == 3
    assert all(row["execution"] == "vmap_batched_chains" for row in manifests)
    frames = []
    for directory in directories:
        assert (directory / "warmup_summary.json").is_file()
        assert (directory / "tuned_parameters.npz").is_file()
        assert (directory / "sampling_state.pkl").is_file()
        assert (directory / "chain_manifest.json").is_file()
        frame = pd.read_parquet(
            directory / "chunks" / "part_000000.parquet"
        )
        assert len(frame) == 6
        frames.append(frame)
    assert not np.array_equal(
        frames[0][["x_00", "x_01"]].to_numpy(),
        frames[1][["x_00", "x_01"]].to_numpy(),
    )


def test_chain_summary_serializes_nonfinite_smoke_diagnostics_as_null(
    tmp_path: Path,
) -> None:
    chain_dirs = []
    for chain, value in enumerate((0.0, 1.0)):
        directory = tmp_path / f"chain_{chain:02d}"
        chunks = directory / "chunks"
        chunks.mkdir(parents=True)
        pd.DataFrame({"x_00": np.full(10, value)}).to_parquet(
            chunks / "part_000000.parquet", index=False
        )
        chain_dirs.append(directory)

    diagnostics, summary = combine_chain_diagnostics(
        chain_dirs, parameter_names=("constant_within_chain",)
    )

    assert np.isinf(diagnostics.loc[0, "rhat"])
    assert summary["max_rhat"] is None
    assert summary["finite_rhat_parameters"] == 0
    assert summary["passes_rhat_1_01"] is False
    assert '"max_rhat": null' in json.dumps(summary, allow_nan=False)


def test_adjusted_mclmc_adapter_supports_both_blackjax_signatures() -> None:
    def explicit_adaptation(mclmc_kernel, logdensity_fn, num_steps):
        del mclmc_kernel, logdensity_fn, num_steps

    def explicit_build_kernel():
        return lambda **kwargs: kwargs

    explicit_module = SimpleNamespace(
        adjusted_mclmc_find_L_and_step_size=explicit_adaptation,
        mcmc=SimpleNamespace(
            adjusted_mclmc=SimpleNamespace(build_kernel=explicit_build_kernel)
        ),
    )
    kernel, extra, api = _adjusted_mclmc_adaptation_adapter(
        explicit_module, _normal_logdensity
    )
    assert api == "explicit_logdensity"
    assert extra == {"logdensity_fn": _normal_logdensity}
    explicit_result = kernel(
        "key", "state", _normal_logdensity, 0.1, "mass", (2.0,)
    )
    assert explicit_result["logdensity_fn"] is _normal_logdensity
    assert explicit_result["integration_steps_params"] == (2.0,)

    def closed_adaptation(mclmc_kernel, num_steps):
        del mclmc_kernel, num_steps

    def closed_build_kernel(logdensity_fn, *, inverse_mass_matrix):
        assert logdensity_fn is _normal_logdensity
        assert inverse_mass_matrix == "mass"
        return lambda key, state, step, integration_steps: (
            key,
            state,
            step,
            integration_steps,
        )

    closed_module = SimpleNamespace(
        adjusted_mclmc_find_L_and_step_size=closed_adaptation,
        mcmc=SimpleNamespace(
            adjusted_mclmc=SimpleNamespace(build_kernel=closed_build_kernel)
        ),
    )
    kernel, extra, api = _adjusted_mclmc_adaptation_adapter(
        closed_module, _normal_logdensity
    )
    assert api == "closed_logdensity"
    assert extra == {}
    closed_result = kernel("key", "state", 2.2, 0.1, "mass")
    assert int(closed_result[-1]) == 3


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


def test_adjusted_mclmc_supports_fixed_geometry_smoke(tmp_path: Path) -> None:
    pytest.importorskip("blackjax")
    manifest = run_adjusted_mclmc_chain(
        _normal_logdensity,
        jnp.array([0.2, -0.1]),
        seed=13,
        settings=MCLMCSettings(
            tune_steps=0,
            sample_chunks=(3,),
            thinning=1,
            initial_step_size=5.0e-2,
        ),
        out_dir=tmp_path / "mclmc_fixed_smoke",
    )

    assert manifest["adaptation_mode"] == "fixed_geometry_smoke"
    assert manifest["actual_tuning_integrator_steps"] == 0
    assert manifest["integration_steps_per_transition"] == 4
    assert manifest["stored_samples"] == 3


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
