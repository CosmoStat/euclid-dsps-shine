from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.amortized.config import (
    amortized_config,
    require_amortized_dependencies,
)
from euclid_dsps.amortized.population_vem import (
    ArrayShardContract,
    fixed_reference_selection_terms,
    is_array_bank_shard_complete,
    make_pmap_fixed_reference_prior_step,
    merge_array_bank_shards,
    read_array_bank_shard,
    selection_calibration_summary,
    write_array_bank_shard,
)
from euclid_dsps.config import load_config

eqx, optax = require_amortized_dependencies()

SHA = "a" * 64
ROOT = Path(__file__).resolve().parents[1]


class _LocationNormalPrior(eqx.Module):
    location: jax.Array
    latent_dim: int = eqx.field(static=True)

    def log_prob(self, value):
        return -0.5 * jnp.sum(jnp.square(value - self.location), axis=-1)


def _q_contract(*, rows_sha: str = SHA) -> ArrayShardContract:
    return ArrayShardContract(
        kind="q_train",
        dataset_sha256=SHA,
        checkpoint_sha256="b" * 64,
        latent_transform_sha256="c" * 64,
        code_commit="deadbeef",
        truth_used=False,
        draws_per_object=3,
        feature_stats_sha256="d" * 64,
        row_indices_sha256=rows_sha,
    )


def test_fixed_reference_selection_terms_matches_direct_estimate() -> None:
    log_beta = jnp.log(jnp.asarray([0.1, 0.2, 0.4, 0.8], dtype=jnp.float32))
    terms = fixed_reference_selection_terms(
        jnp.zeros(4, dtype=jnp.float32),
        jnp.zeros(4, dtype=jnp.float32),
        log_beta,
    )
    weights = np.asarray([0.1, 0.2, 0.4, 0.8])
    expected_ess = weights.sum() ** 2 / np.square(weights).sum()
    assert np.isclose(float(terms.alpha), weights.mean(), rtol=1.0e-6)
    assert np.isclose(float(terms.ess), expected_ess, rtol=1.0e-6)
    assert np.isclose(
        float(terms.maximum_normalized_weight),
        weights.max() / weights.sum(),
        rtol=1.0e-6,
    )
    assert bool(terms.finite)


def test_array_bank_is_atomic_resumable_and_row_complete(tmp_path: Path) -> None:
    root = tmp_path / "bank"
    arrays = {
        "row_index": np.asarray([7, 3], dtype=np.int64),
        "x": np.ones((2, 3, 4), dtype=np.float32),
        "log_q": np.zeros((2, 3), dtype=np.float32),
    }
    receipt = write_array_bank_shard(root, 0, arrays, _q_contract())
    resumed = write_array_bank_shard(root, 0, arrays, _q_contract())
    assert receipt == resumed
    shard = root / "shards" / "shard_00000"
    assert is_array_bank_shard_complete(shard, validate_arrays=True)
    assert np.array_equal(read_array_bank_shard(shard)["row_index"], [7, 3])
    manifest = merge_array_bank_shards(
        root,
        expected_shards=1,
        expected_row_indices=np.asarray([3, 7]),
    )
    assert manifest["status"] == "complete"
    assert manifest["shard_count"] == 1


def test_array_bank_rejects_truth_and_resume_provenance_changes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bank"
    arrays = {
        "row_index": np.asarray([1], dtype=np.int64),
        "x": np.ones((1, 3, 2), dtype=np.float32),
        "log_q": np.zeros((1, 3), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="truth is forbidden"):
        write_array_bank_shard(
            root,
            0,
            arrays,
            ArrayShardContract(**{**_q_contract().__dict__, "truth_used": True}),
        )
    write_array_bank_shard(root, 0, arrays, _q_contract())
    with pytest.raises(ValueError, match="provenance mismatch"):
        write_array_bank_shard(root, 0, arrays, _q_contract(rows_sha="e" * 64))


def test_selection_calibration_summary_gates_global_and_redshift_errors() -> None:
    selected = np.asarray(([False] * 5 + [True] * 5) * 40)
    redshift = np.repeat(np.linspace(0.1, 2.0, 40), 10)
    calibrated = selection_calibration_summary(
        selected.astype(float),
        selected,
        redshift,
        probability_bins=2,
        redshift_bins=4,
        minimum_redshift_bin_objects=50,
    )
    assert calibrated["status"] == "PASS"
    biased = selection_calibration_summary(
        np.full(len(selected), 0.9),
        selected,
        redshift,
        probability_bins=2,
        redshift_bins=4,
        minimum_redshift_bin_objects=50,
    )
    assert biased["status"] == "FAIL"
    assert biased["global_absolute_error"] > 0.03


def test_pmap_prior_step_updates_all_replicas_with_finite_fixed_bank() -> None:
    devices = jax.local_devices()
    count = len(devices)
    source = _LocationNormalPrior(jnp.zeros(2), latent_dim=2)
    candidate = _LocationNormalPrior(jnp.zeros(2), latent_dim=2)
    optimizer = optax.sgd(0.05)
    optimizer_state = optimizer.init(eqx.filter(candidate, eqx.is_inexact_array))

    def replicate(tree):
        return jax.tree_util.tree_map(
            lambda value: (
                jnp.broadcast_to(value, (count, *value.shape))
                if eqx.is_array(value)
                else value
            ),
            tree,
        )

    step = make_pmap_fixed_reference_prior_step(
        optimizer=optimizer,
        minimum_reference_ess_fraction=0.1,
        maximum_alpha_relative_mc_error=1.0,
        maximum_kl_per_dimension=1.0,
    )
    updated, _, metrics = step(
        replicate(candidate),
        replicate(source),
        replicate(optimizer_state),
        jnp.full((count, 2, 3, 2), 0.4),
        jnp.ones((count, 2), dtype=jnp.bool_),
        jnp.zeros((count, 8, 2)),
        jnp.zeros((count, 8)),
        jnp.zeros((count, 8)),
        jnp.full((count,), 0.1),
    )
    assert np.all(np.asarray(metrics.update_applied))
    assert np.all(np.asarray(metrics.reference_ess_fraction) >= 0.99)
    assert np.all(np.asarray(updated.location) > 0.0)


def test_population_vem_refresh_is_small_prior_frozen_avi() -> None:
    config = amortized_config(
        load_config(
            ROOT / "configs/experiments/feniks_sc_drws_r29_population_vem_refresh.yaml"
        )
    )
    assert config["objective"]["mode"] == "stochastic_elbo"
    assert config["objective"]["selection_correction"]["enabled"] is False
    assert config["objective"]["sleep"]["enabled"] is False
    assert config["prior"]["train_jointly"] is False
    assert config["training"]["epochs"] == 2
    assert config["training"]["n_samples"] == 2
    assert config["training"]["jax_batch_size"] == 256
    assert config["output"]["save_posterior_preview"] is False


def test_population_vem_submission_uses_bounded_parallel_h100_stages() -> None:
    submit = (ROOT / "scripts/submit_feniks_sc_drws_population_vem.sh").read_text()
    bank = (ROOT / "scripts/feniks_sc_drws_population_vem_bank_h100.slurm").read_text()
    prior = (
        ROOT / "scripts/feniks_sc_drws_population_vem_prior_h100.slurm"
    ).read_text()
    refresh = (
        ROOT / "scripts/feniks_sc_drws_population_vem_refresh_h100.slurm"
    ).read_text()
    final = (
        ROOT / "scripts/feniks_sc_drws_population_vem_finalize_h100.slurm"
    ).read_text()
    assert "--array=0-35%24" in submit
    assert "--array=0-15%16" in submit
    assert 'export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"' in submit
    assert submit.index("export PYTHONPATH=") < submit.index(
        "python scripts/prepare_feniks_sc_drws_population_vem.py"
    )
    assert "imported population_vem outside the active" in submit
    assert "git worktree add --detach" in submit
    assert "afterok:$BANK_JOB" in submit
    assert "afterok:$REFRESH_JOB" in submit
    assert "#SBATCH --gres=gpu:1" in bank
    assert "#SBATCH --gres=gpu:4" in prior
    assert "#SBATCH --gres=gpu:4" in refresh
    assert "#SBATCH --gres=gpu:1" in final
    assert "--n-samples 2" in refresh
