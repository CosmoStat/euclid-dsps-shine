from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from euclid_dsps.amortized.avi_experiments import Arm
from euclid_dsps.amortized.expert_audit import (
    gate_diagnostics,
    pairwise_mean_separation,
    responsibility_diagnostics,
)
from scripts.feniks_avi_experiments import initialize_candidate, read, sha, write
from scripts.feniks_avi_next_experiments import NEXT_ARMS, prepare


def test_gate_and_responsibility_diagnostics_detect_collapse():
    probabilities = np.array([[0.97, 0.01, 0.01, 0.01]] * 8)
    gate = gate_diagnostics(probabilities)
    assert gate["experts"] == 4
    assert gate["median_active_experts"] == 1
    assert gate["winning_fraction"] == [1.0, 0.0, 0.0, 0.0]
    responsibilities = np.broadcast_to(probabilities[0, :, None, None], (4, 16, 8))
    resp = responsibility_diagnostics(responsibilities)
    assert resp["median_max_responsibility"] == pytest.approx(0.97)
    with pytest.raises(ValueError, match="normalized"):
        gate_diagnostics(probabilities * 2)


def test_pairwise_separation_splits_physical_and_sfh_coordinates():
    means = np.zeros((2, 4, 15))
    means[1, :, :5] = 2.0
    means[1, :, 5:] = 0.5
    rows = pairwise_mean_separation(means)
    assert rows[0]["physical_5d_median_rms"] == pytest.approx(2.0)
    assert rows[0]["sfh_10d_median_rms"] == pytest.approx(0.5)


def test_candidate_uses_requested_expert_count(monkeypatch):
    import equinox as eqx

    from tests.test_avi_experiments import model

    jax.config.update("jax_enable_x64", True)
    source = model()
    monkeypatch.setattr(
        "euclid_dsps.amortized.train.build_amortized_model",
        lambda *args, **kwargs: source,
    )
    for count in (2, 4, 8):
        candidate = initialize_candidate(
            source, {"unused": True}, None, Arm("test", experts=count), 7
        )
        assert candidate.n_components == count
        assert len(candidate.experts) == count
        leaves = jax.tree_util.tree_leaves(eqx.filter(candidate, eqx.is_array))
        assert leaves


def test_next_prepare_freezes_upstream_b_and_contract(tmp_path, monkeypatch):
    training = tmp_path / "training"
    training.mkdir()
    for name in (
        "source_config.yaml",
        "train.npy",
        "validation.npy",
        "teachers.npz",
        "teachers.json",
    ):
        (training / name).write_bytes(b"fixture")
    (training / "arms/B_experts").mkdir(parents=True)
    (training / "arms/B_experts/encoder.eqx").write_bytes(b"encoder")
    write(training / "arms/B_experts/FINAL.json", {"status": "TRAINING_COMPLETE"})
    write(training / "MANIFEST.json", {"fixture": True})
    upstream = {
        "source": {"checkpoint": "x", "feature_stats": "y"},
        "seed": 3,
        "train_rows": 37641,
        "validation_rows": 512,
        "validation_catalog": "test.parquet",
        "global_batch": 256,
        "local_microbatch": 32,
        "accumulation": 2,
        "gpus": 4,
        "particles": 128,
        "decoder_draw_block": 8,
        "validation_particles": 512,
        "learning_rate": 2e-5,
        "warmup_fraction": 0.05,
        "bootstrap_epochs": 6,
        "cycle": ["sleep", "sleep", "wake"],
    }
    monkeypatch.setattr(
        "scripts.feniks_avi_experiments.check_inputs", lambda root: upstream
    )
    monkeypatch.chdir(tmp_path)
    for folder in ("euclid_dsps", "scripts", "configs"):
        (tmp_path / folder).mkdir()
    out = tmp_path / "next"
    prepare(training, out)
    manifest = read(out / "MANIFEST.json")
    assert [arm["name"] for arm in manifest["arms"]] == [a.name for a in NEXT_ARMS]
    assert manifest["posterior_weight_contract"].startswith("exact full_15d")
    assert manifest["selection_in_object_weights"] is False
    assert manifest["prior_sweeps"] == 5
    assert manifest["hashes"][
        str((training / "arms/B_experts/encoder.eqx").resolve())
    ] == sha(training / "arms/B_experts/encoder.eqx")
    assert json.loads((out / "MANIFEST.json").read_text())["truth_used"] is False


def test_launch_contract_uses_four_h100s_and_dependency():
    slurm = Path("scripts/feniks_avi_next.slurm").read_text()
    submit = Path("scripts/submit_feniks_avi_next_experiments.sh").read_text()
    assert "#SBATCH --gres=gpu:4" in slurm
    assert '--array="0-3%$CONCURRENCY"' in submit
    assert "afterok:${PREFLIGHT%%;*}" in submit
    assert "afterok:${TRAIN%%;*}" in submit


def test_prior_bank_keeps_exact_joint_weights(monkeypatch):
    from types import SimpleNamespace

    import jax.numpy as jnp

    from euclid_dsps.amortized.posterior_target import PosteriorTargetValues
    from euclid_dsps.amortized.train import LossBatch
    from scripts.feniks_avi_next_experiments import _prior_bank
    from tests.test_avi_experiments import model

    jax.config.update("jax_enable_x64", True)
    source = model()
    candidate = initialize_candidate(
        source, {"unused": True}, None, Arm("test", experts=4), 7
    )

    def target(active_model, x, batch, *args):
        flux = x[..., :1]
        loglike = -0.5 * jnp.sum((flux - batch.flux) ** 2, axis=-1)
        logprior = active_model.prior.log_prob(x)
        return PosteriorTargetValues(
            loglike + logprior,
            loglike,
            logprior,
            jnp.ones(loglike.shape, bool),
            flux,
            flux,
        )

    monkeypatch.setattr(
        "euclid_dsps.amortized.posterior_target.posterior_log_target", target
    )
    runtime = SimpleNamespace(
        latent_spec=None,
        context=None,
        model_args=None,
        parameter_names=("a", "b"),
        likelihood_config={},
        calibration_config={},
    )
    batch = LossBatch(
        jnp.ones((1, 8, 1)),
        jnp.ones((1, 8, 1)),
        jnp.ones((1, 8, 1), bool),
        jnp.ones((1, 8, 3)),
        jnp.zeros((1, 8, 0)),
    )
    evaluate = _prior_bank(source, candidate, runtime, 32, (jax.local_devices()[0],))
    particles, weights, metrics = evaluate(
        source, candidate, batch, jax.random.split(jax.random.PRNGKey(3), 1)
    )
    assert particles.shape == (1, 32, 8, 2)
    assert weights.shape == (1, 32, 8)
    np.testing.assert_allclose(np.asarray(weights).sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(metrics).all()
