from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from euclid_dsps.amortized.adaptive_smc import (
    build_adaptive_smc_kernels,
    run_adaptive_smc,
)
from euclid_dsps.amortized.config import amortized_config
from euclid_dsps.config import load_config


def _normal_logpdf(value, mean, sigma=1.0):
    return -0.5 * jnp.sum(((value - mean) / sigma) ** 2, axis=-1)


def test_adaptive_smc_transports_proposal_and_preserves_normalization() -> None:
    key = jax.random.PRNGKey(12)
    particles = jax.random.normal(key, (512, 2, 1))
    target_mean = jnp.asarray([[[1.0], [-1.0]]])

    result = run_adaptive_smc(
        key=jax.random.PRNGKey(13),
        initial_particles=particles,
        proposal_logdensity_fn=lambda value: _normal_logpdf(value, 0.0),
        target_logdensity_fn=lambda value: _normal_logpdf(
            value, target_mean, sigma=0.5
        ),
        target_ess_fraction=0.6,
        max_stages=32,
        mala_steps=2,
        mala_step_size=0.15,
        mala_particle_chunk_size=64,
    )

    means = jnp.sum(result.weights[..., None] * result.particles, axis=0)[:, 0]
    np.testing.assert_allclose(np.asarray(means), [1.0, -1.0], atol=0.15)
    np.testing.assert_allclose(np.asarray(result.weights.sum(axis=0)), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(result.final_beta), 1.0, atol=1e-6)
    assert result.beta_to.shape[0] >= 2
    assert np.all(result.pre_resample_ess > 0.0)
    assert np.all((result.mala_acceptance >= 0.0) & (result.mala_acceptance <= 1.0))


def test_adaptive_smc_rejects_invalid_mala_particle_chunk_size() -> None:
    particles = jnp.zeros((8, 1, 1))
    with pytest.raises(ValueError, match="chunk size must be positive"):
        run_adaptive_smc(
            key=jax.random.PRNGKey(1),
            initial_particles=particles,
            proposal_logdensity_fn=lambda value: _normal_logpdf(value, 0.0),
            target_logdensity_fn=lambda value: _normal_logpdf(value, 1.0),
            mala_particle_chunk_size=0,
        )


def test_adaptive_smc_reuses_kernels_with_dynamic_density_arguments() -> None:
    def logq(value, _mean):
        return _normal_logpdf(value, 0.0)

    def target(value, mean):
        return _normal_logpdf(value, mean, sigma=0.7)

    kernels = build_adaptive_smc_kernels(
        proposal_logdensity_fn=logq,
        target_logdensity_fn=target,
        mala_step_size=0.1,
    )
    means = []
    for seed, target_mean in ((20, 0.8), (21, -0.8)):
        result = run_adaptive_smc(
            key=jax.random.PRNGKey(seed),
            initial_particles=jax.random.normal(
                jax.random.PRNGKey(seed + 100), (256, 1, 1)
            ),
            proposal_logdensity_fn=logq,
            target_logdensity_fn=target,
            density_args=(jnp.asarray([[[target_mean]]]),),
            kernels=kernels,
            target_ess_fraction=0.5,
            max_stages=32,
            mala_steps=1,
            mala_step_size=0.1,
        )
        means.append(float(jnp.sum(result.weights[..., None] * result.particles)))
    np.testing.assert_allclose(means, [0.8, -0.8], atol=0.2)


def test_popcosmos_error_floor_variants_change_only_declared_likelihood() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "configs/experiments/popcosmos_native15d_rws.yaml",
        root / "configs/experiments/popcosmos_native15d_rws_floor02.yaml",
        root / "configs/experiments/popcosmos_native15d_rws_floor05.yaml",
    ]
    floors = []
    for path in paths:
        config = load_config(path)
        floors.append(amortized_config(config)["likelihood"]["error_floor_frac"])
        assert config["fit"]["flux_error_floor_frac"] == floors[-1]
        assert len(config["bands"]) == 26
        assert amortized_config(config)["encoder"]["latent_dim"] == 15
    assert floors == [0.0, 0.02, 0.05]


@pytest.mark.skipif(
    importlib.util.find_spec("equinox") is None, reason="equinox is not installed"
)
def test_weighted_refresh_updates_encoder_but_not_prior() -> None:
    from euclid_dsps.amortized.elbo import AmortizedModel
    from euclid_dsps.amortized.encoder import GaussianEncoder
    from euclid_dsps.amortized.flows import StandardNormalPrior
    from euclid_dsps.amortized.proposal_refresh import (
        refresh_encoder_from_weighted_particles,
    )
    from euclid_dsps.calibration import GlobalSedScaleState

    encoder = GaussianEncoder(
        jax.random.PRNGKey(1),
        input_dim=2,
        latent_dim=1,
        hidden_sizes=(8,),
        activation="gelu",
        log_std_min=-6.0,
        log_std_max=2.0,
        initial_log_std=-1.0,
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=1),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    features = jnp.asarray(
        [[-1.0, 1.0], [-0.5, 1.0], [0.5, 1.0], [1.0, 1.0]],
        dtype=jnp.float32,
    )
    centers = features[:, :1]
    particles = jnp.broadcast_to(centers[None, :, :], (16, 4, 1))
    particles = particles + 0.05 * jax.random.normal(
        jax.random.PRNGKey(2), particles.shape
    )
    weights = jnp.full((16, 4), 1.0 / 16.0)
    prior_before = [
        np.asarray(value).copy() for value in jax.tree_util.tree_leaves(model.prior)
    ]

    result = refresh_encoder_from_weighted_particles(
        model,
        features=features,
        particles=particles,
        weights=weights,
        epochs=8,
        object_batch_size=2,
        learning_rate=2.0e-3,
        validation_fraction=0.25,
        seed=3,
    )

    prior_after = [
        np.asarray(value) for value in jax.tree_util.tree_leaves(result.model.prior)
    ]
    assert all(
        np.array_equal(before, after)
        for before, after in zip(prior_before, prior_after, strict=True)
    )
    assert result.best_validation_nll <= result.initial_validation_nll


def test_smc_slurm_contract_separates_pilot_refresh_and_em() -> None:
    root = Path(__file__).resolve().parents[1]
    pilot = (root / "scripts/popcosmos_posthoc_smc_h100.slurm").read_text()
    refresh = (root / "scripts/popcosmos_posthoc_smc_refresh_h100.slurm").read_text()
    submit = (root / "scripts/submit_popcosmos_posthoc_smc_pilot.sh").read_text()
    assert (
        "array_tasks=$((${#SMC_VARIANTS[@]} * ${#SMC_SEEDS[@]} * N_SHARDS))" in submit
    )
    assert 'ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-12}"' in submit
    assert 'CALIBRATION_INDICES="$SOURCE_ROOT/train/validation_indices.npy"' in submit
    assert "proposal_probe_indices.npy" in submit
    assert "floor_0p00" in pilot and "floor_0p02" in pilot and "floor_0p05" in pilot
    assert 'SMC_VARIANTS_CSV="${SMC_VARIANTS_CSV:-' in pilot
    assert (
        '--variants "$SMC_VARIANTS_CSV"'
        in (root / "scripts/popcosmos_posthoc_smc_finalize.slurm").read_text()
    )
    assert "--constraint=h100" in pilot
    assert "--gres=gpu:1" in pilot
    assert "selection_status" in refresh
    assert "moderate_k${PROBE_SAMPLES}_importance" in refresh
    assert "posthoc_empirical_bayes" not in pilot
    assert "ALLOW_LOW_ESS" not in pilot


def test_smc_shards_are_combined_with_exact_cohort(tmp_path: Path) -> None:
    root = tmp_path / "seed_260817"
    common = {
        "algorithm": "adaptive q-to-target SMC",
        "density_space": "latent_x",
        "particles_per_object": 16,
        "seed": 260817,
        "likelihood": {"type": "student_t"},
        "target_contract": "target",
        "proposal_contract": "proposal",
        "selection_contract": "photometry-only",
        "target_ess_fraction": 0.5,
        "mala_steps": 2,
        "mala_step_size": 0.02,
        "bands": ["a"],
        "git_commit": "abc",
        "wall_seconds": 2.0,
        "inputs": {"config": {}, "row_indices": {}},
    }
    for shard_index, row_index in enumerate((10, 20)):
        shard = root / f"shard_{shard_index:03d}"
        shard.mkdir(parents=True)
        (shard / "DONE").touch()
        (shard / "smc_summary.json").write_text(json.dumps(common))
        pd.DataFrame(
            {
                "row_index": [row_index],
                "log_evidence": [1.0],
                "final_ess_fraction": [0.5],
                "unique_ancestor_fraction": [0.5],
                "max_final_weight": [0.08],
                "mean_mala_acceptance": [0.5],
                "weighted_chi2_per_valid_band": [1.0],
                "weighted_reduced_chi2": [1.0],
                "weighted_fraction_abs_gt_5": [0.0],
            }
        ).to_parquet(shard / "smc_object_diagnostics.parquet", index=False)
        pd.DataFrame(
            {
                "row_index": [row_index],
                "stage": [1],
                "beta_to": [1.0],
            }
        ).to_parquet(shard / "smc_stage_diagnostics.parquet", index=False)
        pd.DataFrame(
            {
                "row_index": [row_index],
                "band": ["a"],
                "weighted_abs_chi": [1.0],
                "weighted_frac_abs_gt_5": [0.0],
            }
        ).to_parquet(shard / "posterior_predictive_band_objects.parquet", index=False)
        np.save(shard / "row_indices.npy", np.asarray([row_index], dtype=np.int64))
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/combine_popcosmos_posthoc_smc_shards.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--expected-shards",
            "2",
            "--expected-objects",
            "2",
        ],
        check=True,
    )
    assert (root / "DONE").is_file()
    np.testing.assert_array_equal(np.load(root / "row_indices.npy"), [10, 20])
    summary = json.loads((root / "smc_summary.json").read_text())
    assert summary["support_gate"]["status"] == "PASS"
    assert summary["sharding"]["n_shards"] == 2


def test_smc_summary_selects_only_stable_adequate_variant(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    evidence = {"floor_0p00": 1.0, "floor_0p02": 3.0, "floor_0p05": 2.0}
    for variant, center in evidence.items():
        for seed, offset in ((260817, -0.05), (260818, 0.05)):
            run = root / variant / f"seed_{seed}"
            run.mkdir(parents=True)
            (run / "DONE").touch()
            pd.DataFrame(
                {
                    "row_index": [10, 20],
                    "log_evidence": [center + offset, center + 0.2 + offset],
                }
            ).to_parquet(run / "smc_object_diagnostics.parquet", index=False)
            (run / "smc_summary.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "support_gate": {"status": "PASS"},
                        "metrics": {
                            "mean_log_evidence": center + offset,
                            "median_log_evidence": center + offset,
                            "median_final_ess_fraction": 0.5,
                            "median_unique_ancestor_fraction": 0.4,
                            "median_max_final_weight": 0.01,
                            "median_mala_acceptance": 0.5,
                            "median_chi2_per_valid_band": 2.0,
                            "median_reduced_chi2": 3.0,
                            "median_fraction_abs_gt_5": 0.02,
                        },
                    }
                ),
                encoding="utf-8",
            )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/summarize_popcosmos_posthoc_smc.py"
    )
    subprocess.run(
        [sys.executable, str(script), "--root", str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((root / "pilot_selection/selection_summary.json").read_text())
    assert summary["selection_status"] == "PASS"
    assert summary["selected_variant"] == "floor_0p02"
    assert summary["spectroscopy_used"] is False


def test_smc_summary_accepts_preselected_variant(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    variant = "floor_0p05"
    for seed, offset in ((260817, -0.05), (260818, 0.05)):
        run = root / variant / f"seed_{seed}"
        run.mkdir(parents=True)
        (run / "DONE").touch()
        pd.DataFrame(
            {"row_index": [10, 20], "log_evidence": [2.0 + offset, 2.2 + offset]}
        ).to_parquet(run / "smc_object_diagnostics.parquet", index=False)
        (run / "smc_summary.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "support_gate": {"status": "PASS"},
                    "metrics": {
                        "mean_log_evidence": 2.0 + offset,
                        "median_log_evidence": 2.0 + offset,
                        "median_chi2_per_valid_band": 2.0,
                        "median_fraction_abs_gt_5": 0.02,
                    },
                }
            )
        )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/summarize_popcosmos_posthoc_smc.py"
    )
    subprocess.run(
        [sys.executable, str(script), "--root", str(root), "--variants", variant],
        check=True,
    )
    summary = json.loads((root / "pilot_selection/selection_summary.json").read_text())
    assert summary["selection_status"] == "PASS"
    assert summary["selected_variant"] == variant
    assert summary["evaluated_variants"] == [variant]
